# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
The file has been adapted from two fairscale files:
 (1) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/moe_layer.py
 (2) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/top2gate.py
 Git commit hash: 34df606902a240567a0d898037ece55c2f1336cf
 We retain the following license from the original files:
"""

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from deepspeed.utils.timer import SynchronizedWallClockTimer
from deepspeed.utils import logger
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from deepspeed.utils import groups
from .mappings import drop_tokens, gather_tokens
import numpy as np
import time
import nvtx
import random
import os
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#os.environ['NCCL_BUFFSIZE'] = ''
# hansol
torch.set_printoptions(profile="full")

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

TOPK_GATE_TIMER = 'topk_gate'
MOE_TIMER = 'moe'
FIRST_ALLTOALL_TIMER = '1st_a2a'
SECOND_ALLTOALL_TIMER = '2nd_a2a'

uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe
    TUTEL_INSTALLED = True
except:
    # Fail silently so we don't spam logs unnecessarily if user isn't using tutel
    TUTEL_INSTALLED = False
    pass


def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(low=torch.tensor(1.0 - epsilon, device=device),
                                                      high=torch.tensor(1.0 + epsilon,
                                                                        device=device)).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


from deepspeed import comm as dist

# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx: Any,
            # TODO: replace with DS process group
            group: torch.distributed.ProcessGroup,
            input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))

class _AllToAll_MOE(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any,
            # TODO: replace with DS process group
            group: torch.distributed.ProcessGroup,
            input_slices,
            output_slices,
            input: Tensor,
            output: Tensor) -> Tensor:  # type: ignore
        #print("input_slices : ", input_slices )
        #print("output_slices : ", output_slices )
        ctx.group = group
        ctx.input_slices = input_slices
        ctx.output_slices = output_slices
        with nvtx.annotate("Input in AllToAll_MOE", color="blue"):
            input = input.contiguous()
        output = torch.empty_like(output)
        ctx.output = torch.empty_like(input)
            #output_slices = input_slices.transpose()
        #torch.cuda.memory_stats(dist.get_local_rank())
        #print("mem_allocat check 1",torch.cuda.memory_stats(dist.get_local_rank()))
        #time.sleep(1)
        #print("mem_allocat check after 1",torch.cuda.memory_stats(dist.get_local_rank()))   

        with nvtx.annotate("All to All single in AllToAll_MOE", color="blue"):
            dist.all_to_all_single(output, input, output_slices, input_slices, group=group, async_op=True)
        #time.sleep(1)
        #print("mem_allocat check 2 after sleep",torch.cuda.memory_stats(dist.get_local_rank()))
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, None, None, Tensor,None]:
        # hansol
        #return (None, None, None, _AllToAll_MOE.apply(ctx.group, ctx.input_slices, ctx.output_slices, *grad_output))
        # hansol
        return (None, None, None, _AllToAll_MOE.apply(ctx.group, ctx.output_slices, ctx.input_slices, *grad_output, ctx.output),None)


# einsum rewrites are on par or more performant
# switch can be bubbled up in future
USE_EINSUM = True


# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.
def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == 's,se->se':
        return a.reshape(a.shape[0], -1) * b
    elif rule == 'se,sc->sec':
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == 'se,se->s':
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == 'sec,sm->ecm':
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == 'sec,ecm->sm':
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == 'ks,ksm->sm':
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)


# The following functions are extracted and scripted
# because otherwise during a torch.jit.trace, the non-Tensor
# values used in the calculations get recorded as constants.
# torch.jit.script coerces them into Tensors and preserves
# their dynamic shapes. This enables ONNX export.
# We can't script the entire top1gating function because it
# includes stateful caching logic which is incompatible with ONNX.


@torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity


@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]


@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()




def top1gating(layer_count,
                mask,
                logits: Tensor,
                batch,
                seq_len,
               capacity_factor: float,
               min_capacity: int,
               used_token: Tensor = None,
               noisy_gate_policy: Optional[str] = None,
               drop_tokens: bool = False,
               use_rts: bool = True,
               use_tutel: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    if noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # everything is in fp32 in this function
    
    gates = F.softmax(logits, dim=1)
    
    '''
     #205 hht local gating
    #gates = F.softmax(logits, dim=1)
    device_number = dist.get_rank()
    tensor = torch.zeros_like(logits)
    #if device_number == 0:
        #device_number = 0
    #else :
    #    device_number = 1
    #if device_number == 0 :
    #    device_number = 2
    #else :
      #  device_number = 3
    if device_number == 4:
        tensor[:512, 16] = 1
        tensor[512:1024, 17] = 1
        tensor[1024:1536, 18] = 1
        tensor[1536:, 19] =1
    if device_number == 5:
        tensor[:512, 20] = 1
        tensor[512:1024, 21] = 1
        tensor[1024:1536, 22] = 1
        tensor[1536:, 23] =1
    gates = tensor
    # gates.device = logits.device
    #hht local gating
    '''

    #capacity = _capacity(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))
    capacity = 0
    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(logits_w_noise if noisy_gate_policy == 'RSample' else gates, dim=1)
    num_experts = int(gates.shape[1])
    # hansol
    if mask is not None:
        reshaped_mask = mask.reshape(-1)

        indices1_s_copy = indices1_s.clone().detach()
        indices1_s_copy[~reshaped_mask] = -1
    else :
        indices1_s_copy = indices1_s.clone().detach()
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # mask only used tokens
    if used_token is not None:
        mask1 = einsum("s,se->se", used_token.squeeze(1).squeeze(1).reshape(-1), mask1)
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # if we don't want to drop any tokens

    '''
    if not drop_tokens:
        new_capacity = torch.max(exp_counts).to(logits.device)
        dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=dist.get_world_group())
        capacity = new_capacity
    '''
    #print("capacity : ", capacity)
    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts

    # Random Token Selection
    if use_rts:
        uniform = exp_selection_uniform_map.get(logits.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform(low=torch.tensor(0.0, device=logits.device),
                                                          high=torch.tensor(1.0, device=logits.device)).rsample
            exp_selection_uniform_map[logits.device] = uniform

        mask1_rand = mask1 * uniform(mask1.shape)
    else:
        mask1_rand = mask1

    assert logits.shape[
        0] >= min_capacity, "No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size."
    '''
    # hansol 여기서 떨궈지는 듯?
    top_idx = _top_idx(mask1_rand, capacity)

    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    mask1 = new_mask1
    '''

    # drop the padding
    # if mask is not None:
    #     mask1[~reshaped_mask] = torch.tensor([0, 0, 0, 0], dtype=torch.long, device=mask1.device)

    if use_tutel:
        # Tutel doesn't support index values masked with zero
        # so we need to replace masked indices with -1
        # print("using tutel ! ")
        indices_mask = mask1.sum(dim=1) * num_experts - 1
        indices1_s = torch.min(indices1_s, indices_mask)
        #if device_number == 0 :
            #print("indices_mask : ", indices_mask, indices_mask.size())
            #print("indices1_s : ", indices1_s, indices1_s.size())

    # Compute locations in capacity buffer
    if use_tutel:
        locations1 = tutel_moe.fast_cumsum_sub_one(mask1)
    else:
        locations1 = torch.cumsum(mask1, dim=0) - 1

    if use_tutel:
        gates1_s = (gates * mask1).sum(dim=1)
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        return l_aux, capacity, num_experts, [
            indices1_s,
        ], [
            locations1_s,
        ], [
            gates1_s,
        ], exp_counts, mask1, indices1_s 
    # hansol

    # Store the capacity location for each token

    locations1_s = torch.sum(locations1 * mask1, dim=1)
    
    mask1_float = mask1.float()
    gates = gates * mask1_float

    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    combine_weights = einsum("se,sc->sec", gates, locations1_sc)
    dispatch_mask = combine_weights.bool()

    
    # hansol
    # return l_aux, combine_weights, dispatch_mask, exp_counts#, counts#, result # hansol
    return l_aux, combine_weights, dispatch_mask, mask1, exp_counts, indices1_s#, counts#, result # hansol


def top2gating(logits: Tensor, capacity_factor: float, min_capacity: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    capacity = _capacity(gates, torch.tensor(capacity_factor * 2), torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # hansol
    # print("mask1 : ", mask1)
    # print("mask2 : ", mask2)
    counts = torch.zeros(1,4).to(torch.int)
    indices = (mask1 == 1).nonzero(as_tuple=True)[1]
    max_index = indices.max().item() + 1
    counts = torch.bincount(indices, minlength=max_index)
    if counts.size(0) < 4:
        padding = torch.zeros(4 - counts.size(0), dtype=counts.dtype).to(indices.device)
        counts = torch.cat((counts, padding), dim=0)
    # print(counts)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = einsum("se,se->s", gates, mask1_float)
    gates2_s = einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    locations2_sc = _one_hot_to_float(locations2_s, capacity)
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts # , counts # hansol


class TopKGate(Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 k: int = 1,
                 capacity_factor: float = 1.0,
                 eval_capacity_factor: float = 1.0,
                 min_capacity: int = 8,
                 noisy_gate_policy: Optional[str] = None,
                 drop_tokens: bool = False,
                 use_rts: bool = True) -> None:
        super().__init__()

        # Only top-1 and top-2 are supported at the moment.
        if k != 1 and k != 2:
            raise ValueError('Only top-1 and top-2 gatings are supported.')
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float()
        self.k = k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.noisy_gate_policy = noisy_gate_policy
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False
        self.gate_time = 0.0
        self.drop_tokens = drop_tokens
        self.use_rts = use_rts

    def forward(self,
                layer_count,
                batch,
                mask,
                seq_len,
                input: torch.Tensor,
                used_token: torch.Tensor = None,
                use_tutel: bool = False) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).start()

        if self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
        input_fp32 = input.float()
        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
        logits = self.wg(input_fp32)

        if self.k == 1:
            gate_output = top1gating(layer_count, mask, logits, batch, seq_len, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, used_token, self.noisy_gate_policy if self.training else None,
                                     self.drop_tokens, self.use_rts, use_tutel)

        else:
            gate_output = top2gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity)

        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).stop()
            self.gate_time = self.timers(TOPK_GATE_TIMER).elapsed(reset=False)

        return gate_output

def split_and_pad(input_data, split_sizes,num_local_experts, d_model):

    result_2d = []
     
    split_indices = torch.cumsum(split_sizes.reshape(-1), dim=0)
    split_indices = torch.cat((torch.tensor([0]), split_indices))
    # 8행 5열의 배열 생성
    for i in range(8):
        start_index = i * num_local_experts  # 각 행의 시작 인덱스
        row = split_indices[start_index:start_index + num_local_experts + 1]  # 5개의 요소 선택
        result_2d.append(row)
    result_tensor = torch.cat(result_2d, dim=0).reshape(-1,num_local_experts+1)
    data_chunks = []
    for i in range(8):
            
        row_chunks = [input_data[start:end] for start, end in zip(result_tensor[i, :-1], result_tensor[i, 1:])]
        data_chunks.extend(row_chunks)
    
    #print(len(data_chunks))
    padded_data = torch.nn.utils.rnn.pad_sequence(data_chunks, batch_first=True, padding_value=0)
    padded_data = padded_data.reshape(8, num_local_experts, padded_data.size(1), d_model)
    return padded_data

    # hansol

def slicing_zero(input_data, split_sizes,num_local_experts, d_model):
    capacity = split_sizes.max().item()
    split_sizes = split_sizes.reshape(-1)
    tmp_index = torch.full((len(split_sizes),), capacity)
    tmp_index = tmp_index-split_sizes
    result_list = []
    current_value = 0

    for i in range(len(split_sizes)*2):
        if i % 2 == 0:
            current_value += split_sizes[int(i/2)]
        else:
            current_value += tmp_index[int(i/2)]
        result_list.append(current_value.item())

    result_tensor = torch.tensor(result_list)
    result_tensor = torch.cat((torch.tensor([0]), result_tensor[:-1]), dim=0).reshape(-1,8)

    data_chunks = []
    for i in range(num_local_experts*2):
        row_chunks = [input_data[start:end] for start, end in zip(result_tensor[i, ::2], result_tensor[i, 1::2])]
        data_chunks.extend(row_chunks)
    data_chunks = torch.cat(data_chunks)

    return data_chunks

def zeropad_output(input_data, split_sizes,num_local_experts, d_model, capacity):
    result_2d = []
    
    split_indices = torch.cumsum(split_sizes, dim=0)
    split_indices = torch.cat((torch.tensor([0]), split_indices))

    for i in range(8):
        start_index = i * num_local_experts  
        row = split_indices[start_index:start_index + num_local_experts + 1]  
        result_2d.append(row)
    result_tensor = torch.cat(result_2d, dim=0).reshape(-1,num_local_experts+1)
    data_chunks = []
    for i in range(8):
        row_chunks = [input_data[start:end] for start, end in zip(result_tensor[i, :-1], result_tensor[i, 1:])]
        data_chunks.extend(row_chunks)
    padded_data = torch.nn.utils.rnn.pad_sequence(data_chunks, batch_first=True, padding_value=0.0)
    padded_data = F.pad(padded_data, (0, 0,0, capacity-padded_data.size(1)),'constant',0)


    return padded_data

class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self,
                 gate: Module,
                 experts: Module,
                 ep_group_name,
                 ep_size,
                 num_local_experts: int,
                 use_tutel: bool = False) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_group = None
        self.ep_size = ep_size
        self.ep_group_name = ep_group_name
        self.num_local_experts = num_local_experts
        self.time_falltoall = 0.0
        self.time_salltoall = 0.0
        self.time_moe = 0.0
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False

        # hansol
        self.mask1 = None
        self.result = None

        self.use_tutel = use_tutel and TUTEL_INSTALLED and gate.k == 1

        if self.use_tutel:
            logger.info('Using Tutel optimizations.')
        elif use_tutel and not TUTEL_INSTALLED:
            logger.warning("Tutel optimization requested but not installed. "
                           "Proceeding without Tutel.")
        elif use_tutel and TUTEL_INSTALLED and gate.k != 1:
            logger.warning("To enable Tutel optimization, use top-1 instead of top-2 gate. "
                           "Proceeding without Tutel.")

        self.count = 0
    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group


    def forward(self, layer_count, mask, batch, new_centroids, *input: Tensor, **kwargs: Any) -> Tensor:

        if self.wall_clock_breakdown:
            self.timers(MOE_TIMER).start()

        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = input[0].reshape(-1, d_model)
        if self.use_tutel:
            # hansol
            # self.l_aux, C, E, indices_, locations_, gates_, self.exp_counts = self.gate(batch, reshaped_input, input[1], True)
            self.l_aux, C, E, indices_, locations_, gates_, self.exp_counts, self.mask1, self.result = self.gate(layer_count, batch, mask, input[0].size(0), reshaped_input, input[1], True)

            #######################################################################
            #num_local_experts = 2
            num_experts = 32 
            first_column = new_centroids[:,layer_count//3]
            #print("first_column : ", first_column)
            E = E * 2 
            padding_first_column = torch.full((first_column.size(0) // self.num_local_experts, self.num_local_experts), -1).to(new_centroids.device)
            reshaped_first_column = first_column.reshape(-1,self.num_local_experts).to(new_centroids.device)
            result_first_column = torch.cat((reshaped_first_column, padding_first_column), dim=1).reshape(-1,self.num_local_experts*2).to(new_centroids.device)
            full_range = torch.arange(self.num_local_experts*8).to(new_centroids.device)
            unpopular_expert = full_range[~torch.isin(full_range, first_column)]
            #print("no expert in first_column : ", unpopular_expert)

            indices = unpopular_expert // self.num_local_experts 
            indices.to(new_centroids.device)
            positions = (unpopular_expert % self.num_local_experts) + self.num_local_experts
            positions.to(new_centroids.device)
            
            result_first_column[indices, positions] = unpopular_expert
            final_first_column = result_first_column.reshape(-1)

            #print("final_first_column : ", final_first_column)


            indices_for_centroids = torch.full((len(indices_[0]),), -1, dtype=torch.long).to(new_centroids.device)
            matches = torch.nonzero(indices_[0].unsqueeze(1)==final_first_column, as_tuple=False).to(new_centroids.device)

            unique_values, counts = torch.unique(matches[:, 0], return_counts=True)
            duplicated_values = unique_values[counts > 1]
            non_duplicated_values = unique_values[counts == 1]
            duplicated_mask = torch.isin(matches[:, 0], duplicated_values)
            local_matches = matches[duplicated_mask][:,1]//(self.num_local_experts*2)
            values_to_remove = matches[duplicated_mask][local_matches == dist.get_rank()]
            mask_for_indices = ~torch.isin(matches[duplicated_mask][:, 0], values_to_remove[:,0])
            filtered_tensor = matches[duplicated_mask][mask_for_indices]
            non_duplicated_mask = torch.isin(matches[:, 0], non_duplicated_values)
            filtered_non_duplicated = matches[non_duplicated_mask]
            random_indices = torch.randperm(values_to_remove.size(0))
            shuffled_values_to_remove = values_to_remove[random_indices]
            result_tensor = torch.cat((shuffled_values_to_remove, filtered_tensor, filtered_non_duplicated), dim=0)
            if result_tensor.size(0) > 0 :
                indices_for_centroids[result_tensor[:,0]] = result_tensor[:, 1]

            #print("indices_for_centroids : ", indices_for_centroids, layer_count//3, dist.get_rank())
            #print("reshaped intput : ", reshaped_input.size())
            mask_centroids = F.one_hot(indices_for_centroids, num_classes=self.num_local_experts*8*2)

            self.exp_counts = torch.sum(mask_centroids, dim=0).detach().to('cpu')


            ##############################################################

            '''
            centroid_values = set(new_centroids[:,layer_count//3].flatten().tolist())
            unpopular_experts = set(range(32)) - centroid_values
            unpopular_experts = sorted(list(unpopular_experts))
            unpopular_experts_local = list(filter(lambda x: (dist.get_rank()*4) <= x <= (dist.get_rank()*4+3),unpopular_experts))
            '''


            ##############################################################

            with nvtx.annotate("All gather", color="red"):
                device_number = dist.get_rank()
                exp_counts_gpu = self.exp_counts.to(reshaped_input.device).detach().clone()
                gathered_lists = [torch.zeros_like(exp_counts_gpu, device=reshaped_input.device) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_lists, exp_counts_gpu)

            combined_tensor = torch.cat(gathered_lists)
            C = torch.max(combined_tensor)

            S, M = reshaped_input.size(0), reshaped_input.size(1)


            #############################################################
            with nvtx.annotate("dispatcher", color="blue"):
                if not hasattr(self, '_tutel_dispatcher'):
                    #print("fast_dispatcher ")
                    self._tutel_dispatcher = tutel_moe.fast_dispatcher(E, C, M, dispatch_dtype=reshaped_input.dtype)
                # hansol
                #self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
                self._tutel_dispatcher.update([indices_for_centroids], locations_, gates_, capacity=C)
                dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
        else:
            # hansol
            # self.l_aux, combine_weights, dispatch_mask, self.exp_counts, counts, expert_layer = self.gate(layer_count, batch, mask, input[0].size(0), reshaped_input, input[1]) #hansol
            # self.l_aux, combine_weights, dispatch_mask, self.exp_counts, counts  = self.gate(layer_count, batch, mask, input[0].size(0), reshaped_input, input[1])

            self.l_aux, combine_weights, dispatch_mask, self.mask1, self.exp_counts, self.result  = self.gate(layer_count, batch, mask, input[0].size(0), reshaped_input, input[1])
            dispatched_input = einsum("sec,sm->ecm", dispatch_mask.type_as(input[0]), reshaped_input)



        # print("mask : ", dispatch_mask.size())
        if self.wall_clock_breakdown:
            self.timers(FIRST_ALLTOALL_TIMER).start()
        # print("dispatch ", dispatched_input.size())
        if groups._get_expert_model_parallel_world_size() == 1:
            # If the non-expert is tensor-parallel, it will create
            # duplicate tokens on the tensor-parallel ranks.
            # Since our experts are not tensor-parallel, these duplicates
            # need to be dropped to ensure correctness.
            # this also doubles up as a communication optimization as we are
            # reducing the all-to-all communication volume.
            dispatched_input = drop_tokens(dispatched_input, dim=1)
        #start = time.time()
        
        # hansol
        with nvtx.annotate("Computation1", color="white"):
            non_zero_dispatched_input = slicing_zero(dispatched_input.reshape(-1,d_model), self.exp_counts, self.num_local_experts*2, d_model).to(dispatched_input.device).clone()

            ################## change 4 to 8 
            #non_zero_dispatched_input = dispatched_input[~(dispatched_input == 0).all(axis=1)].clone()
            summed_tensor = self.exp_counts.view(8, self.num_local_experts*2).sum(dim=1).clone()
        with nvtx.annotate("output_lists", color="white"):
            output_lists = [tensor[device_number*self.num_local_experts*2:device_number*self.num_local_experts*2+self.num_local_experts*2].tolist() for tensor in gathered_lists]
        with nvtx.annotate("input_slices", color="white"):
            input_slices = summed_tensor.tolist()
        with nvtx.annotate("output_tensor", color="white"):
            output_tensor = torch.tensor(output_lists).clone()
        with nvtx.annotate("output_tensor_sum", color="white"):
            output_tensor_sum = output_tensor.sum(dim=1).clone()
        with nvtx.annotate("non_zero_output", color="white"):
            non_zero_output = torch.empty((int(output_tensor_sum.sum(dim=0)),768),device = dispatched_input.device).clone()
        with nvtx.annotate("output_slices", color="white"):
            output_slices = output_tensor_sum.tolist()
         
        with nvtx.annotate("AllToAll", color="purple"):

            #print("before  : ", input_slices, output_slices, non_zero_dispatched_input.size(), non_zero_output.size(), dist.get_rank())
            dispatched_input = _AllToAll_MOE.apply(self.ep_group, input_slices, output_slices, non_zero_dispatched_input,non_zero_output)
            #dispatched_input = _AllToAll.apply(self.ep_group, dispatched_input)
        # hansol
        #print("output_tensor_sum : ", output_tensor_sum, layer_count//3, dist.get_rank())
        #print("gathered_lists : ", gathered_lists)
        #print("dispatched_input : ", dispatched_input[:,0], dispatched_input.size())
        #end = time.time()
        #print("first AllToAll : ",end-start, dispatched_input.device)
        
        if self.wall_clock_breakdown:
            self.timers(FIRST_ALLTOALL_TIMER).stop()
            self.time_falltoall = self.timers(FIRST_ALLTOALL_TIMER).elapsed(reset=False)
        # hansol
        #dispatched_input = split_and_pad(dispatched_input, output_tensor)
        # hansol
        # Re-shape after all-to-all: ecm -> gecm
        
        dispatched_input = split_and_pad(dispatched_input,output_tensor,self.num_local_experts*2,d_model).to(dispatched_input.device)


        #dispatched_input = dispatched_input.reshape(self.ep_size, self.num_local_experts, -1, d_model)
        #print("dispatched_input after resphaped : ", dispatched_input[:,:,:,0], dispatched_input.size())
        with nvtx.annotate("Expert", color="blue"):
            expert_output = self.experts(output_tensor ,dispatched_input)
        with nvtx.annotate("Expert1", color="blue"):
            expert_output = slicing_zero(expert_output.reshape(-1,d_model), output_tensor, self.num_local_experts*2, d_model).to(expert_output.device).clone()
        if self.wall_clock_breakdown:
            self.timers(SECOND_ALLTOALL_TIMER).start()

        
        with nvtx.annotate("AllToAll2", color="red"):
            '''
            print("output_slices : ", output_slices, layer_count//3, dist.get_rank())
            print("input_slices : ", input_slices, layer_count//3, dist.get_rank())
            print("indices : ", indices_[0], layer_count//3, dist.get_rank())
            print("indices_for_centroids : ", indices_for_centroids, layer_count//3, dist.get_rank())
            '''
            #print("expert_output : ", expert_output[:,0], layer_count//3, dist.get_rank())
            #print("non_zero_dispatched_input : ", non_zero_dispatched_input[:,0], layer_count//3, dist.get_rank())
            expert_output = _AllToAll_MOE.apply(self.ep_group, output_slices, input_slices, expert_output, torch.empty_like(non_zero_dispatched_input))
            
        if self.wall_clock_breakdown:
            self.timers(SECOND_ALLTOALL_TIMER).stop()
            self.time_salltoall = self.timers(SECOND_ALLTOALL_TIMER).elapsed(reset=False)
        with nvtx.annotate("zeropad", color="white"): 
            combined_output = zeropad_output(expert_output, self.exp_counts,self.num_local_experts*2 ,d_model, C)
        # Re-shape back: gecm -> ecm
        #expert_output = expert_output.reshape(self.ep_size * self.num_local_experts, -1, d_model)
        if groups._get_expert_model_parallel_world_size() == 1:
            # the dropped duplicate tokens need to be gathered on each
            # tensor parallel rank again for the tensor-parallel
            # non-expert of the next layer.
            expert_output = gather_tokens(expert_output, dim=1)
        '''
        if self.use_tutel:
            combined_output = self._tutel_dispatcher.decode(expert_output.view(E * C, M))
        else:
            combined_output = einsum("sec,ecm->sm", combine_weights.type_as(input[0]), expert_output)
        
        a = combined_output.reshape(input[0].shape)
        '''
        with nvtx.annotate("zeropad2", color="white"):
            combined_output = self._tutel_dispatcher.decode(combined_output.view(E*C,M))
        # hansol here
        a = combined_output.reshape(input[0].shape)
        if self.wall_clock_breakdown:
            self.timers(MOE_TIMER).stop()
            self.time_moe = self.timers(MOE_TIMER).elapsed(reset=False)
        # hansol
        #counts, expert_layer
        # , dispatch_mask, mask, self.exp_counts
        return a
        # hansol
