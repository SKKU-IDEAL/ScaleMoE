# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
from typing import List, Optional

import torch
from torch import nn
from deepspeed import comm as dist


class Experts(nn.Module):

    def __init__(self, expert: nn.Module, num_local_experts: int = 1, expert_group_name: Optional[str] = None) -> None:
        super(Experts, self).__init__()

        self.deepspeed_experts = nn.ModuleList([copy.deepcopy(expert) for _ in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        self.group_name = expert_group_name

        ##################### hansol
        self.unpopular_list = []
        ##################### hansol

        # TODO: revisit allreduce for moe.gate...
        for expert in self.deepspeed_experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for param in expert.parameters():
                param.allreduce = False
                param.group_name = expert_group_name

    def make_unpopular_experts(self, unpopular_experts_local, hidden, layer_num):
        for i,num in enumerate(unpopular_experts_local):
            self.deepspeed_experts.append(nn.Sequential(nn.Linear(hidden, 4 * hidden, bias=False),
                nn.GELU(),
                nn.Dropout(0),
                nn.Linear(4 * hidden, hidden, bias=False).to('cpu')))
            param_name_0 = f'unpopular_layer{layer_num}_expert{num.item()}_0'
            param_name_3 = f'unpopular_layer{layer_num}_expert{num.item()}_3'
            unpopular_weight_0 = getattr(self, param_name_0)
            unpopular_weight_3 = getattr(self, param_name_3)
            self.deepspeed_experts[i+self.num_local_experts][0].weight = unpopular_weight_0
            self.deepspeed_experts[i+self.num_local_experts][3].weight = unpopular_weight_3

        
            for param in self.deepspeed_experts[i+self.num_local_experts].parameters():
                param.allreduce = False
                param.group_name = self.group_name

    def forward(self, output_tensor, inputs: torch.Tensor) -> torch.Tensor:
        # hansol chunks = inputs.chunk(self.num_local_experts, dim=1)
        # for chunk, expert in zip(chunks, self.deepspeed_experts[:4]):
        chunks = inputs[:,:self.num_local_experts,:,:].chunk(self.num_local_experts, dim=1)
        expert_outputs: List[torch.Tensor] = []
        for chunk, expert in zip(chunks, self.deepspeed_experts[:self.num_local_experts]):
            out = expert(chunk)
            if isinstance(out, tuple):
                out = out[0]  # Ignore the bias term for now
            expert_outputs += [out]

        expert_outputs = torch.cat(expert_outputs, dim=1)
        # hansol
        output_tensor_sum = torch.sum(output_tensor[:,self.num_local_experts:], dim=0)
        indices = torch.nonzero(output_tensor_sum, as_tuple=True)[0] 
        expert_output_unpopular = torch.zeros_like(inputs[:,self.num_local_experts:,:,:])
        if indices.size(0) > 0:
            for i, index in enumerate(indices) :
                output_for_unpopular = self.deepspeed_experts[i+self.num_local_experts](inputs[:,index+self.num_local_experts,:,:])
                expert_output_unpopular[:,index,:,:] = output_for_unpopular


        return torch.cat((expert_outputs,expert_output_unpopular), dim=1)
