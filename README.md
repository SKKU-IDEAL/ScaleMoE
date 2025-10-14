# ScaleMoE
**The framework for the paper "ScaleMoE: A Fast and Scalable Distributed Training Framework for Large-Scale Mixture-of-Experts Models" in PACT 2025.**

We will continue updating the framework codes and annotations in the future.

This repository is based on the [DeepSpeed training framework](https://github.com/microsoft/DeepSpeed) and the [Tutel MoE](https://github.com/microsoft/tutel).  
Please refer to the official Microsoft DeepSpeed and Tutel libraries to understand the fundamentals of DeepSpeed and Tutel.

# Overview

• **All-to-all communication optimization.** We propose *adaptive all-to-all communication* to minimize communication volume by removing unnecessary zero padding.

• **Balanced expert selection.** We propose *dynamic expert clustering*, facilitating more balanced expert selection.

• **Heterogeneous network-aware data placement.** We propose *topology-aware expert remapping* to fully leverage any type of network configuration.

# How to use

### Environment setting
To use ScaleMoE, users need to install DeepSpeed and Tutel first.

For a quick and consistent setup, we provide a Dockerfile that automatically builds a container including all required dependencies (e.g., CUDA, PyTorch, DeepSpeed, and Tutel).
Users who prefer an easy setup can simply build the Docker image and start a container to run ScaleMoE without manually configuring the environment.

Please refer to the docker/ directory for detailed instructions and preconfigured files.

```bash
# Build the docker image
docker build -t scalemoe:latest -f docker/Dockerfile_cuda12.2 .

# Run the container
docker run --gpus all -it --rm scalemoe:latest
```

If you want to use your own environment, please install DeepSpeed and Tutel.

```bash
#Install deepspeed
pip install deepspeed

#Install tutel
git clone https://github.com/microsoft/tutel --branch main
python3 -m pip uninstall tutel -y
python3 ./tutel/setup.py install --user
```

### MoE block Usage
See folder scalemoe for details
- **`baseline`**  
  Runs the **baseline MoE implementation**.  
  Use this script if you want to reproduce the standard Tutel as baseline behavior with minimal modifications.
  Use different scripts depending on the CUDA version selected when creating Docker.
  
- **`run_adaptive.sh`**  
  Runs the **Adaptive MoE implementation**.  
  This script introduces 

- **`run_scalable.sh`**  
  Runs the **K-means and GA MoE implementation**.  
  This script introduces
   
#### Usage
Run the desired script with your configuration. For example:
```bash
bash scalemoe/scripts/baseline/run_cuda11.sh
bash scalemoe/scripts/run_adaptive.sh
bash scalemoe/scripts/run_scalable.sh
```

### Examples
ScaleMoE provides out-of-the-box support for running large-scale Mixture-of-Experts (MoE) models such as **BERT-MoE** and **GPT-MoE**. 
You can find example training scripts and configuration files under the [`models/`](./models) directory.
We provide a MoE-enhanced BERT implementation based on DeepSpeed and Tutel. 
This example demonstrates how ScaleMoE optimizations—_such as **adaptive all-to-all communication**, **dynamic expert clustering**, and **topology-aware expert remapping**_—can significantly accelerate BERT training on large-scale distributed environments.
#### **BERT**: Bidirectional Encoder Representations from Transformers
To run the **BERT-MoE** example:
- Prepare data
  ```bash
  bash models/BERT/data/getdata.sh
  ```
- Prepare checkpoints
  ```bash
  bash models/BERT/prepare_ck.sh
  ```
- Run baseline
  ```bash
  bash models/BERT/tutel_run.sh
  ```
- Run **adaptive all-to-all communication**
  ```bash
  bash models/BERT/adaptive_run.sh
  ```
- Run **dynamic expert clustering**
  ```bash
  bash models/BERT/kmean_run.sh
  ```
- Run **topology-aware expert remapping**
  ```bash
  bash models/BERT/ga_run.sh
  ```
Wait for the execution to complete, and you will obtain the logs and the exploration results in the logs folder like [`models/BERT/logs/`](./models/BERT/logs).

# Publication
This work has been published in [PACT'25](https://pact2025.github.io/program/).
## Citations
```bibtex
@inproceedings{
}
```
