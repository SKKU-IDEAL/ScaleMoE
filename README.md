## ScaleMoE
**The framework for the paper "ScaleMoE: A Fast and Scalable Distributed Training Framework for Large-Scale Mixture-of-Experts Models" in PACT 2025.**

We will continue updating the framework codes and annotations in the future.

This repository is based on the [DeepSpeed training framework](https://github.com/microsoft/DeepSpeed).  
Please refer to the official Microsoft DeepSpeed library to understand the fundamentals of DeepSpeed.

## Overview

• **All-to-all communication optimization.** We propose *adaptive all-to-all communication* to minimize communication volume by removing unnecessary zero padding.

• **Balanced expert selection.** We propose *dynamic expert clustering*, facilitating more balanced expert selection.

• **Heterogeneous network-aware data placement.** We propose *topology-aware expert remapping* to fully leverage any type of network configuration.

## How to use

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

