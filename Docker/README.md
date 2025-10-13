# Dockerfiles for CUDA-based Experimental Environments

This directory contains two Dockerfiles used to build experimental Docker images targeting different CUDA versions:

- `Dockerfile.cuda11`: For environments based on CUDA 11.x
- `Dockerfile.cuda12`: For environments based on CUDA 12.x

These Dockerfiles are designed to create consistent and reproducible containers for running GPU-accelerated experiments, with dependencies tailored to each CUDA version.

## How to Build

To build the Docker images, use the following commands:

```bash
# For CUDA 11
docker build -f Dockerfile_cuda11.7 -t my-experiment-cuda11 .

# For CUDA 12
docker build -f Dockerfile_cuda12.2 -t my-experiment-cuda12 .
