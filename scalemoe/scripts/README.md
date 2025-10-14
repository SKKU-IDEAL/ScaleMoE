# Scripts for MoE Experiments

This folder contains the main entry-point scripts for running different Mixture-of-Experts (MoE) training strategies.  
Follow the instructions below to choose the appropriate script.

## Available Scripts

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

## Usage
Run the desired script with your configuration. For example:
```bash
bash baseline/run_cuda11.sh
bash run_adaptive.sh
bash run_scalable.sh
