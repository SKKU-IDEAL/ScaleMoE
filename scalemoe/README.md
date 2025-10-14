This repository is based on the [DeepSpeed training framework](https://github.com/microsoft/DeepSpeed).  
Please refer to the official Microsoft DeepSpeed library to understand the fundamentals of DeepSpeed.

## Overview
This repository primarily modifies the **Mixture-of-Experts (MoE)** component within DeepSpeed.  
Each subdirectory contains different scripts for specific experiments or modifications.

## Usage
1. Before starting experiments, **make sure that shuffling in the PyTorch data iterator is disabled**.
   - Example:
     ```python
     train_loader = torch.utils.data.DataLoader(dataset, shuffle=False, ...)
     ```
2. Follow the instructions to run the corresponding scripts.
   - 1dh&2dh
   - baseline
   - adaptive
   - k-means&ga
   

## Notes
- The core DeepSpeed features remain unchanged; only the MoE-related parts are modified.
- For general usage, refer to the original DeepSpeed documentation.

