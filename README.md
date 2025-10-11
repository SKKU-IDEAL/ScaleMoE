## ScaleMoE
**The framework for the paper "ScaleMoE: A Fast and Scalable Distributed Training Framework for Large-Scale Mixture-of-Experts Models" in PACT 2025.**

We will continue updating the framework codes and annotations in the future.

This repository is based on the [DeepSpeed training framework](https://github.com/microsoft/DeepSpeed).  
Please refer to the official Microsoft DeepSpeed library to understand the fundamentals of DeepSpeed.

## Overview

• **All-to-all communication optimization.** We propose *adaptive all-to-all communication* to minimize communication volume by removing unnecessary zero padding.

• **Balanced expert selection.** We propose *dynamic expert clustering*, facilitating more balanced expert selection.

• **Heterogeneous network-aware data placement.** We propose *topology-aware expert remapping* to fully leverage any type of network configuration.
