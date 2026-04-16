# CuFlash-Attn Documentation

Welcome to the CuFlash-Attn documentation — a high-performance CUDA C++ implementation of FlashAttention.

## What is CuFlash-Attn?

CuFlash-Attn is a from-scratch implementation of the FlashAttention algorithm, designed for:

- **Educational purposes** - Learn how FlashAttention works internally
- **Research** - Modify and experiment with attention mechanisms
- **Production** - Clean API for integration into larger systems

## Key Features

::: tip O(N) Memory Complexity
FlashAttention reduces memory usage from O(N²) to O(N), enabling longer sequences on the same hardware.
:::

::: info FP16 & FP32 Support
Both forward and backward passes support half-precision (FP16) and single-precision (FP32) computations.
:::

::: warning CUDA Required
A CUDA-capable GPU with Compute Capability 7.0+ is required (V100 or newer recommended).
:::

## Navigation

- [Quick Start Guide](/en/guide/quick-start) - Get up and running in 5 minutes
- [Building from Source](/en/guide/building) - Detailed build instructions
- [API Reference](/en/api/) - Complete API documentation
- [Algorithm Deep Dive](/en/algorithm) - Understanding FlashAttention
- [Troubleshooting](/en/troubleshooting) - Common issues and solutions

## Links

- [GitHub Repository](https://github.com/LessUp/cuflash-attn)
- [Release Notes](https://github.com/LessUp/cuflash-attn/releases)
- [中文文档](/zh/)
