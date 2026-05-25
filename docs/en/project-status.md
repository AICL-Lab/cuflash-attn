# Project Status

CuFlash-Attn is maintained as a **stable v0.3.0 reference implementation** for learning, auditing, and lightweight integration.

## What stays in scope

- CUDA C++ FlashAttention from scratch
- Forward and backward passes for `float` and `half`
- Supported `head_dim` values: `32`, `64`, `128`
- Public C++ API and C ABI for `ctypes`-style integration
- Bilingual technical documentation and GitHub Pages publishing

## Maintenance posture

This repository now prefers:

1. **clarity over process**
2. **deletion over framework sprawl**
3. **stable behavior over speculative features**
4. **one canonical source per topic**

## Canonical references

- [Quick Start](/en/guide/quick-start)
- [API Reference](/en/api-reference)
- [Architecture](/en/architecture)
- [Repository README](https://github.com/AICL-Lab/cuflash-attn/blob/master/README.md)
- [CHANGELOG.md](https://github.com/AICL-Lab/cuflash-attn/blob/master/CHANGELOG.md)

## Validation boundaries

- Full CUDA builds require a working toolkit and `nvcc`
- GPU tests may be unavailable on documentation-only environments
- Docs and workflow cleanup should still keep the docs build and repository layout coherent
