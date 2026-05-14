# Project Status

CuFlash-Attn is maintained as a **stable v0.3.0 baseline** and an **archive-ready reference implementation**. The current work focuses on governance cleanup, documentation quality, workflow simplification, and bug fixes rather than feature expansion.

## What this project covers

- From-scratch CUDA C++ implementation of FlashAttention
- Forward and backward passes for `float` and `half`
- Supported `head_dim` values: `32`, `64`, `128`
- Public C++ API plus C ABI for Python `ctypes` integration
- OpenSpec-tracked design and verification rules

## Maintenance posture

This repository is intentionally optimized for:

- **clarity over breadth**: no speculative feature growth
- **stable integration surface**: examples, docs, and ABI stay aligned
- **lightweight engineering**: preset-based builds, focused CI, bilingual docs
- **handoff readiness**: contributors and follow-up models can continue from explicit specs and control docs

## Development workflow

The preferred workflow is:

1. Read the relevant files in `openspec/specs/`
2. Capture changes through an OpenSpec change when scope or behavior shifts
3. Build with CMake presets only
4. Run verification appropriate to the environment
5. Use review before landing non-trivial changes

## Validation boundaries

- Local CUDA builds require a working toolkit and `nvcc`
- GPU tests are skipped automatically on systems without a CUDA device
- Documentation and workflow cleanup can be validated without a GPU

## Canonical references

- [Quick Start](/en/guide/quick-start)
- [Building from Source](/en/building)
- [API Reference](/en/api-reference)
- [Troubleshooting](/en/troubleshooting)
- [CHANGELOG.md](https://github.com/AICL-Lab/cuflash-attn/blob/master/CHANGELOG.md)
- [OpenSpec Specifications](https://github.com/AICL-Lab/cuflash-attn/tree/master/openspec/specs)
