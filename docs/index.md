---
layout: home

hero:
  name: "CuFlash-Attn"
  text: "High-Performance CUDA FlashAttention"
  tagline: A from-scratch implementation with O(N) memory, FP32/FP16 support, and full training capabilities
  image:
    src: /hero-logo.svg
    alt: CuFlash-Attn
  actions:
    - theme: brand
      text: Get Started
      link: /en/guide/quick-start
    - theme: alt
      text: View on GitHub
      link: https://github.com/LessUp/cuflash-attn
    - theme: alt
      text: 中文文档
      link: https://lessup.github.io/cuflash-attn/zh/

features:
  - icon: ⚡
    title: O(N) Memory Complexity
    details: Linear memory usage instead of quadratic. Handles sequences up to 16K+ efficiently.
  - icon: 🔢
    title: FP32 & FP16 Support
    details: Full precision control with FP32 accumulation for FP16 operations. Numerically stable.
  - icon: 🔁
    title: Forward & Backward
    details: Complete training support with optimized backward pass using recomputation strategy.
  - icon: 🎭
    title: Causal Masking
    details: Built-in efficient causal attention for autoregressive models like GPT.
  - icon: 🚀
    title: Multi-Architecture
    details: Optimized for NVIDIA GPUs from V100 (sm_70) to H100 (sm_90).
  - icon: 🔧
    title: Easy Integration
    details: Clean C++ API with C ABI for Python ctypes. Header-only optional.
---

<style>
.VPHero .name {
  background: linear-gradient(135deg, #3f83f8 0%, #60a5fa 50%, #a78bfa 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.VPHero .tagline {
  max-width: 600px;
  margin: 1rem auto;
}

:root {
  --vp-home-hero-name-color: transparent;
  --vp-home-hero-name-background: linear-gradient(135deg, #3f83f8 0%, #60a5fa 100%);
  --vp-home-hero-image-background-image: linear-gradient(135deg, #3f83f8 0%, #a78bfa 100%);
  --vp-home-hero-image-filter: blur(40px);
}
</style>

## Quick Start

```bash
# Clone the repository
git clone https://github.com/LessUp/cuflash-attn.git
cd cuflash-attn

# Build with CMake preset
cmake --preset release
cmake --build --preset release

# Run tests
ctest --preset release --output-on-failure
```

## Usage Example

```cpp
#include "cuflash/flash_attention.h"

// Forward pass with causal masking
auto err = cuflash::flash_attention_forward(
    d_Q, d_K, d_V, d_O, d_L,
    batch_size, num_heads, seq_len, head_dim,
    scale,      // 1.0f / sqrt(head_dim)
    true,       // causal
    stream      // CUDA stream
);
```

## Performance

| Sequence Length | Memory (Standard) | Memory (FlashAttention) | Savings |
|----------------|-------------------|------------------------|---------|
| 1,024 | 4 MB | 8 KB | 99.8% |
| 4,096 | 64 MB | 32 KB | 99.95% |
| 16,384 | 1 GB | 128 KB | 99.99% |

## Documentation

| Resource | Description |
|----------|-------------|
| [Quick Start](/en/guide/quick-start) | Get up and running in 5 minutes |
| [API Reference](/en/api-reference) | Complete C++ and C ABI API documentation |
| [Algorithm Deep Dive](/en/algorithm) | Understanding FlashAttention |
| [Building from Source](/en/building) | Detailed build instructions |
| [Troubleshooting](/en/troubleshooting) | Common issues and solutions |
| [中文文档](/zh/) | Chinese documentation |

## Specifications

This project follows **Spec-Driven Development (SDD)**. All implementation details are documented in `/specs/`:

| Document | Description |
|----------|-------------|
| [Product Requirements](https://github.com/LessUp/cuflash-attn/blob/master/specs/product/001-flash-attention-core.md) | Feature definitions and acceptance criteria |
| [Architecture RFC](https://github.com/LessUp/cuflash-attn/blob/master/specs/rfc/001-core-architecture.md) | Technical design and architecture |
| [API Specification](https://github.com/LessUp/cuflash-attn/blob/master/specs/api/001-public-api.md) | Public API definitions |
| [Testing Specification](https://github.com/LessUp/cuflash-attn/blob/master/specs/testing/001-test-specification.md) | Testing strategy and requirements |
