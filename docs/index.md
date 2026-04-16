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
      link: /zh/

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
#include "flash_attention.h"

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

- [English Guide](/en/) - Complete documentation in English
- [中文文档](/zh/) - 简体中文文档
- [API Reference](/en/api/) - Detailed API documentation
- [Algorithm](/en/algorithm) - FlashAttention deep dive
