---
layout: home
title: Documentation

hero:
  name: "CuFlash-Attn"
  text: "High-Performance CUDA FlashAttention"
  tagline: A from-scratch implementation with O(N) memory complexity, FP32/FP16 dual precision, and complete training support
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

features:
  - icon: ⚡
    title: O(N) Memory
    details: Linear memory complexity instead of quadratic. Handle sequences up to 16K+ efficiently without OOM.
  - icon: 🔢
    title: FP32 & FP16
    details: Full precision control with FP32 accumulation for FP16 operations. Numerically stable forward and backward.
  - icon: 🔁
    title: Training Ready
    details: Complete forward and backward passes with optimized gradient computation using recomputation strategy.
  - icon: 🎭
    title: Causal Masking
    details: Built-in efficient causal attention support for autoregressive models like GPT and LLaMA.
  - icon: 🚀
    title: Multi-Architecture
    details: Optimized CUDA kernels for NVIDIA GPUs from V100 (sm_70) to H100 (sm_90).
  - icon: 🔧
    title: Easy Integration
    details: Clean C++ API with C ABI for Python ctypes. Header-only option available for simple use cases.
---

<script setup>
import { onMounted } from 'vue'

onMounted(() => {
  localStorage.setItem('preferred-lang', 'en')
})
</script>

<style>
:root {
  --vp-home-hero-name-color: transparent;
  --vp-home-hero-name-background: linear-gradient(135deg, #3f83f8 0%, #60a5fa 50%, #a78bfa 100%);
  --vp-home-hero-image-background-image: linear-gradient(135deg, #3f83f8 20%, #a78bfa 80%);
  --vp-home-hero-image-filter: blur(50px);
}

.VPHero .name {
  font-size: 3.5rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.02em;
}

.VPHero .text {
  font-size: 1.75rem !important;
  font-weight: 600 !important;
  color: var(--vp-c-text-2) !important;
}

.VPHero .tagline {
  font-size: 1.125rem !important;
  color: var(--vp-c-text-3) !important;
  max-width: 560px;
}
</style>

## Quick Start

Get CuFlash-Attn up and running in under 5 minutes:

::: code-group

```bash [Clone & Build]
# Clone the repository
git clone https://github.com/LessUp/cuflash-attn.git
cd cuflash-attn

# Build with CMake preset
cmake --preset release
cmake --build --preset release

# Run tests
ctest --preset release --output-on-failure
```

```cpp [Basic Usage]
#include "cuflash/flash_attention.h"

// Forward pass with causal masking
auto err = cuflash::flash_attention_forward(
    d_Q, d_K, d_V, d_O, d_L,
    batch_size,    // B
    num_heads,     // H  
    seq_len,       // N
    head_dim,      // D
    scale,         // 1.0f / sqrt(head_dim)
    true,          // causal
    stream         // CUDA stream
);
```

:::

## Performance

Memory efficiency comparison between Standard Attention and FlashAttention:

| Sequence Length | Standard Attention | FlashAttention | **Savings** |
|----------------|-------------------|----------------|-------------|
| 1,024 | 4 MB | 8 KB | **99.8%** |
| 4,096 | 64 MB | 32 KB | **99.95%** |
| 16,384 | 1 GB | 128 KB | **99.99%** |

## Documentation

| Resource | Description |
|----------|-------------|
| [Quick Start Guide](/en/guide/quick-start) | Get up and running in 5 minutes |
| [Building from Source](/en/building) | Detailed build instructions with all options |
| [API Reference](/en/api-reference) | Complete C++ and C ABI documentation |
| [Algorithm Deep Dive](/en/algorithm) | Understanding FlashAttention internals |
| [Troubleshooting](/en/troubleshooting) | Common issues and solutions |

## Specifications

This project follows **OpenSpec** methodology. All implementation details are in `openspec/specs/`:

- [Design Specification](https://github.com/LessUp/cuflash-attn/blob/master/openspec/specs/design/flash-attention-design.md) — Requirements & algorithms
- [Verification Specification](https://github.com/LessUp/cuflash-attn/blob/master/openspec/specs/verification/flash-attention-verification.md) — API & test specs

## GPU Support

| Architecture | Compute | Example GPUs | Status |
|--------------|---------|--------------|--------|
| Volta | sm_70 | V100 | ✅ Supported |
| Turing | sm_75 | RTX 2080 Ti | ✅ Supported |
| Ampere | sm_80, sm_86 | A100, RTX 3090 | ✅ Supported |
| Ada Lovelace | sm_89 | RTX 4090 | ✅ Supported |
| Hopper | sm_90 | H100 | ✅ Supported |
