---
layout: home
title: Documentation

hero:
  name: "CuFlash-Attn"
  text: "CUDA FlashAttention Reference"
  tagline: O(N) memory • FP32/FP16 • Forward/Backward • Archive-ready v0.3.0
  image:
    src: /hero-logo.svg
    alt: CuFlash-Attn
  actions:
    - theme: brand
      text: Get Started →
      link: /en/guide/quick-start
    - theme: alt
      text: View on GitHub
      link: https://github.com/LessUp/cuflash-attn

features:
  - icon: ⚡
    title: Linear Memory
    details: Handle 16K+ token sequences with O(N) memory via FlashAttention tiling — 99.9% less than standard attention.
  - icon: 🎯
    title: Reference Quality
    details: Clean, educational CUDA C++ implementation. No framework dependencies. Easy to understand, modify, and integrate.
  - icon: 🔢
    title: Full Precision Support
    details: FP32 and FP16 with numerically-aware accumulation. Forward and backward passes for complete training support.
  - icon: 🎭
    title: Causal Masking
    details: Built-in support for autoregressive models. Enable with a single boolean flag in the API.
  - icon: 🚀
    title: Multi-GPU Architecture
    details: Optimized kernels for V100 through H100 (sm_70 → sm_90). Production-ready CUDA performance.
  - icon: 📦
    title: Python Ready
    details: C ABI bindings for ctypes integration. Works with PyTorch, NumPy, or raw GPU memory pointers.
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

## Why CuFlash-Attn?

::: tip Choose this library when
You want to **understand** FlashAttention internals, **experiment** with attention mechanisms, or **integrate** without heavy framework dependencies.
:::

### Quick Comparison

| Feature | CuFlash-Attn | PyTorch SDPA | FlashAttention-2 |
|---------|:------------:|:------------:|:----------------:|
| Educational code | ✅ | ❌ | ⚠️ |
| No dependencies | ✅ | ❌ PyTorch | ❌ |
| Python binding | ✅ ctypes | ✅ native | ✅ |
| Training support | ✅ | ✅ | ✅ |
| Customizable | ✅ easy | ⚠️ hard | ⚠️ |

## Quick Start

Get running in under 5 minutes:

::: code-group

```bash [Clone & Build]
git clone https://github.com/LessUp/cuflash-attn.git
cd cuflash-attn

cmake --preset release
cmake --build --preset release

ctest --preset release --output-on-failure
```

```cpp [C++ Usage]
#include "cuflash/flash_attention.h"

auto err = cuflash::flash_attention_forward(
    d_Q, d_K, d_V, d_O, d_L,
    batch_size, num_heads, seq_len, head_dim,
    scale, true, stream
);
```

```python [Python Binding]
import ctypes
lib = ctypes.CDLL("./build/release/libcuflash_attn.so")

# Call via C ABI
lib.cuflash_attention_forward_f32(
    q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
    B, H, N, D, scale, True, None
)
```

:::

## Memory Efficiency

| Seq Length | Standard Attention | FlashAttention | Savings |
|:----------:|:------------------:|:--------------:|:-------:|
| 1,024 | 4 MB | 8 KB | **99.8%** |
| 4,096 | 64 MB | 32 KB | **99.95%** |
| 16,384 | 1 GB | 128 KB | **99.99%** |

## Documentation

| Resource | Description |
|----------|-------------|
| [Quick Start Guide](/en/guide/quick-start) | Preset-based build path |
| [Building from Source](/en/building) | Platforms, presets, overrides |
| [API Reference](/en/api-reference) | Complete C++ and C ABI docs |
| [Algorithm Deep Dive](/en/algorithm) | Tiling, online softmax, recomputation |
| [Troubleshooting](/en/troubleshooting) | Common issues and solutions |

## Project Status

**Stable v0.3.0 baseline** — Archive-ready reference implementation. Current focus: documentation quality, workflow simplification, and bug fixes.

See [Project Status](/en/project-status) for maintenance posture and governance rules.

## OpenSpec Specification

This project follows **OpenSpec** methodology. Canonical requirements:

- [Design Spec](https://github.com/LessUp/cuflash-attn/blob/master/openspec/specs/design/flash-attention-design.md)
- [Verification Spec](https://github.com/LessUp/cuflash-attn/blob/master/openspec/specs/verification/flash-attention-verification.md)
