---
layout: home
title: Documentation

hero:
  name: "CuFlash-Attn"
  text: "OpenSpec-Driven CUDA FlashAttention Reference"
  tagline: Stable v0.3.0 baseline for integration, verification, and archive-ready handoff
  image:
    src: /hero-logo.svg
    alt: CuFlash-Attn
  actions:
    - theme: brand
      text: Get Started
      link: /en/guide/quick-start
    - theme: alt
      text: Project Status
      link: /en/project-status

features:
  - icon: ⚡
    title: O(N) Memory
    details: "FlashAttention tiling plus online softmax keeps activation memory linear in sequence length."
  - icon: 🔢
    title: FP32 & FP16
    details: "Forward and backward paths are implemented for both float and half with numerically aware accumulation."
  - icon: 🔁
    title: Stable Integration Surface
    details: "C++ namespace API and C ABI examples are kept aligned with the shipped headers and docs."
  - icon: 🎭
    title: Spec-Tracked Behavior
    details: "Design and verification live in OpenSpec so behavior, tests, and docs share the same source of truth."
  - icon: 🚀
    title: Multi-Architecture
    details: "Documented support covers NVIDIA GPUs from V100 (sm_70) through H100 (sm_90)."
  - icon: 🔧
    title: Handoff Ready
    details: "Lightweight CI, preset-only builds, and bilingual docs make final maintenance and model handoff predictable."
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

## Project Status

CuFlash-Attn is no longer positioned as a fast-moving feature playground. It is maintained as a **stable reference implementation** with a narrow scope:

- fix correctness, packaging, workflow, and documentation drift
- preserve a reliable `v0.3.0` integration surface
- keep the repository easy to review, teach from, and hand off

See [Project Status](/en/project-status) for the maintenance posture and workflow guardrails.

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
| [Quick Start Guide](/en/guide/quick-start) | Preset-based path from clone to first build |
| [Building from Source](/en/building) | Supported presets, overrides, and platform notes |
| [API Reference](/en/api-reference) | Complete C++ and C ABI documentation |
| [Algorithm Deep Dive](/en/algorithm) | FlashAttention tiling, online softmax, and recomputation |
| [Troubleshooting](/en/troubleshooting) | Common build and runtime issues |
| [Project Status](/en/project-status) | Scope, maintenance posture, and handoff rules |

## Specifications

This project follows **OpenSpec** methodology. Canonical requirements live in `openspec/specs/`:

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
