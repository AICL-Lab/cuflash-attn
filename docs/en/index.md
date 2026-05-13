---
layout: home
title: Documentation

hero:
  name: "CuFlash-Attn"
  text: "From-Scratch CUDA FlashAttention"
  tagline: O(N) memory complexity · FP32/FP16 · Forward & Backward · Educational & Production-Ready
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
---

<style>
.VPHero {
  background: #000000;
}
.VPHero .name {
  color: #ffffff !important;
}
.VPHero .text {
  color: #94a3b8 !important;
}
.VPHero .tagline {
  color: #64748b !important;
}

.home-features {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1.5rem;
  padding: 4rem 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.home-feature-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  border-radius: 4px;
  padding: 1.5rem;
  transition: border-color 0.15s ease;
  position: relative;
  overflow: hidden;
}

.home-feature-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: var(--vp-c-brand-1);
  transform: scaleX(0);
  transition: transform 0.15s ease;
}

.home-feature-card:hover {
  border-color: var(--vp-c-brand-1);
}

.home-feature-card:hover::before {
  transform: scaleX(1);
}

.home-feature-card h3 {
  font-size: 1.125rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
  color: var(--vp-c-text-1);
}

.home-feature-card p {
  font-size: 0.875rem;
  line-height: 1.6;
  color: var(--vp-c-text-2);
  margin-bottom: 0.75rem;
}

.home-feature-card a {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--vp-c-brand-1);
  text-decoration: none;
}

.home-feature-card a:hover {
  text-decoration: underline;
}

.citation-bar {
  background: var(--vp-c-bg-alt);
  border-top: 1px solid var(--vp-c-border);
  padding: 2rem;
}

.citation-bar .container {
  max-width: 1200px;
  margin: 0 auto;
}

.citation-bar h4 {
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--vp-c-text-3);
  margin-bottom: 1rem;
}

.citation-bar .citation-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
}

.citation-bar .citation-item {
  font-size: 0.8rem;
  line-height: 1.5;
  color: var(--vp-c-text-2);
  padding: 0.75rem;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  border-radius: 4px;
}

.citation-bar .citation-item a {
  color: var(--vp-c-brand-1);
  font-weight: 600;
}
</style>

<div class="home-features">
  <div class="home-feature-card">
    <h3>O(N) Memory</h3>
    <p>Handle 16K+ token sequences on a single GPU via FlashAttention tiling. No O(N²) attention matrices stored in HBM.</p>
    <a href="/cuflash-attn/en/algorithm">Algorithm Details &rarr;</a>
  </div>
  <div class="home-feature-card">
    <h3>Zero Dependencies</h3>
    <p>Pure CUDA C++ with no PyTorch, no Cutlass, no Triton. Understand every line. Modify every detail.</p>
    <a href="/cuflash-attn/en/design/kernel-deep-dive">Kernel Deep Dive &rarr;</a>
  </div>
  <div class="home-feature-card">
    <h3>Full Training Support</h3>
    <p>Forward and backward passes with gradient recomputation. FP32 and FP16 with numerically-safe accumulation.</p>
    <a href="/cuflash-attn/en/api-reference">API Reference &rarr;</a>
  </div>
  <div class="home-feature-card">
    <h3>Multi-Architecture</h3>
    <p>Optimized kernels for Volta through Hopper (sm_70 &rarr; sm_90). V100, A100, H100, and consumer GPUs.</p>
    <a href="/cuflash-attn/en/performance/benchmarks">Benchmarks &rarr;</a>
  </div>
</div>

## Quick Start

Build and run in under 5 minutes:

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
| [Quick Start](/en/guide/quick-start) | Preset-based build and first steps |
| [Building](/en/building) | Platforms, presets, and CMake overrides |
| [Algorithm](/en/algorithm) | Tiling, online softmax, recomputation |
| [Kernel Deep Dive](/en/design/kernel-deep-dive) | Shared memory, warp scheduling, vectorized loads |
| [Design Decisions](/en/design/design-decisions) | ADR-style rationale for key choices |
| [API Reference](/en/api-reference) | Complete C++ and C ABI documentation |
| [Benchmarks](/en/performance/benchmarks) | Reproducible performance data |
| [Roofline Analysis](/en/performance/roofline-analysis) | Bandwidth vs compute bounds |
| [Related Work](/en/research/related-work) | Papers and implementations compared |

<div class="citation-bar">
  <div class="container">
    <h4>Core References</h4>
    <div class="citation-list">
      <div class="citation-item">
        <strong>FlashAttention</strong> — Dao et al., NeurIPS 2022.<br>
        <a href="https://arxiv.org/abs/2205.14135">arXiv:2205.14135</a>
      </div>
      <div class="citation-item">
        <strong>FlashAttention-2</strong> — Dao, ICLR 2024.<br>
        <a href="https://arxiv.org/abs/2307.08691">arXiv:2307.08691</a>
      </div>
      <div class="citation-item">
        <strong>Online Softmax</strong> — Milakov & Gimelshein.<br>
        <a href="https://arxiv.org/abs/1805.02867">arXiv:1805.02867</a>
      </div>
    </div>
  </div>
</div>
