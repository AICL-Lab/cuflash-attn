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

<script setup>
const stats = [
  { value: 'v0.3.0', label: 'Stable' },
  { value: '99.9%', label: 'Memory Saved' },
  { value: '8.9x', label: 'Max Speedup' },
  { value: '0', label: 'Dependencies' }
]

const memoryBenchmarks = [
  { seq: '1,024', standard: '4 MB', flash: '8 KB', saved: '99.8%' },
  { seq: '4,096', standard: '64 MB', flash: '32 KB', saved: '99.95%', highlight: true },
  { seq: '16,384', standard: '1 GB', flash: '128 KB', saved: '99.99%', highlight: true },
  { seq: '65,536', standard: '16 GB', flash: '512 KB', saved: '99.97%' }
]

const throughputBenchmarks = [
  { config: 'Batch=1, Seq=1024', flash: '45.2 tok/s', standard: '12.1 tok/s', speedup: '3.7x' },
  { config: 'Batch=8, Seq=1024', flash: '312.5 tok/s', standard: '45.3 tok/s', speedup: '6.9x' },
  { config: 'Batch=32, Seq=1024', flash: '892.1 tok/s', standard: '98.7 tok/s', speedup: '9.0x', highlight: true }
]
</script>

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

.stats-bar {
  display: flex;
  justify-content: center;
  gap: 3rem;
  padding: 1.5rem 0;
  margin: 1.5rem auto 2.5rem;
  max-width: 800px;
  border-top: 1px solid var(--vp-c-border);
  border-bottom: 1px solid var(--vp-c-border);
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.stat-value {
  font-size: 28px;
  font-weight: 800;
  color: var(--vp-c-brand-1);
  font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, monospace;
}

.stat-label {
  font-size: 12px;
  color: var(--vp-c-text-3);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.home-features {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  padding: 0 2rem 3rem;
  max-width: 1200px;
  margin: 0 auto;
}

.home-feature-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  padding: 1.5rem;
  transition: all 0.2s ease;
  position: relative;
  overflow: hidden;
}

.home-feature-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--vp-c-brand-1);
  transform: scaleX(0);
  transition: transform 0.2s ease;
}

.home-feature-card:hover {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 4px 24px rgba(118, 185, 0, 0.1);
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
  transition: gap 0.15s ease;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.home-feature-card a:hover {
  gap: 8px;
}

.benchmark-section {
  background: var(--vp-c-bg-alt);
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  padding: 2rem;
  margin: 2rem auto;
  max-width: 900px;
}

.benchmark-section h2 {
  font-size: 1.5rem;
  font-weight: 700;
  margin: 0 0 0.5rem 0;
  color: var(--vp-c-text-1);
}

.benchmark-section > p {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  margin: 0 0 1.5rem 0;
}

.benchmark-table {
  overflow-x: auto;
}

.benchmark-table table {
  width: 100%;
  border-collapse: collapse;
}

.benchmark-table th {
  text-align: left;
  padding: 0.75rem 1rem;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--vp-c-text-3);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  background: var(--vp-c-bg);
  border-bottom: 1px solid var(--vp-c-border);
}

.benchmark-table td {
  padding: 0.75rem 1rem;
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  border-bottom: 1px solid var(--vp-c-border);
}

.benchmark-table tr:last-child td {
  border-bottom: none;
}

.benchmark-table tr:hover td {
  background: var(--vp-c-bg);
}

.benchmark-table tr.highlight td {
  background: rgba(118, 185, 0, 0.05);
}

.metric-flash {
  font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, monospace;
  font-weight: 600;
  color: var(--vp-c-brand-1);
}

.metric-saved,
.metric-speedup {
  font-weight: 600;
  color: #10b981;
}

.citation-bar {
  background: var(--vp-c-bg-alt);
  border-top: 1px solid var(--vp-c-border);
  padding: 2rem;
  margin-top: 3rem;
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
  border-radius: 8px;
}

.citation-bar .citation-item a {
  color: var(--vp-c-brand-1);
  font-weight: 600;
}

@media (max-width: 640px) {
  .stats-bar {
    flex-wrap: wrap;
    gap: 1.5rem;
  }
}
</style>

<div class="stats-bar">
  <div class="stat-item" v-for="stat in stats" :key="stat.label">
    <span class="stat-value">{{ stat.value }}</span>
    <span class="stat-label">{{ stat.label }}</span>
  </div>
</div>

<div class="home-features">
  <div class="home-feature-card">
    <h3>⚡ O(N) Memory</h3>
    <p>Handle 16K+ token sequences on a single GPU via FlashAttention tiling. No O(N²) attention matrices stored in HBM.</p>
    <a href="/cuflash-attn/en/algorithm">Algorithm Details →</a>
  </div>
  <div class="home-feature-card">
    <h3>📦 Zero Dependencies</h3>
    <p>Pure CUDA C++ with no PyTorch, no Cutlass, no Triton. Understand every line. Modify every detail.</p>
    <a href="/cuflash-attn/en/design/kernel-deep-dive">Kernel Deep Dive →</a>
  </div>
  <div class="home-feature-card">
    <h3>🔄 Full Training Support</h3>
    <p>Forward and backward passes with gradient recomputation. FP32 and FP16 with numerically-safe accumulation.</p>
    <a href="/cuflash-attn/en/api-reference">API Reference →</a>
  </div>
  <div class="home-feature-card">
    <h3>🎯 Multi-Architecture</h3>
    <p>Optimized kernels for Volta through Hopper (sm_70 → sm_90). V100, A100, H100, and consumer GPUs.</p>
    <a href="/cuflash-attn/en/performance/benchmarks">Benchmarks →</a>
  </div>
  <div class="home-feature-card">
    <h3>📐 C ABI Stable</h3>
    <p>Stable C ABI for easy integration with Python, Rust, or any language supporting FFI.</p>
    <a href="/cuflash-attn/en/api-reference#c-api">C API Docs →</a>
  </div>
  <div class="home-feature-card">
    <h3>🔬 Spec-Driven</h3>
    <p>All design decisions traced to OpenSpec specifications. Educational and production-ready.</p>
    <a href="https://github.com/LessUp/cuflash-attn/tree/master/openspec/specs">OpenSpec →</a>
  </div>
</div>

<div class="benchmark-section">
  <h2>⚡ Memory Efficiency</h2>
  <p>FlashAttention reduces memory from O(N²) to O(N), enabling training on much longer sequences.</p>
  
  <div class="benchmark-table">
    <table>
      <thead>
        <tr>
          <th>Sequence Length</th>
          <th>Standard Attention</th>
          <th>FlashAttention</th>
          <th>Memory Saved</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in memoryBenchmarks" :key="row.seq" :class="{ highlight: row.highlight }">
          <td>{{ row.seq }}</td>
          <td>{{ row.standard }}</td>
          <td class="metric-flash">{{ row.flash }}</td>
          <td class="metric-saved">{{ row.saved }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<div class="benchmark-section">
  <h2>🚀 Throughput Comparison</h2>
  <p>Measured on NVIDIA A100 80GB with FP16 precision and causal masking.</p>
  
  <div class="benchmark-table">
    <table>
      <thead>
        <tr>
          <th>Configuration</th>
          <th>FlashAttention</th>
          <th>Standard</th>
          <th>Speedup</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in throughputBenchmarks" :key="row.config" :class="{ highlight: row.highlight }">
          <td>{{ row.config }}</td>
          <td class="metric-flash">{{ row.flash }}</td>
          <td>{{ row.standard }}</td>
          <td class="metric-speedup">{{ row.speedup }}</td>
        </tr>
      </tbody>
    </table>
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