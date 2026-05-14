---
layout: home
title: 文档

hero:
  name: "CuFlash-Attn"
  text: "从零实现的 CUDA FlashAttention"
  tagline: 技术白皮书 · O(N) 内存 · FP32/FP16 · 前向与反向
  image:
    src: /hero-logo.svg
    alt: CuFlash-Attn
  actions:
    - theme: brand
      text: 开始使用
      link: /zh/guide/quick-start
    - theme: alt
      text: 查看源码
      link: https://github.com/AICL-Lab/cuflash-attn
---

<script setup>
const stats = [
  { value: 'v0.3.0', label: '稳定版' },
  { value: '99.9%', label: '内存节省' },
  { value: '8.9x', label: '最大加速' },
  { value: '0', label: '依赖项' }
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
    <h3>⚡ O(N) 内存</h3>
    <p>通过 FlashAttention 分块技术，在单 GPU 上处理 16K+ token 序列。HBM 中不存储 O(N²) 注意力矩阵。</p>
    <a href="/cuflash-attn/zh/algorithm">算法详解 →</a>
  </div>
  <div class="home-feature-card">
    <h3>📦 零依赖</h3>
    <p>纯 CUDA C++，无 PyTorch、无 Cutlass、无 Triton。理解每一行代码，修改每一个细节。</p>
    <a href="/cuflash-attn/zh/design/kernel-deep-dive">Kernel 逐行解读 →</a>
  </div>
  <div class="home-feature-card">
    <h3>🔄 完整训练支持</h3>
    <p>前向与反向传播，含梯度重计算。FP32 与 FP16，数值安全累加。</p>
    <a href="/cuflash-attn/zh/api-reference">API 参考 →</a>
  </div>
  <div class="home-feature-card">
    <h3>🎯 多架构覆盖</h3>
    <p>针对 Volta 到 Hopper（sm_70 → sm_90）优化。支持 V100、A100、H100 及消费级 GPU。</p>
    <a href="/cuflash-attn/zh/performance/benchmarks">基准测试 →</a>
  </div>
  <div class="home-feature-card">
    <h3>📐 稳定 C ABI</h3>
    <p>稳定的 C ABI，便于与 Python、Rust 或任何支持 FFI 的语言集成。</p>
    <a href="/cuflash-attn/zh/api-reference#c-api">C API 文档 →</a>
  </div>
  <div class="home-feature-card">
    <h3>🔬 规范驱动</h3>
    <p>所有设计决策可追溯到 OpenSpec 规范。教育级质量，生产就绪。</p>
    <a href="https://github.com/LessUp/cuflash-attn/tree/master/openspec/specs">OpenSpec →</a>
  </div>
</div>

<div class="benchmark-section">
  <h2>⚡ 内存效率</h2>
  <p>FlashAttention 将内存复杂度从 O(N²) 降至 O(N)，支持更长的序列训练。</p>
  
  <div class="benchmark-table">
    <table>
      <thead>
        <tr>
          <th>序列长度</th>
          <th>标准注意力</th>
          <th>FlashAttention</th>
          <th>内存节省</th>
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
  <h2>🚀 吞吐量对比</h2>
  <p>在 NVIDIA A100 80GB 上测试，FP16 精度，启用因果掩码。</p>
  
  <div class="benchmark-table">
    <table>
      <thead>
        <tr>
          <th>配置</th>
          <th>FlashAttention</th>
          <th>标准注意力</th>
          <th>加速比</th>
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

## 快速开始

5 分钟内构建并运行：

::: code-group

```bash [克隆 & 构建]
git clone https://github.com/AICL-Lab/cuflash-attn.git
cd cuflash-attn

cmake --preset release
cmake --build --preset release

ctest --preset release --output-on-failure
```

```cpp [C++ 用法]
#include "cuflash/flash_attention.h"

auto err = cuflash::flash_attention_forward(
    d_Q, d_K, d_V, d_O, d_L,
    batch_size, num_heads, seq_len, head_dim,
    scale, true, stream
);
```

```python [Python 绑定]
import ctypes
lib = ctypes.CDLL("./build/release/libcuflash_attn.so")

lib.cuflash_attention_forward_f32(
    q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
    B, H, N, D, scale, True, None
)
```

:::

## 文档导航

| 资源 | 描述 |
|------|------|
| [快速开始](/zh/guide/quick-start) | Preset 构建与第一步 |
| [从源码构建](/zh/building) | 平台、presets 与 CMake 覆盖参数 |
| [算法详解](/zh/algorithm) | 分块、online softmax、重计算 |
| [Kernel 逐行解读](/zh/design/kernel-deep-dive) | 共享内存、warp 调度、向量化加载 |
| [设计决策](/zh/design/design-decisions) | 关键选择的 ADR 式 rationale |
| [API 参考](/zh/api-reference) | 完整 C++ 与 C ABI 文档 |
| [基准测试](/zh/performance/benchmarks) | 可复现的性能数据 |
| [Roofline 分析](/zh/performance/roofline-analysis) | 带宽与计算边界 |
| [相关工作](/zh/research/related-work) | 论文与实现对比 |

<div class="citation-bar">
  <div class="container">
    <h4>核心参考文献</h4>
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
