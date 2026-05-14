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
    <h3>O(N) 内存</h3>
    <p>通过 FlashAttention 分块技术，在单 GPU 上处理 16K+ token 序列。HBM 中不存储 O(N²) 注意力矩阵。</p>
    <a href="/cuflash-attn/zh/algorithm">算法详解 &rarr;</a>
  </div>
  <div class="home-feature-card">
    <h3>零依赖</h3>
    <p>纯 CUDA C++，无 PyTorch、无 Cutlass、无 Triton。理解每一行代码，修改每一个细节。</p>
    <a href="/cuflash-attn/zh/design/kernel-deep-dive">Kernel 逐行解读 &rarr;</a>
  </div>
  <div class="home-feature-card">
    <h3>完整训练支持</h3>
    <p>前向与反向传播，含梯度重计算。FP32 与 FP16，数值安全累加。</p>
    <a href="/cuflash-attn/zh/api-reference">API 参考 &rarr;</a>
  </div>
  <div class="home-feature-card">
    <h3>多架构覆盖</h3>
    <p>针对 Volta 到 Hopper（sm_70 &rarr; sm_90）优化。支持 V100、A100、H100 及消费级 GPU。</p>
    <a href="/cuflash-attn/zh/performance/benchmarks">基准测试 &rarr;</a>
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

## 内存效率

| 序列长度 | 标准注意力 | FlashAttention | 节省 |
|:--------:|:---------:|:--------------:|:----:|
| 1,024 | 4 MB | 8 KB | **99.8%** |
| 4,096 | 64 MB | 32 KB | **99.95%** |
| 16,384 | 1 GB | 128 KB | **99.99%** |

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
