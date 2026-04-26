---
layout: home
title: 文档

hero:
  name: "CuFlash-Attn"
  text: "OpenSpec 驱动的 CUDA FlashAttention 参考实现"
  tagline: 稳定的 v0.3.0 基线，面向集成、验证与可归档交接
  image:
    src: /hero-logo.svg
    alt: CuFlash-Attn
  actions:
    - theme: brand
      text: 开始使用
      link: /zh/guide/quick-start
    - theme: alt
      text: 项目状态
      link: /zh/project-status

features:
  - icon: ⚡
    title: O(N) 内存
    details: "FlashAttention 分块与 online softmax 让激活内存随序列长度线性增长。"
  - icon: 🔢
    title: FP32 & FP16
    details: "`float` 与 `half` 的前向、反向路径齐备，并兼顾数值稳定性。"
  - icon: 🔁
    title: 稳定集成面
    details: "C++ 命名空间 API、C ABI 与示例代码保持同步，方便直接接入上层工程。"
  - icon: 🎭
    title: 规范可追踪
    details: "设计与验证规则收敛到 OpenSpec，代码、测试与文档围绕同一事实源演进。"
  - icon: 🚀
    title: 多架构
    details: "文档化支持范围覆盖 V100 (sm_70) 到 H100 (sm_90) 的 NVIDIA GPU。"
  - icon: 🔧
    title: 易于交接
    details: "轻量 CI、preset-only 构建和双语文档让后续维护与模型接手更可控。"
---

<script setup>
import { onMounted } from 'vue'

onMounted(() => {
  localStorage.setItem('preferred-lang', 'zh')
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

## 快速开始

5 分钟内运行 CuFlash-Attn：

::: code-group

```bash [克隆 & 构建]
# 克隆仓库
git clone https://github.com/LessUp/cuflash-attn.git
cd cuflash-attn

# 使用 CMake preset 构建
cmake --preset release
cmake --build --preset release

# 运行测试
ctest --preset release --output-on-failure
```

```cpp [基本用法]
#include "cuflash/flash_attention.h"

// 带因果掩码的前向传播
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

## 项目状态

CuFlash-Attn 不再被视为快速扩张功能面的实验仓库，而是一个**稳定的参考实现**。当前治理目标很明确：

- 修复正确性、打包、工作流和文档漂移
- 保持可靠的 `v0.3.0` 集成基线
- 让仓库更容易被 review、教学使用和后续接手

维护姿态与流程边界见 [项目状态](/zh/project-status)。

## 性能

标准注意力机制与 FlashAttention 的内存效率对比：

| 序列长度 | 标准注意力 | FlashAttention | **节省** |
|----------|-----------|----------------|---------|
| 1,024 | 4 MB | 8 KB | **99.8%** |
| 4,096 | 64 MB | 32 KB | **99.95%** |
| 16,384 | 1 GB | 128 KB | **99.99%** |

## 文档导航

| 资源 | 描述 |
|------|------|
| [快速开始指南](/zh/guide/quick-start) | 从克隆到首次构建的 preset 路径 |
| [从源码构建](/zh/building) | 支持的 preset、覆盖参数与平台说明 |
| [API 参考](/zh/api-reference) | 完整的 C++ 与 C ABI 接口文档 |
| [算法详解](/zh/algorithm) | FlashAttention 分块、online softmax 与重计算策略 |
| [故障排除](/zh/troubleshooting) | 常见构建与运行问题 |
| [项目状态](/zh/project-status) | 范围、维护姿态与交接规则 |

## 规范文档

本项目遵循 **OpenSpec** 规范驱动开发方法。权威需求定义集中在 `openspec/specs/`：

- [设计规范](https://github.com/LessUp/cuflash-attn/blob/master/openspec/specs/design/flash-attention-design.md) — 需求与算法设计
- [验证规范](https://github.com/LessUp/cuflash-attn/blob/master/openspec/specs/verification/flash-attention-verification.md) — API 定义与测试规范

## GPU 支持

| 架构 | 计算能力 | 示例 GPU | 状态 |
|------|---------|----------|------|
| Volta | sm_70 | V100 | ✅ 支持 |
| Turing | sm_75 | RTX 2080 Ti | ✅ 支持 |
| Ampere | sm_80, sm_86 | A100, RTX 3090 | ✅ 支持 |
| Ada Lovelace | sm_89 | RTX 4090 | ✅ 支持 |
| Hopper | sm_90 | H100 | ✅ 支持 |
