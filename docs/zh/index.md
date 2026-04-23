---
layout: home
title: 文档

hero:
  name: "CuFlash-Attn"
  text: "高性能 CUDA FlashAttention"
  tagline: 从零实现的 FlashAttention 算法，具备 O(N) 内存复杂度、FP32/FP16 双精度支持，以及完整的训练功能
  image:
    src: /hero-logo.svg
    alt: CuFlash-Attn
  actions:
    - theme: brand
      text: 开始使用
      link: /zh/guide/quick-start
    - theme: alt
      text: 查看 GitHub
      link: https://github.com/LessUp/cuflash-attn

features:
  - icon: ⚡
    title: O(N) 内存
    details: 线性内存复杂度，告别二次增长。轻松处理 16K+ 长度序列，不再担心显存溢出。
  - icon: 🔢
    title: FP32 & FP16
    details: 完整的精度控制，FP16 计算配合 FP32 累加。数值稳定的前向和反向传播。
  - icon: 🔁
    title: 训练就绪
    details: 完整的前向和反向传播，采用重计算策略优化梯度计算。
  - icon: 🎭
    title: 因果掩码
    details: 内置高效的因果注意力支持，适用于 GPT、LLaMA 等自回归模型。
  - icon: 🚀
    title: 多架构
    details: 针对 NVIDIA GPU 优化的 CUDA 内核，支持从 V100 (sm_70) 到 H100 (sm_90)。
  - icon: 🔧
    title: 易于集成
    details: 简洁的 C++ API 和 C ABI 接口，支持 Python ctypes。提供头文件-only 选项。
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
| [快速开始指南](/zh/guide/quick-start) | 5 分钟上手指南 |
| [从源码构建](/zh/building) | 详细的构建说明和选项 |
| [API 参考](/zh/api-reference) | 完整的 C++ 和 C ABI 接口文档 |
| [算法详解](/zh/algorithm) | 深入理解 FlashAttention 内部原理 |
| [故障排除](/zh/troubleshooting) | 常见问题与解决方案 |

## 规范文档

本项目遵循 **OpenSpec** 规范驱动开发方法。所有实现细节都记录在 `openspec/specs/` 目录下：

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
