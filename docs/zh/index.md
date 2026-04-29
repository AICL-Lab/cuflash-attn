---
layout: home
title: 文档

hero:
  name: "CuFlash-Attn"
  text: "CUDA FlashAttention 参考实现"
  tagline: O(N) 内存 • FP32/FP16 • 前向/反向 • 可归档级 v0.3.0
  image:
    src: /hero-logo.svg
    alt: CuFlash-Attn
  actions:
    - theme: brand
      text: 开始使用 →
      link: /zh/guide/quick-start
    - theme: alt
      text: 查看源码
      link: https://github.com/LessUp/cuflash-attn

features:
  - icon: ⚡
    title: 线性内存
    details: 通过 FlashAttention 分块处理 16K+ token 序列，内存复杂度 O(N) —— 比标准注意力节省 99.9%。
  - icon: 🎯
    title: 参考级质量
    details: 清晰、教育性的 CUDA C++ 实现。无框架依赖。易于理解、修改和集成。
  - icon: 🔢
    title: 完整精度支持
    details: FP32 和 FP16，数值感知累加。前向和反向传播完整支持训练流程。
  - icon: 🎭
    title: 因果掩码
    details: 内置自回归模型支持。API 中一个布尔参数即可启用。
  - icon: 🚀
    title: 多 GPU 架构
    details: 优化的 kernel 覆盖 V100 到 H100（sm_70 → sm_90）。生产级 CUDA 性能。
  - icon: 📦
    title: Python 就绪
    details: C ABI 绑定支持 ctypes 集成。可与 PyTorch、NumPy 或原生 GPU 内存指针配合使用。
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

## 为什么选择 CuFlash-Attn？

::: tip 适用场景
你想**理解** FlashAttention 内部原理，**实验**注意力机制，或在没有重型框架依赖的情况下**集成**到项目中。
:::

### 快速对比

| 特性 | CuFlash-Attn | PyTorch SDPA | FlashAttention-2 |
|------|:------------:|:------------:|:----------------:|
| 教育性代码 | ✅ | ❌ | ⚠️ |
| 无依赖 | ✅ | ❌ PyTorch | ❌ |
| Python 绑定 | ✅ ctypes | ✅ 原生 | ✅ |
| 训练支持 | ✅ | ✅ | ✅ |
| 可定制 | ✅ 简单 | ⚠️ 困难 | ⚠️ |

## 快速开始

5 分钟内运行：

::: code-group

```bash [克隆 & 构建]
git clone https://github.com/LessUp/cuflash-attn.git
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

# 通过 C ABI 调用
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
| [快速开始指南](/zh/guide/quick-start) | Preset 构建路径 |
| [从源码构建](/zh/building) | 平台、presets、覆盖参数 |
| [API 参考](/zh/api-reference) | 完整 C++ 和 C ABI 文档 |
| [算法详解](/zh/algorithm) | 分块、online softmax、重计算 |
| [故障排除](/zh/troubleshooting) | 常见问题与解决方案 |

## 项目状态

**稳定的 v0.3.0 基线** —— 可归档级参考实现。当前重点：文档质量、工作流简化、Bug 修复。

详见 [项目状态](/zh/project-status) 了解维护姿态与治理规则。

## OpenSpec 规范

本项目遵循 **OpenSpec** 规范驱动方法。权威需求定义：

- [设计规范](https://github.com/LessUp/cuflash-attn/blob/master/openspec/specs/design/flash-attention-design.md)
- [验证规范](https://github.com/LessUp/cuflash-attn/blob/master/openspec/specs/verification/flash-attention-verification.md)
