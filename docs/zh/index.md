# CuFlash-Attn 文档

欢迎使用 CuFlash-Attn 文档 — 一个高性能的 CUDA C++ FlashAttention 实现。

## 什么是 CuFlash-Attn？

CuFlash-Attn 是一个从零实现的 FlashAttention 算法，专为以下用途设计：

- **教育目的** - 学习 FlashAttention 的内部工作原理
- **研究** - 修改和实验注意力机制
- **生产** - 简洁的 API 便于集成到大型系统

## 主要特性

::: tip O(N) 内存复杂度
FlashAttention 将内存使用从 O(N²) 降低到 O(N)，使得在相同硬件上能够处理更长的序列。
:::

::: info FP16 & FP32 支持
前向和反向传播都支持半精度（FP16）和单精度（FP32）计算。
:::

::: warning 需要 CUDA
需要支持 CUDA 的 GPU，计算能力 7.0+（推荐使用 V100 或更新）。
:::

## 导航

- [快速开始指南](/zh/guide/quick-start) - 5 分钟上手
- [从源码构建](/zh/guide/building) - 详细构建说明
- [API 参考](/zh/api/) - 完整 API 文档
- [算法详解](/zh/algorithm) - 深入理解 FlashAttention
- [故障排除](/zh/troubleshooting) - 常见问题与解决方案

## 链接

- [GitHub 仓库](https://github.com/LessUp/cuflash-attn)
- [发布版本](https://github.com/LessUp/cuflash-attn/releases)
- [English Docs](/en/)
