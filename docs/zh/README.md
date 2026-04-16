# CuFlash-Attn 文档

CuFlash-Attn 的完整文档——一个从零实现的 CUDA C++ FlashAttention。

---

## 📚 文档章节

### 入门指南

- [**构建指南**](building.md) — 安装、构建配置和环境设置
- [**快速开始**](#快速开始) — 几分钟内上手运行

### 核心文档

- [**API 参考**](api-reference.md) — 完整的 C++ 和 C ABI API 文档
- [**算法详解**](algorithm.md) — 深入理解 FlashAttention 实现
- [**故障排除**](troubleshooting.md) — 常见问题与解决方案

### 其他资源

- [GitHub 仓库](https://github.com/LessUp/cuflash-attn)
- [更新日志](../../CHANGELOG.md)
- [English Docs](../en/README.md) — English Documentation

---

## 快速开始

```cpp
#include "flash_attention.h"

// 前向传播
auto err = cuflash::flash_attention_forward(
    d_Q, d_K, d_V, d_O, d_L,
    batch_size, num_heads, seq_len, head_dim,
    scale, causal, stream
);

if (err != cuflash::FlashAttentionError::SUCCESS) {
    std::cerr << cuflash::get_error_string(err) << std::endl;
}
```

---

## 主要特性

| 特性 | 说明 |
|------|------|
| ⚡ **O(N) 内存** | 线性内存复杂度而非 O(N²) |
| 🔢 **FP32 & FP16** | 完整支持两种精度类型 |
| 🔁 **前向 & 反向** | 完整的训练支持 |
| 🎭 **因果掩码** | 支持自回归模型 |
| 🔧 **C++ & C ABI** | 通过 ctypes 轻松集成 Python |
| 🏎️ **CUDA 优化** | 多架构支持（sm_70 - sm_90） |

---

## 架构支持

| GPU 架构 | 计算能力 | 支持状态 |
|----------|----------|----------|
| Volta (V100) | sm_70 | ✅ |
| Turing (RTX 2080 Ti) | sm_75 | ✅ |
| Ampere (A100) | sm_80 | ✅ |
| Ampere (RTX 3090) | sm_86 | ✅ |
| Ada Lovelace (RTX 4090) | sm_89 | ✅ |
| Hopper (H100) | sm_90 | ✅ |

---

## 环境要求

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 兼容编译器

---

## 语言

- [English](../en/README.md)
- [简体中文](README.md)（当前）

---

## 许可证

MIT 许可证 — 详见 [LICENSE](../../LICENSE)。
