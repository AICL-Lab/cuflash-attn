# CuFlash-Attn

[![CI](https://img.shields.io/github/actions/workflow/status/LessUp/cuflash-attn/ci.yml?branch=master&style=flat-square&logo=github&label=CI)](https://github.com/LessUp/cuflash-attn/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/LessUp/cuflash-attn/pages.yml?branch=master&style=flat-square&logo=githubpages&logoColor=white&label=Docs)](https://lessup.github.io/cuflash-attn/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](LICENSE)

[English](README.md) | 简体中文

> 从零实现的 CUDA C++ FlashAttention —— 主要用于教育和参考目的。

CuFlash-Attn 提供了 FlashAttention 算法的高效、IO 感知实现，具有 O(N) 内存复杂度，并支持 FP32 和 FP16 精度。

---

## ✨ 主要特性

| 特性 | 说明 |
|------|------|
| ⚡ **O(N) 内存** | 相比标准注意力的 O(N²)，线性内存复杂复杂度 |
| 🔢 **双精度支持** | 完整支持前向和反向传播的 FP32 和 FP16 |
| 🔁 **完整训练** | 前向和反向传播，包含梯度计算 |
| 🎭 **因果掩码** | 内置自回归模型支持 |
| 🔧 **易于集成** | C++ 和 C ABI 接口，便于 Python 通过 ctypes 调用 |
| 🏎️ **多架构 CUDA** | 针对 sm_70 至 sm_90 优化（V100 到 H100） |
| 📚 **双语文档** | 完整的中英文文档 |

---

## 📋 环境要求

| 要求 | 最低版本 | 说明 |
|------|----------|------|
| CUDA Toolkit | 11.0 | 包含 nvcc 编译器 |
| CMake | 3.18 | 构建系统 |
| C++ 编译器 | C++17 | GCC 7+、Clang 5+ 或 MSVC 2017+ |
| GPU | 计算能力 7.0+ | 推荐使用 V100 或更新 |

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/LessUp/cuflash-attn.git
cd cuflash-attn

# 使用 CMake preset 构建（推荐）
cmake --preset release
cmake --build --preset release

# 运行测试
ctest --preset release --output-on-failure
```

### 基本用法

```cpp
#include "cuflash/flash_attention.h"

// 使用因果掩码计算注意力
float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

auto err = cuflash::flash_attention_forward(
    d_Q, d_K, d_V,     // 输入张量 [B, H, N, D]
    d_O, d_L,          // 输出和 logsumexp
    batch_size, num_heads, seq_len, head_dim,
    scale,             // 注意力缩放因子
    true,              // 启用因果掩码
    stream             // CUDA 流（可选）
);

if (err != cuflash::FlashAttentionError::SUCCESS) {
    std::cerr << "错误: " << cuflash::get_error_string(err) << std::endl;
}
```

完整示例请参见 [examples](examples/) 目录。

---

## 📖 文档

| 文档 | 说明 |
|------|------|
| [API 参考 (英文)](docs/en/api-reference.md) | 完整的 C++ 和 C ABI API 文档 |
| [算法详解](docs/zh/algorithm.md) | 深入了解 FlashAttention 实现 |
| [构建指南](docs/zh/building.md) | 详细的构建说明 |
| [故障排除](docs/zh/troubleshooting.md) | 常见问题与解决方案 |

**完整文档站点：** [https://lessup.github.io/cuflash-attn/](https://lessup.github.io/cuflash-attn/)

---

## ⚙️ 支持的配置

| 参数 | 支持的值 |
|------|----------|
| `head_dim` | 32、64、128 |
| 数据类型 | `float` (FP32)、`half` (FP16) |
| 因果掩码 | 可选 |
| 批大小 | ≥ 1 |
| 序列长度 | ≥ 1 |
| 头数 | ≥ 1 |

### GPU 架构支持

| 架构 | 计算能力 | GPU |
|------|----------|------|
| Volta | sm_70 | V100 |
| Turing | sm_75 | RTX 2080 Ti |
| Ampere | sm_80、sm_86 | A100、RTX 3090 |
| Ada Lovelace | sm_89 | RTX 4090 |
| Hopper | sm_90 | H100 |

---

## 🧪 测试

```bash
# 运行所有测试
ctest --preset release --output-on-failure

# 运行特定测试
ctest --preset release -R ForwardTest

# PyTorch 对比测试
python tests/test_pytorch_comparison.py
```

---

## ⚡ 性能特征

### 内存复杂度

| 方法 | 前向内存 | 反向内存 |
|------|----------|----------|
| 标准 Attention | O(N²) | O(N²) |
| **FlashAttention** | **O(N)** | **O(N)** |

### 实际内存节省

| 序列长度 | 标准 Attention | FlashAttention | 节省 |
|-----------------|----------|----------------|---------|
| 1,024 | 4 MB | 8 KB | 99.8% |
| 4,096 | 64 MB | 32 KB | 99.95% |
| 16,384 | 1 GB | 128 KB | 99.99% |

---

## 🏗️ 项目结构

```
├── specs/                        # 规范文档 (Single Source of Truth)
│   ├── product/                  # 产品需求与验收标准
│   ├── rfc/                      # 技术设计文档 (RFCs)
│   ├── api/                      # API 规范
│   └── testing/                  # 测试规范
├── include/
│   └── flash_attention.h         # 公共 API 头文件
├── src/
│   ├── flash_attention_api.cu    # API 实现
│   ├── flash_attention_forward.cu # FP32 前向内核
│   ├── flash_attention_backward.cu # FP32 反向内核
│   ├── flash_attention_fp16.cu   # FP16 前向内核
│   └── flash_attention_backward_fp16.cu # FP16 反向
├── docs/
│   ├── en/                       # 英文文档
│   └── zh/                       # 中文文档
├── tests/                        # 测试套件
├── examples/                     # 使用示例
└── CMakePresets.json            # 构建预设
```

**规范文档：**
- [产品需求](specs/product/001-flash-attention-core.md) - 功能定义与验收标准
- [架构 RFC](specs/rfc/001-core-architecture.md) - 技术设计与架构
- [API 规范](specs/api/001-public-api.md) - 公共 API 定义
- [测试规范](specs/testing/001-test-specification.md) - 测试策略与要求

---

## 🤝 贡献

欢迎贡献！请确保：

1. 代码遵循现有风格（运行 `clang-format`）
2. 使用 `ctest` 通过测试
3. API 变更时更新文档

---

## 📄 许可证

本项目采用 MIT 许可证 —— 详见 [LICENSE](LICENSE) 文件。

---

## 📚 参考文献

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (Dao 等，NeurIPS 2022)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (Dao，ICLR 2024)

---

## 📝 更新日志

版本历史和更新请参见 [CHANGELOG.md](CHANGELOG.md)。

---

<p align="center">
  <sub>用 ❤️ 打造的高效注意力计算</sub>
</p>
