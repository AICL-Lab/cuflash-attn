# CuFlash-Attn

> **从零实现的高性能 CUDA C++ FlashAttention**

[![CI](https://img.shields.io/github/actions/workflow/status/LessUp/cuflash-attn/ci.yml?branch=master&style=flat-square&logo=github&label=CI)](https://github.com/LessUp/cuflash-attn/actions/workflows/ci.yml)
[![CodeQL](https://img.shields.io/github/actions/workflow/status/LessUp/cuflash-attn/codeql.yml?branch=master&style=flat-square&logo=github&label=CodeQL)](https://github.com/LessUp/cuflash-attn/actions/workflows/codeql.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/LessUp/cuflash-attn/pages.yml?branch=master&style=flat-square&logo=githubpages&logoColor=white&label=文档)](https://lessup.github.io/cuflash-attn/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/github/v/release/LessUp/cuflash-attn?style=flat-square&label=版本)](https://github.com/LessUp/cuflash-attn/releases)

[English](README.md) · [简体中文](README.zh-CN.md) · [文档](https://lessup.github.io/cuflash-attn/zh/) · [API 参考](https://lessup.github.io/cuflash-attn/zh/api-reference)

---

## 🎯 项目简介

CuFlash-Attn 是一个**从零实现的 FlashAttention 算法**，专为**教育学习**、**研究实验**和**生产集成**而优化。

### 为什么选择 CuFlash-Attn？

| 挑战 | 解决方案 |
|------|----------|
| 📚 **学习 FlashAttention** | 清晰、文档完善的 CUDA 内核，逐步的算法实现 |
| 🔬 **研究与实验** | 修改注意力机制，无需复杂的框架依赖 |
| 🚀 **生产就绪** | C++ API 配合 C ABI 绑定，通过 ctypes 无缝集成 Python |
| ⚡ **GPU 优化** | 多架构支持，从 V100 (sm_70) 到 H100 (sm_90) |

---

## ✨ 主要特性

| 特性 | 说明 |
|------|------|
| ⚡ **O(N) 内存** | 线性内存复杂度，相比标准注意力的 O(N²) — 支持 16K+ 序列 |
| 🔢 **双精度支持** | FP32 & FP16，前向和反向传播完整支持 |
| 🔁 **完整训练** | 完整的前向/反向传播，包含梯度计算 |
| 🎭 **因果掩码** | 内置自回归模型支持（GPT 风格） |
| 🔧 **易于集成** | 简洁的 C++ API + C ABI，便于 Python ctypes 集成 |
| 🏎️ **多架构** | 优化的 CUDA 内核，支持 sm_70 → sm_90（V100 → H100） |
| 🧪 **全面测试** | 单元测试、集成测试、压力测试、PyTorch 对比测试 |
| 📊 **性能基准** | Google Benchmark 集成，用于性能追踪 |
| 📚 **双语文档** | 完整的中英文文档 |

---

## 🚀 快速开始

### 环境要求

- **GPU**: NVIDIA GPU，计算能力 7.0+（V100、RTX 20/30/40、A100、H100）
- **CUDA Toolkit**: 11.0 或更高版本
- **CMake**: 3.18 或更高版本
- **编译器**: GCC 7+、Clang 5+ 或 MSVC 2017+（需要 C++17 支持）

### 安装

```bash
# 克隆仓库
git clone https://github.com/LessUp/cuflash-attn.git
cd cuflash-attn

# 使用预设构建（Release 模式）
cmake --preset release
cmake --build --preset release

# 运行测试
ctest --preset release --output-on-failure
```

### 第一个程序

```cpp
#include <cuda_runtime.h>
#include "cuflash/flash_attention.h"
#include <iostream>

int main() {
    const int B = 2, H = 8, N = 1024, D = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    
    // 分配设备内存
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, B * H * N * D * sizeof(float));
    cudaMalloc(&d_K, B * H * N * D * sizeof(float));
    cudaMalloc(&d_V, B * H * N * D * sizeof(float));
    cudaMalloc(&d_O, B * H * N * D * sizeof(float));
    cudaMalloc(&d_L, B * H * N * sizeof(float));
    
    // 用你的数据初始化 Q、K、V...
    
    // 使用因果掩码计算 FlashAttention
    auto err = cuflash::flash_attention_forward(
        d_Q, d_K, d_V, d_O, d_L,
        B, H, N, D, scale,
        true  // 因果掩码
    );
    
    if (err != cuflash::FlashAttentionError::SUCCESS) {
        std::cerr << "错误: " << cuflash::get_error_string(err) << std::endl;
        return 1;
    }
    
    // d_O 现在包含注意力输出
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_L);
    return 0;
}
```

📖 **更多示例**: 请参见 [examples/](examples/) 目录中的完整程序。

---

## 📊 性能

### 内存效率

| 序列长度 | 标准 Attention | FlashAttention | **节省** |
|---------|---------------|----------------|---------|
| 1,024 | 4 MB | 8 KB | **99.8%** |
| 4,096 | 64 MB | 32 KB | **99.95%** |
| 16,384 | 1 GB | 128 KB | **99.99%** |

### 基准测试

在你的硬件上运行性能基准测试：

```bash
cmake --preset release
cmake --build --preset release
./build/release/benchmarks/cuflash_attn_bench
```

基准测试源代码请参见 [benchmarks/](benchmarks/)。

---

## 📖 文档

### 快速链接

| 资源 | 链接 |
|------|------|
| 📘 **完整文档** | [https://lessup.github.io/cuflash-attn/zh/](https://lessup.github.io/cuflash-attn/zh/) |
| 🔌 **API 参考** | [中文 API 文档](https://lessup.github.io/cuflash-attn/zh/api-reference) |
| 🧠 **算法详解** | [深入理解 FlashAttention](https://lessup.github.io/cuflash-attn/zh/algorithm) |
| 🔧 **构建指南** | [从源码构建](https://lessup.github.io/cuflash-attn/zh/building) |
| ❓ **故障排除** | [常见问题与解决方案](https://lessup.github.io/cuflash-attn/zh/troubleshooting) |

### 文档语言

- 🇬🇧 [English Documentation](https://lessup.github.io/cuflash-attn/)
- 🇨🇳 [中文文档](https://lessup.github.io/cuflash-attn/zh/)

---

## ⚙️ 配置

### 支持的参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `head_dim` | 32, 64, 128 | 内核优化必需 |
| **数据类型** | FP32 (`float`), FP16 (`half`) | 前向和反向都支持 |
| **因果掩码** | 可选 | 运行时启用/禁用 |
| **批大小** | ≥ 1 | 任意正整数 |
| **序列长度** | ≥ 1 | 优化用于 1K-16K+ |
| **头数** | ≥ 1 | 任意正整数 |

### GPU 架构支持

| 架构 | 计算能力 | 示例 GPU |
|------|---------|----------|
| Volta | sm_70 | V100 |
| Turing | sm_75 | RTX 2080 Ti |
| Ampere | sm_80, sm_86 | A100, RTX 3090 |
| Ada Lovelace | sm_89 | RTX 4090 |
| Hopper | sm_90 | H100 |

**默认构建目标**: sm_80, sm_86（A100 + RTX 30xx/40xx）

自定义使用: `cmake -B build -DCMAKE_CUDA_ARCHITECTURES="90"`

---

## 🏗️ 项目结构

```
cuflash-attn/
├── benchmarks/                 # 性能基准测试（Google Benchmark）
├── cmake/                      # CMake 模块和打包配置
├── docs/                       # VitePress 文档站点
│   ├── en/                     # 英文文档
│   ├── zh/                     # 中文文档
│   └── public/                 # 静态资源（logo、favicon）
├── examples/                   # 完整使用示例
├── include/cuflash/            # 公共 API 头文件
│   ├── flash_attention.h       # 主 API，包含 C++ 和 C ABI
│   ├── export.h                # 可见性宏
│   └── version.h.in            # 版本头文件模板
├── specs/                      # 规范驱动开发文档
│   ├── product/                # 产品需求
│   ├── rfc/                    # 技术设计（RFCs）
│   ├── api/                    # API 规范
│   └── testing/                # 测试规范
├── src/                        # 实现代码
│   ├── api/                    # API 调度层
│   ├── forward/                # 前向传播内核实现
│   ├── backward/               # 反向传播内核实现
│   └── kernels/                # 内部内核工具（.cuh）
├── tests/                      # 测试套件
│   ├── unit/                   # 单元测试（8 个文件）
│   ├── integration/            # 集成测试 + PyTorch 对比
│   └── package_smoke/          # 包冒烟测试
├── CMakeLists.txt              # 主构建配置
├── CMakePresets.json           # 构建预设（release、debug、asan）
└── .github/workflows/          # CI/CD 工作流
    ├── ci.yml                  # 矩阵构建、测试、基准测试
    ├── codeql.yml              # 安全扫描
    ├── pages.yml               # 文档部署
    └── release.yml             # 发布自动化
```

---

## 🧪 测试与质量

### 测试分类

```bash
# 运行所有测试
ctest --preset release --output-on-failure

# 运行特定测试类别
ctest --preset release -R ForwardTest    # 前向传播测试
ctest --preset release -R BackwardTest   # 反向传播测试
ctest --preset release -R StressTest     # 压力与边界测试
ctest --preset release -R PyTorch        # PyTorch 对比测试（需要 GPU + PyTorch）
```

### 代码质量工具

- ✅ **clang-format**: 自动化代码格式化（CI 强制执行）
- ✅ **clang-tidy**: 静态分析，50+ 检查
- ✅ **CodeQL**: 每周安全扫描
- ✅ **Sanitizers**: AddressSanitizer & UBSan 支持（调试构建）

```bash
# 使用 AddressSanitizer 构建
cmake --preset debug-asan
cmake --build --preset debug-asan
ctest --preset debug-asan
```

---

## 🤝 贡献

欢迎贡献！本项目遵循**规范驱动开发（SDD）**方法。

### 开始贡献

1. **首先阅读规范** 📖 — 所有需求都在 [/specs/](specs/) 中
2. **Fork 并克隆** 仓库
3. **创建分支** 用于你的功能或修复
4. **编写测试** 验证你的更改
5. **更新文档** 如果 API 变更
6. **提交拉取请求**

### 开发工作流

```bash
# 提交前格式化代码
find . -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" | xargs clang-format -i

# 本地运行测试
cmake --preset release && cmake --build --preset release
ctest --preset release --output-on-failure

# 可选：运行 clang-tidy
clang-tidy src/api/flash_attention_api.cu -- -Iinclude
```

📋 **详细指南**: 请参见 [CONTRIBUTING.md](CONTRIBUTING.md)

🤖 **AI 贡献者**: 阅读 [AGENTS.md](AGENTS.md) 了解 SDD 工作流说明。

---

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

## 📚 参考文献

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao 等，NeurIPS 2022
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao，ICLR 2024

---

## 📈 版本历史

详细的版本历史和更新请参见 [CHANGELOG.md](CHANGELOG.md)。

---

<p align="center">
  <sub>用 ❤️ 打造的高效注意力计算</sub><br>
  <sub>规范驱动开发 · CUDA C++ · 开源</sub>
</p>
