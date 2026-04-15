# CuFlash-Attn

[![CI](https://img.shields.io/github/actions/workflow/status/LessUp/cuflash-attn/ci.yml?branch=master&style=flat-square&logo=github&label=CI)](https://github.com/LessUp/cuflash-attn/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/LessUp/cuflash-attn/pages.yml?branch=master&style=flat-square&logo=githubpages&logoColor=white&label=Docs)](https://lessup.github.io/cuflash-attn/)

[English](README.md) | 简体中文

从零实现的 CUDA C++ FlashAttention。本项目主要作为参考/教学实现，用于展示 FlashAttention 算法；对于追求极致性能的生产环境，建议使用 FlashAttention-2 等成熟库。

## 特性

- **前向传播**: O(N) 内存复杂度的高效注意力计算（支持 FP32 和 FP16）
- **反向传播**: 基于重计算策略的梯度计算（支持 FP32 和 FP16）
- **因果掩码**: 支持自回归模型
- **Online Softmax**: 无需存储 O(N²) 注意力矩阵的数值稳定 softmax

## 已知限制

- **head_dim 支持**: 仅支持 32、64、128
- **高共享内存使用**: head_dim=128 时可能需要支持扩展共享内存的 GPU
- **DIMENSION_MISMATCH 错误**: 当前未主动检查（API 未接收各张量的形状元数据）

## 环境要求

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 编译器
- （可选）PyTorch 用于对比测试

## 构建

### 使用 CMake Presets（推荐）

```bash
cmake --preset default      # Debug 构建 + 测试
cmake --build --preset default
ctest --preset default

cmake --preset release      # 优化构建
cmake --build --preset release
```

### 手动构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

如果 CMake 找不到 CUDA，手动指定：

```bash
cmake .. -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### 构建选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `BUILD_TESTS` | ON | 构建测试套件 |
| `ENABLE_RAPIDCHECK` | OFF | 启用 RapidCheck 属性测试 |
| `BUILD_SHARED_LIBS` | ON | 构建共享库 |
| `BUILD_EXAMPLES` | ON | 构建示例程序 |
| `ENABLE_FAST_MATH` | OFF | 启用 `--use_fast_math`（更快但精度较低） |

## 使用

### C++ API

```cpp
#include "flash_attention.h"

// 前向传播
cuflash::FlashAttentionError err = cuflash::flash_attention_forward(
    Q, K, V,           // 输入张量 [batch, heads, seq_len, head_dim]
    O, L,              // 输出张量和 logsumexp
    batch_size, num_heads, seq_len, head_dim,
    scale,             // 通常为 1/sqrt(head_dim)
    causal,            // 启用因果掩码
    stream             // CUDA 流（可选）
);

// 反向传播
err = cuflash::flash_attention_backward(
    Q, K, V, O, L, dO, // 输入和上游梯度
    dQ, dK, dV,        // 输出梯度
    batch_size, num_heads, seq_len, head_dim,
    scale, causal, stream
);
```

### 支持的配置

| 参数 | 支持范围 |
|------|---------|
| `head_dim` | 32, 64, 128 |
| 数据类型 | `float` (FP32)，`half` (FP16，前后向均支持) |
| 因果掩码 | 可选 |

## 运行测试

```bash
ctest --preset default --output-on-failure
```

GoogleTest 通过 CMake FetchContent 自动下载，无需手动安装。

### PyTorch 对比测试

```bash
python tests/test_pytorch_comparison.py
```

先构建共享库。Preset 构建产物位于 `build/<preset>/`，例如 `build/default/` 或 `build/release/`。也可通过环境变量 `CUFLASH_LIB=/absolute/path/to/libcuflash_attn.so` 指定库路径。

## 算法

基于 FlashAttention 算法：

1. **分块（Tiling）**: 将 Q, K, V 分成适合 SRAM 的块
2. **Online Softmax**: 增量计算 softmax，不存储完整注意力矩阵
3. **重计算（Recomputation）**: 反向传播中重新计算注意力权重而非存储

### 内存复杂度

| 方法 | 前向额外内存 | 反向额外内存 |
|------|-------------|-------------|
| 标准 Attention | O(N²) | O(N²) |
| FlashAttention | O(N) | O(N) |

## 项目结构

```
├── include/
│   └── flash_attention.h          # 公共 API 头文件
├── src/
│   ├── flash_attention_api.cu     # API 实现
│   ├── flash_attention_forward.cu # FP32 前向 kernel
│   ├── flash_attention_backward.cu# FP32 反向 kernel
│   ├── flash_attention_fp16.cu    # FP16 前向 kernel
│   ├── flash_attention_backward_fp16.cu # FP16 反向 kernel
│   ├── kernel_launch_utils.cuh    # Kernel 启动工具
│   ├── online_softmax.cuh         # Online softmax 工具
│   └── matmul.cuh                 # 矩阵乘法辅助
├── tests/
│   ├── test_forward.cu            # 前向传播测试
│   ├── test_backward.cu           # 反向传播测试
│   ├── test_causal_mask.cu        # 因果掩码测试
│   ├── test_online_softmax.cu     # Online softmax 测试
│   ├── test_error_handling.cu     # 错误处理测试
│   ├── test_dtype.cu              # 数据类型测试
│   ├── test_numerical_stability.cu# 数值稳定性测试
│   └── test_pytorch_comparison.py # PyTorch 对比测试
├── examples/
│   └── basic_usage.cu             # 使用示例
├── CMakeLists.txt
└── CMakePresets.json              # 构建预设
```

## 错误处理

```cpp
cuflash::FlashAttentionError err = cuflash::flash_attention_forward(...);
if (err != cuflash::FlashAttentionError::SUCCESS) {
    std::cerr << cuflash::get_error_string(err) << std::endl;
}
```

### 错误码

| 错误码 | 说明 |
|--------|------|
| `SUCCESS` | 操作成功 |
| `INVALID_DIMENSION` | 维度参数无效（≤ 0） |
| `DIMENSION_MISMATCH` | 预留错误码，当前未返回 |
| `NULL_POINTER` | 输入或输出指针为空 |
| `CUDA_ERROR` | CUDA 运行时错误 |
| `OUT_OF_MEMORY` | GPU 显存不足 |
| `UNSUPPORTED_HEAD_DIM` | head_dim 必须为 32, 64 或 128 |
| `UNSUPPORTED_DTYPE` | 该操作不支持的数据类型 |

## GPU 架构支持

| 架构 | 计算能力 | 代表 GPU |
|------|---------|---------|
| Volta | sm_70 | V100 |
| Turing | sm_75 | RTX 2080 Ti |
| Ampere | sm_80, sm_86 | A100, RTX 3090 |
| Ada Lovelace | sm_89 | RTX 4090 |
| Hopper | sm_90 | H100 |

## 许可证

MIT License

## 参考文献

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
