# CuFlash-Attn

[![CI](https://img.shields.io/github/actions/workflow/status/LessUp/cuflash-attn/ci.yml?branch=main&style=flat-square&logo=github&label=CI)](https://github.com/LessUp/cuflash-attn/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/LessUp/cuflash-attn/pages.yml?branch=main&style=flat-square&logo=githubpages&logoColor=white&label=Docs)](https://lessup.github.io/cuflash-attn/)

[English](README.md) | 简体中文

从零实现的高性能 CUDA C++ FlashAttention。

## 特性

- **前向传播**: O(N) 内存复杂度的高效注意力计算
- **反向传播**: 基于重计算策略的梯度计算
- **因果掩码**: 支持自回归模型
- **FP32 & FP16**: 支持单精度和半精度
- **Online Softmax**: 无需存储 O(N²) 注意力矩阵的数值稳定 softmax

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

### 构建选项

- `BUILD_TESTS=ON/OFF`: 构建测试套件（默认: ON）
- `ENABLE_RAPIDCHECK=ON/OFF`: 启用 RapidCheck 属性测试（默认: OFF）
- `BUILD_SHARED_LIBS=ON/OFF`: 构建 Python ctypes 共享库（默认: ON）
- `ENABLE_FAST_MATH=ON/OFF`: 启用 `--use_fast_math`（更快但精度较低，默认: OFF）

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
```

### 支持的配置

- **head_dim**: 32, 64, 128
- **数据类型**: float32, float16
- **因果掩码**: 可选

## 算法

基于 FlashAttention 算法：

1. **分块**: 将 Q, K, V 分成适合 SRAM 的块
2. **Online Softmax**: 增量计算 softmax，不存储完整注意力矩阵
3. **重计算**: 反向传播中重新计算注意力权重而非存储

### 内存复杂度

- 标准 Attention: O(N²)
- FlashAttention: O(N)

## 项目结构

```
├── include/flash_attention.h      # 公共 API 头文件
├── src/
│   ├── flash_attention_api.cu     # API 实现
│   ├── flash_attention_forward.cu # 前向 kernel
│   ├── flash_attention_backward.cu# 反向 kernel
│   ├── flash_attention_fp16.cu    # FP16 支持
│   ├── online_softmax.cuh         # Online softmax 工具
│   └── matmul.cuh                 # 矩阵乘法辅助
├── tests/                         # 测试套件
├── examples/basic_usage.cu        # 使用示例
├── CMakeLists.txt
└── CMakePresets.json
```

## 错误处理

```cpp
cuflash::FlashAttentionError err = cuflash::flash_attention_forward(...);
if (err != cuflash::FlashAttentionError::SUCCESS) {
    std::cerr << cuflash::get_error_string(err) << std::endl;
}
```

### 错误码

- `SUCCESS`: 操作成功
- `INVALID_DIMENSION`: 维度参数无效
- `DIMENSION_MISMATCH`: Q, K, V 维度不匹配
- `NULL_POINTER`: 输入或输出指针为空
- `CUDA_ERROR`: CUDA 运行时错误
- `OUT_OF_MEMORY`: GPU 显存不足
- `UNSUPPORTED_HEAD_DIM`: head_dim 必须为 32, 64 或 128
- `UNSUPPORTED_DTYPE`: 该操作不支持的数据类型

## 许可证

MIT License

## 参考文献

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
