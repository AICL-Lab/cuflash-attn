# 快速开始

本指南帮助你快速上手 CuFlash-Attn。

## 前置条件

- **CUDA Toolkit** 11.0+ (推荐 12.0+)
- **CMake** 3.18+
- **C++17** 兼容编译器
- **GPU**: NVIDIA GPU，计算能力 7.0+ (V100 或更新)

## 安装

```bash
# 克隆仓库
git clone https://github.com/LessUp/cuflash-attn.git
cd cuflash-attn

# 使用 CMake 预设构建
cmake --preset release
cmake --build --preset release

# 运行测试
ctest --preset release --output-on-failure
```

## 基本用法

```cpp
#include "cuflash/flash_attention.h"

// 前向传播 - 带因果掩码
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

## 反向传播

```cpp
auto err = cuflash::flash_attention_backward(
    d_Q, d_K, d_V,     // 原始输入
    d_O, d_L,          // 前向传播输出
    d_dO,              // 输出梯度
    d_dQ, d_dK, d_dV,  // 输入梯度
    batch_size, num_heads, seq_len, head_dim,
    scale,
    true,              // 与前向传播相同的因果设置
    stream
);
```

## 下一步

- [API 参考](/zh/api-reference) - 完整 API 文档
- [从源码构建](/zh/building) - 详细构建说明
- [算法详解](/zh/algorithm) - 深入理解 FlashAttention
