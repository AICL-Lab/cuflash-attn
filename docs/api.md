# API 参考

CuFlash-Attn 提供简洁的 C++ API，所有函数和类型定义在 `cuflash` 命名空间中。

## 头文件

```cpp
#include "flash_attention.h"
```

## 前向传播

### `flash_attention_forward` (FP32)

```cpp
cuflash::FlashAttentionError flash_attention_forward(
    const float* Q,          // 查询张量
    const float* K,          // 键张量
    const float* V,          // 值张量
    float* O,                // 输出张量
    float* L,                // logsumexp（反向传播需要）
    int batch_size,          // 批大小
    int num_heads,           // 注意力头数
    int seq_len,             // 序列长度
    int head_dim,            // 头维度（32, 64, 128）
    float scale,             // 缩放因子，通常 1/√head_dim
    bool causal,             // 是否启用因果掩码
    cudaStream_t stream = 0  // CUDA 流
);
```

### `flash_attention_forward` (FP16)

```cpp
cuflash::FlashAttentionError flash_attention_forward(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    half* L,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream = 0
);
```

## 反向传播

### `flash_attention_backward` (FP32)

```cpp
cuflash::FlashAttentionError flash_attention_backward(
    const float* Q,          // 查询张量
    const float* K,          // 键张量
    const float* V,          // 值张量
    const float* O,          // 前向输出
    const float* L,          // 前向 logsumexp
    const float* dO,         // 上游梯度
    float* dQ,               // Q 的梯度（输出）
    float* dK,               // K 的梯度（输出）
    float* dV,               // V 的梯度（输出）
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream = 0
);
```

### `flash_attention_backward` (FP16)

```cpp
cuflash::FlashAttentionError flash_attention_backward(
    const half* Q,
    const half* K,
    const half* V,
    const half* O,
    const half* L,
    const half* dO,
    half* dQ,
    half* dK,
    half* dV,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream = 0
);
```

> **注意**：FP16 反向传播当前未实现，调用将返回 `UNSUPPORTED_DTYPE`。

### C ABI 接口（用于 Python ctypes）

为方便从 Python 等语言调用，库提供了 C 语言 ABI 接口：

```c
// FP32 前向
cuflash::FlashAttentionError cuflash_forward_f32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream
);

// FP32 反向
cuflash::FlashAttentionError cuflash_backward_f32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* L, const float* dO,
    float* dQ, float* dK, float* dV,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream
);

// FP16 版本类似：cuflash_forward_f16 / cuflash_backward_f16
```

这些函数具有 C 链接（`extern "C"`），可以直接通过 Python `ctypes` 调用。

## 张量布局

所有张量使用 **行优先（row-major）** 布局：

```
Q, K, V, O: [batch_size, num_heads, seq_len, head_dim]
L:          [batch_size, num_heads, seq_len]
```

内存中的偏移计算：

```cpp
// 访问 Q[b][h][s][d]
size_t offset = ((b * num_heads + h) * seq_len + s) * head_dim + d;
```

## 错误处理

### `FlashAttentionError` 枚举

```cpp
enum class FlashAttentionError {
    SUCCESS = 0,
    INVALID_DIMENSION,      // 维度参数无效（≤ 0）
    DIMENSION_MISMATCH,     // Q, K, V 维度不匹配（预留，当前未主动检查）
    NULL_POINTER,           // 输入或输出指针为空
    CUDA_ERROR,             // CUDA 运行时错误
    OUT_OF_MEMORY,          // GPU 显存不足
    UNSUPPORTED_HEAD_DIM,   // head_dim 必须为 32, 64 或 128
    UNSUPPORTED_DTYPE       // 该操作不支持的数据类型
};
```

**注意**：`DIMENSION_MISMATCH` 已预留但当前未实现主动检查，因为 API 未接收每个张量的独立形状信息。

### `get_error_string`

```cpp
const char* get_error_string(FlashAttentionError error);
```

返回错误码对应的可读字符串。

### 使用示例

```cpp
auto err = cuflash::flash_attention_forward(
    d_Q, d_K, d_V, d_O, d_L,
    batch_size, num_heads, seq_len, head_dim,
    1.0f / std::sqrt(static_cast<float>(head_dim)),
    /*causal=*/true
);

if (err != cuflash::FlashAttentionError::SUCCESS) {
    std::cerr << "FlashAttention error: "
              << cuflash::get_error_string(err) << std::endl;
    // 处理错误...
}
```

## 支持的配置

| 参数 | 支持范围 |
|------|---------|
| `head_dim` | 32, 64, 128 |
| 数据类型 | `float` (FP32), `half` (FP16) |
| 因果掩码 | 可选（`bool causal`） |
| 批大小 | ≥ 1 |
| 注意力头数 | ≥ 1 |
| 序列长度 | ≥ 1 |

## 构建选项

| CMake 选项 | 默认值 | 说明 |
|-----------|--------|------|
| `BUILD_TESTS` | ON | 构建测试套件 |
| `ENABLE_RAPIDCHECK` | OFF | 启用 RapidCheck 属性测试 |
| `BUILD_SHARED_LIBS` | ON | 构建共享库（Python ctypes 可用） |
| `ENABLE_FAST_MATH` | OFF | 启用 `--use_fast_math`（更快但精度较低） |

## GPU 架构支持

| 架构 | 计算能力 | 代表 GPU |
|------|---------|---------|
| Volta | sm_70 | V100 |
| Turing | sm_75 | RTX 2080 Ti |
| Ampere | sm_80, sm_86 | A100, RTX 3090 |
| Ada Lovelace | sm_89 | RTX 4090 |
| Hopper | sm_90 | H100 |
