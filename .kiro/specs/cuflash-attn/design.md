# Design Document: CuFlash-Attn

## Overview

CuFlash-Attn 是一个从零实现的 CUDA C++ FlashAttention 库。本设计基于 FlashAttention 论文的核心思想，通过分块计算（tiling）和在线 softmax 技术，实现 IO 感知的高效注意力计算。

### 核心设计原则

| 原则 | 说明 |
|------|------|
| **IO 感知** | 最小化 HBM 访问次数，充分利用 SRAM |
| **分块计算** | 将大矩阵分割成适合共享内存的小块 |
| **在线算法** | 使用在线 softmax 避免存储 O(N²) 的注意力矩阵 |
| **重计算策略** | 反向传播时重新计算注意力权重而非存储 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User API Layer                          │
│  flash_attention_forward() / flash_attention_backward()      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Kernel Launcher                           │
│  - 参数验证                                                   │
│  - Grid/Block 配置                                           │
│  - 共享内存分配                                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CUDA Kernels                              │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ Forward Kernel  │  │ Backward Kernel │                   │
│  │  - Tiling       │  │  - Recompute    │                   │
│  │  - Online Softmax│  │  - Gradient Calc│                   │
│  │  - Causal Mask  │  │  - Causal Mask  │                   │
│  └─────────────────┘  └─────────────────┘                   │
│                                                              │
│  FP32 Kernels: forward.cu, backward.cu                       │
│  FP16 Kernels: fp16.cu, backward_fp16.cu                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Memory Management                         │
│  - 共享内存管理                                               │
│  - 寄存器分配                                                 │
│  - HBM 访问优化                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Components and Interfaces

### 1. API 接口 (flash_attention.h)

```cpp
// 前向传播接口 (FP32)
FlashAttentionError flash_attention_forward(
    const float* Q,           // [batch, heads, seq_len, head_dim]
    const float* K,
    const float* V,
    float* O,                 // 输出
    float* L,                 // logsumexp（反向传播需要）
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,              // 通常为 1/sqrt(head_dim)
    bool causal,
    cudaStream_t stream = 0
);

// 前向传播接口 (FP16)
FlashAttentionError flash_attention_forward(
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

// 反向传播接口 (FP32/FP16 重载)
FlashAttentionError flash_attention_backward(
    const float* Q, const float* K, const float* V,
    const float* O, const float* L, const float* dO,
    float* dQ, float* dK, float* dV,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream = 0
);
```

### 2. Kernel 模板

```cpp
// 前向传播 Kernel
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128)
    flash_attention_forward_kernel(
        const float* __restrict__ Q,
        const float* __restrict__ K,
        const float* __restrict__ V,
        float* __restrict__ O,
        float* __restrict__ L,
        int seq_len, float scale, bool causal
    );

// 反向传播 Kernel
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128)
    flash_attention_backward_dq_kernel(...);

template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128)
    flash_attention_backward_dkdv_kernel(...);
```

---

## Data Models

### 张量布局

所有张量采用 NHSD 布局（batch, heads, seq_len, head_dim），内存连续存储：

```
Memory Layout: [batch_0, head_0, seq_0, dim_0..dim_d]
                       [batch_0, head_0, seq_1, dim_0..dim_d]
                       ...
                       [batch_0, head_1, seq_0, dim_0..dim_d]
                       ...
```

### 分块配置

| head_dim | BLOCK_M | BLOCK_N | 共享内存需求 |
|----------|---------|---------|-------------|
| 32 | 64 | 64 | ~33 KB |
| 64 | 64 | 64 | ~50 KB |
| 128 | 32 | 32 | ~42 KB |

### 在线 Softmax 状态

```cpp
struct OnlineSoftmaxState {
    float m;  // 当前最大值
    float l;  // 归一化因子 (sum of exp)
    
    __device__ void init() {
        m = -INFINITY;
        l = 0.0f;
    }
    
    __device__ void update(float new_m, float new_l) {
        float m_new = max(m, new_m);
        l = l * exp(m - m_new) + new_l * exp(new_m - m_new);
        m = m_new;
    }
};
```

---

## Algorithm Details

### 前向传播算法

```
Algorithm: FlashAttention Forward
Input: Q, K, V ∈ R^(N×d), scale factor s
Output: O ∈ R^(N×d), L ∈ R^N (logsumexp)

1. 将 Q 分成 T_q = ceil(N/B_m) 个块
2. 将 K, V 分成 T_kv = ceil(N/B_n) 个块

3. For each Q block i = 0..T_q-1 (并行):
   a. 从 HBM 加载 Q_i 到 SRAM
   b. 初始化: O_i = 0, m_i = -∞, l_i = 0
   
   c. For each K,V block j = 0..T_kv-1:
      - 如果 causal 且 j*B_n > (i+1)*B_m: 跳过
      - 从 HBM 加载 K_j, V_j 到 SRAM
      - 计算 S_ij = Q_i @ K_j^T * scale
      - 如果 causal: 应用掩码
      - 更新在线 softmax 状态
      - 更新 O_i
   
   d. 最终归一化: O_i = O_i / l_i
   e. 写回 O_i, L_i = m_i + log(l_i) 到 HBM
```

### 反向传播算法

```
Algorithm: FlashAttention Backward
Input: Q, K, V, O, L, dO
Output: dQ, dK, dV

1. 计算 D = rowsum(dO ⊙ O)  // 用于梯度计算

2. For each K,V block j:
   a. 加载 K_j, V_j 到 SRAM
   b. 初始化 dK_j = 0, dV_j = 0
   
   c. For each Q block i:
      - 如果 causal 且不相关: 跳过
      - 加载 Q_i, O_i, dO_i, L_i, D_i
      - 重计算 P_ij = exp(Q_i @ K_j^T * scale - L_i)
      - 计算 dV_j += P_ij^T @ dO_i
      - 计算 dS_ij = P_ij ⊙ (dO_i @ V_j^T - D_i)
      - 计算 dQ_i += dS_ij @ K_j * scale
      - 计算 dK_j += dS_ij^T @ Q_i * scale
   
   d. 写回 dK_j, dV_j 到 HBM

3. 写回所有 dQ 块到 HBM
```

---

## FP16 支持

### 实现策略

FP16 输入在 kernel 内部转换为 FP32 进行计算，输出时再转换回 FP16：

| 阶段 | 数据类型 |
|------|----------|
| 输入 | `half` |
| 内部计算 | `float` (FP32) |
| 输出 | `half` |

### 支持矩阵

| 数据类型 | 前向传播 | 反向传播 |
|----------|----------|----------|
| FP32 (`float`) | ✅ | ✅ |
| FP16 (`half`) | ✅ | ✅ |

---

## Correctness Properties

### Property 1: 前向传播数值等价性

*For any* 有效的 Q, K, V 输入矩阵，FlashAttention 前向传播的输出应与标准注意力计算 `softmax(QK^T * scale) @ V` 的结果在 1e-3 误差范围内一致。

**Validates: Requirements 1.1, 1.2, 1.5, 7.5, 8.1**

### Property 2: 反向传播梯度等价性

*For any* 有效的 Q, K, V, dO 输入，FlashAttention 反向传播计算的 dQ, dK, dV 梯度应与标准注意力反向传播的梯度在 1e-3 误差范围内一致。

**Validates: Requirements 2.1, 2.3, 2.4, 8.2**

### Property 3: 在线 Softmax 等价性

*For any* 输入向量序列，在线 softmax 算法的最终结果应与标准 softmax 计算的结果数值等价。

**Validates: Requirements 4.3**

### Property 4: 数值稳定性

*For any* 包含极端值的有效输入，计算结果不应产生 NaN 或 Inf。

**Validates: Requirements 4.4, 8.3**

### Property 5: 因果掩码正确性

*For any* 启用因果掩码的注意力计算，位置 i 的输出应仅依赖于位置 0 到 i 的输入。

**Validates: Requirements 5.1**

### Property 6: 数据类型支持

*For any* 有效输入，API 应正确处理 FP32 和 FP16 数据类型。

**Validates: Requirements 7.4**

### Property 7: 无效输入错误处理

*For any* 无效输入，API 应返回描述性错误信息而非崩溃。

**Validates: Requirements 7.3**

---

## Error Handling

### 错误类型

```cpp
enum class FlashAttentionError {
    SUCCESS = 0,
    INVALID_DIMENSION,      // 维度参数无效
    DIMENSION_MISMATCH,     // 预留，当前未返回
    NULL_POINTER,           // 空指针输入
    CUDA_ERROR,             // CUDA 运行时错误
    OUT_OF_MEMORY,          // 显存不足
    UNSUPPORTED_HEAD_DIM,   // 不支持的 head_dim
    UNSUPPORTED_DTYPE       // 不支持的数据类型
};
```

### 错误处理策略

| 策略 | 说明 |
|------|------|
| **参数验证** | 在 kernel 启动前验证所有参数 |
| **CUDA 错误检查** | 使用宏包装 CUDA API 调用 |
| **边界检查** | kernel 内部检查数组边界 |
| **错误传播** | 通过返回值传播错误状态 |

---

## Testing Strategy

### 测试框架

- **Google Test**: C++ 单元测试框架
- **RapidCheck**: 属性测试库（可选）
- **PyTorch**: 参考实现进行数值验证

### 测试类型

| 类型 | 说明 |
|------|------|
| 单元测试 | 验证具体功能和边界条件 |
| 属性测试 | 验证通用正确性属性 |
| 集成测试 | PyTorch 对比测试 |
| 数值稳定性测试 | 极端值输入测试 |

---

## Implementation Notes

### 性能优化

| 优化 | 说明 |
|------|------|
| **向量化访存** | `float4` 向量化加载/存储 |
| **Launch Bounds** | `__launch_bounds__(128)` 控制资源使用 |
| **动态共享内存** | 运行时根据 head_dim 调整 |
| **流安全** | 反向传播维护显式 workspace 生命周期 |

### 支持的配置

| 参数 | 支持范围 |
|------|----------|
| head_dim | 32, 64, 128 |
| 数据类型 | FP32, FP16 |
| 因果掩码 | 可选 |

### 限制

- 不支持 head_dim > 128
- 不支持 dropout
- 不支持相对位置编码
