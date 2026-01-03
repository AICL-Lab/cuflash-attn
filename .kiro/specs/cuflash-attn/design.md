# Design Document: CuFlash-Attn

## Overview

CuFlash-Attn 是一个从零实现的 CUDA C++ FlashAttention 库。本设计基于 FlashAttention 论文的核心思想，通过分块计算（tiling）和在线 softmax 技术，实现 IO 感知的高效注意力计算。

### 核心设计原则

1. **IO 感知**: 最小化 HBM 访问次数，充分利用 SRAM
2. **分块计算**: 将大矩阵分割成适合共享内存的小块
3. **在线算法**: 使用在线 softmax 避免存储 O(N²) 的注意力矩阵
4. **重计算策略**: 反向传播时重新计算注意力权重而非存储

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

## Components and Interfaces

### 1. API 接口 (flash_attention.h)

```cpp
// 前向传播接口
void flash_attention_forward(
    const float* Q,           // [batch, heads, seq_len, head_dim]
    const float* K,           // [batch, heads, seq_len, head_dim]
    const float* V,           // [batch, heads, seq_len, head_dim]
    float* O,                 // [batch, heads, seq_len, head_dim]
    float* L,                 // [batch, heads, seq_len] - logsumexp for backward
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,              // 通常为 1/sqrt(head_dim)
    bool causal,              // 是否使用因果掩码
    cudaStream_t stream = 0
);

// 反向传播接口
void flash_attention_backward(
    const float* Q,
    const float* K,
    const float* V,
    const float* O,
    const float* L,           // 前向传播保存的 logsumexp
    const float* dO,          // 上游梯度
    float* dQ,
    float* dK,
    float* dV,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream = 0
);
```

### 2. 前向传播 Kernel

```cpp
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void flash_attention_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,
    int seq_len,
    float scale,
    bool causal
);
```

### 3. 反向传播 Kernel

```cpp
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void flash_attention_backward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ L,
    const float* __restrict__ dO,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    int seq_len,
    float scale,
    bool causal
);
```

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

### 分块参数

```cpp
struct TilingConfig {
    static constexpr int BLOCK_M = 64;    // Q 块的行数
    static constexpr int BLOCK_N = 64;    // K/V 块的行数
    static constexpr int HEAD_DIM = 64;   // 支持的 head_dim (可模板化)
    
    // 共享内存需求
    // Q_tile: BLOCK_M * HEAD_DIM * sizeof(float)
    // K_tile: BLOCK_N * HEAD_DIM * sizeof(float)
    // V_tile: BLOCK_N * HEAD_DIM * sizeof(float)
    // S_tile: BLOCK_M * BLOCK_N * sizeof(float)
};
```

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

## Algorithm Details

### 前向传播算法

```
Algorithm: FlashAttention Forward
Input: Q, K, V ∈ R^(N×d), scale factor s
Output: O ∈ R^(N×d), L ∈ R^N (logsumexp)

1. 将 Q 分成 T_q = ceil(N/B_m) 个块
2. 将 K, V 分成 T_kv = ceil(N/B_n) 个块

3. For each Q block i = 0..T_q-1:
   a. 从 HBM 加载 Q_i 到 SRAM
   b. 初始化: O_i = 0, m_i = -∞, l_i = 0
   
   c. For each K,V block j = 0..T_kv-1:
      - 如果 causal 且 j*B_n > (i+1)*B_m: 跳过
      - 从 HBM 加载 K_j, V_j 到 SRAM
      - 计算 S_ij = Q_i @ K_j^T * scale
      - 如果 causal: 应用掩码
      - 计算块内 m_ij = rowmax(S_ij)
      - 计算 P_ij = exp(S_ij - m_ij)
      - 计算块内 l_ij = rowsum(P_ij)
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
      - 重计算 S_ij = Q_i @ K_j^T * scale
      - 重计算 P_ij = exp(S_ij - L_i)
      - 计算 dV_j += P_ij^T @ dO_i
      - 计算 dP_ij = dO_i @ V_j^T
      - 计算 dS_ij = P_ij ⊙ (dP_ij - D_i)
      - 计算 dQ_i += dS_ij @ K_j * scale
      - 计算 dK_j += dS_ij^T @ Q_i * scale
   
   d. 写回 dK_j, dV_j 到 HBM

3. 写回所有 dQ 块到 HBM
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property 1: 前向传播数值等价性

*For any* 有效的 Q, K, V 输入矩阵（任意 batch_size, num_heads, seq_len, head_dim 组合）和任意 scale 值，FlashAttention 前向传播的输出应与标准注意力计算 `softmax(QK^T * scale) @ V` 的结果在 1e-3 误差范围内一致。

**Validates: Requirements 1.1, 1.2, 1.5, 7.5, 8.1**

### Property 2: 反向传播梯度等价性

*For any* 有效的 Q, K, V, dO 输入，FlashAttention 反向传播计算的 dQ, dK, dV 梯度应与标准注意力反向传播的梯度在 1e-3 误差范围内一致，且输出维度与输入维度匹配。

**Validates: Requirements 2.1, 2.3, 2.4, 8.2**

### Property 3: 在线 Softmax 等价性

*For any* 输入向量序列，在线 softmax 算法（分块处理并维护 m, l 状态）的最终结果应与标准 softmax 计算的结果数值等价。

**Validates: Requirements 4.3**

### Property 4: 数值稳定性

*For any* 包含极端值（接近浮点数边界、非常大或非常小的值）的有效输入，计算结果不应产生 NaN 或 Inf，且应保持合理的数值范围。

**Validates: Requirements 4.4, 8.3**

### Property 5: 因果掩码正确性

*For any* 启用因果掩码的注意力计算，位置 i 的输出应仅依赖于位置 0 到 i 的输入，即修改位置 j > i 的 K, V 值不应影响位置 i 的输出。

**Validates: Requirements 5.1**

### Property 6: 数据类型支持

*For any* 有效输入，API 应正确处理 float32 和 float16 数据类型，且两种类型的计算结果应在各自精度范围内与参考实现一致。

**Validates: Requirements 7.4**

### Property 7: 无效输入错误处理

*For any* 无效输入（如维度不匹配、空输入、负数维度等），API 应返回描述性错误信息而非崩溃或产生未定义行为。

**Validates: Requirements 7.3**

## Error Handling

### 错误类型

```cpp
enum class FlashAttentionError {
    SUCCESS = 0,
    INVALID_DIMENSION,      // 维度参数无效
    DIMENSION_MISMATCH,     // Q, K, V 维度不匹配
    NULL_POINTER,           // 空指针输入
    CUDA_ERROR,             // CUDA 运行时错误
    OUT_OF_MEMORY,          // 显存不足
    UNSUPPORTED_HEAD_DIM,   // 不支持的 head_dim
    UNSUPPORTED_DTYPE       // 不支持的数据类型
};
```

### 错误处理策略

1. **参数验证**: 在 kernel 启动前验证所有参数
2. **CUDA 错误检查**: 使用宏包装 CUDA API 调用
3. **边界检查**: kernel 内部检查数组边界
4. **错误传播**: 通过返回值传播错误状态

```cpp
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        return FlashAttentionError::CUDA_ERROR; \
    } \
} while(0)
```

## Testing Strategy

### 测试框架

- 使用 Google Test 作为 C++ 测试框架
- 使用 [rapidcheck](https://github.com/emil-e/rapidcheck) 作为属性测试库
- 使用 PyTorch 作为参考实现进行数值验证

### 单元测试

1. **在线 Softmax 测试**: 验证在线算法与标准实现的等价性
2. **分块计算测试**: 验证边界条件处理
3. **因果掩码测试**: 验证掩码逻辑正确性
4. **错误处理测试**: 验证各种错误情况的处理

### 属性测试

每个属性测试配置为运行至少 100 次迭代，使用随机生成的输入。

**测试标注格式**: `// Feature: cuflash-attn, Property N: [property description]`

### 集成测试

1. **PyTorch 对比测试**: 与 `torch.nn.functional.scaled_dot_product_attention` 对比
2. **端到端测试**: 在完整 Transformer 层中测试
3. **性能基准测试**: 与标准实现对比内存使用和计算时间

### 测试数据生成

```cpp
// 使用 rapidcheck 生成随机测试数据
rc::gen::inRange(1, 8);      // batch_size
rc::gen::inRange(1, 16);     // num_heads
rc::gen::inRange(1, 2048);   // seq_len
rc::gen::element(32, 64, 128); // head_dim
```

## Implementation Notes

### 性能优化

1. **共享内存使用**: 最大化 SRAM 利用率
2. **寄存器分配**: 合理分配寄存器避免溢出
3. **内存合并访问**: 确保全局内存访问合并
4. **Warp 级原语**: 使用 warp shuffle 进行归约

### 支持的配置

- head_dim: 32, 64, 128 (模板特化)
- 数据类型: float32, float16
- Block sizes: BLOCK_M=64, BLOCK_N=64 (可调)

### 限制

- 当前实现不支持 head_dim > 128
- 不支持 dropout（可后续扩展）
- 不支持相对位置编码（可后续扩展）
