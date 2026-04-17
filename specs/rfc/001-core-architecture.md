# RFC-001: CuFlash-Attn Core Architecture

## Status

**Accepted** ✅

## Overview

This RFC defines the core architecture of CuFlash-Attn, a from-scratch CUDA C++ implementation of the FlashAttention library. The design is based on the core ideas from the FlashAttention papers, implementing IO-aware attention computation through tiling and online softmax techniques.

### Core Design Principles

| Principle | Description |
|-----------|-------------|
| **IO-Awareness** | Minimize HBM accesses, maximize SRAM utilization |
| **Tiled Computation** | Partition large matrices into shared-memory-friendly blocks |
| **Online Algorithms** | Use online softmax to avoid storing O(N²) attention matrices |
| **Recomputation** | Recompute attention weights during backward pass rather than storing them |

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
│  - Parameter validation                                       │
│  - Grid/Block configuration                                  │
│  - Shared memory allocation                                   │
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
│  - Shared memory management                                   │
│  - Register allocation                                        │
│  - HBM access optimization                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Components and Interfaces

### 1. API Interface (flash_attention.h)

```cpp
// Forward pass interface (FP32)
FlashAttentionError flash_attention_forward(
    const float* Q,           // [batch, heads, seq_len, head_dim]
    const float* K,
    const float* V,
    float* O,                 // Output
    float* L,                 // Logsumexp (needed for backward pass)
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,              // Typically 1/sqrt(head_dim)
    bool causal,
    cudaStream_t stream = 0
);

// Forward pass interface (FP16)
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

// Backward pass interface (FP32/FP16 overloads)
FlashAttentionError flash_attention_backward(
    const float* Q, const float* K, const float* V,
    const float* O, const float* L, const float* dO,
    float* dQ, float* dK, float* dV,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream = 0
);
```

### 2. Kernel Templates

```cpp
// Forward pass kernel
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

// Backward pass kernels
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128)
    flash_attention_backward_dq_kernel(...);

template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128)
    flash_attention_backward_dkdv_kernel(...);
```

---

## Data Models

### Tensor Layout

All tensors use NHSD layout (batch, heads, seq_len, head_dim), stored contiguously in memory:

```
Memory Layout: [batch_0, head_0, seq_0, dim_0..dim_d]
                       [batch_0, head_0, seq_1, dim_0..dim_d]
                       ...
                       [batch_0, head_1, seq_0, dim_0..dim_d]
                       ...
```

### Block Configuration

| head_dim | BLOCK_M | BLOCK_N | Shared Memory Requirement |
|----------|---------|---------|---------------------------|
| 32 | 64 | 64 | ~33 KB |
| 64 | 64 | 64 | ~50 KB |
| 128 | 32 | 32 | ~42 KB |

### Online Softmax State

```cpp
struct OnlineSoftmaxState {
    float m;  // Current maximum
    float l;  // Normalization factor (sum of exp)

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

### Forward Pass Algorithm

```
Algorithm: FlashAttention Forward
Input: Q, K, V ∈ R^(N×d), scale factor s
Output: O ∈ R^(N×d), L ∈ R^N (logsumexp)

1. Partition Q into T_q = ceil(N/B_m) blocks
2. Partition K, V into T_kv = ceil(N/B_n) blocks

3. For each Q block i = 0..T_q-1 (in parallel):
   a. Load Q_i from HBM to SRAM
   b. Initialize: O_i = 0, m_i = -∞, l_i = 0

   c. For each K,V block j = 0..T_kv-1:
      - If causal and j*B_n > (i+1)*B_m: skip
      - Load K_j, V_j from HBM to SRAM
      - Compute S_ij = Q_i @ K_j^T * scale
      - If causal: apply mask
      - Update online softmax state
      - Update O_i

   d. Final normalization: O_i = O_i / l_i
   e. Write back O_i, L_i = m_i + log(l_i) to HBM
```

### Backward Pass Algorithm

```
Algorithm: FlashAttention Backward
Input: Q, K, V, O, L, dO
Output: dQ, dK, dV

1. Compute D = rowsum(dO ⊙ O)  // For gradient computation

2. For each K,V block j:
   a. Load K_j, V_j to SRAM
   b. Initialize dK_j = 0, dV_j = 0

   c. For each Q block i:
      - If causal and not relevant: skip
      - Load Q_i, O_i, dO_i, L_i, D_i
      - Recompute P_ij = exp(Q_i @ K_j^T * scale - L_i)
      - Compute dV_j += P_ij^T @ dO_i
      - Compute dS_ij = P_ij ⊙ (dO_i @ V_j^T - D_i)
      - Compute dQ_i += dS_ij @ K_j * scale
      - Compute dK_j += dS_ij^T @ Q_i * scale

   d. Write back dK_j, dV_j to HBM

3. Write back all dQ blocks to HBM
```

---

## FP16 Support

### Implementation Strategy

FP16 inputs are converted to FP32 internally for computation, then converted back to FP16 for output:

| Stage | Data Type |
|-------|-----------|
| Input | `half` |
| Internal computation | `float` (FP32) |
| Output | `half` |

### Support Matrix

| Data Type | Forward Pass | Backward Pass |
|-----------|--------------|---------------|
| FP32 (`float`) | ✅ | ✅ |
| FP16 (`half`) | ✅ | ✅ |

---

## Correctness Properties

### Property 1: Forward Pass Numerical Equivalence

*For any* valid Q, K, V input matrices, FlashAttention forward output should match standard attention computation `softmax(QK^T * scale) @ V` within 1e-3 error tolerance.

**Validates: Requirements 1.1, 1.2, 1.5, 7.5, 8.1**

### Property 2: Backward Pass Gradient Equivalence

*For any* valid Q, K, V, dO inputs, FlashAttention backward computed dQ, dK, dV gradients should match standard attention backward gradients within 1e-3 error tolerance.

**Validates: Requirements 2.1, 2.3, 2.4, 8.2**

### Property 3: Online Softmax Equivalence

*For any* input vector sequence, the online softmax algorithm's final result should be numerically equivalent to standard softmax computation.

**Validates: Requirements 4.3**

### Property 4: Numerical Stability

*For any* valid input containing extreme values, computation should not produce NaN or Inf.

**Validates: Requirements 4.4, 8.3**

### Property 5: Causal Mask Correctness

*For any* attention computation with causal masking enabled, output at position i should only depend on inputs at positions 0 to i.

**Validates: Requirements 5.1**

### Property 6: Data Type Support

*For any* valid input, the API should correctly handle both FP32 and FP16 data types.

**Validates: Requirements 7.4**

### Property 7: Invalid Input Error Handling

*For any* invalid input, the API should return descriptive error messages rather than crashing.

**Validates: Requirements 7.3**

---

## Error Handling

### Error Types

```cpp
enum class FlashAttentionError {
    SUCCESS = 0,
    INVALID_DIMENSION,      // Invalid dimension parameters
    DIMENSION_MISMATCH,     // Reserved, currently not returned
    NULL_POINTER,           // Null pointer input
    CUDA_ERROR,             // CUDA runtime error
    OUT_OF_MEMORY,          // Out of memory
    UNSUPPORTED_HEAD_DIM,   // Unsupported head_dim value
    UNSUPPORTED_DTYPE       // Unsupported data type
};
```

### Error Handling Strategy

| Strategy | Description |
|----------|-------------|
| **Parameter validation** | Validate all parameters before kernel launch |
| **CUDA error checking** | Wrap CUDA API calls with error-checking macros |
| **Boundary checking** | Check array boundaries inside kernels |
| **Error propagation** | Propagate error status through return values |

---

## Testing Strategy

### Test Frameworks

- **Google Test**: C++ unit testing framework
- **RapidCheck**: Property-based testing library (optional)
- **PyTorch**: Reference implementation for numerical validation

### Test Types

| Type | Description |
|------|-------------|
| Unit Tests | Verify specific functionality and boundary conditions |
| Property Tests | Verify general correctness properties |
| Integration Tests | PyTorch comparison tests |
| Numerical Stability Tests | Extreme value input testing |

---

## Implementation Notes

### Performance Optimizations

| Optimization | Description |
|--------------|-------------|
| **Vectorized memory access** | `float4` vectorized loads/stores |
| **Launch bounds** | `__launch_bounds__(128)` to control resource usage |
| **Dynamic shared memory** | Runtime adjustment based on head_dim |
| **Stream safety** | Backward pass maintains explicit workspace lifecycle |

### Supported Configurations

| Parameter | Supported Range |
|-----------|-----------------|
| head_dim | 32, 64, 128 |
| Data types | FP32, FP16 |
| Causal masking | Optional |

### Limitations

- Does not support head_dim > 128
- Does not support dropout
- Does not support relative position encoding
