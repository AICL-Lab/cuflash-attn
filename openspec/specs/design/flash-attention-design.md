---
openspec:
  type: design-spec
  status: accepted
  migrated_from:
    - specs/product/001-flash-attention-core.md
    - specs/rfc/001-core-architecture.md
  last_updated: 2026-04-23
---

# Design Specification: CuFlash-Attn Flash Attention

This design specification combines product requirements and technical design for the CuFlash-Attn library.

---

# Part I: Product Requirements (产品需求)

## Overview

CuFlash-Attn is a high-performance FlashAttention library implemented from scratch in CUDA C++. This project aims to implement the core functionality of the FlashAttention algorithm, using tiling and online softmax techniques to efficiently compute attention mechanisms in Transformer models on GPUs while significantly reducing memory usage.

---

## Glossary

| Term | Description |
|------|-------------|
| **FlashAttention** | An IO-aware exact attention algorithm that reduces HBM accesses through tiling and recomputation strategies |
| **Attention_Kernel** | CUDA kernel that executes attention computations |
| **Query_Matrix (Q)** | Query matrix with shape [batch_size, num_heads, seq_len, head_dim] |
| **Key_Matrix (K)** | Key matrix with shape [batch_size, num_heads, seq_len, head_dim] |
| **Value_Matrix (V)** | Value matrix with shape [batch_size, num_heads, seq_len, head_dim] |
| **Output_Matrix (O)** | Output matrix with shape [batch_size, num_heads, seq_len, head_dim] |
| **Block_Size** | Size of each block during tiled computation |
| **Online_Softmax** | Technique for computing softmax without storing the full attention matrix |
| **Tiling** | Strategy for partitioning large matrices into smaller blocks for computation |
| **HBM** | High Bandwidth Memory (GPU VRAM) |
| **SRAM** | GPU on-chip shared memory |
| **Causal_Mask** | Causal masking to prevent attending to future positions in autoregressive models |

---

## Requirements

### REQ-1: Forward Pass Core Computation

**User Story:** As a deep learning developer, I want to efficiently compute the forward pass of the attention mechanism so I can use it in Transformer models.

| ID | Acceptance Criteria |
|----|---------------------|
| 1.1 | WHEN Q, K, V are provided THEN the Kernel SHALL compute `softmax(QK^T / sqrt(d_k)) * V` and output O |
| 1.2 | WHEN input dimensions are [B, H, N, D] THEN the Kernel SHALL correctly handle all dimensions |
| 1.3 | WHEN seq_len exceeds Block_Size THEN the Kernel SHALL use a tiling strategy |
| 1.4 | WHEN computing softmax THEN the Kernel SHALL use Online_Softmax technique |
| 1.5 | THE Kernel SHALL produce output numerically equivalent to standard attention (error < 1e-3) |

### REQ-2: Backward Pass Computation

**User Story:** As a deep learning developer, I want to compute gradients for the attention mechanism so I can train models.

| ID | Acceptance Criteria |
|----|---------------------|
| 2.1 | WHEN forward output and dO are provided THEN the Kernel SHALL compute dQ, dK, dV gradients |
| 2.2 | WHEN computing backward pass THEN the Kernel SHALL use a recomputation strategy |
| 2.3 | THE Kernel SHALL output gradients numerically equivalent to standard backward propagation (error < 1e-3) |
| 2.4 | WHEN backward pass completes THEN the Kernel SHALL return dQ, dK, dV gradient matrices |

### REQ-3: Tiling Strategy

**User Story:** As a systems developer, I want an efficient tiling strategy implemented to maximize GPU utilization.

| ID | Acceptance Criteria |
|----|---------------------|
| 3.1 | THE Tiling strategy SHALL partition Q, K, V into SRAM-friendly blocks |
| 3.2 | WHEN Block_Size is configured THEN Tiling SHALL ensure blocks fit in shared memory |
| 3.3 | WHEN processing boundary blocks THEN Tiling SHALL correctly handle cases where seq_len is not evenly divisible |

### REQ-4: Online Softmax Implementation

**User Story:** As an algorithm developer, I want online softmax implemented so we don't need to store the full attention matrix.

| ID | Acceptance Criteria |
|----|---------------------|
| 4.1 | THE Online_Softmax SHALL maintain running maximum m and normalization factor l |
| 4.2 | WHEN a new block is processed THEN Online_Softmax SHALL update m and l |
| 4.3 | WHEN all blocks are processed THEN the result SHALL be numerically equivalent to standard softmax |
| 4.4 | THE Online_Softmax SHALL prevent numerical overflow and underflow |

### REQ-5: Causal Masking Support

**User Story:** As an NLP developer, I want causal masking support so I can use this in autoregressive language models.

| ID | Acceptance Criteria |
|----|---------------------|
| 5.1 | WHEN Causal_Mask is enabled THEN the Kernel SHALL set weights at position j > i to negative infinity |
| 5.2 | WHEN using Causal_Mask THEN the Kernel SHALL skip blocks that don't require computation |

### REQ-6: Memory Management

**User Story:** As a systems developer, I want efficient GPU memory management to support longer sequence lengths.

| ID | Acceptance Criteria |
|----|---------------------|
| 6.1 | THE Memory_Manager SHALL allocate only O(N) additional VRAM |
| 6.2 | WHEN forward pass executes THEN it SHALL NOT allocate O(N²) attention matrix storage |
| 6.3 | THE Memory_Manager SHALL correctly manage shared memory |
| 6.4 | WHEN CUDA memory allocation fails THEN it SHALL return a clear error message |

### REQ-7: API Interface Design

**User Story:** As a library user, I want a clean and easy-to-use API so I can easily integrate it into existing projects.

| ID | Acceptance Criteria |
|----|---------------------|
| 7.1 | THE API SHALL provide `flash_attention_forward` function |
| 7.2 | THE API SHALL provide `flash_attention_backward` function |
| 7.3 | WHEN input parameters are invalid THEN the API SHALL return descriptive error messages |
| 7.4 | THE API SHALL support FP16 and FP32 data types |
| 7.5 | THE API SHALL provide an optional scale parameter |

### REQ-8: Numerical Precision Validation

**User Story:** As a QA engineer, I want to verify the numerical precision of the implementation to ensure computational correctness.

| ID | Acceptance Criteria |
|----|---------------------|
| 8.1 | FOR ALL valid inputs, forward output SHALL differ from reference implementation by < 1e-3 |
| 8.2 | FOR ALL valid inputs, backward gradients SHALL differ from reference implementation by < 1e-3 |
| 8.3 | WHEN input contains extreme values THEN computation SHALL remain numerically stable |
| 8.4 | THE implementation SHALL pass PyTorch standard attention comparison tests |

---

## Requirements Traceability Matrix

| Requirement | Test Coverage |
|-------------|---------------|
| REQ-1 | Property 1 (Forward Pass Numerical Equivalence) |
| REQ-2 | Property 2 (Backward Pass Gradient Equivalence) |
| REQ-3 | Unit Tests (Tiling Computation Boundaries) |
| REQ-4 | Property 3 (Online Softmax Equivalence), Property 4 (Numerical Stability) |
| REQ-5 | Property 5 (Causal Mask Correctness) |
| REQ-6 | Error Handling Tests |
| REQ-7 | API Smoke Tests, Property 6 (Data Type Support) |
| REQ-8 | PyTorch Comparison Tests, All Property Tests |

---

## Implementation Phases

All implementation phases are marked as **completed** ✅:

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Project infrastructure (directories, CMake, types) | ✅ |
| Phase 2 | Online softmax (device functions, property tests) | ✅ |
| Phase 3 | Forward pass (matmul helpers, kernel, causal mask, API, tests) | ✅ |
| Phase 4 | Backward pass (auxiliary computation, kernel, causal mask, API, tests) | ✅ |
| Phase 5 | FP16 support (forward, backward, type conversion, tests) | ✅ |
| Phase 6 | Numerical stability and error handling (stability tests, input validation, error handling tests) | ✅ |
| Phase 7 | Integration and documentation (PyTorch comparison, examples, README, docs site) | ✅ |

---

## Future Enhancements (Optional)

| Feature | Priority | Description |
|---------|----------|-------------|
| Dropout Support | Low | Add dropout functionality |
| Relative Position Encoding | Low | Support relative position encoding |
| head_dim > 128 | Low | Extend support for larger head dimensions |
| Multi-Stream Parallelism | Medium | Support parallel computation across multiple CUDA streams |

---

# Part II: Technical Design (技术设计)

## Status

**Accepted** ✅

## Overview

This section defines the core architecture of CuFlash-Attn, a from-scratch CUDA C++ implementation of the FlashAttention library. The design is based on the core ideas from the FlashAttention papers, implementing IO-aware attention computation through tiling and online softmax techniques.

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
