# Product Requirements: CuFlash-Attn Core Features

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
