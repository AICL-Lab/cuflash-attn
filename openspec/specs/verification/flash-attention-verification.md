---
openspec:
  type: verification-spec
  status: accepted
  migrated_from:
    - specs/api/001-public-api.md
    - specs/testing/001-test-specification.md
  last_updated: 2026-04-23
---

# Verification Specification: CuFlash-Attn

This verification specification combines API definitions and testing specifications for the CuFlash-Attn library.

---

# Part I: API Specification (接口规范)

## Overview

This document defines the public API for CuFlash-Attn, including C++ and C ABI interfaces for integration with Python and other languages via ctypes.

---

## Core API

### Forward Pass

#### FP32 Forward

```cpp
FlashAttentionError flash_attention_forward(
    const float* Q,           // [batch, heads, seq_len, head_dim]
    const float* K,
    const float* V,
    float* O,                 // Output tensor
    float* L,                 // Logsumexp (required for backward pass)
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,              // Typically 1/sqrt(head_dim)
    bool causal,              // Enable causal masking
    cudaStream_t stream = 0   // Optional CUDA stream
);
```

**Parameters:**

| Parameter | Type | Direction | Description |
|-----------|------|-----------|-------------|
| Q | `const float*` | Input | Query tensor |
| K | `const float*` | Input | Key tensor |
| V | `const float*` | Input | Value tensor |
| O | `float*` | Output | Output tensor |
| L | `float*` | Output | Logsumexp values |
| batch_size | `int` | Input | Batch size |
| num_heads | `int` | Input | Number of attention heads |
| seq_len | `int` | Input | Sequence length |
| head_dim | `int` | Input | Head dimension (32, 64, or 128) |
| scale | `float` | Input | Scaling factor |
| causal | `bool` | Input | Enable causal masking |
| stream | `cudaStream_t` | Input | CUDA stream (optional) |

**Return Value:**

Returns `FlashAttentionError::SUCCESS` on success, or an error code on failure.

#### FP16 Forward

```cpp
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
```

---

### Backward Pass

#### FP32 Backward

```cpp
FlashAttentionError flash_attention_backward(
    const float* Q,           // Input query tensor
    const float* K,           // Input key tensor
    const float* V,           // Input value tensor
    const float* O,           // Output tensor from forward
    const float* L,           // Logsumexp from forward
    const float* dO,          // Gradient of output
    float* dQ,                // Output gradient of Q
    float* dK,                // Output gradient of K
    float* dV,                // Output gradient of V
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream = 0
);
```

#### FP16 Backward

```cpp
FlashAttentionError flash_attention_backward(
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

---

## Error Handling

### Error Enum

```cpp
enum class FlashAttentionError {
    SUCCESS = 0,               // Success
    INVALID_DIMENSION,         // Invalid dimension parameters
    DIMENSION_MISMATCH,        // Dimension mismatch (reserved)
    NULL_POINTER,              // Null pointer input
    CUDA_ERROR,                // CUDA runtime error
    OUT_OF_MEMORY,             // Out of memory
    UNSUPPORTED_HEAD_DIM,      // Unsupported head_dim value
    UNSUPPORTED_DTYPE          // Unsupported data type
};
```

### Error String Conversion

```cpp
const char* get_error_string(FlashAttentionError error);
```

Returns a human-readable string describing the error.

---

## Tensor Layout

### Memory Format

All tensors use NHSD layout: `(batch, heads, seq_len, head_dim)`

Memory is stored contiguously in row-major order:

```
index = ((batch * num_heads + head) * seq_len + seq) * head_dim + dim
```

### Supported Head Dimensions

| head_dim | BLOCK_M | BLOCK_N | Shared Memory |
|----------|---------|---------|---------------|
| 32 | 64 | 64 | ~33 KB |
| 64 | 64 | 64 | ~50 KB |
| 128 | 32 | 32 | ~42 KB |

---

## C ABI Interface

For Python integration via ctypes, the C ABI provides typed wrappers (return value is the
integer representation of `FlashAttentionError`):

```c
// FP32 variants
int cuflash_attention_forward_f32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream
);

int cuflash_attention_backward_f32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* L, const float* dO,
    float* dQ, float* dK, float* dV,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream
);

// FP16 variants (same signatures with half* pointers)
int cuflash_attention_forward_f16(...);
int cuflash_attention_backward_f16(...);

// Error helper (C ABI; takes the int error code returned above)
const char* cuflash_error_string(int error_code);
```

---

## Usage Examples

### Basic FP32 Forward with Causal Masking

```cpp
#include "cuflash/flash_attention.h"

float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

auto err = cuflash::flash_attention_forward(
    d_Q, d_K, d_V,     // Input tensors
    d_O, d_L,          // Output and logsumexp
    batch_size, num_heads, seq_len, head_dim,
    scale,
    true,              // Enable causal masking
    stream             // CUDA stream (optional)
);

if (err != cuflash::FlashAttentionError::SUCCESS) {
    std::cerr << "Error: " << cuflash::get_error_string(err) << std::endl;
}
```

### Backward Pass

```cpp
auto err = cuflash::flash_attention_backward(
    d_Q, d_K, d_V,     // Original inputs
    d_O, d_L,          // Forward outputs
    d_dO,              // Output gradients
    d_dQ, d_dK, d_dV,  // Input gradients
    batch_size, num_heads, seq_len, head_dim,
    scale,
    true,              // Same causal setting as forward
    stream
);
```

---

## Thread Safety

- All API functions are thread-safe when using different CUDA streams
- Multiple streams can be used for concurrent execution
- Shared state is not maintained between calls

---

## Performance Characteristics

### Memory Complexity

| Method | Forward Memory | Backward Memory |
|--------|----------------|-----------------|
| Standard Attention | O(N²) | O(N²) |
| **FlashAttention** | **O(N)** | **O(N)** |

### Supported Configurations

| Parameter | Supported Values |
|-----------|------------------|
| `head_dim` | 32, 64, 128 |
| Data Types | `float` (FP32), `half` (FP16) |
| Causal Masking | Optional |
| Batch Size | ≥ 1 |
| Sequence Length | ≥ 1 |
| Number of Heads | ≥ 1 |

---

# Part II: Testing Specification (测试规范)

## Overview

This document defines the testing strategy and specifications for CuFlash-Attn. All tests are designed to validate the correctness properties defined in the design specification.

---

## Test Frameworks

| Framework | Purpose |
|-----------|---------|
| **Google Test** | C++ unit testing framework |
| **RapidCheck** | Property-based testing (optional) |
| **PyTorch** | Reference implementation for numerical validation |

---

## Correctness Properties

### Property 1: Forward Pass Numerical Equivalence

**Statement:** For any valid Q, K, V input matrices, FlashAttention forward output should match standard attention computation `softmax(QK^T * scale) @ V` within 1e-3 error tolerance.

**Validates:** REQ-1.1, REQ-1.2, REQ-1.5, REQ-7.5, REQ-8.1

**Test Strategy:**
- Generate random Q, K, V matrices
- Compute output using FlashAttention
- Compute reference output using standard attention
- Compare outputs with max absolute error < 1e-3

### Property 2: Backward Pass Gradient Equivalence

**Statement:** For any valid Q, K, V, dO inputs, FlashAttention backward computed dQ, dK, dV gradients should match standard attention backward gradients within 1e-3 error tolerance.

**Validates:** REQ-2.1, REQ-2.3, REQ-2.4, REQ-8.2

**Test Strategy:**
- Generate random Q, K, V, dO matrices
- Compute gradients using FlashAttention backward
- Compute reference gradients using standard attention backward
- Compare gradients with max absolute error < 1e-3

### Property 3: Online Softmax Equivalence

**Statement:** For any input vector sequence, the online softmax algorithm's final result should be numerically equivalent to standard softmax computation.

**Validates:** REQ-4.3

**Test Strategy:**
- Generate random input vectors
- Compute online softmax result
- Compute standard softmax result
- Compare results with numerical equivalence

### Property 4: Numerical Stability

**Statement:** For any valid input containing extreme values, computation should not produce NaN or Inf.

**Validates:** REQ-4.4, REQ-8.3

**Test Strategy:**
- Generate inputs with extreme values (very large, very small)
- Verify no NaN or Inf in outputs
- Test edge cases near numerical limits

### Property 5: Causal Mask Correctness

**Statement:** For any attention computation with causal masking enabled, output at position i should only depend on inputs at positions 0 to i.

**Validates:** REQ-5.1

**Test Strategy:**
- Enable causal masking
- Verify that position i output is independent of positions > i
- Test boundary conditions at mask edges

### Property 6: Data Type Support

**Statement:** For any valid input, the API should correctly handle both FP32 and FP16 data types.

**Validates:** REQ-7.4

**Test Strategy:**
- Test all properties with FP32 inputs
- Test all properties with FP16 inputs
- Verify type conversion correctness

### Property 7: Invalid Input Error Handling

**Statement:** For any invalid input, the API should return descriptive error messages rather than crashing.

**Validates:** REQ-7.3

**Test Strategy:**
- Test with null pointers
- Test with invalid dimensions
- Test with unsupported head_dim values
- Verify appropriate error codes are returned

---

## Test Categories

### Unit Tests

| Test | Description |
|------|-------------|
| `OnlineSoftmaxTest` | Test online softmax correctness |
| `MatMulTest` | Test blocked matrix multiplication |
| `CausalMaskTest` | Test causal mask application |
| `BoundaryTest` | Test boundary handling for non-divisible seq_len |

### Property Tests

| Test | Property |
|------|----------|
| `ForwardPropertyTest` | Property 1: Forward numerical equivalence |
| `BackwardPropertyTest` | Property 2: Backward gradient equivalence |
| `OnlineSoftmaxPropertyTest` | Property 3: Online softmax equivalence |
| `StabilityPropertyTest` | Property 4: Numerical stability |
| `CausalPropertyTest` | Property 5: Causal mask correctness |
| `DTypePropertyTest` | Property 6: Data type support |
| `ErrorPropertyTest` | Property 7: Invalid input error handling |

### Integration Tests

| Test | Description |
|------|-------------|
| `PyTorchComparisonTest` | Compare against PyTorch standard attention |
| `EndToEndTest` | Full forward + backward pipeline |
| `MultiHeadTest` | Test with multiple attention heads |
| `BatchTest` | Test with batch size > 1 |

### Performance Tests

| Test | Description |
|------|-------------|
| `MemoryUsageTest` | Verify O(N) memory complexity |
| `SpeedBenchmark` | Benchmark against standard attention |
| `ScalingTest` | Test scaling with sequence length |

---

## Test Configuration Matrix

### Supported head_dim Values

| head_dim | BLOCK_M | BLOCK_N | Tests |
|----------|---------|---------|-------|
| 32 | 64 | 64 | All properties |
| 64 | 64 | 64 | All properties |
| 128 | 32 | 32 | All properties |

### Data Type Matrix

| Data Type | Forward | Backward | Tests |
|-----------|---------|----------|-------|
| FP32 (`float`) | ✅ | ✅ | All properties |
| FP16 (`half`) | ✅ | ✅ | All properties |

### Causal Masking

| Causal | Tests |
|--------|-------|
| true | All properties with causal masking |
| false | All properties without causal masking |

---

## Test Execution

### Running All Tests

```bash
ctest --preset release --output-on-failure
```

### Running Specific Tests

```bash
# Run forward tests only
ctest --preset release -R ForwardTest

# Run backward tests only
ctest --preset release -R BackwardTest

# Run property tests only
ctest --preset release -R PropertyTest
```

### PyTorch Comparison

```bash
python tests/test_pytorch_comparison.py
```

---

## Coverage Requirements

| Category | Target Coverage |
|----------|-----------------|
| Code Coverage | > 90% line coverage |
| Property Coverage | 100% of correctness properties |
| Configuration Coverage | All supported head_dim, dtypes, causal settings |
| Edge Case Coverage | Boundary conditions, extreme values, error cases |

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
