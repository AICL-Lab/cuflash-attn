# API Specification: CuFlash-Attn Public API

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

For Python integration via ctypes, the C ABI provides:

```c
// C-compatible forward function
CUFLASH_API int cuflash_flash_attention_forward(
    const void* Q,
    const void* K,
    const void* V,
    void* O,
    void* L,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    int causal,
    int dtype,  // 0=FP32, 1=FP16
    void* stream
);

// C-compatible backward function
CUFLASH_API int cuflash_flash_attention_backward(
    const void* Q,
    const void* K,
    const void* V,
    const void* O,
    const void* L,
    const void* dO,
    void* dQ,
    void* dK,
    void* dV,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    int causal,
    int dtype,
    void* stream
);
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
