# API Reference

Complete C++ API reference for CuFlash-Attn. All functions and types are defined in the `cuflash` namespace.

---

## Table of Contents

- [Header File](#header-file)
- [Forward Pass](#forward-pass)
  - [FP32 Forward](#flash_attention_forward-fp32)
  - [FP16 Forward](#flash_attention_forward-fp16)
- [Backward Pass](#backward-pass)
  - [FP32 Backward](#flash_attention_backward-fp32)
  - [FP16 Backward](#flash_attention_backward-fp16)
- [C ABI Interface](#c-abi-interface)
- [Tensor Layout](#tensor-layout)
- [Error Handling](#error-handling)
- [Type Support](#type-support)
- [Build Options](#build-options)
- [GPU Architecture Support](#gpu-architecture-support)

---

## Header File

```cpp
#include "cuflash/flash_attention.h"
```

All public APIs are exposed through this single header file.

---

## Forward Pass

### `flash_attention_forward` (FP32)

Computes FlashAttention forward pass with FP32 precision.

```cpp
FlashAttentionError flash_attention_forward(
    const float* Q,          // Query tensor [B, H, N, D]
    const float* K,          // Key tensor [B, H, N, D]
    const float* V,          // Value tensor [B, H, N, D]
    float* O,                // Output tensor [B, H, N, D]
    float* L,                // logsumexp [B, H, N] (required for backward)
    int batch_size,          // Batch size B
    int num_heads,           // Number of attention heads H
    int seq_len,             // Sequence length N
    int head_dim,            // Head dimension D (32, 64, or 128)
    float scale,             // Scale factor, typically 1.0f / sqrt(D)
    bool causal,             // Enable causal masking
    cudaStream_t stream = 0  // CUDA stream (0 for default)
);
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `Q` | `const float*` | Query tensor on device memory |
| `K` | `const float*` | Key tensor on device memory |
| `V` | `const float*` | Value tensor on device memory |
| `O` | `float*` | Output tensor on device memory |
| `L` | `float*` | logsumexp values on device memory |
| `batch_size` | `int` | Number of sequences in batch |
| `num_heads` | `int` | Number of attention heads |
| `seq_len` | `int` | Length of input sequences |
| `head_dim` | `int` | Dimension of each head (32, 64, or 128) |
| `scale` | `float` | Attention scale factor |
| `causal` | `bool` | Whether to apply causal (autoregressive) masking |
| `stream` | `cudaStream_t` | CUDA stream for asynchronous execution |

**Returns:** `FlashAttentionError::SUCCESS` on success, error code otherwise.

---

### `flash_attention_forward` (FP16)

Computes FlashAttention forward pass with FP16 precision. Internal computations use FP32 for numerical stability, outputs are converted back to FP16.

```cpp
FlashAttentionError flash_attention_forward(
    const half* Q,           // Query tensor [B, H, N, D]
    const half* K,           // Key tensor [B, H, N, D]
    const half* V,           // Value tensor [B, H, N, D]
    half* O,                 // Output tensor [B, H, N, D]
    half* L,                 // logsumexp [B, H, N]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream = 0
);
```

**Precision Handling:**
- Input/Output: FP16 (16-bit half precision)
- Internal computation: FP32 (32-bit single precision)
- Final result: FP16

This approach provides numerical stability comparable to FP32 while reducing memory bandwidth requirements.

---

## Backward Pass

### `flash_attention_backward` (FP32)

Computes gradients for FlashAttention backward pass with FP32 precision.

```cpp
FlashAttentionError flash_attention_backward(
    const float* Q,          // Query tensor from forward pass
    const float* K,          // Key tensor from forward pass
    const float* V,          // Value tensor from forward pass
    const float* O,          // Output tensor from forward pass
    const float* L,          // logsumexp from forward pass
    const float* dO,         // Upstream gradient [B, H, N, D]
    float* dQ,               // Gradient w.r.t. Q (output)
    float* dK,               // Gradient w.r.t. K (output)
    float* dV,               // Gradient w.r.t. V (output)
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream = 0
);
```

**Gradient Computation:**
- Uses recomputation strategy (recomputes attention weights during backward)
- Does not store O(N²) attention matrices
- Memory complexity: O(N) instead of O(N²)

**Requirements:**
- `O` and `L` must be from a corresponding forward pass call
- `dQ`, `dK`, `dV` must be pre-allocated on device memory

---

### `flash_attention_backward` (FP16)

Computes gradients for FlashAttention backward pass with FP16 precision.

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

**Implementation Notes:**
- Internal accumulation uses FP32 to prevent overflow
- Final gradients are converted to FP16
- Numerical stability comparable to FP32 backward pass

---

## C ABI Interface

C-compatible functions for calling from Python via `ctypes` or other languages.

### FP32 Interface

```c
// Forward pass - C ABI
int cuflash_attention_forward_f32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream
);

// Backward pass - C ABI
int cuflash_attention_backward_f32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* L, const float* dO,
    float* dQ, float* dK, float* dV,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream
);
```

### FP16 Interface

```c
// Forward pass - C ABI
int cuflash_attention_forward_f16(
    const half* Q, const half* K, const half* V,
    half* O, half* L,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream
);

// Backward pass - C ABI
int cuflash_attention_backward_f16(
    const half* Q, const half* K, const half* V,
    const half* O, const half* L, const half* dO,
    half* dQ, half* dK, half* dV,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream
);
```

**Return Value:** Integer value of `FlashAttentionError` enum.

---

## Tensor Layout

All tensors use **row-major (C-style)** memory layout.

### Tensor Shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| `Q`, `K`, `V`, `O` | `[batch_size, num_heads, seq_len, head_dim]` | Input/output tensors |
| `dQ`, `dK`, `dV`, `dO` | `[batch_size, num_heads, seq_len, head_dim]` | Gradient tensors |
| `L` | `[batch_size, num_heads, seq_len]` | logsumexp values |

### Memory Offset Calculation

```cpp
// Access Q[b][h][s][d]
size_t offset = ((b * num_heads + h) * seq_len + s) * head_dim + d;

// Access L[b][h][s]
size_t offset = (b * num_heads + h) * seq_len + s;
```

### Data Type Details

- **float**: 32-bit IEEE 754 single precision
- **half**: 16-bit IEEE 754 half precision (CUDA native)
- All pointers must point to contiguous device memory

---

## Error Handling

### `FlashAttentionError` Enum

```cpp
enum class FlashAttentionError {
    SUCCESS = 0,                   // Operation completed successfully
    INVALID_DIMENSION,             // Dimension parameters invalid (≤ 0)
    DIMENSION_MISMATCH,            // Reserved for future use
    NULL_POINTER,                  // Input or output pointer is null
    CUDA_ERROR,                    // CUDA runtime error occurred
    OUT_OF_MEMORY,                 // Insufficient GPU memory
    UNSUPPORTED_HEAD_DIM,          // head_dim not in {32, 64, 128}
    UNSUPPORTED_DTYPE              // Data type not supported
};
```

### `get_error_string`

```cpp
const char* get_error_string(FlashAttentionError error);
```

Returns a human-readable string for the error code.

### Error Handling Example

```cpp
#include "cuflash/flash_attention.h"
#include <iostream>

int main() {
    // ... allocate device memory for d_Q, d_K, d_V, d_O, d_L ...
    
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    auto err = cuflash::flash_attention_forward(
        d_Q, d_K, d_V, d_O, d_L,
        batch_size, num_heads, seq_len, head_dim,
        scale,
        /*causal=*/true
    );
    
    if (err != cuflash::FlashAttentionError::SUCCESS) {
        std::cerr << "FlashAttention error: "
                  << cuflash::get_error_string(err) << std::endl;
        return 1;
    }
    
    // Backward pass
    err = cuflash::flash_attention_backward(
        d_Q, d_K, d_V, d_O, d_L, d_dO,
        d_dQ, d_dK, d_dV,
        batch_size, num_heads, seq_len, head_dim,
        scale, true
    );
    
    if (err != cuflash::FlashAttentionError::SUCCESS) {
        std::cerr << "Backward pass error: "
                  << cuflash::get_error_string(err) << std::endl;
        return 1;
    }
    
    return 0;
}
```

---

## Type Support

### Supported Configurations

| Parameter | Supported Values |
|-----------|------------------|
| `head_dim` | 32, 64, 128 |
| Data types | `float` (FP32), `half` (FP16) |
| Causal masking | Optional (`bool causal`) |
| Batch size | ≥ 1 |
| Number of heads | ≥ 1 |
| Sequence length | ≥ 1 |

### Data Type Support Matrix

| Data Type | Forward Pass | Backward Pass |
|-----------|--------------|---------------|
| `float` (FP32) | ✅ Full support | ✅ Full support |
| `half` (FP16) | ✅ Full support | ✅ Full support |

---

## Build Options

| CMake Option | Default | Description |
|--------------|---------|-------------|
| `BUILD_TESTS` | ON | Build GoogleTest test suite |
| `ENABLE_RAPIDCHECK` | OFF | Enable RapidCheck property-based tests |
| `BUILD_SHARED_LIBS` | ON | Build as shared library |
| `BUILD_EXAMPLES` | ON | Build example programs |
| `ENABLE_FAST_MATH` | OFF | Enable `--use_fast_math` (faster, less precise) |

### Example Configurations

```bash
# High-performance release build
cmake --preset release-fast-math
cmake --build --preset release-fast-math

# Debug build with all tests
cmake --preset default \
      -DENABLE_RAPIDCHECK=ON
cmake --build --preset default

# Static library only
cmake --preset minimal \
      -DBUILD_SHARED_LIBS=OFF
cmake --build --preset minimal
```

---

## GPU Architecture Support

### Supported CUDA Architectures

| Architecture | Compute Capability | Representative GPUs |
|--------------|-------------------|--------------------|
| Volta | sm_70 | V100 |
| Turing | sm_75 | RTX 2080 Ti |
| Ampere | sm_80, sm_86 | A100, RTX 3090 |
| Ada Lovelace | sm_89 | RTX 4090 |
| Hopper | sm_90 | H100 |

### Architecture-Specific Tuning

Default builds support all architectures. For specific deployment:

```bash
# Target only RTX 3090 / A100
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=86

# Target multiple architectures
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES="80;86;89"
```

### Shared Memory Requirements

| head_dim | SRAM Required | Typical Block Size |
|----------|---------------|-------------------|
| 32 | ~32 KB | 64 × 64 |
| 64 | ~64 KB | 64 × 64 |
| 128 | ~128 KB | 32 × 32 |

Note: head_dim=128 requires GPUs with extended shared memory support.

---

## Thread Safety

### Forward Pass
- Fully thread-safe for concurrent calls with different streams
- No shared mutable state

### Backward Pass
- Uses an internal static workspace for intermediate buffer allocation
- **Thread-safe** for single-threaded CUDA context usage (the common case)
- **Not thread-safe** for concurrent backward calls from multiple host threads

#### Multi-Stream Concurrent Usage

For multi-stream concurrent backward pass execution, you have two options:

1. **Sequential execution per thread** (recommended):
   ```cpp
   // Safe: each thread uses its own stream sequentially
   cudaStream_t stream1, stream2;
   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);
   
   // Thread 1: sequential calls on stream1
   flash_attention_backward(..., stream1);
   flash_attention_backward(..., stream1);  // Safe
   
   // Thread 2: would need synchronization with Thread 1
   ```

2. **Synchronize between threads**:
   - Use `cudaStreamSynchronize()` before another thread calls backward pass
   - Or use external mutex for backward pass calls

::: warning Reference Implementation Note
This design is acceptable for educational and single-stream production use cases.
For multi-threaded training pipelines, consider managing workspaces externally.
:::

## Memory Management

- All tensor allocations are caller's responsibility
- No dynamic memory allocation during kernel execution
- Workspace memory is managed internally with stream-safe allocation
