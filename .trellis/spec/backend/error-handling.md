# Error Handling

> How errors are handled in CuFlash-Attn CUDA library.

---

## Overview

CuFlash-Attn uses **error codes** (not exceptions) for error handling. All functions return a `FlashAttentionError` enum value. This design is compatible with both C++ and C ABI consumers, and avoids exception overhead in GPU code paths.

---

## Error Types

Defined in `include/cuflash/flash_attention.h`:

```cpp
enum class FlashAttentionError {
    SUCCESS = 0,               // Operation completed successfully
    INVALID_DIMENSION,         // Dimensions must be positive
    DIMENSION_MISMATCH,        // Q, K, V tensors have incompatible shapes
    NULL_POINTER,              // Input or output pointer is null
    CUDA_ERROR,                // CUDA runtime error occurred
    OUT_OF_MEMORY,             // Insufficient GPU memory
    UNSUPPORTED_HEAD_DIM,      // head_dim must be 32, 64, or 128
    UNSUPPORTED_DTYPE          // Data type not supported
};
```

### Error String Conversion

```cpp
const char* get_error_string(FlashAttentionError error);
// Returns human-readable string for error code
```

---

## Error Handling Patterns

### 1. Parameter Validation (API Layer)

All public APIs validate parameters before launching kernels:

```cpp
// From src/api/flash_attention_api.cu
static FlashAttentionError validate_params(const void* Q, const void* K, const void* V,
                                           const void* O, const void* L, int batch_size,
                                           int num_heads, int seq_len, int head_dim) {
    // Check null pointers
    if (!Q || !K || !V || !O || !L) {
        return FlashAttentionError::NULL_POINTER;
    }

    // Check dimensions
    if (batch_size <= 0 || num_heads <= 0 || seq_len <= 0 || head_dim <= 0) {
        return FlashAttentionError::INVALID_DIMENSION;
    }

    // Check supported head_dim
    if (head_dim != 32 && head_dim != 64 && head_dim != 128) {
        return FlashAttentionError::UNSUPPORTED_HEAD_DIM;
    }

    return FlashAttentionError::SUCCESS;
}
```

### 2. CUDA Error Handling

Always check CUDA API return values:

```cpp
cudaError_t err = cudaMalloc(&d_ptr, size);
if (err != cudaSuccess) {
    return FlashAttentionError::OUT_OF_MEMORY;
}
```

### 3. Early Return Pattern

```cpp
FlashAttentionError flash_attention_forward(...) {
    // Validate parameters first
    FlashAttentionError err = validate_params(Q, K, V, O, L, ...);
    if (err != FlashAttentionError::SUCCESS) {
        return err;  // Early return on validation failure
    }

    // Launch kernel
    return launch_flash_attention_forward(...);
}
```

---

## C ABI Error Handling

C ABI wrappers return `int` (cast from `FlashAttentionError`):

```cpp
extern "C" {
    // Returns int (cast from FlashAttentionError)
    CUFLASH_EXPORT int cuflash_attention_forward_f32(...);
    
    // Get human-readable error string
    CUFLASH_EXPORT const char* cuflash_error_string(int error_code);
}
```

Python ctypes usage:

```python
result = lib.cuflash_attention_forward_f32(...)
if result != 0:
    error_msg = lib.cuflash_error_string(result)
    raise RuntimeError(f"FlashAttention failed: {error_msg}")
```

---

## Error Propagation Flow

```
User Call
    │
    ▼
┌─────────────────────────────────┐
│  Public API (flash_attention.h) │
│  - Parameter validation         │
│  - Return FlashAttentionError   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  API Dispatch (src/api/)        │
│  - Type dispatch                │
│  - Stream handling              │
│  - CUDA error checking          │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Kernel Launch (src/forward/)   │
│  - Assumes validated inputs     │
│  - Returns CUDA errors up       │
└─────────────────────────────────┘
```

---

## Common Mistakes

### ❌ Throwing Exceptions

```cpp
// WRONG: Exceptions are not C-compatible
if (!Q) throw std::invalid_argument("Q is null");
```

### ✅ Returning Error Codes

```cpp
// CORRECT: Return error code
if (!Q) return FlashAttentionError::NULL_POINTER;
```

### ❌ Ignoring CUDA Errors

```cpp
// WRONG: Unchecked cudaMalloc
cudaMalloc(&d_ptr, size);
```

### ✅ Checking CUDA Errors

```cpp
// CORRECT: Check and propagate
cudaError_t err = cudaMalloc(&d_ptr, size);
if (err != cudaSuccess) {
    return FlashAttentionError::OUT_OF_MEMORY;
}
```

### ❌ Using cudaMemset (Synchronous)

```cpp
// WRONG: Breaks stream ordering
cudaMemset(d_L, 0, size);
```

### ✅ Using cudaMemsetAsync

```cpp
// CORRECT: Maintains stream ordering
cudaMemsetAsync(d_L, 0, size, stream);
```

---

## Testing Error Handling

Unit tests in `tests/unit/test_error_handling.cu`:

```cpp
// Validates error code for null pointer
TEST(ErrorHandlingTest, NullPointerReturnsError) {
    float* Q = nullptr;
    auto err = cuflash::flash_attention_forward(
        Q, K, V, O, L, B, H, N, D, scale, causal
    );
    EXPECT_EQ(err, cuflash::FlashAttentionError::NULL_POINTER);
}

// Validates error for unsupported head_dim
TEST(ErrorHandlingTest, UnsupportedHeadDim) {
    auto err = cuflash::flash_attention_forward(
        Q, K, V, O, L, B, H, N, 48, scale, causal  // 48 not supported
    );
    EXPECT_EQ(err, cuflash::FlashAttentionError::UNSUPPORTED_HEAD_DIM);
}
```

---

## Summary

| Principle | Rule |
|-----------|------|
| **Return type** | Always `FlashAttentionError` (C++), `int` (C ABI) |
| **Validation** | Check at API layer before kernel launch |
| **CUDA errors** | Check every `cudaMalloc`, `cudaMemcpy`, etc. |
| **Stream safety** | Use async operations (`cudaMemsetAsync`) |
| **Exceptions** | NEVER throw in library code |
| **Tests** | Verify all error paths in `test_error_handling.cu` |
