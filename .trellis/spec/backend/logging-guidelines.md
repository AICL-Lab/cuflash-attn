# Logging Guidelines

> Logging conventions for CuFlash-Attn CUDA library.

---

## Overview

CuFlash-Attn is a **headless GPU library** designed for integration into larger systems. Traditional logging (printf, std::cout) is **not used** in library code to avoid:

1. Performance overhead in GPU kernels
2. Output pollution in embedded contexts
3. Thread safety concerns

Instead, the library uses **error codes** for diagnostics and leaves logging to the consumer.

---

## Logging Strategy

### Library Layer: No Logging

```cpp
// CORRECT: Return error code, let consumer decide
FlashAttentionError flash_attention_forward(...) {
    FlashAttentionError err = validate_params(...);
    if (err != FlashAttentionError::SUCCESS) {
        return err;  // No logging here
    }
    return launch_kernel(...);
}
```

### Consumer Layer: Consumer Handles Logging

```cpp
// Python consumer example
result = lib.cuflash_attention_forward_f32(...)
if result != 0:
    error_msg = lib.cuflash_error_string(result)
    logging.error(f"FlashAttention failed: {error_msg}")  # Consumer logs
    raise RuntimeError(error_msg)
```

```cpp
// C++ consumer example
auto err = cuflash::flash_attention_forward(...);
if (err != cuflash::FlashAttentionError::SUCCESS) {
    std::cerr << "Error: " << cuflash::get_error_string(err) << std::endl;
    return err;
}
```

---

## Debugging Mechanisms

### 1. Error Code Inspection

Use `get_error_string()` for human-readable error messages:

```cpp
const char* get_error_string(FlashAttentionError error);
// Returns: "Success", "Null pointer", "CUDA error occurred", etc.
```

### 2. CUDA Error Diagnosis

For `CUDA_ERROR`, check CUDA runtime state:

```cpp
cudaError_t cuda_err = cudaGetLastError();
if (cuda_err != cudaSuccess) {
    // Consumer can log: cudaGetErrorString(cuda_err)
}
```

### 3. Validation-First Design

All errors are caught at API boundary before kernel launch:

```cpp
// Clear error messages from validation
if (!Q) return FlashAttentionError::NULL_POINTER;  // "Null pointer: input or output pointer is null"
if (head_dim != 32 && head_dim != 64 && head_dim != 128) {
    return FlashAttentionError::UNSUPPORTED_HEAD_DIM;  // "Unsupported head_dim: must be 32, 64, or 128"
}
```

---

## Development-Time Debugging

### Conditional Debug Output (Development Only)

```cpp
#ifdef CUFLASH_DEBUG
#define CUFLASH_DEBUG_PRINT(fmt, ...) printf("[CuFlash] " fmt "\n", ##__VA_ARGS__)
#else
#define CUFLASH_DEBUG_PRINT(fmt, ...)  // No-op in release
#endif
```

Usage (development builds only):

```cpp
CUFLASH_DEBUG_PRINT("Launching kernel: B=%d, H=%d, N=%d", B, H, N);
```

**Warning**: Never enable `CUFLASH_DEBUG` in production builds - it adds synchronization overhead.

### CUDA Profiling

For performance debugging, use NVIDIA profiling tools:

```bash
# Profile kernel execution
nsys profile ./cuflash_attn_bench

# Detailed GPU profiling
ncu ./cuflash_attn_bench
```

---

## Anti-patterns

### ❌ Printf in Library Code

```cpp
// WRONG: printf in library code
if (!Q) {
    printf("Error: Q is null\n");
    return FlashAttentionError::NULL_POINTER;
}
```

### ✅ Return Error Code

```cpp
// CORRECT: Let consumer handle output
if (!Q) {
    return FlashAttentionError::NULL_POINTER;
}
```

### ❌ std::cout in Kernel

```cpp
// WRONG: cout in GPU kernel (won't work, but shows intent)
__global__ void kernel(...) {
    std::cout << "Thread " << threadIdx.x << std::endl;  // Invalid!
}
```

### ✅ Debug with printf (Development Only)

```cpp
// ACCEPTABLE: printf in kernel for debugging (requires CUDA support)
__global__ void kernel(...) {
#ifdef CUFLASH_DEBUG
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[DEBUG] Kernel launched\n");
    }
#endif
}
```

---

## Testing Error Messages

Unit tests verify error codes, not log output:

```cpp
// From tests/unit/test_error_handling.cu
TEST(ErrorHandlingTest, GetErrorString) {
    EXPECT_STREQ(
        cuflash::get_error_string(cuflash::FlashAttentionError::SUCCESS),
        "Success"
    );
    EXPECT_STREQ(
        cuflash::get_error_string(cuflash::FlashAttentionError::NULL_POINTER),
        "Null pointer: input or output pointer is null"
    );
}
```

---

## Summary

| Principle | Rule |
|-----------|------|
| **Library code** | No logging, return error codes |
| **Consumer code** | Handle logging based on error codes |
| **Error messages** | Use `get_error_string()` for human-readable text |
| **Debugging** | Use `CUFLASH_DEBUG` flag (never in production) |
| **Performance** | Use NVIDIA profiling tools (nsys, ncu) |
| **Tests** | Verify error codes, not log output |

---

## Why This Matters

1. **Performance**: No I/O overhead in critical GPU code paths
2. **Flexibility**: Consumers choose logging framework (spdlog, glog, Python logging)
3. **Embeddable**: Library can be used in silent/embedded contexts
4. **Thread-safe**: No shared output streams to manage
