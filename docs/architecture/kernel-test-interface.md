# Kernel Utilities Test Interface Design

## Problem Statement

Current testing of kernel utilities (`matmul.cuh`, `online_softmax.cuh`) has issues:

1. **Test isolation violation**: Tests directly include private headers from `src/kernels/`
2. **CPU simulation vs GPU execution**: `test_online_softmax.cu` uses CPU simulation, not testing the actual GPU device functions
3. **Untested modules**: `matmul.cuh` has no direct tests; only tested indirectly through forward/backward kernels

## Proposed Solution: Test Kernel Wrappers

### 1. Create Test Interface Headers

Create `src/kernels/test_kernels.cuh` with GPU kernel wrappers for testing:

```cpp
// src/kernels/test_kernels.cuh
#pragma once

#include <cuda_runtime.h>
#include "cuflash/flash_attention.h"

namespace cuflash {
namespace test_kernels {

// Test kernel for matmul_ABt
__global__ void test_matmul_ABt_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K, float scale
);

// Test kernel for load_tile_to_shared (FP32)
__global__ void test_load_tile_fp32_kernel(
    const float* src, float* dst,
    int rows, int cols, int stride
);

// Test kernel for load_tile_to_shared (FP16)
__global__ void test_load_tile_fp16_kernel(
    const half* src, float* dst,
    int rows, int cols, int stride
);

// Test kernel for store_tile_from_shared (FP32)
__global__ void test_store_tile_fp32_kernel(
    const float* src, float* dst,
    int rows, int cols, int stride
);

// Test kernel for store_tile_from_shared (FP16)
__global__ void test_store_tile_fp16_kernel(
    const float* src, half* dst,
    int rows, int cols, int stride
);

// Test kernel for OnlineSoftmaxState operations
__global__ void test_online_softmax_kernel(
    const float* input, float* output_max, float* output_sum,
    int n, int block_size
);

// Host-side test entry points
FlashAttentionError test_matmul_ABt(
    const float* A, const float* B, float* C,
    int M, int N, int K, float scale, cudaStream_t stream
);

FlashAttentionError test_tile_load_store_fp32(
    const float* src, float* dst,
    int rows, int cols, int stride, cudaStream_t stream
);

FlashAttentionError test_tile_load_store_fp16(
    const half* src, float* dst,
    int rows, int cols, int stride, cudaStream_t stream
);

FlashAttentionError test_online_softmax(
    const float* input, float* output_max, float* output_sum,
    int n, int block_size, cudaStream_t stream
);

}  // namespace test_kernels
}  // namespace cuflash
```

### 2. Create Test Implementation File

Create `src/kernels/test_kernels.cu`:

```cpp
#include "test_kernels.cuh"
#include "matmul.cuh"
#include "online_softmax.cuh"

namespace cuflash {
namespace test_kernels {

__global__ void test_matmul_ABt_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K, float scale
) {
    extern __shared__ float smem[];
    // Load A and B to shared memory
    // Call matmul_ABt<M, N, K>
    // Store C back
}

// ... other kernel implementations

FlashAttentionError test_matmul_ABt(...) {
    // Launch kernel with proper configuration
    // Check for errors
    return FlashAttentionError::SUCCESS;
}

}  // namespace test_kernels
}  // namespace cuflash
```

### 3. Update Test Files

Update `tests/unit/test_online_softmax.cu`:

```cpp
// Remove: #include "../../src/kernels/online_softmax.cuh"
// Add: #include "test_kernels.cuh"

// Use GPU test instead of CPU simulation
TEST(OnlineSoftmaxGPUTest, BasicEquivalence) {
    std::vector<float> input = {...};
    // Allocate device memory
    // Call test_kernels::test_online_softmax()
    // Compare results
}
```

### 4. Update CMakeLists.txt

Add test kernel source:

```cmake
set(SOURCES
    src/api/flash_attention_api.cu
    src/forward/flash_attention_forward_typed.cu
    src/backward/flash_attention_backward_typed.cu
    src/kernels/test_kernels.cu  # Add this
)
```

## Benefits

| Aspect | Before | After |
|--------|--------|-------|
| Test isolation | Violates private include | Uses public test interface |
| Test accuracy | CPU simulation | Actual GPU execution |
| Coverage | matmul.cuh untested | All utilities tested |
| Debuggability | Hard to isolate issues | Unit-level GPU tests |

## Implementation Priority

1. **Phase 1**: Create `test_kernels.cuh` with basic wrappers
2. **Phase 2**: Implement test kernels for `matmul.cuh`
3. **Phase 3**: Migrate `test_online_softmax.cu` to GPU testing
4. **Phase 4**: Add property-based GPU tests

## Note on GPU Requirement

This improvement requires a CUDA-capable GPU for development and testing.
The CPU simulation tests can be kept as a fallback for environments without GPU.
