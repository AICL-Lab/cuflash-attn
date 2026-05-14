#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace cuflash {
namespace impl {

// Type conversion utilities for unified FP32/FP16 kernels
__device__ __forceinline__ float to_float(float val) {
    return val;
}
__device__ __forceinline__ float to_float(half val) {
    return __half2float(val);
}

__device__ __forceinline__ void store_float(float* ptr, float val) {
    *ptr = val;
}
__device__ __forceinline__ void store_float(half* ptr, float val) {
    *ptr = __float2half(val);
}

// Check alignment for float4 vectorized loads (16-byte alignment)
__device__ __forceinline__ bool is_aligned_16(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

// Check alignment for half2 vectorized loads (8-byte alignment)
__device__ __forceinline__ bool is_aligned_8(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 0x7) == 0;
}

// =============================================================================
// Load tile from global memory to shared memory
// =============================================================================

// Load a tile from global memory to shared memory (FP32 specialization)
// Handles boundary conditions when seq_len is not divisible by block size
// Uses float4 vectorized loads when alignment permits
template<int BLOCK_ROWS, int BLOCK_COLS>
__device__ __forceinline__ void load_tile_to_shared(const float* __restrict__ src,
                                                    float* __restrict__ dst, int row_start,
                                                    int col_start, int max_rows, int max_cols,
                                                    int src_stride) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int total_elements = BLOCK_ROWS * BLOCK_COLS;

    const bool can_vectorize = (BLOCK_COLS % 4 == 0) && (src_stride % 4 == 0) &&
                               (col_start % 4 == 0) && is_aligned_16(src) && is_aligned_16(dst);

    // Use float4 vectorized loads only when pointer/stride alignment is guaranteed.
    if constexpr (BLOCK_COLS % 4 == 0) {
        if (can_vectorize) {
            const int total_vec = total_elements / 4;
            for (int i = tid; i < total_vec; i += num_threads) {
                int elem_idx = i * 4;
                int local_row = elem_idx / BLOCK_COLS;
                int local_col = elem_idx % BLOCK_COLS;
                int global_row = row_start + local_row;
                int global_col = col_start + local_col;

                float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (global_row < max_rows && global_col + 3 < max_cols) {
                    val = *reinterpret_cast<const float4*>(
                        &src[global_row * src_stride + global_col]);
                } else if (global_row < max_rows) {
                    if (global_col < max_cols)
                        val.x = src[global_row * src_stride + global_col];
                    if (global_col + 1 < max_cols)
                        val.y = src[global_row * src_stride + global_col + 1];
                    if (global_col + 2 < max_cols)
                        val.z = src[global_row * src_stride + global_col + 2];
                    if (global_col + 3 < max_cols)
                        val.w = src[global_row * src_stride + global_col + 3];
                }
                *reinterpret_cast<float4*>(&dst[local_row * BLOCK_COLS + local_col]) = val;
            }
        } else {
            for (int i = tid; i < total_elements; i += num_threads) {
                int local_row = i / BLOCK_COLS;
                int local_col = i % BLOCK_COLS;
                int global_row = row_start + local_row;
                int global_col = col_start + local_col;

                if (global_row < max_rows && global_col < max_cols) {
                    dst[local_row * BLOCK_COLS + local_col] =
                        src[global_row * src_stride + global_col];
                } else {
                    dst[local_row * BLOCK_COLS + local_col] = 0.0f;
                }
            }
        }
    } else {
        for (int i = tid; i < total_elements; i += num_threads) {
            int local_row = i / BLOCK_COLS;
            int local_col = i % BLOCK_COLS;
            int global_row = row_start + local_row;
            int global_col = col_start + local_col;

            if (global_row < max_rows && global_col < max_cols) {
                dst[local_row * BLOCK_COLS + local_col] = src[global_row * src_stride + global_col];
            } else {
                dst[local_row * BLOCK_COLS + local_col] = 0.0f;
            }
        }
    }
}

// Load a tile from global memory to shared memory (FP16 specialization)
// Converts half to float during load, uses float2 (half2) vectorization
template<int BLOCK_ROWS, int BLOCK_COLS>
__device__ __forceinline__ void load_tile_to_shared(const half* __restrict__ src,
                                                    float* __restrict__ dst, int row_start,
                                                    int col_start, int max_rows, int max_cols,
                                                    int src_stride) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int total_elements = BLOCK_ROWS * BLOCK_COLS;

    // Check if we can use half2 vectorized loads (8-byte alignment for half2)
    const bool can_vectorize =
        (BLOCK_COLS % 2 == 0) && (src_stride % 2 == 0) && (col_start % 2 == 0) && is_aligned_8(src);

    if constexpr (BLOCK_COLS % 2 == 0) {
        if (can_vectorize) {
            const int total_vec = total_elements / 2;
            for (int i = tid; i < total_vec; i += num_threads) {
                int elem_idx = i * 2;
                int local_row = elem_idx / BLOCK_COLS;
                int local_col = elem_idx % BLOCK_COLS;
                int global_row = row_start + local_row;
                int global_col = col_start + local_col;

                if (global_row < max_rows && global_col + 1 < max_cols) {
                    half2 val =
                        *reinterpret_cast<const half2*>(&src[global_row * src_stride + global_col]);
                    dst[local_row * BLOCK_COLS + local_col] = __half2float(val.x);
                    dst[local_row * BLOCK_COLS + local_col + 1] = __half2float(val.y);
                } else if (global_row < max_rows) {
                    dst[local_row * BLOCK_COLS + local_col] =
                        (global_col < max_cols)
                            ? __half2float(src[global_row * src_stride + global_col])
                            : 0.0f;
                    dst[local_row * BLOCK_COLS + local_col + 1] =
                        (global_col + 1 < max_cols)
                            ? __half2float(src[global_row * src_stride + global_col + 1])
                            : 0.0f;
                } else {
                    dst[local_row * BLOCK_COLS + local_col] = 0.0f;
                    dst[local_row * BLOCK_COLS + local_col + 1] = 0.0f;
                }
            }
        } else {
            for (int i = tid; i < total_elements; i += num_threads) {
                int local_row = i / BLOCK_COLS;
                int local_col = i % BLOCK_COLS;
                int global_row = row_start + local_row;
                int global_col = col_start + local_col;

                if (global_row < max_rows && global_col < max_cols) {
                    dst[local_row * BLOCK_COLS + local_col] =
                        __half2float(src[global_row * src_stride + global_col]);
                } else {
                    dst[local_row * BLOCK_COLS + local_col] = 0.0f;
                }
            }
        }
    } else {
        for (int i = tid; i < total_elements; i += num_threads) {
            int local_row = i / BLOCK_COLS;
            int local_col = i % BLOCK_COLS;
            int global_row = row_start + local_row;
            int global_col = col_start + local_col;

            if (global_row < max_rows && global_col < max_cols) {
                dst[local_row * BLOCK_COLS + local_col] =
                    __half2float(src[global_row * src_stride + global_col]);
            } else {
                dst[local_row * BLOCK_COLS + local_col] = 0.0f;
            }
        }
    }
}

// =============================================================================
// Store tile from shared memory to global memory
// =============================================================================

// Store a tile from shared memory to global memory (FP32 specialization)
// Uses float4 vectorized stores when alignment permits
template<int BLOCK_ROWS, int BLOCK_COLS>
__device__ __forceinline__ void store_tile_from_shared(const float* __restrict__ src,
                                                       float* __restrict__ dst, int row_start,
                                                       int col_start, int max_rows, int max_cols,
                                                       int dst_stride) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int total_elements = BLOCK_ROWS * BLOCK_COLS;

    const bool can_vectorize = (BLOCK_COLS % 4 == 0) && (dst_stride % 4 == 0) &&
                               (col_start % 4 == 0) && is_aligned_16(src) && is_aligned_16(dst);

    if constexpr (BLOCK_COLS % 4 == 0) {
        if (can_vectorize) {
            const int total_vec = total_elements / 4;
            for (int i = tid; i < total_vec; i += num_threads) {
                int elem_idx = i * 4;
                int local_row = elem_idx / BLOCK_COLS;
                int local_col = elem_idx % BLOCK_COLS;
                int global_row = row_start + local_row;
                int global_col = col_start + local_col;

                if (global_row < max_rows && global_col + 3 < max_cols) {
                    float4 val =
                        *reinterpret_cast<const float4*>(&src[local_row * BLOCK_COLS + local_col]);
                    *reinterpret_cast<float4*>(&dst[global_row * dst_stride + global_col]) = val;
                } else if (global_row < max_rows) {
                    if (global_col < max_cols)
                        dst[global_row * dst_stride + global_col] =
                            src[local_row * BLOCK_COLS + local_col];
                    if (global_col + 1 < max_cols)
                        dst[global_row * dst_stride + global_col + 1] =
                            src[local_row * BLOCK_COLS + local_col + 1];
                    if (global_col + 2 < max_cols)
                        dst[global_row * dst_stride + global_col + 2] =
                            src[local_row * BLOCK_COLS + local_col + 2];
                    if (global_col + 3 < max_cols)
                        dst[global_row * dst_stride + global_col + 3] =
                            src[local_row * BLOCK_COLS + local_col + 3];
                }
            }
        } else {
            for (int i = tid; i < total_elements; i += num_threads) {
                int local_row = i / BLOCK_COLS;
                int local_col = i % BLOCK_COLS;
                int global_row = row_start + local_row;
                int global_col = col_start + local_col;

                if (global_row < max_rows && global_col < max_cols) {
                    dst[global_row * dst_stride + global_col] =
                        src[local_row * BLOCK_COLS + local_col];
                }
            }
        }
    } else {
        for (int i = tid; i < total_elements; i += num_threads) {
            int local_row = i / BLOCK_COLS;
            int local_col = i % BLOCK_COLS;
            int global_row = row_start + local_row;
            int global_col = col_start + local_col;

            if (global_row < max_rows && global_col < max_cols) {
                dst[global_row * dst_stride + global_col] = src[local_row * BLOCK_COLS + local_col];
            }
        }
    }
}

// Store a tile from shared memory to global memory (FP16 specialization)
// Converts float to half during store, uses half2 vectorization
template<int BLOCK_ROWS, int BLOCK_COLS>
__device__ __forceinline__ void store_tile_from_shared(const float* __restrict__ src,
                                                       half* __restrict__ dst, int row_start,
                                                       int col_start, int max_rows, int max_cols,
                                                       int dst_stride) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int total_elements = BLOCK_ROWS * BLOCK_COLS;

    const bool can_vectorize =
        (BLOCK_COLS % 2 == 0) && (dst_stride % 2 == 0) && (col_start % 2 == 0) && is_aligned_8(dst);

    if constexpr (BLOCK_COLS % 2 == 0) {
        if (can_vectorize) {
            const int total_vec = total_elements / 2;
            for (int i = tid; i < total_vec; i += num_threads) {
                int elem_idx = i * 2;
                int local_row = elem_idx / BLOCK_COLS;
                int local_col = elem_idx % BLOCK_COLS;
                int global_row = row_start + local_row;
                int global_col = col_start + local_col;

                if (global_row < max_rows && global_col + 1 < max_cols) {
                    half2 val;
                    val.x = __float2half(src[local_row * BLOCK_COLS + local_col]);
                    val.y = __float2half(src[local_row * BLOCK_COLS + local_col + 1]);
                    *reinterpret_cast<half2*>(&dst[global_row * dst_stride + global_col]) = val;
                } else if (global_row < max_rows) {
                    if (global_col < max_cols)
                        dst[global_row * dst_stride + global_col] =
                            __float2half(src[local_row * BLOCK_COLS + local_col]);
                    if (global_col + 1 < max_cols)
                        dst[global_row * dst_stride + global_col + 1] =
                            __float2half(src[local_row * BLOCK_COLS + local_col + 1]);
                }
            }
        } else {
            for (int i = tid; i < total_elements; i += num_threads) {
                int local_row = i / BLOCK_COLS;
                int local_col = i % BLOCK_COLS;
                int global_row = row_start + local_row;
                int global_col = col_start + local_col;

                if (global_row < max_rows && global_col < max_cols) {
                    dst[global_row * dst_stride + global_col] =
                        __float2half(src[local_row * BLOCK_COLS + local_col]);
                }
            }
        }
    } else {
        for (int i = tid; i < total_elements; i += num_threads) {
            int local_row = i / BLOCK_COLS;
            int local_col = i % BLOCK_COLS;
            int global_row = row_start + local_row;
            int global_col = col_start + local_col;

            if (global_row < max_rows && global_col < max_cols) {
                dst[global_row * dst_stride + global_col] =
                    __float2half(src[local_row * BLOCK_COLS + local_col]);
            }
        }
    }
}

// =============================================================================
// Matmul operations (shared memory tile operations)
// =============================================================================

// Compute C = A @ B^T where A is MxK and B is NxK
// Result C is MxN
// All matrices are in shared memory
template<int M, int N, int K>
__device__ __forceinline__ void matmul_ABt(const float* __restrict__ A,  // MxK
                                           const float* __restrict__ B,  // NxK
                                           float* __restrict__ C,        // MxN
                                           float scale = 1.0f) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int total_elements = M * N;

    for (int i = tid; i < total_elements; i += num_threads) {
        int row = i / N;
        int col = i % N;

        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = sum * scale;
    }
}

// Compute C = A @ B where A is MxK and B is KxN
template<int M, int N, int K>
__device__ __forceinline__ void matmul_AB(const float* __restrict__ A,  // MxK
                                          const float* __restrict__ B,  // KxN
                                          float* __restrict__ C,        // MxN
                                          float scale = 1.0f) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int total_elements = M * N;

    for (int i = tid; i < total_elements; i += num_threads) {
        int row = i / N;
        int col = i % N;

        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum * scale;
    }
}

// Compute C += A @ B (accumulate)
template<int M, int N, int K>
__device__ __forceinline__ void matmul_AB_acc(const float* __restrict__ A,
                                              const float* __restrict__ B, float* __restrict__ C,
                                              float scale = 1.0f) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int total_elements = M * N;

    for (int i = tid; i < total_elements; i += num_threads) {
        int row = i / N;
        int col = i % N;

        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] += sum * scale;
    }
}

// Compute C = A^T @ B where A is KxM and B is KxN
template<int M, int N, int K>
__device__ __forceinline__ void matmul_AtB(const float* __restrict__ A,  // KxM
                                           const float* __restrict__ B,  // KxN
                                           float* __restrict__ C,        // MxN
                                           float scale = 1.0f) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int total_elements = M * N;

    for (int i = tid; i < total_elements; i += num_threads) {
        int row = i / N;
        int col = i % N;

        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < K; k++) {
            sum += A[k * M + row] * B[k * N + col];
        }
        C[row * N + col] = sum * scale;
    }
}

// =============================================================================
// Tiling Configuration
// =============================================================================
// Centralized tiling configuration for FlashAttention kernels.
// Different configurations for forward and backward passes due to memory
// constraints in backward pass requiring smaller blocks.

/// Tiling configuration for forward pass.
/// Uses larger blocks for better memory throughput.
struct ForwardTilingConfig {
    // Standard block sizes for head_dim 32 and 64
    static constexpr int BLOCK_M = 64;  // Q block rows
    static constexpr int BLOCK_N = 64;  // K/V block rows

    // Smaller blocks for head_dim 128 (shared memory constraint)
    static constexpr int BLOCK_M_HD128 = 32;
    static constexpr int BLOCK_N_HD128 = 32;

    static constexpr int NUM_THREADS = 128;
    static constexpr int WARP_SIZE = 32;
};

/// Tiling configuration for backward pass.
/// Uses smaller blocks to accommodate additional gradient tensors in shared memory.
struct BackwardTilingConfig {
    // Standard block sizes for head_dim 32 and 64
    static constexpr int BLOCK_M = 64;  // Q block rows
    static constexpr int BLOCK_N = 64;  // K/V block rows

    // Smaller blocks for head_dim 128 (more aggressive due to dQ, dK, dV)
    static constexpr int BLOCK_M_HD128 = 16;
    static constexpr int BLOCK_N_HD128 = 32;

    static constexpr int NUM_THREADS = 128;
    static constexpr int WARP_SIZE = 32;
};

/// Supported head dimensions - single source of truth.
/// Used for validation and kernel dispatch.
inline constexpr int SUPPORTED_HEAD_DIMS[] = {32, 64, 128};

/// Check if a head_dim value is supported.
inline constexpr bool is_supported_head_dim(int head_dim) {
    return head_dim == 32 || head_dim == 64 || head_dim == 128;
}

// Legacy alias for backward compatibility
using TilingConfig = ForwardTilingConfig;

}  // namespace impl
}  // namespace cuflash
