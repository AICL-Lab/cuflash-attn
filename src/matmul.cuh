#pragma once

#include <cuda_runtime.h>

namespace cuflash {

// Tiling configuration
struct TilingConfig {
    static constexpr int BLOCK_M = 64;    // Q block rows
    static constexpr int BLOCK_N = 64;    // K/V block rows
    static constexpr int BLOCK_K = 64;    // Head dimension tile
    static constexpr int NUM_THREADS = 128;
    static constexpr int WARP_SIZE = 32;
};

// Load a tile from global memory to shared memory
// Handles boundary conditions when seq_len is not divisible by block size
// Uses float4 vectorized loads when alignment permits
template<int BLOCK_ROWS, int BLOCK_COLS>
__device__ __forceinline__ void load_tile_to_shared(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int row_start,
    int col_start,
    int max_rows,
    int max_cols,
    int src_stride
) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int total_elements = BLOCK_ROWS * BLOCK_COLS;
    
    // Use float4 vectorized loads when BLOCK_COLS is divisible by 4 and col_start is aligned
    if constexpr (BLOCK_COLS % 4 == 0) {
        const int total_vec = total_elements / 4;
        for (int i = tid; i < total_vec; i += num_threads) {
            int elem_idx = i * 4;
            int local_row = elem_idx / BLOCK_COLS;
            int local_col = elem_idx % BLOCK_COLS;
            int global_row = row_start + local_row;
            int global_col = col_start + local_col;
            
            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_row < max_rows && global_col + 3 < max_cols) {
                val = *reinterpret_cast<const float4*>(&src[global_row * src_stride + global_col]);
            } else if (global_row < max_rows) {
                if (global_col < max_cols) val.x = src[global_row * src_stride + global_col];
                if (global_col + 1 < max_cols) val.y = src[global_row * src_stride + global_col + 1];
                if (global_col + 2 < max_cols) val.z = src[global_row * src_stride + global_col + 2];
                if (global_col + 3 < max_cols) val.w = src[global_row * src_stride + global_col + 3];
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
                dst[local_row * BLOCK_COLS + local_col] = src[global_row * src_stride + global_col];
            } else {
                dst[local_row * BLOCK_COLS + local_col] = 0.0f;
            }
        }
    }
}

// Store a tile from shared memory to global memory
// Uses float4 vectorized stores when alignment permits
template<int BLOCK_ROWS, int BLOCK_COLS>
__device__ __forceinline__ void store_tile_from_shared(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int row_start,
    int col_start,
    int max_rows,
    int max_cols,
    int dst_stride
) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int total_elements = BLOCK_ROWS * BLOCK_COLS;
    
    if constexpr (BLOCK_COLS % 4 == 0) {
        const int total_vec = total_elements / 4;
        for (int i = tid; i < total_vec; i += num_threads) {
            int elem_idx = i * 4;
            int local_row = elem_idx / BLOCK_COLS;
            int local_col = elem_idx % BLOCK_COLS;
            int global_row = row_start + local_row;
            int global_col = col_start + local_col;
            
            if (global_row < max_rows && global_col + 3 < max_cols) {
                float4 val = *reinterpret_cast<const float4*>(&src[local_row * BLOCK_COLS + local_col]);
                *reinterpret_cast<float4*>(&dst[global_row * dst_stride + global_col]) = val;
            } else if (global_row < max_rows) {
                if (global_col < max_cols) dst[global_row * dst_stride + global_col] = src[local_row * BLOCK_COLS + local_col];
                if (global_col + 1 < max_cols) dst[global_row * dst_stride + global_col + 1] = src[local_row * BLOCK_COLS + local_col + 1];
                if (global_col + 2 < max_cols) dst[global_row * dst_stride + global_col + 2] = src[local_row * BLOCK_COLS + local_col + 2];
                if (global_col + 3 < max_cols) dst[global_row * dst_stride + global_col + 3] = src[local_row * BLOCK_COLS + local_col + 3];
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

// Compute C = A @ B^T where A is MxK and B is NxK
// Result C is MxN
// All matrices are in shared memory
template<int M, int N, int K>
__device__ __forceinline__ void matmul_ABt(
    const float* __restrict__ A,  // MxK
    const float* __restrict__ B,  // NxK
    float* __restrict__ C,        // MxN
    float scale = 1.0f
) {
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
__device__ __forceinline__ void matmul_AB(
    const float* __restrict__ A,  // MxK
    const float* __restrict__ B,  // KxN
    float* __restrict__ C,        // MxN
    float scale = 1.0f
) {
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
__device__ __forceinline__ void matmul_AB_acc(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    float scale = 1.0f
) {
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
__device__ __forceinline__ void matmul_AtB(
    const float* __restrict__ A,  // KxM
    const float* __restrict__ B,  // KxN
    float* __restrict__ C,        // MxN
    float scale = 1.0f
) {
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

} // namespace cuflash
