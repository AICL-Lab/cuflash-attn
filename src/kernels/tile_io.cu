// Tile I/O Kernel Implementation
// Provides GPU kernel wrappers for tile load/store operations

#include "cuflash/kernels/tile_io.cuh"
#include "impl/tile_io.cuh"

namespace cuflash {
namespace kernels {

// =============================================================================
// Load Kernels
// =============================================================================

template<int BLOCK_ROWS, int BLOCK_COLS>
__global__ void load_tile_kernel_fp32(const float* __restrict__ src, float* __restrict__ dst,
                                      int row_start, int col_start, int max_rows, int max_cols,
                                      int src_stride) {
    extern __shared__ float smem[];
    impl::load_tile_to_shared<BLOCK_ROWS, BLOCK_COLS>(src, smem, row_start, col_start, max_rows,
                                                      max_cols, src_stride);
    __syncthreads();

    // Copy from shared memory to output
    const int tid = threadIdx.x;
    const int total = BLOCK_ROWS * BLOCK_COLS;
    for (int i = tid; i < total; i += blockDim.x) {
        dst[i] = smem[i];
    }
}

template<int BLOCK_ROWS, int BLOCK_COLS>
__global__ void load_tile_kernel_fp16(const half* __restrict__ src, float* __restrict__ dst,
                                      int row_start, int col_start, int max_rows, int max_cols,
                                      int src_stride) {
    extern __shared__ float smem[];
    impl::load_tile_to_shared<BLOCK_ROWS, BLOCK_COLS>(src, smem, row_start, col_start, max_rows,
                                                      max_cols, src_stride);
    __syncthreads();

    const int tid = threadIdx.x;
    const int total = BLOCK_ROWS * BLOCK_COLS;
    for (int i = tid; i < total; i += blockDim.x) {
        dst[i] = smem[i];
    }
}

// =============================================================================
// Store Kernels
// =============================================================================

template<int BLOCK_ROWS, int BLOCK_COLS>
__global__ void store_tile_kernel_fp32(const float* __restrict__ src, float* __restrict__ dst,
                                       int row_start, int col_start, int max_rows, int max_cols,
                                       int dst_stride) {
    extern __shared__ float smem[];

    // Copy input to shared memory
    const int tid = threadIdx.x;
    const int total = BLOCK_ROWS * BLOCK_COLS;
    for (int i = tid; i < total; i += blockDim.x) {
        smem[i] = src[i];
    }
    __syncthreads();

    impl::store_tile_from_shared<BLOCK_ROWS, BLOCK_COLS>(smem, dst, row_start, col_start, max_rows,
                                                         max_cols, dst_stride);
}

template<int BLOCK_ROWS, int BLOCK_COLS>
__global__ void store_tile_kernel_fp16(const float* __restrict__ src, half* __restrict__ dst,
                                       int row_start, int col_start, int max_rows, int max_cols,
                                       int dst_stride) {
    extern __shared__ float smem[];

    const int tid = threadIdx.x;
    const int total = BLOCK_ROWS * BLOCK_COLS;
    for (int i = tid; i < total; i += blockDim.x) {
        smem[i] = src[i];
    }
    __syncthreads();

    impl::store_tile_from_shared<BLOCK_ROWS, BLOCK_COLS>(smem, dst, row_start, col_start, max_rows,
                                                         max_cols, dst_stride);
}

// =============================================================================
// Round-trip Kernel
// =============================================================================

template<int BLOCK_ROWS, int BLOCK_COLS>
__global__ void load_store_roundtrip_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                            int row_start, int col_start, int max_rows,
                                            int max_cols, int stride) {
    extern __shared__ float smem[];

    // Load
    impl::load_tile_to_shared<BLOCK_ROWS, BLOCK_COLS>(src, smem, row_start, col_start, max_rows,
                                                      max_cols, stride);
    __syncthreads();

    // Store back
    impl::store_tile_from_shared<BLOCK_ROWS, BLOCK_COLS>(smem, dst, row_start, col_start, max_rows,
                                                         max_cols, stride);
}

// =============================================================================
// Host Entry Points
// =============================================================================

// Common validation
static FlashAttentionError validate_tile_params(int row_start, int col_start, int max_rows,
                                                int max_cols, int stride) {
    if (max_rows <= 0 || max_cols <= 0 || stride <= 0) {
        return FlashAttentionError::INVALID_DIMENSION;
    }
    if (row_start < 0 || col_start < 0) {
        return FlashAttentionError::INVALID_DIMENSION;
    }
    if (row_start >= max_rows || col_start >= max_cols) {
        return FlashAttentionError::INVALID_DIMENSION;
    }
    return FlashAttentionError::SUCCESS;
}

// FP32 Load
template<int BLOCK_ROWS, int BLOCK_COLS>
FlashAttentionError load_tile(const float* src, float* dst, int row_start, int col_start,
                              int max_rows, int max_cols, int src_stride, cudaStream_t stream) {
    if (!src || !dst) {
        return FlashAttentionError::NULL_POINTER;
    }

    FlashAttentionError err =
        validate_tile_params(row_start, col_start, max_rows, max_cols, src_stride);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }

    size_t smem_size = BLOCK_ROWS * BLOCK_COLS * sizeof(float);
    load_tile_kernel_fp32<BLOCK_ROWS, BLOCK_COLS><<<1, 128, smem_size, stream>>>(
        src, dst, row_start, col_start, max_rows, max_cols, src_stride);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }
    return FlashAttentionError::SUCCESS;
}

// FP16 Load
template<int BLOCK_ROWS, int BLOCK_COLS>
FlashAttentionError load_tile(const half* src, float* dst, int row_start, int col_start,
                              int max_rows, int max_cols, int src_stride, cudaStream_t stream) {
    if (!src || !dst) {
        return FlashAttentionError::NULL_POINTER;
    }

    FlashAttentionError err =
        validate_tile_params(row_start, col_start, max_rows, max_cols, src_stride);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }

    size_t smem_size = BLOCK_ROWS * BLOCK_COLS * sizeof(float);
    load_tile_kernel_fp16<BLOCK_ROWS, BLOCK_COLS><<<1, 128, smem_size, stream>>>(
        src, dst, row_start, col_start, max_rows, max_cols, src_stride);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }
    return FlashAttentionError::SUCCESS;
}

// FP32 Store
template<int BLOCK_ROWS, int BLOCK_COLS>
FlashAttentionError store_tile(const float* src, float* dst, int row_start, int col_start,
                               int max_rows, int max_cols, int dst_stride, cudaStream_t stream) {
    if (!src || !dst) {
        return FlashAttentionError::NULL_POINTER;
    }

    FlashAttentionError err =
        validate_tile_params(row_start, col_start, max_rows, max_cols, dst_stride);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }

    size_t smem_size = BLOCK_ROWS * BLOCK_COLS * sizeof(float);
    store_tile_kernel_fp32<BLOCK_ROWS, BLOCK_COLS><<<1, 128, smem_size, stream>>>(
        src, dst, row_start, col_start, max_rows, max_cols, dst_stride);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }
    return FlashAttentionError::SUCCESS;
}

// FP16 Store
template<int BLOCK_ROWS, int BLOCK_COLS>
FlashAttentionError store_tile(const float* src, half* dst, int row_start, int col_start,
                               int max_rows, int max_cols, int dst_stride, cudaStream_t stream) {
    if (!src || !dst) {
        return FlashAttentionError::NULL_POINTER;
    }

    FlashAttentionError err =
        validate_tile_params(row_start, col_start, max_rows, max_cols, dst_stride);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }

    size_t smem_size = BLOCK_ROWS * BLOCK_COLS * sizeof(float);
    store_tile_kernel_fp16<BLOCK_ROWS, BLOCK_COLS><<<1, 128, smem_size, stream>>>(
        src, dst, row_start, col_start, max_rows, max_cols, dst_stride);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }
    return FlashAttentionError::SUCCESS;
}

// Round-trip
template<int BLOCK_ROWS, int BLOCK_COLS>
FlashAttentionError load_store_tile_roundtrip(const float* src, float* dst, int row_start,
                                              int col_start, int max_rows, int max_cols, int stride,
                                              cudaStream_t stream) {
    if (!src || !dst) {
        return FlashAttentionError::NULL_POINTER;
    }

    FlashAttentionError err =
        validate_tile_params(row_start, col_start, max_rows, max_cols, stride);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }

    size_t smem_size = BLOCK_ROWS * BLOCK_COLS * sizeof(float);
    load_store_roundtrip_kernel<BLOCK_ROWS, BLOCK_COLS>
        <<<1, 128, smem_size, stream>>>(src, dst, row_start, col_start, max_rows, max_cols, stride);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }
    return FlashAttentionError::SUCCESS;
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

// Common tile sizes used in FlashAttention: 32, 64, 128
// HEAD_DIM values: 32, 64, 128

// 64x64 tiles (standard)
template FlashAttentionError load_tile<64, 64>(const float*, float*, int, int, int, int, int,
                                               cudaStream_t);
template FlashAttentionError load_tile<64, 64>(const half*, float*, int, int, int, int, int,
                                               cudaStream_t);
template FlashAttentionError store_tile<64, 64>(const float*, float*, int, int, int, int, int,
                                                cudaStream_t);
template FlashAttentionError store_tile<64, 64>(const float*, half*, int, int, int, int, int,
                                                cudaStream_t);
template FlashAttentionError load_store_tile_roundtrip<64, 64>(const float*, float*, int, int, int,
                                                               int, int, cudaStream_t);

// 64x32 tiles (HEAD_DIM=32)
template FlashAttentionError load_tile<64, 32>(const float*, float*, int, int, int, int, int,
                                               cudaStream_t);
template FlashAttentionError load_tile<64, 32>(const half*, float*, int, int, int, int, int,
                                               cudaStream_t);
template FlashAttentionError store_tile<64, 32>(const float*, float*, int, int, int, int, int,
                                                cudaStream_t);
template FlashAttentionError store_tile<64, 32>(const float*, half*, int, int, int, int, int,
                                                cudaStream_t);

// 64x128 tiles (HEAD_DIM=128)
template FlashAttentionError load_tile<64, 128>(const float*, float*, int, int, int, int, int,
                                                cudaStream_t);
template FlashAttentionError load_tile<64, 128>(const half*, float*, int, int, int, int, int,
                                                cudaStream_t);
template FlashAttentionError store_tile<64, 128>(const float*, float*, int, int, int, int, int,
                                                 cudaStream_t);
template FlashAttentionError store_tile<64, 128>(const float*, half*, int, int, int, int, int,
                                                 cudaStream_t);

// 32x32 tiles (smaller tiles for HEAD_DIM=128)
template FlashAttentionError load_tile<32, 32>(const float*, float*, int, int, int, int, int,
                                               cudaStream_t);
template FlashAttentionError load_tile<32, 32>(const half*, float*, int, int, int, int, int,
                                               cudaStream_t);
template FlashAttentionError store_tile<32, 32>(const float*, float*, int, int, int, int, int,
                                                cudaStream_t);
template FlashAttentionError store_tile<32, 32>(const float*, half*, int, int, int, int, int,
                                                cudaStream_t);

// 32x64 tiles
template FlashAttentionError load_tile<32, 64>(const float*, float*, int, int, int, int, int,
                                               cudaStream_t);
template FlashAttentionError load_tile<32, 64>(const half*, float*, int, int, int, int, int,
                                               cudaStream_t);
template FlashAttentionError store_tile<32, 64>(const float*, float*, int, int, int, int, int,
                                                cudaStream_t);
template FlashAttentionError store_tile<32, 64>(const float*, half*, int, int, int, int, int,
                                                cudaStream_t);

// 32x128 tiles (HEAD_DIM=128 with smaller BLOCK_M)
template FlashAttentionError load_tile<32, 128>(const float*, float*, int, int, int, int, int,
                                                cudaStream_t);
template FlashAttentionError load_tile<32, 128>(const half*, float*, int, int, int, int, int,
                                                cudaStream_t);
template FlashAttentionError store_tile<32, 128>(const float*, float*, int, int, int, int, int,
                                                 cudaStream_t);
template FlashAttentionError store_tile<32, 128>(const float*, half*, int, int, int, int, int,
                                                 cudaStream_t);

}  // namespace kernels
}  // namespace cuflash
