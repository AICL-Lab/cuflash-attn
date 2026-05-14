// Matmul Kernel Implementation
// Provides GPU kernel wrappers for tile-level matrix multiplication operations

#include "cuflash/kernels/matmul.cuh"
#include "impl/tile_io.cuh"

namespace cuflash {
namespace kernels {

// =============================================================================
// Matmul Kernels
// =============================================================================
// Each kernel loads matrices to shared memory, computes the product,
// and stores the result back to global memory.

// Common configuration
constexpr int MATMUL_THREADS = 128;

// -----------------------------------------------------------------------------
// C = A @ B^T kernel
// -----------------------------------------------------------------------------

template<int M, int N, int K>
__global__ void matmul_ABt_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, float scale) {
    extern __shared__ float smem[];
    float* A_tile = smem;            // M * K
    float* B_tile = A_tile + M * K;  // N * K
    float* C_tile = B_tile + N * K;  // M * N

    const int tid = threadIdx.x;

    // Load A tile (M x K) - treat as 2D with stride K
    for (int i = tid; i < M * K; i += blockDim.x) {
        A_tile[i] = A[i];
    }

    // Load B tile (N x K) - B is stored as [N, K] for B^T
    for (int i = tid; i < N * K; i += blockDim.x) {
        B_tile[i] = B[i];
    }
    __syncthreads();

    // Compute C = A @ B^T using impl function
    impl::matmul_ABt<M, N, K>(A_tile, B_tile, C_tile, scale);
    __syncthreads();

    // Store C tile
    for (int i = tid; i < M * N; i += blockDim.x) {
        C[i] = C_tile[i];
    }
}

// -----------------------------------------------------------------------------
// C = A @ B kernel
// -----------------------------------------------------------------------------

template<int M, int N, int K>
__global__ void matmul_AB_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                 float* __restrict__ C, float scale) {
    extern __shared__ float smem[];
    float* A_tile = smem;            // M * K
    float* B_tile = A_tile + M * K;  // K * N
    float* C_tile = B_tile + K * N;  // M * N

    const int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += blockDim.x) {
        A_tile[i] = A[i];
    }
    for (int i = tid; i < K * N; i += blockDim.x) {
        B_tile[i] = B[i];
    }
    __syncthreads();

    impl::matmul_AB<M, N, K>(A_tile, B_tile, C_tile, scale);
    __syncthreads();

    for (int i = tid; i < M * N; i += blockDim.x) {
        C[i] = C_tile[i];
    }
}

// -----------------------------------------------------------------------------
// C += A @ B kernel (accumulate)
// -----------------------------------------------------------------------------

template<int M, int N, int K>
__global__ void matmul_AB_acc_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                     float* __restrict__ C, float scale) {
    extern __shared__ float smem[];
    float* A_tile = smem;            // M * K
    float* B_tile = A_tile + M * K;  // K * N
    float* C_tile = B_tile + K * N;  // M * N

    const int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += blockDim.x) {
        A_tile[i] = A[i];
    }
    for (int i = tid; i < K * N; i += blockDim.x) {
        B_tile[i] = B[i];
    }
    // Load C for accumulation
    for (int i = tid; i < M * N; i += blockDim.x) {
        C_tile[i] = C[i];
    }
    __syncthreads();

    impl::matmul_AB_acc<M, N, K>(A_tile, B_tile, C_tile, scale);
    __syncthreads();

    for (int i = tid; i < M * N; i += blockDim.x) {
        C[i] = C_tile[i];
    }
}

// -----------------------------------------------------------------------------
// C = A^T @ B kernel
// -----------------------------------------------------------------------------

template<int M, int N, int K>
__global__ void matmul_AtB_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, float scale) {
    extern __shared__ float smem[];
    float* A_tile = smem;            // K * M (A is stored as [K, M])
    float* B_tile = A_tile + K * M;  // K * N
    float* C_tile = B_tile + K * N;  // M * N

    const int tid = threadIdx.x;

    for (int i = tid; i < K * M; i += blockDim.x) {
        A_tile[i] = A[i];
    }
    for (int i = tid; i < K * N; i += blockDim.x) {
        B_tile[i] = B[i];
    }
    __syncthreads();

    impl::matmul_AtB<M, N, K>(A_tile, B_tile, C_tile, scale);
    __syncthreads();

    for (int i = tid; i < M * N; i += blockDim.x) {
        C[i] = C_tile[i];
    }
}

// =============================================================================
// Host Entry Points
// =============================================================================

// Validation helper
static FlashAttentionError validate_matmul_params(const float* A, const float* B, const float* C) {
    if (!A || !B || !C) {
        return FlashAttentionError::NULL_POINTER;
    }
    return FlashAttentionError::SUCCESS;
}

// C = A @ B^T
template<int M, int N, int K>
FlashAttentionError matmul_ABt(const float* A, const float* B, float* C, float scale,
                               cudaStream_t stream) {
    FlashAttentionError err = validate_matmul_params(A, B, C);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }

    size_t smem_size = (M * K + N * K + M * N) * sizeof(float);
    matmul_ABt_kernel<M, N, K><<<1, MATMUL_THREADS, smem_size, stream>>>(A, B, C, scale);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }
    return FlashAttentionError::SUCCESS;
}

// C = A @ B
template<int M, int N, int K>
FlashAttentionError matmul_AB(const float* A, const float* B, float* C, float scale,
                              cudaStream_t stream) {
    FlashAttentionError err = validate_matmul_params(A, B, C);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }

    size_t smem_size = (M * K + K * N + M * N) * sizeof(float);
    matmul_AB_kernel<M, N, K><<<1, MATMUL_THREADS, smem_size, stream>>>(A, B, C, scale);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }
    return FlashAttentionError::SUCCESS;
}

// C += A @ B
template<int M, int N, int K>
FlashAttentionError matmul_AB_acc(const float* A, const float* B, float* C, float scale,
                                  cudaStream_t stream) {
    FlashAttentionError err = validate_matmul_params(A, B, C);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }

    size_t smem_size = (M * K + K * N + M * N) * sizeof(float);
    matmul_AB_acc_kernel<M, N, K><<<1, MATMUL_THREADS, smem_size, stream>>>(A, B, C, scale);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }
    return FlashAttentionError::SUCCESS;
}

// C = A^T @ B
template<int M, int N, int K>
FlashAttentionError matmul_AtB(const float* A, const float* B, float* C, float scale,
                               cudaStream_t stream) {
    FlashAttentionError err = validate_matmul_params(A, B, C);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }

    size_t smem_size = (K * M + K * N + M * N) * sizeof(float);
    matmul_AtB_kernel<M, N, K><<<1, MATMUL_THREADS, smem_size, stream>>>(A, B, C, scale);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }
    return FlashAttentionError::SUCCESS;
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

// Common tile sizes for FlashAttention
// M (Q rows), N (KV rows), K (head_dim): (64, 64, 32), (64, 64, 64), (32, 32, 128)

// 64x64x32 (head_dim=32)
template FlashAttentionError matmul_ABt<64, 64, 32>(const float*, const float*, float*, float,
                                                    cudaStream_t);
template FlashAttentionError matmul_AB<64, 64, 32>(const float*, const float*, float*, float,
                                                   cudaStream_t);
template FlashAttentionError matmul_AB_acc<64, 64, 32>(const float*, const float*, float*, float,
                                                       cudaStream_t);
template FlashAttentionError matmul_AtB<64, 64, 32>(const float*, const float*, float*, float,
                                                    cudaStream_t);

// 64x64x64 (head_dim=64)
template FlashAttentionError matmul_ABt<64, 64, 64>(const float*, const float*, float*, float,
                                                    cudaStream_t);
template FlashAttentionError matmul_AB<64, 64, 64>(const float*, const float*, float*, float,
                                                   cudaStream_t);
template FlashAttentionError matmul_AB_acc<64, 64, 64>(const float*, const float*, float*, float,
                                                       cudaStream_t);
template FlashAttentionError matmul_AtB<64, 64, 64>(const float*, const float*, float*, float,
                                                    cudaStream_t);

// 32x32x128 (head_dim=128, smaller tiles)
template FlashAttentionError matmul_ABt<32, 32, 128>(const float*, const float*, float*, float,
                                                     cudaStream_t);
template FlashAttentionError matmul_AB<32, 32, 128>(const float*, const float*, float*, float,
                                                    cudaStream_t);
template FlashAttentionError matmul_AB_acc<32, 32, 128>(const float*, const float*, float*, float,
                                                        cudaStream_t);
template FlashAttentionError matmul_AtB<32, 32, 128>(const float*, const float*, float*, float,
                                                     cudaStream_t);

// 64x64x128 (head_dim=128, larger tiles - may need more shared memory)
template FlashAttentionError matmul_ABt<64, 64, 128>(const float*, const float*, float*, float,
                                                     cudaStream_t);
template FlashAttentionError matmul_AB<64, 64, 128>(const float*, const float*, float*, float,
                                                    cudaStream_t);
template FlashAttentionError matmul_AB_acc<64, 64, 128>(const float*, const float*, float*, float,
                                                        cudaStream_t);
template FlashAttentionError matmul_AtB<64, 64, 128>(const float*, const float*, float*, float,
                                                     cudaStream_t);

// Additional sizes for flexibility
// 32x64 variants
template FlashAttentionError matmul_ABt<32, 64, 64>(const float*, const float*, float*, float,
                                                    cudaStream_t);
template FlashAttentionError matmul_AB<32, 64, 64>(const float*, const float*, float*, float,
                                                   cudaStream_t);
template FlashAttentionError matmul_AB_acc<32, 64, 64>(const float*, const float*, float*, float,
                                                       cudaStream_t);
template FlashAttentionError matmul_AtB<32, 64, 64>(const float*, const float*, float*, float,
                                                    cudaStream_t);

// 64x32 variants
template FlashAttentionError matmul_ABt<64, 32, 64>(const float*, const float*, float*, float,
                                                    cudaStream_t);
template FlashAttentionError matmul_AB<64, 32, 64>(const float*, const float*, float*, float,
                                                   cudaStream_t);
template FlashAttentionError matmul_AB_acc<64, 32, 64>(const float*, const float*, float*, float,
                                                       cudaStream_t);
template FlashAttentionError matmul_AtB<64, 32, 64>(const float*, const float*, float*, float,
                                                    cudaStream_t);

}  // namespace kernels
}  // namespace cuflash
