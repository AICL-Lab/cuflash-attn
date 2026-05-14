// Online Softmax Kernel Implementation
// Provides GPU kernel wrappers for online softmax operations

#include "cuflash/kernels/online_softmax.cuh"
#include "impl/online_softmax.cuh"

namespace cuflash {
namespace kernels {

// =============================================================================
// Kernel Definitions
// =============================================================================

constexpr int SOFTMAX_THREADS = 128;

// -----------------------------------------------------------------------------
// Init Kernel
// -----------------------------------------------------------------------------

__global__ void online_softmax_init_kernel(float* __restrict__ state_m, float* __restrict__ state_l,
                                           int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        state_m[row] = -INFINITY;
        state_l[row] = 0.0f;
    }
}

// -----------------------------------------------------------------------------
// Update Kernel
// -----------------------------------------------------------------------------

__global__ void online_softmax_update_kernel(const float* __restrict__ block_max,
                                             const float* __restrict__ block_sum,
                                             float* __restrict__ state_m,
                                             float* __restrict__ state_l, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        impl::OnlineSoftmaxState state;
        state.m = state_m[row];
        state.l = state_l[row];
        state.update(block_max[row], block_sum[row]);
        state_m[row] = state.m;
        state_l[row] = state.l;
    }
}

// -----------------------------------------------------------------------------
// Finalize Kernel
// -----------------------------------------------------------------------------

__global__ void online_softmax_finalize_kernel(const float* __restrict__ state_m,
                                               const float* __restrict__ state_l,
                                               float* __restrict__ logsumexp,
                                               float* __restrict__ normalizer, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        impl::OnlineSoftmaxState state;
        state.m = state_m[row];
        state.l = state_l[row];
        logsumexp[row] = state.logsumexp();
        normalizer[row] = state.get_normalizer();
    }
}

// -----------------------------------------------------------------------------
// Forward Kernel (complete operation)
// -----------------------------------------------------------------------------

template<int BLOCK_SIZE>
__global__ void online_softmax_forward_kernel(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              float* __restrict__ logsumexp, int rows, int cols) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    if (row >= rows)
        return;

    // Use shared memory for reductions
    float* reduce_smem = smem;

    // Process blocks
    impl::OnlineSoftmaxState state;
    state.init();

    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    int num_blocks = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int b = 0; b < num_blocks; b++) {
        int start = b * BLOCK_SIZE;
        int end = min(start + BLOCK_SIZE, cols);
        int block_len = end - start;

        // Compute block max and sum
        float block_max = -INFINITY;
        float block_sum = 0.0f;

        // Each thread processes multiple elements
        for (int i = threadIdx.x; i < block_len; i += blockDim.x) {
            float val = row_input[start + i];
            block_max = fmaxf(block_max, val);
        }

        block_max = impl::block_reduce_sum<SOFTMAX_THREADS>(block_max, reduce_smem);
        // Actually need max reduction - reuse shared memory
        block_max = impl::block_reduce_max<SOFTMAX_THREADS>(block_max, reduce_smem);

        // Compute exp sum
        for (int i = threadIdx.x; i < block_len; i += blockDim.x) {
            float val = row_input[start + i];
            block_sum += expf(val - block_max);
        }
        block_sum = impl::block_reduce_sum<SOFTMAX_THREADS>(block_sum, reduce_smem);

        // Update state
        if (threadIdx.x == 0) {
            state.update(block_max, block_sum);
        }
        __syncthreads();
    }

    // Final normalization
    float l_inv = state.get_normalizer();

    for (int b = 0; b < num_blocks; b++) {
        int start = b * BLOCK_SIZE;
        int end = min(start + BLOCK_SIZE, cols);
        int block_len = end - start;

        // Need to recompute block max for this block
        float block_max = -INFINITY;
        for (int i = threadIdx.x; i < block_len; i += blockDim.x) {
            float val = row_input[start + i];
            block_max = fmaxf(block_max, val);
        }
        block_max = impl::block_reduce_max<SOFTMAX_THREADS>(block_max, reduce_smem);

        // Compute and store output
        float rescale = expf(block_max - state.m);
        for (int i = threadIdx.x; i < block_len; i += blockDim.x) {
            float val = row_input[start + i];
            row_output[start + i] = expf(val - block_max) * rescale * l_inv;
        }
    }

    // Store logsumexp
    if (threadIdx.x == 0) {
        logsumexp[row] = state.logsumexp();
    }
}

// =============================================================================
// Host Entry Points
// =============================================================================

// Validation helper
static FlashAttentionError validate_online_softmax_params(const float* ptr, int rows) {
    if (!ptr) {
        return FlashAttentionError::NULL_POINTER;
    }
    if (rows <= 0) {
        return FlashAttentionError::INVALID_DIMENSION;
    }
    return FlashAttentionError::SUCCESS;
}

// Init
FlashAttentionError online_softmax_init(float* state_m, float* state_l, int rows,
                                        cudaStream_t stream) {
    FlashAttentionError err = validate_online_softmax_params(state_m, rows);
    if (err != FlashAttentionError::SUCCESS)
        return err;
    if (!state_l)
        return FlashAttentionError::NULL_POINTER;

    int blocks = (rows + SOFTMAX_THREADS - 1) / SOFTMAX_THREADS;
    online_softmax_init_kernel<<<blocks, SOFTMAX_THREADS, 0, stream>>>(state_m, state_l, rows);

    cudaError_t cuda_err = cudaGetLastError();
    return (cuda_err == cudaSuccess) ? FlashAttentionError::SUCCESS
                                     : FlashAttentionError::CUDA_ERROR;
}

// Update
FlashAttentionError online_softmax_update(const float* block_max, const float* block_sum,
                                          float* state_m, float* state_l, int rows,
                                          cudaStream_t stream) {
    FlashAttentionError err = validate_online_softmax_params(block_max, rows);
    if (err != FlashAttentionError::SUCCESS)
        return err;
    if (!block_sum || !state_m || !state_l)
        return FlashAttentionError::NULL_POINTER;

    int blocks = (rows + SOFTMAX_THREADS - 1) / SOFTMAX_THREADS;
    online_softmax_update_kernel<<<blocks, SOFTMAX_THREADS, 0, stream>>>(block_max, block_sum,
                                                                         state_m, state_l, rows);

    cudaError_t cuda_err = cudaGetLastError();
    return (cuda_err == cudaSuccess) ? FlashAttentionError::SUCCESS
                                     : FlashAttentionError::CUDA_ERROR;
}

// Finalize
FlashAttentionError online_softmax_finalize(const float* state_m, const float* state_l,
                                            float* logsumexp, float* normalizer, int rows,
                                            cudaStream_t stream) {
    FlashAttentionError err = validate_online_softmax_params(state_m, rows);
    if (err != FlashAttentionError::SUCCESS)
        return err;
    if (!state_l || !logsumexp)
        return FlashAttentionError::NULL_POINTER;

    int blocks = (rows + SOFTMAX_THREADS - 1) / SOFTMAX_THREADS;
    online_softmax_finalize_kernel<<<blocks, SOFTMAX_THREADS, 0, stream>>>(
        state_m, state_l, logsumexp, normalizer, rows);

    cudaError_t cuda_err = cudaGetLastError();
    return (cuda_err == cudaSuccess) ? FlashAttentionError::SUCCESS
                                     : FlashAttentionError::CUDA_ERROR;
}

// Forward (convenience)
FlashAttentionError online_softmax_forward(const float* input, float* output, float* logsumexp,
                                           int rows, int cols, int block_size,
                                           cudaStream_t stream) {
    if (!input || !output || !logsumexp) {
        return FlashAttentionError::NULL_POINTER;
    }
    if (rows <= 0 || cols <= 0 || block_size <= 0) {
        return FlashAttentionError::INVALID_DIMENSION;
    }

    size_t smem_size = SOFTMAX_THREADS / 32 * sizeof(float);

    // Dispatch based on block size
    if (block_size <= 32) {
        online_softmax_forward_kernel<32>
            <<<rows, SOFTMAX_THREADS, smem_size, stream>>>(input, output, logsumexp, rows, cols);
    } else if (block_size <= 64) {
        online_softmax_forward_kernel<64>
            <<<rows, SOFTMAX_THREADS, smem_size, stream>>>(input, output, logsumexp, rows, cols);
    } else {
        online_softmax_forward_kernel<128>
            <<<rows, SOFTMAX_THREADS, smem_size, stream>>>(input, output, logsumexp, rows, cols);
    }

    cudaError_t cuda_err = cudaGetLastError();
    return (cuda_err == cudaSuccess) ? FlashAttentionError::SUCCESS
                                     : FlashAttentionError::CUDA_ERROR;
}

// Explicit template instantiations
template __global__ void online_softmax_forward_kernel<32>(const float*, float*, float*, int, int);
template __global__ void online_softmax_forward_kernel<64>(const float*, float*, float*, int, int);
template __global__ void online_softmax_forward_kernel<128>(const float*, float*, float*, int, int);

}  // namespace kernels
}  // namespace cuflash
