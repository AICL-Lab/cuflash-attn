// Online Softmax Kernel Implementation
// Provides GPU kernel wrappers for online softmax operations

#include "cuflash/kernels/online_softmax.cuh"
#include "impl/online_softmax.cuh"
#include "primitive_api_utils.cuh"

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
    float* state_smem = reduce_smem + SOFTMAX_THREADS / 32;

    // Process blocks
    if (threadIdx.x == 0) {
        state_smem[0] = -INFINITY;
        state_smem[1] = 0.0f;
    }
    __syncthreads();

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

        block_max = impl::block_reduce_max<SOFTMAX_THREADS>(block_max, reduce_smem);

        // Compute exp sum
        for (int i = threadIdx.x; i < block_len; i += blockDim.x) {
            float val = row_input[start + i];
            block_sum += expf(val - block_max);
        }
        block_sum = impl::block_reduce_sum<SOFTMAX_THREADS>(block_sum, reduce_smem);

        // Update state
        if (threadIdx.x == 0) {
            impl::OnlineSoftmaxState state;
            state.m = state_smem[0];
            state.l = state_smem[1];
            state.update(block_max, block_sum);
            state_smem[0] = state.m;
            state_smem[1] = state.l;
        }
        __syncthreads();
    }

    // Final normalization
    float l_inv = 1.0f / state_smem[1];

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
        float rescale = expf(block_max - state_smem[0]);
        for (int i = threadIdx.x; i < block_len; i += blockDim.x) {
            float val = row_input[start + i];
            row_output[start + i] = expf(val - block_max) * rescale * l_inv;
        }
    }

    // Store logsumexp
    if (threadIdx.x == 0) {
        logsumexp[row] = state_smem[0] + logf(state_smem[1]);
    }
}

// =============================================================================
// Host Entry Points
// =============================================================================

// Init
FlashAttentionError online_softmax_init(float* state_m, float* state_l, int rows,
                                        cudaStream_t stream) {
    FlashAttentionError err = detail::validate_non_null({state_m, state_l});
    if (err != FlashAttentionError::SUCCESS)
        return err;
    err = detail::validate_positive_dimensions({rows});
    if (err != FlashAttentionError::SUCCESS)
        return err;

    int blocks = (rows + SOFTMAX_THREADS - 1) / SOFTMAX_THREADS;
    online_softmax_init_kernel<<<blocks, SOFTMAX_THREADS, 0, stream>>>(state_m, state_l, rows);

    return detail::finish_kernel_launch();
}

// Update
FlashAttentionError online_softmax_update(const float* block_max, const float* block_sum,
                                          float* state_m, float* state_l, int rows,
                                          cudaStream_t stream) {
    FlashAttentionError err = detail::validate_non_null({block_max, block_sum, state_m, state_l});
    if (err != FlashAttentionError::SUCCESS)
        return err;
    err = detail::validate_positive_dimensions({rows});
    if (err != FlashAttentionError::SUCCESS)
        return err;

    int blocks = (rows + SOFTMAX_THREADS - 1) / SOFTMAX_THREADS;
    online_softmax_update_kernel<<<blocks, SOFTMAX_THREADS, 0, stream>>>(block_max, block_sum,
                                                                         state_m, state_l, rows);

    return detail::finish_kernel_launch();
}

// Finalize
FlashAttentionError online_softmax_finalize(const float* state_m, const float* state_l,
                                            float* logsumexp, float* normalizer, int rows,
                                            cudaStream_t stream) {
    FlashAttentionError err = detail::validate_non_null({state_m, state_l, logsumexp, normalizer});
    if (err != FlashAttentionError::SUCCESS)
        return err;
    err = detail::validate_positive_dimensions({rows});
    if (err != FlashAttentionError::SUCCESS)
        return err;

    int blocks = (rows + SOFTMAX_THREADS - 1) / SOFTMAX_THREADS;
    online_softmax_finalize_kernel<<<blocks, SOFTMAX_THREADS, 0, stream>>>(
        state_m, state_l, logsumexp, normalizer, rows);

    return detail::finish_kernel_launch();
}

// Forward (convenience)
FlashAttentionError online_softmax_forward(const float* input, float* output, float* logsumexp,
                                           int rows, int cols, int block_size,
                                           cudaStream_t stream) {
    FlashAttentionError err = detail::validate_non_null({input, output, logsumexp});
    if (err != FlashAttentionError::SUCCESS)
        return err;
    err = detail::validate_positive_dimensions({rows, cols, block_size});
    if (err != FlashAttentionError::SUCCESS)
        return err;

    size_t smem_size = (SOFTMAX_THREADS / 32 + 2) * sizeof(float);

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

    return detail::finish_kernel_launch();
}

// Explicit template instantiations
template __global__ void online_softmax_forward_kernel<32>(const float*, float*, float*, int, int);
template __global__ void online_softmax_forward_kernel<64>(const float*, float*, float*, int, int);
template __global__ void online_softmax_forward_kernel<128>(const float*, float*, float*, int, int);

}  // namespace kernels
}  // namespace cuflash
