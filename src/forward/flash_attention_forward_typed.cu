// Unified Flash Attention Forward Kernel - FP32/FP16 Template
// This file consolidates flash_attention_forward.cu and flash_attention_fp16.cu

#include <float.h>

#include "cuflash/flash_attention.h"
#include "impl/tile_io.cuh"
#include "kernel_launch_utils.cuh"

namespace cuflash {

// Unified forward kernel template
// InputT: float or half - the type of input/output tensors
// Computation always uses float internally for numerical stability
// BLOCK_M: number of Q rows per block
// BLOCK_N: number of K/V rows per block
// HEAD_DIM: dimension of each head
template<typename InputT, int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128)
    flash_attention_forward_kernel(const InputT* __restrict__ Q, const InputT* __restrict__ K,
                                   const InputT* __restrict__ V, InputT* __restrict__ O,
                                   InputT* __restrict__ L, int seq_len, float scale, bool causal) {
    // Block indices
    const int batch_head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;

    // Pointers for this batch/head
    const InputT* Q_ptr = Q + batch_head_idx * seq_len * HEAD_DIM;
    const InputT* K_ptr = K + batch_head_idx * seq_len * HEAD_DIM;
    const InputT* V_ptr = V + batch_head_idx * seq_len * HEAD_DIM;
    InputT* O_ptr = O + batch_head_idx * seq_len * HEAD_DIM;
    InputT* L_ptr = L + batch_head_idx * seq_len;

    // Q block start row
    const int q_start = q_block_idx * BLOCK_M;
    if (q_start >= seq_len)
        return;

    // Number of K/V blocks
    const int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;

    // Shared memory allocation - always float for computation
    extern __shared__ float smem[];
    float* Q_tile = smem;                         // BLOCK_M x HEAD_DIM
    float* K_tile = Q_tile + BLOCK_M * HEAD_DIM;  // BLOCK_N x HEAD_DIM
    float* V_tile = K_tile + BLOCK_N * HEAD_DIM;  // BLOCK_N x HEAD_DIM
    float* S_tile = V_tile + BLOCK_N * HEAD_DIM;  // BLOCK_M x BLOCK_N
    float* O_tile = S_tile + BLOCK_M * BLOCK_N;   // BLOCK_M x HEAD_DIM
    float* m_tile = O_tile + BLOCK_M * HEAD_DIM;  // BLOCK_M
    float* l_tile = m_tile + BLOCK_M;             // BLOCK_M

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Load Q tile (handles FP32/FP16 conversion via template overload)
    impl::load_tile_to_shared<BLOCK_M, HEAD_DIM>(Q_ptr, Q_tile, q_start, 0, seq_len, HEAD_DIM,
                                                 HEAD_DIM);

    // Initialize O, m, l
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += num_threads) {
        O_tile[i] = 0.0f;
    }
    for (int i = tid; i < BLOCK_M; i += num_threads) {
        m_tile[i] = -INFINITY;
        l_tile[i] = 0.0f;
    }
    __syncthreads();

    // Iterate over K/V blocks
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;

        // Causal: skip blocks that are entirely in the future
        if (causal && kv_start > q_start + BLOCK_M - 1) {
            break;
        }

        // Load K and V tiles (handles FP32/FP16 conversion)
        impl::load_tile_to_shared<BLOCK_N, HEAD_DIM>(K_ptr, K_tile, kv_start, 0, seq_len, HEAD_DIM,
                                                     HEAD_DIM);
        impl::load_tile_to_shared<BLOCK_N, HEAD_DIM>(V_ptr, V_tile, kv_start, 0, seq_len, HEAD_DIM,
                                                     HEAD_DIM);
        __syncthreads();

        // Compute S = Q @ K^T * scale
        impl::matmul_ABt<BLOCK_M, BLOCK_N, HEAD_DIM>(Q_tile, K_tile, S_tile, scale);
        __syncthreads();

        // Apply causal mask if needed
        if (causal) {
            for (int i = tid; i < BLOCK_M * BLOCK_N; i += num_threads) {
                int q_idx = i / BLOCK_N;
                int k_idx = i % BLOCK_N;
                int global_q = q_start + q_idx;
                int global_k = kv_start + k_idx;
                if (global_k > global_q) {
                    S_tile[i] = -INFINITY;
                }
            }
            __syncthreads();
        }

        // Compute row-wise max and sum for this block
        for (int row = tid; row < BLOCK_M; row += num_threads) {
            if (q_start + row >= seq_len)
                continue;

            float row_max = -INFINITY;
            for (int j = 0; j < BLOCK_N; j++) {
                if (kv_start + j < seq_len) {
                    row_max = fmaxf(row_max, S_tile[row * BLOCK_N + j]);
                }
            }

            float row_sum = 0.0f;
            for (int j = 0; j < BLOCK_N; j++) {
                if (kv_start + j < seq_len) {
                    S_tile[row * BLOCK_N + j] = expf(S_tile[row * BLOCK_N + j] - row_max);
                    row_sum += S_tile[row * BLOCK_N + j];
                } else {
                    S_tile[row * BLOCK_N + j] = 0.0f;
                }
            }

            // Update online softmax state
            float m_old = m_tile[row];
            float l_old = l_tile[row];
            float m_new = fmaxf(m_old, row_max);
            float l_new = l_old * expf(m_old - m_new) + row_sum * expf(row_max - m_new);

            // Rescale existing O
            float rescale = expf(m_old - m_new);
            for (int d = 0; d < HEAD_DIM; d++) {
                O_tile[row * HEAD_DIM + d] *= rescale;
            }

            // Add contribution from this block: P @ V
            float p_scale = expf(row_max - m_new);
            for (int d = 0; d < HEAD_DIM; d++) {
                float sum = 0.0f;
                for (int j = 0; j < BLOCK_N; j++) {
                    sum += S_tile[row * BLOCK_N + j] * V_tile[j * HEAD_DIM + d];
                }
                O_tile[row * HEAD_DIM + d] += sum * p_scale;
            }

            m_tile[row] = m_new;
            l_tile[row] = l_new;
        }
        __syncthreads();
    }

    // Final normalization and write output
    for (int row = tid; row < BLOCK_M; row += num_threads) {
        int global_row = q_start + row;
        if (global_row >= seq_len)
            continue;

        float l_inv = 1.0f / l_tile[row];
        for (int d = 0; d < HEAD_DIM; d++) {
            O_ptr[global_row * HEAD_DIM + d] =
                InputT(O_tile[row * HEAD_DIM + d] * l_inv);  // Implicit float->half if needed
        }

        // Store logsumexp for backward pass
        L_ptr[global_row] = InputT(m_tile[row] + logf(l_tile[row]));
    }
}

// Explicit template instantiations for FP32
template __global__ void flash_attention_forward_kernel<float, 64, 64, 32>(
    const float*, const float*, const float*, float*, float*, int, float, bool);
template __global__ void flash_attention_forward_kernel<float, 64, 64, 64>(
    const float*, const float*, const float*, float*, float*, int, float, bool);
template __global__ void flash_attention_forward_kernel<float, 32, 32, 128>(
    const float*, const float*, const float*, float*, float*, int, float, bool);

// Explicit template instantiations for FP16
template __global__ void flash_attention_forward_kernel<half, 64, 64, 32>(const half*, const half*,
                                                                          const half*, half*, half*,
                                                                          int, float, bool);
template __global__ void flash_attention_forward_kernel<half, 64, 64, 64>(const half*, const half*,
                                                                          const half*, half*, half*,
                                                                          int, float, bool);
template __global__ void flash_attention_forward_kernel<half, 32, 32, 128>(const half*, const half*,
                                                                           const half*, half*,
                                                                           half*, int, float, bool);

// Unified launch function template
template<typename InputT>
FlashAttentionError launch_flash_attention_forward_typed(const InputT* Q, const InputT* K,
                                                         const InputT* V, InputT* O, InputT* L,
                                                         int batch_size, int num_heads, int seq_len,
                                                         int head_dim, float scale, bool causal,
                                                         cudaStream_t stream);

// FP32 specialization
template<>
FlashAttentionError launch_flash_attention_forward_typed<float>(
    const float* Q, const float* K, const float* V, float* O, float* L, int batch_size,
    int num_heads, int seq_len, int head_dim, float scale, bool causal, cudaStream_t stream) {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_M_HD128 = 32;
    constexpr int BLOCK_N_HD128 = 32;

    const int batch_heads = batch_size * num_heads;
    const int num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    const int num_q_blocks_hd128 = (seq_len + BLOCK_M_HD128 - 1) / BLOCK_M_HD128;
    dim3 grid(num_q_blocks, batch_heads);
    dim3 grid_hd128(num_q_blocks_hd128, batch_heads);
    dim3 block(128);

    size_t smem_size = (BLOCK_M * head_dim +  // Q_tile
                        BLOCK_N * head_dim +  // K_tile
                        BLOCK_N * head_dim +  // V_tile
                        BLOCK_M * BLOCK_N +   // S_tile
                        BLOCK_M * head_dim +  // O_tile
                        BLOCK_M +             // m_tile
                        BLOCK_M) *
                       sizeof(float);
    size_t smem_size_hd128 =
        (BLOCK_M_HD128 * head_dim + BLOCK_N_HD128 * head_dim + BLOCK_N_HD128 * head_dim +
         BLOCK_M_HD128 * BLOCK_N_HD128 + BLOCK_M_HD128 * head_dim + BLOCK_M_HD128 + BLOCK_M_HD128) *
        sizeof(float);

    FlashAttentionError status = FlashAttentionError::SUCCESS;

    if (head_dim == 32) {
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_forward_kernel<float, BLOCK_M, BLOCK_N, 32>),
            smem_size);
        if (status != FlashAttentionError::SUCCESS) {
            return status;
        }
        flash_attention_forward_kernel<float, BLOCK_M, BLOCK_N, 32>
            <<<grid, block, smem_size, stream>>>(Q, K, V, O, L, seq_len, scale, causal);
    } else if (head_dim == 64) {
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_forward_kernel<float, BLOCK_M, BLOCK_N, 64>),
            smem_size);
        if (status != FlashAttentionError::SUCCESS) {
            return status;
        }
        flash_attention_forward_kernel<float, BLOCK_M, BLOCK_N, 64>
            <<<grid, block, smem_size, stream>>>(Q, K, V, O, L, seq_len, scale, causal);
    } else if (head_dim == 128) {
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_forward_kernel<float, BLOCK_M_HD128, BLOCK_N_HD128, 128>),
            smem_size_hd128);
        if (status != FlashAttentionError::SUCCESS) {
            return status;
        }
        flash_attention_forward_kernel<float, BLOCK_M_HD128, BLOCK_N_HD128, 128>
            <<<grid_hd128, block, smem_size_hd128, stream>>>(Q, K, V, O, L, seq_len, scale, causal);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }

    return FlashAttentionError::SUCCESS;
}

// FP16 specialization
template<>
FlashAttentionError launch_flash_attention_forward_typed<half>(
    const half* Q, const half* K, const half* V, half* O, half* L, int batch_size, int num_heads,
    int seq_len, int head_dim, float scale, bool causal, cudaStream_t stream) {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_M_HD128 = 32;
    constexpr int BLOCK_N_HD128 = 32;

    const int batch_heads = batch_size * num_heads;
    const int num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    const int num_q_blocks_hd128 = (seq_len + BLOCK_M_HD128 - 1) / BLOCK_M_HD128;
    dim3 grid(num_q_blocks, batch_heads);
    dim3 grid_hd128(num_q_blocks_hd128, batch_heads);
    dim3 block(128);

    // Shared memory is always float-sized for internal computation
    size_t smem_size = (BLOCK_M * head_dim + BLOCK_N * head_dim + BLOCK_N * head_dim +
                        BLOCK_M * BLOCK_N + BLOCK_M * head_dim + BLOCK_M + BLOCK_M) *
                       sizeof(float);
    size_t smem_size_hd128 =
        (BLOCK_M_HD128 * head_dim + BLOCK_N_HD128 * head_dim + BLOCK_N_HD128 * head_dim +
         BLOCK_M_HD128 * BLOCK_N_HD128 + BLOCK_M_HD128 * head_dim + BLOCK_M_HD128 + BLOCK_M_HD128) *
        sizeof(float);

    FlashAttentionError status = FlashAttentionError::SUCCESS;

    if (head_dim == 32) {
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_forward_kernel<half, BLOCK_M, BLOCK_N, 32>),
            smem_size);
        if (status != FlashAttentionError::SUCCESS) {
            return status;
        }
        flash_attention_forward_kernel<half, BLOCK_M, BLOCK_N, 32>
            <<<grid, block, smem_size, stream>>>(Q, K, V, O, L, seq_len, scale, causal);
    } else if (head_dim == 64) {
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_forward_kernel<half, BLOCK_M, BLOCK_N, 64>),
            smem_size);
        if (status != FlashAttentionError::SUCCESS) {
            return status;
        }
        flash_attention_forward_kernel<half, BLOCK_M, BLOCK_N, 64>
            <<<grid, block, smem_size, stream>>>(Q, K, V, O, L, seq_len, scale, causal);
    } else if (head_dim == 128) {
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_forward_kernel<half, BLOCK_M_HD128, BLOCK_N_HD128, 128>),
            smem_size_hd128);
        if (status != FlashAttentionError::SUCCESS) {
            return status;
        }
        flash_attention_forward_kernel<half, BLOCK_M_HD128, BLOCK_N_HD128, 128>
            <<<grid_hd128, block, smem_size_hd128, stream>>>(Q, K, V, O, L, seq_len, scale, causal);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }

    return FlashAttentionError::SUCCESS;
}

// Public API entry points - maintain backward compatibility
FlashAttentionError launch_flash_attention_forward(const float* Q, const float* K, const float* V,
                                                   float* O, float* L, int batch_size,
                                                   int num_heads, int seq_len, int head_dim,
                                                   float scale, bool causal, cudaStream_t stream) {
    return launch_flash_attention_forward_typed(Q, K, V, O, L, batch_size, num_heads, seq_len,
                                                head_dim, scale, causal, stream);
}

FlashAttentionError launch_flash_attention_forward_fp16(const half* Q, const half* K, const half* V,
                                                        half* O, half* L, int batch_size,
                                                        int num_heads, int seq_len, int head_dim,
                                                        float scale, bool causal,
                                                        cudaStream_t stream) {
    return launch_flash_attention_forward_typed(Q, K, V, O, L, batch_size, num_heads, seq_len,
                                                head_dim, scale, causal, stream);
}

}  // namespace cuflash
