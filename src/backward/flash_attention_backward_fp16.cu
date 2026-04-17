// Flash Attention FP16 Backward Kernel Implementation

#include <cuda_fp16.h>

#include "cuflash/flash_attention.h"
#include "kernel_launch_utils.cuh"
#include "workspace_utils.cuh"

namespace cuflash {

// Helper functions for half precision conversion
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

__device__ __forceinline__ half float_to_half(float f) {
    return __float2half(f);
}

// Compute D = rowsum(dO * O) for each row (FP16 inputs, FP32 output)
template<int BLOCK_SIZE, int HEAD_DIM>
__global__ void __launch_bounds__(128)
    compute_D_kernel_fp16(const half* __restrict__ dO,  // [batch*heads, seq_len, head_dim]
                          const half* __restrict__ O,   // [batch*heads, seq_len, head_dim]
                          float* __restrict__ D,        // [batch*heads, seq_len]
                          int seq_len) {
    const int batch_head_idx = blockIdx.y;
    const int row_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row_idx >= seq_len)
        return;

    const half* dO_row = dO + batch_head_idx * seq_len * HEAD_DIM + row_idx * HEAD_DIM;
    const half* O_row = O + batch_head_idx * seq_len * HEAD_DIM + row_idx * HEAD_DIM;

    float sum = 0.0f;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        sum += half_to_float(dO_row[d]) * half_to_float(O_row[d]);
    }

    D[batch_head_idx * seq_len + row_idx] = sum;
}

// Compute dQ for one q-block (FP16 inputs/outputs, FP32 shared memory computation)
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128)
    flash_attention_backward_dq_kernel_fp16(const half* __restrict__ Q, const half* __restrict__ K,
                                            const half* __restrict__ V, const half* __restrict__ L,
                                            const half* __restrict__ dO,
                                            const float* __restrict__ D, half* __restrict__ dQ,
                                            int seq_len, float scale, bool causal) {
    const int batch_head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;

    const half* Q_ptr = Q + batch_head_idx * seq_len * HEAD_DIM;
    const half* K_ptr = K + batch_head_idx * seq_len * HEAD_DIM;
    const half* V_ptr = V + batch_head_idx * seq_len * HEAD_DIM;
    const half* L_ptr = L + batch_head_idx * seq_len;
    const half* dO_ptr = dO + batch_head_idx * seq_len * HEAD_DIM;
    const float* D_ptr = D + batch_head_idx * seq_len;
    half* dQ_ptr = dQ + batch_head_idx * seq_len * HEAD_DIM;

    const int q_start = q_block_idx * BLOCK_M;
    if (q_start >= seq_len)
        return;

    const int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ float smem[];
    float* Q_tile = smem;                          // BLOCK_M x HEAD_DIM
    float* dO_tile = Q_tile + BLOCK_M * HEAD_DIM;  // BLOCK_M x HEAD_DIM
    float* K_tile = dO_tile + BLOCK_M * HEAD_DIM;  // BLOCK_N x HEAD_DIM
    float* V_tile = K_tile + BLOCK_N * HEAD_DIM;   // BLOCK_N x HEAD_DIM
    float* S_tile = V_tile + BLOCK_N * HEAD_DIM;   // BLOCK_M x BLOCK_N (reused for dS)
    float* dQ_tile = S_tile + BLOCK_M * BLOCK_N;   // BLOCK_M x HEAD_DIM
    float* L_tile = dQ_tile + BLOCK_M * HEAD_DIM;  // BLOCK_M
    float* D_tile = L_tile + BLOCK_M;              // BLOCK_M

    // Load Q tile from half* (convert to float)
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += num_threads) {
        int local_row = i / HEAD_DIM;
        int local_col = i % HEAD_DIM;
        int global_row = q_start + local_row;
        if (global_row < seq_len) {
            Q_tile[i] = half_to_float(Q_ptr[global_row * HEAD_DIM + local_col]);
        } else {
            Q_tile[i] = 0.0f;
        }
    }

    // Load dO tile from half*
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += num_threads) {
        int local_row = i / HEAD_DIM;
        int local_col = i % HEAD_DIM;
        int global_row = q_start + local_row;
        if (global_row < seq_len) {
            dO_tile[i] = half_to_float(dO_ptr[global_row * HEAD_DIM + local_col]);
        } else {
            dO_tile[i] = 0.0f;
        }
    }

    // Initialize dQ tile to zero
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += num_threads) {
        dQ_tile[i] = 0.0f;
    }

    // Load L (half*) and D (float*)
    for (int i = tid; i < BLOCK_M; i += num_threads) {
        int global_idx = q_start + i;
        L_tile[i] = (global_idx < seq_len) ? half_to_float(L_ptr[global_idx]) : 0.0f;
        D_tile[i] = (global_idx < seq_len) ? D_ptr[global_idx] : 0.0f;
    }
    __syncthreads();

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;

        if (causal && kv_start > q_start + BLOCK_M - 1) {
            break;
        }

        // Load K and V tiles from half*
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += num_threads) {
            int local_row = i / HEAD_DIM;
            int local_col = i % HEAD_DIM;
            int global_row = kv_start + local_row;
            if (global_row < seq_len) {
                K_tile[i] = half_to_float(K_ptr[global_row * HEAD_DIM + local_col]);
                V_tile[i] = half_to_float(V_ptr[global_row * HEAD_DIM + local_col]);
            } else {
                K_tile[i] = 0.0f;
                V_tile[i] = 0.0f;
            }
        }
        __syncthreads();

        // Compute S = Q @ K^T * scale
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += num_threads) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            float sum = 0.0f;
            for (int k = 0; k < HEAD_DIM; k++) {
                sum += Q_tile[row * HEAD_DIM + k] * K_tile[col * HEAD_DIM + k];
            }
            S_tile[i] = sum * scale;
        }
        __syncthreads();

        // Compute dS = P * (dP - D) where P = exp(S - L)
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += num_threads) {
            int q_idx = i / BLOCK_N;
            int k_idx = i % BLOCK_N;
            int global_q = q_start + q_idx;
            int global_k = kv_start + k_idx;

            if (global_q >= seq_len || global_k >= seq_len || (causal && global_k > global_q)) {
                S_tile[i] = 0.0f;
            } else {
                float dP = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    dP += dO_tile[q_idx * HEAD_DIM + d] * V_tile[k_idx * HEAD_DIM + d];
                }
                float p = expf(S_tile[i] - L_tile[q_idx]);
                S_tile[i] = p * (dP - D_tile[q_idx]);
            }
        }
        __syncthreads();

        // Accumulate dQ += dS @ K * scale
        for (int q = tid; q < BLOCK_M; q += num_threads) {
            int global_q = q_start + q;
            if (global_q >= seq_len)
                continue;

            for (int d = 0; d < HEAD_DIM; d++) {
                float sum = 0.0f;
                for (int k = 0; k < BLOCK_N; k++) {
                    sum += S_tile[q * BLOCK_N + k] * K_tile[k * HEAD_DIM + d];
                }
                dQ_tile[q * HEAD_DIM + d] += sum * scale;
            }
        }
        __syncthreads();
    }

    // Store dQ tile to half*
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += num_threads) {
        int local_row = i / HEAD_DIM;
        int local_col = i % HEAD_DIM;
        int global_row = q_start + local_row;
        if (global_row < seq_len) {
            dQ_ptr[global_row * HEAD_DIM + local_col] = float_to_half(dQ_tile[i]);
        }
    }
}

// Compute dK and dV for one kv-block (FP16 inputs/outputs, FP32 shared memory)
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128) flash_attention_backward_dkdv_kernel_fp16(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
    const half* __restrict__ L, const half* __restrict__ dO, const float* __restrict__ D,
    half* __restrict__ dK, half* __restrict__ dV, int seq_len, float scale, bool causal) {
    const int batch_head_idx = blockIdx.y;
    const int kv_block_idx = blockIdx.x;

    const half* Q_ptr = Q + batch_head_idx * seq_len * HEAD_DIM;
    const half* K_ptr = K + batch_head_idx * seq_len * HEAD_DIM;
    const half* V_ptr = V + batch_head_idx * seq_len * HEAD_DIM;
    const half* L_ptr = L + batch_head_idx * seq_len;
    const half* dO_ptr = dO + batch_head_idx * seq_len * HEAD_DIM;
    const float* D_ptr = D + batch_head_idx * seq_len;
    half* dK_ptr = dK + batch_head_idx * seq_len * HEAD_DIM;
    half* dV_ptr = dV + batch_head_idx * seq_len * HEAD_DIM;

    const int kv_start = kv_block_idx * BLOCK_N;
    if (kv_start >= seq_len)
        return;

    const int num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ float smem[];
    float* K_tile = smem;                           // BLOCK_N x HEAD_DIM
    float* V_tile = K_tile + BLOCK_N * HEAD_DIM;    // BLOCK_N x HEAD_DIM
    float* Q_tile = V_tile + BLOCK_N * HEAD_DIM;    // BLOCK_M x HEAD_DIM
    float* dO_tile = Q_tile + BLOCK_M * HEAD_DIM;   // BLOCK_M x HEAD_DIM
    float* S_tile = dO_tile + BLOCK_M * HEAD_DIM;   // BLOCK_M x BLOCK_N (reused for dS)
    float* dK_tile = S_tile + BLOCK_M * BLOCK_N;    // BLOCK_N x HEAD_DIM
    float* dV_tile = dK_tile + BLOCK_N * HEAD_DIM;  // BLOCK_N x HEAD_DIM
    float* L_tile = dV_tile + BLOCK_N * HEAD_DIM;   // BLOCK_M
    float* D_tile = L_tile + BLOCK_M;               // BLOCK_M

    // Load K and V tiles from half*
    for (int i = tid; i < BLOCK_N * HEAD_DIM; i += num_threads) {
        int local_row = i / HEAD_DIM;
        int local_col = i % HEAD_DIM;
        int global_row = kv_start + local_row;
        if (global_row < seq_len) {
            K_tile[i] = half_to_float(K_ptr[global_row * HEAD_DIM + local_col]);
            V_tile[i] = half_to_float(V_ptr[global_row * HEAD_DIM + local_col]);
        } else {
            K_tile[i] = 0.0f;
            V_tile[i] = 0.0f;
        }
    }

    // Initialize dK and dV tiles to zero
    for (int i = tid; i < BLOCK_N * HEAD_DIM; i += num_threads) {
        dK_tile[i] = 0.0f;
        dV_tile[i] = 0.0f;
    }
    __syncthreads();

    for (int q_block = 0; q_block < num_q_blocks; q_block++) {
        int q_start = q_block * BLOCK_M;

        if (causal && q_start + BLOCK_M - 1 < kv_start) {
            continue;
        }

        // Load Q and dO tiles from half*
        for (int i = tid; i < BLOCK_M * HEAD_DIM; i += num_threads) {
            int local_row = i / HEAD_DIM;
            int local_col = i % HEAD_DIM;
            int global_row = q_start + local_row;
            if (global_row < seq_len) {
                Q_tile[i] = half_to_float(Q_ptr[global_row * HEAD_DIM + local_col]);
                dO_tile[i] = half_to_float(dO_ptr[global_row * HEAD_DIM + local_col]);
            } else {
                Q_tile[i] = 0.0f;
                dO_tile[i] = 0.0f;
            }
        }

        // Load L (half*) and D (float*)
        for (int i = tid; i < BLOCK_M; i += num_threads) {
            int global_idx = q_start + i;
            L_tile[i] = (global_idx < seq_len) ? half_to_float(L_ptr[global_idx]) : 0.0f;
            D_tile[i] = (global_idx < seq_len) ? D_ptr[global_idx] : 0.0f;
        }
        __syncthreads();

        // Compute S = Q @ K^T * scale
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += num_threads) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            float sum = 0.0f;
            for (int k = 0; k < HEAD_DIM; k++) {
                sum += Q_tile[row * HEAD_DIM + k] * K_tile[col * HEAD_DIM + k];
            }
            S_tile[i] = sum * scale;
        }
        __syncthreads();

        // Compute P = exp(S - L) with causal masking
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += num_threads) {
            int q_idx = i / BLOCK_N;
            int k_idx = i % BLOCK_N;
            int global_q = q_start + q_idx;
            int global_k = kv_start + k_idx;

            if (global_q >= seq_len || global_k >= seq_len) {
                S_tile[i] = 0.0f;
            } else if (causal && global_k > global_q) {
                S_tile[i] = 0.0f;
            } else {
                S_tile[i] = expf(S_tile[i] - L_tile[q_idx]);
            }
        }
        __syncthreads();

        // Accumulate dV += P^T @ dO
        for (int k = tid; k < BLOCK_N; k += num_threads) {
            if (kv_start + k >= seq_len)
                continue;
            for (int d = 0; d < HEAD_DIM; d++) {
                float sum = 0.0f;
                for (int q = 0; q < BLOCK_M; q++) {
                    if (q_start + q < seq_len) {
                        sum += S_tile[q * BLOCK_N + k] * dO_tile[q * HEAD_DIM + d];
                    }
                }
                dV_tile[k * HEAD_DIM + d] += sum;
            }
        }
        __syncthreads();

        // Compute dS = P * (dP - D) where dP = dO @ V^T
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += num_threads) {
            int q_idx = i / BLOCK_N;
            int k_idx = i % BLOCK_N;
            int global_q = q_start + q_idx;
            int global_k = kv_start + k_idx;

            if (global_q >= seq_len || global_k >= seq_len) {
                S_tile[i] = 0.0f;
            } else if (causal && global_k > global_q) {
                S_tile[i] = 0.0f;
            } else {
                float dP = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    dP += dO_tile[q_idx * HEAD_DIM + d] * V_tile[k_idx * HEAD_DIM + d];
                }
                S_tile[i] = S_tile[i] * (dP - D_tile[q_idx]);
            }
        }
        __syncthreads();

        // Accumulate dK += dS^T @ Q * scale
        for (int k = tid; k < BLOCK_N; k += num_threads) {
            if (kv_start + k >= seq_len)
                continue;
            for (int d = 0; d < HEAD_DIM; d++) {
                float sum = 0.0f;
                for (int q = 0; q < BLOCK_M; q++) {
                    if (q_start + q < seq_len) {
                        sum += S_tile[q * BLOCK_N + k] * Q_tile[q * HEAD_DIM + d];
                    }
                }
                dK_tile[k * HEAD_DIM + d] += sum * scale;
            }
        }
        __syncthreads();
    }

    // Store dK and dV tiles to half*
    for (int i = tid; i < BLOCK_N * HEAD_DIM; i += num_threads) {
        int local_row = i / HEAD_DIM;
        int local_col = i % HEAD_DIM;
        int global_row = kv_start + local_row;
        if (global_row < seq_len) {
            dK_ptr[global_row * HEAD_DIM + local_col] = float_to_half(dK_tile[i]);
            dV_ptr[global_row * HEAD_DIM + local_col] = float_to_half(dV_tile[i]);
        }
    }
}

// Explicit template instantiations
template __global__ void compute_D_kernel_fp16<128, 32>(const half*, const half*, float*, int);
template __global__ void compute_D_kernel_fp16<128, 64>(const half*, const half*, float*, int);
template __global__ void compute_D_kernel_fp16<128, 128>(const half*, const half*, float*, int);

template __global__ void flash_attention_backward_dq_kernel_fp16<64, 64, 32>(
    const half*, const half*, const half*, const half*, const half*, const float*, half*, int,
    float, bool);
template __global__ void flash_attention_backward_dq_kernel_fp16<64, 64, 64>(
    const half*, const half*, const half*, const half*, const half*, const float*, half*, int,
    float, bool);
template __global__ void flash_attention_backward_dq_kernel_fp16<64, 64, 128>(
    const half*, const half*, const half*, const half*, const half*, const float*, half*, int,
    float, bool);
template __global__ void flash_attention_backward_dq_kernel_fp16<16, 32, 128>(
    const half*, const half*, const half*, const half*, const half*, const float*, half*, int,
    float, bool);

template __global__ void flash_attention_backward_dkdv_kernel_fp16<64, 64, 32>(
    const half*, const half*, const half*, const half*, const half*, const float*, half*, half*,
    int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel_fp16<64, 64, 64>(
    const half*, const half*, const half*, const half*, const half*, const float*, half*, half*,
    int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel_fp16<64, 64, 128>(
    const half*, const half*, const half*, const half*, const half*, const float*, half*, half*,
    int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel_fp16<16, 32, 128>(
    const half*, const half*, const half*, const half*, const half*, const float*, half*, half*,
    int, float, bool);

namespace {

DeviceFloatWorkspace& backward_workspace_fp16() {
    static DeviceFloatWorkspace workspace;
    return workspace;
}

}  // namespace

FlashAttentionError launch_flash_attention_backward_fp16(
    const half* Q, const half* K, const half* V, const half* O, const half* L, const half* dO,
    half* dQ, half* dK, half* dV, int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream) {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_M_HD128 = 16;
    constexpr int BLOCK_N_HD128 = 32;

    int batch_heads = batch_size * num_heads;

    DeviceFloatWorkspace& workspace = backward_workspace_fp16();
    FlashAttentionError workspace_status =
        workspace.reserve(static_cast<size_t>(batch_heads) * static_cast<size_t>(seq_len));
    if (workspace_status != FlashAttentionError::SUCCESS) {
        return workspace_status;
    }

    float* D = workspace.data();
    if (D == nullptr) {
        return FlashAttentionError::CUDA_ERROR;
    }

    int d_blocks = (seq_len + 127) / 128;
    dim3 d_grid(d_blocks, batch_heads);

    if (head_dim == 32) {
        compute_D_kernel_fp16<128, 32><<<d_grid, 128, 0, stream>>>(dO, O, D, seq_len);
    } else if (head_dim == 64) {
        compute_D_kernel_fp16<128, 64><<<d_grid, 128, 0, stream>>>(dO, O, D, seq_len);
    } else if (head_dim == 128) {
        compute_D_kernel_fp16<128, 128><<<d_grid, 128, 0, stream>>>(dO, O, D, seq_len);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }

    int num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    int num_q_blocks_hd128 = (seq_len + BLOCK_M_HD128 - 1) / BLOCK_M_HD128;
    int num_kv_blocks_hd128 = (seq_len + BLOCK_N_HD128 - 1) / BLOCK_N_HD128;
    dim3 dq_grid(num_q_blocks, batch_heads);
    dim3 dkdv_grid(num_kv_blocks, batch_heads);
    dim3 dq_grid_hd128(num_q_blocks_hd128, batch_heads);
    dim3 dkdv_grid_hd128(num_kv_blocks_hd128, batch_heads);
    dim3 block(128);

    size_t dq_smem_size =
        (BLOCK_M * head_dim + BLOCK_M * head_dim + BLOCK_N * head_dim + BLOCK_N * head_dim +
         BLOCK_M * BLOCK_N + BLOCK_M * head_dim + BLOCK_M + BLOCK_M) *
        sizeof(float);
    size_t dkdv_smem_size =
        (BLOCK_N * head_dim + BLOCK_N * head_dim + BLOCK_M * head_dim + BLOCK_M * head_dim +
         BLOCK_M * BLOCK_N + BLOCK_N * head_dim + BLOCK_N * head_dim + BLOCK_M + BLOCK_M) *
        sizeof(float);
    size_t dq_smem_size_hd128 =
        (BLOCK_M_HD128 * head_dim + BLOCK_M_HD128 * head_dim + BLOCK_N_HD128 * head_dim +
         BLOCK_N_HD128 * head_dim + BLOCK_M_HD128 * BLOCK_N_HD128 + BLOCK_M_HD128 * head_dim +
         BLOCK_M_HD128 + BLOCK_M_HD128) *
        sizeof(float);
    size_t dkdv_smem_size_hd128 =
        (BLOCK_N_HD128 * head_dim + BLOCK_N_HD128 * head_dim + BLOCK_M_HD128 * head_dim +
         BLOCK_M_HD128 * head_dim + BLOCK_M_HD128 * BLOCK_N_HD128 + BLOCK_N_HD128 * head_dim +
         BLOCK_N_HD128 * head_dim + BLOCK_M_HD128 + BLOCK_M_HD128) *
        sizeof(float);

    FlashAttentionError status = FlashAttentionError::SUCCESS;

    if (head_dim == 32) {
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_backward_dq_kernel_fp16<BLOCK_M, BLOCK_N, 32>),
            dq_smem_size);
        if (status != FlashAttentionError::SUCCESS) {
            return status;
        }
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_backward_dkdv_kernel_fp16<BLOCK_M, BLOCK_N, 32>),
            dkdv_smem_size);
        if (status != FlashAttentionError::SUCCESS) {
            return status;
        }

        flash_attention_backward_dq_kernel_fp16<BLOCK_M, BLOCK_N, 32>
            <<<dq_grid, block, dq_smem_size, stream>>>(Q, K, V, L, dO, D, dQ, seq_len, scale,
                                                       causal);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return FlashAttentionError::CUDA_ERROR;
        }

        flash_attention_backward_dkdv_kernel_fp16<BLOCK_M, BLOCK_N, 32>
            <<<dkdv_grid, block, dkdv_smem_size, stream>>>(Q, K, V, L, dO, D, dK, dV, seq_len,
                                                           scale, causal);
    } else if (head_dim == 64) {
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_backward_dq_kernel_fp16<BLOCK_M, BLOCK_N, 64>),
            dq_smem_size);
        if (status != FlashAttentionError::SUCCESS) {
            return status;
        }
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_backward_dkdv_kernel_fp16<BLOCK_M, BLOCK_N, 64>),
            dkdv_smem_size);
        if (status != FlashAttentionError::SUCCESS) {
            return status;
        }

        flash_attention_backward_dq_kernel_fp16<BLOCK_M, BLOCK_N, 64>
            <<<dq_grid, block, dq_smem_size, stream>>>(Q, K, V, L, dO, D, dQ, seq_len, scale,
                                                       causal);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return FlashAttentionError::CUDA_ERROR;
        }

        flash_attention_backward_dkdv_kernel_fp16<BLOCK_M, BLOCK_N, 64>
            <<<dkdv_grid, block, dkdv_smem_size, stream>>>(Q, K, V, L, dO, D, dK, dV, seq_len,
                                                           scale, causal);
    } else if (head_dim == 128) {
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_backward_dq_kernel_fp16<BLOCK_M_HD128, BLOCK_N_HD128, 128>),
            dq_smem_size_hd128);
        if (status != FlashAttentionError::SUCCESS) {
            return status;
        }
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_backward_dkdv_kernel_fp16<BLOCK_M_HD128, BLOCK_N_HD128, 128>),
            dkdv_smem_size_hd128);
        if (status != FlashAttentionError::SUCCESS) {
            return status;
        }

        flash_attention_backward_dq_kernel_fp16<BLOCK_M_HD128, BLOCK_N_HD128, 128>
            <<<dq_grid_hd128, block, dq_smem_size_hd128, stream>>>(Q, K, V, L, dO, D, dQ, seq_len,
                                                                   scale, causal);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return FlashAttentionError::CUDA_ERROR;
        }

        flash_attention_backward_dkdv_kernel_fp16<BLOCK_M_HD128, BLOCK_N_HD128, 128>
            <<<dkdv_grid_hd128, block, dkdv_smem_size_hd128, stream>>>(Q, K, V, L, dO, D, dK, dV,
                                                                       seq_len, scale, causal);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }

    return FlashAttentionError::SUCCESS;
}

// Internal FP16 backward entry point (validation already done by API layer)
FlashAttentionError flash_attention_backward_fp16(const half* Q, const half* K, const half* V,
                                                  const half* O, const half* L, const half* dO,
                                                  half* dQ, half* dK, half* dV, int batch_size,
                                                  int num_heads, int seq_len, int head_dim,
                                                  float scale, bool causal, cudaStream_t stream) {
    return launch_flash_attention_backward_fp16(Q, K, V, O, L, dO, dQ, dK, dV, batch_size,
                                                num_heads, seq_len, head_dim, scale, causal,
                                                stream);
}

}  // namespace cuflash
