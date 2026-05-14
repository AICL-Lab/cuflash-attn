// Unified Flash Attention Backward Kernel - FP32/FP16 Template
// This file consolidates flash_attention_backward.cu and flash_attention_backward_fp16.cu

#include <float.h>

#include "cuflash/flash_attention.h"
#include "impl/online_softmax.cuh"
#include "impl/tile_io.cuh"
#include "kernel_launch_utils.cuh"
#include "workspace_utils.cuh"

namespace cuflash {

// Compute D = rowsum(dO * O) for each row - unified template
template<typename InputT, int BLOCK_SIZE, int HEAD_DIM>
__global__ void __launch_bounds__(128)
    compute_D_kernel(const InputT* __restrict__ dO, const InputT* __restrict__ O,
                     float* __restrict__ D, int seq_len) {
    const int batch_head_idx = blockIdx.y;
    const int row_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row_idx >= seq_len)
        return;

    const InputT* dO_row = dO + batch_head_idx * seq_len * HEAD_DIM + row_idx * HEAD_DIM;
    const InputT* O_row = O + batch_head_idx * seq_len * HEAD_DIM + row_idx * HEAD_DIM;

    float sum = 0.0f;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        sum += impl::to_float(dO_row[d]) * impl::to_float(O_row[d]);
    }

    D[batch_head_idx * seq_len + row_idx] = sum;
}

// Compute dQ for one q-block - unified template
template<typename InputT, int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128)
    flash_attention_backward_dq_kernel(const InputT* __restrict__ Q, const InputT* __restrict__ K,
                                       const InputT* __restrict__ V, const InputT* __restrict__ L,
                                       const InputT* __restrict__ dO, const float* __restrict__ D,
                                       InputT* __restrict__ dQ, int seq_len, float scale,
                                       bool causal) {
    const int batch_head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;

    const InputT* Q_ptr = Q + batch_head_idx * seq_len * HEAD_DIM;
    const InputT* K_ptr = K + batch_head_idx * seq_len * HEAD_DIM;
    const InputT* V_ptr = V + batch_head_idx * seq_len * HEAD_DIM;
    const InputT* L_ptr = L + batch_head_idx * seq_len;
    const InputT* dO_ptr = dO + batch_head_idx * seq_len * HEAD_DIM;
    const float* D_ptr = D + batch_head_idx * seq_len;
    InputT* dQ_ptr = dQ + batch_head_idx * seq_len * HEAD_DIM;

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

    impl::load_tile_to_shared<BLOCK_M, HEAD_DIM>(Q_ptr, Q_tile, q_start, 0, seq_len, HEAD_DIM,
                                                 HEAD_DIM);
    impl::load_tile_to_shared<BLOCK_M, HEAD_DIM>(dO_ptr, dO_tile, q_start, 0, seq_len, HEAD_DIM,
                                                 HEAD_DIM);

    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += num_threads) {
        dQ_tile[i] = 0.0f;
    }
    for (int i = tid; i < BLOCK_M; i += num_threads) {
        int global_idx = q_start + i;
        L_tile[i] = (global_idx < seq_len) ? impl::to_float(L_ptr[global_idx]) : 0.0f;
        D_tile[i] = (global_idx < seq_len) ? D_ptr[global_idx] : 0.0f;
    }
    __syncthreads();

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;

        if (causal && kv_start > q_start + BLOCK_M - 1) {
            break;
        }

        impl::load_tile_to_shared<BLOCK_N, HEAD_DIM>(K_ptr, K_tile, kv_start, 0, seq_len, HEAD_DIM,
                                                     HEAD_DIM);
        impl::load_tile_to_shared<BLOCK_N, HEAD_DIM>(V_ptr, V_tile, kv_start, 0, seq_len, HEAD_DIM,
                                                     HEAD_DIM);
        __syncthreads();

        impl::matmul_ABt<BLOCK_M, BLOCK_N, HEAD_DIM>(Q_tile, K_tile, S_tile, scale);
        __syncthreads();

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

    impl::store_tile_from_shared<BLOCK_M, HEAD_DIM>(dQ_tile, dQ_ptr, q_start, 0, seq_len, HEAD_DIM,
                                                    HEAD_DIM);
}

// Compute dK and dV for one kv-block - unified template
template<typename InputT, int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128)
    flash_attention_backward_dkdv_kernel(const InputT* __restrict__ Q, const InputT* __restrict__ K,
                                         const InputT* __restrict__ V, const InputT* __restrict__ L,
                                         const InputT* __restrict__ dO, const float* __restrict__ D,
                                         InputT* __restrict__ dK, InputT* __restrict__ dV,
                                         int seq_len, float scale, bool causal) {
    const int batch_head_idx = blockIdx.y;
    const int kv_block_idx = blockIdx.x;

    const InputT* Q_ptr = Q + batch_head_idx * seq_len * HEAD_DIM;
    const InputT* K_ptr = K + batch_head_idx * seq_len * HEAD_DIM;
    const InputT* V_ptr = V + batch_head_idx * seq_len * HEAD_DIM;
    const InputT* L_ptr = L + batch_head_idx * seq_len;
    const InputT* dO_ptr = dO + batch_head_idx * seq_len * HEAD_DIM;
    const float* D_ptr = D + batch_head_idx * seq_len;
    InputT* dK_ptr = dK + batch_head_idx * seq_len * HEAD_DIM;
    InputT* dV_ptr = dV + batch_head_idx * seq_len * HEAD_DIM;

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

    impl::load_tile_to_shared<BLOCK_N, HEAD_DIM>(K_ptr, K_tile, kv_start, 0, seq_len, HEAD_DIM,
                                                 HEAD_DIM);
    impl::load_tile_to_shared<BLOCK_N, HEAD_DIM>(V_ptr, V_tile, kv_start, 0, seq_len, HEAD_DIM,
                                                 HEAD_DIM);

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

        impl::load_tile_to_shared<BLOCK_M, HEAD_DIM>(Q_ptr, Q_tile, q_start, 0, seq_len, HEAD_DIM,
                                                     HEAD_DIM);
        impl::load_tile_to_shared<BLOCK_M, HEAD_DIM>(dO_ptr, dO_tile, q_start, 0, seq_len, HEAD_DIM,
                                                     HEAD_DIM);

        for (int i = tid; i < BLOCK_M; i += num_threads) {
            int global_idx = q_start + i;
            L_tile[i] = (global_idx < seq_len) ? impl::to_float(L_ptr[global_idx]) : 0.0f;
            D_tile[i] = (global_idx < seq_len) ? D_ptr[global_idx] : 0.0f;
        }
        __syncthreads();

        impl::matmul_ABt<BLOCK_M, BLOCK_N, HEAD_DIM>(Q_tile, K_tile, S_tile, scale);
        __syncthreads();

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

    impl::store_tile_from_shared<BLOCK_N, HEAD_DIM>(dK_tile, dK_ptr, kv_start, 0, seq_len, HEAD_DIM,
                                                    HEAD_DIM);
    impl::store_tile_from_shared<BLOCK_N, HEAD_DIM>(dV_tile, dV_ptr, kv_start, 0, seq_len, HEAD_DIM,
                                                    HEAD_DIM);
}

// Explicit template instantiations for FP32
template __global__ void compute_D_kernel<float, 128, 32>(const float*, const float*, float*, int);
template __global__ void compute_D_kernel<float, 128, 64>(const float*, const float*, float*, int);
template __global__ void compute_D_kernel<float, 128, 128>(const float*, const float*, float*, int);

template __global__ void flash_attention_backward_dq_kernel<float, 64, 64, 32>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*, int,
    float, bool);
template __global__ void flash_attention_backward_dq_kernel<float, 64, 64, 64>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*, int,
    float, bool);
template __global__ void flash_attention_backward_dq_kernel<float, 64, 64, 128>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*, int,
    float, bool);
template __global__ void flash_attention_backward_dq_kernel<float, 16, 32, 128>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*, int,
    float, bool);

template __global__ void flash_attention_backward_dkdv_kernel<float, 64, 64, 32>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*,
    float*, int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel<float, 64, 64, 64>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*,
    float*, int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel<float, 64, 64, 128>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*,
    float*, int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel<float, 16, 32, 128>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*,
    float*, int, float, bool);

// Explicit template instantiations for FP16
template __global__ void compute_D_kernel<half, 128, 32>(const half*, const half*, float*, int);
template __global__ void compute_D_kernel<half, 128, 64>(const half*, const half*, float*, int);
template __global__ void compute_D_kernel<half, 128, 128>(const half*, const half*, float*, int);

template __global__ void flash_attention_backward_dq_kernel<half, 64, 64, 32>(
    const half*, const half*, const half*, const half*, const half*, const float*, half*, int,
    float, bool);
template __global__ void flash_attention_backward_dq_kernel<half, 64, 64, 64>(
    const half*, const half*, const half*, const half*, const half*, const float*, half*, int,
    float, bool);
template __global__ void flash_attention_backward_dq_kernel<half, 64, 64, 128>(
    const half*, const half*, const half*, const half*, const half*, const float*, half*, int,
    float, bool);
template __global__ void flash_attention_backward_dq_kernel<half, 16, 32, 128>(
    const half*, const half*, const half*, const half*, const half*, const float*, half*, int,
    float, bool);

template __global__ void flash_attention_backward_dkdv_kernel<half, 64, 64, 32>(
    const half*, const half*, const half*, const half*, const half*, const float*, half*, half*,
    int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel<half, 64, 64, 64>(
    const half*, const half*, const half*, const half*, const half*, const float*, half*, half*,
    int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel<half, 64, 64, 128>(
    const half*, const half*, const half*, const half*, const half*, const float*, half*, half*,
    int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel<half, 16, 32, 128>(
    const half*, const half*, const half*, const half*, const half*, const float*, half*, half*,
    int, float, bool);

namespace {

// Thread-safe workspace accessor using thread-local storage
// Each thread gets its own workspace, eliminating race conditions
DeviceFloatWorkspace& backward_workspace() {
    thread_local DeviceFloatWorkspace workspace;
    return workspace;
}

}  // namespace

// Unified launch function template
template<typename InputT>
FlashAttentionError launch_flash_attention_backward_typed(
    const InputT* Q, const InputT* K, const InputT* V, const InputT* O, const InputT* L,
    const InputT* dO, InputT* dQ, InputT* dK, InputT* dV, int batch_size, int num_heads,
    int seq_len, int head_dim, float scale, bool causal, cudaStream_t stream);

// FP32 specialization
template<>
FlashAttentionError launch_flash_attention_backward_typed<float>(
    const float* Q, const float* K, const float* V, const float* O, const float* L, const float* dO,
    float* dQ, float* dK, float* dV, int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream) {
    using Config = impl::BackwardTilingConfig;
    constexpr int BLOCK_M = Config::BLOCK_M;
    constexpr int BLOCK_N = Config::BLOCK_N;
    constexpr int BLOCK_M_HD128 = Config::BLOCK_M_HD128;
    constexpr int BLOCK_N_HD128 = Config::BLOCK_N_HD128;

    int batch_heads = batch_size * num_heads;

    // Thread-safe workspace access
    DeviceFloatWorkspace& workspace = backward_workspace();
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
        compute_D_kernel<float, 128, 32><<<d_grid, 128, 0, stream>>>(dO, O, D, seq_len);
    } else if (head_dim == 64) {
        compute_D_kernel<float, 128, 64><<<d_grid, 128, 0, stream>>>(dO, O, D, seq_len);
    } else if (head_dim == 128) {
        compute_D_kernel<float, 128, 128><<<d_grid, 128, 0, stream>>>(dO, O, D, seq_len);
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
                flash_attention_backward_dq_kernel<float, BLOCK_M, BLOCK_N, 32>),
            dq_smem_size);
        if (status != FlashAttentionError::SUCCESS)
            return status;
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_backward_dkdv_kernel<float, BLOCK_M, BLOCK_N, 32>),
            dkdv_smem_size);
        if (status != FlashAttentionError::SUCCESS)
            return status;

        flash_attention_backward_dq_kernel<float, BLOCK_M, BLOCK_N, 32>
            <<<dq_grid, block, dq_smem_size, stream>>>(Q, K, V, L, dO, D, dQ, seq_len, scale,
                                                       causal);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return FlashAttentionError::CUDA_ERROR;

        flash_attention_backward_dkdv_kernel<float, BLOCK_M, BLOCK_N, 32>
            <<<dkdv_grid, block, dkdv_smem_size, stream>>>(Q, K, V, L, dO, D, dK, dV, seq_len,
                                                           scale, causal);
    } else if (head_dim == 64) {
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_backward_dq_kernel<float, BLOCK_M, BLOCK_N, 64>),
            dq_smem_size);
        if (status != FlashAttentionError::SUCCESS)
            return status;
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_backward_dkdv_kernel<float, BLOCK_M, BLOCK_N, 64>),
            dkdv_smem_size);
        if (status != FlashAttentionError::SUCCESS)
            return status;

        flash_attention_backward_dq_kernel<float, BLOCK_M, BLOCK_N, 64>
            <<<dq_grid, block, dq_smem_size, stream>>>(Q, K, V, L, dO, D, dQ, seq_len, scale,
                                                       causal);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return FlashAttentionError::CUDA_ERROR;

        flash_attention_backward_dkdv_kernel<float, BLOCK_M, BLOCK_N, 64>
            <<<dkdv_grid, block, dkdv_smem_size, stream>>>(Q, K, V, L, dO, D, dK, dV, seq_len,
                                                           scale, causal);
    } else if (head_dim == 128) {
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_backward_dq_kernel<float, BLOCK_M_HD128, BLOCK_N_HD128, 128>),
            dq_smem_size_hd128);
        if (status != FlashAttentionError::SUCCESS)
            return status;
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_backward_dkdv_kernel<float, BLOCK_M_HD128, BLOCK_N_HD128, 128>),
            dkdv_smem_size_hd128);
        if (status != FlashAttentionError::SUCCESS)
            return status;

        flash_attention_backward_dq_kernel<float, BLOCK_M_HD128, BLOCK_N_HD128, 128>
            <<<dq_grid_hd128, block, dq_smem_size_hd128, stream>>>(Q, K, V, L, dO, D, dQ, seq_len,
                                                                   scale, causal);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return FlashAttentionError::CUDA_ERROR;

        flash_attention_backward_dkdv_kernel<float, BLOCK_M_HD128, BLOCK_N_HD128, 128>
            <<<dkdv_grid_hd128, block, dkdv_smem_size_hd128, stream>>>(Q, K, V, L, dO, D, dK, dV,
                                                                       seq_len, scale, causal);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }

    return FlashAttentionError::SUCCESS;
}

// FP16 specialization
template<>
FlashAttentionError launch_flash_attention_backward_typed<half>(
    const half* Q, const half* K, const half* V, const half* O, const half* L, const half* dO,
    half* dQ, half* dK, half* dV, int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream) {
    using Config = impl::BackwardTilingConfig;
    constexpr int BLOCK_M = Config::BLOCK_M;
    constexpr int BLOCK_N = Config::BLOCK_N;
    constexpr int BLOCK_M_HD128 = Config::BLOCK_M_HD128;
    constexpr int BLOCK_N_HD128 = Config::BLOCK_N_HD128;

    int batch_heads = batch_size * num_heads;

    DeviceFloatWorkspace& workspace = backward_workspace();
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
        compute_D_kernel<half, 128, 32><<<d_grid, 128, 0, stream>>>(dO, O, D, seq_len);
    } else if (head_dim == 64) {
        compute_D_kernel<half, 128, 64><<<d_grid, 128, 0, stream>>>(dO, O, D, seq_len);
    } else if (head_dim == 128) {
        compute_D_kernel<half, 128, 128><<<d_grid, 128, 0, stream>>>(dO, O, D, seq_len);
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
                flash_attention_backward_dq_kernel<half, BLOCK_M, BLOCK_N, 32>),
            dq_smem_size);
        if (status != FlashAttentionError::SUCCESS)
            return status;
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_backward_dkdv_kernel<half, BLOCK_M, BLOCK_N, 32>),
            dkdv_smem_size);
        if (status != FlashAttentionError::SUCCESS)
            return status;

        flash_attention_backward_dq_kernel<half, BLOCK_M, BLOCK_N, 32>
            <<<dq_grid, block, dq_smem_size, stream>>>(Q, K, V, L, dO, D, dQ, seq_len, scale,
                                                       causal);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return FlashAttentionError::CUDA_ERROR;

        flash_attention_backward_dkdv_kernel<half, BLOCK_M, BLOCK_N, 32>
            <<<dkdv_grid, block, dkdv_smem_size, stream>>>(Q, K, V, L, dO, D, dK, dV, seq_len,
                                                           scale, causal);
    } else if (head_dim == 64) {
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_backward_dq_kernel<half, BLOCK_M, BLOCK_N, 64>),
            dq_smem_size);
        if (status != FlashAttentionError::SUCCESS)
            return status;
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_backward_dkdv_kernel<half, BLOCK_M, BLOCK_N, 64>),
            dkdv_smem_size);
        if (status != FlashAttentionError::SUCCESS)
            return status;

        flash_attention_backward_dq_kernel<half, BLOCK_M, BLOCK_N, 64>
            <<<dq_grid, block, dq_smem_size, stream>>>(Q, K, V, L, dO, D, dQ, seq_len, scale,
                                                       causal);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return FlashAttentionError::CUDA_ERROR;

        flash_attention_backward_dkdv_kernel<half, BLOCK_M, BLOCK_N, 64>
            <<<dkdv_grid, block, dkdv_smem_size, stream>>>(Q, K, V, L, dO, D, dK, dV, seq_len,
                                                           scale, causal);
    } else if (head_dim == 128) {
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_backward_dq_kernel<half, BLOCK_M_HD128, BLOCK_N_HD128, 128>),
            dq_smem_size_hd128);
        if (status != FlashAttentionError::SUCCESS)
            return status;
        status = prepare_dynamic_smem_launch(
            reinterpret_cast<const void*>(
                flash_attention_backward_dkdv_kernel<half, BLOCK_M_HD128, BLOCK_N_HD128, 128>),
            dkdv_smem_size_hd128);
        if (status != FlashAttentionError::SUCCESS)
            return status;

        flash_attention_backward_dq_kernel<half, BLOCK_M_HD128, BLOCK_N_HD128, 128>
            <<<dq_grid_hd128, block, dq_smem_size_hd128, stream>>>(Q, K, V, L, dO, D, dQ, seq_len,
                                                                   scale, causal);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return FlashAttentionError::CUDA_ERROR;

        flash_attention_backward_dkdv_kernel<half, BLOCK_M_HD128, BLOCK_N_HD128, 128>
            <<<dkdv_grid_hd128, block, dkdv_smem_size_hd128, stream>>>(Q, K, V, L, dO, D, dK, dV,
                                                                       seq_len, scale, causal);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }

    return FlashAttentionError::SUCCESS;
}

// Public API entry points - maintain backward compatibility
FlashAttentionError launch_flash_attention_backward(const float* Q, const float* K, const float* V,
                                                    const float* O, const float* L, const float* dO,
                                                    float* dQ, float* dK, float* dV, int batch_size,
                                                    int num_heads, int seq_len, int head_dim,
                                                    float scale, bool causal, cudaStream_t stream) {
    return launch_flash_attention_backward_typed(Q, K, V, O, L, dO, dQ, dK, dV, batch_size,
                                                 num_heads, seq_len, head_dim, scale, causal,
                                                 stream);
}

FlashAttentionError launch_flash_attention_backward_fp16(
    const half* Q, const half* K, const half* V, const half* O, const half* L, const half* dO,
    half* dQ, half* dK, half* dV, int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream) {
    return launch_flash_attention_backward_typed(Q, K, V, O, L, dO, dQ, dK, dV, batch_size,
                                                 num_heads, seq_len, head_dim, scale, causal,
                                                 stream);
}

}  // namespace cuflash
