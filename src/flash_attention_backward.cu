// Flash Attention Backward Kernel Implementation

#include "flash_attention.h"
#include "online_softmax.cuh"
#include "matmul.cuh"
#include <float.h>

namespace cuflash {

// Compute D = rowsum(dO * O) for each row
// This is needed for the backward pass gradient computation
template<int BLOCK_SIZE, int HEAD_DIM>
__global__ void compute_D_kernel(
    const float* __restrict__ dO,     // [batch*heads, seq_len, head_dim]
    const float* __restrict__ O,      // [batch*heads, seq_len, head_dim]
    float* __restrict__ D,            // [batch*heads, seq_len]
    int seq_len
) {
    const int batch_head_idx = blockIdx.y;
    const int row_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    if (row_idx >= seq_len) return;
    
    const float* dO_row = dO + batch_head_idx * seq_len * HEAD_DIM + row_idx * HEAD_DIM;
    const float* O_row = O + batch_head_idx * seq_len * HEAD_DIM + row_idx * HEAD_DIM;
    
    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        sum += dO_row[d] * O_row[d];
    }
    
    D[batch_head_idx * seq_len + row_idx] = sum;
}

// Backward kernel template
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void flash_attention_backward_kernel(
    const float* __restrict__ Q,      // [batch*heads, seq_len, head_dim]
    const float* __restrict__ K,      // [batch*heads, seq_len, head_dim]
    const float* __restrict__ V,      // [batch*heads, seq_len, head_dim]
    const float* __restrict__ O,      // [batch*heads, seq_len, head_dim]
    const float* __restrict__ L,      // [batch*heads, seq_len]
    const float* __restrict__ dO,     // [batch*heads, seq_len, head_dim]
    const float* __restrict__ D,      // [batch*heads, seq_len]
    float* __restrict__ dQ,           // [batch*heads, seq_len, head_dim]
    float* __restrict__ dK,           // [batch*heads, seq_len, head_dim]
    float* __restrict__ dV,           // [batch*heads, seq_len, head_dim]
    int seq_len,
    float scale,
    bool causal
) {
    const int batch_head_idx = blockIdx.y;
    const int kv_block_idx = blockIdx.x;
    
    // Pointers for this batch/head
    const float* Q_ptr = Q + batch_head_idx * seq_len * HEAD_DIM;
    const float* K_ptr = K + batch_head_idx * seq_len * HEAD_DIM;
    const float* V_ptr = V + batch_head_idx * seq_len * HEAD_DIM;
    const float* L_ptr = L + batch_head_idx * seq_len;
    const float* dO_ptr = dO + batch_head_idx * seq_len * HEAD_DIM;
    const float* D_ptr = D + batch_head_idx * seq_len;
    float* dQ_ptr = dQ + batch_head_idx * seq_len * HEAD_DIM;
    float* dK_ptr = dK + batch_head_idx * seq_len * HEAD_DIM;
    float* dV_ptr = dV + batch_head_idx * seq_len * HEAD_DIM;
    
    const int kv_start = kv_block_idx * BLOCK_N;
    if (kv_start >= seq_len) return;
    
    const int num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // Shared memory
    extern __shared__ float smem[];
    float* K_tile = smem;                                    // BLOCK_N x HEAD_DIM
    float* V_tile = K_tile + BLOCK_N * HEAD_DIM;            // BLOCK_N x HEAD_DIM
    float* Q_tile = V_tile + BLOCK_N * HEAD_DIM;            // BLOCK_M x HEAD_DIM
    float* dO_tile = Q_tile + BLOCK_M * HEAD_DIM;           // BLOCK_M x HEAD_DIM
    float* S_tile = dO_tile + BLOCK_M * HEAD_DIM;           // BLOCK_M x BLOCK_N
    float* dS_tile = S_tile + BLOCK_M * BLOCK_N;            // BLOCK_M x BLOCK_N
    float* dK_tile = dS_tile + BLOCK_M * BLOCK_N;           // BLOCK_N x HEAD_DIM
    float* dV_tile = dK_tile + BLOCK_N * HEAD_DIM;          // BLOCK_N x HEAD_DIM
    float* L_tile = dV_tile + BLOCK_N * HEAD_DIM;           // BLOCK_M
    float* D_tile = L_tile + BLOCK_M;                       // BLOCK_M
    
    // Load K and V tiles for this block
    load_tile_to_shared<BLOCK_N, HEAD_DIM>(K_ptr, K_tile, kv_start, 0, seq_len, HEAD_DIM, HEAD_DIM);
    load_tile_to_shared<BLOCK_N, HEAD_DIM>(V_ptr, V_tile, kv_start, 0, seq_len, HEAD_DIM, HEAD_DIM);
    
    // Initialize dK and dV accumulators
    for (int i = tid; i < BLOCK_N * HEAD_DIM; i += num_threads) {
        dK_tile[i] = 0.0f;
        dV_tile[i] = 0.0f;
    }
    __syncthreads();
    
    // Iterate over Q blocks
    for (int q_block = 0; q_block < num_q_blocks; q_block++) {
        int q_start = q_block * BLOCK_M;
        
        // Causal: skip if this Q block doesn't attend to this K/V block
        if (causal && q_start < kv_start) {
            continue;
        }
        
        // Load Q, dO, L, D tiles
        load_tile_to_shared<BLOCK_M, HEAD_DIM>(Q_ptr, Q_tile, q_start, 0, seq_len, HEAD_DIM, HEAD_DIM);
        load_tile_to_shared<BLOCK_M, HEAD_DIM>(dO_ptr, dO_tile, q_start, 0, seq_len, HEAD_DIM, HEAD_DIM);
        
        for (int i = tid; i < BLOCK_M; i += num_threads) {
            int global_idx = q_start + i;
            L_tile[i] = (global_idx < seq_len) ? L_ptr[global_idx] : 0.0f;
            D_tile[i] = (global_idx < seq_len) ? D_ptr[global_idx] : 0.0f;
        }
        __syncthreads();
        
        // Recompute S = Q @ K^T * scale
        matmul_ABt<BLOCK_M, BLOCK_N, HEAD_DIM>(Q_tile, K_tile, S_tile, scale);
        __syncthreads();
        
        // Apply causal mask and compute P = softmax(S)
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
                // P_ij = exp(S_ij - L_i)
                S_tile[i] = expf(S_tile[i] - L_tile[q_idx]);
            }
        }
        __syncthreads();
        
        // dV += P^T @ dO
        for (int k = tid; k < BLOCK_N; k += num_threads) {
            if (kv_start + k >= seq_len) continue;
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
        
        // dP = dO @ V^T
        // dS = P * (dP - D)
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += num_threads) {
            int q_idx = i / BLOCK_N;
            int k_idx = i % BLOCK_N;
            int global_q = q_start + q_idx;
            int global_k = kv_start + k_idx;
            
            if (global_q >= seq_len || global_k >= seq_len) {
                dS_tile[i] = 0.0f;
            } else if (causal && global_k > global_q) {
                dS_tile[i] = 0.0f;
            } else {
                // dP_ij = sum_d(dO_i,d * V_j,d)
                float dP = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    dP += dO_tile[q_idx * HEAD_DIM + d] * V_tile[k_idx * HEAD_DIM + d];
                }
                // dS_ij = P_ij * (dP_ij - D_i)
                dS_tile[i] = S_tile[i] * (dP - D_tile[q_idx]);
            }
        }
        __syncthreads();
        
        // dQ += dS @ K * scale (accumulated atomically to global memory)
        for (int q = tid; q < BLOCK_M; q += num_threads) {
            int global_q = q_start + q;
            if (global_q >= seq_len) continue;
            
            for (int d = 0; d < HEAD_DIM; d++) {
                float sum = 0.0f;
                for (int k = 0; k < BLOCK_N; k++) {
                    sum += dS_tile[q * BLOCK_N + k] * K_tile[k * HEAD_DIM + d];
                }
                atomicAdd(&dQ_ptr[global_q * HEAD_DIM + d], sum * scale);
            }
        }
        
        // dK += dS^T @ Q * scale
        for (int k = tid; k < BLOCK_N; k += num_threads) {
            if (kv_start + k >= seq_len) continue;
            for (int d = 0; d < HEAD_DIM; d++) {
                float sum = 0.0f;
                for (int q = 0; q < BLOCK_M; q++) {
                    if (q_start + q < seq_len) {
                        sum += dS_tile[q * BLOCK_N + k] * Q_tile[q * HEAD_DIM + d];
                    }
                }
                dK_tile[k * HEAD_DIM + d] += sum * scale;
            }
        }
        __syncthreads();
    }
    
    // Write dK and dV to global memory
    store_tile_from_shared<BLOCK_N, HEAD_DIM>(dK_tile, dK_ptr, kv_start, 0, seq_len, HEAD_DIM, HEAD_DIM);
    store_tile_from_shared<BLOCK_N, HEAD_DIM>(dV_tile, dV_ptr, kv_start, 0, seq_len, HEAD_DIM, HEAD_DIM);
}


// Explicit template instantiations
template __global__ void compute_D_kernel<128, 32>(const float*, const float*, float*, int);
template __global__ void compute_D_kernel<128, 64>(const float*, const float*, float*, int);
template __global__ void compute_D_kernel<128, 128>(const float*, const float*, float*, int);

template __global__ void flash_attention_backward_kernel<64, 64, 32>(
    const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, float*, float*, float*, int, float, bool);
template __global__ void flash_attention_backward_kernel<64, 64, 64>(
    const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, float*, float*, float*, int, float, bool);
template __global__ void flash_attention_backward_kernel<64, 64, 128>(
    const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, float*, float*, float*, int, float, bool);

// Launch backward kernel
FlashAttentionError launch_flash_attention_backward(
    const float* Q, const float* K, const float* V,
    const float* O, const float* L, const float* dO,
    float* dQ, float* dK, float* dV,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream
) {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    
    int batch_heads = batch_size * num_heads;
    
    // Allocate temporary D buffer
    float* D = nullptr;
    cudaError_t err = cudaMalloc(&D, batch_heads * seq_len * sizeof(float));
    if (err != cudaSuccess) {
        return err == cudaErrorMemoryAllocation ? FlashAttentionError::OUT_OF_MEMORY
                                                 : FlashAttentionError::CUDA_ERROR;
    }
    
    // Initialize dQ to zero
    err = cudaMemset(dQ, 0, batch_heads * seq_len * head_dim * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(D);
        return FlashAttentionError::CUDA_ERROR;
    }
    
    // Compute D = rowsum(dO * O)
    int d_blocks = (seq_len + 127) / 128;
    dim3 d_grid(d_blocks, batch_heads);
    
    if (head_dim == 32) {
        compute_D_kernel<128, 32><<<d_grid, 128, 0, stream>>>(dO, O, D, seq_len);
    } else if (head_dim == 64) {
        compute_D_kernel<128, 64><<<d_grid, 128, 0, stream>>>(dO, O, D, seq_len);
    } else if (head_dim == 128) {
        compute_D_kernel<128, 128><<<d_grid, 128, 0, stream>>>(dO, O, D, seq_len);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(D);
        return FlashAttentionError::CUDA_ERROR;
    }
    
    // Launch backward kernel
    int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(num_kv_blocks, batch_heads);
    dim3 block(128);
    
    size_t smem_size = (BLOCK_N * head_dim +      // K_tile
                        BLOCK_N * head_dim +      // V_tile
                        BLOCK_M * head_dim +      // Q_tile
                        BLOCK_M * head_dim +      // dO_tile
                        BLOCK_M * BLOCK_N +       // S_tile
                        BLOCK_M * BLOCK_N +       // dS_tile
                        BLOCK_N * head_dim +      // dK_tile
                        BLOCK_N * head_dim +      // dV_tile
                        BLOCK_M +                 // L_tile
                        BLOCK_M) * sizeof(float); // D_tile
    
    if (head_dim == 32) {
        flash_attention_backward_kernel<BLOCK_M, BLOCK_N, 32><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, L, dO, D, dQ, dK, dV, seq_len, scale, causal);
    } else if (head_dim == 64) {
        flash_attention_backward_kernel<BLOCK_M, BLOCK_N, 64><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, L, dO, D, dQ, dK, dV, seq_len, scale, causal);
    } else if (head_dim == 128) {
        flash_attention_backward_kernel<BLOCK_M, BLOCK_N, 128><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, L, dO, D, dQ, dK, dV, seq_len, scale, causal);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(D);
        return FlashAttentionError::CUDA_ERROR;
    }

    cudaFree(D);
    return FlashAttentionError::SUCCESS;
}

} // namespace cuflash
