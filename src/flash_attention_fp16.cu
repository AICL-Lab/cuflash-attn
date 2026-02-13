// Flash Attention FP16 Support

#include "flash_attention.h"
#include <cuda_fp16.h>

namespace cuflash {

// Helper functions for half precision conversion
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

__device__ __forceinline__ half float_to_half(float f) {
    return __float2half(f);
}

// FP16 Forward kernel - converts to FP32 internally for computation
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128) flash_attention_forward_fp16_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    half* __restrict__ L,
    int seq_len,
    float scale,
    bool causal
) {
    const int batch_head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;
    
    const half* Q_ptr = Q + batch_head_idx * seq_len * HEAD_DIM;
    const half* K_ptr = K + batch_head_idx * seq_len * HEAD_DIM;
    const half* V_ptr = V + batch_head_idx * seq_len * HEAD_DIM;
    half* O_ptr = O + batch_head_idx * seq_len * HEAD_DIM;
    half* L_ptr = L + batch_head_idx * seq_len;
    
    const int q_start = q_block_idx * BLOCK_M;
    if (q_start >= seq_len) return;
    
    const int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // Shared memory - use float for computation
    extern __shared__ float smem[];
    float* Q_tile = smem;
    float* K_tile = Q_tile + BLOCK_M * HEAD_DIM;
    float* V_tile = K_tile + BLOCK_N * HEAD_DIM;
    float* S_tile = V_tile + BLOCK_N * HEAD_DIM;
    float* O_tile = S_tile + BLOCK_M * BLOCK_N;
    float* m_tile = O_tile + BLOCK_M * HEAD_DIM;
    float* l_tile = m_tile + BLOCK_M;
    
    // Load Q tile (convert to float)
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
        
        if (causal && kv_start > q_start + BLOCK_M - 1) break;
        
        // Load K and V tiles (convert to float)
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
        
        // Apply causal mask
        if (causal) {
            for (int i = tid; i < BLOCK_M * BLOCK_N; i += num_threads) {
                int q_idx = i / BLOCK_N;
                int k_idx = i % BLOCK_N;
                if (kv_start + k_idx > q_start + q_idx) {
                    S_tile[i] = -INFINITY;
                }
            }
            __syncthreads();
        }
        
        // Online softmax and output accumulation
        for (int row = tid; row < BLOCK_M; row += num_threads) {
            if (q_start + row >= seq_len) continue;
            
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
            
            float m_old = m_tile[row];
            float l_old = l_tile[row];
            float m_new = fmaxf(m_old, row_max);
            float l_new = l_old * expf(m_old - m_new) + row_sum * expf(row_max - m_new);
            
            float rescale = expf(m_old - m_new);
            for (int d = 0; d < HEAD_DIM; d++) {
                O_tile[row * HEAD_DIM + d] *= rescale;
            }
            
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
    
    // Final normalization and write output (convert back to half)
    for (int row = tid; row < BLOCK_M; row += num_threads) {
        int global_row = q_start + row;
        if (global_row >= seq_len) continue;
        
        float l_inv = 1.0f / l_tile[row];
        for (int d = 0; d < HEAD_DIM; d++) {
            O_ptr[global_row * HEAD_DIM + d] = float_to_half(O_tile[row * HEAD_DIM + d] * l_inv);
        }
        L_ptr[global_row] = float_to_half(m_tile[row] + logf(l_tile[row]));
    }
}

// Template instantiations
template __global__ void flash_attention_forward_fp16_kernel<64, 64, 32>(
    const half*, const half*, const half*, half*, half*, int, float, bool);
template __global__ void flash_attention_forward_fp16_kernel<64, 64, 64>(
    const half*, const half*, const half*, half*, half*, int, float, bool);
template __global__ void flash_attention_forward_fp16_kernel<64, 64, 128>(
    const half*, const half*, const half*, half*, half*, int, float, bool);


// Launch FP16 forward kernel
void launch_flash_attention_forward_fp16(
    const half* Q, const half* K, const half* V,
    half* O, half* L,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream
) {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    
    int num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(num_q_blocks, batch_size * num_heads);
    dim3 block(128);
    
    size_t smem_size = (BLOCK_M * head_dim +
                        BLOCK_N * head_dim +
                        BLOCK_N * head_dim +
                        BLOCK_M * BLOCK_N +
                        BLOCK_M * head_dim +
                        BLOCK_M +
                        BLOCK_M) * sizeof(float);
    
    if (head_dim == 32) {
        flash_attention_forward_fp16_kernel<BLOCK_M, BLOCK_N, 32><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, L, seq_len, scale, causal);
    } else if (head_dim == 64) {
        flash_attention_forward_fp16_kernel<BLOCK_M, BLOCK_N, 64><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, L, seq_len, scale, causal);
    } else if (head_dim == 128) {
        flash_attention_forward_fp16_kernel<BLOCK_M, BLOCK_N, 128><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, L, seq_len, scale, causal);
    }
}

// Update API to support FP16
FlashAttentionError flash_attention_forward_fp16(
    const half* Q, const half* K, const half* V,
    half* O, half* L,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream
) {
    if (!Q || !K || !V || !O || !L) {
        return FlashAttentionError::NULL_POINTER;
    }
    if (batch_size <= 0 || num_heads <= 0 || seq_len <= 0 || head_dim <= 0) {
        return FlashAttentionError::INVALID_DIMENSION;
    }
    if (head_dim != 32 && head_dim != 64 && head_dim != 128) {
        return FlashAttentionError::UNSUPPORTED_HEAD_DIM;
    }
    
    launch_flash_attention_forward_fp16(Q, K, V, O, L,
        batch_size, num_heads, seq_len, head_dim, scale, causal, stream);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }
    
    return FlashAttentionError::SUCCESS;
}

} // namespace cuflash
