#include "flash_attention.h"

namespace cuflash {

// Forward declaration
FlashAttentionError launch_flash_attention_forward(const float* Q, const float* K, const float* V,
                                                   float* O, float* L, int batch_size,
                                                   int num_heads, int seq_len, int head_dim,
                                                   float scale, bool causal, cudaStream_t stream);

FlashAttentionError launch_flash_attention_backward(const float* Q, const float* K, const float* V,
                                                    const float* O, const float* L, const float* dO,
                                                    float* dQ, float* dK, float* dV, int batch_size,
                                                    int num_heads, int seq_len, int head_dim,
                                                    float scale, bool causal, cudaStream_t stream);

FlashAttentionError flash_attention_forward_fp16(const half* Q, const half* K, const half* V,
                                                 half* O, half* L, int batch_size, int num_heads,
                                                 int seq_len, int head_dim, float scale,
                                                 bool causal, cudaStream_t stream);

FlashAttentionError flash_attention_backward_fp16(const half* Q, const half* K, const half* V,
                                                  const half* O, const half* L, const half* dO,
                                                  half* dQ, half* dK, half* dV, int batch_size,
                                                  int num_heads, int seq_len, int head_dim,
                                                  float scale, bool causal, cudaStream_t stream);

FlashAttentionError launch_flash_attention_forward_fp16(const half* Q, const half* K, const half* V,
                                                        half* O, half* L, int batch_size,
                                                        int num_heads, int seq_len, int head_dim,
                                                        float scale, bool causal,
                                                        cudaStream_t stream);

const char* get_error_string(FlashAttentionError error) {
    switch (error) {
        case FlashAttentionError::SUCCESS:
            return "Success";
        case FlashAttentionError::INVALID_DIMENSION:
            return "Invalid dimension: dimensions must be positive";
        case FlashAttentionError::DIMENSION_MISMATCH:
            return "Dimension mismatch: reserved for richer shape validation APIs";
        case FlashAttentionError::NULL_POINTER:
            return "Null pointer: input or output pointer is null";
        case FlashAttentionError::CUDA_ERROR:
            return "CUDA error occurred";
        case FlashAttentionError::OUT_OF_MEMORY:
            return "Out of memory: insufficient GPU memory";
        case FlashAttentionError::UNSUPPORTED_HEAD_DIM:
            return "Unsupported head_dim: must be 32, 64, or 128";
        case FlashAttentionError::UNSUPPORTED_DTYPE:
            return "Unsupported data type";
        default:
            return "Unknown error";
    }
}

// Validate input parameters (all pointers are const since we only check for null)
static FlashAttentionError validate_params(const void* Q, const void* K, const void* V,
                                           const void* O, const void* L, int batch_size,
                                           int num_heads, int seq_len, int head_dim) {
    // Check null pointers
    if (!Q || !K || !V || !O || !L) {
        return FlashAttentionError::NULL_POINTER;
    }

    // Check dimensions
    if (batch_size <= 0 || num_heads <= 0 || seq_len <= 0 || head_dim <= 0) {
        return FlashAttentionError::INVALID_DIMENSION;
    }

    // Check supported head_dim
    if (head_dim != 32 && head_dim != 64 && head_dim != 128) {
        return FlashAttentionError::UNSUPPORTED_HEAD_DIM;
    }

    return FlashAttentionError::SUCCESS;
}

// Validate backward-specific parameters (Q,K,V,O,L plus dO,dQ,dK,dV)
static FlashAttentionError validate_params_bwd(const void* Q, const void* K, const void* V,
                                               const void* O, const void* L, const void* dO,
                                               void* dQ, void* dK, void* dV, int batch_size,
                                               int num_heads, int seq_len, int head_dim) {
    if (!dO || !dQ || !dK || !dV) {
        return FlashAttentionError::NULL_POINTER;
    }
    return validate_params(Q, K, V, O, L, batch_size, num_heads, seq_len, head_dim);
}

FlashAttentionError flash_attention_forward(const float* Q, const float* K, const float* V,
                                            float* O, float* L, int batch_size, int num_heads,
                                            int seq_len, int head_dim, float scale, bool causal,
                                            cudaStream_t stream) {
    // Validate parameters
    FlashAttentionError err =
        validate_params(Q, K, V, O, L, batch_size, num_heads, seq_len, head_dim);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }

    return launch_flash_attention_forward(Q, K, V, O, L, batch_size, num_heads, seq_len, head_dim,
                                          scale, causal, stream);
}

FlashAttentionError flash_attention_backward(const float* Q, const float* K, const float* V,
                                             const float* O, const float* L, const float* dO,
                                             float* dQ, float* dK, float* dV, int batch_size,
                                             int num_heads, int seq_len, int head_dim, float scale,
                                             bool causal, cudaStream_t stream) {
    FlashAttentionError err = validate_params_bwd(Q, K, V, O, L, dO, dQ, dK, dV, batch_size,
                                                  num_heads, seq_len, head_dim);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }

    // Launch kernel
    FlashAttentionError launch_err =
        launch_flash_attention_backward(Q, K, V, O, L, dO, dQ, dK, dV, batch_size, num_heads,
                                        seq_len, head_dim, scale, causal, stream);

    return launch_err;
}

// Half precision versions
FlashAttentionError flash_attention_forward(const half* Q, const half* K, const half* V, half* O,
                                            half* L, int batch_size, int num_heads, int seq_len,
                                            int head_dim, float scale, bool causal,
                                            cudaStream_t stream) {
    FlashAttentionError err =
        validate_params(Q, K, V, O, L, batch_size, num_heads, seq_len, head_dim);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }

    return flash_attention_forward_fp16(Q, K, V, O, L, batch_size, num_heads, seq_len, head_dim,
                                        scale, causal, stream);
}

FlashAttentionError flash_attention_backward(const half* Q, const half* K, const half* V,
                                             const half* O, const half* L, const half* dO, half* dQ,
                                             half* dK, half* dV, int batch_size, int num_heads,
                                             int seq_len, int head_dim, float scale, bool causal,
                                             cudaStream_t stream) {
    FlashAttentionError err = validate_params_bwd(Q, K, V, O, L, dO, dQ, dK, dV, batch_size,
                                                  num_heads, seq_len, head_dim);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }

    return flash_attention_backward_fp16(Q, K, V, O, L, dO, dQ, dK, dV, batch_size, num_heads,
                                         seq_len, head_dim, scale, causal, stream);
}

}  // namespace cuflash

extern "C" {

int cuflash_attention_forward_f32(const float* Q, const float* K, const float* V, float* O,
                                  float* L, int batch_size, int num_heads, int seq_len,
                                  int head_dim, float scale, bool causal, cudaStream_t stream) {
    return static_cast<int>(cuflash::flash_attention_forward(
        Q, K, V, O, L, batch_size, num_heads, seq_len, head_dim, scale, causal, stream));
}

int cuflash_attention_backward_f32(const float* Q, const float* K, const float* V, const float* O,
                                   const float* L, const float* dO, float* dQ, float* dK, float* dV,
                                   int batch_size, int num_heads, int seq_len, int head_dim,
                                   float scale, bool causal, cudaStream_t stream) {
    return static_cast<int>(cuflash::flash_attention_backward(Q, K, V, O, L, dO, dQ, dK, dV,
                                                              batch_size, num_heads, seq_len,
                                                              head_dim, scale, causal, stream));
}

int cuflash_attention_forward_f16(const half* Q, const half* K, const half* V, half* O, half* L,
                                  int batch_size, int num_heads, int seq_len, int head_dim,
                                  float scale, bool causal, cudaStream_t stream) {
    return static_cast<int>(cuflash::flash_attention_forward(
        Q, K, V, O, L, batch_size, num_heads, seq_len, head_dim, scale, causal, stream));
}

int cuflash_attention_backward_f16(const half* Q, const half* K, const half* V, const half* O,
                                   const half* L, const half* dO, half* dQ, half* dK, half* dV,
                                   int batch_size, int num_heads, int seq_len, int head_dim,
                                   float scale, bool causal, cudaStream_t stream) {
    return static_cast<int>(cuflash::flash_attention_backward(Q, K, V, O, L, dO, dQ, dK, dV,
                                                              batch_size, num_heads, seq_len,
                                                              head_dim, scale, causal, stream));
}
}
