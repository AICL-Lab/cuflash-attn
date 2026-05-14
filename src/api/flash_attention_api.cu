#include "cuflash/flash_attention.h"

namespace cuflash {

// Forward declarations for unified typed launch functions
template<typename InputT>
FlashAttentionError launch_flash_attention_forward_typed(const InputT* Q, const InputT* K,
                                                         const InputT* V, InputT* O, InputT* L,
                                                         int batch_size, int num_heads, int seq_len,
                                                         int head_dim, float scale, bool causal,
                                                         cudaStream_t stream);

template<typename InputT>
FlashAttentionError launch_flash_attention_backward_typed(
    const InputT* Q, const InputT* K, const InputT* V, const InputT* O, const InputT* L,
    const InputT* dO, InputT* dQ, InputT* dK, InputT* dV, int batch_size, int num_heads,
    int seq_len, int head_dim, float scale, bool causal, cudaStream_t stream);

// Error string lookup
const char* get_error_string(FlashAttentionError error) {
    switch (error) {
        case FlashAttentionError::SUCCESS:
            return "Success";
        case FlashAttentionError::INVALID_DIMENSION:
            return "Invalid dimension: dimensions must be positive";
        case FlashAttentionError::DIMENSION_MISMATCH:
            return "Dimension mismatch: Q, K, V tensors have incompatible shapes";
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

// Validate common parameters shared by forward and backward passes
static FlashAttentionError validate_common_params(const void* Q, const void* K, const void* V,
                                                  const void* O, const void* L, int batch_size,
                                                  int num_heads, int seq_len, int head_dim,
                                                  float scale) {
    // Check null pointers
    if (!Q || !K || !V || !O || !L) {
        return FlashAttentionError::NULL_POINTER;
    }

    // Check dimensions are positive
    if (batch_size <= 0 || num_heads <= 0 || seq_len <= 0 || head_dim <= 0) {
        return FlashAttentionError::INVALID_DIMENSION;
    }

    // Check supported head_dim values
    if (head_dim != 32 && head_dim != 64 && head_dim != 128) {
        return FlashAttentionError::UNSUPPORTED_HEAD_DIM;
    }

    // Check scale is finite and non-negative
    if (!isfinite(scale) || scale < 0.0f) {
        return FlashAttentionError::INVALID_DIMENSION;
    }

    return FlashAttentionError::SUCCESS;
}

// Validate forward pass parameters
static FlashAttentionError validate_params_forward(const void* Q, const void* K, const void* V,
                                                   const void* O, const void* L, int batch_size,
                                                   int num_heads, int seq_len, int head_dim,
                                                   float scale) {
    return validate_common_params(Q, K, V, O, L, batch_size, num_heads, seq_len, head_dim, scale);
}

// Validate backward pass parameters (includes additional gradient pointers)
static FlashAttentionError validate_params_backward(const void* Q, const void* K, const void* V,
                                                    const void* O, const void* L, const void* dO,
                                                    const void* dQ, const void* dK, const void* dV,
                                                    int batch_size, int num_heads, int seq_len,
                                                    int head_dim, float scale) {
    // Check backward-specific null pointers first
    if (!dO || !dQ || !dK || !dV) {
        return FlashAttentionError::NULL_POINTER;
    }

    return validate_common_params(Q, K, V, O, L, batch_size, num_heads, seq_len, head_dim, scale);
}

// ===== C++ namespace API =====

// Forward pass (FP32)
FlashAttentionError flash_attention_forward(const float* Q, const float* K, const float* V,
                                            float* O, float* L, int batch_size, int num_heads,
                                            int seq_len, int head_dim, float scale, bool causal,
                                            cudaStream_t stream) {
    FlashAttentionError err =
        validate_params_forward(Q, K, V, O, L, batch_size, num_heads, seq_len, head_dim, scale);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }

    return launch_flash_attention_forward_typed<float>(Q, K, V, O, L, batch_size, num_heads,
                                                       seq_len, head_dim, scale, causal, stream);
}

// Forward pass (FP16)
FlashAttentionError flash_attention_forward(const half* Q, const half* K, const half* V, half* O,
                                            half* L, int batch_size, int num_heads, int seq_len,
                                            int head_dim, float scale, bool causal,
                                            cudaStream_t stream) {
    FlashAttentionError err =
        validate_params_forward(Q, K, V, O, L, batch_size, num_heads, seq_len, head_dim, scale);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }

    return launch_flash_attention_forward_typed<half>(Q, K, V, O, L, batch_size, num_heads, seq_len,
                                                      head_dim, scale, causal, stream);
}

// Backward pass (FP32)
FlashAttentionError flash_attention_backward(const float* Q, const float* K, const float* V,
                                             const float* O, const float* L, const float* dO,
                                             float* dQ, float* dK, float* dV, int batch_size,
                                             int num_heads, int seq_len, int head_dim, float scale,
                                             bool causal, cudaStream_t stream) {
    FlashAttentionError err = validate_params_backward(Q, K, V, O, L, dO, dQ, dK, dV, batch_size,
                                                       num_heads, seq_len, head_dim, scale);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }

    return launch_flash_attention_backward_typed<float>(Q, K, V, O, L, dO, dQ, dK, dV, batch_size,
                                                        num_heads, seq_len, head_dim, scale, causal,
                                                        stream);
}

// Backward pass (FP16)
FlashAttentionError flash_attention_backward(const half* Q, const half* K, const half* V,
                                             const half* O, const half* L, const half* dO, half* dQ,
                                             half* dK, half* dV, int batch_size, int num_heads,
                                             int seq_len, int head_dim, float scale, bool causal,
                                             cudaStream_t stream) {
    FlashAttentionError err = validate_params_backward(Q, K, V, O, L, dO, dQ, dK, dV, batch_size,
                                                       num_heads, seq_len, head_dim, scale);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }

    return launch_flash_attention_backward_typed<half>(Q, K, V, O, L, dO, dQ, dK, dV, batch_size,
                                                       num_heads, seq_len, head_dim, scale, causal,
                                                       stream);
}

}  // namespace cuflash

// ===== C ABI wrappers =====

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

const char* cuflash_error_string(int error_code) {
    return cuflash::get_error_string(static_cast<cuflash::FlashAttentionError>(error_code));
}

}  // extern "C"
