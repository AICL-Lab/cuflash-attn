#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace cuflash {

// Error codes
enum class FlashAttentionError {
    SUCCESS = 0,
    INVALID_DIMENSION,      // 维度参数无效
    DIMENSION_MISMATCH,     // Q, K, V 维度不匹配
    NULL_POINTER,           // 空指针输入
    CUDA_ERROR,             // CUDA 运行时错误
    OUT_OF_MEMORY,          // 显存不足
    UNSUPPORTED_HEAD_DIM,   // 不支持的 head_dim
    UNSUPPORTED_DTYPE       // 不支持的数据类型
};

// Get error message string
const char* get_error_string(FlashAttentionError error);

// Forward pass
// Q, K, V: [batch_size, num_heads, seq_len, head_dim]
// O: [batch_size, num_heads, seq_len, head_dim]
// L: [batch_size, num_heads, seq_len] - logsumexp for backward
FlashAttentionError flash_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    float* L,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream = 0
);

// Backward pass
FlashAttentionError flash_attention_backward(
    const float* Q,
    const float* K,
    const float* V,
    const float* O,
    const float* L,
    const float* dO,
    float* dQ,
    float* dK,
    float* dV,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream = 0
);

// Half precision versions
FlashAttentionError flash_attention_forward(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    half* L,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream = 0
);

FlashAttentionError flash_attention_backward(
    const half* Q,
    const half* K,
    const half* V,
    const half* O,
    const half* L,
    const half* dO,
    half* dQ,
    half* dK,
    half* dV,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream = 0
);

} // namespace cuflash
