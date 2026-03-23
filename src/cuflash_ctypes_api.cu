// C ABI wrapper for Python ctypes access
// Exposes flash_attention functions with C linkage for cross-language binding

#include "flash_attention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" {

// FP32 Forward
cuflash::FlashAttentionError cuflash_forward_f32(
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
    cudaStream_t stream
) {
    return cuflash::flash_attention_forward(
        Q, K, V, O, L,
        batch_size, num_heads, seq_len, head_dim,
        scale, causal, stream
    );
}

// FP32 Backward
cuflash::FlashAttentionError cuflash_backward_f32(
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
    cudaStream_t stream
) {
    return cuflash::flash_attention_backward(
        Q, K, V, O, L, dO, dQ, dK, dV,
        batch_size, num_heads, seq_len, head_dim,
        scale, causal, stream
    );
}

// FP16 Forward
cuflash::FlashAttentionError cuflash_forward_f16(
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
    cudaStream_t stream
) {
    return cuflash::flash_attention_forward(
        Q, K, V, O, L,
        batch_size, num_heads, seq_len, head_dim,
        scale, causal, stream
    );
}

// FP16 Backward
cuflash::FlashAttentionError cuflash_backward_f16(
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
    cudaStream_t stream
) {
    return cuflash::flash_attention_backward(
        Q, K, V, O, L, dO, dQ, dK, dV,
        batch_size, num_heads, seq_len, head_dim,
        scale, causal, stream
    );
}

} // extern "C"
