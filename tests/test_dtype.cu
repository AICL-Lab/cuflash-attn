// Data Type Support Tests
// Feature: cuflash-attn, Property 6: 数据类型支持

#include <gtest/gtest.h>
#if CUFLASH_ENABLE_RAPIDCHECK
 #include <rapidcheck.h>
 #include <rapidcheck/gtest.h>
#endif
#include <vector>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "flash_attention.h"

namespace cuflash {
namespace test {

// Test FP16 forward pass produces results close to FP32
// (FP16 backward returns UNSUPPORTED_DTYPE - see FP16BackwardUnsupported)
TEST(DTypeTest, FP16ForwardMatchesFP32) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 16;
    const int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // Create FP32 inputs
    std::vector<float> h_Q_f32(qkv_size), h_K_f32(qkv_size), h_V_f32(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q_f32[i] = dist(gen);
        h_K_f32[i] = dist(gen);
        h_V_f32[i] = dist(gen);
    }
    
    // Convert to FP16
    std::vector<half> h_Q_f16(qkv_size), h_K_f16(qkv_size), h_V_f16(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q_f16[i] = __float2half(h_Q_f32[i]);
        h_K_f16[i] = __float2half(h_K_f32[i]);
        h_V_f16[i] = __float2half(h_V_f32[i]);
    }
    
    // FP32 computation
    float *d_Q_f32, *d_K_f32, *d_V_f32, *d_O_f32, *d_L_f32;
    cudaMalloc(&d_Q_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_K_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_V_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_O_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_L_f32, l_size * sizeof(float));
    
    cudaMemcpy(d_Q_f32, h_Q_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_f32, h_K_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_f32, h_V_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    
    auto err = flash_attention_forward(d_Q_f32, d_K_f32, d_V_f32, d_O_f32, d_L_f32,
        batch_size, num_heads, seq_len, head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();
    
    std::vector<float> h_O_f32(qkv_size);
    cudaMemcpy(h_O_f32.data(), d_O_f32, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // FP16 computation
    half *d_Q_f16, *d_K_f16, *d_V_f16, *d_O_f16, *d_L_f16;
    cudaMalloc(&d_Q_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_K_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_V_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_O_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_L_f16, l_size * sizeof(half));
    
    cudaMemcpy(d_Q_f16, h_Q_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_f16, h_K_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_f16, h_V_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    
    err = flash_attention_forward(d_Q_f16, d_K_f16, d_V_f16, d_O_f16, d_L_f16,
        batch_size, num_heads, seq_len, head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();
    
    std::vector<half> h_O_f16(qkv_size);
    cudaMemcpy(h_O_f16.data(), d_O_f16, qkv_size * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Compare FP16 and FP32 results (allow larger tolerance for FP16)
    float max_diff = 0.0f;
    for (size_t i = 0; i < qkv_size; i++) {
        float diff = std::abs(h_O_f32[i] - __half2float(h_O_f16[i]));
        max_diff = std::max(max_diff, diff);
    }
    
    // FP16 has lower precision, allow 1e-2 tolerance
    EXPECT_LT(max_diff, 1e-2f) << "Max diff between FP32 and FP16: " << max_diff;
    
    cudaFree(d_Q_f32); cudaFree(d_K_f32); cudaFree(d_V_f32); cudaFree(d_O_f32); cudaFree(d_L_f32);
    cudaFree(d_Q_f16); cudaFree(d_K_f16); cudaFree(d_V_f16); cudaFree(d_O_f16); cudaFree(d_L_f16);
}

// Test that FP16 backward returns UNSUPPORTED_DTYPE (documented behavior)
TEST(DTypeTest, FP16BackwardUnsupported) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 8;
    const int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // Create FP16 inputs
    std::vector<half> h_Q_f16(qkv_size), h_K_f16(qkv_size), h_V_f16(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        float val = dist(gen);
        h_Q_f16[i] = __float2half(val);
        h_K_f16[i] = __float2half(val);
        h_V_f16[i] = __float2half(val);
    }
    
    // Allocate device memory
    half *d_Q, *d_K, *d_V, *d_O, *d_L;
    half *d_dO, *d_dQ, *d_dK, *d_dV;
    cudaMalloc(&d_Q, qkv_size * sizeof(half));
    cudaMalloc(&d_K, qkv_size * sizeof(half));
    cudaMalloc(&d_V, qkv_size * sizeof(half));
    cudaMalloc(&d_O, qkv_size * sizeof(half));
    cudaMalloc(&d_L, l_size * sizeof(half));
    cudaMalloc(&d_dO, qkv_size * sizeof(half));
    cudaMalloc(&d_dQ, qkv_size * sizeof(half));
    cudaMalloc(&d_dK, qkv_size * sizeof(half));
    cudaMalloc(&d_dV, qkv_size * sizeof(half));
    
    cudaMemcpy(d_Q, h_Q_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    
    // FP16 backward should return UNSUPPORTED_DTYPE
    auto err = flash_attention_backward(d_Q, d_K, d_V, d_O, d_L, d_dO, d_dQ, d_dK, d_dV,
        batch_size, num_heads, seq_len, head_dim, scale, false, 0);
    EXPECT_EQ(err, FlashAttentionError::UNSUPPORTED_DTYPE);
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
    cudaFree(d_dO); cudaFree(d_dQ); cudaFree(d_dK); cudaFree(d_dV);
}

// Test head_dim=128 (the largest supported dimension)
TEST(DTypeTest, HeadDim128) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 16;
    const int head_dim = 128;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;
    
    std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = dist(gen);
        h_K[i] = dist(gen);
        h_V[i] = dist(gen);
    }
    
    std::vector<float> h_O(qkv_size), h_L(l_size);
    
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_L, l_size * sizeof(float));
    
    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    
    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L,
        batch_size, num_heads, seq_len, head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    
    cudaDeviceSynchronize();
    cudaMemcpy(h_O.data(), d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Check output is finite
    for (size_t i = 0; i < qkv_size; i++) {
        EXPECT_TRUE(std::isfinite(h_O[i])) << "Output at index " << i << " is not finite";
    }
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
}


TEST(DTypeTest, FP16BackwardReturnsUnsupportedDType) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 8;
    const int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    half dummy = __float2half(0.0f);

    auto err = flash_attention_backward(
        &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy,
        batch_size, num_heads, seq_len, head_dim, scale, false, 0);

    EXPECT_EQ(err, FlashAttentionError::UNSUPPORTED_DTYPE);
}

#if CUFLASH_ENABLE_RAPIDCHECK
// Property test: FP16 results should be close to FP32
// Feature: cuflash-attn, Property 6: 数据类型支持
// Validates: Requirements 7.4
RC_GTEST_PROP(DTypeProperty, FP16ClosesToFP32, ()) {
    int batch_size = 1;
    int num_heads = 1;
    int seq_len = *rc::gen::inRange(4, 33);
    int head_dim = *rc::gen::element(32, 64, 128); // Include 128 for full coverage
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;
    
    std::vector<float> h_Q_f32(qkv_size), h_K_f32(qkv_size), h_V_f32(qkv_size);
    std::vector<half> h_Q_f16(qkv_size), h_K_f16(qkv_size), h_V_f16(qkv_size);
    
    for (size_t i = 0; i < qkv_size; i++) {
        float val = *rc::gen::inRange(-100, 100) * 0.01f;
        h_Q_f32[i] = val;
        h_Q_f16[i] = __float2half(val);
        val = *rc::gen::inRange(-100, 100) * 0.01f;
        h_K_f32[i] = val;
        h_K_f16[i] = __float2half(val);
        val = *rc::gen::inRange(-100, 100) * 0.01f;
        h_V_f32[i] = val;
        h_V_f16[i] = __float2half(val);
    }
    
    // Allocate and run FP32
    float *d_Q_f32, *d_K_f32, *d_V_f32, *d_O_f32, *d_L_f32;
    cudaMalloc(&d_Q_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_K_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_V_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_O_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_L_f32, l_size * sizeof(float));
    
    cudaMemcpy(d_Q_f32, h_Q_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_f32, h_K_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_f32, h_V_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    
    auto err = flash_attention_forward(d_Q_f32, d_K_f32, d_V_f32, d_O_f32, d_L_f32,
        batch_size, num_heads, seq_len, head_dim, scale, false, 0);
    RC_ASSERT(err == FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();
    
    std::vector<float> h_O_f32(qkv_size);
    cudaMemcpy(h_O_f32.data(), d_O_f32, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Allocate and run FP16
    half *d_Q_f16, *d_K_f16, *d_V_f16, *d_O_f16, *d_L_f16;
    cudaMalloc(&d_Q_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_K_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_V_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_O_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_L_f16, l_size * sizeof(half));
    
    cudaMemcpy(d_Q_f16, h_Q_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_f16, h_K_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_f16, h_V_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    
    err = flash_attention_forward(d_Q_f16, d_K_f16, d_V_f16, d_O_f16, d_L_f16,
        batch_size, num_heads, seq_len, head_dim, scale, false, 0);
    RC_ASSERT(err == FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();
    
    std::vector<half> h_O_f16(qkv_size);
    cudaMemcpy(h_O_f16.data(), d_O_f16, qkv_size * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Compare
    float max_diff = 0.0f;
    for (size_t i = 0; i < qkv_size; i++) {
        float diff = std::abs(h_O_f32[i] - __half2float(h_O_f16[i]));
        max_diff = std::max(max_diff, diff);
    }
    RC_ASSERT(max_diff < 1e-2f);
    
    cudaFree(d_Q_f32); cudaFree(d_K_f32); cudaFree(d_V_f32); cudaFree(d_O_f32); cudaFree(d_L_f32);
    cudaFree(d_Q_f16); cudaFree(d_K_f16); cudaFree(d_V_f16); cudaFree(d_O_f16); cudaFree(d_L_f16);
}

#endif

} // namespace test
} // namespace cuflash
