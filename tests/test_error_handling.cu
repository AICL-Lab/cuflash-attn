// Error Handling Tests
// Feature: cuflash-attn, Property 7: 无效输入错误处理

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <cuda_runtime.h>
#include "flash_attention.h"

namespace cuflash {
namespace test {

// Test null pointer handling
TEST(ErrorHandlingTest, NullPointers) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 16;
    const int head_dim = 32;
    const float scale = 1.0f;
    
    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;
    
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_L, l_size * sizeof(float));
    
    // Test null Q
    auto err = flash_attention_forward(nullptr, d_K, d_V, d_O, d_L,
        batch_size, num_heads, seq_len, head_dim, scale, false, 0);
    EXPECT_EQ(err, FlashAttentionError::NULL_POINTER);
    
    // Test null K
    err = flash_attention_forward(d_Q, nullptr, d_V, d_O, d_L,
        batch_size, num_heads, seq_len, head_dim, scale, false, 0);
    EXPECT_EQ(err, FlashAttentionError::NULL_POINTER);
    
    // Test null V
    err = flash_attention_forward(d_Q, d_K, nullptr, d_O, d_L,
        batch_size, num_heads, seq_len, head_dim, scale, false, 0);
    EXPECT_EQ(err, FlashAttentionError::NULL_POINTER);
    
    // Test null O
    err = flash_attention_forward(d_Q, d_K, d_V, nullptr, d_L,
        batch_size, num_heads, seq_len, head_dim, scale, false, 0);
    EXPECT_EQ(err, FlashAttentionError::NULL_POINTER);
    
    // Test null L
    err = flash_attention_forward(d_Q, d_K, d_V, d_O, nullptr,
        batch_size, num_heads, seq_len, head_dim, scale, false, 0);
    EXPECT_EQ(err, FlashAttentionError::NULL_POINTER);
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
}


// Test invalid dimensions
TEST(ErrorHandlingTest, InvalidDimensions) {
    const float scale = 1.0f;
    
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, 1024 * sizeof(float));
    cudaMalloc(&d_K, 1024 * sizeof(float));
    cudaMalloc(&d_V, 1024 * sizeof(float));
    cudaMalloc(&d_O, 1024 * sizeof(float));
    cudaMalloc(&d_L, 64 * sizeof(float));
    
    // Test zero batch_size
    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L,
        0, 1, 16, 32, scale, false, 0);
    EXPECT_EQ(err, FlashAttentionError::INVALID_DIMENSION);
    
    // Test negative batch_size
    err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L,
        -1, 1, 16, 32, scale, false, 0);
    EXPECT_EQ(err, FlashAttentionError::INVALID_DIMENSION);
    
    // Test zero num_heads
    err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L,
        1, 0, 16, 32, scale, false, 0);
    EXPECT_EQ(err, FlashAttentionError::INVALID_DIMENSION);
    
    // Test zero seq_len
    err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L,
        1, 1, 0, 32, scale, false, 0);
    EXPECT_EQ(err, FlashAttentionError::INVALID_DIMENSION);
    
    // Test zero head_dim
    err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L,
        1, 1, 16, 0, scale, false, 0);
    EXPECT_EQ(err, FlashAttentionError::INVALID_DIMENSION);
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
}

// Test unsupported head_dim
TEST(ErrorHandlingTest, UnsupportedHeadDim) {
    const float scale = 1.0f;
    
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, 1024 * sizeof(float));
    cudaMalloc(&d_K, 1024 * sizeof(float));
    cudaMalloc(&d_V, 1024 * sizeof(float));
    cudaMalloc(&d_O, 1024 * sizeof(float));
    cudaMalloc(&d_L, 64 * sizeof(float));
    
    // Test unsupported head_dim (not 32, 64, or 128)
    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L,
        1, 1, 16, 48, scale, false, 0);
    EXPECT_EQ(err, FlashAttentionError::UNSUPPORTED_HEAD_DIM);
    
    err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L,
        1, 1, 16, 256, scale, false, 0);
    EXPECT_EQ(err, FlashAttentionError::UNSUPPORTED_HEAD_DIM);
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
}

// Test error string function
TEST(ErrorHandlingTest, ErrorStrings) {
    EXPECT_STREQ(get_error_string(FlashAttentionError::SUCCESS), "Success");
    EXPECT_STREQ(get_error_string(FlashAttentionError::NULL_POINTER), 
        "Null pointer: input or output pointer is null");
    EXPECT_STREQ(get_error_string(FlashAttentionError::INVALID_DIMENSION),
        "Invalid dimension: dimensions must be positive");
    EXPECT_STREQ(get_error_string(FlashAttentionError::UNSUPPORTED_HEAD_DIM),
        "Unsupported head_dim: must be 32, 64, or 128");
}


// Property test: Invalid inputs should return errors, not crash
// Feature: cuflash-attn, Property 7: 无效输入错误处理
// Validates: Requirements 7.3
RC_GTEST_PROP(ErrorHandlingProperty, InvalidInputsReturnErrors, ()) {
    // Generate invalid dimension scenarios
    int scenario = *rc::gen::inRange(0, 5);
    
    int batch_size = 1;
    int num_heads = 1;
    int seq_len = 16;
    int head_dim = 32;
    
    switch (scenario) {
        case 0: batch_size = *rc::gen::inRange(-10, 0); break;
        case 1: num_heads = *rc::gen::inRange(-10, 0); break;
        case 2: seq_len = *rc::gen::inRange(-10, 0); break;
        case 3: head_dim = *rc::gen::inRange(-10, 0); break;
        case 4: head_dim = *rc::gen::element(16, 48, 96, 256); break;  // Unsupported
    }
    
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, 1024 * sizeof(float));
    cudaMalloc(&d_K, 1024 * sizeof(float));
    cudaMalloc(&d_V, 1024 * sizeof(float));
    cudaMalloc(&d_O, 1024 * sizeof(float));
    cudaMalloc(&d_L, 64 * sizeof(float));
    
    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L,
        batch_size, num_heads, seq_len, head_dim, 1.0f, false, 0);
    
    // Should return an error, not SUCCESS
    RC_ASSERT(err != FlashAttentionError::SUCCESS);
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
}

// Property test: Valid inputs should succeed
RC_GTEST_PROP(ErrorHandlingProperty, ValidInputsSucceed, ()) {
    int batch_size = *rc::gen::inRange(1, 4);
    int num_heads = *rc::gen::inRange(1, 8);
    int seq_len = *rc::gen::inRange(1, 65);
    int head_dim = *rc::gen::element(32, 64, 128);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;
    
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_L, l_size * sizeof(float));
    
    // Initialize with zeros
    cudaMemset(d_Q, 0, qkv_size * sizeof(float));
    cudaMemset(d_K, 0, qkv_size * sizeof(float));
    cudaMemset(d_V, 0, qkv_size * sizeof(float));
    
    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L,
        batch_size, num_heads, seq_len, head_dim, scale, false, 0);
    
    RC_ASSERT(err == FlashAttentionError::SUCCESS);
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
}

} // namespace test
} // namespace cuflash
