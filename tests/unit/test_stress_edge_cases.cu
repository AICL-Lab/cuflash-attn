// Stress Tests and Edge Case Tests
// Validates robustness under extreme and boundary conditions

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cuflash/flash_attention.h"

namespace cuflash {
namespace test {

namespace {

// Helper: allocate device memory and fill with data
template<typename T>
std::vector<T*> allocate_device_tensors(const std::vector<size_t>& element_counts) {
    std::vector<T*> ptrs;
    ptrs.reserve(element_counts.size());
    for (size_t count : element_counts) {
        T* d_ptr = nullptr;
        cudaMalloc(&d_ptr, count * sizeof(T));
        ptrs.push_back(d_ptr);
    }
    return ptrs;
}

void free_device_memory(const std::vector<void*>& ptrs) {
    for (auto* ptr : ptrs) {
        cudaFree(ptr);
    }
}

void launch_kernel_and_sync(const std::function<void(cudaStream_t)>& kernel_func,
                            cudaStream_t stream) {
    kernel_func(stream);
    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess) << "CUDA kernel failed: " << cudaGetErrorString(err);
}

}  // namespace

// ============================================================================
// Edge Case Tests - Minimum Dimensions
// ============================================================================

TEST(EdgeCaseTest, MinimumValidDimensions) {
    // batch_size=1, num_heads=1, seq_len=1, head_dim=32 (minimum supported)
    int batch_size = 1, num_heads = 1, seq_len = 1, head_dim = 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    auto devs = allocate_device_tensors<float>({qkv_size, qkv_size, qkv_size, qkv_size, l_size});
    float *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2];
    float *d_O = devs[3], *d_L = devs[4];

    // Initialize with simple values
    std::vector<float> h_q(qkv_size, 0.5f);
    std::vector<float> h_k(qkv_size, 0.3f);
    std::vector<float> h_v(qkv_size, 0.7f);

    cudaMemcpy(d_Q, h_q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_k.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_v.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    auto err = cuflash::flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads,
                                                seq_len, head_dim, scale, false, stream);

    EXPECT_EQ(err, cuflash::FlashAttentionError::SUCCESS);

    cudaStreamDestroy(stream);
    free_device_memory(devs);
}

TEST(EdgeCaseTest, MinimumHeadDim) {
    // head_dim=32 is the minimum supported
    int batch_size = 2, num_heads = 4, seq_len = 64, head_dim = 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    auto devs = allocate_device_tensors<float>({qkv_size, qkv_size, qkv_size, qkv_size, l_size});
    float *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2];
    float *d_O = devs[3], *d_L = devs[4];

    // Fill with random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::vector<float> h_data(qkv_size);
    for (auto& v : h_data)
        v = dis(gen);

    cudaMemcpy(d_Q, h_data.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_data.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_data.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    auto err = cuflash::flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads,
                                                seq_len, head_dim, scale, true, stream);

    EXPECT_EQ(err, cuflash::FlashAttentionError::SUCCESS);

    cudaStreamDestroy(stream);
    free_device_memory(devs);
}

// ============================================================================
// Edge Case Tests - Invalid Parameters
// ============================================================================

TEST(EdgeCaseTest, NullPointerInput) {
    auto err = cuflash::flash_attention_forward(nullptr, nullptr, nullptr, nullptr, nullptr, 1, 1,
                                                64, 64, 0.125f, false, 0);
    EXPECT_EQ(err, cuflash::FlashAttentionError::NULL_POINTER);
}

TEST(EdgeCaseTest, UnsupportedHeadDim) {
    // head_dim=48 is not supported (only 32, 64, 128)
    int batch_size = 1, num_heads = 1, seq_len = 64, head_dim = 48;

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    auto devs = allocate_device_tensors<float>({qkv_size, qkv_size, qkv_size, qkv_size, l_size});
    float *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2];
    float *d_O = devs[3], *d_L = devs[4];

    auto err = cuflash::flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads,
                                                seq_len, head_dim, 0.125f, false, 0);

    EXPECT_EQ(err, cuflash::FlashAttentionError::UNSUPPORTED_HEAD_DIM);

    free_device_memory(devs);
}

TEST(EdgeCaseTest, ZeroSequenceLength) {
    auto devs = allocate_device_tensors<float>({1, 1, 1, 1, 1});
    float *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2];
    float *d_O = devs[3], *d_L = devs[4];

    auto err =
        cuflash::flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, 1, 1, 0, 64, 0.125f, false, 0);

    EXPECT_EQ(err, cuflash::FlashAttentionError::INVALID_DIMENSION);

    free_device_memory(devs);
}

TEST(EdgeCaseTest, ZeroBatchSize) {
    auto devs = allocate_device_tensors<float>({1, 1, 1, 1, 1});
    float *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2];
    float *d_O = devs[3], *d_L = devs[4];

    auto err =
        cuflash::flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, 0, 1, 64, 64, 0.125f, false, 0);

    EXPECT_EQ(err, cuflash::FlashAttentionError::INVALID_DIMENSION);

    free_device_memory(devs);
}

// ============================================================================
// Stress Tests - Large Sequences
// ============================================================================

TEST(StressTest, LargeSequenceForward) {
    // Test seq_len=8192 with head_dim=128 (large memory usage)
    int batch_size = 1, num_heads = 8, seq_len = 8192, head_dim = 128;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    auto devs = allocate_device_tensors<float>({qkv_size, qkv_size, qkv_size, qkv_size, l_size});
    float *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2];
    float *d_O = devs[3], *d_L = devs[4];

    // Fill with random data
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
    std::vector<float> h_data(qkv_size);
    for (auto& v : h_data)
        v = dis(gen);

    cudaMemcpy(d_Q, h_data.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_data.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_data.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    auto err = cuflash::flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads,
                                                seq_len, head_dim, scale, true, stream);

    EXPECT_EQ(err, cuflash::FlashAttentionError::SUCCESS);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    free_device_memory(devs);
}

TEST(StressTest, LargeSequenceBackward) {
    int batch_size = 1, num_heads = 8, seq_len = 4096, head_dim = 64;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    // Q, K, V, O, L, dO, dQ, dK, dV
    auto devs = allocate_device_tensors<float>(
        {qkv_size, qkv_size, qkv_size, qkv_size, l_size, qkv_size, qkv_size, qkv_size, qkv_size});

    float *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2], *d_O = devs[3];
    float *d_L = devs[4], *d_dO = devs[5], *d_dQ = devs[6], *d_dK = devs[7];
    float* d_dV = devs[8];

    std::mt19937 gen(456);
    std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
    std::vector<float> h_data(qkv_size);
    for (auto& v : h_data)
        v = dis(gen);
    std::vector<float> h_l(l_size, 0.0f);

    cudaMemcpy(d_Q, h_data.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_data.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_data.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_O, h_data.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, h_l.data(), l_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO, h_data.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    auto err = cuflash::flash_attention_backward(d_Q, d_K, d_V, d_O, d_L, d_dO, d_dQ, d_dK, d_dV,
                                                 batch_size, num_heads, seq_len, head_dim, scale,
                                                 true, stream);

    EXPECT_EQ(err, cuflash::FlashAttentionError::SUCCESS);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    free_device_memory(devs);
}

// ============================================================================
// Stress Tests - Repeated Execution (Memory Leak Detection)
// ============================================================================

TEST(StressTest, RepeatedExecutionNoLeak) {
    int batch_size = 2, num_heads = 4, seq_len = 256, head_dim = 64;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    auto devs = allocate_device_tensors<float>({qkv_size, qkv_size, qkv_size, qkv_size, l_size});
    float *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2];
    float *d_O = devs[3], *d_L = devs[4];

    std::mt19937 gen(789);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::vector<float> h_data(qkv_size);
    for (auto& v : h_data)
        v = dis(gen);

    cudaMemcpy(d_Q, h_data.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_data.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_data.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    // Check memory before and after repeated executions
    size_t free_before = 0, total = 0;
    cudaMemGetInfo(&free_before, &total);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    const int iterations = 100;
    for (int i = 0; i < iterations; ++i) {
        auto err = cuflash::flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads,
                                                    seq_len, head_dim, scale, false, stream);
        ASSERT_EQ(err, cuflash::FlashAttentionError::SUCCESS);
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    size_t free_after = 0;
    cudaMemGetInfo(&free_after, &total);

    // Memory should not leak across iterations
    // Allow small tolerance for CUDA runtime overhead
    size_t memory_diff =
        (free_after > free_before) ? (free_after - free_before) : (free_before - free_after);
    EXPECT_LT(memory_diff, 1024 * 1024)  // Less than 1MB difference
        << "Potential memory leak detected: " << memory_diff << " bytes difference after "
        << iterations << " iterations";

    free_device_memory(devs);
}

// ============================================================================
// Multi-Head Stress Tests
// ============================================================================

TEST(StressTest, MultiHeadLarge) {
    // Many heads with moderate sequence length
    int batch_size = 1, num_heads = 32, seq_len = 512, head_dim = 64;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    auto devs = allocate_device_tensors<float>({qkv_size, qkv_size, qkv_size, qkv_size, l_size});
    float *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2];
    float *d_O = devs[3], *d_L = devs[4];

    std::mt19937 gen(101);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    std::vector<float> h_data(qkv_size);
    for (auto& v : h_data)
        v = dis(gen);

    cudaMemcpy(d_Q, h_data.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_data.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_data.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    auto err = cuflash::flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads,
                                                seq_len, head_dim, scale, true, stream);

    EXPECT_EQ(err, cuflash::FlashAttentionError::SUCCESS);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    free_device_memory(devs);
}

// ============================================================================
// Causal Mask Boundary Tests
// ============================================================================

TEST(EdgeCaseTest, CausalMaskSeqLenOne) {
    // With seq_len=1, causal mask should have no effect (single token attends to itself)
    int batch_size = 1, num_heads = 1, seq_len = 1, head_dim = 64;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    auto devs = allocate_device_tensors<float>({qkv_size, qkv_size, qkv_size, qkv_size, l_size});
    float *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2];
    float *d_O = devs[3], *d_L = devs[4];

    std::vector<float> h_q(qkv_size, 1.0f);
    std::vector<float> h_k(qkv_size, 1.0f);
    std::vector<float> h_v(qkv_size, 1.0f);

    cudaMemcpy(d_Q, h_q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_k.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_v.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // Run with causal mask
    auto err_causal = cuflash::flash_attention_forward(
        d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads, seq_len, head_dim, scale, true, stream);
    EXPECT_EQ(err_causal, cuflash::FlashAttentionError::SUCCESS);

    std::vector<float> h_o_causal(qkv_size);
    cudaMemcpy(h_o_causal.data(), d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Run without causal mask
    err_causal = cuflash::flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads,
                                                  seq_len, head_dim, scale, false, stream);
    EXPECT_EQ(err_causal, cuflash::FlashAttentionError::SUCCESS);

    std::vector<float> h_o_non_causal(qkv_size);
    cudaMemcpy(h_o_non_causal.data(), d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Results should be identical for seq_len=1
    for (size_t i = 0; i < qkv_size; ++i) {
        EXPECT_NEAR(h_o_causal[i], h_o_non_causal[i], 1e-5f);
    }

    cudaStreamDestroy(stream);
    free_device_memory(devs);
}

}  // namespace test
}  // namespace cuflash
