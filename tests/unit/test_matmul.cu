// Matmul Unit Tests
// Tests for cuflash::kernels::matmul_* operations
//
// NOTE: These tests validate the standalone kernel API for matmul operations.
// The production forward/backward kernels use impl::matmul_* directly in
// shared memory for better performance. The kernels::matmul_* API provides
// a testable interface for the same underlying algorithms.
//
// Test coverage: The numerical correctness of impl::matmul_* is indirectly
// validated through the end-to-end tests in tests/integration/.

#include <gtest/gtest.h>
#if CUFLASH_ENABLE_RAPIDCHECK
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#endif
#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <vector>

#include "cuflash/kernels/matmul.cuh"

namespace cuflash {
namespace test {

// =============================================================================
// Test Utilities
// =============================================================================

class MatmulTest : public ::testing::Test {
   protected:
    cudaStream_t stream;

    void SetUp() override { cudaStreamCreate(&stream); }

    void TearDown() override { cudaStreamDestroy(stream); }

    // CPU reference implementations
    std::vector<float> matmul_ABt_cpu(const std::vector<float>& A, const std::vector<float>& B,
                                      int M, int N, int K, float scale) {
        std::vector<float> C(M * N, 0.0f);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[j * K + k];  // B transposed
                }
                C[i * N + j] = sum * scale;
            }
        }
        return C;
    }

    std::vector<float> matmul_AB_cpu(const std::vector<float>& A, const std::vector<float>& B,
                                     int M, int N, int K, float scale) {
        std::vector<float> C(M * N, 0.0f);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum * scale;
            }
        }
        return C;
    }

    std::vector<float> matmul_AtB_cpu(const std::vector<float>& A, const std::vector<float>& B,
                                      int M, int N, int K, float scale) {
        std::vector<float> C(M * N, 0.0f);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[k * M + i] * B[k * N + j];  // A transposed
                }
                C[i * N + j] = sum * scale;
            }
        }
        return C;
    }
};

// =============================================================================
// matmul_ABt Tests (Attention Score Computation)
// =============================================================================

TEST_F(MatmulTest, ABt_Basic_64x64x32) {
    constexpr int M = 64, N = 64, K = 32;

    std::vector<float> h_A(M * K), h_B(N * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_A)
        v = dist(gen);
    for (auto& v : h_B)
        v = dist(gen);

    float scale = 0.17678f;  // 1/sqrt(32)

    // CPU reference
    auto h_C_expected = matmul_ABt_cpu(h_A, h_B, M, N, K, scale);

    // GPU computation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

    FlashAttentionError err = kernels::matmul_ABt<M, N, K>(d_A, d_B, d_C, scale, stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    std::vector<float> h_C(M * N);
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    for (size_t i = 0; i < h_C.size(); i++) {
        EXPECT_NEAR(h_C[i], h_C_expected[i], 1e-4f) << "Mismatch at index " << i;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(MatmulTest, ABt_HeadDim64) {
    constexpr int M = 64, N = 64, K = 64;

    std::vector<float> h_A(M * K, 1.0f), h_B(N * K, 1.0f);
    float scale = 0.125f;  // 1/sqrt(64)

    auto h_C_expected = matmul_ABt_cpu(h_A, h_B, M, N, K, scale);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

    FlashAttentionError err = kernels::matmul_ABt<M, N, K>(d_A, d_B, d_C, scale, stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    std::vector<float> h_C(M * N);
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // All elements should be K * scale = 64 * 0.125 = 8.0
    for (float v : h_C) {
        EXPECT_NEAR(v, 8.0f, 1e-4f);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(MatmulTest, ABt_HeadDim128_SmallTile) {
    constexpr int M = 32, N = 32, K = 128;

    std::vector<float> h_A(M * K), h_B(N * K);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : h_A)
        v = dist(gen);
    for (auto& v : h_B)
        v = dist(gen);

    float scale = 0.08839f;  // ~1/sqrt(128)

    auto h_C_expected = matmul_ABt_cpu(h_A, h_B, M, N, K, scale);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

    FlashAttentionError err = kernels::matmul_ABt<M, N, K>(d_A, d_B, d_C, scale, stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    std::vector<float> h_C(M * N);
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < h_C.size(); i++) {
        EXPECT_NEAR(h_C[i], h_C_expected[i], 1e-3f) << "Mismatch at index " << i;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// =============================================================================
// matmul_AB Tests (Attention Output Computation)
// =============================================================================

TEST_F(MatmulTest, AB_Basic_64x64x32) {
    constexpr int M = 64, N = 32, K = 64;

    std::vector<float> h_A(M * K), h_B(K * N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_A)
        v = dist(gen);
    for (auto& v : h_B)
        v = dist(gen);

    auto h_C_expected = matmul_AB_cpu(h_A, h_B, M, N, K, 1.0f);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    FlashAttentionError err = kernels::matmul_AB<M, N, K>(d_A, d_B, d_C, 1.0f, stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    std::vector<float> h_C(M * N);
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < h_C.size(); i++) {
        EXPECT_NEAR(h_C[i], h_C_expected[i], 1e-4f) << "Mismatch at index " << i;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// =============================================================================
// matmul_AB_acc Tests (Accumulation)
// =============================================================================

TEST_F(MatmulTest, AB_acc_Accumulates) {
    constexpr int M = 32, N = 64, K = 32;

    std::vector<float> h_A(M * K, 1.0f), h_B(K * N, 1.0f);
    std::vector<float> h_C_init(M * N, 10.0f);  // Initial values

    // Expected: C_new = 10.0 + K * 1.0 * 1.0 = 10.0 + 32 = 42.0

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C_init.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    FlashAttentionError err = kernels::matmul_AB_acc<M, N, K>(d_A, d_B, d_C, 1.0f, stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    std::vector<float> h_C(M * N);
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (float v : h_C) {
        EXPECT_NEAR(v, 42.0f, 1e-4f);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// =============================================================================
// matmul_AtB Tests (Gradient Computation)
// =============================================================================

TEST_F(MatmulTest, AtB_Basic) {
    constexpr int M = 32, N = 64, K = 64;  // A is KxM, B is KxN

    std::vector<float> h_A(K * M), h_B(K * N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_A)
        v = dist(gen);
    for (auto& v : h_B)
        v = dist(gen);

    auto h_C_expected = matmul_AtB_cpu(h_A, h_B, M, N, K, 1.0f);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, K * M * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), K * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    FlashAttentionError err = kernels::matmul_AtB<M, N, K>(d_A, d_B, d_C, 1.0f, stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    std::vector<float> h_C(M * N);
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < h_C.size(); i++) {
        EXPECT_NEAR(h_C[i], h_C_expected[i], 1e-4f) << "Mismatch at index " << i;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(MatmulTest, NullPointerReturnsError) {
    float* d_null = nullptr;
    float* d_valid;
    cudaMalloc(&d_valid, 64 * 64 * sizeof(float));

    FlashAttentionError err =
        kernels::matmul_ABt<64, 64, 32>(d_null, d_valid, d_valid, 1.0f, stream);
    EXPECT_EQ(err, FlashAttentionError::NULL_POINTER);

    err = kernels::matmul_ABt<64, 64, 32>(d_valid, d_null, d_valid, 1.0f, stream);
    EXPECT_EQ(err, FlashAttentionError::NULL_POINTER);

    err = kernels::matmul_ABt<64, 64, 32>(d_valid, d_valid, d_null, 1.0f, stream);
    EXPECT_EQ(err, FlashAttentionError::NULL_POINTER);

    cudaFree(d_valid);
}

// =============================================================================
// Scale Factor Tests
// =============================================================================

TEST_F(MatmulTest, ScaleFactorApplied) {
    constexpr int M = 32, N = 32, K = 32;

    std::vector<float> h_A(M * K, 1.0f), h_B(N * K, 1.0f);

    float *d_A, *d_B, *d_C1, *d_C2;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C1, M * N * sizeof(float));
    cudaMalloc(&d_C2, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

    // With scale = 1.0
    kernels::matmul_ABt<M, N, K>(d_A, d_B, d_C1, 1.0f, stream);

    // With scale = 0.5
    kernels::matmul_ABt<M, N, K>(d_A, d_B, d_C2, 0.5f, stream);

    std::vector<float> h_C1(M * N), h_C2(M * N);
    cudaMemcpy(h_C1.data(), d_C1, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2.data(), d_C2, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // C2 should be exactly half of C1
    for (size_t i = 0; i < h_C1.size(); i++) {
        EXPECT_NEAR(h_C2[i], h_C1[i] * 0.5f, 1e-5f);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C1);
    cudaFree(d_C2);
}

// =============================================================================
// Property Tests (if RapidCheck enabled)
// =============================================================================

#if CUFLASH_ENABLE_RAPIDCHECK

RC_GTEST_PROP(MatmulProperty, ABt_Linearity, ()) {
    constexpr int M = 32, N = 32, K = 32;

    // Generate random matrices
    std::vector<float> h_A(M * K), h_B(N * K);
    for (auto& v : h_A)
        v = *rc::gen::inRange(-100, 100) * 0.01f;
    for (auto& v : h_B)
        v = *rc::gen::inRange(-100, 100) * 0.01f;

    float scale1 = *rc::gen::inRange(1, 100) * 0.01f;
    float scale2 = *rc::gen::inRange(1, 100) * 0.01f;

    float *d_A, *d_B, *d_C1, *d_C2;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C1, M * N * sizeof(float));
    cudaMalloc(&d_C2, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    kernels::matmul_ABt<M, N, K>(d_A, d_B, d_C1, scale1, stream);
    kernels::matmul_ABt<M, N, K>(d_A, d_B, d_C2, scale2, stream);

    std::vector<float> h_C1(M * N), h_C2(M * N);
    cudaMemcpy(h_C1.data(), d_C1, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2.data(), d_C2, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Check linearity: C2 = C1 * (scale2 / scale1)
    float ratio = scale2 / scale1;
    for (size_t i = 0; i < h_C1.size(); i++) {
        RC_ASSERT(std::abs(h_C2[i] - h_C1[i] * ratio) < 1e-3f);
    }

    cudaStreamDestroy(stream);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C1);
    cudaFree(d_C2);
}

#endif  // CUFLASH_ENABLE_RAPIDCHECK

}  // namespace test
}  // namespace cuflash
