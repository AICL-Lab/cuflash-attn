// Online Softmax Unit Tests
// Tests for cuflash::kernels::online_softmax_* operations
// Uses the PUBLIC API (not internal implementation)

#include <gtest/gtest.h>
#if CUFLASH_ENABLE_RAPIDCHECK
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#endif
#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <vector>

#include "cuflash/kernels/online_softmax.cuh"

namespace cuflash {
namespace test {

// =============================================================================
// Test Utilities
// =============================================================================

class OnlineSoftmaxTest : public ::testing::Test {
   protected:
    cudaStream_t stream;

    void SetUp() override { cudaStreamCreate(&stream); }

    void TearDown() override { cudaStreamDestroy(stream); }

    // CPU reference: standard softmax
    std::vector<float> reference_softmax(const std::vector<float>& input, int rows, int cols) {
        std::vector<float> output(rows * cols);
        for (int r = 0; r < rows; r++) {
            float max_val = -INFINITY;
            for (int c = 0; c < cols; c++) {
                max_val = std::max(max_val, input[r * cols + c]);
            }
            float sum = 0.0f;
            for (int c = 0; c < cols; c++) {
                output[r * cols + c] = std::exp(input[r * cols + c] - max_val);
                sum += output[r * cols + c];
            }
            float inv_sum = 1.0f / sum;
            for (int c = 0; c < cols; c++) {
                output[r * cols + c] *= inv_sum;
            }
        }
        return output;
    }

    // CPU reference: logsumexp
    std::vector<float> reference_logsumexp(const std::vector<float>& input, int rows, int cols) {
        std::vector<float> output(rows);
        for (int r = 0; r < rows; r++) {
            float max_val = -INFINITY;
            for (int c = 0; c < cols; c++) {
                max_val = std::max(max_val, input[r * cols + c]);
            }
            float sum = 0.0f;
            for (int c = 0; c < cols; c++) {
                sum += std::exp(input[r * cols + c] - max_val);
            }
            output[r] = max_val + std::log(sum);
        }
        return output;
    }
};

// =============================================================================
// High-level API Tests: online_softmax_forward
// =============================================================================

TEST_F(OnlineSoftmaxTest, Forward_Basic) {
    constexpr int ROWS = 4;
    constexpr int COLS = 8;

    std::vector<float> h_input = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                                  -1.0f, 0.0f,  1.0f,  0.0f,  -1.0f, 0.0f,  1.0f,  0.0f,
                                  0.1f,  0.2f,  0.3f,  0.4f,  0.5f,  0.6f,  0.7f,  0.8f,
                                  10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f};

    auto h_output_expected = reference_softmax(h_input, ROWS, COLS);
    auto h_logsumexp_expected = reference_logsumexp(h_input, ROWS, COLS);

    float *d_input, *d_output, *d_logsumexp;
    cudaMalloc(&d_input, ROWS * COLS * sizeof(float));
    cudaMalloc(&d_output, ROWS * COLS * sizeof(float));
    cudaMalloc(&d_logsumexp, ROWS * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);

    FlashAttentionError err =
        kernels::online_softmax_forward(d_input, d_output, d_logsumexp, ROWS, COLS, 4, stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    std::vector<float> h_output(ROWS * COLS), h_logsumexp(ROWS);
    cudaMemcpy(h_output.data(), d_output, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_logsumexp.data(), d_logsumexp, ROWS * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare softmax output
    for (size_t i = 0; i < h_output.size(); i++) {
        EXPECT_NEAR(h_output[i], h_output_expected[i], 1e-5f) << "Softmax mismatch at index " << i;
    }

    // Compare logsumexp
    for (int r = 0; r < ROWS; r++) {
        EXPECT_NEAR(h_logsumexp[r], h_logsumexp_expected[r], 1e-4f)
            << "Logsumexp mismatch at row " << r;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_logsumexp);
}

TEST_F(OnlineSoftmaxTest, Forward_LargeInput) {
    constexpr int ROWS = 64;
    constexpr int COLS = 128;

    std::vector<float> h_input(ROWS * COLS);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (auto& v : h_input)
        v = dist(gen);

    auto h_output_expected = reference_softmax(h_input, ROWS, COLS);

    float *d_input, *d_output, *d_logsumexp;
    cudaMalloc(&d_input, ROWS * COLS * sizeof(float));
    cudaMalloc(&d_output, ROWS * COLS * sizeof(float));
    cudaMalloc(&d_logsumexp, ROWS * sizeof(float));
    cudaMemcpy(d_input, h_input.data(), ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);

    FlashAttentionError err =
        kernels::online_softmax_forward(d_input, d_output, d_logsumexp, ROWS, COLS, 32, stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    std::vector<float> h_output(ROWS * COLS);
    cudaMemcpy(h_output.data(), d_output, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < h_output.size(); i++) {
        EXPECT_NEAR(h_output[i], h_output_expected[i], 1e-4f) << "Mismatch at index " << i;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_logsumexp);
}

TEST_F(OnlineSoftmaxTest, Forward_NumericalStability) {
    // Test with large values that would cause overflow in naive implementation
    constexpr int ROWS = 2;
    constexpr int COLS = 4;

    std::vector<float> h_input = {1000.0f,  1001.0f, 1002.0f,  999.0f,
                                  -1000.0f, -999.0f, -1001.0f, -998.0f};

    auto h_output_expected = reference_softmax(h_input, ROWS, COLS);

    float *d_input, *d_output, *d_logsumexp;
    cudaMalloc(&d_input, ROWS * COLS * sizeof(float));
    cudaMalloc(&d_output, ROWS * COLS * sizeof(float));
    cudaMalloc(&d_logsumexp, ROWS * sizeof(float));
    cudaMemcpy(d_input, h_input.data(), ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);

    FlashAttentionError err =
        kernels::online_softmax_forward(d_input, d_output, d_logsumexp, ROWS, COLS, 2, stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    std::vector<float> h_output(ROWS * COLS);
    cudaMemcpy(h_output.data(), d_output, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost);

    // Check no NaN or Inf
    for (size_t i = 0; i < h_output.size(); i++) {
        EXPECT_FALSE(std::isnan(h_output[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(h_output[i])) << "Inf at index " << i;
        EXPECT_NEAR(h_output[i], h_output_expected[i], 1e-4f);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_logsumexp);
}

// =============================================================================
// Low-level API Tests: Primitives
// =============================================================================

TEST_F(OnlineSoftmaxTest, Primitives_InitUpdateFinalize) {
    constexpr int ROWS = 4;

    // Simulate processing two blocks
    // Block 1: values [1, 2, 3, 4] for each row
    // Block 2: values [5, 6, 7, 8] for each row

    float *d_state_m, *d_state_l;
    float *d_block_max1, *d_block_sum1;
    float *d_block_max2, *d_block_sum2;
    float *d_logsumexp, *d_normalizer;

    cudaMalloc(&d_state_m, ROWS * sizeof(float));
    cudaMalloc(&d_state_l, ROWS * sizeof(float));
    cudaMalloc(&d_block_max1, ROWS * sizeof(float));
    cudaMalloc(&d_block_sum1, ROWS * sizeof(float));
    cudaMalloc(&d_block_max2, ROWS * sizeof(float));
    cudaMalloc(&d_block_sum2, ROWS * sizeof(float));
    cudaMalloc(&d_logsumexp, ROWS * sizeof(float));
    cudaMalloc(&d_normalizer, ROWS * sizeof(float));

    // Block 1 statistics: max=4, sum=exp(1-4)+exp(2-4)+exp(3-4)+exp(4-4)
    std::vector<float> h_max1(ROWS, 4.0f);
    float sum1 = std::exp(1 - 4) + std::exp(2 - 4) + std::exp(3 - 4) + std::exp(4 - 4);
    std::vector<float> h_sum1(ROWS, sum1);
    cudaMemcpy(d_block_max1, h_max1.data(), ROWS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_sum1, h_sum1.data(), ROWS * sizeof(float), cudaMemcpyHostToDevice);

    // Block 2 statistics: max=8, sum=exp(5-8)+exp(6-8)+exp(7-8)+exp(8-8)
    std::vector<float> h_max2(ROWS, 8.0f);
    float sum2 = std::exp(5 - 8) + std::exp(6 - 8) + std::exp(7 - 8) + std::exp(8 - 8);
    std::vector<float> h_sum2(ROWS, sum2);
    cudaMemcpy(d_block_max2, h_max2.data(), ROWS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_sum2, h_sum2.data(), ROWS * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize
    FlashAttentionError err = kernels::online_softmax_init(d_state_m, d_state_l, ROWS, stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    // Update with block 1
    err = kernels::online_softmax_update(d_block_max1, d_block_sum1, d_state_m, d_state_l, ROWS,
                                         stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    // Update with block 2
    err = kernels::online_softmax_update(d_block_max2, d_block_sum2, d_state_m, d_state_l, ROWS,
                                         stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    // Finalize
    err = kernels::online_softmax_finalize(d_state_m, d_state_l, d_logsumexp, d_normalizer, ROWS,
                                           stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    // Check results
    std::vector<float> h_logsumexp(ROWS);
    cudaMemcpy(h_logsumexp.data(), d_logsumexp, ROWS * sizeof(float), cudaMemcpyDeviceToHost);

    // Expected: full data [1,2,3,4,5,6,7,8], max=8, sum=exp(1-8)+...+exp(8-8)
    float expected_max = 8.0f;
    float expected_sum = std::exp(1 - 8) + std::exp(2 - 8) + std::exp(3 - 8) + std::exp(4 - 8) +
                         std::exp(5 - 8) + std::exp(6 - 8) + std::exp(7 - 8) + std::exp(8 - 8);
    float expected_logsumexp = expected_max + std::log(expected_sum);

    for (int r = 0; r < ROWS; r++) {
        EXPECT_NEAR(h_logsumexp[r], expected_logsumexp, 1e-4f) << "Row " << r;
    }

    cudaFree(d_state_m);
    cudaFree(d_state_l);
    cudaFree(d_block_max1);
    cudaFree(d_block_sum1);
    cudaFree(d_block_max2);
    cudaFree(d_block_sum2);
    cudaFree(d_logsumexp);
    cudaFree(d_normalizer);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(OnlineSoftmaxTest, NullPointerReturnsError) {
    float* d_null = nullptr;
    float* d_valid;
    cudaMalloc(&d_valid, 4 * sizeof(float));

    FlashAttentionError err = kernels::online_softmax_init(d_null, d_valid, 4, stream);
    EXPECT_EQ(err, FlashAttentionError::NULL_POINTER);

    err = kernels::online_softmax_forward(d_null, d_valid, d_valid, 4, 4, 2, stream);
    EXPECT_EQ(err, FlashAttentionError::NULL_POINTER);

    cudaFree(d_valid);
}

TEST_F(OnlineSoftmaxTest, InvalidDimensionReturnsError) {
    float* d_valid;
    cudaMalloc(&d_valid, 4 * sizeof(float));

    FlashAttentionError err = kernels::online_softmax_init(d_valid, d_valid, 0, stream);
    EXPECT_EQ(err, FlashAttentionError::INVALID_DIMENSION);

    err = kernels::online_softmax_forward(d_valid, d_valid, d_valid, 0, 4, 2, stream);
    EXPECT_EQ(err, FlashAttentionError::INVALID_DIMENSION);

    err = kernels::online_softmax_forward(d_valid, d_valid, d_valid, 4, 0, 2, stream);
    EXPECT_EQ(err, FlashAttentionError::INVALID_DIMENSION);

    cudaFree(d_valid);
}

// =============================================================================
// Sum-to-One Property Test
// =============================================================================

TEST_F(OnlineSoftmaxTest, OutputSumsToOne) {
    constexpr int ROWS = 16;
    constexpr int COLS = 64;

    std::vector<float> h_input(ROWS * COLS);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& v : h_input)
        v = dist(gen);

    float *d_input, *d_output, *d_logsumexp;
    cudaMalloc(&d_input, ROWS * COLS * sizeof(float));
    cudaMalloc(&d_output, ROWS * COLS * sizeof(float));
    cudaMalloc(&d_logsumexp, ROWS * sizeof(float));
    cudaMemcpy(d_input, h_input.data(), ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);

    FlashAttentionError err =
        kernels::online_softmax_forward(d_input, d_output, d_logsumexp, ROWS, COLS, 16, stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    std::vector<float> h_output(ROWS * COLS);
    cudaMemcpy(h_output.data(), d_output, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost);

    // Each row should sum to 1
    for (int r = 0; r < ROWS; r++) {
        float sum = 0.0f;
        for (int c = 0; c < COLS; c++) {
            sum += h_output[r * COLS + c];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Row " << r << " does not sum to 1";
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_logsumexp);
}

// =============================================================================
// Property Tests (if RapidCheck enabled)
// =============================================================================

#if CUFLASH_ENABLE_RAPIDCHECK

RC_GTEST_PROP(OnlineSoftmaxProperty, Forward_Equivalence, ()) {
    constexpr int ROWS = 8;
    constexpr int COLS = 32;

    std::vector<float> h_input(ROWS * COLS);
    for (auto& v : h_input) {
        v = *rc::gen::inRange(-100, 100) * 0.1f;
    }

    float *d_input, *d_output, *d_logsumexp;
    cudaMalloc(&d_input, ROWS * COLS * sizeof(float));
    cudaMalloc(&d_output, ROWS * COLS * sizeof(float));
    cudaMalloc(&d_logsumexp, ROWS * sizeof(float));
    cudaMemcpy(d_input, h_input.data(), ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int block_size = *rc::gen::inRange(1, 33);

    FlashAttentionError err = kernels::online_softmax_forward(d_input, d_output, d_logsumexp, ROWS,
                                                              COLS, block_size, stream);
    RC_ASSERT(err == FlashAttentionError::SUCCESS);

    std::vector<float> h_output(ROWS * COLS);
    cudaMemcpy(h_output.data(), d_output, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost);

    // Check all values in [0, 1]
    for (float v : h_output) {
        RC_ASSERT(v >= 0.0f);
        RC_ASSERT(v <= 1.0f);
        RC_ASSERT(!std::isnan(v));
        RC_ASSERT(!std::isinf(v));
    }

    // Check rows sum to 1
    for (int r = 0; r < ROWS; r++) {
        float sum = 0.0f;
        for (int c = 0; c < COLS; c++) {
            sum += h_output[r * COLS + c];
        }
        RC_ASSERT(std::abs(sum - 1.0f) < 1e-4f);
    }

    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_logsumexp);
}

#endif  // CUFLASH_ENABLE_RAPIDCHECK

}  // namespace test
}  // namespace cuflash
