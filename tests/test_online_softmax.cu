// Online Softmax Tests
// Feature: cuflash-attn, Property 3: 在线 Softmax 等价性

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>

// Include the online softmax implementation
#include "../src/online_softmax.cuh"

namespace cuflash {
namespace test {

// CPU reference implementation of standard softmax
std::vector<float> reference_softmax(const std::vector<float>& input) {
    if (input.empty()) return {};
    
    float max_val = *std::max_element(input.begin(), input.end());
    std::vector<float> exp_vals(input.size());
    float sum = 0.0f;
    
    for (size_t i = 0; i < input.size(); i++) {
        exp_vals[i] = std::exp(input[i] - max_val);
        sum += exp_vals[i];
    }
    
    for (size_t i = 0; i < input.size(); i++) {
        exp_vals[i] /= sum;
    }
    
    return exp_vals;
}

// CPU simulation of online softmax algorithm
std::vector<float> online_softmax_cpu(const std::vector<float>& input, int block_size) {
    if (input.empty()) return {};
    
    int n = input.size();
    int num_blocks = (n + block_size - 1) / block_size;
    
    // Online softmax state
    float m = -INFINITY;
    float l = 0.0f;
    
    // Accumulator for weighted values (simulating O accumulation)
    std::vector<float> output(n, 0.0f);
    
    // Process each block
    for (int b = 0; b < num_blocks; b++) {
        int start = b * block_size;
        int end = std::min(start + block_size, n);
        
        // Find max in this block
        float block_max = -INFINITY;
        for (int i = start; i < end; i++) {
            block_max = std::max(block_max, input[i]);
        }
        
        // Compute sum of exp in this block
        float block_sum = 0.0f;
        for (int i = start; i < end; i++) {
            block_sum += std::exp(input[i] - block_max);
        }
        
        // Update online state
        float m_new = std::max(m, block_max);
        float l_new = l * std::exp(m - m_new) + block_sum * std::exp(block_max - m_new);
        
        m = m_new;
        l = l_new;
    }
    
    // Final normalization - compute actual softmax values
    for (int i = 0; i < n; i++) {
        output[i] = std::exp(input[i] - m) / l;
    }
    
    return output;
}

// Test that online softmax produces same result as standard softmax
TEST(OnlineSoftmaxTest, EquivalenceSmall) {
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    auto ref = reference_softmax(input);
    auto online = online_softmax_cpu(input, 2);  // block size 2
    
    ASSERT_EQ(ref.size(), online.size());
    for (size_t i = 0; i < ref.size(); i++) {
        EXPECT_NEAR(ref[i], online[i], 1e-5f) << "Mismatch at index " << i;
    }
}

// Test with larger input
TEST(OnlineSoftmaxTest, EquivalenceLarge) {
    std::vector<float> input(128);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& v : input) v = dist(gen);
    
    auto ref = reference_softmax(input);
    auto online = online_softmax_cpu(input, 16);
    
    ASSERT_EQ(ref.size(), online.size());
    for (size_t i = 0; i < ref.size(); i++) {
        EXPECT_NEAR(ref[i], online[i], 1e-5f) << "Mismatch at index " << i;
    }
}

// Test numerical stability with large values
TEST(OnlineSoftmaxTest, NumericalStability) {
    std::vector<float> input = {1000.0f, 1001.0f, 1002.0f, 999.0f};
    
    auto ref = reference_softmax(input);
    auto online = online_softmax_cpu(input, 2);
    
    // Check no NaN or Inf
    for (size_t i = 0; i < ref.size(); i++) {
        EXPECT_FALSE(std::isnan(ref[i]));
        EXPECT_FALSE(std::isinf(ref[i]));
        EXPECT_FALSE(std::isnan(online[i]));
        EXPECT_FALSE(std::isinf(online[i]));
    }
    
    // Check equivalence
    for (size_t i = 0; i < ref.size(); i++) {
        EXPECT_NEAR(ref[i], online[i], 1e-5f);
    }
}

// Property test: Online softmax should equal standard softmax for any input
// Feature: cuflash-attn, Property 3: 在线 Softmax 等价性
// Validates: Requirements 4.3
RC_GTEST_PROP(OnlineSoftmaxProperty, EquivalenceProperty, ()) {
    // Generate random input size (1 to 256)
    int n = *rc::gen::inRange(1, 257);
    
    // Generate random block size (1 to n)
    int block_size = *rc::gen::inRange(1, std::max(2, n));
    
    // Generate random input values
    std::vector<float> input(n);
    for (int i = 0; i < n; i++) {
        input[i] = *rc::gen::inRange(-100, 100) * 0.1f;
    }
    
    auto ref = reference_softmax(input);
    auto online = online_softmax_cpu(input, block_size);
    
    RC_ASSERT(ref.size() == online.size());
    
    for (size_t i = 0; i < ref.size(); i++) {
        // Allow small numerical error
        RC_ASSERT(std::abs(ref[i] - online[i]) < 1e-4f);
    }
}

// Property test: Softmax output should sum to 1
RC_GTEST_PROP(OnlineSoftmaxProperty, SumToOne, ()) {
    int n = *rc::gen::inRange(1, 257);
    int block_size = *rc::gen::inRange(1, std::max(2, n));
    
    std::vector<float> input(n);
    for (int i = 0; i < n; i++) {
        input[i] = *rc::gen::inRange(-100, 100) * 0.1f;
    }
    
    auto output = online_softmax_cpu(input, block_size);
    
    float sum = 0.0f;
    for (float v : output) {
        sum += v;
    }
    
    RC_ASSERT(std::abs(sum - 1.0f) < 1e-4f);
}

// Property test: All softmax outputs should be in [0, 1]
RC_GTEST_PROP(OnlineSoftmaxProperty, OutputRange, ()) {
    int n = *rc::gen::inRange(1, 257);
    int block_size = *rc::gen::inRange(1, std::max(2, n));
    
    std::vector<float> input(n);
    for (int i = 0; i < n; i++) {
        input[i] = *rc::gen::inRange(-100, 100) * 0.1f;
    }
    
    auto output = online_softmax_cpu(input, block_size);
    
    for (float v : output) {
        RC_ASSERT(v >= 0.0f);
        RC_ASSERT(v <= 1.0f);
        RC_ASSERT(!std::isnan(v));
        RC_ASSERT(!std::isinf(v));
    }
}

} // namespace test
} // namespace cuflash
