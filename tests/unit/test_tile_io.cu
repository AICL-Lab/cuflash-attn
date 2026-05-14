// Tile I/O Unit Tests
// Tests for cuflash::kernels::load_tile and store_tile operations
//
// NOTE: These tests validate the standalone kernel API for tile I/O operations.
// The production forward/backward kernels use impl::load_tile_to_shared and
// impl::store_tile_from_shared directly for better performance (avoiding
// kernel launch overhead for each tile). The kernels::* API provides a
// testable interface for the same underlying algorithms.
//
// Test coverage: The correctness of impl::* tile functions is validated through
// end-to-end tests in tests/integration/.

#include <gtest/gtest.h>
#if CUFLASH_ENABLE_RAPIDCHECK
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <vector>

#include "cuflash/kernels/tile_io.cuh"

namespace cuflash {
namespace test {

// =============================================================================
// Test Utilities
// =============================================================================

class TileIOTest : public ::testing::Test {
   protected:
    cudaStream_t stream;

    void SetUp() override { cudaStreamCreate(&stream); }

    void TearDown() override { cudaStreamDestroy(stream); }

    // CPU reference: extract a tile from a matrix
    std::vector<float> extract_tile_cpu(const std::vector<float>& matrix, int row_start,
                                        int col_start, int tile_rows, int tile_cols, int max_rows,
                                        int max_cols, int stride) {
        std::vector<float> tile(tile_rows * tile_cols, 0.0f);
        for (int r = 0; r < tile_rows; r++) {
            for (int c = 0; c < tile_cols; c++) {
                int global_row = row_start + r;
                int global_col = col_start + c;
                if (global_row < max_rows && global_col < max_cols) {
                    tile[r * tile_cols + c] = matrix[global_row * stride + global_col];
                }
            }
        }
        return tile;
    }
};

// =============================================================================
// Load Tile Tests
// =============================================================================

// Test FP32 load with 64x64 tile
TEST_F(TileIOTest, LoadTileFP32_64x64) {
    constexpr int BLOCK_ROWS = 64;
    constexpr int BLOCK_COLS = 64;
    constexpr int MAX_ROWS = 128;
    constexpr int MAX_COLS = 128;

    // Create source matrix
    std::vector<float> h_src(MAX_ROWS * MAX_COLS);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& v : h_src)
        v = dist(gen);

    // Allocate device memory
    float *d_src, *d_dst;
    cudaMalloc(&d_src, MAX_ROWS * MAX_COLS * sizeof(float));
    cudaMalloc(&d_dst, BLOCK_ROWS * BLOCK_COLS * sizeof(float));

    cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Load tile from position (32, 32)
    int row_start = 32, col_start = 32;
    FlashAttentionError err = kernels::load_tile<BLOCK_ROWS, BLOCK_COLS>(
        d_src, d_dst, row_start, col_start, MAX_ROWS, MAX_COLS, MAX_COLS, stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    // Copy result back
    std::vector<float> h_dst(BLOCK_ROWS * BLOCK_COLS);
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare with CPU reference
    auto expected = extract_tile_cpu(h_src, row_start, col_start, BLOCK_ROWS, BLOCK_COLS, MAX_ROWS,
                                     MAX_COLS, MAX_COLS);
    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_FLOAT_EQ(h_dst[i], expected[i]) << "Mismatch at index " << i;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

// Test FP16 load with conversion
TEST_F(TileIOTest, LoadTileFP16_64x64) {
    constexpr int BLOCK_ROWS = 64;
    constexpr int BLOCK_COLS = 64;
    constexpr int MAX_ROWS = 128;
    constexpr int MAX_COLS = 64;  // head_dim size

    // Create source matrix in FP16
    std::vector<half> h_src(MAX_ROWS * MAX_COLS);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_src)
        v = __float2half(dist(gen));

    // Allocate device memory
    half* d_src;
    float* d_dst;
    cudaMalloc(&d_src, MAX_ROWS * MAX_COLS * sizeof(half));
    cudaMalloc(&d_dst, BLOCK_ROWS * BLOCK_COLS * sizeof(float));

    cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(half), cudaMemcpyHostToDevice);

    // Load tile
    int row_start = 0, col_start = 0;
    FlashAttentionError err = kernels::load_tile<BLOCK_ROWS, BLOCK_COLS>(
        d_src, d_dst, row_start, col_start, MAX_ROWS, MAX_COLS, MAX_COLS, stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    // Copy result back
    std::vector<float> h_dst(BLOCK_ROWS * BLOCK_COLS);
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify conversion
    for (int i = 0; i < BLOCK_ROWS * BLOCK_COLS; i++) {
        float expected = __half2float(h_src[i]);
        EXPECT_NEAR(h_dst[i], expected, 1e-3f) << "Mismatch at index " << i;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

// Test boundary handling (tile extends past matrix)
TEST_F(TileIOTest, LoadTileBoundaryHandling) {
    constexpr int BLOCK_ROWS = 64;
    constexpr int BLOCK_COLS = 64;
    constexpr int MAX_ROWS = 100;  // Not divisible by 64
    constexpr int MAX_COLS = 100;

    std::vector<float> h_src(MAX_ROWS * MAX_COLS, 1.0f);
    float *d_src, *d_dst;
    cudaMalloc(&d_src, MAX_ROWS * MAX_COLS * sizeof(float));
    cudaMalloc(&d_dst, BLOCK_ROWS * BLOCK_COLS * sizeof(float));
    cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Load tile that extends past the boundary
    int row_start = 64;  // Only 36 valid rows
    int col_start = 64;  // Only 36 valid cols
    FlashAttentionError err = kernels::load_tile<BLOCK_ROWS, BLOCK_COLS>(
        d_src, d_dst, row_start, col_start, MAX_ROWS, MAX_COLS, MAX_COLS, stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    std::vector<float> h_dst(BLOCK_ROWS * BLOCK_COLS);
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Valid region should be 1.0, invalid should be 0.0
    for (int r = 0; r < BLOCK_ROWS; r++) {
        for (int c = 0; c < BLOCK_COLS; c++) {
            int idx = r * BLOCK_COLS + c;
            bool valid = (r < 36 && c < 36);  // MAX_ROWS - 64 = 36
            if (valid) {
                EXPECT_FLOAT_EQ(h_dst[idx], 1.0f) << "Valid region at " << idx;
            } else {
                EXPECT_FLOAT_EQ(h_dst[idx], 0.0f) << "Invalid region at " << idx;
            }
        }
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

// =============================================================================
// Store Tile Tests
// =============================================================================

TEST_F(TileIOTest, StoreTileFP32_64x64) {
    constexpr int BLOCK_ROWS = 64;
    constexpr int BLOCK_COLS = 64;
    constexpr int MAX_ROWS = 128;
    constexpr int MAX_COLS = 128;

    // Create tile data
    std::vector<float> h_tile(BLOCK_ROWS * BLOCK_COLS);
    for (int i = 0; i < BLOCK_ROWS * BLOCK_COLS; i++) {
        h_tile[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_tile, *d_dst;
    cudaMalloc(&d_tile, BLOCK_ROWS * BLOCK_COLS * sizeof(float));
    cudaMalloc(&d_dst, MAX_ROWS * MAX_COLS * sizeof(float));
    cudaMemset(d_dst, 0, MAX_ROWS * MAX_COLS * sizeof(float));

    cudaMemcpy(d_tile, h_tile.data(), h_tile.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Store tile at position (64, 64)
    int row_start = 64, col_start = 64;
    FlashAttentionError err = kernels::store_tile<BLOCK_ROWS, BLOCK_COLS>(
        d_tile, d_dst, row_start, col_start, MAX_ROWS, MAX_COLS, MAX_COLS, stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    // Copy result back
    std::vector<float> h_dst(MAX_ROWS * MAX_COLS);
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify stored region
    for (int r = 0; r < BLOCK_ROWS; r++) {
        for (int c = 0; c < BLOCK_COLS; c++) {
            int tile_idx = r * BLOCK_COLS + c;
            int dst_idx = (row_start + r) * MAX_COLS + (col_start + c);
            EXPECT_FLOAT_EQ(h_dst[dst_idx], h_tile[tile_idx])
                << "Mismatch at (" << r << ", " << c << ")";
        }
    }

    // Verify other regions are zero
    for (int r = 0; r < MAX_ROWS; r++) {
        for (int c = 0; c < MAX_COLS; c++) {
            if (r < row_start || r >= row_start + BLOCK_ROWS || c < col_start ||
                c >= col_start + BLOCK_COLS) {
                int idx = r * MAX_COLS + c;
                EXPECT_FLOAT_EQ(h_dst[idx], 0.0f)
                    << "Non-tile region at (" << r << ", " << c << ")";
            }
        }
    }

    cudaFree(d_tile);
    cudaFree(d_dst);
}

TEST_F(TileIOTest, StoreTileFP16_32x128) {
    constexpr int BLOCK_ROWS = 32;
    constexpr int BLOCK_COLS = 128;  // HEAD_DIM = 128
    constexpr int MAX_ROWS = 64;
    constexpr int MAX_COLS = 128;

    // Create tile data in float
    std::vector<float> h_tile(BLOCK_ROWS * BLOCK_COLS);
    for (int i = 0; i < BLOCK_ROWS * BLOCK_COLS; i++) {
        h_tile[i] = static_cast<float>(i) / 100.0f;
    }

    // Allocate device memory
    float* d_tile;
    half* d_dst;
    cudaMalloc(&d_tile, BLOCK_ROWS * BLOCK_COLS * sizeof(float));
    cudaMalloc(&d_dst, MAX_ROWS * MAX_COLS * sizeof(half));
    cudaMemset(d_dst, 0, MAX_ROWS * MAX_COLS * sizeof(half));

    cudaMemcpy(d_tile, h_tile.data(), h_tile.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Store tile
    FlashAttentionError err = kernels::store_tile<BLOCK_ROWS, BLOCK_COLS>(
        d_tile, d_dst, 0, 0, MAX_ROWS, MAX_COLS, MAX_COLS, stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    // Copy result back
    std::vector<half> h_dst(MAX_ROWS * MAX_COLS);
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // Verify conversion and storage
    for (int i = 0; i < BLOCK_ROWS * BLOCK_COLS; i++) {
        float expected = __half2float(h_dst[i]);
        EXPECT_NEAR(expected, h_tile[i], 1e-3f) << "Mismatch at index " << i;
    }

    cudaFree(d_tile);
    cudaFree(d_dst);
}

// =============================================================================
// Round-trip Tests
// =============================================================================

TEST_F(TileIOTest, RoundTripLoadStore) {
    constexpr int BLOCK_ROWS = 64;
    constexpr int BLOCK_COLS = 64;
    constexpr int MAX_ROWS = 128;
    constexpr int MAX_COLS = 128;

    // Create source matrix
    std::vector<float> h_src(MAX_ROWS * MAX_COLS);
    std::vector<float> h_dst(MAX_ROWS * MAX_COLS, 0.0f);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (auto& v : h_src)
        v = dist(gen);

    float *d_src, *d_dst;
    cudaMalloc(&d_src, MAX_ROWS * MAX_COLS * sizeof(float));
    cudaMalloc(&d_dst, MAX_ROWS * MAX_COLS * sizeof(float));
    cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_dst, 0, MAX_ROWS * MAX_COLS * sizeof(float));

    // Round-trip at position (16, 16)
    int row_start = 16, col_start = 16;
    FlashAttentionError err = kernels::load_store_tile_roundtrip<BLOCK_ROWS, BLOCK_COLS>(
        d_src, d_dst, row_start, col_start, MAX_ROWS, MAX_COLS, MAX_COLS, stream);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the tile region matches source
    for (int r = 0; r < BLOCK_ROWS; r++) {
        for (int c = 0; c < BLOCK_COLS; c++) {
            int global_row = row_start + r;
            int global_col = col_start + c;
            int src_idx = global_row * MAX_COLS + global_col;
            int dst_idx = global_row * MAX_COLS + global_col;
            EXPECT_FLOAT_EQ(h_dst[dst_idx], h_src[src_idx])
                << "Round-trip mismatch at (" << global_row << ", " << global_col << ")";
        }
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(TileIOTest, NullPointerReturnsError) {
    float* d_null = nullptr;
    float* d_valid;
    cudaMalloc(&d_valid, 64 * 64 * sizeof(float));

    FlashAttentionError err =
        kernels::load_tile<64, 64>(d_null, d_valid, 0, 0, 128, 128, 128, stream);
    EXPECT_EQ(err, FlashAttentionError::NULL_POINTER);

    err = kernels::load_tile<64, 64>(d_valid, d_null, 0, 0, 128, 128, 128, stream);
    EXPECT_EQ(err, FlashAttentionError::NULL_POINTER);

    cudaFree(d_valid);
}

TEST_F(TileIOTest, InvalidDimensionReturnsError) {
    float *d_src, *d_dst;
    cudaMalloc(&d_src, 128 * 128 * sizeof(float));
    cudaMalloc(&d_dst, 64 * 64 * sizeof(float));

    // Negative start
    FlashAttentionError err =
        kernels::load_tile<64, 64>(d_src, d_dst, -1, 0, 128, 128, 128, stream);
    EXPECT_EQ(err, FlashAttentionError::INVALID_DIMENSION);

    // Start past bounds
    err = kernels::load_tile<64, 64>(d_src, d_dst, 128, 0, 128, 128, 128, stream);
    EXPECT_EQ(err, FlashAttentionError::INVALID_DIMENSION);

    // Zero dimensions
    err = kernels::load_tile<64, 64>(d_src, d_dst, 0, 0, 0, 128, 128, stream);
    EXPECT_EQ(err, FlashAttentionError::INVALID_DIMENSION);

    cudaFree(d_src);
    cudaFree(d_dst);
}

// =============================================================================
// Property Tests (if RapidCheck enabled)
// =============================================================================

#if CUFLASH_ENABLE_RAPIDCHECK

RC_GTEST_PROP(TileIOProperty, LoadStorePreservesData, ()) {
    constexpr int BLOCK_ROWS = 64;
    constexpr int BLOCK_COLS = 64;
    constexpr int MAX_ROWS = 128;
    constexpr int MAX_COLS = 128;

    // Generate random tile position
    int row_start = *rc::gen::inRange(0, MAX_ROWS);
    int col_start = *rc::gen::inRange(0, MAX_COLS);

    // Generate random data
    std::vector<float> h_src(MAX_ROWS * MAX_COLS);
    for (auto& v : h_src) {
        v = *rc::gen::inRange(-1000, 1000) * 0.001f;
    }

    float *d_src, *d_dst;
    cudaMalloc(&d_src, MAX_ROWS * MAX_COLS * sizeof(float));
    cudaMalloc(&d_dst, MAX_ROWS * MAX_COLS * sizeof(float));
    cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_dst, 0, MAX_ROWS * MAX_COLS * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    FlashAttentionError err = kernels::load_store_tile_roundtrip<BLOCK_ROWS, BLOCK_COLS>(
        d_src, d_dst, row_start, col_start, MAX_ROWS, MAX_COLS, MAX_COLS, stream);

    RC_ASSERT(err == FlashAttentionError::SUCCESS);

    std::vector<float> h_dst(MAX_ROWS * MAX_COLS);
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify tile region
    int valid_rows = std::min(BLOCK_ROWS, MAX_ROWS - row_start);
    int valid_cols = std::min(BLOCK_COLS, MAX_COLS - col_start);

    for (int r = 0; r < valid_rows; r++) {
        for (int c = 0; c < valid_cols; c++) {
            int src_idx = (row_start + r) * MAX_COLS + (col_start + c);
            int dst_idx = (row_start + r) * MAX_COLS + (col_start + c);
            RC_ASSERT(std::abs(h_dst[dst_idx] - h_src[src_idx]) < 1e-5f);
        }
    }

    cudaStreamDestroy(stream);
    cudaFree(d_src);
    cudaFree(d_dst);
}

#endif  // CUFLASH_ENABLE_RAPIDCHECK

}  // namespace test
}  // namespace cuflash
