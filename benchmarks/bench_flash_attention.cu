#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "cuflash/flash_attention.h"

// Helper: allocate device memory and fill with random data
template<typename T>
static std::vector<T*> allocate_and_init(const std::vector<size_t>& sizes) {
    std::vector<T*> ptrs;
    ptrs.reserve(sizes.size());
    for (size_t size : sizes) {
        T* d_ptr = nullptr;
        size_t bytes = size * sizeof(T);
        cudaMalloc(&d_ptr, bytes);
        // Fill with random values in [-1, 1]
        std::vector<T> h_data(size);
        for (size_t i = 0; i < size; ++i) {
            h_data[i] = static_cast<T>(
                2.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 1.0f);
        }
        cudaMemcpy(d_ptr, h_data.data(), bytes, cudaMemcpyHostToDevice);
        ptrs.push_back(d_ptr);
    }
    return ptrs;
}

static void FreeDeviceMemory(void* ptr) {
    cudaFree(ptr);
}

// FP32 Forward Benchmark
static void BM_Forward_FP32(benchmark::State& state) {
    int seq_len = state.range(0);
    int head_dim = state.range(1);
    int batch_size = 1;
    int num_heads = 8;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = static_cast<size_t>(batch_size) * num_heads * seq_len * head_dim;
    size_t l_size = static_cast<size_t>(batch_size) * num_heads * seq_len;

    auto devs = allocate_and_init<float>({qkv_size, qkv_size, qkv_size,  // Q, K, V
                                          qkv_size,                      // O
                                          l_size});                      // L

    float *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2];
    float *d_O = devs[3], *d_L = devs[4];

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    for (auto _ : state) {
        auto err = cuflash::flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads,
                                                    seq_len, head_dim, scale, false, stream);
        if (err != cuflash::FlashAttentionError::SUCCESS) {
            state.SkipWithError("flash_attention_forward failed");
            break;
        }
        cudaStreamSynchronize(stream);
    }

    cudaStreamDestroy(stream);
    for (auto* ptr : devs) {
        cudaFree(ptr);
    }
}
BENCHMARK(BM_Forward_FP32)
    ->Args({256, 64})
    ->Args({512, 64})
    ->Args({1024, 64})
    ->Args({2048, 64})
    ->Args({4096, 64})
    ->Args({4096, 128})
    ->Unit(benchmark::kMillisecond);

// FP32 Backward Benchmark
static void BM_Backward_FP32(benchmark::State& state) {
    int seq_len = state.range(0);
    int head_dim = state.range(1);
    int batch_size = 1;
    int num_heads = 8;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = static_cast<size_t>(batch_size) * num_heads * seq_len * head_dim;
    size_t l_size = static_cast<size_t>(batch_size) * num_heads * seq_len;

    // Q, K, V, O, L, dO, dQ, dK, dV
    auto devs = allocate_and_init<float>(
        {qkv_size, qkv_size, qkv_size, qkv_size, l_size, qkv_size, qkv_size, qkv_size, qkv_size});

    float *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2], *d_O = devs[3];
    float *d_L = devs[4], *d_dO = devs[5], *d_dQ = devs[6], *d_dK = devs[7];
    float* d_dV = devs[8];

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    for (auto _ : state) {
        auto err = cuflash::flash_attention_backward(d_Q, d_K, d_V, d_O, d_L, d_dO, d_dQ, d_dK,
                                                     d_dV, batch_size, num_heads, seq_len, head_dim,
                                                     scale, false, stream);
        if (err != cuflash::FlashAttentionError::SUCCESS) {
            state.SkipWithError("flash_attention_backward failed");
            break;
        }
        cudaStreamSynchronize(stream);
    }

    cudaStreamDestroy(stream);
    for (auto* ptr : devs) {
        cudaFree(ptr);
    }
}
BENCHMARK(BM_Backward_FP32)
    ->Args({256, 64})
    ->Args({512, 64})
    ->Args({1024, 64})
    ->Args({2048, 64})
    ->Args({4096, 64})
    ->Args({4096, 128})
    ->Unit(benchmark::kMillisecond);

// FP16 Forward Benchmark
static void BM_Forward_FP16(benchmark::State& state) {
    int seq_len = state.range(0);
    int head_dim = state.range(1);
    int batch_size = 1;
    int num_heads = 8;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = static_cast<size_t>(batch_size) * num_heads * seq_len * head_dim;
    size_t l_size = static_cast<size_t>(batch_size) * num_heads * seq_len;

    auto devs = allocate_and_init<half>({qkv_size, qkv_size, qkv_size,  // Q, K, V
                                         qkv_size,                      // O
                                         l_size});                      // L

    half *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2];
    half *d_O = devs[3], *d_L = devs[4];

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    for (auto _ : state) {
        auto err = cuflash::flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads,
                                                    seq_len, head_dim, scale, false, stream);
        if (err != cuflash::FlashAttentionError::SUCCESS) {
            state.SkipWithError("flash_attention_forward (FP16) failed");
            break;
        }
        cudaStreamSynchronize(stream);
    }

    cudaStreamDestroy(stream);
    for (auto* ptr : devs) {
        cudaFree(ptr);
    }
}
BENCHMARK(BM_Forward_FP16)
    ->Args({256, 64})
    ->Args({512, 64})
    ->Args({1024, 64})
    ->Args({2048, 64})
    ->Args({4096, 64})
    ->Args({4096, 128})
    ->Unit(benchmark::kMillisecond);

// Causal Mask Forward Benchmark
static void BM_Forward_Causal(benchmark::State& state) {
    int seq_len = state.range(0);
    int head_dim = state.range(1);
    int batch_size = 1;
    int num_heads = 8;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = static_cast<size_t>(batch_size) * num_heads * seq_len * head_dim;
    size_t l_size = static_cast<size_t>(batch_size) * num_heads * seq_len;

    auto devs = allocate_and_init<float>({qkv_size, qkv_size, qkv_size,  // Q, K, V
                                          qkv_size,                      // O
                                          l_size});                      // L

    float *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2];
    float *d_O = devs[3], *d_L = devs[4];

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    for (auto _ : state) {
        auto err = cuflash::flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads,
                                                    seq_len, head_dim, scale, true, stream);
        if (err != cuflash::FlashAttentionError::SUCCESS) {
            state.SkipWithError("flash_attention_forward (causal) failed");
            break;
        }
        cudaStreamSynchronize(stream);
    }

    cudaStreamDestroy(stream);
    for (auto* ptr : devs) {
        cudaFree(ptr);
    }
}
BENCHMARK(BM_Forward_Causal)
    ->Args({256, 64})
    ->Args({512, 64})
    ->Args({1024, 64})
    ->Args({2048, 64})
    ->Args({4096, 64})
    ->Args({4096, 128})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
