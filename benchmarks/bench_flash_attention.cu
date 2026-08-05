#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
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

// Report achieved compute and memory throughput so results can be read against
// the hardware roofline, not just wall-clock.
//   FLOP model: forward = 4 * B*H*N^2*D (the two N x N matmuls QK^T and PV);
//               backward ~= 2.5x forward (dQ, dK, dV, dS, dP).
//   Byte model: FlashAttention streams Q, K, V, O once each = 4 * B*H*N*D elems.
static void report_metrics(benchmark::State& state, int batch_size, int num_heads, int seq_len,
                           int head_dim, size_t elem_size, bool backward) {
    const double n = static_cast<double>(seq_len);
    const double elems = static_cast<double>(batch_size) * num_heads * n * head_dim;
    double flops = 4.0 * elems * n;
    if (backward) {
        flops *= 2.5;
    }
    const double bytes = 4.0 * elems * static_cast<double>(elem_size);
    state.counters["TFLOP/s"] =
        benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::OneK::kIs1000) /
        1e12;
    state.counters["HBM GB/s"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::OneK::kIs1000) /
        1e9;
}

// =============================================================================
// Naive (materialized) attention baseline
// =============================================================================
// Forms the full N x N score matrix. This is the "standard" attention that
// FlashAttention exists to avoid; it is included ONLY as a comparison point
// (and intentionally runs out of memory at large N), not as a recommended
// implementation.
__global__ void naive_attention_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                       const float* __restrict__ V, float* __restrict__ O,
                                       int seq_len, int head_dim, float scale, bool causal) {
    const int bh = blockIdx.y;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    if (row >= seq_len)
        return;

    const float* Q_row =
        Q + static_cast<size_t>(bh) * seq_len * head_dim + static_cast<size_t>(row) * head_dim;
    const float* K_base = K + static_cast<size_t>(bh) * seq_len * head_dim;
    const float* V_base = V + static_cast<size_t>(bh) * seq_len * head_dim;
    float* O_row =
        O + static_cast<size_t>(bh) * seq_len * head_dim + static_cast<size_t>(row) * head_dim;

    extern __shared__ float scores[];  // seq_len floats
    __shared__ float red[128];

    // scores[j] = scale * <Q_row, K_j>, with the causal mask applied.
    float local_max = -INFINITY;
    for (int j = tid; j < seq_len; j += nthreads) {
        float s = -INFINITY;
        if (!(causal && j > row)) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += Q_row[d] * K_base[static_cast<size_t>(j) * head_dim + d];
            }
            s = dot * scale;
        }
        scores[j] = s;
        local_max = fmaxf(local_max, s);
    }

    red[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s)
            red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    float row_max = red[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < seq_len; j += nthreads) {
        float e = expf(scores[j] - row_max);
        scores[j] = e;
        local_sum += e;
    }
    red[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s)
            red[tid] += red[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / red[0];
    __syncthreads();

    for (int d = tid; d < head_dim; d += nthreads) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            acc += scores[j] * V_base[static_cast<size_t>(j) * head_dim + d];
        }
        O_row[d] = acc * inv_sum;
    }
}

static void BM_NaiveForward_FP32(benchmark::State& state) {
    int seq_len = state.range(0);
    int head_dim = state.range(1);
    int batch_size = 1;
    int num_heads = 8;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = static_cast<size_t>(batch_size) * num_heads * seq_len * head_dim;

    auto devs = allocate_and_init<float>({qkv_size, qkv_size, qkv_size, qkv_size});  // Q, K, V, O
    float *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2], *d_O = devs[3];

    dim3 grid(seq_len, batch_size * num_heads);
    size_t smem = static_cast<size_t>(seq_len) * sizeof(float);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    for (auto _ : state) {
        naive_attention_kernel<<<grid, 128, smem, stream>>>(d_Q, d_K, d_V, d_O, seq_len, head_dim,
                                                            scale, false);
        cudaStreamSynchronize(stream);
    }

    // The naive path also materializes the N x N score matrix (written + read).
    const double n = static_cast<double>(seq_len);
    const double elems = static_cast<double>(batch_size) * num_heads * n * head_dim;
    const double flops = 4.0 * elems * n;
    const double bytes = 4.0 * elems * sizeof(float) +
                         2.0 * static_cast<double>(batch_size) * num_heads * n * n * sizeof(float);
    state.counters["TFLOP/s"] =
        benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::OneK::kIs1000) /
        1e12;
    state.counters["HBM GB/s"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::OneK::kIs1000) /
        1e9;

    cudaStreamDestroy(stream);
    for (auto* ptr : devs) {
        cudaFree(ptr);
    }
}
// Keep N modest: the baseline allocates an O(N^2) score matrix.
BENCHMARK(BM_NaiveForward_FP32)
    ->Args({256, 64})
    ->Args({512, 64})
    ->Args({1024, 64})
    ->Args({2048, 64})
    ->Args({4096, 64})
    ->Unit(benchmark::kMillisecond);

// =============================================================================
// FlashAttention forward / backward
// =============================================================================

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

    report_metrics(state, batch_size, num_heads, seq_len, head_dim, sizeof(float), false);

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

    report_metrics(state, batch_size, num_heads, seq_len, head_dim, sizeof(float), true);

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

// Reduced-precision (FP16/BF16) forward. L (logsumexp) is always FP32 even for
// reduced-precision inputs, so it is allocated separately; see
// include/cuflash/flash_attention.h for the rationale.
template<typename InputT>
static void BM_Forward_ReducedPrec(benchmark::State& state) {
    int seq_len = state.range(0);
    int head_dim = state.range(1);
    int batch_size = 1;
    int num_heads = 8;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = static_cast<size_t>(batch_size) * num_heads * seq_len * head_dim;
    size_t l_size = static_cast<size_t>(batch_size) * num_heads * seq_len;

    auto devs = allocate_and_init<InputT>({qkv_size, qkv_size, qkv_size,  // Q, K, V
                                           qkv_size});                    // O
    auto l_bufs = allocate_and_init<float>({l_size});                     // L (FP32)

    InputT *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2];
    InputT* d_O = devs[3];
    float* d_L = l_bufs[0];

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    for (auto _ : state) {
        auto err = cuflash::flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads,
                                                    seq_len, head_dim, scale, false, stream);
        if (err != cuflash::FlashAttentionError::SUCCESS) {
            state.SkipWithError("flash_attention_forward (reduced precision) failed");
            break;
        }
        cudaStreamSynchronize(stream);
    }

    report_metrics(state, batch_size, num_heads, seq_len, head_dim, sizeof(InputT), false);

    cudaStreamDestroy(stream);
    for (auto* ptr : devs) {
        cudaFree(ptr);
    }
    for (auto* ptr : l_bufs) {
        cudaFree(ptr);
    }
}

static void BM_Forward_FP16(benchmark::State& state) {
    BM_Forward_ReducedPrec<half>(state);
}
BENCHMARK(BM_Forward_FP16)
    ->Args({256, 64})
    ->Args({512, 64})
    ->Args({1024, 64})
    ->Args({2048, 64})
    ->Args({4096, 64})
    ->Args({4096, 128})
    ->Unit(benchmark::kMillisecond);

static void BM_Forward_BF16(benchmark::State& state) {
    BM_Forward_ReducedPrec<__nv_bfloat16>(state);
}
BENCHMARK(BM_Forward_BF16)
    ->Args({256, 64})
    ->Args({512, 64})
    ->Args({1024, 64})
    ->Args({2048, 64})
    ->Args({4096, 64})
    ->Args({4096, 128})
    ->Unit(benchmark::kMillisecond);

// Reduced-precision (FP16/BF16) backward with FP32 L, same rationale as above.
template<typename InputT>
static void BM_Backward_ReducedPrec(benchmark::State& state) {
    int seq_len = state.range(0);
    int head_dim = state.range(1);
    int batch_size = 1;
    int num_heads = 8;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = static_cast<size_t>(batch_size) * num_heads * seq_len * head_dim;
    size_t l_size = static_cast<size_t>(batch_size) * num_heads * seq_len;

    // Q, K, V, O, dO, dQ, dK, dV
    auto devs = allocate_and_init<InputT>(
        {qkv_size, qkv_size, qkv_size, qkv_size, qkv_size, qkv_size, qkv_size, qkv_size});
    auto l_bufs = allocate_and_init<float>({l_size});  // L (FP32)

    InputT *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2], *d_O = devs[3];
    InputT *d_dO = devs[4], *d_dQ = devs[5], *d_dK = devs[6];
    InputT* d_dV = devs[7];
    float* d_L = l_bufs[0];

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    for (auto _ : state) {
        auto err = cuflash::flash_attention_backward(d_Q, d_K, d_V, d_O, d_L, d_dO, d_dQ, d_dK,
                                                     d_dV, batch_size, num_heads, seq_len, head_dim,
                                                     scale, false, stream);
        if (err != cuflash::FlashAttentionError::SUCCESS) {
            state.SkipWithError("flash_attention_backward (reduced precision) failed");
            break;
        }
        cudaStreamSynchronize(stream);
    }

    report_metrics(state, batch_size, num_heads, seq_len, head_dim, sizeof(InputT), true);

    cudaStreamDestroy(stream);
    for (auto* ptr : devs) {
        cudaFree(ptr);
    }
    for (auto* ptr : l_bufs) {
        cudaFree(ptr);
    }
}

static void BM_Backward_FP16(benchmark::State& state) {
    BM_Backward_ReducedPrec<half>(state);
}
BENCHMARK(BM_Backward_FP16)
    ->Args({256, 64})
    ->Args({512, 64})
    ->Args({1024, 64})
    ->Args({2048, 64})
    ->Args({4096, 64})
    ->Args({4096, 128})
    ->Unit(benchmark::kMillisecond);

static void BM_Backward_BF16(benchmark::State& state) {
    BM_Backward_ReducedPrec<__nv_bfloat16>(state);
}
BENCHMARK(BM_Backward_BF16)
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

    // Causal masking does roughly half the work on average.
    const double n = static_cast<double>(seq_len);
    const double elems = static_cast<double>(batch_size) * num_heads * n * head_dim;
    const double flops = 0.5 * 4.0 * elems * n;
    const double bytes = 4.0 * elems * sizeof(float);
    state.counters["TFLOP/s"] =
        benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::OneK::kIs1000) /
        1e12;
    state.counters["HBM GB/s"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::OneK::kIs1000) /
        1e9;

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
