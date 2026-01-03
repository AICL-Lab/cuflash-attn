#pragma once

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace cuflash {

// Online Softmax State
// Maintains running max (m) and sum of exponentials (l) for numerically stable softmax
struct OnlineSoftmaxState {
    float m;  // Current maximum value
    float l;  // Sum of exp(x - m)
    
    __device__ __forceinline__ void init() {
        m = -INFINITY;
        l = 0.0f;
    }
    
    // Update state with a new block's statistics
    // new_m: max value in the new block
    // new_l: sum of exp(x - new_m) in the new block
    __device__ __forceinline__ void update(float new_m, float new_l) {
        float m_max = fmaxf(m, new_m);
        // Rescale both sums to the new maximum
        l = l * expf(m - m_max) + new_l * expf(new_m - m_max);
        m = m_max;
    }
    
    // Get the logsumexp value: m + log(l)
    __device__ __forceinline__ float logsumexp() const {
        return m + logf(l);
    }
    
    // Get the normalization factor for final output
    __device__ __forceinline__ float get_normalizer() const {
        return 1.0f / l;
    }
};

// Compute row-wise max for a tile stored in registers
// tile: array of values for one row
// n: number of elements
template<int N>
__device__ __forceinline__ float row_max(const float* tile) {
    float max_val = -INFINITY;
    #pragma unroll
    for (int i = 0; i < N; i++) {
        max_val = fmaxf(max_val, tile[i]);
    }
    return max_val;
}

// Compute row-wise sum of exp(x - max) for a tile
// tile: array of values for one row
// max_val: the maximum value (for numerical stability)
// n: number of elements
template<int N>
__device__ __forceinline__ float row_sum_exp(const float* tile, float max_val) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < N; i++) {
        sum += expf(tile[i] - max_val);
    }
    return sum;
}

// Apply softmax in-place to a tile (after computing max and sum)
// tile: array of values for one row
// max_val: the maximum value
// sum_exp: sum of exp(x - max)
template<int N>
__device__ __forceinline__ void apply_softmax(float* tile, float max_val, float sum_exp) {
    float inv_sum = 1.0f / sum_exp;
    #pragma unroll
    for (int i = 0; i < N; i++) {
        tile[i] = expf(tile[i] - max_val) * inv_sum;
    }
}

// Warp-level reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for max using shared memory
template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warp_reduce_max(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < BLOCK_SIZE / 32) ? shared[lane] : -INFINITY;
    if (wid == 0) val = warp_reduce_max(val);
    
    return val;
}

// Block-level reduction for sum using shared memory
template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < BLOCK_SIZE / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

} // namespace cuflash
