#pragma once

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace cuflash {
namespace impl {

// =============================================================================
// Online Softmax State
// =============================================================================

/// State for numerically stable streaming softmax.
/// Maintains running max (m) and sum of exponentials (l).
struct OnlineSoftmaxState {
    float m;  // Current maximum value
    float l;  // Sum of exp(x - m)

    __device__ __forceinline__ void init() {
        m = -INFINITY;
        l = 0.0f;
    }

    /// Update state with a new block's statistics.
    /// new_m: max value in the new block
    /// new_l: sum of exp(x - new_m) in the new block
    __device__ __forceinline__ void update(float new_m, float new_l) {
        float m_max = fmaxf(m, new_m);
        // Rescale both sums to the new maximum
        l = l * expf(m - m_max) + new_l * expf(new_m - m_max);
        m = m_max;
    }

    /// Get the logsumexp value: m + log(l)
    __device__ __forceinline__ float logsumexp() const { return m + logf(l); }

    /// Get the normalization factor for final output
    __device__ __forceinline__ float get_normalizer() const { return 1.0f / l; }

    /// Update from local block statistics and return the rescale factor for existing O.
    /// This combines update() with computing the necessary scaling for output accumulation.
    /// @param new_m max value in the new block
    /// @param new_l sum of exp(x - new_m) in the new block
    /// @param rescale_existing output: factor to multiply existing O by
    /// @param scale_new output: factor to multiply new P@V by
    __device__ __forceinline__ void update_with_rescale(float new_m, float new_l,
                                                        float& rescale_existing, float& scale_new) {
        float m_old = m;
        float l_old = l;
        float m_new = fmaxf(m_old, new_m);
        l = l_old * expf(m_old - m_new) + new_l * expf(new_m - m_new);
        m = m_new;

        // Compute rescaling factors
        rescale_existing = expf(m_old - m_new);  // For existing O
        scale_new = expf(new_m - m_new);         // For new P @ V contribution
    }
};

// =============================================================================
// Row-wise Statistics (for device code)
// =============================================================================

/// Compute row-wise max for a tile stored in registers.
template<int N>
__device__ __forceinline__ float row_max(const float* tile) {
    float max_val = -INFINITY;
#pragma unroll
    for (int i = 0; i < N; i++) {
        max_val = fmaxf(max_val, tile[i]);
    }
    return max_val;
}

/// Compute row-wise sum of exp(x - max) for a tile.
template<int N>
__device__ __forceinline__ float row_sum_exp(const float* tile, float max_val) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < N; i++) {
        sum += expf(tile[i] - max_val);
    }
    return sum;
}

/// Apply softmax in-place to a tile (after computing max and sum).
template<int N>
__device__ __forceinline__ void apply_softmax(float* tile, float max_val, float sum_exp) {
    float inv_sum = 1.0f / sum_exp;
#pragma unroll
    for (int i = 0; i < N; i++) {
        tile[i] = expf(tile[i] - max_val) * inv_sum;
    }
}

// =============================================================================
// Block-level Reductions
// =============================================================================

/// Block-level max reduction using shared memory.
template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp-level reduction
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / 32) ? shared[lane] : -INFINITY;
    if (wid == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
        }
    }

    return val;
}

/// Block-level sum reduction using shared memory.
template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp-level reduction
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / 32) ? shared[lane] : 0.0f;
    if (wid == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
    }

    return val;
}

}  // namespace impl
}  // namespace cuflash
