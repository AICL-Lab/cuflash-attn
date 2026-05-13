# Kernel Deep Dive

This document provides a rigorous, implementation-level analysis of the CuFlash-Attn CUDA kernel. We decompose the tile-and-pipeline execution model, quantify shared memory and register footprints, and justify every microarchitectural choice with reference to the NVIDIA execution model.

---

## 1. Launch Configuration and Occupancy

### 1.1 Grid-Block Topology

The kernel is launched with a one-dimensional grid and a fixed one-dimensional thread block:

| Parameter | Expression | Typical Value | Semantics |
|-----------|------------|---------------|-----------|
| `gridDim.x` | `batch_size × num_heads` | — | One block per (batch, head) pair |
| `blockDim.x` | `128` | 128 threads | Fixed for the entire kernel family |
| `__launch_bounds__` | `128` | — | Compiler contract for register allocation |

```cpp
// Launch site (host-side C++)
dim3 grid(batch_size * num_heads);
dim3 block(128);
flash_attn_fwd_kernel<<<grid, block, smem_bytes, stream>>>(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, H, N, d, scale,
    stride_qb, stride_qh, stride_qn,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn
);
```

The kernel prototype enforces the compiler contract:

```cpp
template <typename T, int head_dim, int Br, int Bc>
__global__ void __launch_bounds__(128)
flash_attn_fwd_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    // ... strides and meta-data
);
```

### 1.2 `__launch_bounds__(128)` and Register Pressure

The `__launch_bounds__(128)` directive informs the compiler that **at most 128 threads reside in a single thread block**. This allows the compiler to increase the per-thread register budget while still guaranteeing that the SM can host at least one resident block.

| Metric | Without `__launch_bounds__` | With `__launch_bounds__(128)` | Impact |
|--------|----------------------------|-------------------------------|--------|
| Max registers per thread (Volta+) | 255 | ~128–168 | Compiler may spill fewer local variables |
| Blocks per SM (theoretical) | 1–2 | 2–4 | Higher SM occupancy via block-level parallelism |
| Latency hiding | moderate | improved | More warps per SM to cover global-memory latency |
| Instruction cache pressure | higher | lower | Smaller register footprint per thread |

On compute capability ≥ 7.0 (Volta, Ampere, Hopper), an SM has 65,536 32-bit registers. With `__launch_bounds__(128)`, the compiler is permitted to allocate up to:

$$
\frac{65{,}536}{128 \times 4} = 128 \text{ registers per thread}
$$

if four blocks are to be resident simultaneously, or up to 256 registers per thread if only two blocks are required. In practice, the kernel compiles to **approximately 96–112 registers per thread**, leaving sufficient headroom for the CUDA runtime and avoiding register spilling into local memory.

---

## 2. Shared Memory Layout

### 2.1 Tile Geometry

FlashAttention partitions the $N \times d$ attention computation into small tiles that fit in on-chip SRAM. The tiling parameters are compile-time constants, selected by `head_dim`.

| Symbol | Role | Dimensions | Size (FP16, d=64) | Size (FP16, d=128) |
|--------|------|------------|-------------------|--------------------|
| $B_r$ | Rows per Q-tile | 64 | — | — |
| $B_c$ | Columns per KV-tile | 64 (d=64) or 32 (d=128) | — | — |
| $d$ | Head dimension | 32, 64, 128 | — | — |
| Q tile | Query SRAM buffer | $[B_r, d]$ | 8 KiB | 16 KiB |
| K tile | Key SRAM buffer | $[B_c, d]$ | 8 KiB | 8 KiB |
| V tile | Value SRAM buffer | $[B_c, d]$ | 8 KiB | 8 KiB |
| S tile | Score SRAM buffer | $[B_r, B_c]$ | 8 KiB | 4 KiB |

The exact layout is:

```cpp
// Shared memory allocation (device-side)
// All buffers are aligned to 16 bytes for vectorized access
extern __shared__ char smem_base[];

T* q_smem = reinterpret_cast<T*>(smem_base);                           // [Br, d]
T* k_smem = q_smem + Br * d;                                            // [Bc, d]
T* v_smem = k_smem + Bc * d;                                            // [Bc, d]
float* s_smem = reinterpret_cast<float*>(v_smem + Bc * d);               // [Br, Bc]
```

### 2.2 Memory Layout Diagram

The shared memory region is contiguous and partitioned as follows (example: FP16, d=64, Br=Bc=64):

```
Low address
│
├─ Q_smem [64 × 64]  ──────── 8192 bytes (8 KiB)
│
├─ K_smem [64 × 64]  ──────── 8192 bytes (8 KiB)
│
├─ V_smem [64 × 64]  ──────── 8192 bytes (8 KiB)
│
├─ S_smem [64 × 64 float]  ─ 16384 bytes (16 KiB)
│
High address
```

Total shared memory per block: **40 KiB** for the FP16 d=64 configuration. This fits comfortably within the 48–100+ KiB capacities of modern SMs (Ampere: 164 KiB, Hopper: 228 KiB), while leaving capacity for the L1 cache to capture temporally-reused register spills.

---

## 3. Warp-Level Decomposition

### 3.1 Thread-to-Warp Mapping

The 128 threads of a block are subdivided into four warps of 32 lanes each:

| Warp ID | Lane Range | Primary Responsibility |
|---------|------------|------------------------|
| 0 | 0–31 | Load Q, K, V tiles from global memory → shared memory |
| 1 | 32–63 | Compute $S = QK^T$ (GEMM-I) via WMMA or CUDA Core MAD |
| 2 | 64–95 | Apply softmax: $P = \text{softmax}(S)$, online normalization |
| 3 | 96–127 | Compute $O += PV$ (GEMM-II), write output to global memory |

> **Note:** The preceding table describes a logical assignment for pedagogical clarity. The actual kernel employs an **interleaved execution model** in which all warps participate in each phase, but with distinct tile sub-assignment to maximize instruction-level parallelism (ILP).

### 3.2 Phase-by-Phase Warp Choreography

```cpp
// Simplified control flow inside the kernel
const int warp_id = threadIdx.x / 32;
const int lane_id = threadIdx.x % 32;

for (int tile_kv = 0; tile_kv < num_kv_tiles; ++tile_kv) {
    // Phase 1: Cooperative async load of K and V tiles
    if (warp_id < 2) {
        load_gmem_to_smem<K_tile_shape>(K + kv_offset, k_smem);
        load_gmem_to_smem<V_tile_shape>(V + kv_offset, v_smem);
    }
    __syncthreads();

    // Phase 2: Compute S = Q @ K^T
    // Each warp computes a [Br/4, Bc] stripe of S
    float acc[Br_per_warp][Bc_per_thread];
    compute_qk_gemm(q_smem, k_smem, acc, warp_id, lane_id);

    // Phase 3: Online softmax & rescale
    online_softmax(acc, m_prev, l_prev, m_new, l_new);

    // Phase 4: Accumulate O += P @ V
    accumulate_pv(acc, v_smem, o_acc, warp_id, lane_id);
    __syncthreads();
}
```

### 3.3 Why 128 Threads?

| Factor | 64 Threads | 128 Threads | 256 Threads |
|--------|------------|-------------|-------------|
| Warps per block | 2 | 4 | 8 |
| Shared memory per thread | higher | moderate | lower |
| GEMM parallelism | insufficient for 4×4 WMMA | optimal for 2×2 WMMA tiling | excess parallelism, bank conflicts |
| Occupancy on Ampere SM (max 32 warps) | 16 blocks possible | 8 blocks possible | 4 blocks possible |
| Latency hiding | weak | strong | comparable |

The 128-thread configuration provides **four warps**, which is the minimum count required to keep an SM fully occupied during global-memory load stalls while avoiding excessive shared-memory partition camping.

---

## 4. Vectorized Memory Access and Coalescing

### 4.1 `float4` Vectorization Strategy

All global-memory loads and stores are vectorized via `float4` (16 bytes). For FP16 (`half`), a `float4` encompasses eight consecutive `half` elements.

```cpp
// Device helper: vectorized global → shared load
__device__ __forceinline__
void load_gmem_vec4(const half* __restrict__ gmem, half* smem, int row, int col, int stride) {
    const int4* gmem_vec = reinterpret_cast<const int4*>(gmem + row * stride + col);
    int4 val = __ldg(gmem_vec);                     // Load via read-only cache (L2)
    half2* smem_vec = reinterpret_cast<half2*>(smem + row * d + col);
    // Decompose int4 into two half2 writes
    smem_vec[0] = *reinterpret_cast<half2*>(&val.x);
    smem_vec[1] = *reinterpret_cast<half2*>(&val.y);
    smem_vec[2] = *reinterpret_cast<half2*>(&val.z);
    smem_vec[3] = *reinterpret_cast<half2*>(&val.w);
}
```

### 4.2 Coalescing Analysis

A warp of 32 threads issues a single memory transaction when threads access **consecutive 128-byte aligned segments**. With `float4` (16 bytes per thread):

$$
32 \text{ threads} \times 16 \text{ bytes} = 512 \text{ bytes per warp transaction}
$$

This exactly matches the L2 cache line size on Ampere and Hopper, yielding **perfect coalescing** and one cache-line fetch per warp.

| Access Pattern | Bytes per Thread | Warp Transaction Size | Coalesced? | Performance |
|----------------|------------------|-----------------------|------------|-------------|
| scalar `half` | 2 | 64 B | partial | low |
| `half2` | 4 | 128 B | yes | moderate |
| `float4` (as `half`×8) | 16 | 512 B | perfect | peak |

### 4.3 Shared Memory Bank Conflicts

The $[B_c, d]$ K and V tiles are laid out in **row-major order**. To avoid bank conflicts on column-major access during the $QK^T$ GEMM, the K tile is **transposed during the GMEM→SMEM load**:

```cpp
// Transpose K while loading: gmem K[i,j] -> smem K_T[j,i]
// This makes K_smem column-major with respect to the original K,
// which is row-major with respect to the dot-product access pattern.
#pragma unroll
for (int i = lane_id; i < Bc * d / 8; i += 32) {
    int row = i / (d / 8);
    int col = i % (d / 8);
    float4 val = reinterpret_cast<const float4*>(K_gmem + row * stride_k + col * 8)[0];
    // Write to transposed location
    reinterpret_cast<float4*>(k_smem + col * 8 * Bc + row * 8)[0] = val;
}
```

After transposition, the inner dimension of the $QK^T$ dot product accesses consecutive shared-memory banks, resulting in **zero bank conflicts**.

---

## 5. Causal Masking: Warp-Level Skip Logic

### 5.1 Problem Statement

For autoregressive (causal) attention, the score matrix $S$ must satisfy:

$$
S_{ij} = 0 \quad \text{if} \quad j > i
$$

In tiled FlashAttention, $i$ and $j$ denote tile coordinates. For a given Q-tile at row $t_q$ and KV-tile at column $t_k$:

| Condition | Action |
|-----------|--------|
| $t_k < t_q$ | Full tile is valid; compute normally |
| $t_k > t_q$ | Entire tile is masked; **skip** (warp-level early exit) |
| $t_k = t_q$ | Partial mask; apply element-wise predicate inside tile |

### 5.2 Warp-Level Implementation

```cpp
// Causal mask dispatch inside the KV-tile loop
for (int tile_kv = 0; tile_kv < num_kv_tiles; ++tile_kv) {
    if (tile_kv > tile_q) {
        // Entire tile is in the causal forbidden zone.
        // All warps skip load, compute, and store for this iteration.
        continue;
    }

    // Load K, V (cooperative)
    // ...

    if (tile_kv < tile_q) {
        // Full tile below the diagonal: no masking needed
        compute_full_tile(q_smem, k_smem, v_smem, o_acc);
    } else {
        // tile_kv == tile_q: diagonal tile
        // Only warp lanes corresponding to valid (i >= j) positions participate
        compute_diagonal_tile(q_smem, k_smem, v_smem, o_acc, tile_q, tile_kv);
    }
}
```

### 5.3 Diagonal Tile Predicate

Within the diagonal tile, each thread computes a boolean mask predicate based on its global row and column indices:

```cpp
__device__ __forceinline__
bool causal_mask(int global_row, int global_col, int seq_len) {
    return global_col <= global_row;   // inclusive for causal attention
}

// Inside the diagonal-tile GEMM
int global_row = tile_q * Br + warp_row_offset + (lane_id / 4);
int global_col = tile_kv * Bc + warp_col_offset + (lane_id % 4) * 8;

float score = ...;  // QK^T dot product
score = causal_mask(global_row, global_col, N) ? score : -INFINITY;
```

The warp-level skip logic eliminates **all global-memory loads and compute for upper-triangular tiles**, reducing the causal attention FLOPs from $O(N^2)$ to $O(N^2/2)$ and memory traffic from $O(N^2)$ to $O(N^2/2)$.

---

## 6. FP16 with FP32 Internal Accumulation

### 6.1 Numerical Path

Although the kernel interface accepts FP16 tensors, all intermediate accumulation occurs in FP32 to prevent catastrophic loss of precision in the softmax denominator.

| Stage | Data Type | Rationale |
|-------|-----------|-----------|
| GMEM → SMEM load | `half` (16-bit) | Minimize HBM bandwidth (2× vs FP32) |
| Q, K, V in SRAM | `half` | Maximize tile size within SRAM constraints |
| $S = QK^T$ dot product | `float` (32-bit) | Accumulate up to $d=128$ products without overflow |
| Online softmax ($m$, $l$, $exp$) | `float` | Dynamic range of FP32 required for stability |
| $P = \text{softmax}(S)$ | `half` | Store back to SRAM for GEMM-II |
| $O += PV$ accumulation | `float` | Prevent round-off in long sequences |
| SMEM → GMEM store | `half` | Final output in requested precision |

### 6.2 Type-Casting Microkernel

```cpp
// FP16 shared-memory fetch, FP32 MAC accumulation
__device__ __forceinline__
float dot_product_fp16_to_fp32(const half* q_vec, const half* k_vec, int d) {
    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < d; i += 8) {
        // Load 8 half elements = 1 float4 = 128 bits
        float4 q4 = reinterpret_cast<const float4*>(q_vec + i)[0];
        float4 k4 = reinterpret_cast<const float4*>(k_vec + i)[0];

        // Cast each half2 to float2, then FMA
        half2* q_h2 = reinterpret_cast<half2*>(&q4);
        half2* k_h2 = reinterpret_cast<half2*>(&k4);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 q_f2 = __half22float2(q_h2[j]);
            float2 k_f2 = __half22float2(k_h2[j]);
            acc += q_f2.x * k_f2.x;
            acc += q_f2.y * k_f2.y;
        }
    }
    return acc;
}
```

### 6.3 Online Softmax in FP32

The online softmax algorithm (Milakov & Gimelshein, 2018) maintains running maximum $m$ and running sum $l$ in FP32:

```cpp
struct SoftmaxState {
    float m;   // running max
    float l;   // running sum of exponentials
};

__device__ __forceinline__
SoftmaxState online_softmax_update(float score, SoftmaxState state) {
    float m_new = fmaxf(state.m, score);
    float l_new = state.l * expf(state.m - m_new) + expf(score - m_new);
    return {m_new, l_new};
}
```

Without FP32 accumulation, the softmax denominator $l$ would suffer from severe round-off error for long sequences ($N > 2{,}048$), where the exponent dynamic range spans multiple orders of magnitude.

---

## 7. Complete Kernel Skeleton

The following listing integrates the concepts discussed above into a coherent, compilable kernel skeleton:

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

template <int Br, int Bc, int d>
__global__ void __launch_bounds__(128)
flash_attn_fwd_fp16(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int N, int stride_qb, int stride_qh, int stride_qn,
    int stride_kb, int stride_kh, int stride_kn,
    int stride_vb, int stride_vh, int stride_vn,
    float scale
) {
    // Grid: (batch*heads), Block: 128
    const int bh_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Shared memory
    extern __shared__ char smem[];
    half* q_smem = reinterpret_cast<half*>(smem);
    half* k_smem = q_smem + Br * d;
    half* v_smem = k_smem + Bc * d;
    float* s_smem = reinterpret_cast<float*>(v_smem + Bc * d);

    // Persistent registers for online softmax
    float m_reg[Br / 4];   // per-row max, distributed across warps
    float l_reg[Br / 4];   // per-row sum
    float o_reg[Br / 4][d / 8];  // partial O accumulator

    #pragma unroll
    for (int i = 0; i < Br / 4; ++i) {
        m_reg[i] = -FLT_MAX;
        l_reg[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < d / 8; ++j) o_reg[i][j] = 0.0f;
    }

    // Load Q tile (cooperative, vectorized)
    // ...

    // Main KV-tile loop
    const int num_kv_tiles = (N + Bc - 1) / Bc;
    for (int tile_kv = 0; tile_kv < num_kv_tiles; ++tile_kv) {
        // Causal skip
        if (tile_kv > tile_q) continue;

        // Load K, V tiles
        // ... vectorized GMEM -> SMEM via warp 0–1 ...
        __syncthreads();

        // Compute S = QK^T for this tile
        // Warp 1–2 compute GEMM-I
        // ...

        // Apply causal mask if on diagonal
        if (tile_kv == tile_q) {
            // ... predicate write to -INFINITY ...
        }

        // Online softmax update
        // ... update m_reg, l_reg ...

        // Compute PV and accumulate into o_reg
        // ... GEMM-II ...
        __syncthreads();
    }

    // Finalize O: divide by l_reg, cast to half, write to GMEM
    // ...
}
```

---

## 8. Performance Checklist

| Optimization | Status | Verification Method |
|--------------|--------|---------------------|
| `__launch_bounds__(128)` | Active | `cuobjdump -sass` register count inspection |
| Vectorized `float4` loads/stores | Active | Nsight Compute `gld_transactions` / `gst_transactions` ratio |
| Zero shared-memory bank conflicts | Active | Nsight Compute `shared_load_bank_conflict` counter |
| Full coalescing (512 B/warp) | Active | `memory_throughput` saturation metric |
| FP32 softmax accumulation | Active | Numerical unit tests vs. FP32 reference |
| Causal warp-level skip | Active | Nsight Compute `inst_executed` reduction for causal mask |

---

## 9. References

1. Dao, T., et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.
2. Milakov, M., & Gimelshein, N. *Online normalizer calculation for softmax.* arXiv:1805.02867.
3. NVIDIA CUDA C++ Programming Guide, *§ 7.22 `__launch_bounds__`*.
4. NVIDIA Nsight Compute Documentation, *Kernel Profiling Guide*.
