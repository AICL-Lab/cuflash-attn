# Tensor Core Migration Plan

> Status: **Phase 2 (forward) implemented** — the WMMA forward kernel for
> FP16/BF16 landed in v0.5.0 (`src/forward/flash_attention_forward_wmma.cu`)
> with runtime dispatch and the scalar path kept as fallback. Numerical
> verification on real hardware (test suite + compute-sanitizer + benchmarks)
> is pending via the GPU workflow; backward WMMA (Phase 2 continued) is next.
> Every phase is designed to be landed and verified independently on real
> hardware.

## Why this exists

CuFlash-Attn is correct but slow. The kernels compute every matmul as a naive
scalar loop — one thread produces one output element and walks the `K`/`HEAD_DIM`
axis serially (`src/kernels/impl/tile_io.cuh`, `matmul_ABt`), and the fused
softmax/`P@V` section assigns **one whole query row to one thread**
(`src/forward/flash_attention_forward_typed.cu`). No Tensor Cores, no
`cp.async`, no register tiling, no warp-level cooperation.

For a reduced-precision attention kernel, Tensor Cores are where ~90% of the
achievable throughput lives. Until they are used, the "high-performance" framing
in the README is aspirational rather than true. This document is the path to
making it true, broken into stages that each compile, pass the existing suite,
and show a measurable speedup before the next begins.

**Hard rule for every phase:** land it behind the existing tests
(`test_forward`, `test_backward`, `test_dtype`, the PyTorch comparison) and
`scripts/run_compute_sanitizer.sh` before moving on. Do not stack unverified
changes.

## Phase 0 — Baseline and measurement (do this first)

Before optimizing, make the gap visible and reproducible:

1. Run `benchmarks/bench_flash_attention.cu` (now reports `TFLOP/s` and
   `HBM GB/s`) on the target GPU and record numbers in
   `docs/en/performance/benchmarks.md` with the exact device/driver/CUDA.
2. Add the naive materialized baseline (already in the benchmark) and the
   official `flash-attn` / PyTorch SDPA numbers for the same shapes.
3. Profile one forward launch with Nsight Compute; note achieved occupancy,
   SM throughput, and memory throughput. This tells you whether Phase 1 or
   Phase 2 buys more.

Exit criterion: a checked-in table of baseline TFLOP/s per shape.

## Phase 1 — Register tiling and warp cooperation (no Tensor Cores)

Cheapest large win, and it de-risks everything after. Replace the
"one-thread-one-row" inner loops with a tiled scheme where each thread owns a
small `MxN` register block of the output and the block cooperates per warp.

- Keep FP32 in shared memory as today, but have each thread accumulate a
  `TM x TN` micro-tile (e.g. `4 x 4`) of `S` and of `O`, reading `Q`/`K`/`V`
  into registers.
- This alone typically yields several× over the scalar version by cutting
  instruction overhead and shared-memory traffic, and it is the scaffolding the
  Tensor Core phases reuse.
- Pad shared-memory rows (or swizzle indices) to remove bank conflicts, which
  the current `A[row*K+k]` / `B[col*K+k]` access pattern has when `K` is a
  multiple of 32.

Exit criterion: forward+backward pass the suite and compute-sanitizer; benchmark
shows the speedup; numbers recorded.

## Phase 2 — WMMA for FP16/BF16 (sm_70+)

Introduce Tensor Cores via the `nvcuda::wmma` API for the reduced-precision
paths (FP32 stays on the Phase-1 CUDA-core path).

- Store `Q`/`K`/`V` tiles in shared memory **in the input precision** (half /
  bf16), not up-converted to float — this halves shared-memory pressure and is
  what lets block sizes grow.
- Compute `S = Q @ Kᵀ` and `O += P @ V` with `wmma::fragment`
  (`m16n16k16` for half; bf16 fragments need sm_80+). Accumulate into
  `float` fragments.
- Keep the online-softmax state (`m`, `l`) and the deferred normalization
  exactly as now; only the two matmuls change.
- The backward's three matmuls (`dV = Pᵀ@dO`, `dP = dO@Vᵀ`, `dQ/dK = dS·…`)
  get the same treatment.

Sketch of the `QKᵀ` step:

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;   // Q tile
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;   // K tile (Kᵀ via col_major)
fragment<accumulator, 16, 16, 16, float>        c_frag;   // S tile
fill_fragment(c_frag, 0.0f);

// each warp owns one 16x16 S sub-tile; loop over the K dimension in 16-wide steps
for (int k0 = 0; k0 < HEAD_DIM; k0 += 16) {
    load_matrix_sync(a_frag, Q_smem + row0 * HEAD_DIM + k0, HEAD_DIM);
    load_matrix_sync(b_frag, K_smem + col0 * HEAD_DIM + k0, HEAD_DIM);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
}
// c_frag now holds scale * S for this warp's 16x16 block; apply scale + softmax
```

Exit criterion: FP16/BF16 forward+backward match the FP32 reference within the
existing tolerances; compute-sanitizer clean; benchmark shows a large multiple
over Phase 1 at `head_dim ∈ {64,128}`.

## Phase 3 — Overlap load and compute with `cp.async` (sm_80+)

Hide global-memory latency by double-buffering the K/V tiles:

- Use `cuda::memcpy_async` / `cp.async` to prefetch the next K/V tile into a
  second shared-memory buffer while the current tile is being consumed.
- Coordinate with `cuda::pipeline` barriers instead of whole-block
  `__syncthreads`.
- This is where block sizes and occupancy are tuned against the opt-in shared
  memory limit (already plumbed through `prepare_dynamic_smem_launch`).

Exit criterion: higher SM throughput in Nsight Compute; benchmark improvement;
suite + sanitizer green.

## Phase 4 — Hopper: WGMMA + TMA (sm_90, optional)

For H100-class throughput, move to warpgroup MMA (`wgmma`) and the Tensor Memory
Accelerator for async bulk copies, in the FlashAttention-3 style. This is the
largest effort and should only start once Phases 1–3 are solid and measured.
Consider depending on CUTLASS here rather than hand-writing PTX.

## Cross-cutting concerns

- **Dispatch by arch at runtime.** Keep the scalar/Phase-1 path as the fallback
  for any arch without the needed Tensor Core instruction; select the kernel in
  `launch_flash_attention_forward_typed` via a compiled-arch check.
- **`head_dim` coverage.** WMMA fragments are 16-wide; `head_dim ∈ {32,64,128}`
  all divide cleanly. Keep the existing `is_supported_head_dim` gate.
- **Numerical parity.** Accumulation stays FP32 throughout; the only precision
  change is the storage format of the shared-memory tiles, which the existing
  FP16/BF16-vs-FP32 tests already police.
- **L stays FP32.** The logsumexp output is `float*` (see CHANGELOG); do not
  regress this when restructuring shared memory.

## What "done" means

The README's performance table is reproducible from `benchmarks/` on a named
device, shows CuFlash-Attn within a small factor of the official FlashAttention
at `head_dim=128`, and every number is backed by a checked-in benchmark run
rather than an estimate.
