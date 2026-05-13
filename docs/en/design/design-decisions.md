# Design Decisions

This document records the architectural decisions (ADRs) that shape CuFlash-Attn. Each entry follows the format: **Context**, **Decision**, **Consequences** (pros/cons), and **References**.

---

## ADR-001: Restriction of `head_dim` to {32, 64, 128}

### Context

FlashAttention relies on aggressive static tiling of SRAM buffers. The tile dimensions—$B_r$ (query rows per tile), $B_c$ (key/value columns per tile), and $d$ (head dimension)—must be compile-time constants to enable:

- Loop unrolling by the compiler, eliminating dynamic bounds checks.
- Static shared-memory sizing via `extern __shared__` or fixed-size arrays.
- Register allocation planning: the number of accumulator registers is $B_r \times B_c \times 4$ bytes (FP32).

Allowing arbitrary `head_dim` would require dynamic shared-memory arithmetic, partial loop unrolling, and unpredictable register pressure, all of which degrade kernel performance and complicate the instruction schedule.

### Decision

`head_dim` is restricted to the set `{32, 64, 128}`. Each value selects a distinct template instantiation of the kernel family:

```cpp
template <int head_dim>
struct KernelTraits;

template <>
struct KernelTraits<64> {
    static constexpr int Br = 64;
    static constexpr int Bc = 64;
    static constexpr int smem_size = 48 * 1024;  // bytes
};

template <>
struct KernelTraits<128> {
    static constexpr int Br = 64;
    static constexpr int Bc = 32;   // halved to fit SRAM
    static constexpr int smem_size = 64 * 1024;  // bytes
};
```

The host dispatch layer routes to the correct instantiation via a `switch` or function-pointer table:

```cpp
FlashAttentionError dispatch_forward(int head_dim, ...) {
    switch (head_dim) {
        case 32:  return flash_attn_fwd<32>(...);
        case 64:  return flash_attn_fwd<64>(...);
        case 128: return flash_attn_fwd<128>(...);
        default:  return FlashAttentionError::UNSUPPORTED_HEAD_DIM;
    }
}
```

### Consequences

| Pros | Cons |
|------|------|
| Maximum compiler optimization: fully unrolled inner loops, no runtime division/modulo | Models with `head_dim = 80` (e.g., some LLaMA variants) require padding or are unsupported |
| Predictable shared-memory footprint; no risk of oversubscribing SM SRAM | Library size increases linearly with the number of template instantiations (3× code bloat) |
| Simpler warp-level microkernels: every thread knows its tile boundaries at compile time | Cannot dynamically tune tile sizes for exotic sequence-length / head-dim combinations |
| Verifiable correctness: each instantiation is tested in isolation | Adding a new head_dim requires a new specialization, not just a runtime parameter |

### References

- Dao, T., et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.
- NVIDIA CUDA C++ Best Practices Guide, *§ Kernel Loop Unrolling*.

---

## ADR-002: FP16 Input with FP32 Internal Accumulation

### Context

Modern GPUs (Ampere, Hopper) provide 2× peak TFLOPS for FP16 Tensor Cores compared to FP32 CUDA Cores, and HBM bandwidth is also 2× higher for 16-bit elements. However, the attention softmax is numerically unstable in reduced precision: summing exponentials across thousands of tokens causes catastrophic cancellation and overflow in FP16.

Standard cuDNN/cuBLAS approaches solve this by keeping the compute type (accumulation precision) separate from the data type (memory precision).

### Decision

All FP16 kernels use **FP32 for every intermediate accumulation**, while keeping GMEM traffic in FP16:

| Stage | Precision | Storage Location |
|-------|-----------|------------------|
| Q, K, V loads | `half` | Global Memory → Shared Memory |
| $S = QK^T$ MAC | `float` | Registers |
| Online softmax ($m$, $l$) | `float` | Registers |
| $P = \text{softmax}(S)$ | `half` | Shared Memory (transient) |
| $O += PV$ MAC | `float` | Registers |
| Final output store | `half` | Global Memory |

This matches the `CUDA_R_16F` / `CUDA_R_32F` pattern used by cuBLAS GEMM APIs.

### Consequences

| Pros | Cons |
|------|------|
| Numerical equivalence to FP32 reference within `< 1e-3` relative error | Doubles register footprint for accumulator arrays (FP32 vs FP16) |
| Exploits 2× HBM bandwidth of FP16 for the memory-bound Q/K/V loads | Requires explicit `__half2float` / `__float2half` conversions in the inner loop |
| Prevents softmax denominator overflow for sequences up to 64k tokens | Slightly higher instruction count; however, the kernel is memory-bound, so IPC impact is negligible |
| Compatible with mixed-precision training pipelines (PyTorch AMP) | — |

### References

- Micikevicius, P., et al. *Mixed Precision Training.* ICLR 2018.
- NVIDIA A100 Tensor Core GPU Architecture whitepaper, *§ FP16 Accumulation Modes*.

---

## ADR-003: `ctypes` over `pybind11` for Python Bindings

### Context

CuFlash-Attn exposes a plain C ABI (`extern "C"`) to maximize interoperability. A Python wrapper is required for integration with PyTorch and NumPy. The two primary candidates are:

1. **pybind11**: C++ header-only library; feature-rich, type-safe, but introduces a C++ dependency and significant compile-time overhead.
2. **ctypes**: Python standard-library module; loads shared objects dynamically, zero build dependencies, but requires manual type signatures.

### Decision

Use `ctypes` for the Python wrapper. The C ABI is intentionally minimal:

```c
// include/cuflash/cuflash.h
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CUFLASH_SUCCESS = 0,
    CUFLASH_ERROR_INVALID_SHAPE = 1,
    CUFLASH_ERROR_CUDA_ERROR = 2,
    // ...
} cuflash_status_t;

cuflash_status_t cuflash_attention_forward(
    const void* q, const void* k, const void* v,
    void* out,
    int batch, int heads, int seq_len, int head_dim,
    float scale,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
```

The Python shim is ~80 lines of `ctypes` boilerplate:

```python
# python/cuflash/__init__.py
import ctypes
import torch

_lib = ctypes.CDLL("libcuflash.so")
_lib.cuflash_attention_forward.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float,
    ctypes.c_void_p,  # cudaStream_t
]
_lib.cuflash_attention_forward.restype = ctypes.c_int

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    scale: float = None) -> torch.Tensor:
    # ... shape validation, output allocation, ctypes call ...
    status = _lib.cuflash_attention_forward(
        q.data_ptr(), k.data_ptr(), v.data_ptr(),
        out.data_ptr(),
        B, H, N, d,
        scale or (d ** -0.5),
        torch.cuda.current_stream().cuda_stream,
    )
    if status != 0:
        raise RuntimeError(f"CuFlash-Attn forward failed with status {status}")
    return out
```

### Consequences

| Pros | Cons |
|------|------|
| Zero C++ build dependencies; no pybind11 submodule or CMake integration | Manual type safety: mismatched `argtypes` cause undefined behavior at runtime |
| Pure-Python wrapper; editable install without compilation | Error messages are less descriptive than pybind11's automatic type conversion errors |
| Smaller binary size; no C++ exception-handling machinery in `.so` | No automatic Python object lifetime management (must manage `data_ptr()` carefully) |
| Aligns with the project's "minimal surface area" philosophy | Adding new API variants requires manual ctypes signature updates |

### References

- Python Standard Library, `ctypes` module documentation.
- pybind11 documentation, *§ Build system integration*.

---

## ADR-004: OpenSpec-Driven Development

### Context

CUDA kernel projects are notoriously brittle: a small tweak to tile size or memory layout can silently break numerical correctness or halve occupancy. Without a formal specification, changes are justified by "it worked in my benchmark," leading to irreproducible performance and hidden regressions.

### Decision

Adopt **OpenSpec** as the single source of truth for all requirements, API contracts, and verification criteria. Every behavioral change is tracked as a change proposal:

```
openspec/
├── specs/
│   ├── design/flash-attention-design.md       # REQ-1.1, REQ-1.2, ...
│   └── verification/flash-attention-verification.md  # Test criteria, tolerances
├── changes/
│   ├── active/          # Proposed or in-progress changes
│   └── archive/         # Completed, reviewed, and merged changes
└── config.yaml          # Project rules, anti-patterns, context
```

The workflow is mandatory:

1. `/opsx:propose <name>` — create `proposal.md + design.md + tasks.md`
2. Read the relevant spec in `openspec/specs/`
3. `/opsx:apply <name>` — implement the change
4. `/verify` — format check, build, test
5. `/opsx:archive <name>` — close the change loop

### Consequences

| Pros | Cons |
|------|------|
| Every code change is traceable to a written requirement | Higher upfront documentation cost (offset by reduced debug time) |
| Prevents "gold-plating": no features exist without a spec | Contributors must learn the OpenSpec CLI workflow |
| Enables deterministic code review: reviewer checks spec compliance, not just style | Rapid prototyping is slower; hotfixes still require a lightweight change record |
| Creates a permanent audit trail for safety-critical or research-reproducible deployments | — |

### References

- OpenSpec project documentation: `openspec/README.md`.
- `openspec/config.yaml` — project-specific rules and anti-patterns.

---

## ADR-005: Fixed Block Size of 128 Threads

### Context

The choice of `blockDim.x` governs:

- Occupancy: 128 threads = 4 warps, which allows 8 blocks per SM on Ampere (32 warps / 4 warps = 8), or 4 blocks on Hopper with 128 threads per block.
- Shared memory per thread: halving the block size doubles the SRAM budget per thread, but halves GEMM parallelism.
- Instruction cache pressure: smaller blocks reduce cache thrashing; larger blocks increase ILP but reduce the number of concurrent blocks.

### Decision

Fix `blockDim.x = 128` for **all kernel variants**, enforced by `__launch_bounds__(128)`:

```cpp
__global__ void __launch_bounds__(128) flash_attn_fwd_kernel(...)
```

This is coupled to the warp-level assignment:

| Warp ID | Lane Range | Logical Role |
|---------|------------|--------------|
| 0 | 0–31 | GMEM → SMEM load (Q, K, V) |
| 1 | 32–63 | GEMM-I: $S = QK^T$ |
| 2 | 64–95 | Softmax: online normalization |
| 3 | 96–127 | GEMM-II: $O += PV$ and GMEM store |

> In practice, all warps participate in each phase with strip-mined tile sub-assignments to maximize ILP.

### Consequences

| Pros | Cons |
|------|------|
| Simplifies register allocation: compiler knows exactly 128 threads, no edge cases | Not optimal for very small head dimensions (e.g., $d=8$), where 64 threads would suffice |
| Maximizes Ampere SM occupancy (8 blocks × 4 warps = 32 warps, fully subscribed) | Less flexible than autotuned block sizes found in Triton or CUTLASS |
| 128 is a multiple of 32 (warp size) and 4 (float4 vector width), avoiding lane-id branching | Fixed cost even when $N < 128$; threads are masked but still consume scheduling slots |
| Aligns with WMMA fragment dimensions (16×16×16) for future Tensor Core integration | — |

### References

- NVIDIA CUDA Occupancy Calculator, *Ampere SM resource partitioning*.
- Vasily Volkov, *Better Performance at Lower Occupancy* (GTC 2010).

---

## ADR-006: O(N) SRAM Tiling over Standard O(N²) Attention

### Context

Standard attention materializes the full $N \times N$ score matrix $S$ and the $N \times N$ attention matrix $P$ in HBM:

$$
\text{Memory} = O(N \cdot d) \text{ for } Q,K,V + O(N^2) \text{ for } S,P + O(N \cdot d) \text{ for } O
$$

For $N = 16{,}384$ and FP16, $S$ alone consumes $512$ MiB, exceeding the SRAM of a single SM by three orders of magnitude. The resulting algorithm is **memory-bound**: compute units sit idle while HBM feeds the $N^2$ intermediate tensors.

### Decision

Implement **IO-aware tiling**: the $N \times N$ attention computation is decomposed into smaller $B_r \times B_c$ tiles that fit in shared memory. Each tile is loaded, computed, and discarded without ever writing $S$ or $P$ to HBM:

```
Standard Attention:        FlashAttention (IO-Aware):
┌─────────────┐          ┌─────────────┐
│  Q K V O    │          │  Q K V O    │
│    ↓        │          │    ↓        │
│  S (N×N)    │          │  SRAM tiles │
│    ↓        │          │    ↓        │
│  P (N×N)    │          │  (no HBM)   │
│    ↓        │          │    ↓        │
│  O          │          │  O          │
└─────────────┘          └─────────────┘
      HMB                       HBM
```

The online softmax algorithm (Milakov & Gimelshein, 2018) maintains the correct output $O$ incrementally, using only $O(B_r)$ additional registers per block for the running statistics $m$ and $l$.

### Consequences

| Pros | Cons |
|------|------|
| Memory complexity reduced from $O(N^2)$ to $O(N)$; supports arbitrary sequence lengths bounded only by HBM capacity for Q/K/V | Higher arithmetic intensity: the kernel is now compute-bound on small sequence lengths ($N < 2{,}048$) |
| HBM bandwidth is the bottleneck only for Q/K/V/O, not for $N^2$ intermediates | More complex kernel logic: online softmax rescaling, tile synchronization, and causal masking require careful correctness proofs |
| Enables attention on longer sequences without gradient checkpointing or memory sharding | Slightly higher FLOP count due to online rescaling (empirically < 3% overhead) |
| Backward pass also benefits: $dQ$, $dK$, $dV$ are computed tile-by-tile without materializing $dP$ | Kernel fusion limits flexibility; cannot easily insert custom operations between $S$ and $P$ |

### References

- Dao, T., et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.
- Milakov, M., & Gimelshein, N. *Online normalizer calculation for softmax.* arXiv:1805.02867.
- Rabe, M. N., & Staats, C. *Self-Attention Does Not Need $O(n^2)$ Memory.* arXiv:2112.05682.

---

## Appendix: Decision Summary Matrix

| ID | Decision | Primary Driver | Secondary Trade-off |
|----|----------|----------------|---------------------|
| ADR-001 | `head_dim ∈ {32, 64, 128}` | Compile-time tiling | Reduced model compatibility |
| ADR-002 | FP16 → FP32 accumulation | Numerical stability | Register pressure |
| ADR-003 | `ctypes` over `pybind11` | Zero-dependency build | Manual type safety |
| ADR-004 | OpenSpec governance | Correctness traceability | Documentation overhead |
| ADR-005 | `blockDim.x = 128` fixed | Occupancy / warp mapping | Inflexibility for tiny problems |
| ADR-006 | O(N) SRAM tiling | Memory scalability | Kernel complexity |
