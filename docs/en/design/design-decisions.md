# Design Decisions

This document records the current architectural decisions that still matter after the repository slimdown. It intentionally keeps only decisions that directly affect the CUDA implementation or the public integration surface.

---

## ADR-001: Restrict `head_dim` to {32, 64, 128}

**Context**  
The kernels rely on compile-time tiling choices to keep shared-memory usage and register pressure predictable.

**Decision**  
Support only `head_dim` values `32`, `64`, and `128`, and dispatch to explicit template instantiations.

**Consequences**

- Maximum compiler optimization and predictable shared-memory sizing
- Simpler host-side validation and narrower test surface
- Unsupported model variants must pad or adapt externally

---

## ADR-002: Use FP32 accumulation for FP16 paths

**Context**  
Attention score accumulation and online softmax are numerically fragile in half precision.

**Decision**  
Store inputs/outputs as `half` where needed, but perform internal accumulation and normalization in `float`.

**Consequences**

- Better numerical stability for long sequences
- Slightly higher internal register/shared-memory pressure
- Stable behavior across FP32 and FP16 entry points

---

## ADR-003: Keep the Python integration surface at the C ABI level

**Context**  
The project is intentionally lightweight and should remain easy to embed without adding large binding frameworks.

**Decision**  
Expose a C ABI and document `ctypes`-style usage rather than adopting pybind11 or a custom runtime layer.

**Consequences**

- Smaller dependency surface
- Easier integration from multiple languages
- Less ergonomic runtime type checking than higher-level binding frameworks

---

## ADR-004: Fix the core thread-block shape at 128 threads

**Context**  
Kernel occupancy, warp assignment, and shared-memory tiling have all been tuned around a 128-thread block shape.

**Decision**  
Keep `blockDim.x = 128` and continue to pair it with explicit launch configuration checks.

**Consequences**

- Predictable scheduling and occupancy characteristics
- Fewer tuning dimensions to maintain
- Any future change requires retuning and revalidation across kernels

---

## ADR-005: Prefer explicit repository docs over external process frameworks

**Context**  
The repository previously accumulated overlapping AI/process frameworks that duplicated guidance and drifted from the actual codebase.

**Decision**  
Keep contributor guidance in the repository’s native docs (`README`, `CONTRIBUTING`, GitHub workflow files, and VitePress pages) and remove external control layers.

**Consequences**

- Lower maintenance overhead
- Fewer contradictory instructions
- Less ceremony when updating docs and workflows
