# Directory Structure

> How CUDA backend code is organized in CuFlash-Attn.

---

## Overview

CuFlash-Attn is a **pure CUDA C++ library** with no traditional backend/frontend split. The codebase follows a layered architecture optimized for GPU kernel development and Python binding integration.

---

## Directory Layout

```
cuflash-attn/
├── include/cuflash/          # Public API headers
│   ├── flash_attention.h     # Main API (C++ namespace + C ABI)
│   ├── export.h              # Visibility macros
│   └── version.h.in          # Version header template
├── src/                      # Implementation
│   ├── api/                  # API dispatch layer
│   │   └── flash_attention_api.cu  # Parameter validation, dispatch to kernels
│   ├── forward/              # Forward pass kernels
│   │   ├── flash_attention_forward.cu   # FP32 forward kernel
│   │   └── flash_attention_fp16.cu      # FP16 forward kernel
│   ├── backward/             # Backward pass kernels
│   │   ├── flash_attention_backward.cu  # FP32 backward kernel
│   │   └── flash_attention_backward_fp16.cu  # FP16 backward kernel
│   └── kernels/              # Internal kernel utilities (.cuh)
│       ├── online_softmax.cuh      # Online softmax algorithm
│       ├── matmul.cuh              # Matrix multiplication utilities
│       ├── kernel_launch_utils.cuh # Grid/block calculation helpers
│       └── workspace_utils.cuh     # Workspace allocation helpers
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests (8 files)
│   │   ├── test_forward.cu
│   │   ├── test_backward.cu
│   │   ├── test_online_softmax.cu
│   │   ├── test_dtype.cu
│   │   ├── test_causal_mask.cu
│   │   ├── test_numerical_stability.cu
│   │   ├── test_stress_edge_cases.cu
│   │   └── test_error_handling.cu
│   ├── integration/          # Integration tests
│   │   └── test_api_smoke.cpp
│   └── package_smoke/        # Package smoke tests
│       └── main.cpp
├── benchmarks/               # Performance benchmarks
│   └── cuflash_attn_bench.cu
├── examples/                 # Usage examples
│   ├── basic_usage.cu        # C++ example
│   └── python_binding.py     # Python ctypes example
├── docs/                     # VitePress documentation site
├── openspec/                 # OpenSpec source of truth
│   ├── specs/                # Accepted design + verification specs
│   └── changes/              # Change proposals, designs, task lists
└── cmake/                    # CMake modules & packaging
```

---

## Module Organization

### Layered Architecture

| Layer | Directory | Purpose |
|-------|-----------|---------|
| **Public API** | `include/cuflash/` | Headers for external consumers |
| **API Dispatch** | `src/api/` | Parameter validation, kernel selection |
| **Kernel Implementation** | `src/forward/`, `src/backward/` | Core algorithm implementation |
| **Internal Utilities** | `src/kernels/` | Reusable kernel components (.cuh only) |
| **Tests** | `tests/unit/`, `tests/integration/` | Verification and regression tests |

### Key Principles

1. **`.cuh` files in `src/kernels/`** — Internal utilities, never exposed in public headers
2. **`.cu` files in `src/forward/` and `src/backward/`** — Kernel implementations
3. **Single API header** — `include/cuflash/flash_attention.h` exports all public symbols
4. **Test naming** — `test_<feature>.cu` for unit tests, `test_api_smoke.cpp` for integration

---

## Naming Conventions

### Files

| Type | Pattern | Example |
|------|---------|---------|
| Public header | `<module>.h` | `flash_attention.h` |
| Internal header | `<utility>.cuh` | `online_softmax.cuh` |
| Kernel source | `flash_attention_<pass>[_<dtype>].cu` | `flash_attention_forward.cu`, `flash_attention_fp16.cu` |
| Unit test | `test_<feature>.cu` | `test_forward.cu`, `test_causal_mask.cu` |
| Benchmark | `<lib>_bench.cu` | `cuflash_attn_bench.cu` |

### Directories

- Use lowercase with underscores: `flash_attention`, `online_softmax`
- Group by **pass direction** (`forward/`, `backward/`) for kernels
- Group by **test type** (`unit/`, `integration/`, `package_smoke/`)

---

## Adding New Code

### Adding a New Kernel

1. Create `src/forward/` or `src/backward/` directory if needed
2. Add `<kernel_name>.cu` with kernel implementation
3. Add declaration in `include/cuflash/flash_attention.h`
4. Add dispatch logic in `src/api/flash_attention_api.cu`
5. Create `tests/unit/test_<kernel_name>.cu`

### Adding Internal Utilities

1. Create `src/kernels/<utility>.cuh` (NEVER in `include/`)
2. Use `__device__` or `__host__ __device__` functions
3. Document thread safety and memory requirements

---

## Examples

### Well-organized module: `src/forward/`

```
src/forward/
├── flash_attention_forward.cu   # FP32 forward kernel
└── flash_attention_fp16.cu      # FP16 forward kernel
```

- Clear separation by data type
- Each file implements a single kernel variant
- Shared utilities pulled from `src/kernels/`

### Well-organized test: `tests/unit/test_forward.cu`

- Tests for numerical correctness
- Edge case handling
- References spec IDs: `// Validates REQ-1.1, Property 1`

---

## Anti-patterns

| ❌ Don't | ✅ Do Instead |
|---------|---------------|
| Put `.cuh` in `include/cuflash/` | Keep internal utilities in `src/kernels/` |
| Create deeply nested directories | Keep structure flat (max 2-3 levels) |
| Mix FP32 and FP16 in same file | Separate files per dtype |
| Add test utilities to `src/` | Keep test helpers in `tests/` |
