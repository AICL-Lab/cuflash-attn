# Quality Guidelines

> Code quality standards for CuFlash-Attn CUDA library.

---

## Overview

CuFlash-Attn follows **spec-driven development** with strict quality gates. All code changes must pass linting, testing, and formatting checks before commit.

---

## Forbidden Patterns

### CUDA Safety

| ❌ Forbidden | Reason | ✅ Alternative |
|-------------|--------|----------------|
| `cudaMemset()` | Synchronous, breaks stream ordering | `cudaMemsetAsync(ptr, val, size, stream)` |
| Unchecked `cudaMalloc` | Silent memory failures | Check return value, return `OUT_OF_MEMORY` |
| `throw` exceptions | Not C-compatible, overhead | Return `FlashAttentionError` enum |
| Raw `new`/`delete` for GPU memory | Unmanaged GPU memory | Use `cudaMalloc`/`cudaFree` with RAII |
| `__syncthreads()` in divergent code | Deadlock risk | Ensure all threads reach barrier |

### Memory Safety

| ❌ Forbidden | Reason |
|-------------|--------|
| Out-of-bounds array access | Undefined behavior on GPU |
| Use-after-free | Dangling pointers crash kernels |
| Memory leaks in error paths | GPU memory exhaustion |
| Missing `cudaFree` in workspace utilities | Resource leaks |

### Code Quality

| ❌ Forbidden | Reason |
|-------------|--------|
| `using namespace std;` in headers | Namespace pollution |
| `#pragma once` missing | Multiple include errors |
| Magic numbers without constants | Unmaintainable code |
| Missing spec ID in tests | Cannot trace to requirements |

---

## Required Patterns

### 1. CMake Presets Only

Always use presets, never direct `cmake -B`:

```bash
# CORRECT
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure

# WRONG
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### 2. clang-format Before Commit

```bash
# Required before every commit
find . \( -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" \) \
  ! -path "*/build/*" | xargs clang-format -i
```

Style: Google style, `IndentWidth=4`, `ColumnLimit=100` (see `.clang-format`)

### 3. Spec ID in Test Comments

```cpp
// Validates REQ-1.1, Property 1 - Numerical equivalence with PyTorch
TEST(ForwardTest, NumericalEquivalence) {
    // Test implementation...
}
```

### 4. Error Handling Pattern

```cpp
// Every public API must validate and return error codes
FlashAttentionError api_function(...) {
    FlashAttentionError err = validate_params(...);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }
    return launch_kernel(...);
}
```

### 5. Stream-Safe CUDA Operations

```cpp
// CORRECT: Async with explicit stream
cudaMemsetAsync(d_L, 0, size, stream);
kernel<<<grid, block, 0, stream>>>(...);
cudaStreamSynchronize(stream);  // Only when needed

// WRONG: Synchronous operations
cudaMemset(d_L, 0, size);  // Breaks stream ordering
```

---

## Testing Requirements

### Test Categories

| Category | Location | Purpose |
|----------|----------|---------|
| Unit tests | `tests/unit/*.cu` | Individual component correctness |
| Integration | `tests/integration/` | End-to-end API validation |
| Stress tests | `tests/unit/test_stress_edge_cases.cu` | Boundary conditions |
| PyTorch comparison | `tests/integration/` | Numerical accuracy verification |
| Package smoke | `tests/package_smoke/` | Installation verification |

### Test Requirements

1. **All new features** must have unit tests
2. **All API changes** must update existing tests
3. **All tests** must pass before commit (or be skipped when no GPU)
4. **Test naming**: `test_<feature>_<scenario>.cu`

### Running Tests

```bash
# All tests (auto-skips if no GPU)
ctest --preset release --output-on-failure

# Specific test category
ctest --preset release -R ForwardTest
ctest --preset release -R BackwardTest
ctest --preset release -R StressTest
```

---

## Code Review Checklist

Before submitting code, verify:

### Build & Test

- [ ] `cmake --preset release && cmake --build --preset release` succeeds
- [ ] `ctest --preset release --output-on-failure` passes (or skips when no GPU)
- [ ] `clang-format` has been run on all changed files

### CUDA-Specific

- [ ] All `cudaMalloc` calls checked for errors
- [ ] All `cudaMemsetAsync` used instead of `cudaMemset`
- [ ] Workspace memory freed in all exit paths (success and error)
- [ ] Stream parameter passed through all async operations

### Spec Compliance

- [ ] Changes align with `openspec/specs/`
- [ ] New features have spec IDs in test comments
- [ ] API changes update `include/cuflash/flash_attention.h`

### Documentation

- [ ] Public API documented with Doxygen comments
- [ ] README updated if user-facing behavior changes
- [ ] CHANGELOG updated for notable changes

---

## CI/CD Quality Gates

GitHub Actions enforces these checks:

| Workflow | Checks |
|----------|--------|
| `ci.yml` | Build (release), tests, clang-format, clang-tidy |
| `codeql.yml` | Security analysis (weekly) |
| `pages.yml` | Documentation deployment |

### Required Checks for Merge

- ✅ Build succeeds on all target architectures
- ✅ All tests pass (or skip when no GPU)
- ✅ clang-format check passes
- ✅ clang-tidy warnings resolved

---

## Common Mistakes

### ❌ Skipping clang-format

```bash
# WRONG: Commit without formatting
git add .
git commit -m "Add feature"

# CORRECT: Format first
find . -name "*.cu" -o -name "*.cuh" | xargs clang-format -i
git add .
git commit -m "Add feature"
```

### ❌ Missing CUDA Error Check

```cpp
// WRONG: Unchecked allocation
cudaMalloc(&d_workspace, workspace_size);

// CORRECT: Check and propagate
cudaError_t err = cudaMalloc(&d_workspace, workspace_size);
if (err != cudaSuccess) {
    return FlashAttentionError::OUT_OF_MEMORY;
}
```

### ❌ Testing Without Spec ID

```cpp
// WRONG: No traceability
TEST(ForwardTest, CausalMask) { ... }

// CORRECT: Reference spec
// Validates REQ-2.1, Property 3 - Causal masking correctness
TEST(ForwardTest, CausalMask) { ... }
```

---

## Summary

| Gate | Requirement |
|------|-------------|
| **Build** | `cmake --preset release` succeeds |
| **Format** | clang-format run on all changes |
| **Tests** | All pass or skip (no GPU) |
| **Specs** | Changes traceable to `openspec/specs/` |
| **Errors** | All CUDA calls checked |
| **Streams** | Async operations only |
