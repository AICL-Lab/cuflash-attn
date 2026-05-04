# Backend Development Guidelines

> Best practices for CUDA backend development in CuFlash-Attn.

---

## Overview

This directory contains guidelines for CuFlash-Attn's CUDA C++ backend. The project is a **pure GPU library** without traditional backend/frontend separation.

---

## Guidelines Index

| Guide | Description | Status |
|-------|-------------|--------|
| [Directory Structure](./directory-structure.md) | Module organization and file layout | ✅ Complete |
| [Database Guidelines](./database-guidelines.md) | Not applicable (no database) | ✅ N/A |
| [Error Handling](./error-handling.md) | Error codes, CUDA error handling | ✅ Complete |
| [Quality Guidelines](./quality-guidelines.md) | Code standards, forbidden patterns | ✅ Complete |
| [Logging Guidelines](./logging-guidelines.md) | Error code strategy (no traditional logging) | ✅ Complete |

---

## Pre-Development Checklist

Before writing any CUDA code, read:

1. **Directory Structure** - Understand layered architecture (API → Kernel → Utilities)
2. **Error Handling** - Required error code patterns for all public APIs
3. **Quality Guidelines** - Forbidden patterns (cudaMemset, unchecked cudaMalloc, etc.)

---

## Key Principles

### 1. Spec-Driven Development

All implementations must trace to `openspec/specs/`. Test comments must reference spec IDs:

```cpp
// Validates REQ-1.1, Property 1 - Numerical equivalence
TEST(ForwardTest, NumericalEquivalence) { ... }
```

### 2. Error Code Returns

All public APIs return `FlashAttentionError` enum, never throw exceptions:

```cpp
FlashAttentionError flash_attention_forward(...) {
    FlashAttentionError err = validate_params(...);
    if (err != FlashAttentionError::SUCCESS) {
        return err;
    }
    return launch_kernel(...);
}
```

### 3. Stream-Safe CUDA Operations

Always use async operations with explicit stream:

```cpp
// CORRECT
cudaMemsetAsync(d_L, 0, size, stream);

// WRONG
cudaMemset(d_L, 0, size);  // Breaks stream ordering
```

---

## Quick Reference

| What | Where |
|------|-------|
| Public API | `include/cuflash/flash_attention.h` |
| API Dispatch | `src/api/flash_attention_api.cu` |
| Forward Kernels | `src/forward/*.cu` |
| Backward Kernels | `src/backward/*.cu` |
| Internal Utilities | `src/kernels/*.cuh` |
| Unit Tests | `tests/unit/*.cu` |

---

**Language**: All documentation written in **English**.
