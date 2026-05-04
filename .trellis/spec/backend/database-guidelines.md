# Database Guidelines

> Not applicable to CuFlash-Attn.

---

## Overview

CuFlash-Attn is a **pure CUDA C++ library** for GPU-accelerated attention computation. It does **not use any database** system.

---

## Not Applicable

This guideline file is **not applicable** because:

1. **No persistent storage**: CuFlash-Attn operates on GPU memory only
2. **No ORM**: No object-relational mapping required
3. **No migrations**: No database schema to evolve
4. **No transactions**: No transactional data operations

---

## Related Guidelines

For data management patterns relevant to this project, see:

- [Directory Structure](./directory-structure.md) - Memory management patterns
- [Error Handling](./error-handling.md) - CUDA error handling (not DB errors)
- [Quality Guidelines](./quality-guidelines.md) - Memory safety patterns

---

## If You Need Database Integration

Consumers of CuFlash-Attn may need to store attention weights or results in databases. Such integration is **outside the library's scope** and should be handled at the application layer.

Example consumer architecture:

```
Application Layer
├── Database (PostgreSQL, MongoDB, etc.)
├── CuFlash-Attn (GPU attention computation)
└── Data Pipeline (load/store tensors)
```
