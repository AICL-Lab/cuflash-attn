# AGENTS.md - AI Agent Instructions

## Project: CuFlash-Attn

High-performance CUDA C++ FlashAttention implementation from scratch. This is a CMake-based C++/CUDA project with Spec-Driven Development (SDD) workflow.

---

## Critical Workflow: Spec-Driven Development (SDD)

**ALL implementation must follow specs in `/specs/` as the single source of truth.**

### Before Writing Any Code

1. **Read relevant specs FIRST:**
   - `/specs/product/` - Product requirements & acceptance criteria
   - `/specs/rfc/` - Technical design documents
   - `/specs/api/` - API specifications
   - `/specs/testing/` - Testing specifications

2. **If user request conflicts with specs:** STOP and ask whether to update specs first

3. **For new features/API changes:** Propose spec updates BEFORE coding. Wait for approval.

4. **Test against spec:** Reference spec IDs in test comments (e.g., `// Validates RFC-001, Property 1`)

### Code Generation Rules

- API changes → Update `/specs/api/`
- New features → Update `/specs/product/`
- Architecture changes → Create/update `/specs/rfc/`
- **Never** add features not defined in specs (no gold-plating)

---

## Build System

CMake with presets. **Requires CUDA 11.0+ and C++17.**

```bash
# Standard release build
cmake --preset release
cmake --build --preset release

# Run tests
ctest --preset release --output-on-failure

# Run specific test categories
ctest --preset release -R ForwardTest    # Forward pass tests
ctest --preset release -R BackwardTest   # Backward pass tests
ctest --preset release -R StressTest     # Stress & edge cases
ctest --preset release -R PyTorch        # PyTorch comparison (requires GPU + PyTorch)

# Debug with AddressSanitizer
cmake --preset debug-asan
cmake --build --preset debug-asan
ctest --preset debug-asan

# Build all GPU architectures (sm_70-90)
cmake --preset all-architectures
cmake --build --preset all-architectures
```

### Build Presets Available

| Preset | Use Case |
|--------|----------|
| `release` | Standard optimized build (default) |
| `release-fast-math` | With `--use_fast_math` (less precise, faster) |
| `debug-asan` | Debug with AddressSanitizer |
| `minimal` | No tests/examples (smallest binary) |
| `all-architectures` | sm_70,75,80,86,89,90 (V100 through H100) |

**Default CUDA architectures:** sm_80, sm_86 (A100 + RTX 30xx/40xx)

---

## Project Structure

```
cuflash-attn/
├── specs/                      # SDD: Single Source of Truth
│   ├── product/                # Product requirements
│   ├── rfc/                    # Technical design (RFCs)
│   ├── api/                    # API specifications
│   └── testing/                # Testing specs
├── include/cuflash/            # Public API headers
│   ├── flash_attention.h       # Main API (C++ + C ABI)
│   ├── export.h                # Visibility macros
│   └── version.h.in            # Version template
├── src/                        # Implementation
│   ├── api/                    # API dispatch layer
│   ├── forward/                # Forward kernels
│   ├── backward/               # Backward kernels
│   └── kernels/                # Internal kernel utilities (.cuh)
├── tests/
│   ├── unit/                   # Unit tests (8 files)
│   ├── integration/            # Integration + PyTorch comparison
│   └── package_smoke/          # Package smoke tests
├── benchmarks/                 # Google Benchmark
├── examples/                   # Usage examples
├── docs/                       # VitePress documentation
└── cmake/                      # CMake modules
```

---

## API Architecture

**Dual API design:** C++ namespace + C ABI for Python ctypes

- `cuflash::flash_attention_forward()` / `cuflash::flash_attention_backward()` - C++ API
- `cuflash_attention_forward_f32()` / `cuflash_attention_forward_f16()` - C ABI

**Supported types:** FP32 (`float`) and FP16 (`half`)

**Key constraints:**
- `head_dim` must be 32, 64, or 128 (kernel optimization requirement)
- Layout: [batch_size, num_heads, seq_len, head_dim]
- Returns `FlashAttentionError` enum for error handling

---

## Code Quality Commands

```bash
# Format (REQUIRED before commit - enforced in CI)
find . -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" | xargs clang-format -i

# clang-tidy (optional)
clang-tidy src/api/flash_attention_api.cu -- -Iinclude
```

**CI enforces:** clang-format v17 with LLVM style

---

## Testing Notes

- **Unit tests:** Google Test framework
- **PyTorch comparison:** Requires `pip install -r requirements-dev.txt` and GPU
- **Test discovery:** Uses `gtest_discover_tests` with 60s timeout
- **Package smoke test:** Tests that installed library can be found via CMake

---

## Documentation

Built with VitePress (Node.js 18+):

```bash
cd docs
npm ci
npm run docs:build
npm run docs:dev     # Dev server
```

Published to GitHub Pages at `https://lessup.github.io/cuflash-attn/`

---

## Anti-Patterns to Avoid

| ❌ DON'T | ✅ DO |
|----------|-------|
| Write code without reading specs | Read `/specs/` first |
| Add features not in specs | Propose spec updates before code |
| Skip formatting before commit | Run clang-format |
| Invent API designs | Follow `/specs/api/` exactly |
| Write tests without spec references | Reference spec IDs in comments |
| Build without preset | Use `cmake --preset release` |

---

## Quick Reference

```bash
# Full development cycle
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
find . -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" | xargs clang-format -i

# Benchmarks
./build/release/benchmarks/cuflash_attn_bench

# Example
./build/release/example
```

---

## When in Doubt

1. Check `/specs/` for the relevant document
2. If spec unclear or missing, **ask user** - don't assume
3. Follow existing patterns in `/src/` and `/tests/`
