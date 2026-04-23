# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### 🔧 Chore | 工程整治

#### CI/CD
- Fix `pages.yml`: remove non-existent root `package.json`/`package-lock.json` triggers
- Fix `release.yml`: unify to `cmake --preset release` (remove redundant `-B` flag)
- Fix `docs/.vitepress/config.js`: correct "Specs" nav links from `specs/` to `openspec/specs`

#### Tooling
- Add `.clangd` LSP configuration (CUDA paths, sm_86, diagnostics)
- Add `CMAKE_EXPORT_COMPILE_COMMANDS=ON` for LSP `compile_commands.json` generation
- Update `.vscode/settings.json` with full clangd/CUDA development settings
- Add `.github/copilot-instructions.md` (project-level Copilot instructions, Chinese responses)
- Add `.github/pull_request_template.md`
- Add `.github/ISSUE_TEMPLATE/bug_report.md` (CUDA-specific bug template)

#### Documentation
- Rewrite `AGENTS.md`: high-density CUDA traps, build commands, tool collaboration
- Rewrite `openspec/config.yaml`: Google style (not LLVM), branch strategy, AI tools
- Fix `docs/en/index.md` and `docs/zh/index.md`: update broken `specs/` links to `openspec/specs/`
- Rewrite `CONTRIBUTING.md`: from generic template to project-specific 40-line guide

#### Cleanup
- Delete `QWEN.md` (stale v0.1.0, wrong paths)
- Delete `specs.archived/` (old spec structure, superseded by `openspec/specs/`)
- Delete `CODE_OF_CONDUCT.md` (generic template, no value)

---

## [0.2.0] - 2026-04-16

### 🚀 Highlights | 亮点

This release introduces complete FP16 backward pass support and a thoroughly restructured bilingual documentation system.

本版本引入完整的 FP16 反向传播支持和全面重构的双语文档系统。

### ✨ Added | 新增

#### Features | 功能
- **FP16 Backward Pass** (`src/flash_attention_backward_fp16.cu`)
  - Complete FP16 gradient computation
  - Numerical stability through FP32 internal accumulation
  - C ABI interface for Python ctypes integration
  - Full test coverage

- **Bilingual Documentation System** | 双语文档系统
  - Restructured `docs/en/` and `docs/zh/` directories
  - Professional English API reference with comprehensive examples
  - Detailed algorithm deep dive in both languages
  - New troubleshooting guides for common issues
  - Complete build guides with cross-platform instructions

#### Documentation | 文档
- **API Reference (English)** (`docs/en/api-reference.md`)
  - Complete C++ and C ABI documentation
  - Tensor layout specifications with offset calculations
  - Error handling examples and best practices
  - Thread safety and memory management guidelines
  - GPU architecture support matrix

- **Algorithm Documentation** (`docs/en/algorithm.md`, `docs/zh/algorithm.md`)
  - Standard attention bottleneck explanation
  - Core FlashAttention concepts (tiling, online softmax, recomputation)
  - Step-by-step forward and backward algorithms
  - Causal masking strategy with efficiency analysis
  - FP16 implementation details and precision handling
  - Memory complexity analysis with real-world comparisons

- **Build Guide** (`docs/en/building.md`, `docs/zh/building.md`)
  - CMake presets for common configurations
  - Cross-platform build instructions (Linux, Windows, Docker)
  - GPU architecture targeting guide
  - Troubleshooting common build issues

- **Troubleshooting Guide** (`docs/en/troubleshooting.md`, `docs/zh/troubleshooting.md`)
  - Build issue resolution
  - Runtime error diagnosis
  - Performance optimization tips
  - Numerical accuracy considerations
  - Complete error code reference

### 🔧 Changed | 变更

- **Project Documentation Restructure**
  - Migrated all docs to language-specific subdirectories
  - Updated HonKit configuration for bilingual support
  - Enhanced navigation and cross-references

- **README Updates**
  - Professionalized English README with badges and clear structure
  - Synchronized Chinese README with consistent formatting
  - Added quick start examples and feature highlights
  - Improved project structure visualization

### 🐛 Fixed | 修复

- None in this release (all fixes included in previous versions)

### 🗑️ Removed | 移除

- Deleted legacy `docs/*.md` files (migrated to new structure)
- Removed unused `src/cuflash_ctypes_api.cu` (duplicate C ABI)

---

## [0.1.0] - 2026-03-13

### ✨ Added | 新增

- **Complete Documentation Suite**
  - API reference documentation (`docs/api.md`)
  - FlashAttention algorithm deep dive (`docs/algorithm.md`)
  - Build guide with CMake instructions (`docs/building.md`)
  - HonKit documentation site with Chinese search support

### 🔧 Changed | 变更

- **CI Workflow Improvement**
  - Switched to CPU-safe mode for CI (format checking only)
  - Reason: GitHub Hosted Runners don't provide GPU, CUDA builds were failing
  - Now runs clang-format static checks only

### 🔨 CI/CD | 持续集成

- Unified GitHub Actions workflow naming conventions
- Added `permissions` and `concurrency` configurations
- Pages workflow with `paths` filtering and sparse-checkout

---

## [0.1.0-alpha.2] - 2026-03-10

### ✨ Added | 新增

- **Standardized CI Workflow** (`.github/workflows/ci.yml`)
  - Support for `push`, `pull_request`, `workflow_dispatch` triggers
  - Added clang-format code format checking

### 🔧 Changed | 变更

- Renamed Pages workflow: `docs.yml` → `pages.yml`

---

## [0.1.0-alpha.1] - 2026-02-13

### ✨ Added | 新增

- **FP16 Forward Pass** (`src/flash_attention_fp16.cu`)
  - FP16 forward kernel implementation
  - FP16 API integration with full `half` type support

### ⚡ Performance Optimizations | 性能优化

- **Vectorized Memory Access**
  - `float4` vectorized loads/stores for improved global memory bandwidth
  
- **Launch Bounds**
  - Added `__launch_bounds__(128)` to all CUDA kernels for occupancy optimization
  
- **Fast Math Option**
  - Optional `--use_fast_math` compiler flag for faster execution

### 🐛 Bug Fixes | 错误修复

**Stream Safety in Backward Pass:**
- Changed `cudaMemset` to `cudaMemsetAsync` for in-stream ordering
- Added `cudaStreamSynchronize` to prevent premature workspace deallocation
- Added `cudaMalloc` return value checking

### 🏗️ Build Configuration | 构建配置

- Added SM 89 (Ada Lovelace, RTX 4090) support
- Added SM 90 (Hopper, H100) support

---

## Version Summary | 版本摘要

| Version | Key Features | Release Date |
|---------|--------------|--------------|
| [0.2.0] | FP16 backward, bilingual docs, troubleshooting guide | 2026-04-16 |
| [0.1.0] | Complete docs, CPU-safe CI, HonKit site | 2026-03-13 |
| [0.1.0-alpha.2] | Standardized CI, format checking | 2026-03-10 |
| [0.1.0-alpha.1] | FP16 forward, performance optimizations | 2026-02-13 |

---

## Release Links | 发布链接

- [v0.2.0](https://github.com/LessUp/cuflash-attn/releases/tag/v0.2.0)
- [v0.1.0](https://github.com/LessUp/cuflash-attn/releases/tag/v0.1.0)
- [v0.1.0-alpha.2](https://github.com/LessUp/cuflash-attn/releases/tag/v0.1.0-alpha.2)
- [v0.1.0-alpha.1](https://github.com/LessUp/cuflash-attn/releases/tag/v0.1.0-alpha.1)

---

[Unreleased]: https://github.com/LessUp/cuflash-attn/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/LessUp/cuflash-attn/releases/tag/v0.2.0
[0.1.0]: https://github.com/LessUp/cuflash-attn/releases/tag/v0.1.0
[0.1.0-alpha.2]: https://github.com/LessUp/cuflash-attn/releases/tag/v0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/LessUp/cuflash-attn/releases/tag/v0.1.0-alpha.1
