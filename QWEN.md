# QWEN.md - CuFlash-Attn Project Context

## Project Overview

**CuFlash-Attn** is a high-performance CUDA C++ implementation of the FlashAttention algorithm, built from scratch. The project provides an efficient, IO-aware attention implementation with O(N) memory complexity (compared to O(N²) in standard attention), supporting both FP32 and FP16 precision for training and inference.

### Key Characteristics

- **Purpose**: Educational resource, research experimentation, and production integration
- **Core Algorithm**: FlashAttention with tiling strategy and online softmax
- **Memory Complexity**: O(N) linear vs O(N²) quadratic
- **Precision**: FP32 (float) and FP16 (half) for forward and backward passes
- **Features**: Causal masking, multi-GPU architecture support (sm_70 to sm_90)
- **License**: MIT
- **Version**: 0.1.0

### Target GPUs

Supports NVIDIA GPUs from V100 (sm_70) to H100 (sm_90), including:
- Volta (sm_70): V100
- Turing (sm_75): RTX 2080 Ti
- Ampere (sm_80, sm_86): A100, RTX 3090
- Ada Lovelace (sm_89): RTX 4090
- Hopper (sm_90): H100

**Default build targets**: sm_80, sm_86 (A100 + RTX 30xx/40xx)

---

## Technology Stack

| Category | Technology |
|----------|-----------|
| **Languages** | CUDA C++ (C++17, CUDA C++17) |
| **Build System** | CMake 3.18+ with Ninja generator |
| **Testing** | Google Test (gtest), CTest |
| **Benchmarking** | Google Benchmark |
| **Documentation** | VitePress (Vue-based static site generator) |
| **CI/CD** | GitHub Actions (matrix builds, CodeQL, deployment) |
| **Code Quality** | clang-format, clang-tidy, AddressSanitizer, UBSan |
| **Package Management** | CMake FetchContent (gtest, benchmark, rapidcheck) |

---

## Project Structure

```
cuflash-attn/
├── .github/workflows/          # CI/CD pipelines
│   ├── ci.yml                  # Matrix builds (CUDA 12.2/12.4, Release/Debug)
│   ├── codeql.yml              # Weekly security scanning
│   ├── pages.yml               # Docs deployment to GitHub Pages
│   └── release.yml             # Release automation
├── .vitepress/                 # VitePress theme and config
│   ├── config.js               # Documentation site configuration
│   └── theme/                  # Custom Vue components and CSS
├── benchmarks/                 # Performance benchmarks
│   └── bench_flash_attention.cu # Google Benchmark integration
├── cmake/                      # CMake modules
│   ├── cuflash_attnConfig.cmake.in  # Package config template
│   └── run_package_smoke.cmake      # Package smoke test script
├── docs/                       # Documentation site (VitePress)
│   ├── en/                     # English documentation
│   ├── zh/                     # Chinese documentation
│   ├── public/                 # Static assets (logos, favicons)
│   └── package.json            # Node.js dependencies for docs
├── examples/                   # Usage examples
│   └── basic_usage.cu          # Simple demonstration
├── include/cuflash/            # Public API headers
│   ├── flash_attention.h       # Main API (C++ and C ABI)
│   ├── export.h                # Visibility macros (CUFLASH_EXPORT)
│   └── version.h.in            # Version header template (auto-generated)
├── specs/                      # Spec-Driven Development documents
│   ├── product/                # Product requirements
│   ├── rfc/                    # Technical design (RFCs)
│   ├── api/                    # API specifications
│   ├── testing/                # Testing specifications
│   └── index.md                # Specifications overview
├── src/                        # Implementation
│   ├── api/                    # API dispatch layer
│   │   └── flash_attention_api.cu
│   ├── forward/                # Forward kernels
│   │   ├── flash_attention_forward.cu   # FP32 forward
│   │   └── flash_attention_fp16.cu      # FP16 forward
│   ├── backward/               # Backward kernels
│   │   ├── flash_attention_backward.cu   # FP32 backward
│   │   └── flash_attention_backward_fp16.cu  # FP16 backward
│   └── kernels/                # Internal utilities (.cuh)
│       ├── kernel_launch_utils.cuh
│       ├── matmul.cuh
│       ├── online_softmax.cuh
│       └── workspace_utils.cuh
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests (8 files)
│   ├── integration/            # Integration tests + PyTorch comparison
│   └── package_smoke/          # Package smoke tests
├── CMakeLists.txt              # Main build configuration
├── CMakePresets.json           # Build presets (6 presets)
├── AGENTS.md                   # AI workflow instructions (SDD)
├── CONTRIBUTING.md             # Contribution guidelines
├── .clang-format               # Code formatting rules
├── .clang-tidy                 # Static analysis configuration
└── .gitignore                  # Git ignore patterns
```

---

## Building and Running

### Prerequisites

- NVIDIA GPU with Compute Capability 7.0+ (V100 or newer recommended)
- CUDA Toolkit 11.0+ (tested with 12.4.1)
- CMake 3.18+
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)

### Quick Build

```bash
# Configure and build (Release mode)
cmake --preset release
cmake --build --preset release

# Run tests
ctest --preset release --output-on-failure
```

### Available Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `release` | Release build with tests | Default development |
| `default` | Debug build with tests | Debugging |
| `minimal` | Release, no tests/examples | Quick validation |
| `release-fast-math` | Release with --use_fast_math | Performance testing |
| `debug-asan` | Debug with AddressSanitizer | Memory debugging |
| `all-architectures` | All GPU architectures (sm_70-90) | Distribution builds |

### Build Examples

```bash
# Debug build
cmake --preset default
cmake --build --preset default

# Build with all GPU architectures
cmake --preset all-architectures
cmake --build --preset all-architectures

# Minimal build (no tests)
cmake --preset minimal
cmake --build --preset minimal

# Build with AddressSanitizer
cmake --preset debug-asan
cmake --build --preset debug-asan
ctest --preset debug-asan
```

### Custom CUDA Architecture

```bash
# Build for specific architecture only
cmake -B build-custom -DCMAKE_CUDA_ARCHITECTURES="90"
cmake --build build-custom
```

---

## API Usage

### C++ API

```cpp
#include "cuflash/flash_attention.h"

// Forward pass (FP32)
auto err = cuflash::flash_attention_forward(
    d_Q, d_K, d_V,  // Input tensors [B, H, N, D]
    d_O, d_L,       // Output and logsumexp
    batch_size, num_heads, seq_len, head_dim,
    scale,          // 1.0f / sqrt(head_dim)
    causal,         // Enable causal masking
    stream          // CUDA stream (optional)
);

// Backward pass (FP32)
err = cuflash::flash_attention_backward(
    d_Q, d_K, d_V, d_O, d_L, d_dO,
    d_dQ, d_dK, d_dV,
    batch_size, num_heads, seq_len, head_dim,
    scale, causal, stream
);
```

### C ABI (for Python ctypes)

```c
// Returns integer error code (0 = success)
int err = cuflash_attention_forward_f32(
    d_Q, d_K, d_V, d_O, d_L,
    batch_size, num_heads, seq_len, head_dim,
    scale, causal, stream
);
```

### Supported Parameters

| Parameter | Values | Notes |
|-----------|--------|-------|
| `head_dim` | 32, 64, 128 | Required for kernel optimization |
| Data Types | `float` (FP32), `half` (FP16) | Both forward & backward |
| Causal Mask | Optional | Runtime enable/disable |
| Batch Size | ≥ 1 | Any positive integer |
| Sequence Length | ≥ 1 | Optimized for 1K-16K+ |
| Number of Heads | ≥ 1 | Any positive integer |

---

## Testing

### Run Tests

```bash
# All tests
ctest --preset release --output-on-failure

# Specific test categories
ctest --preset release -R ForwardTest    # Forward pass tests
ctest --preset release -R BackwardTest   # Backward pass tests
ctest --preset release -R StressTest     # Stress & edge cases
ctest --preset release -R PyTorch        # PyTorch comparison (requires GPU + PyTorch)
```

### Test Categories

- **Unit Tests**: Forward, backward, online softmax, causal mask, error handling, dtype, numerical stability, stress/edge cases
- **Integration Tests**: API smoke test, PyTorch numerical comparison
- **Package Tests**: Build and install smoke tests

### PyTorch Comparison

```bash
# Requires Python 3.8+ and PyTorch
pip install -r tests/integration/requirements.txt
python tests/integration/test_pytorch_comparison.py
```

---

## Code Quality Tools

### Formatting

```bash
# Format all C/C++/CUDA files
find . -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" | xargs clang-format -i
```

- **clang-format**: LLVM style, enforced in CI
- **clang-tidy**: 50+ static analysis checks configured

### Static Analysis

```bash
# Run clang-tidy on specific file
clang-tidy src/api/flash_attention_api.cu -- -Iinclude -Ibuild/include
```

### Sanitizers

```bash
# AddressSanitizer (memory errors, leaks)
cmake --preset debug-asan
cmake --build --preset debug-asan
ctest --preset debug-asan
```

---

## Documentation

### Build Documentation

```bash
cd docs
npm ci
npm run docs:build
npm run docs:dev  # Local development server
```

### Documentation Links

- **English**: https://lessup.github.io/cuflash-attn/
- **Chinese**: https://lessup.github.io/cuflash-attn/zh/
- **API Reference**: https://lessup.github.io/cuflash-attn/en/api-reference

---

## Spec-Driven Development (SDD)

This project strictly follows **Spec-Driven Development** methodology. The `/specs/` directory serves as the **Single Source of Truth** for all implementation details.

### Specification Structure

| Directory | Purpose |
|-----------|---------|
| `/specs/product/` | Product requirements and acceptance criteria |
| `/specs/rfc/` | Technical design documents (RFCs) |
| `/specs/api/` | API interface specifications |
| `/specs/testing/` | Testing specifications and requirements |

### Workflow for Contributors

1. **Read specs first** - Always review relevant spec documents before implementing
2. **Spec-first updates** - Propose spec changes before code changes
3. **Implementation** - Code must exactly match spec definitions
4. **Test against spec** - Validate acceptance criteria from specs

See [AGENTS.md](AGENTS.md) for detailed AI workflow instructions.

---

## CI/CD Workflows

### GitHub Actions

| Workflow | Trigger | Description |
|----------|---------|-------------|
| `ci.yml` | Push/PR to master | Matrix builds (CUDA 12.2/12.4, Release/Debug), tests, format check, docs build |
| `codeql.yml` | Weekly (Mon 6AM) + Push/PR | Security scanning for C++ and JavaScript |
| `pages.yml` | Push to master | Build and deploy documentation to GitHub Pages |
| `release.yml` | Tag push (v*) | Create GitHub release with build artifacts |

### Caching

CI uses CMake dependency caching to speed up builds by 40%+. Cache keys are based on OS, CUDA version, build type, and CMakeLists.txt hash.

---

## Development Conventions

### Coding Style

- **Language**: C++17 with CUDA C++ extensions
- **Formatting**: LLVM style via clang-format
- **Naming**: Follows clang-tidy conventions
  - Namespaces: `lower_case`
  - Classes: `CamelCase`
  - Functions: `lower_case`
  - Constants: `UPPER_CASE`
- **Visibility**: Hidden by default, exported via `CUFLASH_EXPORT` macro

### Commit Messages

Follow conventional commits:

```
type(scope): description

Examples:
feat(api): add FP16 backward support
fix(kernel): correct causal mask boundary condition
docs(guide): update installation instructions
test(backward): add gradient check for edge cases
```

### Contribution Guidelines

1. Fork and create a branch
2. Read relevant specs
3. Write tests for new functionality
4. Implement changes
5. Format code with clang-format
6. Run tests locally
7. Submit pull request with clear description

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## Performance Characteristics

### Memory Complexity

| Method | Forward Memory | Backward Memory |
|--------|----------------|-----------------|
| Standard Attention | O(N²) | O(N²) |
| **FlashAttention** | **O(N)** | **O(N)** |

### Real-World Savings

| Sequence Length | Standard | FlashAttention | Savings |
|-----------------|----------|----------------|---------|
| 1,024 | 4 MB | 8 KB | 99.8% |
| 4,096 | 64 MB | 32 KB | 99.95% |
| 16,384 | 1 GB | 128 KB | 99.99% |

### Running Benchmarks

```bash
cmake --preset release
cmake --build --preset release
./build/release/benchmarks/cuflash_attn_bench
```

---

## Common Commands Reference

| Task | Command |
|------|---------|
| **Build (Release)** | `cmake --preset release && cmake --build --preset release` |
| **Run Tests** | `ctest --preset release --output-on-failure` |
| **Format Code** | `find . -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" \| xargs clang-format -i` |
| **Build Docs** | `cd docs && npm ci && npm run docs:build` |
| **Run Benchmarks** | `./build/release/benchmarks/cuflash_attn_bench` |
| **Clean Build** | `rm -rf build/` |

---

## Key Files

| File | Purpose |
|------|---------|
| `include/cuflash/flash_attention.h` | Main public API header (C++ and C ABI) |
| `CMakeLists.txt` | Build configuration and target definitions |
| `CMakePresets.json` | Predefined build configurations |
| `specs/index.md` | Specifications overview and navigation |
| `.github/workflows/ci.yml` | CI pipeline definition |
| `.vitepress/config.js` | Documentation site configuration |
| `AGENTS.md` | AI contributor workflow (SDD instructions) |
| `CONTRIBUTING.md` | Human contributor guidelines |

---

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao et al., NeurIPS 2022
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao, ICLR 2024
