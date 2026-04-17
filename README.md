# CuFlash-Attn

> **High-performance CUDA C++ FlashAttention implementation from scratch**

[![CI](https://img.shields.io/github/actions/workflow/status/LessUp/cuflash-attn/ci.yml?branch=master&style=flat-square&logo=github&label=CI)](https://github.com/LessUp/cuflash-attn/actions/workflows/ci.yml)
[![CodeQL](https://img.shields.io/github/actions/workflow/status/LessUp/cuflash-attn/codeql.yml?branch=master&style=flat-square&logo=github&label=CodeQL)](https://github.com/LessUp/cuflash-attn/actions/workflows/codeql.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/LessUp/cuflash-attn/pages.yml?branch=master&style=flat-square&logo=githubpages&logoColor=white&label=Docs)](https://lessup.github.io/cuflash-attn/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/github/v/release/LessUp/cuflash-attn?style=flat-square&label=version)](https://github.com/LessUp/cuflash-attn/releases)

[English](README.md) · [简体中文](README.zh-CN.md) · [Documentation](https://lessup.github.io/cuflash-attn/) · [API Reference](https://lessup.github.io/cuflash-attn/en/api-reference)

---

## 🎯 Overview

CuFlash-Attn is a **from-scratch implementation** of the FlashAttention algorithm, optimized for **educational purposes**, **research experimentation**, and **production integration**.

### Why CuFlash-Attn?

| Challenge | Solution |
|-----------|----------|
| 📚 **Learn FlashAttention** | Clean, well-documented CUDA kernels with step-by-step algorithm implementation |
| 🔬 **Research & Experiment** | Modify attention mechanisms without complex framework dependencies |
| 🚀 **Production Ready** | C++ API with C ABI bindings for seamless Python integration via ctypes |
| ⚡ **GPU Optimized** | Multi-architecture support from V100 (sm_70) to H100 (sm_90) |

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| ⚡ **O(N) Memory** | Linear memory complexity vs O(N²) in standard attention — handle 16K+ sequences |
| 🔢 **Dual Precision** | FP32 & FP16 support for both forward and backward passes |
| 🔁 **Full Training** | Complete forward/backward with gradient computation |
| 🎭 **Causal Masking** | Built-in support for autoregressive models (GPT-style) |
| 🔧 **Easy Integration** | Clean C++ API + C ABI for Python ctypes integration |
| 🏎️ **Multi-Arch** | Optimized CUDA kernels for sm_70 → sm_90 (V100 → H100) |
| 🧪 **Comprehensive Tests** | Unit tests, integration tests, stress tests, PyTorch comparison |
| 📊 **Benchmarks** | Google Benchmark integration for performance tracking |
| 📚 **Bilingual Docs** | Complete English & Chinese documentation |

---

## 🚀 Quick Start

### Prerequisites

- **GPU**: NVIDIA GPU with Compute Capability 7.0+ (V100, RTX 20/30/40, A100, H100)
- **CUDA Toolkit**: 11.0 or later
- **CMake**: 3.18 or later
- **Compiler**: GCC 7+, Clang 5+, or MSVC 2017+ (C++17 support required)

### Installation

```bash
# Clone repository
git clone https://github.com/LessUp/cuflash-attn.git
cd cuflash-attn

# Build with preset (Release mode)
cmake --preset release
cmake --build --preset release

# Run tests
ctest --preset release --output-on-failure
```

### Your First Program

```cpp
#include <cuda_runtime.h>
#include "cuflash/flash_attention.h"
#include <iostream>

int main() {
    const int B = 2, H = 8, N = 1024, D = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, B * H * N * D * sizeof(float));
    cudaMalloc(&d_K, B * H * N * D * sizeof(float));
    cudaMalloc(&d_V, B * H * N * D * sizeof(float));
    cudaMalloc(&d_O, B * H * N * D * sizeof(float));
    cudaMalloc(&d_L, B * H * N * sizeof(float));
    
    // Initialize Q, K, V with your data...
    
    // Compute FlashAttention with causal masking
    auto err = cuflash::flash_attention_forward(
        d_Q, d_K, d_V, d_O, d_L,
        B, H, N, D, scale,
        true  // causal masking
    );
    
    if (err != cuflash::FlashAttentionError::SUCCESS) {
        std::cerr << "Error: " << cuflash::get_error_string(err) << std::endl;
        return 1;
    }
    
    // d_O now contains the attention output
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_L);
    return 0;
}
```

📖 **More examples**: See [examples/](examples/) directory for complete programs.

---

## 📊 Performance

### Memory Efficiency

| Sequence Length | Standard Attention | FlashAttention | **Savings** |
|----------------|-------------------|----------------|-------------|
| 1,024 | 4 MB | 8 KB | **99.8%** |
| 4,096 | 64 MB | 32 KB | **99.95%** |
| 16,384 | 1 GB | 128 KB | **99.99%** |

### Benchmark Results

Run performance benchmarks on your hardware:

```bash
cmake --preset release
cmake --build --preset release
./build/release/benchmarks/cuflash_attn_bench
```

See [benchmarks/](benchmarks/) for benchmark source code.

---

## 📖 Documentation

### Quick Links

| Resource | Link |
|----------|------|
| 📘 **Full Documentation** | [https://lessup.github.io/cuflash-attn/](https://lessup.github.io/cuflash-attn/) |
| 🔌 **API Reference** | [English API Docs](https://lessup.github.io/cuflash-attn/en/api-reference) |
| 🧠 **Algorithm Deep Dive** | [FlashAttention Explained](https://lessup.github.io/cuflash-attn/en/algorithm) |
| 🔧 **Build Guide** | [Building from Source](https://lessup.github.io/cuflash-attn/en/building) |
| ❓ **Troubleshooting** | [Common Issues & Solutions](https://lessup.github.io/cuflash-attn/en/troubleshooting) |

### Documentation Languages

- 🇬🇧 [English Documentation](https://lessup.github.io/cuflash-attn/)
- 🇨🇳 [中文文档](https://lessup.github.io/cuflash-attn/zh/)

---

## ⚙️ Configuration

### Supported Parameters

| Parameter | Values | Notes |
|-----------|--------|-------|
| `head_dim` | 32, 64, 128 | Required for kernel optimization |
| **Data Types** | FP32 (`float`), FP16 (`half`) | Both forward & backward |
| **Causal Mask** | Optional | Enabled/disabled at runtime |
| **Batch Size** | ≥ 1 | Any positive integer |
| **Sequence Length** | ≥ 1 | Optimized for 1K-16K+ |
| **Number of Heads** | ≥ 1 | Any positive integer |

### GPU Architecture Support

| Architecture | Compute | Example GPUs |
|--------------|---------|--------------|
| Volta | sm_70 | V100 |
| Turing | sm_75 | RTX 2080 Ti |
| Ampere | sm_80, sm_86 | A100, RTX 3090 |
| Ada Lovelace | sm_89 | RTX 4090 |
| Hopper | sm_90 | H100 |

**Default build targets**: sm_80, sm_86 (A100 + RTX 30xx/40xx)

Customize with: `cmake -B build -DCMAKE_CUDA_ARCHITECTURES="90"`

---

## 🏗️ Project Structure

```
cuflash-attn/
├── benchmarks/                 # Performance benchmarks (Google Benchmark)
├── cmake/                      # CMake modules & packaging
├── docs/                       # VitePress documentation site
│   ├── en/                     # English documentation
│   ├── zh/                     # Chinese documentation
│   └── public/                 # Static assets (logos, favicons)
├── examples/                   # Complete usage examples
├── include/cuflash/            # Public API headers
│   ├── flash_attention.h       # Main API with C++ and C ABI
│   ├── export.h                # Visibility macros
│   └── version.h.in            # Version header template
├── specs/                      # Spec-Driven Development documents
│   ├── product/                # Product requirements
│   ├── rfc/                    # Technical design (RFCs)
│   ├── api/                    # API specifications
│   └── testing/                # Testing specifications
├── src/                        # Implementation
│   ├── api/                    # API dispatch layer
│   ├── forward/                # Forward kernel implementations
│   ├── backward/               # Backward kernel implementations
│   └── kernels/                # Internal kernel utilities (.cuh)
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests (8 files)
│   ├── integration/            # Integration tests + PyTorch comparison
│   └── package_smoke/          # Package smoke tests
├── CMakeLists.txt              # Main build configuration
├── CMakePresets.json           # Build presets (release, debug, asan)
└── .github/workflows/          # CI/CD workflows
    ├── ci.yml                  # Matrix builds, tests, benchmarks
    ├── codeql.yml              # Security scanning
    ├── pages.yml               # Docs deployment
    └── release.yml             # Release automation
```

---

## 🧪 Testing & Quality

### Test Categories

```bash
# Run all tests
ctest --preset release --output-on-failure

# Run specific test categories
ctest --preset release -R ForwardTest    # Forward pass tests
ctest --preset release -R BackwardTest   # Backward pass tests
ctest --preset release -R StressTest     # Stress & edge cases
ctest --preset release -R PyTorch        # PyTorch comparison (requires GPU + PyTorch)
```

### Code Quality Tools

- ✅ **clang-format**: Automated code formatting (enforced in CI)
- ✅ **clang-tidy**: Static analysis with 50+ checks
- ✅ **CodeQL**: Weekly security scanning
- ✅ **Sanitizers**: AddressSanitizer & UBSan support (debug builds)

```bash
# Build with AddressSanitizer
cmake --preset debug-asan
cmake --build --preset debug-asan
ctest --preset debug-asan
```

---

## 🤝 Contributing

Contributions are welcome! This project follows **Spec-Driven Development (SDD)** methodology.

### Getting Started

1. **Read the specs first** 📖 — All requirements are in [/specs/](specs/)
2. **Fork & clone** the repository
3. **Create a branch** for your feature or fix
4. **Write tests** that validate your changes
5. **Update documentation** if API changes
6. **Submit a pull request**

### Development Workflow

```bash
# Format code before committing
find . -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" | xargs clang-format -i

# Run tests locally
cmake --preset release && cmake --build --preset release
ctest --preset release --output-on-failure

# Optional: Run clang-tidy
clang-tidy src/api/flash_attention_api.cu -- -Iinclude
```

📋 **Detailed guidelines**: See [CONTRIBUTING.md](CONTRIBUTING.md)

🤖 **For AI Contributors**: Read [AGENTS.md](AGENTS.md) for SDD workflow instructions.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📚 References

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao et al., NeurIPS 2022
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao, ICLR 2024

---

## 📈 Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and updates.

---

<p align="center">
  <sub>Built with ❤️ for efficient attention computation</sub><br>
  <sub>Spec-Driven Development · CUDA C++ · Open Source</sub>
</p>
