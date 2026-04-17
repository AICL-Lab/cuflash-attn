# CuFlash-Attn

[![CI](https://img.shields.io/github/actions/workflow/status/LessUp/cuflash-attn/ci.yml?branch=master&style=flat-square&logo=github&label=CI)](https://github.com/LessUp/cuflash-attn/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/LessUp/cuflash-attn/pages.yml?branch=master&style=flat-square&logo=githubpages&logoColor=white&label=Docs)](https://lessup.github.io/cuflash-attn/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](LICENSE)

English | [简体中文](README.zh-CN.md)

> A FlashAttention implementation in CUDA C++ from scratch — primarily for educational purposes and as a reference implementation.

CuFlash-Attn provides an efficient, IO-aware implementation of the FlashAttention algorithm with O(N) memory complexity and support for both FP32 and FP16 precision.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| ⚡ **O(N) Memory** | Linear memory complexity versus O(N²) in standard attention |
| 🔢 **Dual Precision** | Full FP32 and FP16 support for forward and backward passes |
| 🔁 **Complete Training** | Both forward and backward passes with gradient computation |
| 🎭 **Causal Masking** | Built-in support for autoregressive models |
| 🔧 **Easy Integration** | C++ and C ABI interfaces for Python via ctypes |
| 🏎️ **Multi-Arch CUDA** | Optimized for sm_70 through sm_90 (V100 to H100) |
| 📚 **Bilingual Docs** | Complete documentation in English and Chinese |

---

## 📋 Requirements

| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| CUDA Toolkit | 11.0 | Includes nvcc compiler |
| CMake | 3.18 | Build system |
| C++ Compiler | C++17 | GCC 7+, Clang 5+, or MSVC 2017+ |
| GPU | Compute Capability 7.0+ | V100 or newer recommended |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/LessUp/cuflash-attn.git
cd cuflash-attn

# Build with CMake preset (recommended)
cmake --preset release
cmake --build --preset release

# Run tests
ctest --preset release --output-on-failure
```

### Basic Usage

```cpp
#include "cuflash/flash_attention.h"

// Compute attention with causal masking
float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

auto err = cuflash::flash_attention_forward(
    d_Q, d_K, d_V,     // Input tensors [B, H, N, D]
    d_O, d_L,          // Output and logsumexp
    batch_size, num_heads, seq_len, head_dim,
    scale,             // Attention scale factor
    true,              // Enable causal masking
    stream             // CUDA stream (optional)
);

if (err != cuflash::FlashAttentionError::SUCCESS) {
    std::cerr << "Error: " << cuflash::get_error_string(err) << std::endl;
}
```

See the [examples](examples/) directory for complete examples.

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [English API Reference](docs/en/api-reference.md) | Complete C++ and C ABI API documentation |
| [算法详解 (Algorithm)](docs/zh/algorithm.md) | Deep dive into FlashAttention implementation |
| [构建指南 (Build Guide)](docs/zh/building.md) | Detailed build instructions |
| [故障排除 (Troubleshooting)](docs/zh/troubleshooting.md) | Common issues and solutions |

**Full documentation site:** [https://lessup.github.io/cuflash-attn/](https://lessup.github.io/cuflash-attn/)

---

## ⚙️ Supported Configurations

| Parameter | Supported Values |
|-----------|------------------|
| `head_dim` | 32, 64, 128 |
| Data Types | `float` (FP32), `half` (FP16) |
| Causal Masking | Optional |
| Batch Size | ≥ 1 |
| Sequence Length | ≥ 1 |
| Number of Heads | ≥ 1 |

### GPU Architecture Support

| Architecture | Compute | GPUs |
|--------------|---------|------|
| Volta | sm_70 | V100 |
| Turing | sm_75 | RTX 2080 Ti |
| Ampere | sm_80, sm_86 | A100, RTX 3090 |
| Ada Lovelace | sm_89 | RTX 4090 |
| Hopper | sm_90 | H100 |

---

## 🧪 Testing

```bash
# Run all tests
ctest --preset release --output-on-failure

# Run specific test
ctest --preset release -R ForwardTest

# PyTorch comparison
python tests/test_pytorch_comparison.py
```

---

## ⚡ Performance Characteristics

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

---

## 🏗️ Project Structure

```
├── specs/                        # Specification documents (Single Source of Truth)
│   ├── product/                  # Product requirements & acceptance criteria
│   ├── rfc/                      # Technical design documents (RFCs)
│   ├── api/                      # API specifications
│   └── testing/                  # Testing specifications
├── include/
│   └── flash_attention.h         # Public API header
├── src/
│   ├── flash_attention_api.cu    # API implementation
│   ├── flash_attention_forward.cu # FP32 forward kernel
│   ├── flash_attention_backward.cu # FP32 backward kernel
│   ├── flash_attention_fp16.cu   # FP16 forward kernel
│   └── flash_attention_backward_fp16.cu # FP16 backward
├── docs/                         # User-facing documentation
│   ├── setup/                    # Environment setup guides
│   ├── tutorials/                # Usage tutorials
│   ├── architecture/             # Architecture documentation
│   └── assets/                   # Static assets
├── tests/                        # Test suite
├── examples/                     # Usage examples
└── CMakePresets.json            # Build presets
```

**Specification Documents:**
- [Product Requirements](specs/product/001-flash-attention-core.md) - Feature definitions and acceptance criteria
- [Core Architecture RFC](specs/rfc/001-core-architecture.md) - Technical design and architecture
- [API Specification](specs/api/001-public-api.md) - Public API definitions
- [Testing Specification](specs/testing/001-test-specification.md) - Testing strategy and requirements

---

## 🤝 Contributing

Contributions are welcome! This project follows **Spec-Driven Development (SDD)** methodology. Please ensure:

1. **Read the specs first** - All implementation details are defined in `/specs/` directory
2. Code follows the existing style (run `clang-format`)
3. Tests pass with `ctest`
4. Documentation is updated for API changes

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on how to contribute, including how to work with specs.

**For AI Contributors:** Read [AGENTS.md](AGENTS.md) for the SDD workflow instructions.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📚 References

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (Dao et al., NeurIPS 2022)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (Dao, ICLR 2024)

---

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

---

<p align="center">
  <sub>Built with ❤️ for efficient attention computation</sub>
</p>
