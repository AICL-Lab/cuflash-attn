# CuFlash-Attn Documentation

Complete documentation for CuFlash-Attn — a CUDA C++ implementation of FlashAttention from scratch.

---

## 📚 Documentation Sections

### Getting Started

- [**Build Guide**](building.md) — Installation, build configuration, and environment setup
- [**Quick Start**](#quick-start) — Get up and running in minutes

### Core Documentation

- [**API Reference**](api-reference.md) — Complete C++ and C ABI API documentation
- [**Algorithm Deep Dive**](algorithm.md) — Understanding FlashAttention implementation
- [**Troubleshooting**](troubleshooting.md) — Common issues and solutions

### Additional Resources

- [GitHub Repository](https://github.com/LessUp/cuflash-attn)
- [Changelog](../../CHANGELOG.md)
- [中文文档](../zh/README.md) — 中文文档

---

## Quick Start

```cpp
#include "flash_attention.h"

// Forward pass
auto err = cuflash::flash_attention_forward(
    d_Q, d_K, d_V, d_O, d_L,
    batch_size, num_heads, seq_len, head_dim,
    scale, causal, stream
);

if (err != cuflash::FlashAttentionError::SUCCESS) {
    std::cerr << cuflash::get_error_string(err) << std::endl;
}
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| ⚡ **O(N) Memory** | Linear memory complexity instead of O(N²) |
| 🔢 **FP32 & FP16** | Full support for both precision types |
| 🔁 **Forward & Backward** | Complete training support |
| 🎭 **Causal Masking** | Autoregressive model support |
| 🔧 **C++ & C ABI** | Easy Python integration via ctypes |
| 🏎️ **CUDA Optimized** | Multi-architecture support (sm_70 - sm_90) |

---

## Architecture Support

| GPU Architecture | Compute Capability | Support Status |
|-----------------|-------------------|----------------|
| Volta (V100) | sm_70 | ✅ |
| Turing (RTX 2080 Ti) | sm_75 | ✅ |
| Ampere (A100) | sm_80 | ✅ |
| Ampere (RTX 3090) | sm_86 | ✅ |
| Ada Lovelace (RTX 4090) | sm_89 | ✅ |
| Hopper (H100) | sm_90 | ✅ |

---

## Requirements

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compatible compiler

---

## Language

- [English](README.md) (current)
- [简体中文](../zh/README.md)

---

## License

MIT License — See [LICENSE](../../LICENSE) for details.
