# CuFlash-Attn

[![CI](https://img.shields.io/github/actions/workflow/status/LessUp/cuflash-attn/ci.yml?branch=master&style=flat-square&logo=github&label=CI)](https://github.com/LessUp/cuflash-attn/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/LessUp/cuflash-attn/pages.yml?branch=master&style=flat-square&logo=githubpages&logoColor=white&label=Docs)](https://lessup.github.io/cuflash-attn/)

English | [简体中文](README.zh-CN.md)

A FlashAttention implementation in CUDA C++ from scratch. This is primarily a reference/educational implementation demonstrating the FlashAttention algorithm; for production workloads requiring maximum performance, consider using established libraries like FlashAttention-2.

## Features

- **Forward Pass**: Efficient attention computation with O(N) memory complexity (FP32 and FP16)
- **Backward Pass**: Gradient computation using recomputation strategy (FP32 and FP16)
- **Causal Masking**: Support for autoregressive models
- **Online Softmax**: Numerically stable softmax without storing O(N²) attention matrix

## Known Limitations

- **head_dim support**: Only 32, 64, and 128 are supported
- **High shared memory usage**: May require GPUs with extended shared memory support for head_dim=128
- **DIMENSION_MISMATCH error**: Currently not actively checked (API does not receive per-tensor shape metadata)

## Requirements

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compatible compiler
- (Optional) PyTorch for comparison tests

## Building

### Using CMake Presets (recommended)

```bash
cmake --preset default      # Debug build with tests
cmake --build --preset default
ctest --preset default

cmake --preset release      # Optimized build
cmake --build --preset release
```

### Manual build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

If CMake cannot find CUDA, configure it explicitly:

```bash
cmake .. -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | ON | Build test suite |
| `ENABLE_RAPIDCHECK` | OFF | Enable RapidCheck property tests |
| `BUILD_SHARED_LIBS` | ON | Build shared library |
| `BUILD_EXAMPLES` | ON | Build example binary |
| `ENABLE_FAST_MATH` | OFF | Enable `--use_fast_math` (faster but less precise) |

## Usage

### C++ API

```cpp
#include "flash_attention.h"

// Forward pass
cuflash::FlashAttentionError err = cuflash::flash_attention_forward(
    Q, K, V,           // Input tensors [batch, heads, seq_len, head_dim]
    O, L,              // Output tensor and logsumexp
    batch_size, num_heads, seq_len, head_dim,
    scale,             // Usually 1/sqrt(head_dim)
    causal,            // Enable causal masking
    stream             // CUDA stream (optional)
);

// Backward pass
err = cuflash::flash_attention_backward(
    Q, K, V, O, L, dO, // Inputs and upstream gradient
    dQ, dK, dV,        // Output gradients
    batch_size, num_heads, seq_len, head_dim,
    scale, causal, stream
);
```

### Supported Configurations

| Parameter | Supported Values |
|-----------|------------------|
| `head_dim` | 32, 64, 128 |
| Data types | `float` (FP32), `half` (FP16, both forward and backward) |
| Causal masking | Optional |

## Running Tests

```bash
ctest --preset default --output-on-failure
```

GoogleTest is automatically fetched via CMake FetchContent — no manual installation required.

### PyTorch Comparison Tests

```bash
python tests/test_pytorch_comparison.py
```

Build the shared library first. Preset builds place artifacts under `build/<preset>/`, for example `build/default/` or `build/release/`. You can also override the library path with `CUFLASH_LIB=/absolute/path/to/libcuflash_attn.so`.

## Algorithm

This implementation follows the FlashAttention algorithm:

1. **Tiling**: Divide Q, K, V into blocks that fit in SRAM
2. **Online Softmax**: Compute softmax incrementally without storing full attention matrix
3. **Recomputation**: In backward pass, recompute attention weights instead of storing

### Memory Complexity

| Method | Forward Memory | Backward Memory |
|--------|----------------|-----------------|
| Standard Attention | O(N²) | O(N²) |
| FlashAttention | O(N) | O(N) |

## Project Structure

```
├── include/
│   └── flash_attention.h          # Public API header
├── src/
│   ├── flash_attention_api.cu     # API implementation
│   ├── flash_attention_forward.cu # FP32 forward kernel
│   ├── flash_attention_backward.cu# FP32 backward kernel
│   ├── flash_attention_fp16.cu    # FP16 forward kernel
│   ├── flash_attention_backward_fp16.cu # FP16 backward kernel
│   ├── kernel_launch_utils.cuh    # Kernel launch utilities
│   ├── online_softmax.cuh         # Online softmax utilities
│   └── matmul.cuh                 # Matrix multiplication helpers
├── tests/
│   ├── test_forward.cu            # Forward pass tests
│   ├── test_backward.cu           # Backward pass tests
│   ├── test_causal_mask.cu        # Causal masking tests
│   ├── test_online_softmax.cu     # Online softmax tests
│   ├── test_error_handling.cu     # Error handling tests
│   ├── test_dtype.cu              # Data type tests
│   ├── test_numerical_stability.cu# Numerical stability tests
│   └── test_pytorch_comparison.py # PyTorch comparison
├── examples/
│   └── basic_usage.cu             # Usage example
├── CMakeLists.txt
└── CMakePresets.json              # Build presets
```

## Error Handling

```cpp
cuflash::FlashAttentionError err = cuflash::flash_attention_forward(...);
if (err != cuflash::FlashAttentionError::SUCCESS) {
    std::cerr << cuflash::get_error_string(err) << std::endl;
}
```

### Error Codes

| Error Code | Description |
|------------|-------------|
| `SUCCESS` | Operation completed successfully |
| `INVALID_DIMENSION` | Dimension parameters are invalid (≤ 0) |
| `DIMENSION_MISMATCH` | Reserved for future use |
| `NULL_POINTER` | Input or output pointer is null |
| `CUDA_ERROR` | CUDA runtime error occurred |
| `OUT_OF_MEMORY` | Insufficient GPU memory |
| `UNSUPPORTED_HEAD_DIM` | head_dim must be 32, 64, or 128 |
| `UNSUPPORTED_DTYPE` | Data type not supported for this operation |

## GPU Architecture Support

| Architecture | Compute Capability | Representative GPU |
|--------------|-------------------|-------------------|
| Volta | sm_70 | V100 |
| Turing | sm_75 | RTX 2080 Ti |
| Ampere | sm_80, sm_86 | A100, RTX 3090 |
| Ada Lovelace | sm_89 | RTX 4090 |
| Hopper | sm_90 | H100 |

## License

MIT License

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
