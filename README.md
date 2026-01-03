# CuFlash-Attn

A high-performance FlashAttention implementation in CUDA C++ from scratch.

## Features

- **Forward Pass**: Efficient attention computation with O(N) memory complexity
- **Backward Pass**: Gradient computation using recomputation strategy
- **Causal Masking**: Support for autoregressive models
- **FP32 & FP16**: Support for both single and half precision
- **Online Softmax**: Numerically stable softmax without storing O(N²) attention matrix

## Requirements

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compatible compiler
- (Optional) PyTorch for comparison tests

## Building

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build Options

- `BUILD_TESTS=ON/OFF`: Build test suite (default: ON)

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

- **head_dim**: 32, 64, 128
- **Data types**: float32, float16
- **Causal masking**: Optional

## Running Tests

```bash
cd build
ctest --output-on-failure
```

### PyTorch Comparison Tests

```bash
python tests/test_pytorch_comparison.py
```


## Algorithm

This implementation follows the FlashAttention algorithm:

1. **Tiling**: Divide Q, K, V into blocks that fit in SRAM
2. **Online Softmax**: Compute softmax incrementally without storing full attention matrix
3. **Recomputation**: In backward pass, recompute attention weights instead of storing

### Memory Complexity

- Standard Attention: O(N²) for attention matrix
- FlashAttention: O(N) - only stores output and logsumexp

## Project Structure

```
├── include/
│   └── flash_attention.h      # Public API header
├── src/
│   ├── flash_attention_api.cu     # API implementation
│   ├── flash_attention_forward.cu # Forward kernel
│   ├── flash_attention_backward.cu# Backward kernel
│   ├── flash_attention_fp16.cu    # FP16 support
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
└── README.md
```

## Error Handling

```cpp
cuflash::FlashAttentionError err = cuflash::flash_attention_forward(...);
if (err != cuflash::FlashAttentionError::SUCCESS) {
    std::cerr << cuflash::get_error_string(err) << std::endl;
}
```

### Error Codes

- `SUCCESS`: Operation completed successfully
- `INVALID_DIMENSION`: Dimension parameters are invalid
- `NULL_POINTER`: Input or output pointer is null
- `UNSUPPORTED_HEAD_DIM`: head_dim must be 32, 64, or 128
- `CUDA_ERROR`: CUDA runtime error occurred

## License

MIT License

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
