# Quick Start

Get CuFlash-Attn up and running in minutes.

## Prerequisites

- NVIDIA GPU with Compute Capability 7.0+ (V100, RTX 20/30/40 series, A100, H100)
- CUDA Toolkit 11.0 or later
- CMake 3.18 or later
- C++17 compatible compiler

## Installation

### Clone the Repository

```bash
git clone https://github.com/LessUp/cuflash-attn.git
cd cuflash-attn
```

### Build with CMake Presets

```bash
# Configure (Release build)
cmake --preset release

# Build
cmake --build --preset release

# Run tests
ctest --preset release --output-on-failure
```

### Custom Preset Overrides

```bash
# Release build with a custom architecture target
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build --preset release -j$(nproc)
```

## Your First Program

```cpp
#include <cuda_runtime.h>
#include "cuflash/flash_attention.h"
#include <iostream>
#include <cmath>

int main() {
    // Configuration
    const int B = 2, H = 8, N = 1024, D = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const bool causal = true;
    
    // Allocate and initialize tensors (simplified)
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, B * H * N * D * sizeof(float));
    cudaMalloc(&d_K, B * H * N * D * sizeof(float));
    cudaMalloc(&d_V, B * H * N * D * sizeof(float));
    cudaMalloc(&d_O, B * H * N * D * sizeof(float));
    cudaMalloc(&d_L, B * H * N * sizeof(float));
    
    // Initialize Q, K, V with your data...
    
    // Compute FlashAttention
    auto err = cuflash::flash_attention_forward(
        d_Q, d_K, d_V, d_O, d_L,
        B, H, N, D, scale, causal
    );
    
    if (err != cuflash::FlashAttentionError::SUCCESS) {
        std::cerr << "Error: " << cuflash::get_error_string(err) << std::endl;
        return 1;
    }
    
    // d_O now contains the output
    
    // Cleanup
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_L);
    
    return 0;
}
```

## Next Steps

- Learn about [API Reference](/en/api-reference)
- Read the [algorithm explanation](/en/algorithm)
- Check [building options](/en/building)
