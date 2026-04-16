# Build Guide

Complete guide for building CuFlash-Attn from source.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [CMake Presets](#cmake-presets)
- [Manual Build](#manual-build)
- [Build Options](#build-options)
- [Running Tests](#running-tests)
- [GPU Architecture Configuration](#gpu-architecture-configuration)
- [Cross-Platform Notes](#cross-platform-notes)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Dependency | Minimum Version | Notes |
|------------|-----------------|-------|
| **CUDA Toolkit** | 11.0 | Includes nvcc compiler and CUDA libraries |
| **CMake** | 3.18 | Build system generator |
| **C++ Compiler** | C++17 | GCC 7+, Clang 5+, MSVC 2017+ |
| **Python** (optional) | 3.8+ | For PyTorch comparison tests |
| **PyTorch** (optional) | 2.0+ | For verification tests |

### Verifying Prerequisites

```bash
# Check CUDA version
nvcc --version

# Check CMake version
cmake --version

# Check C++ compiler version (Linux)
g++ --version
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/LessUp/cuflash-attn.git
cd cuflash-attn

# Build using preset (recommended)
cmake --preset release
cmake --build --preset release

# Run tests
ctest --preset release --output-on-failure
```

---

## CMake Presets

The project provides predefined CMake presets for common build configurations:

| Preset | Build Type | Tests | Purpose |
|--------|-----------|-------|---------|
| `default` | Debug | ✅ | Development and debugging |
| `release` | Release | ✅ | Production use |
| `release-fast-math` | Release | ✅ | Maximum performance (reduced precision) |
| `minimal` | Release | ❌ | Minimal build size |

### Preset Build Commands

```bash
# Debug build (with tests and examples)
cmake --preset default
cmake --build --preset default

# Optimized release build
cmake --preset release
cmake --build --preset release

# Release with fast math optimizations
cmake --preset release-fast-math
cmake --build --preset release-fast-math

# Minimal build (no tests, no examples)
cmake --preset minimal
cmake --build --preset minimal
```

---

## Manual Build

For custom configurations:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Specifying CUDA Path

If CMake cannot find CUDA, specify the path explicitly:

**Linux/macOS:**
```bash
cmake .. -DCUDAToolkit_ROOT=/usr/local/cuda \
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

**Windows (PowerShell):**
```powershell
cmake .. -DCUDAToolkit_ROOT="$env:CUDA_PATH" `
         -DCMAKE_CUDA_COMPILER="$env:CUDA_PATH\bin\nvcc.exe"
```

### Cross-Platform Library Extensions

| Platform | Shared Library Extension | Static Library Extension |
|----------|------------------------|-------------------------|
| Linux | `.so` | `.a` |
| macOS | `.dylib` | `.a` |
| Windows | `.dll` | `.lib` |

---

## Build Options

| CMake Option | Default | Description |
|--------------|---------|-------------|
| `BUILD_TESTS` | ON | Build GoogleTest test suite |
| `ENABLE_RAPIDCHECK` | OFF | Enable RapidCheck property-based tests |
| `BUILD_SHARED_LIBS` | ON | Build as shared library (`*.so`/`.dll`/`.dylib`) |
| `BUILD_EXAMPLES` | ON | Build example programs |
| `ENABLE_FAST_MATH` | OFF | Enable `--use_fast_math` compiler flag |

### ENABLE_FAST_MATH

Enables aggressive mathematical optimizations that trade precision for speed:

```bash
cmake .. -DENABLE_FAST_MATH=ON
```

**Effects:**
- Faster `expf()` and `logf()` operations
- Slightly reduced numerical precision
- Generally acceptable for deep learning training

### Example Configurations

```bash
# High-performance release build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DENABLE_FAST_MATH=ON \
         -DBUILD_SHARED_LIBS=OFF

# Debug build with all tests
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DENABLE_RAPIDCHECK=ON

# Static library only
cmake .. -DBUILD_SHARED_LIBS=OFF \
         -DBUILD_TESTS=OFF \
         -DBUILD_EXAMPLES=OFF
```

---

## Running Tests

### CTest (Recommended)

```bash
# Run all tests with preset
ctest --preset release --output-on-failure

# Run specific test
ctest --preset release -R ForwardTest

# Run with verbose output
ctest --preset release -V
```

### GoogleTest Direct

Tests are located in `build/<preset>/tests/`:

```bash
# Run all tests
./build/release/tests/cuflash_attn_tests

# Run specific test suite
./build/release/tests/cuflash_attn_tests --gtest_filter="ForwardTest*"

# List all available tests
./build/release/tests/cuflash_attn_tests --gtest_list_tests
```

### PyTorch Comparison Tests

Verify numerical correctness against PyTorch reference:

```bash
# Build shared library first
cmake --preset release

# Run comparison tests
python tests/test_pytorch_comparison.py
```

**Library Path Resolution:**
1. Environment variable `CUFLASH_LIB`
2. `build/default/` directory
3. `build/release/` directory

**Custom Library Path:**
```bash
CUFLASH_LIB=/path/to/libcuflash_attn.so python tests/test_pytorch_comparison.py
```

---

## GPU Architecture Configuration

### Default Supported Architectures

| Compute Capability | Architecture | Representative GPUs |
|-------------------|--------------|---------------------|
| sm_70 | Volta | V100 |
| sm_75 | Turing | RTX 2080 Ti |
| sm_80 | Ampere | A100 |
| sm_86 | Ampere | RTX 3090 |
| sm_89 | Ada Lovelace | RTX 4090 |
| sm_90 | Hopper | H100 |

### Targeting Specific Architectures

```bash
# Single architecture (faster compilation)
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86  # RTX 3090/A100

# Multiple architectures
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86;89"

# Architecture ranges
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80-virtual"  # Virtual architecture
```

### Architecture Selection Guide

| Use Case | Recommended Configuration | Reason |
|----------|---------------------------|--------|
| Development | `-DCMAKE_CUDA_ARCHITECTURES=86` | Faster compilation |
| A100 Cluster | `-DCMAKE_CUDA_ARCHITECTURES=80` | Optimal for target |
| H100 Cluster | `-DCMAKE_CUDA_ARCHITECTURES=90` | Optimal for target |
| Public Release | Default (all architectures) | Maximum compatibility |

### Checking Your GPU Architecture

```bash
# Using nvidia-smi (shows current GPU)
nvidia-smi

# Using deviceQuery (sample from CUDA SDK)
/usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# Using nvcc
nvcc -arch=sm_70 --run - <<< '__global__ void k(){} int main(){k<<<1,1>>>();}'
```

---

## Cross-Platform Notes

### Linux

Standard workflow works out of the box:

```bash
sudo apt-get install cmake g++  # Ubuntu/Debian
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### macOS

CUDA support on macOS is limited to older versions. NVIDIA no longer provides macOS drivers for newer GPUs.

### Windows

**Using Visual Studio 2019/2022:**

```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

**Using Ninja (faster):**

```cmd
mkdir build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

**Common Windows Issues:**
- Ensure CUDA bin directory is in PATH
- Use x64 Native Tools Command Prompt for Visual Studio

### Docker

```dockerfile
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y cmake g++ git

WORKDIR /workspace
COPY . .

RUN cmake --preset release && \
    cmake --build --preset release
```

Run with:
```bash
docker build -t cuflash-attn .
docker run --gpus all cuflash-attn ./build/release/tests/cuflash_attn_tests
```

---

## Troubleshooting

### CMake Cannot Find CUDA

```bash
# Explicitly set CUDA paths
cmake .. -DCUDAToolkit_ROOT=/usr/local/cuda \
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### Compilation Errors with Unknown Architecture

Error: `Unknown CUDA architecture sm_XX`

**Solution:** Update CUDA toolkit or specify supported architecture:
```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES="70;75;80"
```

### Link Errors for Shared Library

Ensure `LD_LIBRARY_PATH` includes the build directory:
```bash
export LD_LIBRARY_PATH=$PWD/build/release:$LD_LIBRARY_PATH
```

### Out of Memory During Build

Reduce parallel jobs:
```bash
cmake --build . -j2  # Use 2 parallel jobs instead of all cores
```

### Test Failures on GPU-less Systems

The CI workflow is configured to run only format checks when no GPU is available. To run tests locally, ensure you have a CUDA-capable GPU.

---

## Next Steps

- Read the [API Reference](api-reference.md) for usage examples
- Explore the [Algorithm Documentation](algorithm.md) for implementation details
- Check [Troubleshooting](troubleshooting.md) for common issues
