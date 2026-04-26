# Troubleshooting Guide

Common issues and solutions when building and using CuFlash-Attn.

---

## Table of Contents

- [Build Issues](#build-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Numerical Accuracy](#numerical-accuracy)
- [Error Code Reference](#error-code-reference)
- [Getting Help](#getting-help)

---

## Build Issues

### CMake Cannot Find CUDA

**Symptoms:**
```
CMake Error: Could not find CUDA
```

**Solutions:**

1. **Verify CUDA Installation:**
   ```bash
   nvcc --version
   nvidia-smi
   ```

2. **Explicitly Set CUDA Paths:**
   ```bash
   cmake --preset release \
         -DCUDAToolkit_ROOT=/usr/local/cuda \
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
   ```

3. **Common Path Locations:**
   | OS | Default CUDA Path |
   |---|-------------------|
   | Linux | `/usr/local/cuda` |
   | Windows | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8` |

4. **Set Environment Variables:**
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

---

### Unsupported GPU Architecture

**Symptoms:**
```
nvcc fatal : Unsupported GPU architecture 'sm_89'
```

**Cause:** CUDA toolkit version doesn't support your GPU architecture.

**Solutions:**

1. **Update CUDA Toolkit** (recommended for newer GPUs)

2. **Target Compatible Architecture:**
   ```bash
   # Check supported architectures
   nvcc --help | grep "gpu-architecture"
   
   # Build for supported architecture
   cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=80  # Adjust to your CUDA version
   ```

3. **Compatibility Matrix:**
   | Architecture | Minimum CUDA Version |
   |--------------|---------------------|
   | sm_70 (V100) | CUDA 9.0 |
   | sm_75 (Turing) | CUDA 10.0 |
   | sm_80 (A100) | CUDA 11.0 |
   | sm_86 (RTX 3090) | CUDA 11.1 |
   | sm_89 (RTX 4090) | CUDA 11.8 |
   | sm_90 (H100) | CUDA 12.0 |

---

### Out of Memory During Build

**Symptoms:**
```
nvcc fatal : Memory allocation failure
```

**Solutions:**

1. **Reduce Parallel Jobs:**
   ```bash
   cmake --build --preset release -j2  # Use only 2 parallel jobs
   ```

2. **Reduce Target Architectures:**
   ```bash
   # Build for single architecture
   cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=86
   ```

3. **Close Other Applications:** Free up system memory

---

### Linker Errors

**Symptoms:**
```
undefined reference to `cuflash::flash_attention_forward'
```

**Solutions:**

1. **Verify Library Build:**
   ```bash
   ls -la build/release/libcuflash_attn*
   ```

2. **Set Library Path:**
   ```bash
   export LD_LIBRARY_PATH=$PWD/build/release:$LD_LIBRARY_PATH
   ```

3. **Link Correctly in Your Project:**
   ```cmake
   target_link_libraries(your_target cuflash_attn)
   ```

---

## Runtime Errors

### CUDA Out of Memory

**Error Code:** `FlashAttentionError::OUT_OF_MEMORY`

**Symptoms:**
```
CUDA error: out of memory
```

**Solutions:**

1. **Check Available Memory:**
   ```bash
   nvidia-smi
   ```

2. **Reduce Batch Size or Sequence Length:**
   | Configuration | Memory Impact |
   |---------------|---------------|
   | batch_size | Linear |
   | seq_len | Linear |
   | num_heads | Linear |
   | head_dim | Fixed (32/64/128) |

3. **Memory Estimation Formulas:**
   ```
   Forward: ~4 × batch × heads × seq_len × head_dim (bytes for FP32)
   Backward: ~6 × batch × heads × seq_len × head_dim (bytes for FP32)
   ```

4. **Free GPU Memory Before Running:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

### Unsupported Head Dimension

**Error Code:** `FlashAttentionError::UNSUPPORTED_HEAD_DIM`

**Valid Values:** 32, 64, 128

**Solutions:**

1. **Check Your Configuration:**
   ```cpp
   if (head_dim != 32 && head_dim != 64 && head_dim != 128) {
       // Not supported
   }
   ```

2. **Workarounds:**
   - Pad to nearest supported dimension
   - Use multiple calls for heads of different sizes

---

### Invalid Dimension Parameters

**Error Code:** `FlashAttentionError::INVALID_DIMENSION`

**Cause:** batch_size, num_heads, seq_len, or head_dim ≤ 0

**Solution:** Verify all dimension parameters are positive integers.

---

### Null Pointer

**Error Code:** `FlashAttentionError::NULL_POINTER`

**Cause:** One or more input/output pointers are `nullptr`

**Solution:** Verify all tensor pointers are properly allocated:
```cpp
cudaMalloc(&d_Q, batch_size * num_heads * seq_len * head_dim * sizeof(float));
// ... allocate all required tensors
```

---

### CUDA Runtime Errors

**Error Code:** `FlashAttentionError::CUDA_ERROR`

**Common Causes:**

1. **Invalid Memory Access:**
   - Pointers not allocated on device memory
   - Memory corruption from other operations

2. **Kernel Launch Failure:**
   - Too many threads or blocks for GPU
   - Resource conflicts

3. **Debug Steps:**
   ```bash
   # Enable CUDA error checking
   export CUDA_LAUNCH_BLOCKING=1
   
   # Run with cuda-memcheck
   cuda-memcheck ./your_program
   
   # Use compute-sanitizer (CUDA 11+)
   compute-sanitizer ./your_program
   ```

---

## Performance Issues

### Slower Than Expected

**Diagnostic Steps:**

1. **Verify GPU Utilization:**
   ```bash
   nvidia-smi -l 1  # Monitor GPU usage
   ```

2. **Check Architecture Match:**
   - Ensure binary is compiled for your GPU architecture
   - Rebuild with correct `CMAKE_CUDA_ARCHITECTURES`

3. **Enable Fast Math (if precision allows):**
   ```bash
   cmake --preset release-fast-math
   cmake --build --preset release-fast-math
   ```

4. **Profile Kernel Execution:**
   ```bash
   # Use Nsight Compute
   ncu ./your_program
   
   # Use Nsight Systems
   nsys profile ./your_program
   ```

---

### High Memory Usage

**Note:** This is expected for head_dim=128 due to larger shared memory requirements.

**Optimization Options:**

1. **Use FP16 Instead of FP32:**
   - Halves memory usage
   - Minimal accuracy impact for most applications

2. **Reduce Block Size (if customized):**
   - Smaller blocks → less shared memory per block
   - May impact performance

---

## Numerical Accuracy

### Results Differ from PyTorch

**Expected Differences:**

| Aspect | Expected Behavior |
|--------|-------------------|
| Small numerical differences | ±1e-5 for FP32, ±1e-3 for FP16 |
| FP16 accumulation | Higher variance due to rounding |

**Diagnostic Steps:**

1. **Check Scale Factor:**
   ```cpp
   float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
   // Should match PyTorch default
   ```

2. **Verify Causal Masking:**
   - Ensure `causal` parameter matches between implementations

3. **Use Same Data Type:**
   - Compare FP32 to FP32
   - Compare FP16 to FP16

---

### INF/NaN in Output

**Causes:**

1. **Input Contains INF/NaN:**
   ```python
   import torch
   assert not torch.isnan(Q).any()
   assert not torch.isinf(Q).any()
   ```

2. **Extreme QK Values:**
   - Enable causal masking if sequence is autoregressive
   - Check scale factor calculation

3. **FP16 Overflow:**
   - Use FP32 for problematic inputs
   - Enable gradient clipping in training

---

### Gradient Mismatch in Backward Pass

**Common Causes:**

1. **Missing Logsumexp (L):**
   - Must pass L from forward pass to backward pass
   - L must be from the same forward call

2. **Incorrect Gradient Flow:**
   - dO should be gradients w.r.t. O (forward output)
   - dQ, dK, dV are output parameters

---

## Error Code Reference

| Error Code | Value | Meaning | Resolution |
|------------|-------|---------|------------|
| `SUCCESS` | 0 | Operation successful | N/A |
| `INVALID_DIMENSION` | 1 | Dimension ≤ 0 | Check all dimension parameters |
| `DIMENSION_MISMATCH` | 2 | Reserved | Not currently used |
| `NULL_POINTER` | 3 | Null pointer passed | Verify all allocations |
| `CUDA_ERROR` | 4 | CUDA runtime error | Check CUDA context and memory |
| `OUT_OF_MEMORY` | 5 | GPU OOM | Reduce problem size or free memory |
| `UNSUPPORTED_HEAD_DIM` | 6 | head_dim not in {32,64,128} | Use supported dimension |
| `UNSUPPORTED_DTYPE` | 7 | Data type not supported | Use float or half |

```cpp
// Comprehensive error handling example
auto err = cuflash::flash_attention_forward(...);
switch (err) {
    case cuflash::FlashAttentionError::SUCCESS:
        break;
    case cuflash::FlashAttentionError::OUT_OF_MEMORY:
        std::cerr << "GPU out of memory. Try reducing batch size.\n";
        break;
    case cuflash::FlashAttentionError::UNSUPPORTED_HEAD_DIM:
        std::cerr << "head_dim must be 32, 64, or 128.\n";
        break;
    default:
        std::cerr << "Error: " << cuflash::get_error_string(err) << "\n";
}
```

---

## Getting Help

### Before Asking

1. **Check Error Code:** Use `get_error_string()` for detailed message
2. **Verify Setup:** Run `nvidia-smi` and `nvcc --version`
3. **Test Basic Functionality:** Run built-in tests with `ctest`

### Reporting Issues

When reporting issues, include:

1. **System Information:**
   ```bash
   nvidia-smi
   nvcc --version
   cmake --version
   ```

2. **Error Output:** Full error message and stack trace
3. **Minimal Reproduction:** Small code snippet that triggers the issue
4. **Build Configuration:** CMake cache or preset used

### Resources

- [API Reference](api-reference.md)
- [Build Guide](building.md)
- [Algorithm Documentation](algorithm.md)
- [GitHub Issues](https://github.com/LessUp/cuflash-attn/issues)
