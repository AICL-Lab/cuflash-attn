# Testing Specification: CuFlash-Attn

## Overview

This document defines the testing strategy and specifications for CuFlash-Attn. All tests are designed to validate the correctness properties defined in the core architecture RFC.

---

## Test Frameworks

| Framework | Purpose |
|-----------|---------|
| **Google Test** | C++ unit testing framework |
| **RapidCheck** | Property-based testing (optional) |
| **PyTorch** | Reference implementation for numerical validation |

---

## Correctness Properties

### Property 1: Forward Pass Numerical Equivalence

**Statement:** For any valid Q, K, V input matrices, FlashAttention forward output should match standard attention computation `softmax(QK^T * scale) @ V` within 1e-3 error tolerance.

**Validates:** REQ-1.1, REQ-1.2, REQ-1.5, REQ-7.5, REQ-8.1

**Test Strategy:**
- Generate random Q, K, V matrices
- Compute output using FlashAttention
- Compute reference output using standard attention
- Compare outputs with max absolute error < 1e-3

### Property 2: Backward Pass Gradient Equivalence

**Statement:** For any valid Q, K, V, dO inputs, FlashAttention backward computed dQ, dK, dV gradients should match standard attention backward gradients within 1e-3 error tolerance.

**Validates:** REQ-2.1, REQ-2.3, REQ-2.4, REQ-8.2

**Test Strategy:**
- Generate random Q, K, V, dO matrices
- Compute gradients using FlashAttention backward
- Compute reference gradients using standard attention backward
- Compare gradients with max absolute error < 1e-3

### Property 3: Online Softmax Equivalence

**Statement:** For any input vector sequence, the online softmax algorithm's final result should be numerically equivalent to standard softmax computation.

**Validates:** REQ-4.3

**Test Strategy:**
- Generate random input vectors
- Compute online softmax result
- Compute standard softmax result
- Compare results with numerical equivalence

### Property 4: Numerical Stability

**Statement:** For any valid input containing extreme values, computation should not produce NaN or Inf.

**Validates:** REQ-4.4, REQ-8.3

**Test Strategy:**
- Generate inputs with extreme values (very large, very small)
- Verify no NaN or Inf in outputs
- Test edge cases near numerical limits

### Property 5: Causal Mask Correctness

**Statement:** For any attention computation with causal masking enabled, output at position i should only depend on inputs at positions 0 to i.

**Validates:** REQ-5.1

**Test Strategy:**
- Enable causal masking
- Verify that position i output is independent of positions > i
- Test boundary conditions at mask edges

### Property 6: Data Type Support

**Statement:** For any valid input, the API should correctly handle both FP32 and FP16 data types.

**Validates:** REQ-7.4

**Test Strategy:**
- Test all properties with FP32 inputs
- Test all properties with FP16 inputs
- Verify type conversion correctness

### Property 7: Invalid Input Error Handling

**Statement:** For any invalid input, the API should return descriptive error messages rather than crashing.

**Validates:** REQ-7.3

**Test Strategy:**
- Test with null pointers
- Test with invalid dimensions
- Test with unsupported head_dim values
- Verify appropriate error codes are returned

---

## Test Categories

### Unit Tests

| Test | Description |
|------|-------------|
| `OnlineSoftmaxTest` | Test online softmax correctness |
| `MatMulTest` | Test blocked matrix multiplication |
| `CausalMaskTest` | Test causal mask application |
| `BoundaryTest` | Test boundary handling for non-divisible seq_len |

### Property Tests

| Test | Property |
|------|----------|
| `ForwardPropertyTest` | Property 1: Forward numerical equivalence |
| `BackwardPropertyTest` | Property 2: Backward gradient equivalence |
| `OnlineSoftmaxPropertyTest` | Property 3: Online softmax equivalence |
| `StabilityPropertyTest` | Property 4: Numerical stability |
| `CausalPropertyTest` | Property 5: Causal mask correctness |
| `DTypePropertyTest` | Property 6: Data type support |
| `ErrorPropertyTest` | Property 7: Invalid input error handling |

### Integration Tests

| Test | Description |
|------|-------------|
| `PyTorchComparisonTest` | Compare against PyTorch standard attention |
| `EndToEndTest` | Full forward + backward pipeline |
| `MultiHeadTest` | Test with multiple attention heads |
| `BatchTest` | Test with batch size > 1 |

### Performance Tests

| Test | Description |
|------|-------------|
| `MemoryUsageTest` | Verify O(N) memory complexity |
| `SpeedBenchmark` | Benchmark against standard attention |
| `ScalingTest` | Test scaling with sequence length |

---

## Test Configuration Matrix

### Supported head_dim Values

| head_dim | BLOCK_M | BLOCK_N | Tests |
|----------|---------|---------|-------|
| 32 | 64 | 64 | All properties |
| 64 | 64 | 64 | All properties |
| 128 | 32 | 32 | All properties |

### Data Type Matrix

| Data Type | Forward | Backward | Tests |
|-----------|---------|----------|-------|
| FP32 (`float`) | ✅ | ✅ | All properties |
| FP16 (`half`) | ✅ | ✅ | All properties |

### Causal Masking

| Causal | Tests |
|--------|-------|
| true | All properties with causal masking |
| false | All properties without causal masking |

---

## Test Execution

### Running All Tests

```bash
ctest --preset release --output-on-failure
```

### Running Specific Tests

```bash
# Run forward tests only
ctest --preset release -R ForwardTest

# Run backward tests only
ctest --preset release -R BackwardTest

# Run property tests only
ctest --preset release -R PropertyTest
```

### PyTorch Comparison

```bash
python tests/test_pytorch_comparison.py
```

---

## Coverage Requirements

| Category | Target Coverage |
|----------|-----------------|
| Code Coverage | > 90% line coverage |
| Property Coverage | 100% of correctness properties |
| Configuration Coverage | All supported head_dim, dtypes, causal settings |
| Edge Case Coverage | Boundary conditions, extreme values, error cases |

---

## Requirements Traceability Matrix

| Requirement | Test Coverage |
|-------------|---------------|
| REQ-1 | Property 1 (Forward Pass Numerical Equivalence) |
| REQ-2 | Property 2 (Backward Pass Gradient Equivalence) |
| REQ-3 | Unit Tests (Tiling Computation Boundaries) |
| REQ-4 | Property 3 (Online Softmax Equivalence), Property 4 (Numerical Stability) |
| REQ-5 | Property 5 (Causal Mask Correctness) |
| REQ-6 | Error Handling Tests |
| REQ-7 | API Smoke Tests, Property 6 (Data Type Support) |
| REQ-8 | PyTorch Comparison Tests, All Property Tests |
