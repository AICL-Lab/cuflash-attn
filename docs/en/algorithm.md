# FlashAttention Algorithm Deep Dive

FlashAttention is an IO-aware algorithm for computing exact attention with reduced memory complexity from O(N²) to O(N), while achieving significant speedup in practice.

---

## Table of Contents

- [Standard Attention Bottleneck](#standard-attention-bottleneck)
- [Core FlashAttention Concepts](#core-flashattention-concepts)
  - [Tiling](#1-tiling)
  - [Online Softmax](#2-online-softmax)
  - [Recomputation](#3-recomputation)
- [Forward Pass Algorithm](#forward-pass-algorithm)
- [Backward Pass Algorithm](#backward-pass-algorithm)
- [Causal Masking](#causal-masking)
- [FP16 Implementation](#fp16-implementation)
- [Memory Complexity Analysis](#memory-complexity-analysis)
- [Implementation Highlights](#implementation-highlights)
- [References](#references)

---

## Standard Attention Bottleneck

Standard self-attention computation:

```
S = Q × K^T           # [N, N] — Attention score matrix
P = softmax(S)        # [N, N] — Attention weight matrix
O = P × V             # [N, d] — Output
```

**Core Problem:** Intermediate matrices S and P have O(N²) size, must be stored in HBM (device memory). For large sequence lengths N:

| Issue | Impact |
|-------|--------|
| **Memory Usage** | N=4096, 32 heads → ~2 GB just for attention matrices |
| **Bandwidth Bottleneck** | GPU computation is much faster than HBM bandwidth; time dominated by data movement |
| **IO Operations** | S and P each require write-to and read-from HBM: 4 O(N²) operations total |

---

## Core FlashAttention Concepts

### 1. Tiling

Divide Q, K, V into blocks that fit in SRAM (shared memory):

```
Q = [Q_1, Q_2, ..., Q_Tr]    Each block [B_r, d]
K = [K_1, K_2, ..., K_Tc]    Each block [B_c, d]
V = [V_1, V_2, ..., V_Tc]    Each block [B_c, d]
```

**Block Size Selection:**

| GPU Architecture | SRAM Size | Typical B_r × B_c |
|------------------|-----------|-------------------|
| Volta (V100) | 96 KB | 64 × 64 |
| Ampere (A100) | 164 KB | 128 × 64 |
| Hopper (H100) | 228 KB | 128 × 128 |

**Why Tiling Works:**
- Each block fits in fast SRAM (L1/shared memory)
- Avoids repeated HBM accesses for intermediate results
- Enables parallel processing of independent blocks

### 2. Online Softmax

Standard softmax requires two passes (find max → compute exp sum → normalize). FlashAttention uses online softmax to update incrementally in a single pass:

```python
for each KV block j:
    S_ij = Q_i × K_j^T                 # Local attention scores
    m_new = max(m_old, rowmax(S_ij))   # Update global maximum
    P = exp(S_ij - m_new)              # Local softmax numerator
    l_new = exp(m_old - m_new) × l_old + rowsum(P)  # Update normalizer
    O_i = (exp(m_old - m_new) × O_i + P × V_j) / l_new  # Update output
```

**Key Insight:** When processing a new KV block, previous outputs must be corrected by `exp(m_old - m_new)` because the global maximum may have changed.

**Numerical Stability:** Tracking running maximum ensures no exp() overflow even for large attention scores.

### 3. Recomputation

Standard backward pass stores O(N²) attention matrix P for gradient computation. FlashAttention's strategy:

| Phase | Storage | Memory |
|-------|---------|--------|
| **Forward** | Output O and logsumexp L only | O(N) |
| **Backward** | Recompute attention weights from Q, K, V, O, L | O(N) |

**Trade-off:** Increases computation (~33% more FLOPs) but significantly reduces HBM IO, resulting in overall speedup.

---

## Forward Pass Algorithm

```
Input: Q, K, V ∈ R^(N×d), scale
Output: O ∈ R^(N×d), L ∈ R^N

Initialize: O = 0, m = -∞, l = 0

For each Q block i (parallel):
    Load Q_i to SRAM
    For each KV block j:
        Load K_j, V_j to SRAM
        S_ij = scale × Q_i × K_j^T           # Compute in SRAM
        m_new = max(m_i, rowmax(S_ij))
        P = exp(S_ij - m_new)
        l_new = exp(m_i - m_new) × l_i + rowsum(P)
        O_i = (l_i × exp(m_i - m_new) × O_i + P × V_j) / l_new
        m_i = m_new, l_i = l_new
    L_i = m_i + log(l_i)                      # Store logsumexp
```

**Key Operations:**
1. **Parallel over Q blocks:** Each output block computed independently
2. **Sequential over KV blocks:** Accumulate attention across all keys
3. **Output correction:** Adjust running sum when new maximum found

---

## Backward Pass Algorithm

```
Input: Q, K, V, O, L, dO
Output: dQ, dK, dV

For each KV block j:
    Load K_j, V_j to SRAM
    Initialize dK_j = 0, dV_j = 0
    For each Q block i:
        Load Q_i, O_i, dO_i, L_i to SRAM
        S_ij = scale × Q_i × K_j^T
        P_ij = exp(S_ij - L_i)               # Recompute attention weights
        D_i = rowsum(dO_i ⊙ O_i)             # Diagonal term
        dV_j += P_ij^T × dO_i                # V gradient
        dP_ij = dO_i × V_j^T
        dS_ij = P_ij ⊙ (dP_ij - D_i)         # Softmax gradient
        dQ_i += scale × dS_ij × K_j          # Q gradient
        dK_j += scale × dS_ij^T × Q_i        # K gradient
```

**Gradient Flow:**
1. **dV:** Weighted sum of gradients using attention weights
2. **dQ, dK:** Through softmax Jacobian using recomputed P
3. **Memory efficient:** No O(N²) storage needed

---

## Causal Masking

For autoregressive models (like GPT), position i can only attend to positions ≤ i. FlashAttention's block structure enables efficient causal masking:

| Case | Handling |
|------|----------|
| **Full skip** | KV block start column > Q block end row → skip entire block |
| **Partial mask** | Apply mask within block (set to -∞) |

**Efficiency Gain:** Approximately 50% of blocks can be skipped entirely, reducing computation by half.

**Implementation:**
```
for Q block i:
    for KV block j:
        if block_start_j > block_end_i:
            continue  # Entire block masked, skip
        elif block needs partial masking:
            apply mask during softmax computation
```

---

## FP16 Implementation

This implementation fully supports FP16 (half precision) for both forward and backward passes.

### Implementation Strategy

FP16 inputs are converted to FP32 internally for computation, then converted back to FP16 for output:

```
Input: half* Q, K, V
Internal: float (FP32) computation
Output: half* O, L
```

### Numerical Precision

| Operation | Precision |
|-----------|-----------|
| Matrix multiplication (Q × K^T) | FP32 |
| Softmax computation | FP32 |
| Accumulation | FP32 |
| Final output | FP16 |

**Benefits:**
- Numerical stability comparable to FP32
- Reduced memory bandwidth (2× smaller tensors)
- Supported on all modern GPUs (compute capability ≥ 5.3)

---

## Memory Complexity Analysis

| Method | Forward Memory | Backward Memory | HBM IO |
|--------|----------------|-----------------|--------|
| Standard Attention | O(N²) | O(N²) | O(N² + Nd) |
| FlashAttention | O(N) | O(N) | O(N²d / M) |

Where M is SRAM size. When M = Θ(Nd), IO complexity approaches O(Nd), which is optimal.

### Real Memory Savings

| Sequence Length | Standard Attention | FlashAttention | Savings |
|-----------------|-------------------|----------------|---------|
| 1,024 | 4 MB | 8 KB | 99.8% |
| 4,096 | 64 MB | 32 KB | 99.95% |
| 16,384 | 1 GB | 128 KB | 99.99% |

---

## Implementation Highlights

### Block Configuration

| head_dim | BLOCK_M | BLOCK_N | Notes |
|----------|---------|---------|-------|
| 32 | 64 | 64 | Standard configuration |
| 64 | 64 | 64 | Standard configuration |
| 128 | 32 | 32 | Reduced for larger shared memory needs |

### Optimization Techniques

| Technique | Benefit |
|-----------|---------|
| **Vectorized Memory Access** | `float4` loads/stores for better bandwidth |
| **Launch Bounds** | `__launch_bounds__(128)` controls register pressure |
| **Dynamic Shared Memory** | Runtime allocation based on head_dim |
| **Stream Safety** | Explicit workspace lifetime management |
| **Warp-level Primitives** | `__shfl_sync` for reduction operations |

### Data Type Support

| Data Type | Forward | Backward |
|-----------|---------|----------|
| FP32 (float) | ✅ | ✅ |
| FP16 (half) | ✅ | ✅ |

---

## References

1. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
   - Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
   - NeurIPS 2022
   - [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

2. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
   - Tri Dao
   - ICLR 2024
   - [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

3. **Online normalizer calculation for softmax**
   - Maxim Milakov, Natalia Gimelshein
   - [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)

4. **NVIDIA CUDA Programming Guide - Shared Memory**
   - [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
