# FlashAttention Algorithm Deep Dive

FlashAttention is an IO-aware exact attention algorithm that reduces memory complexity from $O(N^2)$ to $O(N)$ while matching standard attention numerically.

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

Standard self-attention is defined as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

This expands to three materialized intermediate matrices:

$$
S = QK^T \in \mathbb{R}^{N \times N}, \quad P = \text{softmax}(S) \in \mathbb{R}^{N \times N}, \quad O = PV \in \mathbb{R}^{N \times d}
$$

**Core Problem:** $S$ and $P$ have $O(N^2)$ size and must reside in HBM (device memory). For large $N$:

| Issue | Impact |
|-------|--------|
| **Memory Usage** | $N=4096$, 32 heads $\Rightarrow$ ~2 GB just for $S$ and $P$ |
| **Bandwidth Bottleneck** | GPU compute $\gg$ HBM bandwidth; time dominated by data movement |
| **IO Operations** | $S$ and $P$ each require write-to and read-from HBM: 4 $O(N^2)$ operations total |

![Tiling Overview](/diagrams/tiling-overview.svg)

*Figure 1: Q/K/V tiling into SRAM blocks. Intermediate $S$ and $P$ never touch HBM.*

---

## Core FlashAttention Concepts

### 1. Tiling

Divide $Q$, $K$, $V$ into blocks that fit in SRAM (shared memory / L1 cache):

$$
Q = [Q_1, Q_2, \ldots, Q_{T_r}], \quad Q_i \in \mathbb{R}^{B_r \times d}
$$

$$
K = [K_1, K_2, \ldots, K_{T_c}], \quad K_j \in \mathbb{R}^{B_c \times d}
$$

$$
V = [V_1, V_2, \ldots, V_{T_c}], \quad V_j \in \mathbb{R}^{B_c \times d}
$$

**Block Size Selection:**

| GPU Architecture | SRAM Size | Typical $B_r \times B_c$ |
|------------------|-----------|--------------------------|
| Volta (V100) | 96 KB | $64 \times 64$ |
| Ampere (A100) | 164 KB | $128 \times 64$ |
| Hopper (H100) | 228 KB | $128 \times 128$ |

**Why Tiling Works:**
- Each block fits in fast SRAM ($\sim$19 TB/s) instead of slow HBM ($\sim$2 TB/s).
- Avoids repeated HBM accesses for intermediate results.
- Enables parallel processing of independent Q blocks.

### 2. Online Softmax

Standard softmax requires two passes over each row (find $\max$ $\to$ compute $\exp$ sum $\to$ normalize). FlashAttention uses **online softmax** to update incrementally in a single pass over KV blocks:

For each KV block $j$ processed by Q block $i$:

$$
m_{ij}^{\text{new}} = \max(m_{ij}^{\text{old}}, \text{rowmax}(S_{ij}))
$$

$$
\tilde{P}_{ij} = \exp(S_{ij} - m_{ij}^{\text{new}})
$$

$$
l_{ij}^{\text{new}} = \exp(m_{ij}^{\text{old}} - m_{ij}^{\text{new}}) \cdot l_{ij}^{\text{old}} + \text{rowsum}(\tilde{P}_{ij})
$$

$$
O_{ij}^{\text{new}} = \frac{\exp(m_{ij}^{\text{old}} - m_{ij}^{\text{new}}) \cdot O_{ij}^{\text{old}} + \tilde{P}_{ij} V_j}{l_{ij}^{\text{new}}}
$$

![Online Softmax State Machine](/diagrams/online-softmax-state-machine.svg)

*Figure 2: Online softmax state updates. When a new KV block reveals a larger row max, previous outputs are rescaled by $\exp(m_{\text{old}} - m_{\text{new}})$.*

**Key Insight:** When processing a new KV block, previous outputs must be corrected by $\exp(m_{\text{old}} - m_{\text{new}})$ because the global row maximum may have changed.

**Numerical Stability:** Tracking the running maximum ensures $\exp(\cdot)$ never overflows, even for large attention scores.

### 3. Recomputation

Standard backward pass stores the $O(N^2)$ attention matrix $P$ for gradient computation. FlashAttention's strategy:

| Phase | Storage | Memory |
|-------|---------|--------|
| **Forward** | Output $O$ and logsumexp $L$ only | $O(N)$ |
| **Backward** | Recompute $P$ from $Q, K, V, O, L$ on-the-fly | $O(N)$ |

**Trade-off:** Increases computation by $\sim$33% extra FLOPs, but significantly reduces HBM IO, resulting in overall speedup.

![Backward Recompute Flow](/diagrams/backward-recompute-flow.svg)

*Figure 3: Backward pass recomputes $P_{ij}$ in SRAM from forward outputs. No $O(N^2)$ matrix is stored.*

---

## Forward Pass Algorithm

```
Input:  Q, K, V ∈ R^(N×d), scale = 1/√d
Output: O ∈ R^(N×d), L ∈ R^N

Initialize: O = 0, m = -∞, l = 0  (per row)

For each Q block i (parallel over i = 1..T_r):
    Load Q_i to SRAM
    For each KV block j = 1..T_c (sequential):
        Load K_j, V_j to SRAM
        S_ij = scale × Q_i × K_j^T           # [B_r, B_c] in SRAM
        m_new = max(m_i, rowmax(S_ij))        # Update row max
        P = exp(S_ij - m_new)                 # Local softmax numerator
        l_new = exp(m_i - m_new) × l_i + rowsum(P)
        O_i = (exp(m_i - m_new) × O_i + P × V_j) / l_new
        m_i = m_new, l_i = l_new
    L_i = m_i + log(l_i)                      # Store logsumexp
```

**Key Operations:**
1. **Parallel over Q blocks:** Each output block computed independently by one CUDA block.
2. **Sequential over KV blocks:** Accumulate attention across all keys.
3. **Output correction:** Adjust running sum when a new maximum is found.

---

## Backward Pass Algorithm

```
Input:  Q, K, V, O, L, dO
Output: dQ, dK, dV

For each KV block j:
    Load K_j, V_j to SRAM
    Initialize dK_j = 0, dV_j = 0
    For each Q block i:
        Load Q_i, O_i, dO_i, L_i to SRAM
        S_ij = scale × Q_i × K_j^T
        P_ij = exp(S_ij - L_i)                # Recompute attention weights
        D_i = rowsum(dO_i ⊙ O_i)              # Diagonal term
        dV_j += P_ij^T × dO_i                 # V gradient
        dP_ij = dO_i × V_j^T
        dS_ij = P_ij ⊙ (dP_ij - D_i)          # Softmax Jacobian
        dQ_i += scale × dS_ij × K_j           # Q gradient
        dK_j += scale × dS_ij^T × Q_i        # K gradient
```

**Gradient Flow:**
1. **dV:** Weighted sum of upstream gradients using recomputed attention weights.
2. **dQ, dK:** Through softmax Jacobian using recomputed $P$.
3. **Memory efficient:** No $O(N^2)$ storage needed at any point.

---

## Causal Masking

For autoregressive models, position $i$ can only attend to positions $\leq i$. FlashAttention's block structure enables efficient causal masking:

| Case | Handling |
|------|----------|
| **Full skip** | KV block start column $>$ Q block end row $\Rightarrow$ skip entire block |
| **Partial mask** | Apply mask within block (set to $-\infty$) |

**Efficiency Gain:** Approximately 50% of blocks can be skipped entirely, reducing computation by half.

![Causal Masking Blocks](/diagrams/causal-masking-blocks.svg)

*Figure 4: Causal masking at block granularity. Lower-triangular blocks are fully computed; diagonal blocks are partially masked; upper-triangular blocks are skipped.*

---

## FP16 Implementation

This implementation fully supports FP16 (half precision) for both forward and backward passes.

### Implementation Strategy

FP16 inputs are converted to FP32 internally for computation, then converted back to FP16 for output:

$$
\text{Input: } \texttt{half}^* \; Q, K, V \xrightarrow{\text{load}} \text{FP32 registers} \xrightarrow{\text{compute}} \text{FP32 accumulator} \xrightarrow{\text{store}} \texttt{half}^* \; O, L
$$

### Numerical Precision

| Operation | Precision |
|-----------|-----------|
| Matrix multiplication ($Q \times K^T$) | FP32 |
| Softmax computation | FP32 |
| Accumulation | FP32 |
| Final output | FP16 |

**Benefits:**
- Numerical stability comparable to FP32.
- Reduced memory bandwidth (2$\times$ smaller tensors).
- Supported on all modern GPUs (compute capability $\geq$ 5.3).

---

## Memory Complexity Analysis

| Method | Forward Memory | Backward Memory | HBM IO |
|--------|----------------|-----------------|--------|
| Standard Attention | $O(N^2)$ | $O(N^2)$ | $O(N^2 + Nd)$ |
| FlashAttention | $O(N)$ | $O(N)$ | $O\left(\frac{N^2 d^2}{M}\right)$ |

Where $M$ is SRAM size. When $M = \Theta(Nd)$, IO complexity approaches $O(Nd)$, which is optimal since the inputs and outputs alone are $\Theta(Nd)$.

### Real Memory Savings

| Sequence Length | Standard Attention | FlashAttention | Savings |
|-----------------|--------------------|----------------|---------|
| 1,024 | 4 MB | 8 KB | **99.8%** |
| 4,096 | 64 MB | 32 KB | **99.95%** |
| 16,384 | 1 GB | 128 KB | **99.99%** |

---

## Implementation Highlights

### Block Configuration

| head_dim | $B_r$ | $B_c$ | SRAM per Block |
|----------|-------|-------|----------------|
| 32 | 64 | 64 | $\sim$32 KB |
| 64 | 64 | 64 | $\sim$64 KB |
| 128 | 32 | 32 | $\sim$128 KB |

### Optimization Techniques

| Technique | Benefit |
|-----------|---------|
| **Vectorized Memory Access** | `float4` loads/stores for coalesced bandwidth |
| **Launch Bounds** | `__launch_bounds__(128)` controls register pressure |
| **Dynamic Shared Memory** | Runtime allocation based on `head_dim` |
| **Stream Safety** | Explicit workspace lifetime management |
| **Warp-level Primitives** | `__shfl_sync` for intra-warp reduction |

### Data Type Support

| Data Type | Forward | Backward |
|-----------|---------|----------|
| FP32 (`float`) | Full | Full |
| FP16 (`half`) | Full | Full |

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
