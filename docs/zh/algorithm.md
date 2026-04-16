# FlashAttention 算法详解

FlashAttention 是一种 IO 感知的算法，用于以 O(N) 而非 O(N²) 的内存复杂度计算精确的注意力，同时在实践中实现显著的加速。

---

## 目录

- [标准 Attention 瓶颈](#标准-attention-瓶颈)
- [FlashAttention 核心概念](#flashattention-核心概念)
  - [分块](#1-分块)
  - [Online Softmax](#2-online-softmax)
  - [重计算](#3-重计算)
- [前向传播算法](#前向传播算法)
- [反向传播算法](#反向传播算法)
- [因果掩码](#因果掩码)
- [FP16 实现](#fp16-实现)
- [内存复杂度分析](#内存复杂度分析)
- [实现要点](#实现要点)
- [参考文献](#参考文献)

---

## 标准 Attention 瓶颈

标准自注意力计算：

```
S = Q × K^T           # [N, N] — 注意力得分矩阵
P = softmax(S)        # [N, N] — 注意力权重矩阵
O = P × V             # [N, d] — 输出
```

**核心问题：** 中间矩阵 S 和 P 具有 O(N²) 大小，必须存储在 HBM（设备内存）中。对于大的序列长度 N：

| 问题 | 影响 |
|------|------|
| **内存使用** | N=4096、32 个头时，仅注意力矩阵就需要约 2 GB |
| **带宽瓶颈** | GPU 计算速度远快于 HBM 带宽；时间主要花在数据移动上 |
| **IO 操作** | S 和 P 各自需要写入和读取 HBM：总共 4 次 O(N²) 操作 |

---

## FlashAttention 核心概念

### 1. 分块

将 Q、K、V 分成可以放入 SRAM（共享内存）的块：

```
Q = [Q_1, Q_2, ..., Q_Tr]    每块 [B_r, d]
K = [K_1, K_2, ..., K_Tc]    每块 [B_c, d]
V = [V_1, V_2, ..., V_Tc]    每块 [B_c, d]
```

**块大小选择：**

| GPU 架构 | SRAM 大小 | 典型 B_r × B_c |
|----------|-----------|----------------|
| Volta (V100) | 96 KB | 64 × 64 |
| Ampere (A100) | 164 KB | 128 × 64 |
| Hopper (H100) | 228 KB | 128 × 128 |

**分块为何有效：**
- 每个块可以放入快速的 SRAM（L1/共享内存）
- 避免中间结果的重复 HBM 访问
- 实现独立块的并行处理

### 2. Online Softmax

标准 softmax 需要两遍扫描（查找最大值 → 计算 exp 和 → 归一化）。FlashAttention 使用 online softmax 在单次遍历中增量更新：

```python
对每个 KV 块 j:
    S_ij = Q_i × K_j^T                 # 局部注意力得分
    m_new = max(m_old, rowmax(S_ij))   # 更新全局最大值
    P = exp(S_ij - m_new)              # 局部 softmax 分子
    l_new = exp(m_old - m_new) × l_old + rowsum(P)  # 更新归一化因子
    O_i = (exp(m_old - m_new) × O_i + P × V_j) / l_new  # 更新输出
```

**关键洞察：** 处理新的 KV 块时，必须通过 `exp(m_old - m_new)` 修正之前的输出，因为全局最大值可能已改变。

**数值稳定性：** 跟踪运行最大值确保即使对于大的注意力得分也不会发生 exp() 溢出。

### 3. 重计算

标准反向传播存储 O(N²) 的注意力矩阵 P 用于梯度计算。FlashAttention 的策略：

| 阶段 | 存储 | 内存 |
|------|------|------|
| **前向传播** | 仅输出 O 和 logsumexp L | O(N) |
| **反向传播** | 从 Q、K、V、O、L 重新计算注意力权重 | O(N) |

**权衡：** 增加计算量（约 33% 更多 FLOPs）但显著减少 HBM IO，从而实现整体加速。

---

## 前向传播算法

```
输入: Q, K, V ∈ R^(N×d), scale
输出: O ∈ R^(N×d), L ∈ R^N

初始化: O = 0, m = -∞, l = 0

对每个 Q 块 i（并行）:
    加载 Q_i 到 SRAM
    对每个 KV 块 j:
        加载 K_j、V_j 到 SRAM
        S_ij = scale × Q_i × K_j^T           # 在 SRAM 中计算
        m_new = max(m_i, rowmax(S_ij))
        P = exp(S_ij - m_new)
        l_new = exp(m_i - m_new) × l_i + rowsum(P)
        O_i = (l_i × exp(m_i - m_new) × O_i + P × V_j) / l_new
        m_i = m_new, l_i = l_new
    L_i = m_i + log(l_i)                      # 存储 logsumexp
```

**关键操作：**
1. **Q 块并行：** 每个输出块独立计算
2. **KV 块串行：** 在所有键上累加注意力
3. **输出修正：** 发现新的最大值时调整运行和

---

## 反向传播算法

```
输入: Q, K, V, O, L, dO
输出: dQ, dK, dV

对每个 KV 块 j:
    加载 K_j、V_j 到 SRAM
    初始化 dK_j = 0, dV_j = 0
    对每个 Q 块 i:
        加载 Q_i、O_i、dO_i、L_i 到 SRAM
        S_ij = scale × Q_i × K_j^T
        P_ij = exp(S_ij - L_i)               # 重新计算注意力权重
        D_i = rowsum(dO_i ⊙ O_i)             # 对角项
        dV_j += P_ij^T × dO_i                # V 的梯度
        dP_ij = dO_i × V_j^T
        dS_ij = P_ij ⊙ (dP_ij - D_i)         # Softmax 梯度
        dQ_i += scale × dS_ij × K_j          # Q 的梯度
        dK_j += scale × dS_ij^T × Q_i        # K 的梯度
```

**梯度流：**
1. **dV：** 使用注意力权重的梯度加权和
2. **dQ、dK：** 通过 softmax 雅可比矩阵使用重新计算的 P
3. **内存高效：** 不需要 O(N²) 存储

---

## 因果掩码

对于自回归模型（如 GPT），位置 i 只能关注位置 ≤ i。FlashAttention 的块结构实现了高效的因果掩码：

| 情况 | 处理方式 |
|------|----------|
| **完全跳过** | KV 块起始列 > Q 块结束行时 → 跳过整个块 |
| **部分掩码** | 在块内应用掩码（设为 -∞） |

**效率提升：** 大约 50% 的块可以完全跳过，减少一半的计算量。

**实现：**
```
对 Q 块 i:
    对 KV 块 j:
        if block_start_j > block_end_i:
            continue  # 整个块被掩码，跳过
        elif 块需要部分掩码:
            在 softmax 计算期间应用掩码
```

---

## FP16 实现

本实现完全支持前向和反向传播的 FP16（半精度）。

### 实现策略

FP16 输入在内部转换为 FP32 进行计算，然后再转换回 FP16 输出：

```
输入: half* Q, K, V
内部: float (FP32) 计算
输出: half* O, L
```

### 数值精度

| 操作 | 精度 |
|------|------|
| 矩阵乘法 (Q × K^T) | FP32 |
| Softmax 计算 | FP32 |
| 累加 | FP32 |
| 最终输出 | FP16 |

**优势：**
- 与 FP32 相当的数值稳定性
- 减少内存带宽（张量大小减半）
- 支持所有现代 GPU（计算能力 ≥ 5.3）

---

## 内存复杂度分析

| 方法 | 前向内存 | 反向内存 | HBM IO |
|------|---------|---------|--------|
| 标准 Attention | O(N²) | O(N²) | O(N² + Nd) |
| FlashAttention | O(N) | O(N) | O(N²d / M) |

其中 M 是 SRAM 大小。当 M = Θ(Nd) 时，IO 复杂度接近 O(Nd)，这是最优的。

### 实际内存节省

| 序列长度 | 标准 Attention | FlashAttention | 节省 |
|-----------------|-------------------|----------------|---------|
| 1,024 | 4 MB | 8 KB | 99.8% |
| 4,096 | 64 MB | 32 KB | 99.95% |
| 16,384 | 1 GB | 128 KB | 99.99% |

---

## 实现要点

### 块配置

| head_dim | BLOCK_M | BLOCK_N | 说明 |
|----------|---------|---------|------|
| 32 | 64 | 64 | 标准配置 |
| 64 | 64 | 64 | 标准配置 |
| 128 | 32 | 32 | 针对更大的共享内存需求减小 |

### 优化技术

| 技术 | 收益 |
|-----------|---------|
| **向量化内存访问** | `float4` 加载/存储以获得更好的带宽 |
| **Launch Bounds** | `__launch_bounds__(128)` 控制寄存器压力 |
| **动态共享内存** | 基于 head_dim 的运行时分配 |
| **流安全** | 显式工作空间生命周期管理 |
| **Warp 级原语** | `__shfl_sync` 用于归约操作 |

### 数据类型支持

| 数据类型 | 前向传播 | 反向传播 |
|-----------|---------|----------|
| FP32 (float) | ✅ | ✅ |
| FP16 (half) | ✅ | ✅ |

---

## 参考文献

1. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
   - Tri Dao、Daniel Y. Fu、Stefano Ermon、Atri Rudra、Christopher Ré
   - NeurIPS 2022
   - [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

2. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
   - Tri Dao
   - ICLR 2024
   - [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

3. **Online normalizer calculation for softmax**
   - Maxim Milakov、Natalia Gimelshein
   - [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)

4. **NVIDIA CUDA 编程指南 - 共享内存**
   - [CUDA C++ 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
