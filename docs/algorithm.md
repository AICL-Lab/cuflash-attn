# FlashAttention 算法详解

FlashAttention 是一种 IO-aware 的精确注意力算法，通过分块计算和重计算策略，将注意力机制的内存复杂度从 O(N²) 降低到 O(N)，同时在实践中显著提升计算速度。

## 目录

- [标准 Attention 的瓶颈](#标准-attention-的瓶颈)
- [FlashAttention 的核心思想](#flashattention-的核心思想)
- [前向传播算法](#前向传播算法)
- [反向传播算法](#反向传播算法)
- [因果掩码](#因果掩码)
- [FP16 支持](#fp16-支持)
- [内存复杂度对比](#内存复杂度对比)
- [本项目实现特点](#本项目实现特点)
- [参考文献](#参考文献)

---

## 标准 Attention 的瓶颈

标准 Self-Attention 的计算流程：

```
S = Q × Kᵀ           # [N, N] — 注意力得分矩阵
P = softmax(S)        # [N, N] — 注意力权重矩阵
O = P × V             # [N, d] — 输出
```

**核心问题**：中间矩阵 S 和 P 的大小为 O(N²)，必须在 HBM（显存）中存储。当序列长度 N 较大时：

| 问题 | 影响 |
|------|------|
| **显存占用** | N=4096, heads=32 时，仅注意力矩阵就需要 ~2 GB |
| **带宽瓶颈** | GPU 计算速度远快于 HBM 读写速度，大量时间花在数据搬运上 |
| **IO 次数** | S、P 矩阵各需一次写入和一次读取 HBM，共 4 次 O(N²) IO |

---

## FlashAttention 的核心思想

### 1. 分块（Tiling）

将 Q、K、V 分成大小为 B 的块，使每个块能完全放入 SRAM（共享内存）：

```
Q = [Q₁, Q₂, ..., Qₜ]    每块 [B_r, d]
K = [K₁, K₂, ..., Kₜ]    每块 [B_c, d]
V = [V₁, V₂, ..., Vₜ]    每块 [B_c, d]
```

块大小选择依据：

| GPU 架构 | SRAM 大小 | 典型 B_r × B_c |
|----------|-----------|----------------|
| Volta (V100) | 96 KB | 64 × 64 |
| Ampere (A100) | 164 KB | 128 × 64 |
| Hopper (H100) | 228 KB | 128 × 128 |

### 2. Online Softmax

标准 softmax 需要两遍扫描（计算 max → 计算 exp sum → 归一化），FlashAttention 使用 online softmax 在单遍扫描中增量更新：

```
对每个 KV 块 j:
    Sᵢⱼ = Qᵢ × Kⱼᵀ                    # 局部注意力得分
    mᵢⱼ = max(mᵢ,ⱼ₋₁, rowmax(Sᵢⱼ))    # 更新全局最大值
    Pᵢⱼ = exp(Sᵢⱼ - mᵢⱼ)              # 局部 softmax 分子
    lᵢⱼ = exp(mᵢ,ⱼ₋₁ - mᵢⱼ) × lᵢ,ⱼ₋₁ + rowsum(Pᵢⱼ)  # 更新归一化因子
    Oᵢ  = diag(exp(mᵢ,ⱼ₋₁ - mᵢⱼ))⁻¹ × Oᵢ + Pᵢⱼ × Vⱼ  # 增量更新输出
```

**关键**：每处理一个新的 KV 块时，之前的输出需要通过 `exp(m_old - m_new)` 进行修正。

### 3. 重计算（Recomputation）

在反向传播中，标准实现需要存储 O(N²) 的注意力矩阵 P 用于梯度计算。FlashAttention 的策略是：

| 阶段 | 存储内容 | 内存占用 |
|------|----------|----------|
| **前向传播** | 输出 O 和 logsumexp L | O(N) |
| **反向传播** | 从 Q、K、V、O、L 重新计算注意力权重 | O(N) |

虽然增加了计算量（~33% FLOPs），但大幅减少了 HBM IO，总体速度更快。

---

## 前向传播算法

```
输入: Q, K, V ∈ ℝ^{N×d}, scale
输出: O ∈ ℝ^{N×d}, L ∈ ℝ^{N}

初始化: O = 0, m = -∞, l = 0

对每个 Q 块 i (并行):
    加载 Qᵢ 到 SRAM
    对每个 KV 块 j:
        加载 Kⱼ, Vⱼ 到 SRAM
        Sᵢⱼ = scale × Qᵢ × Kⱼᵀ           # 在 SRAM 中计算
        m_new = max(mᵢ, rowmax(Sᵢⱼ))
        P = exp(Sᵢⱼ - m_new)
        l_new = exp(mᵢ - m_new) × lᵢ + rowsum(P)
        Oᵢ = (lᵢ × exp(mᵢ - m_new) × Oᵢ + P × Vⱼ) / l_new
        mᵢ = m_new, lᵢ = l_new
    Lᵢ = mᵢ + log(lᵢ)                     # 存储 logsumexp
```

---

## 反向传播算法

```
输入: Q, K, V, O, L, dO
输出: dQ, dK, dV

对每个 KV 块 j:
    加载 Kⱼ, Vⱼ 到 SRAM
    初始化 dKⱼ = 0, dVⱼ = 0
    对每个 Q 块 i:
        加载 Qᵢ, Oᵢ, dOᵢ, Lᵢ 到 SRAM
        Sᵢⱼ = scale × Qᵢ × Kⱼᵀ
        Pᵢⱼ = exp(Sᵢⱼ - Lᵢ)              # 重计算注意力权重
        Dᵢ = rowsum(dOᵢ ⊙ Oᵢ)
        dVⱼ += Pᵢⱼᵀ × dOᵢ
        dPᵢⱼ = dOᵢ × Vⱼᵀ
        dSᵢⱼ = Pᵢⱼ ⊙ (dPᵢⱼ - Dᵢ)
        dQᵢ += scale × dSᵢⱼ × Kⱼ
        dKⱼ += scale × dSᵢⱼᵀ × Qᵢ
```

---

## 因果掩码

在自回归模型（如 GPT）中，位置 i 只能关注位置 ≤ i 的 token。FlashAttention 的分块结构天然支持高效因果掩码：

| 情况 | 处理方式 |
|------|----------|
| **完全跳过** | KV 块的起始列 > Q 块的末尾行时，整个块被掩码，直接跳过 |
| **部分掩码** | 在块内逐元素应用掩码（设为 -∞） |

这比标准实现更高效，因为可以跳过约一半的块计算。

---

## FP16 支持

本项目完整支持 FP16 数据类型的前向和反向传播：

### 实现策略

FP16 输入在 kernel 内部转换为 FP32 进行计算，输出时再转换回 FP16：

```
输入: half* Q, K, V
内部计算: float (FP32)
输出: half* O, L
```

### 数值精度

| 操作 | 精度 |
|------|------|
| 矩阵乘法 (Q × Kᵀ) | FP32 |
| Softmax 计算 | FP32 |
| 累加操作 | FP32 |
| 最终输出 | FP16 |

这种设计在保持数值稳定性的同时，减少了显存带宽需求。

---

## 内存复杂度对比

| 方法 | 前向额外内存 | 反向额外内存 | HBM IO |
|------|-------------|-------------|--------|
| 标准 Attention | O(N²) | O(N²) | O(N² + Nd) |
| FlashAttention | O(N) | O(N) | O(N²d / M) |

其中 M 是 SRAM 大小。当 M = Θ(Nd) 时，IO 复杂度为 O(Nd)，达到最优。

### 实际内存节省示例

| 序列长度 | 标准 Attention | FlashAttention | 节省比例 |
|----------|----------------|----------------|----------|
| 1024 | 4 MB | 8 KB | 99.8% |
| 4096 | 64 MB | 32 KB | 99.95% |
| 16384 | 1 GB | 128 KB | 99.99% |

---

## 本项目实现特点

### 分块配置

| head_dim | BLOCK_M | BLOCK_N | 说明 |
|----------|---------|---------|------|
| 32 | 64 | 64 | 标准配置 |
| 64 | 64 | 64 | 标准配置 |
| 128 | 32 | 32 | 适配更大共享内存需求 |

### 优化技术

| 技术 | 说明 |
|------|------|
| **向量化访存** | `float4` 向量化加载/存储路径 |
| **Launch Bounds** | `__launch_bounds__(128)` 控制线程块资源使用 |
| **动态共享内存** | 运行时根据 head_dim 调整共享内存大小 |
| **流安全** | 反向传播维护显式 workspace 生命周期 |

### 数据类型支持

| 数据类型 | 前向传播 | 反向传播 |
|----------|----------|----------|
| FP32 (`float`) | ✅ | ✅ |
| FP16 (`half`) | ✅ | ✅ |

---

## 参考文献

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (Dao, 2023)
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (Milakov & Gimelshein, 2018)
