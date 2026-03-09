# FlashAttention 算法详解

FlashAttention 是一种 IO-aware 的精确注意力算法，通过分块计算和重计算策略，将注意力机制的内存复杂度从 O(N²) 降低到 O(N)，同时在实践中显著提升计算速度。

## 标准 Attention 的瓶颈

标准 Self-Attention 的计算流程：

```
S = Q × Kᵀ           # [N, N] — 注意力得分矩阵
P = softmax(S)        # [N, N] — 注意力权重矩阵
O = P × V             # [N, d] — 输出
```

**核心问题**：中间矩阵 S 和 P 的大小为 O(N²)，必须在 HBM（显存）中存储。当序列长度 N 较大时：

- **显存占用**：N=4096, heads=32 时，仅注意力矩阵就需要 ~2 GB
- **带宽瓶颈**：GPU 计算速度远快于 HBM 读写速度，大量时间花在数据搬运上
- **IO 次数**：S、P 矩阵各需一次写入和一次读取 HBM，共 4 次 O(N²) IO

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

关键：每处理一个新的 KV 块时，之前的输出需要通过 `exp(m_old - m_new)` 进行修正。

### 3. 重计算（Recomputation）

在反向传播中，标准实现需要存储 O(N²) 的注意力矩阵 P 用于梯度计算。FlashAttention 的策略是：

- **前向传播**：只存储输出 O 和 logsumexp L（各 O(N)）
- **反向传播**：从 Q、K、V、O、L 重新计算注意力权重

虽然增加了计算量（~33% FLOPs），但大幅减少了 HBM IO，总体速度更快。

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

## 因果掩码

在自回归模型（如 GPT）中，位置 i 只能关注位置 ≤ i 的 token。FlashAttention 的分块结构天然支持高效因果掩码：

- **完全跳过**：当 KV 块的起始列 > Q 块的末尾行时，整个块被掩码，直接跳过
- **部分掩码**：在块内逐元素应用掩码（设为 -∞）

这比标准实现更高效，因为可以跳过约一半的块计算。

## 内存复杂度对比

| 方法 | 前向额外内存 | 反向额外内存 | HBM IO |
|------|-------------|-------------|--------|
| 标准 Attention | O(N²) | O(N²) | O(N² + Nd) |
| FlashAttention | O(N) | O(N) | O(N²d / M) |

其中 M 是 SRAM 大小。当 M = Θ(Nd) 时，IO 复杂度为 O(Nd)，达到最优。

## 本项目实现特点

- **分块大小**: 根据 head_dim 自动选择最优 BLOCK_M / BLOCK_N
- **支持 head_dim**: 32, 64, 128
- **向量化访存**: `float4` 向量化加载/存储，提升带宽利用率
- **Launch Bounds**: `__launch_bounds__(128)` 优化寄存器分配
- **FP16 支持**: 利用 half 精度减少带宽压力
- **流安全**: 所有异步操作使用 `cudaMemsetAsync`，释放前同步

## 参考文献

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (Dao, 2023)
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (Milakov & Gimelshein, 2018)
