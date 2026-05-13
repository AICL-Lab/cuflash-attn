# Roofline 分析

> **版本**: v0.3.0  
> **适用范围**: CuFlash-Attn 前向/反向 kernel，FP16，causal/non-causal  
> **前置阅读**: [基准测试](./benchmarks.md)（含实测带宽与耗时数据）

---

## 1. Roofline 模型简介

Roofline 模型是一种**面向吞吐量**的性能分析框架，它将算法性能受限于两个互斥资源：

1. **内存带宽（Bandwidth Roof）**——单位时间内可从 HBM（High Bandwidth Memory）读写数据的最大字节数，记为 $\beta_{peak}$（GB/s）。
2. **峰值算力（Compute Roof）**——单位时间内 Tensor Core / CUDA Core 可完成的浮点运算数，记为 $\pi_{peak}$（TFLOPS）。

算法在这两个极限之间处于哪种 regime，由其**算术强度（Arithmetic Intensity, AI）**决定：

$$
AI = \frac{\text{总浮点运算数 (FLOPs)}}{\text{总 HBM 访存量 (Bytes)}}
$$

单位为 $\text{FLOP}/\text{Byte}$。Roofline 的"屋顶"形状为分段函数：

$$
P_{roofline}(AI) = \min(\pi_{peak},\; \beta_{peak} \times AI)
$$

其几何意义如下：

| 概念 | 定义 | 图示位置 |
|------|------|---------|
| **Memory-bound regime** | $AI < AI_{ridge}$，性能由斜率为 $\beta_{peak}$ 的直线限制 | Roofline 左侧斜线区域 |
| **Compute-bound regime** | $AI > AI_{ridge}$，性能由水平线 $\pi_{peak}$ 限制 | Roofline 右侧平顶区域 |
| **Ridge Point（脊点）** | $AI_{ridge} = \pi_{peak} / \beta_{peak}$，带宽与算力限制的交界 | 斜线与水平线的交点 |

> **工程直觉**: 若算法位于 ridge point 左侧，再增加 Tensor Core 算力也**无济于事**；必须减少 HBM 流量或提高 $AI$。FlashAttention 的核心价值正是通过 tiling 与在线 softmax 将 Attention 从 ridge point 的极左侧向右推移，但仍处于 memory-bound 区间。

---

## 2. 目标 GPU 的理论峰值

以下数值均为厂商标称的**dense FP16 Tensor Core**峰值，非稀疏、非低精度（INT8/FP8）。

| GPU | 架构 | HBM 带宽 $\beta_{peak}$ | FP16 算力 $\pi_{peak}$ | Ridge Point $AI_{ridge}$ | TDP |
|:---|:---|:---:|:---:|:---:|:---:|
| NVIDIA V100 | Volta (`sm_70`) | 900 GB/s | 31.4 TFLOPS | 34.9 FLOP/Byte | 300 W |
| NVIDIA A100 | Ampere (`sm_80`) | 2,039 GB/s | 312 TFLOPS | 153 FLOP/Byte | 400 W |
| NVIDIA H100 | Hopper (`sm_90`) | 3,350 GB/s | 989 TFLOPS | 295 FLOP/Byte | 700 W |

### 2.1 Ridge Point 的工程含义

| GPU | Ridge Point 解读 |
|:---|:---|
| V100 | 每从 HBM 读取 1 Byte，必须至少做 35 次 FP16 运算才能"回本"进入 compute-bound。否则性能被带宽锁死。 |
| A100 | Ampere 的 Tensor Core 算力提升近 10×，但带宽仅提升 2.3×，ridge point 大幅右移至 153。这意味着大量传统 kernel（GEMM 以外的）在 A100 上更容易落入 memory-bound。 |
| H100 | Hopper 的 ridge point 达到 295。FlashAttention-3 引入的 TMA + WGMMA 本质上是**在硬件层面进一步减少 HBM 流量**，从而将有效 $AI$ 向右推移，逼近 ridge point。 |

---

## 3. FlashAttention 算术强度推导

### 3.1 标准 Attention 的计算与访存

对于输入 $Q, K, V \in \mathbb{R}^{B \times H \times N \times d}$，标准 Attention（无 tiling，materialize 中间矩阵）的计算流程为：

$$
S = QK^T \in \mathbb{R}^{B \times H \times N \times N}, \quad P = \text{softmax}(S) \in \mathbb{R}^{B \times H \times N \times N}, \quad O = PV \in \mathbb{R}^{B \times H \times N \times d}
$$

- **总 FLOPs**: $FLOPs_{std} = 2 \cdot B \cdot H \cdot N^2 \cdot d \;(\text{GEMM } S) + 5 \cdot B \cdot H \cdot N^2 \;(\text{softmax}) + 2 \cdot B \cdot H \cdot N^2 \cdot d \;(\text{GEMM } O)$

  简化后主导项为：
  $$
  FLOPs_{std} \approx 4 \cdot B \cdot H \cdot N^2 \cdot d
  $$

- **总 HBM 访存**: 需读写 $Q, K, V, S, P, O$ 共 6 个张量。其中 $S, P$ 为 $N \times N$。

  $$
  Bytes_{std} \approx 2 \cdot B \cdot H \cdot N \cdot d \; (Q,K,V,O) + 4 \cdot B \cdot H \cdot N^2 \; (S,P)
  $$

  当 $N \gg d$ 时（如 $N=8192, d=64$），$Bytes_{std}$ 由 $O(N^2)$ 项主导。

- **算术强度**:
  $$
  AI_{std} = \frac{4 \cdot B \cdot H \cdot N^2 \cdot d}{4 \cdot B \cdot H \cdot N^2 + \text{低阶项}} \approx d \quad \text{当 } N \to \infty
  $$

  代入 $d=64$：
  $$
  AI_{std} \approx 64 \; \text{FLOP/Byte}
  $$

### 3.2 FlashAttention 的计算与访存

FlashAttention（以本实现 v0.3.0 为例，采用 online softmax + tiling，无中间矩阵 materialize）的核心不变量为：

- 将 $Q, K, V$ 分块为 SRAM 可容纳的 tile（如 $B_r \times d$, $B_c \times d$）。
- 仅输出 $O$ 写回 HBM；中间量 $S, P$ 在 SRAM 内生成、消费、丢弃。
- Online softmax 维护两个统计量：row max $m$ 与 row sum $l$。

**访存分析**:

| 数据 | 大小 | 方向 | 次数 | 说明 |
|------|------|------|------|------|
| $Q$ | $B \cdot H \cdot N \cdot d$ | HBM $\to$ SRAM | 1 | 逐 tile 读取 |
| $K$ | $B \cdot H \cdot N \cdot d$ | HBM $\to$ SRAM | $\lceil N / B_c \rceil$ | 外循环重载 |
| $V$ | $B \cdot H \cdot N \cdot d$ | HBM $\to$ SRAM | $\lceil N / B_c \rceil$ | 与 $K$ 同步加载 |
| $O$ | $B \cdot H \cdot N \cdot d$ | SRAM $\to$ HBM | 1 | 最终输出 |
| $m, l$ | $B \cdot H \cdot N$ | SRAM $\leftrightarrow$ HBM | 0 (SRAM 驻留) | 本实现在 tile 迭代中驻留 SRAM |

因此，对于 causal mask 场景（本实现支持），$K, V$ 的读取次数因下三角结构减半，总 HBM 流量近似为：

$$
Bytes_{FA} \approx \underbrace{2 \cdot B \cdot H \cdot N \cdot d}_{Q,O} + \underbrace{2 \cdot B \cdot H \cdot N \cdot d \cdot \frac{N}{B_c} \cdot \frac{1}{2}}_{K,V\;\text{causal 减半}} \approx 2 \cdot B \cdot H \cdot N \cdot d + B \cdot H \cdot N \cdot d \cdot \frac{N}{B_c}
$$

在典型 tiling 参数下（$B_c = 128$ 或 256），当 $N \gg B_c$ 时，第二项（$K, V$ 重载）不可忽略，但仍远小于 $O(N^2)$ 的 materialized $S, P$。

**更简洁的上界估算**（参考 FlashAttention 原始论文）：

$$
Bytes_{FA} \approx \Theta(B \cdot H \cdot N \cdot d)
$$

即 FlashAttention 的 HBM 流量从 $O(N^2)$ 降至 $O(N)$。

**算术强度**:

$$
AI_{FA} = \frac{FLOPs_{FA}}{Bytes_{FA}} \approx \frac{4 \cdot B \cdot H \cdot N^2 \cdot d}{c \cdot B \cdot H \cdot N \cdot d} = \frac{4N}{c}
$$

其中 $c$ 为与 tiling 大小相关的常数（$c \approx 4 \sim 8$，取决于 $B_c, B_r$ 与 causal 掩码减少的访存）。

代入 $N=8192, c=6$：

$$
AI_{FA} \approx \frac{4 \times 8192}{6} \approx 5460 \; \text{FLOP/Byte}
$$

> **注意**: 上述 $AI_{FA}$ 是**理论上限**，假设 $K, V$ 完全复用、无额外 index 计算开销。实际 kernel 中，causal mask 的边界判断、softmax 的 online rescaling、以及 SRAM bank conflict 会导致有效 $AI$ 下降 20%–40%。

### 3.3 为什么 FlashAttention 仍是 Memory-bound

尽管 $AI_{FA} \approx 5460$ 看起来远大于 A100 的 ridge point（153），但在 Roofline 模型中必须区分**算法算术强度**与**有效算术强度**：

| 因素 | 对 $AI$ 的影响 | 说明 |
|------|--------------|------|
| Causal mask 不规则访存 | 降低 10%–20% | 下三角导致每个 query tile 需处理的 key tile 数量递减，warp 利用率不均 |
| Online softmax 额外 FLOPs | 提升 $AI$ | 重缩放、max 更新、log-sum-exp 增加少量计算，但不显著增加访存 |
| SRAM $\to$ Register / Shared Mem 流量 | **不纳入 HBM 流量** | Roofline 模型若使用 HBM-only 字节数，会高估 $AI$；若使用**全部内存层级流量**（含 shared memory），$AI$ 会大幅下降 |
| 小 head_dim（$d=32$） | 降低 $AI$ | 每个元素的计算量减少，tiling 粒度受限 |

**工程结论**: 在严格的 HBM-only Roofline 意义下，FlashAttention 的实测 $AI_{effective}$ 落在 **50–150 FLOP/Byte** 区间（见第 5 节实测表）。这意味着：

- 对于 V100（$AI_{ridge}=35$），FlashAttention 接近 ridge point，部分配置已触及 compute-bound 边缘。
- 对于 A100/H100（$AI_{ridge}=153/295$），FlashAttention **仍位于 memory-bound 区域**，但已非常接近 ridge point。

> **面试核心论点**: FlashAttention 的优化目标不是"变成 compute-bound"，而是"在 memory-bound 中做到最好"——通过 tiling 消除 $O(N^2)$ 的 HBM 流量，使得性能由**带宽上限** $P = \beta_{peak} \times AI$ 决定，而非由 $O(N^2)$ 的容量瓶颈决定。

---

## 4. Tiling 如何提高算术强度并减少 HBM 流量

### 4.1 无 Tiling 的访存灾难

以 $N=16384, d=64, B=1, H=8$ 为例：

| 指标 | 标准 Attention | FlashAttention (tiled) |
|:---|---:|---:|
| $S = QK^T$ 大小 | $8 \times 16384^2 \times 2 \text{ Bytes} = 4.29 \text{ GB}$ | 0（SRAM 内消纳） |
| $P = \text{softmax}(S)$ 大小 | $4.29 \text{ GB}$ | 0（SRAM 内消纳） |
| 总 HBM 激活内存 | ~8.6 GB（仅 $S, P$）+ 64 MB（$Q,K,V,O$） | ~260 MB（仅 $Q,K,V,O$ 与 tile buffer） |
| HBM 流量（读+写）| ~17.2 GB（单次前向） | ~520 MB（单次前向） |
| 算术强度 $AI$ | $\approx d = 64$ | $\approx 200$–$800$（有效值） |

Tiling 的内存减幅达到 **30×–60×**，这是 FlashAttention 能处理长序列的根本原因。

### 4.2 Tiling 的算术强度提升机制

Tiling 提高 $AI$ 的本质是**数据复用（Data Reuse）**：

$$
AI = \frac{\text{FLOPs}}{\text{HBM Bytes}} = \frac{\text{FLOPs per tile} \times \text{num tiles}}{\text{HBM Bytes per tile} \times \text{num tiles}} \xrightarrow{\text{reuse}} \frac{\text{FLOPs per tile}}{\text{HBM Bytes per tile} / \text{reuse factor}}
$$

在 FlashAttention 中：

- 一个 $Q$ tile（$B_r \times d$）与所有 $K$ tiles 计算内积，产生 $B_r \times N$ 的局部 $S$ 行。
- 每个 $K$ tile（$B_c \times d$）被加载到 SRAM 后，服务于**多个** $Q$ tiles（若 non-causal）或**递减数量**的 $Q$ tiles（若 causal）。
- 计算量随 $B_r \times B_c \times d$ 增长，而 HBM 流量仅随 $B_r \times d + B_c \times d$ 增长。

**SRAM 容量约束**:

设 SRAM 大小为 $M_{SRAM}$（A100 每 SM 为 164 KB，可被多个 block 分区使用），则 tiling 需满足：

$$
\underbrace{B_r \times d}_{Q\;tile} + \underbrace{2 \times B_c \times d}_{K,V\;tiles} + \underbrace{B_r \times B_c}_{S\;tile} + \underbrace{B_r}_{m\;vector} + \underbrace{B_r}_{l\;vector} + \underbrace{B_r \times d}_{O\;accumulator} \leq M_{SRAM}
$$

本实现 v0.3.0 选取 $B_r = 128, B_c = 128, d=64$，则 SRAM 占用约为：

$$
128 \times 64 + 2 \times 128 \times 64 + 128 \times 128 + 128 + 128 + 128 \times 64 = 8\text{K} + 16\text{K} + 16\text{K} + 0.5\text{K} + 0.5\text{K} + 8\text{K} \approx 49\text{KB}
$$

远小于 164 KB，留有余量给编译器插入的临时变量与 bank conflict 规避 padding。

---

## 5. 实测带宽利用率与 Roofline 定位

### 5.1 有效带宽利用率

以下数据基于 [基准测试](./benchmarks.md) 的实测 kernel-only 时间，结合 `nvprof` / `ncu` 采集的 HBM 流量统计。测试配置：batch=1, heads=8, head_dim=64, causal FP16。

| GPU | seq_len | 实测时间 (ms) | 理论 FLOPs | 实测 TFLOPS | 理论 HBM 流量 (GB) | 有效带宽 (GB/s) | 峰值带宽利用率 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| V100 | 1,024 | 0.42 | 2.15 | 2.1 | 0.23 | 548 | 61% |
| V100 | 4,096 | 5.82 | 34.4 | 2.8 | 0.92 | 630 | 70% |
| V100 | 8,192 | 22.50 | 137.4 | 3.0 | 1.84 | 651 | 72% |
| V100 | 16,384 | 88.0 | 549.8 | 3.1 | 3.68 | 670 | 74% |
| A100 | 1,024 | 0.19 | 2.15 | 4.5 | 0.23 | 1,211 | 59% |
| A100 | 4,096 | 2.18 | 34.4 | 7.5 | 0.92 | 1,631 | 80% |
| A100 | 8,192 | 7.80 | 137.4 | 8.5 | 1.84 | 1,855 | 91% |
| A100 | 16,384 | 28.5 | 549.8 | 9.3 | 3.68 | 1,957 | 96% |
| H100 | 1,024 | 0.11 | 2.15 | 8.2 | 0.23 | 2,091 | 62% |
| H100 | 4,096 | 1.15 | 34.4 | 14.2 | 0.92 | 3,020 | 90% |
| H100 | 8,192 | 3.85 | 137.4 | 17.3 | 1.84 | 3,247 | 97% |
| H100 | 16,384 | 13.2 | 549.8 | 20.1 | 3.68 | 3,350 | **100%** |

### 5.2 Roofline 图上定位

基于上表计算有效算术强度 $AI_{eff} = \text{实测 TFLOPS} \times 10^{12} / (\text{有效带宽} \times 10^9)$，并在 Roofline 坐标系中标定：

| GPU | seq_len | $AI_{eff}$ (FLOP/Byte) | Roofline Regime | 距离 Ridge Point |
|:---|:---:|:---:|:---|:---|
| V100 | 1,024 | 3.8 | Deep memory-bound | 9.2× 低于 ridge |
| V100 | 4,096 | 4.4 | Deep memory-bound | 7.9× 低于 ridge |
| V100 | 8,192 | 4.6 | Deep memory-bound | 7.6× 低于 ridge |
| V100 | 16,384 | 4.6 | Deep memory-bound | 7.6× 低于 ridge |
| A100 | 1,024 | 3.7 | Deep memory-bound | 41× 低于 ridge |
| A100 | 4,096 | 4.6 | Deep memory-bound | 33× 低于 ridge |
| A100 | 8,192 | 4.6 | Deep memory-bound | 33× 低于 ridge |
| A100 | 16,384 | 4.8 | Deep memory-bound | 32× 低于 ridge |
| H100 | 1,024 | 3.9 | Deep memory-bound | 76× 低于 ridge |
| H100 | 4,096 | 4.7 | Deep memory-bound | 63× 低于 ridge |
| H100 | 8,192 | 5.3 | Deep memory-bound | 56× 低于 ridge |
| H100 | 16,384 | 6.0 | Deep memory-bound | 49× 低于 ridge |

> **关键洞察**: $AI_{eff}$ 仅约 4–6 FLOP/Byte，远低于所有 GPU 的 ridge point。这意味着本实现 v0.3.0 的**有效**性能受限于带宽，但带宽利用率随 seq_len 增加而提高（因为固定开销被摊薄）。

### 5.3 为什么 $AI_{eff}$ 与理论 $AI_{FA}$ 差距巨大

理论上节推导 $AI_{FA} \approx 5460$ FLOP/Byte，而实测仅 4–6 FLOP/Byte，差距约 **1000×**。原因如下：

| 因素 | 影响量级 | 解释 |
|------|:-------:|------|
| **HBM 流量定义差异** | $\times 50$–$100$ | 理论推导中 $Bytes_{FA}$ 仅计 $Q,K,V,O$；但实测中 ncu 统计的 HBM 流量包含：atomics、reduction scratchpad、kernel 启动参数、以及 PyTorch/CUDA context 的隐性流量。更关键的是，**shared memory 流量未被计入**，而 FlashAttention 的 tile 计算在 SRAM 内产生大量内部流量。 |
| **Causal mask 不规则性** | $\times 1.5$–$2$ | Causal mask 导致大量 warp 内线程闲置（padding 至三角形边界），有效 FLOPs 降低。 |
| **Online softmax 额外访存** | $\times 1.2$ | $m, l$ 向量的频繁读写（即使驻留 SRAM，也有 register spilling 到 local memory 的情况）。 |
| **短序列固定开销** | $\times 2$–$4$ | `seq_len=1K` 时，kernel launch、grid setup、边界条件判断的 overhead 占比极高。 |

**修正后的 Roofline 分析应采用如下口径**：

$$
AI_{HBM\text{-}only} = \frac{4 \cdot B \cdot H \cdot N^2 \cdot d}{2 \cdot B \cdot H \cdot N \cdot d \; (Q,O) + 2 \cdot B \cdot H \cdot N \cdot d \; (K,V\;\text{单次})} \approx N
$$

若以 $N=8192$ 计算，$AI_{HBM\text{-}only} \approx 8192$ FLOP/Byte，仍高于 ridge point。实测差距主要来源于：

1. **本实现 v0.3.0 尚未实现 FlashAttention-2 的 split-K / sequence-parallel 优化**，导致 $K, V$ 的重载次数高于理论下限。
2. **Google Benchmark 的 timer 精度与 warm-up 策略**在短序列下引入系统误差。
3. **FP16 的 Tensor Core 利用率**: 本实现的 warp-level GEMM 使用手工编排的 HMMA 指令，但在小 $d$（32/64）时无法充分填满 MMA 单元，导致实际算力远低于 $\pi_{peak}$。

---

## 6. 标准 Attention vs FlashAttention 的 Roofline 对比

### 6.1 同一坐标系下的定位

以 A100（$\beta_{peak}=2039$ GB/s, $\pi_{peak}=312$ TFLOPS, $AI_{ridge}=153$）为基准：

```
Performance (TFLOPS)
    |
312 |______________________________  Compute Roof (Flat)
    |                             /
    |                           /
    |                         /
    |                       /   <-- Ridge Point @ AI=153
    |                     /
    |                   /
    |                 /
    |               /
    |             /  <-- Bandwidth Roof (Slope = 2039 GB/s)
    |           /
    |         /
    |       /
    |     /
    |   /
    | /
    +-----------------------------------> AI (FLOP/Byte)
      1    10    50   100   153   500   1000

Standard Attention (seq=16K):  X @ AI≈64,  P≈0.13 TFLOPS
FlashAttention v0.3.0 (seq=16K): O @ AI≈5*  P≈9.3 TFLOPS
FlashAttention-2 (参考):        △ @ AI≈80*, P≈80+ TFLOPS

* 有效 AI（含全部内存层级）
```

### 6.2 对比汇总表

| 维度 | 标准 Attention (Materialized) | CuFlash-Attn v0.3.0 | FlashAttention-2/3 (生产级) |
|:---|:---|:---|:---|
| $AI$ (HBM-only) | $O(d) \approx 64$ | $O(N/d) \approx 5000$ | $O(N/B_c) \approx 5000$ |
| $AI_{eff}$ (全内存层级) | $\approx 3$–$5$ | $\approx 4$–$6$ | $\approx 50$–$150$ |
| HBM 流量 scaling | $O(N^2)$ | $O(N)$ | $O(N)$ |
| A100 峰值带宽利用率 | 20%–35% | 60%–96% | 85%–110% |
| A100 实测 TFLOPS | 1.5–3.0 | 4.5–9.3 | 80–150+ |
| 最大 seq_len (40GB) | ~8K–16K | ~64K | ~128K–256K |
| Roofline Regime | Deep memory-bound, 低效 | Memory-bound, 中等效率 | Near ridge point / 部分 compute-bound |

### 6.3 定性结论

1. **标准 Attention** 位于 Roofline 极左下角。即使给它无限算力，也无法突破 $P = \beta \times AI$ 的斜线限制；且 $AI$ 固定为 $O(d)$，不随 $N$ 增长，**不具备 scaling 潜力**。

2. **CuFlash-Attn v0.3.0** 通过 tiling 将 $AI$ 提升数个数量级，但受限于参考级实现的手工程度，未能完全消除多余 HBM 流量与 warp 闲置。其性能位于 Roofline 斜线上段，距离 ridge point 仍有一个数量级的差距。

3. **FlashAttention-2/3** 通过以下手段进一步右移 $AI$：
   - **Split-K / Sequence Parallel**: 将 $K, V$ 的冗余加载分摊到多个 warp group。
   - **Grouped GEMM / Warp Specialization**: 减少 softmax 与 GEMM 之间的流水线气泡。
   - **TMA (Hopper) / cp.async (Ampere)**: 异步预取隐藏 HBM 延迟。
   - **精确 causal mask 处理**: 避免 tile 内的无效计算与访存。

   这些优化使得生产级 FlashAttention 在 A100 上可达到 ridge point 附近，在 H100 上配合 TMA 甚至部分进入 compute-bound regime。

---

## 7. 优化路线图（从 Roofline 视角）

| 阶段 | 目标 $AI_{eff}$ | 手段 | 预期 A100 带宽利用率 | 难度 |
|:---|:---:|:---|:---:|:---:|
| v0.3.0 (当前) | 4–6 | 基础 tiling + online softmax | 60%–96% | 基线 |
| v0.4.0 | 15–25 | `cp.async` 预取、更优 warp 调度、causal mask 边界优化 | 85%–100% | 中 |
| v0.5.0 | 40–80 | Split-K sequence parallel、warp-group 级 reduction、减少 bank conflict | 95%–110% | 高 |
| v1.0.0 (未来) | 100+ | CUTLASS 集成或 TMA/WGMMA 重写（Hopper） | 接近 ridge point | 极高 |

---

## 8. 参考公式速查

| 符号 | 定义 | 单位 |
|:---|:---|:---|
| $N$ | 序列长度（`seq_len`） | — |
| $d$ | 头维度（`head_dim`） | — |
| $B$ | batch size | — |
| $H$ | 注意力头数 | — |
| $B_r, B_c$ | Query / Key-Value tile 大小 | — |
| $\beta_{peak}$ | HBM 峰值带宽 | GB/s |
| $\pi_{peak}$ | FP16 Tensor Core 峰值算力 | TFLOPS |
| $AI$ | 算术强度 = FLOPs / Bytes | FLOP/Byte |
| $AI_{ridge}$ | Ridge point = $\pi_{peak} / \beta_{peak}$ | FLOP/Byte |
| $P_{roofline}$ | Roofline 性能上限 = $\min(\pi_{peak}, \beta_{peak} \times AI)$ | TFLOPS |

---

## 9. 推荐阅读

1. Williams, S., Waterman, A., & Patterson, D. (2009). *Roofline: An insightful visual performance model for multicore architectures*. Communications of the ACM.
2. Dao, T., et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS.
3. Dao, T., et al. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*.
4. NVIDIA. (2022). *CUDA C++ Programming Guide* — Compute Capability 8.0/9.0 Architecture Details.
