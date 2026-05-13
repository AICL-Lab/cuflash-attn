# 设计决策

本文档以 ADR（Architecture Decision Record）格式记录 CuFlash-Attn 的关键技术决策，每项决策包含背景、决策内容、利弊权衡与引用来源。所有记录均为最终状态（Accepted），不再修改。

---

## 目录

- [ADR-1: head_dim 限制为 {32, 64, 128}](#adr-1-head_dim-限制为-32-64-128)
- [ADR-2: FP16 内部使用 FP32 Accumulation](#adr-2-fp16-内部使用-fp32-accumulation)
- [ADR-3: ctypes 而非 pybind11](#adr-3-ctypes-而非-pybind11)
- [ADR-4: OpenSpec 驱动开发](#adr-4-openspec-驱动开发)
- [ADR-5: 固定 block size 128](#adr-5-固定-block-size-128)
- [ADR-6: O(N) Tiling 优于标准 O(N²) Attention](#adr-6-on-tiling-优于标准-on²-attention)

---

## ADR-1: head_dim 限制为 {32, 64, 128}

### Context

FlashAttention kernel 的核心约束是共享内存容量。每个 block 需同时驻留 Q_tile、K_tile、V_tile、S_tile、O_tile 及 softmax 状态向量。共享内存占用与 `BLOCK_M × HEAD_DIM`、`BLOCK_N × HEAD_DIM`、`BLOCK_M × BLOCK_N` 均成正比。若将 `head_dim` 泛化为运行时任意值，则 tile 尺寸无法在编译期确定，导致：

1. 循环边界变为运行时变量，编译器无法展开内层 dot-product 循环；
2. 共享内存偏移无法以常量表达式计算，寄存器寻址效率下降；
3. 共享内存总量不可预测，难以在启动前完成容量检查。

Transformer 生态中，head_dim 的取值具有高度聚类性：标准 BERT/GPT 使用 64，ViT 变体使用 64 或 128，部分轻量模型使用 32。96、160 等非 2 的幂次值虽存在，但占比极低。

### Decision

`head_dim` 仅支持编译期枚举值 `{32, 64, 128}`，通过模板显式实例化：

```cpp
template __global__ void flash_attention_forward_kernel<64, 64, 32>(...);
template __global__ void flash_attention_forward_kernel<64, 64, 64>(...);
template __global__ void flash_attention_forward_kernel<32, 32, 128>(...);
```

运行时通过 `if (head_dim == 32) ... else if (head_dim == 64) ... else if (head_dim == 128)` 分发到对应模板特化。

### Consequences

| 维度 | 影响 |
|------|------|
| **性能** | ✅ 编译期常量使内层循环完全展开，dot-product 可被编译器优化为 FMA 指令链。共享内存索引退化为常量偏移，消除动态地址计算指令。 |
| **可维护性** | ✅ 模板实例化代码路径有限（前向 3 条 × 2 数据类型 + 反向 4 条 × 2 数据类型），测试覆盖完备。 |
| **通用性** | ❌ 不支持 96、160、256 等非常见 head_dim，用户需 padding 或自行修改 kernel。 |
| **编译时间** | ⚠️ 模板实例化数量线性增长，当前规模下编译时间可控（< 30s）。若未来扩展至 8+ 个 head_dim，需考虑拆分编译单元。 |

### References

- `src/forward/flash_attention_forward.cu:169-178` — 模板显式实例化
- `src/api/flash_attention_api.cu:66-68` — 运行时 head_dim 校验
- `openspec/specs/design/flash-attention-design.md` — Block Configuration 章节

---

## ADR-2: FP16 内部使用 FP32 Accumulation

### Context

FP16（IEEE 754 half-precision）的表示范围约为 `±6.55×10⁴`，精度为 10 位有效位（约 3 位十进制）。Attention 计算中的典型风险点包括：

1. `Q @ K^T` 的 dot-product：序列长度较大时，累加项数 `head_dim` 可达 128，FP16 累加会导致显著舍入误差；
2. `exp(x)`：当 attention score 较大（如 `x > 10`）时，FP16 的有限动态范围易导致上溢；
3. Online softmax 的迭代修正：`exp(m_old - m_new)` 中，若 `m_old` 与 `m_new` 差距大，FP16 可能下溢为 0。

FlashAttention-2 官方实现同样采用 FP16 输入、FP32 TensorCore/FFMA 累加的策略，仅在最终写回时降级为 FP16。

### Decision

FP16 kernel 的共享内存与寄存器累加器全部使用 `float`，仅在 GMEM 边界进行 `half ↔ float` 转换：

```cpp
// Load: GMEM half → SMEM float
Q_tile[i] = __half2float(Q_ptr[global_row * HEAD_DIM + local_col]);

// Compute: 全部在 float 域
float sum = 0.0f;
for (int k = 0; k < HEAD_DIM; k++) {
    sum += Q_tile[row * HEAD_DIM + k] * K_tile[col * HEAD_DIM + k];
}

// Store: SMEM float → GMEM half
O_ptr[global_row * HEAD_DIM + d] = __float2half(O_tile[...] * l_inv);
```

### Consequences

| 维度 | 影响 |
|------|------|
| **数值稳定性** | ✅ 与纯 FP32 实现的相对误差 < 1e-3，通过全部集成测试。FP16 下溢/上溢风险被完全消除。 |
| **带宽** | ✅ GMEM 读写带宽减半（2 bytes/element vs 4 bytes），在长序列场景下显著降低 HBM 压力。 |
| **计算吞吐** | ⚠️ 未使用 TensorCore，FP32 FFMA 吞吐为 FP16 的 1/2（Ampere 架构）。这是教育/参考库的可接受权衡。 |
| **共享内存** | ⚠️ 共享内存以 float 分配，FP16 kernel 的 SMEM 占用与 FP32 相同，无法利用 FP16 压缩 SMEM。 |
| **寄存器** | ⚠️ 额外引入 `__half2float` / `__float2half` 转换指令，但 Volta+ 上为单周期吞吐指令，无实质影响。 |

### References

- `src/forward/flash_attention_fp16.cu:11-17` — `half_to_float` / `float_to_half` 辅助函数
- `src/forward/flash_attention_fp16.cu:54-63` — Load 阶段转换
- `openspec/specs/verification/flash-attention-verification.md` — Property 6: Data Type Support
- Dao, Tri. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR 2024.

---

## ADR-3: ctypes 而非 pybind11

### Context

项目需要提供 Python 调用接口，供深度学习开发者集成。可选方案：

1. **pybind11**：C++ 头文件库，提供 C++ 到 Python 的自动绑定，语法简洁，支持 `torch.Tensor` 的直接透传。
2. **ctypes**：Python 标准库，通过 `CDLL` 加载共享库，手动声明函数签名与类型转换。
3. **Cython**：编译型 Python 扩展，性能最优，但引入额外构建依赖。

CuFlash-Attn 的项目定位是"从零实现的 CUDA C++ 参考库"，核心约束为"零外部依赖"。

### Decision

使用 `ctypes` 作为唯一 Python 绑定机制。示例代码位于 `examples/python_binding.py`：

```python
import ctypes
lib = ctypes.CDLL("./build/release/libcuflash_attn.so")

lib.cuflash_attention_forward_f32.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_bool, ctypes.c_void_p,
]
lib.cuflash_attention_forward_f32.restype = ctypes.c_int
```

### Consequences

| 维度 | 影响 |
|------|------|
| **依赖** | ✅ 零外部依赖。用户无需 `pip install pybind11` 或编译 Cython 扩展。纯标准库即可调用。 |
| **构建** | ✅ 无需额外的 Python 头文件路径或 `setup.py` 编译逻辑。CMake 仅产出 `.so`，Python 直接加载。 |
| **兼容性** | ✅ 同时兼容 CuPy、PyTorch、Numba 等框架，只要这些框架能提供 `c_void_p` 级别的设备指针。 |
| **类型安全** | ❌ 函数签名在 Python 端手动声明，若参数顺序或类型错误，崩溃发生在 C 层，调试困难。 |
| **工程效率** | ❌ 新增 C API 时，需同步更新 Python 端的 `argtypes` / `restype` 声明，维护成本高于 pybind11 的自动推导。 |
| **性能** | ⚠️ 函数调用本身为 C 层直接调用，无额外包装开销；主要开销在 Python 循环中调用多次的场景，但 FlashAttention 为粗粒度 kernel launch，单次调用即完成整个 attention 计算。 |

### References

- `examples/python_binding.py` — 完整 ctypes 绑定示例
- `src/api/flash_attention_api.cu:150-187` — C ABI 导出接口
- `openspec/config.yaml` — 零依赖哲学（anti-pattern: 引入非标准库绑定）

---

## ADR-4: OpenSpec 驱动开发

### Context

参考级 CUDA 项目面临的核心风险是"代码与文档漂移"：实现细节、接口语义、测试覆盖随时间发散，最终仓库中唯一可信的来源是代码本身，而设计 rationale 散落于 PR 描述和 Issue 回复中。传统的 Wiki 或独立 README 无法强制要求代码变更前更新文档。

### Decision

采用 OpenSpec 规范优先治理模型：

1. `openspec/specs/` 为唯一真相来源（Single Source of Truth），包含产品需求、技术设计、API 契约、验证标准；
2. 任何行为或 API 变更必须先创建变更提案（`/opsx:propose`），包含 `proposal.md`、`design.md`、`tasks.md`；
3. 代码实现后通过 `/verify` 检查格式、构建、测试，最终归档到 `openspec/changes/archive/`；
4. 测试注释必须引用 spec ID（如 `// Validates REQ-1.1`）。

```
openspec/
├── specs/
│   ├── design/flash-attention-design.md      # 需求 + 技术设计
│   └── verification/flash-attention-verification.md  # API + 测试规范
├── changes/
│   └── archive/                               # 已完成变更归档
└── config.yaml                                # 项目规则与 anti-patterns
```

### Consequences

| 维度 | 影响 |
|------|------|
| **可追溯性** | ✅ 每一行关键代码均可追溯到具体需求（REQ-X.Y）与验证属性（Property N）。 |
| **协作一致性** | ✅ AI 代理、人类贡献者、CI 系统共享同一套规范语义，减少"我以为 / 你以为"的沟通成本。 |
| **变更成本** | ⚠️ 轻量 bug 修复亦需标注 spec ID，初期增加文档开销；长期看，回归测试与新人 onboarding 成本显著降低。 |
| **灵活性** | ❌ 禁止 gold-plating：规范中未定义的功能不得实现。这对参考库是 feature 而非 bug，但限制了实验性分支的敏捷性。 |
| **归档价值** | ✅ 项目进入稳定基线（v0.3.0）后，OpenSpec archive 本身即构成完整的技术演进史。 |

### References

- `openspec/specs/index.md` — 规范索引
- `openspec/config.yaml` — 项目规则
- `AGENTS.md` — OpenSpec 工作流完整描述
- `CLAUDE.md` — 规范优先协作指南

---

## ADR-5: 固定 block size 128

### Context

CUDA kernel 的 block size 决定每 block 线程数，直接影响：

1. **Occupancy**：每 SM 可驻留的 block 数 = floor(SM 最大线程数 / block_size)。Ampere 每 SM 2048 线程，block=128 时上限 16 blocks/SM。
2. **共享内存带宽**：共享内存按 bank 组织，128 线程对应 4 warps，可同时发起 4 路独立访问而不 bank conflict（当访问模式跨步为 1 时）。
3. **Warp-level 原语效率**：`__shfl_xor_sync` 在 warp 内通信，block size 为 warp size 整数倍时保证 warp 完整。
4. **寄存器分配**：`__launch_bounds__(128)` 向编译器提供寄存器预算的上界约束。

常见备选值：64（occupancy 高但并行度不足）、256（共享内存/寄存器压力大）、512（仅适合无 SMEM 的纯计算 kernel）。

### Decision

所有 kernel（前向、反向、辅助 kernel `compute_D_kernel`）统一使用 `blockDim.x = 128`，并标注 `__launch_bounds__(128)`：

```cpp
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128)
flash_attention_forward_kernel(...);
```

### Consequences

| 维度 | 影响 |
|------|------|
| **Occupancy** | ✅ 128 为 32 的整数倍，warp 完整；Ampere/Hopper 上每 SM 可驻留最多 16 个 block，SM 利用率充足。 |
| **共享内存** | ✅ 128 线程的 load/store 循环步长（`i += 128`）恰好覆盖较大 tile，共享内存 bank conflict 概率低。 |
| **寄存器** | ✅ `__launch_bounds__(128)` 使编译器在 128-thread 假设下收紧寄存器预算，提升 block 并发数。 |
| **灵活性** | ❌ 固定 128 意味着无法根据 head_dim 或 seq_len 动态选择 64/256 以获得更优局部性。但这简化了 launcher 逻辑，对参考库是可接受权衡。 |
| **辅助 kernel** | ⚠️ `compute_D_kernel` 使用 128 线程处理逐行 dot-product，每行独立计算，128 的 block size 略有 over-provision（每行仅需一个线程），但保持统一简化调度。 |

### References

- `src/forward/flash_attention_forward.cu:17` — `__launch_bounds__(128)`
- `src/forward/flash_attention_forward.cu:195` — `dim3 block(128)`
- `src/backward/flash_attention_backward.cu:16` — 反向 kernel 同样固定 128
- `docs/zh/design/kernel-deep-dive.md` — Launch Bounds 与寄存器压力章节

---

## ADR-6: O(N) Tiling 优于标准 O(N²) Attention

### Context

标准 Attention 的瓶颈不在计算而在 IO：

- `S = Q @ K^T` 产生 `[seq_len, seq_len]` 矩阵，必须写入 HBM；
- `P = softmax(S)` 读取 S、写入 P，再次产生 O(N²) HBM 流量；
- `O = P @ V` 读取 P，写入 O。

当 `N = 4096, d = 64` 时，中间矩阵 S 与 P 各占 64 MB，而 Q/K/V 各仅 1 MB。HBM 带宽（A100 约 2 TB/s）远低于 TensorCore 算力（312 TFLOPS），IO 成为绝对瓶颈。

FlashAttention 的核心洞察是：若将 Q、K、V 切分为 SRAM 可容纳的小块，则中间结果 S、P 永不离片，HBM 流量从 O(N²) 降至 O(N²d / M)，其中 M 为 SRAM 容量。

### Decision

采用分块（tiling）+ Online Softmax 策略，kernel 内部循环迭代 KV blocks，共享内存中完成局部 attention 计算，仅向 HBM 输出最终 O 与 logsumexp L：

```cpp
for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
    load K_j, V_j to SRAM;
    S_ij = Q_i @ K_j^T * scale;          // SRAM 内
    online_softmax_update(S_ij);         // SRAM 内
    O_i = rescale(O_i) + P_ij @ V_j;   // SRAM 内
}
write O_i, L_i to HBM;
```

### Consequences

| 维度 | 影响 |
|------|------|
| **内存复杂度** | ✅ 前向/反向均为 O(N) 额外内存，支持 16K+ 序列长度在单卡运行。 |
| **HBM IO** | ✅ IO 复杂度从 O(N² + Nd) 降至 O(N²d / M)。当 M ≈ Nd（SRAM 容量接近 Q/K/V 总大小）时，接近 O(Nd) 最优。 |
| **计算量** | ⚠️ 反向传播需重新计算 P_ij（而非从 HBM 读取），增加约 33% FLOPs。但 HBM 访问延迟远高于 FFMA 计算延迟，净效果为加速。 |
| **Kernel 复杂度** | ⚠️ 需管理 Online Softmax 状态（m, l）的跨块更新，实现复杂度显著高于标准 attention。 |
| **硬件依赖** | ⚠️ 收益与 SRAM 容量强相关：Volta (96 KB) 收益小于 Hopper (228 KB)。本实现通过调整 BLOCK_M/BLOCK_N 适配不同架构。 |

### 量化对比

| 序列长度 N | 标准 Attention HBM 写 (S+P) | FlashAttention HBM 写 (仅 O+L) | HBM 读省却比例 |
|:----------:|:---------------------------:|:------------------------------:|:--------------:|
| 1,024 | 8 MB | ~16 KB | 99.8% |
| 4,096 | 128 MB | ~64 KB | 99.95% |
| 16,384 | 2 GB | ~256 KB | 99.99% |

### References

- `docs/zh/algorithm.md` — 标准 Attention 瓶颈与 FlashAttention 核心概念
- `src/forward/flash_attention_forward.cu:69-151` — 前向 tiling + online softmax 实现
- Dao et al., NeurIPS 2022, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) — IO-Awareness 理论分析
- `openspec/specs/design/flash-attention-design.md` — REQ-3: Tiling Strategy, REQ-4: Online Softmax

---

## 变更日志

| 日期 | 变更 | 作者 |
|------|------|------|
| 2026-04-29 | 创建 ADR-1 ~ ADR-6，归档 v0.3.0 全部关键决策 | CuFlash-Attn Maintainers |
