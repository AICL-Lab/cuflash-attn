# Related Work

This document surveys the academic foundations and engineering ecosystems that surround CuFlash-Attn. Every paper and repository listed here has directly informed design decisions, kernel tiling strategies, or API boundaries in this project.

---

## Table of Contents

- [Academic Papers](#academic-papers)
- [Related Repositories](#related-repositories)
- [Positioning: Why CuFlash-Attn Exists](#positioning-why-cuflash-attn-exists)

---

## Academic Papers

Papers are listed in strict chronological order to illustrate the evolution of ideas from foundational softmax numerics through distributed attention mechanisms.

---

### Online normalizer calculation for softmax

- **Authors:** Maxim Milakov, Natalia Gimelshein  
- **Venue:** arXiv preprint  
- **Year:** 2018  
- **Link:** [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)

**Core contribution:** Introduces a streaming, single-pass algorithm for computing softmax normalization statistics without materializing the full score matrix. Maintains a running maximum $m$ and running exponential sum $l$ that can be updated incrementally as new data arrives.

**Relation to this project:** The online softmax update is the mathematical backbone of FlashAttention's tiling strategy. CuFlash-Attn implements the Milakov–Gimelshein recurrence verbatim in device code, using FP32 accumulators to prevent round-off error during long-sequence reduction.

---

### Multi-Query Attention

- **Authors:** Noam Shazeer  
- **Venue:** Internal technical report (widely cited)  
- **Year:** 2019  
- **Link:** [PDF](https://arxiv.org/abs/1911.02150)

**Core contribution:** Proposes sharing a single key–value head across multiple query heads, reducing decoder memory bandwidth and KV-cache size by a factor equal to the number of query heads.

**Relation to this project:** While CuFlash-Attn currently implements standard multi-head attention, the kernel's block-parallel structure over `(batch, head)` pairs means that MQ attention can be supported with only host-side stride adjustments. The tiling logic itself is agnostic to whether K/V heads are shared.

---

### Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM

- **Authors:** Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amir Phanishayee, Matei Zaharia  
- **Venue:** SC (International Conference for High Performance Computing, Networking, Storage and Analysis)  
- **Year:** 2021  
- **Link:** [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)

**Core contribution:** Demonstrates that attention layers are memory-bandwidth bound at scale, and that tensor-parallel sharding of the MLP and attention projections is necessary but not sufficient for throughput scaling. Establishes the empirical cost model that motivates attention-specific kernel optimization.

**Relation to this project:** Megatron-LM's profiling data validate the HBM-bandwidth bottleneck that FlashAttention addresses. CuFlash-Attn's O(N) memory claim is meaningful precisely because the standard attention O(N²) footprint becomes the dominant training constraint at the sequence lengths (2K–128K) studied in this paper.

---

### FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

- **Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher R\'e  
- **Venue:** NeurIPS (Conference on Neural Information Processing Systems)  
- **Year:** 2022  
- **Link:** [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

**Core contribution:** Introduces an IO-aware tiling algorithm that computes exact attention in $O(N)$ HBM memory by fusing the $QK^T$, softmax, and $PV$ operations into a single kernel. Eliminates materialization of the $N \times N$ score and probability matrices through SRAM-resident tiles and online softmax recomputation.

**Relation to this project:** This is the primary algorithmic reference. CuFlash-Attn re-implements the forward and backward tiling algorithms from scratch in CUDA C++, using the exact block-sequencing, online softmax rescaling, and causal-mask skipping logic described in the paper.

---

### PagedAttention: vLLM

- **Authors:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica  
- **Venue:** SOSP (ACM Symposium on Operating Systems Principles)  
- **Year:** 2023  
- **Link:** [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

**Core contribution:** Proposes block-sparse KV-cache paging, analogous to virtual memory, to eliminate memory fragmentation and enable efficient sharing of KV caches across multiple decoding sequences.

**Relation to this project:** PagedAttention operates at the systems layer above the kernel. CuFlash-Attn's forward kernel accepts arbitrary pointer strides, which means a PagedAttention-style block table can be passed directly to the API without requiring internal buffer reformatting. The projects are complementary: CuFlash-Attn optimizes the per-block kernel; PagedAttention optimizes the block allocation policy.

---

### Grouped-Query Attention

- **Authors:** Joshua Ainslie, Tao Lu, Michiel de Jong, Yury Zemlyanskiy, Santiago Ontanon, Sumit Sanghai  
- **Venue:** arXiv preprint  
- **Year:** 2023  
- **Link:** [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)

**Core contribution:** Generalizes multi-query attention by grouping query heads into $G$ groups, each sharing one key–value head. Interpolates between full multi-head attention ($G = H$) and multi-query attention ($G = 1$), offering a controllable quality–efficiency trade-off.

**Relation to this project:** Like MQA, GQA is a host-side stride/scheduling concern for CuFlash-Attn. The kernel launch grid already iterates over `(batch, head)`; mapping $G$ KV heads to $H$ query heads requires only index arithmetic in the host wrapper, with zero kernel modifications.

---

### Ring Attention with Blockwise Transformers for Near-Infinite Context

- **Authors:** Hao Liu, Matei Zaharia, Pieter Abbeel  
- **Venue:** arXiv preprint  
- **Year:** 2023  
- **Link:** [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)

**Core contribution:** Distributes attention across a ring of devices, where each GPU computes local attention blocks and passes KV tiles to its neighbor. Achieves context lengths far exceeding single-GPU HBM capacity without approximating the attention computation.

**Relation to this project:** Ring Attention extends FlashAttention's single-device tiling to the cluster scale. CuFlash-Attn's kernel is the natural building block for a ring-style system: it is already block-parallel over KV tiles, so ring communication can be overlapped with the inner-GEMM execution via CUDA streams.

---

### FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

- **Authors:** Tri Dao  
- **Venue:** ICLR (International Conference on Learning Representations)  
- **Year:** 2024  
- **Link:** [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

**Core contribution:** Improves parallelism by splitting the KV sequence across warps/SMs rather than the Q sequence alone, reducing idle threads in the original FlashAttention warp schedule. Also decouples the head-dimension loop from the sequence loop to better saturate tensor cores.

**Relation to this project:** FlashAttention-2's scheduling insights inform CuFlash-Attn's warp-decomposition strategy (see [Kernel Deep Dive](/en/design/kernel-deep-dive)). While CuFlash-Attn currently uses the FlashAttention-1 block-outer-loop structure for clarity, the warp-level work partitioning follows the FA-2 principle of minimizing divergent execution within a thread block.

---

## Related Repositories

The following table compares actively maintained implementations against CuFlash-Attn on dimensions that matter to researchers and systems engineers: language, feature set, training support, and modifiability.

| Repository | Language | Stars-level | Key Feature | Training Support | Differentiation from CuFlash-Attn |
|------------|----------|-------------|-------------|------------------|-----------------------------------|
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | CUDA C++ / Python | Very High (>10k) | Production FlashAttention-1/2 with fused kernels, FP8, var-len, ALiBi | Full (forward + backward) | Production-oriented; heavy Cutlass dependency; dense feature matrix (PagedAttention, sliding window, etc.) that obscures the core algorithm. |
| [facebookresearch/xformers](https://github.com/facebookresearch/xformers) | CUDA / Triton / Python | High (>5k) | Modular transformer components with memory-efficient attention | Partial (forward focused) | PyTorch-centric; wraps multiple backends (Cutlass, Triton, CUDA); not a standalone reference implementation. |
| [pytorch/pytorch](https://github.com/pytorch/pytorch) (SDPA) | C++ / CUDA / Python | Very High (>80k) | `torch.nn.functional.scaled_dot_product_attention` with backend dispatch | Full (via FlashAttention/Cutlass backends) | Meta-framework dispatching to vendor kernels; source is entangled with PyTorch runtime; not educational. |
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | CUDA C++ | Very High (>5k) | Template-based GEMM and epilogue fusion for all NVIDIA architectures | N/A (library, not end-to-end) | Cutlass is a building block, not an attention kernel. CuFlash-Attn intentionally avoids Cutlass to demonstrate manual tiling. |
| [openai/triton](https://github.com/openai/triton) (tutorials) | Python (Triton DSL) | Very High (>11k) | Python-first GPU kernel language with FlashAttention tutorial | Tutorial-level only | Triton abstracts warp scheduling and memory layout; excellent for rapid prototyping, but hides the CUDA execution model that CuFlash-Attn exposes. |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) (inference kernels) | CUDA C++ / Python | High (>5k) | ZeRO-3, inference-optimized attention, MoE support | Inference only (for attn kernels) | Tightly coupled to DeepSpeed's parallelism runtime; kernels are not extractable as a standalone library. |

**Key observation:** No existing repository occupies the intersection of *standalone C++*, *zero external dependencies*, *full training support*, and *pedagogical clarity*. CuFlash-Attn fills this niche.

---

## Positioning: Why CuFlash-Attn Exists

CuFlash-Attn was created to satisfy three requirements that existing implementations do not simultaneously meet:

### 1. Educational Clarity

Reading the FlashAttention paper and then reading the Dao-AILab production code requires traversing thousands of lines of Cutlass template metaprogramming, Python–C++ interop, and feature-branch divergence. CuFlash-Attn provides a single, linear CUDA C++ kernel file where every variable name, shared-memory offset, and warp boundary corresponds one-to-one with the pseudocode in the original paper.

**Intended audience:**
- Graduate students implementing attention for the first time
- Researchers who need to modify the softmax reduction or tiling strategy for a new architecture
- Interview candidates who want to demonstrate end-to-end CUDA systems knowledge

### 2. Zero Dependencies

CuFlash-Attn builds with only:
- CMake (≥ 3.18)
- A CUDA Toolkit (≥ 11.4)
- A C++17 compiler

There is no PyTorch, no Cutlass, no Triton, no Python, and no third-party linear-algebra library. This means:
- The build completes in seconds, not minutes.
- The binary can be deployed in environments where Python is unavailable (embedded inference runtimes, HPC clusters with custom MPI stacks).
- Every FLOP and every byte of HBM traffic is accountable to code written in this repository.

### 3. Easy to Modify

Because the kernel is hand-written rather than generated, modifications are localized and traceable:

| Desired Change | Where to Edit | Lines of Code |
|----------------|---------------|---------------|
| Change tile size $B_r \times B_c$ | Template parameters in `flash_attention_kernel.cuh` | 1 |
| Add a custom softmax temperature schedule | `online_softmax_update` device function | 5 |
| Swap causal mask for a custom block-sparse pattern | Causal skip logic inside the KV-tile loop | 10 |
| Introduce a new data type (e.g., BF16) | Casting microkernel + host API dispatch | 30 |
| Integrate with a custom autograd framework | Host-side C API (`cuflash_attention_forward`) | 20 |

In a production framework, the same changes require navigating template specializations, Python bindings, and vendor-library version constraints.

---

## Summary

CuFlash-Attn does not compete with production frameworks on feature count or out-of-the-box distributed training support. It competes on **clarity**, **buildability**, and **hackability**. If your goal is to understand exactly how FlashAttention works on an NVIDIA GPU—and to have the confidence to change it—CuFlash-Attn is the reference implementation to start from.
