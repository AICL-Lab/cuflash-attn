# References

This page collects all bibliographic entries cited throughout the CuFlash-Attn documentation. Entries are grouped by category and formatted in a BibTeX-inspired style for easy inclusion in academic manuscripts, technical reports, and interview preparation materials.

---

## Table of Contents

- [Core Papers](#core-papers)
- [Implementation References](#implementation-references)
- [Performance Optimization References](#performance-optimization-references)
- [CUDA Programming References](#cuda-programming-references)

---

## Core Papers

These papers define the algorithmic and architectural foundations on which CuFlash-Attn is built.

---

### FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Citation:**
```bibtex
@inproceedings{dao2022flashattention,
  title={FlashAttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022},
  url={https://arxiv.org/abs/2205.14135}
}
```

**Formatted entry:**
- **Title:** FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- **Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher R\'e
- **Venue:** NeurIPS 2022
- **Year:** 2022
- **URL:** [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)

**Used in:** [Algorithm](/en/algorithm), [Kernel Deep Dive](/en/design/kernel-deep-dive), [Related Work](/en/research/related-work)

---

### FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

**Citation:**
```bibtex
@inproceedings{dao2024flashattention2,
  title={FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024},
  url={https://arxiv.org/abs/2307.08691}
}
```

**Formatted entry:**
- **Title:** FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
- **Authors:** Tri Dao
- **Venue:** ICLR 2024
- **Year:** 2024
- **URL:** [https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)

**Used in:** [Related Work](/en/research/related-work)

---

### Online normalizer calculation for softmax

**Citation:**
```bibtex
@article{milakov2018online,
  title={Online normalizer calculation for softmax},
  author={Milakov, Maxim and Gimelshein, Natalia},
  journal={arXiv preprint arXiv:1805.02867},
  year={2018},
  url={https://arxiv.org/abs/1805.02867}
}
```

**Formatted entry:**
- **Title:** Online normalizer calculation for softmax
- **Authors:** Maxim Milakov, Natalia Gimelshein
- **Venue:** arXiv preprint
- **Year:** 2018
- **URL:** [https://arxiv.org/abs/1805.02867](https://arxiv.org/abs/1805.02867)

**Used in:** [Algorithm](/en/algorithm), [Kernel Deep Dive](/en/design/kernel-deep-dive), [Related Work](/en/research/related-work)

---

## Implementation References

These papers and reports describe systems, models, and architectural patterns that contextualize the practical need for memory-efficient attention kernels.

---

### Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM

**Citation:**
```bibtex
@inproceedings{narayanan2021megatron,
  title={Efficient Large-Scale Language Model Training on {GPU} Clusters Using {Megatron-LM}},
  author={Narayanan, Deepak and Shoeybi, Mohammad and Casper, Jared and LeGresley, Patrick and Patwary, Mostofa and Korthikanti, Vijay and Vainbrand, Dmitri and Kashinkunti, Prethvi and Bernauer, Julie and Catanzaro, Bryan and Phanishayee, Amir and Zaharia, Matei},
  booktitle={Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC)},
  year={2021},
  url={https://arxiv.org/abs/2104.04473}
}
```

**Formatted entry:**
- **Title:** Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM
- **Authors:** Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amir Phanishayee, Matei Zaharia
- **Venue:** SC 2021
- **Year:** 2021
- **URL:** [https://arxiv.org/abs/2104.04473](https://arxiv.org/abs/2104.04473)

**Used in:** [Related Work](/en/research/related-work)

---

### PagedAttention: vLLM

**Citation:**
```bibtex
@inproceedings{kwon2023pagedattention,
  title={Efficient Memory Management for Large Language Model Serving with {PagedAttention}},
  author={Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph E. and Zhang, Hao and Stoica, Ion},
  booktitle={Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP)},
  year={2023},
  url={https://arxiv.org/abs/2309.06180}
}
```

**Formatted entry:**
- **Title:** Efficient Memory Management for Large Language Model Serving with PagedAttention
- **Authors:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
- **Venue:** SOSP 2023
- **Year:** 2023
- **URL:** [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

**Used in:** [Related Work](/en/research/related-work)

---

### Ring Attention with Blockwise Transformers for Near-Infinite Context

**Citation:**
```bibtex
@article{liu2023ringattention,
  title={Ring Attention with Blockwise Transformers for Near-Infinite Context},
  author={Liu, Hao and Zaharia, Matei and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2310.01889},
  year={2023},
  url={https://arxiv.org/abs/2310.01889}
}
```

**Formatted entry:**
- **Title:** Ring Attention with Blockwise Transformers for Near-Infinite Context
- **Authors:** Hao Liu, Matei Zaharia, Pieter Abbeel
- **Venue:** arXiv preprint
- **Year:** 2023
- **URL:** [https://arxiv.org/abs/2310.01889](https://arxiv.org/abs/2310.01889)

**Used in:** [Related Work](/en/research/related-work)

---

### Multi-Query Attention

**Citation:**
```bibtex
@article{shazeer2019multiquery,
  title={Fast Transformer Decoding: One Write-Head is All You Need},
  author={Shazeer, Noam},
  journal={arXiv preprint arXiv:1911.02150},
  year={2019},
  url={https://arxiv.org/abs/1911.02150}
}
```

**Formatted entry:**
- **Title:** Fast Transformer Decoding: One Write-Head is All You Need (Multi-Query Attention)
- **Authors:** Noam Shazeer
- **Venue:** arXiv preprint
- **Year:** 2019
- **URL:** [https://arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150)

**Used in:** [Related Work](/en/research/related-work)

---

### Grouped-Query Attention

**Citation:**
```bibtex
@article{ainslie2023gqa,
  title={GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
  author={Ainslie, Joshua and Lu, Tao and de Jong, Michiel and Zemlyanskiy, Yury and Ontanon, Santiago and Sanghai, Sumit},
  journal={arXiv preprint arXiv:2305.13245},
  year={2023},
  url={https://arxiv.org/abs/2305.13245}
}
```

**Formatted entry:**
- **Title:** GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- **Authors:** Joshua Ainslie, Tao Lu, Michiel de Jong, Yury Zemlyanskiy, Santiago Ontanon, Sumit Sanghai
- **Venue:** arXiv preprint
- **Year:** 2023
- **URL:** [https://arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)

**Used in:** [Related Work](/en/research/related-work)

---

## Performance Optimization References

These works establish the empirical and theoretical grounding for GPU performance engineering decisions made in CuFlash-Attn.

---

### Roofline Model for GPU Performance Analysis

While CuFlash-Attn does not cite a single canonical roofline paper, the roofline analyses in our [Performance](/en/performance/roofline-analysis) documentation are grounded in the methodology established by Williams, Waterman, and Patterson (2009), extended to GPU architectures by subsequent NVIDIA and academic works.

**Recommended reading:**
- Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An Insightful Visual Performance Model for Multicore Architectures. *Communications of the ACM*, 52(4), 65–76. [https://doi.org/10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785)

---

### Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking

**Citation:**
```bibtex
@inproceedings{jang2019volta,
  title={Dissecting the {NVIDIA} {Volta} {GPU} Architecture via Microbenchmarking},
  author={Jang, Haicheng and Kim, Wonjeon and Chun, Sungho and Lee, Jungho and Lee, Hyungon and Noh, Seungryul and Jung, Myoungsoo},
  booktitle={2019 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)},
  year={2019},
  url={https://arxiv.org/abs/1804.06826}
}
```

**Formatted entry:**
- **Title:** Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking
- **Authors:** Haicheng Jang, Wonjeon Kim, Sungho Chun, Jungho Lee, Hyungon Lee, Seungryul Noh, Myoungsoo Jung
- **Venue:** ISPASS 2019
- **Year:** 2019
- **URL:** [https://arxiv.org/abs/1804.06826](https://arxiv.org/abs/1804.06826)

**Note:** The shared-memory capacity figures, warp scheduling rules, and L2 cache-line size assumptions used in CuFlash-Attn's kernel design are validated against the microbenchmarking data in this paper and its Ampere/Hopper successors.

---

## CUDA Programming References

These are authoritative NVIDIA documents used to justify device-side programming choices in CuFlash-Attn.

---

### CUDA C++ Programming Guide

**Citation:**
```bibtex
@manual{nvidia2024cudaguide,
  title={{CUDA C++ Programming Guide}},
  author={{NVIDIA Corporation}},
  edition={Version 12.4},
  year={2024},
  url={https://docs.nvidia.com/cuda/cuda-c-programming-guide/}
}
```

**Formatted entry:**
- **Title:** CUDA C++ Programming Guide
- **Authors:** NVIDIA Corporation
- **Venue:** Official Documentation
- **Year:** 2024 (Version 12.4)
- **URL:** [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

**Relevant sections:**
- § 7.22 `__launch_bounds__`
- § 7.3 Shared Memory
- § 7.13 Warp Shuffle Functions
- § 5.2.3 Vectorized Memory Access

---

### NVIDIA Nsight Compute Documentation

**Citation:**
```bibtex
@manual{nvidia2024nsight,
  title={{NVIDIA Nsight Compute: Kernel Profiling Guide}},
  author={{NVIDIA Corporation}},
  edition={Version 2024.1},
  year={2024},
  url={https://docs.nvidia.com/nsight-compute/}
}
```

**Formatted entry:**
- **Title:** NVIDIA Nsight Compute: Kernel Profiling Guide
- **Authors:** NVIDIA Corporation
- **Venue:** Official Documentation
- **Year:** 2024
- **URL:** [https://docs.nvidia.com/nsight-compute/index.html](https://docs.nvidia.com/nsight-compute/index.html)

**Relevant counters:**
- `gld_transactions`, `gst_transactions` — coalescing verification
- `shared_load_bank_conflict` — shared memory conflict detection
- `memory_throughput` — bandwidth saturation analysis
- `inst_executed` — instruction count reduction for causal-mask skipping

---

## How to Cite CuFlash-Attn

If you use CuFlash-Attn in academic work, software benchmarks, or technical blogging, please cite the project as follows:

```bibtex
@software{cuflashattn2024,
  title={{CuFlash-Attn}: From-Scratch {CUDA} {FlashAttention}},
  author={{LessUp Organization}},
  year={2024},
  url={https://github.com/LessUp/cuflash-attn},
  note={Version 0.3.0. Educational CUDA C++ reference implementation of FlashAttention.}
}
```

---

*Last updated: 2024. This reference list is maintained in sync with the [Related Work](/en/research/related-work) page. If a paper cited in the documentation is missing from this page, please file a documentation issue.*
