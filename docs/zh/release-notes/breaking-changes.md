# 破坏性变更与迁移指南

本页记录各版本之间的破坏性变更，并提供迁移指导。

---

## v0.3.0 → 未来版本

目前没有计划的破坏性变更。v0.3.0 稳定基线是未来开发的基础。

---

## 迁移指南

### API 稳定性

CuFlash-Attn 遵循语义化版本控制：

- **主版本号** (0.x.x → 1.x.x)：可能包含破坏性 API 变更
- **次版本号** (0.3.x → 0.4.x)：新功能，向后兼容
- **修订号** (0.3.0 → 0.3.1)：仅修复 bug，完全兼容

### 当前 API 契约

v0.3.0 API 已稳定：

```cpp
// C++ API - 自 v0.3.0 起稳定
FlashAttentionError flash_attention_forward(
    const float* d_Q, const float* d_K, const float* d_V,
    float* d_O, float* d_L,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool is_causal, cudaStream_t stream
);
```

```c
// C ABI - 自 v0.3.0 起稳定
CUFlashAttnError cuflash_attention_forward_f32(
    const void* q, const void* k, const void* v,
    void* o, void* l,
    int B, int H, int N, int D,
    float scale, int is_causal, void* stream
);
```

### 内存布局

内存布局已固定，不会变更：

```
输入/输出: [batch, num_heads, seq_len, head_dim]
Log-sum-exp: [batch, num_heads, seq_len]
```

---

## 弃用策略

功能弃用最少提前一个版本通知：

1. **公告**：在更新日志和文档中标注弃用
2. **警告**：添加运行时警告（如适用）
3. **移除**：在下一个主版本中移除

---

## 版本兼容性

| CuFlash-Attn | CUDA | GPU 架构 | 状态 |
|:------------:|:----:|:--------:|:----:|
| v0.3.0 | 11.0+ | sm_70 - sm_90 | 稳定 |
| v0.2.x | 11.0+ | sm_70 - sm_80 | EOL |
| v0.1.x | 11.0+ | sm_70 - sm_75 | EOL |

::: tip 注意
v0.3.0 是首个稳定基线。早期版本已终止支持。
:::
