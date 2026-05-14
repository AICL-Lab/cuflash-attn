# Breaking Changes and Migration

This page documents breaking changes between versions and provides migration guidance.

---

## v0.3.0 → Future

No breaking changes currently planned. The v0.3.0 stable baseline is the foundation for future development.

---

## Migration Guidelines

### API Stability

CuFlash-Attn follows semantic versioning:

- **Major version** (0.x.x → 1.x.x): May contain breaking API changes
- **Minor version** (0.3.x → 0.4.x): New features, backward compatible
- **Patch version** (0.3.0 → 0.3.1): Bug fixes only, fully compatible

### Current API Contract

The v0.3.0 API is stable:

```cpp
// C++ API - stable since v0.3.0
FlashAttentionError flash_attention_forward(
    const float* d_Q, const float* d_K, const float* d_V,
    float* d_O, float* d_L,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool is_causal, cudaStream_t stream
);
```

```c
// C ABI - stable since v0.3.0
CUFlashAttnError cuflash_attention_forward_f32(
    const void* q, const void* k, const void* v,
    void* o, void* l,
    int B, int H, int N, int D,
    float scale, int is_causal, void* stream
);
```

### Memory Layout

The memory layout is fixed and will not change:

```
Input/Output: [batch, num_heads, seq_len, head_dim]
Log-sum-exp:  [batch, num_heads, seq_len]
```

---

## Deprecation Policy

Features are deprecated with a minimum one-version notice:

1. **Announce**: Deprecation noted in changelog and documentation
2. **Warn**: Runtime warning added (if applicable)
3. **Remove**: Feature removed in next major version

---

## Version Compatibility

| CuFlash-Attn | CUDA | GPU Architecture | Status |
|:------------:|:----:|:----------------:|:------:|
| v0.3.0 | 11.0+ | sm_70 - sm_90 | Stable |
| v0.2.x | 11.0+ | sm_70 - sm_80 | EOL |
| v0.1.x | 11.0+ | sm_70 - sm_75 | EOL |

::: tip Note
v0.3.0 is the first stable baseline. Earlier versions are end-of-life.
:::
