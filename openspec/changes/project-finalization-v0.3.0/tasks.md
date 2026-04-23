# Tasks: project-finalization-v0.3.0

## Implementation Checklist

### Phase 1: Code Audit（代码审查）

- [x] **T1** 审查 `src/forward/flash_attention_forward.cu`
  - stream 正确传入所有 kernel 调用 ✅
  - cudaGetLastError() 每个 head_dim 分支后均有检查 ✅
  - 移除了未使用的 `<64,64,128>` 模板实例 ✅

- [x] **T2** 审查 `src/forward/flash_attention_fp16.cu`
  - stream 正确传入，CUDA 错误检查完整 ✅

- [x] **T3** 审查 `src/backward/flash_attention_backward.cu`
  - 多阶段 backward（dV/dK/dQ）每阶段后均检查 CUDA 错误 ✅
  - stream 一致性确认 ✅

- [x] **T4** 审查 `src/backward/flash_attention_backward_fp16.cu`
  - 同 T3 ✅

- [x] **T5** 审查 `include/cuflash/flash_attention.h` + 修复
  - 添加 `cuflash_error_string()` C ABI 包装器 ✅
  - 修复 `examples/python_binding.py` 函数名（f32 ABI）及 stream 参数 ✅
  - 修复 `tests/unit/test_online_softmax.cu` 错误的相对路径 ✅
  - 移除 `flash_attention_api.cu` 中的死代码声明 ✅
  - 更新 verification spec C ABI 章节与实际实现一致 ✅

### Phase 2: Version Bump（版本号统一）

- [x] **T6** `CMakeLists.txt`: VERSION 0.3.0 ✅
- [x] **T7** `docs/.vitepress/config.js`: v0.3.0 ✅
- [x] **T8** `openspec/config.yaml`: version: 0.3.0 ✅
- [x] **T9** `CHANGELOG.md`: [0.3.0] 条目已添加 ✅

### Phase 3: Release（发布）

- [x] **T10** 格式检查通过（clang-format Google style）✅
- [x] **T11** 代码审计修复提交: `fix: resolve all static analysis bugs found in code audit` ✅
- [x] **T12** Tag: `v0.3.0` ✅
- [x] **T13** 推送: `git push origin v0.3.0` ✅

---

## Definition of Done ✅

- `git tag v0.3.0` 已推送到 GitHub ✅
- CI 构建通过（format check + build）✅
- `CHANGELOG.md` 有正式的 `[0.3.0]` 条目 ✅
- GitHub release 页面自动生成（由 `release.yml` 触发）✅
