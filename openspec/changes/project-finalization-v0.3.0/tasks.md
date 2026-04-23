# Tasks: project-finalization-v0.3.0

## Implementation Checklist

### Phase 1: Code Audit（代码审查）

- [ ] **T1** 审查 `src/forward/flash_attention_forward.cu`
  - 确认 `stream` 参数正确传入所有 kernel 调用
  - 确认 `cudaGetLastError()` 之后正确返回 FlashAttentionError

- [ ] **T2** 审查 `src/forward/flash_attention_fp16.cu`
  - 同 T1

- [ ] **T3** 审查 `src/backward/flash_attention_backward.cu`
  - 确认多阶段 backward（dV/dK/dQ）每个阶段后检查 CUDA 错误
  - 确认 stream 一致性

- [ ] **T4** 审查 `src/backward/flash_attention_backward_fp16.cu`
  - 同 T3

- [ ] **T5** 审查 `include/cuflash/flash_attention.h`
  - 确认 API 声明与实现一致

### Phase 2: Version Bump（版本号统一）

- [ ] **T6** `CMakeLists.txt`: `VERSION 0.2.0` → `VERSION 0.3.0`
- [ ] **T7** `docs/.vitepress/config.js`: 两处 `v0.2.0` → `v0.3.0`
- [ ] **T8** `openspec/config.yaml`: `version: 0.2.0` → `version: 0.3.0`
- [ ] **T9** `CHANGELOG.md`: `[Unreleased]` → `[0.3.0] - <today>`

### Phase 3: Release（发布）

- [ ] **T10** 运行格式检查: `find src include -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" | xargs clang-format --dry-run -Werror`
- [ ] **T11** 提交版本变更: `git commit -m "chore(release): bump to v0.3.0"`
- [ ] **T12** 打 tag: `git tag -a v0.3.0 -m "Release v0.3.0 — finalization release"`
- [ ] **T13** 推送 tag: `git push origin v0.3.0`

---

## Definition of Done

- `git tag v0.3.0` 已推送到 GitHub
- CI 构建通过（format check + build）
- `CHANGELOG.md` 有正式的 `[0.3.0]` 条目
- GitHub release 页面自动生成（由 `release.yml` 触发）
