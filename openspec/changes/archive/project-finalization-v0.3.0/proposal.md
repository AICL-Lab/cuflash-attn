# Change Proposal: project-finalization-v0.3.0

## Intent（意图）

**为什么：** 项目经历了多轮小模型开发，导致代码漂移和架构腐化。本次整治已完成工程化层面的全面清理（见 `chore: comprehensive project overhaul`），现需完成代码层面的最终收尾，发布 v0.3.0 稳定版。

**做什么：** 代码审查、流安全性验证、版本号统一、最终发布。

---

## Scope（范围）

### In Scope ✅
- 版本号从 0.2.0 → 0.3.0（CMakeLists.txt、docs/config.js）
- stream safety 审查（确认所有 kernel 正确使用传入的 stream）
- 内存管理审查（确认无内存泄漏、无越界访问）
- CHANGELOG.md 最终确认
- `git tag v0.3.0` + 触发 release workflow

### Out of Scope ❌
- 新功能开发
- API 变更
- 性能优化（可以做，但不是必须）

---

## Key Files

| 文件 | 变更内容 |
|------|----------|
| `CMakeLists.txt` | VERSION 0.2.0 → 0.3.0 |
| `docs/.vitepress/config.js` | nav version 标签 v0.2.0 → v0.3.0 |
| `openspec/config.yaml` | version: 0.2.0 → 0.3.0 |
| `CHANGELOG.md` | 将 [Unreleased] 替换为 [0.3.0] + 日期 |

---

## Context

- 当前 master: `3169fd9` (chore: comprehensive project overhaul)
- 目标 tag: `v0.3.0`
- 无 GPU 的 CI 环境：`test_main.cpp` 已有 `cudaGetDeviceCount()` 保护，无 GPU 时跳过测试
- 代码层无未完成的 TODO/FIXME
