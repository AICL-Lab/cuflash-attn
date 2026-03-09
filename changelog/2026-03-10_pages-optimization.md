# 2026-03-10 GitHub Pages 优化

## Summary

优化 HonKit 文档站，新增专题文档页面，改进构建流程与搜索体验。

## Changes

### 文档内容
- **新增 `docs/algorithm.md`** — FlashAttention 算法详解（分块、Online Softmax、重计算策略、因果掩码、复杂度对比）
- **新增 `docs/api.md`** — 完整 API 参考（FP32/FP16 前向/反向函数签名、张量布局、错误码、GPU 架构表）
- **新增 `docs/building.md`** — 构建指南（CMake Presets、手动构建、构建选项、PyTorch 对比测试）
- **扩展 `SUMMARY.md`** — 从 3 个链接扩展到 6 个页面，按"文档 / Changelog"分组

### 构建配置
- **`book.json`** — 语言改为 `zh-hans`，新增插件：`search-pro`（中文搜索支持）、`copy-code-button`、`expandable-chapters`、`back-to-top-button`
- **`pages.yml`** — 新增 sparse-checkout（仅检出文档相关文件，跳过 src/tests/examples）、安装 HonKit 插件、paths 触发器新增 `docs/**` 和 `README.zh-CN.md`

### Bug 修复
- **README.md / README.zh-CN.md** — Docs badge 修复 `docs.yml` → `pages.yml`（指向实际存在的 workflow），新增 CI badge
