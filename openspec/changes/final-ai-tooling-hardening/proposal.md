# Change Proposal: final-ai-tooling-hardening

## Intent（意图）

**为什么：** 当前仓库已有 `AGENTS.md`、`.github/copilot-instructions.md`、`.clangd`、`.vscode/settings.json` 和本地 `CLAUDE.local.md`，但缺少正式的 `CLAUDE.md`，AI 指令之间也存在过渡期重复与边界不清。

**做什么：** 正式化项目级 AI 指令和本地开发辅助配置，明确 Copilot / Claude / Codex / OpenSpec 的协作边界，并压缩不必要的工具链复杂度。

## Scope（范围）

### In Scope ✅

- `AGENTS.md`
- `CLAUDE.md`
- `.github/copilot-instructions.md`
- `.clangd`
- `.vscode/settings.json`
- 相关 AI / LSP / plugin 配置文件

### Out of Scope ❌

- OpenSpec 需求与验证正文（交给 `final-docs-governance`）
- CI workflow 实现细节（交给 `final-ci-rationalization`）
