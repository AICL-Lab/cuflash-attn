# Change Proposal: final-docs-governance

## Intent（意图）

**为什么：** 仓库已经完成 `v0.3.0` 发布，但 README、CHANGELOG、CONTRIBUTING、OpenSpec 入口文档和 AI 控制文档仍保留收尾过渡态叙事，文档层级之间也存在职责重叠。若不先治理文档真相源，后续代码、CI 和 GitHub 展示面的清理会继续失去统一口径。

**做什么：** 收敛项目文档层级，清理过时与低价值内容，统一项目定位、版本叙事和协作约束，让文档真正服务于“归档级旗舰样板”和后续模型接手。

## Scope（范围）

### In Scope ✅

- `README.md` / `README.zh-CN.md`
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `openspec/specs/**`
- `AGENTS.md`
- `CLAUDE.md`
- `.github/copilot-instructions.md`

### Out of Scope ❌

- docs 站点视觉与信息架构重构（交给 `final-pages-redesign`）
- workflow / release 逻辑修复（交给 `final-ci-rationalization`）
