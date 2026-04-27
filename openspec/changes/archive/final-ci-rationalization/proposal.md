# Change Proposal: final-ci-rationalization

## Intent（意图）

**为什么：** 当前 workflow 集合已经覆盖 CI、CodeQL、Pages、Release，但仍存在发布期遗留缺陷和过度设计倾向。若不精简自动化，项目会继续承担不必要的维护复杂度。

**做什么：** 修复现有 workflow 缺陷，收紧触发条件，统一到 preset-only 的构建约定，并让 CI/CD 与“无 GPU runner + 归档级稳定仓库”的现实相匹配。

## Scope（范围）

### In Scope ✅

- `.github/workflows/ci.yml`
- `.github/workflows/codeql.yml`
- `.github/workflows/pages.yml`
- `.github/workflows/release.yml`
- 与 workflow 紧密耦合的 release / docs build 文案

### Out of Scope ❌

- GitHub About / topics / homepage 元数据更新（交给 `final-handoff-backlog` 或远端同步步骤）
- README / OpenSpec 核心文案治理（交给 `final-docs-governance`）
