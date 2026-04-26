# Change Proposal: final-pages-redesign

## Intent（意图）

**为什么：** 当前 VitePress 站点已经能构建，但仍带有发布冲刺阶段遗留的复杂度和叙事噪音。作为项目对外窗口，它需要从“可访问”提升到“可信、克制、信息架构清晰”的归档级展示站。

**做什么：** 重构 docs 首页、导航、页面职责和外部链接策略，评估 PWA / GA / Algolia 等功能的真实价值，并将站点收敛为高质量的技术展示面。

## Scope（范围）

### In Scope ✅

- `docs/.vitepress/**`
- `docs/index.md`
- `docs/en/**`
- `docs/zh/**`
- `docs/public/**`

### Out of Scope ❌

- OpenSpec 真相源重写（交给 `final-docs-governance`）
- workflow 触发条件与 GitHub Pages 发布逻辑（交给 `final-ci-rationalization`）
