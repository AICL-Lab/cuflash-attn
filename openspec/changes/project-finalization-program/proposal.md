# Change Proposal: project-finalization-program

## Intent（意图）

**为什么：** `project-finalization-v0.3.0` 已完成上一轮版本发布收尾，但仓库仍处于“已发布 yet 未完全归档”的状态：项目规范、AI 指令、对外文档、GitHub 自动化、站点体验和协作流程之间仍有收尾期漂移，尚未形成可长期冻结、可直接交给后续模型接手的最终治理结构。

**做什么：** 建立一个总控型 OpenSpec change，定义本轮最终治理的目标状态、破坏性清理边界、子 change 拆分、执行顺序、完成标准和交接方式。该 change 不直接承载所有实现细节，而是作为后续 Phase 1-4 的总导航与约束源。

---

## Scope（范围）

### In Scope ✅

- 明确项目已从“发布 v0.3.0”转入“归档级旗舰样板”治理阶段
- 定义本轮允许的破坏性清理边界
- 拆分后续子 change，避免继续堆叠巨型杂糅改动
- 明确文档、工作流、AI 配置、代码审计和 GitHub 展示面的治理顺序
- 为后续 GLM 接手提供结构化任务入口

### Out of Scope ❌

- 在本 change 内一次性完成所有代码、文档、CI/CD 和 GitHub metadata 改动
- 新功能开发或性能导向优化
- 任何未通过后续子 change 明确承接的“顺手扩展”

---

## Success Criteria（成功标准）

1. 仓库中存在一个可审阅的总控 change，清楚说明为什么需要最终治理而不是继续扩展。
2. 该 change 能把后续工作拆为若干独立、可归档、可 review 的子 change。
3. `openspec/config.yaml` 等项目级真相源不再停留在“目标发布 v0.3.0”的过渡叙事。
4. 后续执行者无需反推历史，即可理解：
   - 当前项目状态
   - 治理优先级
   - 哪些改动允许破坏兼容
   - 哪些流程必须保留（OpenSpec、/review、preset-only build 等）

---

## Planned Child Changes（计划中的子变更）

1. `final-docs-governance`
   - 收敛 README / CHANGELOG / CONTRIBUTING / OpenSpec 核心文档 / AI 控制文档
2. `final-pages-redesign`
   - 重构 GitHub Pages / VitePress 站点的信息架构与对外叙事
3. `final-ci-rationalization`
   - 精简 workflow，修复 release / CodeQL / Pages 的过度设计与缺陷
4. `final-ai-tooling-hardening`
   - 正式化 `CLAUDE.md`、重写 `AGENTS.md`、更新 Copilot / LSP / 插件策略
5. `final-bug-sweep`
   - 对代码、示例、测试、打包面进行最终 bug 清扫
6. `final-handoff-backlog`
   - 输出交接给 GLM 的最终 backlog 与执行约束

---

## Notes（备注）

- 本 change 允许为“规范纯度和最终稳定性”进行必要的破坏性清理。
- 本仓库继续遵守单分支策略：直接在 `master` 工作，不建立长期功能分支。
- 执行阶段避免 `/fleet`；优先使用长会话串行推进，在关键节点通过 `/review` 做质量闸门。
