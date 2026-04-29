# Design: project-finalization-program

## Overview

本设计文档定义 CuFlash-Attn 最终治理程序的组织方式，而不是某个单一功能实现。核心目标是把仓库从“完成 v0.3.0 发布后的收尾状态”推进到“可以随时归档、规则收敛、协作面稳定、便于模型接手”的最终状态。

## Design Principles

### 1. Governance before implementation

优先治理真相源与规则源：

- `openspec/specs/**`
- `openspec/config.yaml`
- `README*.md`
- `AGENTS.md`
- `CLAUDE.md`
- `.github/copilot-instructions.md`
- `.github/workflows/*.yml`

只有这些基础约束收敛后，代码、测试、示例和 GitHub 展示面的改动才不会继续漂移。

### 2. Break the work into archiveable units

后续治理以子 change 为单位推进。每个子 change 必须：

- 范围单一
- 能单独 review
- 能单独 verify
- 能单独 archive

禁止再创建覆盖文档、CI、AI、代码、发布面所有层的“超大收尾 diff”。

### 3. Final-state bias over compatibility bias

本轮治理允许必要的破坏性清理，只要：

- 结果更稳定
- 规范更一致
- 对外叙事更干净
- 接手成本更低

不为了维持历史兼容而继续保留明显的过渡态结构、过期文档或复杂工作流。

### 4. Keep the project attractive, not noisy

项目对外定位是“归档级旗舰样板”，而不是“仅供内部收尾的私有仓库”。因此：

- 文档站与 README 仍需具备展示性和教学价值；
- 但不保留纯营销性、模板化或过度工程化内容；
- GitHub About、Pages、文档首页应传达可信、克制、完整的技术形象。

## Execution Model

### Phase order

1. 总控 change
2. 文档治理
3. CI/GitHub 流程收敛
4. Pages/站点重构
5. AI 工具链硬化
6. 代码与测试 bug sweep
7. 最终 backlog / handoff

### Review gates

- 每个子 change 完成后执行 `/review`
- 任何 API、规范或发布面改动都必须同步更新对应文档
- 任何偏离 preset-only 构建、OpenSpec-first 或单分支策略的设计都应视为回归

## Deliverables

本总控 change 的直接产物为：

1. 一个正式的治理 proposal（`proposal.md`）
2. 一个说明治理方式与边界的 design（`design.md`）
3. 一个总控 checklist（`tasks.md`）
4. 一个已经更新到最终治理状态的 `openspec/config.yaml`

## Completion Criteria

当以下条件满足时，本 change 可视为完成：

1. 总控 proposal / design / tasks 已写入仓库
2. `openspec/config.yaml` 已从“发布 v0.3.0”转为“最终治理 / 归档级稳定态”语境
3. 所有后续工作都已能映射到明确的子 change，而不是继续游离在规范之外
