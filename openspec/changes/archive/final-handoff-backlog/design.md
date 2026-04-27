# Design: final-handoff-backlog

## Goal

把最终治理后的剩余动作压缩成一个可直接交给 GLM 或人工执行的 backlog，避免接手者重新
扫描整个历史上下文。

## 1. Current Change Status

| Change | Status | Notes |
|--------|--------|-------|
| `project-finalization-program` | Implemented | 作为总控治理框架 |
| `final-docs-governance` | Implemented | README / 控制文档 / 版本叙事已收敛 |
| `final-pages-redesign` | Implemented | docs 站已轻量化并加入 project-status |
| `final-ci-rationalization` | Implemented | workflow 已精简并修复关键缺陷 |
| `final-ai-tooling-hardening` | Implemented | LSP / MCP / plugin 边界已固化 |
| `final-bug-sweep` | Implemented | 示例与关键长尾缺陷已收口 |
| `final-handoff-backlog` | Implemented by this doc | 负责最终交接矩阵 |

## 2. Archive-Ready Backlog

| Backlog ID | Owner | Definition of Done | Depends On | Needs | Destructive | `/review` | OpenSpec change |
|------------|-------|--------------------|------------|-------|-------------|-----------|-----------------|
| `cuda-verification-sweep` | GLM / human on CUDA machine | `cmake --preset release && cmake --build --preset release && ctest --preset release --output-on-failure` 成功；如环境允许，再跑 integration / package smoke | 当前治理改动已落地 | CUDA Toolkit + `nvcc` + GPU | No | Recommended | `project-finalization-program` |
| `github-surface-smoke` | GLM / human with repo access | 确认 GitHub Pages、repo description/topics、release workflow 页面表现符合当前文档叙事 | 仓库推送到远端 | GitHub write/admin 权限 | No | Optional | `final-ci-rationalization` |
| `final-review-gate` | Human or senior model | 对治理类 diff 做最后一次 `/review` 或等价人工审查，确认无新增复杂度 | 前两项完成或明确受环境限制 | Review capability | No | Required | `project-finalization-program` |
| `archive-openspec-changes` | GLM / human | 将已完成 change 按流程 `/opsx:archive`，保留可追踪历史 | final review 完成 | OpenSpec write access | No | Optional | 各自对应 change |

## 3. Work Better Left to GLM

这些任务适合后续模型继续，而不值得在当前会话里为了环境限制反复试错：

1. **CUDA 机器上的完整编译验证**
2. **远端 GitHub 表面的视觉与 workflow 烟雾测试**
3. **最终 `/review` 后的归档节奏控制**

## 4. Human / Permission Boundaries

| Topic | Why a human may still be needed |
|-------|----------------------------------|
| GitHub Pages / release 最终观感 | 需要浏览器确认与仓库管理权限 |
| 发布、tag、是否真正 archive 仓库 | 属于项目所有者决定 |
| CUDA 硬件可用性 | 取决于机器与驱动环境，不是仓库本身能解决 |

## 5. Acceptance Gate for “Final State”

仓库可以被视为“可归档稳定态”的条件：

1. 控制文档、README、OpenSpec、docs 站叙事一致
2. workflow 不再包含明显冗余或坏链路
3. AI 工具链遵循“本地优先、按需远端、默认不加插件”的边界
4. CUDA 机器上的验证结果要么通过，要么明确记录为环境限制
5. 所有治理类 change 都有可追踪的 proposal / tasks / design / archive 记录
