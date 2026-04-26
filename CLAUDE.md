# CLAUDE.md — Claude AI 协作指南

> **语言**: 中文交互。代码注释和 API 文档保持英文。

## 项目定位

CuFlash-Attn 是**从零实现的 CUDA C++ FlashAttention 参考库**，当前稳定在 `v0.3.0` 基线，正在进行最终治理与可归档稳定化。

- **范围明确**：O(N) 内存 FlashAttention，FP32/FP16，前向+反向
- **规范驱动**：所有设计由 `openspec/specs/` 定义，代码遵循规范
- **收尾姿态**：修复漂移、精简工程化、提升交接就绪度（不扩功能）

## 与 Claude 协作时的关键规则

### 1. 规范优先

- **所有实现前**必须先读 `openspec/specs/` 中的相关规范
- 如果请求与规范冲突，停止询问是否先更新规范
- 不添加规范中未定义的功能（no gold-plating）

### 2. OpenSpec 变更流程

所有**行为或 API 变化**都要通过变更提案：

```
/opsx:propose <name>  →  读规范  →  /opsx:apply  →  /verify  →  /opsx:archive
```

**文档/工作流/治理类变更**可以轻量推进，但要同时：
- 更新相关控制文档（AGENTS.md, CONTRIBUTING.md, config.yaml 等）
- 在 PR 模板中明确标注为"Governance cleanup"
- 必要时先运行 `/review`

### 3. 代码风格与验证

**格式化**（Google style, IndentWidth=4, ColumnLimit=100）：
```bash
find . \( -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" \) \
  ! -path "*/build/*" | xargs clang-format -i
```

**构建与测试**（preset-only，无 GPU 时自动跳过）：
```bash
cmake --preset release && cmake --build --preset release
ctest --preset release --output-on-failure
```

### 4. CUDA 关键陷阱

- **流安全**：用 `cudaMemsetAsync` 而非 `cudaMemset`
- **错误处理**：检查所有 `cudaMalloc` 返回值，返回 `FlashAttentionError`，不抛异常
- **内存布局**：`[batch, num_heads, seq_len, head_dim]` 不可变
- **head_dim**：仅支持 32、64、128

### 5. 单线推进，避免分支膨胀

- 始终在 `master` 工作
- 避免长期未合并分支或 `/fleet` 式并行扩散
- 一次只聚焦推进一个 OpenSpec change
- 跨文件改动、规范变更、工作流调整前运行 `/review`

## 常见工作场景

### 修复 Bug

1. 在 `openspec/specs/` 中找相关需求
2. 写测试并标注 spec ID（`// Validates REQ-X.Y`）
3. 修复代码，验证所有测试通过
4. 格式化并 commit

### 改动 API

1. 先更新 `openspec/specs/verification/` 中的 API 定义
2. 创建或更新对应的 OpenSpec change
3. 同步更新 `include/cuflash/` 头文件和 C ABI
4. 添加/更新带 spec 引用的测试
5. 运行 `/review` 后 commit

### 修正文档/工作流

1. 识别需要改动的控制文档（README, CONTRIBUTING.md, AGENTS.md, config.yaml 等）
2. 在当前治理 change 框架内更新（或新建 governance 类 change）
3. 确保所有改动在同一 commit 或相邻 commit 中
4. 必要时运行 `/review`

## 目录与文件职责

| 路径 | 职责 |
|------|------|
| `openspec/specs/` | 规范真相源（需求、API、验证标准） |
| `openspec/changes/` | 变更追踪（design, tasks, 完成后归档） |
| `openspec/config.yaml` | 项目规则、context、anti-patterns |
| `include/cuflash/` | 公开 API（C++ namespace） |
| `src/api/`, `src/forward/`, `src/backward/` | 实现 |
| `tests/unit/`, `tests/integration/` | 验证（需要 spec ID 注释） |
| `docs/` | VitePress 双语文档站 |
| `CONTRIBUTING.md` | 贡献者指南（流程、工具、风格） |
| `AGENTS.md` | 多代理/CLI 合作指南（代码注释保留中文翻译） |
| `.github/copilot-instructions.md` | GitHub Copilot 指令 |
| `.github/pull_request_template.md` | PR 模板（包含 OpenSpec 决策声明） |

## 与其他工具的协作

| 工具 | 最佳用途 |
|------|---------|
| **GitHub Copilot** | 快速代码补全、文档编写、`/review` 快速审查 |
| **Claude Code** | 复杂重构、多文件规范更新、OpenSpec 提案设计 |
| **Codex CLI** | 批处理任务、自动化工作流触发 |

## 不要做（Anti-patterns）

| ❌ | ✅ |
|----|-----|
| 不读规范直接写代码 | 先读 `openspec/specs/` |
| 添加未授权的功能 | 先通过 OpenSpec 流程 |
| 并行堆积未合并分支 | 保持单线、可审查的节奏 |
| 跳过 `/review` 就 commit | 规范/工作流改动前必 review |
| 使用 `cudaMemset` | 用 `cudaMemsetAsync` + `cudaStreamSynchronize` |
| 不检查 CUDA 返回值 | 每个 `cudaMalloc` 都检查 |
| 直接 `cmake -B build` | 用 `cmake --preset` |

## 成功指标

项目完成最终治理后，应该达到：

- ✅ 规范与代码、文档、测试同步更新
- ✅ 所有变更可追踪（通过 OpenSpec changes）
- ✅ 开发流程可重复（preset、format、review gates）
- ✅ 交接文档完整（README, CONTRIBUTING.md, 控制文档, 双语 docs）
- ✅ CI/CD 精简且可维护（无过度复杂的矩阵）
- ✅ 代码示例与文档对齐（Python binding 示例、快速开始指南）

---

**最后更新**: 2026-04-26  
**维护状态**: v0.3.0 稳定基线，最终治理阶段  
**规范来源**: `openspec/specs/`
