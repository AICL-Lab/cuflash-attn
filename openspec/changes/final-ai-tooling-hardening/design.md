# Design: final-ai-tooling-hardening

## Goal

把本地编辑器、AI 工具和远端能力的边界固定下来，避免在最终治理阶段继续叠加“看起来更强、
实际上更重”的工具链复杂度。

## 1. Local Editing Baseline

### Preferred stack

| Need | Decision | Why |
|------|----------|-----|
| C++ / CUDA navigation | `clangd` | 直接消费 `compile_commands.json`，跨编辑器可复用 |
| Configure/build entrypoint | CMake presets | 与仓库规则一致，避免 `cmake -B` 漂移 |
| VS Code integration | CMake Tools + clangd | 足够覆盖 configure、build、index、format |

### Compile database contract

1. 运行 `cmake --preset release`
2. 生成 `build/release/compile_commands.json`
3. `clangd` 从该目录读取实际编译参数

### Degraded mode without `nvcc`

- **允许继续进行**：OpenSpec、README、控制文档、workflow、GitHub metadata、绝大多数头文件整理
- **会退化的能力**：`.cu` 语义诊断、CMake configure、完整编译错误定位
- **处理原则**：把这类退化视为环境限制，而不是自动归咎于仓库代码

## 2. Capability Routing: CLI Skills vs MCP

| 场景 | 默认机制 | 原因 |
|------|----------|------|
| 本地仓库探索、计划执行、review、handoff | CLI Skills | 低上下文、可复用、不会引入额外常驻依赖 |
| GitHub Actions、repo metadata、PR / issue / workflow 状态 | MCP 或 `gh` | 这些信息不在本地仓库中，远端读取更直接 |
| 单次远端管理动作（topics、description、release 验证） | `gh` 优先 | 命令可审计、脚本化简单 |
| 需要结构化远端查询/下载日志 | MCP 优先 | 比手写 API 请求更稳、更省手工解析 |

### Non-goals

- 不为本地代码编辑引入常驻 MCP 服务
- 不为低频动作增加常开后台代理
- 不把“可以接更多工具”误当成“应该接更多工具”

## 3. Copilot Plugin Decision

**结论：当前不引入项目级 Copilot Plugin。**

理由：

1. 现有项目指令（`AGENTS.md`、`CLAUDE.md`、`.github/copilot-instructions.md`）已经覆盖规则注入
2. OpenSpec change 流程已经提供结构化的任务容器
3. CLI Skills 已能覆盖计划、review、verify、handoff 等高频流程
4. GitHub 远端状态已可通过 MCP / `gh` 获取，无需再叠加插件层

### Revisit criteria

只有同时满足以下条件时，才重新评估项目级 Copilot Plugin：

1. 存在高频、重复、跨多个会话的痛点
2. 现有 instructions + CLI skills + `gh` / MCP 无法稳定覆盖
3. 引入插件不会要求长期常驻服务或显著增加上下文负担

## 4. Files that Encode the Decision

| File | Role |
|------|------|
| `.clangd` | clangd fallback 与 compile database 约定 |
| `.vscode/settings.json` | VS Code 使用 clangd + presets 的默认行为 |
| `.vscode/extensions.json` | 推荐最小扩展集合，避免冲突 |
| `CONTRIBUTING.md` | 给人类贡献者的本地编辑基线 |
| `AGENTS.md` / `CLAUDE.md` / `.github/copilot-instructions.md` | 给 AI 工具的边界说明 |
| `openspec/config.yaml` | 项目级规则与 quick reference |
