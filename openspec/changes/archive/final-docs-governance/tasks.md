# Tasks: final-docs-governance

- [x] 统一 README / OpenSpec / CHANGELOG / CONTRIBUTING 的项目定位与版本叙事
- [x] 删除或改写低价值、重复、过期的控制文档内容
- [x] 建立正式 `CLAUDE.md` 并理顺 AI 指令边界
- [x] 确保所有控制文档都能作为 GLM 接手入口，而不是历史记录堆积

## 执行结果

### 2026-04-27 文档治理

1. **路径引用修复**:
   - README.md: `[/specs/](specs/)` → `[openspec/specs/](openspec/specs/)`
   - README.zh-CN.md: 同上修复

2. **文档职责边界确认**:
   - `CLAUDE.md`: Claude Code 协作指南，中等详细
   - `AGENTS.md`: 多代理协作准则，最详细的 CUDA 陷阱
   - `copilot-instructions.md`: Copilot 指令，精简列表形式
   - 三者各有职责，边界清晰，无整份复制

3. **版本叙事统一**:
   - README / CHANGELOG / CONTRIBUTING 均引用 v0.3.0 稳定基线
   - OpenSpec specs/ 作为唯一真相源

4. **GLM 接手入口**:
   - `CLAUDE.md` 提供完整项目规则和约束
   - `openspec/config.yaml` 提供 OpenSpec 配置
   - `openspec/specs/index.md` 提供规范结构
   - 各控制文档职责清晰，可作为独立入口
