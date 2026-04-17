# AGENTS.md - AI Agent Workflow Instructions

## Project Philosophy: Spec-Driven Development (SDD)

本项目严格遵循**规范驱动开发（Spec-Driven Development）**范式。所有的代码实现必须以 `/specs` 目录下的规范文档为唯一事实来源（Single Source of Truth）。

---

## Directory Context (目录说明)

| Directory | Purpose | 说明 |
|-----------|---------|------|
| `/specs/product/` | Product feature definitions and acceptance criteria | 产品功能定义与验收标准 |
| `/specs/rfc/` | Technical design documents (Request for Comments) | 技术设计文档 |
| `/specs/api/` | API interface specifications (human and machine-readable) | API 接口定义（人类可读与机器可读） |
| `/specs/db/` | Database/schema specifications (if applicable) | 数据库 Schema 设计规范（如适用） |
| `/specs/testing/` | Testing specifications and BDD test cases | 测试规范与 BDD 测试用例 |
| `/docs/` | User-facing documentation (setup, tutorials, architecture) | 用户文档（安装、教程、架构说明） |

---

## AI Agent Workflow Instructions (AI 工作流指令)

当你（AI）被要求开发一个新功能、修改现有功能或修复 Bug 时，**必须严格按照以下工作流执行，不可跳过任何步骤**：

### Step 1: Review Specs (审查与分析)

**在编写任何代码之前，首先阅读 `/specs` 目录下相关的规范文档：**

- 产品需求文档：`/specs/product/`
- 技术设计文档：`/specs/rfc/`
- API 定义：`/specs/api/`
- 测试规范：`/specs/testing/`

**重要规则：**
- 如果用户指令与现有 Spec 冲突，请**立即停止编码**，并指出冲突点，询问用户是否需要先更新 Spec
- 不要在没有上下文的情况下"自由发挥"，强制第一步读取 `/specs` 可以锚定思考范围

### Step 2: Spec-First Update (规范优先)

**如果这是一个新功能，或者需要改变现有的接口/数据库结构，必须首先提议修改或创建相应的 Spec 文档：**

- 新增 API 端点 → 更新 `/specs/api/openapi.yaml` 或创建新的 API 规范
- 新功能 → 创建/更新 `/specs/product/` 文档
- 架构变更 → 在 `/specs/rfc/` 创建 RFC

**等待用户确认 Spec 的修改后，才能进入代码编写阶段。**

### Step 3: Implementation (代码实现)

**编写代码时，必须 100% 遵守 Spec 中的定义：**

- 变量命名规范
- API 路径和端点
- 数据类型和结构
- HTTP 状态码和错误格式
- 认证/授权模式

**禁止事项 (No Gold-Plating)：**
- 不要在代码中擅自添加 Spec 中未定义的功能
- 如果遇到不确定的技术细节，请查阅 `/specs/rfc/` 下的架构约定，不要自行捏造设计模式

### Step 4: Test Against Spec (测试验证)

**根据 `/specs` 中的验收标准（Acceptance Criteria）编写测试：**

- 编写单元测试和集成测试，确保覆盖 Spec 中描述的所有边界情况
- 测试应该验证实现满足 Spec 的正确性属性
- 在测试注释中引用 Spec 文档 ID（例如 `// Validates RFC-001, Property 1`）

---

## Code Generation Rules (代码生成规则)

1. **API 变更**：任何对外部暴露的 API 变更，必须同步修改 `/specs/api/` 规范
2. **数据库变更**：任何 Schema 变更，必须先反映到 `/specs/db/`
3. **功能新增**：新功能必须在 `/specs/product/` 中定义后才能实现
4. **技术决策**：遇到不确定的技术细节，必须遵循 `/specs/rfc/` 中的架构约定
5. **禁止捏造规范**：不要在运行时创建规范。如果没有规范，提议创建一个并等待批准

---

## Project Structure Overview (项目结构)

```
cuflash-attn/
├── specs/                      # Single Source of Truth (规范文档)
│   ├── product/                # Product requirements & acceptance
│   │   └── 001-flash-attention-core.md
│   ├── rfc/                    # Technical design documents (RFCs)
│   │   └── 001-core-architecture.md
│   ├── api/                    # API specifications
│   │   └── 001-public-api.md
│   ├── db/                     # Database/schema specs (if applicable)
│   │   └── README.md
│   └── testing/                # Testing specifications
│       └── 001-test-specification.md
├── docs/                       # User-facing documentation
│   ├── setup/                  # Environment setup guides
│   ├── tutorials/              # Usage tutorials
│   ├── architecture/           # High-level architecture
│   ├── en/                     # English documentation
│   ├── zh/                     # Chinese documentation
│   └── assets/                 # Static assets (images, diagrams)
├── include/                    # Public API headers
│   └── flash_attention.h
├── src/                        # Implementation source code
├── tests/                      # Test suite
├── examples/                   # Usage examples
├── README.md                   # Project entry (English)
├── README.zh-CN.md             # Project entry (Chinese)
├── CONTRIBUTING.md             # Contribution guidelines
├── CHANGELOG.md                # Version history
├── LICENSE                     # MIT License
└── AGENTS.md                   # This file - AI workflow instructions
```

---

## Common Commands (常用命令)

### Building (构建)

```bash
cmake --preset release
cmake --build --preset release
```

### Testing (测试)

```bash
ctest --preset release --output-on-failure
```

### Formatting (格式化)

```bash
find . -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" | xargs clang-format -i
```

---

## Anti-Patterns to Avoid (避免的反模式)

| ❌ DO NOT | ✅ DO |
|-----------|-------|
| Write code without reading specs first | Read specs before writing any code |
| Add features not defined in specs (gold-plating) | Propose spec updates before code changes |
| Invent API designs that aren't documented | Follow spec definitions exactly |
| Skip spec updates when changing interfaces | Wait for user approval on spec modifications |
| Write tests that don't cover spec acceptance criteria | Write tests that validate spec compliance |

---

## Questions? (有问题？)

如果你对任何规范或需求不确定，**请向用户寻求澄清，而不是做假设**。提问总比实现错误要好。

If you're unsure about any spec or requirement, **ask the user for clarification** rather than making assumptions. It's better to ask than to implement incorrectly.
