# Specifications Overview

CuFlash-Attn follows **OpenSpec** methodology. All implementation details, requirements, and design decisions are documented in this directory, which serves as the **Single Source of Truth** for the project.

---

## Directory Structure

```
openspec/specs/
├── design/               # 设计规格（产品需求 + 技术设计）
│   └── flash-attention-design.md
└── verification/         # 验证规格（API + 测试）
    └── flash-attention-verification.md
```

---

## Specification Documents

### Design Specification (设计规格)

| Document | Description |
|----------|-------------|
| [flash-attention-design.md](design/flash-attention-design.md) | 产品需求、技术架构、算法设计、正确性属性 |

**覆盖内容 (Coverage):**
- **产品需求 (Product Requirements):** REQ-1 至 REQ-8
  - REQ-1: Forward Pass Core Computation
  - REQ-2: Backward Pass Computation
  - REQ-3: Tiling Strategy
  - REQ-4: Online Softmax Implementation
  - REQ-5: Causal Masking Support
  - REQ-6: Memory Management
  - REQ-7: API Interface Design
  - REQ-8: Numerical Precision Validation

- **技术设计 (Technical Design):**
  - System architecture and component design
  - API interfaces (C++ and C ABI)
  - Data models and tensor layouts
  - Forward and backward algorithms
  - FP16 support strategy
  - Error handling
  - Correctness properties (Property 1-7)

---

### Verification Specification (验证规格)

| Document | Description |
|----------|-------------|
| [flash-attention-verification.md](verification/flash-attention-verification.md) | API 定义、测试规范、覆盖要求 |

**覆盖内容 (Coverage):**
- **API Specification (接口规范):**
  - Forward pass API (FP32/FP16)
  - Backward pass API (FP32/FP16)
  - Error handling and types
  - Tensor layout conventions
  - C ABI interface for Python integration
  - Usage examples

- **Testing Specification (测试规范):**
  - Property 1: Forward Pass Numerical Equivalence
  - Property 2: Backward Pass Gradient Equivalence
  - Property 3: Online Softmax Equivalence
  - Property 4: Numerical Stability
  - Property 5: Causal Mask Correctness
  - Property 6: Data Type Support
  - Property 7: Invalid Input Error Handling

---

## OpenSpec Workflow

### Creating Changes (创建变更)

```
/opsx:propose <change-name>   # Create new change proposal
```

This creates:
```
openspec/changes/<change-name>/
├── proposal.md    # Intent (why and what) - 意图（为什么和做什么）
├── specs/         # Delta specs (ADDED/MODIFIED/REMOVED) - 增量规格
├── design.md      # Technical approach (optional) - 技术方案（可选）
└── tasks.md       # Implementation checklist - 实现清单
```

### Implementing Changes (实现变更)

```
/opsx:apply <change-name>     # Begin implementation
```

### Completing Changes (完成变更)

```
/opsx:archive <change-name>   # Archive completed change
```

---

## How to Use Specs (如何使用规格)

### For Developers (开发者指南)

1. **Read specs first** - 在实现功能或修复 Bug 之前，先阅读相关规范文档
2. **Follow spec definitions** - 严格按照规范定义实现，不要添加未定义的功能
3. **Propose spec updates** - 在修改接口或添加功能之前，先提议更新规范
4. **Write tests** - 编写测试验证实现符合验收标准

### For Contributors (贡献者指南)

1. Review relevant specs in pull requests
2. Ensure code changes comply with spec definitions
3. Update specs when proposing API or feature changes

---

## Requirements Traceability Matrix

| Requirement | Design Section | Test Coverage |
|-------------|----------------|---------------|
| REQ-1 (Forward Pass) | Forward Algorithm | Property 1 |
| REQ-2 (Backward Pass) | Backward Algorithm | Property 2 |
| REQ-3 (Tiling Strategy) | Block Configuration | Unit Tests |
| REQ-4 (Online Softmax) | Online Softmax State | Property 3, 4 |
| REQ-5 (Causal Masking) | Algorithm Details | Property 5 |
| REQ-6 (Memory Management) | Memory Management | Error Handling Tests |
| REQ-7 (API Interface) | API Interface | Property 6, 7 |
| REQ-8 (Numerical Precision) | Correctness Properties | All Properties |

---

## Configuration

See `openspec/config.yaml` for project rules and context.

See `AGENTS.md` for AI agent workflow instructions.
