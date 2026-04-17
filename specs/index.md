# Specifications Overview

CuFlash-Attn follows **Spec-Driven Development (SDD)** methodology. All implementation details, requirements, and design decisions are documented in the `/specs/` directory, which serves as the **Single Source of Truth** for the project.

---

## Directory Structure

```
specs/
├── product/            # Product requirements & acceptance criteria
│   └── 001-flash-attention-core.md
├── rfc/                # Technical design documents (RFCs)
│   └── 001-core-architecture.md
├── api/                # API specifications
│   └── 001-public-api.md
├── db/                 # Database/schema specifications (if applicable)
│   └── README.md
└── testing/            # Testing specifications
    └── 001-test-specification.md
```

---

## Specification Documents

### Product Requirements (产品需求)

| Document | Description |
|----------|-------------|
| [001-flash-attention-core.md](product/001-flash-attention-core.md) | Core feature definitions, user stories, and acceptance criteria for the FlashAttention library |

**Requirements Covered:**
- REQ-1: Forward Pass Core Computation
- REQ-2: Backward Pass Computation
- REQ-3: Tiling Strategy
- REQ-4: Online Softmax Implementation
- REQ-5: Causal Masking Support
- REQ-6: Memory Management
- REQ-7: API Interface Design
- REQ-8: Numerical Precision Validation

---

### Technical Design / RFCs (技术设计)

| Document | Status | Description |
|----------|--------|-------------|
| [001-core-architecture.md](rfc/001-core-architecture.md) | ✅ Accepted | Core architecture, components, algorithms, and correctness properties |

**Design Covered:**
- System architecture and component design
- API interfaces (C++ and C ABI)
- Data models and tensor layouts
- Forward and backward algorithms
- FP16 support strategy
- Error handling
- Testing strategy

---

### API Specifications (接口规范)

| Document | Description |
|----------|-------------|
| [001-public-api.md](api/001-public-api.md) | Complete API definition including C++ interfaces, C ABI bindings, and usage examples |

**API Covered:**
- Forward pass API (FP32/FP16)
- Backward pass API (FP32/FP16)
- Error handling and types
- Tensor layout conventions
- C ABI interface for Python integration

---

### Testing Specifications (测试规范)

| Document | Description |
|----------|-------------|
| [001-test-specification.md](testing/001-test-specification.md) | Testing strategy, correctness properties, and test coverage requirements |

**Properties Covered:**
- Property 1: Forward Pass Numerical Equivalence
- Property 2: Backward Pass Gradient Equivalence
- Property 3: Online Softmax Equivalence
- Property 4: Numerical Stability
- Property 5: Causal Mask Correctness
- Property 6: Data Type Support
- Property 7: Invalid Input Error Handling

---

## How to Use Specs

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

## Spec-Driven Workflow

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Step 1          │    │  Step 2          │    │  Step 3          │    │  Step 4          │
│  Review Specs    │───▶│  Spec-First      │───▶│  Implementation  │───▶│  Test Against    │
│  审查规范         │    │  Update 规范优先  │    │  代码实现         │    │  Spec 测试验证    │
└──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
```

See [AGENTS.md](../AGENTS.md) for detailed AI agent workflow instructions.

---

## Requirements Traceability Matrix

| Requirement | RFC Section | Test Coverage |
|-------------|-------------|---------------|
| REQ-1 (Forward Pass) | Forward Algorithm | Property 1 |
| REQ-2 (Backward Pass) | Backward Algorithm | Property 2 |
| REQ-3 (Tiling Strategy) | Block Configuration | Unit Tests |
| REQ-4 (Online Softmax) | Online Softmax State | Property 3, 4 |
| REQ-5 (Causal Masking) | Algorithm Details | Property 5 |
| REQ-6 (Memory Management) | Memory Management | Error Handling Tests |
| REQ-7 (API Interface) | API Interface | Property 6, 7 |
| REQ-8 (Numerical Precision) | Correctness Properties | All Properties |
