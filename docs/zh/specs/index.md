# OpenSpec 规范文档

CuFlash-Attn 采用 **OpenSpec** 方法论开发 — 所有设计决策和验证标准都以可执行规范的形式文档化。

---

## 什么是 OpenSpec？

OpenSpec 是一种规范驱动开发方法：

1. **需求先行** — 在任何实现之前记录需求
2. **规范可执行** — 测试注释引用规范 ID
3. **设计可追溯** — 每行代码都可追溯到需求

---

## 规范索引

### 设计规范

| 规范 ID | 标题 | 描述 |
|---------|------|------|
| [DESIGN-001] | [Flash Attention 设计](https://github.com/AICL-Lab/cuflash-attn/blob/master/openspec/specs/design/flash-attention-design.md) | 核心算法设计和内存模型 |

### 验证规范

| 规范 ID | 标题 | 描述 |
|---------|------|------|
| [VERIF-001] | [Flash Attention 验证](https://github.com/AICL-Lab/cuflash-attn/blob/master/openspec/specs/verification/flash-attention-verification.md) | API 契约和数值精度要求 |

---

## 规范结构

每个规范遵循以下结构：

```markdown
# 规范 ID: SPEC-XXX

## 需求

### REQ-X.Y: 需求标题
需求描述。

**属性：**
1. 属性描述
2. 属性描述

**验证：** 如何验证此需求已满足。
```

---

## 规范与测试的关系

测试文件通过注释引用规范：

```cpp
// 验证 REQ-1.1，属性 1 - 数值等价性
TEST(ForwardTest, NumericalEquivalence) {
    // 测试实现...
}
```

这创建了**双向追溯**：
- 从规范 → 测试（通过规范 ID）
- 从测试 → 规范（通过注释）

---

## 在 GitHub 上浏览规范

所有规范都可在仓库中查看：

- **[查看所有规范](https://github.com/AICL-Lab/cuflash-attn/tree/master/openspec/specs)**

::: info
OpenSpec 方法论确保 CuFlash-Attn 在演进过程中保持可维护、可测试和文档完善。
:::
