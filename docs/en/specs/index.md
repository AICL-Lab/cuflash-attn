# OpenSpec Specifications

CuFlash-Attn is developed following the **OpenSpec** methodology — all design decisions and verification criteria are documented as executable specifications.

---

## What is OpenSpec?

OpenSpec is a specification-driven development approach where:

1. **Requirements are documented first** — Before any implementation
2. **Specs are executable** — Test comments reference spec IDs
3. **Design is traceable** — Every line of code traces to a requirement

---

## Specification Index

### Design Specifications

| Spec ID | Title | Description |
|---------|-------|-------------|
| [DESIGN-001] | [Flash Attention Design](https://github.com/AICL-Lab/cuflash-attn/blob/master/openspec/specs/design/flash-attention-design.md) | Core algorithm design and memory model |

### Verification Specifications

| Spec ID | Title | Description |
|---------|-------|-------------|
| [VERIF-001] | [Flash Attention Verification](https://github.com/AICL-Lab/cuflash-attn/blob/master/openspec/specs/verification/flash-attention-verification.md) | API contract and numerical accuracy requirements |

---

## Spec Structure

Each specification follows this structure:

```markdown
# Spec ID: SPEC-XXX

## Requirements

### REQ-X.Y: Requirement Title
Description of the requirement.

**Properties:**
1. Property description
2. Property description

**Verification:** How to verify this requirement is met.
```

---

## How Specs Relate to Tests

Test files reference specifications using comments:

```cpp
// Validates REQ-1.1, Property 1 - Numerical equivalence
TEST(ForwardTest, NumericalEquivalence) {
    // Test implementation...
}
```

This creates a **bidirectional trace**:
- From spec → test (via spec ID)
- From test → spec (via comment)

---

## Browse Specs on GitHub

All specifications are available in the repository:

- **[View all specs](https://github.com/AICL-Lab/cuflash-attn/tree/master/openspec/specs)**

::: info
The OpenSpec methodology ensures that CuFlash-Attn remains maintainable, testable, and well-documented as it evolves.
:::
