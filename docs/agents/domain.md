# Domain Docs Configuration

## Layout

**Single-context**

This repo uses one global context for domain knowledge.

## Files

| File | Purpose |
|------|---------|
| `CONTEXT.md` | Domain language, concepts, and terminology for this project |
| `docs/adr/` | Architecture Decision Records (ADRs) |

## Consumer Rules

Skills like `improve-codebase-architecture`, `diagnose`, and `tdd` will:

1. Read `CONTEXT.md` to understand domain terminology before suggesting changes
2. Read ADRs from `docs/adr/` to respect past architectural decisions
3. Suggest new ADRs when proposing significant architectural changes

## Current Status

- `CONTEXT.md` — **Does not exist yet**. Consider creating one to document this project's domain (CUDA, FlashAttention, memory management patterns, etc.).
- `docs/adr/` — **Does not exist yet**. Consider creating this directory and adding ADRs for major decisions.
