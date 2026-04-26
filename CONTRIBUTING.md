# Contributing to CuFlash-Attn

CuFlash-Attn is in **final-governance / archive-ready stabilization**. Contributions should bias toward
spec alignment, documentation quality, workflow simplification, and bug cleanup over feature expansion.

## Prerequisites

- NVIDIA GPU with Compute Capability 7.0+ (V100+)
- CUDA Toolkit 12.x
- CMake 3.18+, GCC 9+, C++17

## Development Workflow (OpenSpec)

This project uses **OpenSpec** methodology. All changes must follow the spec-driven cycle:

```
/opsx:propose <change-name>  →  review specs  →  /opsx:apply  →  /verify  →  /opsx:archive
```

1. **Read specs first**: `openspec/specs/` is the single source of truth
2. **Propose before coding**: Use `/opsx:propose <name>` to create a change proposal
3. **Reference spec IDs in tests**: e.g., `// Validates REQ-1.1, Property 1`
4. **Format before commit**: `find . -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" | xargs clang-format -i`
5. **Run review before concluding major work**: use `/review` on meaningful changesets

## Repository Operating Mode

CuFlash-Attn is in a **final-governance** phase. Prefer short, explicit cleanup work over broad feature work:

1. **Behavior or API change** → create or update an OpenSpec change first
2. **Docs / workflow / metadata cleanup** → land it under an existing governance-oriented change when possible
3. **Avoid speculative expansion** → if the work does not improve correctness, maintainability, docs quality, or handoff readiness, it is probably out of scope
4. **Use `/review` before landing non-trivial diffs** → especially for cross-file refactors, workflow edits, and API-adjacent changes
5. **Stay in one focused lane** → finish one OpenSpec change cleanly before starting another

## Build & Test

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
```

## Code Style

- **Format**: Google style via clang-format (`.clang-format`)
- **Naming**: namespaces `lower_case`, classes `CamelCase`, functions `lower_case`
- **Commits**: Conventional commits — `feat(scope): description`, `fix(scope): description`

## Branch Strategy

- Work directly on `master`
- Use `git tag v0.x.x` for releases
- Keep feature work in short-lived commits, not long-lived branches
- Avoid branch and workflow sprawl; prefer one focused OpenSpec change at a time
- Avoid `/fleet`-style parallel branch proliferation for this repository
