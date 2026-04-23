# Contributing to CuFlash-Attn

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

