## Summary

<!-- What changed and why? -->

## Change Type

- [ ] Bug fix
- [ ] Refactor / cleanup
- [ ] Docs / workflow
- [ ] Build / CI

## Testing

- [ ] `cmake --preset release && cmake --build --preset release`
- [ ] `ctest --preset release --output-on-failure`（有 CUDA / GPU 时）
- [ ] `cd docs && npm run docs:build`
- [ ] `find . \( -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" \) ! -path "*/build/*" | xargs clang-format -i`

## Notes

<!-- Optional follow-up work, compatibility notes, or cleanup rationale -->
