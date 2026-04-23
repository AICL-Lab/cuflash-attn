## 关联规范 / Related Spec

<!-- 引用 openspec/specs/ 中的相关规范 ID，例如：REQ-1.1 -->

## 变更类型 / Change Type

- [ ] Bug fix
- [ ] Feature (已有 spec 支持)
- [ ] Docs / Tests only

## 变更摘要 / Summary

<!-- 简要说明做了什么 -->

## 测试 / Testing

- [ ] `cmake --preset release && cmake --build --preset release`
- [ ] `ctest --preset release --output-on-failure`（有 GPU 时）
- [ ] 代码已格式化：`find . -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" | grep -v build | xargs clang-format -i`
- [ ] 测试注释中引用了 spec ID（如 `// Validates REQ-1.1`）
