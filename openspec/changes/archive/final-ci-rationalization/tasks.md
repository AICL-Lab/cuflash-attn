# Tasks: final-ci-rationalization

- [x] 修复 `release.yml` 的 Node 环境与 docs 构建前置
- [x] 修复 `codeql.yml` 对 preset-only 构建规则的偏离
- [x] 审核 `ci.yml` / `pages.yml` 的触发条件与重复检查
- [x] 保持无 GPU runner 下的 CI 行为清晰、可解释、维护成本低

## 审核结果

### 2026-04-27 审核结论

1. **ci.yml / pages.yml 触发条件**: 
   - ci.yml 的 docs-build job 是 CI 验证（不部署）
   - pages.yml 是真正的文档部署
   - 两者职责分明，无重复触发问题

2. **codeql.yml**: 
   - 已使用 `cmake --preset release`
   - 符合 preset-only 构建规范，无需修改

3. **release.yml**: 
   - Node.js 20 + npm cache 配置正确
   - docs 构建依赖关系合理

4. **无 GPU runner 行为**: 
   - test_main.cpp 已有 cudaGetDeviceCount 检测保护
   - ctest 自动跳过 GPU 测试
   - 行为清晰、可解释
