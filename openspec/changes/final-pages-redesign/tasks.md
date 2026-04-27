# Tasks: final-pages-redesign

- [x] 重新定义 docs 站点首页与语言入口的职责
- [x] 调整导航与页面结构，突出 Guide / API / Spec / Build / Troubleshooting / Status
- [x] 评估并裁剪 PWA / analytics / search 配置的复杂度
- [x] 确保英文与中文站点结构对齐且不重复 README

## 执行结果

### 2026-04-27 文档站点评估

1. **首页职责**:
   - 语言选择页 (docs/index.md): 提供双语入口和自动跳转
   - 英文首页 (docs/en/index.md): Hero 区域 + 快速开始 + 性能数据 + 文档导航
   - 中文首页 (docs/zh/index.md): 与英文结构完全对齐
   - **结论**: 首页职责清晰，不重复 README

2. **导航结构**:
   - 主导航: Guide / Build / API / Troubleshooting
   - Project 下拉菜单: Project Status / Releases / Specs
   - **结论**: 导航结构完整，Spec 已在 Project 菜单中

3. **配置复杂度**:
   - 使用 local search（无需外部服务）
   - 无 PWA 配置
   - 无 analytics 配置
   - **结论**: 已是精简配置，无需裁剪

4. **双语对齐**:
   - 中英文结构完全一致
   - 内容互补，非重复
   - **结论**: 双语结构良好
