# Changelog

所有重要的项目变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [Unreleased]

### 新增
- FP16 反向传播完整实现（`src/flash_attention_backward_fp16.cu`）

### 变更
- 重构全部文档，统一中英文 README 结构
- 更新所有文档反映 FP16 反向传播已支持
- 更新 CI workflow 使用 `Jimver/cuda-toolkit@v0.2.23`
- 更新 Pages workflow 使用 `actions/upload-pages-artifact@v4`
- 所有源代码格式化以通过 clang-format 检查

### 移除
- 删除未使用的 `src/cuflash_ctypes_api.cu`（重复的 C ABI 实现）

---

## [0.1.0] - 2026-03-13

### 新增
- 完整的 API 参考文档（`docs/api.md`）
- FlashAttention 算法详解文档（`docs/algorithm.md`）
- 构建指南文档（`docs/building.md`）
- HonKit 文档站支持，包含中文搜索

### 变更
- CI workflow 调整为 CPU-safe 模式（仅格式检查）
  - 原因：GitHub Hosted Runner 不提供 GPU，CUDA 构建长期失败
  - 现在只运行 clang-format 静态检查

### CI/CD
- 统一 GitHub Actions 工作流命名规范
- 添加 `permissions` 和 `concurrency` 配置
- Pages workflow 添加 `paths` 过滤和 sparse-checkout

---

## [0.1.0-alpha.2] - 2026-03-10

### 新增
- 标准化 CI 工作流（`.github/workflows/ci.yml`）
  - 支持 `push`、`pull_request`、`workflow_dispatch` 触发
  - 添加 clang-format 代码格式检查

### 变更
- 重命名 Pages workflow：`docs.yml` → `pages.yml`

---

## [0.1.0-alpha.1] - 2026-02-13

### 新增
- FP16 前向传播实现（`src/flash_attention_fp16.cu`）
- FP16 前向传播 API 连接，正式支持 `half` 类型输入

### 性能优化
- **向量化内存访问**: `float4` 向量化加载/存储，提升全局内存带宽
- **Launch Bounds**: 所有 CUDA kernel 添加 `__launch_bounds__(128)`
- **Fast Math**: 可选的 `--use_fast_math` 编译选项

### Bug 修复
- **反向传播流安全问题**:
  - `cudaMemset` 改为 `cudaMemsetAsync` 确保流内有序执行
  - 添加 `cudaStreamSynchronize` 防止过早释放临时缓冲区
  - 添加 `cudaMalloc` 返回值检查

### 构建配置
- 新增 SM 89 (Ada Lovelace, RTX 4090) 支持
- 新增 SM 90 (Hopper, H100) 支持

---

## 版本说明

| 版本 | 主要特性 |
|------|----------|
| Unreleased | FP16 反向传播、文档重构 |
| 0.1.0 | 完整文档、CPU-safe CI |
| 0.1.0-alpha.2 | 标准化 CI/CD |
| 0.1.0-alpha.1 | FP16 前向、性能优化 |

---

[Unreleased]: https://github.com/LessUp/cuflash-attn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/LessUp/cuflash-attn/releases/tag/v0.1.0
[0.1.0-alpha.2]: https://github.com/LessUp/cuflash-attn/compare/v0.1.0-alpha.1...v0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/LessUp/cuflash-attn/releases/tag/v0.1.0-alpha.1
