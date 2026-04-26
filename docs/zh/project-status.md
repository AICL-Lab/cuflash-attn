# 项目状态

CuFlash-Attn 当前以 **稳定的 v0.3.0 基线** 维护，并定位为 **可归档的参考实现仓库**。现阶段工作的重点是治理收口、文档质量、工作流简化和缺陷修复，而不是继续扩展功能面。

## 项目覆盖范围

- 从零实现的 CUDA C++ FlashAttention
- `float` 与 `half` 的前向、反向传播
- 支持的 `head_dim`：`32`、`64`、`128`
- 公开的 C++ API，以及面向 Python `ctypes` 的 C ABI
- 通过 OpenSpec 追踪的设计与验证规则

## 维护姿态

仓库明确优先以下目标：

- **清晰优先于堆功能**：不做投机式功能扩张
- **集成面稳定**：示例、文档、ABI 保持一致
- **工程轻量**：基于 preset 的构建、聚焦 CI、双语文档
- **便于交接**：贡献者和后续模型都能从规范与控制文档直接继续

## 开发流程

推荐工作流如下：

1. 先阅读 `openspec/specs/` 中的相关规范
2. 当行为或范围发生变化时，通过 OpenSpec change 记录
3. 仅使用 CMake preset 进行构建
4. 根据当前环境运行对应的验证
5. 非平凡改动在落地前先做 review

## 验证边界

- 本地 CUDA 构建需要可用的 toolkit 和 `nvcc`
- 无 CUDA 设备的环境会自动跳过 GPU 测试
- 文档与工作流治理无需 GPU 即可完成验证

## 权威入口

- [快速开始](/zh/guide/quick-start)
- [从源码构建](/zh/building)
- [API 参考](/zh/api-reference)
- [故障排除](/zh/troubleshooting)
- [CHANGELOG.md](https://github.com/LessUp/cuflash-attn/blob/master/CHANGELOG.md)
- [OpenSpec 规范](https://github.com/LessUp/cuflash-attn/tree/master/openspec/specs)
