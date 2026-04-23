# GitHub Copilot 项目指令 / Project Instructions

**语言**: 始终使用中文回复，代码注释和 API 文档保持英文。
**Language**: Always respond in Chinese. Keep code comments and API docs in English.

---

## 项目概述

CuFlash-Attn 是从零实现的 CUDA C++ FlashAttention 库，当前版本 **v0.2.0**，目标是尽快完善收尾发布 v0.3.0。

- **核心算法**: FlashAttention tiling + online softmax，O(N) 内存复杂度
- **技术栈**: CUDA C++17 / CMake 3.18+ / Google Test / VitePress
- **API 设计**: 双 API（C++ namespace + C ABI for Python ctypes）
- **支持类型**: FP32 (`float`) + FP16 (`half`)，前向和反向传播

---

## 开发工作流（OpenSpec）

```
/opsx:propose <name>  →  实现  →  /verify  →  /opsx:archive
```

**必须先读规范**: `openspec/specs/` 是唯一真相来源，修改前必须查阅。

---

## 关键约束（避免这些错误）

### CUDA 特有陷阱
- **Stream safety**: 使用 `cudaMemsetAsync`（而非 `cudaMemset`）以维持流顺序
- **内存对齐**: Q/K/V 张量 layout 必须是 `[batch, heads, seq_len, head_dim]`
- **head_dim 限制**: 只支持 32、64、128（kernel tile 大小硬编码）
- **cudaMalloc 返回值**: 必须检查并向上报告 `FlashAttentionError`
- **工作空间**: 反向传播需要 workspace，使用后必须释放

### 构建规则
```bash
# 始终使用 preset，不要直接调用 cmake -B
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure   # 无 GPU 时自动跳过
```

### 代码格式
```bash
# 提交前必须格式化（CI 强制检查）
find . -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" | \
  grep -v build | xargs clang-format -i
```
格式规范: Google style，IndentWidth=4，ColumnLimit=100（见 `.clang-format`）

### API 错误处理
```cpp
// 始终返回 FlashAttentionError，不要 throw 异常
// 0 = success, 非零 = 失败原因
if (err != cudaSuccess) return FlashAttentionError::CUDA_ERROR;
```

---

## 目录结构（关键路径）

```
openspec/specs/         ← 规范文档（唯一真相来源）
include/cuflash/        ← 公开 API 头文件
src/api/                ← API 分发层
src/forward/            ← FP32/FP16 前向 kernel
src/backward/           ← FP32/FP16 反向 kernel
src/kernels/            ← 内部 .cuh 工具（不对外暴露）
tests/unit/             ← 8 个单元测试文件
tests/integration/      ← PyTorch 数值对比测试
```

---

## 测试引用规范

```cpp
// 测试注释必须引用 spec ID
// Validates REQ-1.1, Property 1 - 数值等价性
TEST(ForwardTest, NumericalEquivalence) { ... }
```

---

## 不要做（Anti-patterns）

| ❌ 禁止 | ✅ 应该 |
|---------|---------|
| 直接写代码不读规范 | 先读 `openspec/specs/` |
| 添加 spec 未定义的功能 | 先提 spec 变更再写代码 |
| 提交前不格式化 | 运行 clang-format |
| 创建长期分支 | 直接在 master 工作 |
| 使用同步 CUDA ops（cudaMemset）| 用 async 版本（cudaMemsetAsync）|
