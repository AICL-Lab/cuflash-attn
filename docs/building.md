# 构建指南

本指南介绍如何从源码构建 CuFlash-Attn。

## 目录

- [环境要求](#环境要求)
- [使用 CMake Presets](#使用-cmake-presets推荐)
- [手动构建](#手动构建)
- [构建选项](#构建选项)
- [运行测试](#运行测试)
- [目标 GPU 架构](#目标-gpu-架构)

---

## 环境要求

| 依赖 | 最低版本 | 说明 |
|------|----------|------|
| **CUDA Toolkit** | 11.0 | NVIDIA GPU 编译器 |
| **CMake** | 3.18 | 构建系统 |
| **C++ 编译器** | C++17 | GCC 7+, Clang 5+, MSVC 2017+ |
| **PyTorch** (可选) | 2.0.0 | 用于对比测试 |

---

## 使用 CMake Presets（推荐）

项目提供了预定义的 CMake Presets，简化构建流程：

| Preset | 构建类型 | 测试 | 用途 |
|--------|----------|------|------|
| `default` | Debug | ✅ | 开发调试 |
| `release` | Release | ✅ | 生产使用 |
| `release-fast-math` | Release | ✅ | 高性能（精度略降） |
| `minimal` | Release | ❌ | 最小构建 |

### 构建命令

```bash
# Debug 构建（含测试）
cmake --preset default
cmake --build --preset default
ctest --preset default

# Release 优化构建
cmake --preset release
cmake --build --preset release

# Release + fast math
cmake --preset release-fast-math
cmake --build --preset release-fast-math

# 最小构建（不含测试和示例）
cmake --preset minimal
cmake --build --preset minimal
```

---

## 手动构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### 指定 CUDA 路径

如果 CMake 找不到 CUDA，手动指定：

```bash
# Linux/macOS
cmake .. -DCUDAToolkit_ROOT=/usr/local/cuda \
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# Windows (PowerShell)
cmake .. -DCUDAToolkit_ROOT="$env:CUDA_PATH" `
         -DCMAKE_CUDA_COMPILER="$env:CUDA_PATH\bin\nvcc.exe"
```

### 跨平台说明

| 平台 | 共享库扩展名 |
|------|-------------|
| Linux | `.so` |
| macOS | `.dylib` |
| Windows | `.dll` |

Python 测试脚本会自动检测平台并加载对应库文件。

---

## 构建选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `BUILD_TESTS` | ON | 构建 GoogleTest 测试套件 |
| `ENABLE_RAPIDCHECK` | OFF | 启用 RapidCheck 属性测试 |
| `BUILD_SHARED_LIBS` | ON | 构建共享库 |
| `BUILD_EXAMPLES` | ON | 构建示例程序 |
| `ENABLE_FAST_MATH` | OFF | 启用 `--use_fast_math`（加速 expf/logf，精度略降） |

### 示例配置

```bash
# 高性能 Release 构建
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DENABLE_FAST_MATH=ON \
         -DENABLE_RAPIDCHECK=ON

# 仅构建静态库
cmake .. -DBUILD_SHARED_LIBS=OFF \
         -DBUILD_TESTS=OFF \
         -DBUILD_EXAMPLES=OFF
```

---

## 运行测试

### 单元测试

```bash
# 使用 CMake Preset（推荐）
ctest --preset default --output-on-failure

# 或指定构建目录
ctest --test-dir build/default --output-on-failure

# 运行特定测试
ctest --preset default -R ForwardTest
```

GoogleTest 通过 CMake FetchContent 自动下载，无需手动安装。

### PyTorch 对比测试

```bash
python tests/test_pytorch_comparison.py
```

**要求**:
- 以 `BUILD_SHARED_LIBS=ON` 构建
- 安装 PyTorch >= 2.0.0

**库路径查找顺序**:
1. 环境变量 `CUFLASH_LIB`
2. `build/default/` 目录
3. `build/release/` 目录

---

## 目标 GPU 架构

默认构建支持以下 CUDA 架构：

| 计算能力 | 架构代号 | 代表 GPU |
|---------|----------|----------|
| sm_70 | Volta | V100 |
| sm_75 | Turing | RTX 2080 Ti |
| sm_80 | Ampere | A100 |
| sm_86 | Ampere | RTX 3090 |
| sm_89 | Ada Lovelace | RTX 4090 |
| sm_90 | Hopper | H100 |

### 自定义目标架构

```bash
# 仅支持 RTX 3090 / A100
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86

# 支持多个架构
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86;89"
```

### 架构选择建议

| 使用场景 | 推荐配置 |
|----------|----------|
| 开发调试 | `-DCMAKE_CUDA_ARCHITECTURES=86` (快速编译) |
| A100 集群 | `-DCMAKE_CUDA_ARCHITECTURES=80` |
| H100 集群 | `-DCMAKE_CUDA_ARCHITECTURES=90` |
| 通用发布 | 默认（支持全部架构） |
