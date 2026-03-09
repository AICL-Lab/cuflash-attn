# 构建指南

## 环境要求

- **CUDA Toolkit** 11.0+
- **CMake** 3.18+
- **C++17** 兼容编译器（GCC 7+, Clang 5+, MSVC 2017+）
- **（可选）** PyTorch >= 2.0.0，用于对比测试

## 使用 CMake Presets（推荐）

项目提供了预定义的 CMake Presets，简化构建流程：

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

# 最小构建（不含测试）
cmake --preset minimal
cmake --build --preset minimal
```

## 手动构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### 指定 CUDA 路径

如果 CMake 找不到 CUDA，手动指定：

```bash
cmake .. -DCUDAToolkit_ROOT=/usr/local/cuda \
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

## 构建选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `BUILD_TESTS` | ON | 构建 GoogleTest 测试套件 |
| `ENABLE_RAPIDCHECK` | OFF | 启用 RapidCheck 属性测试 |
| `BUILD_SHARED_LIBS` | ON | 构建 `.so` 共享库（Python ctypes 可用） |
| `ENABLE_FAST_MATH` | OFF | 启用 `--use_fast_math`（加速 expf/logf，精度略降） |

示例：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DENABLE_FAST_MATH=ON \
         -DENABLE_RAPIDCHECK=ON
```

## 运行测试

```bash
# 使用 CMake Preset
ctest --preset default --output-on-failure

# 或手动
cd build
ctest --output-on-failure
```

GoogleTest 通过 CMake FetchContent 自动下载，无需手动安装。

### PyTorch 对比测试

确保以 `BUILD_SHARED_LIBS=ON` 构建，然后：

```bash
python tests/test_pytorch_comparison.py
```

该脚本会加载 `build/libcuflash_attn.so`，与 PyTorch 的标准注意力实现对比数值精度。

## 目标 GPU 架构

默认构建支持以下 CUDA 架构：

| 计算能力 | 架构 | 代表 GPU |
|---------|------|---------|
| sm_70 | Volta | V100 |
| sm_75 | Turing | RTX 2080 Ti |
| sm_80 | Ampere | A100 |
| sm_86 | Ampere | RTX 3090 |
| sm_89 | Ada Lovelace | RTX 4090 |
| sm_90 | Hopper | H100 |

如需仅编译特定架构以加速构建：

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86
```
