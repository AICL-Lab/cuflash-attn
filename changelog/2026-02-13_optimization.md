# Changelog - 2026-02-13 性能优化与 Bug 修复

## 功能连接
- **FP16 Forward API 连接**: 将已实现的 FP16 forward kernel (`flash_attention_fp16.cu`) 连接到公共 API (`flash_attention_api.cu`)，FP16 forward pass 现在可以正常使用，不再返回 `UNSUPPORTED_DTYPE`

## Bug 修复
- **Backward pass 流安全问题修复**: 
  - `cudaMemset` 改为 `cudaMemsetAsync` 以确保流内有序执行
  - 在 `cudaFree(D)` 前添加 `cudaStreamSynchronize(stream)` 防止内核未完成时释放临时缓冲区
  - 添加 `cudaMalloc` 返回值检查，避免分配失败后继续执行

## 性能优化
- **float4 向量化内存访问** (`matmul.cuh`): `load_tile_to_shared` 和 `store_tile_from_shared` 在 `BLOCK_COLS` 为 4 的倍数时使用 `float4` 向量化加载/存储，提升全局内存带宽利用率
- **`__launch_bounds__(128)`**: 为所有 CUDA 内核添加 launch bounds 提示，帮助编译器优化寄存器分配，提高 occupancy
- **`--use_fast_math` 编译选项**: 为库和测试启用快速数学运算，加速 `expf`、`logf`、`fmaxf` 等高频数学函数

## 构建配置
- **CUDA 架构扩展**: 新增 SM 89 (Ada Lovelace, RTX 4090) 和 SM 90 (Hopper, H100) 支持

## 影响文件
- `src/flash_attention_api.cu`
- `src/flash_attention_forward.cu`
- `src/flash_attention_backward.cu`
- `src/flash_attention_fp16.cu`
- `src/matmul.cuh`
- `CMakeLists.txt`
