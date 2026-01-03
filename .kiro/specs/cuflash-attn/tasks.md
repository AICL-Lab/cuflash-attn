# Implementation Plan: CuFlash-Attn

## Overview

本实现计划将 FlashAttention 设计分解为增量式的编码任务。每个任务构建在前一个任务之上，确保代码始终可编译和测试。

## Tasks

- [x] 1. 项目结构和基础设施
  - [x] 1.1 创建项目目录结构和 CMake 构建系统
    - 创建 `src/`, `include/`, `tests/` 目录
    - 配置 CMakeLists.txt 支持 CUDA 编译
    - 配置 Google Test 和 rapidcheck 依赖
    - _Requirements: 7.1, 7.2_

  - [x] 1.2 定义核心数据类型和错误处理
    - 创建 `include/flash_attention.h` 头文件
    - 定义 `FlashAttentionError` 枚举
    - 定义 CUDA_CHECK 宏
    - _Requirements: 7.3, 6.4_

- [x] 2. 在线 Softmax 实现
  - [x] 2.1 实现在线 Softmax 设备函数
    - 创建 `src/online_softmax.cuh`
    - 实现 `OnlineSoftmaxState` 结构体
    - 实现 `init()` 和 `update()` 方法
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 2.2 编写在线 Softmax 属性测试
    - **Property 3: 在线 Softmax 等价性**
    - **Validates: Requirements 4.3**

- [x] 3. 前向传播 Kernel 实现
  - [x] 3.1 实现分块矩阵乘法辅助函数
    - 创建 `src/matmul.cuh`
    - 实现共享内存加载函数
    - 实现分块矩阵乘法
    - _Requirements: 3.1, 3.2_

  - [x] 3.2 实现前向传播 Kernel 核心逻辑
    - 创建 `src/flash_attention_forward.cu`
    - 实现 `flash_attention_forward_kernel` 模板
    - 实现 Q, K, V 分块加载
    - 实现在线 softmax 累积
    - 实现输出写回
    - _Requirements: 1.1, 1.3, 1.4, 3.3_

  - [x] 3.3 实现因果掩码支持
    - 在前向 kernel 中添加因果掩码逻辑
    - 实现块级跳过优化
    - _Requirements: 5.1, 5.2_

  - [x] 3.4 实现前向传播 API 函数
    - 创建 `src/flash_attention_api.cu`
    - 实现参数验证
    - 实现 grid/block 配置
    - 实现 `flash_attention_forward` 函数
    - _Requirements: 7.1, 7.3, 7.5_

  - [x] 3.5 编写前向传播属性测试
    - **Property 1: 前向传播数值等价性**
    - **Validates: Requirements 1.1, 1.2, 1.5, 7.5, 8.1**

  - [x] 3.6 编写因果掩码属性测试
    - **Property 5: 因果掩码正确性**
    - **Validates: Requirements 5.1**

- [x] 4. Checkpoint - 前向传播验证
  - 确保所有测试通过，如有问题请询问用户

- [x] 5. 反向传播 Kernel 实现
  - [x] 5.1 实现反向传播辅助计算
    - 创建 `src/flash_attention_backward.cu`
    - 实现 D = rowsum(dO ⊙ O) 计算
    - _Requirements: 2.1_

  - [x] 5.2 实现反向传播 Kernel 核心逻辑
    - 实现 `flash_attention_backward_kernel` 模板
    - 实现注意力权重重计算
    - 实现 dQ, dK, dV 梯度计算
    - _Requirements: 2.1, 2.2, 2.4_

  - [x] 5.3 实现反向传播因果掩码支持
    - 在反向 kernel 中添加因果掩码逻辑
    - _Requirements: 5.1_

  - [x] 5.4 实现反向传播 API 函数
    - 实现参数验证
    - 实现 `flash_attention_backward` 函数
    - _Requirements: 7.2, 7.3_

  - [x] 5.5 编写反向传播属性测试
    - **Property 2: 反向传播梯度等价性**
    - **Validates: Requirements 2.1, 2.3, 2.4, 8.2**

- [x] 6. Checkpoint - 反向传播验证
  - 确保所有测试通过，如有问题请询问用户

- [x] 7. 数值稳定性和数据类型支持
  - [x] 7.1 添加 float16 支持
    - 添加 half 类型模板特化
    - 实现 float16 到 float32 的转换辅助函数
    - _Requirements: 7.4_

  - [x] 7.2 编写数据类型支持属性测试
    - **Property 6: 数据类型支持**
    - **Validates: Requirements 7.4**

  - [x] 7.3 编写数值稳定性属性测试
    - **Property 4: 数值稳定性**
    - **Validates: Requirements 4.4, 8.3**

- [x] 8. 错误处理完善
  - [x] 8.1 完善输入验证
    - 添加维度检查
    - 添加空指针检查
    - 添加 head_dim 支持检查
    - _Requirements: 7.3_

  - [x] 8.2 编写错误处理属性测试
    - **Property 7: 无效输入错误处理**
    - **Validates: Requirements 7.3**

- [x] 9. 集成测试和文档
  - [x] 9.1 创建 PyTorch 对比测试
    - 编写 Python 测试脚本
    - 与 torch.nn.functional.scaled_dot_product_attention 对比
    - _Requirements: 8.4_

  - [x] 9.2 编写使用示例和 README
    - 创建 examples/ 目录
    - 编写基本使用示例
    - 编写 README.md
    - _Requirements: 7.1, 7.2_

- [x] 10. Final Checkpoint - 完整验证
  - 确保所有测试通过，如有问题请询问用户

## Notes

- 每个任务都引用了具体的需求以便追溯
- Checkpoint 任务用于增量验证
- 属性测试验证通用正确性属性
- 单元测试验证具体示例和边界情况
