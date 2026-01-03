# Requirements Document

## Introduction

CuFlash-Attn 是一个基于 CUDA C++ 从零实现的高性能 FlashAttention 库。该项目旨在实现 FlashAttention 算法的核心功能，通过分块计算和在线 softmax 技术，在 GPU 上高效计算 Transformer 模型中的注意力机制，同时显著减少显存占用。

## Glossary

- **FlashAttention**: 一种 IO 感知的精确注意力算法，通过分块计算和重计算策略减少 HBM 访问次数
- **Attention_Kernel**: 执行注意力计算的 CUDA 核函数
- **Query_Matrix (Q)**: 查询矩阵，形状为 [batch_size, num_heads, seq_len, head_dim]
- **Key_Matrix (K)**: 键矩阵，形状为 [batch_size, num_heads, seq_len, head_dim]
- **Value_Matrix (V)**: 值矩阵，形状为 [batch_size, num_heads, seq_len, head_dim]
- **Output_Matrix (O)**: 输出矩阵，形状为 [batch_size, num_heads, seq_len, head_dim]
- **Block_Size**: 分块计算时每个块的大小
- **Online_Softmax**: 在线计算 softmax 的技术，无需存储完整的注意力矩阵
- **Tiling**: 将大矩阵分割成小块进行计算的策略
- **HBM**: High Bandwidth Memory，GPU 高带宽显存
- **SRAM**: GPU 片上共享内存
- **Causal_Mask**: 因果掩码，用于自回归模型中防止关注未来位置

## Requirements

### Requirement 1: 前向传播核心计算

**User Story:** 作为深度学习开发者，我希望能够高效计算注意力机制的前向传播，以便在 Transformer 模型中使用。

#### Acceptance Criteria

1. WHEN Query_Matrix, Key_Matrix, Value_Matrix 被提供 THEN Attention_Kernel SHALL 计算 softmax(QK^T / sqrt(d_k)) * V 并输出 Output_Matrix
2. WHEN 输入矩阵维度为 [batch_size, num_heads, seq_len, head_dim] THEN Attention_Kernel SHALL 正确处理所有维度
3. WHEN seq_len 超过 Block_Size THEN Attention_Kernel SHALL 使用分块策略进行计算
4. WHEN 计算 softmax THEN Attention_Kernel SHALL 使用 Online_Softmax 技术避免存储完整的注意力矩阵
5. THE Attention_Kernel SHALL 输出与标准注意力计算数值等价的结果（允许浮点误差在 1e-3 范围内）

### Requirement 2: 反向传播计算

**User Story:** 作为深度学习开发者，我希望能够计算注意力机制的梯度，以便进行模型训练。

#### Acceptance Criteria

1. WHEN 前向传播的输出和上游梯度 dO 被提供 THEN Backward_Kernel SHALL 计算 dQ, dK, dV 梯度
2. WHEN 计算反向传播 THEN Backward_Kernel SHALL 使用重计算策略而非存储中间结果
3. THE Backward_Kernel SHALL 输出与标准反向传播数值等价的梯度（允许浮点误差在 1e-3 范围内）
4. WHEN 反向传播完成 THEN Backward_Kernel SHALL 返回 dQ, dK, dV 三个梯度矩阵

### Requirement 3: 分块计算策略

**User Story:** 作为系统开发者，我希望实现高效的分块计算策略，以便最大化 GPU 利用率并减少显存占用。

#### Acceptance Criteria

1. THE Tiling_Strategy SHALL 将 Q, K, V 矩阵分割成适合 SRAM 的小块
2. WHEN Block_Size 被配置 THEN Tiling_Strategy SHALL 确保每个块能完全加载到共享内存
3. WHEN 处理边界块 THEN Tiling_Strategy SHALL 正确处理 seq_len 不能被 Block_Size 整除的情况
4. THE Tiling_Strategy SHALL 最小化 HBM 到 SRAM 的数据传输次数

### Requirement 4: 在线 Softmax 实现

**User Story:** 作为算法开发者，我希望实现在线 softmax 计算，以便在不存储完整注意力矩阵的情况下计算正确的 softmax 结果。

#### Acceptance Criteria

1. THE Online_Softmax SHALL 维护运行时的最大值 m 和归一化因子 l
2. WHEN 新的块被处理 THEN Online_Softmax SHALL 更新 m 和 l 以保持数值稳定性
3. WHEN 所有块处理完成 THEN Online_Softmax SHALL 产生与标准 softmax 数值等价的结果
4. THE Online_Softmax SHALL 避免数值溢出和下溢问题

### Requirement 5: 因果掩码支持

**User Story:** 作为 NLP 开发者，我希望支持因果掩码，以便在自回归语言模型中使用 FlashAttention。

#### Acceptance Criteria

1. WHEN Causal_Mask 选项启用 THEN Attention_Kernel SHALL 将位置 j > i 的注意力权重设为负无穷
2. WHEN 使用 Causal_Mask THEN Attention_Kernel SHALL 跳过不需要计算的块以提高效率
3. THE Causal_Mask 实现 SHALL 与非掩码版本共享核心计算逻辑

### Requirement 6: 内存管理

**User Story:** 作为系统开发者，我希望高效管理 GPU 内存，以便支持更长的序列长度。

#### Acceptance Criteria

1. THE Memory_Manager SHALL 仅分配 O(N) 的额外显存用于输出和中间状态
2. WHEN 前向传播执行 THEN Memory_Manager SHALL 不分配 O(N²) 的注意力矩阵存储
3. THE Memory_Manager SHALL 正确管理共享内存的分配和使用
4. WHEN CUDA 内存分配失败 THEN Memory_Manager SHALL 返回明确的错误信息

### Requirement 7: API 接口设计

**User Story:** 作为库用户，我希望有简洁易用的 API 接口，以便轻松集成到现有项目中。

#### Acceptance Criteria

1. THE API SHALL 提供 flash_attention_forward 函数用于前向传播
2. THE API SHALL 提供 flash_attention_backward 函数用于反向传播
3. WHEN 输入参数无效 THEN API SHALL 返回描述性错误信息
4. THE API SHALL 支持 float16 和 float32 数据类型
5. THE API SHALL 提供可选的 scale 参数用于自定义缩放因子

### Requirement 8: 数值精度验证

**User Story:** 作为质量保证工程师，我希望验证实现的数值精度，以确保计算结果的正确性。

#### Acceptance Criteria

1. FOR ALL 有效输入，flash_attention_forward 的输出 SHALL 与参考实现的差异在 1e-3 范围内
2. FOR ALL 有效输入，flash_attention_backward 的梯度 SHALL 与参考实现的差异在 1e-3 范围内
3. WHEN 输入包含极端值（接近浮点数边界）THEN 计算 SHALL 保持数值稳定
4. THE 实现 SHALL 通过与 PyTorch 标准注意力的对比测试
