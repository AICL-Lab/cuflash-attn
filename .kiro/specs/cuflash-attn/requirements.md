# Requirements Document: CuFlash-Attn

## Introduction

CuFlash-Attn 是一个基于 CUDA C++ 从零实现的高性能 FlashAttention 库。该项目旨在实现 FlashAttention 算法的核心功能，通过分块计算和在线 softmax 技术，在 GPU 上高效计算 Transformer 模型中的注意力机制，同时显著减少显存占用。

---

## Glossary

| 术语 | 说明 |
|------|------|
| **FlashAttention** | 一种 IO 感知的精确注意力算法，通过分块计算和重计算策略减少 HBM 访问次数 |
| **Attention_Kernel** | 执行注意力计算的 CUDA 核函数 |
| **Query_Matrix (Q)** | 查询矩阵，形状为 [batch_size, num_heads, seq_len, head_dim] |
| **Key_Matrix (K)** | 键矩阵，形状为 [batch_size, num_heads, seq_len, head_dim] |
| **Value_Matrix (V)** | 值矩阵，形状为 [batch_size, num_heads, seq_len, head_dim] |
| **Output_Matrix (O)** | 输出矩阵，形状为 [batch_size, num_heads, seq_len, head_dim] |
| **Block_Size** | 分块计算时每个块的大小 |
| **Online_Softmax** | 在线计算 softmax 的技术，无需存储完整的注意力矩阵 |
| **Tiling** | 将大矩阵分割成小块进行计算的策略 |
| **HBM** | High Bandwidth Memory，GPU 高带宽显存 |
| **SRAM** | GPU 片上共享内存 |
| **Causal_Mask** | 因果掩码，用于自回归模型中防止关注未来位置 |

---

## Requirements

### REQ-1: 前向传播核心计算

**User Story:** 作为深度学习开发者，我希望能够高效计算注意力机制的前向传播，以便在 Transformer 模型中使用。

| ID | 验收标准 |
|----|----------|
| 1.1 | WHEN Q, K, V 被提供 THEN Kernel SHALL 计算 `softmax(QK^T / sqrt(d_k)) * V` 并输出 O |
| 1.2 | WHEN 输入维度为 [B, H, N, D] THEN Kernel SHALL 正确处理所有维度 |
| 1.3 | WHEN seq_len 超过 Block_Size THEN Kernel SHALL 使用分块策略 |
| 1.4 | WHEN 计算 softmax THEN Kernel SHALL 使用 Online_Softmax 技术 |
| 1.5 | THE Kernel SHALL 输出与标准注意力数值等价的结果（误差 < 1e-3） |

### REQ-2: 反向传播计算

**User Story:** 作为深度学习开发者，我希望能够计算注意力机制的梯度，以便进行模型训练。

| ID | 验收标准 |
|----|----------|
| 2.1 | WHEN 前向输出和 dO 被提供 THEN Kernel SHALL 计算 dQ, dK, dV 梯度 |
| 2.2 | WHEN 计算反向传播 THEN Kernel SHALL 使用重计算策略 |
| 2.3 | THE Kernel SHALL 输出与标准反向传播数值等价的梯度（误差 < 1e-3） |
| 2.4 | WHEN 反向传播完成 THEN Kernel SHALL 返回 dQ, dK, dV 三个梯度矩阵 |

### REQ-3: 分块计算策略

**User Story:** 作为系统开发者，我希望实现高效的分块计算策略，以便最大化 GPU 利用率。

| ID | 验收标准 |
|----|----------|
| 3.1 | THE Tiling 策略 SHALL 将 Q, K, V 分割成适合 SRAM 的小块 |
| 3.2 | WHEN Block_Size 配置后 THEN Tiling SHALL 确保块能完全加载到共享内存 |
| 3.3 | WHEN 处理边界块 THEN Tiling SHALL 正确处理 seq_len 不能被整除的情况 |

### REQ-4: 在线 Softmax 实现

**User Story:** 作为算法开发者，我希望实现在线 softmax 计算，以便不存储完整注意力矩阵。

| ID | 验收标准 |
|----|----------|
| 4.1 | THE Online_Softmax SHALL 维护运行时最大值 m 和归一化因子 l |
| 4.2 | WHEN 新块被处理 THEN Online_Softmax SHALL 更新 m 和 l |
| 4.3 | WHEN 所有块处理完成 THEN 结果 SHALL 与标准 softmax 数值等价 |
| 4.4 | THE Online_Softmax SHALL 避免数值溢出和下溢 |

### REQ-5: 因果掩码支持

**User Story:** 作为 NLP 开发者，我希望支持因果掩码，以便在自回归语言模型中使用。

| ID | 验收标准 |
|----|----------|
| 5.1 | WHEN Causal_Mask 启用 THEN Kernel SHALL 将位置 j > i 的权重设为负无穷 |
| 5.2 | WHEN 使用 Causal_Mask THEN Kernel SHALL 跳过不需要计算的块 |

### REQ-6: 内存管理

**User Story:** 作为系统开发者，我希望高效管理 GPU 内存，以便支持更长的序列长度。

| ID | 验收标准 |
|----|----------|
| 6.1 | THE Memory_Manager SHALL 仅分配 O(N) 的额外显存 |
| 6.2 | WHEN 前向传播执行 THEN 不 SHALL 分配 O(N²) 的注意力矩阵存储 |
| 6.3 | THE Memory_Manager SHALL 正确管理共享内存 |
| 6.4 | WHEN CUDA 内存分配失败 THEN SHALL 返回明确的错误信息 |

### REQ-7: API 接口设计

**User Story:** 作为库用户，我希望有简洁易用的 API 接口，以便轻松集成到现有项目中。

| ID | 验收标准 |
|----|----------|
| 7.1 | THE API SHALL 提供 `flash_attention_forward` 函数 |
| 7.2 | THE API SHALL 提供 `flash_attention_backward` 函数 |
| 7.3 | WHEN 输入参数无效 THEN API SHALL 返回描述性错误信息 |
| 7.4 | THE API SHALL 支持 FP16 和 FP32 数据类型 |
| 7.5 | THE API SHALL 提供可选的 scale 参数 |

### REQ-8: 数值精度验证

**User Story:** 作为质量保证工程师，我希望验证实现的数值精度，以确保计算结果的正确性。

| ID | 验收标准 |
|----|----------|
| 8.1 | FOR ALL 有效输入，forward 输出 SHALL 与参考实现差异 < 1e-3 |
| 8.2 | FOR ALL 有效输入，backward 梯度 SHALL 与参考实现差异 < 1e-3 |
| 8.3 | WHEN 输入包含极端值 THEN 计算 SHALL 保持数值稳定 |
| 8.4 | THE 实现 SHALL 通过 PyTorch 标准注意力对比测试 |

---

## Requirements Traceability Matrix

| 需求 | 测试覆盖 |
|------|----------|
| REQ-1 | Property 1 (前向传播数值等价性) |
| REQ-2 | Property 2 (反向传播梯度等价性) |
| REQ-3 | 单元测试 (分块计算边界) |
| REQ-4 | Property 3 (在线 Softmax 等价性), Property 4 (数值稳定性) |
| REQ-5 | Property 5 (因果掩码正确性) |
| REQ-6 | 错误处理测试 |
| REQ-7 | API 烟雾测试, Property 6 (数据类型支持) |
| REQ-8 | PyTorch 对比测试, 所有属性测试 |
