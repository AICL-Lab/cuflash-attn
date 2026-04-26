# AGENTS.md — CuFlash-Attn AI 代理指令

> **回复语言**: 始终使用**中文**回复。代码注释和 API 文档保持英文。

---

## 项目本质

CuFlash-Attn 是从零实现的 CUDA C++ FlashAttention 库，使用 **OpenSpec** 规范驱动开发。当前目标：**完成最终治理收敛，形成可归档的稳定样板**。

| 属性 | 值 |
|------|-----|
| 版本 | 0.3.0（稳定基线） |
| GPU 要求 | Compute Capability 7.0+（V100→H100, sm_70-sm_90） |
| 默认架构 | sm_80, sm_86（A100 + RTX 30/40 系列） |
| head_dim | 只支持 **32、64、128**（kernel tile 硬编码） |
| 数据类型 | FP32 + FP16，含前向和反向传播 |
| 张量 layout | `[batch_size, num_heads, seq_len, head_dim]` |
| 当前工作重点 | 文档/流程/AI 工具链收敛 + 长尾缺陷清理 |

---

## OpenSpec 工作流

**所有代码变更必须通过 OpenSpec 流程：**

```
1. /opsx:propose <name>     创建变更提案（proposal.md + design.md + tasks.md）
2. 阅读相关规范              openspec/specs/ 是唯一真相来源
3. /opsx:apply <name>       实现变更
4. /verify                  格式检查 + 构建 + 测试
5. /opsx:archive <name>     归档完成的变更
```

规范文档位置：
- `openspec/specs/design/flash-attention-design.md` — 产品需求 + 技术设计
- `openspec/specs/verification/flash-attention-verification.md` — API 规范 + 测试规范

### 轻量收尾工作流

项目当前处于最终治理阶段，推荐以下短链路：

1. 先判断是 **行为变更** 还是 **治理收尾**
2. 行为、API、测试语义变化：先建立或更新 OpenSpec change
3. 文档、工作流、GitHub metadata、AI 指令收敛：优先归入治理类 change，而不是额外扩功能
4. 单次只推进一个聚焦的 change，避免多条长期并行分支
5. 重要改动落地前执行 `/review`

---

## 构建与测试

```bash
# 标准构建（始终使用 preset，禁止直接 cmake -B）
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure  # 无 GPU 时自动跳过

# Debug + AddressSanitizer
cmake --preset debug-asan && cmake --build --preset debug-asan

# 格式化（提交前必须执行，CI 强制检查）
find . \( -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" \) \
  ! -path "*/build/*" | xargs clang-format -i
```

**CI 说明**: GitHub runner 无 GPU，ctest 自动跳过所有 GPU 测试（`test_main.cpp` 中有 cudaGetDeviceCount 检查）。

---

## CUDA 关键陷阱（Critical Pitfalls）

### Stream Safety（流安全）
```cuda
// ❌ 错误：cudaMemset 在流外同步，破坏流顺序
cudaMemset(workspace, 0, size);

// ✅ 正确：async 版本维持流内顺序
cudaMemsetAsync(workspace, 0, size, stream);
cudaStreamSynchronize(stream);  // 在 workspace 释放前
```

### 错误处理
```cpp
// 每个 CUDA 调用后必须检查返回值
if (cudaMalloc(&ptr, size) != cudaSuccess) {
    return FlashAttentionError::CUDA_ERROR;
}
// 不要 throw，通过 FlashAttentionError 枚举向上传递
```

### 内存布局（不可变约定）
```
张量 offset = ((batch * num_heads + head) * seq_len + seq) * head_dim + dim
// Q/K/V/O 全部遵循此 offset 计算
```

### Kernel 启动参数
```cuda
// 块大小固定为 128（__launch_bounds__ 设置）
// Grid = (num_heads * batch_size) 个块
// 不要修改 tiling 逻辑而不更新规范
```

---

## 代码规范

- **格式**: Google style + IndentWidth=4 + ColumnLimit=100（`.clang-format`）
- **命名**: namespace `lower_case`，class `CamelCase`，function `lower_case`，const `UPPER_CASE`
- **可见性**: 默认隐藏，导出用 `CUFLASH_EXPORT`（`include/cuflash/export.h`）
- **Commit**: Conventional commits — `feat(api):`、`fix(kernel):`、`docs(guide):`

---

## 测试规范

```cpp
// 测试注释必须引用 spec ID
// Validates REQ-1.1, Property 1 - 数值等价性（误差 < 1e-3）
TEST(ForwardTest, NumericalEquivalence) { ... }
```

**测试目录结构:**
```
tests/unit/        8 个单元测试（forward/backward/causal/fp16/softmax/error/stress）
tests/integration/ PyTorch 数值对比（需要 GPU + PyTorch）
tests/package_smoke/ 安装验证
```

---

## Git 分支策略

- **始终在 master 工作**，禁止创建长期未合并的功能分支
- 使用 `git tag v0.x.x` 标记版本，通过 tag push 触发 release workflow
- 每次 push 自动触发 CI（format check + build + test skip on no GPU）
- 避免 `/fleet` 和云端/本地大量未合并分支，优先使用长会话串行推进

---

## 工具协作模式

| 工具 | 最佳使用场景 |
|------|-------------|
| **Copilot** (本工具) | 快速代码补全、文档编写、`/review` 代码审查 |
| **Claude Code** | 复杂跨文件重构、`/opsx:propose`、`/opsx:apply` |
| **Codex CLI** | 批处理任务、自动化 PR 处理 |

**Review 流程**: 实现后运行 `/review` 进行代码审查再提交。

**收尾原则**:
- 优先修正规范漂移、文档错误、CI 复杂度、示例失真
- 不做与最终稳定态无关的功能扩张
- 如果变更跨越规范、代码、文档三个面，必须把它们一起收口

### AI / LSP 工具边界

- 本地编辑默认使用 `clangd` + CMake Tools，依赖 `cmake --preset release` 生成
  `build/release/compile_commands.json`
- 没有 `nvcc` 的机器允许继续推进 docs / specs / workflow / AI 指令收尾，但 `.cu`
  语义诊断和 configure/build 会退化
- CLI Skills 优先用于 plan / review / verify / handoff 这类低上下文流程
- MCP / `gh` 只在需要远端 GitHub 状态、Actions、metadata 时按需使用
- 默认不配置项目级 Copilot Plugin；若未来引入，必须证明现有 instructions + skills +
  `gh` / MCP 无法覆盖高频痛点

---

## 禁止模式（Anti-patterns）

| ❌ 禁止 | ✅ 应该 |
|---------|---------|
| 不读规范就写代码 | 先读 `openspec/specs/`，再实现 |
| 添加 spec 未定义的功能 | 先 `/opsx:propose` 更新规范 |
| 使用同步 CUDA 操作（cudaMemset） | 用 cudaMemsetAsync |
| 不检查 CUDA 错误 | 每次调用都检查，返回 FlashAttentionError |
| 直接 cmake -B 构建 | 使用 cmake --preset |
| 创建长期功能分支 | 直接在 master 上 commit |
| 提交前不格式化 | 运行 clang-format |
| 修改 head_dim 支持范围不更新 kernel tiling | 同步更新 kernel 和规范 |
