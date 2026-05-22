# Primitive Layer Deepening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 收敛 `online_softmax`、`matmul`、`tile_io` 的 shallow standalone primitive layer，把 host-side validation、kernel launch、type conversion 统一到更深的内部 module。

**Architecture:** 保持 public header 暂时稳定，不在这一轮同时引入 spec/API breakage。先把重复的 host-side validation、dynamic shared-memory launch、kernel launch error handling 折叠进一个内部 helper seam，再把 FP32/FP16 转换收敛到单一 type adapter policy；这样 wrapper 继续兼容，但 implementation 更深、locality 更好。

**Tech Stack:** CUDA C++17, CMake presets, GoogleTest

---

## File Map

- Create: `src/kernels/primitive_api_utils.cuh`
- Create: `src/kernels/impl/type_adapter.cuh`
- Modify: `src/kernels/online_softmax.cu`
- Modify: `src/kernels/matmul.cu`
- Modify: `src/kernels/tile_io.cu`
- Modify: `src/kernels/impl/tile_io.cuh`
- Modify: `src/forward/flash_attention_forward_typed.cu`
- Modify: `src/backward/flash_attention_backward_typed.cu`
- Test: `tests/unit/test_online_softmax.cu`
- Test: `tests/unit/test_matmul.cu`
- Test: `tests/unit/test_dtype.cu`

### Task 1: Centralize primitive wrapper validation and launch helpers

**Files:**
- Create: `src/kernels/primitive_api_utils.cuh`
- Modify: `src/kernels/online_softmax.cu`
- Modify: `src/kernels/matmul.cu`
- Modify: `src/kernels/tile_io.cu`

- [ ] **Step 1: Write the failing test**

Use existing characterization tests as the guard surface:

```cpp
EXPECT_EQ(kernels::online_softmax_forward(nullptr, d_valid, d_valid, 4, 4, 2, stream),
          FlashAttentionError::NULL_POINTER);
EXPECT_EQ(kernels::matmul_ABt<64, 64, 32>(nullptr, d_valid, d_valid, 1.0f, stream),
          FlashAttentionError::NULL_POINTER);
EXPECT_EQ(kernels::load_tile<64, 64>(d_src, d_dst, -1, 0, 128, 128, 128, stream),
          FlashAttentionError::INVALID_DIMENSION);
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
ctest --preset release --output-on-failure -R "OnlineSoftmaxTest|MatmulTest|TileIOTest"
```

Expected: 当前机器无 `nvcc`，会卡在 configure/build；在有 CUDA 环境时，这一步用于证明 refactor 前后行为回归可观测。

- [ ] **Step 3: Write minimal implementation**

Create one internal helper seam:

```cpp
inline FlashAttentionError validate_non_null(std::initializer_list<const void*> ptrs);
inline FlashAttentionError validate_positive_dimensions(std::initializer_list<int> dims);
inline FlashAttentionError validate_tile_window(int row_start, int col_start,
                                                int max_rows, int max_cols, int stride);
template <typename KernelFunc>
inline FlashAttentionError prepare_kernel_launch(KernelFunc kernel, size_t smem_size);
inline FlashAttentionError finish_kernel_launch();
```

Then replace file-local helpers like:

```cpp
FlashAttentionError err = detail::validate_non_null({A, B, C});
if (err != FlashAttentionError::SUCCESS) return err;
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
ctest --preset release --output-on-failure -R "OnlineSoftmaxTest|MatmulTest|TileIOTest"
```

Expected: 在有 CUDA 环境时，既有 null/dimension tests 继续 PASS。

- [ ] **Step 5: Commit**

```bash
git add src/kernels/primitive_api_utils.cuh src/kernels/online_softmax.cu src/kernels/matmul.cu src/kernels/tile_io.cu
git commit -m "refactor(kernels): centralize primitive wrapper helpers"
```

### Task 2: Unify FP32/FP16 conversion policy

**Files:**
- Create: `src/kernels/impl/type_adapter.cuh`
- Modify: `src/kernels/impl/tile_io.cuh`
- Modify: `src/forward/flash_attention_forward_typed.cu`
- Modify: `src/backward/flash_attention_backward_typed.cu`
- Test: `tests/unit/test_dtype.cu`

- [ ] **Step 1: Write the failing test**

Use the current FP16 characterization tests:

```cpp
TEST(DTypeTest, FP16ForwardMatchesFP32);
TEST(DTypeTest, FP16BackwardMatchesFP32);
```

Keep the finite-gradient checks as explicit acceptance criteria:

```cpp
EXPECT_TRUE(std::isfinite(__half2float(h_dQ[i])));
EXPECT_TRUE(std::isfinite(__half2float(h_dK[i])));
EXPECT_TRUE(std::isfinite(__half2float(h_dV[i])));
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
ctest --preset release --output-on-failure -R DTypeTest
```

Expected: 当前机器依旧被 `nvcc` 阻塞；在有 CUDA 环境时，这一步用于确认 refactor 没把 FP16/FP32 路径改坏。

- [ ] **Step 3: Write minimal implementation**

Create one trait:

```cpp
template <typename T>
struct TypeAdapter;

template <>
struct TypeAdapter<float> {
    __device__ static float to_compute(float value) { return value; }
    __device__ static float from_compute(float value) { return value; }
};

template <>
struct TypeAdapter<half> {
    __device__ static float to_compute(half value) { return __half2float(value); }
    __device__ static half from_compute(float value) { return __float2half(value); }
};
```

Then replace scattered conversions like:

```cpp
sum += impl::TypeAdapter<InputT>::to_compute(dO_row[d]) *
       impl::TypeAdapter<InputT>::to_compute(O_row[d]);
L_ptr[global_row] =
    impl::TypeAdapter<InputT>::from_compute(m_tile[row] + logf(l_tile[row]));
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
ctest --preset release --output-on-failure -R DTypeTest
```

Expected: 在有 CUDA 环境时，FP16/FP32 tests PASS。

- [ ] **Step 5: Commit**

```bash
git add src/kernels/impl/type_adapter.cuh src/kernels/impl/tile_io.cuh src/forward/flash_attention_forward_typed.cu src/backward/flash_attention_backward_typed.cu
git commit -m "refactor(kernels): unify primitive type conversion"
```

### Task 3: Preserve and extend the regression surface

**Files:**
- Modify: `tests/unit/test_online_softmax.cu`
- Modify: `tests/unit/test_matmul.cu`

- [ ] **Step 1: Write the failing test**

Keep the two key regressions:

```cpp
TEST_F(OnlineSoftmaxTest, Forward_MultiBlockCrossWarpMatchesReference);
TEST_F(OnlineSoftmaxTest, FinalizeNullNormalizerReturnsError);
TEST_F(MatmulTest, ABt_HeadDim128_LargeTile);
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
ctest --preset release --output-on-failure -R "OnlineSoftmaxTest|MatmulTest"
```

Expected: 在有 CUDA 环境时，旧实现会在 cross-warp/multi-block correctness 和 large-smem launch 上失败。

- [ ] **Step 3: Write minimal implementation**

No new production API. Just keep tests aligned with the bug fixes:

```cpp
EXPECT_NEAR(h_output[i], h_output_expected[i], 1e-4f);
ASSERT_EQ(err, FlashAttentionError::SUCCESS);
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
ctest --preset release --output-on-failure -R "OnlineSoftmaxTest|MatmulTest"
```

Expected: 在有 CUDA 环境时，new regressions PASS。

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_online_softmax.cu tests/unit/test_matmul.cu
git commit -m "test(kernels): lock down primitive wrapper regressions"
```

### Task 4: Final hygiene and verification

**Files:**
- Modify: all touched files

- [ ] **Step 1: Format modified files**

Run:

```bash
clang-format -i src/kernels/primitive_api_utils.cuh src/kernels/impl/type_adapter.cuh src/kernels/impl/tile_io.cuh src/kernels/online_softmax.cu src/kernels/matmul.cu src/kernels/tile_io.cu src/forward/flash_attention_forward_typed.cu src/backward/flash_attention_backward_typed.cu tests/unit/test_online_softmax.cu tests/unit/test_matmul.cu
```

- [ ] **Step 2: Run diff sanity checks**

Run:

```bash
git --no-pager diff --check
git --no-pager diff --stat
```

Expected: no whitespace errors; diff concentrated in primitive-layer files.

- [ ] **Step 3: Run repo verification**

Run:

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
```

Expected: 当前机器因缺 `nvcc` 仍会阻塞在 configure；有 CUDA 环境后，这三步必须全绿。

- [ ] **Step 4: Commit**

```bash
git add src tests docs/superpowers/plans/2026-05-22-primitive-layer-deepening.md
git commit -m "refactor(kernels): deepen standalone primitive layer"
```
