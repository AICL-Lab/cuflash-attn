#pragma once

#include <initializer_list>

#include "cuflash/flash_attention.h"
#include "kernel_launch_utils.cuh"

namespace cuflash {
namespace kernels {
namespace detail {

inline FlashAttentionError validate_non_null(std::initializer_list<const void*> pointers) {
    for (const void* pointer : pointers) {
        if (pointer == nullptr) {
            return FlashAttentionError::NULL_POINTER;
        }
    }
    return FlashAttentionError::SUCCESS;
}

inline FlashAttentionError validate_positive_dimensions(std::initializer_list<int> dimensions) {
    for (int dimension : dimensions) {
        if (dimension <= 0) {
            return FlashAttentionError::INVALID_DIMENSION;
        }
    }
    return FlashAttentionError::SUCCESS;
}

inline FlashAttentionError validate_tile_window(int row_start, int col_start, int max_rows,
                                                int max_cols, int stride) {
    FlashAttentionError status = validate_positive_dimensions({max_rows, max_cols, stride});
    if (status != FlashAttentionError::SUCCESS) {
        return status;
    }

    if (row_start < 0 || col_start < 0 || row_start >= max_rows || col_start >= max_cols) {
        return FlashAttentionError::INVALID_DIMENSION;
    }

    return FlashAttentionError::SUCCESS;
}

template<typename KernelFunc>
inline FlashAttentionError prepare_kernel_launch(KernelFunc kernel, size_t smem_size) {
    return prepare_dynamic_smem_launch(reinterpret_cast<const void*>(kernel), smem_size);
}

inline FlashAttentionError finish_kernel_launch() {
    return (cudaGetLastError() == cudaSuccess) ? FlashAttentionError::SUCCESS
                                               : FlashAttentionError::CUDA_ERROR;
}

}  // namespace detail
}  // namespace kernels
}  // namespace cuflash
