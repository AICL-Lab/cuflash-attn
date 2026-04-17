#pragma once

#include "cuflash/flash_attention.h"

namespace cuflash {

inline FlashAttentionError query_max_dynamic_shared_memory_per_block(int* max_dynamic_smem) {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }

    int default_limit = 0;
    err = cudaDeviceGetAttribute(&default_limit, cudaDevAttrMaxSharedMemoryPerBlock, device);
    if (err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }

    int optin_limit = default_limit;
    err = cudaDeviceGetAttribute(&optin_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (err != cudaSuccess) {
        optin_limit = default_limit;
    }

    *max_dynamic_smem = optin_limit > default_limit ? optin_limit : default_limit;
    return FlashAttentionError::SUCCESS;
}

inline FlashAttentionError prepare_dynamic_smem_launch(const void* kernel_func, size_t smem_size) {
    int max_dynamic_smem = 0;
    FlashAttentionError status = query_max_dynamic_shared_memory_per_block(&max_dynamic_smem);
    if (status != FlashAttentionError::SUCCESS) {
        return status;
    }

    if (smem_size > static_cast<size_t>(max_dynamic_smem)) {
        return FlashAttentionError::CUDA_ERROR;
    }

    if (smem_size > 48 * 1024) {
        cudaError_t err = cudaFuncSetAttribute(
            kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_size));
        if (err != cudaSuccess) {
            return FlashAttentionError::CUDA_ERROR;
        }
    }

    return FlashAttentionError::SUCCESS;
}

}  // namespace cuflash
