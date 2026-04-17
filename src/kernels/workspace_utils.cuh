#pragma once

#include <cuda_runtime.h>

#include "cuflash/flash_attention.h"

namespace cuflash {

// RAII-managed device memory workspace for intermediate buffers
// Used by backward pass kernels to store the D (denominator) array
class DeviceFloatWorkspace {
   public:
    DeviceFloatWorkspace() = default;

    ~DeviceFloatWorkspace() {
        if (buffer_ != nullptr) {
            cudaFree(buffer_);
        }
    }

    // Disable copy and move
    DeviceFloatWorkspace(const DeviceFloatWorkspace&) = delete;
    DeviceFloatWorkspace& operator=(const DeviceFloatWorkspace&) = delete;
    DeviceFloatWorkspace(DeviceFloatWorkspace&&) = delete;
    DeviceFloatWorkspace& operator=(DeviceFloatWorkspace&&) = delete;

    // Reserve memory if current capacity is insufficient
    FlashAttentionError reserve(size_t required_elements) {
        if (required_elements <= capacity_) {
            return FlashAttentionError::SUCCESS;
        }

        float* new_buffer = nullptr;
        cudaError_t err = cudaMalloc(&new_buffer, required_elements * sizeof(float));
        if (err != cudaSuccess) {
            return err == cudaErrorMemoryAllocation ? FlashAttentionError::OUT_OF_MEMORY
                                                    : FlashAttentionError::CUDA_ERROR;
        }

        if (buffer_ != nullptr) {
            cudaFree(buffer_);
        }

        buffer_ = new_buffer;
        capacity_ = required_elements;
        return FlashAttentionError::SUCCESS;
    }

    float* data() const { return buffer_; }
    size_t capacity() const { return capacity_; }

   private:
    float* buffer_ = nullptr;
    size_t capacity_ = 0;
};

}  // namespace cuflash
