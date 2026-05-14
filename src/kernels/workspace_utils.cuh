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

    // Reserve memory if current capacity is sufficient
    // Returns SUCCESS if capacity is already sufficient or allocation succeeds
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

    // Clear the buffer (free memory)
    void clear() {
        if (buffer_ != nullptr) {
            cudaFree(buffer_);
            buffer_ = nullptr;
            capacity_ = 0;
        }
    }

   private:
    float* buffer_ = nullptr;
    size_t capacity_ = 0;
};

// Thread-safe workspace manager using thread-local storage
// Each thread gets its own workspace, eliminating race conditions
namespace workspace {

// Get thread-local workspace for backward pass
// This is thread-safe: each thread has its own buffer
inline DeviceFloatWorkspace& get_thread_local_workspace() {
    thread_local DeviceFloatWorkspace workspace;
    return workspace;
}

// Reserve workspace for given size (uses thread-local workspace)
inline FlashAttentionError reserve(size_t required_elements) {
    return get_thread_local_workspace().reserve(required_elements);
}

// Get pointer to current thread's workspace data
inline float* data() {
    return get_thread_local_workspace().data();
}

}  // namespace workspace

}  // namespace cuflash
