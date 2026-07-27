#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace cuflash {
namespace impl {

template<typename T>
struct TypeAdapter;

template<>
struct TypeAdapter<float> {
    __device__ __forceinline__ static float to_compute(float value) { return value; }
    __device__ __forceinline__ static float from_compute(float value) { return value; }
};

template<>
struct TypeAdapter<half> {
    __device__ __forceinline__ static float to_compute(half value) { return __half2float(value); }
    __device__ __forceinline__ static half from_compute(float value) { return __float2half(value); }
};

template<>
struct TypeAdapter<__nv_bfloat16> {
    __device__ __forceinline__ static float to_compute(__nv_bfloat16 value) {
        return __bfloat162float(value);
    }
    __device__ __forceinline__ static __nv_bfloat16 from_compute(float value) {
        return __float2bfloat16_rn(value);
    }
};

}  // namespace impl
}  // namespace cuflash
