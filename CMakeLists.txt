cmake_minimum_required(VERSION 3.18)
project(cuflash_attn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Options
option(BUILD_TESTS "Build tests" ON)
option(ENABLE_RAPIDCHECK "Enable RapidCheck property tests" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(ENABLE_FAST_MATH "Enable --use_fast_math (faster but less precise)" OFF)

# Source files
set(SOURCES
    src/flash_attention_api.cu
    src/flash_attention_forward.cu
    src/flash_attention_backward.cu
    src/flash_attention_fp16.cu
    src/cuflash_ctypes_api.cu
)

# Library
add_library(cuflash_attn ${SOURCES})
target_link_libraries(cuflash_attn CUDA::cudart)
target_include_directories(cuflash_attn PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(cuflash_attn PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Set CUDA flags
# Note: Optimization level (-O2/-O3) is controlled by CMAKE_BUILD_TYPE.
target_compile_options(cuflash_attn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        $<$<BOOL:${ENABLE_FAST_MATH}>:--use_fast_math>
    >
)

# Tests
if(BUILD_TESTS)
    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
        GIT_SHALLOW    TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    if(ENABLE_RAPIDCHECK)
        FetchContent_Declare(
            rapidcheck
            GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
            GIT_TAG        ff6af6fc683159deb51c543b065eba14dfcf329b
            GIT_SHALLOW    FALSE
        )
        FetchContent_MakeAvailable(rapidcheck)
    endif()

    enable_testing()

    # Test executable
    add_executable(cuflash_attn_tests
        tests/test_online_softmax.cu
        tests/test_forward.cu
        tests/test_backward.cu
        tests/test_causal_mask.cu
        tests/test_error_handling.cu
        tests/test_dtype.cu
        tests/test_numerical_stability.cu
    )

    target_link_libraries(cuflash_attn_tests
        cuflash_attn
        gtest_main
        CUDA::cudart
    )

    if(ENABLE_RAPIDCHECK)
        target_link_libraries(cuflash_attn_tests rapidcheck)
        target_compile_definitions(cuflash_attn_tests PRIVATE CUFLASH_ENABLE_RAPIDCHECK=1)
    else()
        target_compile_definitions(cuflash_attn_tests PRIVATE CUFLASH_ENABLE_RAPIDCHECK=0)
    endif()

    target_compile_options(cuflash_attn_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
        >
    )

    include(GoogleTest)
    gtest_discover_tests(cuflash_attn_tests)
endif()

# Example executable
add_executable(example examples/basic_usage.cu)
target_link_libraries(example cuflash_attn CUDA::cudart)
