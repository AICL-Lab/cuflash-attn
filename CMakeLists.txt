cmake_minimum_required(VERSION 3.18)
project(cuflash_attn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Options
option(BUILD_TESTS "Build tests" ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/flash_attention_api.cu
    src/flash_attention_forward.cu
    src/flash_attention_backward.cu
    src/flash_attention_fp16.cu
)

# Library
add_library(cuflash_attn STATIC ${SOURCES})
target_link_libraries(cuflash_attn CUDA::cudart)
target_include_directories(cuflash_attn PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Set CUDA flags
target_compile_options(cuflash_attn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
    >
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    
    # Fetch GoogleTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    
    # Fetch rapidcheck
    FetchContent_Declare(
        rapidcheck
        GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(rapidcheck)
    
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
        GTest::gtest_main
        rapidcheck
        CUDA::cudart
    )
    
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
