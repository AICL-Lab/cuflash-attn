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
option(ENABLE_RAPIDCHECK "Enable RapidCheck property tests" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

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
add_library(cuflash_attn ${SOURCES})
target_link_libraries(cuflash_attn CUDA::cudart)
target_include_directories(cuflash_attn PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(cuflash_attn PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Set CUDA flags
target_compile_options(cuflash_attn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
    >
)

# Tests
if(BUILD_TESTS)
    find_package(GTest QUIET)

    if(NOT GTest_FOUND)
        message(WARNING "GTest not found. Tests will be skipped. Install GTest or configure with -DBUILD_TESTS=OFF.")
    else()
        enable_testing()

        # Fetch rapidcheck
        if(ENABLE_RAPIDCHECK)
            include(FetchContent)
            FetchContent_Declare(
                rapidcheck
                GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
                GIT_TAG master
            )
            FetchContent_MakeAvailable(rapidcheck)
        endif()

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
endif()

# Example executable
add_executable(example examples/basic_usage.cu)
target_link_libraries(example cuflash_attn CUDA::cudart)
