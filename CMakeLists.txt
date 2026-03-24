cmake_minimum_required(VERSION 3.18)
project(cuflash_attn VERSION 0.1.0 LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

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
option(BUILD_EXAMPLES "Build examples" ON)
option(ENABLE_FAST_MATH "Enable --use_fast_math (faster but less precise)" OFF)

# Source files
set(SOURCES
    src/flash_attention_api.cu
    src/flash_attention_forward.cu
    src/flash_attention_backward.cu
    src/flash_attention_fp16.cu
)

# Library
add_library(cuflash_attn ${SOURCES})
target_link_libraries(cuflash_attn PUBLIC CUDA::cudart)
target_include_directories(cuflash_attn PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(cuflash_attn PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME cuflash_attn
    CUDA_SEPARABLE_COMPILATION ON
)

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

    add_executable(cuflash_attn_tests
        tests/test_main.cpp
        tests/test_online_softmax.cu
        tests/test_forward.cu
        tests/test_backward.cu
        tests/test_causal_mask.cu
        tests/test_error_handling.cu
        tests/test_dtype.cu
        tests/test_numerical_stability.cu
    )
    target_link_libraries(cuflash_attn_tests PRIVATE cuflash_attn gtest)
    target_compile_options(cuflash_attn_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
        >
    )
    set_target_properties(cuflash_attn_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    if(ENABLE_RAPIDCHECK)
        target_link_libraries(cuflash_attn_tests PRIVATE rapidcheck)
        target_compile_definitions(cuflash_attn_tests PRIVATE CUFLASH_ENABLE_RAPIDCHECK=1)
    else()
        target_compile_definitions(cuflash_attn_tests PRIVATE CUFLASH_ENABLE_RAPIDCHECK=0)
    endif()

    add_executable(cuflash_attn_api_smoke tests/test_api_smoke.cpp)
    target_link_libraries(cuflash_attn_api_smoke PRIVATE cuflash_attn)
    set_target_properties(cuflash_attn_api_smoke PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    if(BUILD_TESTING)
        add_test(NAME cuflash_attn_api_smoke COMMAND cuflash_attn_api_smoke)

        include(GoogleTest)
        gtest_discover_tests(cuflash_attn_tests DISCOVERY_TIMEOUT 60)
    endif()

    add_custom_target(check
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS cuflash_attn_tests cuflash_attn_api_smoke
    )
endif()

# Example executable
if(BUILD_EXAMPLES)
    add_executable(example examples/basic_usage.cu)
    target_link_libraries(example PRIVATE cuflash_attn)
    set_target_properties(example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(BUILD_TESTING AND BUILD_EXAMPLES)
    add_test(NAME cuflash_attn_example_build COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target example)
    set_tests_properties(cuflash_attn_example_build PROPERTIES LABELS "smoke")
endif()

if(BUILD_TESTING AND BUILD_SHARED_LIBS)
    set(CUFLASH_PACKAGE_SMOKE_BUILD_DIR ${CMAKE_BINARY_DIR}/package-smoke)
    set(CUFLASH_PACKAGE_SMOKE_INSTALL_DIR ${CMAKE_BINARY_DIR}/package-install)

    add_test(
        NAME cuflash_attn_package_smoke
        COMMAND ${CMAKE_COMMAND}
            -Dproject_build_dir=${CMAKE_BINARY_DIR}
            -Dsource_dir=${CMAKE_SOURCE_DIR}/tests/package_smoke
            -Dbinary_dir=${CUFLASH_PACKAGE_SMOKE_BUILD_DIR}
            -Dinstall_dir=${CUFLASH_PACKAGE_SMOKE_INSTALL_DIR}
            -Dgenerator=${CMAKE_GENERATOR}
            -Dbuild_type=${CMAKE_BUILD_TYPE}
            -Dcuda_architectures=${CMAKE_CUDA_ARCHITECTURES}
            -Dctest_command=${CMAKE_CTEST_COMMAND}
            -P ${CMAKE_SOURCE_DIR}/cmake/run_package_smoke.cmake
    )
    set_tests_properties(cuflash_attn_package_smoke PROPERTIES LABELS "smoke;package")
endif()

# Installation
install(TARGETS cuflash_attn
    EXPORT cuflash_attnTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT cuflash_attnTargets
    FILE cuflash_attnTargets.cmake
    NAMESPACE cuflash::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuflash_attn
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuflash_attnConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cuflash_attnConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuflash_attn
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cuflash_attnConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/cuflash_attnConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cuflash_attnConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuflash_attn
)

export(EXPORT cuflash_attnTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/cuflash_attnTargets.cmake
    NAMESPACE cuflash::
)

export(PACKAGE cuflash_attn)
