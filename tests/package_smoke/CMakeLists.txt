cmake_minimum_required(VERSION 3.18)
project(cuflash_attn_package_smoke LANGUAGES CXX CUDA)

find_package(cuflash_attn REQUIRED CONFIG)

add_executable(cuflash_attn_package_smoke main.cpp)
target_link_libraries(cuflash_attn_package_smoke PRIVATE cuflash::cuflash_attn)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME cuflash_attn_package_smoke COMMAND cuflash_attn_package_smoke)
endif()
