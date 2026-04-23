#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstring>
#include <iostream>

namespace {

bool is_listing_tests(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gtest_list_tests") == 0) {
            return true;
        }
    }
    return false;
}

bool cuda_device_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

}  // namespace

int main(int argc, char** argv) {
    const bool listing_tests = is_listing_tests(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);

    if (!listing_tests && !cuda_device_available()) {
        std::cout << "CUDA device not available; skipping GPU test executable" << std::endl;
        return 0;
    }

    return RUN_ALL_TESTS();
}
