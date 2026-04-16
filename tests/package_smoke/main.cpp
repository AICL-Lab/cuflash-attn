#include <iostream>

#include "flash_attention.h"

int main() {
    float q = 0.0f;
    float o = 0.0f;
    float l = 0.0f;

    auto err =
        cuflash::flash_attention_forward(nullptr, &q, &q, &o, &l, 1, 1, 1, 32, 1.0f, false, 0);
    if (err != cuflash::FlashAttentionError::NULL_POINTER) {
        std::cerr << "Installed package smoke check failed: expected NULL_POINTER, got "
                  << static_cast<int>(err) << std::endl;
        return 1;
    }

    std::cout << cuflash::get_error_string(cuflash::FlashAttentionError::SUCCESS) << std::endl;
    return 0;
}
