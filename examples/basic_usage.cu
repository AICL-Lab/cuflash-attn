// Basic usage example for CuFlash-Attn

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "flash_attention.h"

int main() {
    // Example dimensions
    const int batch_size = 2;
    const int num_heads = 8;
    const int seq_len = 128;
    const int head_dim = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    std::cout << "CuFlash-Attn Example" << std::endl;
    std::cout << "===================" << std::endl;
    std::cout << "batch_size: " << batch_size << std::endl;
    std::cout << "num_heads: " << num_heads << std::endl;
    std::cout << "seq_len: " << seq_len << std::endl;
    std::cout << "head_dim: " << head_dim << std::endl;
    std::cout << "scale: " << scale << std::endl;
    std::cout << std::endl;
    
    // Calculate sizes
    const size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    const size_t l_size = batch_size * num_heads * seq_len;
    
    // Allocate host memory
    std::vector<float> h_Q(qkv_size);
    std::vector<float> h_K(qkv_size);
    std::vector<float> h_V(qkv_size);
    std::vector<float> h_O(qkv_size);
    std::vector<float> h_L(l_size);
    
    // Initialize with random values
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_L, l_size * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run forward pass
    auto error = cuflash::flash_attention_forward(
        d_Q, d_K, d_V, d_O, d_L,
        batch_size, num_heads, seq_len, head_dim,
        scale, false, 0
    );
    
    if (error != cuflash::FlashAttentionError::SUCCESS) {
        std::cerr << "Error: " << cuflash::get_error_string(error) << std::endl;
        return 1;
    }
    
    // Copy result back
    cudaMemcpy(h_O.data(), d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Flash Attention forward pass completed successfully!" << std::endl;
    std::cout << "Output[0]: " << h_O[0] << std::endl;
    std::cout << "Output shape: [" << batch_size << ", " << num_heads << ", " 
              << seq_len << ", " << head_dim << "]" << std::endl;
    
    // Example with causal masking
    std::cout << std::endl << "Running with causal masking..." << std::endl;
    
    error = cuflash::flash_attention_forward(
        d_Q, d_K, d_V, d_O, d_L,
        batch_size, num_heads, seq_len, head_dim,
        scale, true,  // causal = true
        0
    );
    
    if (error != cuflash::FlashAttentionError::SUCCESS) {
        std::cerr << "Error: " << cuflash::get_error_string(error) << std::endl;
        return 1;
    }
    
    cudaDeviceSynchronize();
    std::cout << "Causal attention completed successfully!" << std::endl;
    
    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
    
    return 0;
}
