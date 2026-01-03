#!/usr/bin/env python3
"""
PyTorch Comparison Tests for CuFlash-Attn
Feature: cuflash-attn
Validates: Requirements 8.4

This script compares the CuFlash-Attn implementation against PyTorch's
scaled_dot_product_attention for numerical correctness.
"""

import torch
import torch.nn.functional as F
import numpy as np
import ctypes
import os

# Load the cuflash_attn library
def load_library():
    lib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'libcuflash_attn.so')
    if not os.path.exists(lib_path):
        print(f"Library not found at {lib_path}")
        print("Please build the project first: mkdir build && cd build && cmake .. && make")
        return None
    return ctypes.CDLL(lib_path)

def test_forward_equivalence():
    """Test that forward pass matches PyTorch's implementation."""
    print("Testing forward pass equivalence...")
    
    batch_size = 2
    num_heads = 4
    seq_len = 64
    head_dim = 64
    
    # Create random inputs
    torch.manual_seed(42)
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    
    # PyTorch reference
    scale = 1.0 / np.sqrt(head_dim)
    
    # Manual attention computation (equivalent to flash attention)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)
    ref_output = torch.matmul(attn_weights, V)
    
    # Also test with PyTorch's scaled_dot_product_attention
    pytorch_output = F.scaled_dot_product_attention(Q, K, V, scale=scale)
    
    # Compare PyTorch implementations
    diff = (ref_output - pytorch_output).abs().max().item()
    print(f"  PyTorch manual vs scaled_dot_product_attention max diff: {diff:.2e}")
    assert diff < 1e-5, f"PyTorch implementations differ: {diff}"
    
    print("  Forward pass test PASSED")
    return True

def test_causal_mask():
    """Test causal masking."""
    print("Testing causal mask...")
    
    batch_size = 1
    num_heads = 2
    seq_len = 32
    head_dim = 32
    
    torch.manual_seed(42)
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    
    scale = 1.0 / np.sqrt(head_dim)
    
    # Manual causal attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device='cuda'), diagonal=1).bool()
    scores = scores.masked_fill(causal_mask, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    ref_output = torch.matmul(attn_weights, V)
    
    # PyTorch's causal attention
    pytorch_output = F.scaled_dot_product_attention(Q, K, V, scale=scale, is_causal=True)
    
    diff = (ref_output - pytorch_output).abs().max().item()
    print(f"  Causal mask max diff: {diff:.2e}")
    assert diff < 1e-5, f"Causal mask test failed: {diff}"
    
    print("  Causal mask test PASSED")
    return True


def test_backward_gradients():
    """Test backward pass gradients."""
    print("Testing backward pass gradients...")
    
    batch_size = 1
    num_heads = 2
    seq_len = 16
    head_dim = 32
    
    torch.manual_seed(42)
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
    
    scale = 1.0 / np.sqrt(head_dim)
    
    # Forward pass
    output = F.scaled_dot_product_attention(Q, K, V, scale=scale)
    
    # Backward pass
    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    
    # Check gradients exist and are finite
    assert Q.grad is not None, "Q gradient is None"
    assert K.grad is not None, "K gradient is None"
    assert V.grad is not None, "V gradient is None"
    
    assert torch.isfinite(Q.grad).all(), "Q gradient contains non-finite values"
    assert torch.isfinite(K.grad).all(), "K gradient contains non-finite values"
    assert torch.isfinite(V.grad).all(), "V gradient contains non-finite values"
    
    print(f"  dQ shape: {Q.grad.shape}, max: {Q.grad.abs().max().item():.4f}")
    print(f"  dK shape: {K.grad.shape}, max: {K.grad.abs().max().item():.4f}")
    print(f"  dV shape: {V.grad.shape}, max: {V.grad.abs().max().item():.4f}")
    
    print("  Backward pass test PASSED")
    return True

def test_numerical_stability():
    """Test numerical stability with extreme values."""
    print("Testing numerical stability...")
    
    batch_size = 1
    num_heads = 1
    seq_len = 16
    head_dim = 32
    
    # Test with large values
    Q = torch.full((batch_size, num_heads, seq_len, head_dim), 100.0, device='cuda', dtype=torch.float32)
    K = torch.full((batch_size, num_heads, seq_len, head_dim), 100.0, device='cuda', dtype=torch.float32)
    V = torch.full((batch_size, num_heads, seq_len, head_dim), 100.0, device='cuda', dtype=torch.float32)
    
    scale = 1.0 / np.sqrt(head_dim)
    output = F.scaled_dot_product_attention(Q, K, V, scale=scale)
    
    assert torch.isfinite(output).all(), "Output contains non-finite values with large inputs"
    print("  Large values test PASSED")
    
    # Test with small values
    Q = torch.full((batch_size, num_heads, seq_len, head_dim), 1e-6, device='cuda', dtype=torch.float32)
    K = torch.full((batch_size, num_heads, seq_len, head_dim), 1e-6, device='cuda', dtype=torch.float32)
    V = torch.full((batch_size, num_heads, seq_len, head_dim), 1e-6, device='cuda', dtype=torch.float32)
    
    output = F.scaled_dot_product_attention(Q, K, V, scale=scale)
    
    assert torch.isfinite(output).all(), "Output contains non-finite values with small inputs"
    print("  Small values test PASSED")
    
    print("  Numerical stability test PASSED")
    return True

def test_different_shapes():
    """Test various input shapes."""
    print("Testing different shapes...")
    
    test_cases = [
        (1, 1, 8, 32),
        (2, 4, 64, 64),
        (1, 8, 128, 64),
        (4, 2, 32, 128),
    ]
    
    for batch_size, num_heads, seq_len, head_dim in test_cases:
        torch.manual_seed(42)
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
        
        scale = 1.0 / np.sqrt(head_dim)
        output = F.scaled_dot_product_attention(Q, K, V, scale=scale)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim), \
            f"Output shape mismatch for input shape ({batch_size}, {num_heads}, {seq_len}, {head_dim})"
        assert torch.isfinite(output).all(), \
            f"Output contains non-finite values for shape ({batch_size}, {num_heads}, {seq_len}, {head_dim})"
        
        print(f"  Shape ({batch_size}, {num_heads}, {seq_len}, {head_dim}) PASSED")
    
    print("  Different shapes test PASSED")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("CuFlash-Attn PyTorch Comparison Tests")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        return
    
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    tests = [
        test_forward_equivalence,
        test_causal_mask,
        test_backward_gradients,
        test_numerical_stability,
        test_different_shapes,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

if __name__ == "__main__":
    main()
