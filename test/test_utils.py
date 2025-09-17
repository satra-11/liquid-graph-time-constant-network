import torch
import pytest

from lgtcn.utils import compute_support_powers
from lgtcn.utils import GraphFilter


@pytest.mark.parametrize("N,in_dim,out_dim,K", [(4, 6, 8, 0), (5, 10, 7, 2)])
def test_graph_filter_shape_and_grad(N, in_dim, out_dim, K):
    S = torch.eye(N)
    S_powers = compute_support_powers(S, K)
    # Extract the 2D matrices from the batched format
    S_powers_2d = [sp.squeeze(0) for sp in S_powers]
    filt = GraphFilter(in_dim, out_dim, K)

    x = torch.randn(N, in_dim, requires_grad=True)
    y = filt(x, S_powers_2d)

    # shape
    assert y.shape == (N, out_dim)

    # grad flows
    y.sum().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in filt.parameters())


def test_compute_support_powers_basic():
    """Test basic functionality of compute_support_powers."""
    N = 4
    K = 2
    S = torch.rand(N, N)
    
    powers = compute_support_powers(S, K)
    
    # Should return K+1 powers: [I, S, S^2]
    assert len(powers) == K + 1
    
    # Check shapes
    for i, power in enumerate(powers):
        assert power.shape == (1, N, N)
    
    # Check that first power is identity (approximately)
    eye_diff = torch.abs(powers[0].squeeze(0) - torch.eye(N))
    assert torch.all(eye_diff < 1e-6)


def test_compute_support_powers_k_zero():
    """Test compute_support_powers with K=0."""
    N = 3
    K = 0
    S = torch.rand(N, N)
    
    powers = compute_support_powers(S, K)
    
    # Should return only identity matrix
    assert len(powers) == 1
    assert powers[0].shape == (1, N, N)
    
    # Check that it's identity
    eye_diff = torch.abs(powers[0].squeeze(0) - torch.eye(N))
    assert torch.all(eye_diff < 1e-6)


def test_compute_support_powers_batch():
    """Test compute_support_powers with batched input."""
    batch_size = 2
    N = 3
    K = 1
    S = torch.rand(batch_size, N, N)
    
    powers = compute_support_powers(S, K)
    
    # Should return K+1 powers: [I, S]
    assert len(powers) == K + 1
    
    # Check shapes
    for power in powers:
        assert power.shape == (batch_size, N, N)
    
    # Check that first power is identity for each batch
    for b in range(batch_size):
        eye_diff = torch.abs(powers[0][b] - torch.eye(N))
        assert torch.all(eye_diff < 1e-6)


def test_graph_filter_batch_processing():
    """Test GraphFilter with batch processing."""
    batch_size = 3
    N, in_dim, out_dim, K = 4, 5, 6, 1
    
    filt = GraphFilter(in_dim, out_dim, K)
    
    # Test with batch input
    x = torch.randn(batch_size, N, in_dim, requires_grad=True)
    S = torch.eye(N).expand(batch_size, -1, -1)  # Expand S to match batch size
    S_powers = compute_support_powers(S, K)
    
    y = filt(x, S_powers)
    
    # Check shape
    assert y.shape == (batch_size, N, out_dim)
    
    # Check gradients
    y.sum().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in filt.parameters())


def test_graph_filter_no_k0():
    """Test GraphFilter with include_k0=False."""
    N, in_dim, out_dim, K = 4, 3, 5, 2
    
    filt = GraphFilter(in_dim, out_dim, K, include_k0=False)
    
    x = torch.randn(N, in_dim)
    S = torch.eye(N)
    S_powers = compute_support_powers(S, K)
    S_powers_2d = [sp.squeeze(0) for sp in S_powers]
    
    y = filt(x, S_powers_2d)
    
    # Check shape
    assert y.shape == (N, out_dim)
    assert torch.isfinite(y).all()


def test_graph_filter_different_k_values():
    """Test GraphFilter with different K values."""
    N, in_dim, out_dim = 5, 4, 6
    
    for K in [0, 1, 3, 5]:
        filt = GraphFilter(in_dim, out_dim, K)
        
        x = torch.randn(N, in_dim)
        S = torch.rand(N, N)
        S_powers = compute_support_powers(S, K)
        S_powers_2d = [sp.squeeze(0) for sp in S_powers]
        
        y = filt(x, S_powers_2d)
        
        assert y.shape == (N, out_dim)
        assert torch.isfinite(y).all()


def test_compute_support_powers_large_k():
    """Test compute_support_powers with larger K values."""
    N = 4
    K = 5
    S = torch.rand(N, N) * 0.1  # Small values to prevent overflow
    
    powers = compute_support_powers(S, K)
    
    assert len(powers) == K + 1
    
    for power in powers:
        assert power.shape == (1, N, N)
        assert torch.isfinite(power).all()


def test_graph_filter_edge_cases():
    """Test GraphFilter edge cases."""
    # Test with very small dimensions
    N, in_dim, out_dim, K = 1, 1, 1, 0
    
    filt = GraphFilter(in_dim, out_dim, K)
    x = torch.randn(N, in_dim)
    S = torch.eye(N)
    S_powers = compute_support_powers(S, K)
    S_powers_2d = [sp.squeeze(0) for sp in S_powers]
    
    y = filt(x, S_powers_2d)
    
    assert y.shape == (N, out_dim)
    assert torch.isfinite(y).all()