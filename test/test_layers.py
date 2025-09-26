import torch
import pytest

from utils import compute_support_powers
from layers import (LGTCNLayer, CfGCNLayer, LTCNLayer)


@pytest.mark.parametrize("LayerCls", [LGTCNLayer, CfGCNLayer])
def test_gnn_layer_clamp_and_grad(LayerCls):
    N, Din, H, K = 4, 8, 16, 2
    S = torch.rand(N, N)                  # random dense graph
    S.fill_diagonal_(0)
    S_powers = compute_support_powers(S, K)
    S_powers_2d = [sp.squeeze(0) for sp in S_powers]

    layer = LayerCls(Din, H, K)
    x = torch.zeros(N, H, requires_grad=True)   # start at zero
    u = torch.randn(N, Din)

    out = layer(x, u, S_powers_2d)                 # forward once

    # outputs are in [-1, 1]   (Lemma 1 / tanh clamp)
    assert torch.all(out <= 1.0 + 1e-6)
    assert torch.all(out >= -1.0 - 1e-6)

    # gradients propagate
    out.mean().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in layer.parameters())


def test_lgtc_time_integration():
    """LGTCNLayer should integrate multiple sub-steps without NaNs/Inf."""
    N, Din, H, K = 3, 5, 12, 1
    S = torch.eye(N)
    S_powers = compute_support_powers(S, K)
    S_powers_2d = [sp.squeeze(0) for sp in S_powers]

    layer = LGTCNLayer(Din, H, K)
    x = torch.rand(N, H)
    u = torch.randn(N, Din)

    x_next = layer(x, u, S_powers_2d, dt=0.05, n_steps=10)
    assert torch.isfinite(x_next).all()


def test_ltcn_layer_basic():
    """LTCNLayer basic functionality test."""
    in_dim, k, num_blocks = 5, 8, 3
    N = num_blocks * k  # total hidden dimension
    
    layer = LTCNLayer(in_dim, k, num_blocks)
    
    # Test forward pass
    y = torch.randn(N)
    u_t = torch.randn(in_dim)
    
    y_next = layer(y, u_t, dt=0.01, n_steps=1)
    
    # Check output shape
    assert y_next.shape == (N,)
    assert torch.isfinite(y_next).all()


def test_ltcn_layer_batch():
    """LTCNLayer batch processing test."""
    batch_size = 4
    in_dim, k, num_blocks = 3, 6, 2
    N = num_blocks * k
    
    layer = LTCNLayer(in_dim, k, num_blocks)
    
    # Test batch forward pass
    y = torch.randn(batch_size, N)
    u_t = torch.randn(batch_size, in_dim)
    
    y_next = layer(y, u_t, dt=0.02, n_steps=5)
    
    # Check output shape
    assert y_next.shape == (batch_size, N)
    assert torch.isfinite(y_next).all()


def test_ltcn_layer_activations():
    """Test different activation functions for LTCNLayer."""
    in_dim, k, num_blocks = 4, 5, 2
    N = num_blocks * k
    
    for activation in ["tanh", "relu", "sigmoid", "htanh"]:
        layer = LTCNLayer(in_dim, k, num_blocks, activation=activation)
        
        y = torch.randn(N)
        u_t = torch.randn(in_dim)
        
        y_next = layer(y, u_t, dt=0.01, n_steps=1)
        
        assert y_next.shape == (N,)
        assert torch.isfinite(y_next).all()


def test_ltcn_layer_clamping():
    """Test output clamping for LTCNLayer."""
    in_dim, k, num_blocks = 3, 4, 2
    N = num_blocks * k
    
    layer = LTCNLayer(in_dim, k, num_blocks, clamp_output=1.0)
    
    # Start with large values
    y = torch.randn(N) * 10
    u_t = torch.randn(in_dim) * 10
    
    y_next = layer(y, u_t, dt=0.05, n_steps=10)
    
    # Check clamping
    assert torch.all(y_next <= 1.0 + 1e-6)
    assert torch.all(y_next >= -1.0 - 1e-6)


def test_ltcn_layer_gradient_flow():
    """Test gradient flow through LTCNLayer."""
    in_dim, k, num_blocks = 3, 4, 2
    N = num_blocks * k
    
    layer = LTCNLayer(in_dim, k, num_blocks)
    
    y = torch.randn(N, requires_grad=True)
    u_t = torch.randn(in_dim, requires_grad=True)
    
    y_next = layer(y, u_t, dt=0.01, n_steps=1)
    loss = y_next.sum()
    
    loss.backward()
    
    # Check gradients exist
    assert y.grad is not None
    assert u_t.grad is not None
    assert all(p.grad is not None for p in layer.parameters())


def test_ltcn_layer_no_input():
    """Test LTCNLayer with no external input (u_t=None)."""
    in_dim, k, num_blocks = 3, 4, 2
    N = num_blocks * k
    
    layer = LTCNLayer(in_dim, k, num_blocks)
    
    y = torch.randn(N)
    
    y_next = layer(y, u_t=None, dt=0.01, n_steps=1)
    
    assert y_next.shape == (N,)
    assert torch.isfinite(y_next).all()


def test_cfgcn_layer_time_parameter():
    """Test CfGCNLayer with different time parameters."""
    N, Din, H, K = 4, 6, 8, 2
    S = torch.eye(N)
    S_powers = compute_support_powers(S, K)
    S_powers_2d = [sp.squeeze(0) for sp in S_powers]
    
    layer = CfGCNLayer(Din, H, K)
    x = torch.randn(N, H)
    u = torch.randn(N, Din)
    
    # Test different time values
    for t in [0.1, 0.5, 1.0, 2.0]:
        y = layer(x, u, S_powers_2d, t=t)
        
        assert y.shape == (N, H)
        assert torch.all(y <= 1.0 + 1e-6)
        assert torch.all(y >= -1.0 - 1e-6)
        assert torch.isfinite(y).all()


def test_lgtcn_layer_multiple_steps():
    """Test LGTCNLayer with multiple integration steps."""
    N, Din, H, K = 5, 4, 10, 1
    S = torch.eye(N)
    S_powers = compute_support_powers(S, K)
    S_powers_2d = [sp.squeeze(0) for sp in S_powers]
    
    layer = LGTCNLayer(Din, H, K)
    x = torch.randn(N, H)
    u = torch.randn(N, Din)
    
    # Test different step counts
    for n_steps in [1, 5, 10, 20]:
        y = layer(x, u, S_powers_2d, dt=0.01, n_steps=n_steps)
        
        assert y.shape == (N, H)
        assert torch.all(y <= 1.0 + 1e-6)
        assert torch.all(y >= -1.0 - 1e-6)
        assert torch.isfinite(y).all()


def test_all_layers_consistency():
    """Test that all layers maintain consistent behavior patterns."""
    N, Din, H, K = 6, 5, 12, 2
    S = torch.rand(N, N)
    S.fill_diagonal_(0)
    S_powers = compute_support_powers(S, K)
    S_powers_2d = [sp.squeeze(0) for sp in S_powers]
    
    # Test LGTCNLayer and CfGCNLayer
    for LayerCls in [LGTCNLayer, CfGCNLayer]:
        layer = LayerCls(Din, H, K)
        x = torch.randn(N, H)
        u = torch.randn(N, Din)
        
        y = layer(x, u, S_powers_2d)
        
        # Basic checks
        assert y.shape == (N, H)
        assert torch.isfinite(y).all()
        assert torch.all(y <= 1.0 + 1e-6)
        assert torch.all(y >= -1.0 - 1e-6)
    
    # Test LTCNLayer separately (different interface)
    ltcn_layer = LTCNLayer(Din, H // 2, 2)  # H // 2 per block, 2 blocks = H total
    y_ltcn = torch.randn(H)
    u_ltcn = torch.randn(Din)
    
    y_next = ltcn_layer(y_ltcn, u_ltcn)
    
    assert y_next.shape == (H,)
    assert torch.isfinite(y_next).all()
