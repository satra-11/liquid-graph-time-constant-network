import torch
import pytest

from lgtcn.utils import compute_support_powers
from lgtcn.layers import GraphFilter
from lgtcn.layers import LGTCNLayer
from lgtcn.layers import CfGCNLayer


@pytest.mark.parametrize("N,in_dim,out_dim,K", [(4, 6, 8, 0), (5, 10, 7, 2)])
def test_graph_filter_shape_and_grad(N, in_dim, out_dim, K):
    S = torch.eye(N)
    S_powers = compute_support_powers(S, K)
    filt = GraphFilter(in_dim, out_dim, K)

    x = torch.randn(N, in_dim, requires_grad=True)
    y = filt(x, S_powers)

    # shape
    assert y.shape == (N, out_dim)

    # grad flows
    y.sum().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in filt.parameters())


@pytest.mark.parametrize("LayerCls", [LGTCNLayer, CfGCNLayer])
def test_gnn_layer_clamp_and_grad(LayerCls):
    N, Din, H, K = 4, 8, 16, 2
    S = torch.rand(N, N)                  # random dense graph
    S.fill_diagonal_(0)
    S_powers = compute_support_powers(S, K)

    layer = LayerCls(Din, H, K)
    x = torch.zeros(N, H, requires_grad=True)   # start at zero
    u = torch.randn(N, Din)

    out = layer(x, u, S_powers)                 # forward once

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

    layer = LGTCNLayer(Din, H, K)
    x = torch.rand(N, H)
    u = torch.randn(N, Din)

    x_next = layer(x, u, S_powers, dt=0.05, n_steps=10)
    assert torch.isfinite(x_next).all()
