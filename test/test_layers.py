import torch
import pytest

from src.utils.compute_s_powers import compute_s_powers
from src.core.layers import (
    LGTCNLayer,
    CfGCNLayer,
    LTCNLayer,
    NeuralODELayer,
    NeuralGraphODELayer,
)
from src.core.models import (
    CfGCNController,
    NeuralODEController,
    NeuralGraphODEController,
)


@pytest.mark.parametrize("LayerCls", [LGTCNLayer, CfGCNLayer])
def test_gnn_layer_clamp_and_grad(LayerCls):
    N, Din, H, K = 4, 8, 16, 2
    S = torch.rand(N, N)  # random dense graph
    S.fill_diagonal_(0)
    S_powers = compute_s_powers(S, K)
    S_powers_2d = [sp.squeeze(0) for sp in S_powers]

    # CfGCNLayerはresidual=Falseでレガシー動作をテスト
    if LayerCls == CfGCNLayer:
        layer = LayerCls(Din, H, K, residual=False)
    else:
        layer = LayerCls(Din, H, K)
    x = torch.zeros(N, H, requires_grad=True)  # start at zero
    u = torch.randn(N, Din)

    out = layer(x, u, S_powers_2d)  # forward once

    # outputs are in [-1, 1] (Lemma 1 / tanh clamp) when residual=False
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
    S_powers = compute_s_powers(S, K)
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
    S_powers = compute_s_powers(S, K)
    S_powers_2d = [sp.squeeze(0) for sp in S_powers]

    # residual=Falseでテスト
    layer = CfGCNLayer(Din, H, K, residual=False)
    x = torch.randn(N, H)
    u = torch.randn(N, Din)

    # Test different time values
    for t in [0.1, 0.5, 1.0, 2.0]:
        y = layer(x, u, S_powers_2d, t=t)

        assert y.shape == (N, H)
        # 有限値であること（入力がランダムの場合、出力は[-1,1]外になる可能性がある）
        assert torch.isfinite(y).all()


def test_lgtcn_layer_multiple_steps():
    """Test LGTCNLayer with multiple integration steps."""
    N, Din, H, K = 5, 4, 10, 1
    S = torch.eye(N)
    S_powers = compute_s_powers(S, K)
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
    S_powers = compute_s_powers(S, K)
    S_powers_2d = [sp.squeeze(0) for sp in S_powers]

    # Test LGTCNLayer（clampあり）
    layer_lgtcn = LGTCNLayer(Din, H, K)
    x = torch.randn(N, H)
    u = torch.randn(N, Din)
    y = layer_lgtcn(x, u, S_powers_2d)
    assert y.shape == (N, H)
    assert torch.isfinite(y).all()
    assert torch.all(y <= 1.0 + 1e-6)
    assert torch.all(y >= -1.0 - 1e-6)

    # Test CfGCNLayer（residual=Falseでテスト）
    layer_cfgcn = CfGCNLayer(Din, H, K, residual=False)
    y = layer_cfgcn(x, u, S_powers_2d)
    assert y.shape == (N, H)
    assert torch.isfinite(y).all()
    # 入力がランダムの場合、CfGCNLayerの出力は[-1,1]外になる可能性がある

    # Test LTCNLayer separately (different interface)
    ltcn_layer = LTCNLayer(Din, H // 2, 2)  # H // 2 per block, 2 blocks = H total
    y_ltcn = torch.randn(H)
    u_ltcn = torch.randn(Din)

    y_next = ltcn_layer(y_ltcn, u_ltcn)

    assert y_next.shape == (H,)
    assert torch.isfinite(y_next).all()


@pytest.mark.parametrize("matrix_type", ["adjacency", "laplacian", "random_walk"])
def test_cfgcn_controller_matrix_types(matrix_type):
    """Test CfGCNController with different matrix types."""
    B, T, C, H_frame, W_frame = 2, 3, 3, 64, 64
    hidden_dim, K, output_dim = 16, 2, 1
    N_nodes = 64  # Based on AdaptiveAvgPool2d((8, 8)) -> 8*8 = 64 nodes

    controller = CfGCNController(
        frame_height=H_frame,
        frame_width=W_frame,
        hidden_dim=hidden_dim,
        K=K,
        output_dim=output_dim,
        matrix_type=matrix_type,
    )

    frames = torch.randn(B, T, C, H_frame, W_frame)
    # Create a dummy adjacency matrix for testing
    adjacency = torch.randint(0, 2, (B, T, N_nodes, N_nodes), dtype=torch.float32)
    # Ensure it's symmetric for Laplacian/Random Walk (though not strictly required for this test)
    adjacency = adjacency + adjacency.transpose(-1, -2)
    adjacency = adjacency.clamp(0, 1)

    controls, final_hidden = controller(frames, adjacency=adjacency)

    # Check output shapes
    assert controls.shape == (B, T, output_dim)
    # final_hidden: (B, num_layers, N, hidden_dim)
    assert final_hidden.shape[0] == B
    assert final_hidden.shape[-2] == N_nodes
    assert final_hidden.shape[-1] == hidden_dim

    # Check for finite values
    assert torch.isfinite(controls).all()
    assert torch.isfinite(final_hidden).all()


# ============== Neural ODE Tests ==============


def test_neural_ode_layer_basic():
    """Test NeuralODELayer basic functionality."""
    in_dim, hidden_dim = 8, 16

    layer = NeuralODELayer(in_dim, hidden_dim, num_hidden_layers=2)

    y = torch.randn(hidden_dim)
    u_t = torch.randn(in_dim)

    y_next = layer(y, u_t, dt=0.1, n_steps=1)

    assert y_next.shape == (hidden_dim,)
    assert torch.isfinite(y_next).all()


def test_neural_ode_layer_batch():
    """Test NeuralODELayer with batch input."""
    batch_size = 4
    in_dim, hidden_dim = 5, 10

    layer = NeuralODELayer(in_dim, hidden_dim)

    y = torch.randn(batch_size, hidden_dim)
    u_t = torch.randn(batch_size, in_dim)

    y_next = layer(y, u_t, dt=0.05, n_steps=2)

    assert y_next.shape == (batch_size, hidden_dim)
    assert torch.isfinite(y_next).all()


def test_neural_ode_layer_no_input():
    """Test NeuralODELayer with no external input."""
    hidden_dim = 12

    layer = NeuralODELayer(8, hidden_dim)

    y = torch.randn(hidden_dim)

    y_next = layer(y, u_t=None, dt=0.1, n_steps=1)

    assert y_next.shape == (hidden_dim,)
    assert torch.isfinite(y_next).all()


def test_neural_ode_layer_gradient_flow():
    """Test gradient flow through NeuralODELayer."""
    in_dim, hidden_dim = 6, 10

    layer = NeuralODELayer(in_dim, hidden_dim)

    y = torch.randn(hidden_dim, requires_grad=True)
    u_t = torch.randn(in_dim, requires_grad=True)

    y_next = layer(y, u_t, dt=0.1, n_steps=1)
    loss = y_next.sum()

    loss.backward()

    assert y.grad is not None
    assert u_t.grad is not None
    assert all(p.grad is not None for p in layer.parameters())


def test_neural_graph_ode_layer_basic():
    """Test NeuralGraphODELayer basic functionality."""
    N, Din, H, K = 4, 8, 16, 2
    S = torch.rand(N, N)
    S.fill_diagonal_(0)
    S_powers = compute_s_powers(S, K)
    S_powers_2d = [sp.squeeze(0) for sp in S_powers]

    layer = NeuralGraphODELayer(Din, H, K, solver="euler")
    x = torch.randn(N, H)
    u = torch.randn(N, Din)

    out = layer(x, u, S_powers_2d, dt=0.05, n_steps=1)

    assert out.shape == (N, H)
    assert torch.isfinite(out).all()
    # Check output clamping
    assert torch.all(out <= 1.0 + 1e-6)
    assert torch.all(out >= -1.0 - 1e-6)


def test_neural_graph_ode_layer_batch():
    """Test NeuralGraphODELayer with batch input."""
    B, N, Din, H, K = 3, 5, 6, 10, 1
    S = torch.rand(B, N, N)
    for i in range(B):
        S[i].fill_diagonal_(0)
    S_powers = compute_s_powers(S, K)

    layer = NeuralGraphODELayer(Din, H, K, solver="euler")
    x = torch.randn(B, N, H)
    u = torch.randn(B, N, Din)

    out = layer(x, u, S_powers, dt=0.05, n_steps=1)

    assert out.shape == (B, N, H)
    assert torch.isfinite(out).all()


def test_neural_graph_ode_layer_gradient_flow():
    """Test gradient flow through NeuralGraphODELayer."""
    N, Din, H, K = 4, 6, 8, 2
    S = torch.eye(N)
    S_powers = compute_s_powers(S, K)
    S_powers_2d = [sp.squeeze(0) for sp in S_powers]

    layer = NeuralGraphODELayer(Din, H, K, solver="euler")
    x = torch.randn(N, H, requires_grad=True)
    u = torch.randn(N, Din)

    out = layer(x, u, S_powers_2d)

    out.mean().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in layer.parameters())


def test_neural_ode_controller():
    """Test NeuralODEController end-to-end."""
    B, T, C, H_frame, W_frame = 2, 3, 3, 64, 64
    hidden_dim, output_dim = 16, 6

    controller = NeuralODEController(
        frame_height=H_frame,
        frame_width=W_frame,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    frames = torch.randn(B, T, C, H_frame, W_frame)

    controls, final_hidden = controller(frames)

    assert controls.shape == (B, T, output_dim)
    assert final_hidden.shape == (B, hidden_dim)
    assert torch.isfinite(controls).all()
    assert torch.isfinite(final_hidden).all()


@pytest.mark.parametrize("matrix_type", ["adjacency", "laplacian", "random_walk"])
def test_neural_graph_ode_controller_matrix_types(matrix_type):
    """Test NeuralGraphODEController with different matrix types."""
    B, T, C, H_frame, W_frame = 2, 3, 3, 64, 64
    hidden_dim, K, output_dim = 16, 2, 1
    N_nodes = 64

    controller = NeuralGraphODEController(
        frame_height=H_frame,
        frame_width=W_frame,
        hidden_dim=hidden_dim,
        K=K,
        output_dim=output_dim,
        matrix_type=matrix_type,
        solver="euler",
    )

    frames = torch.randn(B, T, C, H_frame, W_frame)
    adjacency = torch.randint(0, 2, (B, T, N_nodes, N_nodes), dtype=torch.float32)
    adjacency = adjacency + adjacency.transpose(-1, -2)
    adjacency = adjacency.clamp(0, 1)

    controls, final_hidden = controller(frames, adjacency=adjacency)

    assert controls.shape == (B, T, output_dim)
    assert final_hidden.shape[0] == B
    assert final_hidden.shape[-2] == N_nodes
    assert final_hidden.shape[-1] == hidden_dim

    assert torch.isfinite(controls).all()
    assert torch.isfinite(final_hidden).all()
