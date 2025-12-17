"""Tests for flocking module."""

import pytest
import torch

from src.flocking import (
    FlockingEnvironment,
    FlockingDataset,
    FlockingLGTCN,
    FlockingLTCN,
)


class TestFlockingEnvironment:
    """Tests for FlockingEnvironment."""

    def test_init(self):
        """Test environment initialization."""
        env = FlockingEnvironment(num_agents=10)
        assert env.num_agents == 10
        assert env.dt == 0.05
        assert env.comm_range == 4.0
        assert env.collision_range == 1.0

    def test_reset(self):
        """Test environment reset."""
        env = FlockingEnvironment(num_agents=8)
        obs, adj = env.reset()

        assert obs.shape == (8, 10)
        assert adj.shape == (8, 8)
        assert env.positions.shape == (8, 2)
        assert env.velocities.shape == (8, 2)
        assert 0 <= env.leader_idx < 8

    def test_step(self):
        """Test environment step."""
        env = FlockingEnvironment(num_agents=5)
        env.reset()

        actions = torch.zeros(5, 2)
        obs, adj, done = env.step(actions)

        assert obs.shape == (5, 10)
        assert adj.shape == (5, 5)
        assert done is False

    def test_expert_action(self):
        """Test expert controller produces valid actions."""
        env = FlockingEnvironment(num_agents=6, max_accel=5.0)
        env.reset()

        expert_actions = env.compute_expert_action()

        assert expert_actions.shape == (6, 2)
        # Actions should be within saturation limits
        assert expert_actions.abs().max() <= 5.0

    def test_adjacency_symmetric(self):
        """Test adjacency matrix is symmetric."""
        env = FlockingEnvironment(num_agents=10)
        env.reset()

        adj = env.compute_adjacency()

        # Adjacency should be symmetric
        assert torch.allclose(adj, adj.T)

    def test_observations_leader_follower(self):
        """Test leader and follower have different observations."""
        env = FlockingEnvironment(num_agents=4)
        env.reset()

        obs = env.get_observations()

        # Leader has one-hot [1, 0]
        leader_obs = obs[env.leader_idx]
        assert leader_obs[8].item() == 1.0
        assert leader_obs[9].item() == 0.0

        # Followers have one-hot [0, 1]
        for i in range(4):
            if i != env.leader_idx:
                assert obs[i, 8].item() == 0.0
                assert obs[i, 9].item() == 1.0

    def test_flocking_error(self):
        """Test flocking error computation."""
        env = FlockingEnvironment(num_agents=5)
        env.reset()

        # Set all velocities to same value
        env.velocities = torch.ones(5, 2)

        error = env.compute_flocking_error()
        assert error == pytest.approx(0.0, abs=1e-6)


class TestFlockingDataset:
    """Tests for FlockingDataset."""

    def test_init_creates_data(self):
        """Test dataset initialization collects trajectories."""
        dataset = FlockingDataset(
            num_trajectories=3,
            trajectory_length=0.5,
            dt=0.1,
            agent_counts=[4, 6],
        )

        assert len(dataset) == 3

    def test_getitem_shapes(self):
        """Test dataset returns correct shapes."""
        dataset = FlockingDataset(
            num_trajectories=2,
            trajectory_length=0.5,
            dt=0.1,
            agent_counts=[5],
        )

        obs, adj, actions = dataset[0]

        # trajectory_length=0.5, dt=0.1 -> 5 steps
        assert obs.shape[0] == 5
        assert obs.shape[1] == 5  # num_agents
        assert obs.shape[2] == 10  # observation dim

        assert adj.shape[0] == 5
        assert adj.shape[1] == 5
        assert adj.shape[2] == 5

        assert actions.shape[0] == 5
        assert actions.shape[1] == 5
        assert actions.shape[2] == 2


class TestFlockingLGTCN:
    """Tests for FlockingLGTCN model."""

    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        model = FlockingLGTCN(
            input_dim=10,
            hidden_dim=32,
            output_dim=2,
            K=2,
        )

        B, T, N = 2, 5, 8
        obs = torch.randn(B, T, N, 10)
        adj = torch.ones(B, T, N, N)

        actions, hidden = model(obs, adj)

        assert actions.shape == (B, T, N, 2)
        assert hidden.shape == (B, 1, N, 32)  # num_layers=1

    def test_gradient_flow(self):
        """Test gradients flow through the model."""
        model = FlockingLGTCN(
            input_dim=10,
            hidden_dim=16,
            output_dim=2,
            K=2,
        )

        B, T, N = 1, 3, 4
        obs = torch.randn(B, T, N, 10, requires_grad=True)
        adj = torch.ones(B, T, N, N)

        actions, _ = model(obs, adj)
        loss = actions.sum()
        loss.backward()

        assert obs.grad is not None
        assert obs.grad.abs().sum() > 0


class TestFlockingLTCN:
    """Tests for FlockingLTCN model."""

    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        model = FlockingLTCN(
            input_dim=10,
            hidden_dim=32,
            output_dim=2,
        )

        B, T, N = 2, 5, 8
        obs = torch.randn(B, T, N, 10)

        # LTCN doesn't use adjacency
        actions, hidden = model(obs, None)

        assert actions.shape == (B, T, N, 2)
        assert hidden.shape == (B, N, 32)

    def test_gradient_flow(self):
        """Test gradients flow through the model."""
        model = FlockingLTCN(
            input_dim=10,
            hidden_dim=16,
            output_dim=2,
        )

        B, T, N = 1, 3, 4
        obs = torch.randn(B, T, N, 10, requires_grad=True)

        actions, _ = model(obs, None)
        loss = actions.sum()
        loss.backward()

        assert obs.grad is not None
        assert obs.grad.abs().sum() > 0


class TestModelsComparison:
    """Tests comparing LGTCN and LTCN."""

    def test_same_output_dim(self):
        """Test both models produce same output dimension."""
        lgtcn = FlockingLGTCN(input_dim=10, hidden_dim=32, output_dim=2)
        ltcn = FlockingLTCN(input_dim=10, hidden_dim=32, output_dim=2)

        B, T, N = 2, 4, 6
        obs = torch.randn(B, T, N, 10)
        adj = torch.ones(B, T, N, N)

        lgtcn_out, _ = lgtcn(obs, adj)
        ltcn_out, _ = ltcn(obs, None)

        assert lgtcn_out.shape == ltcn_out.shape
