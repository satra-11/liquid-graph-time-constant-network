"""Flocking dataset for imitation learning."""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional
import random

from .environment import FlockingEnvironment


class FlockingDataset(Dataset):
    """Dataset for flocking imitation learning.

    Collects trajectories using the expert controller and stores
    (observation, adjacency, expert_action) tuples for training.
    """

    def __init__(
        self,
        num_trajectories: int = 60,
        trajectory_length: float = 2.5,
        dt: float = 0.05,
        agent_counts: Optional[list[int]] = None,
        comm_range: float = 4.0,
        collision_range: float = 1.0,
        max_accel: float = 5.0,
        device: str = "cpu",
    ):
        """Initialize dataset by collecting trajectories.

        Args:
            num_trajectories: Number of trajectories to collect
            trajectory_length: Duration of each trajectory (seconds)
            dt: Sampling time (seconds)
            agent_counts: List of possible agent counts (default: [4,6,10,12,15])
            comm_range: Communication range R
            collision_range: Collision avoidance distance R_CA
            max_accel: Maximum acceleration
            device: Torch device
        """
        if agent_counts is None:
            agent_counts = [4, 6, 10, 12, 15]

        self.num_trajectories = num_trajectories
        self.trajectory_length = trajectory_length
        self.dt = dt
        self.agent_counts = agent_counts
        self.device = torch.device(device)

        # Number of steps per trajectory
        self.steps_per_trajectory = int(trajectory_length / dt)

        # Collect data
        self.data: list[dict] = []
        self._collect_trajectories(
            num_trajectories, comm_range, collision_range, max_accel
        )

    def _collect_trajectories(
        self,
        num_trajectories: int,
        comm_range: float,
        collision_range: float,
        max_accel: float,
    ) -> None:
        """Collect trajectories using expert controller."""
        for traj_idx in range(num_trajectories):
            # Random agent count
            num_agents = random.choice(self.agent_counts)

            env = FlockingEnvironment(
                num_agents=num_agents,
                dt=self.dt,
                comm_range=comm_range,
                collision_range=collision_range,
                max_accel=max_accel,
                device=str(self.device),
            )

            trajectory = self._collect_single_trajectory(env)
            self.data.append(trajectory)

    def _collect_single_trajectory(self, env: FlockingEnvironment) -> dict:
        """Collect a single trajectory using expert controller.

        Returns:
            dict with keys:
                - observations: (T, N, 10)
                - adjacencies: (T, N, N)
                - actions: (T, N, 2)
                - num_agents: int
        """
        observations = []
        adjacencies = []
        actions = []

        obs, adj = env.reset()

        for _ in range(self.steps_per_trajectory):
            # Get expert action
            expert_action = env.compute_expert_action()

            # Store data
            observations.append(obs)
            adjacencies.append(adj)
            actions.append(expert_action)

            # Step environment
            obs, adj, _ = env.step(expert_action)

        return {
            "observations": torch.stack(observations),  # (T, N, 10)
            "adjacencies": torch.stack(adjacencies),  # (T, N, N)
            "actions": torch.stack(actions),  # (T, N, 2)
            "num_agents": env.num_agents,
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a trajectory.

        Returns:
            observations: (T, N, 10)
            adjacencies: (T, N, N)
            actions: (T, N, 2)
        """
        traj = self.data[idx]
        return traj["observations"], traj["adjacencies"], traj["actions"]

    def add_dagger_trajectories(
        self,
        model: torch.nn.Module,
        env_config: dict,
        num_trajectories: int = 10,
    ) -> None:
        """Add trajectories using DAGGER algorithm.

        Runs the learned model to collect states, then labels with expert actions.

        Args:
            model: Trained model to roll out
            env_config: Environment configuration
            num_trajectories: Number of trajectories to add
        """
        model.eval()

        for _ in range(num_trajectories):
            num_agents = random.choice(self.agent_counts)

            env = FlockingEnvironment(
                num_agents=num_agents,
                dt=self.dt,
                device=str(self.device),
                **env_config,
            )

            trajectory = self._collect_dagger_trajectory(model, env)
            self.data.append(trajectory)

    def _collect_dagger_trajectory(
        self,
        model: torch.nn.Module,
        env: FlockingEnvironment,
    ) -> dict:
        """Collect trajectory using model, label with expert."""
        observations = []
        adjacencies = []
        actions = []

        obs, adj = env.reset()
        hidden = None

        with torch.no_grad():
            for _ in range(self.steps_per_trajectory):
                # Get model action (for rollout)
                obs_batch = obs.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 10)
                adj_batch = adj.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)

                model_action, hidden = model(obs_batch, adj_batch, hidden)
                model_action = model_action.squeeze(0).squeeze(0)  # (N, 2)

                # Get expert action (for label)
                expert_action = env.compute_expert_action()

                # Store with expert label
                observations.append(obs)
                adjacencies.append(adj)
                actions.append(expert_action)

                # Step with model action (not expert!)
                obs, adj, _ = env.step(model_action)

        return {
            "observations": torch.stack(observations),
            "adjacencies": torch.stack(adjacencies),
            "actions": torch.stack(actions),
            "num_agents": env.num_agents,
        }


def collate_flocking_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Custom collate function for variable agent counts.

    Since different trajectories may have different numbers of agents,
    we pad to the maximum and return agent counts for masking.

    Returns:
        observations: (B, T, N_max, 10)
        adjacencies: (B, T, N_max, N_max)
        actions: (B, T, N_max, 2)
        agent_counts: list of actual agent counts
    """
    observations, adjacencies, actions = zip(*batch)

    # Find max agents
    max_agents = max(o.shape[1] for o in observations)
    T = observations[0].shape[0]

    B = len(batch)
    obs_padded = torch.zeros(B, T, max_agents, 10)
    adj_padded = torch.zeros(B, T, max_agents, max_agents)
    act_padded = torch.zeros(B, T, max_agents, 2)
    agent_counts = []

    for i, (obs, adj, act) in enumerate(batch):
        N = obs.shape[1]
        obs_padded[i, :, :N, :] = obs
        adj_padded[i, :, :N, :N] = adj
        act_padded[i, :, :N, :] = act
        agent_counts.append(N)

    return obs_padded, adj_padded, act_padded, agent_counts


def setup_flocking_dataloaders(
    num_trajectories: int = 60,
    batch_size: int = 8,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    **dataset_kwargs,
) -> tuple[DataLoader, DataLoader, DataLoader, FlockingDataset]:
    """Create train/val/test dataloaders.

    Args:
        num_trajectories: Total number of trajectories
        batch_size: Batch size
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        **dataset_kwargs: Additional arguments for FlockingDataset

    Returns:
        train_loader, val_loader, test_loader, full_dataset
    """
    dataset = FlockingDataset(
        num_trajectories=num_trajectories,
        **dataset_kwargs,
    )

    # Split
    total = len(dataset)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_flocking_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_flocking_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_flocking_batch,
    )

    return train_loader, val_loader, test_loader, dataset
