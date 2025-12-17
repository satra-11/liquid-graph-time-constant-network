"""Flocking simulation environment based on arXiv:2404.13982."""

import torch
from typing import Optional


class FlockingEnvironment:
    """Leader-Follower Flocking environment with expert controller.

    Implements the flocking control scenario from the LGTC paper:
    - N agents with double integrator dynamics
    - One agent is the leader, others are followers
    - Expert controller uses Olfati-Saber potential for collision avoidance
    """

    def __init__(
        self,
        num_agents: int = 10,
        dt: float = 0.05,
        comm_range: float = 4.0,
        collision_range: float = 1.0,
        max_accel: float = 5.0,
        target_range: float = 20.0,
        w_p: float = 1.0,
        device: str = "cpu",
    ):
        """Initialize the flocking environment.

        Args:
            num_agents: Number of agents (N)
            dt: Sampling time T (seconds)
            comm_range: Communication range R (meters)
            collision_range: Collision avoidance distance R_CA (meters)
            max_accel: Maximum acceleration (m/s^2)
            target_range: Target position range (meters)
            w_p: Gain for leader controller
            device: Torch device
        """
        self.num_agents = num_agents
        self.dt = dt
        self.comm_range = comm_range
        self.collision_range = collision_range
        self.max_accel = max_accel
        self.target_range = target_range
        self.w_p = w_p
        self.device = torch.device(device)

        # State variables (initialized in reset)
        self.positions: torch.Tensor = torch.empty(0)
        self.velocities: torch.Tensor = torch.empty(0)
        self.leader_idx: int = 0
        self.target: torch.Tensor = torch.empty(0)

    def reset(
        self,
        num_agents: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reset environment with random initial conditions.

        Args:
            num_agents: Override number of agents (optional)

        Returns:
            observations: (N, 10) feature vectors for each agent
            adjacency: (N, N) communication adjacency matrix
        """
        if num_agents is not None:
            self.num_agents = num_agents

        N = self.num_agents

        # Initialize positions with inter-distance 0.6-1.0m
        # Use grid-based initialization with random offsets
        self.positions = self._init_positions(N)

        # Initialize velocities uniformly from [-2, 2] m/s
        self.velocities = (torch.rand(N, 2, device=self.device) - 0.5) * 4.0

        # Select random leader
        self.leader_idx = torch.randint(0, N, (1,)).item()

        # Set random target within 20m of leader
        leader_pos = self.positions[self.leader_idx]
        target_offset = (torch.rand(2, device=self.device) - 0.5) * self.target_range
        self.target = leader_pos + target_offset

        return self.get_observations(), self.compute_adjacency()

    def _init_positions(self, n: int) -> torch.Tensor:
        """Initialize agent positions with proper spacing.

        Places agents in a grid pattern with random offsets,
        ensuring inter-distance is between 0.6 and 1.0 meters.
        """
        # Grid size
        grid_size = int(n**0.5) + 1
        positions = []

        for i in range(n):
            row = i // grid_size
            col = i % grid_size
            # Base position with 0.8m spacing (middle of 0.6-1.0)
            base_x = col * 0.8
            base_y = row * 0.8
            # Add random offset (±0.1m to keep in 0.6-1.0 range)
            offset = (torch.rand(2, device=self.device) - 0.5) * 0.2
            positions.append(
                torch.tensor([base_x, base_y], device=self.device) + offset
            )

        return torch.stack(positions)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Execute one simulation step.

        Args:
            actions: (N, 2) acceleration inputs for each agent

        Returns:
            observations: (N, 10) feature vectors
            adjacency: (N, N) communication adjacency matrix
            done: Whether episode is complete
        """
        # Saturate actions
        actions = torch.clamp(actions, -self.max_accel, self.max_accel)

        # Double integrator dynamics
        # v(t+1) = v(t) + u(t) * dt
        # r(t+1) = r(t) + v(t) * dt
        self.velocities = self.velocities + actions * self.dt
        self.positions = self.positions + self.velocities * self.dt

        return self.get_observations(), self.compute_adjacency(), False

    def get_observations(self) -> torch.Tensor:
        """Compute observation features for each agent.

        Returns:
            observations: (N, 10) where each row contains:
                - velocity (2D)
                - average neighbor velocity (2D)
                - error to leader or target (2D)
                - one-hot encoding for leader/follower (2D)
                - relative position to neighbors sum (2D)
        """
        N = self.num_agents
        obs = torch.zeros(N, 10, device=self.device)

        for i in range(N):
            # Own velocity
            obs[i, 0:2] = self.velocities[i]

            # Compute neighbors within collision range (sensing neighbors)
            neighbors = self._get_sensing_neighbors(i)

            if len(neighbors) > 0:
                # Average neighbor velocity
                neighbor_vels = self.velocities[neighbors]
                obs[i, 2:4] = neighbor_vels.mean(dim=0)

                # Relative position to neighbors (sum)
                neighbor_pos = self.positions[neighbors]
                rel_pos = neighbor_pos - self.positions[i]
                obs[i, 6:8] = rel_pos.sum(dim=0)
            else:
                obs[i, 2:4] = 0.0
                obs[i, 6:8] = 0.0

            # Leader/follower specific features
            if i == self.leader_idx:
                # Leader: error to target
                obs[i, 4:6] = self.target - self.positions[i]
                # One-hot: [1, 0] for leader
                obs[i, 8:10] = torch.tensor([1.0, 0.0], device=self.device)
            else:
                # Follower: error to leader
                obs[i, 4:6] = self.positions[self.leader_idx] - self.positions[i]
                # One-hot: [0, 1] for follower
                obs[i, 8:10] = torch.tensor([0.0, 1.0], device=self.device)

        return obs

    def _get_sensing_neighbors(self, agent_idx: int) -> list[int]:
        """Get indices of agents within collision range."""
        neighbors = []
        for j in range(self.num_agents):
            if j != agent_idx:
                dist = torch.norm(self.positions[agent_idx] - self.positions[j])
                if dist < self.collision_range:
                    neighbors.append(j)
        return neighbors

    def compute_adjacency(self) -> torch.Tensor:
        """Compute communication adjacency matrix.

        Returns:
            adjacency: (N, N) binary matrix where A[i,j]=1 if agents
                i and j are within communication range R
        """
        N = self.num_agents
        adjacency = torch.zeros(N, N, device=self.device)

        for i in range(N):
            for j in range(N):
                if i != j:
                    dist = torch.norm(self.positions[i] - self.positions[j])
                    if dist < self.comm_range:
                        adjacency[i, j] = 1.0

        return adjacency

    def compute_expert_action(self) -> torch.Tensor:
        """Compute expert controller actions using Olfati-Saber potential.

        Returns:
            actions: (N, 2) optimal acceleration for each agent
        """
        N = self.num_agents
        actions = torch.zeros(N, 2, device=self.device)

        # Compute average velocity
        avg_vel = self.velocities.mean(dim=0)

        for i in range(N):
            if i == self.leader_idx:
                # Leader controller
                actions[i] = self._leader_controller(i, avg_vel)
            else:
                # Follower controller
                actions[i] = self._follower_controller(i, avg_vel)

        # Saturate
        actions = torch.clamp(actions, -self.max_accel, self.max_accel)

        return actions

    def _follower_controller(self, idx: int, avg_vel: torch.Tensor) -> torch.Tensor:
        """Follower expert controller (Eq. 22a in paper).

        u_f = (1/N) * sum(v_i) - v_i - sum(grad_CA)
        """
        # Velocity alignment term
        u = avg_vel - self.velocities[idx]

        # Collision avoidance term
        for j in range(self.num_agents):
            if j != idx:
                u = u - self._collision_avoidance_gradient(idx, j)

        return u

    def _leader_controller(self, idx: int, avg_vel: torch.Tensor) -> torch.Tensor:
        """Leader expert controller (Eq. 22b in paper).

        u_l = W_p * (d - r_l) + (1/N) * sum(v_i) - v_l - sum(grad_CA)
        """
        # Target attraction term
        u = self.w_p * (self.target - self.positions[idx])

        # Velocity alignment term
        u = u + avg_vel - self.velocities[idx]

        # Collision avoidance term
        for j in range(self.num_agents):
            if j != idx:
                u = u - self._collision_avoidance_gradient(idx, j)

        return u

    def _collision_avoidance_gradient(self, i: int, j: int) -> torch.Tensor:
        """Compute gradient of collision avoidance potential.

        Uses log-barrier style potential from Olfati-Saber:
        If ||r_ij|| < R_CA: gradient points away from j
        """
        r_ij = self.positions[i] - self.positions[j]
        dist = torch.norm(r_ij)

        if dist < 1e-6:
            # Avoid division by zero
            return torch.zeros(2, device=self.device)

        if dist < self.collision_range:
            # Strong repulsion when too close
            # Gradient: (R_CA / dist - 1) * (r_ij / dist)
            magnitude = self.collision_range / dist - 1.0
            direction = r_ij / dist
            return -magnitude * direction
        else:
            # Weak attraction to maintain connectivity
            # Gradient: (dist / R_CA - 1) * (r_ij / dist)
            magnitude = (dist / self.collision_range - 1.0) * 0.1
            direction = r_ij / dist
            return magnitude * direction

    def compute_leader_error(self) -> float:
        """Compute normalized leader error (ef/es)."""
        leader_pos = self.positions[self.leader_idx]
        current_dist = torch.norm(leader_pos - self.target)
        return current_dist.item()

    def compute_flocking_error(self) -> float:
        """Compute flocking error (velocity alignment)."""
        avg_vel = self.velocities.mean(dim=0)
        errors = torch.norm(self.velocities - avg_vel, dim=1)
        return errors.mean().item()
