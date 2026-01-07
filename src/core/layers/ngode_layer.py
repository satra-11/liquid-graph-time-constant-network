import torch
import torch.nn as nn
from torchdiffeq import odeint

from .graph_filter import GraphFilter


class GraphODEFunc(nn.Module):
    """Graph ODE dynamics function: dy/dt = f(t, y, S_powers).

    Uses GraphFilter for graph-aware dynamics.
    """

    def __init__(self, hidden_dim: int, K: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.K = K

        # Graph-aware transformation
        self.graph_filter = GraphFilter(hidden_dim, hidden_dim, K)

        # Additional MLP for nonlinear dynamics
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable decay parameter
        self.decay = nn.Parameter(torch.ones(1, hidden_dim) * 0.1)

        # Store S_powers for use in forward (set by parent layer)
        self.S_powers = None

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute dy/dt = -decay * y + graph_filter(y) + mlp(y)."""
        if self.S_powers is None:
            raise RuntimeError("S_powers must be set before calling forward")

        # Graph convolution
        graph_out = self.graph_filter(y, self.S_powers)

        # Nonlinear transformation
        mlp_out = self.mlp(y)

        # Dynamics: decay + graph + nonlinear
        dy = -self.decay * y + graph_out + mlp_out

        return dy


class NeuralGraphODELayer(nn.Module):
    """Neural Graph ODE layer combining graph convolutions with ODE solver.

    Compatible with LGTCNLayer/CfGCNLayer interface for easy swapping.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        K: int,
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.K = K
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

        # Input transformation using graph filter
        self.input_filter = GraphFilter(in_dim, hidden_dim, K)

        # ODE function
        self.ode_func = GraphODEFunc(hidden_dim, K)

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        S_powers,
        dt: float = 0.05,
        n_steps: int = 1,
    ) -> torch.Tensor:
        """Integrate the Graph ODE from t=0 to t=dt*n_steps.

        Args:
            x: Current hidden state (N, hidden_dim) or (B, N, hidden_dim)
            u: Input at current time (N, in_dim) or (B, N, in_dim)
            S_powers: List of graph shift operator powers [I, S, S^2, ..., S^K]
            dt: Time step size
            n_steps: Number of integration steps

        Returns:
            Next hidden state with same shape as x
        """
        # Transform input using graph filter and add to state (no gating)
        u_proj = self.input_filter(u, S_powers)
        x = x + u_proj

        # Set S_powers for ODE function
        self.ode_func.S_powers = S_powers

        # Time span for integration
        t_span = torch.tensor([0.0, dt * n_steps], device=x.device, dtype=x.dtype)

        # Solve ODE
        solution = odeint(
            self.ode_func,
            x,
            t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )

        # Return final state, clamped to [-1, 1] for stability
        output = solution[-1]
        output = torch.clamp(output, -1.0, 1.0)

        return output
