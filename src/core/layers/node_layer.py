import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    """ODE dynamics function: dy/dt = f(t, y).

    Uses an MLP to model the dynamics.
    """

    def __init__(self, hidden_dim: int, num_hidden_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        layers = []
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute dy/dt = f(t, y)."""
        return self.net(y)


class NeuralODELayer(nn.Module):
    """Neural ODE layer using torchdiffeq.odeint.

    Integrates the ODE dynamics over a time interval to produce the next state.
    Compatible with LTCNLayer interface for easy swapping.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_hidden_layers: int = 2,
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # ODE function
        self.ode_func = ODEFunc(hidden_dim, num_hidden_layers)

    def forward(
        self,
        y: torch.Tensor,
        u_t: torch.Tensor | None,
        dt: float = 0.1,
        n_steps: int = 1,
    ) -> torch.Tensor:
        """Integrate the ODE from t=0 to t=dt*n_steps.

        Args:
            y: Current hidden state (..., hidden_dim)
            u_t: Input at current time (..., in_dim), can be None
            dt: Time step size
            n_steps: Number of integration steps (used to compute final time)

        Returns:
            Next hidden state (..., hidden_dim)
        """
        # Incorporate input into state via simple addition (no gating)
        if u_t is not None:
            u_proj = self.input_proj(u_t)
            y = y + u_proj

        # Time span for integration
        t_span = torch.tensor([0.0, dt * n_steps], device=y.device, dtype=y.dtype)

        # Solve ODE
        # odeint returns shape (time_points, *batch_dims, hidden_dim)
        solution = odeint(
            self.ode_func,
            y,
            t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )

        # Return final state (last time point)
        return solution[-1]
