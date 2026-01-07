import torch
import torch.nn as nn


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

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Compute dy/dt = f(y)."""
        return self.net(y)


class NeuralODELayer(nn.Module):
    """Neural ODE layer using Euler integration.

    Integrates the ODE dynamics over a time interval to produce the next state.
    Uses the same Euler method as LTCNLayer for fair comparison.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

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
        """Integrate the ODE using Euler method.

        Args:
            y: Current hidden state (..., hidden_dim)
            u_t: Input at current time (..., in_dim), can be None
            dt: Time step size
            n_steps: Number of Euler integration steps

        Returns:
            Next hidden state (..., hidden_dim)
        """
        # Incorporate input into state via simple addition (no gating)
        if u_t is not None:
            u_proj = self.input_proj(u_t)
            y = y + u_proj

        # Euler integration: y_{n+1} = y_n + dt * f(y_n)
        for _ in range(n_steps):
            dydt = self.ode_func(y)
            y = y + dt * dydt

        return y
