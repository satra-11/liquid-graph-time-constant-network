import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_filter import GraphFilter


class LGTCNLayer(nn.Module):
    """Continuous-time LGTC layer with hybrid Euler/RK-like solver (eq. 8)."""

    def __init__(self, in_dim: int, hidden_dim: int, K: int):
        super().__init__()
        self.K = K
        self.hidden_dim = hidden_dim

        self.A_hat = GraphFilter(hidden_dim, hidden_dim, K)
        self.B_hat = GraphFilter(in_dim, hidden_dim, K)
        self.A_state = GraphFilter(hidden_dim, hidden_dim, K)
        self.B_state = GraphFilter(in_dim, hidden_dim, K)

        # Biases / time-constant parameters
        self.bx = nn.Parameter(torch.zeros(1, hidden_dim))
        self.bu = nn.Parameter(torch.zeros(1, hidden_dim))
        self.b = nn.Parameter(
            torch.ones(1, hidden_dim) * 0.1
        )  # positive to satisfy Thm.2

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        S_powers,
        dt: float = 5e-2,
        n_steps: int = 1,
    ):
        """Propagate the hidden state through time."""
        # Expand biases if we are in batch mode
        if x.ndim == 3:
            bx = self.bx.expand(x.size(0), -1, -1)
            bu = self.bu.expand(u.size(0), -1, -1)
            b = self.b.expand(x.size(0), -1, -1)
        else:
            bx, bu, b = self.bx, self.bu, self.b

        for _ in range(n_steps):
            f = F.relu(self.A_hat(x, S_powers) + bx) + F.relu(
                self.B_hat(u, S_powers) + bu
            )

            sigma_c = torch.tanh(self.B_state(u, S_powers))

            dx = -(b + f) * x
            dx = dx - self.A_state(x, S_powers)
            dx = dx + f * sigma_c

            x = x + dt * dx
            x = torch.clamp(x, -1.0, 1.0)
        return x
