import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_filter import GraphFilter


class LGTCNLayer(nn.Module):
    """Continuous-time LGTC layer with hybrid Euler/RK-like solver (eq. 14)."""

    def __init__(self, in_dim: int, hidden_dim: int, K: int):
        super().__init__()
        self.K = K
        self.hidden_dim = hidden_dim

        # Filters
        self.A_hat = GraphFilter(hidden_dim, hidden_dim, K)  # on state x
        self.B_hat = GraphFilter(in_dim,    hidden_dim, K)  # on input u
        self.A_state = GraphFilter(hidden_dim, hidden_dim, K)
        self.B_state = GraphFilter(in_dim,    hidden_dim, K)

        # Biases / time-constant parameters
        self.bx = nn.Parameter(torch.zeros(1, hidden_dim))
        self.bu = nn.Parameter(torch.zeros(1, hidden_dim))
        self.b  = nn.Parameter(torch.ones(1, hidden_dim) * 0.1)  # positive to satisfy Thm.2

    def forward(self, x: torch.Tensor, u: torch.Tensor, S_powers, dt: float = 5e-2, n_steps: int = 1):
        """Propagate the hidden state through time."""
        # Expand biases if we are in batch mode
        if x.ndim == 3:
            bx = self.bx.expand(x.size(0), -1, -1)
            bu = self.bu.expand(u.size(0), -1, -1)
            b = self.b.expand(x.size(0), -1, -1)
        else:
            bx, bu, b = self.bx, self.bu, self.b

        for _ in range(n_steps):
            # gating term (paper eq. 8)
            f = F.relu(self.A_hat(x, S_powers) + bx) + \
                F.relu(self.B_hat(u, S_powers) + bu)

            # input modulation
            sigma_u = torch.tanh(self.B_state(u, S_powers))

            # state derivative
            dx = - (b + f) * x                    # -(b+f) ∘ x
            dx = dx - self.A_state(x, S_powers)        # − Σ S^k x A_k
            dx = dx + f * sigma_u                     # + f ∘ tanh(B S(u))

            # Euler step (hybrid solver)
            x = x + dt * dx
            # keep within [-1,1] per Lemma 1
            x = torch.clamp(x, -1.0, 1.0)
        return x
