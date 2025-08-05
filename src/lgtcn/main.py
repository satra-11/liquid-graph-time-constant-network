import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lgtcn.utils import compute_support_powers
# ------------------------------------------------------------
# Graph Filter Bank (eq. 2 in the paper)
# ------------------------------------------------------------

class GraphFilter(nn.Module):
    """Implements H_S(x) = sum_{k=0..K} S^k x W_k."""

    def __init__(self, in_dim: int, out_dim: int, K: int):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.Tensor(K + 1, in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, S_powers):
        out = 0.0
        for k in range(self.K + 1):
            xk = S_powers[k] @ x  # (N, in_dim)
            out = out + xk @ self.weight[k]
        return out

# ------------------------------------------------------------
# Liquid-Graph Time-Constant ODE Layer (eqs. 8 & 14)
# ------------------------------------------------------------

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
        """Propagate the hidden state through time.

        Args:
            x:         (N, hidden_dim) hidden state at t.
            u:         (N, in_dim)     exogenous input features at t (kept constant during the step).
            S_powers:  list of S^k   pre-computed support powers.
            dt:        integration step (s).
            n_steps:   how many sub-steps to integrate over dt*n_steps.
        Returns:
            x_new (N, hidden_dim)
        """
        for _ in range(n_steps):
            # gating term (paper eq. 8)
            f = F.relu(self.A_hat(x, S_powers) + self.bx) + \
                F.relu(self.B_hat(u, S_powers) + self.bu)

            # input modulation
            sigma_u = torch.tanh(self.B_state(u, S_powers))

            # state derivative
            dx = - (self.b + f) * x                    # -(b+f) ∘ x
            dx = dx - self.A_state(x, S_powers)        # − Σ S^k x A_k
            dx = dx + f * sigma_u                     # + f ∘ tanh(B S(u))

            # Euler step (hybrid solver)
            x = x + dt * dx
            # keep within [-1,1] per Lemma 1
            x = torch.clamp(x, -1.0, 1.0)
        return x

# ------------------------------------------------------------
# Closed-Form Graph Time-Constant (CfGC) Layer  (eq. 20)
# ------------------------------------------------------------

class CfGCNLayer(nn.Module):
    """Discrete closed-form approximation that avoids ODE solving."""

    def __init__(self, in_dim: int, hidden_dim: int, K: int, eps: float = 1e-3):
        super().__init__()
        self.K = K
        self.hidden_dim = hidden_dim
        self.eps = eps

        # Filters share structure with LGTCNLayer
        self.A_hat = GraphFilter(hidden_dim, hidden_dim, K)
        self.B_hat = GraphFilter(in_dim,    hidden_dim, K)
        self.A_state = GraphFilter(hidden_dim, hidden_dim, K)
        self.B_state = GraphFilter(in_dim,    hidden_dim, K)

        self.bx = nn.Parameter(torch.zeros(1, hidden_dim))
        self.bu = nn.Parameter(torch.zeros(1, hidden_dim))
        self.b  = nn.Parameter(torch.ones(1, hidden_dim) * 0.1)

    def forward(self, x: torch.Tensor, u: torch.Tensor, S_powers, t: float = 1.0):
        f_sigma = F.relu(self.B_hat(u, S_powers) + self.bu) + \
                  F.relu(self.A_hat(x, S_powers) + self.bx)

        f_x = F.relu(self.A_hat(x, S_powers) + self.bx)

        # crude fi term following paper eq. 20 (element-wise division is safe with eps)
        fi_num = self.A_state(x, S_powers)
        fi = - fi_num / (x + self.eps)

        coeff = torch.sigmoid(- (self.b + f_x + fi) * t + math.pi)
        sigma_u = torch.tanh(self.B_state(u, S_powers))
        x_new = (x * coeff - sigma_u) * torch.sigmoid(2 * f_sigma) + sigma_u
        return torch.clamp(x_new, -1.0, 1.0)

# ------------------------------------------------------------
# Controller / Policy Network used in the paper (Fig. 2 setup)
# ------------------------------------------------------------

class LGTCNController(nn.Module):
    """Full network: MLP encoder  -> (LGTC / CfGC layer) -> MLP decoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 50, K: int = 2,
                 output_dim: int = 2, use_closed_form: bool = False):
        super().__init__()
        self.use_closed_form = use_closed_form
        self.K = K

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU())

        if use_closed_form:
            self.gnn = CfGCNLayer(128, hidden_dim, K)
        else:
            self.gnn = LGTCNLayer(128, hidden_dim, K)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim))

    def forward(self, w: torch.Tensor, S: torch.Tensor, x_state: torch.Tensor | None = None):
        """Compute control action.

        Args:
            w       – (N, input_dim)  local features per agent.
            S       – (N, N) support matrix (same for every feature).
            x_state – (N, hidden_dim) optional persistent hidden state.
        Returns (u, x_next)
        """
        N = w.size(0)
        if x_state is None:
            x_state = w.new_zeros((N, self.gnn.hidden_dim))

        S_powers = compute_support_powers(S, self.K)
        w_emb = self.encoder(w)
        x_next = self.gnn(x_state, w_emb, S_powers)
        u = self.decoder(x_next)  # raw acceleration; clamp as desired outside
        return u, x_next

# ------------------------------------------------------------
# Minimal usage example
# ------------------------------------------------------------

if __name__ == "__main__":
    N, F_in = 5, 10
    features = torch.randn(N, F_in)
    support = torch.eye(N)  # identity graph for a quick smoke test

    ctrl = LGTCNController(input_dim=F_in, hidden_dim=32, K=2, output_dim=2)

    u, x_next = ctrl(features, support)
    print("control action shape:", u.shape)  # (N, 2)
