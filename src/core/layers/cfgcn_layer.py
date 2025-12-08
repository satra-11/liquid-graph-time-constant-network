import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_filter import GraphFilter


class CfGCNLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        K: int,
        eps: float = 0.1,
        residual: bool = True,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.K = K
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.residual = residual
        self.residual_scale = residual_scale

        self.A_hat = GraphFilter(hidden_dim, hidden_dim, K)
        self.B_hat = GraphFilter(in_dim, hidden_dim, K)
        self.A_state = GraphFilter(hidden_dim, hidden_dim, K, include_k0=False)
        self.B_state = GraphFilter(in_dim, hidden_dim, K)

        self.bx = nn.Parameter(torch.zeros(1, hidden_dim))
        self.bu = nn.Parameter(torch.zeros(1, hidden_dim))
        self.b = nn.Parameter(torch.ones(1, hidden_dim) * 0.1)

        # 入力次元と隠れ次元が異なる場合の射影層 (residual使用時のみ)
        if residual and in_dim != hidden_dim:
            self.input_proj = nn.Linear(in_dim, hidden_dim)
        else:
            self.input_proj = None

    def forward(self, x: torch.Tensor, u: torch.Tensor, S_powers, t: float = 1.0):
        A_hat_out = self.A_hat(x, S_powers)
        f_x = F.relu(A_hat_out + self.bx)
        f_sigma = F.relu(self.B_hat(u, S_powers) + self.bu) + f_x

        fi = self.A_state(x, S_powers)

        coeff = torch.sigmoid(-(self.b + f_x + fi) * t + math.pi)
        sigma_u = torch.tanh(self.B_state(u, S_powers))
        x_new = (x * coeff - sigma_u) * torch.sigmoid(2 * f_sigma) + sigma_u

        # Residual connection: 入力からのスキップ接続を追加
        if self.residual:
            if self.input_proj is not None:
                u_proj = self.input_proj(u)
            else:
                u_proj = u
            x_new = x_new + self.residual_scale * u_proj

        return x_new
