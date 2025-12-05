import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_filter import GraphFilter


class CfGCNLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, K: int, eps: float = 1e-3):
        super().__init__()
        self.K = K
        self.hidden_dim = hidden_dim
        self.eps = eps

        self.A_hat = GraphFilter(hidden_dim, hidden_dim, K)
        self.B_hat = GraphFilter(in_dim, hidden_dim, K)
        self.A_state = GraphFilter(hidden_dim, hidden_dim, K, include_k0=False)
        self.B_state = GraphFilter(in_dim, hidden_dim, K)

        self.bx = nn.Parameter(torch.zeros(1, hidden_dim))
        self.bu = nn.Parameter(torch.zeros(1, hidden_dim))
        self.b = nn.Parameter(torch.ones(1, hidden_dim) * 0.1)

    def forward(self, x: torch.Tensor, u: torch.Tensor, S_powers, t: float = 1.0):
        A_hat_out = self.A_hat(x, S_powers)
        f_x = F.relu(A_hat_out + self.bx)
        f_sigma = F.relu(self.B_hat(u, S_powers) + self.bu) + f_x

        fi = self.A_state(x, S_powers)

        # 修正: シグモイド入力から math.pi を削除し、勾配消失を防止
        decay_input = self.b + f_x + fi
        coeff = torch.sigmoid(-t * decay_input)

        sigma_u = torch.tanh(self.B_state(u, S_powers))

        # 修正: ゲーティング機構を改良し、勾配の流れを改善
        gate = torch.sigmoid(f_sigma)
        x_new = x * coeff + (1 - coeff) * sigma_u * gate

        # 出力範囲を[-1, 1]に制限
        x_new = torch.clamp(x_new, -1.0, 1.0)

        return x_new
