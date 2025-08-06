import torch
import torch.nn as nn


class GraphFilter(nn.Module):
    """Implements H_S(x) = sum_{k=0..K} S^k x W_k."""

    def __init__(self, in_dim: int, out_dim: int, K: int):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.Tensor(K + 1, in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, S_powers):
        # If x is 3D (B, N, D), use batch matrix multiplication
        if x.ndim == 3:
            out = 0.0
            for k in range(self.K + 1):
                # S_powers[k] is (B, N, N), x is (B, N, in_dim)
                xk = torch.bmm(S_powers[k], x)
                # self.weight[k] is (in_dim, out_dim), needs to be (B, in_dim, out_dim)
                out = out + torch.bmm(xk, self.weight[k].expand(x.size(0), -1, -1))
            return out
        
        # Original 2D logic
        out = 0.0
        for k in range(self.K + 1):
            xk = S_powers[k] @ x  # (N, in_dim)
            out = out + xk @ self.weight[k]
        return out
