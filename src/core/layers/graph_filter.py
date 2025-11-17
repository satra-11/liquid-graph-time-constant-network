import torch
import torch.nn as nn


class GraphFilter(nn.Module):
    """Implements H_S(x) = sum_{k=0..K} S^k x W_k(eq. 2)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        K: int,
        include_k0: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.K = K
        self.include_k0 = include_k0
        self.weight = nn.Parameter(torch.empty(K + 1, in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, S_powers):
        start_k = 0 if self.include_k0 else 1

        # Normalize S_powers to list/tuple
        if not isinstance(S_powers, (list, tuple)):
            raise TypeError(
                "S_powers must be a list/tuple of length K+1 with powers [I, S, ..., S^K]."
            )

        # Batch
        if x.ndim == 3:
            # x: (B, N, Din)
            B, N, Din = x.shape
            out = torch.zeros(
                B, N, self.weight.size(-1), device=x.device, dtype=x.dtype
            )

            for k in range(start_k, self.K + 1):
                Sk = S_powers[k]
                # Support (N,N) or (B,N,N)
                if Sk.ndim == 2:
                    # Broadcast (N,N) -> (B,N,N) via bmm by repeating x
                    xk = torch.bmm(Sk.expand(B, -1, -1), x)  # (B,N,Din)
                elif Sk.ndim == 3:
                    xk = torch.bmm(Sk, x)  # (B,N,Din)
                else:
                    raise ValueError(f"S_powers[{k}] must be 2D or 3D, got {Sk.ndim}D.")

                Wk = self.weight[k]  # (Din, Dout)
                # out += xk @ Wk  using einsum (no expand)
                out = out + torch.einsum("bni,id->bnd", xk, Wk)

            return out

        elif x.ndim == 2:
            # x: (N, Din)
            N, Din = x.shape
            out = torch.zeros(N, self.weight.size(-1), device=x.device, dtype=x.dtype)

            for k in range(start_k, self.K + 1):
                Sk = S_powers[k]
                if Sk.ndim != 2:
                    raise ValueError(
                        f"For 2D x, S_powers[{k}] must be 2D (N,N); got {Sk.ndim}D."
                    )
                xk = Sk @ x  # (N, Din)
                Wk = self.weight[k]  # (Din, Dout)
                out = out + xk @ Wk  # (N, Dout)

            return out

        else:
            raise ValueError(f"x must be 2D or 3D, got shape {x.shape}.")
