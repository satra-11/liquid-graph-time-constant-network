import torch
import torch.nn as nn

from lgtcn.layers.lgtcn_layer import LGTCNLayer
from lgtcn.layers.cfgcn_layer import CfGCNLayer
from lgtcn.utils.graph import compute_support_powers


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
        """Compute control action."""
        if w.ndim == 3:
            B, N, _ = w.shape
            if x_state is None:
                x_state = w.new_zeros((B, N, self.gnn.hidden_dim))
        else:
            N = w.size(0)
            if x_state is None:
                x_state = w.new_zeros((N, self.gnn.hidden_dim))

        S_powers = compute_support_powers(S, self.K)
        w_emb = self.encoder(w)
        x_next = self.gnn(x_state, w_emb, S_powers)
        u = self.decoder(x_next)
        return u, x_next
