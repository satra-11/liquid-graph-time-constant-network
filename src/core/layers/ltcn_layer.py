import torch
import torch.nn as nn
import torch.nn.functional as F


class LTCNLayer(nn.Module):
    """
    Strict LTC layer faithful to the MATLAB ltc_def dynamics.

    N = num_blocks * k
    - Block 1 (index 0) uses external input u_t:  net_out_0 = act(u_t @ W_in + b_in)
    - Blocks j>=1 use previous block state:       net_out_j = act(y_{j-1} @ W_fwd[j-1] + b_fwd[j-1])
    - Recurrent per block:                        net_recurr_j = act(y_j @ W_rec[j] + b_rec[j])
    - Dynamics per neuron i in block j:
        dy_i/dt = -y_i * ( 1/tau_i + |net_out_i| + |net_recurr_i| )
                  + (E_l[j]   @ net_out_j)_i
                  + (E_l_r[j] @ net_recurr_j)_i
    Euler step: y <- y + dt * dy/dt
    """

    def __init__(
        self,
        in_dim: int,
        k: int,
        num_blocks: int,
        activation: str = "tanh",
        clamp_output: bool | float = False,  # False or float range like 1.0
    ):
        super().__init__()
        assert num_blocks >= 1 and k >= 1
        self.in_dim = in_dim
        self.k = k
        self.num_blocks = num_blocks
        self.N = num_blocks * k
        self.clamp_output = clamp_output

        # --- activation selector (sigmoid / relu / tanh / htanh)
        act = activation.lower()
        if act not in {"sigmoid", "relu", "tanh", "htanh"}:
            raise ValueError("activation must be one of: sigmoid, relu, tanh, htanh")
        self.activation = act

        # --- tau: ensure positivity via softplus
        self._tau_raw = nn.Parameter(torch.full((self.N,), 0.0))  # softplus(0)=~0.693
        self.tau_eps = 1e-6

        # --- Input transform for first block (j=0): u_t -> k
        self.W_in = nn.Linear(in_dim, k)

        # --- Forward transforms between blocks: for j=1..B-1, y_{j-1} -> k
        self.W_fwd = nn.ModuleList([nn.Linear(k, k) for _ in range(num_blocks - 1)])

        # --- Recurrent per block: y_j -> k
        self.W_rec = nn.ModuleList([nn.Linear(k, k) for _ in range(num_blocks)])

        # --- E matrices (per block): combine net_out_j and net_recurr_j into drive term
        #     Shapes: (B, k, k) so that (E[j] @ vec_k)
        self.E_l = nn.Parameter(torch.zeros(num_blocks, k, k))
        self.E_l_r = nn.Parameter(torch.zeros(num_blocks, k, k))

        self.reset_parameters()

    def reset_parameters(self):
        # He/Xavier-ish inits; tweak as needed
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.zeros_(self.W_in.bias)
        for lin in self.W_fwd:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        for lin in self.W_rec:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        # E matrices small random
        nn.init.xavier_uniform_(self.E_l)
        nn.init.xavier_uniform_(self.E_l_r)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        # "htanh"
        return F.hardtanh(x)

    def forward(
        self,
        y: torch.Tensor,  # (..., N)
        u_t: torch.Tensor | None,  # (..., in_dim)  (required for block 0 each step)
        dt: float = 5e-2,
        n_steps: int = 1,
    ) -> torch.Tensor:
        """
        y:   current hidden state (batch..., N) with N = num_blocks * k
        u_t: input at current time (already interpolated). If None and num_blocks>0, block0 net_out=0.
        """
        assert y.shape[-1] == self.N
        if u_t is not None:
            assert u_t.shape[:-1] == y.shape[:-1] and u_t.shape[-1] == self.in_dim

        # reshape to (..., B, k)
        *batch, _ = y.shape
        B, k = self.num_blocks, self.k
        y = y.view(*batch, B, k)

        for _ in range(n_steps):
            # tau positive
            tau = F.softplus(self._tau_raw) + self.tau_eps  # shape (N,)
            tau = tau.view(B, k)
            if len(batch) > 0:
                tau = tau.view(*([1] * len(batch)), B, k).expand(*batch, B, k)

            # ---- net_out per block
            net_out_list = []
            # block 0 from input
            if u_t is None:
                net0 = torch.zeros(*batch, k, device=y.device, dtype=y.dtype)
            else:
                net0 = self._act(self.W_in(u_t))
            net_out_list.append(net0)

            # blocks 1..B-1 from previous block state
            for j in range(1, B):
                prev = y[..., j - 1, :]  # (..., k)
                net_j = self._act(self.W_fwd[j - 1](prev))
                net_out_list.append(net_j)
            net_out = torch.stack(net_out_list, dim=-2)  # (..., B, k)

            # ---- net_recurr per block (from same block state)
            net_recurr_list = []
            for j in range(B):
                cur = y[..., j, :]  # (..., k)
                net_r = self._act(self.W_rec[j](cur))
                net_recurr_list.append(net_r)
            net_recurr = torch.stack(net_recurr_list, dim=-2)  # (..., B, k)

            # ---- decay term: (1/tau) + |net_out| + |net_recurr|
            decay = (1.0 / tau) + net_out.abs() + net_recurr.abs()  # (..., B, k)

            # ---- E-lin combinations: (E_l[j] @ net_out_j) and (E_l_r[j] @ net_recurr_j)
            # reshape for bmm: (..., B, k, k) @ (..., B, k, 1) -> (..., B, k, 1)
            E = self.E_l.view(*([1] * len(batch)), B, k, k).expand(*batch, B, k, k)
            Er = self.E_l_r.view(*([1] * len(batch)), B, k, k).expand(*batch, B, k, k)

            out_term = torch.matmul(E, net_out.unsqueeze(-1)).squeeze(-1)  # (..., B, k)
            recurr_term = torch.matmul(Er, net_recurr.unsqueeze(-1)).squeeze(
                -1
            )  # (..., B, k)

            dydt = -y * decay + out_term + recurr_term  # (..., B, k)
            y = y + dt * dydt

            if self.clamp_output:
                lim = (
                    self.clamp_output
                    if isinstance(self.clamp_output, (int, float))
                    else 1.0
                )
                y = y.clamp(-lim, lim)

        return y.view(*batch, B * k)
