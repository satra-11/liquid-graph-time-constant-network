"""Flocking models based on LTCN and LGTCN."""

import torch
import torch.nn as nn
from typing import Optional

from src.core.layers import CfGCNLayer, LTCNLayer
from src.utils import compute_s_powers, compute_laplacian


class FlockingLGTCN(nn.Module):
    """LGTCN-based flocking controller.

    Architecture (from paper):
    - FC(128) -> FC(128) -> CfGCNLayer(F=50, K=2) -> FC(128) -> FC(128) -> out(2)

    Uses normalized Laplacian as support matrix.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 50,
        output_dim: int = 2,
        K: int = 2,
        fc_dim: int = 128,
        num_layers: int = 1,
    ):
        """Initialize FlockingLGTCN.

        Args:
            input_dim: Input feature dimension (10 for flocking)
            hidden_dim: Hidden state dimension F
            output_dim: Output dimension (2 for 2D acceleration)
            K: Filter length for graph convolution
            fc_dim: Fully connected layer dimension
            num_layers: Number of CfGCN layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.K = K
        self.num_layers = num_layers

        # Input encoder: FC(128) -> ReLU -> FC(128) -> ReLU
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, hidden_dim),
            nn.ReLU(),
        )

        # CfGCN layers
        self.cfgcn_layers = nn.ModuleList(
            [CfGCNLayer(hidden_dim, hidden_dim, K) for _ in range(num_layers)]
        )

        # Output decoder: FC(128) -> ReLU -> FC(128) -> ReLU -> FC(2)
        self.output_decoder = nn.Sequential(
            nn.Linear(hidden_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, output_dim),
        )

    def forward(
        self,
        observations: torch.Tensor,
        adjacency: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            observations: (B, T, N, input_dim) observation features
            adjacency: (B, T, N, N) adjacency matrices
            hidden_state: Optional (B, num_layers, N, hidden_dim) hidden states

        Returns:
            actions: (B, T, N, output_dim) predicted actions
            final_hidden: (B, num_layers, N, hidden_dim) final hidden states
        """
        B, T, N, _ = observations.shape

        # Initialize hidden state
        if hidden_state is None:
            hidden_state = (
                torch.randn(
                    B, self.num_layers, N, self.hidden_dim, device=observations.device
                )
                * 0.01
            )

        current_hiddens = hidden_state.unbind(dim=1)
        actions = []

        for t in range(T):
            # Get observation and adjacency at time t
            obs_t = observations[:, t]  # (B, N, input_dim)
            adj_t = adjacency[:, t]  # (B, N, N)

            # Encode input
            x_t = self.input_encoder(obs_t)  # (B, N, hidden_dim)

            # Compute normalized Laplacian
            L_t = compute_laplacian(adj_t)  # (B, N, N)

            # Compute S powers
            S_powers = compute_s_powers(L_t, self.K)

            # Process through CfGCN layers
            h_prev_layers = current_hiddens
            h_next_layers = []

            # First layer
            h_in = h_prev_layers[0]
            u_in = x_t
            h_out = self.cfgcn_layers[0](h_in, u_in, S_powers)
            h_next_layers.append(h_out)

            # Remaining layers
            for i in range(1, self.num_layers):
                h_in = h_prev_layers[i]
                u_in = h_next_layers[i - 1]
                h_out = self.cfgcn_layers[i](h_in, u_in, S_powers)
                h_next_layers.append(h_out)

            current_hiddens = tuple(h_next_layers)

            # Decode output
            final_h = h_next_layers[-1]  # (B, N, hidden_dim)
            action_t = self.output_decoder(final_h)  # (B, N, output_dim)
            actions.append(action_t)

        actions = torch.stack(actions, dim=1)  # (B, T, N, output_dim)
        final_hidden = torch.stack(current_hiddens, dim=1)

        return actions, final_hidden


class FlockingLTCN(nn.Module):
    """LTCN-based flocking controller (baseline without graph structure).

    Architecture:
    - FC(128) -> FC(128) -> LTCNLayer(F=50) -> FC(128) -> FC(128) -> out(2)

    Processes each agent independently (no message passing).
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 48,  # Must be divisible by num_blocks
        output_dim: int = 2,
        fc_dim: int = 128,
        num_blocks: int = 4,
    ):
        """Initialize FlockingLTCN.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension (must be divisible by num_blocks)
            output_dim: Output dimension
            fc_dim: Fully connected layer dimension
            num_blocks: Number of blocks in LTCNLayer
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks

        # Ensure hidden_dim is divisible by num_blocks
        k_per_block = hidden_dim // num_blocks
        # LTCNLayer's N = num_blocks * k
        self.ltcn_hidden_dim = num_blocks * k_per_block
        self.hidden_dim = self.ltcn_hidden_dim

        # Input encoder: outputs ltcn_hidden_dim
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, self.ltcn_hidden_dim),
            nn.ReLU(),
        )

        # LTCN layer: in_dim matches encoder output
        self.ltcn_layer = LTCNLayer(self.ltcn_hidden_dim, k_per_block, num_blocks)

        # Output decoder
        self.output_decoder = nn.Sequential(
            nn.Linear(self.ltcn_hidden_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, output_dim),
        )

    def forward(
        self,
        observations: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            observations: (B, T, N, input_dim) observation features
            adjacency: Ignored (for API compatibility)
            hidden_state: Optional (B, N, hidden_dim) hidden states

        Returns:
            actions: (B, T, N, output_dim) predicted actions
            final_hidden: (B, N, hidden_dim) final hidden states
        """
        B, T, N, _ = observations.shape

        # Initialize hidden state
        if hidden_state is None:
            hidden_state = torch.zeros(
                B, N, self.hidden_dim, device=observations.device
            )

        # Process each agent with shared LTCN
        # Flatten batch and agents: (B, N, ...) -> (B*N, ...)
        current_hidden = hidden_state.reshape(B * N, self.hidden_dim)

        actions = []

        for t in range(T):
            obs_t = observations[:, t]  # (B, N, input_dim)
            obs_flat = obs_t.reshape(B * N, self.input_dim)

            # Encode input
            x_t = self.input_encoder(obs_flat)  # (B*N, ltcn_hidden_dim)

            # LTCN step
            next_hidden = self.ltcn_layer(current_hidden, x_t, dt=0.05, n_steps=1)

            # Decode output
            action_t = self.output_decoder(next_hidden)  # (B*N, output_dim)
            action_t = action_t.reshape(B, N, self.output_dim)
            actions.append(action_t)

            current_hidden = next_hidden

        actions = torch.stack(actions, dim=1)  # (B, T, N, output_dim)
        final_hidden = current_hidden.reshape(B, N, self.hidden_dim)

        return actions, final_hidden
