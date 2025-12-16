import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.core.layers.ngode_layer import NeuralGraphODELayer
from src.utils import compute_s_powers, compute_laplacian, compute_random_walk_matrix


class NeuralGraphODEController(nn.Module):
    """映像データからNeural Graph ODEを使って制御信号を生成するコントローラー.

    CfGCNControllerと同様のインターフェースで、CfGCNLayerの代わりにNeuralGraphODELayerを使用。
    """

    def __init__(
        self,
        frame_height: int = 64,
        frame_width: int = 64,
        hidden_dim: int = 64,
        K: int = 2,
        output_dim: int = 6,
        matrix_type: str = "adjacency",
        num_layers: int = 1,
        solver: str = "dopri5",
    ):
        super().__init__()
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.hidden_dim = hidden_dim
        self.K = K
        self.output_dim = output_dim
        self.matrix_type = matrix_type
        self.num_layers = num_layers
        self.node_encoder = nn.Linear(128, hidden_dim)

        # CNN feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        # Neural Graph ODE temporal processors (stacked layers)
        self.temporal_processor = nn.ModuleList(
            [
                NeuralGraphODELayer(hidden_dim, hidden_dim, K, solver=solver)
                for _ in range(num_layers)
            ]
        )

        # Control output decoder
        self.control_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim),
        )

        # Learnable output scaling (like CfGCNController)
        self.output_scale = nn.Parameter(torch.ones(self.output_dim) * 10.0)
        self.output_bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(
        self,
        frames: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Neural Graph ODE controller.

        Args:
            frames: Input frames (B, T, C, H, W) or pre-extracted features (B, T, 128, 8, 8)
            adjacency: Optional adjacency matrix (B, T, N, N)
            hidden_state: Optional initial hidden state (B, num_layers, N, hidden_dim)

        Returns:
            controls: Predicted controls (B, T, output_dim)
            final_hidden: Final hidden state (B, num_layers, N, hidden_dim)
        """
        B, T, C, H, W = frames.shape

        # Check if input is pre-extracted features
        if C == 128 and H == 8 and W == 8:
            features = frames
        else:
            frames_flat = frames.view(-1, C, H, W)
            features = self.feature_extractor(frames_flat)
            features = features.view(B, T, 128, 8, 8)

        # Reshape features for node encoding
        features = features.permute(0, 1, 3, 4, 2)
        node_feats = features.reshape(B, T, 64, 128)
        node_feats = self.node_encoder(node_feats)

        # Default adjacency matrix (fully connected)
        if adjacency is None:
            N = node_feats.size(2)
            adjacency = torch.ones(B, T, N, N, device=frames.device)

        controls = []

        N = node_feats.size(2)
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = (
                torch.randn(
                    B, self.num_layers, N, self.hidden_dim, device=frames.device
                )
                * 0.01
            )

        current_hiddens = hidden_state.unbind(dim=1)

        # Process each time step
        for t in range(T):
            xt = node_feats[:, t, :, :]  # (B, N, hidden_dim)

            A_t = adjacency[:, t, :, :]

            # Compute graph shift operator
            if self.matrix_type == "adjacency":
                S_t = A_t
            elif self.matrix_type == "laplacian":
                S_t = compute_laplacian(A_t)
            elif self.matrix_type == "random_walk":
                S_t = compute_random_walk_matrix(A_t)
            else:
                raise ValueError(f"Unknown matrix type: {self.matrix_type}")

            S_powers = compute_s_powers(S_t, self.K)

            h_prev_layers = current_hiddens
            h_next_layers = []

            # First layer uses input
            h_in = h_prev_layers[0]
            u_in = xt
            h_out = self.temporal_processor[0](h_in, u_in, S_powers)
            h_next_layers.append(h_out)

            # Subsequent layers use previous layer output
            for i in range(1, self.num_layers):
                h_in = h_prev_layers[i]
                u_in = h_next_layers[i - 1]
                h_out = self.temporal_processor[i](h_in, u_in, S_powers)
                h_next_layers.append(h_out)

            # Decode control from final layer output
            final_layer_output = h_next_layers[-1]
            control = self.control_decoder(final_layer_output.mean(dim=1))
            control = control * self.output_scale + self.output_bias
            controls.append(control)
            current_hiddens = tuple(h_next_layers)

        controls = torch.stack(controls, dim=1)
        final_hidden = torch.stack(current_hiddens, dim=1)

        return controls, final_hidden
