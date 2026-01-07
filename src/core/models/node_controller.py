import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.core.layers.node_layer import NeuralODELayer


class NeuralODEController(nn.Module):
    """映像データからNeural ODEを使って制御信号を生成するコントローラー.

    LTCNControllerと同様のインターフェースで、LTCNLayerの代わりにNeuralODELayerを使用。
    パラメータ数はLTCNと同程度になるよう調整済み。
    """

    def __init__(
        self,
        frame_height: int = 64,
        frame_width: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 6,
        num_hidden_layers: int = 1,  # Reduced to match LTCN param count
        mlp_hidden_dim: int = 10,  # Internal MLP width for fair comparison
    ):
        super().__init__()
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
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

        # Neural ODE temporal processor
        self.temporal_processor = NeuralODELayer(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

        # Control output decoder
        self.control_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim),
        )

    def forward(
        self, frames: torch.Tensor, hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Neural ODE controller.

        Args:
            frames: Input frames (B, T, C, H, W) or pre-extracted features (B, T, 128, 8, 8)
            hidden_state: Optional initial hidden state (B, hidden_dim)

        Returns:
            controls: Predicted controls (B, T, output_dim)
            final_hidden: Final hidden state (B, hidden_dim)
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

        controls = []

        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = torch.zeros(B, self.hidden_dim, device=frames.device)

        current_hidden = hidden_state

        # Process each time step
        for t in range(T):
            # Average over spatial nodes
            x_t = node_feats[:, t].mean(dim=1)  # (B, hidden_dim)

            # Neural ODE integration
            next_hidden = self.temporal_processor(
                current_hidden, x_t, dt=0.1, n_steps=1
            )

            # Decode control signal
            control = self.control_decoder(next_hidden)
            controls.append(control)
            current_hidden = next_hidden

        controls = torch.stack(controls, dim=1)
        final_hidden = current_hidden

        return controls, final_hidden
