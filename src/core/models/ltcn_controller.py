import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.core.layers import LTCNLayer


class LTCNController(nn.Module):
    """映像データからLTCNを使って制御信号を生成するコントローラー"""

    def __init__(
        self,
        frame_height: int = 64,
        frame_width: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 6,
        num_layers: int = 4,
    ):
        super().__init__()
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.node_encoder = nn.Linear(128, hidden_dim)

        assert (
            hidden_dim % num_layers == 0
        ), "hidden_dim must be divisible by num_layers"

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        k_per_block = hidden_dim // num_layers
        num_blocks = num_layers
        self.temporal_processor = LTCNLayer(hidden_dim, k_per_block, num_blocks)

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
        B, T, C, H, W = frames.shape

        if C == 128 and H == 8 and W == 8:
            features = frames
        else:
            frames_flat = frames.view(-1, C, H, W)
            features = self.feature_extractor(frames_flat)
            features = features.view(B, T, 128, 8, 8)

        features = features.permute(0, 1, 3, 4, 2)
        node_feats = features.reshape(B, T, 64, 128)
        node_feats = self.node_encoder(node_feats)

        controls = []

        if hidden_state is None:
            hidden_state = torch.zeros(
                B, self.temporal_processor.N, device=frames.device
            )

        current_hidden = hidden_state

        for t in range(T):
            x_t = node_feats[:, t].mean(dim=1)
            next_hidden = self.temporal_processor(
                current_hidden, x_t, dt=0.1, n_steps=1
            )

            control = self.control_decoder(next_hidden)
            controls.append(control)
            current_hidden = next_hidden

        controls = torch.stack(controls, dim=1)
        final_hidden = current_hidden

        return controls, final_hidden
