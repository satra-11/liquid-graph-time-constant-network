import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.core.layers import CfGCNLayer
from src.utils import compute_s_powers, compute_laplacian, compute_random_walk_matrix


class CfGCNController(nn.Module):
    """映像データからCfGCNを使って制御信号を生成するコントローラー"""

    def __init__(
        self,
        frame_height: int = 64,
        frame_width: int = 64,
        hidden_dim: int = 64,
        K: int = 2,
        output_dim: int = 6,
        matrix_type: str = "adjacency",
        num_layers: int = 1,
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

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.temporal_processor = nn.ModuleList(
            [CfGCNLayer(hidden_dim, hidden_dim, K) for _ in range(num_layers)]
        )

        self.control_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim),
        )

        # 学習可能な出力スケーリングパラメータ
        # 正規化されたセンサーデータのスケール(std≈1)に合わせて設定
        # モデル出力が約0.1程度なので、10倍程度のスケールが必要
        self.output_scale = nn.Parameter(torch.ones(self.output_dim) * 10.0)
        self.output_bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(
        self,
        frames: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
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

        attentions = []

        if adjacency is None:
            N = node_feats.size(2)
            adjacency = torch.ones(B, T, N, N, device=frames.device)

        controls = []

        N = node_feats.size(2)
        if hidden_state is None:
            # 小さなランダム値で初期化（ゼロ初期化だとCfGCNLayerの除算が不安定になる）
            hidden_state = (
                torch.randn(
                    B, self.num_layers, N, self.hidden_dim, device=frames.device
                )
                * 0.01
            )

        current_hiddens = hidden_state.unbind(dim=1)

        for t in range(T):
            xt = node_feats[:, t, :, :]

            A_t = adjacency[:, t, :, :]
            attentions.append(A_t)

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

            # 1st layer
            h_in = h_prev_layers[0]
            u_in = xt
            h_out = self.temporal_processor[0](h_in, u_in, S_powers)
            h_next_layers.append(h_out)

            # 2nd layer to N
            for i in range(1, self.num_layers):
                h_in = h_prev_layers[i]
                u_in = h_next_layers[i - 1]
                h_out = self.temporal_processor[i](h_in, u_in, S_powers)
                h_next_layers.append(h_out)

            final_layer_output = h_next_layers[-1]
            control = self.control_decoder(final_layer_output.mean(dim=1))
            # 出力スケーリングを適用
            control = control * self.output_scale + self.output_bias
            controls.append(control)
            current_hiddens = tuple(h_next_layers)

        controls = torch.stack(controls, dim=1)
        final_hidden = torch.stack(current_hiddens, dim=1)
        attentions = torch.stack(attentions, dim=1)

        return controls, final_hidden
