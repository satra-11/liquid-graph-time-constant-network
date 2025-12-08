#!/usr/bin/env python
"""
勾配の流れを確認するデバッグスクリプト
"""

import torch
import torch.nn as nn
import os
import sys

# パスを追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.models import CfGCNController
from src.driving.data import setup_dataloaders


def check_gradients():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # データセット作成（小さいバッチで確認）
    train_loader, _, _, _ = setup_dataloaders(
        data_dir="./data/raw",
        sequence_length=20,
        batch_size=16,  # 小さいバッチ
        processed_dir=None,  # 生画像から読み込む
    )

    # モデル作成
    model = CfGCNController(
        frame_height=64,
        frame_width=64,
        hidden_dim=64,
        output_dim=6,
        K=2,
        num_layers=1,
    ).to(device)

    criterion = nn.MSELoss()

    # 1バッチだけ取得
    frames, sensors, _, _ = next(iter(train_loader))
    frames = frames.to(device)
    sensors = sensors.to(device)

    print("\n=== Input Data ===")
    print(f"frames shape: {frames.shape}, dtype: {frames.dtype}")
    print(f"frames mean: {frames.mean():.4f}, std: {frames.std():.4f}")
    print(f"sensors shape: {sensors.shape}, dtype: {sensors.dtype}")
    print(f"sensors mean: {sensors.mean():.4f}, std: {sensors.std():.4f}")
    print(f"sensors min: {sensors.min():.4f}, max: {sensors.max():.4f}")

    # Forward pass
    model.train()
    predictions, hidden = model(frames)

    print("\n=== Model Output ===")
    print(f"predictions shape: {predictions.shape}")
    print(f"predictions mean: {predictions.mean():.6f}, std: {predictions.std():.6f}")
    print(f"predictions min: {predictions.min():.6f}, max: {predictions.max():.6f}")
    print(f"hidden shape: {hidden.shape}")
    print(f"hidden mean: {hidden.mean():.6f}, std: {hidden.std():.6f}")

    # Loss計算
    pred_last = predictions[:, -1, :]
    target_last = sensors[:, -1, :]
    loss = criterion(pred_last, target_last)

    print("\n=== Loss ===")
    print(f"Loss value: {loss.item():.6f}")
    print(
        f"Target variance (expected loss if output is constant): {target_last.var():.6f}"
    )

    # Backward pass
    loss.backward()

    # 勾配を確認
    print("\n=== Gradient Analysis ===")
    print(
        f"{'Layer':<50} {'grad_mean':>12} {'grad_std':>12} {'grad_max':>12} {'param_mean':>12}"
    )
    print("-" * 100)

    zero_grad_layers = []
    small_grad_layers = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            grad_std = param.grad.std().item()
            grad_max = param.grad.abs().max().item()
            param_mean = param.abs().mean().item()

            status = ""
            if grad_max < 1e-8:
                status = " [ZERO GRAD!]"
                zero_grad_layers.append(name)
            elif grad_max < 1e-5:
                status = " [VERY SMALL]"
                small_grad_layers.append(name)

            print(
                f"{name:<50} {grad_mean:>12.2e} {grad_std:>12.2e} {grad_max:>12.2e} {param_mean:>12.4f}{status}"
            )
        else:
            print(f"{name:<50} {'NO GRAD':>12}")

    # サマリー
    print("\n=== Summary ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if zero_grad_layers:
        print(f"\n⚠️ Layers with ZERO gradients ({len(zero_grad_layers)}):")
        for name in zero_grad_layers:
            print(f"  - {name}")

    if small_grad_layers:
        print(f"\n⚠️ Layers with VERY SMALL gradients ({len(small_grad_layers)}):")
        for name in small_grad_layers:
            print(f"  - {name}")

    # 中間層の活性化値を確認
    print("\n=== Activation Analysis ===")
    with torch.no_grad():
        # CfGCNLayerの内部状態を確認
        B, T, C, H, W = frames.shape
        if C == 128 and H == 8 and W == 8:
            features = frames
        else:
            frames_flat = frames.view(-1, C, H, W)
            features = model.feature_extractor(frames_flat)
            features = features.view(B, T, 128, 8, 8)

        features = features.permute(0, 1, 3, 4, 2)
        node_feats = features.reshape(B, T, 64, 128)
        node_feats = model.node_encoder(node_feats)

        print(f"node_feats mean: {node_feats.mean():.4f}, std: {node_feats.std():.4f}")
        print(f"node_feats min: {node_feats.min():.4f}, max: {node_feats.max():.4f}")

        # hidden stateの初期化を確認
        N = node_feats.size(2)
        hidden_state = (
            torch.randn(B, model.num_layers, N, model.hidden_dim, device=device) * 0.01
        )
        print(
            f"initial hidden mean: {hidden_state.mean():.6f}, std: {hidden_state.std():.6f}"
        )


if __name__ == "__main__":
    check_gradients()
