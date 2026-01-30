#!/usr/bin/env python
"""
モデルの外乱（Corruption）耐性評価スクリプト

異なる外乱（ノイズ、白飛び、トンネル出口など）におけるモデルの頑健性を評価します。
"""

import argparse
from pathlib import Path
import json
import random
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.core.models import LTCNController, CfGCNController, NeuralGraphODEController
from src.driving.data import setup_dataloaders
from src.utils import (
    add_gaussian_noise,
    add_static_bias,
    add_overexposure,
    simulate_tunnel_exit,
)


class CorruptionRobustnessEvaluator:
    """外乱耐性評価クラス"""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")

    def evaluate_robustness(
        self,
        model: nn.Module,
        test_data: dict[str, torch.Tensor],
        levels: list[float],
        corruption_type: str = "noise",
    ) -> dict[str, Any]:
        """評価を実行"""
        results: dict[str, Any] = {
            "metrics": {},
            "metadata": {"corruption_type": corruption_type},
        }

        for level in levels:
            print(f"Testing {corruption_type} level: {level}")

            clean_frames = test_data["clean_frames"]
            sensors = test_data["sensors"]

            # 外乱の適用
            corrupted_frames = self._apply_corruption(
                clean_frames, level, corruption_type
            )

            # モデルテスト
            metrics = self._evaluate_model(
                model, clean_frames, corrupted_frames, sensors
            )
            results["metrics"][f"level_{level}"] = metrics

        # ロバスト性スコア (Slope)
        results["robustness_score"] = self._calculate_robustness_score(
            results["metrics"]
        )

        return results

    def _apply_corruption(
        self, frames: torch.Tensor, level: float, corruption_type: str
    ) -> torch.Tensor:
        """フレームに外乱を適用"""
        if corruption_type == "noise":
            # ガウシアンノイズ (level = std)
            return torch.stack([add_gaussian_noise(f, std=level) for f in frames])
        elif corruption_type == "bias":
            # Level 1: Static Bias (level = bias value)
            return torch.stack([add_static_bias(f, bias=level) for f in frames])
        elif corruption_type == "overexposure":
            # Level 2: Contrast/Overexposure (level = factor)
            return torch.stack([add_overexposure(f, factor=level) for f in frames])
        elif corruption_type == "tunnel":
            # Level 3: Tunnel Exit (level = peak_intensity)
            corrupted_batch = []
            for i in range(frames.shape[0]):
                # 個別のシーケンス [T, H, W, C]
                seq = frames[i]
                # トンネル出口の位置をシーケンスの真ん中あたりに設定
                exit_idx = seq.shape[0] // 2
                corrupted_seq = simulate_tunnel_exit(
                    seq, exit_idx=exit_idx, peak_intensity=level, transition_duration=5
                )
                corrupted_batch.append(corrupted_seq)
            return torch.stack(corrupted_batch)

        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")

    def _evaluate_model(
        self,
        model: nn.Module,
        clean_data: torch.Tensor,
        corrupted_data: torch.Tensor,
        sensors: torch.Tensor,
    ) -> dict[str, float]:
        """単一モデルの評価"""
        clean_data = clean_data.to(self.device)
        corrupted_data = corrupted_data.to(self.device)
        sensors = sensors.to(self.device)

        model.eval()
        with torch.no_grad():
            # クリーンデータでの予測
            pred_clean, _ = model(clean_data)
            # 外乱付きデータでの予測
            pred_corrupted, _ = model(corrupted_data)

            # シーケンスの最後のタイムステップで評価
            pred_clean_last = pred_clean[:, -1, :]
            pred_corrupted_last = pred_corrupted[:, -1, :]
            sensors_last = sensors[:, -1, :]

            # 外乱付きデータでの誤差
            control_mse = nn.MSELoss()(pred_corrupted_last, sensors_last).item()
            control_mae = nn.L1Loss()(pred_corrupted_last, sensors_last).item()

            # 出力の安定性: 外乱による予測の変動
            output_variance = nn.MSELoss()(pred_corrupted_last, pred_clean_last).item()

        return {
            "control_mse": control_mse,
            "control_mae": control_mae,
            "output_variance": output_variance,
        }

    def _calculate_robustness_score(self, metrics: dict[str, Any]) -> dict[str, float]:
        """ロバスト性スコアを計算"""
        levels = []
        mse_values = []
        for key in sorted(metrics.keys()):
            if key.startswith("level_"):
                lvl = float(key.split("_")[1])
                levels.append(lvl)
                mse_values.append(metrics[key]["control_mse"])

        if len(levels) > 1:
            slope = np.polyfit(levels, mse_values, 1)[0]
            return {"slope": float(slope)}
        return {"slope": 0.0}

    def visualize_results(self, results: dict[str, Any], save_path: Path | None = None):
        """評価結果を可視化"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes = axes.flatten()

        c_type = results["metadata"]["corruption_type"]
        metrics_data = results["metrics"]
        levels = [
            float(k.split("_")[1]) for k in metrics_data if k.startswith("level_")
        ]
        levels.sort()

        metrics_to_plot = [
            ("control_mse", f"MSE vs {c_type}"),
            ("control_mae", f"MAE vs {c_type}"),
            ("output_variance", f"Variance vs {c_type}"),
        ]

        for i, (metric, title) in enumerate(metrics_to_plot):
            vals = [metrics_data[f"level_{lvl}"][metric] for lvl in levels]

            axes[i].plot(levels, vals, "b-o", linewidth=2)
            axes[i].set_xlabel("Severity Level")
            axes[i].set_ylabel(metric)
            axes[i].set_title(title)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")


def run_corruption_robustness_evaluation(args: argparse.Namespace):
    """評価実行"""
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Init seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as _:
        mlflow.log_params(vars(args))

        # Data
        _, _, test_loader, _ = setup_dataloaders(
            data_dir=args.data_dir,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            processed_dir=args.processed_dir,
        )

        # Model
        print(f"Creating and loading {args.model_type.upper()} model...")
        if args.model_type == "ltcn":
            model = LTCNController(
                frame_height=64,
                frame_width=64,
                output_dim=6,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers_ltcn,
            )
        elif args.model_type == "lgtcn":
            model = CfGCNController(
                frame_height=64,
                frame_width=64,
                hidden_dim=args.hidden_dim,
                output_dim=6,
                K=args.K,
                num_layers=args.num_layers_cfgcn,
            )
        elif args.model_type == "ngode":
            model = NeuralGraphODEController(
                frame_height=64,
                frame_width=64,
                hidden_dim=args.hidden_dim,
                output_dim=6,
                K=args.K,
                num_layers=args.num_layers_cfgcn,
            )
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")

        # Load Weights
        print(f"Loading model from {args.model_path}")
        ckpt = torch.load(args.model_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model: {e}")
            model = torch.load(args.model_path, map_location=device)

        model.eval()
        model.to(device)

        # Get Test Batch
        batch = next(iter(test_loader))
        frames, sensors = batch[0].to(device), batch[1].to(device)
        test_data = {"clean_frames": frames, "sensors": sensors}

        # Levels
        levels = [float(x) for x in args.levels.split(",")]

        # Run Eval
        evaluator = CorruptionRobustnessEvaluator(device)
        results = evaluator.evaluate_robustness(
            model, test_data, levels, args.corruption_type
        )

        # Save
        r_path = save_dir / f"robustness_{args.model_type}_{args.corruption_type}.json"
        p_path = save_dir / f"robustness_{args.model_type}_{args.corruption_type}.png"

        with open(r_path, "w") as f:
            json.dump(results, f, indent=2)

        evaluator.visualize_results(results, p_path)

        mlflow.log_artifact(str(r_path))
        mlflow.log_artifact(str(p_path))

        print(f"Done. Check {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate corruption robustness")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--save-dir", default="./corruption_results")

    parser.add_argument(
        "--model-type", required=True, choices=["ltcn", "lgtcn", "ngode"]
    )
    parser.add_argument("--model-path", required=True)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers-ltcn", type=int, default=4)
    parser.add_argument("--num-layers-cfgcn", type=int, default=1)
    parser.add_argument("--K", type=int, default=2)

    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    # Corruption Params
    parser.add_argument(
        "--corruption-type",
        choices=["noise", "bias", "overexposure", "tunnel"],
        default="noise",
        help="Type of corruption to apply",
    )
    parser.add_argument(
        "--levels",
        default="0.0,0.1,0.2,0.3",
        help="Comma-separated levels (std, bias, factor, intensity)",
    )

    args = parser.parse_args()
    run_corruption_robustness_evaluation(args)
