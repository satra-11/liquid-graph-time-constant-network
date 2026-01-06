#!/usr/bin/env python
"""
LTCNとNeural ODEのノイズ耐性評価スクリプト

異なるノイズレベルにおけるLTCNとNeural ODEモデルの頑健性を比較評価します。
"""

import time
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

from src.core.models import LTCNController, NeuralODEController
from src.driving.data import setup_dataloaders
from src.utils import add_gaussian_noise


class NoiseRobustnessEvaluator:
    """LTCNとNeural ODEのノイズ耐性評価クラス"""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")

    def evaluate_robustness(
        self,
        ltcn_model: nn.Module,
        node_model: nn.Module,
        test_data: dict[str, torch.Tensor],
        noise_levels: list[float] = [0.0, 0.1, 0.2, 0.3],
    ) -> dict[str, Any]:
        """LTCNとNeural ODEをノイズ耐性で比較評価

        Args:
            ltcn_model: LTCNモデル
            node_model: Neural ODEモデル
            test_data: テストデータ（clean_frames, sensors）
            noise_levels: 評価するノイズレベルのリスト

        Returns:
            評価結果の辞書
        """
        results: dict[str, Any] = {"ltcn": {}, "node": {}, "comparison": {}}

        for noise_level in noise_levels:
            print(f"Testing noise level: {noise_level}")

            clean_frames = test_data["clean_frames"]
            sensors = test_data["sensors"]

            # ノイズ付きフレーム生成（noise_levelに応じてガウシアンノイズを追加）
            noisy_frames = torch.stack(
                [add_gaussian_noise(frame, std=noise_level) for frame in clean_frames]
            )

            # LTCNテスト
            ltcn_metrics = self._evaluate_model(
                ltcn_model, clean_frames, noisy_frames, sensors
            )
            results["ltcn"][f"noise_{noise_level}"] = ltcn_metrics

            # Neural ODEテスト
            node_metrics = self._evaluate_model(
                node_model, clean_frames, noisy_frames, sensors
            )
            results["node"][f"noise_{noise_level}"] = node_metrics

        # 比較サマリー
        results["comparison"] = self._generate_comparison_summary(results)

        return results

    def _evaluate_model(
        self,
        model: nn.Module,
        clean_data: torch.Tensor,
        noisy_data: torch.Tensor,
        sensors: torch.Tensor,
    ) -> dict[str, float]:
        """単一モデルの評価"""
        clean_data = clean_data.to(self.device)
        noisy_data = noisy_data.to(self.device)
        sensors = sensors.to(self.device)

        model.eval()
        with torch.no_grad():
            # クリーンデータでの予測
            pred_clean, _ = model(clean_data)
            # ノイズ付きデータでの予測
            pred_noisy, _ = model(noisy_data)

            # シーケンスの最後のタイムステップで評価
            pred_clean_last = pred_clean[:, -1, :]
            pred_noisy_last = pred_noisy[:, -1, :]
            sensors_last = sensors[:, -1, :]

            # ノイズ付きデータでの誤差
            control_mse = nn.MSELoss()(pred_noisy_last, sensors_last).item()
            control_mae = nn.L1Loss()(pred_noisy_last, sensors_last).item()

            # クリーンデータでの誤差（参考用）
            clean_mse = nn.MSELoss()(pred_clean_last, sensors_last).item()
            clean_mae = nn.L1Loss()(pred_clean_last, sensors_last).item()

            # 出力の安定性: ノイズによる予測の変動
            output_variance = nn.MSELoss()(pred_noisy_last, pred_clean_last).item()

        return {
            "control_mse": control_mse,
            "control_mae": control_mae,
            "clean_mse": clean_mse,
            "clean_mae": clean_mae,
            "output_variance": output_variance,
        }

    def _generate_comparison_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """比較サマリーを生成"""
        summary: dict[str, Any] = {
            "winner_by_metric": {},
            "robustness_score": {},
        }

        metrics = ["control_mse", "control_mae", "output_variance"]

        for metric in metrics:
            ltcn_values = []
            node_values = []

            for key in results["ltcn"]:
                if key.startswith("noise_"):
                    ltcn_val = results["ltcn"][key][metric]
                    node_val = results["node"][key][metric]
                    ltcn_values.append(ltcn_val)
                    node_values.append(node_val)

            ltcn_avg = float(np.mean(ltcn_values))
            node_avg = float(np.mean(node_values))

            # 小さい方が良いメトリクス（MSE, MAE, variance）
            winner = "LTCN" if ltcn_avg < node_avg else "Neural ODE"

            summary["winner_by_metric"][metric] = {
                "winner": winner,
                "ltcn_avg": ltcn_avg,
                "node_avg": node_avg,
                "difference": float(abs(ltcn_avg - node_avg)),
                "ltcn_values": ltcn_values,
                "node_values": node_values,
            }

        # ノイズに対するロバスト性スコア（ノイズレベルの増加に対する誤差増加率）
        for model_name, model_results in [
            ("ltcn", results["ltcn"]),
            ("node", results["node"]),
        ]:
            noise_levels = []
            mse_values = []
            for key in sorted(model_results.keys()):
                if key.startswith("noise_"):
                    noise_level = float(key.split("_")[1])
                    noise_levels.append(noise_level)
                    mse_values.append(model_results[key]["control_mse"])

            if len(noise_levels) > 1:
                # ノイズ増加に対する誤差の傾き（勾配）を計算
                slope = np.polyfit(noise_levels, mse_values, 1)[0]
                summary["robustness_score"][model_name] = {
                    "slope": float(slope),
                    "interpretation": "Lower slope = more robust to noise",
                }

        return summary

    def visualize_comparison(
        self, results: dict[str, Any], save_path: Path | None = None
    ):
        """比較結果を可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        noise_levels = [
            float(k.split("_")[1]) for k in results["ltcn"] if k.startswith("noise_")
        ]
        noise_levels.sort()

        metrics_to_plot = [
            ("control_mse", "Control MSE (Noisy Input)"),
            ("control_mae", "Control MAE (Noisy Input)"),
            ("output_variance", "Output Variance (Noisy vs Clean)"),
            ("clean_mse", "Control MSE (Clean Input)"),
        ]

        for i, (metric, title) in enumerate(metrics_to_plot):
            ltcn_values = []
            node_values = []

            for noise_level in noise_levels:
                key = f"noise_{noise_level}"
                ltcn_val = results["ltcn"][key][metric]
                node_val = results["node"][key][metric]
                ltcn_values.append(ltcn_val)
                node_values.append(node_val)

            axes[i].plot(noise_levels, ltcn_values, "b-o", label="LTCN", linewidth=2)
            axes[i].plot(
                noise_levels, node_values, "r-s", label="Neural ODE", linewidth=2
            )
            axes[i].set_xlabel("Noise Level (std)")
            axes[i].set_ylabel(title)
            axes[i].set_title(title)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        plt.show()


def run_noise_robustness_evaluation(args: argparse.Namespace):
    """ノイズ耐性評価のメイン関数"""
    # デバイス設定
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # シード設定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 保存ディレクトリ作成
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_params(vars(args))

        # データセット作成（テストデータのみ必要）
        _, _, test_loader, _ = setup_dataloaders(
            data_dir=args.data_dir,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            processed_dir=args.processed_dir,
        )

        # モデル作成
        print("Creating models...")
        ltcn_model = LTCNController(
            frame_height=64,
            frame_width=64,
            output_dim=6,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers_ltcn,
        )

        node_model = NeuralODEController(
            frame_height=64,
            frame_width=64,
            output_dim=6,
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_hidden_layers_node,
            solver=args.solver,
        )

        # モデルのロード
        print(f"Loading LTCN model from {args.ltcn_model_path}")
        ltcn_checkpoint = torch.load(args.ltcn_model_path, map_location=device)
        try:
            if "model_state_dict" in ltcn_checkpoint:
                ltcn_model.load_state_dict(ltcn_checkpoint["model_state_dict"])
            else:
                ltcn_model.load_state_dict(ltcn_checkpoint)
        except Exception as e:
            print(f"Error loading LTCN model: {e}")
            ltcn_model = torch.load(args.ltcn_model_path, map_location=device)

        print(f"Loading Neural ODE model from {args.node_model_path}")
        node_checkpoint = torch.load(args.node_model_path, map_location=device)
        try:
            if "model_state_dict" in node_checkpoint:
                node_model.load_state_dict(node_checkpoint["model_state_dict"])
            else:
                node_model.load_state_dict(node_checkpoint)
        except Exception as e:
            print(f"Error loading Neural ODE model: {e}")
            node_model = torch.load(args.node_model_path, map_location=device)

        # モデルをデバイスに転送
        ltcn_model.eval()
        node_model.eval()
        ltcn_model.to(device)
        node_model.to(device)

        # テストデータを1バッチ取得
        with torch.no_grad():
            try:
                test_frames, test_sensors, _, _ = next(iter(test_loader))
            except StopIteration:
                raise RuntimeError(
                    "Test loader が空です。テストデータが読み込まれているか確認してください。"
                )

        # デバイスへ転送
        test_frames = test_frames.to(device)
        test_sensors = test_sensors.to(device)

        test_data = {
            "clean_frames": test_frames,
            "sensors": test_sensors,
        }

        # ノイズレベルの設定
        noise_levels = [float(x) for x in args.noise_levels.split(",")]

        # 評価の実行
        evaluator = NoiseRobustnessEvaluator(device)
        results = evaluator.evaluate_robustness(
            ltcn_model, node_model, test_data, noise_levels=noise_levels
        )

        # 結果を保存
        results_path = save_dir / "noise_robustness_results.json"
        plots_path = save_dir / "noise_robustness_plots.png"

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        evaluator.visualize_comparison(results, plots_path)

        # 評価結果をMLflowに記録
        print("Logging artifacts to MLflow...")
        mlflow.log_artifact(str(plots_path))
        mlflow.log_artifact(str(results_path))

        # 最終的なサマリーメトリクスを記録
        summary = results.get("comparison", {}).get("winner_by_metric", {})
        for metric, values in summary.items():
            mlflow.log_metric(f"LTCN_avg_{metric}", values.get("ltcn_avg", 0))
            mlflow.log_metric(f"NeuralODE_avg_{metric}", values.get("node_avg", 0))

        # ロバスト性スコアを記録
        robustness = results.get("comparison", {}).get("robustness_score", {})
        for model_name, score_info in robustness.items():
            mlflow.log_metric(
                f"{model_name}_robustness_slope", score_info.get("slope", 0)
            )

    print(f"\nEvaluation completed! Results saved to {save_dir}")
    print("To view results, run 'mlflow ui' and open http://localhost:5000")

    # サマリーを出力
    print("\n" + "=" * 60)
    print("NOISE ROBUSTNESS EVALUATION SUMMARY")
    print("=" * 60)
    for metric, values in summary.items():
        print(f"\n{metric}:")
        print(f"  LTCN avg: {values['ltcn_avg']:.6f}")
        print(f"  Neural ODE avg: {values['node_avg']:.6f}")
        print(f"  Winner: {values['winner']}")

    if robustness:
        print("\nRobustness Score (lower slope = more robust):")
        for model_name, score_info in robustness.items():
            print(f"  {model_name}: {score_info['slope']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate noise robustness of LTCN and Neural ODE controllers"
    )
    # Data params
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./noise_robustness_results",
        help="Directory to save results",
    )

    # Model paths
    parser.add_argument(
        "--ltcn-model-path",
        type=str,
        required=True,
        help="Path to LTCN model checkpoint",
    )
    parser.add_argument(
        "--node-model-path",
        type=str,
        required=True,
        help="Path to Neural ODE model checkpoint",
    )

    # Model params (LTCN)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers-ltcn", type=int, default=4)

    # Model params (Neural ODE)
    parser.add_argument("--num-hidden-layers-node", type=int, default=2)
    parser.add_argument(
        "--solver", type=str, default="dopri5", help="ODE solver (dopri5, euler, rk4)"
    )

    # Evaluation params
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--noise-levels",
        type=str,
        default="0.0,0.05,0.1,0.15,0.2,0.25,0.3",
        help="Comma-separated list of noise levels (std dev)",
    )

    # System params
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    start_time = time.time()
    run_noise_robustness_evaluation(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"\nTotal execution time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}."
    )
