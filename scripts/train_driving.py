#!/usr/bin/env python
"""
映像データによる自律走行タスクでのLGTCN/LTCN訓練スクリプト
"""

import argparse
import random
from pathlib import Path
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from src.tasks import (
    NetworkComparator,
)
from src.core.models import CfGCNController, LTCNController
from src.data import HDDLoader


def main():
    parser = argparse.ArgumentParser(
        description="Train driving controllers with LGTCN/LTCN"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-sequences", type=int, default=800)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--corruption-rate", type=float, default=0.2)
    parser.add_argument("--data-dir", type=str, default="./hdd")
    parser.add_argument("--save-dir", type=str, default="./driving_results")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--sensor-sequence",
        type=str,
        default=None,
        help="Sensor sequence name for testing (e.g., 201702271017)",
    )

    args = parser.parse_args()

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

        # データセット作成
        print("Loading dataset from HDD...")

        full_dataset = HDDLoader(
            camera_dir=os.path.join(args.data_dir, "camera"),
            sensor_dir=os.path.join(args.data_dir, "sensor"),
            sequence_length=args.sequence_length,
            exclude_features=["rtk_pos_info", "rtk_track_info"],
        )

        # 訓練・検証・テストに分割
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        # データローダー作成
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )

        # モデル作成
        print("Creating models...")
        lgtcn_model = CfGCNController(
            frame_height=64,
            frame_width=64,
            hidden_dim=args.hidden_dim,
            output_dim=6,
            K=args.K,
        )

        ltcn_model = LTCNController(
            frame_height=64,
            frame_width=64,
            output_dim=6,
            hidden_dim=args.hidden_dim,
        )

        # LGTCN訓練
        print("Training LGTCN...")
        train_model(
            lgtcn_model,
            "LGTCN",
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            device=device,
        )

        # LTCN訓練
        print("Training LTCN...")
        train_model(
            ltcn_model,
            "LTCN",
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            device=device,
        )

        # モデル保存 (MLflow)
        print("Logging models to MLflow...")
        mlflow.pytorch.log_model(lgtcn_model, "lgtcn_model")
        mlflow.pytorch.log_model(ltcn_model, "ltcn_model")

        # 評価のためにテストデータを1バッチ取得
        lgtcn_model.eval()
        ltcn_model.eval()

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

        # evaluate_networks に渡す dict
        test_data = {
            "clean_frames": test_frames,
            "sensors": test_sensors,
        }

        # 評価の実行
        results = evaluate_networks(lgtcn_model, ltcn_model, test_data, device)

        # 結果を保存
        comparator = NetworkComparator(device)
        results_path = save_dir / "comparison_results.json"
        plots_path = save_dir / "comparison_plots.png"

        # `save_results` は存在しないため、直接jsonをダンプ
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        comparator.visualize_comparison(results, plots_path)

        # 評価結果をMLflowに記録
        print("Logging artifacts to MLflow...")
        mlflow.log_artifact(str(plots_path))
        mlflow.log_artifact(str(results_path))

        # 最終的なサマリーメトリクスを記録
        summary = results.get("comparison", {}).get("winner_by_metric", {})
        for metric, values in summary.items():
            mlflow.log_metric(f"LGTCN_avg_{metric}", values.get("lgtcn_avg", 0))
            mlflow.log_metric(f"LTCN_avg_{metric}", values.get("ltcn_avg", 0))

    print(f"Training completed! Results saved to {save_dir}")
    print(
        f"To view results, run 'mlflow ui' in the directory '{Path.cwd()}' and open http://localhost:5000"
    )


def train_model(
    model: LTCNController | CfGCNController,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: torch.device = None,
):
    """モデルを訓練"""
    device = device or torch.device("cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    start_time = time.time()

    for epoch in range(num_epochs):
        # 訓練フェーズ
        model.train()
        epoch_train_loss = 0.0

        for frames, sensors, _, _ in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN - {model_name}]"
        ):
            frames = frames.to(device)
            sensors = sensors.to(device)

            optimizer.zero_grad()

            predictions, _ = model(frames)

            loss = criterion(predictions[:, -1, :], sensors[:, -1, :])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        mlflow.log_metric(f"{model_name} Train Loss", avg_train_loss, step=epoch)

        # 検証フェーズ
        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for frames, sensors, _, _ in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [VAL - {model_name}]"
            ):
                frames = frames.to(device)
                sensors = sensors.to(device)

                predictions, _ = model(frames)

                # 予測の最後のタイムステップと比較
                loss = criterion(predictions[:, -1, :], sensors[:, -1, :])
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        mlflow.log_metric(f"{model_name} Val Loss", avg_val_loss, step=epoch)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}: Train Loss ({model_name}) = {avg_train_loss:.6f}, Val Loss ({model_name}) = {avg_val_loss:.6f}"
            )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Training for {model_name} finished in {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}."
    )


def evaluate_networks(
    lgtcn_model: CfGCNController,
    ltcn_model: LTCNController,
    test_data: dict,
    device: torch.device,
):
    """LGTCNとLTCNを比較評価"""
    comparator = NetworkComparator(device)

    # テストデータ準備
    test_dict = {
        "clean_frames": test_data["clean_frames"],
        "sensors": test_data["sensors"],
        "adjacency": None,
    }

    print("Comparing LGTCN and LTCN...")
    results = comparator.compare_networks(
        lgtcn_model,
        ltcn_model,
        test_dict,
        corruption_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    )

    return results


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Total execution time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}."
    )
