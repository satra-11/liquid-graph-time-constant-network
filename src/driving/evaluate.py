import argparse
import json
import random
from pathlib import Path

import mlflow
import numpy as np
import torch

from src.core.models import (
    CfGCNController,
    LTCNController,
    NeuralODEController,
    NeuralGraphODEController,
)
from src.driving.data import setup_dataloaders
from src.driving.engine import evaluate_model


def run_single_model_evaluation(args: argparse.Namespace):
    """単一モデルを評価する関数"""
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

        # データセット作成 (テストデータのみ必要だが、setup_dataloadersは全部返す)
        _, _, test_loader, _ = setup_dataloaders(
            data_dir=args.data_dir,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            processed_dir=args.processed_dir,
        )

        # モデル作成
        print(f"Creating {args.model.upper()} model...")
        if args.model == "lgtcn":
            model = CfGCNController(
                frame_height=64,
                frame_width=64,
                hidden_dim=args.hidden_dim,
                output_dim=6,
                K=args.K,
                num_layers=args.num_layers_cfgcn,
            )
        elif args.model == "ltcn":
            model = LTCNController(
                frame_height=64,
                frame_width=64,
                output_dim=6,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers_ltcn,
            )
        elif args.model == "node":
            model = NeuralODEController(
                frame_height=64,
                frame_width=64,
                hidden_dim=args.hidden_dim,
                output_dim=6,
            )
        elif args.model == "ngode":
            model = NeuralGraphODEController(
                frame_height=64,
                frame_width=64,
                hidden_dim=args.hidden_dim,
                output_dim=6,
                K=args.K,
                num_layers=args.num_layers_cfgcn,
            )
        else:
            raise ValueError(f"Unknown model type: {args.model}")

        # モデルのロード
        print(f"Loading {args.model.upper()} model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        try:
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback for direct model load if it was saved as full model
            model = torch.load(args.model_path, map_location=device)

        # 評価のためにテストデータを1バッチ取得
        model.eval()
        model.to(device)

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

        # evaluate_single_model に渡す dict
        test_data = {
            "clean_frames": test_frames,
            "sensors": test_sensors,
        }

        # 評価の実行
        results = evaluate_model(model, args.model, test_data, device)

        # 結果を保存
        results_path = save_dir / f"{args.model}_evaluation_results.json"

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # 評価結果をMLflowに記録
        print("Logging artifacts to MLflow...")
        mlflow.log_artifact(str(results_path))

        # メトリクスを記録
        for metric_name, metric_value in results.get("metrics", {}).items():
            mlflow.log_metric(f"{args.model}_{metric_name}", metric_value)

    print(f"Evaluation completed! Results saved to {results_path}")
    print(
        f"To view results, run 'mlflow ui' in the directory '{Path.cwd()}' and open http://localhost:5000"
    )
