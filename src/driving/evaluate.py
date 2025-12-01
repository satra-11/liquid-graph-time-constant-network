import argparse
import json
import random
from pathlib import Path

import mlflow
import numpy as np
import torch

from src.core.models import CfGCNController, LTCNController
from src.driving.data import setup_dataloaders
from src.driving.engine import evaluate_networks
from src.tasks import NetworkComparator


def run_evaluation(args: argparse.Namespace):
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
        print("Creating models...")
        lgtcn_model = CfGCNController(
            frame_height=64,
            frame_width=64,
            hidden_dim=args.hidden_dim,
            output_dim=6,
            K=args.K,
            num_layers=args.num_layers_cfgcn,
        )

        ltcn_model = LTCNController(
            frame_height=64,
            frame_width=64,
            output_dim=6,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers_ltcn,
        )

        # モデルのロード
        print(f"Loading LGTCN model from {args.lgtcn_model_path}")
        lgtcn_checkpoint = torch.load(args.lgtcn_model_path, map_location=device)
        # Checkpoint could be the full checkpoint dict or just state_dict
        # Based on run.py, it saves a dict with "model_state_dict"
        # But user might provide a direct model path if saved differently.
        # run.py saves with mlflow.pytorch.log_model which saves the whole model object usually,
        # but also saves checkpoints manually with torch.save.
        # The user said "lgtcn/driving_results/lgtcn_model.pth".
        # Let's assume it's a state dict or we try to load it.
        # If the user provides the path to the file saved by `torch.save` in `run.py`, it has `model_state_dict`.
        # If it's from mlflow, it might be different.
        # Let's assume it's the checkpoint format from `run.py` first.

        try:
            if "model_state_dict" in lgtcn_checkpoint:
                lgtcn_model.load_state_dict(lgtcn_checkpoint["model_state_dict"])
            else:
                lgtcn_model.load_state_dict(lgtcn_checkpoint)
        except Exception as e:
            print(f"Error loading LGTCN model: {e}")
            # Fallback for direct model load if it was saved as full model
            lgtcn_model = torch.load(args.lgtcn_model_path, map_location=device)

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

        # 評価のためにテストデータを1バッチ取得
        lgtcn_model.eval()
        ltcn_model.eval()
        lgtcn_model.to(device)
        ltcn_model.to(device)

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

    print(f"Evaluation completed! Results saved to {save_dir}")
    print(
        f"To view results, run 'mlflow ui' in the directory '{Path.cwd()}' and open http://localhost:5000"
    )
