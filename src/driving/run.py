import argparse
import random
from pathlib import Path
import json
import numpy as np
import torch
import torch.optim as optim
import mlflow
import mlflow.pytorch

from src.tasks import NetworkComparator
from src.core.models import CfGCNController, LTCNController
from src.driving.data import setup_dataloaders
from src.driving.engine import train_model, evaluate_networks


def run_training(args: argparse.Namespace):
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
        train_loader, val_loader, test_loader, _ = setup_dataloaders(
            data_dir=args.data_dir,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
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

        # オプティマイザの作成
        lgtcn_optimizer = optim.Adam(lgtcn_model.parameters(), lr=args.lr)
        ltcn_optimizer = optim.Adam(ltcn_model.parameters(), lr=args.lr)

        start_epoch_lgtcn = 0
        start_epoch_ltcn = 0

        # チェックポイントからの再開
        if args.resume_from_checkpoint:
            lgtcn_checkpoint_path = (
                Path(args.resume_from_checkpoint) / "LGTCN_checkpoint.pth"
            )
            if lgtcn_checkpoint_path.exists():
                print(f"Resuming LGTCN training from {lgtcn_checkpoint_path}")
                checkpoint = torch.load(lgtcn_checkpoint_path)
                lgtcn_model.load_state_dict(checkpoint["model_state_dict"])
                lgtcn_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch_lgtcn = checkpoint["epoch"]

            ltcn_checkpoint_path = (
                Path(args.resume_from_checkpoint) / "LTCN_checkpoint.pth"
            )
            if ltcn_checkpoint_path.exists():
                print(f"Resuming LTCN training from {ltcn_checkpoint_path}")
                checkpoint = torch.load(ltcn_checkpoint_path)
                ltcn_model.load_state_dict(checkpoint["model_state_dict"])
                ltcn_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch_ltcn = checkpoint["epoch"]

        # LGTCN訓練
        print("Training LGTCN...")
        train_model(
            lgtcn_model,
            "LGTCN",
            train_loader,
            val_loader,
            optimizer=lgtcn_optimizer,
            save_dir=save_dir,
            num_epochs=args.epochs,
            start_epoch=start_epoch_lgtcn,
            device=device,
        )

        # LTCN訓練
        print("Training LTCN...")
        train_model(
            ltcn_model,
            "LTCN",
            train_loader,
            val_loader,
            optimizer=ltcn_optimizer,
            save_dir=save_dir,
            num_epochs=args.epochs,
            start_epoch=start_epoch_ltcn,
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
