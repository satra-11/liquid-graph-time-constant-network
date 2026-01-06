import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import mlflow
import mlflow.pytorch


from src.core.models import CfGCNController, LTCNController
from src.driving.data import setup_dataloaders
from src.driving.engine import train_model


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

        # オプティマイザの作成
        # LGTCNはパラメータグループごとに異なる学習率を設定
        # 勾配が大きい層（feature_extractor, node_encoder）には低い学習率
        lgtcn_param_groups = [
            {
                "params": lgtcn_model.feature_extractor.parameters(),
                "lr": args.lr * 0.1,  # 低い学習率
            },
            {
                "params": lgtcn_model.node_encoder.parameters(),
                "lr": args.lr * 0.1,  # 低い学習率
            },
            {
                "params": lgtcn_model.temporal_processor.parameters(),
                "lr": args.lr,
            },
            {
                "params": lgtcn_model.control_decoder.parameters(),
                "lr": args.lr,
            },
            {
                "params": [lgtcn_model.output_scale, lgtcn_model.output_bias],
                "lr": args.lr,
            },
        ]
        lgtcn_optimizer = optim.Adam(lgtcn_param_groups, weight_decay=1e-4)
        # LGTCNにCosineAnnealingLRスケジューラを追加
        lgtcn_scheduler = CosineAnnealingLR(
            lgtcn_optimizer, T_max=args.epochs, eta_min=1e-6
        )

        ltcn_optimizer = optim.Adam(ltcn_model.parameters(), lr=args.lr)
        # LTCNにもCosineAnnealingLRスケジューラを追加
        ltcn_scheduler = CosineAnnealingLR(
            ltcn_optimizer, T_max=args.epochs, eta_min=1e-6
        )

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

        # LGTCN訓練（全ての改善策を適用）
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
            scheduler=lgtcn_scheduler,
            use_full_sequence_loss=True,  # 全シーケンスに対してLossを計算
            gradient_clip_norm=5.0,  # 勾配クリッピングを緩和
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
            scheduler=ltcn_scheduler,
        )

        # モデル保存 (MLflow)
        print("Logging models to MLflow...")

        # サンプル入力を作成してモデルシグネチャを自動推論
        sample_batch = next(iter(train_loader))
        sample_input = sample_batch[0][:1].to(device)  # (1, seq_len, C, H, W)

        # モデルをCPUに移動してからログ
        lgtcn_model.to("cpu")
        ltcn_model.to("cpu")
        sample_input_cpu = sample_input.cpu()

        mlflow.pytorch.log_model(
            lgtcn_model, "lgtcn_model", input_example=sample_input_cpu.numpy()
        )
        mlflow.pytorch.log_model(
            ltcn_model, "ltcn_model", input_example=sample_input_cpu.numpy()
        )

    print(f"Training completed! Results saved to {save_dir}")
    print(
        f"To view results, run 'mlflow ui' in the directory '{Path.cwd()}' and open http://localhost:5000"
    )
