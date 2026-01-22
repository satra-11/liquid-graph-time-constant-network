import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import mlflow
import mlflow.pytorch


from src.core.models import (
    CfGCNController,
    LTCNController,
    NeuralODEController,
    NeuralGraphODEController,
)
from src.driving.data import setup_dataloaders
from src.driving.engine import train_model


def create_model(model_type: str, args: argparse.Namespace):
    """指定されたタイプのモデルを作成"""
    if model_type == "lgtcn":
        return CfGCNController(
            frame_height=64,
            frame_width=64,
            hidden_dim=args.hidden_dim,
            output_dim=6,
            K=args.K,
            num_layers=args.num_layers_cfgcn,
        )
    elif model_type == "ltcn":
        return LTCNController(
            frame_height=64,
            frame_width=64,
            output_dim=6,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers_ltcn,
        )
    elif model_type == "node":
        return NeuralODEController(
            frame_height=64,
            frame_width=64,
            hidden_dim=args.hidden_dim,
            output_dim=6,
        )
    elif model_type == "ngode":
        return NeuralGraphODEController(
            frame_height=64,
            frame_width=64,
            hidden_dim=args.hidden_dim,
            output_dim=6,
            K=args.K,
            num_layers=args.num_layers_cfgcn,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_optimizer_and_scheduler(model, model_type: str, args: argparse.Namespace):
    """モデルタイプに応じたオプティマイザとスケジューラを作成"""
    if model_type == "lgtcn":
        # LGTCNはパラメータグループごとに異なる学習率を設定
        param_groups = [
            {
                "params": model.feature_extractor.parameters(),
                "lr": args.lr * 0.1,
            },
            {
                "params": model.node_encoder.parameters(),
                "lr": args.lr * 0.1,
            },
            {
                "params": model.temporal_processor.parameters(),
                "lr": args.lr,
            },
            {
                "params": model.control_decoder.parameters(),
                "lr": args.lr,
            },
            {
                "params": [model.output_scale, model.output_bias],
                "lr": args.lr,
            },
        ]
        optimizer = optim.Adam(param_groups, weight_decay=1e-4)
    elif model_type == "ngode":
        # NGODEもLGTCN同様のパラメータグループを設定
        param_groups = [
            {
                "params": model.feature_extractor.parameters(),
                "lr": args.lr * 0.1,
            },
            {
                "params": model.node_encoder.parameters(),
                "lr": args.lr * 0.1,
            },
            {
                "params": model.temporal_processor.parameters(),
                "lr": args.lr,
            },
            {
                "params": model.control_decoder.parameters(),
                "lr": args.lr,
            },
            {
                "params": [model.output_scale, model.output_bias],
                "lr": args.lr,
            },
        ]
        optimizer = optim.Adam(param_groups, weight_decay=1e-4)
    else:
        # LTCN, NODE は標準の設定
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    return optimizer, scheduler


def get_training_config(model_type: str) -> dict:
    """モデルタイプに応じた訓練設定を取得"""
    if model_type in ["lgtcn", "ngode"]:
        return {
            "use_full_sequence_loss": True,
            "gradient_clip_norm": 5.0,
        }
    else:
        # ltcn, node も全シーケンスでLossを計算（評価時と統一）
        return {
            "use_full_sequence_loss": True,
            "gradient_clip_norm": 1.0,
        }


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

    model_type = args.model
    model_name = model_type.upper()

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
        print(f"Creating {model_name} model...")
        model = create_model(model_type, args)

        # オプティマイザとスケジューラを作成
        optimizer, scheduler = create_optimizer_and_scheduler(model, model_type, args)

        start_epoch = 0

        # チェックポイントからの再開
        if args.resume_from_checkpoint:
            checkpoint_path = (
                Path(args.resume_from_checkpoint) / f"{model_name}_checkpoint.pth"
            )
            if checkpoint_path.exists():
                print(f"Resuming {model_name} training from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"]

        # 訓練設定を取得
        training_config = get_training_config(model_type)

        # モデル訓練
        print(f"Training {model_name}...")
        train_model(
            model,
            model_name,
            train_loader,
            val_loader,
            optimizer=optimizer,
            save_dir=save_dir,
            num_epochs=args.epochs,
            start_epoch=start_epoch,
            device=device,
            scheduler=scheduler,
            **training_config,
        )

        # モデル保存 (MLflow)
        print("Logging model to MLflow...")

        # サンプル入力を作成してモデルシグネチャを自動推論
        sample_batch = next(iter(train_loader))
        sample_input = sample_batch[0][:1].to(device)  # (1, seq_len, C, H, W)

        # モデルをCPUに移動してからログ
        model.to("cpu")
        sample_input_cpu = sample_input.cpu()

        # 入力例をfloat32にキャスト (numpyのデフォルトがdoubleになるのを防ぐ)
        input_example = sample_input_cpu.numpy().astype(np.float32)

        # モデルの出力を取得してシグネチャを作成
        with torch.no_grad():
            output = model(sample_input_cpu)
            # Tuple出力の場合は最初の要素（制御信号）を使用
            if isinstance(output, tuple):
                output = output[0]
            output_example = output.numpy()

        signature = mlflow.models.infer_signature(input_example, output_example)

        mlflow.pytorch.log_model(
            model,
            artifact_path=f"{model_type}_model",
            input_example=input_example,
            signature=signature,
        )

    print(f"Training completed! Results saved to {save_dir}")
    print(
        f"To view results, run 'mlflow ui' in the directory '{Path.cwd()}' and open http://localhost:5000"
    )
