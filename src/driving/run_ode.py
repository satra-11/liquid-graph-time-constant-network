import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import mlflow
import mlflow.pytorch


from src.core.models import NeuralODEController, NeuralGraphODEController
from src.driving.data import setup_dataloaders
from src.driving.engine import train_model


def run_training(args: argparse.Namespace):
    """Neural ODE / Neural Graph ODE のトレーニングを実行."""
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
        processed_dir = args.processed_dir
        if processed_dir and not os.path.exists(processed_dir):
            print(
                f"WARNING: Processed directory {processed_dir} not found. Falling back to raw images."
            )
            processed_dir = None

        train_loader, val_loader, test_loader, _ = setup_dataloaders(
            data_dir=args.data_dir,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            processed_dir=processed_dir,
        )

        # モデル作成
        print("Creating Neural ODE models...")

        # Neural Graph ODE Controller
        ngode_model = NeuralGraphODEController(
            frame_height=64,
            frame_width=64,
            hidden_dim=args.hidden_dim,
            output_dim=6,
            K=args.K,
            num_layers=args.num_layers,
            solver=args.solver,
        )

        # Neural ODE Controller
        node_model = NeuralODEController(
            frame_height=64,
            frame_width=64,
            output_dim=6,
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_hidden_layers,
            solver=args.solver,
        )

        # オプティマイザの作成
        # Neural Graph ODE: パラメータグループごとに異なる学習率を設定
        ngode_param_groups = [
            {
                "params": ngode_model.feature_extractor.parameters(),
                "lr": args.lr * 0.1,
            },
            {
                "params": ngode_model.node_encoder.parameters(),
                "lr": args.lr * 0.1,
            },
            {
                "params": ngode_model.temporal_processor.parameters(),
                "lr": args.lr,
            },
            {
                "params": ngode_model.control_decoder.parameters(),
                "lr": args.lr,
            },
            {
                "params": [ngode_model.output_scale, ngode_model.output_bias],
                "lr": args.lr,
            },
        ]
        ngode_optimizer = optim.Adam(ngode_param_groups, weight_decay=1e-4)
        ngode_scheduler = CosineAnnealingLR(
            ngode_optimizer, T_max=args.epochs, eta_min=1e-6
        )

        node_optimizer = optim.Adam(node_model.parameters(), lr=args.lr)

        start_epoch_ngode = 0
        start_epoch_node = 0

        # チェックポイントからの再開
        if args.resume_from_checkpoint:
            ngode_checkpoint_path = (
                Path(args.resume_from_checkpoint) / "NGODE_checkpoint.pth"
            )
            if ngode_checkpoint_path.exists():
                print(
                    f"Resuming Neural Graph ODE training from {ngode_checkpoint_path}"
                )
                checkpoint = torch.load(ngode_checkpoint_path)
                ngode_model.load_state_dict(checkpoint["model_state_dict"])
                ngode_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch_ngode = checkpoint["epoch"]

            node_checkpoint_path = (
                Path(args.resume_from_checkpoint) / "NODE_checkpoint.pth"
            )
            if node_checkpoint_path.exists():
                print(f"Resuming Neural ODE training from {node_checkpoint_path}")
                checkpoint = torch.load(node_checkpoint_path)
                node_model.load_state_dict(checkpoint["model_state_dict"])
                node_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch_node = checkpoint["epoch"]

        # Neural Graph ODE 訓練
        print("Training Neural Graph ODE...")
        train_model(
            ngode_model,
            "NGODE",
            train_loader,
            val_loader,
            optimizer=ngode_optimizer,
            save_dir=save_dir,
            num_epochs=args.epochs,
            start_epoch=start_epoch_ngode,
            device=device,
            scheduler=ngode_scheduler,
            use_full_sequence_loss=True,
            gradient_clip_norm=5.0,
        )

        # Neural ODE 訓練
        print("Training Neural ODE...")
        train_model(
            node_model,
            "NODE",
            train_loader,
            val_loader,
            optimizer=node_optimizer,
            save_dir=save_dir,
            num_epochs=args.epochs,
            start_epoch=start_epoch_node,
            device=device,
        )

        # モデル保存 (MLflow)
        print("Logging models to MLflow...")

        # サンプル入力を作成してモデルシグネチャを自動推論
        sample_batch = next(iter(train_loader))
        sample_input = sample_batch[0][:1].to(device)

        # モデルをCPUに移動してからログ
        ngode_model.to("cpu")
        node_model.to("cpu")
        sample_input_cpu = sample_input.cpu()

        mlflow.pytorch.log_model(
            ngode_model, "ngode_model", input_example=sample_input_cpu.numpy()
        )
        mlflow.pytorch.log_model(
            node_model, "node_model", input_example=sample_input_cpu.numpy()
        )

    print(f"Training completed! Results saved to {save_dir}")
    print(
        f"To view results, run 'mlflow ui' in the directory '{Path.cwd()}' and open http://localhost:5000"
    )


def main():
    parser = argparse.ArgumentParser(description="Neural ODE Training")

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/data/raw",
        help="Path to driving dataset",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="/data/processed",
        help="Path to processed features",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./driving_results_ode",
        help="Directory to save results",
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--sequence-length", type=int, default=20, help="Sequence length"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model arguments
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--K", type=int, default=2, help="Graph filter order")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of layers for Neural Graph ODE",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=2,
        help="MLP hidden layers for Neural ODE",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="dopri5",
        choices=["dopri5", "euler", "rk4", "midpoint"],
        help="ODE solver",
    )

    # Other arguments
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto/cpu/cuda)"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory",
    )

    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
