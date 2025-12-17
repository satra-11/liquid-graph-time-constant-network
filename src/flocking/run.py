"""Training script for flocking models.

Based on arXiv:2404.13982 experiment conditions.
"""

import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import mlflow
import mlflow.pytorch

from .models import FlockingLGTCN, FlockingLTCN
from .data import setup_flocking_dataloaders
from .engine import train_flocking_model, evaluate_flocking_model


def run_flocking_training(args: argparse.Namespace) -> None:
    """Run flocking training pipeline."""
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Seed setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_params(vars(args))

        # Setup dataloaders
        print("Collecting training trajectories...")
        train_loader, val_loader, test_loader, dataset = setup_flocking_dataloaders(
            num_trajectories=args.num_trajectories,
            batch_size=args.batch_size,
            train_ratio=0.7,
            val_ratio=0.1,
            trajectory_length=args.trajectory_length,
            dt=args.dt,
            agent_counts=args.agent_counts,
            comm_range=args.comm_range,
            collision_range=args.collision_range,
            max_accel=args.max_accel,
            device=str(device),
        )
        print(f"Dataset size: {len(dataset)} trajectories")

        # Create models
        print("Creating models...")
        lgtcn_model = FlockingLGTCN(
            input_dim=10,
            hidden_dim=args.hidden_dim,
            output_dim=2,
            K=args.K,
            fc_dim=128,
            num_layers=args.num_layers,
        )

        ltcn_model = FlockingLTCN(
            input_dim=10,
            hidden_dim=args.hidden_dim,
            output_dim=2,
            fc_dim=128,
            num_blocks=args.num_blocks,
        )

        # Create optimizers
        lgtcn_optimizer = optim.Adam(
            lgtcn_model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
        )
        ltcn_optimizer = optim.Adam(
            ltcn_model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
        )

        # Train LGTCN
        print("\n" + "=" * 50)
        print("Training FlockingLGTCN...")
        print("=" * 50)
        train_flocking_model(
            model=lgtcn_model,
            model_name="FlockingLGTCN",
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=lgtcn_optimizer,
            save_dir=save_dir,
            dataset=dataset,
            num_epochs=args.epochs,
            dagger_interval=args.dagger_interval,
            dagger_trajectories=args.dagger_trajectories,
            device=device,
            gradient_clip_norm=args.gradient_clip,
            max_accel=args.max_accel,
        )

        # Train LTCN
        print("\n" + "=" * 50)
        print("Training FlockingLTCN...")
        print("=" * 50)
        train_flocking_model(
            model=ltcn_model,
            model_name="FlockingLTCN",
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=ltcn_optimizer,
            save_dir=save_dir,
            dataset=dataset,
            num_epochs=args.epochs,
            dagger_interval=args.dagger_interval,
            dagger_trajectories=args.dagger_trajectories,
            device=device,
            gradient_clip_norm=args.gradient_clip,
            max_accel=args.max_accel,
        )

        # Evaluate both models
        print("\n" + "=" * 50)
        print("Evaluating models...")
        print("=" * 50)

        lgtcn_results = evaluate_flocking_model(
            lgtcn_model, test_loader, device, args.max_accel
        )
        ltcn_results = evaluate_flocking_model(
            ltcn_model, test_loader, device, args.max_accel
        )

        print(
            f"FlockingLGTCN - MSE: {lgtcn_results['mse']:.6f}, "
            f"RMSE: {lgtcn_results['rmse']:.6f}"
        )
        print(
            f"FlockingLTCN  - MSE: {ltcn_results['mse']:.6f}, "
            f"RMSE: {ltcn_results['rmse']:.6f}"
        )

        mlflow.log_metrics(
            {
                "FlockingLGTCN_test_mse": lgtcn_results["mse"],
                "FlockingLGTCN_test_rmse": lgtcn_results["rmse"],
                "FlockingLTCN_test_mse": ltcn_results["mse"],
                "FlockingLTCN_test_rmse": ltcn_results["rmse"],
            }
        )

        # Log models to MLflow
        print("Logging models to MLflow...")
        lgtcn_model.to("cpu")
        ltcn_model.to("cpu")

        mlflow.pytorch.log_model(lgtcn_model, "flocking_lgtcn_model")
        mlflow.pytorch.log_model(ltcn_model, "flocking_ltcn_model")

    print(f"\nTraining completed! Results saved to {save_dir}")
    print(
        f"To view results, run 'mlflow ui' in '{Path.cwd()}' "
        "and open http://localhost:5000"
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train flocking models (LGTCN and LTCN)"
    )

    # Data arguments
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=60,
        help="Number of training trajectories",
    )
    parser.add_argument(
        "--trajectory-length",
        type=float,
        default=2.5,
        help="Trajectory duration in seconds",
    )
    parser.add_argument(
        "--dt", type=float, default=0.05, help="Sampling time in seconds"
    )
    parser.add_argument(
        "--agent-counts",
        type=int,
        nargs="+",
        default=[4, 6, 10, 12, 15],
        help="Possible agent counts",
    )
    parser.add_argument(
        "--comm-range", type=float, default=4.0, help="Communication range R in meters"
    )
    parser.add_argument(
        "--collision-range",
        type=float,
        default=1.0,
        help="Collision avoidance range R_CA in meters",
    )
    parser.add_argument(
        "--max-accel", type=float, default=5.0, help="Maximum acceleration in m/s^2"
    )

    # Model arguments
    parser.add_argument(
        "--hidden-dim", type=int, default=48, help="Hidden state dimension F"
    )
    parser.add_argument(
        "--K", type=int, default=2, help="Filter length for graph convolution"
    )
    parser.add_argument(
        "--num-layers", type=int, default=1, help="Number of CfGCN layers"
    )
    parser.add_argument(
        "--num-blocks", type=int, default=4, help="Number of blocks in LTCN layer"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--gradient-clip", type=float, default=1.0, help="Gradient clipping norm"
    )
    parser.add_argument(
        "--dagger-interval", type=int, default=20, help="Run DAGGER every N epochs"
    )
    parser.add_argument(
        "--dagger-trajectories",
        type=int,
        default=10,
        help="Number of trajectories to add per DAGGER",
    )

    # Other arguments
    parser.add_argument(
        "--save-dir",
        type=str,
        default="flocking_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto, cpu, cuda)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_flocking_training(args)
