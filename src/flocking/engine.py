"""Training engine for flocking models."""

import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

from .models import FlockingLGTCN, FlockingLTCN
from .data import FlockingDataset


def train_flocking_model(
    model: FlockingLGTCN | FlockingLTCN,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    save_dir: Path,
    dataset: FlockingDataset,
    num_epochs: int = 120,
    dagger_interval: int = 20,
    dagger_trajectories: int = 10,
    start_epoch: int = 0,
    device: torch.device | None = None,
    gradient_clip_norm: float = 1.0,
    max_accel: float = 5.0,
) -> None:
    """Train flocking model with DAGGER algorithm.

    Args:
        model: Model to train (FlockingLGTCN or FlockingLTCN)
        model_name: Name for logging
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        save_dir: Directory to save checkpoints
        dataset: Dataset for DAGGER augmentation
        num_epochs: Total number of epochs
        dagger_interval: Run DAGGER every N epochs
        dagger_trajectories: Number of trajectories to add per DAGGER
        start_epoch: Starting epoch (for resume)
        device: Torch device
        gradient_clip_norm: Max norm for gradient clipping
        max_accel: Maximum acceleration for action clamping
    """
    device = device or torch.device("cpu")
    model = model.to(device)

    criterion = nn.MSELoss()

    start_time = time.time()
    best_val_loss = float("inf")

    for epoch in range(start_epoch, num_epochs):
        # DAGGER: Add expert-labeled trajectories from model rollouts
        if epoch > 0 and epoch % dagger_interval == 0:
            print(f"Running DAGGER at epoch {epoch}...")
            model.eval()
            dataset.add_dagger_trajectories(
                model=model,
                env_config={
                    "comm_range": 4.0,
                    "collision_range": 1.0,
                    "max_accel": max_accel,
                },
                num_trajectories=dagger_trajectories,
            )
            print(f"Dataset size: {len(dataset)}")

        # Training phase
        model.train()
        epoch_train_loss = 0.0

        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN - {model_name}]"
        ):
            obs, adj, expert_actions, agent_counts = batch
            obs = obs.to(device)
            adj = adj.to(device)
            expert_actions = expert_actions.to(device)

            optimizer.zero_grad()

            # Forward pass
            pred_actions, _ = model(obs, adj)

            # Compute loss with masking for variable agent counts
            loss = compute_masked_loss(
                pred_actions, expert_actions, agent_counts, criterion
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clip_norm
            )
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        mlflow.log_metric(f"{model_name} Train Loss", avg_train_loss, step=epoch)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [VAL - {model_name}]"
            ):
                obs, adj, expert_actions, agent_counts = batch
                obs = obs.to(device)
                adj = adj.to(device)
                expert_actions = expert_actions.to(device)

                pred_actions, _ = model(obs, adj)

                loss = compute_masked_loss(
                    pred_actions, expert_actions, agent_counts, criterion
                )
                epoch_val_loss += loss.item()

        if len(val_loader) > 0:
            avg_val_loss = epoch_val_loss / len(val_loader)
        else:
            avg_val_loss = 0.0
        mlflow.log_metric(f"{model_name} Val Loss", avg_val_loss, step=epoch)

        # Log learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        mlflow.log_metric(f"{model_name} LR", current_lr, step=epoch)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}: Train Loss = {avg_train_loss:.6f}, "
                f"Val Loss = {avg_val_loss:.6f}"
            )

        # Save checkpoint
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / f"{model_name}_checkpoint.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
            },
            checkpoint_path,
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = save_dir / f"{model_name}_best.pth"
            torch.save(model.state_dict(), best_path)

    end_time = time.time()
    elapsed = end_time - start_time
    print(
        f"Training for {model_name} finished in "
        f"{time.strftime('%H:%M:%S', time.gmtime(elapsed))}."
    )


def compute_masked_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    agent_counts: list[int],
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute loss with masking for variable agent counts.

    Args:
        predictions: (B, T, N_max, output_dim)
        targets: (B, T, N_max, output_dim)
        agent_counts: List of actual agent counts per batch item
        criterion: Loss function

    Returns:
        Masked loss value
    """
    B, T, N_max, output_dim = predictions.shape

    # Create mask
    mask = torch.zeros(B, T, N_max, device=predictions.device)
    for i, n in enumerate(agent_counts):
        mask[i, :, :n] = 1.0

    # Expand mask for output_dim
    mask = mask.unsqueeze(-1).expand_as(predictions)

    # Apply mask
    masked_pred = predictions * mask
    masked_target = targets * mask

    # Compute loss only on valid entries
    num_valid = mask.sum()
    if num_valid > 0:
        loss = ((masked_pred - masked_target) ** 2).sum() / num_valid
    else:
        loss = torch.tensor(0.0, device=predictions.device)

    return loss


def evaluate_flocking_model(
    model: FlockingLGTCN | FlockingLTCN,
    test_loader: DataLoader,
    device: torch.device,
    max_accel: float = 5.0,
) -> dict:
    """Evaluate flocking model on test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Torch device
        max_accel: Maximum acceleration

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model = model.to(device)

    total_mse = 0.0
    total_samples = 0

    criterion = nn.MSELoss(reduction="sum")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            obs, adj, expert_actions, agent_counts = batch
            obs = obs.to(device)
            adj = adj.to(device)
            expert_actions = expert_actions.to(device)

            pred_actions, _ = model(obs, adj)

            # Clamp predictions
            pred_actions = torch.clamp(pred_actions, -max_accel, max_accel)

            # Compute MSE with masking
            for i, n in enumerate(agent_counts):
                B, T = pred_actions.shape[:2]
                pred_i = pred_actions[i, :, :n, :]
                target_i = expert_actions[i, :, :n, :]
                total_mse += criterion(pred_i, target_i).item()
                total_samples += T * n * 2  # 2 for 2D

    avg_mse = total_mse / total_samples if total_samples > 0 else 0.0

    return {
        "mse": avg_mse,
        "rmse": avg_mse**0.5,
    }
