import time
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

from src.core.models import (
    CfGCNController,
    LTCNController,
    NeuralODEController,
    NeuralGraphODEController,
)
from src.tasks import NetworkComparator


def train_model(
    model: LTCNController
    | CfGCNController
    | NeuralODEController
    | NeuralGraphODEController,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    save_dir: Path,
    num_epochs: int = 100,
    start_epoch: int = 0,
    device: torch.device = None,
    scheduler: optim.lr_scheduler.LRScheduler = None,
    use_full_sequence_loss: bool = False,
    gradient_clip_norm: float = 1.0,
):
    """モデルを訓練"""
    device = device or torch.device("cpu")
    model = model.to(device)

    criterion = nn.MSELoss()

    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
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

            # Lossを計算
            if use_full_sequence_loss:
                loss = criterion(predictions, sensors)
            else:
                loss = criterion(predictions[:, -1, :], sensors[:, -1, :])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clip_norm
            )
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        mlflow.log_metric(f"{model_name} Train Loss", avg_train_loss, step=epoch)

        # 学習率をログ
        current_lr = optimizer.param_groups[0]["lr"]
        mlflow.log_metric(f"{model_name} LR", current_lr, step=epoch)

        # スケジューラのステップ
        if scheduler is not None:
            scheduler.step()

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

                # Lossを計算
                if use_full_sequence_loss:
                    loss = criterion(predictions, sensors)
                else:
                    loss = criterion(predictions[:, -1, :], sensors[:, -1, :])
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        mlflow.log_metric(f"{model_name} Val Loss", avg_val_loss, step=epoch)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}: Train Loss ({model_name}) = {avg_train_loss:.6f}, Val Loss ({model_name}) = {avg_val_loss:.6f}"
            )

        # チェックポイントの保存
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / f"{model_name}_checkpoint.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Training for {model_name} finished in {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}."
    )


def evaluate_model(
    model: LTCNController
    | CfGCNController
    | NeuralODEController
    | NeuralGraphODEController,
    model_name: str,
    test_data: dict,
    device: torch.device,
    corruption_levels: list[float] | None = None,
):
    """単一モデルを評価"""
    if corruption_levels is None:
        corruption_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    model.eval()
    model.to(device)

    clean_frames = test_data["clean_frames"].to(device)
    sensors = test_data["sensors"].to(device)

    results: dict[str, Any] = {
        "model_name": model_name,
        "corruption_levels": corruption_levels,
        "results_by_corruption": {},
        "metrics": {},
    }

    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()

    total_mse = 0.0
    total_mae = 0.0

    with torch.no_grad():
        for corruption_level in corruption_levels:
            # ノイズを追加
            if corruption_level > 0:
                noise = torch.randn_like(clean_frames) * corruption_level
                corrupted_frames = clean_frames + noise
                corrupted_frames = torch.clamp(corrupted_frames, 0, 1)
            else:
                corrupted_frames = clean_frames

            # 予測
            predictions, _ = model(corrupted_frames)

            # メトリクス計算
            mse = mse_criterion(predictions, sensors).item()
            mae = mae_criterion(predictions, sensors).item()

            results["results_by_corruption"][str(corruption_level)] = {
                "mse": mse,
                "mae": mae,
            }

            total_mse += mse
            total_mae += mae

            print(f"  Corruption {corruption_level:.1f}: MSE={mse:.6f}, MAE={mae:.6f}")

    # 平均メトリクスを記録
    num_levels = len(corruption_levels)
    results["metrics"]["avg_mse"] = total_mse / num_levels
    results["metrics"]["avg_mae"] = total_mae / num_levels

    print(
        f"\n{model_name.upper()} Average: MSE={results['metrics']['avg_mse']:.6f}, MAE={results['metrics']['avg_mae']:.6f}"
    )

    return results


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
