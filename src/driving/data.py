import os
from torch.utils.data import DataLoader, random_split, Dataset
from src.data import HDDLoader


def setup_dataloaders(
    data_dir: str,
    sequence_length: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, Dataset]:
    """データセットを準備し、データローダーを作成する"""
    print("Loading dataset from HDD...")

    full_dataset = HDDLoader(
        camera_dir=os.path.join(data_dir, "camera"),
        sensor_dir=os.path.join(data_dir, "sensor"),
        sequence_length=sequence_length,
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
        train_dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=False,
        shuffle=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=False,
        shuffle=False,
        persistent_workers=True,
        prefetch_factor=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=False,
        shuffle=False,
        persistent_workers=True,
        prefetch_factor=4,
    )

    return train_loader, val_loader, test_loader, full_dataset
