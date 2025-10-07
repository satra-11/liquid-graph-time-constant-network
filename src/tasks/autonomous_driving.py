import torch
import numpy as np
import os
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from src.utils import add_whiteout

class DrivingDataset(Dataset):
    """HDDから自律走行データを読み込むためのデータセット"""

    def __init__(
        self,
        camera_dir: str,
        target_dir: str,
        sequence_length: int,
        frame_height: int = 224,
        frame_width: int = 224,
    ):
        self.sequence_length = sequence_length

        self.transform = transforms.Compose([
            transforms.Resize((frame_height, frame_width)),
            transforms.ToTensor(),
        ])

        self.sequences = []
        target_files = sorted(glob(os.path.join(target_dir, '*.npy')))

        for target_file in target_files:
            seq_name = os.path.basename(target_file).replace('.npy', '')
            camera_seq_dir = os.path.join(camera_dir, seq_name)

            if os.path.isdir(camera_seq_dir):
                image_files = sorted(glob(os.path.join(camera_seq_dir, '*.jpg'))) # Assuming .jpg, adjust if needed
                if len(image_files) >= self.sequence_length:
                    self.sequences.append({
                        'name': seq_name,
                        'images': image_files,
                        'target': target_file
                    })

    def __len__(self):
        # 各シーケンスから取り出せるサブシーケンスの総数を返す
        total = 0
        for seq in self.sequences:
            total += len(seq['images']) - self.sequence_length + 1
        return total

    def __getitem__(self, idx):
        # idxから、どのシーケンスのどの開始フレームかを特定する
        seq_idx = 0
        frame_offset = idx
        for i, seq in enumerate(self.sequences):
            num_sub_sequences = len(seq['images']) - self.sequence_length + 1
            if frame_offset < num_sub_sequences:
                seq_idx = i
                break
            frame_offset -= num_sub_sequences

        selected_sequence = self.sequences[seq_idx]
        start_frame = frame_offset

        # 画像シーケンスを読み込む
        image_paths = selected_sequence['images'][start_frame : start_frame + self.sequence_length]

        clean_frames = []
        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            clean_frames.append(self.transform(img))

        clean_frames_tensor = torch.stack(clean_frames) # (T, C, H, W)
        # (T, H, W, C) に変換
        clean_frames_tensor = clean_frames_tensor.permute(0, 2, 3, 1)

        # 汚損フレームを生成
        corrupted_frames_tensor = add_whiteout(frame=clean_frames_tensor.clone())

        # ターゲットを読み込む
        targets_full = np.load(selected_sequence['target'])
        targets = torch.from_numpy(targets_full[start_frame : start_frame + self.sequence_length]).float()

        return clean_frames_tensor, corrupted_frames_tensor, targets
