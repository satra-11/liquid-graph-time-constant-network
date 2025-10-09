from typing import Optional, Sequence, Tuple, List, Dict, Callable
import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class DrivingDataset(Dataset):
    """
    Camera(画像列) と Sensor(数値系列) を同一窓で返すデータセット

    出力:
      video: 画像テンソル [T, C, H, W]（to_channels_first=True なら [C, T, H, W]）
      sensor: センサ行列 [T, D]
      seq_name: シーケンス名
      start: 窓の開始インデックス（間引き後のインデックス）
    """

    def __init__(
        self,
        camera_dir: str,
        sensor_dir: str,
        sequence_length: int,
        sequences: Optional[Sequence[str]] = None,   # 明示列挙。Noneなら両方に存在するseqを自動探索
        stride: int = 1,
        frame_size: Tuple[int, int] = (224, 224),
        camera_transform: Optional[Callable] = None, # 追加transform（ToTensor/Normalize前に入れたい場合は下を上書き）
        imagenet_normalize: bool = True,
        center_crop: Optional[Tuple[int, int]] = None,
        sensor_normalize: bool = False,
        sensor_mean: Optional[np.ndarray] = None,    # shape=(D,)
        sensor_std: Optional[np.ndarray] = None,     # shape=(D,)
        sensor_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert sequence_length > 0 and stride > 0
        self.camera_dir = camera_dir
        self.sensor_dir = sensor_dir
        self.sequence_length = sequence_length
        self.stride = stride
        self.sensor_normalize = sensor_normalize
        self.sensor_mean = sensor_mean
        self.sensor_std = sensor_std
        self.sensor_dtype = sensor_dtype

        # ------------ 画像 transform ------------
        if camera_transform is None:
            tfms = [T.Resize(frame_size)]
            if center_crop is not None:
                tfms.append(T.CenterCrop(center_crop))
            tfms.append(T.ToTensor())
            if imagenet_normalize:
                mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                tfms.append(T.Normalize(mean, std))
            self.camera_tf = T.Compose(tfms)
        else:
            self.camera_tf = camera_transform

        # ------------ シーケンス探索 ------------
        if sequences is None:
            # 両方に存在する seq を集約
            sensor_candidates = {
                os.path.splitext(os.path.basename(p))[0]
                for p in glob(os.path.join(self.sensor_dir, "*.npy"))
            }
            cam_candidates = {
                os.path.basename(d)
                for d in glob(os.path.join(self.camera_dir, "*"))
                if os.path.isdir(d)
            }
            seq_list = sorted(sensor_candidates & cam_candidates)
        else:
            seq_list = list(sequences)

        self.sequences: List[Dict] = []
        for seq in seq_list:
            cam_dir = os.path.join(self.camera_dir, seq)
            sfile = os.path.join(self.sensor_dir, f"{seq}.npy")
            if not (os.path.isdir(cam_dir) and os.path.isfile(sfile)):
                continue

            frames = []
            frames.extend(glob(os.path.join(cam_dir, "*.jpg")))
            if not frames:
                continue
            frames = sorted(frames)

            # センサ読み込み（形だけ確認・長さ合わせは後段でも実施）
            try:
                s = np.load(sfile, allow_pickle=False, mmap_mode="r")
            except Exception:
                continue
            if s.ndim != 2 or s.shape[0] < sequence_length:
                continue
            
            L = min(len(frames), len(s))

            if L < sequence_length:
                continue

            n_subseq = 1 + (L - sequence_length) // self.stride
            if n_subseq <= 0:
                continue

            self.sequences.append({
                "name": seq,
                "frames_all": frames,   # 間引き済み
                "sensor_path": sfile,
                "L": L,
                "n_subseq": n_subseq,
            })

    def __len__(self):
        return sum(s["n_subseq"] for s in self.sequences)

    def _index_to_seq_offset(self, idx: int) -> Tuple[int, int]:
        off = idx
        for i, s in enumerate(self.sequences):
            if off < s["n_subseq"]:
                start = off * self.stride
                return i, start
            off -= s["n_subseq"]
        raise IndexError(idx)

    @staticmethod
    def _sanitize(x: np.ndarray) -> np.ndarray:
        return np.nan_to_num(x, copy=False, posinf=0.0, neginf=0.0)

    def _maybe_norm_sensor(self, x: np.ndarray) -> np.ndarray:
        if not self.sensor_normalize:
            return x
        if self.sensor_mean is None or self.sensor_std is None:
            mu = x.mean(axis=0, keepdims=True)
            sg = x.std(axis=0, keepdims=True) + 1e-8
            return (x - mu) / sg
        mu = self.sensor_mean.reshape(1, -1)
        sg = self.sensor_std.reshape(1, -1) + 1e-8
        return (x - mu) / sg

    def __getitem__(self, idx: int):
        si, start = self._index_to_seq_offset(idx)
        meta = self.sequences[si]
        end = start + self.sequence_length
        end = min(end, meta["L"])  # 念のため

        # ---- 画像読み込み ----
        imgs = []
        for p in meta["frames_all"][start:end]:
            with Image.open(p) as im:
                im = im.convert("RGB")
                imgs.append(self.camera_tf(im))
        video = torch.stack(imgs, dim=0)  # [T, C, H, W]

        s_mem = np.load(meta["sensor_path"], allow_pickle=False, mmap_mode="r")
        s_win = np.asarray(s_mem[start:end], dtype=np.float64).copy()  # [T, D]
        s_win = self._sanitize(s_win)
        s_win = self._maybe_norm_sensor(s_win)
        sensor = torch.as_tensor(s_win, dtype=self.sensor_dtype)  # [T, D]

        return video, sensor, meta["name"], start
