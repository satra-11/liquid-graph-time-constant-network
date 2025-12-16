from typing import Optional, Sequence, Tuple, List, Dict, Callable
import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class HDDLoader(Dataset):
    """
    Camera(画像列) と Sensor(数値系列) を同一窓で返すデータセット

    出力:
      video: 画像テンソル [T, C, H, W]（to_channels_first=True なら [C, T, H, W]）
      sensor: センサ行列 [T, D]
      seq_name: シーケンス名
      start: 窓の開始インデックス（間引き後のインデックス）
    """

    FEATURE_KEYS = [
        "accel_pedal_info",
        "brake_pedal_info",
        "steer_info",
        "vel_info",
        "yaw_info",
        "turn_signal_info",
        "rtk_pos_info",
        "rtk_track_info",
    ]

    def __init__(
        self,
        camera_dir: str,
        sensor_dir: str,
        sequence_length: int,
        sequences: Optional[
            Sequence[str]
        ] = None,  # 明示列挙。Noneなら両方に存在するseqを自動探索
        stride: int = 1,
        frame_size: Tuple[int, int] = (224, 224),
        camera_transform: Optional[
            Callable
        ] = None,  # 追加transform（ToTensor/Normalize前に入れたい場合は下を上書き）
        imagenet_normalize: bool = True,
        center_crop: Optional[Tuple[int, int]] = None,
        sensor_normalize: bool = False,
        sensor_mean: Optional[np.ndarray] = None,  # shape=(D,)
        sensor_std: Optional[np.ndarray] = None,  # shape=(D,)
        sensor_dtype: torch.dtype = torch.float32,
        exclude_features: Optional[List[str]] = None,  # 除外するセンサー特徴量のリスト
        processed_dir: Optional[str] = None,  # 追加: 処理済み特徴量のディレクトリ
    ):
        super().__init__()
        assert sequence_length > 0 and stride > 0
        self.camera_dir = camera_dir
        self.sensor_dir = sensor_dir
        self.processed_dir = processed_dir
        self.sequence_length = sequence_length
        self.stride = stride
        self.sensor_normalize = sensor_normalize
        self.sensor_mean = sensor_mean
        self.sensor_std = sensor_std
        self.sensor_dtype = sensor_dtype
        self.exclude_features = exclude_features or []

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
            if self.processed_dir:
                cam_candidates = {
                    os.path.splitext(os.path.basename(p))[0]
                    for p in glob(os.path.join(self.processed_dir, "*.npy"))
                }
            else:
                cam_candidates = {
                    os.path.basename(d)
                    for d in glob(os.path.join(self.camera_dir, "*"))
                    if os.path.isdir(d)
                }
            seq_list = sorted(sensor_candidates & cam_candidates)

            if not seq_list:
                raise RuntimeError(
                    f"No common sequences found!\n"
                    f"  Sensor dir: {self.sensor_dir} (Found {len(sensor_candidates)} files)\n"
                    f"  Camera/Processed dir: {self.processed_dir or self.camera_dir} (Found {len(cam_candidates)} items)\n"
                    f"  Please check your data paths and ensure sequence names match."
                )
        else:
            seq_list = list(sequences)

        self.sequences: List[Dict] = []
        for seq in seq_list:
            sfile = os.path.join(self.sensor_dir, f"{seq}.npy")

            if self.processed_dir:
                pfile = os.path.join(self.processed_dir, f"{seq}.npy")
                if not (os.path.isfile(pfile) and os.path.isfile(sfile)):
                    continue
                # 特徴量ファイルの長さを取得
                try:
                    # mmap_mode="r" でヘッダだけ読む
                    feat = np.load(pfile, allow_pickle=False, mmap_mode="r")
                    L_frames = feat.shape[0]
                except Exception:
                    continue
            else:
                cam_dir = os.path.join(self.camera_dir, seq)
                if not (os.path.isdir(cam_dir) and os.path.isfile(sfile)):
                    continue
                frames = glob(os.path.join(cam_dir, "*.jpg"))
                if not frames:
                    continue
                frames = sorted(frames)
                L_frames = len(frames)
                pfile = None

            # センサ読み込み（形だけ確認・長さ合わせは後段でも実施）
            try:
                s = np.load(sfile, allow_pickle=False, mmap_mode="r")
            except Exception:
                continue
            if s.ndim != 2 or s.shape[0] < sequence_length:
                continue

            L = min(L_frames, len(s))

            if L < sequence_length:
                continue

            n_subseq = 1 + (L - sequence_length) // self.stride
            if n_subseq <= 0:
                continue

            seq_info = {
                "name": seq,
                "sensor_path": sfile,
                "L": L,
                "n_subseq": n_subseq,
            }
            if self.processed_dir:
                seq_info["processed_path"] = pfile
            else:
                seq_info["frames_all"] = frames

            self.sequences.append(seq_info)

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

        # ---- 画像/特徴量読み込み ----
        if "processed_path" in meta:
            # 特徴量をロード
            p_mem = np.load(meta["processed_path"], allow_pickle=False, mmap_mode="r")
            # [T, 128, 8, 8]
            feat_win = np.asarray(p_mem[start:end], dtype=np.float32).copy()
            video = torch.from_numpy(feat_win)
        else:
            # 画像読み込み
            imgs = []
            for p in meta["frames_all"][start:end]:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    imgs.append(self.camera_tf(im))
            video = torch.stack(imgs, dim=0)  # [T, C, H, W]

        s_mem = np.load(meta["sensor_path"], allow_pickle=False, mmap_mode="r")
        s_win = np.asarray(s_mem[start:end], dtype=np.float64).copy()  # [T, D]

        # 特徴量を除外
        if self.exclude_features:
            exclude_indices = [
                self.FEATURE_KEYS.index(feat)
                for feat in self.exclude_features
                if feat in self.FEATURE_KEYS
            ]
            if exclude_indices:
                s_win = np.delete(s_win, exclude_indices, axis=1)

        s_win = self._sanitize(s_win)
        s_win = self._maybe_norm_sensor(s_win)
        sensor = torch.as_tensor(s_win, dtype=self.sensor_dtype)  # [T, D]

        return video, sensor, meta["name"], start
