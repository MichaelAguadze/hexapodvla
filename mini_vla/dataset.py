"""Dataset for Mini VLA: loads video frames + task labels + actions."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

try:
    import av
except ImportError as exc:
    raise RuntimeError("PyAV required: pip install av") from exc


def discover_episodes(session_dir: Path) -> list[dict]:
    """Find all episode_* subdirs with data.parquet + video.mp4."""
    episodes = []
    for ep_dir in sorted(session_dir.glob("episode_*")):
        parquet = ep_dir / "data.parquet"
        video = ep_dir / "video.mp4"
        if parquet.exists() and video.exists():
            df = pd.read_parquet(parquet)
            if not df.empty:
                episodes.append(
                    {
                        "dir": ep_dir,
                        "num_frames": len(df),
                        "task": str(df["task"].iloc[0]),
                        "task_index": int(df["task_index"].iloc[0]),
                    }
                )
    return episodes


def build_task_mapping(session_dir: Path) -> dict[str, int]:
    """Load task_index → task string mapping from tasks.json."""
    tasks_file = session_dir / "tasks.json"
    if tasks_file.exists():
        raw = json.loads(tasks_file.read_text())
        return {v: int(k) for k, v in raw.items()}
    episodes = discover_episodes(session_dir)
    mapping: dict[str, int] = {}
    for ep in episodes:
        if ep["task"] not in mapping:
            mapping[ep["task"]] = len(mapping)
    return mapping


def build_task_mapping_multi(session_dirs: list[Path]) -> dict[str, int]:
    """Build a unified task mapping from multiple session directories."""
    mapping: dict[str, int] = {}
    for sd in session_dirs:
        per_dir = build_task_mapping(sd)
        for task, idx in per_dir.items():
            if task not in mapping:
                mapping[task] = idx
    return mapping


class _FrameCache:
    """LRU cache for decoded video frames."""

    def __init__(self, image_size: tuple[int, int], max_items: int = 16):
        self.image_size = image_size
        self.max_items = max_items
        self._cache: OrderedDict[Path, list[np.ndarray]] = OrderedDict()

    def get(self, video_path: Path) -> list[np.ndarray]:
        if video_path in self._cache:
            self._cache.move_to_end(video_path)
            return self._cache[video_path]
        frames = self._decode(video_path)
        self._cache[video_path] = frames
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return frames

    def _decode(self, video_path: Path) -> list[np.ndarray]:
        w, h = self.image_size
        decoded = []
        with av.open(str(video_path)) as container:
            for frame in container.decode(video=0):
                img = Image.fromarray(frame.to_ndarray(format="rgb24"))
                img = img.resize((w, h), Image.Resampling.BILINEAR)
                decoded.append(np.asarray(img, dtype=np.uint8))
        return decoded


class MiniVLADataset(Dataset):
    """One sample = (single RGB frame, task_index, action[vx,vy,omega])."""

    def __init__(
        self,
        session_dir: Path | str | list[Path | str],
        task_to_idx: dict[str, int],
        image_size: tuple[int, int] = (160, 120),
        augment: bool = False,
        min_action_norm: float = 0.0,
    ):
        if isinstance(session_dir, list):
            self.session_dirs = [Path(s) for s in session_dir]
        else:
            self.session_dirs = [Path(session_dir)]
        self.task_to_idx = task_to_idx
        self.image_size = image_size
        self.augment = augment
        self.cache = _FrameCache(image_size, max_items=64)

        self.samples: list[tuple[Path, int, int, np.ndarray]] = []
        skipped = 0
        for sd in self.session_dirs:
            episodes = discover_episodes(sd)
            for ep in episodes:
                df = pd.read_parquet(ep["dir"] / "data.parquet")
                actions = np.asarray(df["action"].tolist(), dtype=np.float32)
                task_idx = task_to_idx.get(ep["task"], 0)
                video_path = ep["dir"] / "video.mp4"
                for frame_idx in range(len(actions)):
                    if np.max(np.abs(actions[frame_idx])) < min_action_norm:
                        skipped += 1
                        continue
                    self.samples.append((video_path, frame_idx, task_idx, actions[frame_idx]))
        if skipped > 0:
            print(f"[mini_vla] Filtered out {skipped} idle frames (threshold={min_action_norm})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        video_path, frame_idx, task_idx, action = self.samples[index]
        frames = self.cache.get(video_path)
        rgb = frames[frame_idx]
        image = Image.fromarray(rgb)
        if self.augment:
            image = self._augment(image)
        image_tensor = TF.to_tensor(image)
        return {
            "image": image_tensor,
            "task_idx": torch.tensor(task_idx, dtype=torch.long),
            "action": torch.as_tensor(action, dtype=torch.float32),
        }

    def _augment(self, image: Image.Image) -> Image.Image:
        import random

        image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
        image = TF.adjust_contrast(image, random.uniform(0.7, 1.3))
        image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))
        image = TF.adjust_hue(image, random.uniform(-0.05, 0.05))
        if random.random() < 0.3:
            image = TF.gaussian_blur(image, kernel_size=3)
        return image

    def preload_all(self) -> None:
        """Decode all videos upfront."""
        seen: set[Path] = set()
        for video_path, _, _, _ in self.samples:
            if video_path not in seen:
                self.cache.get(video_path)
                seen.add(video_path)
