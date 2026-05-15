"""Mini VLA model: CNN vision + trainable language embedding → action."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class MiniVLAConfig:
    image_width: int = 160
    image_height: int = 120
    input_channels: int = 3
    vision_dim: int = 128
    language_dim: int = 64
    num_tasks: int = 4
    action_dim: int = 3
    dropout: float = 0.1


class VisionEncoder(nn.Module):
    """Simple CNN that maps an RGB image to a vision_dim vector."""

    def __init__(self, in_channels: int, vision_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, vision_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LanguageEncoder(nn.Module):
    """Trainable embedding table that maps a task index to language_dim."""

    def __init__(self, num_tasks: int, language_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_tasks, language_dim)

    def forward(self, task_idx: torch.Tensor) -> torch.Tensor:
        return self.embedding(task_idx)


class MiniVLA(nn.Module):
    """Vision-Language-Action: concat vision (64d) + language (32d) → action (3d)."""

    def __init__(self, config: MiniVLAConfig | None = None):
        super().__init__()
        self.config = config or MiniVLAConfig()
        self.vision = VisionEncoder(self.config.input_channels, self.config.vision_dim)
        self.language = LanguageEncoder(self.config.num_tasks, self.config.language_dim)
        fused_dim = self.config.vision_dim + self.config.language_dim
        self.action_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(64, self.config.action_dim),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, task_idx: torch.Tensor) -> torch.Tensor:
        v = self.vision(image)
        l = self.language(task_idx)
        fused = torch.cat([v, l], dim=-1)
        return self.action_head(fused)


def save_checkpoint(
    path: Path,
    model: MiniVLA,
    *,
    epoch: int,
    metrics: dict[str, float],
    task_to_idx: dict[str, int],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "model_config": asdict(model.config),
            "model_state_dict": model.state_dict(),
            "task_to_idx": task_to_idx,
        },
        path,
    )


def load_checkpoint(
    path: Path, map_location: str | torch.device | None = None
) -> tuple[MiniVLA, dict[str, object]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    config = MiniVLAConfig(**payload["model_config"])
    model = MiniVLA(config)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
