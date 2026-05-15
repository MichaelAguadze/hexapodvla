"""CNN policy package for TurboPi."""

from __future__ import annotations

DEFAULT_IMAGE_WIDTH = 160
DEFAULT_IMAGE_HEIGHT = 120
DEFAULT_FRAME_HISTORY = 3
DEFAULT_DATA_ROOT = "data/turbopi_cnn/episodes"
LEGACY_DATA_ROOT = "data/turbopi_cnn_loop/episodes"

from .model import LoopCNNModel  # noqa: E402

__all__ = [
    "DEFAULT_IMAGE_WIDTH",
    "DEFAULT_IMAGE_HEIGHT",
    "DEFAULT_FRAME_HISTORY",
    "DEFAULT_DATA_ROOT",
    "LEGACY_DATA_ROOT",
    "LoopCNNModel",
]
