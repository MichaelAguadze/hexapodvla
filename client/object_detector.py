"""YOLO-based object detector for the hexapod camera feed."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: float       # normalised 0→1 across frame width
    center_y: float       # normalised 0→1 across frame height
    area_fraction: float  # bounding-box area as fraction of frame area


class ObjectDetector:
    """Wraps a YOLOv8 model for single-frame inference.

    Args:
        model_name:  ultralytics model filename or path (e.g. "yolov8n.pt").
                     The file is downloaded automatically on first use.
        confidence:  minimum detection confidence to keep [0, 1].
        device:      inference device — "cpu", "cuda", or "mps".
        classes:     if given, only return detections whose label is in this set.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.40,
        device: str = "cpu",
        classes: Optional[List[str]] = None,
    ) -> None:
        from ultralytics import YOLO
        self._model = YOLO(model_name)
        self._conf = confidence
        self._device = device
        self._filter: Optional[set] = set(classes) if classes else None

    @property
    def class_names(self) -> dict:
        """Mapping of class index → label string."""
        return self._model.names

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame and return detections.

        Args:
            frame_bgr: H×W×3 uint8 numpy array in BGR colour order.

        Returns:
            List of Detection objects, one per bounding box.
        """
        h, w = frame_bgr.shape[:2]
        results = self._model(
            frame_bgr,
            conf=self._conf,
            device=self._device,
            verbose=False,
        )
        out: List[Detection] = []
        for r in results:
            for box in r.boxes:
                label = self._model.names[int(box.cls[0])]
                if self._filter and label not in self._filter:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                area = (x2 - x1) * (y2 - y1) / (w * h)
                out.append(Detection(label, conf, x1, y1, x2, y2, cx, cy, area))
        return out
