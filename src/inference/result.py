"""Normalized detector outputs. Downstream code must not depend on Ultralytics types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class Detection:
    """Normalized detection. ``bbox`` is always pixel ``(x1, y1, x2, y2)`` / xyxy."""

    class_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    track_id: int | None = None

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]


@dataclass(frozen=True)
class LatencyBreakdown:
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


@dataclass
class DetectionResult:
    detections: list[Detection] = field(default_factory=list)
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    device: str = "cpu"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    image_size: tuple[int, int] = (0, 0)

    def of_class(self, class_ids: set[int]) -> list[Detection]:
        return [det for det in self.detections if det.class_id in class_ids]
