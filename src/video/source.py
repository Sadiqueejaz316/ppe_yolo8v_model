"""Video source interface used by RTSP, files, and (later) webcams."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

import numpy as np


@dataclass
class VideoFrame:
    image: np.ndarray
    timestamp: datetime
    camera_id: str
    frame_index: int
    source_fps: float = 0.0
    camera_name: str = ""


class VideoSource(ABC):
    """Common ingestion contract. Multi-camera support can wrap many of these."""

    @property
    @abstractmethod
    def camera_id(self) -> str:
        raise NotImplementedError

    @property
    def camera_name(self) -> str:
        return self.camera_id

    @property
    def reconnect_count(self) -> int:
        return 0

    @property
    def dropped_frames(self) -> int:
        return 0

    @property
    def measured_fps(self) -> float:
        return 0.0

    @property
    def connected(self) -> bool:
        return False

    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def frames(self) -> Iterator[VideoFrame]:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    def release(self) -> None:
        self.stop()

    @staticmethod
    def utcnow() -> datetime:
        return datetime.now(timezone.utc)
