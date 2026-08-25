"""File-based video sources for development without a live camera."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import numpy as np

from src.exceptions import InvalidSourceError
from src.video.source import VideoFrame, VideoSource

logger = logging.getLogger(__name__)


class ImageSource(VideoSource):
    def __init__(self, path: str | Path, camera_id: str = "IMG-001", camera_name: str = "Image") -> None:
        self._path = Path(path)
        self._camera_id = camera_id
        self._camera_name = camera_name
        self._image: np.ndarray | None = None
        self._stop = False

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def camera_name(self) -> str:
        return self._camera_name

    @property
    def connected(self) -> bool:
        return self._image is not None

    def connect(self) -> None:
        import cv2

        if not self._path.exists():
            raise InvalidSourceError(f"Image not found: {self._path}")
        image = cv2.imread(str(self._path))
        if image is None:
            raise InvalidSourceError(f"Could not read image: {self._path}")
        self._image = image
        logger.info("IMAGE_LOADED camera=%s path=%s", self.camera_id, self._path)

    def frames(self) -> Iterator[VideoFrame]:
        if self._image is None:
            self.connect()
        assert self._image is not None
        if self._stop:
            return
        yield VideoFrame(
            image=self._image,
            timestamp=self.utcnow(),
            camera_id=self.camera_id,
            frame_index=1,
            source_fps=0.0,
            camera_name=self.camera_name,
        )

    def stop(self) -> None:
        self._stop = True


class VideoFileSource(VideoSource):
    def __init__(self, path: str | Path, camera_id: str = "VID-001", camera_name: str = "Video file") -> None:
        self._path = Path(path)
        self._camera_id = camera_id
        self._camera_name = camera_name
        self._capture = None
        self._stop = False
        self._fps = 0.0
        self._dropped_frames = 0

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def camera_name(self) -> str:
        return self._camera_name

    @property
    def connected(self) -> bool:
        return self._capture is not None

    @property
    def measured_fps(self) -> float:
        return self._fps

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    def connect(self) -> None:
        import cv2

        if not self._path.exists():
            raise InvalidSourceError(f"Video not found: {self._path}")
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            raise InvalidSourceError(f"Could not open video: {self._path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._fps = fps if fps > 1e-3 else 25.0
        self._capture = capture
        logger.info("VIDEO_OPENED camera=%s path=%s fps=%.2f", self.camera_id, self._path, self._fps)

    def frames(self) -> Iterator[VideoFrame]:
        if self._capture is None:
            self.connect()
        assert self._capture is not None
        index = 0
        started = self.utcnow()
        while not self._stop:
            ok, frame = self._capture.read()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                break
            index += 1
            ts = started + timedelta(seconds=(index - 1) / max(self._fps, 1e-3))
            yield VideoFrame(
                image=frame,
                timestamp=ts,
                camera_id=self.camera_id,
                frame_index=index,
                source_fps=self._fps,
                camera_name=self.camera_name,
            )
        self.stop()

    def stop(self) -> None:
        self._stop = True
        capture = self._capture
        self._capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
