"""Latest-frame buffering and inference-rate gating.

Industrial cameras may run at 25/30 FPS while inference is slower. This
module keeps only the newest unread frame so the pipeline stays low-latency.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

from src.video.source import VideoFrame


class LatestFrameBuffer:
    """Single-slot buffer. Putting a new frame drops the unread previous one."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: VideoFrame | None = None
        self._dropped = 0

    @property
    def dropped(self) -> int:
        return self._dropped

    def put(self, frame: VideoFrame) -> None:
        with self._lock:
            if self._frame is not None:
                self._dropped += 1
            self._frame = frame

    def take(self) -> VideoFrame | None:
        with self._lock:
            frame = self._frame
            self._frame = None
            return frame


class InferenceGate:
    """Allow inference at most ``target_fps`` times per second."""

    def __init__(self, target_fps: float) -> None:
        self._interval = 1.0 / max(float(target_fps), 0.1)
        self._last: float | None = None

    def allow(self, now: float | None = None) -> bool:
        ts = time.monotonic() if now is None else now
        if self._last is None or (ts - self._last) >= self._interval:
            self._last = ts
            return True
        return False


def drain_source(source_frames: Iterator[VideoFrame], buffer: LatestFrameBuffer, stop_event: threading.Event) -> None:
    try:
        for frame in source_frames:
            if stop_event.is_set():
                break
            buffer.put(frame)
    except Exception:
        stop_event.set()
        raise
