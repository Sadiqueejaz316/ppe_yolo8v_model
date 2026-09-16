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
    """Single-slot buffer. Putting a new frame drops the unread previous one.

    Lock is held only to replace or take the slot — never during encode, disk,
    or inference.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: VideoFrame | None = None
        self._dropped = 0
        self._ready = threading.Event()

    @property
    def dropped(self) -> int:
        return self._dropped

    def put(self, frame: VideoFrame) -> None:
        with self._lock:
            if self._frame is not None:
                self._dropped += 1
            self._frame = frame
            self._ready.set()

    def take(self, timeout: float | None = None) -> VideoFrame | None:
        if timeout is not None and timeout > 0:
            self._ready.wait(timeout)
        with self._lock:
            frame = self._frame
            self._frame = None
            self._ready.clear()
            return frame

    def wake(self) -> None:
        """Unblock waiters (shutdown). Does not invent a frame."""
        self._ready.set()


class InferenceGate:
    """Allow inference at most ``target_fps`` times per second."""

    def __init__(self, target_fps: float) -> None:
        self._interval = 1.0 / max(float(target_fps), 0.1)
        self._last: float | None = None

    def allow(self, now: float | None = None, *, commit: bool = True) -> bool:
        ts = time.monotonic() if now is None else now
        if self._last is None or (ts - self._last) >= self._interval:
            if commit:
                self._last = ts
            return True
        return False


def drain_source(
    source_frames: Iterator[VideoFrame],
    buffer: LatestFrameBuffer,
    stop_event: threading.Event,
) -> None:
    """Copy frames into the latest-slot buffer until stopped.

    Does not catch exceptions: the caller decides reconnect vs shutdown.
    """
    for frame in source_frames:
        if stop_event.is_set():
            break
        buffer.put(frame)
