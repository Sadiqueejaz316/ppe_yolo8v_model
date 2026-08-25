"""RTSP camera ingestion with reconnect, timeouts, and stale-frame dropping."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from src.config.settings import CameraConfig
from src.exceptions import CameraConnectionError, InvalidSourceError
from src.security import redact_rtsp_url
from src.video.source import VideoFrame, VideoSource

logger = logging.getLogger(__name__)

CaptureFactory = Callable[..., Any]


def _set_capture_option(capture: Any, prop: str, value: float) -> None:
    import cv2

    attr = getattr(cv2, prop, None)
    if attr is None:
        return
    try:
        capture.set(attr, value)
    except Exception:
        logger.debug("CAMERA_PROP_UNSUPPORTED prop=%s", prop)


class RTSPCamera(VideoSource):
    """Single-camera RTSP source. Interface is camera-id based for later fan-out."""

    def __init__(
        self,
        config: CameraConfig,
        capture_factory: CaptureFactory | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not (config.rtsp_url or "").strip():
            raise InvalidSourceError(
                f"Camera {config.id} has an empty RTSP URL. Set RTSP_URL in the environment or .env."
            )
        self._config = config
        self._capture_factory = capture_factory
        self._sleep = sleep
        self._capture: Any = None
        self._stop = threading.Event()
        self._connected = False
        self._reconnect_count = 0
        self._dropped_frames = 0
        self._frame_index = 0
        self._measured_fps = 0.0
        self._last_ok_ts: float | None = None
        self._skip_counter = 0

    @property
    def camera_id(self) -> str:
        return self._config.id

    @property
    def camera_name(self) -> str:
        return self._config.name

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    @property
    def measured_fps(self) -> float:
        return self._measured_fps

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._open(initial=True)

    def stop(self) -> None:
        self._stop.set()
        self._release()

    def frames(self) -> Iterator[VideoFrame]:
        if self._capture is None and not self._connected:
            try:
                self.connect()
            except CameraConnectionError:
                logger.warning(
                    "CAMERA_DISCONNECTED camera=%s reason=initial_open_failed",
                    self.camera_id,
                )
        while not self._stop.is_set():
            if not self._connected:
                self._reconnect()
                if not self._connected:
                    continue
            frame = self._read_frame()
            if frame is None:
                self._handle_disconnect()
                continue
            if not self._accept_skip():
                self._dropped_frames += 1
                continue
            now = time.monotonic()
            if self._last_ok_ts is not None:
                dt = now - self._last_ok_ts
                if dt > 0:
                    instant = 1.0 / dt
                    self._measured_fps = (
                        instant if self._measured_fps <= 0 else (0.85 * self._measured_fps + 0.15 * instant)
                    )
            self._last_ok_ts = now
            self._frame_index += 1
            yield VideoFrame(
                image=frame,
                timestamp=self.utcnow(),
                camera_id=self.camera_id,
                frame_index=self._frame_index,
                source_fps=self._measured_fps,
                camera_name=self.camera_name,
            )

    def _accept_skip(self) -> bool:
        skip = max(0, int(self._config.frame_skip))
        if skip <= 0:
            return True
        self._skip_counter += 1
        if self._skip_counter % (skip + 1) == 0:
            return True
        return False

    def _open(self, initial: bool) -> None:
        self._release()
        url = self._config.rtsp_url.strip()
        redacted = redact_rtsp_url(url)
        if "://" not in url or not url.lower().startswith("rtsp://"):
            logger.error("CAMERA_INVALID_URL camera=%s url=%s", self.camera_id, redacted)
            raise InvalidSourceError(f"Camera {self.camera_id} RTSP URL is invalid")

        capture = self._create_capture(url)
        timeout_ms = max(1.0, float(self._config.connection_timeout)) * 1000.0
        read_ms = max(1.0, float(self._config.read_timeout)) * 1000.0
        _set_capture_option(capture, "CAP_PROP_OPEN_TIMEOUT_MSEC", timeout_ms)
        _set_capture_option(capture, "CAP_PROP_READ_TIMEOUT_MSEC", read_ms)
        _set_capture_option(capture, "CAP_PROP_BUFFERSIZE", 1)

        opened = False
        try:
            opened = bool(capture.isOpened())
        except Exception as exc:
            logger.error("CAMERA_OPEN_EXCEPTION camera=%s error=%s", self.camera_id, exc)
            opened = False

        if not opened:
            try:
                capture.release()
            except Exception:
                pass
            if initial:
                logger.error("CAMERA_OPEN_FAILED camera=%s url=%s", self.camera_id, redacted)
                raise CameraConnectionError(f"Failed to open RTSP camera {self.camera_id}")
            logger.warning("CAMERA_OPEN_FAILED camera=%s url=%s", self.camera_id, redacted)
            self._connected = False
            return

        self._capture = capture
        self._connected = True
        logger.info("CAMERA_CONNECTED camera=%s url=%s", self.camera_id, redacted)

    def _create_capture(self, url: str) -> Any:
        if self._capture_factory is not None:
            return self._capture_factory(url)
        import cv2

        backend = getattr(cv2, "CAP_FFMPEG", 0)
        try:
            return cv2.VideoCapture(url, backend)
        except TypeError:
            return cv2.VideoCapture(url)

    def _read_frame(self) -> np.ndarray | None:
        capture = self._capture
        if capture is None:
            return None
        try:
            ok, frame = capture.read()
        except Exception as exc:
            logger.warning("CAMERA_READ_EXCEPTION camera=%s error=%s", self.camera_id, exc)
            return None
        if not ok or frame is None:
            return None
        if getattr(frame, "size", 0) == 0:
            logger.warning("CAMERA_CORRUPT_FRAME camera=%s", self.camera_id)
            return None
        return frame

    def _handle_disconnect(self) -> None:
        logger.warning("CAMERA_DISCONNECTED camera=%s", self.camera_id)
        self._connected = False
        self._release()
        self._reconnect_count += 1

    def _reconnect(self) -> None:
        delay = max(0.1, float(self._config.reconnect_delay))
        logger.warning(
            "CAMERA_RECONNECTING camera=%s delay_s=%.1f attempt=%s",
            self.camera_id,
            delay,
            self._reconnect_count,
        )
        self._sleep(delay)
        if self._stop.is_set():
            return
        self._open(initial=False)
        if self._connected:
            logger.info("CAMERA_RECONNECTED camera=%s reconnects=%s", self.camera_id, self._reconnect_count)

    def _release(self) -> None:
        capture = self._capture
        self._capture = None
        self._connected = False
        if capture is None:
            return
        try:
            capture.release()
        except Exception:
            logger.debug("CAMERA_RELEASE_FAILED camera=%s", self.camera_id)
