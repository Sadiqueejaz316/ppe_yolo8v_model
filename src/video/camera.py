"""RTSP camera ingestion with reconnect, timeouts, and stale-frame dropping."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from src.config.settings import CameraConfig
from src.exceptions import CameraConnectionError, FRAME_RUNTIME_ERRORS, InvalidSourceError
from src.security import redact_rtsp_url
from src.video.source import VideoFrame, VideoSource

logger = logging.getLogger(__name__)

CaptureFactory = Callable[..., Any]

# Consecutive failed reads before treating the stream as disconnected.
READ_FAILURES_BEFORE_RECONNECT = 3
RECONNECT_BACKOFF_CAP_S = 30.0


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
        self._consecutive_read_failures = 0

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
                    "CAMERA_DISCONNECTED camera=%s reason=initial_open_failed; will retry",
                    self.camera_id,
                )
        while not self._stop.is_set():
            if not self._connected or self._capture is None:
                self._reconnect()
                if not self._connected:
                    continue
            frame = self._read_frame()
            if frame is None:
                if self._stop.is_set():
                    break
                if self._consecutive_read_failures < READ_FAILURES_BEFORE_RECONNECT:
                    self._sleep_interruptible(0.05)
                    continue
                self._handle_disconnect()
                continue
            self._consecutive_read_failures = 0
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

        capture: Any = None
        opened = False
        try:
            capture = self._create_capture(url)
            timeout_ms = max(1.0, float(self._config.connection_timeout)) * 1000.0
            read_ms = max(1.0, float(self._config.read_timeout)) * 1000.0
            _set_capture_option(capture, "CAP_PROP_OPEN_TIMEOUT_MSEC", timeout_ms)
            _set_capture_option(capture, "CAP_PROP_READ_TIMEOUT_MSEC", read_ms)
            # Keep only the newest decoder frame so reconnects do not replay stale video.
            _set_capture_option(capture, "CAP_PROP_BUFFERSIZE", 1)
            opened = bool(capture.isOpened())
        except InvalidSourceError:
            raise
        except FRAME_RUNTIME_ERRORS as exc:
            logger.warning(
                "CAMERA_OPEN_EXCEPTION camera=%s url=%s error=%s",
                self.camera_id,
                redacted,
                exc,
            )
            opened = False

        if not opened:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    logger.debug("CAMERA_RELEASE_FAILED camera=%s", self.camera_id)
            if initial:
                logger.error("CAMERA_OPEN_FAILED camera=%s url=%s", self.camera_id, redacted)
                raise CameraConnectionError(f"Failed to open RTSP camera {self.camera_id}")
            logger.warning("CAMERA_OPEN_FAILED camera=%s url=%s", self.camera_id, redacted)
            self._connected = False
            return

        self._capture = capture
        self._connected = True
        self._consecutive_read_failures = 0
        self._drop_decoder_backlog()
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
            self._consecutive_read_failures += 1
            return None
        try:
            ok, frame = capture.read()
        except FRAME_RUNTIME_ERRORS as exc:
            self._consecutive_read_failures += 1
            logger.warning(
                "CAMERA_READ_EXCEPTION camera=%s consecutive=%s error=%s",
                self.camera_id,
                self._consecutive_read_failures,
                exc,
            )
            return None
        if not ok or frame is None:
            self._consecutive_read_failures += 1
            return None
        if getattr(frame, "size", 0) == 0:
            self._consecutive_read_failures += 1
            logger.warning("CAMERA_CORRUPT_FRAME camera=%s", self.camera_id)
            return None
        return frame

    def _handle_disconnect(self) -> None:
        logger.warning(
            "CAMERA_DISCONNECTED camera=%s consecutive_read_failures=%s",
            self.camera_id,
            self._consecutive_read_failures,
        )
        self._connected = False
        self._release()
        self._consecutive_read_failures = 0

    def _reconnect_delay(self) -> float:
        base = max(0.01, float(self._config.reconnect_delay))
        exponent = min(max(0, self._reconnect_count - 1), 8)
        return min(RECONNECT_BACKOFF_CAP_S, base * (2 ** exponent))

    def _sleep_interruptible(self, seconds: float) -> None:
        if seconds <= 0:
            return
        # Event.wait is interruptible on stop; injected test sleep stays injectable.
        if self._sleep is time.sleep:
            self._stop.wait(seconds)
            return
        self._sleep(seconds)

    def _drop_decoder_backlog(self) -> None:
        """Discard buffered decoder frames so the next yield is the newest."""
        capture = self._capture
        grab = getattr(capture, "grab", None) if capture is not None else None
        if not callable(grab):
            return
        for _ in range(2):
            try:
                if not grab():
                    break
            except FRAME_RUNTIME_ERRORS:
                break

    def _reconnect(self) -> None:
        self._reconnect_count += 1
        delay = self._reconnect_delay()
        logger.warning(
            "CAMERA_RECONNECTING camera=%s delay_s=%.2f attempt=%s",
            self.camera_id,
            delay,
            self._reconnect_count,
        )
        self._sleep_interruptible(delay)
        if self._stop.is_set():
            return
        try:
            self._open(initial=False)
        except CameraConnectionError as exc:
            logger.warning("CAMERA_RECONNECT_FAILED camera=%s error=%s", self.camera_id, exc)
            self._connected = False
            return
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
