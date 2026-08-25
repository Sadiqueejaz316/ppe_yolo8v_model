import time

import numpy as np
import pytest

from src.config.settings import CameraConfig
from src.exceptions import CameraConnectionError, InvalidSourceError
from src.video.camera import RTSPCamera


class FakeCapture:
    def __init__(self, url, opened=True, frames=None, fail_after=None, fail_open_times=0):
        self.url = url
        self._opened = opened
        self._frames = list(frames or [])
        self._fail_after = fail_after
        self._reads = 0
        self.released = False
        self.props = {}
        type(self).open_attempts = getattr(type(self), "open_attempts", 0) + 1
        if fail_open_times and type(self).open_attempts <= fail_open_times:
            self._opened = False

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def read(self):
        self._reads += 1
        if self._fail_after is not None and self._reads > self._fail_after:
            self._opened = False
            return False, None
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self):
        self.released = True
        self._opened = False


def _config(**kwargs) -> CameraConfig:
    values = dict(
        id="CAM-001",
        name="Test",
        rtsp_url="rtsp://user:secret@192.168.1.10:554/stream",
        reconnect_delay=0.01,
        connection_timeout=1,
        read_timeout=1,
    )
    values.update(kwargs)
    return CameraConfig(**values)


def test_empty_rtsp_url_rejected():
    with pytest.raises(InvalidSourceError):
        RTSPCamera(_config(rtsp_url=""))


def test_invalid_rtsp_url_rejected():
    camera = RTSPCamera(_config(rtsp_url="not-a-url"), capture_factory=lambda url: FakeCapture(url, opened=False))
    with pytest.raises(InvalidSourceError):
        camera.connect()


def test_connection_failure():
    camera = RTSPCamera(
        _config(),
        capture_factory=lambda url: FakeCapture(url, opened=False),
    )
    with pytest.raises(CameraConnectionError):
        camera.connect()


def test_reconnect_after_dropped_frames():
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)]
    state = {"opens": 0}

    class ReconnectingCapture(FakeCapture):
        def __init__(self, url):
            state["opens"] += 1
            if state["opens"] == 1:
                super().__init__(url, opened=True, frames=frames[:2], fail_after=2)
            else:
                super().__init__(url, opened=True, frames=frames[2:])

    sleeps = []
    camera = RTSPCamera(_config(), capture_factory=ReconnectingCapture, sleep=lambda s: sleeps.append(s))
    got = []
    for frame in camera.frames():
        got.append(frame)
        if len(got) >= 3:
            camera.stop()
            break
    assert len(got) >= 3
    assert camera.reconnect_count >= 1
    assert sleeps
    assert all(item.camera_id == "CAM-001" for item in got)


def test_rtsp_credentials_not_stored_on_frame():
    frame_img = np.zeros((4, 4, 3), dtype=np.uint8)
    camera = RTSPCamera(
        _config(),
        capture_factory=lambda url: FakeCapture(url, opened=True, frames=[frame_img]),
    )
    camera.connect()
    frame = next(camera.frames())
    camera.stop()
    assert "secret" not in repr(frame)
    assert frame.camera_id == "CAM-001"


def test_clean_shutdown_stops_frame_loop():
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(20)]
    camera = RTSPCamera(
        _config(),
        capture_factory=lambda url: FakeCapture(url, opened=True, frames=frames),
        sleep=lambda _s: None,
    )
    got = []
    for frame in camera.frames():
        got.append(frame)
        if len(got) == 2:
            camera.stop()
    assert len(got) == 2
    assert camera.connected is False


def test_repeated_reconnect():
    frames = [np.zeros((6, 6, 3), dtype=np.uint8) for _ in range(8)]
    state = {"opens": 0}

    class FlakyCapture(FakeCapture):
        def __init__(self, url):
            state["opens"] += 1
            if state["opens"] in {1, 3}:
                super().__init__(url, opened=True, frames=frames[:2], fail_after=2)
            else:
                super().__init__(url, opened=True, frames=frames[2:4], fail_after=2)

    camera = RTSPCamera(_config(), capture_factory=FlakyCapture, sleep=lambda _s: None)
    got = []
    for frame in camera.frames():
        got.append(frame)
        if camera.reconnect_count >= 2 and len(got) >= 4:
            camera.stop()
            break
        if len(got) >= 12:
            camera.stop()
            break
    assert len(got) >= 4
    assert camera.reconnect_count >= 2
