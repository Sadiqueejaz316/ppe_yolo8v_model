import threading
import time
from datetime import datetime, timezone

import numpy as np

from src.video.frame_processor import InferenceGate, LatestFrameBuffer
from src.video.source import VideoFrame


def _frame(index: int) -> VideoFrame:
    return VideoFrame(
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        timestamp=datetime.now(timezone.utc),
        camera_id="CAM-001",
        frame_index=index,
        source_fps=30.0,
        camera_name="test",
    )


def test_put_drops_unread_previous_frame():
    buf = LatestFrameBuffer()
    buf.put(_frame(1))
    buf.put(_frame(2))
    buf.put(_frame(3))
    assert buf.dropped == 2
    taken = buf.take()
    assert taken is not None
    assert taken.frame_index == 3
    assert buf.take() is None


def test_take_timeout_returns_none():
    buf = LatestFrameBuffer()
    started = time.monotonic()
    assert buf.take(timeout=0.05) is None
    assert (time.monotonic() - started) >= 0.04


def test_wake_unblocks_take():
    buf = LatestFrameBuffer()
    result: list[VideoFrame | None] = []

    def _reader() -> None:
        result.append(buf.take(timeout=2.0))

    thread = threading.Thread(target=_reader)
    thread.start()
    time.sleep(0.02)
    buf.wake()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert result == [None]


def test_lock_not_held_across_wait():
    buf = LatestFrameBuffer()
    buf.put(_frame(1))
    # Concurrent put during take must replace, not queue.
    taken = buf.take()
    assert taken is not None
    buf.put(_frame(2))
    buf.put(_frame(3))
    assert buf.take().frame_index == 3


def test_inference_gate_limits_rate():
    gate = InferenceGate(10.0)
    now = 1000.0
    assert gate.allow(now) is True
    assert gate.allow(now + 0.05) is False
    assert gate.allow(now + 0.11) is True


def test_inference_gate_probe_does_not_consume_slot():
    gate = InferenceGate(10.0)
    now = 1000.0

    assert gate.allow(now, commit=False) is True
    assert gate.allow(now + 0.01) is True
