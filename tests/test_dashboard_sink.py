import time
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

from src.events.store import EventStore
from src.events.violation import PPEViolationEvent
from src.metrics.collector import MetricsSnapshot
from src.ops.live import LiveFrameBuffer, LiveStateStore, is_complete_jpeg
from src.ops.sink import DashboardSink
from src.video.source import VideoFrame


def _processed(tmp_image=None, events=None, camera_id="CAM-001"):
    image = tmp_image if tmp_image is not None else np.zeros((16, 16, 3), dtype=np.uint8)
    frame = VideoFrame(
        image=image,
        timestamp=datetime.now(timezone.utc),
        camera_id=camera_id,
        frame_index=1,
        source_fps=12.0,
        camera_name="test",
    )
    metrics = MetricsSnapshot(
        camera_id=camera_id,
        camera_fps=12.0,
        inference_fps=10.0,
        inference_latency_ms=40.0,
        end_to_end_latency_ms=50.0,
        persons=0,
        detections=0,
        violations=0,
        dropped_frames=0,
        reconnect_count=0,
        cpu_percent=None,
        gpu_percent=None,
        camera_connected=True,
    )
    return SimpleNamespace(
        scene_summary={"people": [], "total_people": 0, "compliant": 0, "violations": 0},
        metrics=metrics,
        frame=frame,
        events=events or [],
        annotated=image,
    )


def test_is_complete_jpeg():
    assert is_complete_jpeg(b"\xff\xd8\xff\xd9") is True
    assert is_complete_jpeg(b"\xff\xd8\xff") is False
    assert is_complete_jpeg(b"") is False
    assert is_complete_jpeg(None) is False


def test_live_store_rejects_incomplete_jpeg(tmp_path):
    store = LiveStateStore(tmp_path / "live")
    path = tmp_path / "live" / "CAM-001.jpg"
    path.write_bytes(b"\xff\xd8\xff\x00partial")
    assert store.jpeg_bytes("CAM-001") is None


def test_live_store_atomic_write_readable(tmp_path):
    store = LiveStateStore(tmp_path / "live")
    jpeg = b"\xff\xd8\xff\xd9"
    store.write("CAM-001", {"camera_id": "CAM-001"}, jpeg)
    assert store.jpeg_bytes("CAM-001") == jpeg
    assert not (tmp_path / "live" / "CAM-001.jpg.tmp").exists()


def test_observe_does_not_wait_for_slow_disk(tmp_path):
    events = EventStore(tmp_path / "ppe.sqlite")
    buffer = LiveFrameBuffer()

    class SlowLive(LiveStateStore):
        def write(self, camera_id, payload, jpeg=None):
            time.sleep(0.25)
            super().write(camera_id, payload, jpeg)

    sink = DashboardSink(
        events,
        SlowLive(tmp_path / "live"),
        frame_buffer=buffer,
        async_persistence=True,
    )
    try:
        started = time.perf_counter()
        sink.observe(_processed())
        elapsed = time.perf_counter() - started
        assert elapsed < 0.15
        jpeg = buffer.get_jpeg("CAM-001")
        assert jpeg is not None
        assert is_complete_jpeg(jpeg)
    finally:
        sink.close()


def test_async_event_insert_and_close_joins_worker(tmp_path):
    events = EventStore(tmp_path / "ppe.sqlite")
    live = LiveStateStore(tmp_path / "live")
    sink = DashboardSink(events, live, frame_buffer=LiveFrameBuffer(), async_persistence=True)
    event = PPEViolationEvent.create(
        camera_id="CAM-001",
        timestamp=datetime.now(timezone.utc),
        person_id=7,
        missing_ppe="helmet",
        confidence=0.9,
    )
    processed = _processed(events=[event])
    processed.scene_summary = {
        "people": [{"person_id": 7, "helmet": "MISSING"}],
        "total_people": 1,
        "compliant": 0,
        "violations": 1,
    }
    try:
        sink.observe(processed)
        sink.close()
        row = events.get(event.event_id)
        assert row is not None
        assert row["violation_type"] == "HELMET_MISSING"
        thread = sink._worker_thread
        assert thread is None or not thread.is_alive()
    finally:
        sink.close()


def test_encode_failure_does_not_raise(tmp_path):
    events = EventStore(tmp_path / "ppe.sqlite")
    live = LiveStateStore(tmp_path / "live")
    sink = DashboardSink(events, live, frame_buffer=LiveFrameBuffer(), async_persistence=False)
    processed = _processed()
    processed.annotated = "not-an-image"
    sink.observe(processed)
    payload = live.read("CAM-001")
    assert payload is not None
    assert live.jpeg_bytes("CAM-001") is None
