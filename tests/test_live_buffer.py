import concurrent.futures
import threading

from src.ops.live import LiveFrameBuffer


def test_update_and_get():
    buf = LiveFrameBuffer()
    dummy_jpeg = b"\xff\xd8\xff\xe0test"
    meta = {"timestamp": "2026-09-08T12:00:00Z", "camera_fps": 10.0}

    buf.update("CAM-001", dummy_jpeg, meta)

    entry = buf.get("CAM-001")
    assert entry is not None
    assert entry[0] == dummy_jpeg
    assert entry[1] == meta
    assert buf.get_jpeg("CAM-001") == dummy_jpeg
    assert buf.get_metadata("CAM-001") == meta


def test_newest_replaces_old():
    buf = LiveFrameBuffer()
    buf.update("CAM-001", b"first_frame", {"frame": 1})
    buf.update("CAM-001", b"second_frame", {"frame": 2})

    assert buf.get_jpeg("CAM-001") == b"second_frame"
    assert buf.get_metadata("CAM-001") == {"frame": 2}


def test_camera_ids_are_independent():
    buf = LiveFrameBuffer()
    buf.update("CAM-001", b"cam1_data", {"cam": 1})
    buf.update("CAM-002", b"cam2_data", {"cam": 2})

    assert buf.get_jpeg("CAM-001") == b"cam1_data"
    assert buf.get_jpeg("CAM-002") == b"cam2_data"


def test_missing_camera_returns_none():
    buf = LiveFrameBuffer()
    assert buf.get("CAM-999") is None
    assert buf.get_jpeg("CAM-999") is None
    assert buf.get_metadata("CAM-999") is None


def test_clear():
    buf = LiveFrameBuffer()
    buf.update("CAM-001", b"cam1", {})
    buf.update("CAM-002", b"cam2", {})

    buf.clear("CAM-001")
    assert buf.get("CAM-001") is None
    assert buf.get("CAM-002") is not None

    buf.clear()
    assert buf.get("CAM-002") is None


def test_concurrent_update_read():
    buf = LiveFrameBuffer()
    stop_event = threading.Event()

    def writer(cam_id: str):
        count = 0
        while not stop_event.is_set() and count < 1000:
            buf.update(cam_id, f"jpeg_{count}".encode(), {"count": count})
            count += 1

    def reader(cam_id: str):
        reads = 0
        while not stop_event.is_set() and reads < 1000:
            val = buf.get_jpeg(cam_id)
            if val is not None:
                assert isinstance(val, bytes)
            reads += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(writer, "CAM-001"),
            executor.submit(writer, "CAM-002"),
            executor.submit(reader, "CAM-001"),
            executor.submit(reader, "CAM-002"),
        ]
        concurrent.futures.wait(futures, timeout=5.0)
        stop_event.set()

    for f in futures:
        f.result()
