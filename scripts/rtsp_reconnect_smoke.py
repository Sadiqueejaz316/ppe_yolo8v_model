"""Short RTSP reconnect smoke test that does not load the PPE model."""

from __future__ import annotations

import threading
import time

from src.config.settings import CameraConfig
from src.exceptions import CameraConnectionError
from src.logging_setup import setup_logging
from src.video.camera import RTSPCamera


def main() -> None:
    setup_logging("INFO")
    camera = RTSPCamera(
        CameraConfig(
            id="CAM-001",
            name="Reconnect smoke",
            rtsp_url="rtsp://127.0.0.1:1/invalid",
            reconnect_delay=1.0,
            connection_timeout=2.0,
            read_timeout=2.0,
        )
    )
    timer = threading.Timer(8.0, camera.stop)
    timer.daemon = True
    timer.start()
    started = time.monotonic()
    try:
        camera.connect()
    except CameraConnectionError:
        pass
    for _frame in camera.frames():
        break
    timer.cancel()
    elapsed = time.monotonic() - started
    print(
        f"elapsed={elapsed:.1f}s reconnect_count={camera.reconnect_count} connected={camera.connected}"
    )


if __name__ == "__main__":
    main()
