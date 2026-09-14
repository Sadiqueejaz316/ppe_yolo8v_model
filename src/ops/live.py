"""Filesystem live snapshot for the operator dashboard.

The pipeline writes scene_summary + an annotated JPEG. The API only reads.
No RTSP URLs are stored.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LiveFrameBuffer:
    """Thread-safe per-camera latest-frame in-memory buffer.

    Stores only the most recent JPEG + metadata for each camera.
    Does not accumulate an unbounded queue.
    Reading never touches disk.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: dict[str, tuple[bytes, dict[str, Any] | None]] = {}

    def update(
        self,
        camera_id: str,
        jpeg_bytes: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._frames[camera_id] = (jpeg_bytes, metadata)

    def get(self, camera_id: str) -> tuple[bytes, dict[str, Any] | None] | None:
        with self._lock:
            return self._frames.get(camera_id)

    def get_jpeg(self, camera_id: str) -> bytes | None:
        entry = self.get(camera_id)
        return entry[0] if entry is not None else None

    def get_metadata(self, camera_id: str) -> dict[str, Any] | None:
        entry = self.get(camera_id)
        return entry[1] if entry is not None else None

    def clear(self, camera_id: str | None = None) -> None:
        with self._lock:
            if camera_id is None:
                self._frames.clear()
            else:
                self._frames.pop(camera_id, None)


_GLOBAL_BUFFER: LiveFrameBuffer | None = None
_GLOBAL_BUFFER_LOCK = threading.Lock()


def get_global_frame_buffer() -> LiveFrameBuffer:
    """Return the global process-wide in-memory frame buffer."""
    global _GLOBAL_BUFFER
    with _GLOBAL_BUFFER_LOCK:
        if _GLOBAL_BUFFER is None:
            _GLOBAL_BUFFER = LiveFrameBuffer()
        return _GLOBAL_BUFFER


class LiveStateStore:
    def __init__(self, live_dir: Path) -> None:
        self.root = Path(live_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _json_path(self, camera_id: str) -> Path:
        return self.root / f"{camera_id}.json"

    def _jpeg_path(self, camera_id: str) -> Path:
        return self.root / f"{camera_id}.jpg"

    def _safe_replace(self, src: Path, dst: Path) -> None:
        for attempt in range(5):
            try:
                src.replace(dst)
                return
            except OSError:
                if attempt == 4:
                    try:
                        if src.exists():
                            dst.write_bytes(src.read_bytes())
                            src.unlink(missing_ok=True)
                    except OSError:
                        pass
                    return
                time.sleep(0.01)

    def write(
        self,
        camera_id: str,
        payload: dict[str, Any],
        jpeg: bytes | None = None,
    ) -> None:
        path = self._json_path(camera_id)
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
            self._safe_replace(tmp, path)
        except OSError:
            pass
        if jpeg:
            jpg = self._jpeg_path(camera_id)
            jtmp = jpg.with_suffix(".jpg.tmp")
            try:
                jtmp.write_bytes(jpeg)
                self._safe_replace(jtmp, jpg)
            except OSError:
                pass

    def read(self, camera_id: str) -> dict[str, Any] | None:
        path = self._json_path(camera_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        payload["_mtime"] = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        payload["_has_jpeg"] = self._jpeg_path(camera_id).is_file()
        return payload

    def list_camera_ids(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.json"))

    def jpeg_bytes(self, camera_id: str) -> bytes | None:
        path = self._jpeg_path(camera_id)
        if not path.is_file():
            return None
        try:
            data = path.read_bytes()
        except OSError:
            return None
        return data or None
