"""Adapter: pipeline ProcessedFrame → dashboard stores.

Does not compute PPE compliance. It copies scene_summary and confirmed events.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from src.config.settings import AppConfig
from src.events.store import EventStore
from src.ops.live import LiveFrameBuffer, LiveStateStore, get_global_frame_buffer

logger = logging.getLogger(__name__)


class DashboardSink:
    def __init__(
        self,
        events: EventStore,
        live: LiveStateStore,
        jpeg_quality: int = 90,
        frame_buffer: LiveFrameBuffer | None = None,
        async_persistence: bool = True,
    ) -> None:
        self.events = events
        self.live = live
        self._jpeg_quality = int(jpeg_quality)
        self.frame_buffer = frame_buffer if frame_buffer is not None else get_global_frame_buffer()
        self._async_persistence = async_persistence
        self._queue_lock = threading.Lock()
        self._pending: dict[str, tuple[dict[str, Any], bytes | None]] = {}
        self._has_work = threading.Event()
        self._stop_worker = False
        self._worker_thread: threading.Thread | None = None

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        frame_buffer: LiveFrameBuffer | None = None,
    ) -> DashboardSink:
        events = EventStore(config.resolve_path(config.dashboard.sqlite_path))
        jsonl = config.resolve_path(config.evidence.events_jsonl)
        try:
            events.import_jsonl(jsonl)
        except OSError:
            logger.warning("EVENT_STORE_JSONL_IMPORT_FAILED path=%s", jsonl, exc_info=True)
        live = LiveStateStore(config.resolve_path(config.dashboard.live_dir))
        return cls(
            events,
            live,
            jpeg_quality=config.evidence.jpeg_quality,
            frame_buffer=frame_buffer,
        )

    def _ensure_worker_started(self) -> None:
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="dashboard-sink-persist",
            )
            self._worker_thread.start()

    def _worker_loop(self) -> None:
        while not self._stop_worker:
            self._has_work.wait(timeout=0.1)
            items: list[tuple[str, dict[str, Any], bytes | None]] = []
            with self._queue_lock:
                self._has_work.clear()
                for cam_id, (payload, jpeg) in self._pending.items():
                    items.append((cam_id, payload, jpeg))
                self._pending.clear()

            for cam_id, payload, jpeg in items:
                try:
                    self.live.write(cam_id, payload, jpeg)
                except Exception:
                    logger.exception("DASHBOARD_ASYNC_PERSIST_FAILED camera=%s", cam_id)

    def flush(self) -> None:
        items: list[tuple[str, dict[str, Any], bytes | None]] = []
        with self._queue_lock:
            for cam_id, (payload, jpeg) in self._pending.items():
                items.append((cam_id, payload, jpeg))
            self._pending.clear()
        for cam_id, payload, jpeg in items:
            try:
                self.live.write(cam_id, payload, jpeg)
            except Exception:
                logger.exception("DASHBOARD_ASYNC_PERSIST_FAILED camera=%s", cam_id)

    def close(self) -> None:
        self._stop_worker = True
        self._has_work.set()
        self.flush()

    def observe(self, processed: Any) -> None:
        summary = getattr(processed, "scene_summary", None) or {}
        metrics = getattr(processed, "metrics", None)
        frame = getattr(processed, "frame", None)
        camera_id = getattr(metrics, "camera_id", None) or getattr(frame, "camera_id", None)
        if not camera_id:
            return

        people = list(summary.get("people") or [])
        by_person = {item.get("person_id"): item for item in people if isinstance(item, dict)}
        for event in getattr(processed, "events", None) or []:
            try:
                self.events.insert(event, ppe_snapshot=by_person.get(event.person_id))
            except Exception:
                logger.exception("DASHBOARD_EVENT_STORE_FAILED event=%s", getattr(event, "event_id", None))

        ts = getattr(frame, "timestamp", None)
        timestamp = ts.isoformat() if hasattr(ts, "isoformat") else None
        payload = {
            "timestamp": timestamp,
            "camera_id": camera_id,
            "camera_name": getattr(frame, "camera_name", "") or camera_id,
            "camera_connected": bool(getattr(metrics, "camera_connected", False)),
            "camera_fps": float(getattr(metrics, "camera_fps", 0.0) or 0.0),
            "inference_fps": float(getattr(metrics, "inference_fps", 0.0) or 0.0),
            "inference_latency_ms": float(getattr(metrics, "inference_latency_ms", 0.0) or 0.0),
            "scene_summary": summary,
        }
        # Live preview can use a slightly lower quality than evidence stills.
        live_quality = max(40, min(int(self._jpeg_quality), 75))
        jpeg = _encode_jpeg(getattr(processed, "annotated", None), live_quality)

        # 1. Update in-memory buffer immediately (< 0.01ms)
        cam_key = str(camera_id)
        if jpeg and self.frame_buffer is not None:
            self.frame_buffer.update(cam_key, jpeg, payload)

        # 2. Persist to disk (non-blocking when async_persistence is True)
        if self._async_persistence:
            with self._queue_lock:
                self._pending[cam_key] = (payload, jpeg)
            self._ensure_worker_started()
            self._has_work.set()
        else:
            try:
                self.live.write(cam_key, payload, jpeg)
            except OSError:
                logger.exception("DASHBOARD_LIVE_WRITE_FAILED camera=%s", camera_id)


def _encode_jpeg(image: Any, quality: int) -> bytes | None:
    if image is None:
        return None
    try:
        import cv2

        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return bytes(buffer)
    except Exception:
        return None
