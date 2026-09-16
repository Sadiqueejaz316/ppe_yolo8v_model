"""Adapter: pipeline ProcessedFrame → dashboard stores.

Does not compute PPE compliance. It copies scene_summary and confirmed events.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from src.config.settings import AppConfig
from src.events.store import EventStore
from src.exceptions import FRAME_RUNTIME_ERRORS
from src.ops.live import LiveFrameBuffer, LiveStateStore, get_global_frame_buffer, is_complete_jpeg

logger = logging.getLogger(__name__)

# Confirmed events are rare; still bound the queue so a stuck SQLite cannot grow forever.
MAX_PENDING_EVENTS = 256


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
        self._pending_events: list[tuple[Any, dict[str, Any] | None]] = []
        self._has_work = threading.Event()
        self._stop_worker = threading.Event()
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
        if self._stop_worker.is_set():
            return
        with self._queue_lock:
            if self._stop_worker.is_set():
                return
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._worker_thread = threading.Thread(
                    target=self._worker_loop,
                    daemon=True,
                    name="dashboard-sink-persist",
                )
                self._worker_thread.start()

    def _drain_pending(
        self,
    ) -> tuple[list[tuple[str, dict[str, Any], bytes | None]], list[tuple[Any, dict[str, Any] | None]]]:
        with self._queue_lock:
            self._has_work.clear()
            live_items = [
                (cam_id, payload, jpeg) for cam_id, (payload, jpeg) in self._pending.items()
            ]
            self._pending.clear()
            event_items = list(self._pending_events)
            self._pending_events.clear()
        return live_items, event_items

    def _persist_batch(
        self,
        live_items: list[tuple[str, dict[str, Any], bytes | None]],
        event_items: list[tuple[Any, dict[str, Any] | None]],
    ) -> None:
        for cam_id, payload, jpeg in live_items:
            try:
                self.live.write(cam_id, payload, jpeg)
            except OSError as exc:
                logger.warning("DASHBOARD_ASYNC_PERSIST_FAILED camera=%s error=%s", cam_id, exc)
        for event, snapshot in event_items:
            try:
                self.events.insert(event, ppe_snapshot=snapshot)
            except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
                logger.warning(
                    "DASHBOARD_EVENT_STORE_FAILED event=%s error=%s",
                    getattr(event, "event_id", None),
                    exc,
                )

    def _worker_loop(self) -> None:
        while not self._stop_worker.is_set():
            self._has_work.wait(timeout=0.1)
            live_items, event_items = self._drain_pending()
            self._persist_batch(live_items, event_items)
        live_items, event_items = self._drain_pending()
        self._persist_batch(live_items, event_items)

    def flush(self) -> None:
        live_items, event_items = self._drain_pending()
        self._persist_batch(live_items, event_items)

    def close(self) -> None:
        self._stop_worker.set()
        self._has_work.set()
        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning("DASHBOARD_SINK_WORKER_JOIN_TIMEOUT")
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
        queued_events: list[tuple[Any, dict[str, Any] | None]] = []
        for event in getattr(processed, "events", None) or []:
            queued_events.append((event, by_person.get(event.person_id)))

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
        # Encode outside any shared lock so the vision loop never holds a lock
        # during JPEG work. Disk write happens on the worker thread.
        live_quality = max(40, min(int(self._jpeg_quality), 75))
        jpeg = _encode_jpeg(getattr(processed, "annotated", None), live_quality)

        cam_key = str(camera_id)
        if jpeg and self.frame_buffer is not None:
            self.frame_buffer.update(cam_key, jpeg, payload)

        if self._async_persistence:
            dropped_events = 0
            with self._queue_lock:
                self._pending[cam_key] = (payload, jpeg)
                for item in queued_events:
                    if len(self._pending_events) >= MAX_PENDING_EVENTS:
                        self._pending_events.pop(0)
                        dropped_events += 1
                    self._pending_events.append(item)
            if dropped_events:
                logger.warning(
                    "DASHBOARD_EVENT_QUEUE_FULL camera=%s dropped=%s",
                    camera_id,
                    dropped_events,
                )
            self._ensure_worker_started()
            self._has_work.set()
            return

        try:
            self.live.write(cam_key, payload, jpeg)
        except OSError as exc:
            logger.warning("DASHBOARD_LIVE_WRITE_FAILED camera=%s error=%s", camera_id, exc)
        for event, snapshot in queued_events:
            try:
                self.events.insert(event, ppe_snapshot=snapshot)
            except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
                logger.warning(
                    "DASHBOARD_EVENT_STORE_FAILED event=%s error=%s",
                    getattr(event, "event_id", None),
                    exc,
                )


def _encode_jpeg(image: Any, quality: int) -> bytes | None:
    if image is None:
        return None
    try:
        import cv2

        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            logger.warning("LIVE_JPEG_ENCODE_FAILED reason=imencode_false")
            return None
        data = bytes(buffer)
        if not is_complete_jpeg(data):
            logger.warning("LIVE_JPEG_ENCODE_FAILED reason=incomplete")
            return None
        return data
    except FRAME_RUNTIME_ERRORS as exc:
        logger.warning("LIVE_JPEG_ENCODE_FAILED error=%s", exc)
        return None
