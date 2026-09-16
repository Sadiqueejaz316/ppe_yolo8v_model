"""Assemble dashboard payloads from pipeline outputs. No PPE rule engine here."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.api.labels import PPE_FROM_VIOLATION, violation_label
from src.api.mock import mock_events, mock_live_payload
from src.api.paths import resolve_evidence_file
from src.config.settings import AppConfig, CameraConfig
from src.events.store import EventStore
from src.ops.live import LiveFrameBuffer, LiveStateStore, get_global_frame_buffer


class DashboardService:
    def __init__(
        self,
        config: AppConfig,
        events: EventStore,
        live: LiveStateStore,
        frame_buffer: LiveFrameBuffer | None = None,
    ) -> None:
        self.config = config
        self.events = events
        self.live = live
        self.frame_buffer = frame_buffer if frame_buffer is not None else get_global_frame_buffer()
        self.evidence_root = config.resolve_path(config.evidence.directory)

    @property
    def mock(self) -> bool:
        return bool(self.config.dashboard.mock_data)

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "mock": self.mock,
            "poll_interval_ms": self.config.dashboard.poll_interval_ms,
            "snapshot_poll_interval_ms": getattr(self.config.dashboard, "snapshot_poll_interval_ms", 120),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def cameras(self) -> list[dict[str, Any]]:
        rows = []
        for camera in self.config.cameras:
            live = None if self.mock else self.live.read(camera.id)
            if self.mock:
                live = mock_live_payload() if camera.id == (self.config.cameras[0].id if self.config.cameras else "CAM-001") else None
            rows.append(self._camera_row(camera, live))
        return rows

    def camera(self, camera_id: str) -> dict[str, Any] | None:
        try:
            camera = self.config.camera_by_id(camera_id)
        except KeyError:
            return None
        live = mock_live_payload() if self.mock else self.live.read(camera.id)
        return self._camera_row(camera, live)

    def snapshot_jpeg(self, camera_id: str) -> bytes | None:
        if self.mock:
            return None
        if camera_id not in {c.id for c in self.config.cameras}:
            return None
        if self.frame_buffer is not None:
            buf_jpeg = self.frame_buffer.get_jpeg(camera_id)
            if buf_jpeg is not None:
                return buf_jpeg
        return self.live.jpeg_bytes(camera_id)

    def summary(self) -> dict[str, Any]:
        cameras = self.cameras()
        live_payloads = []
        for camera in self.config.cameras:
            if self.mock:
                live_payloads.append(mock_live_payload())
                break
            payload = self.live.read(camera.id)
            if payload:
                live_payloads.append(payload)

        people: list[dict[str, Any]] = []
        for payload in live_payloads:
            scene = payload.get("scene_summary") or {}
            for person in scene.get("people") or []:
                item = dict(person)
                item["camera_id"] = payload.get("camera_id")
                people.append(item)

        if live_payloads:
            total = sum(int((p.get("scene_summary") or {}).get("total_people") or 0) for p in live_payloads)
            compliant = sum(int((p.get("scene_summary") or {}).get("compliant") or 0) for p in live_payloads)
            timestamp = live_payloads[0].get("timestamp")
            live_available = True
            empty_message = None if people else "Empty scene"
        else:
            total = 0
            compliant = 0
            timestamp = datetime.now(timezone.utc).isoformat()
            live_available = False
            empty_message = "No live camera feed"

        online = sum(1 for row in cameras if row.get("status") in {"ONLINE", "MOCK"})
        active = []
        for person in people:
            if person.get("overall_status") != "NON_COMPLIANT":
                continue
            active.append(
                {
                    "person_id": person.get("person_id"),
                    "camera_id": person.get("camera_id"),
                    "missing_ppe": list(person.get("missing_ppe") or []),
                    "overall_status": person.get("overall_status"),
                    "helmet": person.get("helmet"),
                    "mask": person.get("mask"),
                    "vest": person.get("vest"),
                }
            )

        recent = self.list_events(limit=8)
        return {
            "timestamp": timestamp,
            "mock": self.mock,
            "live_available": live_available,
            "system_ok": (online > 0) or self.mock,
            "message": "DEVELOPMENT MODE — live counts are simulated" if self.mock else empty_message,
            "total_people": total,
            "compliant_people": compliant,
            "violations": total - compliant,
            "cameras_online": online,
            "cameras_total": len(self.config.cameras),
            "active_violations": active,
            "people": people,
            "recent_events": recent,
            "cameras": cameras,
            "poll_interval_ms": self.config.dashboard.poll_interval_ms,
            "snapshot_poll_interval_ms": getattr(self.config.dashboard, "snapshot_poll_interval_ms", 120),
        }

    def list_events(
        self,
        camera_id: str | None = None,
        violation_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if self.mock:
            rows = mock_events()
            if camera_id:
                rows = [row for row in rows if row["camera_id"] == camera_id]
            if violation_type:
                rows = [row for row in rows if row["violation_type"] == violation_type]
            rows = rows[:limit]
        else:
            rows = self.events.list(camera_id=camera_id, violation_type=violation_type, limit=limit)
        live_people = self._live_people_index()
        return [self._event_view(row, live_people) for row in rows]

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        if self.mock:
            row = next((item for item in mock_events() if item["event_id"] == event_id), None)
        else:
            row = self.events.get(event_id)
        if row is None:
            return None
        view = self._event_view(row, self._live_people_index())
        view["ppe"] = row.get("ppe") or view.get("ppe")
        stored = row.get("evidence_path")
        evidence = resolve_evidence_file(
            self.evidence_root,
            event_id,
            stored,
            camera_id=row.get("camera_id"),
            timestamp=row.get("timestamp"),
        )
        view["has_evidence"] = evidence is not None
        view["evidence_url"] = f"/api/evidence/{event_id}" if evidence is not None else None
        return view

    def evidence_file(self, event_id: str) -> Path | None:
        row = self.events.get(event_id)
        if row is None:
            return None
        return resolve_evidence_file(
            self.evidence_root,
            event_id,
            row.get("evidence_path"),
            camera_id=row.get("camera_id"),
            timestamp=row.get("timestamp"),
        )

    def _live_people_index(self) -> dict[tuple[Any, Any], dict[str, Any]]:
        index: dict[tuple[Any, Any], dict[str, Any]] = {}
        for camera in self.config.cameras:
            payload = mock_live_payload() if self.mock else self.live.read(camera.id)
            if not payload:
                continue
            for person in (payload.get("scene_summary") or {}).get("people") or []:
                index[(payload.get("camera_id"), person.get("person_id"))] = person
        return index

    def _event_view(self, row: dict[str, Any], live_people: dict[tuple[Any, Any], dict[str, Any]]) -> dict[str, Any]:
        key = (row.get("camera_id"), row.get("person_id"))
        person = live_people.get(key)
        item = PPE_FROM_VIOLATION.get(str(row.get("violation_type") or ""), "")
        if person is not None:
            missing = set(person.get("missing_ppe") or [])
            status = "Active" if item in missing else "Resolved"
        else:
            status = "Confirmed"
        camera = None
        try:
            camera = self.config.camera_by_id(str(row.get("camera_id")))
        except KeyError:
            pass
        return {
            "event_id": row.get("event_id"),
            "timestamp": row.get("timestamp"),
            "camera_id": row.get("camera_id"),
            "camera_name": camera.name if camera else row.get("camera_id"),
            "location": camera.name if camera else None,
            "zone": row.get("zone"),
            "person_id": row.get("person_id"),
            "violation_type": row.get("violation_type"),
            "violation": violation_label(str(row.get("violation_type") or "")),
            "confidence": row.get("confidence"),
            "status": status,
            "ppe": row.get("ppe") or person,
            "has_evidence": False,
        }

    def _camera_row(self, camera: CameraConfig, live: dict[str, Any] | None) -> dict[str, Any]:
        if self.mock:
            status = "MOCK"
        elif live and live.get("camera_connected"):
            status = "ONLINE"
        elif live:
            status = "OFFLINE"
        else:
            status = "NO LIVE FEED"
        has_snap = False
        if not self.mock:
            if self.frame_buffer is not None and self.frame_buffer.get_jpeg(camera.id) is not None:
                has_snap = True
            elif live and live.get("_has_jpeg"):
                has_snap = True
        return {
            "id": camera.id,
            "name": camera.name,
            "location": camera.name,
            "zone": camera.zone,
            "enabled": camera.enabled,
            "target_fps": camera.target_fps,
            "status": status,
            "camera_fps": live.get("camera_fps") if live else None,
            "inference_fps": live.get("inference_fps") if live else None,
            "has_snapshot": has_snap,
        }
