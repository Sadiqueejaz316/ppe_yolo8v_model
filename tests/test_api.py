from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.paths import resolve_evidence_file
from src.config.settings import load_settings
from src.events.store import EventStore
from src.events.violation import PPEViolationEvent
from src.ops.live import LiveStateStore


def _app(tmp_path: Path, mock: bool = False, seed_event: bool = False, live: bool = False):
    settings = load_settings()
    settings = replace(
        settings,
        dashboard=replace(
            settings.dashboard,
            sqlite_path=str(tmp_path / "ppe.sqlite"),
            live_dir=str(tmp_path / "live"),
            mock_data=mock,
        ),
        evidence=replace(
            settings.evidence,
            directory=str(tmp_path / "evidence"),
            events_jsonl=str(tmp_path / "events.jsonl"),
        ),
        project_root=Path(tmp_path),
    )
    (tmp_path / "evidence").mkdir(parents=True, exist_ok=True)
    if seed_event:
        store = EventStore(tmp_path / "ppe.sqlite")
        event = PPEViolationEvent.create(
            camera_id="CAM-001",
            timestamp=datetime(2026, 8, 25, 16, 4, 31, tzinfo=timezone.utc),
            person_id=17,
            missing_ppe="mask",
            confidence=0.91,
            zone="general",
        )
        evidence = tmp_path / "evidence" / "CAM-001" / "2026-08-25" / f"event-{event.event_id}.jpg"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_bytes(b"\xff\xd8\xff")
        event = PPEViolationEvent(
            event_id=event.event_id,
            camera_id=event.camera_id,
            timestamp=event.timestamp,
            person_id=event.person_id,
            violation_type=event.violation_type,
            confidence=event.confidence,
            evidence_path=str(evidence),
            zone=event.zone,
            bbox=event.bbox,
        )
        store.insert(
            event,
            ppe_snapshot={
                "person_id": 17,
                "helmet": "COMPLIANT",
                "mask": "MISSING",
                "vest": "COMPLIANT",
                "overall_status": "NON_COMPLIANT",
                "missing_ppe": ["mask"],
            },
        )
    if live:
        LiveStateStore(tmp_path / "live").write(
            "CAM-001",
            {
                "timestamp": "2026-08-25T16:04:31+00:00",
                "camera_id": "CAM-001",
                "camera_name": "Plant Entrance",
                "camera_connected": True,
                "camera_fps": 9.8,
                "inference_fps": 8.0,
                "scene_summary": {
                    "total_people": 2,
                    "compliant": 1,
                    "violations": 1,
                    "required_ppe": ["helmet", "safety_vest", "mask"],
                    "people": [
                        {
                            "person_id": 17,
                            "helmet": "COMPLIANT",
                            "mask": "MISSING",
                            "vest": "COMPLIANT",
                            "overall_status": "NON_COMPLIANT",
                            "missing_ppe": ["mask"],
                            "present_ppe": ["helmet", "safety_vest"],
                        },
                        {
                            "person_id": 24,
                            "helmet": "COMPLIANT",
                            "mask": "COMPLIANT",
                            "vest": "COMPLIANT",
                            "overall_status": "COMPLIANT",
                            "missing_ppe": [],
                            "present_ppe": ["helmet", "mask", "safety_vest"],
                        },
                    ],
                },
            },
            jpeg=b"\xff\xd8\xff",
        )
    return TestClient(create_app(settings))


def test_health_ok(tmp_path):
    client = _app(tmp_path)
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert "poll_interval_ms" in body
    assert "snapshot_poll_interval_ms" in body
    assert body["snapshot_poll_interval_ms"] == 120


def test_dashboard_summary_empty_scene(tmp_path):
    client = _app(tmp_path, mock=False)
    body = client.get("/api/dashboard/summary").json()
    assert body["total_people"] == 0
    assert body["violations"] == 0
    assert body["live_available"] is False
    assert body["mock"] is False
    assert "No live camera feed" in (body["message"] or "")


def test_dashboard_summary_active_violations_from_scene_summary(tmp_path):
    client = _app(tmp_path, live=True)
    body = client.get("/api/dashboard/summary").json()
    assert body["total_people"] == 2
    assert body["compliant_people"] == 1
    assert body["violations"] == 1
    assert body["active_violations"][0]["person_id"] == 17
    assert body["active_violations"][0]["missing_ppe"] == ["mask"]
    assert all("class_id" not in person for person in body["people"])


def test_cameras_do_not_expose_rtsp(tmp_path):
    client = _app(tmp_path, live=True)
    rows = client.get("/api/cameras").json()
    assert rows
    dumped = str(rows)
    assert "rtsp" not in dumped.lower()
    assert "password" not in dumped.lower()
    assert rows[0]["id"] == "CAM-001"
    assert rows[0]["status"] == "ONLINE"


def test_events_empty(tmp_path):
    client = _app(tmp_path)
    assert client.get("/api/events").json() == []


def test_events_and_detail(tmp_path):
    client = _app(tmp_path, seed_event=True)
    rows = client.get("/api/events").json()
    assert len(rows) == 1
    assert rows[0]["violation"] == "Missing Mask"
    event_id = rows[0]["event_id"]
    detail = client.get(f"/api/events/{event_id}").json()
    assert detail["person_id"] == 17
    assert detail["ppe"]["mask"] == "MISSING"
    assert detail["has_evidence"] is True
    image = client.get(f"/api/evidence/{event_id}")
    assert image.status_code == 200
    assert image.content.startswith(b"\xff\xd8")


def test_invalid_event_id(tmp_path):
    client = _app(tmp_path)
    assert client.get("/api/events/not-an-id!").status_code == 400
    assert client.get("/api/events/deadbeefdead").status_code == 404
    assert client.get("/api/evidence/deadbeefdead").status_code == 404


def test_missing_evidence_message_via_detail(tmp_path):
    client = _app(tmp_path)
    store = EventStore(tmp_path / "ppe.sqlite")
    event = PPEViolationEvent.create(
        camera_id="CAM-001",
        timestamp=datetime(2026, 8, 25, tzinfo=timezone.utc),
        person_id=9,
        missing_ppe="helmet",
        confidence=0.5,
    )
    store.insert(event)
    client = _app(tmp_path)
    detail = client.get(f"/api/events/{event.event_id}").json()
    assert detail["has_evidence"] is False
    assert client.get(f"/api/evidence/{event.event_id}").status_code == 404


def test_mock_mode_is_labeled(tmp_path):
    client = _app(tmp_path, mock=True)
    body = client.get("/api/dashboard/summary").json()
    assert body["mock"] is True
    assert body["total_people"] == 12
    assert body["compliant_people"] == 9
    assert body["violations"] == 3
    assert "DEVELOPMENT MODE" in body["message"]
    events = client.get("/api/events").json()
    assert len(events) == 3
    assert client.get("/api/evidence/mock00000017").status_code == 404


def test_evidence_path_traversal_rejected(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"nope")
    assert (
        resolve_evidence_file(root, "abc123abc123", str(secret), camera_id="CAM-001") is None
    )
    assert resolve_evidence_file(root, "abc123abc123", str(root / ".." / "secret.jpg")) is None


def test_event_filter_by_camera_and_type(tmp_path):
    client = _app(tmp_path, seed_event=True)
    assert client.get("/api/events", params={"camera_id": "CAM-001"}).json()
    assert client.get("/api/events", params={"violation_type": "HELMET_MISSING"}).json() == []
    assert client.get("/api/events", params={"camera_id": "bad id"}).status_code == 400


def test_snapshot_from_memory_no_disk(tmp_path):
    from src.ops.live import LiveFrameBuffer

    settings = load_settings()
    settings = replace(
        settings,
        dashboard=replace(
            settings.dashboard,
            sqlite_path=str(tmp_path / "ppe.sqlite"),
            live_dir=str(tmp_path / "live"),
            mock_data=False,
        ),
        project_root=Path(tmp_path),
    )
    buffer = LiveFrameBuffer()
    memory_jpeg = b"\xff\xd8\xff\xe0in_memory_only"
    buffer.update("CAM-001", memory_jpeg, {"test": True})

    live_dir = tmp_path / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    disk_jpg = live_dir / "CAM-001.jpg"
    assert not disk_jpg.exists()

    client = TestClient(create_app(settings, frame_buffer=buffer))
    resp = client.get("/api/cameras/CAM-001/snapshot")
    assert resp.status_code == 200
    assert resp.content == memory_jpeg
    assert not disk_jpg.exists()
