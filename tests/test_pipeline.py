from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from src.config.settings import load_settings
from src.inference.detector import Detector
from src.inference.result import Detection, DetectionResult, LatencyBreakdown
from src.pipeline import PPEPipeline
from src.video.source import VideoFrame


CLASS_NAMES = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    7: "Safety Vest",
}


class FakeDetector(Detector):
    def __init__(self, frames: list[list[Detection]] | None = None) -> None:
        self._frames = list(frames or [])
        self.calls = 0

    @property
    def class_names(self) -> dict[int, str]:
        return dict(CLASS_NAMES)

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def model_path(self) -> Path:
        return Path("models/best.pt")

    def predict(self, frame: np.ndarray) -> DetectionResult:
        self.calls += 1
        if self._frames:
            dets = self._frames.pop(0)
        else:
            dets = []
        return DetectionResult(
            detections=list(dets),
            latency=LatencyBreakdown(inference_ms=4.0),
            device="cpu",
            image_size=(int(frame.shape[1]), int(frame.shape[0])),
        )


def _frame(ts: datetime, index: int = 1) -> VideoFrame:
    return VideoFrame(
        image=np.zeros((80, 80, 3), dtype=np.uint8),
        timestamp=ts,
        camera_id="CAM-001",
        frame_index=index,
        source_fps=10.0,
        camera_name="test",
    )


def _person_all_ppe() -> list[Detection]:
    return [
        Detection(5, "Person", 0.9, (10, 10, 50, 70)),
        Detection(0, "Hardhat", 0.9, (18, 10, 38, 22)),
        Detection(7, "Safety Vest", 0.88, (14, 28, 46, 60)),
        Detection(1, "Mask", 0.86, (24, 18, 36, 28)),
    ]


def _person_missing_all() -> list[Detection]:
    return [Detection(5, "Person", 0.9, (10, 10, 50, 70))]


def _person_missing_helmet() -> list[Detection]:
    return [
        Detection(5, "Person", 0.9, (10, 10, 50, 70)),
        Detection(7, "Safety Vest", 0.88, (14, 28, 46, 60)),
        Detection(1, "Mask", 0.86, (24, 18, 36, 28)),
    ]


def test_unknown_zone_raises():
    settings = load_settings()
    camera = replace(settings.cameras[0], zone="does-not-exist")
    with pytest.raises(KeyError, match="does-not-exist"):
        PPEPipeline.build(settings, camera, detector=FakeDetector())


def test_stale_inference_does_not_confirm_violation(tmp_path, monkeypatch):
    settings = load_settings()
    settings = replace(
        settings,
        evidence=replace(settings.evidence, directory=str(tmp_path / "evidence"), events_jsonl=str(tmp_path / "events.jsonl")),
        violations=replace(settings.violations, confirmation_seconds=2.0, cooldown_seconds=30.0),
        visualization=replace(settings.visualization, stable_frames=1),
        project_root=Path(tmp_path),
    )
    camera = settings.cameras[0]
    detector = FakeDetector([_person_missing_helmet(), _person_missing_helmet()])
    pipeline = PPEPipeline.build(settings, camera, detector=detector)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = pipeline.process(_frame(start, 1), infer=True)
    assert first.events == []
    skipped = pipeline.process(_frame(start + timedelta(seconds=3), 2), infer=False)
    assert skipped.events == []
    assert detector.calls == 1
    confirmed = pipeline.process(_frame(start + timedelta(seconds=3), 3), infer=True)
    assert len(confirmed.events) >= 1
    assert any(event.violation_type == "HELMET_MISSING" for event in confirmed.events)


def test_pipeline_emits_evidence_for_confirmed_event(tmp_path):
    settings = load_settings()
    settings = replace(
        settings,
        evidence=replace(settings.evidence, directory=str(tmp_path / "evidence"), events_jsonl=str(tmp_path / "events.jsonl")),
        violations=replace(settings.violations, confirmation_seconds=1.0, cooldown_seconds=30.0),
        visualization=replace(settings.visualization, stable_frames=1),
        project_root=Path(tmp_path),
    )
    detector = FakeDetector([_person_missing_helmet(), _person_missing_helmet()])
    pipeline = PPEPipeline.build(settings, settings.cameras[0], detector=detector)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pipeline.process(_frame(start, 1), infer=True)
    result = pipeline.process(_frame(start + timedelta(seconds=1), 2), infer=True)
    assert result.events
    assert result.events[0].evidence_path
    assert Path(result.events[0].evidence_path).exists()


def test_pipeline_one_event_for_multiple_missing_ppe(tmp_path):
    settings = load_settings()
    settings = replace(
        settings,
        evidence=replace(settings.evidence, directory=str(tmp_path / "evidence"), events_jsonl=str(tmp_path / "events.jsonl")),
        violations=replace(settings.violations, confirmation_seconds=1.0, cooldown_seconds=30.0),
        visualization=replace(settings.visualization, stable_frames=1),
        project_root=Path(tmp_path),
        dashboard=replace(settings.dashboard, sqlite_path=str(tmp_path / "ppe.sqlite"), live_dir=str(tmp_path / "live")),
    )
    detector = FakeDetector([_person_missing_all(), _person_missing_all()])
    pipeline = PPEPipeline.build(settings, settings.cameras[0], detector=detector)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pipeline.process(_frame(start, 1), infer=True)
    result = pipeline.process(_frame(start + timedelta(seconds=1), 2), infer=True)
    assert len(result.events) == 1
    assert result.events[0].evidence_path
    jpgs = list((tmp_path / "evidence").rglob("*.jpg"))
    assert len(jpgs) == 1
    jsonl = Path(tmp_path / "events.jsonl")
    assert jsonl.exists()
    assert len(jsonl.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_pipeline_flicker_compliant_does_not_recapture(tmp_path):
    settings = load_settings()
    settings = replace(
        settings,
        evidence=replace(settings.evidence, directory=str(tmp_path / "evidence"), events_jsonl=str(tmp_path / "events.jsonl")),
        violations=replace(settings.violations, confirmation_seconds=1.0, cooldown_seconds=0.0),
        visualization=replace(settings.visualization, stable_frames=1),
        project_root=Path(tmp_path),
        dashboard=replace(settings.dashboard, sqlite_path=str(tmp_path / "ppe.sqlite"), live_dir=str(tmp_path / "live")),
    )
    detector = FakeDetector(
        [
            _person_missing_helmet(),
            _person_missing_helmet(),
            _person_all_ppe(),
            _person_missing_helmet(),
            _person_missing_helmet(),
        ]
    )
    pipeline = PPEPipeline.build(settings, settings.cameras[0], detector=detector)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pipeline.process(_frame(start, 1), infer=True)
    confirmed = pipeline.process(_frame(start + timedelta(seconds=1), 2), infer=True)
    assert len(confirmed.events) == 1
    pipeline.process(_frame(start + timedelta(seconds=2), 3), infer=True)
    later = pipeline.process(_frame(start + timedelta(seconds=6), 4), infer=True)
    later2 = pipeline.process(_frame(start + timedelta(seconds=7), 5), infer=True)
    assert later.events == []
    assert later2.events == []
    assert len(list((tmp_path / "evidence").rglob("*.jpg"))) == 1


def test_pipeline_scene_summary_is_person_centric(tmp_path):
    settings = load_settings()
    settings = replace(
        settings,
        evidence=replace(settings.evidence, directory=str(tmp_path / "evidence"), events_jsonl=str(tmp_path / "events.jsonl")),
        visualization=replace(settings.visualization, mode="person_summary", stable_frames=1),
        project_root=Path(tmp_path),
    )
    detections = [
        Detection(5, "Person", 0.9, (10, 10, 50, 70)),
        Detection(5, "Person", 0.88, (90, 10, 130, 70)),
        Detection(0, "Hardhat", 0.9, (18, 10, 38, 22)),
        Detection(1, "Mask", 0.86, (24, 18, 36, 28)),
        Detection(7, "Safety Vest", 0.88, (14, 28, 46, 60)),
        Detection(2, "NO-Hardhat", 0.84, (100, 10, 120, 22)),
        Detection(1, "Mask", 0.82, (104, 18, 116, 28)),
        Detection(4, "NO-Safety Vest", 0.8, (94, 28, 126, 60)),
    ]
    pipeline = PPEPipeline.build(
        settings,
        settings.cameras[0],
        detector=FakeDetector([detections]),
        enable_tracking=False,
    )
    result = pipeline.process(_frame(datetime(2026, 1, 1, tzinfo=timezone.utc)), infer=True)
    assert result.scene_summary["total_people"] == 2
    people = {item["person_id"]: item for item in result.scene_summary["people"]}
    assert {item.helmet_state() for item in result.persons} <= {"PRESENT", "MISSING", "UNKNOWN"}
    assert any(item["overall_status"] == "NON_COMPLIANT" for item in people.values())
    from src.viz import build_overlay_plan

    plan = build_overlay_plan(
        result.detections,
        result.persons,
        result.compliance,
        pipeline.taxonomy,
        pipeline.compliance.required_ppe,
        visualization=settings.visualization,
        image_shape=(80, 80),
    )
    assert plan.raw_boxes == ()
    assert len(plan.persons) == 2
