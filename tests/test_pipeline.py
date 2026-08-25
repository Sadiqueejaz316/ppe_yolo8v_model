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
