from datetime import datetime, timedelta, timezone

from src.compliance.rules import ComplianceResult
from src.config.settings import ViolationConfig
from src.events.temporal import TemporalViolationFilter


def _result(person_id: int, missing: tuple[str, ...], present: tuple[str, ...]) -> ComplianceResult:
    return ComplianceResult(
        person_id=person_id,
        compliant=not missing,
        missing_ppe=missing,
        present_ppe=present,
        zone="general",
        confidence=0.92,
    )


def test_one_bad_frame_does_not_emit_violation():
    filt = TemporalViolationFilter(ViolationConfig(confirmation_seconds=2.0, cooldown_seconds=30.0))
    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    events = filt.update(
        [_result(17, ("helmet",), ("mask", "safety_vest"))],
        now,
        "CAM-001",
        {17: (0, 0, 10, 20)},
    )
    assert events == []


def test_persistent_bad_detection_emits_violation():
    filt = TemporalViolationFilter(ViolationConfig(confirmation_seconds=2.0, cooldown_seconds=30.0))
    start = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    boxes = {17: (0.0, 0.0, 10.0, 20.0)}
    missing = [_result(17, ("helmet",), ("mask", "safety_vest"))]
    assert filt.update(missing, start, "CAM-001", boxes) == []
    assert filt.update(missing, start + timedelta(seconds=1.5), "CAM-001", boxes) == []
    events = filt.update(missing, start + timedelta(seconds=2.0), "CAM-001", boxes)
    assert len(events) == 1
    assert events[0].person_id == 17
    assert events[0].violation_type == "HELMET_MISSING"
    assert events[0].camera_id == "CAM-001"


def test_cooldown_emits_only_one_event():
    filt = TemporalViolationFilter(ViolationConfig(confirmation_seconds=2.0, cooldown_seconds=30.0))
    start = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    boxes = {17: (0.0, 0.0, 10.0, 20.0)}
    missing = [_result(17, ("helmet",), ())]
    filt.update(missing, start, "CAM-001", boxes)
    first = filt.update(missing, start + timedelta(seconds=2), "CAM-001", boxes)
    later = []
    for seconds in range(3, 20):
        later.extend(filt.update(missing, start + timedelta(seconds=seconds), "CAM-001", boxes))
    assert len(first) == 1
    assert later == []


def test_recovery_before_confirmation_does_not_emit():
    filt = TemporalViolationFilter(ViolationConfig(confirmation_seconds=2.0, cooldown_seconds=30.0))
    start = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    boxes = {17: (0.0, 0.0, 10.0, 20.0)}
    missing = [_result(17, ("helmet",), ())]
    present = [_result(17, (), ("helmet",))]
    assert filt.update(missing, start, "CAM-001", boxes) == []
    assert filt.update(present, start + timedelta(seconds=1), "CAM-001", boxes) == []
    assert filt.update(missing, start + timedelta(seconds=2.5), "CAM-001", boxes) == []


def test_recovery_after_confirmation_does_not_reemit_during_cooldown():
    filt = TemporalViolationFilter(ViolationConfig(confirmation_seconds=2.0, cooldown_seconds=30.0))
    start = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    boxes = {17: (0.0, 0.0, 10.0, 20.0)}
    missing = [_result(17, ("helmet",), ())]
    present = [_result(17, (), ("helmet",))]
    filt.update(missing, start, "CAM-001", boxes)
    events = filt.update(missing, start + timedelta(seconds=2), "CAM-001", boxes)
    assert len(events) == 1
    assert filt.update(present, start + timedelta(seconds=3), "CAM-001", boxes) == []
    assert filt.update(missing, start + timedelta(seconds=4), "CAM-001", boxes) == []


def test_track_drop_keeps_cooldown():
    filt = TemporalViolationFilter(ViolationConfig(confirmation_seconds=2.0, cooldown_seconds=30.0))
    start = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    boxes = {17: (0.0, 0.0, 10.0, 20.0)}
    missing = [_result(17, ("helmet",), ())]
    filt.update(missing, start, "CAM-001", boxes)
    assert filt.update(missing, start + timedelta(seconds=2), "CAM-001", boxes)
    assert filt.update([], start + timedelta(seconds=2.5), "CAM-001", {}) == []
    later = filt.update(missing, start + timedelta(seconds=3), "CAM-001", boxes)
    assert later == []
