import concurrent.futures
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading

import numpy as np
import pytest

from src.config.settings import EvidenceConfig
from src.events.evidence_dedup import EvidenceDeduplicator, normalize_violation_signature
from src.events.violation import PPEViolationEvent
from src.evidence.capture import EvidenceCapture


def _event(
    camera_id: str = "CAM-001",
    person_id: int = 17,
    missing: str = "helmet",
    ts: datetime | None = None,
    bbox: tuple[float, float, float, float] | None = (100.0, 100.0, 200.0, 400.0),
) -> PPEViolationEvent:
    return PPEViolationEvent.create(
        camera_id=camera_id,
        timestamp=ts or datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc),
        person_id=person_id,
        missing_ppe=missing,
        confidence=0.9,
        bbox=bbox,
    )


# ---------------------------------------------------------------------------
# Core 12 Required Test Cases
# ---------------------------------------------------------------------------

def test_1_first_violation_captures():
    dedup = EvidenceDeduplicator(cooldown_seconds=30.0)
    t100 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    allowed = dedup.check_and_reserve("CAM-001", 17, "helmet", t100)
    assert allowed is True


def test_2_immediate_duplicate_suppressed():
    dedup = EvidenceDeduplicator(cooldown_seconds=30.0)
    t100 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    t105 = t100 + timedelta(seconds=5)

    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t100) is True
    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t105) is False


def test_3_twenty_nine_seconds_suppressed():
    dedup = EvidenceDeduplicator(cooldown_seconds=30.0)
    t100 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    t129 = t100 + timedelta(seconds=29)

    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t100) is True
    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t129) is False


def test_4_thirty_seconds_allowed():
    dedup = EvidenceDeduplicator(cooldown_seconds=30.0, repeat_active_violations=True)
    t100 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    t130 = t100 + timedelta(seconds=30)

    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t100) is True
    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t130) is True


def test_5_different_worker_allowed():
    dedup = EvidenceDeduplicator(cooldown_seconds=30.0)
    t100 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    t105 = t100 + timedelta(seconds=5)

    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t100) is True
    assert dedup.check_and_reserve("CAM-001", 18, "helmet", t105) is True


def test_6_additional_ppe_item_suppressed_as_composite():
    dedup = EvidenceDeduplicator(cooldown_seconds=30.0)
    t100 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    t104 = t100 + timedelta(seconds=4)

    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t100) is True
    # Vest confirms a few seconds later; same worker stays on cooldown.
    assert dedup.check_and_reserve("CAM-001", 17, "safety_vest", t104) is False
    record = next(iter(dedup._records.values()))
    assert record.violation_signature == "helmet+safety_vest"


def test_7_multiple_missing_ppe_produces_one_capture(tmp_path):
    config = EvidenceConfig(directory=str(tmp_path / "evidence"), cooldown_seconds=30.0)
    capture = EvidenceCapture(config, project_root=Path(tmp_path))
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    t100 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)

    # In a real frame, Worker 17 has both helmet and vest missing
    event_helmet = _event(person_id=17, missing="helmet", ts=t100)
    event_vest = _event(person_id=17, missing="safety_vest", ts=t100)

    # Consolidated capture for worker 17:
    signature = ["helmet", "safety_vest"]
    path1 = capture.save(frame, event_helmet, signature=signature)
    assert path1.path is not None
    assert path1.is_new is True

    path2 = capture.save(frame, event_vest, signature=signature)
    assert path2.path == path1.path
    assert path2.is_new is False

    saved_images = list((tmp_path / "evidence").rglob("*.jpg"))
    assert len(saved_images) == 1


def test_8_ppe_ordering_is_normalized():
    sig1 = normalize_violation_signature(["helmet", "vest"])
    sig2 = normalize_violation_signature(["vest", "helmet"])
    sig3 = normalize_violation_signature("helmet+safety_vest")
    sig4 = normalize_violation_signature("vest+helmet")

    assert sig1 == sig2 == sig3 == sig4 == "helmet+safety_vest"


def test_9_resolve_does_not_bypass_cooldown():
    dedup = EvidenceDeduplicator(cooldown_seconds=30.0)
    t100 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    t120 = t100 + timedelta(seconds=20)
    t130 = t100 + timedelta(seconds=30)

    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t100) is True
    dedup.resolve_violation("CAM-001", 17, "helmet")
    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t120) is False
    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t130) is True


def test_10_track_id_changes_spatial_continuity():
    dedup = EvidenceDeduplicator(cooldown_seconds=30.0, spatial_iou_threshold=0.4)
    t100 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    t103 = t100 + timedelta(seconds=3)

    bbox17 = (100.0, 100.0, 200.0, 400.0)
    # Track 21 appears at virtually the same coordinates (IoU > 0.8)
    bbox21 = (102.0, 101.0, 201.0, 402.0)

    # Worker 17 captures evidence at t=100
    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t100, bbox=bbox17) is True

    # Tracker flickers and assigns Track ID 21 to the same physical person at t=103
    # Spatial continuity should recognize this as the same person and suppress!
    assert dedup.check_and_reserve("CAM-001", 21, "helmet", t103, bbox=bbox21) is False


def test_11_multiple_cameras():
    dedup = EvidenceDeduplicator(cooldown_seconds=30.0)
    t100 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    t105 = t100 + timedelta(seconds=5)

    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t100) is True
    # CAM-002 is an independent context
    assert dedup.check_and_reserve("CAM-002", 17, "helmet", t105) is True


def test_12_concurrent_calls():
    dedup = EvidenceDeduplicator(cooldown_seconds=30.0)
    t100 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)

    results = []

    def try_reserve():
        res = dedup.check_and_reserve("CAM-001", 17, "helmet", t100)
        results.append(res)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(try_reserve) for _ in range(16)]
        concurrent.futures.wait(futures)

    # Exactly ONE thread should succeed in reserving the capture slot
    assert results.count(True) == 1
    assert results.count(False) == 15


# ---------------------------------------------------------------------------
# Additional Edge Cases
# ---------------------------------------------------------------------------

def test_prune_removes_expired_records():
    dedup = EvidenceDeduplicator(cooldown_seconds=10.0)
    t0 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    dedup.check_and_reserve("CAM-001", 1, "helmet", t0)
    dedup.check_and_reserve("CAM-001", 2, "helmet", t0)

    # Pruning 100 seconds later should prune both
    t_later = t0 + timedelta(seconds=100)
    pruned = dedup.prune(t_later)
    assert pruned == 2
    assert len(dedup._records) == 0


def test_staggered_items_share_one_capture(tmp_path):
    config = EvidenceConfig(directory=str(tmp_path / "evidence"), cooldown_seconds=30.0)
    capture = EvidenceCapture(config, project_root=Path(tmp_path))
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    t0 = datetime(2026, 9, 9, 14, 32, 1, tzinfo=timezone.utc)
    t3 = t0 + timedelta(seconds=3)

    path1 = capture.save(frame, _event(person_id=17, missing="helmet", ts=t0))
    path2 = capture.save(frame, _event(person_id=17, missing="safety_vest", ts=t3))

    assert path1.path is not None
    assert path1.is_new is True
    assert path2.path == path1.path
    assert path2.is_new is False
    assert len(list((tmp_path / "evidence").rglob("*.jpg"))) == 1


def test_partial_ppe_resolve_keeps_cooldown():
    dedup = EvidenceDeduplicator(cooldown_seconds=30.0)
    t100 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    t105 = t100 + timedelta(seconds=5)

    assert dedup.check_and_reserve("CAM-001", 17, ["helmet", "safety_vest"], t100) is True
    dedup.resolve_violation("CAM-001", 17, "safety_vest")
    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t105) is False


def test_spatial_iou_quarter_suppresses_new_track():
    dedup = EvidenceDeduplicator(cooldown_seconds=30.0, spatial_iou_threshold=0.2)
    t100 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    t103 = t100 + timedelta(seconds=3)
    # IoU = 0.25: 100x100 boxes overlapping by 40px on x
    bbox_a = (0.0, 0.0, 100.0, 100.0)
    bbox_b = (60.0, 0.0, 160.0, 100.0)
    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t100, bbox=bbox_a) is True
    assert dedup.check_and_reserve("CAM-001", 21, "helmet", t103, bbox=bbox_b) is False


def test_spatial_center_inside_suppresses_new_track():
    dedup = EvidenceDeduplicator(cooldown_seconds=30.0, spatial_iou_threshold=0.2)
    t100 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    t103 = t100 + timedelta(seconds=3)
    bbox_a = (0.0, 0.0, 200.0, 400.0)
    bbox_b = (40.0, 40.0, 90.0, 120.0)
    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t100, bbox=bbox_a) is True
    assert dedup.check_and_reserve("CAM-001", 99, "helmet", t103, bbox=bbox_b) is False


def test_far_boxes_are_distinct_workers():
    dedup = EvidenceDeduplicator(cooldown_seconds=30.0, spatial_iou_threshold=0.2)
    t100 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    bbox_a = (0.0, 0.0, 40.0, 80.0)
    bbox_b = (200.0, 0.0, 240.0, 80.0)
    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t100, bbox=bbox_a) is True
    assert dedup.check_and_reserve("CAM-001", 18, "helmet", t100, bbox=bbox_b) is True


def test_present_flicker_does_not_reopen_cooldown():
    dedup = EvidenceDeduplicator(cooldown_seconds=30.0)
    t100 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    t105 = t100 + timedelta(seconds=5)

    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t100) is True
    dedup.resolve_violation("CAM-001", 17, "helmet")
    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t105) is False


def test_cooldown_zero_disables_deduplication():
    dedup = EvidenceDeduplicator(cooldown_seconds=0.0)
    t100 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    t101 = t100 + timedelta(seconds=1)

    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t100) is True
    assert dedup.check_and_reserve("CAM-001", 17, "helmet", t101) is True
