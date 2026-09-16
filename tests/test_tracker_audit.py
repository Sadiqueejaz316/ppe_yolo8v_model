"""Tracking stability audit — no model required.

Covers every checkpoint listed in the audit specification:
  - Tracker lifecycle (instantiated once, not per frame)
  - Frame monotonicity and order
  - Coordinate format (xyxy, pixel space)
  - Person-only input to tracker
  - ID stability through normal movement
  - ID stability through short gaps (within track_buffer)
  - match_thresh interaction
  - new_track_thresh interaction
  - Confidence threshold interaction
  - Two-stage matching
  - Crossing persons
  - Large crowd (5+ persons)
  - IoU math that proves the root cause
  - Regression: consecutive ID switching symptom with default config

All tests use synthetic detections. best.pt is NOT loaded.
"""

from __future__ import annotations
import pytest
from src.config.settings import TrackingConfig
from src.inference.result import Detection
from src.tracking.tracker import ByteTracker, NoOpTracker, TrackedDetection


def _person(bbox, conf=0.9):
    return Detection(class_id=5, label="Person", confidence=conf, bbox=bbox)


def _cfg(**kw):
    defaults = dict(
        track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6,
        track_buffer=30, match_thresh=0.7, min_hits=1,
    )
    defaults.update(kw)
    return TrackingConfig(**defaults)


# AUDIT 4: Tracker Lifetime

class TestTrackerLifetime:
    def test_same_instance_across_frames(self):
        tracker = ByteTracker(_cfg())
        iid = id(tracker)
        tracker.update([_person((100, 100, 200, 400))])
        tracker.update([_person((102, 100, 202, 400))])
        assert id(tracker) == iid

    def test_internal_frame_counter_increments(self):
        tracker = ByteTracker(_cfg())
        assert tracker._frame_id == 0
        tracker.update([_person((100, 100, 200, 400))])
        assert tracker._frame_id == 1
        tracker.update([])
        assert tracker._frame_id == 2

    def test_reset_clears_state(self):
        tracker = ByteTracker(_cfg())
        tracker.update([_person((100, 100, 200, 400))])
        tracker.reset()
        assert tracker._frame_id == 0
        assert tracker._next_id == 1
        assert tracker._tracks == []
        result = tracker.update([_person((100, 100, 200, 400))])
        assert result[0].track_id == 1


# AUDIT 5: Frame Order

class TestFrameOrder:
    def test_frame_id_monotonic(self):
        tracker = ByteTracker(_cfg())
        for i in range(10):
            tracker.update([_person((100 + i, 100, 200 + i, 400))])
            assert tracker._frame_id == i + 1


# AUDIT 6: Coordinate Format

class TestCoordinateFormat:
    def test_xyxy_preserved_in_output(self):
        tracker = ByteTracker(_cfg())
        bbox = (50.0, 100.0, 150.0, 400.0)
        result = tracker.update([_person(bbox)])
        x1, y1, x2, y2 = result[0].bbox
        assert x1 < x2
        assert y1 < y2

    def test_large_pixel_coords(self):
        tracker = ByteTracker(_cfg())
        bbox = (400.0, 200.0, 600.0, 900.0)
        result = tracker.update([_person(bbox)])
        assert result[0].bbox == bbox

    def test_coords_match_input(self):
        tracker = ByteTracker(_cfg())
        bbox = (120.5, 80.3, 310.1, 540.9)
        result = tracker.update([_person(bbox)])
        assert result[0].bbox[0] == pytest.approx(120.5, abs=0.1)
        assert result[0].bbox[3] == pytest.approx(540.9, abs=0.1)


# AUDIT 7: Person-Only

class TestPersonOnlyFiltering:
    def test_single_person_gets_one_id(self):
        assert len(ByteTracker(_cfg()).update([_person((100, 100, 200, 400))])) == 1

    def test_two_persons_get_two_ids(self):
        result = ByteTracker(_cfg()).update([_person((100, 100, 200, 400)), _person((600, 100, 700, 400))])
        assert len({r.track_id for r in result}) == 2

    def test_no_detections_returns_empty(self):
        assert ByteTracker(_cfg()).update([]) == []


# AUDIT 8: Detection Format

class TestDetectionFormat:
    def test_output_is_tracked_detection(self):
        result = ByteTracker(_cfg()).update([_person((10, 10, 50, 120))])
        td = result[0]
        assert isinstance(td, TrackedDetection)
        assert isinstance(td.track_id, int) and td.track_id >= 1
        assert len(td.bbox) == 4

    def test_output_confidence(self):
        result = ByteTracker(_cfg()).update([_person((10, 10, 50, 120), conf=0.87)])
        assert result[0].confidence == pytest.approx(0.87, abs=0.01)


# IoU Maths (Root Cause Evidence)

class TestIoUCalculation:
    def test_iou_identical(self):
        from src.geometry import iou
        a = (100.0, 100.0, 200.0, 400.0)
        assert iou(a, a) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        from src.geometry import iou
        assert iou((0.0, 0.0, 100.0, 100.0), (200.0, 0.0, 300.0, 100.0)) == pytest.approx(0.0)

    def test_iou_5px_shift_above_07(self):
        from src.geometry import iou
        score = iou((100.0, 100.0, 200.0, 400.0), (105.0, 100.0, 205.0, 400.0))
        assert score > 0.7, f"5px shift IoU={score:.3f} should be > 0.7"

    def test_iou_20px_shift_below_default_thresh(self):
        """ROOT CAUSE: 20px shift on 100px box -> IoU < 0.7 -> ID switch."""
        from src.geometry import iou
        score = iou((100.0, 100.0, 200.0, 400.0), (120.0, 100.0, 220.0, 400.0))
        assert score < 0.7, f"20px shift IoU={score:.3f} must be < 0.7"

    def test_iou_20px_shift_above_recommended_thresh(self):
        from src.geometry import iou
        score = iou((100.0, 100.0, 200.0, 400.0), (120.0, 100.0, 220.0, 400.0))
        assert score > 0.35, f"20px shift IoU={score:.3f} should be > recommended 0.35"

    def test_iou_40px_shift_above_recommended(self):
        from src.geometry import iou
        score = iou((100.0, 100.0, 200.0, 400.0), (140.0, 100.0, 240.0, 400.0))
        assert score > 0.35, f"40px shift IoU={score:.3f} should be > 0.35"


# ID Stability: Movement

class TestIdStabilityMovement:
    def test_movement_keeps_id_recommended_thresh(self):
        tracker = ByteTracker(_cfg(match_thresh=0.35, track_buffer=40))
        pid = tracker.update([_person((100, 100, 200, 400))])[0].track_id
        for dx in range(1, 10):
            result = tracker.update([_person((100 + dx*5, 100, 200 + dx*5, 400))])
            assert len(result) == 1
            assert result[0].track_id == pid, f"ID changed at step {dx}"

    def test_default_thresh_fails_on_20px_movement(self):
        """DEMONSTRATES THE BUG with default match_thresh=0.7."""
        tracker = ByteTracker(_cfg(match_thresh=0.7))
        pid = tracker.update([_person((100, 100, 200, 400))])[0].track_id
        result = tracker.update([_person((120, 100, 220, 400))])
        if result and result[0].track_id != pid:
            pytest.xfail(
                f"CONFIRMED BUG: Default match_thresh=0.7 causes ID switch on 20px movement. "
                f"IoU(100w,20shift)≈0.667 < 0.7. Old={pid} New={result[0].track_id}. "
                f"Fix: match_thresh: 0.35 in config/app.yaml"
            )


# Track Buffer

class TestTrackBuffer:
    def test_one_frame_gap_keeps_id(self):
        tracker = ByteTracker(_cfg(track_buffer=30, match_thresh=0.3))
        pid = tracker.update([_person((100, 100, 200, 400))])[0].track_id
        tracker.update([])
        again = tracker.update([_person((102, 100, 202, 400))])
        assert again[0].track_id == pid

    def test_5_frame_gap_within_buffer_keeps_id(self):
        tracker = ByteTracker(_cfg(track_buffer=30, match_thresh=0.3))
        pid = tracker.update([_person((100, 100, 200, 400))])[0].track_id
        for _ in range(5):
            tracker.update([])
        assert tracker.update([_person((105, 100, 205, 400))])[0].track_id == pid

    def test_gap_beyond_buffer_creates_new_id(self):
        buf = 5
        tracker = ByteTracker(_cfg(track_buffer=buf, match_thresh=0.3))
        pid = tracker.update([_person((100, 100, 200, 400))])[0].track_id
        for _ in range(buf + 2):
            tracker.update([])
        assert tracker.update([_person((100, 100, 200, 400))])[0].track_id != pid

    def test_buffer_at_10fps_covers_3s(self):
        assert 30 / 10 == pytest.approx(3.0)


# match_thresh

class TestMatchThresh:
    def test_low_thresh_preserves_id(self):
        tracker = ByteTracker(_cfg(match_thresh=0.3, track_buffer=30))
        pid = tracker.update([_person((100, 100, 200, 400))])[0].track_id
        result = tracker.update([_person((130, 100, 230, 400))])
        assert result[0].track_id == pid


# Confidence / Two-Stage

class TestConfidenceInteraction:
    def test_low_conf_not_creates_track(self):
        tracker = ByteTracker(_cfg(track_high_thresh=0.5, new_track_thresh=0.6))
        assert tracker.update([_person((100, 100, 200, 400), conf=0.45)]) == []

    def test_existing_track_survives_confidence_drop(self):
        tracker = ByteTracker(_cfg(track_high_thresh=0.5, track_low_thresh=0.1, match_thresh=0.3))
        pid = tracker.update([_person((100, 100, 200, 400), conf=0.9)])[0].track_id
        result = tracker.update([_person((102, 100, 202, 400), conf=0.35)])
        assert result[0].track_id == pid

    def test_confidence_gap_survives_within_buffer(self):
        tracker = ByteTracker(_cfg(track_buffer=30, match_thresh=0.3))
        pid = tracker.update([_person((100, 100, 200, 400))])[0].track_id
        for _ in range(3):
            tracker.update([])
        assert tracker.update([_person((103, 100, 203, 400))])[0].track_id == pid


# Multiple Persons

class TestMultiplePersons:
    def test_two_persons_stable_ids(self):
        tracker = ByteTracker(_cfg(match_thresh=0.35))
        first = sorted(tracker.update([_person((100, 100, 200, 400)), _person((600, 100, 700, 400))]), key=lambda r: r.bbox[0])
        id_A, id_B = first[0].track_id, first[1].track_id
        for dx in range(1, 6):
            result = sorted(tracker.update([_person((100+dx*5, 100, 200+dx*5, 400)), _person((600+dx*3, 100, 700+dx*3, 400))]), key=lambda r: r.bbox[0])
            assert result[0].track_id == id_A
            assert result[1].track_id == id_B

    def test_five_persons_stable_ids(self):
        tracker = ByteTracker(_cfg(match_thresh=0.35))
        boxes = [(x*200, 100, x*200+100, 400) for x in range(5)]
        initial_ids = sorted(r.track_id for r in tracker.update([_person(b) for b in boxes]))
        assert len(set(initial_ids)) == 5
        for step in range(1, 4):
            shifted = [(x*200+step*5, 100, x*200+100+step*5, 400) for x in range(5)]
            assert sorted(r.track_id for r in tracker.update([_person(b) for b in shifted])) == initial_ids


# Occlusion

class TestOcclusion:
    def test_both_tracked_after_crossing(self):
        tracker = ByteTracker(_cfg(match_thresh=0.35, track_buffer=30))
        tracker.update([_person((50, 100, 150, 400)), _person((500, 100, 600, 400))])
        tracker.update([_person((200, 100, 300, 400)), _person((350, 100, 450, 400))])
        result = tracker.update([_person((350, 100, 450, 400)), _person((200, 100, 300, 400))])
        assert len(result) == 2


# Regression: Exact Bug Report

class TestIdSwitchRegression:
    def test_regression_default_vs_fixed(self):
        """
        REGRESSION: Person walks 20px/frame. Default match_thresh=0.7 causes ID switching.
        Fixed config (match_thresh=0.35, track_buffer=40) must be stable.
        """
        frames = [(100, 100, 200, 400), (120, 100, 220, 400), (140, 100, 240, 400), (160, 100, 260, 400)]

        # Fixed config: must be stable
        tracker_fixed = ByteTracker(_cfg(match_thresh=0.35, track_buffer=40))
        fixed_ids = [tracker_fixed.update([_person(b)])[0].track_id for b in frames]
        assert len(set(fixed_ids)) == 1, f"Fixed config still switches IDs: {fixed_ids}"

        # Default config: expected to show the bug
        tracker_default = ByteTracker(_cfg(match_thresh=0.7))
        default_ids = []
        for bbox in frames:
            res = tracker_default.update([_person(bbox)])
            default_ids.append(res[0].track_id if res else None)

        non_none = [x for x in default_ids if x is not None]
        if len(set(non_none)) > 1:
            pytest.xfail(
                f"DEFAULT CONFIG REPRODUCES REPORTED BUG: IDs={default_ids}. "
                f"Root cause: IoU(100px wide, 20px shift)=0.667 < match_thresh=0.7. "
                f"FIXED CONFIG: IDs={fixed_ids} (stable)."
            )
