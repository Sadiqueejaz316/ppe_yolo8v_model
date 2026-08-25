from src.security import redact_rtsp_url
from src.tracking.tracker import ByteTracker
from src.config.settings import TrackingConfig
from src.inference.result import Detection


def test_redact_rtsp_credentials():
    url = "rtsp://username:password@192.168.1.100:554/stream"
    redacted = redact_rtsp_url(url)
    assert "password" not in redacted
    assert "username" not in redacted
    assert "192.168.1.100" in redacted
    assert redacted.startswith("rtsp://***:***@")


def test_byte_tracker_assigns_stable_ids():
    tracker = ByteTracker(TrackingConfig(track_high_thresh=0.4, new_track_thresh=0.4, match_thresh=0.3, min_hits=1))
    det = Detection(1, "Person", 0.9, (10, 10, 50, 120))
    first = tracker.update([det])
    second = tracker.update([Detection(1, "Person", 0.88, (12, 12, 52, 122))])
    assert first and second
    assert first[0].track_id == second[0].track_id


def test_byte_tracker_two_persons_keep_distinct_ids():
    tracker = ByteTracker(TrackingConfig(track_high_thresh=0.4, new_track_thresh=0.4, match_thresh=0.3, min_hits=1))
    left = Detection(1, "Person", 0.9, (10, 10, 50, 120))
    right = Detection(1, "Person", 0.9, (200, 10, 240, 120))
    first = tracker.update([left, right])
    second = tracker.update(
        [Detection(1, "Person", 0.88, (12, 12, 52, 122)), Detection(1, "Person", 0.88, (202, 12, 242, 122))]
    )
    assert {item.track_id for item in first} == {item.track_id for item in second}
    assert len({item.track_id for item in first}) == 2


def test_byte_tracker_empty_detections_returns_empty():
    tracker = ByteTracker(TrackingConfig(min_hits=1))
    tracker.update([Detection(1, "Person", 0.9, (10, 10, 50, 120))])
    assert tracker.update([]) == []


def test_byte_tracker_person_disappears():
    tracker = ByteTracker(TrackingConfig(track_high_thresh=0.4, new_track_thresh=0.4, match_thresh=0.3, min_hits=1, track_buffer=2))
    det = Detection(1, "Person", 0.9, (10, 10, 50, 120))
    first = tracker.update([det])
    assert first
    tracker.update([])
    tracker.update([])
    tracker.update([])
    gone = tracker.update([])
    assert gone == []


def test_byte_tracker_person_reappears_keeps_id_within_buffer():
    tracker = ByteTracker(
        TrackingConfig(
            track_high_thresh=0.4,
            new_track_thresh=0.4,
            match_thresh=0.3,
            min_hits=1,
            track_buffer=5,
        )
    )
    det = Detection(1, "Person", 0.9, (10, 10, 50, 120))
    first = tracker.update([det])
    assert first
    person_id = first[0].track_id
    assert tracker.update([]) == []
    again = tracker.update([Detection(1, "Person", 0.88, (12, 11, 51, 121))])
    assert len(again) == 1
    assert again[0].track_id == person_id
