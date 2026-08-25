"""Person tracker abstraction.

ByteTrack is the V1 implementation. The rest of the pipeline only sees
``TrackedDetection`` with a temporary ``track_id``. No biometric identity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.config.settings import TrackingConfig
from src.geometry import iou
from src.inference.result import Detection


@dataclass(frozen=True)
class TrackedDetection:
    detection: Detection
    track_id: int

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.detection.bbox

    @property
    def confidence(self) -> float:
        return self.detection.confidence

    @property
    def label(self) -> str:
        return self.detection.label


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: list[Detection], frame: np.ndarray | None = None) -> list[TrackedDetection]:
        raise NotImplementedError

    def reset(self) -> None:
        return None


class NoOpTracker(Tracker):
    """Assigns sequential IDs within a single frame. Useful for still images."""

    def update(self, detections: list[Detection], frame: np.ndarray | None = None) -> list[TrackedDetection]:
        return [
            TrackedDetection(detection=det, track_id=index + 1)
            for index, det in enumerate(detections)
        ]


@dataclass
class _Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    score: float
    hits: int = 0
    time_since_update: int = 0
    state: str = "new"


def _greedy_match(
    tracks: list[_Track],
    detections: list[Detection],
    thresh: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    pairs: list[tuple[float, int, int]] = []
    for ti, track in enumerate(tracks):
        for di, det in enumerate(detections):
            score = iou(track.bbox, det.bbox)
            if score >= thresh:
                pairs.append((score, ti, di))
    pairs.sort(reverse=True)

    used_t: set[int] = set()
    used_d: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        matches.append((ti, di))
    unmatched_t = [i for i in range(len(tracks)) if i not in used_t]
    unmatched_d = [i for i in range(len(detections)) if i not in used_d]
    return matches, unmatched_t, unmatched_d


class ByteTracker(Tracker):
    """Lightweight ByteTrack-style two-stage IoU tracker for person boxes.

    Tracking IDs are temporary analytics IDs only. They are not worker identities.
    """

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self._config = config or TrackingConfig()
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._frame_id = 0

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1
        self._frame_id = 0

    def update(self, detections: list[Detection], frame: np.ndarray | None = None) -> list[TrackedDetection]:
        self._frame_id += 1
        cfg = self._config
        for track in self._tracks:
            track.time_since_update += 1

        high = [det for det in detections if det.confidence >= cfg.track_high_thresh]
        low = [
            det
            for det in detections
            if cfg.track_low_thresh <= det.confidence < cfg.track_high_thresh
        ]

        confirmed = [t for t in self._tracks if t.state == "tracked"]
        lost = [t for t in self._tracks if t.state != "tracked"]
        pool = confirmed + lost

        matches, unmatched_tracks, unmatched_high = _greedy_match(pool, high, cfg.match_thresh)
        for ti, di in matches:
            self._apply_update(pool[ti], high[di])

        remaining_tracks = [pool[i] for i in unmatched_tracks if pool[i].state == "tracked"]
        matches2, unmatched_tracks2, _ = _greedy_match(remaining_tracks, low, min(0.5, cfg.match_thresh))
        for ti, di in matches2:
            self._apply_update(remaining_tracks[ti], low[di])

        matched_remaining = {remaining_tracks[ti].track_id for ti, _ in matches2}
        for track in remaining_tracks:
            if track.track_id not in matched_remaining:
                track.state = "lost"

        for di in unmatched_high:
            det = high[di]
            if det.confidence < cfg.new_track_thresh:
                continue
            self._tracks.append(
                _Track(
                    track_id=self._next_id,
                    bbox=det.bbox,
                    score=det.confidence,
                    hits=1,
                    time_since_update=0,
                    state="tracked",
                )
            )
            self._next_id += 1

        alive: list[_Track] = []
        for track in self._tracks:
            if track.time_since_update > cfg.track_buffer:
                continue
            alive.append(track)
        self._tracks = alive

        output: list[TrackedDetection] = []
        for track in self._tracks:
            if track.state != "tracked":
                continue
            if track.hits < cfg.min_hits:
                continue
            detection = Detection(
                class_id=-1,
                label="person",
                confidence=track.score,
                bbox=track.bbox,
                track_id=track.track_id,
            )
            output.append(TrackedDetection(detection=detection, track_id=track.track_id))
        return output

    def _apply_update(self, track: _Track, detection: Detection) -> None:
        track.bbox = detection.bbox
        track.score = detection.confidence
        track.hits += 1
        track.time_since_update = 0
        track.state = "tracked"
