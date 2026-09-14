"""Evidence capture deduplication and cooldown mechanism.

Prevents duplicate evidence images for the same worker and violation within
a configurable cooldown window (default 30 seconds), while consolidating
multi-item violations into a single evidence capture and providing
spatial-continuity fallback against ByteTrack ID instability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import threading
from typing import Sequence

from src.geometry import BBox, center, containment_ratio, iou, point_in_bbox

logger = logging.getLogger(__name__)


def normalize_violation_signature(items: str | Sequence[str]) -> str:
    """Normalize and sort PPE violation items into a canonical signature.

    Examples:
        ["helmet", "vest"] -> "helmet+safety_vest"
        ["safety_vest", "helmet"] -> "helmet+safety_vest"
        "HELMET_MISSING" -> "helmet"
        "vest+helmet" -> "helmet+safety_vest"
    """
    raw: list[str]
    if isinstance(items, str):
        raw = [items]
    else:
        raw = [str(p) for p in items if str(p).strip()]
    parts: list[str] = []
    for item in raw:
        cleaned = item.lower().replace("_missing", "").replace(",", "+")
        parts.extend(p.strip() for p in cleaned.split("+") if p.strip())

    normalized: list[str] = []
    for part in parts:
        if part == "vest":
            part = "safety_vest"
        normalized.append(part)

    unique_sorted = sorted(set(normalized))
    return "+".join(unique_sorted) if unique_sorted else "unknown"


@dataclass
class EvidenceRecord:
    camera_id: str
    worker_id: str
    violation_signature: str
    timestamp: datetime
    bbox: BBox | None = None
    evidence_path: str | None = None
    active: bool = True


class EvidenceDeduplicator:
    """Thread-safe evidence deduplicator with per-worker/violation cooldown."""

    def __init__(
        self,
        cooldown_seconds: float = 30.0,
        repeat_active_violations: bool = True,
        spatial_iou_threshold: float = 0.2,
    ) -> None:
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.repeat_active_violations = bool(repeat_active_violations)
        self.spatial_iou_threshold = float(spatial_iou_threshold)
        self._lock = threading.Lock()
        self._records: dict[str, EvidenceRecord] = {}
        self._aliases: dict[tuple[str, str], str] = {}  # (camera_id, alias_id) -> primary_worker_id

    @staticmethod
    def make_key(camera_id: str, worker_id: str | int, signature: str | None = None) -> str:
        """Logical identity: camera + worker (+ optional normalized signature for logs)."""
        base = f"{camera_id}|worker-{worker_id}"
        if signature:
            return f"{base}|{signature}"
        return base

    def _episode_key(self, camera_id: str, worker_id: str) -> str:
        # One evidence episode per worker on a camera; signature lives on the record.
        return self.make_key(camera_id, worker_id)

    def _merge_signature(self, existing: str, incoming: str) -> str:
        return normalize_violation_signature(f"{existing}+{incoming}")

    def evidence_path(self, camera_id: str, worker_id: str | int) -> str | None:
        worker_str = str(worker_id)
        with self._lock:
            primary_id = self._aliases.get((camera_id, worker_str), worker_str)
            record = self._records.get(self._episode_key(camera_id, primary_id))
            return record.evidence_path if record is not None else None

    def check_and_reserve(
        self,
        camera_id: str,
        worker_id: str | int,
        violation_items: str | Sequence[str],
        timestamp: datetime,
        bbox: BBox | None = None,
    ) -> bool:
        """Atomically check if evidence should be captured and reserve the slot if so.

        Cooldown is per (camera_id, worker_id). Additional missing PPE items for the
        same worker merge into the existing composite signature and do not write a
        second image.

        Returns:
            True if evidence capture is allowed (slot reserved).
            False if suppressed by cooldown or spatial continuity.
        """
        signature = normalize_violation_signature(violation_items)
        worker_str = str(worker_id)

        with self._lock:
            # 0. Check worker ID alias (from spatial continuity)
            primary_id = self._aliases.get((camera_id, worker_str), worker_str)
            key = self._episode_key(camera_id, primary_id)

            # 1. Episode for this worker (any PPE signature). Cooldown is strictly
            # time-based: PRESENT flicker / resolve cannot reopen a slot inside 30s.
            record = self._records.get(key)
            if record is not None:
                elapsed = (timestamp - record.timestamp).total_seconds()
                record.violation_signature = self._merge_signature(record.violation_signature, signature)
                if elapsed < self.cooldown_seconds:
                    logger.debug(
                        "EVIDENCE_SUPPRESSED_COOLDOWN camera=%s worker=%s sig=%s elapsed=%.1fs cooldown=%.1fs",
                        camera_id,
                        primary_id,
                        record.violation_signature,
                        elapsed,
                        self.cooldown_seconds,
                    )
                    if bbox is not None:
                        record.bbox = bbox
                    return False
                if record.active and self.repeat_active_violations:
                    record.timestamp = timestamp
                    if bbox is not None:
                        record.bbox = bbox
                    logger.info(
                        "EVIDENCE_REPEAT_ACTIVE_ALLOWED camera=%s worker=%s sig=%s elapsed=%.1fs",
                        camera_id,
                        primary_id,
                        record.violation_signature,
                        elapsed,
                    )
                    return True
                if record.active:
                    return False
                record.active = True
                record.timestamp = timestamp
                record.violation_signature = signature
                if bbox is not None:
                    record.bbox = bbox
                logger.info(
                    "EVIDENCE_NEW_EPISODE_ALLOWED camera=%s worker=%s sig=%s (previously resolved)",
                    camera_id,
                    primary_id,
                    signature,
                )
                return True

            # 2. Spatial continuity fallback for ByteTrack ID switching
            matched = self._best_spatial_match(camera_id, timestamp, bbox)
            if matched is not None:
                existing, overlap = matched
                logger.info(
                    "EVIDENCE_SUPPRESSED_SPATIAL_CONTINUITY camera=%s original_worker=%s new_worker=%s sig=%s iou=%.2f",
                    camera_id,
                    existing.worker_id,
                    worker_str,
                    signature,
                    overlap,
                )
                self._aliases[(camera_id, worker_str)] = existing.worker_id
                if bbox is not None:
                    existing.bbox = bbox
                existing.violation_signature = self._merge_signature(
                    existing.violation_signature, signature
                )
                return False

            # 3. New violation: reserve slot
            self._records[key] = EvidenceRecord(
                camera_id=camera_id,
                worker_id=primary_id,
                violation_signature=signature,
                timestamp=timestamp,
                bbox=bbox,
                active=True,
            )
            return True

    def _spatially_related(self, current: BBox, existing: BBox) -> tuple[bool, float]:
        overlap = iou(current, existing)
        if overlap >= self.spatial_iou_threshold:
            return True, overlap
        if point_in_bbox(center(current), existing) or point_in_bbox(center(existing), current):
            return True, overlap
        if containment_ratio(current, existing) >= 0.5 or containment_ratio(existing, current) >= 0.5:
            return True, overlap
        return False, overlap

    def _best_spatial_match(
        self,
        camera_id: str,
        timestamp: datetime,
        bbox: BBox | None,
    ) -> tuple[EvidenceRecord, float] | None:
        if bbox is None or self.spatial_iou_threshold <= 0:
            return None
        best: tuple[EvidenceRecord, float] | None = None
        for existing in self._records.values():
            if existing.camera_id != camera_id or not existing.active or existing.bbox is None:
                continue
            elapsed = (timestamp - existing.timestamp).total_seconds()
            if elapsed >= self.cooldown_seconds:
                continue
            related, overlap = self._spatially_related(bbox, existing.bbox)
            if not related:
                continue
            if best is None or overlap > best[1]:
                best = (existing, overlap)
        return best

    def record_capture(
        self,
        camera_id: str,
        worker_id: str | int,
        violation_items: str | Sequence[str],
        timestamp: datetime,
        evidence_path: str,
        bbox: BBox | None = None,
    ) -> None:
        """Update recorded capture path for an active record."""
        signature = normalize_violation_signature(violation_items)
        worker_str = str(worker_id)
        with self._lock:
            primary_id = self._aliases.get((camera_id, worker_str), worker_str)
            key = self._episode_key(camera_id, primary_id)
            record = self._records.get(key)
            if record is not None:
                record.evidence_path = evidence_path
                record.timestamp = timestamp
                record.violation_signature = self._merge_signature(record.violation_signature, signature)
                if bbox is not None:
                    record.bbox = bbox
            else:
                self._records[key] = EvidenceRecord(
                    camera_id=camera_id,
                    worker_id=primary_id,
                    violation_signature=signature,
                    timestamp=timestamp,
                    bbox=bbox,
                    evidence_path=evidence_path,
                    active=True,
                )

    def resolve_violation(
        self,
        camera_id: str,
        worker_id: str | int,
        resolved_items: str | Sequence[str],
    ) -> None:
        """Drop resolved items from the episode; clear cooldown only when none remain."""
        signature = normalize_violation_signature(resolved_items)
        resolved_set = {part for part in signature.split("+") if part and part != "unknown"}
        if not resolved_set:
            return
        worker_str = str(worker_id)

        with self._lock:
            primary_id = self._aliases.get((camera_id, worker_str), worker_str)
            record = self._records.get(self._episode_key(camera_id, primary_id))
            if record is None or not record.active:
                return
            record_items = {part for part in record.violation_signature.split("+") if part}
            remaining = record_items - resolved_set
            if remaining == record_items:
                return
            if remaining:
                record.violation_signature = "+".join(sorted(remaining))
                logger.debug(
                    "EVIDENCE_VIOLATION_PARTIAL_RESOLVE camera=%s worker=%s remaining=%s resolved=%s",
                    camera_id,
                    primary_id,
                    record.violation_signature,
                    signature,
                )
                return
            record.active = False
            logger.debug(
                "EVIDENCE_VIOLATION_RESOLVED camera=%s worker=%s sig=%s resolved=%s",
                camera_id,
                primary_id,
                record.violation_signature,
                signature,
            )

    def prune(self, now: datetime | None = None) -> int:
        """Prune old deduplication records to prevent memory growth."""
        cutoff_time = now or datetime.now(timezone.utc)
        max_age = max(60.0, self.cooldown_seconds * 2.5)
        stale_keys: list[str] = []

        with self._lock:
            for key, record in self._records.items():
                elapsed = (cutoff_time - record.timestamp).total_seconds()
                if elapsed >= max_age:
                    stale_keys.append(key)
            for k in stale_keys:
                self._records.pop(k, None)

        return len(stale_keys)

    def reset(self) -> None:
        """Clear all records and aliases."""
        with self._lock:
            self._records.clear()
            self._aliases.clear()
