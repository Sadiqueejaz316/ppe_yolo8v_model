"""Save annotated evidence only for confirmed violations."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from src.config.settings import EvidenceConfig
from src.events.evidence_dedup import EvidenceDeduplicator
from src.events.violation import PPEViolationEvent
from src.exceptions import EvidenceError

logger = logging.getLogger(__name__)


EVIDENCE_NEW = "new"
EVIDENCE_REUSED = "reused"
EVIDENCE_SUPPRESSED = "suppressed"


@dataclass(frozen=True)
class EvidenceSaveResult:
    """Outcome of an evidence capture attempt.

    ``new`` is the only status that may publish an event or insert a dashboard
    row. ``reused`` points at the JPEG of the still-active episode, ``suppressed``
    means the episode is on cooldown with no image to show.
    """

    path: str | None
    status: str

    @property
    def is_new(self) -> bool:
        return self.status == EVIDENCE_NEW

    @property
    def is_reused(self) -> bool:
        return self.status == EVIDENCE_REUSED

    @property
    def is_suppressed(self) -> bool:
        return self.status == EVIDENCE_SUPPRESSED


class EvidenceCapture:
    def __init__(
        self,
        config: EvidenceConfig,
        project_root: Path,
        deduplicator: EvidenceDeduplicator | None = None,
    ) -> None:
        self._config = config
        self._root = (project_root / config.directory).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        if deduplicator is not None:
            self._deduplicator = deduplicator
        else:
            self._deduplicator = EvidenceDeduplicator(
                cooldown_seconds=getattr(config, "cooldown_seconds", 30.0),
                repeat_active_violations=getattr(config, "repeat_active_violations", False),
                spatial_iou_threshold=getattr(config, "spatial_iou_threshold", 0.2),
            )

    @property
    def deduplicator(self) -> EvidenceDeduplicator:
        return self._deduplicator

    def save(
        self,
        frame: np.ndarray,
        event: PPEViolationEvent,
        signature: str | Sequence[str] | None = None,
        force: bool = False,
    ) -> EvidenceSaveResult:
        sig = signature if signature is not None else event.violation_type
        if not force and self._deduplicator.cooldown_seconds > 0:
            allowed = self._deduplicator.check_and_reserve(
                camera_id=event.camera_id,
                worker_id=event.person_id,
                violation_items=sig,
                timestamp=event.timestamp,
                bbox=event.bbox,
            )
            if not allowed:
                existing = self._deduplicator.evidence_path(event.camera_id, event.person_id)
                return EvidenceSaveResult(
                    path=existing,
                    status=EVIDENCE_REUSED if existing else EVIDENCE_SUPPRESSED,
                )

        day = event.timestamp.strftime("%Y-%m-%d")
        directory = self._root / event.camera_id / day
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("EVIDENCE_MKDIR_FAILED error=%s", exc)
            raise EvidenceError(f"Could not create evidence directory: {exc}") from exc

        path = directory / f"event-{event.event_id}.jpg"
        # Write to a temp file and rename: the dashboard must never read a
        # half-written JPEG while the pipeline is encoding.
        # Extension stays .jpg: cv2 picks the encoder from it.
        tmp_path = path.with_name(f".{path.stem}.tmp.jpg")
        try:
            import cv2

            ok = cv2.imwrite(str(tmp_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self._config.jpeg_quality)])
            if not ok:
                raise OSError("cv2.imwrite returned False")
            os.replace(tmp_path, path)
        except OSError as exc:
            logger.error("EVIDENCE_WRITE_FAILED path=%s error=%s", path, exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise EvidenceError(f"Could not write evidence image: {exc}") from exc

        str_path = str(path)
        self._deduplicator.record_capture(
            camera_id=event.camera_id,
            worker_id=event.person_id,
            violation_items=sig,
            timestamp=event.timestamp,
            evidence_path=str_path,
            bbox=event.bbox,
        )
        logger.info("EVIDENCE_SAVED camera=%s event=%s path=%s", event.camera_id, event.event_id, path)
        return EvidenceSaveResult(path=str_path, status=EVIDENCE_NEW)

    def prune(self, now: datetime | None = None) -> int:
        days = max(0, int(self._config.retention_days))
        if days <= 0:
            return 0
        cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
        removed = 0
        if not self._root.exists():
            return 0
        for file in self._root.rglob("*.jpg"):
            try:
                mtime = datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    file.unlink()
                    removed += 1
            except OSError as exc:
                logger.warning("EVIDENCE_PRUNE_FAILED path=%s error=%s", file, exc)
        if removed:
            logger.info("EVIDENCE_PRUNED count=%s retention_days=%s", removed, days)
        return removed
