"""Save annotated evidence only for confirmed violations."""

from __future__ import annotations

import logging
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


@dataclass(frozen=True)
class EvidenceSaveResult:
    path: str | None
    is_new: bool


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
                repeat_active_violations=getattr(config, "repeat_active_violations", True),
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
                return EvidenceSaveResult(
                    path=self._deduplicator.evidence_path(event.camera_id, event.person_id),
                    is_new=False,
                )

        day = event.timestamp.strftime("%Y-%m-%d")
        directory = self._root / event.camera_id / day
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("EVIDENCE_MKDIR_FAILED error=%s", exc)
            raise EvidenceError(f"Could not create evidence directory: {exc}") from exc

        path = directory / f"event-{event.event_id}.jpg"
        try:
            import cv2

            ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self._config.jpeg_quality)])
            if not ok:
                raise OSError("cv2.imwrite returned False")
        except OSError as exc:
            logger.error("EVIDENCE_WRITE_FAILED path=%s error=%s", path, exc)
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
        return EvidenceSaveResult(path=str_path, is_new=True)

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
