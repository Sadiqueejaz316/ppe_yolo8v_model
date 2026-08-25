"""Event publisher abstraction. V1 writes locally; later HTTP/Redis/Kafka."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from src.events.violation import PPEViolationEvent
from src.exceptions import EvidenceError

logger = logging.getLogger(__name__)


class EventPublisher(ABC):
    @abstractmethod
    def publish(self, event: PPEViolationEvent) -> None:
        raise NotImplementedError


class LocalEventPublisher(EventPublisher):
    def __init__(self, jsonl_path: Path | None = None) -> None:
        self._jsonl_path = jsonl_path

    def publish(self, event: PPEViolationEvent) -> None:
        logger.warning(
            "PPE_VIOLATION camera=%s person=%s type=%s confidence=%.2f event=%s evidence=%s",
            event.camera_id,
            event.person_id,
            event.violation_type,
            event.confidence,
            event.event_id,
            event.evidence_path or "",
        )
        print(event.summary())
        print()
        if self._jsonl_path is None:
            return
        try:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict()) + "\n")
        except OSError as exc:
            logger.error("EVENT_LOG_FAILED error=%s", exc)
            raise EvidenceError(f"Could not append event log: {exc}") from exc
