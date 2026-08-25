"""Normalized PPE violation events. Separate from detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4


def _event_id() -> str:
    return uuid4().hex[:12]


@dataclass(frozen=True)
class PPEViolationEvent:
    event_id: str
    camera_id: str
    timestamp: datetime
    person_id: int
    violation_type: str
    confidence: float
    evidence_path: str | None = None
    zone: str = "general"
    bbox: tuple[float, float, float, float] | None = None

    @classmethod
    def create(
        cls,
        camera_id: str,
        timestamp: datetime,
        person_id: int,
        missing_ppe: str,
        confidence: float,
        zone: str = "general",
        bbox: tuple[float, float, float, float] | None = None,
        evidence_path: str | None = None,
    ) -> PPEViolationEvent:
        return cls(
            event_id=_event_id(),
            camera_id=camera_id,
            timestamp=timestamp,
            person_id=person_id,
            violation_type=f"{missing_ppe.upper()}_MISSING",
            confidence=confidence,
            evidence_path=evidence_path,
            zone=zone,
            bbox=bbox,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload

    def summary(self) -> str:
        return (
            f"CAMERA: {self.camera_id}\n"
            f"PERSON: {self.person_id}\n"
            f"VIOLATION: {self.violation_type}\n"
            f"CONFIDENCE: {self.confidence:.2f}\n"
            f"TIME: {self.timestamp.isoformat()}"
        )
