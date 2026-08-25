"""Temporal confirmation and cooldown for PPE violations.

A single bad frame must not generate an event. A persistent miss becomes one
confirmed event, then stays quiet for the cooldown window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src.compliance.rules import ComplianceResult
from src.config.settings import ViolationConfig
from src.events.violation import PPEViolationEvent


@dataclass
class _ItemState:
    missing_since: datetime | None = None
    last_event_at: datetime | None = None


@dataclass
class TemporalViolationFilter:
    config: ViolationConfig
    states: dict[tuple[int, str], _ItemState] = field(default_factory=dict)

    def update(
        self,
        results: list[ComplianceResult],
        timestamp: datetime,
        camera_id: str,
        person_boxes: dict[int, tuple[float, float, float, float]],
    ) -> list[PPEViolationEvent]:
        active_people = {item.person_id for item in results}
        events: list[PPEViolationEvent] = []
        confirmation = timedelta(seconds=self.config.confirmation_seconds)
        cooldown = timedelta(seconds=self.config.cooldown_seconds)

        for result in results:
            missing = set(result.missing_ppe)
            present = set(result.present_ppe)
            watched = missing | present
            for item in watched:
                key = (result.person_id, item)
                state = self.states.setdefault(key, _ItemState())
                if item in present:
                    state.missing_since = None
                    continue
                if state.missing_since is None:
                    state.missing_since = timestamp
                    continue
                if timestamp - state.missing_since < confirmation:
                    continue
                if state.last_event_at is not None and timestamp - state.last_event_at < cooldown:
                    continue
                events.append(
                    PPEViolationEvent.create(
                        camera_id=camera_id,
                        timestamp=timestamp,
                        person_id=result.person_id,
                        missing_ppe=item,
                        confidence=result.confidence,
                        zone=result.zone,
                        bbox=person_boxes.get(result.person_id),
                    )
                )
                state.last_event_at = timestamp

        stale: list[tuple[int, str]] = []
        for key, state in self.states.items():
            if key[0] in active_people:
                continue
            # Track dropped: reset confirmation so absence is not treated as a miss,
            # but keep cooldown so the same ID cannot immediately re-alert.
            state.missing_since = None
            if state.last_event_at is None or timestamp - state.last_event_at >= cooldown:
                stale.append(key)
        for key in stale:
            self.states.pop(key, None)
        return events
