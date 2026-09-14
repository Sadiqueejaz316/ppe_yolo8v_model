"""Hold displayed PPE polarity until several frames agree.

Association can flicker between Hardhat and NO-Hardhat on consecutive frames.
Violation events already wait for ``confirmation_seconds``. This filter is a
shorter, frame-count gate so the operator overlay does not flash.

The first observation for a person + PPE category is accepted immediately.
Later flips require ``stable_frames`` consecutive matching polarities.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.compliance.association import PPEObservation, PersonPPEState


@dataclass
class _Slot:
    polarity: str
    observation: PPEObservation | None
    pending_polarity: str | None = None
    pending_observation: PPEObservation | None = None
    pending_count: int = 0


class PPEStateStabilizer:
    def __init__(self, stable_frames: int = 3) -> None:
        self._stable_frames = max(1, int(stable_frames))
        self._slots: dict[tuple[int, str], _Slot] = {}

    def reset(self) -> None:
        self._slots.clear()

    def update(self, states: list[PersonPPEState]) -> list[PersonPPEState]:
        if self._stable_frames <= 1:
            return states

        live = {state.person_id for state in states}
        self._slots = {key: slot for key, slot in self._slots.items() if key[0] in live}

        stabilized: list[PersonPPEState] = []
        for state in states:
            watched = set(state.observations)
            watched.update(key[1] for key in self._slots if key[0] == state.person_id)
            new_obs: dict[str, PPEObservation] = {}
            for canonical in watched:
                raw_obs = state.observations.get(canonical)
                raw_polarity = state.status_for(canonical)
                displayed = self._advance(state.person_id, canonical, raw_polarity, raw_obs)
                if displayed is not None:
                    new_obs[canonical] = displayed
            stabilized.append(
                replace(state, observations=new_obs)
            )
        return stabilized

    def _advance(
        self,
        person_id: int,
        canonical: str,
        raw_polarity: str,
        raw_obs: PPEObservation | None,
    ) -> PPEObservation | None:
        key = (person_id, canonical)
        slot = self._slots.get(key)
        if slot is None:
            self._slots[key] = _Slot(polarity=raw_polarity, observation=raw_obs)
            return raw_obs

        if raw_polarity == slot.polarity:
            slot.pending_polarity = None
            slot.pending_observation = None
            slot.pending_count = 0
            if raw_obs is not None:
                slot.observation = raw_obs
            return slot.observation

        if slot.pending_polarity != raw_polarity:
            slot.pending_polarity = raw_polarity
            slot.pending_observation = raw_obs
            slot.pending_count = 1
        else:
            slot.pending_count += 1
            if raw_obs is not None:
                slot.pending_observation = raw_obs

        if slot.pending_count >= self._stable_frames:
            slot.polarity = raw_polarity
            slot.observation = slot.pending_observation
            slot.pending_polarity = None
            slot.pending_observation = None
            slot.pending_count = 0
        return slot.observation
