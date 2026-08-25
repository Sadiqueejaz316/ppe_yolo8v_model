"""Compliance evaluation. Separate from object detection on purpose."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from src.compliance.association import PersonPPEState
from src.config.settings import ZoneConfig
from src.taxonomy import ResolvedTaxonomy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComplianceResult:
    person_id: int
    compliant: bool
    missing_ppe: tuple[str, ...]
    present_ppe: tuple[str, ...]
    zone: str
    confidence: float

    @property
    def overall_status(self) -> str:
        return "COMPLIANT" if self.compliant else "NON_COMPLIANT"


class ComplianceEngine:
    def __init__(self, zone: ZoneConfig, taxonomy: ResolvedTaxonomy) -> None:
        supported = set(taxonomy.supported_ppe)
        required = tuple(item for item in zone.required_ppe if item in supported)
        skipped = [item for item in zone.required_ppe if item not in supported]
        if skipped:
            logger.warning(
                "COMPLIANCE_SKIPPED_UNSUPPORTED zone=%s items=%s",
                zone.name,
                skipped,
            )
        self._zone = zone
        self._required = required

    @property
    def required_ppe(self) -> tuple[str, ...]:
        return self._required

    def evaluate(self, person: PersonPPEState) -> ComplianceResult:
        missing: list[str] = []
        present: list[str] = []
        confidences: list[float] = []
        for item in self._required:
            status = person.status_for(item)
            obs = person.observations.get(item)
            if status == "present":
                present.append(item)
                if obs is not None:
                    confidences.append(obs.confidence)
            else:
                # Safety-oriented: required PPE that is not positively associated
                # is treated as missing. Negative classes reinforce this.
                missing.append(item)
                if obs is not None:
                    confidences.append(obs.confidence)
        confidence = max(confidences) if confidences else person.confidence
        return ComplianceResult(
            person_id=person.person_id,
            compliant=not missing,
            missing_ppe=tuple(missing),
            present_ppe=tuple(present),
            zone=self._zone.name,
            confidence=float(confidence),
        )

