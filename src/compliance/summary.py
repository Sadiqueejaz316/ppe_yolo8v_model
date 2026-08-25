"""Operator-facing person PPE snapshot for overlays and a future dashboard.

This is a read model. It does not change detection, association, or compliance.
"""

from __future__ import annotations

from src.compliance.association import PersonPPEState
from src.compliance.rules import ComplianceResult


def _operator_item(person: PersonPPEState, canonical: str, required: set[str]) -> str:
    state = person.item_state(canonical)
    if canonical not in required:
        return state
    if state == "PRESENT":
        return "COMPLIANT"
    return "MISSING"


def person_status_record(
    person: PersonPPEState,
    result: ComplianceResult | None,
    required_ppe: tuple[str, ...],
) -> dict:
    required = set(required_ppe)
    record = {
        "person_id": person.person_id,
        "bbox": person.bbox,
        "helmet": _operator_item(person, "helmet", required),
        "mask": _operator_item(person, "mask", required),
        "vest": _operator_item(person, "safety_vest", required),
        "overall_status": "COMPLIANT" if result is not None and result.compliant else "NON_COMPLIANT",
        "missing_ppe": list(result.missing_ppe) if result is not None else list(required_ppe),
        "present_ppe": list(result.present_ppe) if result is not None else [],
    }
    return record


def build_scene_summary(
    persons: list[PersonPPEState],
    compliance: list[ComplianceResult],
    required_ppe: tuple[str, ...],
) -> dict:
    by_id = {item.person_id: item for item in compliance}
    people = [person_status_record(person, by_id.get(person.person_id), required_ppe) for person in persons]
    compliant = sum(1 for item in people if item["overall_status"] == "COMPLIANT")
    return {
        "total_people": len(people),
        "compliant": compliant,
        "violations": len(people) - compliant,
        "required_ppe": list(required_ppe),
        "people": people,
    }
