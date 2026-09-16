"""Operator-facing labels. Not compliance logic."""

from __future__ import annotations

VIOLATION_LABELS = {
    "HELMET_MISSING": "Missing Helmet",
    "MASK_MISSING": "Missing Mask",
    "SAFETY_VEST_MISSING": "Missing Safety Vest",
}

PPE_FROM_VIOLATION = {
    "HELMET_MISSING": "helmet",
    "MASK_MISSING": "mask",
    "SAFETY_VEST_MISSING": "safety_vest",
}


def violation_label(violation_type: str) -> str:
    return VIOLATION_LABELS.get(violation_type, violation_type.replace("_", " ").title())
