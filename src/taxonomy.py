"""Map actual model class names onto canonical PPE / person labels."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from src.config.settings import TaxonomyConfig

logger = logging.getLogger(__name__)


def _norm(label: str) -> str:
    return "".join(ch.lower() for ch in label if ch.isalnum())


@dataclass(frozen=True)
class ClassHit:
    canonical: str
    polarity: str  # person | positive | negative | other


@dataclass
class ResolvedTaxonomy:
    person_class_ids: set[int]
    positive_ids: dict[str, set[int]]
    negative_ids: dict[str, set[int]]
    by_class_id: dict[int, ClassHit]
    unmatched_labels: dict[int, str]
    supported_ppe: tuple[str, ...]

    def is_person(self, class_id: int) -> bool:
        return class_id in self.person_class_ids

    def ppe_for(self, class_id: int) -> ClassHit | None:
        hit = self.by_class_id.get(class_id)
        if hit is None or hit.polarity not in {"positive", "negative"}:
            return None
        return hit


def resolve_taxonomy(class_names: dict[int, str], config: TaxonomyConfig) -> ResolvedTaxonomy:
    """Match config aliases against the checkpoint's actual names. No guessing."""
    person_ids: set[int] = set()
    positive_ids: dict[str, set[int]] = {key: set() for key in config.ppe}
    negative_ids: dict[str, set[int]] = {key: set() for key in config.ppe}
    by_class_id: dict[int, ClassHit] = {}
    unmatched: dict[int, str] = {}

    person_aliases = {_norm(label) for label in config.person_labels}
    pos_aliases = {
        canonical: {_norm(label) for label in spec.positive} for canonical, spec in config.ppe.items()
    }
    neg_aliases = {
        canonical: {_norm(label) for label in spec.negative} for canonical, spec in config.ppe.items()
    }

    for class_id, label in class_names.items():
        token = _norm(label)
        if token in person_aliases:
            person_ids.add(class_id)
            by_class_id[class_id] = ClassHit(canonical="person", polarity="person")
            continue

        matched = False
        for canonical, aliases in pos_aliases.items():
            if token in aliases:
                positive_ids[canonical].add(class_id)
                by_class_id[class_id] = ClassHit(canonical=canonical, polarity="positive")
                matched = True
                break
        if matched:
            continue
        for canonical, aliases in neg_aliases.items():
            if token in aliases:
                negative_ids[canonical].add(class_id)
                by_class_id[class_id] = ClassHit(canonical=canonical, polarity="negative")
                matched = True
                break
        if not matched:
            unmatched[class_id] = label
            by_class_id[class_id] = ClassHit(canonical=label, polarity="other")

    supported = tuple(
        canonical
        for canonical in config.ppe
        if positive_ids[canonical] or negative_ids[canonical]
    )
    logger.info(
        "TAXONOMY_RESOLVED persons=%s ppe=%s unmatched=%s",
        sorted(person_ids),
        list(supported),
        unmatched,
    )
    return ResolvedTaxonomy(
        person_class_ids=person_ids,
        positive_ids=positive_ids,
        negative_ids=negative_ids,
        by_class_id=by_class_id,
        unmatched_labels=unmatched,
        supported_ppe=supported,
    )
