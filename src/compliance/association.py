"""Spatial association of PPE detections to tracked persons.

Detecting a helmet somewhere in the frame does NOT mean a given worker is
wearing it. This module assigns each PPE box to at most one person using
head/torso regions, IoU, containment, and center-point tests.

Positive/negative classes (Hardhat vs NO-Hardhat) become one state per
PPE category on ``PersonPPEState``. They are not independent objects.

Conflict rule (deterministic, logged at DEBUG):

1. Score every PPE box against each person's configured region
   (helmet/mask → head, vest → torso).
2. Assign each PPE box to at most one person (highest score × confidence).
3. For one person and one PPE category, keep the best observation per polarity.
4. If both polarities remain:
   a. If the better rank is at least ``(1 + conflict_margin)`` times the
      other, take the better rank (``rank = confidence * association_score``).
   b. Else take the higher detector confidence.
   c. Else, if ``prefer_positive_on_tie``, keep present; otherwise keep missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

from src.config.settings import AssociationConfig
from src.geometry import BBox, center, containment_ratio, iou, point_in_bbox, subregion
from src.inference.result import Detection
from src.taxonomy import ResolvedTaxonomy
from src.tracking.tracker import TrackedDetection

logger = logging.getLogger(__name__)

# Operator-facing polarity for one PPE category (not zone compliance).
ITEM_PRESENT = "PRESENT"
ITEM_MISSING = "MISSING"
ITEM_UNKNOWN = "UNKNOWN"


@dataclass
class PPEObservation:
    canonical: str
    polarity: str  # positive | negative
    confidence: float
    bbox: BBox
    label: str
    score: float

    @property
    def rank(self) -> float:
        return self.confidence * self.score


@dataclass
class PersonPPEState:
    person_id: int
    bbox: BBox
    confidence: float
    observations: dict[str, PPEObservation] = field(default_factory=dict)

    def status_for(self, canonical: str) -> str:
        obs = self.observations.get(canonical)
        if obs is None:
            return "not_associated"
        return "present" if obs.polarity == "positive" else "missing"

    def item_state(self, canonical: str) -> str:
        """PRESENT / MISSING / UNKNOWN — association polarity, not zone compliance."""
        status = self.status_for(canonical)
        if status == "present":
            return ITEM_PRESENT
        if status == "missing":
            return ITEM_MISSING
        return ITEM_UNKNOWN

    def helmet_state(self) -> str:
        return self.item_state("helmet")

    def mask_state(self) -> str:
        return self.item_state("mask")

    def vest_state(self) -> str:
        return self.item_state("safety_vest")


def region_for(person_bbox: BBox, canonical: str, config: AssociationConfig) -> BBox:
    kind = (config.regions.get(canonical) or "body").lower()
    if kind == "head":
        return subregion(person_bbox, 0.0, config.head_height_ratio)
    if kind == "torso":
        return subregion(person_bbox, config.torso_y_start, config.torso_y_end)
    return person_bbox


def association_score(ppe_bbox: BBox, region: BBox, config: AssociationConfig) -> float:
    iou_score = iou(ppe_bbox, region)
    contain = containment_ratio(ppe_bbox, region)
    center_score = 1.0 if config.center_in_region and point_in_bbox(center(ppe_bbox), region) else 0.0
    if iou_score < config.min_iou and contain < config.min_containment and center_score <= 0:
        return 0.0
    # Center containment admits borderline PPE boxes, but must not flatten every
    # overlapping-person candidate to the same perfect score.
    center_floor = config.min_score if center_score > 0 else 0.0
    return max(iou_score, contain, center_floor)


def _region_distance(ppe_bbox: BBox, region: BBox) -> float:
    """Pixel center distance used only to break equal association ranks."""
    px, py = center(ppe_bbox)
    rx, ry = center(region)
    return ((px - rx) ** 2 + (py - ry) ** 2) ** 0.5


def resolve_ppe_conflict(
    observations: list[PPEObservation],
    config: AssociationConfig,
    *,
    person_id: int | None = None,
) -> PPEObservation:
    """Pick one polarity for a person + PPE category. Never silent: conflicts are logged."""
    if not observations:
        raise ValueError("resolve_ppe_conflict requires at least one observation")
    if len(observations) == 1:
        return observations[0]

    best_by_polarity: dict[str, PPEObservation] = {}
    for obs in observations:
        current = best_by_polarity.get(obs.polarity)
        if current is None or obs.rank > current.rank or (
            obs.rank == current.rank and obs.confidence > current.confidence
        ):
            best_by_polarity[obs.polarity] = obs

    if len(best_by_polarity) == 1:
        return next(iter(best_by_polarity.values()))

    positive = best_by_polarity.get("positive")
    negative = best_by_polarity.get("negative")
    if positive is None:
        return negative
    if negative is None:
        return positive

    higher, lower = (positive, negative) if positive.rank >= negative.rank else (negative, positive)
    margin = max(0.0, float(config.conflict_margin))
    if lower.rank <= 0 and higher.rank > 0:
        chosen = higher
        reason = "rank"
    elif lower.rank > 0 and higher.rank >= lower.rank * (1.0 + margin):
        chosen = higher
        reason = "rank"
    elif abs(positive.confidence - negative.confidence) > 1e-9:
        chosen = positive if positive.confidence > negative.confidence else negative
        reason = "confidence"
    else:
        chosen = positive if config.prefer_positive_on_tie else negative
        reason = "tie_positive" if config.prefer_positive_on_tie else "tie_negative"

    logger.debug(
        "PPE_CONFLICT person=%s item=%s positive_conf=%.3f negative_conf=%.3f "
        "positive_rank=%.3f negative_rank=%.3f resolved=%s reason=%s",
        person_id,
        positive.canonical,
        positive.confidence,
        negative.confidence,
        positive.rank,
        negative.rank,
        chosen.polarity,
        reason,
    )
    return chosen


def associate_ppe(
    persons: list[TrackedDetection],
    detections: list[Detection],
    taxonomy: ResolvedTaxonomy,
    config: AssociationConfig,
) -> list[PersonPPEState]:
    states = [
        PersonPPEState(person_id=person.track_id, bbox=person.bbox, confidence=person.confidence)
        for person in persons
    ]
    if not states:
        return states

    candidates: list[tuple[float, float, int, PPEObservation]] = []
    for det in detections:
        hit = taxonomy.ppe_for(det.class_id)
        if hit is None:
            continue
        for index, person in enumerate(states):
            region = region_for(person.bbox, hit.canonical, config)
            score = association_score(det.bbox, region, config)
            if score < config.min_score:
                continue
            distance = _region_distance(det.bbox, region)
            obs = PPEObservation(
                canonical=hit.canonical,
                polarity=hit.polarity,
                confidence=det.confidence,
                bbox=det.bbox,
                label=det.label,
                score=score,
            )
            candidates.append((score * det.confidence, distance, index, obs))

    # Overlapping workers produce near-identical scores for the same PPE box;
    # the closer head/torso region owns it instead of whoever was listed first.
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    assigned_boxes: set[tuple[float, float, float, float]] = set()
    grouped: dict[int, list[PPEObservation]] = {index: [] for index in range(len(states))}

    for _, _, person_index, obs in candidates:
        box_key = tuple(round(v, 2) for v in obs.bbox)
        if box_key in assigned_boxes:
            continue
        assigned_boxes.add(box_key)
        grouped[person_index].append(obs)

    for person_index, observations in grouped.items():
        by_canonical: dict[str, list[PPEObservation]] = {}
        for obs in observations:
            by_canonical.setdefault(obs.canonical, []).append(obs)
        person_id = states[person_index].person_id
        for canonical, items in by_canonical.items():
            states[person_index].observations[canonical] = resolve_ppe_conflict(
                items, config, person_id=person_id
            )

    return states
