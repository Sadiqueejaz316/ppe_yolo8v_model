"""Spatial association of PPE detections to tracked persons.

Detecting a helmet somewhere in the frame does NOT mean a given worker is
wearing it. This module assigns each PPE box to at most one person using
head/torso regions, IoU, containment, and center-point tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config.settings import AssociationConfig
from src.geometry import BBox, center, containment_ratio, iou, point_in_bbox, subregion
from src.inference.result import Detection
from src.taxonomy import ResolvedTaxonomy
from src.tracking.tracker import TrackedDetection


@dataclass
class PPEObservation:
    canonical: str
    polarity: str  # positive | negative
    confidence: float
    bbox: BBox
    label: str
    score: float


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
    score = max(iou_score, contain, center_score)
    if iou_score < config.min_iou and contain < config.min_containment and center_score <= 0:
        return 0.0
    return score


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

    candidates: list[tuple[float, int, PPEObservation]] = []
    for det in detections:
        hit = taxonomy.ppe_for(det.class_id)
        if hit is None:
            continue
        for index, person in enumerate(states):
            region = region_for(person.bbox, hit.canonical, config)
            score = association_score(det.bbox, region, config)
            if score < config.min_score:
                continue
            obs = PPEObservation(
                canonical=hit.canonical,
                polarity=hit.polarity,
                confidence=det.confidence,
                bbox=det.bbox,
                label=det.label,
                score=score,
            )
            candidates.append((score * det.confidence, index, obs))

    candidates.sort(key=lambda item: item[0], reverse=True)
    assigned_boxes: set[tuple[float, float, float, float]] = set()

    for _, person_index, obs in candidates:
        box_key = tuple(round(v, 2) for v in obs.bbox)
        if box_key in assigned_boxes:
            continue
        current = states[person_index].observations.get(obs.canonical)
        if current is not None:
            current_rank = current.confidence * current.score
            new_rank = obs.confidence * obs.score
            # Prefer positive evidence when ranks are close; otherwise keep stronger.
            if new_rank < current_rank and not (
                current.polarity == "negative" and obs.polarity == "positive" and new_rank >= current_rank * 0.8
            ):
                continue
        states[person_index].observations[obs.canonical] = obs
        assigned_boxes.add(box_key)

    return states
