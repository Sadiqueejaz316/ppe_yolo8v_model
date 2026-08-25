"""Bounding-box geometry helpers used by association and tracking."""

from __future__ import annotations

from typing import Sequence

BBox = tuple[float, float, float, float]


def as_bbox(values: Sequence[float]) -> BBox:
    x1, y1, x2, y2 = (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def center(bbox: BBox) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def point_in_bbox(point: tuple[float, float], bbox: BBox) -> bool:
    x, y = point
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def intersection_area(a: BBox, b: BBox) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: BBox, b: BBox) -> float:
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = area(a) + area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def containment_ratio(inner: BBox, outer: BBox) -> float:
    """Fraction of ``inner`` that lies inside ``outer``."""
    inner_area = area(inner)
    if inner_area <= 0:
        return 0.0
    return intersection_area(inner, outer) / inner_area


def subregion(bbox: BBox, y0_ratio: float, y1_ratio: float) -> BBox:
    """Vertical slice of a box, ratios relative to box height."""
    x1, y1, x2, y2 = bbox
    height = max(0.0, y2 - y1)
    top = y1 + height * max(0.0, min(1.0, y0_ratio))
    bottom = y1 + height * max(0.0, min(1.0, y1_ratio))
    if bottom < top:
        top, bottom = bottom, top
    return (x1, top, x2, bottom)
