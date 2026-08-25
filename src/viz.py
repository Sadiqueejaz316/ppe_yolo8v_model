"""Annotated overlay for live display and evidence images."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from src.compliance.association import PersonPPEState
from src.compliance.rules import ComplianceResult
from src.inference.result import Detection
from src.metrics.collector import MetricsSnapshot
from src.taxonomy import ResolvedTaxonomy

COLOR_PERSON = (255, 200, 0)
COLOR_OK = (40, 180, 40)
COLOR_MISSING = (40, 40, 230)
COLOR_PPE = (40, 180, 220)
COLOR_OTHER = (160, 160, 160)
COLOR_TEXT = (255, 255, 255)


def _put(image: np.ndarray, text: str, org: tuple[int, int], color=COLOR_TEXT, scale: float = 0.55) -> None:
    import cv2

    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _box(image: np.ndarray, bbox: tuple[float, float, float, float], color, thickness: int = 2) -> None:
    import cv2

    x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def annotate(
    frame: np.ndarray,
    detections: list[Detection],
    persons: list[PersonPPEState],
    compliance: list[ComplianceResult],
    metrics: MetricsSnapshot | None,
    taxonomy: ResolvedTaxonomy,
    required_ppe: tuple[str, ...],
    timestamp: datetime | None = None,
) -> np.ndarray:
    import cv2

    image = frame.copy()
    compliance_by_id = {item.person_id: item for item in compliance}

    for det in detections:
        hit = taxonomy.by_class_id.get(det.class_id)
        if hit is not None and hit.polarity == "person":
            continue
        if hit is not None and hit.polarity == "positive":
            color = COLOR_PPE
        elif hit is not None and hit.polarity == "negative":
            color = COLOR_MISSING
        else:
            color = COLOR_OTHER
        _box(image, det.bbox, color, 1)
        _put(image, f"{det.label} {det.confidence:.2f}", (int(det.x1), max(16, int(det.y1) - 6)), color, 0.45)

    for person in persons:
        result = compliance_by_id.get(person.person_id)
        color = COLOR_OK if result is not None and result.compliant else COLOR_MISSING
        _box(image, person.bbox, color, 2)
        x1, y1 = int(person.bbox[0]), int(person.bbox[1])
        lines = [f"Person #{person.person_id}"]
        for item in required_ppe:
            status = person.status_for(item)
            mark = "OK" if status == "present" else "MISSING"
            lines.append(f"{item}: {mark}")
        for offset, line in enumerate(lines):
            _put(image, line, (x1 + 4, y1 + 16 + offset * 16), color, 0.5)

    y = 22
    if metrics is not None:
        status = "CONNECTED" if metrics.camera_connected else "DISCONNECTED"
        overlay_lines = [
            f"Camera: {metrics.camera_id} [{status}]",
            f"Camera FPS: {metrics.camera_fps:.1f}",
            f"Inference FPS: {metrics.inference_fps:.1f}",
            f"Inference: {metrics.inference_latency_ms:.0f} ms",
            f"E2E: {metrics.end_to_end_latency_ms:.0f} ms",
            f"Persons: {metrics.persons}",
            f"Violations: {metrics.violations}",
            f"Dropped frames: {metrics.dropped_frames}",
        ]
        if metrics.cpu_percent is not None:
            overlay_lines.append(f"CPU: {metrics.cpu_percent:.0f}%")
        if metrics.gpu_percent is not None:
            overlay_lines.append(f"GPU: {metrics.gpu_percent:.0f}%")
        for line in overlay_lines:
            _put(image, line, (12, y), COLOR_TEXT, 0.55)
            y += 20

    if timestamp is not None:
        _put(image, timestamp.isoformat(timespec="seconds"), (12, image.shape[0] - 12), COLOR_TEXT, 0.5)

    # Keep cv2 referenced so type checkers stay quiet if unused in some builds.
    _ = cv2
    return image
