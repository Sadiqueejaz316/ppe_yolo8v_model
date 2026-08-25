"""Person-centric overlay for live display and evidence images.

Production modes draw tracked people and compact PPE status. Raw YOLO boxes
are opt-in (debug / show_raw_detections) so crowded scenes stay readable.

OpenCV Hershey fonts are ASCII-only, so status uses + (present) and x (missing)
instead of Unicode checkmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.compliance.association import PersonPPEState
from src.compliance.rules import ComplianceResult
from src.compliance.summary import build_scene_summary
from src.config.settings import VisualizationConfig
from src.inference.result import Detection
from src.metrics.collector import MetricsSnapshot
from src.taxonomy import ResolvedTaxonomy

COLOR_OK = (40, 180, 40)
COLOR_MISSING = (40, 40, 230)
COLOR_PPE = (40, 180, 220)
COLOR_OTHER = (160, 160, 160)
COLOR_TEXT = (255, 255, 255)
COLOR_LABEL_BG = (20, 20, 20)

DISPLAY_PPE = (("helmet", "Helmet", "H"), ("mask", "Mask", "M"), ("safety_vest", "Vest", "V"))


@dataclass(frozen=True)
class RawBoxOverlay:
    bbox: tuple[float, float, float, float]
    label: str
    color: tuple[int, int, int]


@dataclass(frozen=True)
class PersonOverlay:
    person_id: int
    bbox: tuple[float, float, float, float]
    color: tuple[int, int, int]
    thickness: int
    lines: tuple[str, ...]
    origin: tuple[int, int]
    overall: str


@dataclass(frozen=True)
class OverlayPlan:
    mode: str
    raw_boxes: tuple[RawBoxOverlay, ...]
    persons: tuple[PersonOverlay, ...]
    hud_lines: tuple[str, ...]
    scene_summary: dict


def _put(image: np.ndarray, text: str, org: tuple[int, int], color=COLOR_TEXT, scale: float = 0.55) -> None:
    import cv2

    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _box(image: np.ndarray, bbox: tuple[float, float, float, float], color, thickness: int = 2) -> None:
    import cv2

    x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def _estimate_size(lines: tuple[str, ...], scale: float) -> tuple[int, int]:
    if not lines:
        return (0, 0)
    width = int(max(len(line) for line in lines) * 11 * scale) + 8
    height = int(len(lines) * 18 * scale) + 8
    return (max(width, 8), max(height, 8))


def _rects_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _place_label(
    bbox: tuple[float, float, float, float],
    size: tuple[int, int],
    image_shape: tuple[int, int],
    occupied: list[tuple[int, int, int, int]],
    above: bool,
) -> tuple[int, int]:
    img_h, img_w = image_shape
    x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    tw, th = size
    candidates: list[tuple[int, int]] = []
    if above:
        candidates.append((x1, y1 - th - 2))
    candidates.extend(
        [
            (x1, y1 + 2),
            (x1, y2 + 2),
            (max(0, x2 - tw), y1 - th - 2),
            (x1, max(0, y1 - th - 22)),
        ]
    )
    for ox, oy in candidates:
        ox = max(0, min(ox, max(0, img_w - tw)))
        oy = max(0, min(oy, max(0, img_h - th)))
        rect = (ox, oy, ox + tw, oy + th)
        if not any(_rects_overlap(rect, other) for other in occupied):
            occupied.append(rect)
            return (ox, oy)

    ox = max(0, min(x1, max(0, img_w - tw)))
    oy = max(0, min(max(0, y1 - th - 2), max(0, img_h - th)))
    step = max(8, th)
    for shift in range(24):
        rect = (ox, oy, ox + tw, oy + th)
        if not any(_rects_overlap(rect, other) for other in occupied):
            occupied.append(rect)
            return (ox, oy)
        if oy > 0:
            oy = max(0, oy - step)
        else:
            ox = min(max(0, img_w - tw), ox + step)
            oy = max(0, min(max(0, y1 - th - 2), max(0, img_h - th)))
        if shift == 12:
            oy = min(max(0, img_h - th), y2 + 2)
    occupied.append((ox, oy, ox + tw, oy + th))
    return (ox, oy)


def _ppe_marks(
    person: PersonPPEState,
    required_ppe: tuple[str, ...],
    crowded: bool,
) -> list[str]:
    marks: list[str] = []
    for canonical, full, short in DISPLAY_PPE:
        if canonical not in required_ppe:
            continue
        present = person.status_for(canonical) == "present"
        name = short if crowded else full
        marks.append(f"+{name}" if present else f"x{name}")
    return marks


def _person_lines(
    person: PersonPPEState,
    result: ComplianceResult | None,
    required_ppe: tuple[str, ...],
    viz: VisualizationConfig,
    crowded: bool,
) -> tuple[str, ...]:
    overall = result.overall_status if result is not None else "NON_COMPLIANT"
    pid = f"#{person.person_id}" if crowded else f"Person #{person.person_id}"
    if viz.mode == "minimal":
        parts: list[str] = []
        if viz.show_person_id:
            parts.append(pid)
        if viz.show_overall_status:
            parts.append(overall)
        return (tuple(parts) if parts else (pid,))

    lines: list[str] = []
    if viz.show_person_id:
        lines.append(pid)
    if viz.show_ppe_status:
        marks = _ppe_marks(person, required_ppe, crowded)
        if crowded:
            missing = [mark for mark in marks if mark.startswith("x")]
            if missing:
                lines.append(" ".join(missing))
        elif marks:
            lines.append("  ".join(marks))
    if viz.show_overall_status and not crowded:
        lines.append(overall)
    if not lines:
        lines.append(pid)
    return tuple(lines)


def _raw_overlays(
    detections: list[Detection],
    taxonomy: ResolvedTaxonomy,
    viz: VisualizationConfig,
) -> tuple[RawBoxOverlay, ...]:
    debug = viz.mode == "debug" or viz.show_raw_detections
    if not debug:
        return ()
    boxes: list[RawBoxOverlay] = []
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
        boxes.append(
            RawBoxOverlay(
                bbox=det.bbox,
                label=f"{det.label} {det.confidence:.2f}",
                color=color,
            )
        )
    return tuple(boxes)


def _hud_lines(
    metrics: MetricsSnapshot | None,
    summary: dict,
    viz: VisualizationConfig,
) -> tuple[str, ...]:
    lines: list[str] = []
    if metrics is not None:
        status = "CONNECTED" if metrics.camera_connected else "DISCONNECTED"
        lines.extend(
            [
                f"Camera: {metrics.camera_id} [{status}]",
                f"Camera FPS: {metrics.camera_fps:.1f}",
                f"Inference FPS: {metrics.inference_fps:.1f}",
                f"Inference: {metrics.inference_latency_ms:.0f} ms",
                f"E2E: {metrics.end_to_end_latency_ms:.0f} ms",
                f"Persons: {metrics.persons}",
                f"Violations: {metrics.violations}",
                f"Dropped frames: {metrics.dropped_frames}",
            ]
        )
        if metrics.cpu_percent is not None:
            lines.append(f"CPU: {metrics.cpu_percent:.0f}%")
        if metrics.gpu_percent is not None:
            lines.append(f"GPU: {metrics.gpu_percent:.0f}%")

    lines.append(
        f"PPE STATUS  people={summary['total_people']}  "
        f"ok={summary['compliant']}  bad={summary['violations']}"
    )
    rows = max(0, int(viz.hud_violation_rows))
    shown = 0
    for person in summary.get("people") or []:
        if person.get("overall_status") == "COMPLIANT":
            continue
        missing = person.get("missing_ppe") or []
        short = ",".join(str(item).replace("safety_vest", "vest") for item in missing) or "PPE"
        lines.append(f"  #{person['person_id']} x {short}")
        shown += 1
        if shown >= rows:
            extra = summary["violations"] - shown
            if extra > 0:
                lines.append(f"  ... {extra} more")
            break
    return tuple(lines)


def build_overlay_plan(
    detections: list[Detection],
    persons: list[PersonPPEState],
    compliance: list[ComplianceResult],
    taxonomy: ResolvedTaxonomy,
    required_ppe: tuple[str, ...],
    visualization: VisualizationConfig | None = None,
    metrics: MetricsSnapshot | None = None,
    image_shape: tuple[int, int] = (720, 1280),
) -> OverlayPlan:
    viz = visualization or VisualizationConfig()
    mode = viz.mode if viz.mode in {"person_summary", "minimal", "debug"} else "person_summary"
    if mode != viz.mode:
        viz = VisualizationConfig(
            mode=mode,
            show_raw_detections=viz.show_raw_detections,
            show_person_id=viz.show_person_id,
            show_ppe_status=viz.show_ppe_status,
            show_overall_status=viz.show_overall_status,
            crowd_compact_threshold=viz.crowd_compact_threshold,
            label_above_box=viz.label_above_box,
            stable_frames=viz.stable_frames,
            hud_violation_rows=viz.hud_violation_rows,
        )
    summary = build_scene_summary(persons, compliance, required_ppe)
    crowded = len(persons) >= max(1, viz.crowd_compact_threshold)
    scale = 0.4 if crowded else 0.5
    compliance_by_id = {item.person_id: item for item in compliance}
    occupied: list[tuple[int, int, int, int]] = []
    hud = _hud_lines(metrics, summary, viz)
    if hud:
        hud_w, hud_h = _estimate_size(hud, 0.5)
        occupied.append((8, 4, 8 + hud_w, 4 + hud_h))
    person_overlays: list[PersonOverlay] = []

    ordered = sorted(persons, key=lambda item: (item.bbox[1], item.bbox[0]))
    for person in ordered:
        result = compliance_by_id.get(person.person_id)
        overall = result.overall_status if result is not None else "NON_COMPLIANT"
        color = COLOR_OK if overall == "COMPLIANT" else COLOR_MISSING
        thickness = 2 if overall == "COMPLIANT" else 3
        lines = _person_lines(person, result, required_ppe, viz, crowded)
        size = _estimate_size(lines, scale)
        origin = _place_label(person.bbox, size, image_shape, occupied, viz.label_above_box)
        person_overlays.append(
            PersonOverlay(
                person_id=person.person_id,
                bbox=person.bbox,
                color=color,
                thickness=thickness,
                lines=lines,
                origin=origin,
                overall=overall,
            )
        )

    return OverlayPlan(
        mode=mode,
        raw_boxes=_raw_overlays(detections, taxonomy, viz),
        persons=tuple(person_overlays),
        hud_lines=hud,
        scene_summary=summary,
    )


def annotate(
    frame: np.ndarray,
    detections: list[Detection],
    persons: list[PersonPPEState],
    compliance: list[ComplianceResult],
    metrics: MetricsSnapshot | None,
    taxonomy: ResolvedTaxonomy,
    required_ppe: tuple[str, ...],
    timestamp: datetime | None = None,
    visualization: VisualizationConfig | None = None,
) -> np.ndarray:
    import cv2

    image = frame.copy()
    viz = visualization or VisualizationConfig()
    crowded = len(persons) >= max(1, viz.crowd_compact_threshold)
    scale = 0.4 if crowded else 0.5
    plan = build_overlay_plan(
        detections,
        persons,
        compliance,
        taxonomy,
        required_ppe,
        visualization=viz,
        metrics=metrics,
        image_shape=(int(image.shape[0]), int(image.shape[1])),
    )

    for raw in plan.raw_boxes:
        _box(image, raw.bbox, raw.color, 1)
        _put(image, raw.label, (int(raw.bbox[0]), max(16, int(raw.bbox[1]) - 6)), raw.color, 0.4)

    for person in plan.persons:
        _box(image, person.bbox, person.color, person.thickness)
        ox, oy = person.origin
        tw, th = _estimate_size(person.lines, scale)
        cv2.rectangle(image, (ox, oy), (ox + tw, oy + th), COLOR_LABEL_BG, -1)
        for offset, line in enumerate(person.lines):
            color = person.color
            if line.startswith("x"):
                color = COLOR_MISSING
            elif line.startswith("+"):
                color = COLOR_OK
            _put(image, line, (ox + 4, oy + 14 + offset * int(18 * scale)), color, scale)

    y = 22
    for line in plan.hud_lines:
        _put(image, line, (12, y), COLOR_TEXT, 0.5)
        y += 18

    if timestamp is not None:
        _put(image, timestamp.isoformat(timespec="seconds"), (12, image.shape[0] - 12), COLOR_TEXT, 0.5)

    _ = cv2
    return image
