"""End-to-end per-frame vision pipeline.

Detection → tracking → association → compliance → temporal confirmation → events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
import numpy as np

from src.compliance.association import PersonPPEState, associate_ppe
from src.compliance.rules import ComplianceEngine, ComplianceResult
from src.config.settings import AppConfig, CameraConfig
from src.events.publisher import EventPublisher, LocalEventPublisher
from src.events.temporal import TemporalViolationFilter
from src.events.violation import PPEViolationEvent
from src.evidence.capture import EvidenceCapture
from src.exceptions import EvidenceError, InferenceError
from src.inference.detector import Detector, YOLODetector
from src.inference.result import Detection, DetectionResult
from src.metrics.collector import MetricsSnapshot, PipelineMetrics
from src.taxonomy import ResolvedTaxonomy, resolve_taxonomy
from src.tracking.tracker import ByteTracker, NoOpTracker, TrackedDetection, Tracker
from src.video.source import VideoFrame
from src.viz import annotate

logger = logging.getLogger(__name__)


@dataclass
class ProcessedFrame:
    frame: VideoFrame
    detections: list[Detection]
    persons: list[PersonPPEState]
    tracked: list[TrackedDetection]
    compliance: list[ComplianceResult]
    events: list[PPEViolationEvent]
    annotated: np.ndarray
    inferred: bool
    inference_ms: float
    metrics: MetricsSnapshot


@dataclass
class PPEPipeline:
    config: AppConfig
    camera: CameraConfig
    detector: Detector
    taxonomy: ResolvedTaxonomy
    tracker: Tracker
    compliance: ComplianceEngine
    temporal: TemporalViolationFilter
    evidence: EvidenceCapture
    publisher: EventPublisher
    metrics: PipelineMetrics
    last_detections: list[Detection] = field(default_factory=list)
    last_result: DetectionResult | None = None

    @classmethod
    def build(
        cls,
        config: AppConfig,
        camera: CameraConfig,
        detector: Detector | None = None,
        tracker: Tracker | None = None,
        enable_tracking: bool = True,
    ) -> PPEPipeline:
        conf = camera.confidence_threshold or config.model.confidence_threshold
        det = detector or YOLODetector(
            model_path=config.model.path,
            device=config.model.device,
            confidence_threshold=conf,
            iou_threshold=config.model.iou_threshold,
            imgsz=config.model.imgsz,
            project_root=config.project_root,
        )
        taxonomy = resolve_taxonomy(det.class_names, config.taxonomy)
        if camera.zone not in config.zones:
            known = ", ".join(sorted(config.zones)) or "<none>"
            raise KeyError(
                f"Unknown zone '{camera.zone}' for camera '{camera.id}'. Known zones: {known}"
            )
        zone = config.zones[camera.zone]
        if tracker is None:
            if enable_tracking and config.tracking.enabled:
                tracker = ByteTracker(config.tracking)
            else:
                tracker = NoOpTracker()
        jsonl = config.resolve_path(config.evidence.events_jsonl)
        pipeline = cls(
            config=config,
            camera=camera,
            detector=det,
            taxonomy=taxonomy,
            tracker=tracker,
            compliance=ComplianceEngine(zone, taxonomy),
            temporal=TemporalViolationFilter(config.violations),
            evidence=EvidenceCapture(config.evidence, config.project_root),
            publisher=LocalEventPublisher(jsonl),
            metrics=PipelineMetrics(camera.id),
        )
        if hasattr(det, "warmup"):
            try:
                det.warmup()
            except Exception:
                logger.warning("MODEL_WARMUP_FAILED camera=%s", camera.id, exc_info=True)
        return pipeline

    def process(self, frame: VideoFrame, infer: bool = True) -> ProcessedFrame:
        started = time.perf_counter()
        detection_result: DetectionResult | None = None
        fresh_inference = False
        if infer:
            try:
                detection_result = self.detector.predict(frame.image)
                self.last_result = detection_result
                self.last_detections = detection_result.detections
                fresh_inference = True
            except InferenceError:
                logger.exception("INFERENCE_EXCEPTION camera=%s", frame.camera_id)
                detection_result = self.last_result
        detections = list(self.last_detections)
        latency_ms = detection_result.latency.total_ms if detection_result is not None else 0.0
        if fresh_inference:
            self.metrics.record_inference(latency_ms, True)

        person_dets = [det for det in detections if self.taxonomy.is_person(det.class_id)]
        tracked = self.tracker.update(person_dets, frame.image)
        persons = associate_ppe(tracked, detections, self.taxonomy, self.config.association)
        compliance = [self.compliance.evaluate(person) for person in persons]
        boxes = {person.person_id: person.bbox for person in persons}
        # Confirmation must use fresh detections. Reused boxes with a new timestamp
        # would let a single stale inference confirm a violation.
        if fresh_inference:
            events = self.temporal.update(compliance, frame.timestamp, frame.camera_id, boxes)
        else:
            events = []

        published: list[PPEViolationEvent] = []
        annotated_preview = annotate(
            frame.image,
            detections,
            persons,
            compliance,
            None,
            self.taxonomy,
            self.compliance.required_ppe,
            timestamp=frame.timestamp,
        )
        for event in events:
            evidence_path = None
            try:
                evidence_path = self.evidence.save(annotated_preview, event)
            except EvidenceError:
                logger.exception("EVIDENCE_EXCEPTION camera=%s event=%s", frame.camera_id, event.event_id)
            event = PPEViolationEvent(
                event_id=event.event_id,
                camera_id=event.camera_id,
                timestamp=event.timestamp,
                person_id=event.person_id,
                violation_type=event.violation_type,
                confidence=event.confidence,
                evidence_path=evidence_path,
                zone=event.zone,
                bbox=event.bbox,
            )
            try:
                self.publisher.publish(event)
            except EvidenceError:
                logger.exception("EVENT_PUBLISH_FAILED camera=%s", frame.camera_id)
            published.append(event)

        self.metrics.record_counts(len(persons), len(detections), len(published))
        self.metrics.record_end_to_end((time.perf_counter() - started) * 1000.0)
        snapshot = MetricsSnapshot(
            camera_id=frame.camera_id,
            camera_fps=frame.source_fps,
            inference_fps=self.metrics.inference_fps,
            inference_latency_ms=self.metrics.inference_latency_ms,
            end_to_end_latency_ms=self.metrics.end_to_end_latency_ms,
            persons=len(persons),
            detections=len(detections),
            violations=self.metrics.violations,
            dropped_frames=0,
            reconnect_count=0,
            cpu_percent=None,
            gpu_percent=None,
            camera_connected=True,
        )
        annotated = annotate(
            frame.image,
            detections,
            persons,
            compliance,
            snapshot,
            self.taxonomy,
            self.compliance.required_ppe,
            timestamp=frame.timestamp,
        )
        return ProcessedFrame(
            frame=frame,
            detections=detections,
            persons=persons,
            tracked=tracked,
            compliance=compliance,
            events=published,
            annotated=annotated,
            inferred=infer,
            inference_ms=latency_ms,
            metrics=snapshot,
        )
