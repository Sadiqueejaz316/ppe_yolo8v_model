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
from src.compliance.stability import PPEStateStabilizer
from src.compliance.summary import build_scene_summary
from src.config.settings import AppConfig, CameraConfig
from src.events.publisher import EventPublisher, LocalEventPublisher
from src.events.temporal import TemporalViolationFilter
from src.events.violation import PPEViolationEvent
from src.evidence.capture import EvidenceCapture
from src.exceptions import EvidenceError, FRAME_RUNTIME_ERRORS, InferenceError
from src.inference.detector import Detector, YOLODetector
from src.inference.result import Detection, DetectionResult
from src.metrics.collector import MetricsSnapshot, PipelineMetrics
from src.ops.sink import DashboardSink
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
    scene_summary: dict


@dataclass
class PPEPipeline:
    config: AppConfig
    camera: CameraConfig
    detector: Detector
    taxonomy: ResolvedTaxonomy
    tracker: Tracker
    compliance: ComplianceEngine
    temporal: TemporalViolationFilter
    stabilizer: PPEStateStabilizer
    evidence: EvidenceCapture
    publisher: EventPublisher
    metrics: PipelineMetrics
    dashboard_sink: DashboardSink | None = None
    last_detections: list[Detection] = field(default_factory=list)
    last_result: DetectionResult | None = None
    frames_since_prune: int = 0

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
        sink: DashboardSink | None = None
        try:
            sink = DashboardSink.from_config(config)
        except Exception:
            logger.warning("DASHBOARD_SINK_INIT_FAILED camera=%s", camera.id, exc_info=True)
        pipeline = cls(
            config=config,
            camera=camera,
            detector=det,
            taxonomy=taxonomy,
            tracker=tracker,
            compliance=ComplianceEngine(zone, taxonomy),
            temporal=TemporalViolationFilter(config.violations),
            stabilizer=PPEStateStabilizer(config.visualization.stable_frames),
            evidence=EvidenceCapture(config.evidence, config.project_root),
            publisher=LocalEventPublisher(jsonl),
            metrics=PipelineMetrics(camera.id),
            dashboard_sink=sink,
        )
        if hasattr(det, "warmup"):
            try:
                det.warmup()
            except Exception:
                logger.warning("MODEL_WARMUP_FAILED camera=%s", camera.id, exc_info=True)
        return pipeline

    def _log_layer_counts(
        self,
        frame: VideoFrame,
        detections: list[Detection],
        person_dets: list[Detection],
        tracked: list[TrackedDetection],
        persons: list[PersonPPEState],
    ) -> None:
        """One DEBUG line per frame telling which layer lost a worker.

        ``persons`` vs ``tracks`` isolates the tracker, ``ppe`` vs
        ``associations`` isolates association. Enable with ``--log-level DEBUG``.
        """
        ppe_count = sum(1 for det in detections if self.taxonomy.ppe_for(det.class_id) is not None)
        logger.debug(
            "PIPELINE_LAYERS camera=%s frame=%s persons=%s ppe=%s tracks=%s associations=%s untracked=%s",
            frame.camera_id,
            frame.frame_index,
            len(person_dets),
            ppe_count,
            len(tracked),
            sum(len(person.observations) for person in persons),
            max(0, len(person_dets) - len(tracked)),
        )

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
        persons = self.stabilizer.update(persons)
        compliance = [self.compliance.evaluate(person) for person in persons]
        if logger.isEnabledFor(logging.DEBUG):
            self._log_layer_counts(frame, detections, person_dets, tracked, persons)
        summary = build_scene_summary(persons, compliance, self.compliance.required_ppe)
        boxes = {person.person_id: person.bbox for person in persons}
        # Confirmation must use fresh detections. Reused boxes with a new timestamp
        # would let a single stale inference confirm a violation.
        if fresh_inference:
            events = self.temporal.update(compliance, frame.timestamp, frame.camera_id, boxes)
        else:
            events = []

        published: list[PPEViolationEvent] = []
        annotated_preview: np.ndarray | None = None
        # Only clear the episode when the worker is fully compliant. Per-item
        # PRESENT flicker must not reopen evidence inside the cooldown window.
        dedup = self.evidence.deduplicator
        if fresh_inference:
            dedup.note_active_tracks(
                frame.camera_id, [person.person_id for person in persons], frame.timestamp
            )
        for comp in compliance:
            if comp.compliant and not comp.missing_ppe:
                dedup.resolve_violation(frame.camera_id, comp.person_id, comp.present_ppe)

        # Group confirmed events by person for multi-item consolidation
        events_by_person: dict[int, list[PPEViolationEvent]] = {}
        for event in events:
            events_by_person.setdefault(event.person_id, []).append(event)

        for _person_id, person_events in events_by_person.items():
            signature_items = [ev.violation_type for ev in person_events]
            primary_event = person_events[0]
            if annotated_preview is None:
                annotated_preview = annotate(
                    frame.image,
                    detections,
                    persons,
                    compliance,
                    None,
                    self.taxonomy,
                    self.compliance.required_ppe,
                    timestamp=frame.timestamp,
                    visualization=self.config.visualization,
                )
            saved = None
            try:
                saved = self.evidence.save(
                    annotated_preview,
                    primary_event,
                    signature=signature_items,
                )
            except EvidenceError:
                logger.exception("EVIDENCE_EXCEPTION camera=%s event=%s", frame.camera_id, primary_event.event_id)

            # Reused or suppressed evidence belongs to an episode that already
            # published: no JPEG, no JSONL event, no dashboard row.
            if saved is None or not saved.is_new:
                if saved is not None:
                    logger.debug(
                        "EVENT_NOT_PUBLISHED camera=%s worker=%s status=%s",
                        frame.camera_id,
                        primary_event.person_id,
                        saved.status,
                    )
                continue

            event = PPEViolationEvent(
                event_id=primary_event.event_id,
                camera_id=primary_event.camera_id,
                timestamp=primary_event.timestamp,
                person_id=primary_event.person_id,
                violation_type=primary_event.violation_type,
                confidence=primary_event.confidence,
                evidence_path=saved.path,
                zone=primary_event.zone,
                bbox=primary_event.bbox,
            )
            try:
                self.publisher.publish(event)
            except EvidenceError:
                logger.exception("EVENT_PUBLISH_FAILED camera=%s", frame.camera_id)
            published.append(event)

        # Long live runs would otherwise accumulate one dedup record per track ID.
        if fresh_inference:
            self.frames_since_prune += 1
            if self.frames_since_prune >= 600:
                self.frames_since_prune = 0
                try:
                    dedup.prune(frame.timestamp)
                except Exception:
                    logger.debug("EVIDENCE_DEDUP_PRUNE_FAILED camera=%s", frame.camera_id, exc_info=True)

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
            visualization=self.config.visualization,
        )
        result = ProcessedFrame(
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
            scene_summary=summary,
        )
        # Live JPEG encode + in-memory publish; disk/SQLite stay off this thread.
        if self.dashboard_sink is not None and fresh_inference:
            try:
                self.dashboard_sink.observe(result)
            except FRAME_RUNTIME_ERRORS as exc:
                logger.warning("DASHBOARD_SINK_FAILED camera=%s error=%s", frame.camera_id, exc)
        return result
