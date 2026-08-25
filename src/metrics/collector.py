"""Runtime metrics. Values are measured, never fabricated."""

from __future__ import annotations

from dataclasses import dataclass
import time

from src.video.source import VideoSource


@dataclass
class MetricsSnapshot:
    camera_id: str
    camera_fps: float
    inference_fps: float
    inference_latency_ms: float
    end_to_end_latency_ms: float
    persons: int
    detections: int
    violations: int
    dropped_frames: int
    reconnect_count: int
    cpu_percent: float | None
    gpu_percent: float | None
    camera_connected: bool


class PipelineMetrics:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.inference_latency_ms = 0.0
        self.end_to_end_latency_ms = 0.0
        self.inference_fps = 0.0
        self.persons = 0
        self.detections = 0
        self.violations = 0
        self._last_infer_ts: float | None = None
        self._cpu_ok = True
        self._gpu_ok = True
        self._gpu_inited = False
        try:
            import psutil

            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    def record_inference(self, latency_ms: float, inferred: bool) -> None:
        if not inferred:
            return
        self.inference_latency_ms = float(latency_ms)
        now = time.monotonic()
        if self._last_infer_ts is not None:
            dt = now - self._last_infer_ts
            if dt > 0:
                instant = 1.0 / dt
                self.inference_fps = (
                    instant if self.inference_fps <= 0 else (0.8 * self.inference_fps + 0.2 * instant)
                )
        self._last_infer_ts = now

    def record_end_to_end(self, latency_ms: float) -> None:
        self.end_to_end_latency_ms = float(latency_ms)

    def record_counts(self, persons: int, detections: int, new_violations: int) -> None:
        self.persons = persons
        self.detections = detections
        self.violations += new_violations

    def snapshot(self, source: VideoSource, dropped_extra: int = 0) -> MetricsSnapshot:
        return MetricsSnapshot(
            camera_id=self.camera_id,
            camera_fps=source.measured_fps,
            inference_fps=self.inference_fps,
            inference_latency_ms=self.inference_latency_ms,
            end_to_end_latency_ms=self.end_to_end_latency_ms,
            persons=self.persons,
            detections=self.detections,
            violations=self.violations,
            dropped_frames=source.dropped_frames + dropped_extra,
            reconnect_count=source.reconnect_count,
            cpu_percent=self._cpu_percent(),
            gpu_percent=self._gpu_percent(),
            camera_connected=source.connected,
        )

    def _cpu_percent(self) -> float | None:
        if not self._cpu_ok:
            return None
        try:
            import psutil

            return float(psutil.cpu_percent(interval=None))
        except Exception:
            self._cpu_ok = False
            return None

    def _gpu_percent(self) -> float | None:
        if not self._gpu_ok:
            return None
        try:
            import pynvml

            if not self._gpu_inited:
                pynvml.nvmlInit()
                self._gpu_inited = True
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except Exception:
            self._gpu_ok = False
            return None
