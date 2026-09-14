"""Central configuration loaded from YAML + environment variables."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional at import time
    load_dotenv = None  # type: ignore[assignment]

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _expand_string(value: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2)
        env_val = os.environ.get(key)
        if env_val is not None and env_val != "":
            return env_val
        return default if default is not None else ""

    return _ENV_PATTERN.sub(_replace, value)


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_string(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _as_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def _as_bool(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_str_list(value: Any) -> list[str]:
    if not value:
        return []
    return [str(item) for item in value]


@dataclass(frozen=True)
class ModelConfig:
    path: str = "models/best.pt"
    device: str = "auto"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    imgsz: int = 640


@dataclass(frozen=True)
class InferenceConfig:
    target_inference_fps: float = 10.0


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = True
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.7
    min_hits: int = 1


def _default_association_regions() -> dict[str, str]:
    return {"helmet": "head", "mask": "head", "safety_vest": "torso"}


@dataclass(frozen=True)
class AssociationConfig:
    min_iou: float = 0.05
    min_containment: float = 0.25
    center_in_region: bool = True
    min_score: float = 0.25
    head_height_ratio: float = 0.35
    torso_y_start: float = 0.20
    torso_y_end: float = 0.75
    conflict_margin: float = 0.15
    prefer_positive_on_tie: bool = True
    regions: dict[str, str] = field(default_factory=_default_association_regions)


@dataclass(frozen=True)
class VisualizationConfig:
    """Operator overlay. Detection itself is unchanged."""

    mode: str = "person_summary"
    show_raw_detections: bool = False
    show_person_id: bool = True
    show_ppe_status: bool = True
    show_overall_status: bool = True
    crowd_compact_threshold: int = 8
    label_above_box: bool = True
    stable_frames: int = 3
    hud_violation_rows: int = 8


@dataclass(frozen=True)
class PPEClassAliases:
    positive: tuple[str, ...] = ()
    negative: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaxonomyConfig:
    person_labels: tuple[str, ...] = ("Person", "person")
    ppe: dict[str, PPEClassAliases] = field(default_factory=dict)


@dataclass(frozen=True)
class ZoneConfig:
    name: str
    required_ppe: tuple[str, ...]


@dataclass(frozen=True)
class ViolationConfig:
    confirmation_seconds: float = 2.0
    cooldown_seconds: float = 30.0


@dataclass(frozen=True)
class EvidenceConfig:
    directory: str = "evidence"
    retention_days: int = 30
    jpeg_quality: int = 90
    events_jsonl: str = "evidence/events.jsonl"
    cooldown_seconds: float = 30.0
    repeat_active_violations: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(frozen=True)
class DashboardConfig:
    """Thin operator UI. Does not change detection or compliance."""

    sqlite_path: str = "data/ppe.sqlite"
    live_dir: str = "data/live"
    poll_interval_ms: int = 500
    snapshot_poll_interval_ms: int = 120
    mock_data: bool = False
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass(frozen=True)
class CameraConfig:
    id: str
    name: str
    rtsp_url: str
    enabled: bool = True
    target_fps: float = 10.0
    reconnect_delay: float = 5.0
    connection_timeout: float = 10.0
    read_timeout: float = 10.0
    frame_skip: int = 0
    confidence_threshold: float | None = None
    zone: str = "general"


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    inference: InferenceConfig
    tracking: TrackingConfig
    association: AssociationConfig
    visualization: VisualizationConfig
    taxonomy: TaxonomyConfig
    zones: dict[str, ZoneConfig]
    violations: ViolationConfig
    evidence: EvidenceConfig
    logging: LoggingConfig
    dashboard: DashboardConfig
    cameras: tuple[CameraConfig, ...]
    project_root: Path = PROJECT_ROOT

    def camera_by_id(self, camera_id: str) -> CameraConfig:
        for camera in self.cameras:
            if camera.id == camera_id:
                return camera
        known = ", ".join(camera.id for camera in self.cameras) or "<none>"
        raise KeyError(f"Unknown camera '{camera_id}'. Known cameras: {known}")

    def resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (self.project_root / candidate).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return _expand(data)


def _parse_taxonomy(raw: dict[str, Any]) -> TaxonomyConfig:
    ppe_raw = raw.get("ppe") or {}
    ppe: dict[str, PPEClassAliases] = {}
    for canonical, spec in ppe_raw.items():
        spec = spec or {}
        ppe[str(canonical)] = PPEClassAliases(
            positive=tuple(_as_str_list(spec.get("positive"))),
            negative=tuple(_as_str_list(spec.get("negative"))),
        )
    return TaxonomyConfig(
        person_labels=tuple(_as_str_list(raw.get("person_labels") or ["Person", "person"])),
        ppe=ppe,
    )


def _parse_camera(raw: dict[str, Any]) -> CameraConfig:
    camera_id = str(raw.get("id") or "").strip()
    if not camera_id:
        raise ValueError("Each camera entry must have an id")
    return CameraConfig(
        id=camera_id,
        name=str(raw.get("name") or camera_id),
        rtsp_url=str(raw.get("rtsp_url") or ""),
        enabled=_as_bool(raw.get("enabled"), True),
        target_fps=_as_float(raw.get("target_fps"), 10.0),
        reconnect_delay=_as_float(raw.get("reconnect_delay"), 5.0),
        connection_timeout=_as_float(raw.get("connection_timeout"), 10.0),
        read_timeout=_as_float(raw.get("read_timeout"), 10.0),
        frame_skip=_as_int(raw.get("frame_skip"), 0),
        confidence_threshold=(
            _as_float(raw.get("confidence_threshold"), 0.35)
            if raw.get("confidence_threshold") not in (None, "")
            else None
        ),
        zone=str(raw.get("zone") or "general"),
    )


_VALID_VIZ_MODES = ("person_summary", "minimal", "debug")


def _parse_visualization(raw: dict[str, Any]) -> VisualizationConfig:
    mode = str(raw.get("mode") or "person_summary").strip().lower()
    if mode not in _VALID_VIZ_MODES:
        mode = "person_summary"
    return VisualizationConfig(
        mode=mode,
        show_raw_detections=_as_bool(raw.get("show_raw_detections"), False),
        show_person_id=_as_bool(raw.get("show_person_id"), True),
        show_ppe_status=_as_bool(raw.get("show_ppe_status"), True),
        show_overall_status=_as_bool(raw.get("show_overall_status"), True),
        crowd_compact_threshold=_as_int(raw.get("crowd_compact_threshold"), 8),
        label_above_box=_as_bool(raw.get("label_above_box"), True),
        stable_frames=_as_int(raw.get("stable_frames"), 3),
        hud_violation_rows=_as_int(raw.get("hud_violation_rows"), 8),
    )


def _parse_dashboard(raw: dict[str, Any]) -> DashboardConfig:
    mock_env = os.environ.get("MOCK_DATA")
    return DashboardConfig(
        sqlite_path=str(raw.get("sqlite_path") or "data/ppe.sqlite"),
        live_dir=str(raw.get("live_dir") or "data/live"),
        poll_interval_ms=_as_int(raw.get("poll_interval_ms"), 500),
        snapshot_poll_interval_ms=_as_int(
            raw.get("snapshot_poll_interval_ms", os.environ.get("SNAPSHOT_POLL_MS")),
            120,
        ),
        mock_data=_as_bool(mock_env if mock_env not in (None, "") else raw.get("mock_data"), False),
        host=str(raw.get("host") or os.environ.get("DASHBOARD_HOST") or "127.0.0.1"),
        port=_as_int(raw.get("port", os.environ.get("DASHBOARD_PORT")), 8000),
    )


def load_settings(
    project_root: Path | None = None,
    app_yaml: Path | None = None,
    camera_yaml: Path | None = None,
    dotenv_path: Path | None = None,
) -> AppConfig:
    """Load YAML config and apply `.env` / process environment overrides."""
    root = Path(project_root) if project_root else PROJECT_ROOT
    env_file = Path(dotenv_path) if dotenv_path else root / ".env"
    if load_dotenv is not None and env_file.exists():
        load_dotenv(env_file, override=False)

    app_path = Path(app_yaml) if app_yaml else root / "config" / "app.yaml"
    camera_path = Path(camera_yaml) if camera_yaml else root / "config" / "camera.yaml"
    app_raw = _load_yaml(app_path)
    camera_raw = _load_yaml(camera_path)

    model_raw = app_raw.get("model") or {}
    inference_raw = app_raw.get("inference") or {}
    tracking_raw = app_raw.get("tracking") or {}
    association_raw = app_raw.get("association") or {}
    visualization_raw = app_raw.get("visualization") or {}
    taxonomy_raw = app_raw.get("class_taxonomy") or {}
    zones_raw = app_raw.get("zones") or {}
    violations_raw = app_raw.get("violations") or {}
    evidence_raw = app_raw.get("evidence") or {}
    logging_raw = app_raw.get("logging") or {}
    dashboard_raw = app_raw.get("dashboard") or {}

    zones = {
        str(name): ZoneConfig(name=str(name), required_ppe=tuple(_as_str_list((spec or {}).get("required_ppe"))))
        for name, spec in zones_raw.items()
    }
    if not zones:
        zones = {"general": ZoneConfig(name="general", required_ppe=("helmet", "safety_vest", "mask"))}

    cameras = tuple(_parse_camera(item) for item in (camera_raw.get("cameras") or []))

    return AppConfig(
        model=ModelConfig(
            path=str(model_raw.get("path") or os.environ.get("MODEL_PATH") or "models/best.pt"),
            device=str(model_raw.get("device") or os.environ.get("DEVICE") or "auto"),
            confidence_threshold=_as_float(
                model_raw.get("confidence_threshold", os.environ.get("CONFIDENCE_THRESHOLD")),
                0.35,
            ),
            iou_threshold=_as_float(model_raw.get("iou_threshold", os.environ.get("IOU_THRESHOLD")), 0.45),
            imgsz=_as_int(model_raw.get("imgsz"), 640),
        ),
        inference=InferenceConfig(
            target_inference_fps=_as_float(
                inference_raw.get("target_inference_fps", os.environ.get("TARGET_FPS")),
                10.0,
            )
        ),
        tracking=TrackingConfig(
            enabled=_as_bool(tracking_raw.get("enabled"), True),
            track_high_thresh=_as_float(tracking_raw.get("track_high_thresh"), 0.5),
            track_low_thresh=_as_float(tracking_raw.get("track_low_thresh"), 0.1),
            new_track_thresh=_as_float(tracking_raw.get("new_track_thresh"), 0.6),
            track_buffer=_as_int(tracking_raw.get("track_buffer"), 30),
            match_thresh=_as_float(tracking_raw.get("match_thresh"), 0.7),
            min_hits=_as_int(tracking_raw.get("min_hits"), 1),
        ),
        association=AssociationConfig(
            min_iou=_as_float(association_raw.get("min_iou"), 0.05),
            min_containment=_as_float(association_raw.get("min_containment"), 0.25),
            center_in_region=_as_bool(association_raw.get("center_in_region"), True),
            min_score=_as_float(association_raw.get("min_score"), 0.25),
            head_height_ratio=_as_float(association_raw.get("head_height_ratio"), 0.35),
            torso_y_start=_as_float(association_raw.get("torso_y_start"), 0.20),
            torso_y_end=_as_float(association_raw.get("torso_y_end"), 0.75),
            conflict_margin=_as_float(association_raw.get("conflict_margin"), 0.15),
            prefer_positive_on_tie=_as_bool(association_raw.get("prefer_positive_on_tie"), True),
            regions={
                str(k): str(v)
                for k, v in (
                    association_raw.get("regions") or _default_association_regions()
                ).items()
            },
        ),
        visualization=_parse_visualization(visualization_raw),
        taxonomy=_parse_taxonomy(taxonomy_raw),
        zones=zones,
        violations=ViolationConfig(
            confirmation_seconds=_as_float(
                violations_raw.get("confirmation_seconds", os.environ.get("VIOLATION_CONFIRMATION_SECONDS")),
                2.0,
            ),
            cooldown_seconds=_as_float(
                violations_raw.get("cooldown_seconds", os.environ.get("VIOLATION_COOLDOWN_SECONDS")),
                30.0,
            ),
        ),
        evidence=EvidenceConfig(
            directory=str(evidence_raw.get("directory") or "evidence"),
            retention_days=_as_int(evidence_raw.get("retention_days"), 30),
            jpeg_quality=_as_int(evidence_raw.get("jpeg_quality"), 90),
            events_jsonl=str(evidence_raw.get("events_jsonl") or "evidence/events.jsonl"),
            cooldown_seconds=_as_float(
                evidence_raw.get("cooldown_seconds", os.environ.get("EVIDENCE_COOLDOWN_SECONDS")),
                30.0,
            ),
            repeat_active_violations=_as_bool(
                evidence_raw.get("repeat_active_violations", os.environ.get("REPEAT_ACTIVE_VIOLATIONS")),
                True,
            ),
        ),
        logging=LoggingConfig(level=str(logging_raw.get("level") or os.environ.get("LOG_LEVEL") or "INFO")),
        dashboard=_parse_dashboard(dashboard_raw),
        cameras=cameras,
        project_root=root,
    )
