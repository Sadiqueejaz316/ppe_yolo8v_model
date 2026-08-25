"""Detector abstraction and YOLOv8 adapter.

The rest of the pipeline consumes ``Detection`` objects only. Replacing this
module with YOLOv11 / RT-DETR / ONNX / TensorRT should not require changes
in camera, tracking, association, or compliance code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.exceptions import InferenceError, ModelLoadError, ModelNotFoundError, UnsupportedModelError
from src.geometry import as_bbox
from src.inference.result import Detection, DetectionResult, LatencyBreakdown

logger = logging.getLogger(__name__)

HF_REPO_ID = "Hansung-Cho/yolov8-ppe-detection"
HF_FILENAME = "best.pt"


class Detector(ABC):
    """Interface every PPE detector must implement."""

    @property
    @abstractmethod
    def class_names(self) -> dict[int, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_path(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def predict(self, frame: np.ndarray) -> DetectionResult:
        raise NotImplementedError


def resolve_device(requested: str) -> str:
    """Map DEVICE=auto|cpu|cuda|cuda:0 to a concrete torch device string."""
    choice = (requested or "auto").strip().lower()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ModelLoadError("PyTorch is required to load the PPE model") from exc

    cuda_ok = bool(torch.cuda.is_available())
    if choice in {"auto", ""}:
        return "cuda:0" if cuda_ok else "cpu"
    if choice == "cpu":
        return "cpu"
    if choice.startswith("cuda"):
        if not cuda_ok:
            logger.warning("GPU_UNAVAILABLE requested=%s falling_back=cpu", requested)
            return "cpu"
        return "cuda:0" if choice == "cuda" else choice
    logger.warning("UNKNOWN_DEVICE requested=%s falling_back=cpu", requested)
    return "cpu"


def describe_device(device: str) -> tuple[str, str]:
    """Return (human label, cuda status) for startup logs."""
    if device.startswith("cuda"):
        try:
            import torch

            name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA"
            return f"NVIDIA GPU ({name})", "available"
        except Exception:
            return "NVIDIA GPU", "available"
    return "CPU", "unavailable"


def resolve_model_path(path: str | Path, project_root: Path | None = None) -> Path:
    """Resolve ``best.pt`` without downloading a new checkpoint."""
    candidate = Path(path)
    search: list[Path] = []
    if candidate.is_absolute():
        search.append(candidate)
    else:
        if project_root is not None:
            search.append((project_root / candidate).resolve())
        search.append(Path.cwd() / candidate)
        search.append(candidate)

    for item in search:
        if item.exists() and item.is_file():
            return item.resolve()

    cached = _local_hf_cache()
    if cached is not None:
        logger.warning("MODEL_PATH_FALLBACK using local huggingface cache path=%s", cached)
        return cached

    tried = ", ".join(str(item) for item in search)
    raise ModelNotFoundError(
        f"PPE model not found. Looked at: {tried}. "
        "Copy the existing best.pt into models/best.pt. Do not download a different checkpoint."
    )


def _local_hf_cache() -> Path | None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        cached = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, local_files_only=True)
    except Exception:
        return None
    path = Path(cached)
    return path if path.exists() else None


def boxes_to_detections(
    xyxy: Sequence[Sequence[float]],
    confidences: Sequence[float],
    class_ids: Sequence[int],
    class_names: Mapping[int, str],
    confidence_threshold: float,
) -> list[Detection]:
    """Pure conversion used by the YOLO adapter and unit tests."""
    detections: list[Detection] = []
    for coords, confidence, class_id in zip(xyxy, confidences, class_ids, strict=False):
        conf = float(confidence)
        if conf < confidence_threshold:
            continue
        cid = int(class_id)
        label = str(class_names.get(cid, str(cid)))
        detections.append(
            Detection(
                class_id=cid,
                label=label,
                confidence=conf,
                bbox=as_bbox(coords),
            )
        )
    return detections


class YOLODetector(Detector):
    """Ultralytics YOLOv8 adapter. Keep Ultralytics types inside this class."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        project_root: Path | None = None,
    ) -> None:
        self._model_path = resolve_model_path(model_path, project_root=project_root)
        self._requested_device = device
        self._device = resolve_device(device)
        self._confidence_threshold = float(confidence_threshold)
        self._iou_threshold = float(iou_threshold)
        self._imgsz = int(imgsz)
        self._model = self._load()
        self._class_names = {int(k): str(v) for k, v in dict(self._model.names).items()}
        self._cuda_fallback_used = False
        label, cuda_status = describe_device(self._device)
        logger.info(
            "MODEL_LOADED model=%s device=%s device_label=%s cuda=%s classes=%s",
            self._model_path.name,
            self._device,
            label,
            cuda_status,
            len(self._class_names),
        )

    def _load(self) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelLoadError("ultralytics is required to load best.pt") from exc

        try:
            model = YOLO(str(self._model_path))
        except Exception as exc:
            raise ModelLoadError(f"Failed to load checkpoint: {self._model_path}") from exc

        names = getattr(model, "names", None)
        if not names:
            raise UnsupportedModelError("Checkpoint does not expose class names")
        try:
            model.to(self._device)
        except Exception as exc:
            logger.warning("MODEL_DEVICE_MOVE_FAILED device=%s error=%s; continuing", self._device, exc)
        return model

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._class_names)

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def imgsz(self) -> int:
        overrides = getattr(self._model, "overrides", {}) or {}
        value = overrides.get("imgsz", self._imgsz)
        if isinstance(value, (list, tuple)):
            return int(value[0])
        return int(value or self._imgsz)

    @property
    def architecture(self) -> str:
        inner = getattr(self._model, "model", None)
        yaml_file = getattr(inner, "yaml_file", None) or getattr(inner, "yaml", None)
        if isinstance(yaml_file, dict):
            yaml_file = yaml_file.get("yaml_file") or yaml_file.get("backbone")
        task = getattr(self._model, "task", "detect")
        type_name = type(inner).__name__ if inner is not None else "YOLO"
        return f"YOLOv8 PPE ({type_name}, task={task})"

    def warmup(self, size: int | None = None) -> None:
        side = int(size or self.imgsz)
        dummy = np.zeros((side, side, 3), dtype=np.uint8)
        try:
            self.predict(dummy)
        except Exception as exc:
            logger.warning("MODEL_WARMUP_FAILED error=%s", exc)

    def predict(self, frame: np.ndarray) -> DetectionResult:
        if frame is None or getattr(frame, "size", 0) == 0:
            raise InferenceError("Malformed frame: empty image")
        if getattr(frame, "ndim", 0) not in {2, 3}:
            raise InferenceError(f"Malformed frame: expected 2 or 3 dims, got {getattr(frame, 'ndim', None)}")

        height, width = int(frame.shape[0]), int(frame.shape[1])
        try:
            results = self._model.predict(
                source=frame,
                conf=self._confidence_threshold,
                iou=self._iou_threshold,
                imgsz=self._imgsz,
                device=self._device,
                verbose=False,
            )
        except RuntimeError as exc:
            message = str(exc)
            if ("CUDA" in message or "cuda" in message) and self._device != "cpu" and not self._cuda_fallback_used:
                logger.error("CUDA_ERROR falling_back=cpu error=%s", exc)
                self._cuda_fallback_used = True
                self._device = "cpu"
                try:
                    self._model.to("cpu")
                except Exception:
                    pass
                return self.predict(frame)
            raise InferenceError(f"Inference failed: {exc}") from exc
        except Exception as exc:
            raise InferenceError(f"Inference failed: {exc}") from exc

        if not results:
            return DetectionResult(device=self._device, image_size=(width, height))

        result = results[0]
        speed = getattr(result, "speed", {}) or {}
        latency = LatencyBreakdown(
            preprocess_ms=float(speed.get("preprocess", 0.0) or 0.0),
            inference_ms=float(speed.get("inference", 0.0) or 0.0),
            postprocess_ms=float(speed.get("postprocess", 0.0) or 0.0),
        )
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return DetectionResult(latency=latency, device=self._device, image_size=(width, height))

        xyxy = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        class_ids = [int(value) for value in boxes.cls.cpu().tolist()]
        detections = boxes_to_detections(
            xyxy=xyxy,
            confidences=confidences,
            class_ids=class_ids,
            class_names=self._class_names,
            confidence_threshold=self._confidence_threshold,
        )
        return DetectionResult(
            detections=detections,
            latency=latency,
            device=self._device,
            image_size=(width, height),
        )


# Backwards-friendly alias used in docs / call sites.
PPEDetector = YOLODetector
