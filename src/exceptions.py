"""Domain exceptions for the PPE pipeline."""


class PPEError(Exception):
    """Base error for the PPE detection system."""


class ModelNotFoundError(PPEError):
    """Model checkpoint is missing on disk and not in the local cache."""


class ModelLoadError(PPEError):
    """Checkpoint exists but could not be loaded."""


class UnsupportedModelError(PPEError):
    """Checkpoint is not a supported detection model."""


class CameraError(PPEError):
    """Base camera / video-source error."""


class CameraConnectionError(CameraError):
    """Failed to open or re-open a camera stream."""


class InvalidSourceError(CameraError):
    """Source path or RTSP URL is missing or malformed."""


class InferenceError(PPEError):
    """Detector failed on a frame."""


class EvidenceError(PPEError):
    """Evidence could not be written (disk, permissions, etc.)."""
