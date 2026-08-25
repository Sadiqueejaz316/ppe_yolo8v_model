from pathlib import Path

import pytest

from src.config.settings import load_settings
from src.inference.detector import YOLODetector, resolve_model_path


def test_settings_load_and_expand_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("RTSP_URL", raising=False)
    settings = load_settings()
    assert settings.model.path.endswith("best.pt")
    assert settings.model.device in {"auto", "cpu", "cuda", "cuda:0"} or settings.model.device.startswith("cuda")
    assert "CAM-001" in {cam.id for cam in settings.cameras}


@pytest.mark.skipif(not Path("models/best.pt").exists(), reason="best.pt is not present")
def test_model_loads_and_exposes_classes():
    path = resolve_model_path("models/best.pt", project_root=Path(".").resolve())
    detector = YOLODetector(path, device="cpu", confidence_threshold=0.35)
    assert detector.class_names
    assert all(isinstance(k, int) and isinstance(v, str) for k, v in detector.class_names.items())
