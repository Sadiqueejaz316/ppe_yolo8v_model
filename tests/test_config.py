from pathlib import Path

from src.config.settings import load_settings


def test_load_settings_from_project_root():
    settings = load_settings()
    assert settings.model.path.endswith("best.pt")
    assert settings.cameras
    assert settings.cameras[0].id
    assert "general" in settings.zones
    assert settings.violations.confirmation_seconds > 0
    assert settings.association.min_score > 0
    assert settings.visualization.stable_frames >= 1
    assert settings.association.regions["helmet"] == "head"
    assert settings.association.regions["safety_vest"] == "torso"
    assert settings.dashboard.sqlite_path
    assert settings.dashboard.poll_interval_ms >= 500


def test_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert (settings.project_root / "config" / "app.yaml").exists()
    resolved = settings.resolve_path("models/best.pt")
    assert resolved.is_absolute()
    assert resolved.parent.name == "models"


def test_unknown_camera_raises():
    settings = load_settings()
    try:
        settings.camera_by_id("NO-SUCH-CAM")
        assert False, "expected KeyError"
    except KeyError as exc:
        assert "NO-SUCH-CAM" in str(exc)
