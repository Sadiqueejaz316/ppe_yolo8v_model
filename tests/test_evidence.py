from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from src.config.settings import EvidenceConfig
from src.events.violation import PPEViolationEvent
from src.exceptions import EvidenceError
from src.evidence.capture import EvidenceCapture


def test_evidence_written_only_when_save_called(tmp_path):
    config = EvidenceConfig(directory=str(tmp_path / "evidence"), retention_days=30, jpeg_quality=80)
    capture = EvidenceCapture(config, project_root=Path(tmp_path))
    event = PPEViolationEvent.create(
        camera_id="CAM-001",
        timestamp=datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc),
        person_id=17,
        missing_ppe="helmet",
        confidence=0.92,
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    path = capture.save(frame, event)
    assert path.path is not None
    saved = Path(path.path)
    assert saved.exists()
    assert "CAM-001" in str(saved)
    assert "2026-08-21" in str(saved)
    assert saved.name.startswith("event-")


def test_evidence_filenames_are_unique(tmp_path):
    config = EvidenceConfig(directory=str(tmp_path / "evidence"), retention_days=30, jpeg_quality=80)
    capture = EvidenceCapture(config, project_root=Path(tmp_path))
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    stamp = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    first = Path(
        capture.save(
            frame,
            PPEViolationEvent.create(
                camera_id="CAM-001", timestamp=stamp, person_id=1, missing_ppe="helmet", confidence=0.9
            ),
        ).path
    )
    second = Path(
        capture.save(
            frame,
            PPEViolationEvent.create(
                camera_id="CAM-001", timestamp=stamp, person_id=2, missing_ppe="mask", confidence=0.9
            ),
        ).path
    )
    assert first.exists() and second.exists()
    assert first.name != second.name


def test_evidence_prune_removes_old_files(tmp_path):
    config = EvidenceConfig(directory="evidence", retention_days=1, jpeg_quality=80)
    capture = EvidenceCapture(config, project_root=Path(tmp_path))
    stale = tmp_path / "evidence" / "CAM-001" / "old.jpg"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"x")
    old = datetime.now(timezone.utc) - timedelta(days=10)
    import os

    os.utime(stale, (old.timestamp(), old.timestamp()))
    removed = capture.prune(now=datetime.now(timezone.utc))
    assert removed >= 1
    assert not stale.exists()


def test_evidence_write_failure_raises(tmp_path, monkeypatch):
    import cv2

    monkeypatch.setattr(cv2, "imwrite", lambda *args, **kwargs: False)
    config = EvidenceConfig(directory=str(tmp_path / "evidence"), retention_days=30, jpeg_quality=80)
    capture = EvidenceCapture(config, project_root=Path(tmp_path))
    event = PPEViolationEvent.create(
        camera_id="CAM-001",
        timestamp=datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc),
        person_id=1,
        missing_ppe="helmet",
        confidence=0.9,
    )
    with pytest.raises(EvidenceError):
        capture.save(np.zeros((8, 8, 3), dtype=np.uint8), event)
