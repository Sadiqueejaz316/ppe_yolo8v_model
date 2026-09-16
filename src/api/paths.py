"""Serve evidence files without exposing arbitrary filesystem paths."""

from __future__ import annotations

from pathlib import Path


def is_safe_event_id(event_id: str) -> bool:
    token = (event_id or "").strip()
    return 8 <= len(token) <= 32 and token.replace("_", "").isalnum()


def is_safe_camera_id(camera_id: str) -> bool:
    token = (camera_id or "").strip()
    if not token or len(token) > 64:
        return False
    return all(ch.isalnum() or ch in "._-" for ch in token)


def resolve_evidence_file(
    evidence_root: Path,
    event_id: str,
    stored_path: str | None,
    camera_id: str | None = None,
    timestamp: str | None = None,
) -> Path | None:
    root = evidence_root.resolve()
    candidates: list[Path] = []
    if stored_path:
        candidates.append(Path(stored_path))
    if camera_id and timestamp:
        day = timestamp[:10]
        candidates.append(root / camera_id / day / f"event-{event_id}.jpg")
    if camera_id:
        candidates.append(root / camera_id / f"event-{event_id}.jpg")

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return resolved
    return None
