"""SQLite store for confirmed violation event metadata only.

Images stay on the local filesystem. One row per confirmed event, never per frame.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from src.events.violation import PPEViolationEvent

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    person_id INTEGER NOT NULL,
    violation_type TEXT NOT NULL,
    confidence REAL,
    evidence_path TEXT,
    zone TEXT,
    bbox_json TEXT,
    ppe_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(violation_type);
"""


def _dt(value: datetime | str) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class EventStore:
    def __init__(self, sqlite_path: Path) -> None:
        self.path = Path(sqlite_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def insert(
        self,
        event: PPEViolationEvent,
        ppe_snapshot: dict[str, Any] | None = None,
    ) -> None:
        bbox = json.dumps(list(event.bbox)) if event.bbox is not None else None
        ppe_json = json.dumps(ppe_snapshot) if ppe_snapshot else None
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO events (
                    event_id, camera_id, timestamp, person_id, violation_type,
                    confidence, evidence_path, zone, bbox_json, ppe_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.camera_id,
                    _dt(event.timestamp),
                    int(event.person_id),
                    event.violation_type,
                    float(event.confidence),
                    event.evidence_path,
                    event.zone,
                    bbox,
                    ppe_json,
                ),
            )
            conn.commit()

    def get(self, event_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
        return self._row(row) if row is not None else None

    def list(
        self,
        camera_id: str | None = None,
        violation_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        clauses = ["1=1"]
        params: list[Any] = []
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if violation_type:
            clauses.append("violation_type = ?")
            params.append(violation_type)
        sql = (
            f"SELECT * FROM events WHERE {' AND '.join(clauses)} "
            "ORDER BY timestamp DESC LIMIT ?"
        )
        params.append(limit)
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row(row) for row in rows]

    def import_jsonl(self, jsonl_path: Path) -> int:
        if not jsonl_path.exists():
            return 0
        imported = 0
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_id = str(raw.get("event_id") or "").strip()
                if not event_id:
                    continue
                bbox = raw.get("bbox")
                event = PPEViolationEvent(
                    event_id=event_id,
                    camera_id=str(raw.get("camera_id") or ""),
                    timestamp=_parse_ts(raw.get("timestamp")),
                    person_id=int(raw.get("person_id") or 0),
                    violation_type=str(raw.get("violation_type") or ""),
                    confidence=float(raw.get("confidence") or 0.0),
                    evidence_path=raw.get("evidence_path"),
                    zone=str(raw.get("zone") or "general"),
                    bbox=tuple(bbox) if isinstance(bbox, list) and len(bbox) == 4 else None,
                )
                before = self.get(event_id)
                self.insert(event)
                if before is None and self.get(event_id) is not None:
                    imported += 1
        if imported:
            logger.info("EVENT_STORE_IMPORTED count=%s path=%s", imported, jsonl_path)
        return imported

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        ppe = None
        if row["ppe_json"]:
            try:
                ppe = json.loads(row["ppe_json"])
            except json.JSONDecodeError:
                ppe = None
        bbox = None
        if row["bbox_json"]:
            try:
                parsed = json.loads(row["bbox_json"])
                if isinstance(parsed, list) and len(parsed) == 4:
                    bbox = parsed
            except json.JSONDecodeError:
                bbox = None
        return {
            "event_id": row["event_id"],
            "camera_id": row["camera_id"],
            "timestamp": row["timestamp"],
            "person_id": row["person_id"],
            "violation_type": row["violation_type"],
            "confidence": row["confidence"],
            "evidence_path": row["evidence_path"],
            "zone": row["zone"],
            "bbox": bbox,
            "ppe": ppe,
        }


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value or "")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now()
