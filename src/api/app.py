"""Thin FastAPI layer over pipeline scene_summary, SQLite events, and evidence files."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from src.api.paths import is_safe_camera_id, is_safe_event_id
from src.api.service import DashboardService
from src.config.settings import AppConfig, load_settings
from src.events.store import EventStore
from src.logging_setup import setup_logging
from src.ops.live import LiveFrameBuffer, LiveStateStore

logger = logging.getLogger(__name__)


def create_app(
    config: AppConfig | None = None,
    frame_buffer: LiveFrameBuffer | None = None,
) -> FastAPI:
    cfg = config or load_settings()
    setup_logging(cfg.logging.level)
    events = EventStore(cfg.resolve_path(cfg.dashboard.sqlite_path))
    try:
        events.import_jsonl(cfg.resolve_path(cfg.evidence.events_jsonl))
    except OSError:
        logger.warning("EVENT_STORE_JSONL_IMPORT_FAILED", exc_info=True)
    live = LiveStateStore(cfg.resolve_path(cfg.dashboard.live_dir))
    service = DashboardService(cfg, events, live, frame_buffer=frame_buffer)

    app = FastAPI(title="PPE Operator Dashboard", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:8000",
            "http://localhost:8000",
        ],
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.state.service = service
    app.state.config = cfg

    @app.get("/api/health")
    def health() -> dict:
        return service.health()

    @app.get("/api/dashboard/summary")
    def dashboard_summary() -> dict:
        return service.summary()

    @app.get("/api/cameras")
    def cameras() -> list:
        return service.cameras()

    @app.get("/api/cameras/{camera_id}")
    def camera(camera_id: str) -> dict:
        if not is_safe_camera_id(camera_id):
            raise HTTPException(status_code=400, detail="Invalid camera id")
        row = service.camera(camera_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Unknown camera")
        return row

    @app.get("/api/cameras/{camera_id}/snapshot")
    def camera_snapshot(camera_id: str) -> Response:
        if not is_safe_camera_id(camera_id):
            raise HTTPException(status_code=400, detail="Invalid camera id")
        jpeg = service.snapshot_jpeg(camera_id)
        if not jpeg:
            raise HTTPException(status_code=404, detail="No live snapshot")
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/api/events")
    def events_list(
        camera_id: str | None = Query(default=None),
        violation_type: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> list:
        if camera_id is not None and not is_safe_camera_id(camera_id):
            raise HTTPException(status_code=400, detail="Invalid camera id")
        if violation_type is not None and not violation_type.replace("_", "").isalnum():
            raise HTTPException(status_code=400, detail="Invalid violation type")
        return service.list_events(camera_id=camera_id, violation_type=violation_type, limit=limit)

    @app.get("/api/events/{event_id}")
    def event_detail(event_id: str) -> dict:
        if not is_safe_event_id(event_id):
            raise HTTPException(status_code=400, detail="Invalid event id")
        row = service.get_event(event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Event not found")
        return row

    @app.get("/api/evidence/{event_id}")
    def evidence(event_id: str) -> FileResponse:
        if not is_safe_event_id(event_id):
            raise HTTPException(status_code=400, detail="Invalid evidence id")
        path = service.evidence_file(event_id)
        if path is None:
            raise HTTPException(status_code=404, detail="No evidence available")
        return FileResponse(path, media_type="image/jpeg")

    # NOTE: FastAPI does NOT serve the React frontend.
    # Development: run `npm run dev` in frontend/ (Vite at http://127.0.0.1:5173).
    # Production: serve frontend/dist/ with a static server (nginx/Caddy) that
    # proxies /api/* to this FastAPI process.
    return app


def main() -> None:
    import uvicorn

    config = load_settings()
    uvicorn.run(
        "src.api.app:create_app",
        factory=True,
        host=config.dashboard.host,
        port=config.dashboard.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
