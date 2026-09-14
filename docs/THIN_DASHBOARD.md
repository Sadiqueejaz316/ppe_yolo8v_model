# PPE Operator Dashboard (V1)

## 1. Dashboard purpose

CURRENT: the repository is a computer-vision PPE pipeline (YOLO → track → associate → comply).

V1 IMPLEMENTATION: a thin local operator UI so a person can see live counts, current violations, recent confirmed events, and evidence images.

FUTURE: authentication, multi-camera operations, browser RTSP, analytics.

## 2. Architecture

```
Camera / video
      ↓
PPEPipeline  (unchanged detection / tracking / association / compliance)
      ↓
ProcessedFrame.scene_summary + confirmed PPEViolationEvent
      ↓
DashboardSink  (adapter only)
      ↓
SQLite metadata     +     data/live/{camera_id}.json + .jpg
      ↓
FastAPI  /api/*
      ↓
React  (poll every ~1.5s)
```

The frontend never computes helmet/mask/vest compliance. The API never re-runs association or zone rules.

## 3. Existing data consumed

| Source | Used as |
| --- | --- |
| `ProcessedFrame.scene_summary` | people counts, per-person helmet/mask/vest/overall_status, active violations |
| `PPEViolationEvent` | historical events (already temporally confirmed) |
| `EvidenceCapture` JPEG files | event evidence |
| `config/camera.yaml` | camera id, name/location, zone, target fps (never `rtsp_url`) |
| `MetricsSnapshot` | camera_connected, fps when a live snapshot exists |

## 4. API endpoints

- `GET /api/health`
- `GET /api/dashboard/summary`
- `GET /api/cameras`
- `GET /api/cameras/{camera_id}`
- `GET /api/cameras/{camera_id}/snapshot` (JPEG if the pipeline wrote one)
- `GET /api/events?camera_id=&violation_type=`
- `GET /api/events/{event_id}`
- `GET /api/evidence/{event_id}`

## 5. Database changes

SQLite file: `data/ppe.sqlite` (configurable).

Table `events`: metadata only (`event_id`, camera, timestamp, person, violation_type, confidence, evidence_path, zone, bbox, optional ppe snapshot copied from scene_summary at confirm time).

No image blobs. No one-row-per-frame.

On API or pipeline start, existing `evidence/events.jsonl` rows are imported (`INSERT OR IGNORE`).

## 6. Evidence storage

Unchanged local layout:

`evidence/{camera_id}/{YYYY-MM-DD}/event-{event_id}.jpg`

The evidence API resolves by event id and refuses paths outside the evidence root.

## 7. Frontend structure

```
frontend/src/
  pages/          Dashboard, Events, Event detail, Cameras
  components/     Header, StatusCard, CameraView, ViolationList, ...
  services/api.ts
  hooks/usePoll.ts
```

Routes: `/dashboard`, `/events`, `/events/:id`, `/cameras`.

## 8. Real-time update strategy

CURRENT/V1: HTTP polling (`dashboard.poll_interval_ms`, default 1500).

FUTURE: WebSocket or SSE.

There is no WebRTC/media server. If the pipeline is running it writes an annotated JPEG; the dashboard refreshes that image. True RTSP-in-the-browser is Phase 2.

## 9. Mock / development mode

`MOCK_DATA=true` (or `dashboard.mock_data: true`).

Labeled **DEVELOPMENT MODE — live counts are simulated**. Mock must not be mistaken for a plant feed. Mock events have no evidence files.

## 10. Security considerations

V1 has no login (document as production gap).

- RTSP URLs and passwords never leave the backend
- Evidence access is by event id, with path-traversal checks
- Camera/event ids are validated
- CORS limited to local dashboard origins

FUTURE: authentication (Phase 4), RBAC (Phase 5).

## 11. How to run locally

Backend (from repo root, WSL venv recommended):

```bash
pip install fastapi uvicorn
python -m src.api
```

Mock live counts:

```bash
MOCK_DATA=true python -m src.api
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173 (Vite proxies `/api` to http://127.0.0.1:8000).

Optional: run the existing pipeline in another terminal (`python -m src.main ...`) so `data/live` snapshots and SQLite events update.

## 12. Tests

```bash
python -m pytest tests -q
```

Dashboard tests mock `scene_summary` / events. They do not load `best.pt`.

## 13. Future improvements

1. Polling dashboard (this V1)
2. RTSP → browser-compatible live stream
3. WebSocket/SSE events
4. User authentication
5. RBAC
6. Multi-camera processing (API already uses `camera_id`)
7. Historical analytics
8. Alerts/notifications
9. Edge deployment
10. Optional cloud backup
