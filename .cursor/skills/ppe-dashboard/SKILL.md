---
name: ppe-dashboard
description: Maintains the thin PPE operator dashboard — FastAPI /api/*, SQLite events, live JPEG snapshots, and the React polling UI. Use when changing frontend/, src/api/, src/ops/, docs/THIN_DASHBOARD.md, dashboard YAML, or when the user mentions the operator UI, mock data, snapshots, or evidence viewer. Never recompute helmet/mask/vest compliance in the API or React.
---

# PPE operator dashboard

## Architecture

```
PPEPipeline  →  scene_summary + PPEViolationEvent
      →  DashboardSink  →  SQLite + data/live/{camera_id}.json|.jpg
      →  FastAPI /api/*  →  React (HTTP poll)
```

Frontend never computes compliance. API never re-runs association or zone rules.

## Run locally

```bash
python -m src.api                          # http://127.0.0.1:8000
MOCK_DATA=true python -m src.api            # labeled simulated counts
cd frontend && npm install && npm run dev  # http://127.0.0.1:5173
```

Pipeline in another terminal (`python -m src.main ...`) so live JPEGs and SQLite update. Empty counts are expected if neither pipeline nor `MOCK_DATA` is on.

## API (GET only)

- `GET /api/health`
- `GET /api/dashboard/summary`
- `GET /api/cameras` and `GET /api/cameras/{camera_id}`
- `GET /api/cameras/{camera_id}/snapshot` (JPEG)
- `GET /api/events?camera_id=&violation_type=`
- `GET /api/events/{event_id}`
- `GET /api/evidence/{event_id}`

Never include `rtsp_url`. Validate ids (`src/api/paths.py`). Evidence must resolve inside the evidence root.

SQLite `data/ppe.sqlite` table `events`: metadata only. Import `evidence/events.jsonl` with `INSERT OR IGNORE` on start. No image blobs. No per-frame rows.

## Frontend

| Path | Page |
| --- | --- |
| `/dashboard` | Live counts, camera JPEG, active violations |
| `/events` | Confirmed event table |
| `/events/:id` | Event + evidence image |
| `/cameras` | Camera status |

- Fetch via `frontend/src/services/api.ts`. Poll with `usePoll` (`dashboard.poll_interval_ms`, default 500ms).
- Operator item fields on people: `COMPLIANT` / `MISSING` / `PRESENT` / `UNKNOWN` from the API. Display them; do not re-derive from boxes.
- `MOCK_DATA` / `summary.mock` must show **DEVELOPMENT MODE — live counts are simulated**.
- CSS: tokens in `frontend/src/index.css` (`--safe`, `--attention`, `--warn`, `--panel`, `--bg`). Industrial control-room dark UI.
- After UI changes, verify in the browser (dashboard, events, event detail, cameras) including mock and empty/live-unavailable states.

## Sink behavior

`src/ops/sink.py` is an adapter. Publish live JPEG only on **fresh inference** frames. Browser never opens RTSP.

V1: no auth, no WebSocket/SSE, no WebRTC. Do not add those unless asked. CORS: local Vite/API origins only.

## Additional resources

- Endpoint payload notes: [reference.md](reference.md)
- Spec: `docs/THIN_DASHBOARD.md`
