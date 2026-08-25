# Industrial PPE Monitoring System — Complete Technical Audit and Architecture Report

**Project:** Tata-PPE  
**Audit Date:** 2026-08-24  
**Repository:** `d:\Tata-ppe`  
**Model:** `models/best.pt` (YOLOv8, `Hansung-Cho/yolov8-ppe-detection`, 6.0 MB, 10 classes)  
**Status of model:** Functionally verified — not evaluated or modified in this audit.

---

> [!IMPORTANT]
> **Notation used throughout this document:**
> - **VERIFIED** — directly confirmed by reading source code
> - **INFERRED** — logically inferred from code structure (not directly executed)
> - **RECOMMENDED** — design guidance, not currently implemented
> - **NOT IMPLEMENTED** — explicitly absent from the codebase

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Goals](#2-project-goals)
3. [Current System Overview](#3-current-system-overview)
4. [Current Architecture](#4-current-architecture)
5. [Codebase Structure](#5-codebase-structure)
6. [Component-by-Component Audit](#6-component-by-component-audit)
7. [Detector Architecture](#7-detector-architecture)
8. [Camera Architecture](#8-camera-architecture)
9. [RTSP Architecture](#9-rtsp-architecture)
10. [Tracking](#10-tracking)
11. [PPE Association](#11-ppe-association)
12. [Compliance](#12-compliance)
13. [Temporal Logic](#13-temporal-logic)
14. [Evidence](#14-evidence)
15. [Local Storage](#15-local-storage)
16. [Database](#16-database)
17. [API](#17-api)
18. [Web Interface](#18-web-interface)
19. [Hardware](#19-hardware)
20. [Network Architecture](#20-network-architecture)
21. [Security](#21-security)
22. [Monitoring and System Health](#22-monitoring-and-system-health)
23. [Failure Handling](#23-failure-handling)
24. [Jetson Compatibility](#24-jetson-compatibility)
25. [Multi-Camera Scalability](#25-multi-camera-scalability)
26. [Current vs Target Architecture](#26-current-vs-target-architecture)
27. [Testing Audit](#27-testing-audit)
28. [Gap Analysis](#28-gap-analysis)
29. [Hardware Bill of Materials](#29-hardware-bill-of-materials)
30. [Storage Sizing](#30-storage-sizing)
31. [Project Maturity Score](#31-project-maturity-score)
32. [Recommended Roadmap](#32-recommended-roadmap)
33. [Risks](#33-risks)
34. [Open Questions](#34-open-questions)
35. [Final Recommendations](#35-final-recommendations)
36. [Final Audit Summary Table](#36-final-audit-summary-table)

---

## 1. Executive Summary

The Tata-PPE project is a **well-architected Python prototype** for industrial PPE (Personal Protective Equipment) monitoring in a manufacturing/steel environment. It implements a full computer vision pipeline from frame ingestion to confirmed violation event storage.

**The good:** The codebase demonstrates strong separation of concerns. All major layers — detector, tracker, association, compliance, temporal filtering, evidence — have defined interfaces and independent implementations. The core pipeline logic is solid, testable, and has a meaningful test suite covering 51 tests. The model has been manually verified against test images and a 4-second test video.

**The gap:** The system is a **headless CLI pipeline** with no REST API, no web interface, no relational database, and no live camera integration verified against a physical device. Evidence is written to the local filesystem as JPEG images and a flat JSONL log. This is appropriate for V1, but means there is no way to query, view, or manage historical data beyond reading raw files.

**The immediate next step** is not to add more features — it is to:
1. Test the RTSP path with a real physical camera.
2. Fix the evidence path portability bug (absolute WSL paths in JSONL).
3. Add a SQLite database so events are queryable.
4. Add a minimal FastAPI REST API.
5. Add a basic web dashboard.

Everything else — multi-camera, Jetson, cloud — follows from that foundation.

---

## 2. Project Goals

As stated in the README and confirmed by the code, V1 goals are:

1. Connect a single YOLOv8 PPE checkpoint (`best.pt`) to a live RTSP camera.
2. Evaluate **per-person PPE compliance** rather than raw object detection.
3. Confirm violations only after a **temporal threshold** (default: 2 s).
4. Store annotated evidence images for confirmed violations only.
5. Lay a foundation for future FastAPI, PostgreSQL, and React layers **without rewriting the pipeline**.

**Goals explicitly deferred from V1** (README line 66):
FastAPI, React, PostgreSQL, Redis, Kafka, Kubernetes, face recognition, worker identity, model retraining.

---

## 3. Current System Overview

| Attribute | Value |
|-----------|-------|
| Language | Python 3.10+ (developed on Python 3.12 in WSL) |
| Model | YOLOv8 via Ultralytics (`best.pt`, 6 MB, 10 classes) |
| Runtime | CPU (development); CUDA optional with automatic fallback |
| Environment | WSL on Windows D:\Tata-ppe |
| Camera support | RTSP (code present, untested against physical camera), video file, image |
| Output | Annotated JPEG evidence + flat JSONL event log |
| API | **None** |
| Web UI | **None** |
| Database | **None** |
| Deployment | Manual CLI; no Docker, no service file, no CI/CD |

**VERIFIED:** The pipeline has been run against at least one test video and produced 5 violation events in `evidence/events.jsonl` and 5 JPEG images in `evidence/CAM-001/2026-08-21/`. Evidence paths in JSONL contain `/mnt/d/Tata-ppe/...` confirming execution from WSL.

---

## 4. Current Architecture

The following is the **actual** current architecture, derived entirely from reading the source code.

```
Input Source
    ├─ ImageSource          (src/video/file_source.py:ImageSource)
    ├─ VideoFileSource      (src/video/file_source.py:VideoFileSource)
    └─ RTSPCamera           (src/video/camera.py:RTSPCamera)
          │
          ▼
    VideoFrame              (src/video/source.py:VideoFrame)
    (image: np.ndarray, timestamp, camera_id, frame_index, source_fps)
          │
          ▼
    [InferenceGate]         (src/video/frame_processor.py:InferenceGate)
    Rate-limits inference to target_fps (default 10 FPS)
          │
          ▼
    YOLODetector.predict()  (src/inference/detector.py:YOLODetector)
    Ultralytics YOLO → normalized DetectionResult
          │
          ▼
    DetectionResult         (src/inference/result.py:DetectionResult)
    list[Detection(class_id, label, confidence, bbox)]
          │
          ▼
    ByteTracker.update()    (src/tracking/tracker.py:ByteTracker)
    Person detections → list[TrackedDetection(track_id, bbox)]
          │
          ▼
    associate_ppe()         (src/compliance/association.py:associate_ppe)
    Spatial PPE-to-person assignment → list[PersonPPEState]
          │
          ▼
    ComplianceEngine.evaluate()  (src/compliance/rules.py:ComplianceEngine)
    Per-person zone check → list[ComplianceResult]
          │
          ▼
    TemporalViolationFilter.update()  (src/events/temporal.py:TemporalViolationFilter)
    Confirmation + cooldown → list[PPEViolationEvent]
          │
          ▼
    EvidenceCapture.save()  (src/evidence/capture.py:EvidenceCapture)
    Annotated JPEG → evidence/<camera_id>/<date>/event-<id>.jpg
          │
          ▼
    LocalEventPublisher.publish()  (src/events/publisher.py:LocalEventPublisher)
    Log line + JSONL append → evidence/events.jsonl
          │
          ▼
    annotate() + cv2.imshow()  (src/viz.py + src/main.py)
    Optional display or video file output
```

**What does NOT exist:**
REST API, web interface, relational database, message queue, multi-camera manager,
continuous video recording, system health endpoint, authentication.

---

## 5. Codebase Structure

```
d:\Tata-ppe\
│
├── config/
│   ├── app.yaml              Model, tracking, taxonomy, zones, violations, evidence config
│   └── camera.yaml           Single CAM-001 definition (RTSP_URL from env)
│
├── evidence/                 Runtime output — gitignored except .gitkeep
│   ├── events.jsonl          22 entries from test runs (VERIFIED)
│   └── CAM-001/2026-08-21/   5 JPEG evidence images ~120 KB each (VERIFIED)
│
├── models/
│   └── best.pt               YOLOv8, 6.0 MB (gitignored, present locally)
│
├── scripts/
│   ├── download_model.py     HuggingFace local-only download
│   ├── make_test_video.py    Generates test.mp4 from test.jpg
│   ├── rtsp_reconnect_smoke.py  Reconnect smoke test (no model)
│   └── test_image.py         Thin CLI wrapper
│
├── src/
│   ├── main.py               CLI: argparse, run_pipeline(), _run_live()
│   ├── pipeline.py           PPEPipeline.build() + .process() orchestrator
│   ├── taxonomy.py           Model class names → canonical PPE mapping
│   ├── geometry.py           Pure IoU, containment, center, subregion
│   ├── exceptions.py         PPEError hierarchy (7 exception types)
│   ├── viz.py                OpenCV annotation overlay
│   ├── security.py           RTSP URL credential redaction
│   ├── logging_setup.py      Root logger → stdout
│   ├── config/settings.py    YAML + env → frozen AppConfig dataclass tree
│   ├── inference/
│   │   ├── detector.py       Detector ABC + YOLODetector + helpers
│   │   └── result.py         Detection, DetectionResult, LatencyBreakdown
│   ├── video/
│   │   ├── source.py         VideoSource ABC + VideoFrame
│   │   ├── camera.py         RTSPCamera with reconnect loop
│   │   ├── file_source.py    ImageSource, VideoFileSource
│   │   └── frame_processor.py  LatestFrameBuffer, InferenceGate
│   ├── tracking/tracker.py   Tracker ABC + ByteTracker + NoOpTracker
│   ├── compliance/
│   │   ├── association.py    associate_ppe() spatial scoring
│   │   └── rules.py          ComplianceEngine zone evaluation
│   ├── events/
│   │   ├── temporal.py       TemporalViolationFilter state machine
│   │   ├── violation.py      PPEViolationEvent dataclass
│   │   └── publisher.py      EventPublisher ABC + LocalEventPublisher
│   ├── evidence/capture.py   EvidenceCapture.save() + .prune()
│   └── metrics/collector.py  PipelineMetrics + MetricsSnapshot
│
├── tests/                    51 tests across 14 files (no physical camera required)
│   ├── test_association.py   8 tests
│   ├── test_camera.py        7 tests (mocked FakeCapture)
│   ├── test_cli.py           3 tests
│   ├── test_compliance.py    5 tests
│   ├── test_config.py        3 tests
│   ├── test_detection.py     3 tests
│   ├── test_evidence.py      4 tests
│   ├── test_model_load.py    2 tests (1 skipped without best.pt)
│   ├── test_model_mapping.py 2 tests
│   ├── test_pipeline.py      3 tests (FakeDetector)
│   ├── test_real_image_association.py  1 test (real coords)
│   ├── test_temporal.py      5 tests
│   └── test_tracking_and_security.py  5 tests
│
├── .env.example              RTSP_URL, DEVICE, thresholds template
├── .gitignore                Secrets, venv, evidence, models excluded
├── pyproject.toml            name=tata-ppe, Python>=3.10, pytest config
├── requirements.txt          51 fully pinned dependencies
└── run_all.sh                Bash: process all images in test_images/

NOT PRESENT: Dockerfile, docker-compose.yml, CI config, API code, DB code, frontend.
```

---

## 6. Component-by-Component Audit

### 6.1 Configuration — `src/config/settings.py`

**Status: IMPLEMENTED — Well structured**

- **VERIFIED:** `load_settings()` reads `config/app.yaml` + `config/camera.yaml`, applies `.env` and env overrides.
- **VERIFIED:** Uses `${VAR:-default}` template expansion (custom regex — not shell subprocess).
- **VERIFIED:** Produces frozen dataclass tree: `AppConfig → ModelConfig, TrackingConfig, AssociationConfig, TaxonomyConfig, ZoneConfig, ViolationConfig, EvidenceConfig, LoggingConfig, CameraConfig`.
- **VERIFIED:** Multiple cameras can be listed in `camera.yaml` but only one is used per pipeline invocation.
- **LIMITATION:** No schema validation — invalid YAML values silently fall to defaults.
- **LIMITATION:** `CameraConfig.rtsp_url` is only validated at camera open time, not at config load time.

### 6.2 Inference — `src/inference/detector.py`, `src/inference/result.py`

**Status: IMPLEMENTED — Strong abstraction**

- **VERIFIED:** `Detector` ABC defines `class_names`, `device`, `model_path`, `predict()`.
- **VERIFIED:** `YOLODetector` is the only concrete implementation; uses `ultralytics.YOLO`.
- **VERIFIED:** `Detection` dataclass has zero Ultralytics types — correctly isolated.
- **VERIFIED:** CUDA fallback: on `RuntimeError` with "CUDA" in message, moves model to CPU once.
- **VERIFIED:** `resolve_model_path()` tries candidate paths, then HuggingFace local cache with `local_files_only=True`. Never downloads.
- **VERIFIED:** `warmup()` runs a zero-pixel dummy prediction at startup to prime JIT compilation.
- **LIMITATION:** `PPEDetector = YOLODetector` alias at line 305 is unused — dead code.
- **LIMITATION:** `imgsz` is passed on every `predict()` call. Minor inefficiency for fixed-size inference.

### 6.3 Video Sources — `src/video/`

**Status: IMPLEMENTED — Good abstraction**

- **VERIFIED:** `VideoSource` ABC defines `camera_id`, `connect()`, `frames()`, `stop()`.
- **VERIFIED:** `RTSPCamera` implements reconnect loop, stale-frame dropping, measured FPS, frame skip counter.
- **VERIFIED:** Buffer size set to 1 (`CAP_PROP_BUFFERSIZE=1`) to prefer latest frame over backlog.
- **VERIFIED:** `LatestFrameBuffer` is thread-safe (threading.Lock), single-slot, drops previous unread frame.
- **VERIFIED:** `InferenceGate` rate-limits by `target_fps` using `time.monotonic()`.
- **LIMITATION:** RTSP tested only against `FakeCapture` in unit tests — **not against a physical camera**.
- **LIMITATION:** No `USBCamera` or `ONVIFCamera` implementation.
- **LIMITATION:** `VideoFrame` does not carry explicit resolution fields (inferrable from `image.shape`).

### 6.4 Taxonomy — `src/taxonomy.py`

**Status: IMPLEMENTED — Correct design**

- **VERIFIED:** `resolve_taxonomy()` reads class names from the loaded model, not hardcoded.
- **VERIFIED:** `_norm()` strips non-alphanumeric, lowercases — case-insensitive alias matching.
- **VERIFIED:** Both positive and negative polarities per PPE type.
- **VERIFIED:** Unmatched classes logged as `"other"` — not silently dropped or misclassified.

### 6.5 Tracking — `src/tracking/tracker.py`

**Status: IMPLEMENTED — Custom IoU tracker (not official ByteTrack library)**

- **VERIFIED:** `ByteTracker` implements two-stage greedy IoU matching: high-confidence first, then low-confidence against remaining confirmed tracks.
- **VERIFIED:** `Tracker` ABC allows drop-in replacement without touching pipeline.
- **LIMITATION:** No Kalman filter. Tracks update by direct bbox replacement. ID switches can occur when workers cross paths.
- **LIMITATION:** No re-identification. Worker re-entering frame gets a new track ID.
- **LIMITATION:** `_greedy_match` is O(T×D) — fine for <10 persons per camera.

### 6.6 PPE Association — `src/compliance/association.py`

**Status: IMPLEMENTED — Spatially aware, well-tested**

- **VERIFIED:** `associate_ppe()` uses `score = max(IoU(ppe_bbox, region), containment_ratio, center_point_in_region)`.
- **VERIFIED:** Greedy assignment sorted by `score × confidence` — highest-confidence wins globally.
- **VERIFIED:** Each PPE bbox assigned to at most one person (set deduplication by bbox key).
- **VERIFIED:** Prefers positive evidence over negative when confidence is close.
- **VERIFIED:** Tested against real bounding box coordinates from the actual test image.
- **LIMITATION:** `head_height_ratio=0.35` is global. Shallow camera angles may misclassify head region.
- **LIMITATION:** Greedy approach can fail when one PPE item is equidistant between two workers.

### 6.7 Compliance — `src/compliance/rules.py`

**Status: IMPLEMENTED — Safety-biased correctly**

- **VERIFIED:** Required PPE not positively associated → treated as missing (conservative, correct for safety).
- **VERIFIED:** Negative-class detections (`NO-Hardhat`, etc.) reinforce missing status.
- **VERIFIED:** Zone PPE types not in taxonomy → logged as skipped, not silently required.
- **LIMITATION:** Single zone per camera. No multi-zone polygons or time-of-day switching.

### 6.8 Temporal Logic — `src/events/temporal.py`

**Status: IMPLEMENTED — Correct state machine, thoroughly tested**

| Event | Behavior |
|-------|----------|
| First frame: PPE missing | Set `missing_since`, no event |
| Missing < `confirmation_seconds` | No event |
| Missing ≥ `confirmation_seconds` | Emit event, set `last_event_at` |
| Missing within cooldown | No new event |
| PPE appears | Reset `missing_since` |
| PPE disappears again | New confirmation window |
| Person leaves frame | Clear `missing_since`, preserve cooldown |

- **LIMITATION:** State is in-memory. Crash → all temporal state lost. Confirmation windows restart.
- **LIMITATION:** No distinction between "deliberately removed" and "momentary tracker loss".

### 6.9 Events — `src/events/violation.py`, `src/events/publisher.py`

**Status: IMPLEMENTED — Minimal but correct**

- **VERIFIED:** `PPEViolationEvent` is immutable (frozen dataclass). Fields: `event_id` (12-hex UUID fragment), `camera_id`, `timestamp`, `person_id`, `violation_type`, `confidence`, `evidence_path`, `zone`, `bbox`.
- **VERIFIED:** `EventPublisher` ABC allows future HTTP/Redis/Kafka publishers without changing pipeline.
- **VERIFIED:** `LocalEventPublisher` logs at WARNING level + prints to stdout + appends JSONL.
- **LIMITATION:** JSONL append is synchronous — blocks pipeline thread at high event rates.
- **LIMITATION:** No deduplication across process restarts — JSONL may have duplicates.
- **LIMITATION:** JSONL has no index — O(N) scan to query events.

### 6.10 Evidence — `src/evidence/capture.py`

**Status: IMPLEMENTED — Working but has portability bug**

- **VERIFIED:** `EvidenceCapture.save()` saves annotated JPEG to `evidence/<camera_id>/<YYYY-MM-DD>/event-<id>.jpg`.
- **VERIFIED:** `EvidenceCapture.prune()` deletes JPEGs older than `retention_days` by file mtime.
- **VERIFIED (BUG):** Evidence paths stored in `events.jsonl` are **absolute WSL paths** (`/mnt/d/Tata-ppe/evidence/...`). These are invalid on Linux deployments or Windows without WSL mounted at `/mnt/d`. This must be fixed by storing relative paths.
- **LIMITATION:** Pruning called only at pipeline shutdown. Evidence accumulates if process runs for weeks.
- **LIMITATION:** No SHA-256 hash stored — integrity cannot be verified.
- **LIMITATION:** No thumbnail generation — full-frame JPEG (~120 KB) must be served for any UI.

### 6.11 Metrics — `src/metrics/collector.py`

**Status: IMPLEMENTED — In-memory, console-only**

- **VERIFIED:** Tracks inference FPS, end-to-end latency, person/detection/violation counts.
- **VERIFIED:** CPU percent via `psutil`, GPU percent via `pynvml` — both degrade gracefully if unavailable.
- **LIMITATION:** No HTTP health endpoint. No Prometheus export. No disk or RAM monitoring.
- **LIMITATION:** `violations` counter is cumulative lifetime total, not a rate.

### 6.12 Visualization — `src/viz.py`

**Status: IMPLEMENTED — Functional**

- **VERIFIED:** `annotate()` draws bounding boxes, person ID, PPE status (OK/MISSING), metrics overlay, timestamp.
- **VERIFIED:** Frame is copied before annotation — original not mutated.
- **LIMITATION:** No distinction between evidence-quality annotation (saved to disk) and display-quality annotation (shown on screen). Both use the same annotated frame, which means evidence images always include the metrics overlay.

### 6.13 CLI — `src/main.py`

**Status: IMPLEMENTED — Complete for V1**

- **VERIFIED:** Supports `--mode image|video|rtsp`, `--source`, `--camera`, `--no-display`, `--save-output`, `--max-frames`, `--max-seconds`, `--log-level`.
- **VERIFIED:** SIGINT/SIGTERM handled cleanly via signal handlers.
- **VERIFIED:** RTSP mode uses background ingest thread + `LatestFrameBuffer`.
- **LIMITATION:** No `--config-file` flag for custom app.yaml path (only `--config-root`).
- **LIMITATION:** `_can_display()` creates/destroys a window probe — noisy on headless servers.

---

## 7. Detector Architecture

### Current State

```
Detector (ABC) — src/inference/detector.py
    └── YOLODetector
            └── ultralytics.YOLO → models/best.pt
```

### Interface (VERIFIED)

```python
class Detector(ABC):
    @property def class_names(self) -> dict[int, str]: ...
    @property def device(self) -> str: ...
    @property def model_path(self) -> Path: ...
    def predict(self, frame: np.ndarray) -> DetectionResult: ...
```

### Extensibility

Downstream components (`PPEPipeline`, `associate_ppe`, `ComplianceEngine`, `TemporalViolationFilter`) consume only `DetectionResult` objects — pure Python dataclasses with no Ultralytics types. **A future ONNXDetector or TensorRTDetector only needs to implement these four members. Nothing else changes.**

### Future Detector Backends (NOT IMPLEMENTED)

| Backend | Use Case | Blocker |
|---------|----------|---------|
| `YOLODetector` | Development, CPU/GPU via PyTorch | **CURRENT** |
| `ONNXDetector` | Cross-platform portability, CPU/Intel | Export + onnxruntime |
| `OpenVINODetector` | Intel CPU/iGPU acceleration | OpenVINO toolkit |
| `TensorRTDetector` | NVIDIA Jetson, maximum throughput | TRT engine build on Jetson |

---

## 8. Camera Architecture

### Current State (VERIFIED)

```
VideoSource (ABC) — src/video/source.py
    ├── RTSPCamera           — src/video/camera.py       [IMPLEMENTED]
    ├── VideoFileSource      — src/video/file_source.py  [IMPLEMENTED]
    └── ImageSource          — src/video/file_source.py  [IMPLEMENTED]
```

### Recommended Future Architecture (RECOMMENDED)

```
VideoSource (ABC)
    ├── RTSPCamera           OpenCV / FFmpeg RTSP     [CURRENT]
    ├── ONVIFCamera          ONVIF protocol           [NOT IMPLEMENTED]
    ├── USBCamera            OpenCV VideoCapture(0)   [NOT IMPLEMENTED]
    ├── VideoFileSource      Development use          [CURRENT]
    └── ImageSource          Development use          [CURRENT]
```

### Recommended Camera Interface Additions (RECOMMENDED)

```python
# Current VideoSource interface is sufficient for V1.
# For industrial deployment, add:
def get_metadata(self) -> dict: ...   # resolution, codec, manufacturer
@property def stream_url(self) -> str: ...  # redacted URL for display
```

---

## 9. RTSP Architecture

### Current Implementation (VERIFIED)

`RTSPCamera` uses `cv2.VideoCapture(url, cv2.CAP_FFMPEG)` with:
- `CAP_PROP_BUFFERSIZE=1` — prefer latest frame, drop backlog
- `CAP_PROP_OPEN_TIMEOUT_MSEC` and `CAP_PROP_READ_TIMEOUT_MSEC`
- Reconnect loop: disconnect → sleep → reopen → resume

Background ingest thread + `LatestFrameBuffer` separates capture from inference in RTSP mode.

### RTSP Testing Status

**NOT VERIFIED against a physical camera.** All RTSP tests use `FakeCapture` in `tests/test_camera.py`. Real behavior depends on camera vendor, codec, authentication, and network stack.

### Known RTSP Risks

- Some cameras require H.264 baseline profile; HEVC requires FFmpeg with H.265 support.
- Non-standard Digest/Basic authentication may fail silently.
- `CAP_PROP_OPEN_TIMEOUT_MSEC` may be ignored on some OpenCV builds.
- Windows OpenCV RTSP support is more limited than Linux OpenCV.

### Live Streaming to Browser (NOT IMPLEMENTED)

RTSP cannot be exposed directly to a browser. Options:

| Approach | Latency | Complexity | V1 Suitability |
|----------|---------|------------|----------------|
| RTSP → MJPEG over HTTP | 200–500 ms | Low | ✅ Recommended for V1 |
| RTSP → HLS (FFmpeg) | 3–10 s | Medium | ⚠️ Acceptable |
| RTSP → WebRTC (mediamtx) | <100 ms | High | ❌ Overkill for V1 |

**RECOMMENDATION:** MJPEG streaming from the FastAPI backend. The pipeline already annotates frames; the API simply JPEG-encodes and streams via `multipart/x-mixed-replace`.

---

## 10. Tracking

### Current Implementation (VERIFIED)

`ByteTracker` in `src/tracking/tracker.py` is a custom lightweight IoU tracker (NOT the official ByteTrack library):

1. Split detections: high-confidence (`>= track_high_thresh=0.5`), low-confidence.
2. Greedily match high-confidence to existing tracks by IoU.
3. Match low-confidence to remaining confirmed tracks.
4. Create new tracks for unmatched high-confidence detections above `new_track_thresh=0.6`.
5. Expire tracks not updated for `track_buffer=30` frames.

### Limitations

- No Kalman filter → no velocity prediction → ID switches when workers cross.
- No appearance features → cannot re-identify worker re-entering frame.
- Greedy matching (not Hungarian algorithm) → suboptimal for many simultaneous tracks.

### `Tracker` ABC supports drop-in replacement (VERIFIED)

```python
class Tracker(ABC):
    def update(self, detections: list[Detection], frame: np.ndarray | None) -> list[TrackedDetection]: ...
    def reset(self) -> None: ...
```

Future: `BoT-SORT`, `OC-SORT`, `StrongSORT`.

---

## 11. PPE Association

### Current Implementation (VERIFIED)

```python
score = max(
    iou(ppe_bbox, person_region),
    containment_ratio(ppe_bbox, person_region),
    1.0 if center(ppe_bbox) in person_region else 0.0
)
```

Greedy assignment by `score × confidence`. Each PPE bbox assigned to at most one person.

### Region Mapping (VERIFIED from config/app.yaml)

| PPE Type | Region | Ratio |
|----------|--------|-------|
| helmet | head | Top 35% of person bbox |
| mask | head | Top 35% of person bbox |
| safety_vest | torso | 20%–75% of person bbox height |

### Association Config Defaults (VERIFIED)

| Parameter | Default |
|-----------|---------|
| `min_iou` | 0.05 |
| `min_containment` | 0.25 |
| `min_score` | 0.25 |
| `head_height_ratio` | 0.35 |

---

## 12. Compliance

### Current Implementation (VERIFIED)

`ComplianceEngine.evaluate()` in `src/compliance/rules.py`:
- Required PPE not present → missing (safety-biased: unknown = missing).
- Negative-class detections (`NO-Hardhat`) reinforce missing status.
- Returns `ComplianceResult(compliant: bool, missing_ppe: tuple, present_ppe: tuple, zone, confidence)`.

### Zone (VERIFIED)

```yaml
# config/app.yaml
zones:
  general:
    required_ppe: [helmet, safety_vest, mask]
```

All three PPE types required. One zone only. All cameras assigned to `general`.

---

## 13. Temporal Logic

### State Machine (VERIFIED)

`TemporalViolationFilter` in `src/events/temporal.py`:

| Scenario | Behavior |
|----------|----------|
| First missing frame | Set `missing_since`, no event |
| Missing < 2.0 s | No event |
| Missing ≥ 2.0 s | Emit event; set `last_event_at` |
| Missing within 30 s of last event | No new event (cooldown) |
| PPE appears | Reset `missing_since` |
| PPE disappears again | New 2 s window starts |
| Worker leaves frame | Clear `missing_since`; preserve cooldown |

### Defaults (VERIFIED)

| Parameter | Default | Source |
|-----------|---------|--------|
| `confirmation_seconds` | 2.0 | `config/app.yaml`, env `VIOLATION_CONFIRMATION_SECONDS` |
| `cooldown_seconds` | 30.0 | `config/app.yaml`, env `VIOLATION_COOLDOWN_SECONDS` |

### RECOMMENDED Production Defaults

| Parameter | RECOMMENDED | Rationale |
|-----------|-------------|-----------|
| `confirmation_seconds` | 3–5 s | Reduce false positives from transient occlusions |
| `cooldown_seconds` | 60–120 s | Match safety officer response time |

---

## 14. Evidence

### Current Implementation (VERIFIED)

```
evidence/
    CAM-001/
        2026-08-21/
            event-ba20c2393b97.jpg   (annotated JPEG, ~120 KB)
            event-f4e50f38ed7b.jpg
            ...5 total images
    events.jsonl                     (22 entries, VERIFIED)
```

**VERIFIED BUG:** JSONL paths are absolute WSL paths (`/mnt/d/Tata-ppe/evidence/...`). Invalid on Linux deployments without WSL.

**Fix:** Store relative path only: `CAM-001/2026-08-21/event-ba20c2393b97.jpg`. Resolve against evidence root at query time.

### Evidence Immutability (RECOMMENDED)

For industrial compliance:
- Set file permissions to read-only after writing (`os.chmod(path, 0o444)`).
- Store SHA-256 hash in database.
- Do not delete via API — only via retention pruning job.

---

## 15. Local Storage

### Current State (VERIFIED)

| Data Type | Storage |
|-----------|---------|
| Evidence images | Filesystem: `evidence/<cam>/<date>/event-<id>.jpg` |
| Event log | `evidence/events.jsonl` (flat, append-only) |
| Configuration | `config/app.yaml`, `config/camera.yaml` |
| Model weights | `models/best.pt` |
| System logs | stdout (not persisted to file) |
| Database | **None** |

### Recommended V1 Storage Structure (RECOMMENDED)

```
<project_root>/
├── data/
│   ├── evidence/
│   │   └── CAM-001/
│   │       └── 2026/
│   │           └── 08/
│   │               └── 24/
│   │                   └── event-<id>.jpg
│   ├── thumbnails/          (optional, future)
│   ├── exports/             (CSV/PDF reports, future)
│   └── logs/
│       ├── app.log          (rotating file)
│       └── audit.log        (immutable audit trail)
├── db/
│   └── ppe.db               (SQLite for V1)
└── models/
    └── best.pt
```

---

## 16. Database

### Current State: NOT IMPLEMENTED

Only persistent store is `events.jsonl` (unindexed flat file) and filesystem images.

### Recommendation: SQLite for V1, PostgreSQL for Pilot

**SQLite**: Zero infrastructure, file-based, adequate for single-camera.
**PostgreSQL**: Multi-camera, multi-user, ACID, full SQL — add at pilot stage.

### Recommended V1 Schema (RECOMMENDED)

```sql
CREATE TABLE cameras (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    location TEXT,
    enabled BOOLEAN DEFAULT TRUE,
    zone TEXT DEFAULT 'general',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE violation_events (
    event_id TEXT PRIMARY KEY,
    camera_id TEXT REFERENCES cameras(id),
    timestamp TIMESTAMP NOT NULL,
    person_id INTEGER NOT NULL,
    violation_type TEXT NOT NULL,   -- HELMET_MISSING, MASK_MISSING, etc.
    confidence REAL NOT NULL,
    zone TEXT NOT NULL,
    bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE evidence_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT REFERENCES violation_events(event_id),
    relative_path TEXT NOT NULL,    -- e.g., CAM-001/2026/08/24/event-abc.jpg
    file_size_bytes INTEGER,
    sha256 TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT,
    event_type TEXT NOT NULL,       -- CAMERA_CONNECTED, INFERENCE_ERROR, etc.
    message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_violations_camera_time ON violation_events(camera_id, timestamp);
CREATE INDEX idx_violations_type ON violation_events(violation_type);
```

**Key design decision:** Database stores metadata only. JPEG images stay on the filesystem. This keeps the DB small, fast, and backupable independently of evidence images.

---

## 17. API

### Current State: NOT IMPLEMENTED

No HTTP server. No endpoints. No authentication.

### Recommended V1 API: FastAPI (RECOMMENDED)

#### Core Endpoints

```
# Health
GET /api/health               → status, timestamp, camera_count, db_status

# Cameras
GET /api/cameras              → list of camera summaries with live status
GET /api/cameras/{id}         → camera detail, current FPS, today's violations
POST /api/cameras             → add camera
PATCH /api/cameras/{id}       → update config
DELETE /api/cameras/{id}      → disable camera

# Events
GET /api/events               → paginated, filterable by camera/date/type
GET /api/events/{id}          → single event detail with evidence URL

# Evidence
GET /api/evidence/{event_id}  → JPEG image (stream from filesystem)

# Statistics
GET /api/statistics           → aggregate counts by camera/date/type

# Live
GET /api/cameras/{id}/stream  → MJPEG stream (multipart/x-mixed-replace)
```

#### Example: `GET /api/events` Response

```json
{
  "total": 847,
  "page": 1,
  "per_page": 50,
  "items": [
    {
      "event_id": "ba20c2393b97",
      "camera_id": "CAM-001",
      "camera_name": "Plant Entrance",
      "timestamp": "2026-08-21T16:01:51.002Z",
      "person_id": 1,
      "violation_type": "MASK_MISSING",
      "confidence": 0.842,
      "zone": "general",
      "evidence_url": "/api/evidence/ba20c2393b97"
    }
  ]
}
```

#### Example: `GET /api/health` Response

```json
{
  "status": "healthy",
  "timestamp": "2026-08-24T10:00:00Z",
  "cameras": {
    "CAM-001": {"connected": true, "fps": 9.8, "reconnects": 0}
  },
  "system": {
    "cpu_percent": 42,
    "ram_percent": 67,
    "disk_used_gb": 12.4,
    "disk_total_gb": 500.0
  },
  "database": "connected"
}
```

---

## 18. Web Interface

### Current State: NOT IMPLEMENTED

### Recommended V1 Pages (RECOMMENDED)

| Page | Content | Priority |
|------|---------|----------|
| Dashboard | Camera status cards, today's violations, recent events | P0 |
| Live Feed | MJPEG stream with annotation overlay | P0 |
| Events | Filterable event table with evidence thumbnails | P0 |
| Event Detail | Full evidence image, metadata, person ID | P0 |
| System Health | CPU, RAM, disk, camera FPS, inference FPS | P1 |
| Settings | Camera config, zone config, retention | P2 |
| Statistics | Violations per day/camera/type charts | P2 |

**Technology recommendation:** React + Vite (SPA) for a responsive, real-time dashboard. For V1 simplicity, Jinja2 server-rendered HTML from FastAPI is acceptable and avoids a build step.

---

## 19. Hardware

### Development Hardware (CURRENT — INFERRED)

- Windows with WSL (Ubuntu)
- `torch==2.13.0+cu130` installed but CUDA unavailable at runtime (CPU mode)
- No physical RTSP camera tested

### V1 Pilot Camera Specification (RECOMMENDED)

| Attribute | Minimum | Preferred |
|-----------|---------|-----------|
| Resolution | 1080p | 4MP |
| FPS | 15 | 25 |
| Codec | H.264 | H.264 baseline |
| Protocol | RTSP | RTSP + ONVIF |
| Power | PoE 802.3af | PoE+ 802.3at |
| Protection | IP65 | IP66/67 |
| Temperature | -10 to +50°C | -20 to +60°C |
| WDR | Yes | 120+ dB |
| IR | Yes (20m) | Yes (50m) |

### Camera Placement for PPE Detection

| PPE Type | Camera Requirement |
|----------|--------------------|
| Helmet | Camera must see top of head. Mount 3–5 m high, 20–40° downward angle. Overhead is ideal. |
| Mask | Face must cover 30+ pixels. Work at <8 m for 1080p at standard lens. |
| Vest | Torso is large — most tolerant to angle and distance. |

**What software CANNOT fix if camera placement is poor:**
- Camera angle hides helmet top → helmet always appears missing.
- Backlit scenes without WDR → silhouettes only, no PPE detail.
- Insufficient resolution at working distance → mask detection unreliable.

### V1 Edge Computer Specification (RECOMMENDED)

| Component | Specification |
|-----------|--------------|
| CPU | Intel Core i7 (12th gen) or Ryzen 7 |
| RAM | 16 GB DDR4 |
| OS SSD | 256 GB NVMe |
| Evidence SSD | 1 TB NVMe (separate drive) |
| GPU | Optional: NVIDIA RTX 3060 12GB |
| OS | Ubuntu 22.04 LTS (headless) |

### Should You Use Jetson Nano? — **RECOMMENDATION: NO**

| Issue | Detail |
|-------|--------|
| Status | **Discontinued** (EOL) |
| Memory | 4 GB shared CPU/GPU — tight with PyTorch + OS |
| Performance | ~15–25 FPS for YOLOv8n TensorRT — marginal for safety use |
| Setup | Requires ONNX export + trtexec build — not yet implemented |
| Availability | Long-term replacement parts risk |

### Jetson Orin Nano (8 GB) — RECOMMENDED IF EDGE AI REQUIRED

| Attribute | Value |
|-----------|-------|
| AI Performance | 40 TOPS |
| Memory | 8 GB LPDDR5 (shared) |
| YOLOv8s TensorRT FP16 | ~30–50 FPS (1–2 cameras) |
| Power | 7–25 W |
| Status | Active (2024+) |

---

## 20. Network Architecture

### Recommended V1 Industrial Network (RECOMMENDED)

```
┌─────────────────────────────────────────┐
│  CAMERA VLAN  192.168.10.0/24           │
│                                         │
│  CAM-001  192.168.10.101  RTSP 554/tcp  │
│  CAM-002  192.168.10.102  RTSP 554/tcp  │
│                │                        │
│           PoE Switch (managed, 8-port)  │
└────────────────┬────────────────────────┘
                 │ Trunk (tagged VLAN)
        ┌────────┴──────────┐
        │   Edge Computer   │
        │  NIC-1: 10.x/24   │ ← camera ingestion
        │  NIC-2: 1.x/24    │ ← management
        └────────┬──────────┘
                 │
        ┌────────┴──────────┐
        │  Management LAN   │
        │  192.168.1.0/24   │
        └────────┬──────────┘
                 │
        ┌────────┴──────────┐
        │  Safety Officer   │
        │  Browser (HTTPS)  │
        └───────────────────┘
```

**Key rules:**
- Camera VLAN cannot initiate connections to management LAN.
- RTSP traffic (2–4 Mbps/camera) stays isolated.
- Web UI served only on management NIC.
- No internet access from camera VLAN.

---

## 21. Security

### Current Security Audit

| Item | Status | Finding |
|------|--------|---------|
| RTSP credentials in `.env` | ✅ | Correct — gitignored |
| RTSP URL redaction in logs | ✅ | `security.py:redact_rtsp_url()` used throughout |
| No face recognition | ✅ | Tracking IDs are temporary integers only |
| No biometric data | ✅ | Confirmed in code and README |
| API authentication | ❌ | No API yet |
| Web UI authentication | ❌ | No UI yet |
| Evidence file permissions | ❌ | Default umask — not hardened |
| Log persistence | ❌ | stdout only — not persisted |
| Audit log | ❌ | Not implemented |
| Model integrity check | ❌ | No hash verification of best.pt |

### Recommended Security for V1 Pilot (RECOMMENDED)

1. **HTTPS** for web UI (Nginx reverse proxy + self-signed cert minimum).
2. **API key** or HTTP Basic Auth for the dashboard.
3. **Evidence files read-only** (`os.chmod(path, 0o444)`) after writing.
4. **Rotating audit log** for all API requests and config changes.
5. **RTSP URL never stored in database** — only in `.env` / secrets manager.

---

## 22. Monitoring and System Health

### Current State (VERIFIED)

In-memory `PipelineMetrics` tracks inference FPS, latency, person count, violations, CPU%, GPU%. Printed to console at 1-second intervals in RTSP mode.

### Missing (NOT IMPLEMENTED)

- HTTP health endpoint
- Disk usage monitoring and alert
- RAM monitoring
- Database connectivity check
- Prometheus metrics export
- Alerting (threshold breach)

### Recommended Health Endpoint (RECOMMENDED)

```json
GET /api/health → 200 OK
{
  "status": "healthy",
  "cameras": { "CAM-001": { "connected": true, "fps": 9.8 } },
  "inference": { "latency_ms": 85, "fps": 9.7 },
  "system": {
    "cpu_percent": 42, "ram_percent": 67,
    "disk_used_gb": 12.4, "disk_total_gb": 500.0,
    "gpu_percent": null
  },
  "database": "connected"
}
```

---

## 23. Failure Handling

| Failure | Detection | Response | Recovery |
|---------|-----------|----------|----------|
| Camera disconnect | `cv2.read()` → `(False, None)` | Log, set connected=False | Auto-reconnect with delay |
| Camera initial failure | `isOpened()` False | `CameraConnectionError` | Retry loop |
| Inference exception | `InferenceError` in pipeline | Log, reuse last detections | Next frame retries |
| CUDA OOM | `RuntimeError` with "CUDA" | Log, move model to CPU | Stable on CPU thereafter |
| Evidence write failure | `imwrite` returns False | `EvidenceError`, event published without path | No retry |
| JSONL append failure | `OSError` | `EvidenceError`, event lost | No retry |
| Application crash | Process exit | Temporal state lost | Manual restart |
| Disk full | `OSError` | `EvidenceError` | No precheck (gap) |
| Power failure | Process exit | Same as crash | UPS + auto-start |
| Clock error | Not detected | Wrong timestamps | NTP check not enforced |

### Missing Failure Handlers (NOT IMPLEMENTED)

- Disk-full precheck before evidence write.
- Log rotation (stdout fills disk if redirected to file).
- JSONL integrity check on startup (truncated last line).
- Systemd / Docker restart policy.
- NTP synchronization enforcement.

---

## 24. Jetson Compatibility

### Architecture Assessment

The `Detector` ABC is the **only component that needs to change** for Jetson. The rest of the pipeline is hardware-agnostic Python.

### TensorRT Migration Path (RECOMMENDED — NOT IMPLEMENTED)

```
Step 1: Export to ONNX
    python -c "from ultralytics import YOLO; YOLO('models/best.pt').export(format='onnx')"

Step 2: Convert to TRT engine on Jetson
    trtexec --onnx=models/best.onnx --saveEngine=models/best.engine --fp16

Step 3: Implement TensorRTDetector(Detector)
    class TensorRTDetector(Detector):
        def predict(self, frame) -> DetectionResult: ...
        # Load engine, run inference, return normalized DetectionResult

Step 4: Select backend via config
    model:
      backend: tensorrt   # or pytorch, onnx
      path: models/best.engine
```

**Tracking, association, compliance, temporal, evidence, API, UI: unchanged.**

---

## 25. Multi-Camera Scalability

### V1: 1–2 Cameras

One process per camera. Separate JSONL per camera (avoid concurrent-append corruption).

### V2: 2–4 Cameras (RECOMMENDED NEXT)

```
CameraManager (supervisor process)
    ├── CameraWorker(CAM-001) — subprocess or thread
    ├── CameraWorker(CAM-002)
    └── CameraWorker(CAM-003)
                │ DB write per event
        Shared SQLite/PostgreSQL
                │
            REST API + Web UI
```

### V3: 8–16+ Cameras

```
Camera Ingest Workers → frame queue (multiprocessing)
                              ↓
               Batched Inference Server (1 GPU)
                              ↓
         Per-Camera Tracking/Compliance Workers
                              ↓
                   PostgreSQL Event Database
```

Batching frames from multiple cameras into a single GPU inference call maximizes GPU utilization and is the key architectural change for large-scale deployment.

---

## 26. Current vs Target Architecture

### A. Current Architecture (VERIFIED — What Actually Exists)

```
[CLI: python -m src.main]
        │
[ImageSource | VideoFileSource | RTSPCamera]
        │ VideoFrame
[InferenceGate] → [YOLODetector] → [DetectionResult]
        │
[ByteTracker] → [associate_ppe] → [ComplianceEngine] → [TemporalViolationFilter]
        │
[EvidenceCapture → JPEG file] + [LocalEventPublisher → events.jsonl]
        │
[cv2.imshow / VideoWriter (optional)]

Storage: Filesystem JPEGs + flat JSONL only.
Interface: CLI only.
```

### B. Target V1 Architecture (RECOMMENDED)

```
[Industrial IP Camera]  rtsp://192.168.10.101:554/stream
        │
[RTSPCamera + LatestFrameBuffer + InferenceGate]
        │
[YOLODetector (or ONNX/TRT backend)]
        │
[ByteTracker] → [associate_ppe] → [ComplianceEngine] → [TemporalViolationFilter]
        │
[EvidenceCapture → data/evidence/<cam>/<Y>/<M>/<D>/event-<id>.jpg]
        │ (relative path + SHA-256)
[EventPublisher → SQLite/PostgreSQL violation_events table]
        │
[FastAPI REST API on management NIC]
        │
[React / Jinja2 Web Dashboard]
        │
[GET /api/cameras/{id}/stream → MJPEG live feed]

System Health: GET /api/health → CPU, RAM, disk, camera FPS, inference FPS
Scheduled: Daily evidence pruning (APScheduler)
Deployment: Dockerfile + systemd service
```

---

## 27. Testing Audit

### Coverage Matrix

| Component | Tests | Quality | Gaps |
|-----------|-------|---------|------|
| `YOLODetector` | 2 | Basic | CUDA fallback, malformed frame, warmup |
| `boxes_to_detections` | 2 | Good | Empty input, all-filtered |
| Taxonomy resolution | 2 | Good | — |
| `RTSPCamera` | 7 | Good | Timeout, frame skip, corrupt frame |
| `ImageSource` | 0 | None | Missing file, corrupt image |
| `VideoFileSource` | 0 | None | Missing file, empty video |
| `LatestFrameBuffer` | 0 | None | Thread safety, overflow |
| `InferenceGate` | 0 | None | Rate limiting accuracy |
| `ByteTracker` | 4 | Good | Many tracks, ID wrap-around |
| `NoOpTracker` | 0 | None | Basic |
| `associate_ppe` | 9 | Very Good | Negative polarity edge cases |
| `ComplianceEngine` | 5 | Good | Zone with no required PPE |
| `TemporalViolationFilter` | 5 | Very Good | Multiple persons, clock jump |
| `PPEViolationEvent` | 0 | None | `to_dict`, `summary` |
| `LocalEventPublisher` | 0 | None | JSONL write, disk error |
| `EvidenceCapture` | 4 | Good | Directory permissions |
| `PipelineMetrics` | 0 | None | FPS calculation, snapshot |
| `PPEPipeline` (integration) | 3 | Very Good | Multi-person, multi-camera |
| `load_settings` | 3 | Good | Missing YAML, invalid structure |
| CLI | 3 | Basic | Video mode, RTSP mode |
| Geometry | 1 | Basic | `subregion`, `area`, `center`, `point_in_bbox` |
| `redact_rtsp_url` | 1 | Good | Malformed URLs, no credentials |

**Total: ~51 tests. Tests requiring physical camera: 0. Tests requiring `best.pt`: 1 (auto-skipped).**

---

## 28. Gap Analysis

| Current State | Gap | Required Action | Priority |
|---------------|-----|-----------------|----------|
| RTSP code present but untested on real camera | No hardware validation | Test with industrial IP camera | P0 |
| Evidence paths are absolute WSL paths | Portability bug | Store relative paths in JSONL + future DB | P0 |
| Logs to stdout only | No persistent log | Add `RotatingFileHandler` | P1 |
| No database | Cannot query events | Add SQLite + SQLAlchemy | P1 |
| No REST API | No programmatic access | Implement FastAPI | P1 |
| No web UI | No user interface | Add minimal dashboard | P1 |
| Pruning at shutdown only | Evidence accumulates | Add daily scheduled prune | P1 |
| No Dockerfile | Manual deployment | Add Dockerfile + systemd | P1 |
| No disk-full check | Silent evidence loss | Check disk before write | P2 |
| Single camera JSONL | Concurrent-append risk | Per-camera JSONL or DB | P2 |
| Fixed head region ratio | Camera-angle sensitivity | Per-zone `head_height_ratio` | P2 |
| No thumbnail generation | Heavy API image serving | Generate thumbnails at save time | P2 |
| No live browser stream | Cannot view camera in UI | Add MJPEG endpoint | P2 |
| No health endpoint | No remote monitoring | Add `GET /api/health` | P2 |
| No API authentication | Open to management LAN | Add API key / Basic Auth | P2 |
| No `ImageSource` tests | Silent failures possible | Add tests | P2 |
| No Kalman filter | ID switches on fast motion | Replace with BoT-SORT | P2 |
| Temporal state in-memory | Lost on crash | Persist to DB | P3 |
| No multi-zone polygons | All workers treated equally | Spatial zone detection | P3 |
| No TensorRT backend | Cannot use Jetson optimally | Implement `TensorRTDetector` | P3 |
| No CI/CD | Changes unvalidated | Add GitHub Actions | P3 |
| No cloud backup | Evidence not replicated | Optional S3/blob sync | P3 |

---

## 29. Hardware Bill of Materials

### Conceptual V1 BOM (Single-Camera Pilot)

| Component | V1 Requirement | Specification | Purpose | Upgrade Path |
|-----------|---------------|---------------|---------|--------------|
| IP Camera | 1× | 1080p/25fps, H.264, RTSP, ONVIF, PoE, IP66, WDR 120dB, IR 30m | PPE detection | Add more cameras |
| Varifocal lens | If supported | 2.8–12mm | Adjust FOV | Fixed after installation |
| PoE Switch | 1× (8-port) | 802.3af/at, VLAN, managed | Camera power + isolation | 16-port for 8 cameras |
| Edge Computer | 1× | i7/16GB RAM/256GB OS SSD/1TB evidence SSD, Ubuntu 22.04 | Inference + storage | Add GPU or Jetson |
| Discrete GPU | Optional | NVIDIA RTX 3060 12GB | Accelerate inference | Jetson for embedded |
| Evidence SSD | 1× | 1 TB NVMe (separate from OS) | Evidence images | Scale to 2–4 TB |
| UPS | 1× | 1000 VA / 600W | Power failure protection | Higher capacity |
| Management switch | 1× | 5-port unmanaged | Management LAN | — |
| Industrial enclosure | Optional | IP54 steel cabinet | Hardware protection | Active cooling |
| Monitor | Optional | 24" HDMI | Commissioning only | Remove after go-live |
| Cat6 cabling | As needed | Conduit in plant | PoE + data | — |

---

## 30. Storage Sizing

### Evidence-Only Model (CURRENT DESIGN — RECOMMENDED)

No continuous video recording. Evidence images saved only for confirmed violations.

**Evidence image size:** ~120 KB (annotated 1080p JPEG at quality=90, VERIFIED)

```
daily_images = violations_per_hour × operating_hours × camera_count
daily_storage_MB = daily_images × 0.12 MB
```

| Scenario | Rate | Cameras | Daily Images | Daily Storage | 30-Day Storage |
|----------|------|---------|--------------|---------------|----------------|
| Low | 2/hr × 8h | 1 | 16 | 2 MB | 60 MB |
| Medium | 10/hr × 8h | 1 | 80 | 10 MB | 300 MB |
| Medium | 10/hr × 8h | 4 | 320 | 38 MB | 1.1 GB |
| High | 30/hr × 8h | 4 | 960 | 115 MB | 3.5 GB |
| Worst | 60/hr × 8h | 16 | 7680 | 900 MB | 27 GB |

**A 1 TB SSD covers the worst-case scenario for 37 months.** Evidence-only storage is negligible.

**Database size:** ~500 bytes/event × 10,000 events/month = 5 MB/month → SQLite adequate for years.

### If Continuous Recording Were Added (NOT CURRENT DESIGN)

```
H.264 at 2 Mbps (1080p/25fps):
  Per camera per day: 2 Mbps × 86400 s / 8 bits = 21.6 GB
  4 cameras × 30 days = 2.6 TB
  Required: 4–6 TB RAID for 30-day retention
```

This is why evidence-only storage is strongly preferred for V1.

---

## 31. Project Maturity Score

**Scale: 0=Not implemented, 1=Prototype, 2=Working dev feature, 3=Tested feature, 4=Pilot ready, 5=Production ready**

| Component | Score | Evidence |
|-----------|-------|---------|
| Model (best.pt) | 4/5 | Verified working, correct classes, tested on video |
| Inference (YOLODetector) | 3/5 | Solid abstraction, CUDA fallback, tested; no Jetson/TRT |
| Camera code (RTSPCamera) | 2/5 | Well-written, mock-tested; no physical camera test |
| RTSP | 2/5 | Code present; not verified against real stream |
| Tracking (ByteTracker) | 2/5 | Custom IoU, tested; no Kalman, may ID-switch |
| PPE Association | 3/5 | Spatial scoring, real-coord tested, edge cases covered |
| Compliance | 3/5 | Safety-biased logic, tested |
| Temporal Logic | 3/5 | State machine correct, all cases tested |
| Evidence (filesystem) | 2/5 | Save+prune work; portability bug (absolute WSL paths) |
| Storage | 1/5 | JSONL only; no queryable store |
| Database | 0/5 | Not implemented |
| API | 0/5 | Not implemented |
| Web UI | 0/5 | Not implemented |
| Security | 1/5 | RTSP redaction done; no auth, HTTPS, audit |
| Monitoring | 1/5 | Console metrics only; no health endpoint |
| Hardware | 1/5 | Dev PC only; no plant hardware |
| Deployment | 0/5 | Manual CLI; no Docker/service/CI |
| Testing | 2/5 | 51 tests, good core coverage; gaps in file sources, publisher, CLI |

---

## 32. Recommended Roadmap

### Phase 0 — Stabilization (1–2 weeks)
- Fix evidence path portability bug (relative paths)
- Add rotating file log handler
- Add `ImageSource`, `VideoFileSource`, `InferenceGate` unit tests
- Add disk-space check before evidence write
- Validate JSONL last-line integrity on startup

### Phase 1 — Physical Camera Validation (2–4 weeks)
- Procure industrial IP camera (ONVIF, PoE, 1080p)
- Set up camera network (PoE switch, VLAN, static IPs)
- Test `RTSPCamera` against real stream
- Tune association and temporal parameters for actual camera angle
- Document camera installation standards

### Phase 2 — Database (2–3 weeks)
- SQLite + SQLAlchemy ORM
- Alembic migrations for schema
- Replace JSONL with DB inserts (keep JSONL as fallback log)
- Store relative evidence paths + SHA-256 hash

### Phase 3 — REST API (2–4 weeks)
- FastAPI: health, cameras, events, evidence, statistics
- Pagination and filtering for events
- MJPEG streaming endpoint for live view
- API key authentication
- OpenAPI documentation auto-generated

### Phase 4 — Web Dashboard (3–5 weeks)
- Dashboard: camera status, recent violations
- Events page: filterable table with thumbnails
- Event detail: full evidence image + metadata
- Live feed: MJPEG stream
- System health page

### Phase 5 — Evidence Scheduling (1 week)
- APScheduler daily pruning job
- Disk usage alert threshold

### Phase 6 — Deployment (2 weeks)
- Dockerfile + docker-compose.yml
- Systemd service file for auto-start
- Nginx reverse proxy with HTTPS

### Phase 7 — Multi-Camera (3–4 weeks)
- Camera manager: one pipeline per camera
- Second camera in YAML + physical camera
- Centralized database writes

### Phase 8 — Edge Hardware (4–8 weeks)
- Procure Jetson Orin Nano or x86 + GPU
- Export best.pt to ONNX
- Implement ONNXDetector (cross-platform validation)
- For Jetson: TRT export + TensorRTDetector

### Phase 9 — Production Hardening (4–6 weeks)
- RBAC (admin/supervisor/viewer)
- Audit log
- Evidence immutability
- NTP check at startup
- 2-week sustained load test

---

## 33. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Real camera RTSP incompatible with OpenCV build | Medium | High | Test early; have FFmpeg source build fallback |
| Camera angle prevents helmet detection | High | High | Define and enforce installation standards before procurement |
| Low-light/backlit scenes degrade detection | High | Medium | WDR camera + supplemental IR lighting |
| False violations from poor association at distance | Medium | Medium | Tune `min_score`, `head_height_ratio` per installation |
| Disk full causes silent evidence loss | Low | Medium | Add disk monitoring + alert |
| WSL paths break Linux deployment | Already present | Medium | Fix in Phase 0 |
| ID switches cause missed confirmation windows | Medium | Low | Increase `confirmation_seconds` to compensate |
| Process crash loses temporal state | Medium | Low | Persistent state in DB (Phase 2) |
| best.pt underperforms on real plant workers | Unknown | High | Validate on real footage before pilot commitment |
| Jetson setup complexity delays edge deployment | High | Low | Use x86+GPU for V1; Jetson in Phase 8 |

---

## 34. Open Questions

1. **Camera placement:** What is the exact mounting height and downward angle in the steel plant? This determines `head_height_ratio` tuning.
2. **Lighting:** Is supplemental IR lighting available, or does the camera need strong built-in IR LEDs?
3. **Safety vest color:** Fluorescent yellow/orange (easy to detect) or darker colors?
4. **Network infrastructure:** Is a managed PoE switch and camera VLAN already in place?
5. **Camera model:** Has a specific camera been selected? ONVIF compliance and RTSP URL format must be verified.
6. **Retention policy:** How long must evidence be retained for local labor safety compliance? 30 days may be insufficient.
7. **24/7 operation:** Night shift operation? IR performance in full darkness must be validated.
8. **Access control:** Who sees the web UI? Safety officers only, or supervisors and managers too?
9. **Alert escalation:** Record-and-review only, or real-time alerts (email/SMS/alarm)?
10. **Model validation:** Has best.pt been evaluated on images from the actual plant environment?

---

## 35. Final Recommendations

### What should you build next?

Based on the **actual state** of the repository — not the README's aspirations — here is the precise sequence:

**1. Test RTSP against a physical camera (P0)**
The entire value proposition depends on reading from a real camera. Nothing else matters until this works. Find a 1080p IP camera (any brand with RTSP/H.264), connect it to the development machine, run `python -m src.main --mode rtsp --camera CAM-001`, and fix whatever breaks.

**2. Fix the evidence path portability bug (P0)**
`events.jsonl` stores absolute WSL paths. They break on Linux and Windows without WSL. Fix `EvidenceCapture.save()` to store the path relative to the evidence root. This is a one-function fix.

**3. Add a SQLite database (P1)**
Without queryable event storage, the system is a black box. Add SQLite + SQLAlchemy with the schema in Section 16. This enables everything that comes after.

**4. Add a minimal FastAPI REST API (P1)**
A ~200-line FastAPI app gives you `GET /events`, `GET /cameras`, `GET /health`, and `GET /evidence/{id}`. Deploy alongside the pipeline process.

**5. Add a minimal web dashboard (P1)**
A read-only dashboard (Jinja2 server-rendered, no build step needed) showing camera status, recent violations, and evidence images. Safety officers can now use the system without reading files manually.

**6. Add Dockerfile and systemd service (P1)**
Required for pilot deployment. The system must auto-restart after power failure.

**7. Add scheduled evidence pruning (P1)**
A daily APScheduler job prevents disk accumulation without requiring a process restart.

**8. Validate on real plant footage (concurrent with all phases)**
Run the pipeline on actual footage from the target environment — even a short phone recording at the intended camera height and angle — to validate detection and association before committing to hardware procurement.

---

## 36. Final Audit Summary Table

| Area | Current Status | Maturity | Main Gap | Priority |
|------|----------------|----------|----------|----------|
| Model | YOLOv8 best.pt verified, 10 classes | 4/5 | Not validated on real plant footage | P0 |
| Inference | YOLODetector, CPU/CUDA fallback, warmup | 3/5 | No ONNX/TRT; Jetson untested | P3 |
| Camera | RTSPCamera with reconnect, mock-tested | 2/5 | No physical camera test | P0 |
| RTSP | OpenCV/FFmpeg reconnect loop implemented | 2/5 | Not tested against real camera | P0 |
| Tracking | Custom IoU ByteTracker, tested | 2/5 | No Kalman filter, ID switches possible | P2 |
| PPE Association | Spatial head/torso scoring, well-tested | 3/5 | Fixed head ratio, camera-angle sensitivity | P2 |
| Compliance | Zone-based, safety-biased, tested | 3/5 | No multi-zone polygons | P2 |
| Temporal Logic | Confirm+cooldown state machine, tested | 3/5 | In-memory only (lost on crash) | P3 |
| Evidence | JPEG + JSONL, pruning works | 2/5 | Absolute WSL paths (portability bug) | P0 |
| Storage | Filesystem + JSONL only | 1/5 | No queryable store | P1 |
| Database | Not implemented | 0/5 | SQLite + SQLAlchemy needed | P1 |
| API | Not implemented | 0/5 | FastAPI needed | P1 |
| Web UI | Not implemented | 0/5 | Dashboard needed | P1 |
| Security | RTSP redaction only | 1/5 | No auth, HTTPS, audit log | P2 |
| Monitoring | Console metrics only | 1/5 | No health endpoint, no disk alert | P2 |
| Hardware | Development PC, no plant hardware | 1/5 | Camera + edge compute not procured | P0 |
| Deployment | Manual CLI, no Docker/service | 0/5 | Dockerfile + systemd needed | P1 |
| Testing | 51 tests, good core coverage | 2/5 | File sources, publisher, CLI gaps | P2 |

---

### TOP 10 ACTIONS BEFORE PILOT DEPLOYMENT

| # | Action | Rationale |
|---|--------|-----------|
| 1 | Test RTSP against physical industrial IP camera | Core unvalidated |
| 2 | Fix evidence path portability (relative paths) | Known portability bug |
| 3 | Add SQLite database + SQLAlchemy | Required for event queries |
| 4 | Add FastAPI (events, cameras, health, evidence endpoints) | Required for web UI |
| 5 | Add minimal web dashboard (camera status + events) | Safety officer usability |
| 6 | Add Dockerfile + systemd service | Required for deployment |
| 7 | Add daily scheduled evidence pruning | Prevents disk fill |
| 8 | Add rotating file log handler | Prevents log loss |
| 9 | Validate detection on real plant footage | Unknown model performance |
| 10 | Define camera installation standards (height, angle, distance) | Physical placement determines software correctness |

---

### TOP 10 FUTURE IMPROVEMENTS

| # | Improvement | Phase |
|---|-------------|-------|
| 1 | ONVIF camera support for automated discovery | 7 |
| 2 | BoT-SORT or OC-SORT for stable tracking | 7 |
| 3 | ONNX export + ONNXDetector | 8 |
| 4 | TensorRT backend for Jetson Orin | 8 |
| 5 | Multi-camera manager with shared DB | 7 |
| 6 | RBAC (admin / supervisor / viewer) | 9 |
| 7 | Statistical dashboard (violations per shift/zone) | 4 |
| 8 | Configurable zone polygons drawn on camera image | 9 |
| 9 | Optional cloud backup for evidence images | 9+ |
| 10 | Scheduled PDF compliance report generation | 9+ |

---

*End of report. Audit date: 2026-08-24. All findings derived from direct inspection of source code unless explicitly marked RECOMMENDED, INFERRED, or NOT IMPLEMENTED.*
