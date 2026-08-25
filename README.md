# Industrial PPE Detection System (V1)

Live CCTV/RTSP PPE analytics for a steel/manufacturing plant. V1 connects an existing local YOLOv8 PPE checkpoint to a single industrial camera, then evaluates **per-person compliance** instead of treating raw object detections as the safety decision.

This is a vision pipeline prototype. It is structured so a later FastAPI backend, PostgreSQL store, and React dashboard can be added without rewriting detection, tracking, association, or compliance.

## 1. Project purpose

Monitor workers in camera views and confirm PPE violations only when they persist over time.

The mandatory processing chain is:

```
RTSP / video / image
        ↓
Frame ingestion (reconnect, drop stale frames)
        ↓
PPE object detection (best.pt)
        ↓
Normalized detections
        ↓
Person tracking (temporary IDs)
        ↓
PPE ↔ person association
        ↓
Compliance rules
        ↓
Temporal confirmation + cooldown
        ↓
Violation event + evidence image
```

Detecting a `Hardhat` somewhere in the frame does **not** mean a given worker is wearing a helmet. Association and compliance are separate modules.

## 2. Architecture

```
src/
  config/settings.py          YAML + environment configuration
  inference/detector.py       Detector interface + YOLOv8 adapter
  inference/result.py         Detection dataclass (no Ultralytics types leak out)
  inference/test_model.py     Checkpoint verification CLI
  video/camera.py             RTSP source with reconnect
  video/file_source.py        Image and video-file sources
  video/frame_processor.py    Latest-frame buffer + inference FPS gate
  tracking/tracker.py         Tracker interface + ByteTrack-style IDs
  taxonomy.py                 Map actual model class names → canonical PPE
  compliance/association.py   Spatial PPE-to-person assignment
  compliance/rules.py         Zone required-PPE evaluation
  events/temporal.py          Confirmation + cooldown
  events/violation.py         PPEViolationEvent
  evidence/capture.py         Save annotated frames for confirmed events only
  pipeline.py                 End-to-end per-frame orchestration
  main.py                     image / video / rtsp entry point
```

Interfaces intended for later replacement:

| Interface | V1 implementation | Later |
| --- | --- | --- |
| `VideoSource` | `RTSPCamera`, `VideoFileSource`, `ImageSource` | webcam, multi-camera |
| `Detector` | `YOLODetector` (Ultralytics YOLOv8) | YOLOv11, RT-DETR, ONNX, TensorRT |
| `Tracker` | `ByteTracker` | BoT-SORT, OC-SORT |
| `EventPublisher` | `LocalEventPublisher` (log + JSONL) | HTTP, Redis, Kafka |

V1 does **not** include FastAPI, React, PostgreSQL, Redis, Kafka, Kubernetes, face recognition, or model retraining.

## 3. Requirements

- Python 3.10+ (this repo was developed with Python 3.12 in the existing WSL `.venv`)
- The existing local checkpoint `best.pt` (already tested; do not replace it)
- OpenCV, Ultralytics, PyTorch (see `requirements.txt`)
- FFmpeg-enabled OpenCV for RTSP
- Optional NVIDIA GPU (`DEVICE=cuda`). CPU works and is the fallback.

## 4. Installation

The existing virtualenv was created in WSL:

```bash
cd /mnt/d/Tata-ppe   # or D:\Tata-ppe from Windows
source .venv/bin/activate
pip install -r requirements.txt
```

From Windows PowerShell, run the same interpreter with:

```powershell
wsl -e bash -lc 'cd /mnt/d/Tata-ppe && .venv/bin/python -m src.main --help'
```

Copy environment defaults:

```bash
cp .env.example .env
```

Do not put real RTSP passwords in YAML or source. `.env` is gitignored.

## 5. Model setup

The model originated from `Hansung-Cho/yolov8-ppe-detection` and was already downloaded and tested in this repository.

V1 uses:

```text
models/best.pt
```

That file is a copy of the existing local Hugging Face cache checkpoint. The application will not download a different model. If `models/best.pt` is missing, it may fall back to the **local** Hugging Face cache (`local_files_only=True`) and then fail clearly.

Verify the checkpoint:

```bash
python -m src.inference.test_model --image test_images/test.jpg
```

## 6. Image inference

```bash
python -m src.inference.test_model --image test_images/test.jpg

python -m src.main --mode image --source test_images/test.jpg --no-display --save-output outputs/test_image.jpg
```

A single image shows detections, person IDs, and per-person PPE state. It does **not** emit a confirmed violation: temporal confirmation requires the condition to persist.

## 7. Video inference

A short clip can be generated from the existing test image:

```bash
python scripts/make_test_video.py

python -m src.main --mode video --source test_images/test.mp4 --no-display
```

Useful flags:

```bash
python -m src.main --mode video --source test_images/test.mp4 --no-display --max-frames 30
python -m src.main --mode video --source clip.mp4 --save-output outputs/annotated.mp4
```

Press `q` to quit when a display window is available.

## 8. RTSP configuration

Set the stream in `.env` (never in Git):

```env
RTSP_URL=rtsp://username:password@192.168.1.100:554/stream
DEVICE=auto
CONFIDENCE_THRESHOLD=0.35
TARGET_FPS=10
VIOLATION_CONFIRMATION_SECONDS=2
VIOLATION_COOLDOWN_SECONDS=30
```

Camera metadata lives in `config/camera.yaml`:

```yaml
cameras:
  - id: CAM-001
    name: Plant Entrance
    rtsp_url: ${RTSP_URL}
    enabled: true
    target_fps: 10
    reconnect_delay: 5
    confidence_threshold: 0.35
    zone: general
```

Run:

```bash
python -m src.main --mode rtsp --camera CAM-001
python -m src.main --camera CAM-001
```

Headless / time-boxed test (no physical camera required to exercise reconnect):

```bash
python -m src.main --mode rtsp --camera CAM-001 --no-display --max-seconds 15
```

Logs use a redacted URL (`rtsp://***:***@host/...`). Passwords are never printed.

Reconnect behaviour:

```text
CAMERA_CONNECTED
   → frames
   → network drop
CAMERA_DISCONNECTED
CAMERA_RECONNECTING
CAMERA_RECONNECTED
   → resume
```

The process does not exit because the camera dropped. It keeps retrying.

## 9. GPU setup

```env
DEVICE=auto    # cuda:0 if CUDA is available, else cpu
DEVICE=cpu
DEVICE=cuda
DEVICE=cuda:0
```

At startup the process prints:

```text
Device: CPU                  # or NVIDIA GPU (name)
CUDA: unavailable            # or available
Model: best.pt
```

If CUDA is requested but unavailable, the detector logs `GPU_UNAVAILABLE` and falls back to CPU. A CUDA runtime error during inference also falls back to CPU once.

This development machine ran on **CPU** (`torch 2.13.0+cu130`, `torch.cuda.is_available() == False`).

## 10. Camera configuration

| Setting | Where | Meaning |
| --- | --- | --- |
| `RTSP_URL` | `.env` | Secret stream URL |
| `target_fps` / `TARGET_FPS` | YAML / env | Desired inference rate |
| `reconnect_delay` | `config/camera.yaml` | Seconds between reconnect attempts |
| `connection_timeout` / `read_timeout` | YAML | OpenCV open/read timeouts (ms internally) |
| `frame_skip` | YAML | Optional additional skip after each kept frame |
| `zone` | YAML | Which required-PPE set to apply |

Live ingestion keeps **only the latest unread frame**. Stale frames are dropped so the pipeline prefers low latency over processing a backlog.

## 11. PPE classes supported

Class names and IDs are read from `best.pt`, not assumed in code. This checkpoint reports:

| ID | Model class | Application role |
| --- | --- | --- |
| 0 | Hardhat | helmet present |
| 1 | Mask | mask present |
| 2 | NO-Hardhat | helmet missing evidence |
| 3 | NO-Mask | mask missing evidence |
| 4 | NO-Safety Vest | vest missing evidence |
| 5 | Person | trackable person |
| 6 | Safety Cone | ignored by compliance |
| 7 | Safety Vest | vest present |
| 8 | machinery | ignored by compliance |
| 9 | vehicle | ignored by compliance |

Canonical PPE types used by compliance: `helmet`, `mask`, `safety_vest`.

Aliases are configured in `config/app.yaml` under `class_taxonomy`. If a future checkpoint uses different strings, update that mapping. Do not retrain for V1.

## 12. Compliance logic

The detector answers: **what objects are visible?**

The compliance engine answers: **is this tracked person compliant for the active zone?**

V1 zone (`config/app.yaml`):

```yaml
zones:
  general:
    required_ppe:
      - helmet
      - safety_vest
      - mask
```

Safety-oriented rule: a required item that is not **positively** associated to the person is treated as missing. Negative classes (`NO-Hardhat`, …) reinforce that. Other industrial classes (cones, machinery, vehicles) are not invented into the required set.

## 13. Violation events

A missing item must persist for `violation_confirmation_seconds` (default 2.0) before an event is created. After that, `violation_cooldown_seconds` (default 30) suppresses duplicates for the same person and PPE type.

Example:

```text
CAMERA: CAM-001
PERSON: 2
VIOLATION: HELMET_MISSING
CONFIDENCE: 0.67
TIME: 2026-08-21T16:01:51+00:00
```

Events are also appended to `evidence/events.jsonl`.

On the looping test clip (`test_images/test.mp4`, 4 seconds, 3 workers), V1 confirmed five events after the 2-second window: left worker mask + vest, center worker helmet + mask, right worker mask. A single bad frame does not create an event.

## 14. Evidence storage

Confirmed events only. Path layout:

```text
evidence/
  CAM-001/
    2026-08-21/
      event-<id>.jpg
  events.jsonl
```

Images include bounding boxes, person ID, missing PPE labels, timestamp, and camera ID. Retention is `evidence.retention_days` (default 30).

## 15. Testing

```bash
python -m pytest tests -q
python -m src.inference.test_model --image test_images/test.jpg
```

Coverage includes:

- model class mapping and box conversion
- invalid RTSP URL / open failure / reconnect
- helmet assigned to the correct person, not an unrelated worker
- overlapping workers
- all PPE present → compliant; helmet missing → violation
- one bad frame → no event; persistent miss → one event; cooldown → still one event
- evidence write path

Tests mock cameras and frames. A physical RTSP camera is not required.

## 16. Troubleshooting

| Symptom | What to check |
| --- | --- |
| `PPE model not found` | Place the existing `best.pt` at `models/best.pt`. Do not download a different file. |
| `Camera … has an empty RTSP URL` | Set `RTSP_URL` in `.env` |
| `CAMERA_OPEN_FAILED` looping | Network, credentials, codec, or firewall. Watch `CAMERA_RECONNECTING`. URL is redacted in logs. |
| No display window | WSL/headless has no GUI. Use `--no-display` and `--save-output`. |
| Slow first frame | Ultralytics/PyTorch warmup. V1 warms the model at startup; later frames are faster. |
| GPU requested but CPU used | `torch.cuda.is_available()` is false, or CUDA error triggered fallback. |
| Every frame is a violation | Increase `VIOLATION_CONFIRMATION_SECONDS`. Check association overlay before trusting events. |
| Duplicate alerts | Increase `VIOLATION_COOLDOWN_SECONDS`. |
| Wrong PPE mapping | Print classes with `python -m src.inference.test_model` and edit `class_taxonomy`. |

## Commands cheat sheet

```bash
# Verify checkpoint
python -m src.inference.test_model --image test_images/test.jpg

# Image
python -m src.main --mode image --source test_images/test.jpg --no-display --save-output outputs/test_image.jpg

# Video
python -m src.main --mode video --source test_images/test.mp4 --no-display

# RTSP
python -m src.main --mode rtsp --camera CAM-001

# Tests
python -m pytest tests -q
```

## Security

- RTSP credentials come from the environment / `.env`
- `.env` is gitignored
- logs call `redact_rtsp_url()` before printing stream locations
- no employee identity, no face recognition, tracking IDs are temporary
