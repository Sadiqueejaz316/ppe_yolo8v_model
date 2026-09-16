# PPE pipeline reference

## Checkpoint classes (`models/best.pt`)

Read from the model at load time. This checkpoint reports:

| ID | Model class | Role |
| --- | --- | --- |
| 0 | Hardhat | helmet present |
| 1 | Mask | mask present |
| 2 | NO-Hardhat | helmet missing evidence |
| 3 | NO-Mask | mask missing evidence |
| 4 | NO-Safety Vest | vest missing evidence |
| 5 | Person | trackable person |
| 6 | Safety Cone | ignored |
| 7 | Safety Vest | vest present |
| 8 | machinery | ignored |
| 9 | vehicle | ignored |

Aliases live in `config/app.yaml` `class_taxonomy`. Match is case-insensitive alphanumeric (`Hard-Hat` → `hardhat`).

## V1 zone

```yaml
zones:
  general:
    required_ppe:
      - helmet
      - safety_vest
      - mask
```

## Association regions (`config/app.yaml`)

```yaml
association:
  head_height_ratio: 0.35
  torso_y_start: 0.20
  torso_y_end: 0.75
  regions:
    helmet: head
    mask: head
    safety_vest: torso
```

## Tracking notes

ByteTrack `match_thresh` is **0.35** in `config/app.yaml` (not the 0.7 code default). A high IoU gate causes ID switches when a person moves ~20px on a 100px-wide box. `track_buffer: 40` ≈ 4s occlusion at 10 FPS.

## Run commands

```bash
python -m src.inference.test_model --image test_images/test.jpg
python -m src.main --mode image --source test_images/test.jpg --no-display --save-output outputs/test_image.jpg
python -m src.main --mode video --source test_images/test.mp4 --no-display --max-frames 30
python -m src.main --mode rtsp --camera CAM-001 --no-display --max-seconds 15
```

WSL venv (this repo's usual interpreter):

```bash
wsl -e bash -lc 'cd /mnt/d/Tata-ppe && .venv/bin/python -m src.main --help'
```

Headless / WSL: `--no-display`. Press `q` to quit when a window exists.

## Interfaces to preserve

| Interface | V1 | Later |
| --- | --- | --- |
| `VideoSource` | `RTSPCamera`, `VideoFileSource`, `ImageSource` | webcam, multi-cam |
| `Detector` | `YOLODetector` | YOLOv11, RT-DETR, ONNX, TensorRT |
| `Tracker` | `ByteTracker` | BoT-SORT, OC-SORT |
| `EventPublisher` | `LocalEventPublisher` | HTTP, Redis, Kafka |
