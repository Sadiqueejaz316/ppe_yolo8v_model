---
name: ppe-pipeline
description: Maintains the industrial PPE vision pipeline — YOLO detection, person tracking, PPE-to-person association, zone compliance, temporal confirmation, evidence, and person-centric overlay. Use when changing src/pipeline.py, src/inference, src/tracking, src/compliance, src/events, src/evidence, src/viz.py, src/taxonomy.py, src/video, src/main.py, or config/app.yaml taxonomy and zones.
---

# PPE vision pipeline

## Quick start

1. Keep the chain: detect → track **persons** → associate PPE → `PersonPPEState` → stability hold → `ComplianceEngine` → `TemporalViolationFilter` → evidence + JSONL.
2. Do not treat a raw YOLO box as a safety decision.
3. Do not replace `models/best.pt`. Map class strings in `config/app.yaml` `class_taxonomy`.
4. Run `python -m pytest tests -q` after association, compliance, temporal, or viz changes.

## Module map

| Concern | Module |
| --- | --- |
| Orchestration | `src/pipeline.py` (`PPEPipeline.process`) |
| Detector I/O | `src/inference/result.py` (`Detection` xyxy) |
| YOLOv8 adapter | `src/inference/detector.py` |
| Class names → canonical PPE | `src/taxonomy.py` |
| PPE ↔ person | `src/compliance/association.py` |
| Zone required-PPE | `src/compliance/rules.py` |
| Overlay flicker hold | `src/compliance/stability.py` |
| Dashboard read model | `src/compliance/summary.py` |
| Confirm + cooldown | `src/events/temporal.py` |
| Evidence JPEG | `src/evidence/capture.py` |
| Overlay | `src/viz.py` |
| RTSP / files | `src/video/` |

## Canonical PPE

After association, one state per person per category:

| Model classes | Field | Values |
| --- | --- | --- |
| Hardhat / NO-Hardhat | `helmet` | PRESENT, MISSING, UNKNOWN |
| Mask / NO-Mask | `mask` | PRESENT, MISSING, UNKNOWN |
| Safety Vest / NO-Safety Vest | `vest` / `safety_vest` | PRESENT, MISSING, UNKNOWN |

Cones, machinery, vehicles stay `other` (unmatched). Do not invent them into `zones.general.required_ppe`.

Unknown required items whose classes are missing from the checkpoint are skipped with `COMPLIANCE_SKIPPED_UNSUPPORTED`.

## Association (do not simplify)

1. Score each PPE box against the person's region (helmet/mask → head, vest → torso).
2. Assign each PPE box to **at most one** person (highest score × confidence).
3. Per person and category, keep the best observation per polarity.
4. Conflict (Hardhat **and** NO-Hardhat):
   - If better `rank = confidence × association_score` exceeds the other by `conflict_margin` (default 15%), take it.
   - Else higher detector confidence.
   - Else `prefer_positive_on_tie` → PRESENT.
5. Log `PPE_CONFLICT` at DEBUG.

Safety-oriented compliance: required PPE not **positively** associated is missing. `UNKNOWN` and negative classes both fail the zone check.

## Temporal events

- `violations.confirmation_seconds` (YAML/env) must elapse while the item stays missing.
- Then one `PPEViolationEvent` with `violation_type` like `HELMET_MISSING`.
- `cooldown_seconds` suppresses duplicates for the same `(person_id, item)`.
- Confirm **only** when `fresh_inference` is true. Reused boxes + a new timestamp must not confirm.

Evidence only for confirmed events: `evidence/{camera_id}/{YYYY-MM-DD}/event-{id}.jpg`. Dedup via `src/events/evidence_dedup.py`.

## Visualization

- Default operator modes: `person_summary` or `minimal` (person box + compact PPE / overall).
- `debug` draws raw YOLO boxes **and** the person summary.
- OpenCV Hershey cannot draw Unicode checkmarks; keep ASCII `+Helmet xMask`.
- `ProcessedFrame.scene_summary` is the dashboard contract — change it in `summary.py` and the API/frontend types together.

## Config and secrets

- Camera metadata: `config/camera.yaml`. Stream URL: `.env` `RTSP_URL`.
- Log streams with `redact_rtsp_url()`.
- Device: `DEVICE=auto|cpu|cuda`. CUDA missing → `GPU_UNAVAILABLE` and CPU fallback.

## Additional resources

- Class IDs, conflict examples, and run commands: [reference.md](reference.md)
