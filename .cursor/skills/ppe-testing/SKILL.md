---
name: ppe-testing
description: Writes pytest coverage for the PPE pipeline and dashboard using synthetic Detection objects without loading best.pt or a physical camera. Use when adding or fixing tests under tests/, reproducing association/compliance/temporal bugs, or verifying API/evidence behavior.
---

# PPE tests

## Command

```bash
python -m pytest tests -q
```

From WSL: `wsl -e bash -lc 'cd /mnt/d/Tata-ppe && .venv/bin/python -m pytest tests -q'`.

## Rules

- Do **not** load `models/best.pt` in unit tests except `tests/test_model_load.py`.
- Do **not** require a live RTSP camera. Mock `VideoSource` / OpenCV as existing camera tests do.
- Build people with `Detection(..., label="Person", track_id=...)` wrapped in `TrackedDetection`.
- Resolve taxonomy from a **small** `class_names` dict + `TaxonomyConfig`, same pattern as `tests/test_association.py`.

```python
from src.inference.result import Detection
from src.tracking.tracker import TrackedDetection

det = Detection(class_id=1, label="Person", confidence=0.9, bbox=(0, 0, 100, 200), track_id=17)
person = TrackedDetection(detection=det, track_id=17)
```

## What to cover when you change a module

| Change | Tests |
| --- | --- |
| Association / regions / conflict | `test_association.py` — correct person, not unrelated, overlap, Hardhat vs NO-Hardhat |
| Zone rules | `test_compliance.py` — all present → compliant; required missing → non-compliant |
| Confirm / cooldown | `test_temporal.py` — one bad frame → no event; persist → one; cooldown → still one |
| Overlay hold | `test_stability.py`, `test_visualization.py` — person-centric vs debug |
| Evidence dedup | `test_evidence.py`, `test_evidence_dedup.py` |
| Dashboard | `test_api.py` — health, summary, cameras omit RTSP, evidence path safety |
| Config / taxonomy aliases | `test_config.py`, `test_model_mapping.py` |
| Redaction | `test_tracking_and_security.py` |

## Assertions that match the product

- Association status: `present` / `missing` / `not_associated` via `PersonPPEState.status_for`.
- Operator item state: `PRESENT` / `MISSING` / `UNKNOWN` via `item_state`.
- Compliance: `ComplianceResult.compliant` and `missing_ppe` tuples of canonical names (`helmet`, not `Hardhat`).
- Events: `violation_type == "HELMET_MISSING"` (canonical upper + `_MISSING`).
- Image mode of `src.main` must **not** emit confirmed violations (no temporal persistence).

## Dashboard tests

Mock `scene_summary` / JSONL / live dir. Assert payload shape and security (no `rtsp_url`, rejected `../` evidence). Do not reimplement association in the test.

If a test needs the real checkpoint and it is missing, skip or fail with `PPE model not found` — never download a different `best.pt`.
