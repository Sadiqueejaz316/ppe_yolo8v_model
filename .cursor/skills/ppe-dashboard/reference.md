# Dashboard API contract

## `scene_summary` (pipeline → sink → summary)

```python
{
  "total_people": int,
  "compliant": int,
  "violations": int,
  "required_ppe": ["helmet", "safety_vest", "mask"],
  "people": [
    {
      "person_id": int,
      "bbox": (x1, y1, x2, y2),
      "helmet": "COMPLIANT" | "MISSING" | "PRESENT" | "UNKNOWN",
      "mask": "...",
      "vest": "...",  # canonical safety_vest, operator key vest
      "overall_status": "COMPLIANT" | "NON_COMPLIANT",
      "missing_ppe": ["helmet", ...],
      "present_ppe": [...],
    }
  ],
}
```

Required items map `PRESENT` → operator `COMPLIANT`; anything else required → `MISSING`. Non-required items stay PRESENT/MISSING/UNKNOWN.

## Event row

`violation_type`: `HELMET_MISSING` | `MASK_MISSING` | `SAFETY_VEST_MISSING`.

Human labels: `src/api/labels.py` (`Missing Helmet`, …). Frontend `missingLabel()` maps `safety_vest` → `Safety Vest`.

## Live files

```
data/live/{camera_id}.json
data/live/{camera_id}.jpg
```

JSON includes people counts, fps, `camera_connected`. Snapshot endpoint prefers in-memory `LiveFrameBuffer`, then disk JPEG.

## Mock

`src/api/mock.py` when `MOCK_DATA=true` or `dashboard.mock_data: true`. Snapshot endpoint returns 404 in mock (no fake JPEG). Health/summary set `mock: true`.
