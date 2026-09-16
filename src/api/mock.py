"""Development mock snapshot. Never unlabeled as production data."""

from __future__ import annotations

from datetime import datetime, timezone


def _person(person_id: int, missing: list[str]) -> dict:
    helmet = "MISSING" if "helmet" in missing else "COMPLIANT"
    mask = "MISSING" if "mask" in missing else "COMPLIANT"
    vest = "MISSING" if "safety_vest" in missing else "COMPLIANT"
    overall = "NON_COMPLIANT" if missing else "COMPLIANT"
    return {
        "person_id": person_id,
        "bbox": [0, 0, 10, 10],
        "helmet": helmet,
        "mask": mask,
        "vest": vest,
        "overall_status": overall,
        "missing_ppe": missing,
        "present_ppe": [item for item in ("helmet", "mask", "safety_vest") if item not in missing],
    }


def mock_live_payload() -> dict:
    people = [_person(index, []) for index in range(1, 10)]
    people.extend(
        [
            _person(17, ["mask"]),
            _person(21, ["helmet"]),
            _person(31, ["safety_vest"]),
        ]
    )
    compliant = sum(1 for item in people if item["overall_status"] == "COMPLIANT")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "camera_id": "CAM-001",
        "camera_name": "Plant Entrance",
        "camera_connected": True,
        "camera_fps": 9.8,
        "inference_fps": 9.1,
        "inference_latency_ms": 42.0,
        "scene_summary": {
            "total_people": len(people),
            "compliant": compliant,
            "violations": len(people) - compliant,
            "required_ppe": ["helmet", "safety_vest", "mask"],
            "people": people,
        },
        "_has_jpeg": False,
        "_mock": True,
    }


def mock_events() -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    specs = (
        ("mock00000017", 17, "MASK_MISSING", ["mask"]),
        ("mock00000021", 21, "HELMET_MISSING", ["helmet"]),
        ("mock00000031", 31, "SAFETY_VEST_MISSING", ["safety_vest"]),
    )
    rows = []
    for event_id, person_id, vtype, missing in specs:
        person = _person(person_id, missing)
        rows.append(
            {
                "event_id": event_id,
                "camera_id": "CAM-001",
                "timestamp": now,
                "person_id": person_id,
                "violation_type": vtype,
                "confidence": 0.9,
                "evidence_path": None,
                "zone": "general",
                "bbox": None,
                "ppe": person,
            }
        )
    return rows
