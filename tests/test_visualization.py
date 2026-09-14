import numpy as np

from src.compliance.association import PPEObservation, PersonPPEState
from src.compliance.rules import ComplianceResult
from src.compliance.summary import build_scene_summary
from src.config.settings import PPEClassAliases, TaxonomyConfig, VisualizationConfig
from src.inference.result import Detection
from src.taxonomy import resolve_taxonomy
from src.viz import build_overlay_plan, annotate


def _taxonomy():
    return resolve_taxonomy(
        {
            0: "Hardhat",
            1: "Mask",
            2: "NO-Hardhat",
            3: "NO-Mask",
            4: "NO-Safety Vest",
            5: "Person",
            7: "Safety Vest",
        },
        TaxonomyConfig(
            person_labels=("Person",),
            ppe={
                "helmet": PPEClassAliases(positive=("Hardhat",), negative=("NO-Hardhat",)),
                "mask": PPEClassAliases(positive=("Mask",), negative=("NO-Mask",)),
                "safety_vest": PPEClassAliases(positive=("Safety Vest",), negative=("NO-Safety Vest")),
            },
        ),
    )


REQUIRED = ("helmet", "mask", "safety_vest")


def _obs(name: str, polarity: str = "positive") -> PPEObservation:
    return PPEObservation(name, polarity, 0.9, (0, 0, 10, 10), name, 1.0)


def _person(person_id: int, bbox, present: set[str]) -> PersonPPEState:
    observations = {
        item: _obs(item, "positive" if item in present else "negative")
        for item in REQUIRED
        if item in present or True
    }
    for item in list(observations):
        if item not in present:
            observations[item] = _obs(item, "negative")
    return PersonPPEState(person_id=person_id, bbox=bbox, confidence=0.9, observations=observations)


def _compliant(person_id: int, bbox, missing: tuple[str, ...] = ()) -> tuple[PersonPPEState, ComplianceResult]:
    present = {item for item in REQUIRED if item not in missing}
    person = _person(person_id, bbox, present)
    result = ComplianceResult(
        person_id=person_id,
        compliant=not missing,
        missing_ppe=missing,
        present_ppe=tuple(sorted(present)),
        zone="general",
        confidence=0.9,
    )
    return person, result


def test_person_summary_hides_raw_ppe_boxes():
    taxonomy = _taxonomy()
    person, result = _compliant(17, (40, 80, 160, 300), missing=("mask",))
    detections = [
        Detection(5, "Person", 0.9, (40, 80, 160, 300)),
        Detection(0, "Hardhat", 0.9, (60, 80, 100, 110)),
        Detection(3, "NO-Mask", 0.88, (70, 110, 95, 130)),
        Detection(7, "Safety Vest", 0.91, (50, 140, 150, 240)),
    ]
    plan = build_overlay_plan(
        detections,
        [person],
        [result],
        taxonomy,
        REQUIRED,
        visualization=VisualizationConfig(mode="person_summary"),
        image_shape=(480, 640),
    )
    assert plan.raw_boxes == ()
    assert len(plan.persons) == 1
    text = " ".join(plan.persons[0].lines)
    assert "17" in text
    assert "Mask" in text or "xM" in text
    assert plan.persons[0].overall == "NON_COMPLIANT"


def test_debug_mode_draws_raw_detections():
    taxonomy = _taxonomy()
    person, result = _compliant(17, (40, 80, 160, 300))
    detections = [
        Detection(0, "Hardhat", 0.9, (60, 80, 100, 110)),
        Detection(1, "Mask", 0.88, (70, 110, 95, 130)),
        Detection(7, "Safety Vest", 0.91, (50, 140, 150, 240)),
        Detection(5, "Person", 0.9, (40, 80, 160, 300)),
    ]
    plan = build_overlay_plan(
        detections,
        [person],
        [result],
        taxonomy,
        REQUIRED,
        visualization=VisualizationConfig(mode="debug"),
        image_shape=(480, 640),
    )
    labels = [box.label for box in plan.raw_boxes]
    assert any("Hardhat" in label for label in labels)
    assert any("Mask" in label for label in labels)
    assert any("Safety Vest" in label for label in labels)
    assert not any(label.startswith("Person") for label in labels)


def test_minimal_mode_omits_per_item_ppe():
    taxonomy = _taxonomy()
    person, result = _compliant(24, (40, 80, 160, 300))
    plan = build_overlay_plan(
        [],
        [person],
        [result],
        taxonomy,
        REQUIRED,
        visualization=VisualizationConfig(mode="minimal"),
        image_shape=(480, 640),
    )
    text = " ".join(plan.persons[0].lines)
    assert "Helmet" not in text
    assert "COMPLIANT" in text
    assert "24" in text


def test_scene_summary_matches_operator_contract():
    p17, r17 = _compliant(17, (0, 0, 80, 160), missing=("mask",))
    p21, r21 = _compliant(21, (100, 0, 180, 160), missing=("helmet", "safety_vest"))
    p24, r24 = _compliant(24, (200, 0, 280, 160))
    summary = build_scene_summary([p17, p21, p24], [r17, r21, r24], REQUIRED)
    assert summary["total_people"] == 3
    assert summary["compliant"] == 1
    assert summary["violations"] == 2
    by_id = {item["person_id"]: item for item in summary["people"]}
    assert by_id[17]["helmet"] == "COMPLIANT"
    assert by_id[17]["mask"] == "MISSING"
    assert by_id[17]["vest"] == "COMPLIANT"
    assert by_id[17]["overall_status"] == "NON_COMPLIANT"
    assert by_id[21]["helmet"] == "MISSING"
    assert by_id[24]["overall_status"] == "COMPLIANT"


def test_crowded_scene_one_overlay_per_person():
    taxonomy = _taxonomy()
    persons = []
    compliance = []
    for index in range(20):
        col = index % 5
        row = index // 5
        bbox = (20 + col * 120, 40 + row * 160, 100 + col * 120, 170 + row * 160)
        missing = ("mask",) if index % 3 == 0 else ()
        person, result = _compliant(index + 1, bbox, missing=missing)
        persons.append(person)
        compliance.append(result)
    plan = build_overlay_plan(
        [],
        persons,
        compliance,
        taxonomy,
        REQUIRED,
        visualization=VisualizationConfig(mode="person_summary", crowd_compact_threshold=8),
        image_shape=(720, 1280),
    )
    ids = [item.person_id for item in plan.persons]
    assert len(ids) == 20
    assert len(set(ids)) == 20
    origins = [item.origin for item in plan.persons]
    assert len(set(origins)) == 20


def test_annotate_keeps_frame_shape():
    taxonomy = _taxonomy()
    person, result = _compliant(1, (10, 20, 80, 120), missing=("helmet",))
    frame = np.zeros((160, 200, 3), dtype=np.uint8)
    out = annotate(
        frame,
        [Detection(2, "NO-Hardhat", 0.9, (20, 20, 50, 45))],
        [person],
        [result],
        None,
        taxonomy,
        REQUIRED,
        visualization=VisualizationConfig(mode="person_summary"),
    )
    assert out.shape == frame.shape
