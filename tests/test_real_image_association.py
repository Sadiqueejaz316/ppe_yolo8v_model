from src.compliance.association import associate_ppe, association_score, region_for
from src.config.settings import AssociationConfig, PPEClassAliases, TaxonomyConfig
from src.inference.result import Detection
from src.taxonomy import resolve_taxonomy
from src.tracking.tracker import TrackedDetection


def _taxonomy():
    names = {
        0: "Hardhat",
        1: "Mask",
        2: "NO-Hardhat",
        3: "NO-Mask",
        4: "NO-Safety Vest",
        5: "Person",
        7: "Safety Vest",
    }
    return resolve_taxonomy(
        names,
        TaxonomyConfig(
            person_labels=("Person",),
            ppe={
                "helmet": PPEClassAliases(positive=("Hardhat",), negative=("NO-Hardhat",)),
                "mask": PPEClassAliases(positive=("Mask",), negative=("NO-Mask",)),
                "safety_vest": PPEClassAliases(positive=("Safety Vest",), negative=("NO-Safety Vest",)),
            },
        ),
    )


def _person(track_id: int, bbox) -> TrackedDetection:
    det = Detection(class_id=5, label="Person", confidence=0.88, bbox=bbox, track_id=track_id)
    return TrackedDetection(detection=det, track_id=track_id)


def test_real_test_image_boxes_associate_helmets():
    taxonomy = _taxonomy()
    persons = [
        _person(1, (74.4, 133.1, 286.7, 533.0)),
        _person(2, (270.3, 106.3, 472.8, 532.4)),
        _person(3, (480.7, 116.1, 658.8, 532.7)),
    ]
    detections = [
        Detection(5, "Person", 0.886, (74.4, 133.1, 286.7, 533.0)),
        Detection(5, "Person", 0.881, (270.3, 106.3, 472.8, 532.4)),
        Detection(5, "Person", 0.857, (480.7, 116.1, 658.8, 532.7)),
        Detection(0, "Hardhat", 0.842, (507.0, 122.0, 584.4, 174.8)),
        Detection(0, "Hardhat", 0.838, (139.7, 140.5, 213.6, 193.5)),
        Detection(3, "NO-Mask", 0.822, (522.6, 190.3, 563.3, 221.0)),
        Detection(3, "NO-Mask", 0.755, (154.0, 208.3, 201.0, 244.0)),
        Detection(4, "NO-Safety Vest", 0.743, (105.3, 232.8, 250.6, 377.9)),
        Detection(3, "NO-Mask", 0.728, (344.7, 165.3, 384.5, 198.2)),
        Detection(7, "Safety Vest", 0.723, (492.1, 209.8, 627.2, 450.8)),
        Detection(7, "Safety Vest", 0.549, (291.8, 183.5, 434.7, 407.1)),
        Detection(2, "NO-Hardhat", 0.360, (331.8, 109.8, 391.3, 149.4)),
    ]
    cfg = AssociationConfig()
    left_head = region_for(persons[0].bbox, "helmet", cfg)
    left_hat = (139.7, 140.5, 213.6, 193.5)
    assert association_score(left_hat, left_head, cfg) >= cfg.min_score

    states = associate_ppe(persons, detections, taxonomy, cfg)
    by_id = {item.person_id: item for item in states}
    assert by_id[1].status_for("helmet") == "present", by_id[1].observations
    assert by_id[2].status_for("helmet") == "missing", by_id[2].observations
    assert by_id[3].status_for("helmet") == "present", by_id[3].observations
    assert by_id[1].status_for("safety_vest") == "missing"
    assert by_id[2].status_for("safety_vest") == "present"
    assert by_id[3].status_for("safety_vest") == "present"
