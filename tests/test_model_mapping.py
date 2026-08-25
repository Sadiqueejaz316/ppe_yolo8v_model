from src.config.settings import TaxonomyConfig, PPEClassAliases
from src.inference.detector import boxes_to_detections
from src.taxonomy import resolve_taxonomy


def test_taxonomy_reads_actual_class_names():
    names = {
        0: "Hardhat",
        1: "Mask",
        2: "NO-Hardhat",
        3: "NO-Mask",
        4: "NO-Safety Vest",
        5: "Person",
        6: "Safety Cone",
        7: "Safety Vest",
        8: "machinery",
        9: "vehicle",
    }
    config = TaxonomyConfig(
        person_labels=("Person",),
        ppe={
            "helmet": PPEClassAliases(positive=("Hardhat",), negative=("NO-Hardhat",)),
            "mask": PPEClassAliases(positive=("Mask",), negative=("NO-Mask",)),
            "safety_vest": PPEClassAliases(positive=("Safety Vest",), negative=("NO-Safety Vest",)),
        },
    )
    taxonomy = resolve_taxonomy(names, config)
    assert taxonomy.person_class_ids == {5}
    assert taxonomy.positive_ids["helmet"] == {0}
    assert taxonomy.negative_ids["helmet"] == {2}
    assert "Safety Cone" in taxonomy.unmatched_labels.values()
    assert set(taxonomy.supported_ppe) == {"helmet", "mask", "safety_vest"}


def test_output_conversion_preserves_model_labels():
    names = {0: "Hardhat", 5: "Person"}
    dets = boxes_to_detections(
        xyxy=[(0, 0, 10, 10)],
        confidences=[0.9],
        class_ids=[0],
        class_names=names,
        confidence_threshold=0.1,
    )
    assert dets[0].label == "Hardhat"
    assert dets[0].class_id == 0
