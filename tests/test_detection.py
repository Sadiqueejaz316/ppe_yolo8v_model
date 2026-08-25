from src.geometry import as_bbox, containment_ratio, iou
from src.inference.detector import boxes_to_detections
from src.inference.result import Detection


def test_bbox_conversion_and_confidence_filter():
    names = {0: "Hardhat", 1: "Person"}
    detections = boxes_to_detections(
        xyxy=[(10, 20, 40, 80), (5, 5, 15, 15), (100, 100, 120, 140)],
        confidences=[0.94, 0.10, 0.51],
        class_ids=[0, 1, 1],
        class_names=names,
        confidence_threshold=0.35,
    )
    assert len(detections) == 2
    assert detections[0].label == "Hardhat"
    assert detections[0].bbox == (10.0, 20.0, 40.0, 80.0)
    assert detections[1].label == "Person"
    assert all(det.confidence >= 0.35 for det in detections)


def test_bbox_swaps_inverted_coordinates():
    box = as_bbox((40, 80, 10, 20))
    assert box == (10.0, 20.0, 40.0, 80.0)


def test_iou_and_containment():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 5.0, 15.0, 15.0)
    assert 0.1 < iou(a, b) < 0.2
    inner = (1.0, 1.0, 2.0, 2.0)
    assert containment_ratio(inner, a) == 1.0
