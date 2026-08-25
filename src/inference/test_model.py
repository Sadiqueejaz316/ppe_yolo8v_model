"""Deterministic verification of the local best.pt checkpoint.

Usage:
    python -m src.inference.test_model --image test_images/test.jpg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from src.config.settings import load_settings
from src.inference.detector import YOLODetector, describe_device, resolve_device, resolve_model_path
from src.logging_setup import setup_logging


def _load_image(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _print_report(detector: YOLODetector, image: np.ndarray | None, warmup: bool) -> int:
    device_label, cuda_status = describe_device(detector.device)
    names = detector.class_names

    print("Model:")
    print(f"    {detector.architecture}")
    print()
    print("Path:")
    print(f"    {detector.model_path}")
    print()
    print("Device:")
    print(f"    {detector.device.upper() if detector.device == 'cpu' else detector.device}")
    print(f"    {device_label}")
    print(f"    CUDA: {cuda_status}")
    print()
    print("Input size:")
    print(f"    {detector.imgsz}")
    print()
    print("Classes:")
    for class_id in sorted(names):
        print(f"    {class_id}: {names[class_id]}")
    print()

    if image is None:
        print("Inference:")
        print("    skipped (no image provided)")
        return 0

    if warmup:
        detector.warmup()

    started = time.perf_counter()
    result = detector.predict(image)
    wall_ms = (time.perf_counter() - started) * 1000.0

    print("Inference:")
    print(f"    preprocessing: {result.latency.preprocess_ms:.2f} ms")
    print(f"    inference: {result.latency.inference_ms:.2f} ms")
    print(f"    postprocessing: {result.latency.postprocess_ms:.2f} ms")
    print(f"    total (reported): {result.latency.total_ms:.2f} ms")
    print(f"    total (wall): {wall_ms:.2f} ms")
    fps = 1000.0 / wall_ms if wall_ms > 0 else 0.0
    print(f"    FPS (approx): {fps:.2f}")
    print()
    print(f"Detections: {len(result.detections)}")
    for det in result.detections:
        x1, y1, x2, y2 = (round(v, 1) for v in det.bbox)
        print(f"    {det.label} id={det.class_id} conf={det.confidence:.3f} bbox=({x1}, {y1}, {x2}, {y2})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the local YOLOv8 PPE checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Optional test image")
    parser.add_argument("--model", type=str, default=None, help="Override model path")
    parser.add_argument("--device", type=str, default=None, help="auto|cpu|cuda|cuda:0")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging("INFO")
    settings = load_settings()
    model_path = args.model or settings.model.path
    device = args.device or settings.model.device
    conf = args.conf if args.conf is not None else settings.model.confidence_threshold

    resolved = resolve_model_path(model_path, project_root=settings.project_root)
    concrete_device = resolve_device(device)
    print(f"Device: {'NVIDIA GPU' if concrete_device.startswith('cuda') else 'CPU'}")
    print(f"CUDA: {'available' if concrete_device.startswith('cuda') else 'unavailable'}")
    print(f"Model: {resolved.name}")
    print()

    detector = YOLODetector(
        model_path=model_path,
        device=device,
        confidence_threshold=conf,
        iou_threshold=settings.model.iou_threshold,
        imgsz=settings.model.imgsz,
        project_root=settings.project_root,
    )

    image = None
    if args.image:
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = (settings.project_root / image_path).resolve()
        image = _load_image(image_path)

    return _print_report(detector, image, warmup=not args.no_warmup)


if __name__ == "__main__":
    sys.exit(main())
