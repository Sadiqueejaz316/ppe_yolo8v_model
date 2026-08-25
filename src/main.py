"""Live PPE pipeline entry point.

Examples:
    python -m src.inference.test_model --image test_images/test.jpg
    python -m src.main --mode image --source test_images/test.jpg
    python -m src.main --mode video --source test_images/test.mp4
    python -m src.main --mode rtsp --camera CAM-001
    python -m src.main --camera CAM-001
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

from src.config.settings import AppConfig, CameraConfig, load_settings
from src.exceptions import CameraConnectionError, InvalidSourceError, ModelNotFoundError, PPEError
from src.inference.detector import describe_device, resolve_device
from src.logging_setup import setup_logging
from src.metrics.collector import PipelineMetrics
from src.pipeline import PPEPipeline
from src.video.camera import RTSPCamera
from src.video.file_source import ImageSource, VideoFileSource
from src.video.frame_processor import InferenceGate, LatestFrameBuffer
from src.video.source import VideoSource

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Industrial PPE detection pipeline")
    parser.add_argument("--mode", choices=["image", "video", "rtsp"], default=None)
    parser.add_argument("--source", type=str, default=None, help="Image or video path")
    parser.add_argument("--camera", type=str, default=None, help="Camera id from config/camera.yaml")
    parser.add_argument("--config-root", type=str, default=None)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--save-output", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=0, help="Stop after N seconds (useful for RTSP tests)")
    parser.add_argument("--log-level", type=str, default=None)
    return parser


def _infer_mode(args: argparse.Namespace) -> str:
    if args.mode:
        return args.mode
    if args.camera:
        return "rtsp"
    if args.source:
        suffix = Path(args.source).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return "image"
        return "video"
    return "rtsp"


def _can_display() -> bool:
    if os.environ.get("PPE_NO_DISPLAY"):
        return False
    try:
        import cv2

        cv2.namedWindow("__ppe_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__ppe_probe__")
        return True
    except Exception:
        return False


def _print_startup(config: AppConfig, camera: CameraConfig) -> None:
    device = resolve_device(config.model.device)
    label, cuda_status = describe_device(device)
    print(f"Device: {label}")
    print(f"CUDA: {cuda_status}")
    print(f"Model: {Path(config.model.path).name}")
    print(f"Camera: {camera.id} ({camera.name})")
    print()


def _open_writer(path: str, frame_shape, fps: float):
    import cv2

    height, width = int(frame_shape[0]), int(frame_shape[1])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, max(fps, 5.0), (width, height))
    if not writer.isOpened():
        logger.error("OUTPUT_WRITER_FAILED path=%s", path)
        return None
    return writer


def _build_source(mode: str, args: argparse.Namespace, config: AppConfig, camera: CameraConfig) -> VideoSource:
    if mode == "image":
        if not args.source:
            raise InvalidSourceError("--source is required for image mode")
        path = Path(args.source)
        if not path.is_absolute():
            path = config.project_root / path
        return ImageSource(path, camera_id=camera.id, camera_name=camera.name)
    if mode == "video":
        if not args.source:
            raise InvalidSourceError("--source is required for video mode")
        path = Path(args.source)
        if not path.is_absolute():
            path = config.project_root / path
        return VideoFileSource(path, camera_id=camera.id, camera_name=camera.name)
    return RTSPCamera(camera)


def _print_metrics_console(snapshot, processed) -> None:
    print(
        f"Camera: {snapshot.camera_id} | "
        f"Camera FPS: {snapshot.camera_fps:.1f} | "
        f"Inference FPS: {snapshot.inference_fps:.1f} | "
        f"Inference latency: {snapshot.inference_latency_ms:.0f} ms | "
        f"Persons: {snapshot.persons} | "
        f"Violations: {snapshot.violations} | "
        f"Dropped frames: {snapshot.dropped_frames}",
        flush=True,
    )
    for det in processed.detections:
        print(f"    {det.label} {det.confidence:.2f}")


def run_pipeline(args: argparse.Namespace) -> int:
    root = Path(args.config_root).resolve() if args.config_root else None
    config = load_settings(project_root=root)
    setup_logging(args.log_level or config.logging.level)

    mode = _infer_mode(args)
    if mode in {"image", "video"} and not args.source:
        raise InvalidSourceError("--source is required for image/video mode")
    camera_id = args.camera or (config.cameras[0].id if config.cameras else "CAM-001")
    if config.cameras:
        camera = config.camera_by_id(camera_id)
    else:
        camera = CameraConfig(id=camera_id, name=camera_id, rtsp_url=os.environ.get("RTSP_URL", ""))

    logger.info("CONFIG_LOADED cameras=%s zone=%s model=%s", len(config.cameras), camera.zone, config.model.path)
    _print_startup(config, camera)
    pipeline = PPEPipeline.build(config, camera, enable_tracking=(mode != "image"))
    if mode == "image":
        from src.tracking.tracker import NoOpTracker

        pipeline.tracker = NoOpTracker()

    source = _build_source(mode, args, config, camera)
    try:
        source.connect()
    except CameraConnectionError:
        if mode != "rtsp":
            raise
        logger.warning("CAMERA_DISCONNECTED camera=%s reason=initial_open_failed; will retry", camera.id)

    display = (not args.no_display) and _can_display()
    writer = None
    processed_count = 0
    gate_fps = camera.target_fps if camera.target_fps else config.inference.target_inference_fps
    gate = InferenceGate(gate_fps)
    runtime_metrics = pipeline.metrics

    try:
        if mode == "rtsp":
            processed_count = _run_live(
                source=source,
                pipeline=pipeline,
                runtime_metrics=runtime_metrics,
                gate=gate,
                display=display,
                save_output=args.save_output,
                max_frames=args.max_frames,
                max_seconds=args.max_seconds,
            )
        else:
            started_at = time.monotonic()
            for frame in source.frames():
                infer = True if mode == "image" else gate.allow()
                processed = pipeline.process(frame, infer=infer)
                snapshot = runtime_metrics.snapshot(source)
                processed.metrics = snapshot
                _print_metrics_console(snapshot, processed)
                if args.save_output:
                    if mode == "image":
                        import cv2

                        cv2.imwrite(args.save_output, processed.annotated)
                    else:
                        if writer is None:
                            writer = _open_writer(
                                args.save_output,
                                processed.annotated.shape,
                                config.inference.target_inference_fps,
                            )
                        if writer is not None:
                            writer.write(processed.annotated)
                if display:
                    import cv2

                    cv2.imshow(f"PPE {camera.id}", processed.annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                processed_count += 1
                if args.max_frames and processed_count >= args.max_frames:
                    break
                if args.max_seconds and (time.monotonic() - started_at) >= args.max_seconds:
                    break
    except KeyboardInterrupt:
        logger.info("SHUTDOWN_REQUESTED camera=%s", camera.id)
    finally:
        source.stop()
        if writer is not None:
            writer.release()
        if display:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
        try:
            pipeline.evidence.prune()
        except Exception:
            logger.debug("EVIDENCE_PRUNE_SKIPPED", exc_info=True)

    logger.info("PIPELINE_STOPPED camera=%s frames=%s", camera.id, processed_count)
    return 0


def _run_live(
    source: VideoSource,
    pipeline: PPEPipeline,
    runtime_metrics: PipelineMetrics,
    gate: InferenceGate,
    display: bool,
    save_output: str | None,
    max_frames: int,
    max_seconds: float = 0,
) -> int:
    buffer = LatestFrameBuffer()
    stop = threading.Event()
    writer = None
    processed_count = 0

    def _ingest() -> None:
        try:
            for frame in source.frames():
                if stop.is_set():
                    break
                buffer.put(frame)
        except Exception:
            logger.exception("INGEST_THREAD_FAILED camera=%s", source.camera_id)
            stop.set()

    thread = threading.Thread(target=_ingest, name="rtsp-ingest", daemon=True)
    thread.start()
    last_log = 0.0
    started_at = time.monotonic()
    try:
        while not stop.is_set():
            if max_seconds and (time.monotonic() - started_at) >= max_seconds:
                break
            frame = buffer.take()
            if frame is None:
                time.sleep(0.005)
                if not thread.is_alive():
                    frame = buffer.take()
                    if frame is None:
                        break
                else:
                    continue
            infer = gate.allow()
            processed = pipeline.process(frame, infer=infer)
            snapshot = runtime_metrics.snapshot(source, dropped_extra=buffer.dropped)
            processed.metrics = snapshot
            now = time.monotonic()
            if now - last_log >= 1.0:
                _print_metrics_console(snapshot, processed)
                last_log = now
            if save_output:
                if writer is None:
                    writer = _open_writer(save_output, processed.annotated.shape, max(snapshot.camera_fps, 5.0))
                if writer is not None:
                    writer.write(processed.annotated)
            if display:
                import cv2

                cv2.imshow(f"PPE {source.camera_id}", processed.annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            processed_count += 1
            if max_frames and processed_count >= max_frames:
                break
    finally:
        stop.set()
        source.stop()
        thread.join(timeout=2.0)
        if writer is not None:
            writer.release()
    return processed_count


def main(argv: list[str] | None = None) -> int:
    def _handle_stop(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_stop)
    try:
        signal.signal(signal.SIGTERM, _handle_stop)
    except (ValueError, OSError):
        pass

    args = build_parser().parse_args(argv)
    try:
        return run_pipeline(args)
    except (ModelNotFoundError, InvalidSourceError, CameraConnectionError, PPEError, KeyError) as exc:
        logger.error("PIPELINE_FAILED error=%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
