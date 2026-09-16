"""Comprehensive benchmark script for video pipeline latency audit."""

import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.compliance.association import associate_ppe
from src.compliance.summary import build_scene_summary
from src.config.settings import load_settings
from src.ops.sink import _encode_jpeg
from src.pipeline import PPEPipeline
from src.taxonomy import resolve_taxonomy
from src.video.file_source import VideoFileSource
from src.video.source import VideoFrame
from src.viz import annotate


def measure_video_properties():
    print("============================================================")
    print("1. SOURCE VIDEO PROPERTIES")
    print("============================================================")
    videos = glob.glob("test_images/*.mp4*") + glob.glob("test_images/*/*.mp4*")
    results = {}
    for v in sorted(videos):
        cap = cv2.VideoCapture(v)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = count / fps if fps > 0 else 0
        results[v] = {"fps": fps, "width": w, "height": h, "frames": count, "duration": dur}
        print(f"File: {v}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Resolution: {w}x{h}")
        print(f"  Total frames: {count}")
        print(f"  Duration: {dur:.2f}s")
        cap.release()
    print()
    return results


def run_granular_benchmark(video_path: str, num_frames: int = 100):
    print("============================================================")
    print(f"2. PPE PIPELINE TIMING BREAKDOWN (Source: {video_path})")
    print("============================================================")
    config = load_settings()
    camera = config.cameras[0]
    pipeline = PPEPipeline.build(config, camera)
    source = VideoFileSource(video_path, camera_id=camera.id)
    source.connect()

    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} test frames from {video_path}.")

    # Warmup
    for i in range(5):
        pipeline.process(
            VideoFrame(
                image=frames[i % len(frames)],
                timestamp=datetime.now(timezone.utc),
                camera_id=camera.id,
                frame_index=i + 1,
                source_fps=25.0,
                camera_name="Test",
            )
        )

    t_preprocess = []
    t_yolo_infer = []
    t_postprocess = []
    t_tracking = []
    t_association = []
    t_stabilizer = []
    t_compliance = []
    t_summary = []
    t_viz_preview = []
    t_viz_hud = []
    t_jpeg_encode = []
    t_disk_write = []
    t_total_frame = []
    jpeg_sizes = []

    detector = pipeline.detector

    for idx, img in enumerate(frames):
        t0 = time.perf_counter()

        # 1. Detection
        det_result = detector.predict(img)
        dets = det_result.detections
        t_preprocess.append(det_result.latency.preprocess_ms)
        t_yolo_infer.append(det_result.latency.inference_ms)
        t_postprocess.append(det_result.latency.postprocess_ms)

        # 2. Tracking
        t_tr_start = time.perf_counter()
        person_dets = [d for d in dets if pipeline.taxonomy.is_person(d.class_id)]
        tracked = pipeline.tracker.update(person_dets, img)
        t_tr_end = time.perf_counter()
        t_tracking.append((t_tr_end - t_tr_start) * 1000.0)

        # 3. Association
        t_as_start = time.perf_counter()
        persons = associate_ppe(tracked, dets, pipeline.taxonomy, pipeline.config.association)
        t_as_end = time.perf_counter()
        t_association.append((t_as_end - t_as_start) * 1000.0)

        # 4. Stabilizer
        t_st_start = time.perf_counter()
        persons = pipeline.stabilizer.update(persons)
        t_st_end = time.perf_counter()
        t_stabilizer.append((t_st_end - t_st_start) * 1000.0)

        # 5. Compliance
        t_co_start = time.perf_counter()
        compliance = [pipeline.compliance.evaluate(p) for p in persons]
        t_co_end = time.perf_counter()
        t_compliance.append((t_co_end - t_co_start) * 1000.0)

        # 6. Summary
        t_su_start = time.perf_counter()
        summary = build_scene_summary(persons, compliance, pipeline.compliance.required_ppe)
        t_su_end = time.perf_counter()
        t_summary.append((t_su_end - t_su_start) * 1000.0)

        # 7. Visualization preview
        t_v1_start = time.perf_counter()
        annotated_preview = annotate(
            img,
            dets,
            persons,
            compliance,
            None,
            pipeline.taxonomy,
            pipeline.compliance.required_ppe,
            visualization=pipeline.config.visualization,
        )
        t_v1_end = time.perf_counter()
        t_viz_preview.append((t_v1_end - t_v1_start) * 1000.0)

        # 8. Visualization final (with HUD)
        t_v2_start = time.perf_counter()
        snapshot = pipeline.metrics.snapshot(source)
        annotated_final = annotate(
            img,
            dets,
            persons,
            compliance,
            snapshot,
            pipeline.taxonomy,
            pipeline.compliance.required_ppe,
            visualization=pipeline.config.visualization,
        )
        t_v2_end = time.perf_counter()
        t_viz_hud.append((t_v2_end - t_v2_start) * 1000.0)

        # 9. JPEG Encoding
        t_jp_start = time.perf_counter()
        live_quality = max(40, min(int(pipeline.config.evidence.jpeg_quality), 75))
        jpeg = _encode_jpeg(annotated_final, live_quality)
        t_jp_end = time.perf_counter()
        t_jpeg_encode.append((t_jp_end - t_jp_start) * 1000.0)
        if jpeg is not None:
            jpeg_sizes.append(len(jpeg))

        # 10. Disk write
        t_dw_start = time.perf_counter()
        if pipeline.dashboard_sink:
            pipeline.dashboard_sink.live.write(camera.id, {"test": True}, jpeg)
        t_dw_end = time.perf_counter()
        t_disk_write.append((t_dw_end - t_dw_start) * 1000.0)

        t_end = time.perf_counter()
        t_total_frame.append((t_end - t0) * 1000.0)

    print(f"Timing Breakdown ({len(frames)} frames):")
    print(f"  1. Preprocess:       {np.mean(t_preprocess):.2f} ms (min {np.min(t_preprocess):.2f}, max {np.max(t_preprocess):.2f})")
    print(f"  2. YOLO Inference:   {np.mean(t_yolo_infer):.2f} ms (min {np.min(t_yolo_infer):.2f}, max {np.max(t_yolo_infer):.2f})")
    print(f"  3. Postprocess:      {np.mean(t_postprocess):.2f} ms (min {np.min(t_postprocess):.2f}, max {np.max(t_postprocess):.2f})")
    print(f"  4. Tracking:         {np.mean(t_tracking):.2f} ms (min {np.min(t_tracking):.2f}, max {np.max(t_tracking):.2f})")
    print(f"  5. Association:      {np.mean(t_association):.2f} ms (min {np.min(t_association):.2f}, max {np.max(t_association):.2f})")
    print(f"  6. Stabilizer:       {np.mean(t_stabilizer):.2f} ms (min {np.min(t_stabilizer):.2f}, max {np.max(t_stabilizer):.2f})")
    print(f"  7. Compliance:       {np.mean(t_compliance):.2f} ms (min {np.min(t_compliance):.2f}, max {np.max(t_compliance):.2f})")
    print(f"  8. Scene Summary:    {np.mean(t_summary):.2f} ms (min {np.min(t_summary):.2f}, max {np.max(t_summary):.2f})")
    print(f"  9. Viz (Preview):    {np.mean(t_viz_preview):.2f} ms (min {np.min(t_viz_preview):.2f}, max {np.max(t_viz_preview):.2f})")
    print(f" 10. Viz (HUD):        {np.mean(t_viz_hud):.2f} ms (min {np.min(t_viz_hud):.2f}, max {np.max(t_viz_hud):.2f})")
    print(f"     Viz Total:        {np.mean(t_viz_preview) + np.mean(t_viz_hud):.2f} ms")
    print(f" 11. JPEG Encoding:    {np.mean(t_jpeg_encode):.2f} ms (min {np.min(t_jpeg_encode):.2f}, max {np.max(t_jpeg_encode):.2f})")
    print(f" 12. Disk Write:       {np.mean(t_disk_write):.2f} ms (min {np.min(t_disk_write):.2f}, max {np.max(t_disk_write):.2f})")
    print(f" ------------------------------------------------------------")
    print(f"  Total Frame Time:    {np.mean(t_total_frame):.2f} ms (min {np.min(t_total_frame):.2f}, max {np.max(t_total_frame):.2f})")
    print(f"  Processed FPS:       {1000.0 / np.mean(t_total_frame):.2f} FPS")
    print()

    if jpeg_sizes:
        print("=== JPEG Snapshot Info ===")
        print(f"  JPEG Quality: {live_quality}")
        print(f"  Average File Size: {np.mean(jpeg_sizes) / 1024.0:.1f} KB")
        print(f"  Resolution: {annotated_final.shape[1]}x{annotated_final.shape[0]}")
    print()


def measure_fastapi_latency():
    print("============================================================")
    print("3. FASTAPI SNAPSHOT LATENCY")
    print("============================================================")
    app = create_app()
    client = TestClient(app)

    # Test summary endpoint
    t_sum = []
    for _ in range(50):
        t0 = time.perf_counter()
        resp = client.get("/api/dashboard/summary")
        t_sum.append((time.perf_counter() - t0) * 1000.0)
    print(f"  GET /api/dashboard/summary: avg {np.mean(t_sum):.2f} ms (min {np.min(t_sum):.2f}, max {np.max(t_sum):.2f}), status={resp.status_code}")

    # Test snapshot endpoint
    t_snap = []
    for _ in range(50):
        t0 = time.perf_counter()
        resp = client.get("/api/cameras/CAM-001/snapshot")
        t_snap.append((time.perf_counter() - t0) * 1000.0)
    print(f"  GET /api/cameras/CAM-001/snapshot: avg {np.mean(t_snap):.2f} ms (min {np.min(t_snap):.2f}, max {np.max(t_snap):.2f}), status={resp.status_code}")
    print()


def measure_phase1_live_buffer_improvements():
    print("============================================================")
    print("4. PHASE 1 IN-MEMORY BUFFER & DISK I/O LATENCY BENCHMARK")
    print("============================================================")
    from src.ops.live import LiveFrameBuffer, LiveStateStore

    buf = LiveFrameBuffer()
    test_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 45000  # ~45 KB simulated frame
    meta = {"camera_id": "CAM-001", "timestamp": "2026-09-08T12:00:00Z"}

    # 1. In-memory buffer write latency
    t_mem_write = []
    for _ in range(1000):
        t0 = time.perf_counter()
        buf.update("CAM-001", test_jpeg, meta)
        t_mem_write.append((time.perf_counter() - t0) * 1000.0)
    print(f"  LiveFrameBuffer.update (RAM write):    avg {np.mean(t_mem_write):.4f} ms (p95 {np.percentile(t_mem_write, 95):.4f} ms)")

    # 2. In-memory buffer read latency
    t_mem_read = []
    for _ in range(1000):
        t0 = time.perf_counter()
        _ = buf.get_jpeg("CAM-001")
        t_mem_read.append((time.perf_counter() - t0) * 1000.0)
    print(f"  LiveFrameBuffer.get_jpeg (RAM read):   avg {np.mean(t_mem_read):.4f} ms (p95 {np.percentile(t_mem_read, 95):.4f} ms)")

    # 3. Synchronous disk write latency
    tmp_store = LiveStateStore(Path("data/live_benchmark_tmp"))
    t_disk_write = []
    for _ in range(50):
        t0 = time.perf_counter()
        tmp_store.write("CAM-001", meta, test_jpeg)
        t_disk_write.append((time.perf_counter() - t0) * 1000.0)
    print(f"  Synchronous Disk write (.tmp + replace): avg {np.mean(t_disk_write):.2f} ms (min {np.min(t_disk_write):.2f}, max {np.max(t_disk_write):.2f})")

    # 4. Synchronous disk read latency
    t_disk_read = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = tmp_store.jpeg_bytes("CAM-001")
        t_disk_read.append((time.perf_counter() - t0) * 1000.0)
    print(f"  Synchronous Disk read (file read):      avg {np.mean(t_disk_read):.2f} ms (min {np.min(t_disk_read):.2f}, max {np.max(t_disk_read):.2f})")

    # Clean up benchmark tmp files
    for p in Path("data/live_benchmark_tmp").glob("*"):
        try:
            p.unlink()
        except OSError:
            pass
    try:
        Path("data/live_benchmark_tmp").rmdir()
    except OSError:
        pass

    # 5. FastAPI latency with in-memory buffer
    app_mem = create_app(frame_buffer=buf)
    client_mem = TestClient(app_mem)
    t_api_mem = []
    for _ in range(50):
        t0 = time.perf_counter()
        resp = client_mem.get("/api/cameras/CAM-001/snapshot")
        t_api_mem.append((time.perf_counter() - t0) * 1000.0)
    print(f"  FastAPI snapshot via LiveFrameBuffer:   avg {np.mean(t_api_mem):.2f} ms (min {np.min(t_api_mem):.2f}, max {np.max(t_api_mem):.2f})")

    speedup = np.mean(t_disk_write) / max(np.mean(t_mem_write), 0.0001)
    print(f"\n  Pipeline thread frame delivery speedup: ~{speedup:.0f}x faster (zero synchronous disk wait)")
    print("============================================================\n")


if __name__ == "__main__":
    measure_video_properties()
    # Test on test.mp4 and tata-demo.mp4
    if os.path.exists("test_images/test.mp4"):
        run_granular_benchmark("test_images/test.mp4", num_frames=40)
    if os.path.exists("test_images/tata-demo.mp4"):
        run_granular_benchmark("test_images/tata-demo.mp4", num_frames=60)
    measure_fastapi_latency()
    measure_phase1_live_buffer_improvements()
