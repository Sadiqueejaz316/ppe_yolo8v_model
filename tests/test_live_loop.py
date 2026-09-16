import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

from src.exceptions import InferenceError
from src.main import _run_live
from src.metrics.collector import PipelineMetrics
from src.video.frame_processor import InferenceGate
from src.video.source import VideoFrame, VideoSource


def _frame(index: int, empty: bool = False) -> VideoFrame:
    image = np.zeros((0, 0, 3), dtype=np.uint8) if empty else np.zeros((8, 8, 3), dtype=np.uint8)
    return VideoFrame(
        image=image,
        timestamp=datetime.now(timezone.utc),
        camera_id="CAM-001",
        frame_index=index,
        source_fps=30.0,
        camera_name="test",
    )


class ScriptedSource(VideoSource):
    def __init__(self, frames: list[VideoFrame], pace_s: float = 0.02) -> None:
        self._frames = list(frames)
        self._pace_s = pace_s
        self._stop = threading.Event()
        self._connected = True

    @property
    def camera_id(self) -> str:
        return "CAM-001"

    @property
    def camera_name(self) -> str:
        return "test"

    @property
    def connected(self) -> bool:
        return self._connected and not self._stop.is_set()

    @property
    def measured_fps(self) -> float:
        return 30.0

    def connect(self) -> None:
        return None

    def frames(self):
        for frame in self._frames:
            if self._stop.is_set():
                return
            yield frame
            if self._pace_s:
                time.sleep(self._pace_s)
        while not self._stop.is_set():
            time.sleep(0.01)

    def stop(self) -> None:
        self._stop.set()
        self._connected = False


class RecordingPipeline:
    def __init__(self, fail_on: set[int] | None = None, delay_s: float = 0.0) -> None:
        self.fail_on = fail_on or set()
        self.delay_s = delay_s
        self.indices: list[int] = []
        self.calls = 0

    def process(self, frame: VideoFrame, infer: bool = True):
        self.calls += 1
        if frame.frame_index in self.fail_on:
            raise InferenceError("malformed detection result")
        if self.delay_s:
            time.sleep(self.delay_s)
        self.indices.append(frame.frame_index)
        return SimpleNamespace(
            detections=[],
            annotated=frame.image,
            metrics=None,
        )


def _run(source, pipeline, max_frames=0, max_seconds=1.0, target_fps=100.0) -> int:
    return _run_live(
        source=source,
        pipeline=pipeline,
        runtime_metrics=PipelineMetrics("CAM-001"),
        gate=InferenceGate(target_fps),
        display=False,
        save_output=None,
        max_frames=max_frames,
        max_seconds=max_seconds,
    )


def test_bad_frame_does_not_kill_live_loop():
    source = ScriptedSource([_frame(1), _frame(2), _frame(3), _frame(4)])
    pipeline = RecordingPipeline(fail_on={2})
    processed = _run(source, pipeline, max_frames=3, max_seconds=2.0)
    assert processed == 3
    assert pipeline.calls >= 4
    assert 2 not in pipeline.indices
    assert pipeline.indices == [1, 3, 4]


def test_unexpected_frame_exception_does_not_kill_live_loop():
    class UnexpectedFailurePipeline(RecordingPipeline):
        def process(self, frame: VideoFrame, infer: bool = True):
            if frame.frame_index == 2:
                raise AttributeError("malformed backend result")
            return super().process(frame, infer=infer)

    source = ScriptedSource([_frame(1), _frame(2), _frame(3)])
    pipeline = UnexpectedFailurePipeline()
    processed = _run(source, pipeline, max_frames=2, max_seconds=2.0)

    assert processed == 2
    assert pipeline.indices == [1, 3]


def test_empty_frame_is_skipped():
    source = ScriptedSource([_frame(1, empty=True), _frame(2)])
    pipeline = RecordingPipeline()
    processed = _run(source, pipeline, max_frames=1, max_seconds=2.0)
    assert processed == 1
    assert pipeline.indices == [2]


class ContinuousSource(ScriptedSource):
    def frames(self):
        index = 1
        while not self._stop.is_set():
            yield _frame(index)
            index += 1
            if self._pace_s:
                time.sleep(self._pace_s)


def test_slow_processing_drops_old_frames():
    source = ContinuousSource([], pace_s=0.001)
    pipeline = RecordingPipeline(delay_s=0.03)
    processed = _run(source, pipeline, max_frames=4, max_seconds=3.0)
    assert processed == 4
    assert pipeline.indices[0] >= 1
    # Intermediate camera frames must be skipped rather than queued.
    assert pipeline.indices[-1] > pipeline.indices[0] + 3


def test_live_loop_shutdown_stops_ingest():
    source = ScriptedSource([_frame(1)])
    pipeline = RecordingPipeline()
    processed = _run(source, pipeline, max_frames=1, max_seconds=2.0)
    assert processed == 1
    assert source.connected is False


def test_keyboard_interrupt_cleans_up():
    source = ScriptedSource([_frame(1), _frame(2), _frame(3)])

    class Boom(RecordingPipeline):
        def process(self, frame: VideoFrame, infer: bool = True):
            super().process(frame, infer=infer)
            raise KeyboardInterrupt

    pipeline = Boom()
    try:
        _run(source, pipeline, max_frames=5, max_seconds=2.0)
        raise AssertionError("expected KeyboardInterrupt")
    except KeyboardInterrupt:
        pass
    assert source.connected is False
