"""Synthetic capture emits snapshots and stop() is repeatable."""

import time

import numpy as np

from vision.capture import VideoCaptureThread, render_synthetic_frame


def _wait(qapp, predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_render_synthetic_frame_shape_changes_with_tick():
    first = render_synthetic_frame(80, 60, 0)
    later = render_synthetic_frame(80, 60, 5)
    assert first.shape == (60, 80, 3)
    assert first.dtype == np.uint8
    assert not np.array_equal(first, later)


def test_synthetic_thread_emits_and_stop_is_repeatable(qapp):
    thread = VideoCaptureThread(source="synthetic", width=320, height=180)
    frames: list[tuple[np.ndarray, float]] = []
    errors: list[str] = []
    stopped: list[bool] = []
    thread.frame_ready.connect(lambda frame, stamp: frames.append((frame, stamp)))
    thread.error.connect(errors.append)
    thread.stopped.connect(lambda: stopped.append(True))
    thread.start()
    try:
        assert _wait(qapp, lambda: len(frames) >= 1)
        frame, stamp = frames[0]
        assert frame.shape == (180, 320, 3)
        assert isinstance(stamp, float)
        assert thread.mode == "synthetic"
        assert errors == []
    finally:
        thread.stop()
        thread.stop()
    qapp.processEvents()
    assert not thread.isRunning()
    assert stopped


def test_missing_camera_falls_back_to_synthetic(qapp):
    thread = VideoCaptureThread(source=31, width=64, height=48)
    frames: list[np.ndarray] = []
    errors: list[str] = []
    thread.frame_ready.connect(lambda frame, _stamp: frames.append(frame))
    thread.error.connect(errors.append)
    thread.start()
    try:
        assert _wait(qapp, lambda: len(frames) >= 1)
        assert frames[0].shape == (48, 64, 3)
        assert thread.mode == "synthetic"
        assert errors and "合成帧" in errors[0]
        assert int(frames[0].max()) != int(frames[0].min())
    finally:
        thread.stop(wait_ms=2000)


def test_set_source_before_start_skips_missing_camera(qapp):
    thread = VideoCaptureThread(source=31, width=64, height=48)
    frames: list[np.ndarray] = []
    errors: list[str] = []
    thread.frame_ready.connect(lambda frame, _stamp: frames.append(frame))
    thread.error.connect(errors.append)
    thread.set_source("synthetic")
    thread.start()
    try:
        assert _wait(qapp, lambda: len(frames) >= 1)
        assert thread.source == "synthetic"
        assert thread.mode == "synthetic"
        assert errors == []
    finally:
        thread.stop(wait_ms=2000)


def test_stop_before_start_then_start_still_runs(qapp):
    thread = VideoCaptureThread(source="synthetic", width=32, height=32)
    thread.stop()
    thread.stop()
    frames: list[np.ndarray] = []
    thread.frame_ready.connect(lambda frame, _stamp: frames.append(frame))
    thread.start()
    try:
        assert _wait(qapp, lambda: len(frames) >= 1)
    finally:
        thread.stop()
