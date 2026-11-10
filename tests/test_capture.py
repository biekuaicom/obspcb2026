"""Synthetic capture emits snapshots and stop() is repeatable."""

import threading
import time

import cv2
import numpy as np

from vision import capture as capture_module
from vision.capture import (
    VideoCaptureThread,
    camera_backends,
    open_failure_message,
    render_synthetic_frame,
)


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


def test_camera_backends_follow_platform(monkeypatch):
    monkeypatch.setattr(capture_module.sys, "platform", "win32")
    windows = camera_backends()
    assert [name for name, _api in windows] == ["DirectShow", "Media Foundation"]
    assert windows[0][1] == cv2.CAP_DSHOW
    assert windows[1][1] == cv2.CAP_MSMF
    monkeypatch.setattr(capture_module.sys, "platform", "linux")
    assert camera_backends() == [("默认", None)]


def test_open_failure_message_names_timeouts():
    text = open_failure_message(0, ["DirectShow 打开超时", "Media Foundation 失败"])
    assert "摄像头 0" in text
    assert "合成帧" in text
    assert "DirectShow" in text
    assert "Media Foundation" in text
    assert "超时" in text
    assert open_failure_message(1, ["默认 失败"]) == "无法打开摄像头 1，已改用合成帧。"
    assert "打开超时" in open_failure_message(2, ["默认 打开超时"])
    assert "读帧超时" in open_failure_message(3, ["默认 读帧超时"])


def _patch_camera(monkeypatch, device):
    monkeypatch.setattr(capture_module, "CAMERA_OPEN_TIMEOUT_S", 0.3)
    monkeypatch.setattr(capture_module, "CAMERA_READ_TIMEOUT_S", 0.3)
    monkeypatch.setattr(capture_module, "CAMERA_PROBE_WINDOW_S", 0.05)
    monkeypatch.setattr(capture_module, "camera_device_present", lambda _index: True)
    monkeypatch.setattr(cv2, "VideoCapture", device)


def test_failed_open_uses_default_backend_then_synthetic(qapp, monkeypatch):
    calls: list[tuple] = []

    class Closed:
        def __init__(self, index, *args):
            calls.append((index, args))

        def isOpened(self):
            return False

        def release(self):
            pass

        def read(self):
            return False, None

        def set(self, *_args, **_kwargs):
            return False

    monkeypatch.setattr(capture_module.sys, "platform", "linux")
    _patch_camera(monkeypatch, Closed)
    thread = VideoCaptureThread(source=2, width=32, height=24)
    frames: list[np.ndarray] = []
    errors: list[str] = []
    thread.frame_ready.connect(lambda frame, _stamp: frames.append(frame))
    thread.error.connect(errors.append)
    thread.start()
    try:
        assert _wait(qapp, lambda: len(frames) >= 1 and bool(errors))
        assert calls == [(2, ())]
        assert thread.mode == "synthetic"
        assert errors[0] == "无法打开摄像头 2，已改用合成帧。"
    finally:
        thread.stop()


def test_open_and_read_frames_without_a_real_camera(qapp, monkeypatch):
    class Live:
        def __init__(self, index, *args):
            del index, args

        def isOpened(self):
            return True

        def read(self):
            return True, np.full((24, 32, 3), 9, np.uint8)

        def release(self):
            pass

        def set(self, *_args, **_kwargs):
            return True

    monkeypatch.setattr(capture_module.sys, "platform", "linux")
    _patch_camera(monkeypatch, Live)
    thread = VideoCaptureThread(source=0, width=32, height=24)
    frames: list[np.ndarray] = []
    errors: list[str] = []
    thread.frame_ready.connect(lambda frame, _stamp: frames.append(frame))
    thread.error.connect(errors.append)
    thread.start()
    try:
        assert _wait(qapp, lambda: len(frames) >= 2)
        assert thread.mode == "camera"
        assert errors == []
        assert int(frames[0][0, 0, 0]) == 9
        assert int(frames[1][0, 0, 0]) == 9
    finally:
        thread.stop()
    assert not thread.isRunning()


def test_hung_open_falls_back_and_stop_returns(qapp, monkeypatch):
    hang = threading.Event()
    started = threading.Event()

    class Hung:
        def __init__(self, *_args, **_kwargs):
            started.set()
            hang.wait(timeout=5)

        def isOpened(self):
            return False

        def release(self):
            pass

        def read(self):
            return False, None

        def set(self, *_args, **_kwargs):
            return False

    monkeypatch.setattr(capture_module.sys, "platform", "linux")
    _patch_camera(monkeypatch, Hung)
    thread = VideoCaptureThread(source=0, width=32, height=24)
    frames: list[np.ndarray] = []
    errors: list[str] = []
    thread.frame_ready.connect(lambda frame, _stamp: frames.append(frame))
    thread.error.connect(errors.append)
    thread.start()
    try:
        assert _wait(qapp, lambda: started.is_set(), timeout=2)
        assert _wait(qapp, lambda: thread.mode == "synthetic" and bool(errors), timeout=2)
        assert any("超时" in message and "合成帧" in message for message in errors)
        assert frames == [] or frames[0].shape == (24, 32, 3)
        assert _wait(qapp, lambda: len(frames) >= 1, timeout=2)
        started_stop = time.monotonic()
        thread.stop(wait_ms=2000)
        assert time.monotonic() - started_stop < 1.5
        assert not thread.isRunning()
    finally:
        hang.set()
        thread.stop(wait_ms=2000)


def test_hung_read_falls_back_to_synthetic(qapp, monkeypatch):
    hang = threading.Event()

    class HungRead:
        def __init__(self, *_args, **_kwargs):
            pass

        def isOpened(self):
            return True

        def read(self):
            hang.wait(timeout=5)
            return False, None

        def release(self):
            pass

        def set(self, *_args, **_kwargs):
            return True

    monkeypatch.setattr(capture_module.sys, "platform", "linux")
    _patch_camera(monkeypatch, HungRead)
    thread = VideoCaptureThread(source=0, width=32, height=24)
    frames: list[np.ndarray] = []
    errors: list[str] = []
    thread.frame_ready.connect(lambda frame, _stamp: frames.append(frame))
    thread.error.connect(errors.append)
    thread.start()
    try:
        assert _wait(qapp, lambda: any("读帧超时" in message for message in errors), timeout=2)
        assert thread.mode == "synthetic"
        assert _wait(qapp, lambda: len(frames) >= 1, timeout=2)
        assert frames[0].shape == (24, 32, 3)
    finally:
        hang.set()
        thread.stop(wait_ms=2000)


def test_windows_skips_hung_directshow_and_uses_media_foundation(qapp, monkeypatch):
    hang = threading.Event()
    calls: list[int] = []

    class Selective:
        def __init__(self, index, backend=cv2.CAP_ANY, params=None):
            del index, params
            calls.append(int(backend))
            self.backend = int(backend)
            if self.backend == cv2.CAP_DSHOW:
                hang.wait(timeout=5)

        def isOpened(self):
            return self.backend == cv2.CAP_MSMF

        def read(self):
            return True, np.full((24, 32, 3), 17, np.uint8)

        def release(self):
            pass

        def set(self, *_args, **_kwargs):
            return True

    monkeypatch.setattr(capture_module.sys, "platform", "win32")
    _patch_camera(monkeypatch, Selective)
    thread = VideoCaptureThread(source=0, width=32, height=24)
    frames: list[np.ndarray] = []
    errors: list[str] = []
    thread.frame_ready.connect(lambda frame, _stamp: frames.append(frame))
    thread.error.connect(errors.append)
    opened = time.monotonic()
    thread.start()
    try:
        assert _wait(qapp, lambda: len(frames) >= 1 and thread.mode == "camera", timeout=2)
        assert time.monotonic() - opened < 1.5
        assert calls[0] == cv2.CAP_DSHOW
        assert cv2.CAP_MSMF in calls
        assert int(frames[0][0, 0, 0]) == 17
        assert errors == []
    finally:
        hang.set()
        thread.stop(wait_ms=2000)


def test_windows_both_backends_timeout_falls_back(qapp, monkeypatch):
    hang = threading.Event()

    class Hung:
        def __init__(self, index, backend=cv2.CAP_ANY, params=None):
            del index, backend, params
            hang.wait(timeout=5)

        def isOpened(self):
            return False

        def release(self):
            pass

        def read(self):
            return False, None

        def set(self, *_args, **_kwargs):
            return False

    monkeypatch.setattr(capture_module.sys, "platform", "win32")
    _patch_camera(monkeypatch, Hung)
    thread = VideoCaptureThread(source=0, width=16, height=16)
    errors: list[str] = []
    frames: list[np.ndarray] = []
    thread.error.connect(errors.append)
    thread.frame_ready.connect(lambda frame, _stamp: frames.append(frame))
    thread.start()
    try:
        assert _wait(qapp, lambda: bool(errors) and thread.mode == "synthetic", timeout=3)
        assert "DirectShow" in errors[0]
        assert "Media Foundation" in errors[0]
        assert "合成帧" in errors[0]
        assert _wait(qapp, lambda: len(frames) >= 1, timeout=2)
    finally:
        hang.set()
        thread.stop(wait_ms=2000)
