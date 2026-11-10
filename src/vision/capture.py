"""Camera grab thread.

``VideoCaptureThread`` is the only place that calls ``cv2.VideoCapture.read``.
The UI thread receives a copied BGR snapshot plus a wall-clock timestamp.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

_SYNTHETIC = "synthetic"
_DEFAULT_SIZE = (960, 540)
_SYNTHETIC_FPS = 15.0


def render_synthetic_frame(width: int, height: int, tick: int) -> np.ndarray:
    """Draw a PCB-like BGR frame so the shell runs with no camera.

    The moving rectangle is only a visual heartbeat. It is not a tracked pose.
    """
    width = max(16, int(width))
    height = max(16, int(height))
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = (48, 108, 62)
    step_x = max(width // 8, 24)
    step_y = max(height // 6, 24)
    for row, y in enumerate(range(step_y // 2, height, step_y)):
        y = min(y, height - 1)
        cv2.line(frame, (0, y), (width - 1, y), (40, 132, 176), 1)
        for col, x in enumerate(range(step_x // 2, width, step_x)):
            x = min(x, width - 1)
            color = (30, 90, 200) if (row + col) % 2 == 0 else (20, 70, 160)
            x0 = max(0, x - 8)
            y0 = max(0, y - 5)
            x1 = min(width - 1, x + 8)
            y1 = min(height - 1, y + 5)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1)
    # Moving reticle so the fake stream is obviously alive. It is pixels in the
    # frame, not a QGraphics annotation.
    cx = 10 + int((tick * 7) % max(width - 20, 1))
    cy = 10 + int((tick * 5) % max(height - 20, 1))
    arm = max(6, min(width, height) // 14)
    ink = (245, 245, 245)
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), ink, 1)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), ink, 1)
    cv2.circle(frame, (cx, cy), max(3, arm // 2), ink, 1)
    if width >= 120 and height >= 36:
        cv2.putText(
            frame,
            "SYNTHETIC",
            (8, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (235, 245, 235),
            1,
            cv2.LINE_AA,
        )
    return frame


def camera_device_present(index: int) -> bool:
    """Best-effort check so a missing Linux node does not block inside OpenCV."""
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        return False
    if sys.platform.startswith("linux"):
        return os.path.exists(f"/dev/video{index}")
    return True


class VideoCaptureThread(QThread):
    """OpenCV capture on a ``QThread``. Do not call ``read()`` on the UI thread.

    Signals
    -------
    frame_ready(object, float)
        BGR ``ndarray`` snapshot and ``time.time()`` seconds. Treat the array
        as read-only; mutate a copy.
    error(str)
        Open failure or unexpected loop error. A missing camera falls back to
        synthetic frames after one ``error``.
    stopped()
        Emitted after ``run`` releases the capture, including after ``stop``.

    ``source`` is a camera index, a video path, or the exact string
    ``"synthetic"``. ``set_source`` applies on the next loop.
    """

    frame_ready = pyqtSignal(object, float)
    error = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(
        self,
        source: int | str = 0,
        width: int | None = None,
        height: int | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._source: int | str = source
        self._pending_source: int | str | None = None
        self._width = width
        self._height = height
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._capture: cv2.VideoCapture | None = None
        self._mode = "pending"
        self._tick = 0
        self._open_key: tuple | None = None

    @property
    def mode(self) -> str:
        """``pending``, ``camera``, ``file``, or ``synthetic``."""
        return self._mode

    @property
    def source(self) -> int | str:
        with self._lock:
            return self._source

    def set_source(self, source: int | str) -> None:
        """Queue a source change. The next loop opens it.

        Whether signals should pause across the switch is still an open
        product question; this slice keeps emitting frames.
        """
        with self._lock:
            self._pending_source = source

    def start(self, priority: QThread.Priority = QThread.Priority.InheritPriority) -> None:
        if self.isRunning():
            return
        self._stop_event.clear()
        super().start(priority)

    def stop(self, wait_ms: int = 2000) -> None:
        """Set the exit flag, wait, and release the capture. Safe to repeat."""
        self._stop_event.set()
        if self.isRunning():
            finished = self.wait(max(0, int(wait_ms)))
            if not finished:
                self._release_capture()
                self.wait(500)
        else:
            self._release_capture()

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._ensure_source(self._take_source())
                if self._stop_event.is_set():
                    break
                frame = self._read_frame()
                if frame is None:
                    self._sleep_interruptible(0.05)
                    continue
                self.frame_ready.emit(frame, time.time())
                self._pace()
        except Exception as exc:  # noqa: BLE001 — keep the thread from dying silently
            self.error.emit(f"采集线程异常：{exc}")
        finally:
            self._release_capture()
            self.stopped.emit()

    def _take_source(self) -> int | str:
        with self._lock:
            if self._pending_source is not None:
                self._source = self._pending_source
                self._pending_source = None
                self._open_key = None
            return self._source

    def _frame_size(self) -> tuple[int, int]:
        width = self._width if self._width and self._width > 0 else _DEFAULT_SIZE[0]
        height = self._height if self._height and self._height > 0 else _DEFAULT_SIZE[1]
        return int(width), int(height)

    def _ensure_source(self, source: int | str) -> None:
        key = (source, self._width, self._height)
        if self._open_key == key:
            return
        self._release_capture()
        self._open_key = key
        self._tick = 0
        if source == _SYNTHETIC:
            self._mode = "synthetic"
            return
        if isinstance(source, int):
            self._open_camera(source)
            return
        self._open_file(str(source))

    def _open_camera(self, index: int) -> None:
        if not camera_device_present(index):
            self._mode = "synthetic"
            self.error.emit(f"摄像头 {index} 不可用，已改用合成帧（source='synthetic'）。")
            return
        capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            capture.release()
            self._mode = "synthetic"
            self.error.emit(f"无法打开摄像头 {index}，已改用合成帧。")
            return
        self._apply_size(capture)
        self._capture = capture
        self._mode = "camera"

    def _open_file(self, path: str) -> None:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            self._mode = "synthetic"
            self.error.emit(f"无法打开视频 {path}，已改用合成帧。")
            return
        self._capture = capture
        self._mode = "file"

    def _apply_size(self, capture: cv2.VideoCapture) -> None:
        if self._width and self._width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        if self._height and self._height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))

    def _read_frame(self) -> np.ndarray | None:
        capture = self._capture
        if capture is None:
            self._tick += 1
            width, height = self._frame_size()
            return render_synthetic_frame(width, height, self._tick)
        ok, frame = capture.read()
        if not ok or frame is None:
            if self._mode == "file":
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
            if not ok or frame is None:
                self.error.emit("读帧失败，已改用合成帧。")
                self._release_capture()
                self._mode = "synthetic"
                self._tick += 1
                width, height = self._frame_size()
                return render_synthetic_frame(width, height, self._tick)
        return np.ascontiguousarray(frame).copy()

    def _pace(self) -> None:
        if self._mode == "synthetic":
            self._sleep_interruptible(1.0 / _SYNTHETIC_FPS)
        else:
            self._sleep_interruptible(0.001)

    def _sleep_interruptible(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, seconds)
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self.msleep(max(1, int(min(remaining, 0.05) * 1000)))

    def _release_capture(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()
