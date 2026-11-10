"""Camera grab thread.

``VideoCaptureThread`` is the only place that calls ``cv2.VideoCapture.read``.
The UI thread receives a copied BGR snapshot plus a wall-clock timestamp.

Opening and reading a camera can block inside OpenCV with no way to interrupt
the call. On Windows that shows up as ``VideoCapture(index)`` hanging in the
default backend before the synthetic fallback ever runs. Camera open/read
therefore runs on a private thread, bounded by a timeout. Windows tries
DirectShow and then Media Foundation. A failed, empty, or timed-out open falls
back to synthetic frames and emits one ``error``.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

_SYNTHETIC = "synthetic"
_DEFAULT_SIZE = (960, 540)
_SYNTHETIC_FPS = 15.0
_POLL_S = 0.05

# Bounds so a missing or wedged camera cannot stall the grab thread forever.
# Tests monkeypatch these; they are not part of the StableCaptureSession API.
CAMERA_OPEN_TIMEOUT_S = 5.0
CAMERA_READ_TIMEOUT_S = 3.0
CAMERA_PROBE_WINDOW_S = 0.8


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


def camera_backends(platform: str | None = None) -> list[tuple[str, int | None]]:
    """Return ``(label, apiPreference)`` pairs to try, in order.

    ``apiPreference is None`` means ``cv2.VideoCapture(index)`` with the
    platform default. Windows tries DirectShow first (it usually fails fast)
    and then Media Foundation.
    """
    name = sys.platform if platform is None else platform
    if name.startswith("win"):
        backends: list[tuple[str, int | None]] = []
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(("DirectShow", int(cv2.CAP_DSHOW)))
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(("Media Foundation", int(cv2.CAP_MSMF)))
        return backends or [("默认", None)]
    return [("默认", None)]


def open_failure_message(index: int, failures: list[str]) -> str:
    """One Simplified Chinese line for a camera that never produced a frame."""
    if not failures:
        return f"无法打开摄像头 {index}，已改用合成帧。"
    if len(failures) == 1 and failures[0].startswith("默认"):
        reason = failures[0].removeprefix("默认").strip()
        if reason in ("", "失败"):
            return f"无法打开摄像头 {index}，已改用合成帧。"
        if reason == "打开超时":
            return f"摄像头 {index} 打开超时，已改用合成帧。"
        if reason == "读帧超时":
            return f"摄像头 {index} 读帧超时，已改用合成帧。"
        if reason == "无画面":
            return f"摄像头 {index} 没有画面，已改用合成帧。"
    detail = "、".join(failures)
    if any("超时" in item for item in failures):
        return f"摄像头 {index} 打开失败或超时（{detail}），已改用合成帧。"
    return f"无法打开摄像头 {index}（{detail}），已改用合成帧。"


def _timeout_props() -> list[int]:
    props: list[int] = []
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        props.extend([int(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC), int(CAMERA_OPEN_TIMEOUT_S * 1000)])
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        props.extend([int(cv2.CAP_PROP_READ_TIMEOUT_MSEC), int(CAMERA_READ_TIMEOUT_S * 1000)])
    return props


def _make_capture(index: int, backend: int | None) -> cv2.VideoCapture:
    """Open a device. The default backend keeps the historical one-argument call."""
    if backend is None:
        return cv2.VideoCapture(int(index))
    api = int(backend)
    props = _timeout_props()
    if not props:
        return cv2.VideoCapture(int(index), api)
    try:
        return cv2.VideoCapture(int(index), api, props)
    except TypeError:
        return cv2.VideoCapture(int(index), api)


def open_camera_device(payload: dict) -> cv2.VideoCapture | None:
    """Open one backend and apply the requested size. ``None`` means not opened.

    Runs on the camera worker thread. Callers must not use the returned
    capture from any other thread.
    """
    capture = None
    try:
        capture = _make_capture(int(payload["index"]), payload.get("backend"))
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            return None
        width = payload.get("width")
        height = payload.get("height")
        if width and int(width) > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height and int(height) > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        return capture
    except Exception:
        if capture is not None:
            capture.release()
        raise


class _CameraWorker:
    """Own ``VideoCapture`` on one thread so a hung open/read can be abandoned.

    OpenCV's camera calls are not interruptible and are not safe to share
    across threads. The grab thread only waits on a reply queue. On timeout it
    sets ``_drop`` and never touches that capture again; the worker releases it
    if the blocked call ever returns.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._drop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="camera-worker", daemon=True)
        self._thread.start()

    def call(
        self,
        kind: str,
        payload,
        timeout_s: float,
        cancel: threading.Event | None,
    ) -> tuple[str, object]:
        """Return ``(status, value)``. Status is ``ok``, ``timeout``, ``cancelled``, or ``abandoned``."""
        if self._drop.is_set():
            return "abandoned", None
        reply: queue.Queue = queue.Queue(maxsize=1)
        self._queue.put((kind, payload, reply))
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            if cancel is not None and cancel.is_set():
                self._drop.set()
                return "cancelled", None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._drop.set()
                return "timeout", None
            try:
                status, value = reply.get(timeout=min(_POLL_S, remaining))
            except queue.Empty:
                continue
            return status, value

    def shutdown(self, timeout_s: float = 0.8) -> None:
        """Release the device when the worker is still answering. A dropped worker is left to exit itself."""
        if self._drop.is_set():
            return
        status, _value = self.call("shutdown", None, timeout_s, None)
        if status != "ok":
            self._drop.set()

    def _loop(self) -> None:
        capture: cv2.VideoCapture | None = None
        try:
            while not self._drop.is_set():
                try:
                    kind, payload, reply = self._queue.get(timeout=_POLL_S)
                except queue.Empty:
                    continue
                if self._drop.is_set() and kind != "shutdown":
                    reply.put(("abandoned", None))
                    return
                try:
                    if kind == "shutdown":
                        capture = self._take_release(capture)
                        reply.put(("ok", None))
                        return
                    if kind == "open":
                        capture = self._take_release(capture)
                        capture = open_camera_device(payload if isinstance(payload, dict) else {})
                        reply.put(("ok", capture is not None))
                    elif kind == "read":
                        reply.put(("ok", self._read(capture)))
                    else:
                        reply.put(("error", f"未知命令 {kind}"))
                except Exception as exc:  # noqa: BLE001 — report and keep the worker alive
                    reply.put(("error", exc))
                if self._drop.is_set():
                    return
        finally:
            self._safe_release(capture)

    @staticmethod
    def _read(capture: cv2.VideoCapture | None) -> tuple[bool, np.ndarray | None]:
        if capture is None:
            return False, None
        ok, frame = capture.read()
        if not ok or frame is None:
            return False, None
        return True, np.ascontiguousarray(frame).copy()

    @staticmethod
    def _safe_release(capture: cv2.VideoCapture | None) -> None:
        if capture is None:
            return
        try:
            capture.release()
        except Exception:
            pass

    def _take_release(self, capture: cv2.VideoCapture | None) -> None:
        self._safe_release(capture)
        return None


class VideoCaptureThread(QThread):
    """OpenCV capture on a ``QThread``. Do not call ``read()`` on the UI thread.

    Signals
    -------
    frame_ready(object, float)
        BGR ``ndarray`` snapshot and ``time.time()`` seconds. Treat the array
        as read-only; mutate a copy.
    error(str)
        Open failure or unexpected loop error. A missing, failed, or timed-out
        camera falls back to synthetic frames after one ``error``.
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
        self._worker: _CameraWorker | None = None
        self._primed: np.ndarray | None = None
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
            self._enter_synthetic(f"摄像头 {index} 不可用，已改用合成帧（source='synthetic'）。")
            return
        failures: list[str] = []
        for label, backend in camera_backends():
            if self._stop_event.is_set():
                return
            worker = _CameraWorker()
            adopted = False
            try:
                status, opened = worker.call(
                    "open",
                    {
                        "index": index,
                        "backend": backend,
                        "width": self._width,
                        "height": self._height,
                    },
                    CAMERA_OPEN_TIMEOUT_S,
                    self._stop_event,
                )
                if status == "cancelled" or self._stop_event.is_set():
                    return
                if status == "timeout":
                    failures.append(f"{label} 打开超时")
                    continue
                if status != "ok" or not opened:
                    failures.append(f"{label} 失败")
                    continue
                primed, probe = self._probe_frames(worker)
                if probe == "cancelled" or self._stop_event.is_set():
                    return
                if probe == "frame" and primed is not None:
                    self._worker = worker
                    self._primed = primed
                    self._mode = "camera"
                    adopted = True
                    return
                if probe == "timeout":
                    failures.append(f"{label} 读帧超时")
                else:
                    failures.append(f"{label} 无画面")
            finally:
                if not adopted:
                    worker.shutdown()
        self._enter_synthetic(open_failure_message(index, failures))

    def _probe_frames(self, worker: _CameraWorker) -> tuple[np.ndarray | None, str]:
        """Read until one real frame arrives, the probe window ends, or the read hangs."""
        deadline = time.monotonic() + max(0.0, CAMERA_PROBE_WINDOW_S)
        while not self._stop_event.is_set():
            status, value = worker.call("read", None, CAMERA_READ_TIMEOUT_S, self._stop_event)
            if status == "cancelled":
                return None, "cancelled"
            if status == "timeout":
                return None, "timeout"
            if status == "ok" and isinstance(value, tuple) and value[0] and value[1] is not None:
                return value[1], "frame"
            if time.monotonic() >= deadline:
                return None, "empty"
            if self._stop_event.wait(0.05):
                return None, "cancelled"
        return None, "cancelled"

    def _enter_synthetic(self, message: str) -> None:
        self._mode = "synthetic"
        self._primed = None
        self.error.emit(message)

    def _open_file(self, path: str) -> None:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            self._mode = "synthetic"
            self.error.emit(f"无法打开视频 {path}，已改用合成帧。")
            return
        self._capture = capture
        self._mode = "file"

    def _read_frame(self) -> np.ndarray | None:
        if self._mode == "camera":
            return self._read_camera()
        capture = self._capture
        if capture is None:
            return self._synthetic_frame()
        ok, frame = capture.read()
        if not ok or frame is None:
            if self._mode == "file":
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
            if not ok or frame is None:
                self.error.emit("读帧失败，已改用合成帧。")
                self._release_capture()
                self._mode = "synthetic"
                return self._synthetic_frame()
        return np.ascontiguousarray(frame).copy()

    def _read_camera(self) -> np.ndarray | None:
        if self._primed is not None:
            frame = self._primed
            self._primed = None
            return frame
        worker = self._worker
        if worker is None:
            self._enter_synthetic("读帧失败，已改用合成帧。")
            return self._synthetic_frame()
        status, value = worker.call("read", None, CAMERA_READ_TIMEOUT_S, self._stop_event)
        if status == "cancelled":
            return None
        if status == "timeout":
            self._release_worker()
            self._enter_synthetic("摄像头读帧超时，已改用合成帧。")
            return self._synthetic_frame()
        if status != "ok" or not isinstance(value, tuple) or not value[0] or value[1] is None:
            self._release_worker()
            self._enter_synthetic("读帧失败，已改用合成帧。")
            return self._synthetic_frame()
        return value[1]

    def _synthetic_frame(self) -> np.ndarray:
        self._tick += 1
        width, height = self._frame_size()
        return render_synthetic_frame(width, height, self._tick)

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

    def _release_worker(self) -> None:
        with self._lock:
            worker = self._worker
            self._worker = None
            self._primed = None
        if worker is not None:
            worker.shutdown()

    def _release_capture(self) -> None:
        self._release_worker()
        with self._lock:
            capture = self._capture
            self._capture = None
        if capture is not None:
            capture.release()
