"""Color overview export and the headless / UI scan entry points."""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import QDialog

from app.ui.main_window import MainWindow
from app.ui.scan_dialog import ScanWorkbenchDialog
from vision import capture as capture_module
from vision.stable_capture import (
    OVERVIEW_MAX_PIXELS,
    OVERVIEW_MAX_SIDE,
    StableCaptureSession,
    default_sample_global,
    run_board_demo,
    run_demo_scan,
)


def _board(height=240, width=460, seed=4):
    rng = np.random.default_rng(seed)
    image = np.full((height, width), 176, np.uint8)
    noise = rng.integers(-4, 5, size=image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for row, y in enumerate(range(36, height - 36, 26)):
        for col, x in enumerate(range(32, width - 32, 28)):
            px = int(np.clip(x + int(rng.integers(-4, 5)), 16, width - 16))
            py = int(np.clip(y + int(rng.integers(-4, 5)), 16, height - 16))
            if (row + col) % 3 == 0:
                cv2.circle(image, (px, py), 4, 18, -1)
            else:
                cv2.rectangle(image, (px - 9, py - 3), (px + 9, py + 3), 30, -1)
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Paint a distinct hue on the left and right so the overview must span both tiles.
    color[:, :80, 0] = np.clip(color[:, :80, 0].astype(np.int16) + 80, 0, 255).astype(np.uint8)
    color[:, -80:, 2] = np.clip(color[:, -80:, 2].astype(np.int16) + 80, 0, 255).astype(np.uint8)
    return color


def _settle(session, frame):
    session.observe(frame)
    return session.observe(frame)


def _nonzero_span(mask: np.ndarray) -> int:
    columns = np.flatnonzero(np.any(mask, axis=0))
    if columns.size == 0:
        return 0
    return int(columns[-1] - columns[0] + 1)


def test_run_board_demo_writes_color_overview(tmp_path):
    sample = default_sample_global()
    assert sample is not None and sample.is_file()
    image = cv2.imread(str(sample), cv2.IMREAD_COLOR)
    assert image is not None
    out_dir = tmp_path / "demo"
    summary = run_board_demo(image, out_dir, board_id="SAMPLE-PARTIAL-001", source=str(sample))
    overview_path = Path(summary["overview"])
    assert overview_path.is_file()
    overview = cv2.imread(str(overview_path), cv2.IMREAD_COLOR)
    assert overview is not None
    height, width = overview.shape[:2]
    assert overview.ndim == 3
    assert int(np.count_nonzero(overview)) > 0
    assert max(width, height) <= OVERVIEW_MAX_SIDE
    assert width * height <= OVERVIEW_MAX_PIXELS
    assert summary["shots"] >= 2
    assert summary["full_canvas"] is False
    # Three 480-wide windows stepped by 70/140 px must span more than one tile.
    assert width > 480
    assert _nonzero_span(np.any(overview > 8, axis=2)) > 480
    assert summary["overview_tiles"] == summary["shots"]
    assert summary["overview_size"] == [width, height]


def test_disconnected_view_is_rejected_and_overview_keeps_two_tiles():
    board = _board()
    window = board[24:150, 20:160]
    shifted = board[24:150, 44:184]
    disconnected = board[24:150, 300:440]
    session = StableCaptureSession(min_overlap=0.08)
    session.start()
    _settle(session, window)
    assert session.capture().ok is True
    _settle(session, shifted)
    assert session.capture().ok is True
    overview = session.overview_image()
    assert overview is not None
    height, width = overview.shape[:2]
    assert overview.ndim == 3 and int(np.count_nonzero(overview)) > 0
    assert max(width, height) <= OVERVIEW_MAX_SIDE
    assert width * height <= OVERVIEW_MAX_PIXELS
    assert width > window.shape[1]
    _settle(session, disconnected)
    rejected = session.capture()
    assert rejected.ok is False
    assert "退回" in rejected.reason
    assert len(session.tiles) == 2
    topology = session.topology_db(board_id="SYN", source="synthetic")
    assert topology["full_canvas"] is False
    assert session.chain_dict()["tiles"][0]["id"] == "cap-000"


def test_overview_rebuilds_when_limits_are_tight():
    board = _board(height=240, width=520)
    views = [board[24:150, x : x + 140] for x in (20, 44, 68, 92, 116, 140, 164)]
    session = StableCaptureSession(
        min_overlap=0.08,
        overview_max_side=220,
        overview_max_pixels=80_000,
    )
    session.start()
    accepted = 0
    for view in views:
        _settle(session, view)
        if session.capture().ok:
            accepted += 1
    assert accepted >= 3
    overview = session.overview_image()
    assert overview is not None
    height, width = overview.shape[:2]
    assert max(width, height) <= 220
    assert width * height <= 80_000
    # Later tiles were not dropped: the overview is wider than a single window
    # even after the canvas was scaled down to the cap.
    assert _nonzero_span(np.any(overview > 8, axis=2)) > 140
    assert session.overview_meta()["tiles"] == accepted
    assert session.overview_meta()["full_canvas"] is False


def test_run_demo_scan_writes_overview_without_qt(tmp_path):
    summary = run_demo_scan(tmp_path / "scan")
    overview = Path(summary["overview"])
    assert overview.is_file()
    image = cv2.imread(str(overview), cv2.IMREAD_COLOR)
    assert image is not None
    height, width = image.shape[:2]
    assert max(width, height) <= OVERVIEW_MAX_SIDE
    assert width * height <= OVERVIEW_MAX_PIXELS
    assert int(np.count_nonzero(image)) > 0
    assert (tmp_path / "scan" / "chain.json").is_file()
    assert (tmp_path / "scan" / "topology_db.json").is_file()
    assert summary["full_canvas"] is False
    assert summary["shots"] >= 2
    assert summary["rejects"] >= 1


def _wait(qapp, predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _pixmap_ready(label) -> bool:
    pixmap = label.pixmap()
    return pixmap is not None and not pixmap.isNull()


def test_scan_workbench_shows_live_preview_without_start(qapp):
    frame = np.zeros((48, 64, 3), np.uint8)
    frame[:] = (20, 40, 60)
    dialog = ScanWorkbenchDialog(lambda: frame)
    dialog.show()
    qapp.processEvents()
    try:
        assert dialog.windowTitle() == "扫描建图"
        assert dialog.start_button.text() == "开始扫描"
        assert dialog.session.started is False
        assert not dialog.capture_button.isEnabled()
        assert _pixmap_ready(dialog.live_view)
        assert "等待画面" not in dialog.live_view.text()
        assert "开始扫描" in dialog.status_label.text()
    finally:
        dialog.close()


def test_scan_workbench_replaces_placeholder_while_opening(qapp):
    dialog = ScanWorkbenchDialog(lambda: None, source_status=lambda: ("pending", ""))
    dialog.show()
    qapp.processEvents()
    try:
        assert "等待画面" not in dialog.live_view.text()
        assert "请稍候" in dialog.live_view.text()
        assert dialog.sample_button.isEnabled()
        assert dialog.demo_button.isEnabled()
        assert dialog.close_button.isEnabled()
        assert not dialog.capture_button.isEnabled()
        dialog.close_button.click()
        qapp.processEvents()
        assert not dialog.isVisible()
    finally:
        dialog.close()


def test_scan_workbench_preview_updates_when_frame_arrives(qapp):
    holder: dict = {"frame": None}
    dialog = ScanWorkbenchDialog(lambda: holder["frame"], source_status=lambda: ("pending", ""))
    dialog.show()
    qapp.processEvents()
    try:
        assert "请稍候" in dialog.live_view.text()
        holder["frame"] = np.full((30, 40, 3), 200, np.uint8)
        assert _wait(qapp, lambda: _pixmap_ready(dialog.live_view), timeout=2)
        assert dialog.session.started is False
        assert dialog.start_button.text() == "开始扫描"
        assert not dialog.capture_button.isEnabled()
    finally:
        dialog.close()


def test_scan_workbench_stays_usable_when_camera_failed(qapp):
    notice = "无法打开摄像头 0，已改用合成帧。"
    dialog = ScanWorkbenchDialog(lambda: None, source_status=lambda: ("synthetic", notice))
    dialog.show()
    qapp.processEvents()
    try:
        assert "等待画面" not in dialog.live_view.text()
        assert "摄像头" in dialog.live_view.text()
        assert dialog.sample_button.isEnabled()
        dialog.use_sample_views()
        qapp.processEvents()
        assert dialog.sample_nav.isVisible()
        assert _pixmap_ready(dialog.live_view)
        assert dialog.session.started is False
        assert dialog.close_button.isEnabled()
    finally:
        dialog.close()


def test_open_scan_workbench_previews_without_start_click(qapp):
    window = MainWindow(source="synthetic", width=160, height=120)
    window.show()
    try:
        assert _wait(qapp, lambda: window.current_frame() is not None)
        dialog = window.open_scan_workbench()
        qapp.processEvents()
        assert dialog.session.started is False
        assert dialog.start_button.text() == "开始扫描"
        assert _pixmap_ready(dialog.live_view)
        assert "等待画面" not in dialog.live_view.text()
        dialog.close()
    finally:
        window.close()
        qapp.processEvents()
        assert not window.capture.isRunning()


def test_open_scan_workbench_after_missing_camera_shows_fallback(qapp):
    window = MainWindow(source=31, width=80, height=60)
    window.show()
    try:
        assert _wait(
            qapp,
            lambda: window.current_frame() is not None and "合成帧" in window._capture_notice,
        )
        dialog = window.open_scan_workbench()
        qapp.processEvents()
        assert _pixmap_ready(dialog.live_view)
        assert "合成帧" in dialog.status_label.text()
        assert dialog.session.started is False
        assert dialog.sample_button.isEnabled()
        assert dialog.close_button.isEnabled()
        dialog.close()
    finally:
        window.close()
        qapp.processEvents()
        assert not window.capture.isRunning()


def test_workbench_shows_fallback_when_camera_open_hangs(qapp, monkeypatch):
    hang = threading.Event()

    class Hung:
        def __init__(self, *_args, **_kwargs):
            hang.wait(timeout=5)

        def isOpened(self):
            return False

        def release(self):
            pass

        def read(self):
            return False, None

        def set(self, *_args, **_kwargs):
            return False

    monkeypatch.setattr(capture_module, "CAMERA_OPEN_TIMEOUT_S", 0.3)
    monkeypatch.setattr(capture_module, "CAMERA_READ_TIMEOUT_S", 0.3)
    monkeypatch.setattr(capture_module, "camera_device_present", lambda _index: True)
    monkeypatch.setattr(cv2, "VideoCapture", Hung)
    window = MainWindow(source=0, width=80, height=60)
    window.show()
    try:
        assert _wait(
            qapp,
            lambda: window.current_frame() is not None and "超时" in window._capture_notice,
            timeout=3,
        )
        dialog = window.open_scan_workbench()
        qapp.processEvents()
        assert dialog.session.started is False
        assert _pixmap_ready(dialog.live_view)
        assert "超时" in dialog.status_label.text()
        assert "合成帧" in dialog.status_label.text()
        assert dialog.close_button.isEnabled()
        dialog.close()
    finally:
        hang.set()
        window.close()
        qapp.processEvents()
        assert not window.capture.isRunning()


def test_main_window_scan_action_and_dialog(qapp):
    window = MainWindow(source="synthetic", width=160, height=120)
    window.show()
    try:
        assert window.scan_action.text() == "扫描建图…"
        assert "Ctrl+M" in window.scan_action.shortcut().toString()
        assert window.demo_scan_action.text() == "一键演示扫描"
        assert window.synthetic_action.text() == "演示/合成帧"
        dialog = ScanWorkbenchDialog(window.current_frame, parent=window)
        dialog.show()
        qapp.processEvents()
        assert isinstance(dialog, QDialog)
        assert "扫描" in dialog.windowTitle()
        assert dialog.start_button.text() == "开始扫描"
        assert dialog.capture_button.text() == "拍摄当前区域"
        assert "导出" in dialog.finish_button.text()
        assert "演示" in dialog.demo_button.text()
        dialog.close()
    finally:
        window.close()
        qapp.processEvents()
        assert not window.capture.isRunning()
