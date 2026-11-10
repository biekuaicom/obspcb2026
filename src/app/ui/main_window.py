"""Main window: live pixmap, empty overlay, right dock."""

from __future__ import annotations

import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QDockWidget, QMainWindow

from app.paths import repo_root, sample_project_path
from app.ui.canvas import PCBCanvasView
from app.ui.sidebar import SidebarPanel
from model.schema import SchemaError
from model.store import ProjectStore
from vision.affine_tracker import PCBAffineTracker
from vision.capture import VideoCaptureThread
from vision.local_global import LocalGlobalTracker
from vision.policy import TRACKING_LOST_TTL_S

_STYLE = """
QMainWindow, QMenuBar, QMenu, QStatusBar, QDockWidget, #sidebarPanel, #sidebarPanel QLabel {
    background: #1b2330;
    color: #e7eef8;
}
QMenuBar::item:selected, QMenu::item:selected { background: #2c3b52; }
QDockWidget::title { background: #243044; padding: 6px; }
QStatusBar { background: #151b26; color: #c5d2e4; }
QGraphicsView { background: #101610; border: none; }
QLabel#sidebarTitle { font-size: 16px; font-weight: 600; }
QLabel#hint { color: #9aabc0; }
"""

def missing_mm_warning(project) -> str:
    """D2: pure-pixel boards are allowed, and the shell says so when mm is absent."""
    has_scale = project.pixel_per_mm is not None
    has_positions = all(component.absolute_pos_mm is not None for component in project.components)
    if has_scale and has_positions:
        return ""
    return "纯像素模式：毫米坐标未标，画面只按像素工作。"


_MODE_LABELS = {
    "pending": "等待",
    "synthetic": "合成帧",
    "camera": "摄像头",
    "file": "视频文件",
}


class MainWindow(QMainWindow):
    """Shell window titled 「PCB AR 标注」.

    Trackers are constructed and left idle. Per-frame registration belongs on
    a worker once a real matcher exists; this window does not call ``track``.
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int | None = None,
        height: int | None = None,
        smoke: bool = False,
    ):
        super().__init__()
        self.setWindowTitle("PCB AR 标注")
        self.setStyleSheet(_STYLE)
        self.resize(1100, 720)
        self.setMinimumSize(800, 480)

        self.store = ProjectStore()
        self.affine_tracker = PCBAffineTracker()
        self.local_tracker = LocalGlobalTracker()
        self.canvas = PCBCanvasView()
        self.sidebar = SidebarPanel()
        self.smoke = smoke
        self.smoke_ok = False
        self._shut_down = False
        self._fps = 0.0
        self._fps_count = 0
        self._fps_t0 = time.monotonic()
        self._load_error = ""

        self.setCentralWidget(self.canvas)
        self.dock = QDockWidget("元件参数", self)
        self.dock.setObjectName("paramDock")
        self.dock.setWidget(self.sidebar)
        self.dock.setMinimumWidth(280)
        self.dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)

        file_menu = self.menuBar().addMenu("文件")
        quit_action = file_menu.addAction("退出")
        quit_action.triggered.connect(self.close)

        self._load_sample()
        self.canvas.set_tracking_ok(False)
        self.statusBar().showMessage("正在打开视频…")
        self.statusBar().setToolTip(
            "主路径 LocalGlobal，仿射降级。"
            f"跟踪丢失 TTL {TRACKING_LOST_TTL_S:g}s（计时未接入）。"
            f"Affine ROI {self.affine_tracker.roi_size[0]}×{self.affine_tracker.roi_size[1]}；"
            f"ORB n_features={self.local_tracker.n_features}。"
        )

        self.capture = VideoCaptureThread(source=source, width=width, height=height, parent=self)
        self.capture.frame_ready.connect(self._on_frame)
        self.capture.error.connect(self._on_capture_error)
        self.capture.stopped.connect(self._on_capture_stopped)
        self.capture.start()

        if self.smoke:
            QTimer.singleShot(5000, self._smoke_timeout)

    def shutdown(self) -> None:
        if self._shut_down:
            return
        self._shut_down = True
        self.capture.stop()

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)

    def _load_sample(self) -> None:
        path = sample_project_path()
        if path is None:
            self.sidebar.set_notice("未找到样例工程 JSON。")
            return
        try:
            project = self.store.load(str(path))
        except (OSError, SchemaError, ValueError) as exc:
            self._load_error = str(exc)
            self.sidebar.set_notice(f"样例工程读取失败：{exc}")
            return
        self.setWindowTitle(f"PCB AR 标注 — {project.board_id}")
        root = repo_root()
        map_file = None if root is None else root / project.global_map_path
        if map_file is not None and map_file.is_file():
            notice = f"已载入 {project.board_id}。标注层本轮保持为空。"
        else:
            notice = (
                f"已载入 {project.board_id}。全局图文件本轮不提供，标注层保持为空。"
            )
        mm_warning = missing_mm_warning(project)
        if mm_warning:
            notice = f"{notice} {mm_warning}"
        self.sidebar.set_notice(notice)

    def _on_frame(self, frame, timestamp: float) -> None:
        del timestamp
        self.canvas.set_video_frame(frame)
        self._note_fps()
        self._refresh_status()
        if self.smoke and not self.smoke_ok:
            self.smoke_ok = True
            QTimer.singleShot(0, self.close)

    def _on_capture_error(self, message: str) -> None:
        self.sidebar.set_notice(message)
        self._refresh_status()

    def _on_capture_stopped(self) -> None:
        if self._shut_down:
            return
        self.statusBar().showMessage("采集线程已停止")

    def _note_fps(self) -> None:
        self._fps_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_t0
        if elapsed >= 0.5:
            self._fps = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_t0 = now

    def _refresh_status(self) -> None:
        mode = _MODE_LABELS.get(self.capture.mode, self.capture.mode)
        pixmap = self.canvas.video_item.pixmap()
        size = "—×—" if pixmap.isNull() else f"{pixmap.width()}×{pixmap.height()}"
        fps = f"{self._fps:.1f} FPS" if self._fps else "FPS …"
        if self.store.project is not None:
            project = self.store.project
            board = f"工程 {project.board_id} · 元件 {len(project.components)}"
        elif self._load_error:
            board = "工程未加载"
        else:
            board = "无工程"
        self.statusBar().showMessage(
            f"视频：{mode} {size} · {fps} · {board} · 标注层空 · 跟踪：桩，未配准"
        )

    def _smoke_timeout(self) -> None:
        if not self.smoke_ok:
            self.close()
