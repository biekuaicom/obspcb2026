"""Scan workbench: live red coverage, incremental color overview, sample demo."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from app.paths import repo_root
from vision.stable_capture import (
    StableCaptureSession,
    default_sample_global,
    default_scan_export_dir,
    demo_disconnected_frame,
    demo_scan_frames,
    run_board_demo,
    save_session,
)

FrameProvider = Callable[[], np.ndarray | None]
SourceStatus = Callable[[], tuple[str, str]]

_IDLE_STATUS = "点击「开始扫描」。稳定后会自动拍下第一张。"
_LIVE_STATUS = "已显示实时画面。点击「开始扫描」。稳定后会自动拍下第一张。"
_WAIT_STATUS = "正在等待第一帧画面。"
_DEFAULT_HINT = "红色覆盖跟着板面走。画面里还留着大约一到两成已拍区域，并且位姿确定，就可以拍下一张。位姿不确定时会提示放慢或稍退。"
_PREVIEW_WAIT_S = 5.0

_STYLE = """
QDialog, QLabel, QPushButton, QFrame {
    color: #e7eef8;
    font-size: 13px;
}
QDialog { background: #1b2330; }
QLabel#title { font-size: 18px; font-weight: 600; }
QLabel#hint, QLabel#muted { color: #9aabc0; }
QLabel#live, QLabel#overview {
    background: #101610;
    border: 1px solid #2c3b52;
    border-radius: 6px;
    color: #9aabc0;
}
QFrame#card {
    background: #243044;
    border: 1px solid #2c3b52;
    border-radius: 6px;
}
QPushButton {
    background: #2c3b52;
    border: 1px solid #3d516d;
    border-radius: 4px;
    padding: 8px 12px;
}
QPushButton:hover { background: #3d516d; }
QPushButton:disabled { color: #6d7d90; background: #1b2330; }
QPushButton#primary {
    background: #1f6feb;
    border-color: #388bfd;
    font-weight: 600;
}
QPushButton#primary:hover { background: #388bfd; }
QPushButton#primary:disabled { background: #1b2330; }
"""


def _pixmap_from_bgr(image: np.ndarray, max_width: int, max_height: int) -> QPixmap:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    height, width, channels = rgb.shape
    qimage = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(qimage)
    if max_width > 1 and max_height > 1:
        pixmap = pixmap.scaled(
            max_width,
            max_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return pixmap


class ScanWorkbenchDialog(QDialog):
    """Interactive stable-capture workbench with a live overview thumbnail."""

    def __init__(
        self,
        frame_provider: FrameProvider,
        parent=None,
        *,
        board_id: str = "SCAN-BOARD",
        source_status: SourceStatus | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("扫描建图")
        self.setModal(True)
        self.resize(1180, 760)
        self.setStyleSheet(_STYLE)
        self.frame_provider = frame_provider
        self.source_status = source_status
        self.board_id = board_id
        self._preview_wait_t0: float | None = None
        self.session = StableCaptureSession()
        self._running = False
        self._first_auto_done = False
        self._sample_frames: list[np.ndarray] = []
        self._sample_index = 0
        self._use_sample = False
        self._exported_dir: Path | None = None
        self._overview_version = -1
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._tick)
        self._build_ui()
        self._update_buttons()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(14)

        left = QVBoxLayout()
        title = QLabel("扫描建图 · 稳定间隔拍照")
        title.setObjectName("title")
        left.addWidget(title)
        subtitle = QLabel(
            "停在清晰稳定的画面拍第一张，移动时让红色覆盖还留在画面里再拍下一张。"
            "右侧会立刻把每张彩色关键帧拼进整板总览（受约束的低分辨率预览，不是全分辨率画布）。"
            "没有摄像头、打开失败或超时时会自动改用合成帧，并在右侧说明原因；"
            "也可以点「使用样例画面」或「一键演示（样例）」。"
        )
        subtitle.setObjectName("hint")
        subtitle.setWordWrap(True)
        left.addWidget(subtitle)
        self.live_view = QLabel("等待画面…")
        self.live_view.setObjectName("live")
        self.live_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_view.setWordWrap(True)
        self.live_view.setMinimumSize(640, 420)
        left.addWidget(self.live_view, 1)
        self.live_hint = QLabel(_DEFAULT_HINT)
        self.live_hint.setObjectName("hint")
        self.live_hint.setWordWrap(True)
        left.addWidget(self.live_hint)
        root.addLayout(left, 3)

        right = QVBoxLayout()
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        status_title = QLabel("扫描状态")
        status_title.setStyleSheet("font-weight: 600;")
        card_layout.addWidget(status_title)
        self.status_label = QLabel(_IDLE_STATUS)
        self.status_label.setWordWrap(True)
        card_layout.addWidget(self.status_label)
        self.metrics_label = QLabel("已拍 0 张 · 清晰度 — · 移动 —")
        self.metrics_label.setObjectName("muted")
        self.metrics_label.setWordWrap(True)
        card_layout.addWidget(self.metrics_label)
        self.overlap_label = QLabel("重叠 —")
        self.overlap_label.setObjectName("muted")
        card_layout.addWidget(self.overlap_label)
        right.addWidget(card)

        overview_title = QLabel("增量整板总览")
        overview_title.setObjectName("muted")
        right.addWidget(overview_title)
        self.overview_view = QLabel("每拍一张会立即并入这里，可查看接缝")
        self.overview_view.setObjectName("overview")
        self.overview_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overview_view.setMinimumHeight(200)
        self.overview_view.setWordWrap(True)
        right.addWidget(self.overview_view, 1)

        self.sample_nav = QFrame()
        nav = QHBoxLayout(self.sample_nav)
        nav.setContentsMargins(0, 0, 0, 0)
        self.prev_sample_button = QPushButton("上一区域")
        self.next_sample_button = QPushButton("下一区域")
        self.prev_sample_button.clicked.connect(lambda: self._step_sample(-1))
        self.next_sample_button.clicked.connect(lambda: self._step_sample(1))
        nav.addWidget(self.prev_sample_button)
        nav.addWidget(self.next_sample_button)
        self.sample_nav.hide()
        right.addWidget(self.sample_nav)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("开始扫描")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.start_scan)
        self.capture_button = QPushButton("拍摄当前区域")
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self.capture_now)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.capture_button)
        right.addLayout(buttons)

        self.sample_button = QPushButton("使用样例画面")
        self.sample_button.clicked.connect(self.use_sample_views)
        self.demo_button = QPushButton("一键演示（样例）")
        self.demo_button.setObjectName("primary")
        self.demo_button.clicked.connect(self.run_sample_demo)
        self.finish_button = QPushButton("完成并导出整板图")
        self.finish_button.setEnabled(False)
        self.finish_button.clicked.connect(self.finish_and_export)
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.reject)
        right.addWidget(self.sample_button)
        right.addWidget(self.demo_button)
        right.addWidget(self.finish_button)
        right.addWidget(self.close_button)

        self.export_label = QLabel("")
        self.export_label.setObjectName("muted")
        self.export_label.setWordWrap(True)
        right.addWidget(self.export_label)
        right.addStretch(1)
        root.addLayout(right, 2)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._preview_wait_t0 is None:
            self._preview_wait_t0 = time.monotonic()
        if not self._timer.isActive():
            self._timer.start()
        self._tick()

    def current_frame(self) -> np.ndarray | None:
        if self._use_sample and self._sample_frames:
            return self._sample_frames[self._sample_index]
        try:
            return self.frame_provider()
        except Exception:  # noqa: BLE001
            return None

    def start_scan(self) -> None:
        self.session = StableCaptureSession()
        self.session.start()
        self._running = True
        self._first_auto_done = False
        self._exported_dir = None
        self._overview_version = -1
        self.start_button.setText("重新开始")
        self.status_label.setText("已开始，等待稳定画面以拍摄第一张…")
        self.export_label.setText("")
        self.overview_view.setPixmap(QPixmap())
        self.overview_view.setText("每拍一张会立即并入这里，可查看接缝")
        if not self._timer.isActive():
            self._timer.start()
        self._tick()
        self._update_buttons()

    def use_sample_views(self) -> None:
        frames = self._load_sample_frames()
        if not frames:
            return
        self._sample_frames = frames
        self._sample_index = 0
        self._use_sample = True
        self.sample_nav.show()
        self.sample_button.setText("样例画面已启用")
        self.status_label.setText(
            f"已载入样例 {len(frames)} 个重叠视图。点击「开始扫描」，再用「下一区域」移动并拍摄。"
        )
        if not self._timer.isActive():
            self._timer.start()
        self._tick()
        self._update_buttons()

    def _load_sample_frames(self) -> list[np.ndarray]:
        path = default_sample_global()
        if path is None:
            QMessageBox.warning(self, "没有样例", "找不到 assets/samples/SAMPLE-PARTIAL-001/global_ref.png")
            return []
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.warning(self, "无法读取样例", str(path))
            return []
        frames = demo_scan_frames(image)
        disconnected = demo_disconnected_frame(image)
        if disconnected.size:
            frames.append(disconnected)
        return frames

    def _step_sample(self, delta: int) -> None:
        if not self._sample_frames:
            return
        self._sample_index = int(np.clip(self._sample_index + delta, 0, len(self._sample_frames) - 1))
        self._tick()

    def run_sample_demo(self) -> None:
        path = default_sample_global()
        if path is None:
            QMessageBox.warning(self, "没有样例", "找不到 assets/samples/SAMPLE-PARTIAL-001/global_ref.png")
            return
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.warning(self, "无法读取样例", str(path))
            return
        out_dir = default_scan_export_dir("SAMPLE-PARTIAL-001", repo_root())
        self._timer.stop()
        self.status_label.setText("正在用样例跑完整扫描流程…")
        try:
            summary = run_board_demo(image, out_dir, board_id="SAMPLE-PARTIAL-001", source=str(path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "演示失败", str(exc))
            self._update_buttons()
            return
        self._exported_dir = out_dir
        self.session = StableCaptureSession()
        self._running = False
        overview_path = Path(summary.get("overview") or (out_dir / "overview.png"))
        overview = cv2.imread(str(overview_path), cv2.IMREAD_COLOR)
        coverage = cv2.imread(str(out_dir / "coverage.png"), cv2.IMREAD_COLOR)
        if coverage is not None:
            self._show_image(self.live_view, coverage)
        if overview is not None:
            self._show_image(self.overview_view, overview)
            self._overview_version = -1
        size = summary.get("overview_size") or [0, 0]
        self.status_label.setText(
            f"演示完成：拍摄 {summary.get('shots')} 张，拒绝 {summary.get('rejects')} 次。"
            f"整板总览 {size[0]}×{size[1]}。"
        )
        self.metrics_label.setText(f"已拍 {summary.get('shots')} 张 · 样例一键演示")
        self.overlap_label.setText(f"平均重叠 {summary.get('overlap_mean')}")
        self.export_label.setText(f"已导出到 {out_dir}")
        self._update_buttons()
        QMessageBox.information(
            self,
            "演示完成",
            f"整板总览已写出：\n{overview_path}\n尺寸 {size[0]}×{size[1]} px\n"
            f"同时保存了 chain.json 与 topology_db.json。",
        )

    def capture_now(self) -> None:
        if not self._running:
            return
        decision = self.session.capture()
        self.status_label.setText(decision.reason)
        if decision.ok:
            self._refresh_overview(force=True)
            if len(self.session.tiles) >= 2:
                self.export_label.setText(f"已拍摄 {len(self.session.tiles)} 张，可以完成并导出整板图。")
        self._tick()
        self._update_buttons()

    def finish_and_export(self) -> None:
        if len(self.session.tiles) < 1:
            QMessageBox.information(self, "还没有照片", "请先开始扫描并至少拍摄一张。")
            return
        if len(self.session.tiles) < 2:
            if QMessageBox.question(
                self,
                "只拍了一张",
                "至少两张重叠照片才能看出整板拼图。仍要导出当前结果吗？",
            ) != QMessageBox.StandardButton.Yes:
                return
        root = repo_root()
        out_dir = default_scan_export_dir(self.board_id, root)
        summary = save_session(self.session, out_dir, board_id=self.board_id, source="scan_workbench")
        self._exported_dir = out_dir
        size = summary.get("overview_size") or [0, 0]
        overview = out_dir / "overview.png"
        self.export_label.setText(f"已导出到 {out_dir}")
        self.status_label.setText(f"整板总览已保存：{overview.name}（{size[0]}×{size[1]}）")
        QMessageBox.information(
            self,
            "已导出整板图",
            f"输出目录：\n{out_dir}\n\n"
            f"overview.png  {size[0]}×{size[1]} px\n"
            f"chain.json\n"
            f"topology_db.json",
        )

    def _source_state(self) -> tuple[str, str]:
        if self.source_status is None:
            return "", ""
        try:
            mode, notice = self.source_status()
        except Exception:  # noqa: BLE001 — a broken status callback must not freeze the preview
            return "", ""
        return str(mode or ""), str(notice or "")

    def _apply_notice(self, notice: str, *, idle: bool) -> None:
        if notice:
            self.live_hint.setText(notice)
            if idle:
                self.status_label.setText(notice)
            return
        if self.live_hint.text() != _DEFAULT_HINT:
            self.live_hint.setText(_DEFAULT_HINT)
        if idle:
            self.status_label.setText(_LIVE_STATUS)

    def _show_waiting(self, mode: str, notice: str) -> None:
        """Replace the stuck placeholder once open has produced neither a frame nor an error."""
        self.live_view.clear()
        if self._preview_wait_t0 is None:
            self._preview_wait_t0 = time.monotonic()
        if notice:
            self.live_view.setText(notice)
            self.status_label.setText(
                "摄像头没有可用画面。可以点「使用样例画面」或「一键演示（样例）」，窗口仍可操作。"
            )
            self.live_hint.setText(notice)
            return
        waited = time.monotonic() - self._preview_wait_t0
        if mode in ("", "pending") and waited < _PREVIEW_WAIT_S:
            self.live_view.setText("正在打开摄像头，请稍候…")
            self.status_label.setText(_WAIT_STATUS)
            return
        if mode in ("", "pending"):
            self.live_view.setText("仍未收到摄像头画面。若打开失败或超时，将自动改用合成帧。")
            self.status_label.setText("正在等待画面，窗口仍可操作。")
            return
        self.live_view.setText("没有实时画面。可点「使用样例画面」或「一键演示（样例）」。")
        self.status_label.setText("没有画面")

    def _tick(self) -> None:
        frame = self.current_frame()
        mode, notice = self._source_state()
        if frame is None:
            self._show_waiting(mode, notice)
            self._update_buttons()
            return
        self._preview_wait_t0 = None
        if self._running:
            state = self.session.observe(frame)
            painted = self.session.render(frame)
            self._show_image(self.live_view, painted)
            motion = "—" if state.motion_px is None else f"{state.motion_px:.1f}px"
            overlap = "—" if state.overlap is None else f"{state.overlap:.0%}"
            self.status_label.setText(state.message)
            self.metrics_label.setText(
                f"已拍 {len(self.session.tiles)} 张 · 清晰度 {state.sharpness:.0f} · 移动 {motion}"
            )
            self.overlap_label.setText(f"红区重叠 {overlap}" + (f" · 样例视图 {self._sample_index + 1}/{len(self._sample_frames)}" if self._use_sample else ""))
            if (
                not self._first_auto_done
                and not self.session.tiles
                and state.stable
            ):
                decision = self.session.capture()
                if decision.ok:
                    self._first_auto_done = True
                    self.status_label.setText("第一张已拍摄并并入总览。请移动并保留红色覆盖，再拍下一张。")
                    self._refresh_overview(force=True)
            self._apply_notice(notice, idle=False)
        else:
            self._show_image(self.live_view, frame)
            self._apply_notice(notice, idle=not self._use_sample and self._exported_dir is None)
        self._refresh_overview()
        self._update_buttons()

    def _update_buttons(self) -> None:
        can_shoot = False
        if self._running:
            frame = self.current_frame()
            can_shoot = frame is not None
            if can_shoot and self.session.tiles:
                can_shoot = self.session._tracked and float(self.session._overlap or 0.0) >= self.session.min_overlap
            elif can_shoot and not self.session.tiles:
                # allow the button; capture() still enforces sharpness / two-frame settle
                can_shoot = True
        self.capture_button.setEnabled(can_shoot)
        self.finish_button.setEnabled(len(self.session.tiles) >= 1)
        self.prev_sample_button.setEnabled(self._use_sample and self._sample_index > 0)
        self.next_sample_button.setEnabled(
            self._use_sample and self._sample_index < max(0, len(self._sample_frames) - 1)
        )

    def _refresh_overview(self, force: bool = False) -> None:
        image = self.session.overview_image()
        if image is None:
            return
        if not force and self._overview_version == self.session._overview_version:
            return
        self._overview_version = self.session._overview_version
        self._show_image(self.overview_view, image)

    def _show_image(self, label: QLabel, image: np.ndarray) -> None:
        width = max(120, label.width() - 8)
        height = max(90, label.height() - 8)
        label.setText("")
        label.setPixmap(_pixmap_from_bgr(image, width, height))

    def reject(self) -> None:
        self._timer.stop()
        super().reject()

    def closeEvent(self, event) -> None:
        self._timer.stop()
        super().closeEvent(event)
