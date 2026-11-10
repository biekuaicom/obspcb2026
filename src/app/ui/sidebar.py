"""Right-hand dock contents. Placeholder only — no parameter editing."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from model.schema import STATUS_COLORS
from vision.policy import TRACKING_LOST_TTL_S


class SidebarPanel(QWidget):
    """Dock body.

    Signals match TECH_SPEC §3.5 (extended fields, status, batch status).
    Nothing in this slice emits them. The panel only explains the color
    convention and the global→frame matrix direction.
    """

    extended_edited = pyqtSignal(str, dict)
    status_changed = pyqtSignal(str, str)
    batch_set_status = pyqtSignal(list, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebarPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("元件参数")
        title.setObjectName("sidebarTitle")
        layout.addWidget(title)

        hint = QLabel(
            "占位，还不能编辑。气泡只有 name、value、package。"
            "核对状态在顶层 status。doubt 不在本轮界面里。"
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addWidget(_swatch("未核对 unchecked", STATUS_COLORS["unchecked"]))
        layout.addWidget(_swatch("已核对 checked", STATUS_COLORS["checked"]))

        direction = QLabel(
            "矩阵：全局像素 → 当前帧。主路径是局部–全局，仿射是降级。"
            f"跟踪丢失 {TRACKING_LOST_TTL_S:g}s 后应冻结叠加并提示；计时器还没接上。"
        )
        direction.setObjectName("hint")
        direction.setWordWrap(True)
        layout.addWidget(direction)

        self._notice = QLabel("")
        self._notice.setObjectName("hint")
        self._notice.setWordWrap(True)
        layout.addWidget(self._notice)
        layout.addStretch(1)

    def set_notice(self, text: str) -> None:
        self._notice.setText(text)


def _swatch(text: str, color: str) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    chip = QLabel()
    chip.setFixedSize(14, 14)
    chip.setStyleSheet(f"background: {color}; border-radius: 2px;")
    layout.addWidget(chip)
    layout.addWidget(QLabel(text))
    layout.addStretch(1)
    return row
