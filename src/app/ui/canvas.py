"""Video canvas: pixmap underneath an empty annotation layer."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from app.ui.annotation_items import AnnotationViewModel


class PCBCanvasView(QGraphicsView):
    """Central view.

    Signals match TECH_SPEC §3.5 but are not emitted yet. Inline editing and
    selection sync wait on the open UX decisions. ``set_annotations`` keeps the
    overlay group empty on purpose.
    """

    component_selected = pyqtSignal(str)
    core_param_edited = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pcbCanvas")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QColor("#101610"))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.video_item = QGraphicsPixmapItem()
        self.video_item.setZValue(0)
        self._scene.addItem(self.video_item)
        self.annotation_layer = self._scene.createItemGroup([])
        self.annotation_layer.setZValue(10)
        self._annotations: list[AnnotationViewModel] = []
        self._tracking_ok = False

    @property
    def annotations(self) -> list[AnnotationViewModel]:
        return list(self._annotations)

    @property
    def tracking_ok(self) -> bool:
        return self._tracking_ok

    def set_video_frame(self, bgr: np.ndarray) -> None:
        """Replace the bottom pixmap. ``bgr`` is a BGR ``uint8`` image."""
        if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("set_video_frame expects a BGR image")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888)
        self.video_item.setPixmap(QPixmap.fromImage(image.copy()))
        self._scene.setSceneRect(self.video_item.boundingRect())
        self._fit_video()

    def set_annotations(self, items: list[AnnotationViewModel]) -> None:
        """Remember view models and leave the annotation layer empty."""
        self._annotations = list(items)
        for child in list(self.annotation_layer.childItems()):
            self._scene.removeItem(child)

    def set_tracking_ok(self, ok: bool) -> None:
        self._tracking_ok = bool(ok)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_video()

    def _fit_video(self) -> None:
        if self.video_item.pixmap().isNull():
            return
        self.fitInView(self.video_item, Qt.AspectRatioMode.KeepAspectRatio)
