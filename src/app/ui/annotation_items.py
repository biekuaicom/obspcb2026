"""Graphics items for a later annotation overlay.

The running shell keeps the annotation layer empty. These classes exist so the
TECH_SPEC §3.5 names are importable. They do not implement inline editing.
``unchecked`` is red, ``checked`` is green, ``doubt`` is yellow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import QGraphicsLineItem, QGraphicsRectItem, QGraphicsTextItem

from model.schema import CORE_PARAM_KEYS, STATUS_COLORS


@dataclass
class AnnotationViewModel:
    """Provisional payload for ``PCBCanvasView.set_annotations``.

    Field names can still change with the open UX decisions. The shell stores
    the list and does not turn it into scene items.
    """

    component_id: str
    status: str
    polygon_xy: tuple[tuple[float, float], ...] = ()
    core_params: dict = field(default_factory=dict)


def bubble_text(core_params: dict) -> str:
    """Exactly ``name``, ``value``, ``package``, in that order. Status is not shown."""
    return "\n".join(f"{key}: {core_params.get(key, '')}" for key in CORE_PARAM_KEYS)


def status_color(status: str) -> QColor:
    return QColor(STATUS_COLORS.get(status, STATUS_COLORS["unchecked"]))


class AnnotationRectItem(QGraphicsRectItem):
    """Bounding rect of one component. Not placed on the scene in this slice."""

    def __init__(self, component_id: str, status: str, rect: QRectF | None = None):
        super().__init__(rect or QRectF())
        self.component_id = component_id
        self.status = status
        pen = QPen(status_color(status))
        pen.setCosmetic(True)
        pen.setWidth(2)
        self.setPen(pen)
        self.setZValue(20)


class AnnotationLeaderLine(QGraphicsLineItem):
    """Leader from a component to its bubble. Not placed in this slice."""

    def __init__(self, component_id: str):
        super().__init__()
        self.component_id = component_id
        pen = QPen(QColor("#d7e2c8"))
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setZValue(30)


class AnnotationBubble(QGraphicsTextItem):
    """Shows name, value, and package. Click-to-edit is not implemented."""

    def __init__(self, component_id: str, core_params: dict | None = None):
        params = {key: (core_params or {}).get(key, "") for key in CORE_PARAM_KEYS}
        super().__init__(bubble_text(params))
        self.component_id = component_id
        self.core_params = params
        self.setDefaultTextColor(QColor("#f4f7fb"))
        self.setZValue(40)
