"""The window shows a video pixmap, an empty overlay, and the right dock."""

import time

from PyQt6.QtWidgets import QDockWidget

from app.ui.annotation_items import AnnotationBubble, AnnotationViewModel
from app.ui.canvas import PCBCanvasView
from app.ui.main_window import MainWindow


def _wait(qapp, predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_canvas_video_layer_stays_above_empty_annotations(qapp):
    del qapp
    view = PCBCanvasView()
    import numpy as np

    view.set_video_frame(np.zeros((40, 60, 3), dtype=np.uint8))
    view.set_annotations(
        [AnnotationViewModel(component_id="R1", status="unchecked", core_params={"name": "R1"})]
    )
    assert view.video_item.pixmap().width() == 60
    assert view.video_item.zValue() < view.annotation_layer.zValue()
    assert len(view.annotation_layer.childItems()) == 0
    assert view.annotations[0].component_id == "R1"
    view.set_tracking_ok(False)
    assert view.tracking_ok is False


def test_bubble_shows_at_most_three_core_lines():
    bubble = AnnotationBubble(
        "R1",
        {"name": "R1", "value": "10K", "package": "0805", "extra": "nope"},
    )
    assert bubble.toPlainText().splitlines() == [
        "name: R1",
        "value: 10K",
        "package: 0805",
    ]


def test_main_window_shell(qapp):
    window = MainWindow(source="synthetic", width=160, height=120)
    window.show()
    try:
        assert _wait(qapp, lambda: not window.canvas.video_item.pixmap().isNull())
        assert window.windowTitle().startswith("PCB AR 标注")
        assert len(window.canvas.annotation_layer.childItems()) == 0
        docks = window.findChildren(QDockWidget)
        assert len(docks) == 1
        assert docks[0].windowTitle() == "元件参数"
        assert window.store.project is not None
        assert window.store.project.board_id == "DEMO-PCB-001"
        assert window.canvas.video_item.pixmap().width() == 160
    finally:
        window.close()
        qapp.processEvents()
        assert not window.capture.isRunning()
