"""Entry point. From the repository root::

    PYTHONPATH=src python -m app.main

``--smoke`` opens the window off the caller's Qt platform, waits for one
frame, and exits. It does not package or install anything.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src = Path(__file__).resolve().parents[1]
    entry = str(src)
    if entry not in sys.path:
        sys.path.insert(0, entry)


_bootstrap_src_path()

from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from app.ui.main_window import MainWindow  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCB AR 标注壳（本轮无打包、无跟踪算法）")
    parser.add_argument(
        "--source",
        default=None,
        help="摄像头索引、视频路径，或 synthetic。默认：普通启动为 0，--smoke 为 synthetic",
    )
    parser.add_argument("--width", type=int, default=None, help="请求宽度；合成帧默认 960")
    parser.add_argument("--height", type=int, default=None, help="请求高度；合成帧默认 540")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="收到第一帧后退出。无显示器时配合 QT_QPA_PLATFORM=offscreen",
    )
    return parser.parse_args(argv)


def _parse_source(text: str | None, *, smoke: bool) -> int | str:
    if text is None:
        return "synthetic" if smoke else 0
    if text == "synthetic":
        return "synthetic"
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    source = _parse_source(args.source, smoke=args.smoke)

    app = QApplication([sys.argv[0]])
    app.setApplicationName("PCB AR 标注")
    window = MainWindow(
        source=source,
        width=args.width,
        height=args.height,
        smoke=args.smoke,
    )
    app.aboutToQuit.connect(window.shutdown)
    # Let Python handle Ctrl+C while the Qt loop is running.
    waker = QTimer()
    waker.start(200)
    waker.timeout.connect(lambda: None)
    app._pcb_waker = waker  # type: ignore[attr-defined]
    signal.signal(signal.SIGINT, lambda *_args: app.quit())

    window.show()
    code = app.exec()
    if args.smoke:
        if window.smoke_ok:
            pixmap = window.canvas.video_item.pixmap()
            print(
                f"smoke ok: {pixmap.width()}x{pixmap.height()} mode={window.capture.mode}",
                flush=True,
            )
            return 0
        print("smoke failed: no frame", file=sys.stderr, flush=True)
        return 1
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
