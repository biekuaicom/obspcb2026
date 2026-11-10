"""Compatibility wrapper. The spec entry is ``vision.mosaic.VideoMosaicBuilder``.

``python -m vision.video_mosaic`` still runs a synthetic mosaic for older
commands. New work should call ``python -m vision.mosaic``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from vision.mosaic import (
    MosaicFrame,
    VideoMosaicBuilder,
    make_synthetic_translation_frames,
)

__all__ = [
    "FrameStatus",
    "VideoMosaicMapper",
    "VideoMosaicResult",
    "make_synthetic_translation_frames",
    "run_video_mosaic",
]


@dataclass
class FrameStatus:
    index: int
    accepted: bool
    reason: str
    inliers: int = 0
    matches: int = 0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "accepted": self.accepted,
            "reason": self.reason,
            "inliers": self.inliers,
            "matches": self.matches,
        }


@dataclass
class VideoMosaicResult:
    ok: bool
    mosaic_path: Path | None
    frames_total: int
    frames_accepted: int
    frames_skipped: int
    canvas_size: tuple[int, int]
    coverage_hint: float | None
    message: str
    frames: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "mosaic": None if self.mosaic_path is None else str(self.mosaic_path),
            "frames_total": self.frames_total,
            "frames_accepted": self.frames_accepted,
            "frames_skipped": self.frames_skipped,
            "canvas_size": [self.canvas_size[0], self.canvas_size[1]],
            "coverage_hint": self.coverage_hint,
            "message": self.message,
            "frames": self.frames,
            "mapping_mode": "video_mosaic",
        }


def _status_from(frame: MosaicFrame) -> FrameStatus:
    return FrameStatus(frame.index, frame.accepted, frame.reason, frame.inliers, frame.matches)


class VideoMosaicMapper:
    """Old name for VideoMosaicBuilder. Prefer the builder in vision.mosaic."""

    def __init__(
        self,
        *,
        detector: str = "sift",
        min_inliers: int = 24,
        min_inlier_ratio: float = 0.35,
        max_step_fraction: float = 0.45,
        reanchor_every: int = 8,
        max_canvas: int = 4096,
        margin: int = 48,
    ):
        if detector not in {"sift", "orb"}:
            raise ValueError("detector must be 'sift' or 'orb'")
        self._builder = VideoMosaicBuilder(overlap_hint=0.85, motion="feature")
        self._builder.min_inliers = int(min_inliers)
        self._builder.min_inlier_ratio = float(min_inlier_ratio)
        self._builder.max_step_fraction = float(max_step_fraction)
        self._builder.reanchor_every = max(1, int(reanchor_every))
        self._builder.max_canvas = int(max_canvas)
        self._builder.margin = int(margin)
        if detector == "orb":
            self._builder._detector = cv2.ORB_create(nfeatures=2000)
            self._builder._norm = cv2.NORM_HAMMING
        self.statuses: list[FrameStatus] = []

    def add_frame(self, frame_bgr: np.ndarray) -> FrameStatus:
        status = _status_from(self._builder.push_frame(frame_bgr))
        self.statuses.append(status)
        return status

    def render(self) -> np.ndarray | None:
        return self._builder.render()

    def coverage_hint(self) -> float | None:
        return self._builder.coverage_hint()

    def canvas_size(self) -> tuple[int, int]:
        return self._builder.canvas_size()

    def run_frames(self, frames: list[np.ndarray]) -> VideoMosaicResult:
        for frame in frames:
            self.add_frame(frame)
        accepted = sum(1 for item in self.statuses if item.accepted)
        skipped = len(self.statuses) - accepted
        ok = accepted >= 2 and skipped < accepted
        return VideoMosaicResult(
            ok=ok,
            mosaic_path=None,
            frames_total=len(self.statuses),
            frames_accepted=accepted,
            frames_skipped=skipped,
            canvas_size=self.canvas_size(),
            coverage_hint=self.coverage_hint(),
            message=(
                f"video mosaic accepted {accepted}/{len(self.statuses)} frames"
                if ok
                else f"video mosaic too few accepted frames ({accepted}/{len(self.statuses)})"
            ),
            frames=[item.to_dict() for item in self.statuses],
        )


def run_video_mosaic(frames: list[np.ndarray], out_dir: Path, *, detector: str = "sift") -> VideoMosaicResult:
    mapper = VideoMosaicMapper(detector=detector)
    result = mapper.run_frames(frames)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mosaic = mapper.render()
    mosaic_path = out_dir / "mosaic.png"
    if mosaic is not None:
        cv2.imwrite(str(mosaic_path), mosaic)
        result.mosaic_path = mosaic_path
    (out_dir / "summary.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="旧入口。请改用 python -m vision.mosaic")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("artifacts/video_mosaic"))
    parser.add_argument("--detector", choices=("sift", "orb"), default="sift")
    args = parser.parse_args(argv)
    if args.video is not None and not args.synthetic:
        capture = cv2.VideoCapture(str(args.video))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        if not frames:
            print(f"无法读取视频 {args.video}")
            return 1
    else:
        frames = make_synthetic_translation_frames(count=max(3, args.frames))
    result = run_video_mosaic(frames, args.out, detector=args.detector)
    print(json.dumps(result.to_dict(), ensure_ascii=False))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
