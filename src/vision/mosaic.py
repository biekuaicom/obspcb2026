"""Mapping Mode (D15 / TECH_SPEC §2.0, §3.6).

``VideoMosaicBuilder`` is the public entry. Still-tile compare stays in
``vision.stitch`` (§3.7) and is not the default mapping path.

Each accepted frame has a 3×3 homography from that frame into the mosaic
canvas. Motion is a partial affine (feature match or optical flow), lifted
to 3×3. Weak or degenerate steps are skipped. The first frame sits at the
canvas center; the canvas grows until ``max_canvas``.

Worker thread: matching and blending happen inside ``push_frame`` and
``finalize``. This CLI calls them synchronously because it has no Qt UI
thread. A desktop Mapping page must run the builder off the GUI thread.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from model.schema import BoardProject

_MIN_INLIERS = 24
_MIN_INLIER_RATIO = 0.35
_MIN_SCALE = 0.92
_MAX_SCALE = 1.08
_MAX_CHAIN_SCALE = 1.35
_MIN_CHAIN_SCALE_FLOOR = 1.0 / _MAX_CHAIN_SCALE
MAP_SOURCE = "video_mosaic"


class MappingState(str, Enum):
    """§2.0 mapping states. Values are the names written into meta.mapping."""

    IDLE = "Idle"
    RECORDING = "Mapping/Recording"
    BUILDING = "Building/Merging"
    MOSAIC_READY = "MosaicReady"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    RESCAN = "Rescan"


class MappingController:
    """Minimal mapping state machine for CLI and smoke. UI wiring is a stub."""

    def __init__(self) -> None:
        self.state = MappingState.IDLE
        self.message = ""

    def start_recording(self) -> MappingState:
        if self.state not in {MappingState.IDLE, MappingState.RESCAN}:
            raise RuntimeError(f"cannot record from {self.state.value}")
        self.state = MappingState.RECORDING
        self.message = ""
        return self.state

    def begin_building(self) -> MappingState:
        if self.state != MappingState.RECORDING:
            raise RuntimeError(f"cannot build from {self.state.value}")
        self.state = MappingState.BUILDING
        return self.state

    def mark_ready(self) -> MappingState:
        if self.state != MappingState.BUILDING:
            raise RuntimeError(f"cannot mark ready from {self.state.value}")
        self.state = MappingState.MOSAIC_READY
        return self.state

    def fail(self, message: str = "") -> MappingState:
        if self.state not in {MappingState.RECORDING, MappingState.BUILDING}:
            raise RuntimeError(f"cannot fail from {self.state.value}")
        self.message = message
        self.state = MappingState.FAILED
        return self.state

    def cancel(self) -> MappingState:
        if self.state == MappingState.IDLE:
            raise RuntimeError("cannot cancel from Idle")
        self.state = MappingState.CANCELLED
        self.message = "cancelled"
        return self.state

    def request_rescan(self) -> MappingState:
        if self.state not in {MappingState.MOSAIC_READY, MappingState.FAILED, MappingState.CANCELLED}:
            raise RuntimeError(f"cannot rescan from {self.state.value}")
        self.state = MappingState.RESCAN
        self.message = ""
        return self.state

    def reset(self) -> MappingState:
        self.state = MappingState.IDLE
        self.message = ""
        return self.state


@dataclass
class MosaicFrame:
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
class MosaicBuildProgress:
    state: str
    frames_total: int
    frames_accepted: int
    frames_skipped: int
    canvas_size: tuple[int, int]
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "frames_total": self.frames_total,
            "frames_accepted": self.frames_accepted,
            "frames_used": self.frames_accepted,
            "frames_skipped": self.frames_skipped,
            "canvas_size": [self.canvas_size[0], self.canvas_size[1]],
            "message": self.message,
        }


@dataclass
class MosaicResult:
    ok: bool
    mosaic_path: Path | None
    frames_total: int
    frames_accepted: int
    frames_skipped: int
    canvas_size: tuple[int, int]
    coverage_hint: float | None
    message: str
    frames: list[MosaicFrame] = field(default_factory=list)
    map_source: str = MAP_SOURCE
    global_map_path: str | None = None
    project_path: Path | None = None
    state: str = MappingState.IDLE.value
    scan: dict | None = None
    compare: dict | None = None

    def to_dict(self) -> dict:
        payload = {
            "ok": self.ok,
            "mosaic": None if self.mosaic_path is None else str(self.mosaic_path),
            "frames_total": self.frames_total,
            "frames_accepted": self.frames_accepted,
            "frames_used": self.frames_accepted,
            "frames_skipped": self.frames_skipped,
            "canvas_size": [self.canvas_size[0], self.canvas_size[1]],
            "coverage_hint": self.coverage_hint,
            "message": self.message,
            "frames": [item.to_dict() for item in self.frames],
            "map_source": self.map_source,
            "global_map_path": self.global_map_path,
            "project": None if self.project_path is None else str(self.project_path),
            "state": self.state,
        }
        if self.scan is not None:
            payload["scan"] = self.scan
        if self.compare is not None:
            payload["compare"] = self.compare
        return payload


class VideoMosaicBuilder:
    """Incremental video mosaic. ``motion`` is feature, optical_flow, or auto."""

    def __init__(self, overlap_hint: float = 0.85, motion: str = "feature"):
        if motion not in {"feature", "optical_flow", "auto"}:
            raise ValueError("motion must be 'feature', 'optical_flow', or 'auto'")
        if not 0.0 < float(overlap_hint) < 1.0:
            raise ValueError("overlap_hint must be between 0 and 1")
        self.overlap_hint = float(overlap_hint)
        self.motion = motion
        # 0.85 overlap leaves ~15% new pixels; a small slack keeps a 16% step.
        self.max_step_fraction = min(0.45, max(0.05, (1.0 - self.overlap_hint) + 0.08))
        self.min_inliers = _MIN_INLIERS
        self.min_inlier_ratio = _MIN_INLIER_RATIO
        self.reanchor_every = 8
        self.max_canvas = 4096
        self.margin = 48
        self.controller = MappingController()
        self.frames: list[MosaicFrame] = []
        self._detector = cv2.SIFT_create(nfeatures=2000)
        self._norm = cv2.NORM_L2
        self._accum: np.ndarray | None = None
        self._weight: np.ndarray | None = None
        self._H_ref: np.ndarray | None = None
        self._ref_pack: tuple | None = None
        self._key_pack: tuple | None = None
        self._H_key: np.ndarray | None = None
        self._since_key = 0
        self._prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self.frames.clear()
        self._accum = None
        self._weight = None
        self._H_ref = None
        self._ref_pack = None
        self._key_pack = None
        self._H_key = None
        self._since_key = 0
        self._prev_gray = None
        self.controller.reset()

    def push_frame(self, frame_bgr: np.ndarray) -> MosaicFrame:
        if self.controller.state == MappingState.CANCELLED:
            return self._record(False, "cancelled")
        if self.controller.state in {MappingState.IDLE, MappingState.RESCAN}:
            self.controller.start_recording()
        if self.controller.state != MappingState.RECORDING:
            return self._record(False, f"not recording ({self.controller.state.value})")
        if frame_bgr is None or getattr(frame_bgr, "ndim", 0) != 3 or frame_bgr.size == 0:
            return self._record(False, "empty frame")
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        pack = self._detect(gray)
        if self._accum is None:
            self._start(frame_bgr, pack, gray)
            return self._record(True, "first frame")
        relative, stats, reason = self._estimate_motion(pack, gray, frame_bgr.shape)
        anchor = "previous"
        if relative is None and self.motion != "optical_flow" and self._key_pack is not None and self._H_key is not None:
            relative, stats, reason = self._match_features(pack, self._key_pack, frame_bgr.shape)
            anchor = "keyframe"
        if relative is None:
            self._prev_gray = gray
            return self._record(False, reason, stats[0], stats[1])
        if anchor == "keyframe":
            tentative = self._normalize(self._H_key @ relative)
        else:
            tentative = self._normalize(self._H_ref @ relative)
        if not self._chain_scale_ok(tentative):
            self._prev_gray = gray
            return self._record(False, "cumulative scale drift", stats[0], stats[1])
        corners = _project_corners(frame_bgr.shape, tentative)
        if not self._fit_canvas(corners):
            self._prev_gray = gray
            return self._record(False, "canvas limit", stats[0], stats[1])
        homography = tentative
        self._blend(frame_bgr, homography)
        self._H_ref = homography
        self._ref_pack = pack
        self._prev_gray = gray
        self._since_key += 1
        if self._since_key >= self.reanchor_every:
            self._key_pack = pack
            self._H_key = homography
            self._since_key = 0
        return self._record(True, f"accepted via {anchor}", stats[0], stats[1])

    def progress(self) -> MosaicBuildProgress:
        accepted = sum(1 for item in self.frames if item.accepted)
        return MosaicBuildProgress(
            state=self.controller.state.value,
            frames_total=len(self.frames),
            frames_accepted=accepted,
            frames_skipped=len(self.frames) - accepted,
            canvas_size=self.canvas_size(),
            message=self.controller.message,
        )

    def finalize(
        self,
        out_path: str | Path,
        board_id: str | None = None,
        repo_root: str | Path | None = None,
        mapping_extra: dict | None = None,
    ) -> MosaicResult:
        """Write the mosaic. Optional board_id updates the project JSON."""
        if self.controller.state == MappingState.CANCELLED:
            return self._result(False, None, "cancelled", MappingState.CANCELLED.value)
        if self.controller.state == MappingState.RECORDING:
            self.controller.begin_building()
        elif self.controller.state != MappingState.BUILDING:
            self.controller.fail("finalize outside recording")
            return self._result(False, None, self.controller.message, self.controller.state.value)
        image = self.render()
        accepted = sum(1 for item in self.frames if item.accepted)
        ok = image is not None and accepted >= 2
        mosaic_path = Path(out_path)
        if ok:
            mosaic_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(mosaic_path), image):
                self.controller.fail("could not write mosaic")
                return self._result(False, None, self.controller.message, self.controller.state.value)
        message = (
            f"video mosaic accepted {accepted}/{len(self.frames)} frames"
            if ok
            else f"video mosaic too few accepted frames ({accepted}/{len(self.frames)})"
        )
        if ok:
            self.controller.mark_ready()
        else:
            self.controller.fail(message)
        result = self._result(ok, mosaic_path if ok else None, message, self.controller.state.value)
        if mapping_extra:
            result.scan = mapping_extra.get("scan")
            result.compare = mapping_extra.get("compare")
        if ok and board_id and repo_root is not None:
            project_path, relative = write_mapping_project(
                Path(repo_root),
                board_id,
                mosaic_path,
                result,
                overlap_hint=self.overlap_hint,
                motion=self.motion,
                mapping_extra=mapping_extra,
            )
            result.project_path = project_path
            result.global_map_path = relative
        return result

    def cancel(self) -> MappingState:
        if self.controller.state == MappingState.IDLE:
            self.controller.state = MappingState.CANCELLED
            self.controller.message = "cancelled"
            return self.controller.state
        self.controller.cancel()
        return self.controller.state

    def render(self) -> np.ndarray | None:
        if self._accum is None or self._weight is None:
            return None
        mosaic = np.zeros(self._accum.shape, dtype=np.uint8)
        valid = self._weight > 1e-6
        mosaic[valid] = np.clip(self._accum[valid] / self._weight[valid, None], 0, 255).astype(np.uint8)
        return mosaic

    def coverage_hint(self) -> float | None:
        if self._weight is None or self._weight.size == 0:
            return None
        return float(np.count_nonzero(self._weight > 1e-6) / self._weight.size)

    def canvas_size(self) -> tuple[int, int]:
        if self._accum is None:
            return (0, 0)
        height, width = self._accum.shape[:2]
        return (width, height)

    def _record(self, accepted: bool, reason: str, inliers: int = 0, matches: int = 0) -> MosaicFrame:
        frame = MosaicFrame(len(self.frames), accepted, reason, inliers, matches)
        self.frames.append(frame)
        return frame

    def _result(self, ok: bool, mosaic_path: Path | None, message: str, state: str) -> MosaicResult:
        accepted = sum(1 for item in self.frames if item.accepted)
        return MosaicResult(
            ok=ok,
            mosaic_path=mosaic_path,
            frames_total=len(self.frames),
            frames_accepted=accepted,
            frames_skipped=len(self.frames) - accepted,
            canvas_size=self.canvas_size(),
            coverage_hint=self.coverage_hint(),
            message=message,
            frames=list(self.frames),
            state=state,
        )

    def _start(self, frame_bgr: np.ndarray, pack: tuple, gray: np.ndarray) -> None:
        frame_h, frame_w = frame_bgr.shape[:2]
        canvas_h = min(self.max_canvas, frame_h + 2 * max(self.margin, frame_h))
        canvas_w = min(self.max_canvas, frame_w + 2 * max(self.margin, frame_w))
        self._accum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
        self._weight = np.zeros((canvas_h, canvas_w), dtype=np.float64)
        origin_x = (canvas_w - frame_w) / 2.0
        origin_y = (canvas_h - frame_h) / 2.0
        homography = np.array(
            [[1.0, 0.0, origin_x], [0.0, 1.0, origin_y], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self._blend(frame_bgr, homography)
        self._H_ref = homography
        self._ref_pack = pack
        self._key_pack = pack
        self._H_key = homography
        self._since_key = 0
        self._prev_gray = gray

    def _detect(self, gray: np.ndarray) -> tuple:
        small, scale = _downscale_gray(gray, 720)
        keys, desc = self._detector.detectAndCompute(small, None)
        return small, scale, keys, desc

    def _estimate_motion(self, pack: tuple, gray: np.ndarray, frame_shape: tuple):
        if self.motion == "optical_flow":
            return self._match_flow(gray, frame_shape)
        relative, stats, reason = self._match_features(pack, self._ref_pack, frame_shape)
        if relative is not None or self.motion != "auto":
            return relative, stats, reason
        if reason not in {"too few features", "too few matches", "too few inliers"}:
            return relative, stats, reason
        flow, flow_stats, flow_reason = self._match_flow(gray, frame_shape)
        if flow is not None:
            return flow, flow_stats, flow_reason
        return relative, stats, reason

    def _match_features(self, current: tuple, reference: tuple | None, frame_shape: tuple):
        if reference is None:
            return None, (0, 0), "no reference"
        _cur_img, cur_scale, cur_keys, cur_desc = current
        _ref_img, ref_scale, ref_keys, ref_desc = reference
        if cur_desc is None or ref_desc is None or len(cur_keys) < 8 or len(ref_desc) < 2:
            return None, (0, 0), "too few features"
        if abs(cur_scale - ref_scale) > 1e-6:
            return None, (0, 0), "feature scale mismatch"
        matcher = cv2.BFMatcher(self._norm)
        pairs = matcher.knnMatch(cur_desc, ref_desc, k=2)
        good = []
        for pair in pairs:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < 0.75 * second.distance:
                good.append(first)
        if len(good) < self.min_inliers:
            return None, (0, len(good)), "too few matches"
        src = np.float32([cur_keys[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
        dst = np.float32([ref_keys[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
        return self._accept_affine(src, dst, len(good), cur_scale, frame_shape)

    def _match_flow(self, gray: np.ndarray, frame_shape: tuple):
        if self._prev_gray is None:
            return None, (0, 0), "no reference"
        prev_small, scale = _downscale_gray(self._prev_gray, 720)
        cur_small, cur_scale = _downscale_gray(gray, 720)
        if abs(scale - cur_scale) > 1e-6:
            return None, (0, 0), "feature scale mismatch"
        points = cv2.goodFeaturesToTrack(prev_small, maxCorners=500, qualityLevel=0.01, minDistance=6)
        if points is None or len(points) < self.min_inliers:
            return None, (0, 0 if points is None else len(points)), "too few features"
        nxt, status, _err = cv2.calcOpticalFlowPyrLK(prev_small, cur_small, points, None)
        if nxt is None or status is None:
            return None, (0, len(points)), "too few matches"
        keep = status.reshape(-1) == 1
        if int(keep.sum()) < self.min_inliers:
            return None, (int(keep.sum()), len(points)), "too few matches"
        src = nxt[keep].reshape(-1, 1, 2)
        dst = points[keep].reshape(-1, 1, 2)
        return self._accept_affine(src, dst, int(keep.sum()), scale, frame_shape)

    def _accept_affine(self, src, dst, matches: int, scale: float, frame_shape: tuple):
        affine, mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.995,
        )
        inliers = int(mask.sum()) if mask is not None else 0
        if affine is None or inliers < self.min_inliers:
            return None, (inliers, matches), "too few inliers"
        if inliers / max(matches, 1) < self.min_inlier_ratio:
            return None, (inliers, matches), "inlier ratio low"
        lifted = affine.copy()
        lifted[0, 2] /= scale
        lifted[1, 2] /= scale
        ok, reason = _relative_is_stable(lifted, frame_shape, self.max_step_fraction)
        if not ok:
            return None, (inliers, matches), reason
        homography = np.vstack([lifted, [0.0, 0.0, 1.0]])
        return self._normalize(homography), (inliers, matches), "ok"

    def _chain_scale_ok(self, homography: np.ndarray) -> bool:
        scale, _step, degenerate = _homography_metrics(homography)
        if degenerate:
            return False
        return _MIN_CHAIN_SCALE_FLOOR <= scale <= _MAX_CHAIN_SCALE

    def _fit_canvas(self, corners: np.ndarray) -> bool:
        assert self._accum is not None
        height, width = self._accum.shape[:2]
        min_x, min_y = corners.min(axis=0)
        max_x, max_y = corners.max(axis=0)
        left = max(0, int(math.ceil(self.margin - min_x)))
        top = max(0, int(math.ceil(self.margin - min_y)))
        right = max(0, int(math.ceil(max_x + self.margin - (width - 1))))
        bottom = max(0, int(math.ceil(max_y + self.margin - (height - 1))))
        if width + left + right > self.max_canvas or height + top + bottom > self.max_canvas:
            return False
        if left or top or right or bottom:
            self._expand(left, top, right, bottom)
        return True

    def _expand(self, left: int, top: int, right: int, bottom: int) -> None:
        assert self._accum is not None and self._weight is not None
        height, width = self._accum.shape[:2]
        new_accum = np.zeros((height + top + bottom, width + left + right, 3), dtype=np.float64)
        new_weight = np.zeros((height + top + bottom, width + left + right), dtype=np.float64)
        new_accum[top : top + height, left : left + width] = self._accum
        new_weight[top : top + height, left : left + width] = self._weight
        self._accum = new_accum
        self._weight = new_weight
        shift = np.array([[1.0, 0.0, left], [0.0, 1.0, top], [0.0, 0.0, 1.0]], dtype=np.float64)
        self._H_ref = self._normalize(shift @ self._H_ref)
        if self._H_key is not None:
            self._H_key = self._normalize(shift @ self._H_key)

    def _blend(self, frame_bgr: np.ndarray, homography: np.ndarray) -> None:
        assert self._accum is not None and self._weight is not None
        height, width = self._accum.shape[:2]
        warped = cv2.warpPerspective(frame_bgr, homography, (width, height))
        mask = cv2.warpPerspective(np.full(frame_bgr.shape[:2], 255, np.uint8), homography, (width, height))
        inside = mask > 0
        if not np.any(inside):
            return
        distance = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 3)
        feather = np.clip(distance, 0.0, 24.0) / 24.0
        self._accum += warped.astype(np.float64) * feather[..., None]
        self._weight += feather

    @staticmethod
    def _normalize(homography: np.ndarray) -> np.ndarray:
        matrix = np.asarray(homography, dtype=np.float64).copy()
        if abs(matrix[2, 2]) < 1e-12:
            return matrix
        matrix /= matrix[2, 2]
        return matrix


def write_mapping_project(
    repo_root: Path,
    board_id: str,
    mosaic_path: Path,
    result: MosaicResult,
    *,
    overlap_hint: float,
    motion: str,
    mapping_extra: dict | None = None,
) -> tuple[Path, str]:
    """Update board JSON: global_map_path, map_source, meta.mapping (§2.6)."""
    repo_root = Path(repo_root)
    project_path = repo_root / "assets" / "projects" / f"{board_id}.json"
    project_path.parent.mkdir(parents=True, exist_ok=True)
    relative = Path(os.path.relpath(Path(mosaic_path).resolve(), project_path.parent.resolve())).as_posix()
    mapping = {
        "mode": MAP_SOURCE,
        "state": result.state,
        "frames_total": result.frames_total,
        "frames_used": result.frames_accepted,
        "frames_skipped": result.frames_skipped,
        "canvas_size": [result.canvas_size[0], result.canvas_size[1]],
        "coverage_hint": result.coverage_hint,
        "overlap_hint": overlap_hint,
        "motion": motion,
    }
    if mapping_extra:
        for key, value in mapping_extra.items():
            if value is not None:
                mapping[key] = value
    if project_path.is_file():
        project = BoardProject.from_json(project_path.read_text(encoding="utf-8"))
        project.global_map_path = relative
        project.map_source = MAP_SOURCE
        meta = dict(project.meta or {})
        meta["mapping"] = mapping
        project.meta = meta
    else:
        project = BoardProject(
            board_id=board_id,
            global_map_path=relative,
            components=[],
            name=board_id,
            map_source=MAP_SOURCE,
            meta={"mapping": mapping},
        )
    project_path.write_text(project.to_json() + "\n", encoding="utf-8")
    return project_path, relative


@dataclass
class SerpentineScan:
    """Crop plan for a microscope-style serpentine pass over one global image."""

    window_size: tuple[int, int]
    step: tuple[int, int]
    overlap_x: float
    overlap_y: float
    rows: int
    cols: int
    pattern: str
    origins: list[tuple[int, int]]
    source: str | None = None

    @property
    def frame_count(self) -> int:
        return len(self.origins)

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "source": self.source,
            "frame_count": self.frame_count,
            "window_size": [self.window_size[0], self.window_size[1]],
            "step": {"dx": self.step[0], "dy": self.step[1]},
            "overlap_x": self.overlap_x,
            "overlap_y": self.overlap_y,
            "rows": self.rows,
            "cols": self.cols,
        }


def plan_serpentine_scan(
    width: int,
    height: int,
    overlap: float = 0.85,
    window_size: tuple[int, int] | None = None,
    source: str | None = None,
) -> SerpentineScan:
    """Plan a left-to-right, then right-to-left grid with about ``overlap`` shared pixels.

    The step is also capped so a single move stays inside VideoMosaicBuilder's
    motion gate (about 0.23 of the short window side when overlap_hint is 0.85).
    """
    if not 0.5 <= float(overlap) < 1.0:
        raise ValueError("overlap must be in [0.5, 1)")
    if width < 8 or height < 8:
        raise ValueError("image is too small to scan")
    if window_size is None:
        window_w = min(width, max(32, int(round(width * 0.52))))
        window_h = min(height, max(32, int(round(height * 0.38))))
    else:
        window_w = min(width, max(8, int(window_size[0])))
        window_h = min(height, max(8, int(window_size[1])))
    gap = 1.0 - float(overlap)
    dx = max(1, int(round(window_w * gap)))
    dy = max(1, int(round(window_h * gap)))
    limit = max(1, int(math.floor(0.20 * min(window_w, window_h))))
    dx = min(dx, limit)
    dy = min(dy, limit)
    xs = _axis_origins(width, window_w, dx)
    ys = _axis_origins(height, window_h, dy)
    origins: list[tuple[int, int]] = []
    for row, y in enumerate(ys):
        row_xs = xs if row % 2 == 0 else list(reversed(xs))
        for x in row_xs:
            origins.append((int(x), int(y)))
    return SerpentineScan(
        window_size=(window_w, window_h),
        step=(dx, dy),
        overlap_x=1.0 - dx / float(window_w),
        overlap_y=1.0 - dy / float(window_h),
        rows=len(ys),
        cols=len(xs),
        pattern="serpentine",
        origins=origins,
        source=source,
    )


def crop_serpentine_frames(image: np.ndarray, scan: SerpentineScan) -> list[np.ndarray]:
    window_w, window_h = scan.window_size
    frames: list[np.ndarray] = []
    for x, y in scan.origins:
        frames.append(image[y : y + window_h, x : x + window_w].copy())
    return frames


def _axis_origins(length: int, window: int, step: int) -> list[int]:
    if window >= length:
        return [0]
    origins = [0]
    pos = 0
    while pos + window < length:
        nxt = min(pos + step, length - window)
        if nxt <= pos:
            break
        origins.append(int(nxt))
        pos = nxt
    return origins


def compare_mosaic_to_global(mosaic: np.ndarray, global_bgr: np.ndarray) -> tuple[dict, np.ndarray | None]:
    """Align a rebuilt mosaic onto the source global image and score the overlap.

    One homography cannot undo a bent chain. A high NCC means the scan stayed rigid.
    """
    cropped = _content_crop(mosaic)
    if cropped is None:
        return {
            "ok": False,
            "ncc": None,
            "coverage": 0.0,
            "mean_abs": None,
            "inliers": 0,
            "matches": 0,
            "notes": "mosaic has no content",
        }, None
    homography, inliers, matches = _match_mosaic_homography(cropped, global_bgr)
    if homography is None:
        return {
            "ok": False,
            "ncc": None,
            "coverage": 0.0,
            "mean_abs": None,
            "inliers": inliers,
            "matches": matches,
            "notes": "could not align mosaic to global_ref",
        }, None
    aligned = cv2.warpPerspective(cropped, homography, (global_bgr.shape[1], global_bgr.shape[0]))
    mask = aligned.sum(axis=2) > 8
    coverage = float(mask.mean()) if mask.size else 0.0
    ncc = _masked_ncc(aligned, global_bgr, mask)
    mean_abs = None
    if int(mask.sum()) > 0:
        mean_abs = float(np.mean(cv2.absdiff(aligned, global_bgr)[mask]))
    ok = ncc is not None and ncc >= 0.5 and coverage >= 0.35
    notes = (
        f"inliers={inliers} matches={matches} ncc={None if ncc is None else round(ncc, 3)} "
        f"coverage={round(coverage, 3)}"
    )
    return {
        "ok": ok,
        "ncc": ncc,
        "coverage": coverage,
        "mean_abs": mean_abs,
        "inliers": inliers,
        "matches": matches,
        "notes": notes,
    }, aligned


def write_global_compare_images(
    global_bgr: np.ndarray,
    aligned: np.ndarray | None,
    artifact_dir: Path,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if aligned is None:
        return
    mask = aligned.sum(axis=2) > 8
    overlay = global_bgr.copy()
    blended = cv2.addWeighted(global_bgr, 0.55, aligned, 0.45, 0.0)
    overlay[mask] = blended[mask]
    diff = np.zeros_like(global_bgr)
    diff[mask] = cv2.absdiff(global_bgr, aligned)[mask]
    cv2.imwrite(str(artifact_dir / "aligned.png"), aligned)
    cv2.imwrite(str(artifact_dir / "overlay.png"), overlay)
    cv2.imwrite(str(artifact_dir / "diff.png"), diff)


def _content_crop(image: np.ndarray) -> np.ndarray | None:
    mask = image.sum(axis=2) > 8
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return image[y0:y1, x0:x1].copy()


def _match_mosaic_homography(mosaic: np.ndarray, global_bgr: np.ndarray) -> tuple[np.ndarray | None, int, int]:
    mosaic_gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    global_gray = cv2.cvtColor(global_bgr, cv2.COLOR_BGR2GRAY)
    mosaic_small, mosaic_scale = _downscale_gray(mosaic_gray, 900)
    global_small, global_scale = _downscale_gray(global_gray, 900)
    detector = cv2.SIFT_create(nfeatures=3000)
    keys_src, desc_src = detector.detectAndCompute(mosaic_small, None)
    keys_dst, desc_dst = detector.detectAndCompute(global_small, None)
    if desc_src is None or desc_dst is None or len(keys_src) < 8 or len(keys_dst) < 8:
        return None, 0, 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_src, desc_dst, k=2)
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good.append(first)
    if len(good) < 12:
        return None, 0, len(good)
    src = np.float32([keys_src[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
    dst = np.float32([keys_dst[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if homography is None or inliers < 12:
        return None, inliers, len(good)
    lifted = _lift_homography(homography, mosaic_scale, global_scale)
    return lifted, inliers, len(good)


def _lift_homography(homography: np.ndarray, src_scale: float, dst_scale: float) -> np.ndarray:
    src_to_full = np.array([[1.0 / src_scale, 0.0, 0.0], [0.0, 1.0 / src_scale, 0.0], [0.0, 0.0, 1.0]])
    full_to_dst = np.array([[dst_scale, 0.0, 0.0], [0.0, dst_scale, 0.0], [0.0, 0.0, 1.0]])
    lifted = np.linalg.inv(full_to_dst) @ homography @ np.linalg.inv(src_to_full)
    if abs(lifted[2, 2]) > 1e-12:
        lifted = lifted / lifted[2, 2]
    return lifted


def _masked_ncc(aligned: np.ndarray, global_bgr: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) < 50:
        return None
    left = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)[mask].astype(np.float64)
    right = cv2.cvtColor(global_bgr, cv2.COLOR_BGR2GRAY)[mask].astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom < 1e-6:
        return None
    return float(np.dot(left, right) / denom)


def make_synthetic_translation_frames(
    *,
    count: int = 8,
    frame_size: tuple[int, int] = (140, 200),
    step: tuple[int, int] = (12, 3),
    bad_index: int | None = 3,
) -> list[np.ndarray]:
    """Slide a window across a textured board. One frame can be noise."""
    frame_h, frame_w = frame_size
    world_h = frame_h + step[1] * count + 80
    world_w = frame_w + step[0] * count + 80
    world = np.zeros((world_h, world_w, 3), dtype=np.uint8)
    world[:, :] = (32, 110, 46)
    rng = np.random.default_rng(3)
    for y in range(8, world_h - 8, 16):
        for x in range(8, world_w - 8, 18):
            color = tuple(int(value) for value in rng.integers(20, 240, size=3))
            cv2.rectangle(world, (x, y), (x + 10, y + 8), color, -1)
            cv2.circle(world, (x + 5, y + 4), 2, (255, 255, 255), -1)
    frames = []
    origin_x, origin_y = 30, 24
    for index in range(count):
        if bad_index is not None and index == bad_index:
            noise = rng.integers(0, 255, size=(frame_h, frame_w, 3), dtype=np.uint8)
            frames.append(noise)
            continue
        x0 = origin_x + index * step[0]
        y0 = origin_y + index * step[1]
        frames.append(world[y0 : y0 + frame_h, x0 : x0 + frame_w].copy())
    return frames


def run_mapping(
    frames: list[np.ndarray],
    out_path: Path,
    *,
    board_id: str,
    repo_root: Path,
    overlap_hint: float = 0.85,
    motion: str = "feature",
    summary_path: Path | None = None,
) -> MosaicResult:
    builder = VideoMosaicBuilder(overlap_hint=overlap_hint, motion=motion)
    builder.controller.start_recording()
    for frame in frames:
        builder.push_frame(frame)
    result = builder.finalize(out_path, board_id=board_id, repo_root=repo_root)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _downscale_gray(gray: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = gray.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 0.999:
        return gray, 1.0
    resized = cv2.resize(
        gray,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _project_corners(shape: tuple, homography: np.ndarray) -> np.ndarray:
    height, width = shape[:2]
    corners = np.array([[[0, 0]], [[width - 1, 0]], [[width - 1, height - 1]], [[0, height - 1]]], dtype=np.float64)
    return cv2.perspectiveTransform(corners, homography).reshape(4, 2)


def _relative_is_stable(affine: np.ndarray, frame_shape: tuple, max_step_fraction: float) -> tuple[bool, str]:
    scale, step, degenerate = _affine_metrics(affine)
    if degenerate or not math.isfinite(scale) or not math.isfinite(step):
        return False, "degenerate affine"
    if not (_MIN_SCALE <= scale <= _MAX_SCALE):
        return False, "scale out of range"
    limit = max_step_fraction * float(min(frame_shape[0], frame_shape[1]))
    if step > limit:
        return False, "step too large"
    return True, "ok"


def _homography_metrics(homography: np.ndarray) -> tuple[float, float, bool]:
    return _affine_metrics(homography[:2, :])


def _affine_metrics(affine: np.ndarray) -> tuple[float, float, bool]:
    linear = np.asarray(affine[:2, :2], dtype=np.float64)
    translation = np.asarray(affine[:2, 2], dtype=np.float64)
    determinant = float(np.linalg.det(linear))
    if determinant <= 1e-8:
        return 0.0, float(np.hypot(translation[0], translation[1])), True
    scale = math.sqrt(abs(determinant))
    column_a = float(np.hypot(linear[0, 0], linear[1, 0]))
    column_b = float(np.hypot(linear[0, 1], linear[1, 1]))
    if abs(column_a - column_b) > 0.12 * max(column_a, column_b, 1e-6):
        return scale, float(np.hypot(*translation)), True
    return scale, float(np.hypot(translation[0], translation[1])), False


def _load_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def run_from_global(
    global_path: Path,
    out_path: Path,
    *,
    board_id: str,
    repo_root: Path,
    overlap: float = 0.85,
    motion: str = "feature",
    window_size: tuple[int, int] | None = None,
    artifact_dir: Path | None = None,
) -> MosaicResult:
    """Crop a serpentine scan from one global photo and build the mosaic."""
    image = cv2.imread(str(global_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read global image {global_path}")
    height, width = image.shape[:2]
    scan = plan_serpentine_scan(width, height, overlap=overlap, window_size=window_size, source=str(global_path))
    frames = crop_serpentine_frames(image, scan)
    builder = VideoMosaicBuilder(overlap_hint=overlap, motion=motion)
    builder.controller.start_recording()
    for frame in frames:
        builder.push_frame(frame)
    rendered = builder.render()
    compare: dict | None = None
    aligned = None
    if rendered is not None:
        compare, aligned = compare_mosaic_to_global(rendered, image)
    extra = {"scan": scan.to_dict(), "compare": compare}
    result = builder.finalize(
        out_path,
        board_id=board_id,
        repo_root=repo_root,
        mapping_extra=extra,
    )
    if artifact_dir is not None:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if rendered is not None:
            cv2.imwrite(str(artifact_dir / "mosaic.png"), rendered)
        write_global_compare_images(image, aligned, artifact_dir)
        (artifact_dir / "summary.json").write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def _parse_window(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise ValueError("--window must look like 560x720")
    return int(parts[0]), int(parts[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mapping Mode：VideoMosaicBuilder（§3.6）")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--from-global", type=Path, default=None, help="用整板图裁蛇形扫描帧")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--board-id", default="VIDEO-MAP-001")
    parser.add_argument("--out", type=Path, default=None, help="马赛克 PNG，默认 assets/maps/<board-id>.png")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--overlap-hint", type=float, default=0.85)
    parser.add_argument("--overlap", type=float, default=None, help="与 --overlap-hint 相同，蛇形扫描用这个重叠率")
    parser.add_argument("--window", default=None, help="扫描窗 WxH，缺省按整板比例")
    parser.add_argument("--motion", choices=("feature", "optical_flow", "auto"), default="feature")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    overlap = args.overlap if args.overlap is not None else args.overlap_hint
    out = args.out if args.out is not None else Path("assets/maps") / f"{args.board_id}.png"
    if args.from_global is not None:
        artifact_dir = args.artifact_dir or Path("artifacts/video_mosaic_from_global")
        try:
            window = _parse_window(args.window)
        except ValueError as exc:
            print(str(exc))
            return 1
        try:
            result = run_from_global(
                args.from_global,
                out,
                board_id=args.board_id,
                repo_root=args.repo_root,
                overlap=overlap,
                motion=args.motion,
                window_size=window,
                artifact_dir=artifact_dir,
            )
        except FileNotFoundError as exc:
            print(str(exc))
            return 1
        print(json.dumps(result.to_dict(), ensure_ascii=False))
        return 0 if result.ok else 1
    summary = args.summary or Path("artifacts/video_mosaic/summary.json")
    if args.video is not None and not args.synthetic:
        frames = _load_video(args.video)
        if not frames:
            print(f"无法读取视频 {args.video}")
            return 1
    else:
        frames = make_synthetic_translation_frames(count=max(3, args.frames))
    result = run_mapping(
        frames,
        out,
        board_id=args.board_id,
        repo_root=args.repo_root,
        overlap_hint=overlap,
        motion=args.motion,
        summary_path=summary,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
