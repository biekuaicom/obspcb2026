"""Stable interval capture with red coverage tracking.

Default mapping path: shoot only when the view is sharp and still, then keep
the captured footprint stuck to the board. The red mask is the last tile
projected into the current frame. There is no growing full-board pixel canvas.

Video mosaic remains an optional bypass in ``vision.mosaic``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from vision.topology import (
    SCHEMA_VERSION,
    ConstellationMatcher,
    GeometricLocalizer,
    TopologyExtractor,
)

_WAIT_FRAME = "等待下一帧确认稳定"
_BLUR = "画面模糊，等待清晰"
_MOTION = "画面仍在移动，等待稳定"
_NOT_STARTED = "请先点击开始"
_NO_OVERLAP = "红区重叠不足，请退回重拍"
_LOST = "未能跟上已拍区域，请退回红色覆盖处重拍"
_ACCEPTED = "已拍摄"


@dataclass
class CaptureDecision:
    ok: bool
    reason: str
    overlap: float | None = None
    tile_id: str | None = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "overlap": None if self.overlap is None else round(float(self.overlap), 4),
            "tile_id": self.tile_id,
            "elapsed_ms": round(float(self.elapsed_ms), 2),
        }


@dataclass
class ObserveState:
    started: bool
    sharp: bool
    sharpness: float
    motion_px: float | None
    stable: bool
    message: str
    overlap: float | None = None
    track_ms: float | None = None
    tracked: bool = False

    def to_dict(self) -> dict:
        return {
            "started": self.started,
            "sharp": self.sharp,
            "sharpness": round(float(self.sharpness), 2),
            "motion_px": None if self.motion_px is None else round(float(self.motion_px), 3),
            "stable": self.stable,
            "message": self.message,
            "overlap": None if self.overlap is None else round(float(self.overlap), 4),
            "track_ms": None if self.track_ms is None else round(float(self.track_ms), 2),
            "tracked": self.tracked,
        }


@dataclass
class _Tile:
    tile_id: str
    width: int
    height: int
    raw: np.ndarray
    binary: np.ndarray
    points: list[dict] = field(default_factory=list)
    extracted: object | None = None
    M_to_ref: np.ndarray = field(default_factory=lambda: np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))


class StableCaptureSession:
    """Arm, wait until a frame is sharp and still, then chain overlapping shots."""

    def __init__(
        self,
        *,
        min_sharpness: float = 40.0,
        max_motion_px: float = 8.0,
        min_overlap: float = 0.08,
        max_long_edge: int = 640,
    ):
        if min_sharpness <= 0 or max_motion_px <= 0 or not 0.0 < min_overlap < 1.0:
            raise ValueError("thresholds must be positive and overlap in (0, 1)")
        self.min_sharpness = float(min_sharpness)
        self.max_motion_px = float(max_motion_px)
        self.min_overlap = float(min_overlap)
        self.extractor = TopologyExtractor(max_long_edge=max_long_edge)
        self.matcher = ConstellationMatcher()
        self.localizer = GeometricLocalizer(self.extractor, self.matcher)
        self.started = False
        self.tiles: list[_Tile] = []
        self.edges: list[dict] = []
        self.rejects: list[dict] = []
        self._prev: np.ndarray | None = None
        self._last: np.ndarray | None = None
        self._pose = None
        self._anchor_id: str | None = None
        self._polygon: np.ndarray | None = None
        self._quads: list[np.ndarray] = []
        self._overlap: float | None = None
        self._track_ms: float | None = None
        self._tracked = False
        self.merge_ms: list[float] = []
        self.chain_mask: np.ndarray | None = None
        self.chain_origin: tuple[int, int] = (0, 0)

    def start(self) -> None:
        self.started = True

    def observe(self, frame: np.ndarray) -> ObserveState:
        sharpness = frame_sharpness(frame)
        sharp = sharpness >= self.min_sharpness
        motion = None if self._last is None else interframe_motion_px(self._last, frame)
        self._prev = self._last
        self._last = frame
        stable = bool(self.started and sharp and motion is not None and motion <= self.max_motion_px)
        if not self.started:
            message = _NOT_STARTED
        elif not sharp:
            message = _BLUR
        elif motion is None:
            message = _WAIT_FRAME
        elif motion > self.max_motion_px:
            message = _MOTION
        else:
            message = "稳定，可以拍摄"
        self._tracked = False
        self._overlap = None
        self._polygon = None
        self._quads = []
        self._track_ms = None
        self._anchor_id = None
        if self.started and self.tiles:
            self._track_union(frame)
        return ObserveState(
            started=self.started,
            sharp=sharp,
            sharpness=sharpness,
            motion_px=motion,
            stable=stable,
            message=message,
            overlap=self._overlap,
            track_ms=self._track_ms,
            tracked=self._tracked,
        )

    def capture(self) -> CaptureDecision:
        if not self.started:
            return self._reject(_NOT_STARTED)
        frame = self._last
        if frame is None:
            return self._reject(_WAIT_FRAME)
        sharpness = frame_sharpness(frame)
        if sharpness < self.min_sharpness:
            return self._reject(_BLUR)
        if self._prev is None:
            return self._reject(_WAIT_FRAME)
        motion = interframe_motion_px(self._prev, frame)
        if motion > self.max_motion_px:
            return self._reject(_MOTION)
        if not self.tiles:
            tile = self._store_tile(frame, None, None)
            return CaptureDecision(True, _ACCEPTED, overlap=None, tile_id=tile.tile_id)
        if not self._tracked or self._pose is None or not self._pose.ok or self._anchor_id is None:
            return self._reject(_LOST, overlap=self._overlap, elapsed_ms=self._track_ms or 0.0)
        overlap = float(self._overlap or 0.0)
        if overlap < self.min_overlap:
            return self._reject(_NO_OVERLAP, overlap=overlap, elapsed_ms=self._track_ms or 0.0)
        tile = self._store_tile(frame, self._anchor_id, self._pose.M)
        return CaptureDecision(
            True,
            _ACCEPTED,
            overlap=overlap,
            tile_id=tile.tile_id,
            elapsed_ms=self._track_ms or 0.0,
        )

    def render(self, frame: np.ndarray | None = None) -> np.ndarray:
        """Red coverage on the live frame. Only the current view is painted."""
        source = self._last if frame is None else frame
        canvas = source.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        if not self._quads:
            return canvas
        return paint_red_union(canvas, self._quads)

    def chain_dict(self) -> dict:
        return {
            "tiles": [
                {
                    "id": tile.tile_id,
                    "image_size": {"width": tile.width, "height": tile.height},
                    "points": len(tile.points),
                    "M_to_ref": np.asarray(tile.M_to_ref, dtype=np.float64).round(6).tolist(),
                }
                for tile in self.tiles
            ],
            "edges": self.edges,
            "rejects": self.rejects,
        }

    def topology_db(self, *, board_id: str, source: str) -> dict:
        points = []
        for tile in self.tiles:
            for point in tile.points:
                local_xy = point["xy"]
                chain_xy = _apply_m(tile.M_to_ref, np.array([local_xy], dtype=np.float64))[0]
                item = dict(point)
                item["xy_tile"] = local_xy
                item["xy"] = [round(float(chain_xy[0]), 2), round(float(chain_xy[1]), 2)]
                item["id"] = f"{tile.tile_id}:{point['id']}"
                item["tile_id"] = tile.tile_id
                points.append(item)
        mask_size = None if self.chain_mask is None else [int(self.chain_mask.shape[1]), int(self.chain_mask.shape[0])]
        return {
            "schema_version": SCHEMA_VERSION,
            "board_id": board_id,
            "source": source,
            "kind": "stable_capture_chain",
            "chain_frame": self.tiles[0].tile_id if self.tiles else None,
            "image_size": None,
            "chain_mask_size": mask_size,
            "full_canvas": False,
            "points": points,
            "edges": self.edges,
            "note": "xy 在首张瓦片坐标系；没有全板像素画布",
        }

    @property
    def coverage_quads(self) -> list[np.ndarray]:
        return list(self._quads)

    def _track_union(self, frame: np.ndarray) -> None:
        started = time.perf_counter()
        observed = self.extractor.extract(frame)
        anchor = None
        pose = None
        for tile in reversed(self.tiles):
            if tile.extracted is None:
                continue
            self.matcher.set_topology(tile.extracted)
            candidate = self.matcher.localize(observed)
            if candidate.ok and not alignment_rejected(candidate.M):
                anchor = tile
                pose = candidate
                break
        self._track_ms = (time.perf_counter() - started) * 1000.0
        self._pose = pose
        if anchor is None or pose is None:
            return
        self._anchor_id = anchor.tile_id
        self._quads = self._quads_from_anchor(anchor, pose.M)
        self._polygon = self._quads[-1] if self._quads else None
        self._overlap = union_overlap_ratio(self._quads, frame.shape[1], frame.shape[0])
        self._tracked = True

    def _quads_from_anchor(self, anchor: _Tile, anchor_to_frame: np.ndarray) -> list[np.ndarray]:
        """Map every captured tile into the live frame through the chain frame."""
        ref_to_frame = _compose_m(np.asarray(anchor_to_frame, dtype=np.float64), _invert_m(anchor.M_to_ref))
        quads = []
        for tile in self.tiles:
            tile_to_frame = _compose_m(ref_to_frame, tile.M_to_ref)
            quads.append(project_tile_quad(tile_to_frame, tile.width, tile.height))
        return quads

    def _store_tile(self, frame: np.ndarray, parent: str | None, pose_m: np.ndarray | None) -> _Tile:
        extracted = self.extractor.extract(frame)
        extracted.board_id = f"cap-{len(self.tiles):03d}"
        tile = _Tile(
            tile_id=extracted.board_id,
            width=int(frame.shape[1]),
            height=int(frame.shape[0]),
            raw=frame.copy(),
            binary=binary_lines(frame),
            points=[point.to_dict() for point in extracted.points],
            extracted=extracted,
        )
        if parent is None or pose_m is None:
            tile.M_to_ref = _identity_m()
        else:
            anchor = next(item for item in self.tiles if item.tile_id == parent)
            # pose maps the anchor tile into this frame, so this frame goes back through it.
            tile.M_to_ref = _compose_m(anchor.M_to_ref, _invert_m(np.asarray(pose_m, dtype=np.float64)))
            direction = "previous_tile_to_current_frame" if parent == self.tiles[-1].tile_id else "anchor_tile_to_current_frame"
            angle = pose_angle_deg(pose_m)
            self.edges.append(
                {
                    "src": parent,
                    "dst": tile.tile_id,
                    "M": np.asarray(pose_m, dtype=np.float64).round(6).tolist(),
                    "M_to_ref": tile.M_to_ref.round(6).tolist(),
                    "overlap": round(float(self._overlap or 0.0), 4),
                    "angle_deg": round(angle, 3),
                    "reflection": bool(linear_is_reflection(pose_m)),
                    "direction": direction,
                }
            )
        corrected, edge_stats = compensate_board_edge(frame)
        tile.binary = binary_lines(corrected)
        tile.edge_before = edge_stats.get("preview_before")
        tile.edge_after = edge_stats.get("preview_after")
        tile.edge_stats = {key: value for key, value in edge_stats.items() if not str(key).startswith("preview")}
        if self.edges:
            self.edges[-1]["edge"] = tile.edge_stats
        self.tiles.append(tile)
        merge_started = time.perf_counter()
        self._fuse_chain_mask(tile)
        merge_ms = (time.perf_counter() - merge_started) * 1000.0
        self.merge_ms.append(merge_ms)
        if self.edges:
            self.edges[-1]["merge_ms"] = round(merge_ms, 2)
        self.matcher.set_topology(extracted)
        self._anchor_id = tile.tile_id
        self._pose_after_capture(tile)
        return tile

    def _pose_after_capture(self, tile: _Tile) -> None:
        """The shutter frame is the new tile, so every earlier tile can be drawn at once."""
        self._quads = self._quads_from_anchor(tile, _identity_m())
        self._polygon = self._quads[-1] if self._quads else None
        self._overlap = 1.0
        self._tracked = True

    def _fuse_chain_mask(self, tile: _Tile) -> None:
        """OR this tile's binary into a chain-sized mask. It grows only with captured tiles."""
        corners = project_tile_quad(tile.M_to_ref, tile.width, tile.height)
        min_x = int(np.floor(np.min(corners[:, 0]))) - 1
        min_y = int(np.floor(np.min(corners[:, 1]))) - 1
        max_x = int(np.ceil(np.max(corners[:, 0]))) + 2
        max_y = int(np.ceil(np.max(corners[:, 1]))) + 2
        if self.chain_mask is None:
            self.chain_origin = (min(0, min_x), min(0, min_y))
            width = max(tile.width, max_x - self.chain_origin[0])
            height = max(tile.height, max_y - self.chain_origin[1])
            self.chain_mask = np.zeros((height, width), dtype=np.uint8)
        origin_x, origin_y = self.chain_origin
        need_left = max(0, origin_x - min_x)
        need_top = max(0, origin_y - min_y)
        need_right = max(0, max_x - (origin_x + self.chain_mask.shape[1]))
        need_bottom = max(0, max_y - (origin_y + self.chain_mask.shape[0]))
        grown_w = self.chain_mask.shape[1] + need_left + need_right
        grown_h = self.chain_mask.shape[0] + need_top + need_bottom
        if grown_w > 2048 or grown_h > 2048:
            return
        if need_left or need_top or need_right or need_bottom:
            grown = np.zeros((grown_h, grown_w), dtype=np.uint8)
            grown[need_top : need_top + self.chain_mask.shape[0], need_left : need_left + self.chain_mask.shape[1]] = self.chain_mask
            self.chain_mask = grown
            self.chain_origin = (origin_x - need_left, origin_y - need_top)
        into_mask = _compose_m(_translate_m(-self.chain_origin[0], -self.chain_origin[1]), tile.M_to_ref)
        warped = cv2.warpAffine(
            tile.binary,
            np.asarray(into_mask, dtype=np.float64),
            (self.chain_mask.shape[1], self.chain_mask.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
        self.chain_mask = cv2.bitwise_or(self.chain_mask, warped)

    def _reject(self, reason: str, overlap: float | None = None, elapsed_ms: float = 0.0) -> CaptureDecision:
        self.rejects.append({"reason": reason, "overlap": overlap})
        return CaptureDecision(False, reason, overlap=overlap, elapsed_ms=elapsed_ms)


def frame_sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def interframe_motion_px(prev: np.ndarray, curr: np.ndarray) -> float:
    """Shift magnitude in original pixels. Same-size frames only."""
    left = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) if prev.ndim == 3 else prev
    right = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) if curr.ndim == 3 else curr
    if left.shape != right.shape or left.size == 0:
        return float("inf")
    height, width = left.shape[:2]
    scale = min(1.0, 160.0 / max(height, width))
    if scale < 0.999:
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        left = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
    (shift_x, shift_y), _response = cv2.phaseCorrelate(left.astype(np.float32), right.astype(np.float32))
    return float(np.hypot(shift_x, shift_y) / scale)


def pose_angle_deg(matrix: np.ndarray) -> float:
    linear = np.asarray(matrix, dtype=np.float64)
    return float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))


def linear_is_reflection(matrix: np.ndarray) -> bool:
    linear = np.asarray(matrix, dtype=np.float64)[:2, :2]
    return float(np.linalg.det(linear)) < 0.0


def alignment_rejected(matrix: np.ndarray) -> bool:
    """Mirror flips and large spins are not a handheld board move."""
    if linear_is_reflection(matrix):
        return True
    return abs(pose_angle_deg(matrix)) > 18.0


def compensate_board_edge(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Pull the PCB outline toward a line or a gentle arc. Background stays put."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    found = _longest_board_edge(gray)
    empty = {"kind": "none", "rms_before": None, "rms_after": None, "samples": 0}
    if found is None:
        return image, empty
    xs, observed, axis = found
    kind, target, rms_before = _fit_line_or_arc(xs, observed)
    corrected = _shift_along_edge(image, xs, observed, target, axis)
    measured = _longest_board_edge(cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY) if corrected.ndim == 3 else corrected)
    rms_after = rms_before
    if measured is not None and measured[2] == axis:
        _xs, after, _axis = measured
        if len(after) >= 20:
            _kind, _target, rms_after = _fit_line_or_arc(_xs, after)
    preview_before = _draw_edge_preview(image, xs, observed, target)
    preview_after = _draw_edge_preview(corrected, xs, target, target)
    return corrected, {
        "kind": kind,
        "axis": axis,
        "rms_before": round(float(rms_before), 3),
        "rms_after": round(float(rms_after), 3),
        "samples": int(len(xs)),
        "preview_before": preview_before,
        "preview_after": preview_after,
    }


def _longest_board_edge(gray: np.ndarray):
    mask = (gray < 150).astype(np.uint8) * 255
    if int(mask.sum()) < 200:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    height, width = mask.shape
    candidates = []
    for axis, profile in (
        ("y", _edge_profile(mask, "top")),
        ("y-bottom", _edge_profile(mask, "bottom")),
        ("x", _edge_profile(mask, "left")),
        ("x-right", _edge_profile(mask, "right")),
    ):
        run = _continuous_run(profile, max_jump=8.0)
        if run is None:
            continue
        start, stop = run
        if stop - start < 40:
            continue
        coords = np.arange(start, stop, dtype=np.float64)
        values = profile[start:stop]
        if np.median(values) < 4 or np.median(values) > (height if axis.startswith("y") else width) - 5:
            continue
        inward = {"y": "y+", "y-bottom": "y-", "x": "x+", "x-right": "x-"}[axis]
        candidates.append((stop - start, coords, values, inward))
    if not candidates:
        return None
    _length, coords, values, axis = max(candidates, key=lambda item: item[0])
    return coords, values, axis


def _edge_profile(mask: np.ndarray, side: str) -> np.ndarray:
    height, width = mask.shape
    if side in {"top", "bottom"}:
        profile = np.full(width, np.nan, dtype=np.float64)
        for x in range(width):
            rows = np.flatnonzero(mask[:, x])
            if len(rows):
                profile[x] = float(rows[0] if side == "top" else rows[-1])
        return profile
    profile = np.full(height, np.nan, dtype=np.float64)
    for y in range(height):
        cols = np.flatnonzero(mask[y, :])
        if len(cols):
            profile[y] = float(cols[0] if side == "left" else cols[-1])
    return profile


def _continuous_run(profile: np.ndarray, max_jump: float):
    best = None
    start = None
    previous = None

    def close(stop: int) -> None:
        nonlocal best, start
        if start is not None and stop - start > (0 if best is None else best[1] - best[0]):
            best = (start, stop)
        start = None

    for index, value in enumerate(profile):
        if not np.isfinite(value) or (previous is not None and start is not None and abs(value - previous) > max_jump):
            close(index)
            if not np.isfinite(value):
                previous = None
                continue
        if start is None:
            start = index
        previous = value
    close(len(profile))
    return best


def _fit_line_or_arc(xs: np.ndarray, ys: np.ndarray):
    design = np.vstack([xs, np.ones_like(xs)]).T
    slope, intercept = np.linalg.lstsq(design, ys, rcond=None)[0]
    line = slope * xs + intercept
    line_rms = float(np.sqrt(np.mean((ys - line) ** 2)))
    target = line
    kind = "line"
    if len(xs) >= 30:
        circle = _fit_circle(xs, ys)
        if circle is not None:
            center_x, center_y, radius = circle
            radial = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)
            arc_rms = float(np.sqrt(np.mean((radial - radius) ** 2)))
            if 40.0 <= radius <= 8000.0 and arc_rms + 0.3 < line_rms:
                direction = np.arctan2(ys - center_y, xs - center_x)
                target = np.stack(
                    [center_x + radius * np.cos(direction), center_y + radius * np.sin(direction)],
                    axis=1,
                )
                kind = "arc"
                return kind, target, arc_rms
    return kind, target, line_rms


def _fit_circle(xs: np.ndarray, ys: np.ndarray):
    design = np.column_stack([2.0 * xs, 2.0 * ys, np.ones_like(xs)])
    observed = xs * xs + ys * ys
    solution, _residuals, _rank, _singular = np.linalg.lstsq(design, observed, rcond=None)
    center_x, center_y, offset = solution
    radius_sq = offset + center_x * center_x + center_y * center_y
    if radius_sq <= 1.0:
        return None
    return float(center_x), float(center_y), float(np.sqrt(radius_sq))


def _shift_along_edge(image: np.ndarray, xs: np.ndarray, observed: np.ndarray, target, axis: str) -> np.ndarray:
    height, width = image.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    band = 12.0
    inward = 1.0 if axis.endswith("+") else -1.0
    horizontal = axis.startswith("y")
    target_values = target if getattr(target, "ndim", 1) == 1 else target[:, 1]
    for index, coord in enumerate(xs):
        source = float(observed[index])
        destination = float(target_values[index])
        shift = destination - source
        if abs(shift) < 0.15:
            continue
        start = int(np.floor(destination - (band if inward < 0 else 1)))
        stop = int(np.ceil(destination + (band if inward > 0 else 1)))
        sample = int(round(float(coord)))
        for dest in range(start, stop + 1):
            distance = (dest - destination) * inward
            if distance < -1.0:
                continue
            weight = max(0.0, 1.0 - max(0.0, distance) / band)
            sampled = dest - shift * weight
            if horizontal and 0 <= dest < height and 0 <= sample < width:
                map_y[dest, sample] = sampled
            elif not horizontal and 0 <= dest < width and 0 <= sample < height:
                map_x[sample, dest] = sampled
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _draw_edge_preview(image: np.ndarray, xs: np.ndarray, observed: np.ndarray, target) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    target_values = target if getattr(target, "ndim", 1) == 1 else target[:, 1]
    observed_values = observed if getattr(observed, "ndim", 1) == 1 else observed[:, 1]
    for coord, source, fitted in zip(xs, observed_values, target_values):
        sample = int(round(coord))
        cv2.circle(canvas, (sample, int(round(float(source)))), 1, (0, 0, 255), -1)
    if getattr(target, "ndim", 1) == 2:
        for point in np.round(target).astype(int):
            cv2.circle(canvas, (int(point[0]), int(point[1])), 1, (0, 220, 0), -1)
    else:
        for coord, fitted in zip(xs, target_values):
            cv2.circle(canvas, (int(round(coord)), int(round(fitted))), 1, (0, 220, 0), -1)
    return canvas


def binary_lines(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    odd = 31 if min(gray.shape[:2]) > 40 else 11
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, odd, 5)


def tile_corners(width: int, height: int) -> np.ndarray:
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float64,
    )


def project_tile_quad(matrix_2x3: np.ndarray, width: int, height: int) -> np.ndarray:
    corners = tile_corners(width, height)
    matrix = np.asarray(matrix_2x3, dtype=np.float64)
    return corners @ matrix[:, :2].T + matrix[:, 2]


def union_overlap_ratio(quads: list[np.ndarray], width: int, height: int) -> float:
    if width <= 1 or height <= 1 or not quads:
        return 0.0
    mask = np.zeros((height, width), dtype=np.uint8)
    for quad in quads:
        polygon = np.round(np.asarray(quad, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
        if polygon.shape[0] >= 3 and np.isfinite(quad).all():
            cv2.fillConvexPoly(mask, polygon, 255)
    return float(np.count_nonzero(mask) / float(width * height))


def frame_overlap_ratio(quad: np.ndarray, width: int, height: int) -> float:
    """Share of the live view covered by the projected previous tile."""
    if width <= 1 or height <= 1:
        return 0.0
    frame = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    polygon = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    if polygon.shape[0] < 3 or not np.isfinite(polygon).all():
        return 0.0
    area, _intersection = cv2.intersectConvexConvex(polygon, frame)
    return float(max(0.0, area) / float(width * height))


def paint_red_union(frame_bgr: np.ndarray, quads: list[np.ndarray], alpha: float = 0.45) -> np.ndarray:
    canvas = frame_bgr.copy()
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    for quad in quads:
        polygon = np.round(np.asarray(quad, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
        if polygon.shape[0] >= 3:
            cv2.fillConvexPoly(mask, polygon, 255)
    red = canvas.copy()
    red[:, :, 2] = 255
    red[:, :, 1] = (red[:, :, 1] * 0.25).astype(np.uint8)
    red[:, :, 0] = (red[:, :, 0] * 0.25).astype(np.uint8)
    painted = mask > 0
    canvas[painted] = (canvas[painted] * (1.0 - alpha) + red[painted] * alpha).astype(np.uint8)
    return canvas


def paint_red_coverage(frame_bgr: np.ndarray, quad: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return paint_red_union(frame_bgr, [quad], alpha=alpha)


def _red_pixels(image: np.ndarray) -> int:
    return int(np.count_nonzero(image[:, :, 2] > image[:, :, 0] + 20))


def _identity_m() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _translate_m(x: float, y: float) -> np.ndarray:
    matrix = _identity_m()
    matrix[0, 2] = float(x)
    matrix[1, 2] = float(y)
    return matrix


def _invert_m(matrix_2x3: np.ndarray) -> np.ndarray:
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = np.asarray(matrix_2x3, dtype=np.float64)
    return np.linalg.inv(homography)[:2, :]


def _compose_m(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_h = np.eye(3, dtype=np.float64)
    right_h = np.eye(3, dtype=np.float64)
    left_h[:2, :] = np.asarray(left, dtype=np.float64)
    right_h[:2, :] = np.asarray(right, dtype=np.float64)
    return (left_h @ right_h)[:2, :]


def _apply_m(matrix_2x3: np.ndarray, points: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix_2x3, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    return pts @ matrix[:, :2].T + matrix[:, 2]


def save_session(session: StableCaptureSession, out_dir: Path, *, board_id: str, source: str) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists():
        for old in out_dir.rglob("*"):
            if old.is_file():
                old.unlink()
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    for tile in session.tiles:
        cv2.imwrite(str(tiles_dir / f"{tile.tile_id}.png"), tile.raw)
        cv2.imwrite(str(tiles_dir / f"{tile.tile_id}-binary.png"), tile.binary)
    if session.tiles and session._last is not None and session._polygon is not None:
        cv2.imwrite(str(out_dir / "coverage.png"), session.render(session._last))
    chain = session.chain_dict()
    topology = session.topology_db(board_id=board_id, source=source)
    (out_dir / "chain.json").write_text(json.dumps(chain, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "topology_db.json").write_text(json.dumps(topology, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    overlaps = [edge["overlap"] for edge in session.edges]
    track_times = [edge.get("track_ms") for edge in session.edges if edge.get("track_ms") is not None]
    if session.chain_mask is not None:
        cv2.imwrite(str(out_dir / "chain_mask.png"), session.chain_mask)
    summary = {
        "board_id": board_id,
        "source": source,
        "shots": len(session.tiles),
        "rejects": len(session.rejects),
        "reject_reasons": [item["reason"] for item in session.rejects],
        "overlap_mean": None if not overlaps else round(float(np.mean(overlaps)), 4),
        "track_ms_median": None if not track_times else round(float(np.median(track_times)), 2),
        "merge_ms_median": None if not session.merge_ms else round(float(np.median(session.merge_ms)), 2),
        "chain_points": sum(len(tile.points) for tile in session.tiles),
        "chain_mask_size": None
        if session.chain_mask is None
        else [int(session.chain_mask.shape[1]), int(session.chain_mask.shape[0])],
        "full_canvas": False,
        "chain": str(out_dir / "chain.json"),
        "topology_db": str(out_dir / "topology_db.json"),
        "coverage": str(out_dir / "coverage.png"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _settle(session: StableCaptureSession, frame: np.ndarray) -> ObserveState:
    session.observe(frame)
    return session.observe(frame)


def _rotated_crop(image: np.ndarray, x: int, y: int, width: int, height: int, angle_deg: float):
    crop = image[y : y + height, x : x + width].copy()
    if abs(angle_deg) < 1e-6:
        return crop, 0.0
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    rotated = cv2.warpAffine(crop, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated, pose_angle_deg(matrix)


def _angle_delta(estimated: float, truth: float) -> float:
    delta = (estimated - truth + 180.0) % 360.0 - 180.0
    return float(delta)


def run_board_demo(image: np.ndarray, out_dir: Path, *, board_id: str, source: str) -> dict:
    """Overlapping crops with small rotations; a far jump and a blurred pair are rejected."""
    window_w, window_h = 480, 360
    origin_x, origin_y = 80, 160
    session = StableCaptureSession()
    session.start()
    track_ms: list[float] = []

    def crop_at(x: int, angle: float = 0.0):
        return _rotated_crop(image, x, origin_y, window_w, window_h, angle)

    blurred = cv2.GaussianBlur(crop_at(origin_x)[0], (0, 0), 3.5)
    _settle(session, blurred)
    blur_decision = session.capture()

    first_frame, _truth0 = crop_at(origin_x, 0.0)
    _settle(session, first_frame)
    first = session.capture()

    decisions = [blur_decision, first]
    step_frames = [(0, first_frame, session.render(first_frame))]
    # Absolute rotations of each shot. The relative truth is the difference of those rotations.
    shot_angles = [0.0, 4.0, 1.0]
    for index, (step, angle) in enumerate(zip((70, 140), (4.0, 1.0)), start=1):
        frame, _absolute = crop_at(origin_x + step, angle)
        state = _settle(session, frame)
        if state.track_ms is not None:
            track_ms.append(state.track_ms)
        decision = session.capture()
        decision.elapsed_ms = state.track_ms or 0.0
        decisions.append(decision)
        if session.edges and decision.ok:
            relative = angle - shot_angles[index - 1]
            _unused, truth = _rotated_crop(image, origin_x, origin_y, window_w, window_h, relative)
            session.edges[-1]["track_ms"] = round(state.track_ms or 0.0, 2)
            session.edges[-1]["angle_truth_deg"] = round(truth, 3)
            session.edges[-1]["angle_error_deg"] = round(_angle_delta(session.edges[-1]["angle_deg"], truth), 3)
        step_frames.append((index, frame, session.render(frame) if decision.ok else frame))

    jumped = image[origin_y + 780 : origin_y + 780 + window_h, origin_x : origin_x + window_w].copy()
    state = _settle(session, jumped)
    if state.track_ms is not None:
        track_ms.append(state.track_ms)
    rejected = session.capture()
    rejected.elapsed_ms = state.track_ms or 0.0
    decisions.append(rejected)

    home = crop_at(origin_x)[0]
    home_state = _settle(session, home)
    if home_state.track_ms is not None:
        track_ms.append(home_state.track_ms)
    union = session.render(home)
    last_only = paint_red_coverage(home, session.coverage_quads[-1]) if session.coverage_quads else home
    union_px = _red_pixels(union)
    last_px = _red_pixels(last_only)
    summary = save_session(session, out_dir, board_id=board_id, source=source)
    steps_dir = Path(out_dir) / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    for index, _frame, image_step in step_frames:
        cv2.imwrite(str(steps_dir / f"step_{index:03d}_coverage.png"), image_step)
    cv2.imwrite(str(Path(out_dir) / "coverage.png"), union)
    cv2.imwrite(str(Path(out_dir) / "coverage_last_tile.png"), last_only)
    summary["shots"] = len(session.tiles)
    summary["rejects"] = len(session.rejects)
    summary["reject_reasons"] = [item["reason"] for item in session.rejects]
    overlaps = [edge["overlap"] for edge in session.edges]
    summary["overlap_mean"] = None if not overlaps else round(float(np.mean(overlaps)), 4)
    summary["track_ms_median"] = None if not track_ms else round(float(np.median(track_ms)), 2)
    summary["merge_ms_median"] = None if not session.merge_ms else round(float(np.median(session.merge_ms)), 2)
    summary["chain_points"] = sum(len(tile.points) for tile in session.tiles)
    summary["union_red_px"] = union_px
    summary["last_tile_red_px"] = last_px
    summary["union_over_last"] = None if last_px <= 0 else round(union_px / last_px, 3)
    summary["full_canvas"] = False
    summary["decisions"] = [item.to_dict() for item in decisions]
    angle_errors = [edge["angle_error_deg"] for edge in session.edges if edge.get("angle_error_deg") is not None]
    summary["angle_error_deg_median"] = None if not angle_errors else round(float(np.median(np.abs(angle_errors))), 3)
    summary["angle_error_deg_max"] = None if not angle_errors else round(float(np.max(np.abs(angle_errors))), 3)
    summary["angle_errors_deg"] = [round(float(value), 3) for value in angle_errors]
    befores = [tile.edge_stats.get("rms_before") for tile in session.tiles if getattr(tile, "edge_stats", None) and tile.edge_stats.get("rms_before") is not None]
    afters = [tile.edge_stats.get("rms_after") for tile in session.tiles if getattr(tile, "edge_stats", None) and tile.edge_stats.get("rms_after") is not None]
    summary["edge_rms_before_median"] = None if not befores else round(float(np.median(befores)), 3)
    summary["edge_rms_after_median"] = None if not afters else round(float(np.median(afters)), 3)
    align_dir = Path(out_dir) / "align"
    align_dir.mkdir(parents=True, exist_ok=True)
    for tile in session.tiles:
        if getattr(tile, "edge_before", None) is not None:
            cv2.imwrite(str(align_dir / f"{tile.tile_id}_before.png"), tile.edge_before)
        if getattr(tile, "edge_after", None) is not None:
            cv2.imwrite(str(align_dir / f"{tile.tile_id}_after.png"), tile.edge_after)
    summary["align_dir"] = str(align_dir)
    (Path(out_dir) / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="稳定式间隔拍照 + 红色覆盖跟踪")
    parser.add_argument("--global", dest="global_path", type=Path, required=True)
    parser.add_argument("--board-id", default="SAMPLE-PARTIAL-001")
    parser.add_argument("--out", type=Path, default=Path("artifacts/stable_capture_map"))
    args = parser.parse_args(argv)
    image = cv2.imread(str(args.global_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"无法读取 {args.global_path}")
        return 1
    summary = run_board_demo(image, args.out, board_id=args.board_id, source=str(args.global_path))
    brief = {
        "shots": summary["shots"],
        "rejects": summary["rejects"],
        "reject_reasons": summary["reject_reasons"],
        "overlap_mean": summary["overlap_mean"],
        "track_ms_median": summary["track_ms_median"],
        "merge_ms_median": summary["merge_ms_median"],
        "chain_points": summary["chain_points"],
        "union_over_last": summary["union_over_last"],
        "angle_error_deg_median": summary.get("angle_error_deg_median"),
        "angle_error_deg_max": summary.get("angle_error_deg_max"),
        "edge_rms_before_median": summary.get("edge_rms_before_median"),
        "edge_rms_after_median": summary.get("edge_rms_after_median"),
        "full_canvas": False,
        "summary": str(args.out / "summary.json"),
    }
    print(json.dumps(brief, ensure_ascii=False))
    return 0 if summary["shots"] >= 2 and summary["rejects"] >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
