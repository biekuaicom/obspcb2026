"""Stable interval capture with red coverage tracking.

Default mapping path: shoot only when the view is sharp and still, then keep
the captured footprint stuck to the board. Live pose comes from geometric
features in the first-tile frame, not from the color overview. ``render()``
paints the union of already-captured regions on the current frame only.

A constrained low-resolution color overview is placed after each accepted shot
(``full_canvas`` stays false). Overlap is never averaged: a high alignment
residual keeps the sharper tile or leaves a gap. Labels live in board
coordinates and are drawn after the pose is solved.

Accepted stills, and only those, are also split into typed elements and
corner fiducials. Each still is placed into the first-tile (or fiducial)
board frame by type-to-type matching in the overlap: silkscreen, pads,
holes, outlines, pins, traces, and color blocks. A periodic via lattice
alone does not lock. If the red-overlap elements disagree, the shot is
rejected and is never force-blended. Tiles may arrive out of order; a
frame that does not fit the last shot can still lock to an earlier region.
After each accepted shot, typed and fiducial constraints refine one global
board frame so chain drift does not accumulate. Preview keeps painting the
last locked red union and does not run that separation.

Video mosaic remains an optional bypass in ``vision.mosaic``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from vision.board_lock import (
    SAFE_RESIDUAL_PX,
    composite_registered_tile,
    estimate_feature_pose,
    estimate_similarity_pose,
    extract_feature_pack,
    landmarks_from_snapshot,
    masked_ncc,
    refine_pose_graph,
)
from vision.elements import (
    align_red_region,
    detect_fiducials,
    estimate_typed_pose,
    extract_board_elements,
    fiducial_correspondences,
    is_periodic_lattice,
    typed_correspondences,
    unique_landmark_support,
)
from vision.structure_tree import extract_structure_tree
from vision.topology import (
    SCHEMA_VERSION,
    ConstellationMatcher,
    GeometricLocalizer,
    Pose2D,
    TopologyExtractor,
)

_WAIT_FRAME = "等待下一帧确认稳定"
_BLUR = "画面模糊，等待清晰"
_MOTION = "画面仍在移动，等待稳定"
_NOT_STARTED = "请先点击开始"
_NO_OVERLAP = "红区重叠不足，请退回重拍"
_LOST = "未能跟上已拍区域，请退回红色覆盖处重拍"
_UNCERTAIN = "位姿还不确定，请放慢一点，或稍微退回已拍区域"
_ELEMENT_MISMATCH = "对位不一致，请重拍，或退回已拍区域"
_ACCEPTED = "已拍摄"
_READY = "稳定，可以拍摄"
_READY_OVERLAP = "重叠足够，可以拍摄"
_COVERAGE_AREA = "覆盖面积变化异常，请退回重拍"
_COVERAGE_ISLAND = "覆盖出现孤立区块，请退回重拍"

OVERVIEW_MAX_SIDE = 3200
OVERVIEW_MAX_PIXELS = 8_000_000
OVERVIEW_MARGIN_PX = 24
OVERVIEW_FEATHER_PX = 10

# Live red-coverage consensus. Later shots should add constraints, not false blocks.
_ANCHOR_RECENT = 8
_ANCHOR_MAX_ALL = 12
_ANCHOR_EARLIER_STRIDE = 3
_POSE_CORNER_TOL_PX = 12.0
_POSE_ANGLE_TOL_DEG = 6.0
_AMBIGUOUS_SCORE_RATIO = 0.8
_AMBIGUOUS_INLIER_RATIO = 0.75
_POINT_MATCH_PX = 6.0
_MIN_INFRAME_SUPPORT_POINTS = 3
_MIN_SUPPORT_RATIO = 0.25
_SLIVER_VISIBLE_FRACTION = 0.15
_COVERAGE_TOUCH_PX = 2

# Coverage memory lives in the chain / board frame, not the live camera frame.
# Area and shape may only grow like one new connected tile, plus thin edge slack.
_COVERAGE_EDGE_SLACK_PX = 3
_COVERAGE_GROW_MARGIN = 0.12
_COVERAGE_BLOB_FRAC = 0.04
_COVERAGE_BLOB_MIN_PX = 400


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
    corrected: np.ndarray | None = None
    M_overview: np.ndarray | None = None
    edge_stats: dict | None = None
    edge_before: np.ndarray | None = None
    edge_after: np.ndarray | None = None
    structure: object | None = None
    landmarks: np.ndarray | None = None
    align_residual_px: float = 0.0
    sharpness: float = 0.0
    elements: object | None = None
    feature_pack: object | None = None
    fiducials: list = field(default_factory=list)


@dataclass
class CoverageMemory:
    """Remembered red coverage in chain / ref coordinates."""

    area: int
    origin: tuple[int, int]
    mask: np.ndarray
    contour: list[list[float]]
    bbox: tuple[int, int, int, int]
    moments: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "area": int(self.area),
            "origin": [int(self.origin[0]), int(self.origin[1])],
            "bbox": [int(value) for value in self.bbox],
            "moments": {str(key): float(value) for key, value in self.moments.items()},
            "contour": [[float(point[0]), float(point[1])] for point in self.contour],
        }


@dataclass
class CoverageCheck:
    ok: bool
    reason: str
    memory: CoverageMemory | None = None
    added: int = 0


@dataclass
class _AnchorHypothesis:
    tile: _Tile
    index: int
    pose: object
    score: float
    ref_to_frame: np.ndarray


class StableCaptureSession:
    """Arm, wait until a frame is sharp and still, then chain overlapping shots."""

    def __init__(
        self,
        *,
        min_sharpness: float = 40.0,
        max_motion_px: float = 8.0,
        min_overlap: float = 0.08,
        max_long_edge: int = 640,
        overview_max_side: int = OVERVIEW_MAX_SIDE,
        overview_max_pixels: int = OVERVIEW_MAX_PIXELS,
        overview_margin_px: int = OVERVIEW_MARGIN_PX,
        overview_feather_px: int = OVERVIEW_FEATHER_PX,
        overview_refine: bool = True,
    ):
        if min_sharpness <= 0 or max_motion_px <= 0 or not 0.0 < min_overlap < 1.0:
            raise ValueError("thresholds must be positive and overlap in (0, 1)")
        if overview_max_side <= 0 or overview_max_pixels <= 0 or overview_margin_px < 0 or overview_feather_px <= 0:
            raise ValueError("overview limits must be positive")
        self.min_sharpness = float(min_sharpness)
        self.max_motion_px = float(max_motion_px)
        self.min_overlap = float(min_overlap)
        self.overview_max_side = int(overview_max_side)
        self.overview_max_pixels = int(overview_max_pixels)
        self.overview_margin_px = int(overview_margin_px)
        self.overview_feather_px = int(overview_feather_px)
        self.overview_refine = bool(overview_refine)
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
        self._overview: np.ndarray | None = None
        self._overview_mask: np.ndarray | None = None
        self._overview_origin: tuple[float, float] = (0.0, 0.0)
        self._overview_scale = 1.0
        self._overview_version = 0
        self.overview_refine_count = 0
        self._coverage_memory: CoverageMemory | None = None
        self._overview_residual: np.ndarray | None = None
        self._overview_sharpness: np.ndarray | None = None
        self._track_note = ""
        self._lock_residual = 0.0
        self._ref_to_frame: np.ndarray | None = None
        self._annotations: list[dict] = []
        self.element_alignment = None

    def start(self) -> None:
        self.started = True

    def reset(self) -> None:
        """Clear captured tiles and overview, keeping the same thresholds."""
        self.started = False
        self.tiles = []
        self.edges = []
        self.rejects = []
        self._prev = None
        self._last = None
        self._pose = None
        self._anchor_id = None
        self._polygon = None
        self._quads = []
        self._overlap = None
        self._track_ms = None
        self._tracked = False
        self.merge_ms = []
        self.chain_mask = None
        self.chain_origin = (0, 0)
        self._overview = None
        self._overview_mask = None
        self._overview_origin = (0.0, 0.0)
        self._overview_scale = 1.0
        self._overview_version = 0
        self.overview_refine_count = 0
        self._coverage_memory = None
        self._overview_residual = None
        self._overview_sharpness = None
        self._track_note = ""
        self._lock_residual = 0.0
        self._ref_to_frame = None
        self._annotations = []
        self.element_alignment = None

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
            message = _READY
        self._tracked = False
        self._overlap = None
        self._polygon = None
        self._quads = []
        self._track_ms = None
        self._anchor_id = None
        if self.started and self.tiles:
            self._track_union(frame)
            if self.started and sharp and motion is not None and motion <= self.max_motion_px:
                if not self._tracked:
                    message = self._track_note or _LOST
                elif float(self._overlap or 0.0) < self.min_overlap:
                    message = _NO_OVERLAP
                else:
                    message = _READY_OVERLAP
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
        # Heavy placement runs on the accepted still, not during preview.
        # Typed elements and fiducials are the authority; preview is only a hint.
        started = time.perf_counter()
        query_elements = extract_board_elements(frame, max_long_edge=self.extractor.max_long_edge)
        query_fiducials = detect_fiducials(frame)
        placed = self._place_accepted_still(frame, query_elements, query_fiducials)
        self._track_ms = float(self._track_ms or 0.0) + (time.perf_counter() - started) * 1000.0
        preview_ok = self._tracked and self._pose is not None and self._pose.ok and self._anchor_id is not None
        if placed is not None:
            self._install_typed_pose(frame, placed)
        elif not preview_ok:
            return self._reject(self._track_note or _LOST, overlap=self._overlap, elapsed_ms=self._track_ms or 0.0)
        if not self._tracked or self._pose is None or not self._pose.ok or self._anchor_id is None:
            return self._reject(self._track_note or _LOST, overlap=self._overlap, elapsed_ms=self._track_ms or 0.0)
        overlap = float(self._overlap or 0.0)
        if overlap < self.min_overlap:
            return self._reject(_NO_OVERLAP, overlap=overlap, elapsed_ms=self._track_ms or 0.0)
        candidate_m = self._candidate_m_to_ref(self._anchor_id, self._pose.M)
        coverage = evaluate_coverage_growth(
            self._coverage_memory,
            int(frame.shape[1]),
            int(frame.shape[0]),
            candidate_m,
        )
        if not coverage.ok:
            return self._reject(coverage.reason, overlap=overlap, elapsed_ms=self._track_ms or 0.0)
        anchor = next(item for item in self.tiles if item.tile_id == self._anchor_id)
        query_elements = self._check_red_elements(frame, anchor, self._pose.M, query_elements=query_elements)
        verdict = self.element_alignment
        if verdict is not None and verdict.status == "lattice_only":
            return self._reject(_UNCERTAIN, overlap=overlap, elapsed_ms=self._track_ms or 0.0)
        if verdict is not None and verdict.status == "disagree" and not self._ignore_flaky_secondary(anchor, frame, verdict):
            return self._reject(_ELEMENT_MISMATCH, overlap=overlap, elapsed_ms=self._track_ms or 0.0)
        tile = self._store_tile(frame, self._anchor_id, self._pose.M, coverage_check=coverage, elements=query_elements)
        self._close_loops(tile)
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

    def overview_image(self) -> np.ndarray | None:
        """Color board overview fused so far. Constrained preview, not a full canvas."""
        return self._overview

    def overview_meta(self) -> dict:
        image = self._overview
        return {
            "width": None if image is None else int(image.shape[1]),
            "height": None if image is None else int(image.shape[0]),
            "tiles": len(self.tiles),
            "max_side": int(self.overview_max_side),
            "max_pixels": int(self.overview_max_pixels),
            "scale": None if image is None else round(float(self._overview_scale), 5),
            "refine_count": int(self.overview_refine_count),
            "full_canvas": False,
        }

    def chain_dict(self) -> dict:
        return {
            "tiles": [
                {
                    "id": tile.tile_id,
                    "image_size": {"width": tile.width, "height": tile.height},
                    "points": len(tile.points),
                    "M_to_ref": np.asarray(tile.M_to_ref, dtype=np.float64).round(6).tolist(),
                    "elements": None if getattr(tile, "elements", None) is None else tile.elements.to_dict(),
                }
                for tile in self.tiles
            ],
            "edges": self.edges,
            "rejects": self.rejects,
            "coverage_memory": None if self._coverage_memory is None else self._coverage_memory.to_dict(),
            "annotations": self.board_annotations(),
            "structure_tree": [
                None if getattr(tile, "structure", None) is None else tile.structure.to_dict()
                for tile in self.tiles
            ],
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
        """Lock this frame to the board and project every captured tile."""
        started = time.perf_counter()
        observed = self.extractor.extract(frame)
        structure = extract_structure_tree(frame)
        query_xy, query_w = self._stack_cloud(observed.xy(), landmarks_from_snapshot(structure))
        library_xy, library_w = self._board_cloud()
        estimate = estimate_similarity_pose(
            library_xy,
            query_xy,
            int(frame.shape[1]),
            int(frame.shape[0]),
            library_w=library_w,
            query_w=query_w,
        )
        width = int(frame.shape[1])
        height = int(frame.shape[0])
        # Typed elements and corner fiducials place the still. Untyped ORB/SIFT
        # is not the lock that decides the pose.
        # Preview tracking stays cheap: no element separation here.
        typed = None
        self._track_ms = (time.perf_counter() - started) * 1000.0
        self._lock_residual = float(estimate.residual)
        self._track_note = ""
        self._ref_to_frame = None
        ref_to_frame = None
        hits = int(estimate.hits)
        confidence = float(estimate.precision)
        if typed is not None:
            ref_to_frame = np.asarray(typed["ref_to_frame"], dtype=np.float64)
            self._lock_residual = float(typed["residual"])
            hits = int(typed.get("hits", hits))
            confidence = float(typed.get("ncc", confidence))
        elif estimate.status == "ok" and not alignment_rejected(estimate.matrix):
            ref_to_frame = np.asarray(estimate.matrix, dtype=np.float64)
            self._lock_residual = float(estimate.residual)
        elif estimate.status == "uncertain":
            self._pose = None
            self._track_note = _UNCERTAIN
            return
        if ref_to_frame is None:
            self._pose = None
            return
        anchor, anchor_to_frame = self._select_anchor(ref_to_frame, width, height)
        if anchor is None or anchor_to_frame is None or alignment_rejected(anchor_to_frame):
            self._pose = None
            self._track_note = _UNCERTAIN
            return
        self.matcher.set_topology(anchor.extracted)
        self._ref_to_frame = ref_to_frame
        self._anchor_id = anchor.tile_id
        self._pose = Pose2D(
            M=np.asarray(anchor_to_frame, dtype=np.float64),
            inliers=hits,
            confidence=confidence,
            ok=True,
            message="ok",
            elapsed_ms=float(self._track_ms),
            scale=float(np.linalg.norm(np.asarray(anchor_to_frame)[:, 0])),
            angle_deg=pose_angle_deg(anchor_to_frame),
            query_points=int(len(query_xy)),
            pose_error_px=float(self._lock_residual),
        )
        self._quads = [
            project_tile_quad(_compose_m(ref_to_frame, tile.M_to_ref), tile.width, tile.height) for tile in self.tiles
        ]
        self._quads = self._filter_quads_by_coverage_memory(self._quads, ref_to_frame, width, height)
        self._polygon = self._quads[-1] if self._quads else None
        self._overlap = union_overlap_ratio(self._quads, width, height)
        self._tracked = True

    def _select_anchor(self, ref_to_frame: np.ndarray, width: int, height: int) -> tuple[_Tile | None, np.ndarray | None]:
        """Prefer the previous tile when it still covers a modest part of the frame."""
        best: _Tile | None = None
        best_fraction = -1.0
        best_matrix = None
        previous = self.tiles[-1]
        previous_matrix = _compose_m(ref_to_frame, previous.M_to_ref)
        previous_fraction = frame_overlap_ratio(
            project_tile_quad(previous_matrix, previous.width, previous.height),
            width,
            height,
        )
        for tile in self.tiles:
            matrix = _compose_m(ref_to_frame, tile.M_to_ref)
            fraction = frame_overlap_ratio(project_tile_quad(matrix, tile.width, tile.height), width, height)
            if fraction > best_fraction:
                best = tile
                best_fraction = fraction
                best_matrix = matrix
        # Filename order is only a hint. Prefer the previous tile when it still
        # overlaps about as well as the best match; otherwise use the best one.
        if previous_fraction >= self.min_overlap and previous_fraction + 0.05 >= best_fraction:
            return previous, previous_matrix
        return best, best_matrix

    def _feature_lock(self, frame: np.ndarray, width: int, height: int) -> dict | None:
        """ORB+CLAHE pose against accepted stills, ranked by overlap NCC."""
        if not self.tiles:
            return None
        query_pack = extract_feature_pack(frame)
        hypotheses = []
        for tile in self._feature_anchor_tiles():
            pose = estimate_feature_pose(
                tile.raw,
                frame,
                library_pack=getattr(tile, "feature_pack", None),
                query_pack=query_pack,
            )
            if not pose.ok or alignment_rejected(pose.matrix):
                continue
            ref_to_frame = _compose_m(np.asarray(pose.matrix, dtype=np.float64), _invert_m(tile.M_to_ref))
            item = {
                "tile": tile,
                "pose": pose,
                "ref_to_frame": ref_to_frame,
                "ncc": float(pose.ncc),
                "residual": float(pose.residual),
                "hits": int(pose.hits),
                "area": int(pose.area),
            }
            hypotheses.append(item)
        if not hypotheses:
            return None
        hypotheses.sort(key=lambda item: (item["ncc"], item["hits"]), reverse=True)
        best = hypotheses[0]
        # Two equally good, disagreeing feature poses are not a unique lock.
        sample = tile_corners(width, height)
        for other in hypotheses[1:]:
            close = other["ncc"] >= 0.50 and (best["ncc"] - other["ncc"]) <= 0.08
            if close and not _poses_consistent(best["ref_to_frame"], other["ref_to_frame"], sample):
                return None
        return best

    def _feature_anchor_tiles(self) -> list[_Tile]:
        """Every accepted still. A frame may belong next to an earlier tile, not the last one."""
        tiles = [tile for tile in self.tiles if tile.raw is not None]
        return list(reversed(tiles))

    def _feature_breaks_ambiguity(self, feature: dict) -> bool:
        """Only unique texture may override a constellation that looks like a lattice."""
        return float(feature.get("ncc", 0.0)) >= 0.60 and int(feature.get("hits", 0)) >= 16

    def _place_accepted_still(self, frame: np.ndarray, query_elements, query_fiducials) -> dict | None:
        """Choose a unique board-frame pose for this still.

        Typed marks and fiducials place the shot. A periodic lattice never
        does. Untyped ORB is not the placement authority.
        """
        return self._typed_lock(frame, query_elements, query_fiducials)

    def _typed_lock(self, frame: np.ndarray, query_elements, query_fiducials) -> dict | None:
        """Place this frame into the board frame by typed marks, not untyped keypoints."""
        if not self.tiles or query_elements is None or not getattr(query_elements, "elements", None):
            return None
        width = int(frame.shape[1])
        height = int(frame.shape[0])
        hypotheses = []
        for tile in self._typed_anchor_tiles(query_elements, query_fiducials):
            reference = getattr(tile, "elements", None)
            if (reference is None or not getattr(reference, "elements", None)) and tile.raw is not None:
                reference = extract_board_elements(tile.raw, max_long_edge=self.extractor.max_long_edge)
                tile.elements = reference
            if reference is None or not getattr(reference, "elements", None):
                continue
            result = estimate_typed_pose(
                reference,
                query_elements,
                reference_fiducials=getattr(tile, "fiducials", None),
                query_fiducials=query_fiducials,
            )
            if result.matrix is None or alignment_rejected(result.matrix):
                continue
            if result.status == "lattice_only":
                continue
            if result.status not in {"ok", "disagree"}:
                continue
            matrix = np.asarray(result.matrix, dtype=np.float64)
            overlap = frame_overlap_ratio(project_tile_quad(matrix, tile.width, tile.height), width, height)
            if overlap < self.min_overlap:
                continue
            ncc, area = masked_ncc(tile.raw, frame, matrix) if tile.raw is not None else (0.0, 0)
            if result.status == "disagree" and not self._typed_disagreement_is_flaky(result, ncc):
                continue
            landmarks = unique_landmark_support(
                reference,
                query_elements,
                reference_fiducials=getattr(tile, "fiducials", None),
                query_fiducials=query_fiducials,
            )
            # Only marks that sit on this pose may relax the photometric check.
            # Shared-looking glyphs elsewhere must not excuse a false pad lock.
            unique = result.secondary_hits >= 1 or (
                result.status == "ok"
                and query_fiducials
                and getattr(tile, "fiducials", None)
            )
            if unique:
                # Microscope lighting often weakens NCC when silkscreen or
                # fiducials already agree.
                if result.status != "ok" and ncc < 0.22:
                    continue
            else:
                # Pad/via-only locks must be tight and photometrically strong
                # so a lookalike period is not accepted as a unique pose.
                frame_area = float(max(1, width * height))
                if ncc < 0.55 or result.residual > 2.0 or area < 0.08 * frame_area:
                    continue
                if result.primary_hits < 8 and ncc < 0.85:
                    continue
            hypotheses.append(
                {
                    "tile": tile,
                    "result": result,
                    "ref_to_frame": _compose_m(matrix, _invert_m(tile.M_to_ref)),
                    "ncc": float(ncc),
                    "residual": float(result.residual),
                    "hits": int(result.primary_hits + result.secondary_hits),
                    "area": int(area),
                    "overlap": float(overlap),
                    "landmarks": int(landmarks),
                }
            )
        if not hypotheses:
            return None
        hypotheses.sort(
            key=lambda item: (item["result"].status == "ok", item["ncc"], item["hits"], item["landmarks"], -item["residual"]),
            reverse=True,
        )
        best = hypotheses[0]
        sample = tile_corners(width, height)
        for other in hypotheses[1:]:
            if _poses_consistent(best["ref_to_frame"], other["ref_to_frame"], sample):
                continue
            unique_best = best["result"].secondary_hits >= 1 or best["landmarks"] >= 2
            unique_other = other["result"].secondary_hits >= 1 or other["landmarks"] >= 2
            if unique_best and not unique_other:
                continue
            ncc_close = other["ncc"] >= 0.50 and (best["ncc"] - other["ncc"]) <= 0.08
            typed_close = (
                other["result"].status == "ok"
                and best["result"].status == "ok"
                and other["hits"] >= best["hits"] - 1
                and abs(other["residual"] - best["residual"]) <= 0.8
                and other["ncc"] >= 0.35
            )
            # Two equally good, disagreeing placements are not a unique lock.
            if ncc_close or typed_close:
                return None
        return best

    def _typed_disagreement_is_flaky(self, verdict, ncc: float) -> bool:
        if ncc < 0.55 or float(getattr(verdict, "residual", 99.0)) > 2.5:
            return False
        message = str(getattr(verdict, "message", "") or "")
        if "primary seam disagrees" in message:
            return ncc >= 0.60 and float(verdict.residual) <= 2.0
        if "secondary seam disagrees:" not in message:
            return False
        kinds = {item.strip() for item in message.split(":", 1)[-1].split(",") if item.strip()}
        if not kinds or kinds - {"outline", "trace", "text"}:
            return False
        if "text" in kinds and _coherent_text_shift(verdict):
            return False
        return True

    def _typed_anchor_tiles(self, query_elements, query_fiducials) -> list[_Tile]:
        """Prefer unique-landmark neighbors; still search earlier tiles for out-of-order shots."""
        usable = []
        for tile in self.tiles:
            if tile.raw is None:
                continue
            usable.append(tile)
        if len(usable) <= 12:
            return list(reversed(usable))
        recent = list(reversed(usable[-8:]))
        ranked = []
        for tile in usable:
            score = unique_landmark_support(
                getattr(tile, "elements", None),
                query_elements,
                reference_fiducials=getattr(tile, "fiducials", None),
                query_fiducials=query_fiducials,
            )
            ranked.append((score, tile))
        ranked.sort(key=lambda item: item[0], reverse=True)
        chosen: list[_Tile] = []
        seen: set[int] = set()
        for tile in recent:
            marker = id(tile)
            if marker not in seen:
                seen.add(marker)
                chosen.append(tile)
        for score, tile in ranked:
            if score <= 0:
                continue
            marker = id(tile)
            if marker in seen:
                continue
            seen.add(marker)
            chosen.append(tile)
            if len(chosen) >= 16:
                break
        return chosen

    def _stack_cloud(self, points: np.ndarray, landmarks: np.ndarray | None, transform: np.ndarray | None = None):

        chunks: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        xy = np.asarray(points, dtype=np.float64).reshape(-1, 2) if points is not None and len(points) else np.zeros((0, 2))
        if len(xy):
            if transform is not None:
                xy = _apply_m(transform, xy)
            chunks.append(xy)
            weights.append(np.ones(len(xy), dtype=np.float64))
        marks = np.zeros((0, 2), dtype=np.float64) if landmarks is None else np.asarray(landmarks, dtype=np.float64).reshape(-1, 2)
        if len(marks):
            if transform is not None:
                marks = _apply_m(transform, marks)
            chunks.append(marks)
            weights.append(np.full(len(marks), 4.0))
        if not chunks:
            return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)
        return np.vstack(chunks), np.concatenate(weights)

    def _board_cloud(self):
        chunks: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for tile in self.tiles:
            xy, weight = self._stack_cloud(_tile_xy(tile), getattr(tile, "landmarks", None), tile.M_to_ref)
            if len(xy):
                chunks.append(xy)
                weights.append(weight)
        if not chunks:
            return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)
        return np.vstack(chunks), np.concatenate(weights)

    def _localize_anchors(self, observed) -> list[_AnchorHypothesis]:
        """Score every usable tile localization; do not stop at the first ok pose."""
        hypotheses = []
        for tile in self._candidate_anchor_tiles():
            self.matcher.set_topology(tile.extracted)
            candidate = self.matcher.localize(observed)
            if not candidate.ok or alignment_rejected(candidate.M):
                continue
            index = self.tiles.index(tile)
            hypotheses.append(
                _AnchorHypothesis(
                    tile=tile,
                    index=index,
                    pose=candidate,
                    score=_localization_score(candidate),
                    ref_to_frame=_compose_m(
                        np.asarray(candidate.M, dtype=np.float64),
                        _invert_m(tile.M_to_ref),
                    ),
                )
            )
        hypotheses.sort(key=lambda item: (item.score, item.pose.confidence, item.pose.inliers), reverse=True)
        return hypotheses

    def _candidate_anchor_tiles(self) -> list[_Tile]:
        tiles = [tile for tile in self.tiles if tile.extracted is not None]
        if not tiles:
            return []
        if len(tiles) <= _ANCHOR_MAX_ALL:
            return list(reversed(tiles))
        chosen: list[_Tile] = []
        seen: set[int] = set()

        def add(tile: _Tile) -> None:
            marker = id(tile)
            if marker not in seen:
                seen.add(marker)
                chosen.append(tile)

        for tile in reversed(tiles[-_ANCHOR_RECENT:]):
            add(tile)
        earlier = tiles[:-_ANCHOR_RECENT]
        if earlier:
            add(earlier[0])
            for tile in earlier[_ANCHOR_EARLIER_STRIDE::_ANCHOR_EARLIER_STRIDE]:
                add(tile)
        return chosen

    def _consensus_ref_to_frame(
        self,
        hypotheses: list[_AnchorHypothesis],
        width: int,
        height: int,
    ) -> tuple[_AnchorHypothesis | None, np.ndarray | None]:
        if not hypotheses:
            return None, None
        primary = hypotheses[0]
        sample = _apply_m(primary.tile.M_to_ref, tile_corners(primary.tile.width, primary.tile.height))
        # If the primary tile is tiny, also compare a live-frame-sized box in ref space.
        if float(np.linalg.norm(sample[2] - sample[0])) < 80.0 and width > 1 and height > 1:
            frame_corners = tile_corners(width, height)
            try:
                sample = np.vstack([sample, _apply_m(_invert_m(primary.ref_to_frame), frame_corners)])
            except np.linalg.LinAlgError:
                pass
        supporters = [primary]
        for other in hypotheses[1:]:
            if _poses_consistent(primary.ref_to_frame, other.ref_to_frame, sample):
                supporters.append(other)
                continue
            if _is_strong_rival(other, primary):
                return None, None
        ref_to_frame = _blend_ref_to_frame([(item.ref_to_frame, item.score) for item in supporters])
        return primary, ref_to_frame

    def _verified_coverage_quads(
        self,
        ref_to_frame: np.ndarray,
        observed,
        width: int,
        height: int,
        *,
        primary_index: int,
    ) -> list[np.ndarray]:
        """Paint only tiles that have point support and stay connected to the anchor overlap."""
        observed_xy = observed.xy() if observed is not None else np.zeros((0, 2), dtype=np.float64)
        quads: list[np.ndarray | None] = []
        trusted: list[bool] = []
        for index, tile in enumerate(self.tiles):
            tile_to_frame = _compose_m(ref_to_frame, tile.M_to_ref)
            quad = project_tile_quad(tile_to_frame, tile.width, tile.height)
            quads.append(quad)
            if index == primary_index:
                trusted.append(True)
                continue
            trusted.append(_tile_is_trusted_in_view(tile, tile_to_frame, quad, observed_xy, width, height))
        accepted = [False] * len(self.tiles)
        if 0 <= primary_index < len(self.tiles):
            accepted[primary_index] = True
        mask = np.zeros((height, width), dtype=np.uint8)
        if accepted[primary_index]:
            _fill_quad(mask, quads[primary_index])
        changed = True
        while changed:
            changed = False
            for index, quad in enumerate(quads):
                if accepted[index] or not trusted[index] or quad is None:
                    continue
                if _quad_touches_mask(quad, mask, touch_px=_COVERAGE_TOUCH_PX):
                    accepted[index] = True
                    _fill_quad(mask, quad)
                    changed = True
        return [quads[index] for index, keep in enumerate(accepted) if keep]

    def _quads_from_anchor(self, anchor: _Tile, anchor_to_frame: np.ndarray) -> list[np.ndarray]:
        """Map every captured tile into the live frame through the chain frame."""
        ref_to_frame = _compose_m(np.asarray(anchor_to_frame, dtype=np.float64), _invert_m(anchor.M_to_ref))
        quads = []
        for tile in self.tiles:
            tile_to_frame = _compose_m(ref_to_frame, tile.M_to_ref)
            quads.append(project_tile_quad(tile_to_frame, tile.width, tile.height))
        return quads

    def _store_tile(
        self,
        frame: np.ndarray,
        parent: str | None,
        pose_m: np.ndarray | None,
        coverage_check: CoverageCheck | None = None,
        elements=None,
    ) -> _Tile:
        extracted = self.extractor.extract(frame)
        extracted.board_id = f"cap-{len(self.tiles):03d}"
        structure = extract_structure_tree(frame)
        tile = _Tile(
            tile_id=extracted.board_id,
            width=int(frame.shape[1]),
            height=int(frame.shape[0]),
            raw=frame.copy(),
            binary=binary_lines(frame),
            points=[point.to_dict() for point in extracted.points],
            extracted=extracted,
            structure=structure,
            landmarks=landmarks_from_snapshot(structure),
            align_residual_px=0.0 if parent is None else float(self._lock_residual),
            sharpness=frame_sharpness(frame),
            elements=elements if elements is not None else extract_board_elements(frame, max_long_edge=self.extractor.max_long_edge),
            feature_pack=extract_feature_pack(frame),
            fiducials=detect_fiducials(frame),
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
        # Edge compensation is a preview only. The color tile stays in the same
        # pixels as the geometric pose, otherwise pins and silkscreen shear.
        tile.corrected = np.ascontiguousarray(corrected).copy()
        tile.M_overview = np.asarray(tile.M_to_ref, dtype=np.float64).copy()
        tile.edge_before = edge_stats.get("preview_before")
        tile.edge_after = edge_stats.get("preview_after")
        tile.edge_stats = {key: value for key, value in edge_stats.items() if not str(key).startswith("preview")}
        if self.edges:
            self.edges[-1]["edge"] = tile.edge_stats
        self.tiles.append(tile)
        merge_started = time.perf_counter()
        self._fuse_chain_mask(tile)
        self._fuse_overview(tile)
        merge_ms = (time.perf_counter() - merge_started) * 1000.0
        self.merge_ms.append(merge_ms)
        if self.edges:
            self.edges[-1]["merge_ms"] = round(merge_ms, 2)
        self.matcher.set_topology(extracted)
        self._anchor_id = tile.tile_id
        self._commit_coverage_memory(tile, coverage_check)
        self._pose_after_capture(tile)
        return tile

    def _pose_after_capture(self, tile: _Tile) -> None:
        """The shutter frame is the new tile, so every earlier tile can be drawn at once."""
        self._quads = self._quads_from_anchor(tile, _identity_m())
        if self._coverage_memory is not None and self._quads:
            ref_to_frame = _compose_m(_identity_m(), _invert_m(tile.M_to_ref))
            self._quads = self._filter_quads_by_coverage_memory(
                self._quads,
                ref_to_frame,
                tile.width,
                tile.height,
            )
        self._polygon = self._quads[-1] if self._quads else None
        self._overlap = 1.0
        self._tracked = True

    def _candidate_m_to_ref(self, parent: str, pose_m: np.ndarray) -> np.ndarray:
        anchor = next(item for item in self.tiles if item.tile_id == parent)
        return _compose_m(anchor.M_to_ref, _invert_m(np.asarray(pose_m, dtype=np.float64)))

    def _commit_coverage_memory(self, tile: _Tile, coverage_check: CoverageCheck | None) -> None:
        if coverage_check is not None and coverage_check.ok and coverage_check.memory is not None:
            self._coverage_memory = coverage_check.memory
            return
        self._coverage_memory = build_coverage_memory(
            self._coverage_memory,
            tile.width,
            tile.height,
            tile.M_to_ref,
        )

    def _filter_quads_by_coverage_memory(
        self,
        quads: list[np.ndarray],
        ref_to_frame: np.ndarray,
        width: int,
        height: int,
    ) -> list[np.ndarray]:
        """Drop sudden extra red blocks that the remembered world shape cannot explain."""
        memory = self._coverage_memory
        if memory is None or memory.area <= 0 or not quads or width <= 1 or height <= 1:
            return quads
        projected = project_coverage_memory(memory, ref_to_frame, width, height)
        if not np.any(projected):
            return quads
        actual = np.zeros((height, width), dtype=np.uint8)
        for quad in quads:
            _fill_quad(actual, quad)
        tile_area = int(self.tiles[0].width * self.tiles[0].height) if self.tiles else width * height
        min_area = _coverage_blob_min_area(tile_area)
        blobs = unexplained_coverage_blobs(
            actual,
            (0, 0),
            projected,
            (0, 0),
            slack_px=_COVERAGE_EDGE_SLACK_PX,
            min_area=min_area,
        )
        if not np.any(blobs):
            return quads
        kept: list[np.ndarray] = []
        for quad in quads:
            probe = np.zeros((height, width), dtype=np.uint8)
            _fill_quad(probe, quad)
            extra = int(np.count_nonzero((probe > 0) & (blobs > 0)))
            visible = int(np.count_nonzero(probe))
            if extra >= min_area and visible > 0 and extra >= int(0.2 * visible):
                continue
            kept.append(quad)
        return kept

    @property
    def coverage_memory(self) -> CoverageMemory | None:
        return self._coverage_memory

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


    def _tile_color(self, tile: _Tile) -> np.ndarray:
        image = tile.raw
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def _overview_pose(self, tile: _Tile) -> np.ndarray:
        if tile.M_overview is not None:
            return np.asarray(tile.M_overview, dtype=np.float64)
        return np.asarray(tile.M_to_ref, dtype=np.float64)

    def _world_bounds(self, tiles: list[_Tile] | None = None) -> tuple[float, float, float, float] | None:
        items = self.tiles if tiles is None else tiles
        if not items:
            return None
        corners = [project_tile_quad(self._overview_pose(tile), tile.width, tile.height) for tile in items]
        points = np.vstack(corners)
        return (
            float(np.min(points[:, 0])),
            float(np.min(points[:, 1])),
            float(np.max(points[:, 0])),
            float(np.max(points[:, 1])),
        )

    def _fit_overview_scale(self, span_x: float, span_y: float) -> float:
        span_x = max(float(span_x), 1.0)
        span_y = max(float(span_y), 1.0)
        scale = 1.0
        scale = min(scale, self.overview_max_side / span_x, self.overview_max_side / span_y)
        scale = min(scale, math.sqrt(self.overview_max_pixels / (span_x * span_y)))
        return max(0.05, min(1.0, scale))

    def _canvas_size(self, bounds: tuple[float, float, float, float], scale: float) -> tuple[int, int]:
        width = max(1, int(math.ceil((bounds[2] - bounds[0]) * scale)))
        height = max(1, int(math.ceil((bounds[3] - bounds[1]) * scale)))
        return width, height

    def _plan_overview(self, bounds_raw: tuple[float, float, float, float]) -> tuple[tuple[float, float, float, float], float, tuple[int, int]]:
        min_x, min_y, max_x, max_y = bounds_raw
        raw_w = max(max_x - min_x, 1.0)
        raw_h = max(max_y - min_y, 1.0)
        scale = 1.0
        for _ in range(4):
            margin = self.overview_margin_px / max(scale, 1e-6)
            scale = self._fit_overview_scale(raw_w + 2.0 * margin, raw_h + 2.0 * margin)
        margin = self.overview_margin_px / max(scale, 1e-6)
        bounds = (min_x - margin, min_y - margin, max_x + margin, max_y + margin)
        size = self._canvas_size(bounds, scale)
        while (
            size[0] > self.overview_max_side
            or size[1] > self.overview_max_side
            or size[0] * size[1] > self.overview_max_pixels
        ) and scale > 0.05:
            scale = max(0.05, scale * 0.95)
            margin = self.overview_margin_px / max(scale, 1e-6)
            bounds = (min_x - margin, min_y - margin, max_x + margin, max_y + margin)
            size = self._canvas_size(bounds, scale)
        return bounds, scale, size

    def _world_to_overview(self) -> np.ndarray:
        scale = float(self._overview_scale)
        origin_x, origin_y = self._overview_origin
        return np.array(
            [[scale, 0.0, -origin_x * scale], [0.0, scale, -origin_y * scale]],
            dtype=np.float64,
        )

    def _allocate_overview(self, origin: tuple[float, float], scale: float, size: tuple[int, int]) -> None:
        width, height = size
        self._overview = np.zeros((height, width, 3), dtype=np.uint8)
        self._overview_mask = np.zeros((height, width), dtype=np.uint8)
        self._overview_residual = np.full((height, width), 99.0, dtype=np.float32)
        self._overview_sharpness = np.zeros((height, width), dtype=np.float32)
        self._overview_origin = (float(origin[0]), float(origin[1]))
        self._overview_scale = float(scale)

    def _expand_overview(self, new_origin: tuple[float, float], new_size: tuple[int, int]) -> None:
        assert self._overview is not None and self._overview_mask is not None
        old = self._overview
        old_mask = self._overview_mask
        old_residual = self._overview_residual
        old_sharpness = self._overview_sharpness
        old_origin = self._overview_origin
        scale = self._overview_scale
        new_w, new_h = new_size
        dx = int(round((old_origin[0] - new_origin[0]) * scale))
        dy = int(round((old_origin[1] - new_origin[1]) * scale))
        canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        mask = np.zeros((new_h, new_w), dtype=np.uint8)
        residual = np.full((new_h, new_w), 99.0, dtype=np.float32)
        sharpness = np.zeros((new_h, new_w), dtype=np.float32)
        x0, y0 = max(0, dx), max(0, dy)
        sx0, sy0 = max(0, -dx), max(0, -dy)
        copy_w = min(old.shape[1] - sx0, new_w - x0)
        copy_h = min(old.shape[0] - sy0, new_h - y0)
        if copy_w > 0 and copy_h > 0:
            canvas[y0 : y0 + copy_h, x0 : x0 + copy_w] = old[sy0 : sy0 + copy_h, sx0 : sx0 + copy_w]
            mask[y0 : y0 + copy_h, x0 : x0 + copy_w] = old_mask[sy0 : sy0 + copy_h, sx0 : sx0 + copy_w]
            if old_residual is not None:
                residual[y0 : y0 + copy_h, x0 : x0 + copy_w] = old_residual[sy0 : sy0 + copy_h, sx0 : sx0 + copy_w]
            if old_sharpness is not None:
                sharpness[y0 : y0 + copy_h, x0 : x0 + copy_w] = old_sharpness[sy0 : sy0 + copy_h, sx0 : sx0 + copy_w]
        self._overview = canvas
        self._overview_mask = mask
        self._overview_residual = residual
        self._overview_sharpness = sharpness
        self._overview_origin = (float(new_origin[0]), float(new_origin[1]))

    def _rebuild_overview(self, tiles: list[_Tile], scale: float) -> None:
        bounds_raw = self._world_bounds(tiles)
        if bounds_raw is None:
            self._overview = None
            self._overview_mask = None
            return
        bounds, planned, size = self._plan_overview(bounds_raw)
        scale = min(float(scale), planned)
        bounds = (
            bounds_raw[0] - self.overview_margin_px / max(scale, 1e-6),
            bounds_raw[1] - self.overview_margin_px / max(scale, 1e-6),
            bounds_raw[2] + self.overview_margin_px / max(scale, 1e-6),
            bounds_raw[3] + self.overview_margin_px / max(scale, 1e-6),
        )
        size = self._canvas_size(bounds, scale)
        self._allocate_overview((bounds[0], bounds[1]), scale, size)
        for item in tiles:
            self._blend_overview_tile(item)

    def _grow_overview(self, bounds_raw: tuple[float, float, float, float], settled: list[_Tile]) -> None:
        if self._overview is None:
            return
        margin = self.overview_margin_px / max(self._overview_scale, 1e-6)
        old_x, old_y = self._overview_origin
        old_max_x = old_x + self._overview.shape[1] / max(self._overview_scale, 1e-6)
        old_max_y = old_y + self._overview.shape[0] / max(self._overview_scale, 1e-6)
        new_origin = (min(old_x, bounds_raw[0] - margin), min(old_y, bounds_raw[1] - margin))
        new_max = (max(old_max_x, bounds_raw[2] + margin), max(old_max_y, bounds_raw[3] + margin))
        new_size = self._canvas_size((new_origin[0], new_origin[1], new_max[0], new_max[1]), self._overview_scale)
        exceeds = (
            new_size[0] > self.overview_max_side
            or new_size[1] > self.overview_max_side
            or new_size[0] * new_size[1] > self.overview_max_pixels
        )
        if exceeds:
            _bounds, scale, _size = self._plan_overview(bounds_raw)
            self._rebuild_overview(settled, scale)
            return
        if (
            new_size[0] != self._overview.shape[1]
            or new_size[1] != self._overview.shape[0]
            or new_origin[0] < old_x - 1e-6
            or new_origin[1] < old_y - 1e-6
        ):
            self._expand_overview(new_origin, new_size)

    def _blend_overview_tile(self, tile: _Tile) -> None:
        """Place one tile. Overlap is copied from the better-aligned source, never averaged."""
        if self._overview is None or self._overview_mask is None:
            return
        if self._overview_residual is None or self._overview_sharpness is None:
            self._overview_residual = np.full(self._overview.shape[:2], 99.0, dtype=np.float32)
            self._overview_sharpness = np.zeros(self._overview.shape[:2], dtype=np.float32)
        image = self._tile_color(tile)
        transform = _compose_m(self._world_to_overview(), self._overview_pose(tile))
        composite_registered_tile(
            self._overview,
            self._overview_mask,
            self._overview_residual,
            self._overview_sharpness,
            image,
            transform,
            residual_px=float(getattr(tile, "align_residual_px", 0.0)),
            sharpness=float(getattr(tile, "sharpness", 0.0)),
            safe_residual_px=SAFE_RESIDUAL_PX,
        )
        self._overview_version += 1

    def _refine_overview_tile(self, tile: _Tile) -> None:
        """Light ECC on overlap edges only. Failures are skipped; M_to_ref is unchanged."""
        if (
            not self.overview_refine
            or self._overview is None
            or self._overview_mask is None
            or not np.any(self._overview_mask)
        ):
            return
        image = self._tile_color(tile)
        placed = _compose_m(self._world_to_overview(), self._overview_pose(tile))
        size = (self._overview.shape[1], self._overview.shape[0])
        height, width = image.shape[:2]
        warped = cv2.warpAffine(
            image,
            placed,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        mask = cv2.warpAffine(
            np.full((height, width), 255, dtype=np.uint8),
            placed,
            size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        overlap = ((mask > 0) & (self._overview_mask > 0)).astype(np.uint8)
        if int(np.count_nonzero(overlap)) < 2500:
            return
        x, y, roi_w, roi_h = cv2.boundingRect(overlap)
        pad = 8
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(self._overview.shape[1], x + roi_w + pad)
        y1 = min(self._overview.shape[0], y + roi_h + pad)
        if x1 - x0 < 24 or y1 - y0 < 24:
            return
        template = cv2.cvtColor(self._overview[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        moving = cv2.cvtColor(warped[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), dtype=np.uint8)
        template_edges = cv2.dilate(cv2.Canny(template, 35, 110), kernel, iterations=1)
        moving_edges = cv2.dilate(cv2.Canny(moving, 35, 110), kernel, iterations=1)
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4)
        try:
            score, warp = cv2.findTransformECC(
                template_edges.astype(np.float32),
                moving_edges.astype(np.float32),
                warp,
                cv2.MOTION_EUCLIDEAN,
                criteria,
                overlap[y0:y1, x0:x1],
                5,
            )
        except cv2.error:
            return
        if not np.isfinite(score) or float(score) < 0.18:
            return
        shift = math.hypot(float(warp[0, 2]), float(warp[1, 2]))
        angle = abs(math.degrees(math.atan2(float(warp[1, 0]), float(warp[0, 0]))))
        if shift > 16.0 or angle > 2.5:
            return
        if shift < 0.05 and angle < 0.02:
            return
        correction = _compose_m(_translate_m(x0, y0), _compose_m(np.asarray(warp, dtype=np.float64), _translate_m(-x0, -y0)))
        try:
            tile.M_overview = _compose_m(_invert_m(self._world_to_overview()), _compose_m(correction, placed))
        except np.linalg.LinAlgError:
            return
        self.overview_refine_count += 1

    def _fuse_overview(self, tile: _Tile) -> None:
        """Warp this tile's color into a constrained overview; shrink and rebuild instead of dropping."""
        if tile.M_overview is None:
            tile.M_overview = np.asarray(tile.M_to_ref, dtype=np.float64).copy()
        settled = [item for item in self.tiles if item is not tile]
        bounds_raw = self._world_bounds(self.tiles)
        if bounds_raw is None:
            return
        if self._overview is None or not settled:
            bounds, scale, size = self._plan_overview(self._world_bounds([tile]) or bounds_raw)
            self._allocate_overview((bounds[0], bounds[1]), scale, size)
            self._blend_overview_tile(tile)
            return
        _planned_bounds, planned_scale, _planned_size = self._plan_overview(bounds_raw)
        if planned_scale < self._overview_scale * 0.999:
            self._rebuild_overview(settled, planned_scale)
        self._grow_overview(bounds_raw, settled)
        updated = self._world_bounds(self.tiles)
        if updated is not None:
            self._grow_overview(updated, settled)
        self._blend_overview_tile(tile)

    def _pair_constraint(self, earlier: _Tile, later: _Tile):
        """Typed or fiducial correspondences between two overlapping tiles."""
        typed = None
        if getattr(earlier, "elements", None) is not None and getattr(later, "elements", None) is not None:
            typed = estimate_typed_pose(
                earlier.elements,
                later.elements,
                reference_fiducials=getattr(earlier, "fiducials", None),
                query_fiducials=getattr(later, "fiducials", None),
            )
        if typed is not None and typed.status == "ok" and typed.residual <= 3.2:
            src_xy, dst_xy = typed_correspondences(earlier.elements, later.elements, typed.matrix)
            fid_src, fid_dst = fiducial_correspondences(
                getattr(earlier, "fiducials", None),
                getattr(later, "fiducials", None),
                typed.matrix,
            )
            if len(fid_src):
                src_xy = np.vstack([src_xy, fid_src]) if len(src_xy) else fid_src
                dst_xy = np.vstack([dst_xy, fid_dst]) if len(dst_xy) else fid_dst
            if len(src_xy) >= 4:
                return src_xy, dst_xy
        if typed is not None and typed.status == "ok" and getattr(earlier, "fiducials", None) and getattr(later, "fiducials", None):
            fid_src, fid_dst = fiducial_correspondences(
                earlier.fiducials,
                later.fiducials,
                typed.matrix,
            )
            if len(fid_src) >= 2:
                return fid_src, fid_dst
        # Untyped pad clouds are only used when the field is not a periodic lattice.
        library_xy, library_w = self._stack_cloud(_tile_xy(earlier), getattr(earlier, "landmarks", None))
        frame_xy, frame_w = self._stack_cloud(_tile_xy(later), getattr(later, "landmarks", None))
        if len(library_xy) < 4 or len(frame_xy) < 4 or is_periodic_lattice(library_xy):
            return None
        estimate = estimate_similarity_pose(
            library_xy,
            frame_xy,
            later.width,
            later.height,
            library_w=library_w,
            query_w=frame_w,
        )
        if not estimate.ok or len(estimate.src_xy) < 4:
            return None
        return estimate.src_xy, estimate.dst_xy

    def _close_loops(self, tile: _Tile) -> None:
        """Match overlapping tiles by typed marks and refine one board frame."""
        if len(self.tiles) < 2:
            return
        constraints: list[tuple] = []
        matched: list[_Tile] = []
        new_index = len(self.tiles) - 1
        for earlier_index, earlier in enumerate(self.tiles[:-1]):
            for later_index, later in enumerate(self.tiles[earlier_index + 1 :], start=earlier_index + 1):
                if not self._tiles_may_overlap(earlier, later):
                    continue
                # On a long chain, only the new tile and fiducial pairs are revisited.
                if later_index != new_index and len(self.tiles) > 12:
                    if not (getattr(earlier, "fiducials", None) and getattr(later, "fiducials", None)):
                        continue
                pair = self._pair_constraint(earlier, later)
                if pair is None:
                    continue
                src_xy, dst_xy = pair
                constraints.append((earlier_index, later_index, src_xy, dst_xy))
                if later_index == new_index:
                    matched.append(earlier)
        if not constraints:
            # Shared fiducials can still pull a drifted tile even without a new pair.
            pinned = self._pin_shared_landmarks()
            if pinned:
                self._rebuild_coverage_memory()
                self._pose_after_capture(tile)
                self._rebuild_color_overview()
                self.overview_refine_count += 1
            return
        solved = refine_pose_graph(
            [np.asarray(item.M_to_ref, dtype=np.float64) for item in self.tiles],
            constraints,
        )
        moved = False
        for item, pose in zip(self.tiles, solved):
            pose = np.asarray(pose, dtype=np.float64)
            if float(np.max(np.abs(pose - item.M_to_ref))) > 0.75:
                moved = True
            item.M_to_ref = pose
            item.M_overview = pose.copy()
        if self.edges and self.edges[-1]["dst"] == tile.tile_id:
            self._rewrite_edge_pose(self.edges[-1], tile)
        primary = self.edges[-1]["src"] if self.edges else None
        for earlier in matched:
            if earlier.tile_id == primary:
                continue
            self._append_loop_edge(earlier, tile)
        self._refresh_annotations()
        self._rebuild_coverage_memory()
        self._pose_after_capture(tile)
        pinned = self._pin_shared_landmarks()
        if moved or pinned:
            self._rebuild_color_overview()
            self.overview_refine_count += 1

    def _tiles_may_overlap(self, left: _Tile, right: _Tile, min_ratio: float = 0.06) -> bool:
        quad_a = project_tile_quad(left.M_to_ref, left.width, left.height).astype(np.float32)
        quad_b = project_tile_quad(right.M_to_ref, right.width, right.height).astype(np.float32)
        area, _intersection = cv2.intersectConvexConvex(quad_a, quad_b)
        if area is None:
            return False
        denom = float(max(1, min(left.width * left.height, right.width * right.height)))
        return float(max(0.0, area)) / denom >= min_ratio

    def _rewrite_edge_pose(self, edge: dict, tile: _Tile) -> None:
        source = next(item for item in self.tiles if item.tile_id == edge["src"])
        relative = _compose_m(_invert_m(tile.M_to_ref), source.M_to_ref)
        edge["M"] = np.asarray(relative, dtype=np.float64).round(6).tolist()
        edge["M_to_ref"] = np.asarray(tile.M_to_ref, dtype=np.float64).round(6).tolist()
        edge["angle_deg"] = round(pose_angle_deg(relative), 3)
        edge["reflection"] = bool(linear_is_reflection(relative))

    def _append_loop_edge(self, earlier: _Tile, tile: _Tile) -> None:
        relative = _compose_m(_invert_m(tile.M_to_ref), earlier.M_to_ref)
        quad_src = project_tile_quad(earlier.M_to_ref, earlier.width, earlier.height).astype(np.float32)
        quad_dst = project_tile_quad(tile.M_to_ref, tile.width, tile.height).astype(np.float32)
        area, _intersection = cv2.intersectConvexConvex(quad_src, quad_dst)
        overlap = float(max(0.0, area) / float(max(1, tile.width * tile.height)))
        self.edges.append(
            {
                "src": earlier.tile_id,
                "dst": tile.tile_id,
                "M": np.asarray(relative, dtype=np.float64).round(6).tolist(),
                "M_to_ref": np.asarray(tile.M_to_ref, dtype=np.float64).round(6).tolist(),
                "overlap": round(overlap, 4),
                "angle_deg": round(pose_angle_deg(relative), 3),
                "reflection": bool(linear_is_reflection(relative)),
                "direction": "loop_closure",
            }
        )

    def _pin_shared_landmarks(self) -> bool:
        """Pull the same fiducial or distinctive mark toward one board point.

        The four corner fiducials, once seen, define the board frame. A small
        residual is treated as drift, not as a new correspondence.
        """
        clusters = self._landmark_clusters()
        if not clusters:
            return False
        moved = False
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            target = np.mean(np.vstack([item[2] for item in cluster]), axis=0)
            for tile, _local, board_xy in cluster:
                if tile is self.tiles[0]:
                    continue
                delta = target - board_xy
                distance = float(np.hypot(delta[0], delta[1]))
                if distance < 0.35 or distance > 18.0:
                    continue
                tile.M_to_ref = np.asarray(tile.M_to_ref, dtype=np.float64).copy()
                tile.M_to_ref[0, 2] += float(delta[0])
                tile.M_to_ref[1, 2] += float(delta[1])
                tile.M_overview = tile.M_to_ref.copy()
                moved = True
        return moved

    def _landmark_clusters(self) -> list[list[tuple]]:
        """Group fiducials and distinctive text that land on the same board point."""
        observations: list[tuple] = []
        for tile in self.tiles:
            for fiducial in getattr(tile, "fiducials", None) or []:
                local = np.asarray(fiducial["xy"], dtype=np.float64)
                board = _apply_m(tile.M_to_ref, local.reshape(1, 2))[0]
                observations.append((tile, "fiducial", board, float(fiducial.get("radius", 0.0)), local))
            elements = getattr(tile, "elements", None)
            if elements is None:
                continue
            for element in elements.of_kind("text"):
                if element.partial or element.area < 30.0:
                    continue
                local = np.asarray(element.centroid, dtype=np.float64)
                board = _apply_m(tile.M_to_ref, local.reshape(1, 2))[0]
                observations.append((tile, "text", board, float(element.area), local))
        clusters: list[list[tuple]] = []
        used = [False] * len(observations)
        for index, item in enumerate(observations):
            if used[index]:
                continue
            used[index] = True
            group = [(item[0], item[4], item[2])]
            for other_index, other in enumerate(observations):
                if used[other_index] or other[1] != item[1]:
                    continue
                limit = 18.0 if item[1] == "fiducial" else 10.0
                if float(np.hypot(item[2][0] - other[2][0], item[2][1] - other[2][1])) > limit:
                    continue
                if item[1] == "fiducial" and abs(item[3] - other[3]) / max(item[3], other[3], 1.0) > 0.35:
                    continue
                used[other_index] = True
                group.append((other[0], other[4], other[2]))
            if len(group) >= 2:
                clusters.append(group)
        return clusters

    def _rebuild_coverage_memory(self) -> None:
        memory = None
        for tile in self.tiles:
            memory = build_coverage_memory(memory, tile.width, tile.height, tile.M_to_ref)
        self._coverage_memory = memory

    def _rebuild_color_overview(self) -> None:
        bounds = self._world_bounds(self.tiles)
        if bounds is None:
            return
        _planned, scale, _size = self._plan_overview(bounds)
        self._rebuild_overview(self.tiles, scale)

    def add_annotation(self, text: str, xy, *, tile_id: str | None = None) -> dict:
        """Record a label in board coordinates. Does not paint it into a tile raster."""
        local = (float(xy[0]), float(xy[1]))
        if tile_id is None:
            board = local
            stored_tile = None
            stored_local = None
        else:
            tile = next((item for item in self.tiles if item.tile_id == tile_id), None)
            if tile is None:
                raise KeyError(tile_id)
            board_xy = _apply_m(tile.M_to_ref, np.array([local], dtype=np.float64))[0]
            board = (float(board_xy[0]), float(board_xy[1]))
            stored_tile = tile_id
            stored_local = local
        for item in self._annotations:
            if item["text"] == str(text) and math.hypot(item["xy"][0] - board[0], item["xy"][1] - board[1]) <= 12.0:
                return dict(item)
        record = {
            "text": str(text),
            "xy": [round(board[0], 2), round(board[1], 2)],
            "tile_id": stored_tile,
            "xy_tile": None if stored_local is None else [round(stored_local[0], 2), round(stored_local[1], 2)],
            "kind": "label",
        }
        self._annotations.append(record)
        return dict(record)

    def board_annotations(self) -> list[dict]:
        self._refresh_annotations()
        return [dict(item) for item in self._annotations]

    def draw_annotations(self, image: np.ndarray, transform: np.ndarray | None = None) -> np.ndarray:
        """Draw each board label once. The returned image is a copy; tiles stay untouched."""
        canvas = image.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        matrix = _identity_m() if transform is None else np.asarray(transform, dtype=np.float64)
        for item in self.board_annotations():
            point = _apply_m(matrix, np.array([item["xy"]], dtype=np.float64))[0]
            origin = (int(round(float(point[0]))), int(round(float(point[1]))))
            cv2.putText(
                canvas,
                str(item["text"]),
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _refresh_annotations(self) -> None:
        for item in self._annotations:
            tile_id = item.get("tile_id")
            local = item.get("xy_tile")
            if not tile_id or not local:
                continue
            tile = next((entry for entry in self.tiles if entry.tile_id == tile_id), None)
            if tile is None:
                continue
            board = _apply_m(tile.M_to_ref, np.array([local], dtype=np.float64))[0]
            item["xy"] = [round(float(board[0]), 2), round(float(board[1]), 2)]

    def _ignore_flaky_secondary(self, anchor: _Tile, frame: np.ndarray, verdict) -> bool:
        """Noisy extraction on a strong photometric lock is not a real seam error.

        A coherent silkscreen shift (several same-letter pairs at one offset)
        is still rejected. Random leftover text or pad blobs are not.
        """
        if self._pose is None or not getattr(self._pose, "ok", False):
            return False
        ncc, _area = masked_ncc(anchor.raw, frame, np.asarray(self._pose.M, dtype=np.float64))
        if ncc < 0.55 or float(self._lock_residual) > 2.5:
            return False
        message = str(getattr(verdict, "message", "") or "")
        if "primary seam disagrees" in message:
            return ncc >= 0.60 and float(self._lock_residual) <= 2.0
        if "secondary seam disagrees:" not in message:
            return False
        kinds = {item.strip() for item in message.split(":", 1)[-1].split(",") if item.strip()}
        if not kinds:
            return False
        if kinds - {"outline", "trace", "text"}:
            return False
        if "text" in kinds and _coherent_text_shift(verdict):
            return False
        return True

    def _install_typed_pose(self, frame: np.ndarray, typed: dict) -> None:
        """Commit a typed lock so capture can write the tile into the board frame."""
        width = int(frame.shape[1])
        height = int(frame.shape[0])
        ref_to_frame = np.asarray(typed["ref_to_frame"], dtype=np.float64)
        self._lock_residual = float(typed["residual"])
        anchor, anchor_to_frame = self._select_anchor(ref_to_frame, width, height)
        if anchor is None or anchor_to_frame is None or alignment_rejected(anchor_to_frame):
            # Fall back to the tile that produced the typed or unique-feature match.
            anchor = typed.get("tile")
            if anchor is None:
                return
            result = typed.get("result")
            pose = typed.get("pose")
            if result is not None and getattr(result, "matrix", None) is not None:
                anchor_to_frame = np.asarray(result.matrix, dtype=np.float64)
            elif pose is not None and getattr(pose, "matrix", None) is not None:
                anchor_to_frame = np.asarray(pose.matrix, dtype=np.float64)
            else:
                return
        self.matcher.set_topology(anchor.extracted)
        self._ref_to_frame = ref_to_frame
        self._anchor_id = anchor.tile_id
        self._pose = Pose2D(
            M=np.asarray(anchor_to_frame, dtype=np.float64),
            inliers=int(typed.get("hits", 0)),
            confidence=float(typed.get("ncc", 0.0)),
            ok=True,
            message="ok",
            elapsed_ms=float(self._track_ms or 0.0),
            scale=float(np.linalg.norm(np.asarray(anchor_to_frame)[:, 0])),
            angle_deg=pose_angle_deg(anchor_to_frame),
            query_points=0,
            pose_error_px=float(self._lock_residual),
        )
        self._quads = [
            project_tile_quad(_compose_m(ref_to_frame, tile.M_to_ref), tile.width, tile.height) for tile in self.tiles
        ]
        self._quads = self._filter_quads_by_coverage_memory(self._quads, ref_to_frame, width, height)
        self._polygon = self._quads[-1] if self._quads else None
        self._overlap = union_overlap_ratio(self._quads, width, height)
        self._tracked = True
        self._track_note = ""

    def _check_red_elements(self, frame: np.ndarray, anchor: _Tile, pose_m: np.ndarray, query_elements=None):

        """Separate this still and compare it to the anchor inside the red overlap.

        Preview tracking does not call this. A lattice, or a seam whose text,
        outline, or traces miss the locked pose, rejects the shot.
        """
        query = query_elements if query_elements is not None else extract_board_elements(frame, max_long_edge=self.extractor.max_long_edge)
        reference = getattr(anchor, "elements", None)
        if reference is None or not getattr(reference, "elements", None) or not query.elements:
            self.element_alignment = None
            return query
        self.element_alignment = align_red_region(
            reference,
            query,
            initial_pose=np.asarray(pose_m, dtype=np.float64),
        )
        return query

    def _reject(self, reason: str, overlap: float | None = None, elapsed_ms: float = 0.0) -> CaptureDecision:
        self.rejects.append({"reason": reason, "overlap": overlap})
        return CaptureDecision(False, reason, overlap=overlap, elapsed_ms=elapsed_ms)


def _coherent_text_shift(verdict) -> bool:
    """True when several tight text pairs share one offset, as in a moved silkscreen."""
    offsets = getattr(verdict, "shift_offsets", None) or getattr(verdict, "conflict_offsets", None) or {}
    values = [float(item) for item in offsets.get("text", []) if 6.0 <= float(item) <= 80.0]
    if len(values) < 2:
        return False
    return (max(values) - min(values)) <= 14.0


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


def _localization_score(pose) -> float:
    return float(pose.inliers) + 4.0 * float(pose.confidence)


def _is_strong_rival(other: _AnchorHypothesis, primary: _AnchorHypothesis) -> bool:
    """A second high-scoring pose that disagrees with the primary is ambiguous."""
    if other.score < _AMBIGUOUS_SCORE_RATIO * primary.score:
        return False
    min_inliers = max(8, int(np.ceil(_AMBIGUOUS_INLIER_RATIO * float(primary.pose.inliers))))
    return int(other.pose.inliers) >= min_inliers


def _poses_consistent(ref_a: np.ndarray, ref_b: np.ndarray, sample_xy: np.ndarray) -> bool:
    if sample_xy is None or len(sample_xy) == 0:
        return _pose_angle_delta_deg(ref_a, ref_b) <= _POSE_ANGLE_TOL_DEG
    return (
        _pose_corner_error_px(ref_a, ref_b, sample_xy) <= _POSE_CORNER_TOL_PX
        and _pose_angle_delta_deg(ref_a, ref_b) <= _POSE_ANGLE_TOL_DEG
    )


def _pose_corner_error_px(ref_a: np.ndarray, ref_b: np.ndarray, sample_xy: np.ndarray) -> float:
    left = _apply_m(ref_a, sample_xy)
    right = _apply_m(ref_b, sample_xy)
    return float(np.max(np.linalg.norm(left - right, axis=1)))


def _pose_angle_delta_deg(ref_a: np.ndarray, ref_b: np.ndarray) -> float:
    delta = pose_angle_deg(ref_a) - pose_angle_deg(ref_b)
    return abs((delta + 180.0) % 360.0 - 180.0)


def _blend_ref_to_frame(items: list[tuple[np.ndarray, float]]) -> np.ndarray:
    total = float(sum(max(weight, 1e-6) for _matrix, weight in items)) or 1.0
    blended = np.zeros((2, 3), dtype=np.float64)
    for matrix, weight in items:
        blended += (max(weight, 1e-6) / total) * np.asarray(matrix, dtype=np.float64)
    return blended


def _tile_xy(tile: _Tile) -> np.ndarray:
    if tile.extracted is not None:
        return tile.extracted.xy()
    if tile.points:
        return np.array([point["xy"] for point in tile.points], dtype=np.float64)
    return np.zeros((0, 2), dtype=np.float64)


def _in_frame_mask(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    return (xy[:, 0] >= 0.0) & (xy[:, 0] < float(width)) & (xy[:, 1] >= 0.0) & (xy[:, 1] < float(height))


def _quad_visible_fraction(quad: np.ndarray, width: int, height: int, tile_width: int, tile_height: int) -> float:
    tile_area = float(max(1, int(tile_width) * int(tile_height)))
    frame = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    polygon = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    if polygon.shape[0] < 3 or not np.isfinite(polygon).all():
        return 0.0
    area, _intersection = cv2.intersectConvexConvex(polygon, frame)
    return float(max(0.0, area) / tile_area)


def _tile_is_trusted_in_view(
    tile: _Tile,
    tile_to_frame: np.ndarray,
    quad: np.ndarray,
    observed_xy: np.ndarray,
    width: int,
    height: int,
) -> bool:
    if quad is None or not np.isfinite(quad).all():
        return False
    visible = _quad_visible_fraction(quad, width, height, tile.width, tile.height)
    if visible <= _SLIVER_VISIBLE_FRACTION:
        return True
    library = _tile_xy(tile)
    if len(library) == 0:
        return False
    projected = _apply_m(tile_to_frame, library)
    inside = _in_frame_mask(projected, width, height)
    n_in = int(np.count_nonzero(inside))
    if n_in == 0:
        return False
    if len(observed_xy) == 0:
        return False
    in_pts = projected[inside]
    distances = np.linalg.norm(in_pts[:, None, :] - observed_xy[None, :, :], axis=2)
    hits = int(np.count_nonzero(distances.min(axis=1) <= _POINT_MATCH_PX))
    if n_in < _MIN_INFRAME_SUPPORT_POINTS:
        return hits >= n_in
    needed = max(_MIN_INFRAME_SUPPORT_POINTS, int(np.ceil(_MIN_SUPPORT_RATIO * n_in)))
    return hits >= needed


def _coverage_kernel(radius: int) -> np.ndarray:
    size = 2 * max(1, int(radius)) + 1
    return np.ones((size, size), dtype=np.uint8)


def _coverage_blob_min_area(tile_area: int) -> int:
    return max(_COVERAGE_BLOB_MIN_PX, int(np.ceil(_COVERAGE_BLOB_FRAC * max(1, int(tile_area)))))


def _edge_slack_area(mask: np.ndarray, slack_px: int) -> int:
    if mask is None or not np.any(mask):
        return 0
    eroded = cv2.erode(mask, _coverage_kernel(slack_px), iterations=1)
    return int(np.count_nonzero(mask) - np.count_nonzero(eroded))


def _place_coverage_mask(
    mask: np.ndarray,
    origin: tuple[int, int],
    new_origin: tuple[int, int],
    size: tuple[int, int],
) -> np.ndarray:
    canvas = np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
    if mask is None or mask.size == 0 or size[0] <= 0 or size[1] <= 0:
        return canvas
    dx = int(origin[0] - new_origin[0])
    dy = int(origin[1] - new_origin[1])
    height, width = mask.shape[:2]
    x0 = max(0, dx)
    y0 = max(0, dy)
    sx = max(0, -dx)
    sy = max(0, -dy)
    copy_w = min(width - sx, int(size[0]) - x0)
    copy_h = min(height - sy, int(size[1]) - y0)
    if copy_w > 0 and copy_h > 0:
        canvas[y0 : y0 + copy_h, x0 : x0 + copy_w] = mask[sy : sy + copy_h, sx : sx + copy_w]
    return canvas


def _align_coverage_masks(
    left: np.ndarray,
    left_origin: tuple[int, int],
    right: np.ndarray,
    right_origin: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    min_x = min(int(left_origin[0]), int(right_origin[0]))
    min_y = min(int(left_origin[1]), int(right_origin[1]))
    max_x = max(int(left_origin[0] + left.shape[1]), int(right_origin[0] + right.shape[1]))
    max_y = max(int(left_origin[1] + left.shape[0]), int(right_origin[1] + right.shape[0]))
    size = (max_x - min_x, max_y - min_y)
    left_aligned = _place_coverage_mask(left, left_origin, (min_x, min_y), size)
    right_aligned = _place_coverage_mask(right, right_origin, (min_x, min_y), size)
    return left_aligned, right_aligned, (min_x, min_y)


def _significant_coverage_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if mask is None or mask.size == 0 or not np.any(mask):
        return np.zeros((0, 0) if mask is None or mask.size == 0 else mask.shape, dtype=np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_area):
            out[labels == label] = 255
    return out


def unexplained_coverage_blobs(
    actual: np.ndarray,
    actual_origin: tuple[int, int],
    expected: np.ndarray,
    expected_origin: tuple[int, int],
    *,
    slack_px: int = _COVERAGE_EDGE_SLACK_PX,
    min_area: int = _COVERAGE_BLOB_MIN_PX,
) -> np.ndarray:
    """Large blobs in ``actual`` that expected coverage plus edge slack cannot explain."""
    actual_aligned, expected_aligned, _origin = _align_coverage_masks(
        actual, actual_origin, expected, expected_origin
    )
    dilated = cv2.dilate(expected_aligned, _coverage_kernel(slack_px), iterations=1)
    extra = np.where((actual_aligned > 0) & (dilated == 0), np.uint8(255), np.uint8(0))
    return _significant_coverage_components(extra, min_area)


def summarize_coverage_mask(mask: np.ndarray, origin: tuple[int, int]) -> CoverageMemory:
    area = int(np.count_nonzero(mask))
    origin_x, origin_y = int(origin[0]), int(origin[1])
    bbox = (origin_x, origin_y, 0, 0)
    contour: list[list[float]] = []
    moments = {"m00": 0.0, "cx": float(origin_x), "cy": float(origin_y)}
    if area > 0:
        x, y, width, height = cv2.boundingRect((mask > 0).astype(np.uint8))
        bbox = (origin_x + int(x), origin_y + int(y), int(width), int(height))
        raw = cv2.moments((mask > 0).astype(np.uint8))
        cx = (raw["m10"] / raw["m00"] + origin_x) if raw["m00"] else float(origin_x)
        cy = (raw["m01"] / raw["m00"] + origin_y) if raw["m00"] else float(origin_y)
        moments = {
            "m00": float(raw["m00"]),
            "m10": float(raw["m10"]),
            "m01": float(raw["m01"]),
            "mu20": float(raw["mu20"]),
            "mu11": float(raw["mu11"]),
            "mu02": float(raw["mu02"]),
            "cx": float(cx),
            "cy": float(cy),
        }
        contours, _hierarchy = cv2.findContours(
            (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            best = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(best, True)
            approx = cv2.approxPolyDP(best, max(1.5, 0.008 * perimeter), True)
            contour = [
                [round(float(point[0][0] + origin_x), 2), round(float(point[0][1] + origin_y), 2)]
                for point in approx
            ]
    return CoverageMemory(
        area=area,
        origin=(origin_x, origin_y),
        mask=np.ascontiguousarray(mask.copy()),
        contour=contour,
        bbox=bbox,
        moments=moments,
    )


def build_coverage_memory(
    memory: CoverageMemory | None,
    tile_width: int,
    tile_height: int,
    M_to_ref: np.ndarray,
    *,
    pad: int = 2,
) -> CoverageMemory:
    """OR a tile footprint into chain-space coverage. First shot starts a new mask."""
    quad = project_tile_quad(M_to_ref, tile_width, tile_height)
    if not np.isfinite(quad).all():
        if memory is not None:
            return memory
        return summarize_coverage_mask(np.zeros((1, 1), dtype=np.uint8), (0, 0))
    min_x = int(np.floor(np.min(quad[:, 0]))) - pad
    min_y = int(np.floor(np.min(quad[:, 1]))) - pad
    max_x = int(np.ceil(np.max(quad[:, 0]))) + pad + 1
    max_y = int(np.ceil(np.max(quad[:, 1]))) + pad + 1
    if memory is not None and memory.mask is not None:
        origin_x, origin_y = memory.origin
        min_x = min(min_x, origin_x)
        min_y = min(min_y, origin_y)
        max_x = max(max_x, origin_x + memory.mask.shape[1])
        max_y = max(max_y, origin_y + memory.mask.shape[0])
        canvas = _place_coverage_mask(memory.mask, memory.origin, (min_x, min_y), (max_x - min_x, max_y - min_y))
    else:
        canvas = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
    shifted = np.asarray(quad, dtype=np.float64).copy()
    shifted[:, 0] -= min_x
    shifted[:, 1] -= min_y
    _fill_quad(canvas, shifted)
    return summarize_coverage_mask(canvas, (min_x, min_y))


def evaluate_coverage_growth(
    memory: CoverageMemory | None,
    tile_width: int,
    tile_height: int,
    M_to_ref: np.ndarray,
    *,
    slack_px: int = _COVERAGE_EDGE_SLACK_PX,
) -> CoverageCheck:
    """Accept a candidate tile only if chain-space coverage grows like one connected frame."""
    proposed = build_coverage_memory(memory, tile_width, tile_height, M_to_ref)
    if memory is None or memory.area <= 0:
        return CoverageCheck(True, "", memory=proposed, added=proposed.area)
    tile_area = int(max(1, int(tile_width) * int(tile_height)))
    added = int(proposed.area - memory.area)
    shrink_slack = max(_edge_slack_area(memory.mask, slack_px), int(slack_px))
    grow_margin = max(
        int(np.ceil(_COVERAGE_GROW_MARGIN * tile_area)),
        _edge_slack_area(proposed.mask, slack_px),
    )
    if added < -shrink_slack:
        return CoverageCheck(False, _COVERAGE_AREA, memory=proposed, added=added)
    if added > tile_area + grow_margin:
        return CoverageCheck(False, _COVERAGE_AREA, memory=proposed, added=added)
    old_aligned, new_aligned, _ = _align_coverage_masks(
        memory.mask, memory.origin, proposed.mask, proposed.origin
    )
    dilated_old = cv2.dilate(old_aligned, _coverage_kernel(slack_px), iterations=1)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (new_aligned > 0).astype(np.uint8), connectivity=8
    )
    min_blob = _coverage_blob_min_area(tile_area)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < min_blob:
            continue
        if not np.any((labels == label) & (dilated_old > 0)):
            return CoverageCheck(False, _COVERAGE_ISLAND, memory=proposed, added=added)
    # The capture candidate is already (old_union OR new_tile) in chain space.
    # Symmetric-difference blobs against a drifted live painting are handled in
    # _filter_quads_by_coverage_memory, which compares against this memory.
    return CoverageCheck(True, "", memory=proposed, added=added)


def project_coverage_memory(
    memory: CoverageMemory,
    ref_to_frame: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Warp remembered chain-space coverage into the current live view."""
    if memory is None or memory.mask is None or width <= 1 or height <= 1:
        return np.zeros((max(0, height), max(0, width)), dtype=np.uint8)
    mask_to_frame = _compose_m(ref_to_frame, _translate_m(memory.origin[0], memory.origin[1]))
    return cv2.warpAffine(
        memory.mask,
        np.asarray(mask_to_frame, dtype=np.float64),
        (int(width), int(height)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _fill_quad(mask: np.ndarray, quad: np.ndarray) -> None:
    polygon = np.round(np.asarray(quad, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
    if polygon.shape[0] >= 3 and np.isfinite(quad).all():
        cv2.fillConvexPoly(mask, polygon, 255)


def _quad_touches_mask(quad: np.ndarray, accepted_mask: np.ndarray, *, touch_px: int) -> bool:
    if accepted_mask is None or not np.any(accepted_mask):
        return False
    if quad is None or not np.isfinite(quad).all():
        return False
    radius = max(1, int(touch_px))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    dilated = cv2.dilate(accepted_mask, kernel, iterations=1)
    probe = np.zeros_like(accepted_mask)
    _fill_quad(probe, quad)
    return bool(np.any((probe > 0) & (dilated > 0)))


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


def overview_summary_fields(session: StableCaptureSession, out_dir: Path | None = None) -> dict:
    meta = session.overview_meta()
    fields = {
        "overview_size": None if meta["width"] is None else [meta["width"], meta["height"]],
        "overview_tiles": meta["tiles"],
        "overview_max_side": meta["max_side"],
        "overview_max_pixels": meta["max_pixels"],
        "overview_scale": meta["scale"],
        "overview_refine_count": meta["refine_count"],
        "full_canvas": False,
    }
    if out_dir is not None and session.overview_image() is not None:
        fields["overview"] = str(Path(out_dir) / "overview.png")
    return fields


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
        _write_element_crops(tile, tiles_dir)
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
    if session.overview_image() is not None:
        cv2.imwrite(str(out_dir / "overview.png"), session.overview_image())
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
    summary.update(overview_summary_fields(session, out_dir))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _write_element_crops(tile, tiles_dir: Path) -> None:
    """Small patch per element. Later recognition reads these, not the whole board."""
    elements = getattr(tile, "elements", None)
    if elements is None:
        return
    folder = tiles_dir / "crops" / tile.tile_id
    wrote = False
    for element in elements.elements:
        crop = getattr(element, "crop", None)
        if crop is None or crop.size == 0:
            continue
        if not wrote:
            folder.mkdir(parents=True, exist_ok=True)
            wrote = True
        cv2.imwrite(str(folder / f"{element.element_id}.png"), crop)


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
    summary.update(overview_summary_fields(session, out_dir))
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


DEMO_WINDOW = (480, 360)
DEMO_ORIGIN = (80, 160)
DEMO_STEPS = ((0, 0.0), (70, 4.0), (140, 1.0))


def default_sample_global() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "assets" / "samples" / "SAMPLE-PARTIAL-001" / "global_ref.png"
        if candidate.is_file():
            return candidate
    return None


def default_scan_export_dir(board_id: str, root: Path | None = None) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = root
    if base is None:
        sample = default_sample_global()
        base = sample.parents[3] if sample is not None else Path.cwd()
    return Path(base) / "artifacts" / "scan_maps" / f"{board_id}-{stamp}"


def demo_scan_frames(image: np.ndarray) -> list[np.ndarray]:
    """Overlapping windows used by the board demo (accepted shots only)."""
    window_w, window_h = DEMO_WINDOW
    origin_x, origin_y = DEMO_ORIGIN
    frames = []
    for step, angle in DEMO_STEPS:
        frame, _truth = _rotated_crop(image, origin_x + step, origin_y, window_w, window_h, angle)
        frames.append(frame)
    return frames


def demo_disconnected_frame(image: np.ndarray) -> np.ndarray:
    window_w, window_h = DEMO_WINDOW
    origin_x, origin_y = DEMO_ORIGIN
    return image[origin_y + 780 : origin_y + 780 + window_h, origin_x : origin_x + window_w].copy()


def run_demo_scan(out_dir: Path | None = None, *, board_id: str = "SAMPLE-PARTIAL-001") -> dict:
    """Headless sample scan that writes overview.png. No Qt required."""
    image_path = default_sample_global()
    if image_path is None:
        raise FileNotFoundError("找不到样例 assets/samples/SAMPLE-PARTIAL-001/global_ref.png")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取 {image_path}")
    destination = Path(out_dir) if out_dir is not None else default_scan_export_dir(board_id, image_path.parents[3])
    return run_board_demo(image, destination, board_id=board_id, source=str(image_path))


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
        "overview": summary.get("overview"),
        "overview_size": summary.get("overview_size"),
        "summary": str(args.out / "summary.json"),
    }
    print(json.dumps(brief, ensure_ascii=False))
    return 0 if summary["shots"] >= 2 and summary["rejects"] >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
