"""Topology live matcher (D16 / TECH_SPEC §3.8, DATA_MODEL §2.7).

Docs on disk may still describe ORB ``LocalGlobalTracker`` as the primary
live path. This module is the industrial slice: gray, adaptive threshold,
then a constellation of pad / hole centroids. It does not compare full-board
color. Video mosaic and still-tile stitch stay optional bypasses.

``Pose2D.M`` is 2×3 and maps library pixels → current-frame pixels (D1).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from model.schema import BoardProject, Component

SCHEMA_VERSION = "0.2-topo"
_NO_TOPOLOGY = "未设置拓扑库，无法定位"
_KINDS = ("hole", "pad", "slot")


@dataclass
class GeoPoint:
    id: str
    kind: str
    xy: tuple[float, float]
    area: float = 0.0
    circularity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "xy": [round(float(self.xy[0]), 2), round(float(self.xy[1]), 2)],
            "area": round(float(self.area), 1),
            "circularity": round(float(self.circularity), 3),
        }


@dataclass
class TopologyExtractResult:
    ok: bool
    points: list[GeoPoint]
    width: int
    height: int
    scale: float = 1.0
    message: str = ""
    board_id: str = ""
    source: str = ""
    schema_version: str = SCHEMA_VERSION

    def counts(self) -> dict[str, int]:
        found = {kind: 0 for kind in _KINDS}
        for point in self.points:
            found[point.kind] = found.get(point.kind, 0) + 1
        found["points"] = len(self.points)
        return found

    def xy(self) -> np.ndarray:
        if not self.points:
            return np.zeros((0, 2), dtype=np.float64)
        return np.array([point.xy for point in self.points], dtype=np.float64)

    def to_dict(self) -> dict:
        slots = [point.to_dict() for point in self.points if point.kind == "slot"]
        return {
            "schema_version": self.schema_version,
            "board_id": self.board_id,
            "image_size": {"width": self.width, "height": self.height},
            "source": self.source,
            "detect_scale": self.scale,
            "counts": self.counts(),
            "points": [point.to_dict() for point in self.points],
            "slots": slots,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> TopologyExtractResult:
        size = data.get("image_size") or {}
        points = []
        for item in data.get("points") or []:
            xy = item.get("xy") or [0, 0]
            points.append(
                GeoPoint(
                    id=str(item.get("id", "")),
                    kind=str(item.get("kind", "pad")),
                    xy=(float(xy[0]), float(xy[1])),
                    area=float(item.get("area", 0)),
                    circularity=float(item.get("circularity", 0)),
                )
            )
        return cls(
            ok=bool(points) or bool(data.get("ok", True)),
            points=points,
            width=int(size.get("width", 0)),
            height=int(size.get("height", 0)),
            scale=float(data.get("detect_scale", 1.0)),
            message=str(data.get("message", "")),
            board_id=str(data.get("board_id", "")),
            source=str(data.get("source", "")),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass
class Pose2D:
    """Library → frame. ``M`` is 2×3, same spirit as D1 (global pixel → frame pixel)."""

    M: np.ndarray
    inliers: int
    confidence: float
    ok: bool
    message: str = ""
    elapsed_ms: float = 0.0
    scale: float = 1.0
    angle_deg: float = 0.0
    query_points: int = 0
    pose_error_px: float | None = None
    pose_error_deg: float | None = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "inliers": int(self.inliers),
            "confidence": round(float(self.confidence), 4),
            "scale": round(float(self.scale), 5),
            "angle_deg": round(float(self.angle_deg), 3),
            "elapsed_ms": round(float(self.elapsed_ms), 2),
            "query_points": int(self.query_points),
            "pose_error_px": None if self.pose_error_px is None else round(float(self.pose_error_px), 3),
            "pose_error_deg": None if self.pose_error_deg is None else round(float(self.pose_error_deg), 3),
            "message": self.message,
            "M": np.asarray(self.M, dtype=np.float64).round(6).tolist(),
        }


def _failed_pose(message: str, *, elapsed_ms: float = 0.0, query_points: int = 0) -> Pose2D:
    return Pose2D(
        M=_identity_m(),
        inliers=0,
        confidence=0.0,
        ok=False,
        message=message,
        elapsed_ms=elapsed_ms,
        query_points=query_points,
    )


class TopologyExtractor:
    """Gray + adaptive threshold. Long edges above ``max_long_edge`` are scaled down."""

    def __init__(self, max_long_edge: int = 1280, adaptive_block: int = 31, adaptive_C: int = 5):
        if int(max_long_edge) < 32:
            raise ValueError("max_long_edge must be at least 32")
        if int(adaptive_block) < 3 or int(adaptive_block) % 2 == 0:
            raise ValueError("adaptive_block must be an odd integer >= 3")
        self.max_long_edge = int(max_long_edge)
        self.adaptive_block = int(adaptive_block)
        self.adaptive_C = int(adaptive_C)

    def extract(self, frame_bgr: np.ndarray, roi_xywh=None, *, working_scale: float | None = None) -> TopologyExtractResult:
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return TopologyExtractResult(False, [], 0, 0, message="空帧，无法抽取拓扑")
        image, origin, frame_w, frame_h = _crop_roi(frame_bgr, roi_xywh)
        if image is None:
            return TopologyExtractResult(False, [], frame_w, frame_h, message="roi_xywh 超出画面")
        crop_h, crop_w = image.shape[:2]
        scale = _working_scale(crop_w, crop_h, self.max_long_edge, working_scale)
        view = image
        if scale < 0.999:
            view = cv2.resize(
                image,
                (max(1, int(round(crop_w * scale))), max(1, int(round(crop_h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        raw = _blobs(view, block=self.adaptive_block, c=self.adaptive_C)
        points = []
        for index, (x, y, kind, area, circularity) in enumerate(raw, start=1):
            full_x = x / scale + origin[0]
            full_y = y / scale + origin[1]
            points.append(
                GeoPoint(
                    id=f"{kind}-{index:04d}",
                    kind=kind,
                    xy=(full_x, full_y),
                    area=area / (scale * scale),
                    circularity=circularity,
                )
            )
        return TopologyExtractResult(
            ok=True,
            points=points,
            width=frame_w,
            height=frame_h,
            scale=scale,
            message=f"extracted {len(points)} points",
        )


class ConstellationMatcher:
    """Match a frame constellation to the library subgraph. Needs ``set_topology`` first."""

    def __init__(self, min_inliers: int = 8, conf_thresh: float = 0.35):
        if int(min_inliers) < 2:
            raise ValueError("min_inliers must be at least 2")
        if not 0.0 < float(conf_thresh) <= 1.0:
            raise ValueError("conf_thresh must be in (0, 1]")
        self.min_inliers = int(min_inliers)
        self.conf_thresh = float(conf_thresh)
        self.ransac_thresh = 5.0
        self.iterations = 400
        self.neighbors = 5
        self.topology: TopologyExtractResult | None = None
        self._xy = np.zeros((0, 2), dtype=np.float64)
        self._desc = np.zeros((0, 1), dtype=np.float64)

    def set_topology(self, topology: TopologyExtractResult | dict | None) -> None:
        if topology is None:
            self.topology = None
            self._xy = np.zeros((0, 2), dtype=np.float64)
            self._desc = np.zeros((0, 1), dtype=np.float64)
            return
        if isinstance(topology, dict):
            topology = TopologyExtractResult.from_dict(topology)
        self.topology = topology
        self._xy = topology.xy()
        self._desc = _constellation_descriptors(self._xy, self.neighbors)

    def localize(self, frame_bgr_or_result) -> Pose2D:
        started = time.perf_counter()
        if self.topology is None:
            return _failed_pose(_NO_TOPOLOGY, elapsed_ms=(time.perf_counter() - started) * 1000.0)
        if len(self._xy) == 0:
            return _failed_pose("拓扑库没有可用点", elapsed_ms=(time.perf_counter() - started) * 1000.0)
        if isinstance(frame_bgr_or_result, TopologyExtractResult):
            query_result = frame_bgr_or_result
        else:
            query_result = TopologyExtractor().extract(frame_bgr_or_result, working_scale=self.topology.scale)
        return self._localize_result(query_result, started)

    def _localize_result(self, query_result: TopologyExtractResult, started: float) -> Pose2D:
        query = query_result.xy()
        homography, inliers, scale, angle, message = self._match_points(query)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        confidence = float(inliers) / float(len(query)) if len(query) else 0.0
        ok = (
            homography is not None
            and inliers >= self.min_inliers
            and confidence >= self.conf_thresh
            and 0.75 <= scale <= 1.35
        )
        if homography is None:
            return _failed_pose(message, elapsed_ms=elapsed_ms, query_points=len(query))
        if not ok:
            message = f"弱匹配（inliers={inliers}，confidence={confidence:.2f}）"
        return Pose2D(
            M=homography[:2, :].copy(),
            inliers=int(inliers),
            confidence=confidence,
            ok=ok,
            message="ok" if ok else message,
            elapsed_ms=elapsed_ms,
            scale=float(scale),
            angle_deg=float(angle),
            query_points=int(len(query)),
        )

    def project_components(self, components, pose: Pose2D) -> list[dict]:
        """Project library ``base_pixel_box`` values through ``pose.M`` into the frame."""
        if pose is None or not pose.ok:
            return []
        projected = []
        for component in components:
            box, component_id, label = _component_view(component)
            corners = _box_corners(box)
            mapped = _apply_m(pose.M, corners)
            projected.append(
                {
                    "component_id": component_id,
                    "label": label,
                    "corners": mapped.round(2).tolist(),
                }
            )
        return projected

    def _match_points(self, query_xy: np.ndarray):
        query_xy = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
        if len(self._xy) < self.neighbors + 1 or len(query_xy) < self.neighbors + 1:
            return None, 0, 1.0, 0.0, "拓扑点太少，无法匹配"
        query_desc = _constellation_descriptors(query_xy, self.neighbors)
        pairs = _putative_pairs(query_desc, self._desc, ratio=0.85)
        if len(pairs) < self.min_inliers:
            pairs = _putative_pairs(query_desc, self._desc, ratio=0.93)
        if len(pairs) < 2:
            return None, 0, 1.0, 0.0, "星座对应太少"
        src = self._xy[pairs[:, 1]]
        dst = query_xy[pairs[:, 0]]
        homography, inlier_mask = _ransac_similarity(src, dst, thresh=self.ransac_thresh, iterations=self.iterations)
        if homography is None or int(inlier_mask.sum()) < 2:
            return None, 0, 1.0, 0.0, "RANSAC 未形成稳定位姿"
        refined = _refine_similarity(src[inlier_mask], dst[inlier_mask])
        if refined is not None:
            homography = refined
            inlier_mask = np.linalg.norm(_apply_h(homography, src) - dst, axis=1) <= self.ransac_thresh
        scale, angle = _similarity_parts(homography)
        explained = _explained_query_points(homography, self._xy, query_xy, self.ransac_thresh)
        return homography, explained, scale, angle, "ok"


class GeometricLocalizer:
    """Extract a live frame and localize it against the matcher's topology."""

    def __init__(self, extractor: TopologyExtractor, matcher: ConstellationMatcher):
        self.extractor = extractor
        self.matcher = matcher

    def track(self, frame_bgr: np.ndarray) -> Pose2D:
        if self.matcher.topology is None:
            return _failed_pose(_NO_TOPOLOGY)
        if len(self.matcher._xy) == 0:
            return _failed_pose("拓扑库没有可用点")
        scale = self.matcher.topology.scale
        extracted = self.extractor.extract(frame_bgr, working_scale=scale)
        return self.matcher._localize_result(extracted, time.perf_counter())


def geo_fingerprint(points: list[GeoPoint], anchor_xy: tuple[float, float], *, neighbors: int = 6) -> dict:
    """Relative constellation around one component. Stored on the component, not as color."""
    if not points:
        return {"kind": "constellation", "anchor_xy": [anchor_xy[0], anchor_xy[1]], "neighbors": []}
    xy = np.array([point.xy for point in points], dtype=np.float64)
    anchor = np.array(anchor_xy, dtype=np.float64)
    order = np.argsort(np.linalg.norm(xy - anchor, axis=1))
    picked = []
    for index in order:
        if len(picked) >= neighbors:
            break
        delta = xy[index] - anchor
        if float(np.hypot(delta[0], delta[1])) < 1.0:
            continue
        point = points[int(index)]
        picked.append(
            {
                "id": point.id,
                "kind": point.kind,
                "dxy": [round(float(delta[0]), 2), round(float(delta[1]), 2)],
            }
        )
    return {
        "kind": "constellation",
        "anchor_xy": [round(float(anchor_xy[0]), 2), round(float(anchor_xy[1]), 2)],
        "neighbors": picked,
    }


def translation_m(origin_x: float, origin_y: float) -> np.ndarray:
    matrix = _identity_m()
    matrix[0, 2] = -float(origin_x)
    matrix[1, 2] = -float(origin_y)
    return matrix


def rotated_crop_m(origin_x: float, origin_y: float, width: int, height: int, angle_deg: float) -> np.ndarray:
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    affine = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    linear = affine[:, :2]
    translation = affine[:, 2] - linear @ np.array([origin_x, origin_y], dtype=np.float64)
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = linear
    matrix[:, 2] = translation
    return matrix


def pose_error_px(estimated_m: np.ndarray, truth_m: np.ndarray, points_xy: np.ndarray) -> float:
    if len(points_xy) == 0:
        return float("inf")
    left = _apply_m(estimated_m, points_xy)
    right = _apply_m(truth_m, points_xy)
    return float(np.mean(np.linalg.norm(left - right, axis=1)))


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    gray = _as_gray(image)
    fill = int(np.median(gray))
    border = fill if image.ndim == 2 else (fill, fill, fill)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def run_topology_live(
    image: np.ndarray,
    *,
    board_id: str,
    source: str,
    out_dir: Path,
    repo_root: Path,
    topology_path: Path | None = None,
    project_path: Path | None = None,
    window: tuple[int, int] = (560, 720),
    queries: int = 6,
    rotation_deg: float = 8.0,
    max_long_edge: int = 1280,
) -> dict:
    """Extract a library from the global photo, then localize crops. No mosaic required."""
    started = time.perf_counter()
    repo_root = Path(repo_root)
    out_dir = Path(out_dir)
    extractor = TopologyExtractor(max_long_edge=max_long_edge)
    library = extractor.extract(image)
    library.board_id = board_id
    library.source = source
    extract_ms = (time.perf_counter() - started) * 1000.0
    out_dir.mkdir(parents=True, exist_ok=True)
    topo_path = Path(topology_path) if topology_path is not None else repo_root / "assets" / "topology" / f"{board_id}.json"
    library.save(topo_path)
    library.save(out_dir / "topology.json")
    matcher = ConstellationMatcher()
    matcher.set_topology(library)
    localizer = GeometricLocalizer(extractor, matcher)
    height, width = _as_gray(image).shape[:2]
    window_w, window_h = window
    origins = _spread_origins(width, height, window_w, window_h, queries)
    component = _demo_component(library, origins[0], window) if origins else None
    _write_project(repo_root, board_id, topo_path, component, project_path)
    records = []
    overlay_global = cv2.cvtColor(_as_gray(image), cv2.COLOR_GRAY2BGR)
    library_xy = library.xy()
    drawn = False
    for index, (origin_x, origin_y) in enumerate(origins):
        crop = image[origin_y : origin_y + window_h, origin_x : origin_x + window_w]
        truth = translation_m(origin_x, origin_y)
        pose = localizer.track(crop)
        record = _score(pose, truth, library_xy, index, (origin_x, origin_y), window, 0.0)
        records.append(record)
        _draw_footprint(overlay_global, truth, pose, window_w, window_h)
        if pose.ok and component is not None and not drawn:
            _draw_query_overlay(out_dir / "overlay.png", crop, pose, matcher, component)
            drawn = True
    if origins and abs(rotation_deg) > 0.1:
        origin_x, origin_y = origins[len(origins) // 2]
        crop = image[origin_y : origin_y + window_h, origin_x : origin_x + window_w]
        rotated = rotate_image(crop, rotation_deg)
        truth = rotated_crop_m(origin_x, origin_y, window_w, window_h, rotation_deg)
        pose = localizer.track(rotated)
        records.append(_score(pose, truth, library_xy, len(records), (origin_x, origin_y), window, rotation_deg))
        _draw_footprint(overlay_global, truth, pose, window_w, window_h)
    if not drawn and origins and component is not None:
        crop = image[origins[0][1] : origins[0][1] + window_h, origins[0][0] : origins[0][0] + window_w]
        _draw_query_overlay(out_dir / "overlay.png", crop, records[0]["pose_obj"], matcher, component)
    cv2.imwrite(str(out_dir / "footprints.png"), overlay_global)
    matched = [item for item in records if item["ok"]]
    summary = {
        "board_id": board_id,
        "source": source,
        "schema_version": SCHEMA_VERSION,
        "library": library.counts(),
        "detect_scale": library.scale,
        "topology": str(topo_path),
        "project": str(project_path or (repo_root / "assets" / "projects" / f"{board_id}.json")),
        "queries_total": len(records),
        "queries_matched": len(matched),
        "queries_failed": len(records) - len(matched),
        "pose_error_px": _median([item["pose_error_px"] for item in matched]),
        "pose_error_deg": _median([item["pose_error_deg"] for item in matched]),
        "inliers_median": _median([item["inliers"] for item in matched]),
        "confidence_median": _median([item["confidence"] for item in matched]),
        "timing_ms": {
            "extract_library": round(extract_ms, 2),
            "match_median": _median([item["elapsed_ms"] for item in records]),
            "total": round((time.perf_counter() - started) * 1000.0, 2),
        },
        "window": [window_w, window_h],
        "queries": [{key: value for key, value in item.items() if key != "pose_obj"} for item in records],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _score(pose: Pose2D, truth: np.ndarray, library_xy, index, origin, window, angle_deg) -> dict:
    if pose.ok and len(library_xy):
        inside = _points_inside(truth, library_xy, window[0], window[1])
        if len(inside):
            pose.pose_error_px = pose_error_px(pose.M, truth, inside)
            truth_angle = float(np.degrees(np.arctan2(truth[1, 0], truth[0, 0])))
            delta = abs(float(pose.angle_deg) - truth_angle) % 360.0
            pose.pose_error_deg = min(delta, 360.0 - delta)
    payload = pose.to_dict()
    return {
        "index": index,
        "origin": [int(origin[0]), int(origin[1])],
        "window": [int(window[0]), int(window[1])],
        "rotation_deg": angle_deg,
        "ok": payload["ok"],
        "inliers": payload["inliers"],
        "confidence": payload["confidence"],
        "elapsed_ms": payload["elapsed_ms"],
        "pose_error_px": payload["pose_error_px"],
        "pose_error_deg": payload["pose_error_deg"],
        "message": payload["message"],
        "M": payload["M"],
        "pose_obj": pose,
    }


def _demo_component(library: TopologyExtractResult, origin: tuple[int, int], window: tuple[int, int]) -> Component | None:
    window_w, window_h = window
    x0, y0 = origin
    chosen = None
    for point in library.points:
        if point.kind != "pad":
            continue
        if x0 + 40 <= point.xy[0] <= x0 + window_w - 40 and y0 + 40 <= point.xy[1] <= y0 + window_h - 40:
            chosen = point
            break
    if chosen is None and library.points:
        chosen = library.points[len(library.points) // 2]
    if chosen is None:
        return None
    cx, cy = chosen.xy
    box = (cx - 24.0, cy - 14.0, 48.0, 28.0)
    fingerprint = geo_fingerprint(library.points, (cx, cy))
    return Component(
        component_id="C-LIVE",
        base_pixel_box=box,
        status="unchecked",
        core_params={"name": "C-LIVE", "value": "topo", "package": "pad"},
        geo_fingerprint=fingerprint,
    )


def _write_project(repo_root: Path, board_id: str, topo_path: Path, component: Component | None, project_path: Path | None) -> None:
    project_path = Path(project_path) if project_path is not None else repo_root / "assets" / "projects" / f"{board_id}.json"
    project_path.parent.mkdir(parents=True, exist_ok=True)
    relative = Path(os.path.relpath(Path(topo_path).resolve(), project_path.parent.resolve())).as_posix()
    if project_path.is_file():
        project = BoardProject.from_json(project_path.read_text(encoding="utf-8"))
    else:
        project = BoardProject(
            board_id=board_id,
            global_map_path=f"../maps/{board_id}.png",
            components=[],
            name=board_id,
            global_ref_path=f"../samples/{board_id}/global_ref.png",
        )
    project.topology_db = relative
    if component is not None:
        replaced = False
        for index, existing in enumerate(project.components):
            if existing.component_id == component.component_id or existing.geo_fingerprint:
                project.components[index] = component
                replaced = True
                break
        if not replaced:
            project.components.append(component)
    project_path.write_text(project.to_json() + "\n", encoding="utf-8")


def _core_lines(component) -> list[str]:
    if isinstance(component, dict):
        core = component.get("core_params") or {}
    else:
        core = getattr(component, "core_params", {}) or {}
    lines = []
    for key in ("name", "value", "package"):
        if key in core and str(core[key]) != "":
            lines.append(str(core[key]))
    return lines[:3]


def render_live_overlay(frame: np.ndarray, pose: Pose2D, matcher: ConstellationMatcher, components) -> np.ndarray:
    """Draw core_params bubbles only after a lock. Otherwise a visible status line."""
    canvas = cv2.cvtColor(_as_gray(frame), cv2.COLOR_GRAY2BGR)
    if not pose.ok:
        text = pose.message or "匹配置信度不足，未叠加"
        _status_banner(canvas, text, locked=False)
        return canvas
    for component in components:
        for item in matcher.project_components([component], pose):
            corners = np.array(item["corners"], dtype=np.int32)
            cv2.polylines(canvas, [corners.reshape(-1, 1, 2)], True, (0, 220, 80), 2, cv2.LINE_AA)
            _draw_param_bubble(canvas, corners, _core_lines(component) or [item["label"]])
    _status_banner(canvas, f"已锁定  conf {pose.confidence:.2f}  inliers {pose.inliers}", locked=True)
    return canvas


def _draw_param_bubble(canvas: np.ndarray, corners: np.ndarray, lines: list[str]) -> None:
    anchor = corners[0]
    x = int(np.clip(anchor[0], 4, canvas.shape[1] - 8))
    y = int(np.clip(anchor[1] - 8, 18 * len(lines) + 6, canvas.shape[0] - 4))
    width = 92
    height = 16 * len(lines) + 8
    top = y - height
    cv2.rectangle(canvas, (x, top), (min(canvas.shape[1] - 2, x + width), y), (0, 220, 80), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line[:12],
            (x + 4, top + 14 + index * 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def _status_banner(canvas: np.ndarray, text: str, *, locked: bool) -> None:
    color = (40, 140, 40) if locked else (30, 30, 180)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), color, -1)
    cv2.putText(canvas, text[:64], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_query_overlay(path: Path, frame: np.ndarray, pose: Pose2D, matcher: ConstellationMatcher, component: Component) -> None:
    canvas = render_live_overlay(frame, pose, matcher, [component])
    cv2.imwrite(str(path), canvas)


def _draw_footprint(canvas, truth_m, pose: Pose2D, width: int, height: int) -> None:
    _polyline(canvas, _corners_in_global(truth_m, width, height), (40, 200, 40))
    if pose.ok:
        try:
            _polyline(canvas, _corners_in_global(pose.M, width, height), (0, 180, 255))
        except np.linalg.LinAlgError:
            return


def _polyline(canvas, corners, color) -> None:
    pts = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)


def _corners_in_global(matrix_2x3, width: int, height: int) -> np.ndarray:
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = np.asarray(matrix_2x3, dtype=np.float64)
    inverse = np.linalg.inv(homography)
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float64)
    return _apply_h(inverse, corners)


def _points_inside(matrix_2x3, points_xy, width, height, margin: int = 12) -> np.ndarray:
    if len(points_xy) == 0:
        return points_xy
    projected = _apply_m(matrix_2x3, points_xy)
    keep = (
        (projected[:, 0] >= margin)
        & (projected[:, 1] >= margin)
        & (projected[:, 0] < width - margin)
        & (projected[:, 1] < height - margin)
    )
    return points_xy[keep]


def _spread_origins(width, height, window_w, window_h, count) -> list[tuple[int, int]]:
    max_x = max(0, width - window_w)
    max_y = max(0, height - window_h)
    fractions = ((0.05, 0.04), (0.55, 0.18), (0.12, 0.42), (0.62, 0.48), (0.08, 0.78), (0.48, 0.86))
    origins = []
    for fx, fy in fractions[: max(1, count)]:
        origins.append((int(round(fx * max_x)), int(round(fy * max_y))))
    unique = []
    for origin in origins:
        if origin not in unique:
            unique.append(origin)
    return unique or [(0, 0)]


def _median(values) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return round(float(np.median(numbers)), 3)


def _crop_roi(frame, roi_xywh):
    height, width = frame.shape[:2]
    if roi_xywh is None:
        return frame, (0, 0), width, height
    x, y, roi_w, roi_h = (int(value) for value in roi_xywh)
    if roi_w <= 0 or roi_h <= 0 or x < 0 or y < 0 or x + roi_w > width or y + roi_h > height:
        return None, (0, 0), width, height
    return frame[y : y + roi_h, x : x + roi_w], (x, y), width, height


def _working_scale(width: int, height: int, max_long_edge: int, working_scale: float | None) -> float:
    if working_scale is not None:
        return float(working_scale) if working_scale > 0 else 1.0
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return 1.0
    return float(max_long_edge) / float(long_edge)


def _feature_gray(gray: np.ndarray) -> np.ndarray:
    """Dark holes and bright pads, with illumination taken out.

    A small black-hat plus top-hat keeps pad and via centers stable when the
    frame is only moderately dark or lightly oxidized. The adaptive threshold
    then runs on that response, not on the raw photo.
    """
    height, width = gray.shape[:2]
    # About the size of a via. Larger kernels smear a lightly oxidized pad.
    ksize = 9 if min(height, width) >= 48 else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    response = cv2.max(dark, bright)
    return cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX)


def _blobs(image: np.ndarray, *, block: int, c: int, min_area: float = 30.0, max_area: float = 1200.0):
    gray = _as_gray(image)
    if gray.size == 0 or int(gray.std()) < 1:
        return []
    height, width = gray.shape[:2]
    odd = min(block, max(3, (min(height, width) // 2) * 2 - 1))
    if odd % 2 == 0:
        odd -= 1
    odd = max(3, odd)
    # Illumination-free peaks when the frame actually has a constellation.
    # Sparse drawings fall back to the raw adaptive mask so both stay consistent.
    peaked = _contours_to_points(255 - _feature_gray(gray), odd, c, min_area, max_area, width, height)
    if len(peaked) >= 20:
        return peaked
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return _contours_to_points(blurred, odd, c, min_area, max_area, width, height)


def _contours_to_points(gray_or_blur, odd, c, min_area, max_area, width, height):
    binary = cv2.adaptiveThreshold(gray_or_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, odd, c)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in contours:
        bounds_x, bounds_y, bounds_w, bounds_h = cv2.boundingRect(contour)
        if bounds_x <= 1 or bounds_y <= 1 or bounds_x + bounds_w >= width - 1 or bounds_y + bounds_h >= height - 1:
            continue
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0 if perimeter < 1.0 else float(4.0 * np.pi * area / (perimeter * perimeter))
        moments = cv2.moments(contour)
        if moments["m00"] <= 1.0:
            continue
        x = float(moments["m10"] / moments["m00"])
        y = float(moments["m01"] / moments["m00"])
        if circularity >= 0.62 and area <= 280.0:
            kind = "hole"
        elif circularity < 0.38 and area >= 300.0:
            kind = "slot"
        else:
            kind = "pad"
        found.append((x, y, kind, area, circularity))
    return _suppress(found, radius=3.0)


def _suppress(points, radius: float):
    kept = []
    for point in sorted(points, key=lambda item: -item[3]):
        if any((point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2 < radius * radius for other in kept):
            continue
        kept.append(point)
    return kept


def _constellation_descriptors(xy: np.ndarray, neighbors: int) -> np.ndarray:
    count = len(xy)
    width = max(1, neighbors - 1) * 2
    if count < 3:
        return np.zeros((count, width), dtype=np.float64)
    k = min(neighbors, count - 1)
    distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argpartition(distances, k - 1, axis=1)[:, :k]
    row = np.arange(count)[:, None]
    nearest = nearest[row, np.argsort(distances[row, nearest], axis=1)]
    dist = distances[row, nearest]
    vectors = xy[nearest] - xy[:, None, :]
    angles = np.arctan2(vectors[..., 1], vectors[..., 0])
    angles = (angles - angles[:, :1] + np.pi) % (2.0 * np.pi) - np.pi
    dist = dist / np.maximum(dist[:, :1], 1e-6)
    feat = np.concatenate([dist[:, 1:], angles[:, 1:]], axis=1)
    if feat.shape[1] < width:
        feat = np.pad(feat, ((0, 0), (0, width - feat.shape[1])))
    return feat


def _putative_pairs(query_desc, library_desc, ratio: float) -> np.ndarray:
    if len(query_desc) == 0 or len(library_desc) < 2:
        return np.zeros((0, 2), dtype=np.int32)
    delta = query_desc[:, None, :] - library_desc[None, :, :]
    dist = np.sqrt(np.sum(delta * delta, axis=2))
    nearest = np.argpartition(dist, 1, axis=1)[:, :2]
    row = np.arange(len(query_desc))[:, None]
    order = np.argsort(dist[row, nearest], axis=1)
    best = nearest[row, order][:, 0]
    second = nearest[row, order][:, 1]
    best_d = dist[np.arange(len(query_desc)), best]
    second_d = dist[np.arange(len(query_desc)), second]
    keep = (second_d > 1e-9) & (best_d / np.maximum(second_d, 1e-9) < ratio)
    return np.stack([np.flatnonzero(keep), best[keep]], axis=1).astype(np.int32)


def _ransac_similarity(src, dst, *, thresh: float, iterations: int):
    rng = np.random.default_rng(0)
    count = len(src)
    best_mask = None
    best_h = None
    best_count = 0
    if count < 2:
        return None, np.zeros(0, dtype=bool)
    for _ in range(iterations):
        i, j = rng.choice(count, 2, replace=False)
        homography = _similarity_from_pair(src[i], src[j], dst[i], dst[j])
        if homography is None:
            continue
        error = np.linalg.norm(_apply_h(homography, src) - dst, axis=1)
        mask = error <= thresh
        hits = int(mask.sum())
        if hits > best_count:
            best_count = hits
            best_mask = mask
            best_h = homography
    if best_mask is None:
        return None, np.zeros(count, dtype=bool)
    return best_h, best_mask


def _similarity_from_pair(src_a, src_b, dst_a, dst_b):
    vector_src = np.asarray(src_b, dtype=np.float64) - np.asarray(src_a, dtype=np.float64)
    vector_dst = np.asarray(dst_b, dtype=np.float64) - np.asarray(dst_a, dtype=np.float64)
    length_src = float(np.hypot(*vector_src))
    length_dst = float(np.hypot(*vector_dst))
    if length_src < 8.0 or length_dst < 8.0:
        return None
    scale = length_dst / length_src
    if not 0.7 <= scale <= 1.4:
        return None
    angle = float(np.arctan2(vector_dst[1], vector_dst[0]) - np.arctan2(vector_src[1], vector_src[0]))
    cosine = np.cos(angle) * scale
    sine = np.sin(angle) * scale
    linear = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    translation = np.asarray(dst_a, dtype=np.float64) - linear @ np.asarray(src_a, dtype=np.float64)
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :2] = linear
    homography[:2, 2] = translation
    return homography


def _refine_similarity(src, dst):
    if len(src) < 2:
        return None
    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    centered_src = src - mean_src
    centered_dst = dst - mean_dst
    variance = float(np.sum(centered_src * centered_src) / len(src))
    if variance < 1e-8:
        return None
    covariance = (centered_dst.T @ centered_src) / len(src)
    left, _singular, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left = left.copy()
        left[:, -1] *= -1.0
        rotation = left @ right_t
    scale = float(np.trace(rotation.T @ covariance) / variance)
    if not 0.7 <= scale <= 1.4:
        return None
    translation = mean_dst - scale * (rotation @ mean_src)
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :2] = scale * rotation
    homography[:2, 2] = translation
    return homography


def _explained_query_points(homography, library_xy, query_xy, thresh: float) -> int:
    """How many query blobs land on a library point after the library→frame pose."""
    if len(library_xy) == 0 or len(query_xy) == 0:
        return 0
    projected = _apply_h(homography, library_xy)
    distances = np.linalg.norm(query_xy[:, None, :] - projected[None, :, :], axis=2)
    return int(np.count_nonzero(distances.min(axis=1) <= thresh))


def _similarity_parts(homography: np.ndarray) -> tuple[float, float]:
    linear = np.asarray(homography, dtype=np.float64)[:2, :2]
    scale = float(np.linalg.norm(linear[:, 0]))
    angle = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
    return scale, angle


def _apply_h(homography, points) -> np.ndarray:
    matrix = np.asarray(homography, dtype=np.float64)
    homo = np.concatenate([np.asarray(points, dtype=np.float64), np.ones((len(points), 1))], axis=1)
    mapped = homo @ matrix.T
    return mapped[:, :2] / np.maximum(mapped[:, 2:3], 1e-12)


def _apply_m(matrix_2x3, points) -> np.ndarray:
    matrix = np.asarray(matrix_2x3, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    return pts @ matrix[:, :2].T + matrix[:, 2]


def _identity_m() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _box_corners(box) -> np.ndarray:
    x, y, w, h = (float(value) for value in box)
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)


def _component_view(component) -> tuple[tuple, str, str]:
    if isinstance(component, dict):
        box = component["base_pixel_box"]
        component_id = str(component.get("component_id", ""))
        core = component.get("core_params") or {}
        label = str(core.get("name") or component_id)
        return box, component_id, label
    core = getattr(component, "core_params", {}) or {}
    return component.base_pixel_box, component.component_id, str(core.get("name") or component.component_id)


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def medium_dark(image: np.ndarray, scale: float = 0.62, gamma: float = 1.25) -> np.ndarray:
    """Mid-gray board pulled down, still readable. Not a near-black frame."""
    values = image.astype(np.float32) / 255.0
    return np.clip((values * scale) ** gamma * 255.0, 0, 255).astype(np.uint8)


def mild_oxidize(image: np.ndarray, sigma: float = 0.8, contrast: float = 0.90) -> np.ndarray:
    """Light blur, a little contrast loss, and a warm cast."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma) if sigma > 0 else image
    values = blurred.astype(np.float32)
    mean = values.mean(axis=(0, 1), keepdims=True)
    values = mean + (values - mean) * contrast
    if values.ndim == 3 and values.shape[2] == 3:
        values[:, :, 0] *= 0.92
        values[:, :, 2] = np.clip(values[:, :, 2] * 1.03 + 4.0, 0, 255)
    return np.clip(values, 0, 255).astype(np.uint8)


def _extreme_dark(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32) / 255.0
    return np.clip((values * 0.28) ** 1.8 * 255.0, 0, 255).astype(np.uint8)


def _extreme_oxidize(image: np.ndarray) -> np.ndarray:
    return mild_oxidize(image, sigma=2.4, contrast=0.55)


def _bright_glare(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    center = (int(out.shape[1] * 0.45), int(out.shape[0] * 0.42))
    cv2.ellipse(out, center, (180, 120), 0, 0, 360, (255, 255, 255), -1)
    return cv2.addWeighted(image, 0.45, out, 0.55, 0)


def _burn_patch(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    y, x = int(out.shape[0] * 0.35), int(out.shape[1] * 0.2)
    cv2.rectangle(out, (x, y), (x + 160, y + 90), (12, 12, 12), -1)
    return out


_STRESS_ORIGINS = ((26, 48), (286, 216), (62, 504), (322, 576), (42, 935), (250, 1031), (480, 180))


def run_topology_stress(
    image: np.ndarray,
    *,
    out_dir: Path,
    window: tuple[int, int] = (560, 720),
    max_long_edge: int = 2048,
) -> dict:
    """Clean library versus stressed crops. Medium modes are the acceptance set."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = TopologyExtractor(max_long_edge=max_long_edge)
    library = extractor.extract(image)
    matcher = ConstellationMatcher()
    matcher.set_topology(library)
    localizer = GeometricLocalizer(extractor, matcher)
    modes = {
        "clean": image,
        "medium_dark": medium_dark(image),
        "mild_oxidize": mild_oxidize(image),
        "dark": _extreme_dark(image),
        "oxidize_blur": _extreme_oxidize(image),
        "bright_glare": _bright_glare(image),
        "burn_patch": _burn_patch(image),
    }
    summary = {
        "primary": ["clean", "medium_dark", "mild_oxidize"],
        "window": [window[0], window[1]],
        "library_points": library.counts(),
        "modes": {},
    }
    window_w, window_h = window
    for name, frame in modes.items():
        rows = []
        for origin_x, origin_y in _STRESS_ORIGINS:
            crop = frame[origin_y : origin_y + window_h, origin_x : origin_x + window_w]
            if crop.shape[0] != window_h or crop.shape[1] != window_w:
                continue
            pose = localizer.track(crop)
            error = None
            if pose.ok:
                error = round(pose_error_px(pose.M, translation_m(origin_x, origin_y), library.xy()), 3)
            rows.append(
                {
                    "origin": [origin_x, origin_y],
                    "ok": pose.ok,
                    "inliers": pose.inliers if pose.ok else 0,
                    "confidence": round(pose.confidence, 4) if pose.ok else 0.0,
                    "pose_error_px": error,
                    "elapsed_ms": round(pose.elapsed_ms, 2),
                    "message": pose.message,
                }
            )
        matched = [row for row in rows if row["ok"]]
        summary["modes"][name] = {
            "matched": len(matched),
            "total": len(rows),
            "failed": len(rows) - len(matched),
            "pose_error_px_median": _median([row["pose_error_px"] for row in matched]),
            "confidence_median": _median([row["confidence"] for row in matched]),
            "inliers_median": _median([row["inliers"] for row in matched]),
            "rows": rows,
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def run_topology_live_loop(
    image: np.ndarray,
    *,
    out_dir: Path,
    component: Component,
    window: tuple[int, int] = (560, 720),
    frames: int = 16,
    max_long_edge: int = 2048,
) -> dict:
    """Slide a window across the board and overlay core_params. No camera required."""
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    extractor = TopologyExtractor(max_long_edge=max_long_edge)
    library = extractor.extract(image)
    matcher = ConstellationMatcher()
    matcher.set_topology(library)
    localizer = GeometricLocalizer(extractor, matcher)
    window_w, window_h = window
    sequence = _live_sequence(image, window, frames)
    rows = []
    written = []
    for index, (crop, origin, kind) in enumerate(sequence):
        pose = localizer.track(crop)
        error = None
        if pose.ok:
            error = round(pose_error_px(pose.M, translation_m(*origin), library.xy()), 3)
        canvas = render_live_overlay(crop, pose, matcher, [component])
        path = frames_dir / f"frame_{index:03d}.png"
        cv2.imwrite(str(path), canvas)
        written.append(str(path))
        rows.append(
            {
                "index": index,
                "origin": [int(origin[0]), int(origin[1])],
                "kind": kind,
                "ok": pose.ok,
                "confidence": round(pose.confidence, 4),
                "inliers": pose.inliers,
                "pose_error_px": error,
                "elapsed_ms": round(pose.elapsed_ms, 2),
                "message": pose.message,
                "overlay": str(path),
            }
        )
    empty = ConstellationMatcher()
    blocked = empty.localize(sequence[0][0])
    blocked_canvas = render_live_overlay(sequence[0][0], blocked, empty, [component])
    blocked_path = out_dir / "no_library.png"
    cv2.imwrite(str(blocked_path), blocked_canvas)
    video_path = out_dir / "live_loop.mp4"
    _write_loop_video(written, video_path)
    matched = [row for row in rows if row["ok"]]
    summary = {
        "frames": len(rows),
        "matched": len(matched),
        "failed": len(rows) - len(matched),
        "match_rate": round(len(matched) / len(rows), 3) if rows else 0.0,
        "confidence_median": _median([row["confidence"] for row in matched]),
        "pose_error_px_median": _median([row["pose_error_px"] for row in matched]),
        "elapsed_ms_median": _median([row["elapsed_ms"] for row in rows]),
        "no_library_message": blocked.message,
        "no_library_overlay": str(blocked_path),
        "video": str(video_path) if video_path.is_file() else None,
        "frames_dir": str(frames_dir),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _live_sequence(image: np.ndarray, window: tuple[int, int], count: int):
    window_w, window_h = window
    height, width = image.shape[:2]
    origin_x, origin_y = 40, 40
    step = 28
    sequence = []
    for index in range(count):
        x = min(origin_x + index * step, max(0, width - window_w))
        y = origin_y if index < count - 4 else min(origin_y + 120, max(0, height - window_h))
        crop = image[y : y + window_h, x : x + window_w].copy()
        kind = "clean"
        if index >= count - 4:
            crop = medium_dark(crop)
            kind = "medium_dark"
        sequence.append((crop, (x, y), kind))
    return sequence


def _write_loop_video(frame_paths: list[str], path: Path) -> None:
    if not frame_paths:
        return
    first = cv2.imread(frame_paths[0])
    if first is None:
        return
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        6.0,
        (first.shape[1], first.shape[0]),
    )
    if not writer.isOpened():
        return
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is not None and frame.shape[:2] == first.shape[:2]:
            writer.write(frame)
    writer.release()


def _live_component(library: TopologyExtractResult, project_path: Path | None, board_id: str, window: tuple[int, int]) -> Component:
    core = {"name": "C-LIVE", "value": "10K", "package": "0805"}
    path = project_path or Path("assets/projects") / f"{board_id}.json"
    if path.is_file():
        project = BoardProject.from_json(path.read_text(encoding="utf-8"))
        for component in project.components:
            if component.component_id == "C-LIVE":
                core = {key: str(component.core_params.get(key, "")) for key in ("name", "value", "package")}
                break
    component = _demo_component(library, (40, 40), window)
    if component is None:
        component = Component(
            component_id="C-LIVE",
            base_pixel_box=(80, 80, 48, 28),
            status="unchecked",
            core_params=core,
        )
    component.component_id = "C-LIVE"
    component.core_params = core
    return component


def _parse_window(text: str) -> tuple[int, int]:
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise ValueError("--window must look like 560x720")
    return int(parts[0]), int(parts[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拓扑抽取 + 星座 Live 定位（D16）")
    parser.add_argument("--global", dest="global_path", type=Path, required=True)
    parser.add_argument("--board-id", default="SAMPLE-PARTIAL-001")
    parser.add_argument("--out", type=Path, default=Path("artifacts/topology_live"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--topology", type=Path, default=None)
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--window", default="560x720")
    parser.add_argument("--queries", type=int, default=6)
    parser.add_argument("--rotation", type=float, default=8.0)
    parser.add_argument("--stress", action="store_true", help="写 artifacts/topology_stress，中等暗光与轻度氧化为主")
    parser.add_argument("--live-loop", action="store_true", help="用滑动裁窗模拟 Live，并叠 core_params 气泡")
    parser.add_argument("--loop-frames", type=int, default=16)
    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=2048,
        help="抽取时的长边上限。类默认是 1280；这块样例长边约 1919，冒烟用 2048 保持原分辨率",
    )
    args = parser.parse_args(argv)
    image = cv2.imread(str(args.global_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"无法读取 {args.global_path}")
        return 1
    try:
        window = _parse_window(args.window)
    except ValueError as exc:
        print(str(exc))
        return 1
    if args.stress or args.live_loop:
        code = 0
        if args.stress:
            stress_out = args.out if args.out != Path("artifacts/topology_live") else Path("artifacts/topology_stress")
            stress = run_topology_stress(image, out_dir=stress_out, window=window, max_long_edge=args.max_long_edge)
            brief = {
                name: {
                    "matched": mode["matched"],
                    "total": mode["total"],
                    "confidence_median": mode["confidence_median"],
                    "pose_error_px_median": mode["pose_error_px_median"],
                }
                for name, mode in stress["modes"].items()
            }
            brief["summary"] = str(stress_out / "summary.json")
            print(json.dumps(brief, ensure_ascii=False))
            primary = stress["modes"]
            if primary["clean"]["matched"] < 7 or primary["medium_dark"]["matched"] < 5 or primary["mild_oxidize"]["matched"] < 5:
                code = 1
        if args.live_loop:
            loop_out = Path("artifacts/topology_live_loop")
            library = TopologyExtractor(max_long_edge=args.max_long_edge).extract(image)
            component = _live_component(library, args.project, args.board_id, window)
            loop = run_topology_live_loop(
                image,
                out_dir=loop_out,
                component=component,
                window=window,
                frames=max(4, args.loop_frames),
                max_long_edge=args.max_long_edge,
            )
            print(json.dumps({
                "frames": loop["frames"],
                "matched": loop["matched"],
                "match_rate": loop["match_rate"],
                "confidence_median": loop["confidence_median"],
                "pose_error_px_median": loop["pose_error_px_median"],
                "elapsed_ms_median": loop["elapsed_ms_median"],
                "no_library_message": loop["no_library_message"],
                "summary": str(loop_out / "summary.json"),
                "video": loop["video"],
            }, ensure_ascii=False))
            if loop["matched"] < max(1, int(0.7 * loop["frames"])):
                code = 1
        return code
    summary = run_topology_live(
        image,
        board_id=args.board_id,
        source=str(args.global_path),
        out_dir=args.out,
        repo_root=args.repo_root,
        topology_path=args.topology,
        project_path=args.project,
        window=window,
        queries=max(1, args.queries),
        rotation_deg=args.rotation,
        max_long_edge=args.max_long_edge,
    )
    brief = {key: summary[key] for key in (
        "library",
        "queries_total",
        "queries_matched",
        "queries_failed",
        "pose_error_px",
        "pose_error_deg",
        "inliers_median",
        "confidence_median",
        "timing_ms",
        "topology",
        "project",
    )}
    brief["summary"] = str(args.out / "summary.json")
    print(json.dumps(brief, ensure_ascii=False))
    return 0 if summary["queries_matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
