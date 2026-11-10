"""Load, save, and query a ``BoardProject``.

``list_visible`` is a geometric filter only: it projects each component center
with the same global→frame direction as the trackers, then keeps centers that
fall inside the viewport polygon. It does not estimate a matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from model.schema import (
    CORE_PARAM_KEYS,
    KNOWN_STATUSES,
    BoardProject,
    Component,
    SchemaError,
    validate_project,
)


class ProjectStore:
    """In-memory project plus the path it was loaded from."""

    def __init__(self) -> None:
        self.project: BoardProject | None = None
        self.path: str | None = None

    def load(self, path: str) -> BoardProject:
        text = Path(path).read_text(encoding="utf-8")
        project = BoardProject.from_json(text)
        self.project = project
        self.path = path
        return project

    def save(self, project: BoardProject, path: str | None = None) -> None:
        target = path if path is not None else self.path
        if not target:
            raise ValueError("path is required when no project file is loaded")
        payload = project.to_dict()
        errors = validate_project(payload)
        if errors:
            raise SchemaError(errors)
        Path(target).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.project = project
        self.path = target

    def update_component(
        self,
        component_id: str,
        *,
        core: dict | None = None,
        extended: dict | None = None,
        status: str | None = None,
    ) -> None:
        """Patch bubble fields. Only ``name``, ``value``, and ``package`` are allowed."""
        component = self._require_component(component_id)
        if core is not None:
            if "status" in core:
                raise SchemaError(["status is top-level; do not put it in core_params"])
            unknown = [key for key in core if key not in CORE_PARAM_KEYS]
            if unknown:
                raise SchemaError(
                    ["core_params only allows name, value, package; got " + ", ".join(unknown)]
                )
            merged = {key: component.core_params.get(key, "") for key in CORE_PARAM_KEYS}
            for key, value in core.items():
                if isinstance(value, (dict, list)):
                    raise SchemaError([f"core_params.{key} must not be nested"])
                merged[key] = value
            component.core_params = merged
        if extended is not None:
            if not isinstance(extended, dict):
                raise SchemaError(["extended_params patch must be an object"])
            merged_extended = dict(component.extended_params)
            merged_extended.update(extended)
            component.extended_params = merged_extended
        if status is not None:
            if status not in KNOWN_STATUSES:
                raise SchemaError(["status must be unchecked, checked, or doubt"])
            component.status = status

    def list_visible(self, viewport_polygon, transform) -> list[Component]:
        """Components whose projected box center lies inside ``viewport_polygon``.

        ``transform`` is a 3×3 homography or a 2×3 affine, global pixel →
        current frame pixel. The center of ``base_pixel_box`` is the tested
        point (not the centroid of a perspective quad). Points on the polygon
        edge count as visible. An empty polygon or a missing project yields
        ``[]``. A transform with the wrong shape raises ``ValueError``.
        """
        if self.project is None or transform is None:
            return []
        polygon = _as_polygon(viewport_polygon)
        if polygon is None:
            return []
        homography = _to_homography(transform)
        visible: list[Component] = []
        for component in self.project.components:
            center = _box_center(component.base_pixel_box)
            projected = _project_xy(center, homography)
            if projected is None:
                continue
            if _inside(projected, polygon):
                visible.append(component)
        return visible

    def _require_component(self, component_id: str) -> Component:
        if self.project is None:
            raise RuntimeError("no project loaded")
        for component in self.project.components:
            if component.component_id == component_id:
                return component
        raise KeyError(component_id)


def _as_polygon(viewport_polygon) -> np.ndarray | None:
    if viewport_polygon is None:
        return None
    points = []
    for point in viewport_polygon:
        if len(point) < 2:
            raise ValueError("viewport polygon points must be (x, y)")
        points.append((float(point[0]), float(point[1])))
    if len(points) < 3:
        return None
    return np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)


def _to_homography(transform) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape == (3, 3):
        return matrix
    if matrix.shape == (2, 3):
        return np.vstack([matrix, [0.0, 0.0, 1.0]])
    raise ValueError("transform must be 2x3 or 3x3 (global pixel → frame pixel)")


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, width, height = box
    return (float(x) + float(width) / 2.0, float(y) + float(height) / 2.0)


def _project_xy(point: tuple[float, float], homography: np.ndarray) -> tuple[float, float] | None:
    vector = homography @ np.array([point[0], point[1], 1.0], dtype=np.float64)
    if abs(float(vector[2])) < 1e-9:
        return None
    return (float(vector[0] / vector[2]), float(vector[1] / vector[2]))


def _inside(point: tuple[float, float], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, point, False) >= 0
