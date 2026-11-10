"""Structure-tree recognition model for later single-component matching.

This module encodes the planned data model: binary shape primitives, a
component unit (a QR-code-like minimal record), combination / tree nodes,
and a board-level snapshot. ``extract_structure_tree`` is a first, cheap
extractor that drops tiny or messy blobs, classifies remaining contours,
and records relative positions. It is not the live pose matcher;
``ConstellationMatcher`` remains the default localizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Tiny clutter and speckled gradient crumbs stay out of the recognition map.
_MIN_AREA = 48
_MAX_NOISE_AREA = 24
_MIN_SOLIDITY = 0.35
# Classification tolerances: tight enough to be useful, loose enough not to flake.
_CIRCULARITY_MIN = 0.72
_CIRCLE_ASPECT_MAX = 1.40
_SQUARE_ASPECT_MAX = 1.25
_RECT_EXTENT_MIN = 0.72
_LINE_ASPECT_MIN = 4.0
_BLOCK_MIN_AREA = 400
_NEAR_PX = 160.0

KIND_CIRCLE = "circle"
KIND_SQUARE = "square"
KIND_RECT = "rect"
KIND_LINE = "line"
KIND_BLOCK = "block"
SHAPE_KINDS = (KIND_SQUARE, KIND_RECT, KIND_CIRCLE, KIND_LINE, KIND_BLOCK)


@dataclass
class ShapePrimitive:
    """A clean binary shape: square, rectangle, circle, line, or large block."""

    kind: str
    center: tuple[float, float]
    size: tuple[float, float]
    area: float
    orientation_deg: float = 0.0
    contour: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "center": [round(float(self.center[0]), 2), round(float(self.center[1]), 2)],
            "size": [round(float(self.size[0]), 2), round(float(self.size[1]), 2)],
            "area": round(float(self.area), 1),
            "orientation_deg": round(float(self.orientation_deg), 2),
            "contour": [[round(float(x), 1), round(float(y), 1)] for x, y in self.contour],
        }


@dataclass
class RelativeRelation:
    """Position of one primitive or unit relative to another."""

    src: int
    dst: int
    dx: float
    dy: float
    distance: float

    def to_dict(self) -> dict:
        return {
            "src": int(self.src),
            "dst": int(self.dst),
            "dx": round(float(self.dx), 2),
            "dy": round(float(self.dy), 2),
            "distance": round(float(self.distance), 2),
        }


@dataclass
class ComponentUnit:
    """QR-code-like minimal data unit: one component plus nearby relations.

    Outline, size, and positions relative to pads, small vias, traces, or
    labels form one tolerant record. Matching must not be pixel-exact.
    """

    unit_id: str
    outline: ShapePrimitive | None
    size: tuple[float, float]
    primitives: list[int] = field(default_factory=list)
    relations: list[RelativeRelation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.unit_id,
            "outline": None if self.outline is None else self.outline.to_dict(),
            "size": [round(float(self.size[0]), 2), round(float(self.size[1]), 2)],
            "primitives": list(self.primitives),
            "relations": [item.to_dict() for item in self.relations],
        }


@dataclass
class CombinationNode:
    """Two or more nearby component units that cross-check each other."""

    node_id: str
    members: list[str]
    relations: list[RelativeRelation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.node_id,
            "members": list(self.members),
            "relations": [item.to_dict() for item in self.relations],
        }


@dataclass
class StructureSnapshot:
    """Board-level or view-level structure-tree snapshot."""

    primitives: list[ShapePrimitive] = field(default_factory=list)
    units: list[ComponentUnit] = field(default_factory=list)
    combinations: list[CombinationNode] = field(default_factory=list)
    relations: list[RelativeRelation] = field(default_factory=list)
    image_size: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        return {
            "image_size": None if self.image_size is None else [int(self.image_size[0]), int(self.image_size[1])],
            "primitives": [item.to_dict() for item in self.primitives],
            "units": [item.to_dict() for item in self.units],
            "combinations": [item.to_dict() for item in self.combinations],
            "relations": [item.to_dict() for item in self.relations],
        }

    def kinds(self) -> list[str]:
        return [item.kind for item in self.primitives]


def extract_structure_tree(
    image: np.ndarray,
    *,
    min_area: int = _MIN_AREA,
    near_px: float = _NEAR_PX,
) -> StructureSnapshot:
    """Build a clean recognition map from an image.

    Tiny clutter and messy gradient crumbs are dropped. Remaining contours
    are classified into circle / rect / square / line-like / large block,
    then relative positions are recorded. This is a first extractable slice,
    not a full component-pin-trace tree and not the live pose matcher.
    """
    if image is None or image.size == 0:
        return StructureSnapshot()
    height, width = image.shape[:2]
    binary = _clean_binary(image)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    primitives: list[ShapePrimitive] = []
    for contour in contours:
        primitive = _classify_contour(contour, min_area=min_area)
        if primitive is not None:
            primitives.append(primitive)
    primitives.sort(key=lambda item: (-item.area, item.center[1], item.center[0]))
    relations = _pairwise_relations(primitives)
    units = _component_units(primitives, relations, near_px=near_px)
    combinations = _combination_nodes(units, near_px=near_px)
    return StructureSnapshot(
        primitives=primitives,
        units=units,
        combinations=combinations,
        relations=relations,
        image_size=(int(width), int(height)),
    )


def _clean_binary(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _threshold, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Drop isolated speckles; keep lines and component-scale blocks.
    open_kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)
    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    kept = np.zeros_like(cleaned)
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= _MAX_NOISE_AREA:
            kept[labels == label] = 255
    return kept


def _classify_contour(contour: np.ndarray, *, min_area: int) -> ShapePrimitive | None:
    area = float(cv2.contourArea(contour))
    if area < float(min_area):
        return None
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull)) or 1.0
    solidity = area / hull_area
    if solidity < _MIN_SOLIDITY and area < _BLOCK_MIN_AREA:
        # Messy gradient crumbs: irregular and not a large uniform block.
        return None
    perimeter = float(cv2.arcLength(contour, True)) or 1.0
    circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
    rect = cv2.minAreaRect(contour)
    (center_x, center_y), (raw_w, raw_h), angle = rect
    width = float(max(raw_w, raw_h, 1.0))
    height = float(max(min(raw_w, raw_h), 1.0))
    aspect = width / height
    extent = area / max(width * height, 1.0)
    if aspect >= _LINE_ASPECT_MIN and extent >= 0.45:
        kind = KIND_LINE
    elif circularity >= _CIRCULARITY_MIN and aspect <= _CIRCLE_ASPECT_MAX:
        kind = KIND_CIRCLE
        width = height = float(np.sqrt(area / np.pi) * 2.0)
    elif extent >= _RECT_EXTENT_MIN and aspect <= _SQUARE_ASPECT_MAX:
        kind = KIND_SQUARE
    elif extent >= _RECT_EXTENT_MIN:
        kind = KIND_RECT
    elif area >= _BLOCK_MIN_AREA:
        kind = KIND_BLOCK
    else:
        return None
    approx = cv2.approxPolyDP(contour, max(1.0, 0.01 * perimeter), True)
    points = [[float(pt[0][0]), float(pt[0][1])] for pt in approx]
    return ShapePrimitive(
        kind=kind,
        center=(float(center_x), float(center_y)),
        size=(width, height),
        area=area,
        orientation_deg=float(angle),
        contour=points,
    )


def _pairwise_relations(primitives: list[ShapePrimitive]) -> list[RelativeRelation]:
    relations: list[RelativeRelation] = []
    for src, left in enumerate(primitives):
        for dst in range(src + 1, len(primitives)):
            right = primitives[dst]
            dx = float(right.center[0] - left.center[0])
            dy = float(right.center[1] - left.center[1])
            relations.append(
                RelativeRelation(
                    src=src,
                    dst=dst,
                    dx=dx,
                    dy=dy,
                    distance=float(np.hypot(dx, dy)),
                )
            )
    return relations


def _component_units(
    primitives: list[ShapePrimitive],
    relations: list[RelativeRelation],
    *,
    near_px: float,
) -> list[ComponentUnit]:
    """Treat each outline-like shape as a unit and attach nearby support shapes."""
    units: list[ComponentUnit] = []
    outline_kinds = {KIND_RECT, KIND_SQUARE, KIND_CIRCLE, KIND_BLOCK}
    for index, primitive in enumerate(primitives):
        if primitive.kind not in outline_kinds:
            continue
        nearby = [
            item
            for item in relations
            if item.distance <= near_px and (item.src == index or item.dst == index)
        ]
        members = {index}
        for item in nearby:
            members.add(item.dst if item.src == index else item.src)
        units.append(
            ComponentUnit(
                unit_id=f"u{index:03d}",
                outline=primitive,
                size=primitive.size,
                primitives=sorted(members),
                relations=nearby,
            )
        )
    return units


def _combination_nodes(units: list[ComponentUnit], *, near_px: float) -> list[CombinationNode]:
    """Pair nearby units into combination nodes; more units add cross-checks."""
    nodes: list[CombinationNode] = []
    for index, left in enumerate(units):
        if left.outline is None:
            continue
        for other in units[index + 1 :]:
            if other.outline is None:
                continue
            dx = float(other.outline.center[0] - left.outline.center[0])
            dy = float(other.outline.center[1] - left.outline.center[1])
            distance = float(np.hypot(dx, dy))
            if distance > near_px:
                continue
            nodes.append(
                CombinationNode(
                    node_id=f"c{len(nodes):03d}",
                    members=[left.unit_id, other.unit_id],
                    relations=[
                        RelativeRelation(src=index, dst=units.index(other), dx=dx, dy=dy, distance=distance)
                    ],
                )
            )
    return nodes
