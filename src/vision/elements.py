"""Typed board-element separation and red-region seam checks.

Interval capture already locks a similarity from pad and via centroids. That
lock does not look at silkscreen, outlines, or traces, so a repeated via
lattice can still ghost a label or shift a package. This module is the check
that runs on an accepted still, not on the live preview:

1. Layered OpenCV extractors emit typed candidates (hole, pad, trace, text,
   outline, pin, color). No Mask R-CNN, SAM, or YOLO.
2. A few conflict rules clean the set: a drill inside a pad stays a hole, a
   letter is not a trace, pins are the regular small pads along an outline.
3. ``align_red_region`` solves a similarity (rotation, translation, a little
   scale, no shear) so the shared red overlap coincides, then scores those
   elements. Crops stay on each element for a later patch recognizer.

Acceptance rule
---------------
Score only elements that fall inside the overlap implied by the similarity
(the red region). Match typed-to-typed.

- Primary: at least 4 interior holes, pads, or pins, median residual
  ``<= ACCEPT_RESIDUAL_PX`` (3.2, the same shot-reject residual as the
  geometric lock). Blending is unchanged: overlap is still not averaged
  above 2.5 px (``board_lock.SAFE_RESIDUAL_PX``).
- Secondary families are text, outline, and trace. A family counts when
  both tiles show at least one instance in the red region. Every such
  family must agree. A family whose members all land more than
  ``CONFLICT_RESIDUAL_PX`` (6) away rejects the shot. One agreeing family
  is enough to keep a via field from being "lattice only".
- A periodic via/pad lattice with no agreeing secondary family is
  ``lattice_only`` and must not lock. An irregular pad/hole constellation
  with no secondary family still locks, so boards whose overlap only shows
  pads and vias keep the geometry-lock behavior.
- A shape cut by the frame is partial. Its contour (not its centroid) is
  scored against the complete same-type contour, and a small translation
  may snap the partial outline onto the complete one. The update stays a
  similarity.

``PatchClassifier`` is the recognition hook. The default returns the
classical kind. This slice does not load weights and does not add a runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

# Shot-level agreement. Blending still refuses to average above board_lock.SAFE_RESIDUAL_PX (2.5).
ACCEPT_RESIDUAL_PX = 3.2
CONFLICT_RESIDUAL_PX = 6.0
MIN_PRIMARY_HITS = 4

KIND_HOLE = "hole"
KIND_PAD = "pad"
KIND_TRACE = "trace"
KIND_TEXT = "text"
KIND_OUTLINE = "outline"
KIND_PIN = "pin"
KIND_COLOR = "color"
ELEMENT_KINDS = (KIND_HOLE, KIND_PAD, KIND_TRACE, KIND_TEXT, KIND_OUTLINE, KIND_PIN, KIND_COLOR)
PRIMARY_KINDS = (KIND_HOLE, KIND_PAD, KIND_PIN)
SECONDARY_KINDS = (KIND_TEXT, KIND_OUTLINE, KIND_TRACE)
# Both tiles must show at least this many interior instances before the family can veto.
_SECONDARY_MIN = {KIND_TEXT: 1, KIND_OUTLINE: 1, KIND_TRACE: 1}
_SECONDARY_DESC_MAX = 0.75
_SECONDARY_PAIR_PX = 80.0
_KIND_CAP = {
    KIND_HOLE: 48,
    KIND_PAD: 48,
    KIND_PIN: 32,
    KIND_TEXT: 24,
    KIND_OUTLINE: 16,
    KIND_TRACE: 24,
    KIND_COLOR: 8,
}
_MAX_ELEMENTS = 160


@dataclass
class BoardElement:
    """One separated primitive plus the crop a later recognizer would see."""

    element_id: str
    kind: str
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    descriptor: np.ndarray
    crop: np.ndarray | None
    area: float
    circularity: float
    aspect: float = 1.0
    solidity: float = 1.0
    stroke_width: float = 0.0
    partial: bool = False
    parent_id: str | None = None
    match_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))

    def to_dict(self) -> dict:
        contour = _simplify_contour(self.contour, 24)
        return {
            "id": self.element_id,
            "kind": self.kind,
            "bbox": [int(value) for value in self.bbox],
            "centroid": [round(float(self.centroid[0]), 2), round(float(self.centroid[1]), 2)],
            "descriptor": [round(float(value), 4) for value in np.asarray(self.descriptor).tolist()],
            "contour": [[round(float(point[0]), 1), round(float(point[1]), 1)] for point in contour],
            "area": round(float(self.area), 1),
            "circularity": round(float(self.circularity), 3),
            "aspect": round(float(self.aspect), 3),
            "partial": bool(self.partial),
            "parent_id": self.parent_id,
            "crop_size": None if self.crop is None else [int(self.crop.shape[1]), int(self.crop.shape[0])],
        }


@dataclass
class BoardElementSet:
    """Elements of one accepted still, in that still's pixel frame."""

    elements: list[BoardElement]
    width: int
    height: int
    scale: float = 1.0

    def counts(self) -> dict[str, int]:
        found = {kind: 0 for kind in ELEMENT_KINDS}
        for element in self.elements:
            found[element.kind] = found.get(element.kind, 0) + 1
        found["elements"] = len(self.elements)
        return found

    def of_kind(self, kind: str) -> list[BoardElement]:
        return [element for element in self.elements if element.kind == kind]

    def to_dict(self) -> dict:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "scale": round(float(self.scale), 5),
            "counts": self.counts(),
            "elements": [element.to_dict() for element in self.elements],
            "crops": "in-memory; regenerated from bbox on the tile image for recognition",
        }


@dataclass
class AlignmentResult:
    """Similarity mapping reference pixels into the query frame."""

    status: str
    matrix: np.ndarray
    residual: float = 99.0
    primary_hits: int = 0
    secondary_hits: int = 0
    hits_by_kind: dict = field(default_factory=dict)
    snapped: list = field(default_factory=list)
    conflict_offsets: dict = field(default_factory=dict)
    shift_offsets: dict = field(default_factory=dict)
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class PatchClassifier:
    """Hook for a later CPU patch recognizer.

    ``label`` sees one crop, the classical kind, and the pose-invariant
    descriptor. The default keeps the classical kind. Do not load a weight
    file here; a subclass can, but this slice ships none.
    """

    def label(self, crop: np.ndarray, kind: str, descriptor: np.ndarray) -> str:
        return kind


def extract_board_elements(
    image: np.ndarray,
    *,
    max_long_edge: int = 640,
    classifier: PatchClassifier | None = None,
) -> BoardElementSet:
    """Separate one still into typed primitives and a crop per primitive."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        return BoardElementSet([], 0, 0, 1.0)
    color = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    full_h, full_w = color.shape[:2]
    scale = _working_scale(full_w, full_h, max_long_edge)
    view = color
    if scale < 0.999:
        view = cv2.resize(
            color,
            (max(1, int(round(full_w * scale))), max(1, int(round(full_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) < 2.0:
        return BoardElementSet([], full_w, full_h, scale)
    masks = _feature_masks(gray)
    raw: list[dict] = []
    raw.extend(_components_from_mask(masks["dark"], "dark"))
    raw.extend(_components_from_mask(masks["bright"], "bright"))
    # A smaller kernel keeps letter strokes and thin wires from filling in.
    raw.extend(_components_from_mask(masks["dark_fine"], "dark_fine"))
    raw.extend(_components_from_mask(masks["bright_fine"], "bright_fine"))
    raw.extend(_color_blocks(view))
    raw = _resolve_conflicts(raw)
    raw = _assign_pins(raw)
    raw = _cap_elements(raw)
    elements = [_materialize(item, color, scale) for item in raw]
    elements = [element for element in elements if element is not None]
    result = BoardElementSet(elements, full_w, full_h, scale)
    if classifier is not None:
        recognize_element_crops(result, classifier)
    else:
        _assign_ids(result)
    return result


def recognize_element_crops(
    elements: BoardElementSet,
    classifier: PatchClassifier | None = None,
) -> BoardElementSet:
    """Relabel stored crops. With no classifier the classical kinds stay."""
    if classifier is None:
        _assign_ids(elements)
        return elements
    for element in elements.elements:
        crop = element.crop if element.crop is not None else np.zeros((1, 1, 3), dtype=np.uint8)
        predicted = classifier.label(crop, element.kind, element.descriptor)
        if isinstance(predicted, str) and predicted in ELEMENT_KINDS:
            element.kind = predicted
    _assign_ids(elements)
    return elements


def align_red_region(
    reference: BoardElementSet,
    query: BoardElementSet,
    *,
    initial_pose: np.ndarray | None = None,
    accept_residual_px: float = ACCEPT_RESIDUAL_PX,
    conflict_residual_px: float = CONFLICT_RESIDUAL_PX,
) -> AlignmentResult:
    """Align the query to the reference with a similarity, then check elements.

    ``initial_pose`` maps reference pixels into the query. When it is given
    (the live geometric lock), only that pose and a local snap are scored.
    A different via period is not searched, and a disagreeing seam is not
    forced. Without it, typed holes and pads vote for a similarity and text,
    outlines, and traces break ties.
    """
    if reference is None or query is None or not reference.elements or not query.elements:
        return AlignmentResult("lost", _identity(), message="no elements")
    if initial_pose is not None:
        return _align_from_pose(
            reference,
            query,
            np.asarray(initial_pose, dtype=np.float64),
            accept_residual_px,
            conflict_residual_px,
        )
    return _align_global(reference, query, accept_residual_px, conflict_residual_px)


def detect_fiducials(image: np.ndarray) -> list[dict]:
    """Copper-ring corner marks with a drilled hole. Ordinary pads and caps are ignored."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        return []
    color = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = color.shape[:2]
    min_radius = max(22.0, 0.05 * float(min(height, width)))
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    gold = cv2.inRange(hsv, (8, 25, 80), (40, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    gold = cv2.morphologyEx(gold, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(gold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or not len(contours):
        return []
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    hierarchy = hierarchy[0]
    found: list[dict] = []
    for index, contour in enumerate(contours):
        area = float(cv2.contourArea(contour))
        if area < math.pi * (min_radius * 0.5) ** 2:
            continue
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        if radius < min_radius or radius > 0.5 * min(height, width):
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 0.0 if perimeter < 1.0 else float(4.0 * math.pi * area / (perimeter * perimeter))
        fill = float(area / max(1.0, math.pi * radius * radius))
        has_hole = int(hierarchy[index][2]) != -1
        yy, xx = np.ogrid[:height, :width]
        dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        inner = dist <= radius * 0.40
        ring = (dist <= radius * 1.05) & (dist >= radius * 0.55)
        if int(inner.sum()) < 25 or int(ring.sum()) < 50:
            continue
        inner_v = float(gray[inner].mean())
        ring_v = float(gray[ring].mean())
        ring_gold = float(gold[ring].mean())
        inner_gold = float(gold[inner].mean())
        if ring_gold < 40.0:
            continue
        if circularity < 0.65 or not has_hole:
            continue
        contrast = ring_v - inner_v
        gold_contrast = ring_gold - inner_gold
        if contrast < 8.0 and gold_contrast < 30.0:
            continue
        score = float(contrast + 0.3 * gold_contrast + 40.0 * min(circularity, 1.0) + 15.0)
        if score < 50.0:
            continue
        found.append(
            {
                "xy": (float(center_x), float(center_y)),
                "radius": float(radius),
                "score": score,
                "circularity": float(circularity),
            }
        )
    found.sort(key=lambda item: -item["score"])
    kept: list[dict] = []
    for item in found:
        if any(
            math.hypot(item["xy"][0] - other["xy"][0], item["xy"][1] - other["xy"][1])
            < 0.6 * (item["radius"] + other["radius"])
            for other in kept
        ):
            continue
        kept.append(item)
    return kept[:4]


def estimate_typed_pose(reference: BoardElementSet, query: BoardElementSet, *, reference_fiducials=None, query_fiducials=None) -> AlignmentResult:
    """Similarity from typed elements and corner fiducials. Does not snap partial outlines.

    A via/pad lattice alone is not unique; callers should treat ``lattice_only``
    as uncertain. Disagreement is not rewritten into an ok pose.
    """
    if reference is None or query is None or not reference.elements or not query.elements:
        return AlignmentResult("lost", _identity(), message="no elements")
    candidates = _vote_similarities(reference, query)
    for matrix in _fiducial_votes(reference_fiducials, query_fiducials):
        if all(not _pose_close(matrix, other, translation_px=8.0, angle_deg=3.0) for other in candidates):
            candidates.append(matrix)
    if not candidates:
        return AlignmentResult("lost", _identity(), message="no typed candidate")
    scored: list[AlignmentResult] = []
    for matrix in candidates:
        refined = _refine_similarity_from_matches(reference, query, matrix, ACCEPT_RESIDUAL_PX)
        chosen = refined if refined is not None else matrix
        scored.append(_evaluate(reference, query, chosen, ACCEPT_RESIDUAL_PX, CONFLICT_RESIDUAL_PX, snapped=[]))
    ok = [item for item in scored if item.status == "ok"]
    if ok:
        ok.sort(key=lambda item: (item.secondary_hits, item.primary_hits, -item.residual), reverse=True)
        return ok[0]
    lattices = [item for item in scored if item.status == "lattice_only"]
    if lattices:
        lattices.sort(key=lambda item: (item.primary_hits, -item.residual), reverse=True)
        return lattices[0]
    disagreements = [item for item in scored if item.status == "disagree"]
    if disagreements:
        disagreements.sort(key=lambda item: (-item.residual, item.primary_hits), reverse=True)
        best = disagreements[0]
        # Keep a geometrically tight disagreeing pose; a random far vote is lost.
        if best.residual <= CONFLICT_RESIDUAL_PX and best.primary_hits >= MIN_PRIMARY_HITS:
            return best
        return AlignmentResult("lost", best.matrix, residual=best.residual, message=best.message)
    return AlignmentResult("lost", scored[0].matrix, message="weak element support")


def typed_correspondences(reference: BoardElementSet, query: BoardElementSet, matrix: np.ndarray, *, accept_px: float = ACCEPT_RESIDUAL_PX) -> tuple[np.ndarray, np.ndarray]:
    """Same-kind pairs that sit on this pose, for a later pose-graph constraint."""
    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    try:
        inverse = _invert(matrix)
    except np.linalg.LinAlgError:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    for element in reference.elements:
        if element.partial or element.kind not in PRIMARY_KINDS + SECONDARY_KINDS:
            continue
        if not _inside(matrix, element.centroid, query.width, query.height):
            continue
        pool = [
            other
            for other in query.elements
            if other.kind == element.kind and _inside(inverse, other.centroid, reference.width, reference.height)
        ]
        if element.kind in SECONDARY_KINDS:
            pool = _secondary_partners(element, pool, matrix)
        else:
            pool = [other for other in pool if _descriptor_compatible(element, other)]
        if not pool:
            continue
        nearest = min(pool, key=lambda other: _centroid_distance(element, other, matrix))
        distance = _centroid_distance(element, nearest, matrix)
        if distance > max(6.0, accept_px + 2.0):
            continue
        src.append(np.asarray(element.centroid, dtype=np.float64))
        dst.append(np.asarray(nearest.centroid, dtype=np.float64))
    if not src:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    return np.vstack(src), np.vstack(dst)


def unique_landmark_support(reference: BoardElementSet, query: BoardElementSet, *, reference_fiducials=None, query_fiducials=None) -> int:
    """How many distinctive marks could possibly be the same object in both views."""
    score = 0
    if reference_fiducials and query_fiducials:
        score += 5 * len(_fiducial_votes(reference_fiducials, query_fiducials))
    if reference is None or query is None:
        return score
    kinds = (KIND_TEXT, KIND_OUTLINE, KIND_PIN, KIND_COLOR)
    weights = {KIND_TEXT: 2, KIND_OUTLINE: 2, KIND_PIN: 2, KIND_COLOR: 1}
    for kind in kinds:
        refs = [element for element in reference.of_kind(kind) if not element.partial][:24]
        queries = [element for element in query.of_kind(kind) if not element.partial][:24]
        hits = 0
        for element in refs:
            if any(_descriptor_compatible(element, other) for other in queries):
                hits += 1
        score += hits * weights[kind]
    return int(score)


def fiducial_correspondences(reference_fiducials, query_fiducials, matrix: np.ndarray, *, accept_px: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Same-mark pairs that sit on this pose, for the board-frame pose graph."""
    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    matrix = np.asarray(matrix, dtype=np.float64)
    for left in list(reference_fiducials or []):
        partners = [right for right in list(query_fiducials or []) if _fiducial_compatible(left, right)]
        if not partners:
            continue
        mapped = _apply(matrix, np.asarray(left.get("xy"), dtype=np.float64).reshape(1, 2))[0]
        nearest = min(
            partners,
            key=lambda right: math.hypot(
                mapped[0] - float(np.asarray(right.get("xy"), dtype=np.float64).reshape(2)[0]),
                mapped[1] - float(np.asarray(right.get("xy"), dtype=np.float64).reshape(2)[1]),
            ),
        )
        point = np.asarray(nearest.get("xy"), dtype=np.float64).reshape(2)
        if math.hypot(mapped[0] - point[0], mapped[1] - point[1]) > accept_px:
            continue
        src.append(np.asarray(left.get("xy"), dtype=np.float64).reshape(2))
        dst.append(point)
    if not src:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    return np.vstack(src), np.vstack(dst)


def _fiducial_compatible(left, right) -> bool:
    radius_l = float(left.get("radius", 0.0))
    radius_r = float(right.get("radius", 0.0))
    if radius_l < 1.0 or radius_r < 1.0:
        return False
    return abs(radius_l - radius_r) / max(radius_l, radius_r, 1.0) <= 0.30


def _similarity_from_two_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    src = np.asarray(src, dtype=np.float64).reshape(2, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(2, 2)
    first = src[1] - src[0]
    second = dst[1] - dst[0]
    length_s = float(np.hypot(first[0], first[1]))
    length_d = float(np.hypot(second[0], second[1]))
    if length_s < 8.0 or length_d < 8.0:
        return None
    scale = length_d / length_s
    if not 0.85 <= scale <= 1.15:
        return None
    angle = math.degrees(math.atan2(second[1], second[0]) - math.atan2(first[1], first[0]))
    rotated = _matrix(angle, scale, 0.0, 0.0)
    mapped = _apply(rotated, src[0].reshape(1, 2))[0]
    matrix = _matrix(angle, scale, float(dst[0][0] - mapped[0]), float(dst[0][1] - mapped[1]))
    if not _similarity_ok(matrix):
        return None
    return matrix


def _fiducial_votes(reference_fiducials, query_fiducials) -> list[np.ndarray]:
    refs = list(reference_fiducials or [])
    queries = list(query_fiducials or [])
    if not refs or not queries:
        return []
    found: list[np.ndarray] = []

    def _add(matrix: np.ndarray | None) -> None:
        if matrix is None:
            return
        if any(_pose_close(matrix, other, translation_px=6.0, angle_deg=3.0) for other in found):
            return
        found.append(matrix)

    if len(refs) >= 2 and len(queries) >= 2:
        for left_i, first in enumerate(refs):
            for left_j, second in enumerate(refs):
                if left_j <= left_i:
                    continue
                for right_i, third in enumerate(queries):
                    if not _fiducial_compatible(first, third):
                        continue
                    for right_j, fourth in enumerate(queries):
                        if right_j == right_i or not _fiducial_compatible(second, fourth):
                            continue
                        _add(
                            _similarity_from_two_points(
                                (first.get("xy"), second.get("xy")),
                                (third.get("xy"), fourth.get("xy")),
                            )
                        )
    for left in refs:
        for right in queries:
            if not _fiducial_compatible(left, right):
                continue
            src = np.asarray(left.get("xy"), dtype=np.float64).reshape(2)
            dst = np.asarray(right.get("xy"), dtype=np.float64).reshape(2)
            _add(_matrix(0.0, 1.0, float(dst[0] - src[0]), float(dst[1] - src[1])))
    return found


def is_periodic_lattice(points: np.ndarray) -> bool:

    """True for a regular 2D via/pad grid. A single column or a jittered field is not."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 8:
        return False
    distances = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    pitch = float(np.median(nearest))
    if pitch < 4.0 or not math.isfinite(pitch):
        return False
    if float(np.std(nearest) / pitch) > 0.08:
        return False
    span = pts.max(axis=0) - pts.min(axis=0)
    # Three rows (span = 2 * pitch) of a regular grid are still a lattice.
    if float(span.min()) < 1.95 * pitch:
        return False
    neighbors = np.sum(distances <= pitch * 1.25, axis=1)
    return float(np.median(neighbors)) >= 2.0


def _align_from_pose(reference, query, pose, accept_px, conflict_px) -> AlignmentResult:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (2, 3) or not _similarity_ok(pose):
        return AlignmentResult("lost", _identity(), message="pose is not a similarity")
    best_matrix = pose
    best = _evaluate(reference, query, pose, accept_px, conflict_px, snapped=[])
    refined = _refine_similarity_from_matches(reference, query, pose, accept_px)
    if refined is not None and _pose_close(refined, pose, translation_px=8.0, angle_deg=5.0):
        refined_score = _evaluate(reference, query, refined, accept_px, conflict_px, snapped=[])
        if _prefer(refined_score, best):
            best = refined_score
            best_matrix = refined
    snapped_matrix, snapped_ids = _snap_partials(reference, query, best_matrix, accept_px)
    if snapped_ids and _pose_close(snapped_matrix, best_matrix, translation_px=14.0, angle_deg=1.0):
        snapped_score = _evaluate(reference, query, snapped_matrix, accept_px, conflict_px, snapped=snapped_ids)
        if snapped_score.status in {"ok", "lattice_only"} or _prefer(snapped_score, best):
            best = snapped_score
    return best


def _align_global(reference, query, accept_px, conflict_px) -> AlignmentResult:
    candidates = _vote_similarities(reference, query)
    scored: list[AlignmentResult] = []
    for matrix in candidates:
        refined = _refine_similarity_from_matches(reference, query, matrix, accept_px)
        chosen = refined if refined is not None else matrix
        result = _evaluate(reference, query, chosen, accept_px, conflict_px, snapped=[])
        snapped_matrix, snapped_ids = _snap_partials(reference, query, result.matrix, accept_px)
        if snapped_ids:
            result = _evaluate(reference, query, snapped_matrix, accept_px, conflict_px, snapped=snapped_ids)
        scored.append(result)
    if not scored:
        return AlignmentResult("lost", _identity(), message="no typed candidate")
    ok = [item for item in scored if item.status == "ok"]
    if ok:
        ok.sort(key=lambda item: (item.secondary_hits, item.primary_hits, -item.residual), reverse=True)
        return ok[0]
    lattices = [item for item in scored if item.status == "lattice_only"]
    if lattices:
        lattices.sort(key=lambda item: (item.primary_hits, -item.residual), reverse=True)
        return lattices[0]
    disagreements = [item for item in scored if item.status == "disagree"]
    if disagreements:
        disagreements.sort(key=lambda item: item.primary_hits, reverse=True)
        return disagreements[0]
    return AlignmentResult("lost", scored[0].matrix, message="weak element support")


def _evaluate(reference, query, matrix, accept_px, conflict_px, snapped) -> AlignmentResult:
    matrix = np.asarray(matrix, dtype=np.float64)
    if not _similarity_ok(matrix):
        return AlignmentResult("lost", matrix, message="not a similarity")
    try:
        inverse = _invert(matrix)
    except np.linalg.LinAlgError:
        return AlignmentResult("lost", matrix, message="singular pose")
    ref_red = [
        element
        for element in reference.elements
        if _inside(matrix, element.centroid, query.width, query.height)
    ]
    query_red = [
        element
        for element in query.elements
        if _inside(inverse, element.centroid, reference.width, reference.height)
    ]
    distances: dict[str, list[float]] = {kind: [] for kind in ELEMENT_KINDS}
    deltas: dict[str, list] = {kind: [] for kind in ELEMENT_KINDS}
    desc_dists: dict[str, list] = {kind: [] for kind in ELEMENT_KINDS}
    matched_points: list[np.ndarray] = []
    hits_by_kind = {kind: 0 for kind in ELEMENT_KINDS}
    for element in ref_red:
        if element.partial and element.kind in PRIMARY_KINDS:
            continue
        pool = [other for other in query_red if other.kind == element.kind]
        if element.kind in SECONDARY_KINDS:
            partners = _secondary_partners(element, pool, matrix)
        else:
            partners = [other for other in pool if _descriptor_compatible(element, other)]
        # Only score pairs that look like the same primitive. Unmatched noisy
        # blobs on a real photo are not a seam disagreement.
        distance = _nearest_distance(element, partners, matrix)
        if distance is None:
            continue
        distances[element.kind].append(distance)
        if element.kind in SECONDARY_KINDS and partners:
            nearest = min(partners, key=lambda other: _centroid_distance(element, other, matrix))
            deltas[element.kind].append(_pair_delta(element, nearest, matrix))
            desc_dists[element.kind].append(_descriptor_distance(element, nearest))
        if distance <= accept_px:
            hits_by_kind[element.kind] = hits_by_kind.get(element.kind, 0) + 1
            if element.kind in PRIMARY_KINDS:
                matched_points.append(np.asarray(element.centroid, dtype=np.float64))
    primary = [value for kind in PRIMARY_KINDS for value in distances[kind]]
    primary_hits = sum(hits_by_kind[kind] for kind in PRIMARY_KINDS)
    residual = float(np.median(primary)) if primary else 99.0
    finite = [value for value in primary if value < 40.0]
    if finite:
        residual = float(np.median(finite))
    if len(primary) < MIN_PRIMARY_HITS:
        primary_state = "weak"
    elif primary_hits >= MIN_PRIMARY_HITS and residual <= accept_px:
        primary_state = "agree"
    elif residual >= conflict_px and primary_hits < max(MIN_PRIMARY_HITS, int(0.5 * len(primary))):
        # Compatible same-kind pairs exist, but they do not sit on this pose.
        primary_state = "conflict"
    elif primary_hits < MIN_PRIMARY_HITS:
        # A few close pairs and some leftovers is weak evidence, not a veto.
        primary_state = "inconclusive"
    else:
        primary_state = "inconclusive"
    family_state = {
        kind: _family_state_secondary(
            distances[kind],
            deltas.get(kind, []),
            accept_px,
            conflict_px,
            kind=kind,
            desc_dists=desc_dists.get(kind, []),
        )
        for kind in SECONDARY_KINDS
    }
    secondary_hits = sum(hits_by_kind[kind] for kind in SECONDARY_KINDS if family_state[kind] == "agree")
    lattice = primary_state == "agree" and is_periodic_lattice(np.asarray(matched_points, dtype=np.float64).reshape(-1, 2) if matched_points else np.zeros((0, 2)))
    agreed = [kind for kind, state in family_state.items() if state == "agree"]
    conflicts = [kind for kind, state in family_state.items() if state == "conflict"]
    conflict_offsets = {
        kind: [float(value) for value in distances[kind] if value >= conflict_px]
        for kind in conflicts
    }
    # Looser text pairing (shape-compatible, no tight descriptor cut) so a
    # shared silkscreen shift is visible even when one glyph's ink changed.
    text_shift = []
    for element in ref_red:
        if element.kind != KIND_TEXT or element.partial:
            continue
        pool = [
            other
            for other in query_red
            if other.kind == KIND_TEXT and not other.partial and _descriptor_compatible(element, other)
        ]
        distance = _nearest_distance(element, pool, matrix)
        if distance is not None and 6.0 <= distance <= _SECONDARY_PAIR_PX:
            text_shift.append(float(distance))
    shift_offsets = {"text": text_shift}
    status = "lost"
    message = "weak element support"
    if conflicts:
        status = "disagree"
        message = "secondary seam disagrees: " + ",".join(conflicts)
    elif primary_state == "conflict" and len(primary) >= MIN_PRIMARY_HITS:
        status = "disagree"
        message = "primary seam disagrees"
    elif primary_state == "agree" and lattice and not agreed:
        status = "lattice_only"
        message = "via/pad lattice alone"
    elif primary_state == "agree":
        status = "ok"
        message = "elements agree"
        if lattice and agreed:
            message = "lattice plus " + ",".join(agreed)
    snapped_ids = list(snapped) if status == "ok" else []
    if status == "ok":
        snapped_ids = _snapped_ids(ref_red, query_red, matrix, accept_px) or list(snapped)
    return AlignmentResult(
        status,
        matrix,
        residual=residual,
        primary_hits=primary_hits,
        secondary_hits=secondary_hits,
        hits_by_kind=hits_by_kind,
        snapped=snapped_ids,
        conflict_offsets=conflict_offsets,
        shift_offsets=shift_offsets,
        message=message,
    )


def _pair_delta(element: BoardElement, other: BoardElement, matrix: np.ndarray) -> np.ndarray:
    projected = _apply(matrix, np.asarray([element.centroid], dtype=np.float64))[0]
    return projected - np.asarray(other.centroid, dtype=np.float64)


def _offset_clustered(deltas: list, minimum: int, conflict_px: float, cluster_px: float = 14.0) -> bool:
    """True when several pairs share one large, consistent shift."""
    vectors = [np.asarray(item, dtype=np.float64).reshape(2) for item in deltas]
    big = [item for item in vectors if float(np.linalg.norm(item)) >= conflict_px]
    if len(big) < minimum:
        return False
    used = [False] * len(big)
    for index, anchor in enumerate(big):
        if used[index]:
            continue
        size = 1
        used[index] = True
        for other_index, other in enumerate(big):
            if used[other_index]:
                continue
            if float(np.linalg.norm(anchor - other)) <= cluster_px:
                used[other_index] = True
                size += 1
        if size >= minimum:
            return True
    return False


def _family_state_secondary(
    distances: list[float],
    deltas: list,
    accept_px: float,
    conflict_px: float,
    *,
    kind: str = "",
    desc_dists: list[float] | None = None,
) -> str:
    """Silkscreen disagreement is a shared shift of distinctive marks.

    One leftover blob at a random offset is not enough to reject a still
    whose pads already lock. A single far, tight letter match with no near
    mate still counts, because a test plate may only show two glyphs and
    extract one cleanly.
    """
    if len(distances) < 1:
        return "absent"
    agree = sum(value <= accept_px for value in distances)
    if _offset_clustered(deltas, 2, conflict_px):
        return "conflict"
    desc_dists = list(desc_dists or [99.0] * len(distances))
    if kind == KIND_TEXT and agree == 0 and not any(value <= 16.0 for value in distances):
        far_tight = [
            distance
            for distance, desc in zip(distances, desc_dists)
            if distance >= max(conflict_px, 24.0) and desc <= 0.75
        ]
        if far_tight:
            return "conflict"
    if agree >= 1:
        return "agree"
    return "inconclusive"


def _family_state(distances: list[float], minimum: int, accept_px: float, conflict_px: float) -> str:

    if len(distances) < minimum:
        return "absent"
    agree = sum(value <= accept_px for value in distances)
    conflict = sum(value >= conflict_px for value in distances)
    if conflict >= minimum and agree == 0:
        return "conflict"
    if agree >= minimum or (agree >= 1 and conflict == 0):
        return "agree"
    if agree == 0 and conflict > 0:
        return "conflict"
    return "inconclusive"


def _snapped_ids(ref_red, query_red, matrix, accept_px) -> list[str]:
    found = []
    for element in ref_red:
        pool = [other for other in query_red if other.kind == element.kind]
        if not pool:
            continue
        partial = element.partial or any(other.partial for other in pool)
        if not partial:
            continue
        centroid = _nearest_centroid(element, pool, matrix)
        chamfer = _nearest_distance(element, pool, matrix)
        if chamfer is None or centroid is None:
            continue
        if chamfer <= accept_px and centroid > accept_px + 1.0:
            found.append(element.element_id)
    return found


def _nearest_distance(element: BoardElement, pool: list[BoardElement], matrix: np.ndarray) -> float | None:
    """Centroid for complete shapes. Chamfer only for a shape the frame actually clips."""
    if not pool:
        return None
    if element.partial:
        complete = [other for other in pool if not other.partial] or pool
        return float(min(_pair_chamfer(element, other, matrix) for other in complete))
    complete = [other for other in pool if not other.partial]
    if complete:
        return float(min(_centroid_distance(element, other, matrix) for other in complete))
    return float(min(_pair_chamfer(element, other, matrix) for other in pool))


def _nearest_centroid(element: BoardElement, pool: list[BoardElement], matrix: np.ndarray) -> float | None:
    if not pool:
        return None
    return float(min(_centroid_distance(element, other, matrix) for other in pool))


def _centroid_distance(element: BoardElement, other: BoardElement, matrix: np.ndarray) -> float:
    projected = _apply(matrix, np.asarray([element.centroid], dtype=np.float64))[0]
    return float(np.hypot(projected[0] - other.centroid[0], projected[1] - other.centroid[1]))


def _pair_chamfer(element: BoardElement, other: BoardElement, matrix: np.ndarray) -> float:
    """Distance from the partial visible contour onto the other contour."""
    if element.partial and not other.partial:
        points = _apply(matrix, _match_points(element))
        return _median_segment_distance(points, _match_points(other))
    if other.partial and not element.partial:
        points = _match_points(other)
        complete = _apply(matrix, _match_points(element))
        return _median_segment_distance(points, complete)
    if element.partial and other.partial:
        points = _apply(matrix, _match_points(element))
        return _median_segment_distance(points, _match_points(other))
    return _centroid_distance(element, other, matrix)


def _match_points(element: BoardElement) -> np.ndarray:
    points = element.match_points if len(element.match_points) else element.contour
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return np.asarray([element.centroid], dtype=np.float64)
    return points


def _median_segment_distance(points: np.ndarray, poly: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0 or len(poly) == 0:
        return 99.0
    if len(points) > 32:
        points = points[np.linspace(0, len(points) - 1, 32).astype(int)]
    if len(poly) == 1:
        return float(np.median(np.linalg.norm(points - poly[0], axis=1)))
    start = poly
    end = np.roll(poly, -1, axis=0)
    segment = end - start
    length2 = np.maximum(np.sum(segment * segment, axis=1), 1e-6)
    offset = points[:, None, :] - start[None, :, :]
    t = np.clip(np.sum(offset * segment[None, :, :], axis=2) / length2[None, :], 0.0, 1.0)
    projection = start[None, :, :] + t[:, :, None] * segment[None, :, :]
    distances = np.linalg.norm(points[:, None, :] - projection, axis=2).min(axis=1)
    return float(np.median(distances))


def _snap_partials(reference, query, matrix, accept_px) -> tuple[np.ndarray, list[str]]:
    """Nudge translation so a clipped contour sits on the complete one."""
    partials = [
        element
        for element in list(reference.elements) + list(query.elements)
        if element.partial and element.kind in SECONDARY_KINDS
    ]
    if not partials:
        return matrix, []
    best = np.asarray(matrix, dtype=np.float64).copy()
    best_chamfer = _partial_chamfer(reference, query, best)
    if best_chamfer <= accept_px:
        return best, _snapped_ids(reference.elements, query.elements, best, accept_px)
    best_ids: list[str] = []
    origin = np.asarray(matrix, dtype=np.float64)
    for dy in range(-12, 13, 2):
        for dx in range(-12, 13, 2):
            if dx == 0 and dy == 0:
                continue
            trial = origin.copy()
            trial[0, 2] += dx
            trial[1, 2] += dy
            if not _similarity_ok(trial):
                continue
            if _primary_median(reference, query, trial) > accept_px:
                continue
            chamfer = _partial_chamfer(reference, query, trial)
            if chamfer + 0.15 < best_chamfer:
                best = trial
                best_chamfer = chamfer
    if best_chamfer <= accept_px:
        best_ids = _snapped_ids(
            [element for element in reference.elements if _inside(best, element.centroid, query.width, query.height)],
            query.elements,
            best,
            accept_px,
        )
    return best, best_ids


def _partial_chamfer(reference, query, matrix) -> float:
    values = []
    query_by_kind: dict[str, list[BoardElement]] = {}
    for element in query.elements:
        query_by_kind.setdefault(element.kind, []).append(element)
    for element in reference.elements:
        pool = query_by_kind.get(element.kind, [])
        if not pool:
            continue
        if not element.partial and not any(other.partial for other in pool):
            continue
        values.append(_nearest_distance(element, pool, matrix) or 99.0)
    if not values:
        return 99.0
    return float(np.median(values))


def _primary_median(reference, query, matrix) -> float:
    try:
        inverse = _invert(matrix)
    except np.linalg.LinAlgError:
        return 99.0
    distances = []
    query_primary = [element for element in query.elements if element.kind in PRIMARY_KINDS]
    for element in reference.elements:
        if element.kind not in PRIMARY_KINDS or element.partial:
            continue
        if not _inside(matrix, element.centroid, query.width, query.height):
            continue
        pool = [
            other
            for other in query_primary
            if other.kind == element.kind and _inside(inverse, other.centroid, reference.width, reference.height)
        ]
        distance = _nearest_centroid(element, pool, matrix)
        if distance is not None:
            distances.append(distance)
    if len(distances) < MIN_PRIMARY_HITS:
        return 99.0
    return float(np.median(distances))


def _vote_similarities(reference: BoardElementSet, query: BoardElementSet) -> list[np.ndarray]:
    """Typed translation votes. Holes and pads lead; text and outlines confirm."""
    general: list[tuple[float, np.ndarray]] = []
    distinctive: list[tuple[float, np.ndarray]] = []
    angles = np.linspace(-16.0, 16.0, 9)
    for angle in angles:
        peaks = _translation_peaks(reference, query, float(angle))
        for item in peaks:
            tx, ty, weight = item[0], item[1], item[2]
            source = item[3] if len(item) > 3 else "all"
            matrix = _matrix(float(angle), 1.0, tx, ty)
            if not _similarity_ok(matrix):
                continue
            (distinctive if source == "distinctive" else general).append((weight, matrix, float(angle)))
    general.sort(key=lambda item: item[0], reverse=True)
    near_by: dict[float, list] = {}
    far: list[tuple[float, np.ndarray, float]] = []
    for item in distinctive:
        if abs(item[2]) <= 4.0:
            near_by.setdefault(item[2], []).append(item)
        else:
            far.append(item)
    near: list = []
    for group in near_by.values():
        group.sort(key=lambda item: item[0], reverse=True)
        near.extend(group[:8])
    far.sort(key=lambda item: item[0], reverse=True)
    matrices = []
    for _weight, matrix, *_rest in general[:8] + near + far[:4]:
        if any(_pose_close(matrix, other, translation_px=6.0, angle_deg=3.0) for other in matrices):
            continue
        matrices.append(matrix)
    return matrices


def _translation_peaks(reference, query, angle: float) -> list[tuple[float, float, float]]:
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    buckets: dict[tuple[int, int], float] = {}
    cell = 6.0
    weights = {KIND_HOLE: 3.0, KIND_PAD: 2.0, KIND_PIN: 2.0, KIND_TEXT: 3.0, KIND_OUTLINE: 3.0, KIND_TRACE: 1.5}
    for kind, weight in weights.items():
        refs = [element for element in reference.of_kind(kind) if not element.partial][:40]
        queries = [element for element in query.of_kind(kind) if not element.partial][:40]
        for left in refs:
            rotated = rotation @ np.asarray(left.centroid, dtype=np.float64)
            for right in queries:
                if not _descriptor_compatible(left, right):
                    continue
                tx = float(right.centroid[0] - rotated[0])
                ty = float(right.centroid[1] - rotated[1])
                key = (int(round(tx / cell)), int(round(ty / cell)))
                buckets[key] = buckets.get(key, 0.0) + weight
    if not buckets:
        return []
    ranked = sorted(buckets.items(), key=lambda item: item[1], reverse=True)[:5]
    peaks = []
    seen: set[tuple[int, int]] = set()
    for key, weight in ranked:
        if weight < 4.0:
            continue
        peaks.append((key[0] * cell, key[1] * cell, weight))
        seen.add(key)
    # Pad/via pairings in the non-overlap region can outvote a true modest
    # overlap. Holes, text, and outlines are less periodic, so keep their
    # own peaks too.
    distinctive = {}
    for kind in (KIND_HOLE, KIND_TEXT, KIND_OUTLINE):
        refs = [element for element in reference.of_kind(kind) if not element.partial][:40]
        queries = [element for element in query.of_kind(kind) if not element.partial][:40]
        for left in refs:
            rotated = rotation @ np.asarray(left.centroid, dtype=np.float64)
            for right in queries:
                if not _descriptor_compatible(left, right):
                    continue
                tx = float(right.centroid[0] - rotated[0])
                ty = float(right.centroid[1] - rotated[1])
                key = (int(round(tx / cell)), int(round(ty / cell)))
                distinctive[key] = distinctive.get(key, 0.0) + weights[kind]
    for key, weight in sorted(distinctive.items(), key=lambda item: item[1], reverse=True)[:8]:
        if weight < 9.0 or key in seen:
            continue
        peaks.append((key[0] * cell, key[1] * cell, weight, "distinctive"))
        seen.add(key)
    return peaks


def _refine_similarity_from_matches(reference, query, matrix, accept_px) -> np.ndarray | None:
    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    try:
        inverse = _invert(matrix)
    except np.linalg.LinAlgError:
        return None
    for element in reference.elements:
        if element.partial or element.kind not in PRIMARY_KINDS + SECONDARY_KINDS:
            continue
        if not _inside(matrix, element.centroid, query.width, query.height):
            continue
        pool = [
            other
            for other in query.elements
            if other.kind == element.kind and _inside(inverse, other.centroid, reference.width, reference.height)
        ]
        if not pool:
            continue
        best = min(pool, key=lambda other: _centroid_distance(element, other, matrix))
        distance = _centroid_distance(element, best, matrix)
        if distance > max(6.0, accept_px + 2.0) or not _descriptor_compatible(element, best):
            continue
        src.append(np.asarray(element.centroid, dtype=np.float64))
        dst.append(np.asarray(best.centroid, dtype=np.float64))
    if len(src) < 4:
        return None
    refined = _umeyama(np.vstack(src), np.vstack(dst))
    if refined is None or not _similarity_ok(refined):
        return None
    return refined


def _descriptor_distance(left: BoardElement, right: BoardElement) -> float:
    a = np.asarray(left.descriptor, dtype=np.float64).ravel()
    b = np.asarray(right.descriptor, dtype=np.float64).ravel()
    if a.size == 0 or a.shape != b.shape:
        return 99.0
    return float(np.linalg.norm(a - b))


def _secondary_partners(element: BoardElement, pool: list[BoardElement], matrix: np.ndarray) -> list[BoardElement]:
    """Keep distinctive same-object candidates inside a modest shift radius."""
    found = []
    for other in pool:
        if not _descriptor_compatible(element, other):
            continue
        # Two clipped fragments are not a reliable same-object pair.
        if element.partial and other.partial:
            continue
        if _descriptor_distance(element, other) > _SECONDARY_DESC_MAX:
            continue
        distance = _nearest_distance(element, [other], matrix)
        if distance is None or distance > _SECONDARY_PAIR_PX:
            continue
        found.append(other)
    return found


def _descriptor_compatible(left: BoardElement, right: BoardElement) -> bool:
    if left.area < 1.0 or right.area < 1.0:
        return False
    ratio = left.area / right.area
    if not 0.55 <= ratio <= 1.85:
        return False
    if abs(left.circularity - right.circularity) > 0.4:
        return False
    if abs(left.aspect - right.aspect) > 0.55:
        return False
    return True


def _prefer(candidate: AlignmentResult, incumbent: AlignmentResult) -> bool:
    rank = {"ok": 3, "lattice_only": 2, "disagree": 1, "lost": 0}
    if rank.get(candidate.status, 0) != rank.get(incumbent.status, 0):
        return rank.get(candidate.status, 0) > rank.get(incumbent.status, 0)
    if candidate.secondary_hits != incumbent.secondary_hits:
        return candidate.secondary_hits > incumbent.secondary_hits
    if candidate.primary_hits != incumbent.primary_hits:
        return candidate.primary_hits > incumbent.primary_hits
    return candidate.residual + 0.05 < incumbent.residual


def _feature_masks(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Blob scale fills pads. Fine scale keeps silkscreen strokes and thin traces."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    blob = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fine = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return {
        "dark": _response_mask(cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, blob)),
        "bright": _response_mask(cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, blob)),
        "dark_fine": _response_mask(cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, fine)),
        "bright_fine": _response_mask(cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, fine)),
    }


def _response_mask(response: np.ndarray) -> np.ndarray:
    if float(response.max()) < 8.0:
        return np.zeros(response.shape, dtype=np.uint8)
    blur = cv2.GaussianBlur(response, (3, 3), 0)
    _level, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(mask.mean()) / 255.0 > 0.35:
        cutoff = max(12.0, float(np.percentile(blur, 94)))
        mask = np.where(blur >= cutoff, 255, 0).astype(np.uint8)
    return mask


def _components_from_mask(mask: np.ndarray, source: str) -> list[dict]:
    if mask is None or not np.any(mask):
        return []
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = mask.shape[:2]
    found = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 12 or area > 0.35 * width * height:
            continue
        x, y, w, h = (int(stats[index, column]) for column in range(4))
        if w < 2 or h < 2:
            continue
        roi = np.where(labels[y : y + h, x : x + w] == index, 255, 0).astype(np.uint8)
        item = _describe_component(roi, x, y, w, h, area, centroids[index], width, height, source)
        if item is not None:
            found.append(item)
    return found


def _describe_component(roi, x, y, w, h, area, centroid, width, height, source) -> dict | None:
    contours, hierarchy = cv2.findContours(roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    outer_index = max(range(len(contours)), key=lambda item: cv2.contourArea(contours[item]))
    outer = contours[outer_index]
    filled = float(cv2.contourArea(outer))
    perimeter = float(cv2.arcLength(outer, True))
    circularity = 0.0 if perimeter < 1.0 else float(4.0 * math.pi * filled / (perimeter * perimeter))
    child_area = 0.0
    if hierarchy is not None:
        child = int(hierarchy[0][outer_index][2])
        while child != -1:
            child_area += float(cv2.contourArea(contours[child]))
            child = int(hierarchy[0][child][0])
    hollow = filled > 80.0 and child_area > 0.35 * max(filled, 1.0) and area < 0.65 * filled
    hull = cv2.convexHull(outer)
    hull_area = float(cv2.contourArea(hull))
    solidity = 1.0 if hull_area < 1.0 else float(min(1.0, area / hull_area))
    stroke = _stroke_width(roi)
    aspect = float(max(w, h) / max(1.0, min(w, h)))
    extent = float(area) / float(max(1, w * h))
    touches = x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1
    contour = outer.reshape(-1, 2).astype(np.float64)
    contour[:, 0] += x
    contour[:, 1] += y
    kind = _classify_shape(
        source=source,
        area=float(area),
        filled=filled,
        circularity=circularity,
        aspect=aspect,
        extent=extent,
        solidity=solidity,
        stroke=stroke,
        width_px=w,
        height_px=h,
        hollow=hollow,
        touches=touches,
    )
    if kind is None:
        return None
    return {
        "kind": kind,
        "contour": contour,
        "bbox": (x, y, w, h),
        "centroid": (float(centroid[0]), float(centroid[1])),
        "area": float(area),
        "circularity": circularity,
        "aspect": aspect,
        "solidity": solidity,
        "stroke": stroke,
        "partial": touches,
        "parent_id": None,
        "image_size": (width, height),
    }


def _classify_shape(
    *,
    source: str,
    area: float,
    filled: float,
    circularity: float,
    aspect: float,
    extent: float,
    solidity: float,
    stroke: float,
    width_px: int,
    height_px: int,
    hollow: bool,
    touches: bool = False,
) -> str | None:
    """One component, one kind. Text is decided before traces so a letter is not a wire."""
    long_side = max(width_px, height_px)
    short_side = min(width_px, height_px)
    fine = source.endswith("fine")
    if fine:
        if long_side >= 36 and aspect >= 4.0 and 0.6 <= stroke <= 8.0 and circularity < 0.45 and area >= 20.0:
            return KIND_TRACE
        if source.startswith("bright") and _looks_like_text(
            short_side, long_side, aspect, stroke, area, extent, solidity, circularity
        ):
            return KIND_TEXT
        return None
    roundish = circularity >= 0.66 and aspect < 1.45
    if source == "dark" and roundish and 8.0 <= area <= 480.0 and not hollow:
        return KIND_HOLE
    if source == "bright" and hollow and circularity >= 0.5 and filled < 1400.0 and area <= 900.0:
        return KIND_PAD
    if source == "bright" and roundish and 20.0 <= area <= 2200.0 and not hollow:
        return KIND_PAD
    if hollow and filled >= 350.0 and aspect < 6.0 and long_side >= 24 and circularity < 0.82:
        return KIND_OUTLINE
    if source == "bright" and _looks_like_text(short_side, long_side, aspect, stroke, area, extent, solidity, circularity):
        return KIND_TEXT
    if (
        long_side >= 36
        and short_side <= 8
        and aspect >= 4.0
        and 0.8 <= stroke <= 6.5
        and circularity < 0.4
        and area >= 24.0
        and extent < 0.85
    ):
        return KIND_TRACE
    # A package outline cut by the frame is an open stroke, not a closed hole.
    if touches and stroke <= 8.0 and area >= 120.0 and long_side >= 28 and extent < 0.4 and aspect < 6.0:
        return KIND_OUTLINE
    if 20.0 <= area <= 2800.0 and aspect < 5.5 and extent >= 0.4 and not hollow and solidity >= 0.45:
        return KIND_PAD
    if hollow and filled >= 280.0 and long_side >= 20:
        return KIND_OUTLINE
    return None


def _looks_like_text(short_side, long_side, aspect, stroke, area, extent, solidity, circularity) -> bool:
    """A silkscreen glyph: small, not a filled pad, not a long wire."""
    return (
        6 <= short_side <= 34
        and long_side <= 42
        and aspect < 3.8
        and 1.0 <= stroke <= 8.0
        and 18.0 <= area <= 450.0
        and extent < 0.82
        and solidity < 0.95
        and circularity < 0.62
    )


def _stroke_width(mask: np.ndarray) -> float:
    """Median stroke. The mask is padded so a component that fills its box still has a background."""
    if mask.size == 0 or not np.any(mask):
        return 0.0
    view = mask
    shrink = 1.0
    height, width = mask.shape[:2]
    if max(height, width) > 96:
        shrink = 96.0 / float(max(height, width))
        view = cv2.resize(
            mask,
            (max(1, int(round(width * shrink))), max(1, int(round(height * shrink)))),
            interpolation=cv2.INTER_NEAREST,
        )
    padded = cv2.copyMakeBorder((view > 0).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)
    values = distance[padded > 0].astype(np.float64)
    values = values[np.isfinite(values) & (values < 1e6)]
    if values.size == 0:
        return 0.0
    return float(np.median(values) * 2.0 / max(shrink, 1e-6))


def _color_blocks(image: np.ndarray) -> list[dict]:
    """Deterministic hue clustering. The dominant cluster is the board, not a block."""
    height, width = image.shape[:2]
    scale = min(1.0, 160.0 / max(height, width))
    small = image
    if scale < 0.999:
        small = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)
    saturated = pixels[:, 1] > 40
    if int(np.count_nonzero(saturated)) < 30:
        return []
    samples = pixels[saturated][:, :2]
    centers = _kmeans_centers(samples, k=min(4, max(1, int(samples.shape[0] // 80))))
    assigned = np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=2).argmin(axis=1)
    # A soldermask-colored board is itself saturated. That largest cluster is background.
    background = int(np.bincount(assigned, minlength=len(centers)).argmax()) if float(saturated.mean()) > 0.45 else -1
    labels_full = np.full(pixels.shape[0], -1, dtype=np.int32)
    labels_full[np.flatnonzero(saturated)] = assigned
    small_h, small_w = small.shape[:2]
    label_image = labels_full.reshape(small_h, small_w)
    found = []
    for cluster, center in enumerate(centers):
        if cluster == background:
            continue
        component = np.where(label_image == cluster, 255, 0).astype(np.uint8)
        if float(component.mean()) < 1.0:
            continue
        hue = float(center[0])
        saturation = float(center[1])
        if saturation < 50.0:
            continue
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(component, connectivity=8)
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < 40:
                continue
            x, y, w, h = (int(stats[index, column]) for column in range(4))
            if min(w, h) < 4 or max(w, h) / max(1.0, min(w, h)) > 6.0:
                continue
            if scale < 0.999:
                inv = 1.0 / scale
                x, y, w, h = int(x * inv), int(y * inv), max(1, int(w * inv)), max(1, int(h * inv))
                cx, cy = float(centroids[index][0] * inv), float(centroids[index][1] * inv)
                area = int(area * inv * inv)
            else:
                cx, cy = float(centroids[index][0]), float(centroids[index][1])
            contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)
            found.append(
                {
                    "kind": KIND_COLOR,
                    "contour": contour,
                    "bbox": (x, y, w, h),
                    "centroid": (cx, cy),
                    "area": float(area),
                    "circularity": 0.5,
                    "aspect": float(max(w, h) / max(1.0, min(w, h))),
                    "solidity": 0.8,
                    "stroke": float(min(w, h)),
                    "partial": x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1,
                    "parent_id": None,
                    "image_size": (width, height),
                    "hue": hue,
                }
            )
    return found


def _kmeans_centers(samples: np.ndarray, k: int, iterations: int = 8) -> np.ndarray:
    """Seeded k-means so color blocks do not flicker between runs."""
    rng = np.random.default_rng(0)
    count = len(samples)
    k = max(1, min(int(k), count))
    fit = samples
    if count > 4000:
        fit = samples[rng.choice(count, 4000, replace=False)]
    fit_count = len(fit)
    centers = np.empty((k, fit.shape[1]), dtype=np.float64)
    centers[0] = fit[int(rng.integers(0, fit_count))]
    for index in range(1, k):
        distance = np.min(np.linalg.norm(fit[:, None, :] - centers[None, :index, :], axis=2), axis=1)
        weights = distance * distance
        total = float(weights.sum())
        if total < 1e-6:
            centers[index] = fit[int(rng.integers(0, fit_count))]
            continue
        pick = int(rng.choice(fit_count, p=weights / total))
        centers[index] = fit[pick]
    for _ in range(iterations):
        distance = np.linalg.norm(fit[:, None, :] - centers[None, :, :], axis=2)
        labels = distance.argmin(axis=1)
        for index in range(k):
            chosen = fit[labels == index]
            if len(chosen):
                centers[index] = chosen.mean(axis=0)
    return centers


def _resolve_conflicts(items: list[dict]) -> list[dict]:
    """Hole-in-pad, letter counters, and color that is really a metal pad."""
    items = _dedupe(items)
    texts = [item for item in items if item["kind"] == KIND_TEXT]
    pads = [item for item in items if item["kind"] == KIND_PAD]
    kept = []
    for item in items:
        if item["kind"] == KIND_HOLE and any(_inside_bbox(item["centroid"], text["bbox"], pad=2) and item["area"] < text["area"] for text in texts):
            continue
        if item["kind"] == KIND_PAD and any(
            _inside_bbox(item["centroid"], text["bbox"], pad=2) and _bbox_iou(item["bbox"], text["bbox"]) > 0.3
            for text in texts
        ):
            continue
        if item["kind"] == KIND_COLOR and any(_bbox_iou(item["bbox"], pad["bbox"]) > 0.65 for pad in pads):
            continue
        if item["kind"] == KIND_COLOR and any(
            _inside_bbox(item["centroid"], outline["bbox"], pad=0) or _bbox_iou(item["bbox"], outline["bbox"]) > 0.2
            for outline in items
            if outline["kind"] == KIND_OUTLINE
        ):
            continue
        if item["kind"] == KIND_TRACE and any(_inside_bbox(item["centroid"], text["bbox"], pad=1) for text in texts):
            continue
        kept.append(item)
    for hole in kept:
        if hole["kind"] != KIND_HOLE:
            continue
        host = None
        for pad in kept:
            if pad["kind"] != KIND_PAD:
                continue
            if _inside_bbox(hole["centroid"], pad["bbox"], pad=1) and hole["area"] * 2.5 < pad["area"]:
                host = pad
                break
        if host is not None:
            hole["parent_hint"] = id(host)
            host["child_hint"] = hole
    return kept


def _assign_pins(items: list[dict]) -> list[dict]:
    outlines = [item for item in items if item["kind"] == KIND_OUTLINE]
    if not outlines:
        return items
    claimed: set[int] = set()
    for outline in outlines:
        pads = []
        for index, item in enumerate(items):
            if item["kind"] != KIND_PAD or item["area"] > 220.0:
                continue
            # Stroke crumbs sit on the outline. Pins stand just off it, in a row.
            distance = _distance_to_poly(item["centroid"], outline["contour"])
            if not 6.0 <= distance <= 22.0:
                continue
            pads.append((index, item))
        if len(pads) < 3:
            continue
        order = _dominant_axis([item["centroid"] for _index, item in pads])
        ordered = sorted(pads, key=lambda pair: pair[1]["centroid"][order])
        coords = [pair[1]["centroid"][order] for pair in ordered]
        pitches = np.diff(np.asarray(coords, dtype=np.float64))
        pitches = pitches[pitches > 3.0]
        if len(pitches) < 2:
            continue
        if float(np.std(pitches) / max(float(np.mean(pitches)), 1.0)) > 0.3:
            continue
        for index, item in ordered:
            item["kind"] = KIND_PIN
            item["parent_hint"] = id(outline)
            claimed.add(index)
    return items


def _dominant_axis(points: list[tuple[float, float]]) -> int:
    array = np.asarray(points, dtype=np.float64)
    span = array.max(axis=0) - array.min(axis=0)
    return int(np.argmax(span))


def _distance_to_poly(point: tuple[float, float], poly: np.ndarray) -> float:
    return _median_segment_distance(np.asarray([point], dtype=np.float64), poly)


def _dedupe(items: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for item in sorted(items, key=lambda entry: -entry["area"]):
        if any(
            entry["kind"] == item["kind"]
            and math.hypot(entry["centroid"][0] - item["centroid"][0], entry["centroid"][1] - item["centroid"][1]) < 4.0
            for entry in kept
        ):
            continue
        kept.append(item)
    return kept


def _cap_elements(items: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {kind: [] for kind in ELEMENT_KINDS}
    for item in items:
        grouped.setdefault(item["kind"], []).append(item)
    kept: list[dict] = []
    for kind, bucket in grouped.items():
        bucket.sort(key=lambda entry: -entry["area"])
        kept.extend(bucket[: _KIND_CAP.get(kind, 16)])
    if len(kept) > _MAX_ELEMENTS:
        kept.sort(key=lambda entry: -entry["area"])
        kept = kept[:_MAX_ELEMENTS]
    return kept


def _materialize(item: dict, full: np.ndarray, scale: float) -> BoardElement | None:
    inv = 1.0 / scale if scale > 1e-6 else 1.0
    contour = np.asarray(item["contour"], dtype=np.float64) * inv
    x, y, w, h = item["bbox"]
    bbox = (
        int(round(x * inv)),
        int(round(y * inv)),
        max(1, int(round(w * inv))),
        max(1, int(round(h * inv))),
    )
    centroid = (float(item["centroid"][0] * inv), float(item["centroid"][1] * inv))
    area = float(item["area"]) * inv * inv
    stroke = float(item["stroke"]) * inv
    match = _match_contour(contour, bbox, full.shape[1], full.shape[0], bool(item["partial"]))
    element = BoardElement(
        element_id="",
        kind=str(item["kind"]),
        contour=contour,
        bbox=bbox,
        centroid=centroid,
        descriptor=np.zeros(8, dtype=np.float64),
        crop=_crop(full, bbox),
        area=area,
        circularity=float(item["circularity"]),
        aspect=float(item["aspect"]),
        solidity=float(item["solidity"]),
        stroke_width=stroke,
        partial=bool(item["partial"]),
        parent_id=None,
        match_points=match,
    )
    element.descriptor = _descriptor(element)
    element._parent_hint = item.get("parent_hint")  # type: ignore[attr-defined]
    element._self_hint = id(item)  # type: ignore[attr-defined]
    return element


def _match_contour(contour: np.ndarray, bbox, width: int, height: int, partial: bool) -> np.ndarray:
    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return points
    approx = cv2.approxPolyDP(points.reshape(-1, 1, 2).astype(np.float32), 1.5, True).reshape(-1, 2)
    if len(approx) >= 3:
        points = approx.astype(np.float64)
    if not partial:
        return points
    x, y, w, h = bbox
    keep = []
    for point in points:
        on_frame = point[0] <= 2.0 or point[1] <= 2.0 or point[0] >= width - 3.0 or point[1] >= height - 3.0
        if not on_frame:
            keep.append(point)
    if len(keep) >= 3:
        return np.asarray(keep, dtype=np.float64)
    return points


def _descriptor(element: BoardElement) -> np.ndarray:
    moments = cv2.moments(np.asarray(element.contour, dtype=np.float32).reshape(-1, 1, 2))
    hu = cv2.HuMoments(moments).flatten()
    logged = []
    for value in hu[:3]:
        logged.append(float(-np.sign(value) * math.log10(abs(float(value)) + 1e-12)))
    return np.asarray(
        [
            math.log(element.area + 1.0),
            element.circularity,
            element.aspect,
            element.solidity,
            element.stroke_width / 10.0,
            *logged,
        ],
        dtype=np.float64,
    )


def _assign_ids(elements: BoardElementSet) -> None:
    hints = {getattr(element, "_self_hint", None): element for element in elements.elements}
    grouped: dict[str, list[BoardElement]] = {}
    for element in elements.elements:
        grouped.setdefault(element.kind, []).append(element)
    for kind, bucket in grouped.items():
        bucket.sort(key=lambda element: (element.centroid[1], element.centroid[0]))
        for index, element in enumerate(bucket, start=1):
            element.element_id = f"{kind}-{index:04d}"
    for element in elements.elements:
        parent = hints.get(getattr(element, "_parent_hint", None))
        element.parent_id = None if parent is None else parent.element_id


def _crop(image: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 3, max_side: int = 96) -> np.ndarray:
    height, width = image.shape[:2]
    x, y, w, h = bbox
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(width, x + w + pad)
    y1 = min(height, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        channels = 1 if image.ndim == 2 else image.shape[2]
        return np.zeros((1, 1) if channels == 1 else (1, 1, channels), dtype=np.uint8)
    crop = np.ascontiguousarray(image[y0:y1, x0:x1])
    long_side = max(crop.shape[:2])
    if long_side > max_side:
        ratio = max_side / float(long_side)
        crop = cv2.resize(
            crop,
            (max(1, int(round(crop.shape[1] * ratio))), max(1, int(round(crop.shape[0] * ratio)))),
            interpolation=cv2.INTER_AREA,
        )
    return crop


def _simplify_contour(contour: np.ndarray, limit: int) -> np.ndarray:
    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(points) <= limit:
        return points
    index = np.linspace(0, len(points) - 1, limit).astype(int)
    return points[index]


def _working_scale(width: int, height: int, max_long_edge: int) -> float:
    long_edge = max(width, height)
    if long_edge <= max_long_edge or max_long_edge < 32:
        return 1.0
    return float(max_long_edge) / float(long_edge)


def _inside_bbox(point: tuple[float, float], bbox: tuple[int, int, int, int], pad: int = 0) -> bool:
    x, y, w, h = bbox
    return x - pad <= point[0] <= x + w + pad and y - pad <= point[1] <= y + h + pad


def _bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = left
    bx, by, bw, bh = right
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = float((x1 - x0) * (y1 - y0))
    union = float(aw * ah + bw * bh) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _inside(matrix: np.ndarray, point: tuple[float, float], width: int, height: int, margin: float = 2.0) -> bool:
    projected = _apply(matrix, np.asarray([point], dtype=np.float64))[0]
    return -margin <= projected[0] < width + margin and -margin <= projected[1] < height + margin


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    linear = np.asarray(matrix, dtype=np.float64)
    return pts @ linear[:, :2].T + linear[:, 2]


def _invert(matrix: np.ndarray) -> np.ndarray:
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = np.asarray(matrix, dtype=np.float64)
    return np.linalg.inv(homography)[:2, :]


def _identity() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _matrix(angle_deg: float, scale: float, tx: float, ty: float) -> np.ndarray:
    radians = math.radians(angle_deg)
    cosine = math.cos(radians) * scale
    sine = math.sin(radians) * scale
    return np.array([[cosine, -sine, tx], [sine, cosine, ty]], dtype=np.float64)


def _similarity_ok(matrix: np.ndarray) -> bool:
    linear = np.asarray(matrix, dtype=np.float64)[:2, :2]
    if linear.shape != (2, 2) or not np.isfinite(linear).all():
        return False
    if float(np.linalg.det(linear)) <= 0.0:
        return False
    column0 = linear[:, 0]
    column1 = linear[:, 1]
    norm0 = float(np.linalg.norm(column0))
    norm1 = float(np.linalg.norm(column1))
    if norm0 < 1e-6:
        return False
    if abs(norm0 - norm1) > 0.05 * norm0:
        return False
    if abs(float(np.dot(column0, column1))) > 0.05 * norm0 * norm1:
        return False
    return 0.93 <= norm0 <= 1.07


def _pose_close(left: np.ndarray, right: np.ndarray, *, translation_px: float, angle_deg: float) -> bool:
    shift = math.hypot(float(left[0, 2] - right[0, 2]), float(left[1, 2] - right[1, 2]))
    if shift > translation_px:
        return False
    angle_left = math.degrees(math.atan2(float(left[1, 0]), float(left[0, 0])))
    angle_right = math.degrees(math.atan2(float(right[1, 0]), float(right[0, 0])))
    delta = (angle_left - angle_right + 180.0) % 360.0 - 180.0
    return abs(delta) <= angle_deg


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
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
    if not 0.93 <= scale <= 1.07:
        return None
    translation = mean_dst - scale * (rotation @ mean_src)
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = translation
    return matrix
