"""Board-frame geometric lock, C-region pose-graph loop closure, and non-blended overview.

Live preview pose is a cheap similarity into the first-tile frame. Accepted
stills are placed by typed C-region anchors, then jointly adjusted so a 田
grid cannot accumulate chain drift. Overlap pixels are never averaged: a high
residual keeps the sharper tile or leaves the overlap untouched.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from vision.topology import _refine_similarity

# Overlap is not blended when the geometric residual is above this.
SAFE_RESIDUAL_PX = 2.5
# A shot this far off is not written into the overview at all.
REJECT_RESIDUAL_PX = 3.2

_MIN_HITS = 5
_MIN_PRECISION = 0.70
_MAX_RESIDUAL = 3.2
_RIVAL_PRECISION = 0.72
_RIVAL_HIT_RATIO = 0.55
_SEARCH_THRESH = 4.5
_LANDMARK_WEIGHT = 4.0


class PoseEstimate:
    """Similarity mapping library pixels into the query frame."""

    def __init__(
        self,
        status: str,
        matrix: np.ndarray | None = None,
        *,
        hits: int = 0,
        precision: float = 0.0,
        residual: float = 99.0,
        src_xy: np.ndarray | None = None,
        dst_xy: np.ndarray | None = None,
    ):
        self.status = status
        self.matrix = np.eye(2, 3, dtype=np.float64) if matrix is None else np.asarray(matrix, dtype=np.float64)
        self.hits = int(hits)
        self.precision = float(precision)
        self.residual = float(residual)
        self.src_xy = np.zeros((0, 2), dtype=np.float64) if src_xy is None else np.asarray(src_xy, dtype=np.float64)
        self.dst_xy = np.zeros((0, 2), dtype=np.float64) if dst_xy is None else np.asarray(dst_xy, dtype=np.float64)
        self.ncc = 0.0
        self.area = 0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def landmarks_from_snapshot(snapshot) -> np.ndarray:
    """Rare shapes only. Ordinary pads stay in the topology constellation."""
    if snapshot is None:
        return np.zeros((0, 2), dtype=np.float64)
    found = []
    for primitive in getattr(snapshot, "primitives", []) or []:
        area = float(getattr(primitive, "area", 0.0))
        kind = str(getattr(primitive, "kind", ""))
        if (kind == "circle" and area >= 120.0) or area >= 500.0:
            center = getattr(primitive, "center", None)
            if center is not None:
                found.append((float(center[0]), float(center[1])))
    if not found:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(found, dtype=np.float64)


def estimate_similarity_pose(
    library_xy: np.ndarray,
    query_xy: np.ndarray,
    width: int,
    height: int,
    *,
    library_w: np.ndarray | None = None,
    query_w: np.ndarray | None = None,
    _disjoint_check: bool = True,
) -> PoseEstimate:
    """Lock a partial view with a similarity. Ambiguous lattices stay uncertain.

    Confidence is the fraction of library points that land inside this frame and
    hit a query point — not the fraction of the whole frame those hits cover.
    A repetitive pad field with two equally good shifts is ``uncertain``.
    """
    library_xy = np.asarray(library_xy, dtype=np.float64).reshape(-1, 2)
    query_xy = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    if len(library_xy) < 4 or len(query_xy) < 4 or width <= 1 or height <= 1:
        return PoseEstimate("lost")
    library_w = np.ones(len(library_xy), dtype=np.float64) if library_w is None else np.asarray(library_w, dtype=np.float64)
    query_w = np.ones(len(query_xy), dtype=np.float64) if query_w is None else np.asarray(query_w, dtype=np.float64)
    sample_lib, sample_lw = _spatial_sample(library_xy, library_w, 220)
    sample_qry, sample_qw = _spatial_sample(query_xy, query_w, 220)
    angles = np.linspace(-18.0, 18.0, 13)
    candidates: list[tuple[float, float, float, float]] = []
    for angle in angles:
        candidates.extend(_vote_angle(sample_lib, sample_lw, sample_qry, sample_qw, float(angle)))
        if abs(angle) <= 6.0:
            candidates.extend(_tight_vote(sample_lib, sample_qry, float(angle)))
    # Score the strongest bins first so a weak nearby angle cannot hide the true peak.
    candidates.sort(key=lambda item: item[0], reverse=True)
    scored: list[dict] = []
    seen: list[tuple[float, float, float]] = []
    examined = 0
    for votes, angle, tx, ty in candidates:
        if any(abs(angle - item[0]) <= 3.0 and abs(tx - item[1]) <= 8.0 and abs(ty - item[2]) <= 8.0 for item in seen):
            continue
        forced = abs(angle) <= 1.5 and votes >= 5.0
        if examined >= 180 and not forced:
            continue
        if not forced:
            examined += 1
        seen.append((angle, tx, ty))
        matrix = _matrix(angle, 1.0, tx, ty)
        hits, precision, residual, src_xy, dst_xy = _score_pose(library_xy, query_xy, matrix, width, height)
        if hits < 4:
            continue
        scored.append(
            {
                "votes": votes,
                "angle": angle,
                "hits": hits,
                "precision": precision,
                "residual": residual,
                "matrix": matrix,
                "src_xy": src_xy,
                "dst_xy": dst_xy,
            }
        )
    if not scored:
        return PoseEstimate("lost")
    clusters = _cluster_modes(scored)
    best = clusters[0]
    refined = _maybe_refine(best, library_xy, query_xy, width, height)
    if refined is not None:
        best = refined
        clusters[0] = best
    rival = _rival_mode(best, clusters[1:])
    acceptable = _acceptable(best)
    if acceptable and rival is None and _disjoint_check and _disjoint_constellation(
        best, library_xy, library_w, query_xy, query_w, width, height
    ):
        rival = best
    if acceptable and rival is None:
        return PoseEstimate(
            "ok",
            best["matrix"],
            hits=best["hits"],
            precision=best["precision"],
            residual=best["residual"],
            src_xy=best["src_xy"],
            dst_xy=best["dst_xy"],
        )
    # A real but non-unique geometric agreement is not "no red in view".
    if rival is not None or (best["hits"] >= _MIN_HITS and best["precision"] >= 0.5):
        return PoseEstimate(
            "uncertain",
            best["matrix"],
            hits=best["hits"],
            precision=best["precision"],
            residual=best["residual"],
            src_xy=best["src_xy"],
            dst_xy=best["dst_xy"],
        )
    return PoseEstimate(
        "lost",
        best["matrix"],
        hits=best["hits"],
        precision=best["precision"],
        residual=best["residual"],
        src_xy=best["src_xy"],
        dst_xy=best["dst_xy"],
    )


def refine_pose_graph(poses: list[np.ndarray], constraints: list[tuple], *, iterations: int = 4) -> list[np.ndarray]:
    """Jointly adjust similarity poses in the first-tile frame. Tile 0 stays fixed.

    Each constraint is ``(src_index, dst_index, src_xy, dst_xy)`` with both
    point sets in that tile's own pixels: ``pose_src(src) ≈ pose_dst(dst)``.
    """
    solved = [np.asarray(pose, dtype=np.float64).copy() for pose in poses]
    if len(solved) < 2 or not constraints:
        return solved
    for _ in range(iterations):
        for index in range(1, len(solved)):
            src_pts: list[np.ndarray] = []
            dst_pts: list[np.ndarray] = []
            for src_index, dst_index, src_xy, dst_xy in constraints:
                src_xy = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
                dst_xy = np.asarray(dst_xy, dtype=np.float64).reshape(-1, 2)
                if len(src_xy) < 2 or len(src_xy) != len(dst_xy):
                    continue
                if index == dst_index:
                    src_pts.append(dst_xy)
                    dst_pts.append(_apply(solved[src_index], src_xy))
                elif index == src_index:
                    src_pts.append(src_xy)
                    dst_pts.append(_apply(solved[dst_index], dst_xy))
            if not src_pts:
                continue
            src = np.vstack(src_pts)
            dst = np.vstack(dst_pts)
            if len(src) < 4:
                continue
            updated = _refine_similarity(src, dst)
            if updated is None:
                continue
            matrix = np.asarray(updated[:2, :], dtype=np.float64)
            if not _similarity_ok(matrix):
                continue
            new_residual = _pair_residual(matrix, src, dst)
            old_residual = _pair_residual(solved[index], src, dst)
            if new_residual > old_residual + 0.05:
                continue
            if _translation_jump(solved[index], matrix) > 40.0:
                continue
            if _angle_jump(solved[index], matrix) > (15.0 if old_residual > 4.0 else 8.0):
                continue
            solved[index] = matrix
    return solved


def joint_align_c_region(poses: list[np.ndarray], constraints: list[tuple], *, iterations: int = 12) -> list[np.ndarray]:
    """Jointly align tiles that share C-region green anchors.

    This is the 田 / loop-closure step: every overlapping pair contributes
    typed correspondences in the same first-tile frame, tile 0 stays fixed,
    and the solve stays a similarity (rotation, translation, slight scale).
    Chaining A→B→D→E alone is not enough; the cross constraints lock the grid.
    """
    return refine_pose_graph(poses, constraints, iterations=iterations)


def composite_registered_tile(
    canvas: np.ndarray,
    mask: np.ndarray,
    residual_map: np.ndarray,
    sharpness_map: np.ndarray,
    image: np.ndarray,
    transform: np.ndarray,
    *,
    residual_px: float,
    sharpness: float,
    safe_residual_px: float = SAFE_RESIDUAL_PX,
) -> None:
    """Copy one registered tile. Overlap is never an average of two exposures."""
    if canvas is None or image is None or image.size == 0:
        return
    height, width = canvas.shape[:2]
    warped = cv2.warpAffine(
        image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
        np.asarray(transform, dtype=np.float64),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    src_mask = cv2.warpAffine(
        np.full(image.shape[:2], 255, dtype=np.uint8),
        np.asarray(transform, dtype=np.float64),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if min(height, width) > 8:
        src_mask = cv2.erode(src_mask, np.ones((3, 3), np.uint8), iterations=1)
    incoming = src_mask > 0
    if not np.any(incoming):
        return
    # Misaligned tiles are not blended into the overlap and are not written as a
    # second, shifted copy of the same pads.
    if float(residual_px) > max(safe_residual_px, REJECT_RESIDUAL_PX):
        return
    occupied = mask > 0
    fresh = incoming & ~occupied
    overlap = incoming & occupied
    better = np.zeros(incoming.shape, dtype=bool)
    if np.any(overlap) and float(residual_px) <= safe_residual_px:
        sharper = sharpness > (sharpness_map + 1.0)
        closer = residual_px + 0.15 < residual_map
        similar = np.abs(residual_map - float(residual_px)) <= 0.4
        incumbent_bad = residual_map > safe_residual_px
        better = overlap & (closer | incumbent_bad | (similar & sharper))
    write = fresh | better
    if not np.any(write):
        return
    canvas[write] = warped[write]
    mask[write] = 255
    residual_map[write] = float(residual_px)
    sharpness_map[write] = float(sharpness)



# Photometric lock used on accepted stills when the pad/via constellation is
# too noisy (real photos) but the two views still share unique texture.
_FEATURE_MIN_INLIERS = 8
_FEATURE_MIN_NCC = 0.45
_FEATURE_MAX_ANGLE_DEG = 18.0


def masked_ncc(library: np.ndarray, query: np.ndarray, matrix: np.ndarray, *, min_area: int = 4000, erode: int = 5) -> tuple[float, int]:
    """Normalized correlation of ``library`` warped into ``query`` on the overlap."""
    if library is None or query is None or library.size == 0 or query.size == 0:
        return 0.0, 0
    height, width = query.shape[:2]
    matrix = np.asarray(matrix, dtype=np.float64)
    warped = cv2.warpAffine(
        library if library.ndim == 3 else cv2.cvtColor(library, cv2.COLOR_GRAY2BGR),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask = cv2.warpAffine(
        np.full(library.shape[:2], 255, dtype=np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if erode > 0 and min(height, width) > erode + 2:
        mask = cv2.erode(mask, np.ones((erode, erode), np.uint8), iterations=1)
    both = mask > 0
    area = int(np.count_nonzero(both))
    if area < int(min_area):
        return 0.0, area
    left = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    right = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY) if query.ndim == 3 else query
    a = left[both].astype(np.float32)
    b = right[both].astype(np.float32)
    a -= float(a.mean())
    b -= float(b.mean())
    denom = float(np.sqrt(float((a * a).sum()) * float((b * b).sum())))
    if denom < 1e-6:
        return 0.0, area
    return float((a * b).sum() / denom), area


def extract_feature_pack(image: np.ndarray):
    """CLAHE+ORB keypoints and descriptors for one still. Cached on accepted tiles."""
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    detector = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=11, patchSize=31)
    keypoints, desc = detector.detectAndCompute(equalized, None)
    if desc is None or len(keypoints) < 8:
        return None
    return (keypoints, desc)


def estimate_feature_pose(
    library_bgr: np.ndarray,
    query_bgr: np.ndarray,
    *,
    library_pack=None,
    query_pack=None,
) -> PoseEstimate:
    """Similarity from a still onto the next still using CLAHE+ORB, scored by NCC.

    Periodic via fields rarely give a unique, high-NCC lock. Real silkscreen,
    packages, and connectors do. This is only for stills, not the live preview.
    """
    if library_bgr is None or query_bgr is None or library_bgr.size == 0 or query_bgr.size == 0:
        return PoseEstimate("lost")
    if library_pack is None:
        library_pack = extract_feature_pack(library_bgr)
    if query_pack is None:
        query_pack = extract_feature_pack(query_bgr)
    if library_pack is None or query_pack is None:
        return PoseEstimate("lost")
    keypoints_a, desc_a = library_pack
    keypoints_b, desc_b = query_pack
    if desc_a is None or desc_b is None or len(keypoints_a) < 8 or len(keypoints_b) < 8:
        return PoseEstimate("lost")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = matcher.knnMatch(desc_a, desc_b, k=2)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.80 * second.distance:
            good.append(first)
    if len(good) < 6:
        return PoseEstimate("lost", hits=len(good))
    src = np.float32([keypoints_a[item.queryIdx].pt for item in good])
    dst = np.float32([keypoints_b[item.trainIdx].pt for item in good])
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.5,
        maxIters=4000,
        confidence=0.99,
    )
    if matrix is None or inlier_mask is None:
        return PoseEstimate("lost", hits=len(good))
    keep = inlier_mask.ravel().astype(bool)
    hits = int(keep.sum())
    matrix = np.asarray(matrix, dtype=np.float64)
    if hits < 4 or not _similarity_ok(matrix) or abs(_angle_of(matrix)) > _FEATURE_MAX_ANGLE_DEG:
        estimate = PoseEstimate("lost", matrix, hits=hits)
        return estimate
    src_xy = np.asarray(src[keep], dtype=np.float64)
    dst_xy = np.asarray(dst[keep], dtype=np.float64)
    residual = _pair_residual(matrix, src_xy, dst_xy)
    refined = _maybe_refine_feature(matrix, library_bgr, query_bgr)
    if refined is not None:
        matrix = refined
        residual = _pair_residual(matrix, src_xy, dst_xy)
        hits, _precision, residual, src_xy, dst_xy = _score_pose(
            src_xy, dst_xy, matrix, int(query_bgr.shape[1]), int(query_bgr.shape[0]), thresh=3.5
        ) if len(src_xy) else (hits, 0.0, residual, src_xy, dst_xy)
    ncc, area = masked_ncc(library_bgr, query_bgr, matrix)
    precision = float(hits) / float(max(len(good), 1))
    min_area = max(4000, int(0.06 * query_bgr.shape[0] * query_bgr.shape[1]))
    ok = (
        hits >= _FEATURE_MIN_INLIERS
        and ncc >= _FEATURE_MIN_NCC
        and area >= min_area
        and residual <= 3.5
        and _similarity_ok(matrix)
        and abs(_angle_of(matrix)) <= _FEATURE_MAX_ANGLE_DEG
    )
    estimate = PoseEstimate(
        "ok" if ok else "lost",
        matrix,
        hits=hits,
        precision=precision,
        residual=residual,
        src_xy=src_xy,
        dst_xy=dst_xy,
    )
    estimate.ncc = float(ncc)
    estimate.area = int(area)
    return estimate


def _maybe_refine_feature(matrix: np.ndarray, library: np.ndarray, query: np.ndarray) -> np.ndarray | None:
    """A short Euclidean ECC nudge. Failures keep the ORB pose."""
    left = cv2.cvtColor(library, cv2.COLOR_BGR2GRAY) if library.ndim == 3 else library
    right = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY) if query.ndim == 3 else query
    scale = min(1.0, 320.0 / max(left.shape[0], left.shape[1], right.shape[0], right.shape[1]))
    if scale < 0.999:
        left = cv2.resize(left, (max(1, int(round(left.shape[1] * scale))), max(1, int(round(left.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (max(1, int(round(right.shape[1] * scale))), max(1, int(round(right.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
        warp = np.array([[matrix[0, 0], matrix[0, 1], matrix[0, 2] * scale], [matrix[1, 0], matrix[1, 1], matrix[1, 2] * scale]], dtype=np.float32)
    else:
        warp = np.asarray(matrix, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)
    try:
        _score, warp = cv2.findTransformECC(
            left.astype(np.float32),
            right.astype(np.float32),
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            4,
        )
    except cv2.error:
        return None
    warp = np.asarray(warp, dtype=np.float64)
    if scale < 0.999:
        warp[0, 2] /= scale
        warp[1, 2] /= scale
    if not _similarity_ok(warp):
        return None
    if _translation_jump(matrix, warp) > 16.0 or _angle_jump(matrix, warp) > 2.5:
        return None
    return warp


def _spacing_compatible(library_xy: np.ndarray, query_xy: np.ndarray) -> np.ndarray:
    """True where a query point and a library point have a similar nearest neighbor."""
    library_nn = _nearest_spacing(library_xy)
    query_nn = _nearest_spacing(query_xy)
    ratio = query_nn[:, None] / np.maximum(library_nn[None, :], 1.0)
    return (ratio > 0.70) & (ratio < 1.45)


def _nearest_spacing(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.ones(len(points), dtype=np.float64)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def _tight_vote(library_xy: np.ndarray, query_xy: np.ndarray, angle: float) -> list[tuple[float, float, float, float]]:
    """Translations supported by similarly spaced neighbors.

    A thin overlap can lose the raw vote to accidental pad collisions. Pairs
    whose nearest-neighbor distance agrees still pile up on the true shift.
    """
    if len(library_xy) < 4 or len(query_xy) < 4:
        return []
    library_nn = _nearest_spacing(library_xy)
    query_nn = _nearest_spacing(query_xy)
    ratio = query_nn[:, None] / np.maximum(library_nn[None, :], 1.0)
    keep = (ratio > 0.86) & (ratio < 1.16)
    if int(np.count_nonzero(keep)) < 4:
        return []
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    rotated = library_xy @ rotation.T
    shift_x = query_xy[:, None, 0] - rotated[None, :, 0]
    shift_y = query_xy[:, None, 1] - rotated[None, :, 1]
    cell = 6.0
    qx = np.rint(shift_x[keep] / cell).astype(np.int64)
    qy = np.rint(shift_y[keep] / cell).astype(np.int64)
    keys = (qx + 200_000) * 400_000 + (qy + 200_000)
    uniq, counts = np.unique(keys, return_counts=True)
    order = np.argsort(counts)[::-1][:5]
    found = []
    for index in order:
        if int(counts[index]) < 4:
            continue
        key = int(uniq[index])
        bin_x = key // 400_000 - 200_000
        bin_y = key % 400_000 - 200_000
        found.append((float(counts[index]) + 0.01, angle, bin_x * cell, bin_y * cell))
    return found


def _vote_angle(library_xy, library_w, query_xy, query_w, angle: float) -> list[tuple[float, float, float, float]]:
    cell = 6.0
    radians = math.radians(angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    rotated = library_xy @ rotation.T
    shift_x = query_xy[:, None, 0] - rotated[None, :, 0]
    shift_y = query_xy[:, None, 1] - rotated[None, :, 1]
    weight = np.sqrt(query_w[:, None] * library_w[None, :])
    compatible = _spacing_compatible(library_xy, query_xy)
    if int(np.count_nonzero(compatible)) >= 12:
        weight = weight * compatible
    qx = np.rint(shift_x / cell).astype(np.int64)
    qy = np.rint(shift_y / cell).astype(np.int64)
    keys = (qx + 200_000) * 400_000 + (qy + 200_000)
    flat_keys = keys.ravel()
    flat_w = weight.ravel()
    uniq, inverse = np.unique(flat_keys, return_inverse=True)
    counts = np.bincount(inverse, weights=flat_w)
    if len(counts) == 0:
        return []
    # Keep near-peak bins, not only the single noisiest one. A thin overlap can
    # outvote a random pad collision by precision even when its raw vote is lower.
    peak = float(counts.max())
    chosen = np.flatnonzero(counts >= max(3.0, peak - 4.0))
    if abs(angle) <= 1.5:
        extra = np.flatnonzero(counts >= 5.0)
        if len(extra):
            chosen = np.unique(np.concatenate([chosen, extra]))
    if len(chosen) > 70:
        chosen = chosen[np.argsort(counts[chosen])[-70:]]
    found = []
    for index in chosen:
        key = int(uniq[index])
        bin_x = key // 400_000 - 200_000
        bin_y = key % 400_000 - 200_000
        found.append((float(counts[index]), angle, bin_x * cell, bin_y * cell))
    return found


def _score_pose(library_xy, query_xy, matrix, width: int, height: int, thresh: float = _SEARCH_THRESH):
    predicted = _apply(matrix, library_xy)
    inside = (
        (predicted[:, 0] >= -1.0)
        & (predicted[:, 0] < float(width) + 1.0)
        & (predicted[:, 1] >= -1.0)
        & (predicted[:, 1] < float(height) + 1.0)
    )
    inside_count = int(np.count_nonzero(inside))
    if inside_count == 0 or len(query_xy) == 0:
        empty = np.zeros((0, 2), dtype=np.float64)
        return 0, 0.0, 99.0, empty, empty
    distances = np.linalg.norm(query_xy[:, None, :] - predicted[None, :, :], axis=2)
    nearest = distances.argmin(axis=0)
    mind = distances.min(axis=0)
    hit = inside & (mind <= thresh)
    hits = int(np.count_nonzero(hit))
    precision = float(hits) / float(inside_count)
    residual = float(np.median(mind[hit])) if hits else 99.0
    return hits, precision, residual, library_xy[hit], query_xy[nearest[hit]]


def _maybe_refine(best: dict, library_xy, query_xy, width: int, height: int) -> dict | None:
    if best["hits"] < 4 or len(best["src_xy"]) < 4:
        return None
    refined = _refine_similarity(best["src_xy"], best["dst_xy"])
    if refined is None:
        return None
    matrix = np.asarray(refined[:2, :], dtype=np.float64)
    if not _similarity_ok(matrix):
        return None
    if _angle_jump(best["matrix"], matrix) > 8.0:
        return None
    hits, precision, residual, src_xy, dst_xy = _score_pose(library_xy, query_xy, matrix, width, height)
    if hits < best["hits"] - 1 or precision + 0.05 < best["precision"]:
        return None
    if residual > best["residual"] + 0.4:
        return None
    return {
        "votes": best["votes"],
        "angle": _angle_of(matrix),
        "hits": hits,
        "precision": precision,
        "residual": residual,
        "matrix": matrix,
        "src_xy": src_xy,
        "dst_xy": dst_xy,
    }


def _cluster_modes(scored: list[dict]) -> list[dict]:
    # A perfect match of a small copied patch must not outrank a larger true overlap.
    ordered = sorted(
        scored,
        key=lambda item: (
            item["precision"] >= _MIN_PRECISION,
            item["precision"],
            -item["residual"],
            item["hits"],
        ),
        reverse=True,
    )
    clusters: list[dict] = []
    for item in ordered:
        if any(_modes_agree(item, other) for other in clusters):
            continue
        clusters.append(item)
    return clusters


def _modes_agree(left: dict, right: dict) -> bool:
    """Angle bins of one alignment disagree in (tx, theta) but land on the same pads."""
    points = left["src_xy"] if len(left["src_xy"]) >= 3 else right["src_xy"]
    if len(points) < 3:
        return False
    if len(points) > 48:
        points = points[:48]
    return _pose_separation(left["matrix"], right["matrix"], points) <= 8.0


def _pose_separation(left: np.ndarray, right: np.ndarray, points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    delta = _apply(left, points) - _apply(right, points)
    return float(np.median(np.linalg.norm(delta, axis=1)))


def _disjoint_constellation(best: dict, library_xy, library_w, query_xy, query_w, width: int, height: int) -> bool:
    """A second copy of the same pads, stored somewhere else, is not a unique lock."""
    if len(best["src_xy"]) < 4 or len(library_xy) < 8:
        return False
    distances = np.linalg.norm(library_xy[:, None, :] - best["src_xy"][None, :, :], axis=2).min(axis=1)
    keep = distances > 5.0
    if int(np.count_nonzero(keep)) < 8:
        return False
    second = estimate_similarity_pose(
        library_xy[keep],
        query_xy,
        width,
        height,
        library_w=library_w[keep],
        query_w=query_w,
        _disjoint_check=False,
    )
    needed = max(8, int(math.ceil(0.40 * float(best["hits"]))))
    if second.hits < needed or second.precision < 0.90 or second.residual > 2.5:
        return False
    probe = second.src_xy if len(second.src_xy) >= 3 else best["src_xy"]
    return _pose_separation(best["matrix"], second.matrix, probe) > 12.0


def _rival_mode(best: dict, others: list[dict]) -> dict | None:
    needed = max(6, int(math.ceil(_RIVAL_HIT_RATIO * float(best["hits"]))))
    duplicate = max(8, int(math.ceil(0.40 * float(best["hits"]))))
    for other in others:
        if other["residual"] > 3.5 or _modes_agree(best, other):
            continue
        strong = (
            other["precision"] >= _RIVAL_PRECISION
            and other["hits"] >= needed
            and other["residual"] <= 1.6
        )
        copied = other["precision"] >= 0.90 and other["residual"] <= 1.2 and other["hits"] >= duplicate
        if strong or copied:
            return other
    return None


def _acceptable(item: dict) -> bool:
    angle = abs(_angle_of(item["matrix"]))
    return (
        item["hits"] >= _MIN_HITS
        and item["precision"] >= _MIN_PRECISION
        and item["residual"] <= _MAX_RESIDUAL
        and angle <= 18.5
        and _similarity_ok(item["matrix"])
    )


def _matrix(angle_deg: float, scale: float, tx: float, ty: float) -> np.ndarray:
    radians = math.radians(angle_deg)
    cosine = math.cos(radians) * scale
    sine = math.sin(radians) * scale
    return np.array([[cosine, -sine, tx], [sine, cosine, ty]], dtype=np.float64)


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    return pts @ matrix[:, :2].T + matrix[:, 2]


def _similarity_ok(matrix: np.ndarray) -> bool:
    linear = np.asarray(matrix, dtype=np.float64)[:2, :2]
    if float(np.linalg.det(linear)) <= 0.0:
        return False
    scale = float(np.linalg.norm(linear[:, 0]))
    return 0.93 <= scale <= 1.07 and np.isfinite(matrix).all()


def _pair_residual(matrix: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    if len(src) == 0:
        return 99.0
    return float(np.median(np.linalg.norm(_apply(matrix, src) - dst, axis=1)))


def _translation_jump(left: np.ndarray, right: np.ndarray) -> float:
    return float(math.hypot(float(left[0, 2] - right[0, 2]), float(left[1, 2] - right[1, 2])))


def _angle_of(matrix: np.ndarray) -> float:
    linear = np.asarray(matrix, dtype=np.float64)
    return float(math.degrees(math.atan2(linear[1, 0], linear[0, 0])))


def _angle_jump(left: np.ndarray, right: np.ndarray) -> float:
    return abs(_angle_delta(_angle_of(left), _angle_of(right)))


def _angle_delta(left: float, right: float) -> float:
    return (left - right + 180.0) % 360.0 - 180.0


def _spatial_sample(points: np.ndarray, weights: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= limit:
        return points, weights
    minimum = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - minimum, 1.0)
    aspect = float(span[0] / span[1])
    cols = max(1, int(math.sqrt(limit * aspect)))
    rows = max(1, int(math.ceil(limit / cols)))
    cell = span / np.array([cols, rows], dtype=np.float64)
    ix = np.clip(((points[:, 0] - minimum[0]) / cell[0]).astype(int), 0, cols - 1)
    iy = np.clip(((points[:, 1] - minimum[1]) / cell[1]).astype(int), 0, rows - 1)
    keys = ix * (rows + 1) + iy
    best_index: dict[int, int] = {}
    best_weight: dict[int, float] = {}
    for index, key in enumerate(keys.tolist()):
        weight = float(weights[index])
        if key not in best_index or weight > best_weight[key]:
            best_index[key] = index
            best_weight[key] = weight
    chosen = np.fromiter(best_index.values(), dtype=int, count=len(best_index))
    if len(chosen) > limit:
        chosen = chosen[:limit]
    return points[chosen], weights[chosen]
