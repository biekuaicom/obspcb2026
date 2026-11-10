"""Semi-automatic tile import, mosaic, and global-reference compare.

Filename rules are fixed (do not classify by image size):

- global reference source name: ``微信图片_20261109184831_115_34.jpg``
- local tiles: ``20261109-*.jpg``
- after import the global file is ``global_ref.png``

OpenCV ``Stitcher`` is an optional probe only. The mosaic used for compare is
built by placing SIFT-localized tiles into the global frame, with a pairwise
feature chain as fallback. A weak alignment does not pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from model.schema import BoardProject, resolve_project_relative

GLOBAL_REF_SOURCE_NAME = "微信图片_20261109184831_115_34.jpg"
NORMALIZED_GLOBAL_NAME = "global_ref.png"
TILE_NAME_RE = re.compile(r"^20261109-.+\.(jpg|jpeg)$", re.IGNORECASE)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_SHANGHAI = timezone(timedelta(hours=8))
_MIN_INLIERS = 12
_COMPARE_MIN_INLIERS = 15
_COMPARE_MIN_NCC = 0.2


@dataclass
class TileSet:
    board_id: str
    tile_paths: list[Path]
    global_ref_path: Path | None = None
    manifest_path: Path | None = None


@dataclass
class StitchResult:
    ok: bool
    mosaic_path: Path | None
    message: str = ""
    coverage_hint: float | None = None
    placements: list[dict] = field(default_factory=list)
    compare: dict | None = None


def classify_sample_name(name: str) -> str:
    """Return ``global``, ``tile``, or ``ignore``. Size is not consulted."""
    if name == GLOBAL_REF_SOURCE_NAME or name == NORMALIZED_GLOBAL_NAME:
        return "global"
    if TILE_NAME_RE.match(name):
        return "tile"
    return "ignore"


def discover_raw_assets(board_dir: Path) -> tuple[list[Path], Path | None]:
    """Split a zip folder by filename. Other images are ignored."""
    tiles: list[Path] = []
    global_ref: Path | None = None
    root = Path(board_dir)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    for path in files:
        kind = classify_sample_name(path.name)
        if kind == "global" and global_ref is None:
            global_ref = path
        elif kind == "tile":
            tiles.append(path)
    tiles.sort(key=lambda path: path.name)
    return tiles, global_ref


class SampleAssetImporter:
    """Copy a raw sample folder into ``assets/samples/<board_id>/`` and write a project stub."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)

    def sample_dir(self, board_id: str) -> Path:
        return self.repo_root / "assets" / "samples" / board_id

    def import_board_dir(self, board_dir: Path, board_id: str) -> TileSet:
        """Load ``tiles/`` + ``global_ref.png``, or normalize a raw zip folder into that layout."""
        board_dir = Path(board_dir)
        destination = self.sample_dir(board_id)
        normalized_global = board_dir / NORMALIZED_GLOBAL_NAME
        tiles_dir = board_dir / "tiles"
        if tiles_dir.is_dir() and normalized_global.is_file() and board_dir.resolve() == destination.resolve():
            tile_paths = _images_in(tiles_dir)
        else:
            raw_tiles, raw_global = discover_raw_assets(board_dir)
            if raw_global is None:
                raise FileNotFoundError(
                    f"未找到全局对照图 {GLOBAL_REF_SOURCE_NAME} 或 {NORMALIZED_GLOBAL_NAME}"
                )
            if not raw_tiles:
                raise FileNotFoundError("未找到局部瓦片（文件名须匹配 20261109-*.jpg）")
            tile_paths = _materialize(destination, raw_tiles, raw_global)
            normalized_global = destination / NORMALIZED_GLOBAL_NAME
        manifest_path = destination / "tiles_manifest.json"
        _write_manifest(manifest_path, board_id, tile_paths, normalized_global)
        return TileSet(
            board_id=board_id,
            tile_paths=tile_paths,
            global_ref_path=normalized_global,
            manifest_path=manifest_path,
        )

    def ensure_project_stub(self, board_id: str, tile_set: TileSet) -> Path:
        """Write ``assets/projects/<board_id>.json`` with paths relative to that file (D11)."""
        projects = self.repo_root / "assets" / "projects"
        projects.mkdir(parents=True, exist_ok=True)
        project_path = projects / f"{board_id}.json"
        global_map = f"../maps/{board_id}.png"
        global_ref = f"../samples/{board_id}/{NORMALIZED_GLOBAL_NAME}"
        for relative in (global_map, global_ref):
            resolve_project_relative(project_path, relative, self.repo_root)
        project = BoardProject(
            board_id=board_id,
            global_map_path=global_map,
            components=[],
            name="局部瓦片样例",
            global_ref_path=global_ref,
            fiducials=[],
            meta={
                "sample_tiles_dir": f"../samples/{board_id}/tiles",
                "source_note": (
                    f"imported from 局部图片.zip; global={GLOBAL_REF_SOURCE_NAME}; "
                    "tiles=20261109-*.jpg"
                ),
                "imported_at": _now(),
            },
        )
        project_path.write_text(project.to_json() + "\n", encoding="utf-8")
        if tile_set.manifest_path is None and tile_set.global_ref_path is not None:
            tile_set.manifest_path = tile_set.global_ref_path.parent / "tiles_manifest.json"
        return project_path


class SemiAutoStitchPipeline:
    """Place tiles on the global reference. Stitcher is recorded, not trusted alone."""

    def __init__(self, *, try_stitcher: bool = False, min_inliers: int = _MIN_INLIERS):
        self.try_stitcher = try_stitcher
        self.min_inliers = min_inliers

    def run(self, tile_set: TileSet, out_mosaic: Path) -> StitchResult:
        if tile_set.global_ref_path is None or not Path(tile_set.global_ref_path).is_file():
            return StitchResult(False, None, "缺少 global_ref，不能对照，拼接失败关闭", 0.0)
        if not tile_set.tile_paths:
            return StitchResult(False, None, "没有局部瓦片", 0.0)
        global_bgr = cv2.imread(str(tile_set.global_ref_path), cv2.IMREAD_COLOR)
        if global_bgr is None:
            return StitchResult(False, None, f"无法读取全局对照图 {tile_set.global_ref_path}", 0.0)

        placements, rejected = _localize_tiles(global_bgr, tile_set.tile_paths, self.min_inliers)
        stitcher_note = "OpenCV Stitcher 未作为唯一路径"
        if self.try_stitcher:
            stitcher_note = _probe_stitcher(tile_set.tile_paths[:4])
        out_mosaic = Path(out_mosaic)
        out_mosaic.parent.mkdir(parents=True, exist_ok=True)
        if placements:
            pre_mosaic, coverage_mask = _blend_pre_mosaic(global_bgr.shape, placements)
            holes = _analyze_holes(global_bgr, coverage_mask, placements)
            cropped = _smart_crop(pre_mosaic, holes["crop_bounds"])
            pre_path = out_mosaic.with_name("pre_mosaic.png")
            if not cv2.imwrite(str(pre_path), pre_mosaic) or not cv2.imwrite(str(out_mosaic), cropped):
                return StitchResult(False, None, "写出 mosaic 失败", holes["image_coverage"], placements)
            compared = self.compare_to_ref(pre_path, tile_set.global_ref_path)
            compared["coverage_hint"] = holes["image_coverage"]
            compared["tile_accepted"] = len(placements)
            compared["tile_count"] = len(tile_set.tile_paths)
            compared["stitcher"] = stitcher_note
            compared["holes"] = holes
            compared["rejected"] = rejected
            compared["pre_mosaic"] = str(pre_path)
            compared["pipeline"] = "pre-locate full global_ref, hole analysis, then smart crop and feather blend"
            message = (
                f"先在完整全局图上定位再融合：{len(placements)}/{len(tile_set.tile_paths)} 张瓦片。"
                f" 空洞 {holes['hole_count']} 个（其中裁切造成 {holes['crop_induced_count']}）。"
                f" {stitcher_note}。{compared['notes']}"
            )
            if rejected:
                message += " 未放入：" + "; ".join(item["reason"] for item in rejected[:6])
            return StitchResult(
                ok=bool(compared["ok"]),
                mosaic_path=out_mosaic,
                message=message,
                coverage_hint=holes["image_coverage"],
                placements=placements,
                compare=compared,
            )

        chained = _chain_mosaic(tile_set.tile_paths)
        if chained is None:
            return StitchResult(
                ok=False,
                mosaic_path=None,
                message=(
                    "瓦片无法定位到全局对照图，相邻瓦片特征链也失败。"
                    f"{stitcher_note}。不把失败当成通过。"
                ),
                coverage_hint=0.0,
                placements=[],
            )
        if not cv2.imwrite(str(out_mosaic), chained):
            return StitchResult(False, None, "写出链式 mosaic 失败", 0.0)
        compared = self.compare_to_ref(out_mosaic, tile_set.global_ref_path)
        compared["stitcher"] = stitcher_note
        return StitchResult(
            ok=bool(compared["ok"]),
            mosaic_path=out_mosaic,
            message=f"仅得到瓦片坐标系链式拼接，对照全局未通过。{compared['notes']} {stitcher_note}",
            coverage_hint=0.0,
            compare=compared,
        )

    def compare_to_ref(self, mosaic: Path, global_ref: Path) -> dict:
        """Score mosaic against the global reference. Weak or missing images fail closed."""
        mosaic_bgr = cv2.imread(str(mosaic), cv2.IMREAD_COLOR)
        global_bgr = cv2.imread(str(global_ref), cv2.IMREAD_COLOR)
        if mosaic_bgr is None or global_bgr is None:
            return {"ok": False, "notes": "mosaic 或 global_ref 读不出来", "ncc": None, "inliers": 0}
        aligned, homography, inliers, matches = _align_to_ref(mosaic_bgr, global_bgr)
        ncc = _masked_ncc(aligned, global_bgr) if aligned is not None else None
        ok = (
            aligned is not None
            and inliers >= _COMPARE_MIN_INLIERS
            and ncc is not None
            and ncc >= _COMPARE_MIN_NCC
        )
        notes = (
            f"inliers={inliers} matches={matches} ncc={None if ncc is None else round(ncc, 3)} "
            f"threshold inliers>={_COMPARE_MIN_INLIERS} ncc>={_COMPARE_MIN_NCC}"
        )
        if not ok:
            notes = "对照未通过。" + notes
        else:
            notes = "对照通过。" + notes
        return {
            "ok": ok,
            "notes": notes,
            "ncc": ncc,
            "inliers": inliers,
            "matches": matches,
            "homography": None if homography is None else homography.tolist(),
            "aligned": aligned,
            "global_bgr": global_bgr,
        }


def write_compare_artifacts(result: StitchResult, global_ref: Path, out_dir: Path) -> dict:
    """Write overlay, diff, and summary.json. ``aligned`` arrays are not kept in the json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compared = dict(result.compare or {"ok": False, "notes": result.message, "ncc": None, "inliers": 0})
    aligned = compared.pop("aligned", None)
    global_bgr = compared.pop("global_bgr", None)
    if global_bgr is None:
        global_bgr = cv2.imread(str(global_ref), cv2.IMREAD_COLOR)
    overlay_path = out_dir / "overlay.png"
    diff_path = out_dir / "diff.png"
    if global_bgr is not None:
        overlay = global_bgr.copy()
        for item in result.placements:
            corners = np.asarray(item["corners"], dtype=np.int32)
            cv2.polylines(overlay, [corners], True, (0, 220, 0), 2)
            origin = tuple(int(v) for v in corners[0])
            cv2.putText(overlay, Path(item["tile"]).stem[-6:], origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        for hole in (compared.get("holes") or {}).get("regions", []):
            x, y, box_w, box_h = hole["bbox"]
            color = (0, 0, 255) if hole.get("crop_induced") else (0, 165, 255)
            cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), color, 2)
        cv2.imwrite(str(overlay_path), overlay)
        if aligned is not None:
            diff = cv2.absdiff(aligned, global_bgr)
            cv2.imwrite(str(diff_path), diff)
    summary = {
        "ok": result.ok,
        "message": result.message,
        "mosaic": None if result.mosaic_path is None else str(result.mosaic_path),
        "coverage_hint": result.coverage_hint,
        "global_ref": str(global_ref),
        "tiles": [str(item.get("tile")) for item in result.placements],
        "rejected": [item for item in result.placements],
        "compare": {key: value for key, value in compared.items() if key != "homography"},
        "compare_homography": compared.get("homography"),
        "overlay": str(overlay_path),
        "diff": str(diff_path) if diff_path.is_file() else None,
    }
    # placements already listed; don't duplicate huge corner lists twice in rejected
    summary["placements"] = [
        {"tile": item["tile"], "inliers": item["inliers"], "corners": item["corners"]}
        for item in result.placements
    ]
    summary.pop("rejected", None)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _images_in(tiles_dir: Path) -> list[Path]:
    return sorted(
        (path for path in tiles_dir.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES),
        key=lambda path: path.name,
    )


def _materialize(destination: Path, tiles: list[Path], global_ref: Path) -> list[Path]:
    tiles_dir = destination / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in tiles:
        target = tiles_dir / source.name
        if source.resolve() != target.resolve():
            target.write_bytes(source.read_bytes())
        copied.append(target)
    image = cv2.imread(str(global_ref), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取全局对照图 {global_ref}")
    ok = cv2.imwrite(str(destination / NORMALIZED_GLOBAL_NAME), image)
    if not ok:
        raise OSError(f"无法写出 {destination / NORMALIZED_GLOBAL_NAME}")
    return copied


def _write_manifest(path: Path, board_id: str, tiles: list[Path], global_ref: Path) -> None:
    root = path.parent
    payload = {
        "board_id": board_id,
        "tiles": [path_to_manifest(root, tile) for tile in tiles],
        "global_ref": path_to_manifest(root, global_ref),
        "source_note": (
            f"global filename locked to {GLOBAL_REF_SOURCE_NAME}; "
            "tiles locked to 20261109-*.jpg; not classified by image size"
        ),
        "imported_at": _now(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def path_to_manifest(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _now() -> str:
    return datetime.now(_SHANGHAI).isoformat(timespec="seconds")


def _downscale(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 0.999:
        return image, 1.0
    resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


# Coarse search uses the whole global image at several scales. No tile is
# cropped, and the global image is not windowed, until every tile has been tried.
_SCALE_PAIRS = ((0.5, 0.5), (0.75, 0.5), (0.35, 0.75), (0.5, 1.0), (1.0, 1.0))


def _match_homography(src_bgr: np.ndarray, dst_bgr: np.ndarray, min_inliers: int) -> tuple[np.ndarray | None, int, int]:
    """Single-scale helper kept for compare/chain. Does not crop either image."""
    found = _best_homography(src_bgr, dst_bgr, ((1.0, 1.0), (0.5, 0.5)), min_inliers, cache=None)
    if found is None:
        return None, 0, 0
    return found["H"], found["inliers"], found["matches"]


def _best_homography(tile_bgr, global_bgr, scale_pairs, min_inliers, cache):
    height, width = global_bgr.shape[:2]
    tile_h, tile_w = tile_bgr.shape[:2]
    best = None
    best_inliers = 0
    best_matches = 0
    if cache is None:
        cache = {}
    global_gray = cv2.cvtColor(global_bgr, cv2.COLOR_BGR2GRAY) if global_bgr.ndim == 3 else global_bgr
    tile_gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY) if tile_bgr.ndim == 3 else tile_bgr
    for tile_scale, global_scale in scale_pairs:
        keys_dst, desc_dst = _cached_features(cache, global_gray, global_scale)
        tile_small = _resize_scale(tile_gray, tile_scale)
        keys_src, desc_src = _features(tile_small)
        if desc_src is None or desc_dst is None or len(keys_src) < 8 or len(keys_dst) < 8:
            continue
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        pairs = matcher.knnMatch(desc_src, desc_dst, k=2)
        good = []
        for pair in pairs:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < 0.75 * second.distance:
                good.append(first)
        best_matches = max(best_matches, len(good))
        if len(good) < max(8, min_inliers):
            continue
        src_pts = np.float32([keys_src[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keys_dst[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0
        best_inliers = max(best_inliers, inliers)
        if homography is None or inliers < min_inliers:
            continue
        full = _lift_homography(homography, tile_scale, global_scale)
        corners = _project_corners((tile_h, tile_w), full)
        if not _quad_plausible(corners, width, height):
            continue
        if best is None or inliers > best["inliers"]:
            best = {"H": full, "inliers": inliers, "matches": len(good), "corners": corners, "scale": (tile_scale, global_scale)}
    if best is None and best_inliers:
        return {"H": None, "inliers": best_inliers, "matches": best_matches, "corners": None}
    return best


def _cached_features(cache: dict, gray: np.ndarray, scale: float):
    key = round(float(scale), 3)
    if key not in cache:
        cache[key] = _features(_resize_scale(gray, scale))
    return cache[key]


def _features(gray: np.ndarray):
    sift = cv2.SIFT_create(nfeatures=2500)
    return sift.detectAndCompute(gray, None)


def _resize_scale(image: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return image
    height, width = image.shape[:2]
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def _lift_homography(homography_small: np.ndarray, tile_scale: float, global_scale: float) -> np.ndarray:
    to_small = np.diag([tile_scale, tile_scale, 1.0])
    to_full = np.diag([1.0 / global_scale, 1.0 / global_scale, 1.0])
    return to_full @ homography_small @ to_small


def _refine_on_roi(tile_bgr, global_bgr, homography, min_inliers):
    """After a full-frame hit, re-estimate inside a padded ROI. A failed refine keeps the coarse hit."""
    corners = _project_corners(tile_bgr.shape, homography)
    minimum = corners.min(axis=0)
    maximum = corners.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    x0 = int(np.floor(minimum[0] - 0.2 * span[0]))
    y0 = int(np.floor(minimum[1] - 0.2 * span[1]))
    x1 = int(np.ceil(maximum[0] + 0.2 * span[0]))
    y1 = int(np.ceil(maximum[1] + 0.2 * span[1]))
    height, width = global_bgr.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return homography, None
    roi = global_bgr[y0:y1, x0:x1]
    found = _best_homography(tile_bgr, roi, ((0.75, 1.0), (1.0, 1.0)), min_inliers, cache=None)
    if found is None or found.get("H") is None:
        return homography, None
    shift = np.array([[1.0, 0.0, x0], [0.0, 1.0, y0], [0.0, 0.0, 1.0]], dtype=np.float64)
    lifted = shift @ found["H"]
    refined_corners = _project_corners(tile_bgr.shape, lifted)
    if not _quad_plausible(refined_corners, width, height):
        return homography, None
    return lifted, int(found["inliers"])


def _localize_tiles(global_bgr: np.ndarray, tiles: list[Path], min_inliers: int) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    height, width = global_bgr.shape[:2]
    cache: dict = {}
    for path in tiles:
        tile = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if tile is None:
            rejected.append({"tile": path.name, "reason": f"{path.name}: 读失败", "inliers": 0, "matches": 0})
            continue
        found = _best_homography(tile, global_bgr, _SCALE_PAIRS, min_inliers, cache)
        if found is None or found.get("H") is None:
            inliers = 0 if found is None else found.get("inliers", 0)
            matches = 0 if found is None else found.get("matches", 0)
            rejected.append(
                {
                    "tile": path.name,
                    "reason": f"{path.name}: 全图多尺度未得到稳定单应 inliers={inliers} matches={matches}",
                    "inliers": inliers,
                    "matches": matches,
                }
            )
            continue
        homography = found["H"]
        refined_inliers = None
        homography, refined_inliers = _refine_on_roi(tile, global_bgr, homography, min_inliers)
        corners = _project_corners(tile.shape, homography)
        if not _quad_plausible(corners, width, height):
            rejected.append(
                {
                    "tile": path.name,
                    "reason": f"{path.name}: 细化后单应不可用 inliers={found['inliers']}",
                    "inliers": found["inliers"],
                    "matches": found["matches"],
                }
            )
            continue
        accepted.append(
            {
                "tile": str(path),
                "inliers": found["inliers"] if refined_inliers is None else max(found["inliers"], refined_inliers),
                "matches": found["matches"],
                "corners": corners.tolist(),
                "image": tile,
                "H": homography,
                "refined": refined_inliers is not None,
                "scale": found.get("scale"),
            }
        )
    return accepted, rejected


def _project_corners(shape: tuple, homography: np.ndarray) -> np.ndarray:
    height, width = shape[:2]
    corners = np.array([[[0, 0]], [[width, 0]], [[width, height]], [[0, height]]], dtype=np.float64)
    return cv2.perspectiveTransform(corners, homography).reshape(4, 2)


def _quad_plausible(corners: np.ndarray, width: int, height: int) -> bool:
    if corners is None or not np.isfinite(corners).all():
        return False
    crosses = []
    for index in range(4):
        edge_a = corners[(index + 1) % 4] - corners[index]
        edge_b = corners[(index + 2) % 4] - corners[(index + 1) % 4]
        crosses.append(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])
    crosses = np.asarray(crosses)
    if not (np.all(crosses > 0) or np.all(crosses < 0)):
        return False
    area = abs(float(cv2.contourArea(corners.astype(np.float32))))
    if area < 0.012 * width * height or area > 0.40 * width * height:
        return False
    center = corners.mean(axis=0)
    if not (-0.2 * width <= center[0] <= 1.2 * width and -0.2 * height <= center[1] <= 1.2 * height):
        return False
    return True


def _coverage(shape: tuple, placements: list[dict]) -> float:
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for item in placements:
        corners = np.asarray(item["corners"], dtype=np.int32)
        cv2.fillConvexPoly(mask, corners, 1)
    return float(mask.mean())


def _blend_pre_mosaic(shape: tuple, placements: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Feather-blend every placement on the full global canvas. Overlaps stay."""
    height, width = shape[:2]
    accum = np.zeros((height, width, 3), dtype=np.float64)
    weight = np.zeros((height, width), dtype=np.float64)
    for item in placements:
        warped = cv2.warpPerspective(item["image"], item["H"], (width, height))
        binary = cv2.warpPerspective(np.full(item["image"].shape[:2], 255, np.uint8), item["H"], (width, height))
        inside = binary > 0
        if not np.any(inside):
            continue
        distance = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 3)
        feather = np.clip(distance, 0, 32) / 32.0
        accum += warped.astype(np.float64) * feather[..., None]
        weight += feather
    mosaic = np.zeros((height, width, 3), dtype=np.uint8)
    valid = weight > 1e-6
    mosaic[valid] = np.clip(accum[valid] / weight[valid, None], 0, 255).astype(np.uint8)
    return mosaic, (weight > 1e-6).astype(np.uint8)


def _analyze_holes(global_bgr: np.ndarray, coverage: np.ndarray, placements: list[dict]) -> dict:
    """Holes on the board after the full pre-mosaic. Crop-induced holes touch a tile edge."""
    height, width = global_bgr.shape[:2]
    bright = global_bgr.max(axis=2)
    dark = global_bgr.min(axis=2)
    board = ~((bright > 245) & (dark > 235))
    uncovered = board & (coverage == 0)
    edge = np.zeros((height, width), dtype=np.uint8)
    for item in placements:
        corners = np.asarray(item["corners"], dtype=np.int32)
        cv2.polylines(edge, [corners], True, 1, 4)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(uncovered.astype(np.uint8), 8)
    regions = []
    crop_induced = 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 500:
            continue
        x, y, box_w, box_h = (int(stats[index, column]) for column in range(4))
        component = labels == index
        touches_tile = bool(np.any(component & (edge > 0)))
        induced = touches_tile
        if induced:
            crop_induced += 1
        regions.append(
            {
                "area": area,
                "bbox": [x, y, box_w, box_h],
                "crop_induced": induced,
                "touches_tile_edge": touches_tile,
            }
        )
    covered_board = int(np.count_nonzero(board & (coverage > 0)))
    board_pixels = int(np.count_nonzero(board))
    ys, xs = np.where(coverage > 0)
    if len(xs) == 0:
        bounds = [0, 0, width, height]
    else:
        pad = int(0.02 * max(height, width))
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(width, int(xs.max()) + 1 + pad)
        y1 = min(height, int(ys.max()) + 1 + pad)
        bounds = [x0, y0, x1 - x0, y1 - y0]
    return {
        "image_coverage": float(coverage.mean()),
        "board_pixels": board_pixels,
        "covered_board_pixels": covered_board,
        "board_coverage": 0.0 if board_pixels == 0 else covered_board / board_pixels,
        "hole_count": len(regions),
        "crop_induced_count": crop_induced,
        "regions": regions,
        "crop_bounds": bounds,
    }


def _smart_crop(image: np.ndarray, bounds: list[int]) -> np.ndarray:
    x, y, width, height = bounds
    if width < 8 or height < 8:
        return image
    return image[y : y + height, x : x + width].copy()


def _chain_mosaic(tiles: list[Path]) -> np.ndarray | None:
    images = []
    for path in tiles:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            small, _scale = _downscale(image, 480)
            images.append(small)
    if len(images) < 2:
        return None
    transforms = [np.eye(3, dtype=np.float64)]
    for previous, current in zip(images, images[1:], strict=False):
        homography, inliers, _matches = _match_homography(current, previous, _MIN_INLIERS)
        if homography is None or inliers < _MIN_INLIERS:
            return None
        transforms.append(transforms[-1] @ homography)
    return _compose(images, transforms)


def _compose(images: list[np.ndarray], transforms: list[np.ndarray]) -> np.ndarray | None:
    corners = []
    for image, homography in zip(images, transforms, strict=True):
        projected = _project_corners(image.shape, homography)
        corners.append(projected)
    stacked = np.vstack(corners)
    minimum = np.floor(stacked.min(axis=0)).astype(np.float64)
    maximum = np.ceil(stacked.max(axis=0)).astype(np.float64)
    span = maximum - minimum
    if span[0] > 8000 or span[1] > 8000 or span[0] < 2 or span[1] < 2:
        return None
    shift = np.array([[1, 0, -minimum[0]], [0, 1, -minimum[1]], [0, 0, 1]], dtype=np.float64)
    canvas = np.zeros((int(span[1]), int(span[0]), 3), dtype=np.uint8)
    for image, homography in zip(images, transforms, strict=True):
        warped = cv2.warpPerspective(image, shift @ homography, (canvas.shape[1], canvas.shape[0]))
        mask = warped.sum(axis=2) > 0
        canvas[mask] = warped[mask]
    return canvas


def _score_pair(mosaic: np.ndarray, global_bgr: np.ndarray) -> tuple[int, int]:
    _homography, inliers, matches = _match_homography(mosaic, global_bgr, min_inliers=8)
    return inliers, matches


def _align_to_ref(mosaic: np.ndarray, global_bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, int, int]:
    if mosaic.shape == global_bgr.shape:
        inliers, matches = _score_pair(mosaic, global_bgr)
        return mosaic, np.eye(3, dtype=np.float64), inliers, matches
    homography, inliers, matches = _match_homography(mosaic, global_bgr, min_inliers=8)
    if homography is None:
        return None, None, inliers, matches
    aligned = cv2.warpPerspective(mosaic, homography, (global_bgr.shape[1], global_bgr.shape[0]))
    return aligned, homography, inliers, matches


def _masked_ncc(aligned: np.ndarray, global_bgr: np.ndarray) -> float | None:
    if aligned.shape != global_bgr.shape:
        return None
    mask = aligned.sum(axis=2) > 0
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


def _probe_stitcher(tile_paths: list[Path]) -> str:
    images = []
    for path in tile_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        small, _scale = _downscale(image, 480)
        images.append(small)
    if len(images) < 2:
        return "OpenCV Stitcher 跳过：样本不足"
    try:
        if hasattr(cv2, "Stitcher_create"):
            stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        else:
            stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
        status, _pano = stitcher.stitch(images)
    except cv2.error as exc:
        return f"OpenCV Stitcher 失败：{exc}"
    if status != getattr(cv2, "Stitcher_OK", 0):
        return f"OpenCV Stitcher 未成功（status={status}），已改用特征定位，不把它当唯一结果"
    return "OpenCV Stitcher 在缩小图上返回了全景，仍以特征定位 mosaic 为准"
