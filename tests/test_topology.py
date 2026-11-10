"""Topology live matcher: known blob offsets, and a hard fail when no library is set."""

import json

import cv2
import numpy as np

from model.schema import BoardProject
from vision.topology import (
    SCHEMA_VERSION,
    ConstellationMatcher,
    GeometricLocalizer,
    Pose2D,
    TopologyExtractor,
    geo_fingerprint,
    medium_dark,
    pose_error_px,
    render_live_overlay,
    rotated_crop_m,
    rotate_image,
    translation_m,
)


def _board(height=280, width=360, seed=2):
    """Irregular holes and long pads so each local star is not a copy of the next."""
    rng = np.random.default_rng(seed)
    image = np.full((height, width), 178, np.uint8)
    noise = rng.integers(-4, 5, size=image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    holes = []
    pads = []
    for row, y in enumerate(range(52, height - 52, 32)):
        for col, x in enumerate(range(48, width - 48, 36)):
            px = int(np.clip(x + int(rng.integers(-7, 8)), 20, width - 20))
            py = int(np.clip(y + int(rng.integers(-6, 7)), 20, height - 20))
            if (row + col) % 3 == 0:
                cv2.circle(image, (px, py), 5, 16, -1)
                holes.append((float(px), float(py)))
            else:
                half_w = 12 + int(rng.integers(0, 6))
                cv2.rectangle(image, (px - half_w, py - 4), (px + half_w, py + 4), 28, -1)
                pads.append((float(px), float(py)))
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return color, holes, pads


def test_extract_finds_holes_pads_and_roi_is_in_frame_coordinates():
    image, holes, pads = _board()
    extractor = TopologyExtractor(max_long_edge=1280, adaptive_block=31, adaptive_C=5)
    full = extractor.extract(image)
    assert full.ok is True
    kinds = {point.kind for point in full.points}
    assert "hole" in kinds and "pad" in kinds
    xy = full.xy()
    hole_hits = sum(float(np.linalg.norm(xy - np.array(point), axis=1).min()) < 3.0 for point in holes)
    pad_hits = sum(float(np.linalg.norm(xy - np.array(point), axis=1).min()) < 3.0 for point in pads)
    assert hole_hits >= int(0.8 * len(holes))
    assert pad_hits >= int(0.7 * len(pads))
    roi = (40, 36, 180, 150)
    cropped = extractor.extract(image, roi_xywh=roi)
    assert cropped.points
    assert min(point.xy[0] for point in cropped.points) >= roi[0] - 2


def test_long_edge_is_scaled_back_to_original_pixels():
    image = np.full((120, 1600, 3), 176, np.uint8)
    cv2.circle(image, (800, 60), 8, (20, 20, 20), -1)
    cv2.rectangle(image, (900, 40), (940, 80), (30, 30, 30), -1)
    result = TopologyExtractor(max_long_edge=1280).extract(image)
    assert result.scale == 1280 / 1600
    assert result.width == 1600
    xs = [point.xy[0] for point in result.points]
    assert xs and min(abs(x - 800) for x in xs) < 4.0


def test_crop_localizes_with_library_to_frame_pose():
    image, _holes, _pads = _board()
    extractor = TopologyExtractor()
    library = extractor.extract(image)
    library.board_id = "SYN"
    matcher = ConstellationMatcher(min_inliers=8, conf_thresh=0.35)
    matcher.set_topology(library)
    localizer = GeometricLocalizer(extractor, matcher)
    origin = (50, 40)
    window = (200, 170)
    crop = image[origin[1] : origin[1] + window[1], origin[0] : origin[0] + window[0]]
    pose = localizer.track(crop)
    assert pose.ok is True
    assert pose.M.shape == (2, 3)
    assert pose.inliers >= 8
    assert pose.confidence >= 0.35
    assert pose.elapsed_ms < 500
    truth = translation_m(*origin)
    error = pose_error_px(pose.M, truth, library.xy())
    assert error < 2.0
    sample = library.xy()[0]
    mapped = pose.M[:, :2] @ sample + pose.M[:, 2]
    assert abs(mapped[0] - (sample[0] - origin[0])) < 2.0
    assert abs(mapped[1] - (sample[1] - origin[1])) < 2.0
    projected = matcher.project_components(
        [{"component_id": "R1", "base_pixel_box": [origin[0] + 10, origin[1] + 8, 20, 12], "core_params": {"name": "R1"}}],
        pose,
    )
    assert projected and projected[0]["label"] == "R1"
    assert abs(projected[0]["corners"][0][0] - 10) < 3


def test_rotated_crop_recovers_angle():
    image, _holes, _pads = _board()
    extractor = TopologyExtractor()
    matcher = ConstellationMatcher()
    matcher.set_topology(extractor.extract(image))
    origin = (70, 50)
    window = (180, 160)
    angle = 8.0
    crop = image[origin[1] : origin[1] + window[1], origin[0] : origin[0] + window[0]]
    pose = GeometricLocalizer(extractor, matcher).track(rotate_image(crop, angle))
    assert pose.ok is True
    truth = rotated_crop_m(origin[0], origin[1], window[0], window[1], angle)
    assert pose_error_px(pose.M, truth, matcher.topology.xy()) < 3.5
    truth_angle = float(np.degrees(np.arctan2(truth[1, 0], truth[0, 0])))
    assert abs(pose.angle_deg - truth_angle) < 2.5


def test_medium_dark_crop_still_locks():
    image, _holes, _pads = _board()
    extractor = TopologyExtractor()
    library = extractor.extract(image)
    matcher = ConstellationMatcher()
    matcher.set_topology(library)
    origin = (50, 40)
    window = (200, 170)
    crop = image[origin[1] : origin[1] + window[1], origin[0] : origin[0] + window[0]]
    pose = GeometricLocalizer(extractor, matcher).track(medium_dark(crop, scale=0.55, gamma=1.3))
    assert pose.ok is True
    assert pose_error_px(pose.M, translation_m(*origin), library.xy()) < 4.0


def test_unlocked_overlay_has_status_and_no_bubble():
    frame = np.full((100, 180, 3), 200, np.uint8)
    pose = Pose2D(
        M=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        inliers=0,
        confidence=0.0,
        ok=False,
        message="未设置拓扑库，无法定位",
    )
    canvas = render_live_overlay(
        frame,
        pose,
        ConstellationMatcher(),
        [{"component_id": "C", "base_pixel_box": [20, 30, 40, 16], "core_params": {"name": "NO", "value": "1", "package": "p"}}],
    )
    green = (canvas[:, :, 1] > 200) & (canvas[:, :, 0] < 40)
    assert int(green.sum()) == 0
    assert int(canvas[6, 8, 2]) > 120


def test_missing_topology_fails_closed():
    matcher = ConstellationMatcher()
    frame = np.zeros((80, 90, 3), dtype=np.uint8)
    pose = matcher.localize(frame)
    assert pose.ok is False
    assert pose.M.shape == (2, 3)
    assert "未设置拓扑库" in pose.message
    tracked = GeometricLocalizer(TopologyExtractor(), matcher).track(frame)
    assert tracked.ok is False
    assert "未设置拓扑库" in tracked.message
    assert matcher.project_components([{"component_id": "R1", "base_pixel_box": [0, 0, 4, 4]}], pose) == []


def test_topology_db_and_fingerprint_roundtrip(tmp_path):
    image, _holes, _pads = _board()
    library = TopologyExtractor().extract(image)
    library.board_id = "SYN-BOARD"
    library.source = "synthetic"
    path = tmp_path / "assets" / "topology" / "SYN-BOARD.json"
    library.save(path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["board_id"] == "SYN-BOARD"
    assert payload["points"][0]["kind"] in {"hole", "pad", "slot"}
    anchor = library.points[0].xy
    fingerprint = geo_fingerprint(library.points, anchor)
    assert fingerprint["neighbors"]
    project_path = tmp_path / "assets" / "projects" / "SYN-BOARD.json"
    project_path.parent.mkdir(parents=True)
    project = BoardProject(
        board_id="SYN-BOARD",
        global_map_path="../maps/SYN-BOARD.png",
        components=[],
        topology_db="../topology/SYN-BOARD.json",
    )
    from model.schema import Component

    project.components.append(
        Component(
            component_id="C-LIVE",
            base_pixel_box=(anchor[0] - 10, anchor[1] - 6, 20, 12),
            status="unchecked",
            core_params={"name": "C-LIVE", "value": "topo", "package": "pad"},
            geo_fingerprint=fingerprint,
        )
    )
    project_path.write_text(project.to_json() + "\n", encoding="utf-8")
    loaded = BoardProject.from_json(project_path.read_text())
    assert loaded.topology_db == "../topology/SYN-BOARD.json"
    assert loaded.components[0].geo_fingerprint["kind"] == "constellation"

