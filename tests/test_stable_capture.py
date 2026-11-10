"""Stable capture: sharpness, overlap, and a chain that refuses a disconnected view."""

import cv2
import numpy as np

from vision.stable_capture import (
    StableCaptureSession,
    _Tile,
    _translate_m,
    alignment_rejected,
    binary_lines,
    compensate_board_edge,
    frame_overlap_ratio,
    frame_sharpness,
    interframe_motion_px,
    linear_is_reflection,
    paint_red_coverage,
    pose_angle_deg,
    project_tile_quad,
)


def _board(height=220, width=320, seed=4):
    rng = np.random.default_rng(seed)
    image = np.full((height, width), 176, np.uint8)
    noise = rng.integers(-4, 5, size=image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for row, y in enumerate(range(36, height - 36, 26)):
        for col, x in enumerate(range(32, width - 32, 28)):
            px = int(np.clip(x + int(rng.integers(-4, 5)), 16, width - 16))
            py = int(np.clip(y + int(rng.integers(-4, 5)), 16, height - 16))
            if (row + col) % 3 == 0:
                cv2.circle(image, (px, py), 4, 18, -1)
            else:
                cv2.rectangle(image, (px - 9, py - 3), (px + 9, py + 3), 30, -1)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _settle(session, frame):
    session.observe(frame)
    return session.observe(frame)


def test_sharpness_and_motion_gates():
    frame = _board()
    blurred = cv2.GaussianBlur(frame, (0, 0), 3.5)
    assert frame_sharpness(frame) > 40
    assert frame_sharpness(blurred) < 40
    assert interframe_motion_px(frame, frame) < 1.0
    shifted = np.roll(frame, 18, axis=1)
    assert interframe_motion_px(frame, shifted) > 8.0


def test_overlap_ratio_accepts_partial_quad_and_rejects_outside():
    width, height = 200, 120
    inside = project_tile_quad(
        np.array([[1.0, 0.0, -40.0], [0.0, 1.0, 0.0]]),
        width,
        height,
    )
    ratio = frame_overlap_ratio(inside, width, height)
    assert 0.7 < ratio < 0.9
    outside = project_tile_quad(
        np.array([[1.0, 0.0, 500.0], [0.0, 1.0, 0.0]]),
        width,
        height,
    )
    assert frame_overlap_ratio(outside, width, height) == 0.0


def test_red_coverage_paints_only_the_projected_quad():
    frame = np.full((80, 120, 3), 40, np.uint8)
    quad = np.array([[10, 10], [70, 10], [70, 60], [10, 60]], dtype=np.float64)
    painted = paint_red_coverage(frame, quad)
    assert painted.shape == frame.shape
    red = painted[30, 40, 2] > painted[30, 40, 0] + 40
    clear = painted[5, 5, 2] == frame[5, 5, 2]
    assert red and clear


def test_first_shot_waits_for_stable_and_rejects_blur_or_motion():
    frame = _board()
    session = StableCaptureSession()
    assert session.capture().reason == "请先点击开始"
    session.start()
    session.observe(frame)
    waiting = session.capture()
    assert waiting.ok is False
    assert "下一帧" in waiting.reason

    blurred = cv2.GaussianBlur(frame, (0, 0), 4.0)
    _settle(session, blurred)
    blur = session.capture()
    assert blur.ok is False
    assert "模糊" in blur.reason

    _settle(session, frame)
    moved = np.roll(frame, 24, axis=1)
    session.observe(moved)
    motion = session.capture()
    assert motion.ok is False
    assert "移动" in motion.reason


def test_overlap_chains_and_disconnected_view_is_rejected():
    board = _board(height=240, width=460)
    window = board[24:150, 20:160]
    shifted = board[24:150, 44:184]
    disconnected = board[24:150, 300:440]
    session = StableCaptureSession(min_overlap=0.08)
    session.start()
    _settle(session, window)
    first = session.capture()
    assert first.ok is True
    assert first.tile_id == "cap-000"
    assert binary_lines(window).shape[:2] == window.shape[:2]

    state = _settle(session, shifted)
    assert state.tracked is True
    assert state.overlap is not None and state.overlap >= 0.08
    overlay = session.render(shifted)
    assert overlay.shape == shifted.shape
    red = (overlay[:, :, 2] > overlay[:, :, 0] + 20).sum()
    assert int(red) > 50
    assert int(red) < overlay.shape[0] * overlay.shape[1]
    second = session.capture()
    assert second.ok is True
    assert session.edges[0]["src"] == "cap-000"
    assert session.edges[0]["dst"] == "cap-001"
    assert session.edges[0]["direction"] == "previous_tile_to_current_frame"

    _settle(session, disconnected)
    rejected = session.capture()
    assert rejected.ok is False
    assert "退回" in rejected.reason
    assert len(session.tiles) == 2
    assert max(tile.raw.shape[0] for tile in session.tiles) == window.shape[0]
    assert session.topology_db(board_id="SYN", source="synthetic")["kind"] == "stable_capture_chain"


def test_red_union_is_larger_than_the_latest_tile_and_points_share_a_frame():
    board = _board(height=240, width=460)
    views = [
        board[24:150, 20:170],
        board[24:150, 36:186],
        board[24:150, 52:202],
    ]
    session = StableCaptureSession(min_overlap=0.08)
    session.start()
    for view in views:
        _settle(session, view)
        decision = session.capture()
        assert decision.ok is True
    home = _settle(session, views[0])
    assert home.tracked is True
    assert len(session.coverage_quads) == 3
    union = session.render(views[0])
    last_only = paint_red_coverage(views[0], session.coverage_quads[-1])
    union_px = int(np.count_nonzero(union[:, :, 2] > union[:, :, 0] + 20))
    last_px = int(np.count_nonzero(last_only[:, :, 2] > last_only[:, :, 0] + 20))
    assert union_px > last_px * 1.15
    topology = session.topology_db(board_id="SYN", source="synthetic")
    assert topology["chain_frame"] == "cap-000"
    assert topology["full_canvas"] is False
    first = [point for point in topology["points"] if point["tile_id"] == "cap-000"]
    later = [point for point in topology["points"] if point["tile_id"] == "cap-002"]
    assert first and later
    assert first[0]["xy"] == first[0]["xy_tile"]
    assert later[0]["xy"] != later[0]["xy_tile"]
    assert session.chain_mask is not None
    assert session.chain_mask.shape[0] < board.shape[0] or session.chain_mask.shape[1] <= views[0].shape[1] + 80


def _rotate(frame, angle_deg):
    height, width = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle_deg, 1.0)
    rotated = cv2.warpAffine(frame, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated, pose_angle_deg(matrix)


def test_small_rotation_is_recovered_and_a_mirror_is_rejected():
    board = _board(height=240, width=420)
    base = board[30:160, 24:180]
    rotated, truth = _rotate(board[30:160, 40:196], 5.0)
    mirror = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 40.0]], dtype=np.float64)
    assert linear_is_reflection(mirror)
    assert alignment_rejected(mirror)
    session = StableCaptureSession()
    session.start()
    _settle(session, base)
    assert session.capture().ok is True
    _settle(session, rotated)
    decision = session.capture()
    assert decision.ok is True
    estimated = session.edges[0]["angle_deg"]
    error = (estimated - truth + 180.0) % 360.0 - 180.0
    assert abs(error) < 3.0


def _red_mask(image):
    return image[:, :, 2] > image[:, :, 0] + 20


def test_disconnected_false_tile_is_not_painted_red():
    """A wrong chain pose must not drop an isolated red block on unphotographed pixels."""
    board = _board(height=240, width=520)
    window = board[24:150, 20:180]
    shifted = board[24:150, 44:204]
    mid = board[24:150, 60:220]
    # Pad empty board to the right so the live frame has a large unphotographed region
    # while the left still overlaps the captured tiles enough to track.
    live = np.full((mid.shape[0], mid.shape[1] + 160, 3), 176, np.uint8)
    live[:, : mid.shape[1]] = mid
    session = StableCaptureSession(min_overlap=0.08)
    session.start()
    _settle(session, window)
    assert session.capture().ok is True
    _settle(session, shifted)
    assert session.capture().ok is True

    session.tiles.append(
        _Tile(
            tile_id="cap-false",
            width=90,
            height=100,
            raw=np.zeros((100, 90, 3), np.uint8),
            binary=np.zeros((100, 90), np.uint8),
            points=[],
            extracted=None,
            M_to_ref=_translate_m(230, 12),
        )
    )

    state = _settle(session, live)
    assert state.tracked is True
    overlay = session.render(live)
    red = _red_mask(overlay)
    assert int(red[:, 20:120].sum()) > 200
    # Injected tile projects to roughly x=190..280, past the true overlap ending near x=144.
    isolated = red[20:90, 200:280]
    assert int(isolated.sum()) == 0
    assert all(float(np.max(quad[:, 0])) < 160 for quad in session.coverage_quads)


def test_conflicting_anchor_poses_do_not_paint_or_track():
    """Two strong localizations that imply different chain poses must not accept a false union."""
    board = _board(height=240, width=460)
    window = board[24:150, 20:160]
    shifted = board[24:150, 44:184]
    session = StableCaptureSession(min_overlap=0.08)
    session.start()
    _settle(session, window)
    assert session.capture().ok is True
    _settle(session, shifted)
    assert session.capture().ok is True

    original = session.tiles[0]
    session.tiles.append(
        _Tile(
            tile_id="cap-conflict",
            width=original.width,
            height=original.height,
            raw=original.raw.copy(),
            binary=original.binary.copy(),
            points=list(original.points),
            extracted=original.extracted,
            M_to_ref=_translate_m(180, 40),
        )
    )

    state = _settle(session, window)
    assert state.tracked is False
    assert session.coverage_quads == []
    overlay = session.render(window)
    assert int(_red_mask(overlay).sum()) == 0
    rejected = session.capture()
    assert rejected.ok is False
    assert "退回" in rejected.reason
    assert len(session.tiles) == 3


def test_wavy_board_edge_is_pulled_toward_a_line():
    image = np.full((180, 240), 210, np.uint8)
    for x in range(16, 220):
        edge = int(round(48 + 3.0 * np.sin(x / 7.0)))
        image[edge:150, x] = 36
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    _corrected, stats = compensate_board_edge(color)
    assert stats["kind"] in {"line", "arc"}
    assert stats["rms_before"] is not None and stats["rms_before"] > 1.2
    assert stats["rms_after"] < stats["rms_before"] * 0.6


def _connected_regions(mask):
    count, _labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return int(count) - 1


def test_coverage_memory_grows_with_overlapping_shots_and_stays_connected():
    from vision.stable_capture import evaluate_coverage_growth, unexplained_coverage_blobs

    board = _board(height=240, width=460)
    views = [
        board[24:150, 20:170],
        board[24:150, 36:186],
        board[24:150, 52:202],
    ]
    session = StableCaptureSession(min_overlap=0.08)
    session.start()
    areas = []
    for view in views:
        _settle(session, view)
        decision = session.capture()
        assert decision.ok is True
        memory = session.coverage_memory
        assert memory is not None
        areas.append(memory.area)
        assert memory.area > 0
        assert memory.contour
        assert _connected_regions(memory.mask) == 1

    tile_area = views[0].shape[0] * views[0].shape[1]
    assert areas[0] == areas[0]  # first shot records a baseline
    assert areas[1] > areas[0]
    assert areas[2] > areas[1]
    assert (areas[1] - areas[0]) < tile_area
    assert (areas[2] - areas[1]) < tile_area

    payload = session.chain_dict()
    assert payload["tiles"][0]["id"] == "cap-000"
    assert payload["coverage_memory"]["area"] == areas[-1]
    assert "contour" in payload["coverage_memory"]
    assert "moments" in payload["coverage_memory"]
    assert len(payload["structure_tree"]) == 3

    # Edge-level dilation is normal boundary fluctuation, not a false block.
    memory = session.coverage_memory
    slack = cv2.dilate(memory.mask, np.ones((5, 5), np.uint8), iterations=1)
    blobs = unexplained_coverage_blobs(
        slack,
        memory.origin,
        memory.mask,
        memory.origin,
        slack_px=3,
        min_area=400,
    )
    assert int(np.count_nonzero(blobs)) == 0

    # A large far-away island, or a tile scaled beyond one frame, is rejected
    # and must leave the remembered area unchanged.
    recorded = memory.area
    island = evaluate_coverage_growth(memory, views[0].shape[1], views[0].shape[0], _translate_m(420, 16))
    assert island.ok is False
    assert "孤立" in island.reason
    scaled = np.array([[3.0, 0.0, 8.0], [0.0, 3.0, 10.0]], dtype=np.float64)
    too_big = evaluate_coverage_growth(memory, views[0].shape[1], views[0].shape[0], scaled)
    assert too_big.ok is False
    assert "面积" in too_big.reason
    assert session.coverage_memory.area == recorded


def test_coverage_memory_rejects_bad_pose_on_capture_and_stays_unchanged():
    from dataclasses import replace
    from vision.stable_capture import _compose_m, _invert_m

    board = _board(height=240, width=460)
    window = board[24:150, 20:160]
    shifted = board[24:150, 44:184]
    session = StableCaptureSession(min_overlap=0.08)
    session.start()
    _settle(session, window)
    assert session.capture().ok is True
    remembered = session.coverage_memory.area
    _settle(session, shifted)
    assert session._tracked is True and session._pose is not None
    anchor = next(tile for tile in session.tiles if tile.tile_id == session._anchor_id)
    desired = _translate_m(420, 18)
    inv_pose = _compose_m(_invert_m(anchor.M_to_ref), desired)
    session._pose = replace(session._pose, M=_invert_m(inv_pose))
    rejected = session.capture()
    assert rejected.ok is False
    assert "退回" in rejected.reason
    assert len(session.tiles) == 1
    assert session.coverage_memory.area == remembered

    # A scaled pose that would add more than one frame of coverage is also rejected.
    _settle(session, shifted)
    assert session._tracked is True and session._pose is not None
    anchor = next(tile for tile in session.tiles if tile.tile_id == session._anchor_id)
    scaled = np.array([[0.3, 0.0, 0.0], [0.0, 0.3, 0.0]], dtype=np.float64)
    # M_to_ref = anchor.M_to_ref @ inv(pose.M); small pose scale => large tile in chain space.
    session._pose = replace(session._pose, M=_compose_m(session._pose.M, scaled))
    rejected_scale = session.capture()
    assert rejected_scale.ok is False
    assert "退回" in rejected_scale.reason
    assert len(session.tiles) == 1
    assert session.coverage_memory.area == remembered
