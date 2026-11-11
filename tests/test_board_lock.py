"""Geometric lock, modest overlap, and overlap that is not double-exposed."""

import cv2
import numpy as np

from vision.board_lock import SAFE_RESIDUAL_PX, composite_registered_tile, refine_pose_graph
from vision.stable_capture import StableCaptureSession


def _settle(session, frame):
    session.observe(frame)
    return session.observe(frame)


def _textured_board(height=320, width=820, seed=2):
    """Irregular pads plus one dense column so a thin overlap is still unique."""
    rng = np.random.default_rng(seed)
    image = np.full((height, width), 180, np.uint8)
    noise = rng.integers(-3, 4, size=image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for row, y in enumerate(range(36, height - 36, 22)):
        for col, x in enumerate(range(30, width - 30, 24)):
            px = int(np.clip(x + int(rng.integers(-4, 5)), 12, width - 12))
            py = int(np.clip(y + int(rng.integers(-4, 5)), 12, height - 12))
            if (row * 2 + col) % 4 == 0:
                cv2.circle(image, (px, py), 3, 20, -1)
            else:
                cv2.rectangle(image, (px - 6, py - 2), (px + 6, py + 2), 30, -1)
    # A self-contained column inside the right-hand 20% of a 200px window
    # that starts at x=20. Centers stay clear of both crop borders.
    for index, yy in enumerate(range(28, 150, 11)):
        cx = 176 if index % 2 == 0 else 190
        cv2.circle(image, (cx, yy), 4, 12, -1)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _regular_grid(height=300, width=700, step=18):
    image = np.full((height, width), 180, np.uint8)
    for y in range(30, height - 30, step):
        for x in range(30, width - 30, step):
            cv2.circle(image, (x, y), 3, 20, -1)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_modest_overlap_locks_without_a_two_thirds_red_region():
    image = _textured_board()
    height, width = 160, 200
    origin_y, origin_x = 24, 20
    shift = 160
    true_overlap = (width - shift) / width
    assert 0.10 <= true_overlap <= 0.20
    first = image[origin_y : origin_y + height, origin_x : origin_x + width]
    second = image[origin_y : origin_y + height, origin_x + shift : origin_x + shift + width]
    session = StableCaptureSession(min_overlap=0.08)
    session.start()
    _settle(session, first)
    assert session.capture().ok is True
    state = _settle(session, second)
    decision = session.capture()
    assert decision.ok is True
    assert state.tracked is True
    assert state.overlap is not None
    assert 0.10 <= state.overlap <= 0.25
    assert state.overlap < 0.66
    assert "不确定" not in state.message
    tx = float(session.tiles[-1].M_to_ref[0, 2])
    assert abs(tx - shift) < 4.0
    assert session.overview_meta()["full_canvas"] is False


def test_repetitive_lattice_asks_to_slow_down_instead_of_hiding_a_two_thirds_rule():
    image = _regular_grid()
    height, width = 150, 200
    origin_y, origin_x = 30, 20
    shift = 120
    true_overlap = (width - shift) / width
    assert true_overlap < 0.5
    first = image[origin_y : origin_y + height, origin_x : origin_x + width]
    second = image[origin_y : origin_y + height, origin_x + shift : origin_x + shift + width]
    session = StableCaptureSession(min_overlap=0.08)
    session.start()
    _settle(session, first)
    assert session.capture().ok is True
    state = _settle(session, second)
    decision = session.capture()
    assert decision.ok is False
    assert "不确定" in state.message
    assert "不确定" in decision.reason
    assert state.overlap is None or state.overlap < 0.66
    assert len(session.tiles) == 1


def test_misaligned_overlap_is_not_averaged():
    canvas = np.full((48, 90, 3), 180, np.uint8)
    canvas[:, 20, :] = 10
    mask = np.full(canvas.shape[:2], 255, np.uint8)
    residual = np.full(canvas.shape[:2], 0.4, np.float32)
    sharpness = np.full(canvas.shape[:2], 40.0, np.float32)
    incoming = np.full_like(canvas, 180)
    incoming[:, 26, :] = 10
    composite_registered_tile(
        canvas,
        mask,
        residual,
        sharpness,
        incoming,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        residual_px=SAFE_RESIDUAL_PX + 3.0,
        sharpness=80.0,
    )
    averaged = (np.full_like(canvas, 180).astype(np.float32) * 0.5 + incoming.astype(np.float32) * 0.5)
    averaged[:, 20, :] = 10 * 0.5 + 180 * 0.5

    def dark_columns(image, level=120):
        gray = image[:, :, 0]
        return [index for index in range(gray.shape[1]) if float(gray[:, index].mean()) < level]

    assert dark_columns(canvas) == [20]
    assert 20 in dark_columns(averaged.astype(np.uint8)) and 26 in dark_columns(averaged.astype(np.uint8))
    assert abs(int(canvas[12, 20, 0]) - 10) < 5
    assert abs(int(canvas[12, 26, 0]) - 180) < 5
    # The old feather blend would have pulled the kept pin toward the other exposure.
    assert abs(int(canvas[12, 20, 0]) - (0.45 * 10 + 0.55 * 180)) > 40

    fresh = np.full((48, 90, 3), 180, np.uint8)
    fresh[:, 30, :] = 200
    base = np.full_like(fresh, 40)
    base_mask = np.full(base.shape[:2], 255, np.uint8)
    base_residual = np.full(base.shape[:2], 1.5, np.float32)
    base_sharp = np.full(base.shape[:2], 10.0, np.float32)
    composite_registered_tile(
        base,
        base_mask,
        base_residual,
        base_sharp,
        fresh,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        residual_px=0.3,
        sharpness=50.0,
    )
    assert abs(int(base[8, 30, 0]) - 200) < 5
    assert abs(int(base[8, 30, 0]) - (40 + 200) / 2) > 40


def test_loop_closes_against_a_non_adjacent_tile():
    image = _textured_board(height=280, width=520)
    height, width = 150, 180
    y = 30
    views = [
        image[y : y + height, 20 : 20 + width],
        image[y : y + height, 50 : 50 + width],
        image[y : y + height, 36 : 36 + width],
    ]
    session = StableCaptureSession(min_overlap=0.08)
    session.start()
    for view in views:
        _settle(session, view)
        assert session.capture().ok is True
    closures = [edge for edge in session.edges if edge["direction"] == "loop_closure"]
    assert any(edge["src"] == "cap-000" and edge["dst"] == "cap-002" for edge in closures)
    assert session.edges[0]["direction"] == "previous_tile_to_current_frame"
    tx = float(session.tiles[2].M_to_ref[0, 2])
    assert abs(tx - 16.0) < 5.0
    assert session.topology_db(board_id="SYN", source="synthetic")["full_canvas"] is False


def test_pose_graph_corrects_a_drifted_earlier_tile():
    truth = 40.0
    drifted = 52.0
    poses = [
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        np.array([[1.0, 0.0, drifted], [0.0, 1.0, 0.0]], dtype=np.float64),
    ]
    src = np.array([[10, 12], [36, 18], [70, 40], [22, 66], [58, 72], [84, 30]], dtype=np.float64)
    dst = src - np.array([truth, 0.0])
    solved = refine_pose_graph(poses, [(0, 1, src, dst)])
    assert abs(float(solved[1][0, 2]) - truth) < 1.5
    assert abs(float(solved[0][0, 2])) < 1e-6
    before = abs(drifted - truth)
    after = abs(float(solved[1][0, 2]) - truth)
    assert after < before / 4.0


def test_annotations_live_in_board_coordinates_and_are_not_baked_into_tiles():
    image = _textured_board()
    height, width = 150, 180
    first = image[40 : 40 + height, 24 : 24 + width]
    second = image[40 : 40 + height, 48 : 48 + width]
    session = StableCaptureSession(min_overlap=0.08)
    session.start()
    _settle(session, first)
    assert session.capture().ok is True
    raw_first = session.tiles[0].raw.copy()
    session.add_annotation("U12", (42, 28), tile_id="cap-000")
    assert session.tiles[0].raw.tobytes() == raw_first.tobytes()
    _settle(session, second)
    assert session.capture().ok is True
    shift = float(session.tiles[1].M_to_ref[0, 2])
    raw_second = session.tiles[1].raw.copy()
    session.add_annotation("U12", (42 - shift, 28), tile_id="cap-001")
    assert session.tiles[1].raw.tobytes() == raw_second.tobytes()
    annotations = session.board_annotations()
    assert len(annotations) == 1
    assert annotations[0]["text"] == "U12"
    assert abs(annotations[0]["xy"][0] - 42) < 4.0
    assert abs(annotations[0]["xy"][1] - 28) < 4.0
    assert session.chain_dict()["annotations"][0]["text"] == "U12"
    blank = np.zeros((height, width, 3), np.uint8)
    drawn = session.draw_annotations(blank)
    green = (drawn[:, :, 1] > 200) & (drawn[:, :, 0] < 20) & (drawn[:, :, 2] < 20)
    assert int(green.sum()) > 8
    overview = session.overview_image()
    assert overview is not None
    baked = (overview[:, :, 1] > 250) & (overview[:, :, 0] < 5) & (overview[:, :, 2] < 5)
    assert int(baked.sum()) == 0
    again = session.draw_annotations(np.zeros_like(blank))
    again_green = (again[:, :, 1] > 200) & (again[:, :, 0] < 20) & (again[:, :, 2] < 20)
    assert int(again_green.sum()) == int(green.sum())
