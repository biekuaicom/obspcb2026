"""C-region green-anchor stitching: strip-only votes, four-point lock, 田 joint adjust."""

import cv2
import numpy as np

from vision.board_lock import joint_align_c_region, refine_pose_graph
from vision.elements import (
    ACCEPT_RESIDUAL_PX,
    BoardElement,
    BoardElementSet,
    align_red_region,
    estimate_typed_pose,
    similarity_from_c_anchors,
    vote_c_region_similarities,
)
from vision.stable_capture import StableCaptureSession


def _settle(session, frame):
    session.observe(frame)
    return session.observe(frame)


def _element(kind, xy, area=80.0, circularity=0.92, aspect=1.0):
    x, y = float(xy[0]), float(xy[1])
    return BoardElement(
        element_id=f"{kind}-{x:.1f}-{y:.1f}",
        kind=kind,
        contour=np.array([[x - 3, y - 3], [x + 3, y - 3], [x + 3, y + 3], [x - 3, y + 3]], dtype=np.float64),
        bbox=(int(x - 4), int(y - 4), 8, 8),
        centroid=(x, y),
        descriptor=np.array([area, circularity, aspect, 1.0, 0.0], dtype=np.float64),
        crop=np.zeros((8, 8, 3), np.uint8),
        area=area,
        circularity=circularity,
        aspect=aspect,
    )


def _element_set(width, height, items):
    return BoardElementSet(elements=list(items), width=int(width), height=int(height), scale=1.0)


def _decoy_and_four_anchor_sets(width=240, height=200, shift=192):
    """Four unique C-region holes plus a lookalike pad lattice outside C.

    The lattice is copied at the same local coordinates in both tiles, so a
    whole-image pad vote prefers tx=0. The true overlap is the right strip of
    A and the left strip of B, with shift 192 (20% overlap).
    """
    overlap = width - shift
    assert 0.18 <= overlap / width <= 0.22
    decoy = []
    for x in (70, 100, 130, 160):
        for y in range(30, 180, 20):
            decoy.append(_element("pad", (x, y), area=70.0, circularity=0.55, aspect=1.6))
    anchors_a = ((200, 36), (210, 80), (218, 132), (228, 176))
    reference = _element_set(width, height, decoy + [_element("hole", xy, area=90.0, circularity=0.95) for xy in anchors_a])
    query_decoy = [_element("pad", (x, y), area=70.0, circularity=0.55, aspect=1.6) for x, y in ((e.centroid[0], e.centroid[1]) for e in decoy)]
    anchors_b = [(xy[0] - shift, xy[1]) for xy in anchors_a]
    query = _element_set(width, height, query_decoy + [_element("hole", xy, area=90.0, circularity=0.95) for xy in anchors_b])
    return reference, query, shift, anchors_a, anchors_b


def test_c_region_matching_ignores_lookalike_pads_outside_overlap():
    """Only C-region anchors may lock the pose. Outside lookalike pads vote tx=0."""
    reference, query, shift, anchors_a, anchors_b = _decoy_and_four_anchor_sets()
    votes = vote_c_region_similarities(reference, query)
    assert any(abs(float(matrix[0, 2]) + shift) < 8.0 and abs(float(matrix[1, 2])) < 8.0 for matrix in votes)
    result = estimate_typed_pose(reference, query)
    assert result.ok
    assert result.primary_hits >= 4
    assert result.residual <= ACCEPT_RESIDUAL_PX
    assert abs(float(result.matrix[0, 2]) + shift) < 3.0
    assert abs(float(result.matrix[1, 2])) < 3.0
    # The identity preferred by non-C pads is not an accepting lock.
    wrong = align_red_region(
        reference,
        query,
        initial_pose=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
    )
    assert wrong.status in {"lattice_only", "disagree", "lost"}
    assert wrong.ok is False


def test_four_point_c_region_lock_like_figure_one():
    """Figure 1: four green anchors in a vertical C strip recover a similarity."""
    src = np.array([[186.0, 28.0], [198.0, 70.0], [190.0, 126.0], [208.0, 172.0]], dtype=np.float64)
    angle = np.radians(6.0)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    dst = src @ rotation.T + np.array([-48.0, 5.0])
    matrix = similarity_from_c_anchors(src, dst)
    assert matrix is not None
    recovered = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    assert abs(recovered - 6.0) < 0.5
    mapped = src @ matrix[:, :2].T + matrix[:, 2]
    assert float(np.max(np.linalg.norm(mapped - dst, axis=1))) < 0.6

    reference = _element_set(240, 200, [_element("hole", xy, area=90.0, circularity=0.95) for xy in src])
    query = _element_set(240, 200, [_element("hole", xy, area=90.0, circularity=0.95) for xy in dst])
    result = estimate_typed_pose(reference, query)
    assert result.ok
    assert result.primary_hits == 4
    assert result.residual <= ACCEPT_RESIDUAL_PX
    angle_hat = float(np.degrees(np.arctan2(result.matrix[1, 0], result.matrix[0, 0])))
    assert abs(angle_hat - 6.0) < 1.5
    assert abs(float(result.matrix[0, 2]) + 48.0) < 4.0


def _cross_constraints():
    """Typed C-region anchors for a 200px 田 with 40px (20%) strips and a center crossing."""
    # A top-left, B top-right, D bottom-left, E bottom-right. Origins: (0,0), (160,0), (0,160), (160,160).
    ab = [
        ((170.0, 40.0), (10.0, 40.0)),
        ((185.0, 90.0), (25.0, 90.0)),
        ((195.0, 130.0), (35.0, 130.0)),
        ((180.0, 55.0), (20.0, 55.0)),
    ]
    ad = [
        ((40.0, 170.0), (40.0, 10.0)),
        ((90.0, 185.0), (90.0, 25.0)),
        ((130.0, 195.0), (130.0, 35.0)),
        ((55.0, 180.0), (55.0, 20.0)),
    ]
    be = [
        ((70.0, 170.0), (70.0, 10.0)),
        ((120.0, 185.0), (120.0, 25.0)),
        ((150.0, 195.0), (150.0, 35.0)),
        ((95.0, 180.0), (95.0, 20.0)),
    ]
    de = [
        ((170.0, 70.0), (10.0, 70.0)),
        ((185.0, 120.0), (25.0, 120.0)),
        ((195.0, 150.0), (35.0, 150.0)),
        ((180.0, 95.0), (20.0, 95.0)),
    ]
    # Center crossing visible in all four tiles (A local, then B/D/E locals).
    center_a = [(170.0, 170.0), (190.0, 180.0), (180.0, 195.0), (198.0, 168.0)]

    def _pairs(left, right):
        src = np.array([item[0] for item in left], dtype=np.float64)
        dst = np.array([item[1] for item in right], dtype=np.float64)
        return src, dst

    constraints = [
        (0, 1) + _pairs(ab, ab),
        (0, 2) + _pairs(ad, ad),
        (1, 3) + _pairs(be, be),
        (2, 3) + _pairs(de, de),
    ]
    # Diagonal A-E and B-D through the same crossing points.
    center_b = [(x - 160.0, y) for x, y in center_a]
    center_d = [(x, y - 160.0) for x, y in center_a]
    center_e = [(x - 160.0, y - 160.0) for x, y in center_a]
    constraints.append((0, 3, np.array(center_a), np.array(center_e)))
    constraints.append((1, 2, np.array(center_b), np.array(center_d)))
    return constraints


def test_tian_grid_joint_lock_prevents_chain_drift():
    """Figure 2: joint C-region constraints pull a drifted 田 back; chaining does not."""
    truth = [
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        np.array([[1.0, 0.0, 160.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 160.0]], dtype=np.float64),
        np.array([[1.0, 0.0, 160.0], [0.0, 1.0, 160.0]], dtype=np.float64),
    ]
    # Simulated A→B→D→E chain drift: each step adds a little extra shift.
    drifted = [
        truth[0].copy(),
        np.array([[1.0, 0.0, 166.0], [0.0, 1.0, 1.5]], dtype=np.float64),
        np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 170.0]], dtype=np.float64),
        np.array([[1.0, 0.0, 168.0], [0.0, 1.0, 172.0]], dtype=np.float64),
    ]
    constraints = _cross_constraints()
    # Chain-only A→B, B→E, A→D leaves the accumulated error on E much larger.
    chain_only = refine_pose_graph(drifted, [constraints[0], constraints[1], constraints[2]], iterations=8)
    joint = joint_align_c_region(drifted, constraints)
    assert abs(float(joint[0][0, 2])) < 1e-6 and abs(float(joint[0][1, 2])) < 1e-6

    def _err(poses):
        return max(
            float(np.hypot(pose[0, 2] - truth[index][0, 2], pose[1, 2] - truth[index][1, 2]))
            for index, pose in enumerate(poses)
        )

    before = _err(drifted)
    after_chain = _err(chain_only)
    after_joint = _err(joint)
    assert before > 10.0
    assert after_joint < 2.5
    assert after_joint < after_chain / 2.0
    assert abs(float(joint[3][0, 2]) - 160.0) < 1.5
    assert abs(float(joint[3][1, 2]) - 160.0) < 1.5
    assert abs(float(joint[2][1, 2]) - 160.0) < 1.5


def _stroke_glyph(image, x, y, color=(250, 250, 250)):
    cv2.line(image, (x, y), (x, y + 12), color, 1)
    cv2.line(image, (x, y + 12), (x + 8, y + 12), color, 1)
    cv2.line(image, (x + 8, y), (x + 8, y + 12), color, 1)


def _tian_board():
    """Painted 田 with unique holes and silkscreen only in the cross C region."""
    tile_w, tile_h, overlap = 200, 200, 40
    board_w = tile_w + (tile_w - overlap)
    board_h = tile_h + (tile_h - overlap)
    image = np.full((board_h + 20, board_w + 20, 3), (40, 140, 60), np.uint8)
    rng = np.random.default_rng(11)
    noise = rng.integers(-3, 4, size=image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    origin = 10
    # Large rectangular pads in unique corners, kept out of the 40px cross.
    for x, y in ((40, 40), (90, 80), (55, 120), (270, 45), (320, 85), (300, 125), (40, 260), (90, 305), (55, 335), (270, 260), (320, 305), (300, 335)):
        cv2.rectangle(image, (origin + x - 10, origin + y - 5), (origin + x + 10, origin + y + 5), (186, 194, 206), -1)
    # Anchors stay 10px inside every tile that should see them (overlap is 40px).
    anchors = [
        (172, 36), (186, 58), (170, 88), (184, 118), (176, 148),
        (36, 172), (58, 186), (88, 170), (118, 184), (148, 176),
        (172, 228), (186, 258), (170, 288), (184, 318), (176, 338),
        (228, 172), (258, 186), (288, 170), (318, 184), (338, 176),
        (172, 172), (186, 180), (178, 188), (188, 170),
    ]
    for x, y in anchors:
        cv2.circle(image, (origin + x, origin + y), 4, (16, 16, 16), -1)
    _stroke_glyph(image, origin + 174, origin + 70)
    _stroke_glyph(image, origin + 70, origin + 174)
    _stroke_glyph(image, origin + 174, origin + 174)
    _stroke_glyph(image, origin + 174, origin + 250)
    _stroke_glyph(image, origin + 250, origin + 174)
    crops = {
        "A": image[origin : origin + tile_h, origin : origin + tile_w],
        "B": image[origin : origin + tile_h, origin + 160 : origin + 160 + tile_w],
        "D": image[origin + 160 : origin + 160 + tile_h, origin : origin + tile_w],
        "E": image[origin + 160 : origin + 160 + tile_h, origin + 160 : origin + 160 + tile_w],
    }
    return crops


def test_tian_session_jointly_locks_and_rejects_c_disagreement():
    """Production capture follows the 田 C-region path and refuses a disagreeing C strip."""
    crops = _tian_board()
    session = StableCaptureSession(min_overlap=0.08)
    session.start()
    for key in ("A", "B", "D", "E"):
        _settle(session, crops[key])
        decision = session.capture()
        assert decision.ok is True, f"{key}: {decision.reason}"
    assert len(session.tiles) == 4
    # Cross / loop constraints exist; placement is not a pure A→B→D→E chain.
    looped = [edge for edge in session.edges if edge["direction"] == "loop_closure"]
    assert looped
    pose_e = session.tiles[3].M_to_ref
    assert abs(float(pose_e[0, 2]) - 160.0) < 6.0
    assert abs(float(pose_e[1, 2]) - 160.0) < 6.0
    assert abs(float(session.tiles[1].M_to_ref[0, 2]) - 160.0) < 6.0
    assert abs(float(session.tiles[2].M_to_ref[1, 2]) - 160.0) < 6.0

    # A drifted 田 is pulled back by the shared C-region anchors.
    session.tiles[1].M_to_ref = session.tiles[1].M_to_ref.copy()
    session.tiles[2].M_to_ref = session.tiles[2].M_to_ref.copy()
    session.tiles[3].M_to_ref = session.tiles[3].M_to_ref.copy()
    session.tiles[1].M_to_ref[0, 2] += 7.0
    session.tiles[2].M_to_ref[1, 2] += 8.0
    session.tiles[3].M_to_ref[0, 2] += 7.0
    session.tiles[3].M_to_ref[1, 2] += 8.0
    session._close_loops(session.tiles[-1])
    assert abs(float(session.tiles[3].M_to_ref[0, 2]) - 160.0) < 3.0
    assert abs(float(session.tiles[3].M_to_ref[1, 2]) - 160.0) < 3.0
    assert abs(float(session.tiles[2].M_to_ref[1, 2]) - 160.0) < 3.0


def test_c_region_disagreement_is_rejected():
    """When C-region green anchors disagree, the still is refused and not blended."""
    width, height, shift = 240, 200, 188
    board = np.full((height + 20, width + shift + 20, 3), (40, 140, 60), np.uint8)
    rng = np.random.default_rng(4)
    # Irregular pads across both tiles, including the vertical C strip.
    for row, y in enumerate(range(24, height - 16, 22)):
        for col, x in enumerate(range(20, width + shift - 16, 26)):
            px = int(np.clip(x + int(rng.integers(-4, 5)), 8, board.shape[1] - 8))
            py = int(np.clip(y + int(rng.integers(-3, 4)), 8, height - 8))
            cv2.rectangle(board, (px - 6, py - 2), (px + 6, py + 2), (186, 194, 206), -1)
    # Four silkscreen marks living only in the overlap C strip (board x ≈ 200..240).
    glyphs = ((204, 40), (218, 70), (208, 120), (226, 160))
    for x, y in glyphs:
        cv2.putText(board, "T", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 2, cv2.LINE_AA)
    first = board[0:height, 20 : 20 + width]
    second = board[0:height, 20 + shift : 20 + shift + width].copy()
    # Shift only the C-region glyphs in the second tile; pads stay put.
    second[20:180, 0:50] = (40, 140, 60)
    for col, x in enumerate(range(20, 50, 26)):
        for y in range(24, height - 16, 22):
            cv2.rectangle(second, (x - 6, y - 2), (x + 6, y + 2), (186, 194, 206), -1)
    for x, y in glyphs:
        cv2.putText(second, "T", (x - shift + 20, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 2, cv2.LINE_AA)
    session = StableCaptureSession(min_overlap=0.08)
    session.start()
    _settle(session, first)
    assert session.capture().ok is True
    _settle(session, second)
    decision = session.capture()
    assert decision.ok is False
    assert decision.reason == "对位不一致，请重拍，或退回已拍区域"
    assert len(session.tiles) == 1
    assert session.overview_meta()["full_canvas"] is False
