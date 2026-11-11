"""Typed separation, red-region similarity, and seam rejection."""

import cv2
import numpy as np

from vision.elements import (
    ACCEPT_RESIDUAL_PX,
    PatchClassifier,
    align_red_region,
    extract_board_elements,
    recognize_element_crops,
)
from vision.stable_capture import StableCaptureSession, save_session


def _settle(session, frame):
    session.observe(frame)
    return session.observe(frame)


def _rich_board():
    """Pads, holes, silkscreen, traces, an outline, pins, and one color block.

    Features sit in separate bands so a 11 px kernel does not merge them.
    """
    rng = np.random.default_rng(3)
    image = np.full((420, 780, 3), (40, 140, 60), np.uint8)
    for row, y in enumerate(range(36, 210, 28)):
        for col, x in enumerate(range(36, 740, 34)):
            if 150 <= x <= 210 and 24 <= y <= 70:
                continue
            px = int(np.clip(x + int(rng.integers(-4, 5)), 16, 760))
            py = int(np.clip(y + int(rng.integers(-3, 4)), 16, 200))
            if (row + col * 2) % 5 == 0:
                cv2.circle(image, (px, py), 4, (16, 16, 16), -1)
            else:
                cv2.rectangle(image, (px - 8, py - 3), (px + 8, py + 3), (186, 194, 206), -1)
    cv2.rectangle(image, (148, 30), (196, 40), (205, 208, 216), -1)
    cv2.circle(image, (172, 35), 3, (6, 6, 6), -1)
    for index, glyph in enumerate("U12"):
        cv2.putText(
            image,
            glyph,
            (40 + index * 36, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
    cv2.line(image, (40, 290), (420, 294), (10, 10, 12), 2)
    cv2.line(image, (40, 320), (300, 324), (10, 10, 12), 2)
    cv2.rectangle(image, (470, 250), (610, 340), (235, 235, 235), 2)
    for x in range(486, 600, 20):
        cv2.rectangle(image, (x, 354), (x + 5, 362), (230, 230, 235), -1)
    cv2.rectangle(image, (640, 300), (740, 360), (0, 165, 255), -1)
    return image


def _textured_with_marks():
    """Irregular pads plus two silkscreen glyphs inside a ~20% overlap strip."""
    rng = np.random.default_rng(2)
    height, width = 320, 860
    gray = np.full((height, width), 180, np.uint8)
    noise = rng.integers(-3, 4, size=gray.shape, dtype=np.int16)
    gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for row, y in enumerate(range(36, height - 36, 22)):
        for col, x in enumerate(range(30, width - 30, 24)):
            px = int(np.clip(x + int(rng.integers(-4, 5)), 12, width - 12))
            py = int(np.clip(y + int(rng.integers(-4, 5)), 12, height - 12))
            if (row * 2 + col) % 4 == 0:
                cv2.circle(gray, (px, py), 3, 20, -1)
            else:
                cv2.rectangle(gray, (px - 6, py - 2), (px + 6, py + 2), 30, -1)
    for index, yy in enumerate(range(28, 150, 11)):
        cx = 176 if index % 2 == 0 else 190
        cv2.circle(gray, (cx, yy), 4, 12, -1)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Windows below are x=20..260 and x=208..448 (shift 188). Overlap is board x=208..260.
    _stroke_glyph(image, 216, 70)
    _stroke_glyph(image, 234, 70)
    return image


def _stroke_glyph(image, x, y, color=(250, 250, 250)):
    """Open stroke. Typed as text, and small enough not to steal the pad lock."""
    cv2.line(image, (x, y), (x, y + 12), color, 1)
    cv2.line(image, (x, y + 12), (x + 8, y + 12), color, 1)
    cv2.line(image, (x + 8, y), (x + 8, y + 12), color, 1)


class _RenamePads(PatchClassifier):
    def label(self, crop, kind, descriptor):
        return "pin" if kind == "pad" else kind


def test_typed_separation_on_a_synthetic_tile():
    elements = extract_board_elements(_rich_board())
    counts = elements.counts()
    assert counts["hole"] >= 4
    assert counts["pad"] >= 4
    assert counts["trace"] >= 2
    assert counts["text"] >= 2
    assert counts["outline"] >= 1
    assert counts["pin"] >= 3
    assert counts["color"] >= 1
    annular = [item for item in elements.of_kind("hole") if item.parent_id and item.parent_id.startswith("pad")]
    assert annular, "a drill inside a pad stays a hole parented to that pad"
    for pin in elements.of_kind("pin"):
        assert pin.parent_id and pin.parent_id.startswith("outline")
    for text in elements.of_kind("text"):
        assert text.kind != "trace"
        assert text.crop is not None and text.crop.size > 0
        assert text.descriptor.shape[0] >= 5
        assert len(text.bbox) == 4
    payload = elements.to_dict()
    assert payload["elements"]
    assert "crop_size" in payload["elements"][0]
    assert "crops" not in payload["elements"][0] or isinstance(payload["elements"][0]["crop_size"], list)


def test_similarity_alignment_on_a_red_overlap():
    image = _rich_board()
    origin_x, origin_y = 30, 40
    height, width = 180, 240
    shift_x, shift_y = 48, 10
    first = image[origin_y : origin_y + height, origin_x : origin_x + width]
    second = image[origin_y + shift_y : origin_y + shift_y + height, origin_x + shift_x : origin_x + shift_x + width]
    result = align_red_region(extract_board_elements(first), extract_board_elements(second))
    assert result.ok
    assert result.primary_hits >= 4
    assert result.residual <= ACCEPT_RESIDUAL_PX
    assert abs(float(result.matrix[0, 2]) + shift_x) < 3.0
    assert abs(float(result.matrix[1, 2]) + shift_y) < 3.0
    angle = float(np.degrees(np.arctan2(result.matrix[1, 0], result.matrix[0, 0])))
    assert abs(angle) < 2.0

    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    spin = cv2.getRotationMatrix2D(center, 4.0, 1.0)
    rotated = cv2.warpAffine(first, spin, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    spun = align_red_region(extract_board_elements(first), extract_board_elements(rotated))
    spun_angle = float(np.degrees(np.arctan2(spun.matrix[1, 0], spun.matrix[0, 0])))
    assert spun.ok
    assert abs(spun_angle + 4.0) < 2.0
    assert spun.residual <= ACCEPT_RESIDUAL_PX


def test_elements_that_disagree_are_rejected():
    image = _rich_board()
    first = image[60:320, 16:260]
    second = first.copy()
    # Glyphs are near y=250 on the board, y=190 in this window. Shift them down.
    second[160:230, 8:160] = (40, 140, 60)
    cv2.putText(second, "U", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(second, "1", (56, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(second, "2", (92, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
    pose = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    result = align_red_region(extract_board_elements(first), extract_board_elements(second), initial_pose=pose)
    assert result.status == "disagree"
    assert result.ok is False


def test_partial_outline_snaps_to_the_complete_one():
    image = _rich_board()
    # Outline is x=470..610, y=250..340. One view clips it; the other holds the whole rectangle.
    full = image[220:400, 430:680]
    clipped = image[220:400, 520:770]
    reference = extract_board_elements(full)
    query = extract_board_elements(clipped)
    assert any(not item.partial for item in reference.of_kind("outline"))
    assert any(item.partial for item in query.of_kind("outline"))
    result = align_red_region(reference, query)
    assert result.ok
    assert result.residual <= ACCEPT_RESIDUAL_PX
    assert abs(float(result.matrix[0, 2]) + 90.0) < 4.0
    assert result.snapped
    complete = next(item for item in reference.of_kind("outline") if not item.partial)
    partial = next(item for item in query.of_kind("outline") if item.partial)
    projected = complete.centroid[0] + float(result.matrix[0, 2])
    assert abs(projected - partial.centroid[0]) > ACCEPT_RESIDUAL_PX


def test_periodic_via_grid_alone_is_not_a_lock():
    grid = np.full((260, 640, 3), 180, np.uint8)
    for y in range(24, 240, 18):
        for x in range(24, 620, 18):
            cv2.circle(grid, (x, y), 3, (20, 20, 20), -1)
    first = grid[20:170, 16:216]
    second = grid[20:170, 52:252]
    result = align_red_region(extract_board_elements(first), extract_board_elements(second))
    assert result.status == "lattice_only"
    assert result.ok is False
    assert result.primary_hits >= 8
    assert result.secondary_hits == 0

    session = StableCaptureSession(min_overlap=0.08)
    session.start()
    _settle(session, first)
    assert session.capture().ok is True
    _settle(session, second)
    decision = session.capture()
    assert decision.ok is False
    assert len(session.tiles) == 1


def test_modest_overlap_with_agreeing_elements_still_locks(tmp_path):
    image = _textured_with_marks()
    height, width = 160, 240
    origin_y, origin_x = 24, 20
    shift = 188
    true_overlap = (width - shift) / width
    assert 0.18 <= true_overlap <= 0.24
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
    assert 0.15 <= state.overlap <= 0.28
    assert session.element_alignment is not None
    assert session.element_alignment.ok
    assert session.element_alignment.primary_hits >= 4
    assert session.element_alignment.secondary_hits >= 1
    assert abs(float(session.tiles[-1].M_to_ref[0, 2]) - shift) < 4.0
    assert session.overview_meta()["full_canvas"] is False
    chain = session.chain_dict()
    assert chain["tiles"][0]["elements"]["counts"]["text"] >= 1
    assert chain["tiles"][1]["elements"]["elements"][0]["crop_size"]
    summary = save_session(session, tmp_path / "scan", board_id="SYN", source="elements")
    assert summary["full_canvas"] is False
    crops = list((tmp_path / "scan" / "tiles" / "crops").rglob("*.png"))
    assert crops


def test_session_rejects_a_shifted_silkscreen_without_stitching():
    image = _textured_with_marks()
    height, width = 160, 240
    origin_y, origin_x = 24, 20
    shift = 188
    first = image[origin_y : origin_y + height, origin_x : origin_x + width]
    second = image[origin_y : origin_y + height, origin_x + shift : origin_x + shift + width].copy()
    # Glyphs sit at the left of this window. Cover only the strokes, then redraw them lower.
    second[40:66, 4:50] = (180, 180, 180)
    _stroke_glyph(second, 8, 90)
    _stroke_glyph(second, 26, 90)
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


def test_preview_does_not_run_element_separation(monkeypatch):
    import vision.stable_capture as capture_module

    calls = {"n": 0}
    real = capture_module.extract_board_elements

    def _count(frame, **kwargs):
        calls["n"] += 1
        return real(frame, **kwargs)

    monkeypatch.setattr(capture_module, "extract_board_elements", _count)
    frame = _textured_with_marks()[24:184, 20:260]
    session = StableCaptureSession(min_overlap=0.08)
    session.start()
    _settle(session, frame)
    assert calls["n"] == 0
    assert session.capture().ok is True
    assert calls["n"] == 1
    _settle(session, frame)
    assert calls["n"] == 1


def test_classifier_hook_relabels_crops_without_weights():
    elements = extract_board_elements(_rich_board())
    assert elements.of_kind("pad")
    recognize_element_crops(elements, _RenamePads())
    assert elements.of_kind("pad") == []
    assert elements.of_kind("pin")
    plain = extract_board_elements(_rich_board())
    assert plain.of_kind("pad")



def _paint_fiducial(image, xy, radius=36):
    center = (int(xy[0]), int(xy[1]))
    cv2.circle(image, center, int(radius), (40, 180, 220), -1)
    cv2.circle(image, center, max(4, int(radius * 0.32)), (18, 18, 18), -1)


def test_fiducial_pairs_recover_a_rotated_pose():
    from vision.elements import detect_fiducials, fiducial_correspondences, _fiducial_votes

    image = np.full((360, 520, 3), (40, 140, 60), np.uint8)
    _paint_fiducial(image, (120, 110), 36)
    _paint_fiducial(image, (380, 250), 38)
    first = detect_fiducials(image)
    assert len(first) == 2
    center = ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)
    spin = cv2.getRotationMatrix2D(center, 7.0, 1.0)
    rotated = cv2.warpAffine(
        image,
        spin,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(40, 140, 60),
    )
    second = detect_fiducials(rotated)
    assert len(second) == 2
    votes = _fiducial_votes(first, second)
    assert votes

    def _angle(matrix):
        return float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))

    assert any(abs(_angle(matrix) + 7.0) < 2.5 for matrix in votes)
    src, dst = fiducial_correspondences(first, second, spin)
    assert len(src) == 2
    assert len(dst) == 2


def test_detect_fiducials_ignores_ordinary_pads():
    from vision.elements import detect_fiducials

    image = np.full((240, 320, 3), (40, 140, 60), np.uint8)
    for x in range(30, 300, 24):
        cv2.circle(image, (x, 80), 5, (40, 180, 220), -1)
        cv2.circle(image, (x, 80), 2, (18, 18, 18), -1)
    assert detect_fiducials(image) == []
