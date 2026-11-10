"""Registration contract and projection visibility. Trackers stay stubs."""

import numpy as np
import pytest

from model.schema import BoardProject
from model.store import ProjectStore
from vision.affine_tracker import AffineResult, PCBAffineTracker, identity_affine
from vision.local_global import HomographyResult, LocalGlobalTracker, identity_homography


def _frame():
    return np.zeros((24, 32, 3), dtype=np.uint8)


def _project() -> BoardProject:
    return BoardProject.from_dict(
        {
            "board_id": "VIS",
            "global_map_path": "assets/maps/VIS.png",
            "components": [
                {
                    "component_id": "IN",
                    "base_pixel_box": [0, 0, 10, 10],
                    "status": "unchecked",
                    "core_params": {"name": "IN"},
                },
                {
                    "component_id": "OUT",
                    "base_pixel_box": [100, 100, 10, 10],
                    "status": "checked",
                    "core_params": {"name": "OUT"},
                },
            ],
        }
    )


def test_track_fail_closed_identity():
    affine = PCBAffineTracker(roi_size=(200, 200))
    affine.set_templates([{"id": "F1", "path": "assets/templates/F1.png", "global_pixel": (1, 2)}])
    result = affine.track(_frame())
    assert result.ok is False
    assert result.confidence == 0.0
    assert result.M.dtype == np.float64
    np.testing.assert_allclose(result.M, identity_affine())
    assert "桩" in result.message
    assert len(affine.templates) == 1

    empty = affine.track(np.zeros((0, 0, 3), dtype=np.uint8))
    assert empty.ok is False
    assert "空帧" in empty.message

    affine.reset()
    assert affine.last_result is None
    assert affine.templates

    local = LocalGlobalTracker()
    local.set_global_map(np.zeros((30, 40, 3), dtype=np.uint8))
    assert local.map_shape == (30, 40, 3)
    homography = local.track(_frame())
    assert homography.ok is False
    assert homography.inliers == 0
    assert homography.H.dtype == np.float64
    np.testing.assert_allclose(homography.H, identity_homography())
    assert "桩" in homography.message


def test_success_matrix_projects_global_box_into_frame():
    """A caller-supplied success result moves global pixels into the frame.

    ``track`` does not produce this result yet. The direction is global → frame.
    """
    matrix = identity_homography()
    matrix[0, 2] = 15.0
    matrix[1, 2] = -4.0
    result = HomographyResult(H=matrix, inliers=12, confidence=0.9, ok=True, message="injected")
    assert result.ok is True
    corners = LocalGlobalTracker().project_boxes([(0.0, 0.0, 10.0, 8.0)], result.H)
    assert len(corners) == 1
    assert corners[0].shape == (4, 2)
    np.testing.assert_allclose(corners[0][0], (15.0, -4.0))
    np.testing.assert_allclose(corners[0][1], (25.0, -4.0))
    np.testing.assert_allclose(corners[0][2], (25.0, 4.0))
    np.testing.assert_allclose(corners[0][3], (15.0, 4.0))

    affine = AffineResult(
        M=np.array([[1.0, 0.0, 15.0], [0.0, 1.0, -4.0]], dtype=np.float64),
        confidence=1.0,
        ok=True,
        message="injected",
    )
    assert affine.M.shape == (2, 3)


def test_project_boxes_rejects_bad_homography():
    with pytest.raises(ValueError):
        LocalGlobalTracker().project_boxes([(0, 0, 1, 1)], np.eye(2))


def test_list_visible_filters_projected_centers():
    store = ProjectStore()
    store.project = _project()
    viewport = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
    visible = store.list_visible(viewport, identity_homography())
    assert [item.component_id for item in visible] == ["IN"]

    # OUT's center is (105, 105). Shift by (-100, -100) lands it at (5, 5).
    shift = identity_homography()
    shift[0, 2] = -100.0
    shift[1, 2] = -100.0
    moved = store.list_visible(viewport, shift)
    assert [item.component_id for item in moved] == ["OUT"]

    affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    via_affine = store.list_visible(viewport, affine)
    assert [item.component_id for item in via_affine] == ["IN"]

    assert store.list_visible([], identity_homography()) == []
    assert store.list_visible(viewport, None) == []


def test_list_visible_includes_center_on_the_edge():
    store = ProjectStore()
    store.project = _project()
    # IN center is (5, 5), on the left edge of this polygon.
    edge = [(5.0, 0.0), (5.0, 10.0), (15.0, 10.0), (15.0, 0.0)]
    visible = store.list_visible(edge, identity_homography())
    assert "IN" in [item.component_id for item in visible]
