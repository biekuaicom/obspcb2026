"""JSON shape, status promotion, and store round-trip."""

import json
from pathlib import Path

import numpy as np
import pytest

from calib.xyz import screen_position_px
from model.schema import STATUS_COLORS, BoardProject, SchemaError, validate_project
from model.store import ProjectStore
from vision.affine_tracker import PCBAffineTracker
from vision.local_global import LocalGlobalTracker

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "assets" / "projects" / "DEMO-PCB-001.json"


def test_status_colors_match_data_model():
    assert STATUS_COLORS == {
        "unchecked": "#E53935",
        "checked": "#43A047",
        "doubt": "#FDD835",
    }


def test_sample_project_json_roundtrip():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    project = BoardProject.from_dict(raw)
    assert project.to_dict() == raw
    again = BoardProject.from_json(project.to_json())
    assert again.to_dict() == raw
    component = again.components[0]
    assert component.component_id == "R1"
    assert component.status == "unchecked"
    assert "status" not in component.core_params
    assert component.core_params == {"name": "R1", "value": "10K", "package": "0805"}


def test_store_save_load_and_patch(tmp_path: Path):
    store = ProjectStore()
    store.load(str(SAMPLE))
    store.update_component("R1", core={"value": "4.7K"}, status="checked")
    component = store.project.components[0]
    assert component.status == "checked"
    assert component.core_params["value"] == "4.7K"
    assert component.core_params["name"] == "R1"
    assert len(component.core_params) <= 3

    target = tmp_path / "board.json"
    store.save(store.project, str(target))
    reloaded = ProjectStore()
    project = reloaded.load(str(target))
    assert project.components[0].status == "checked"
    assert project.components[0].core_params["value"] == "4.7K"
    assert project.components[0].extended_params["sop_note"] == "注意焊接温度"


def test_legacy_core_status_is_promoted():
    raw = {
        "board_id": "LEGACY",
        "global_map_path": "assets/maps/LEGACY.png",
        "components": [
            {
                "component_id": "R1",
                "base_pixel_box": [100, 150, 60, 30],
                "core_params": {"name": "R1", "value": "10K", "status": "checked"},
                "extended_params": {"package": "0805"},
            }
        ],
    }
    errors = validate_project(raw)
    assert any("top level" in message for message in errors)

    project = BoardProject.from_dict(raw)
    component = project.components[0]
    assert component.status == "checked"
    assert component.absolute_pos_mm is None
    assert component.core_params == {"name": "R1", "value": "10K", "package": ""}
    assert project.schema_version == "0.1"


def test_unknown_status_is_treated_as_unchecked():
    raw = {
        "board_id": "X",
        "global_map_path": "assets/maps/X.png",
        "components": [
            {
                "component_id": "C1",
                "base_pixel_box": [0, 0, 2, 2],
                "status": "maybe",
                "core_params": {"name": "C1"},
            }
        ],
    }
    assert any("status" in message for message in validate_project(raw))
    project = BoardProject.from_dict(raw)
    assert project.components[0].status == "unchecked"


def test_reject_too_many_core_keys_and_bad_box():
    raw = {
        "board_id": "X",
        "global_map_path": "assets/maps/X.png",
        "components": [
            {
                "component_id": "C1",
                "base_pixel_box": [0, 0, 0, 2],
                "status": "unchecked",
                "core_params": {"a": "1", "b": "2", "c": "3", "d": "4"},
            }
        ],
    }
    with pytest.raises(SchemaError):
        BoardProject.from_dict(raw)


def test_path_escape_is_rejected(tmp_path: Path):
    from model.schema import resolve_project_relative

    repo = tmp_path / "repo"
    project = repo / "assets" / "projects" / "X.json"
    project.parent.mkdir(parents=True)
    project.write_text("{}", encoding="utf-8")
    inside = resolve_project_relative(project, "../maps/X.png", repo)
    assert inside == (repo / "assets" / "maps" / "X.png").resolve()
    with pytest.raises(ValueError, match="escapes"):
        resolve_project_relative(project, "../../../secret.png", repo)


def test_stub_trackers_import_and_fail_closed():
    affine = PCBAffineTracker()
    local = LocalGlobalTracker()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    affine_result = affine.track(frame)
    local_result = local.track(frame)
    assert affine_result.ok is False
    assert local_result.ok is False
    assert affine_result.M.shape == (2, 3)
    assert local_result.H.shape == (3, 3)
    assert affine_result.message
    assert local_result.message


def test_pure_pixel_board_uses_exact_bubble_keys():
    project = BoardProject.from_dict(
        {
            "board_id": "PX",
            "global_map_path": "assets/maps/PX.png",
            "components": [
                {
                    "component_id": "R2",
                    "base_pixel_box": [1, 2, 3, 4],
                    "status": "unchecked",
                    "core_params": {"name": "R2"},
                }
            ],
        }
    )
    assert project.pixel_per_mm is None
    component = project.components[0]
    assert component.absolute_pos_mm is None
    assert component.core_params == {"name": "R2", "value": "", "package": ""}
    from app.ui.main_window import missing_mm_warning

    assert "纯像素" in missing_mm_warning(project)
    with pytest.raises(SchemaError):
        BoardProject.from_dict(
            {
                "board_id": "PX",
                "global_map_path": "assets/maps/PX.png",
                "components": [
                    {
                        "component_id": "R2",
                        "base_pixel_box": [1, 2, 3, 4],
                        "status": "unchecked",
                        "core_params": {"name": "R2", "footprint": "0805"},
                    }
                ],
            }
        )


def test_policy_constants_match_confirmed_decisions():
    from vision.policy import (
        FALLBACK_TRACKER,
        PRIMARY_TRACKER,
        TRACKING_LOST_TTL_S,
        TRANSFORM_DIRECTION,
    )

    assert TRANSFORM_DIRECTION == "global_to_frame"
    assert PRIMARY_TRACKER == "local_global"
    assert FALLBACK_TRACKER == "affine"
    assert TRACKING_LOST_TTL_S == 0.8


def test_xyz_calibration_is_not_implemented():
    with pytest.raises(NotImplementedError):
        screen_position_px((0, 0), (0, 0), 1.0)
