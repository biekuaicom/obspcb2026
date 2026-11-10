"""VideoMosaicBuilder: state machine, skip weak frames, project write-out."""

import json

import cv2
import numpy as np

from model.schema import BoardProject
from vision.mosaic import (
    MAP_SOURCE,
    MappingController,
    MappingState,
    VideoMosaicBuilder,
    make_synthetic_translation_frames,
)


def test_mapping_state_machine_ready_fail_cancel_rescan():
    ready = MappingController()
    assert ready.start_recording() is MappingState.RECORDING
    assert ready.begin_building() is MappingState.BUILDING
    assert ready.mark_ready() is MappingState.MOSAIC_READY
    assert ready.request_rescan() is MappingState.RESCAN
    assert ready.start_recording() is MappingState.RECORDING

    failed = MappingController()
    failed.start_recording()
    assert failed.fail("no lock") is MappingState.FAILED
    assert failed.request_rescan() is MappingState.RESCAN

    cancelled = MappingController()
    cancelled.start_recording()
    assert cancelled.cancel() is MappingState.CANCELLED
    assert cancelled.request_rescan() is MappingState.RESCAN
    cancelled.reset()
    assert cancelled.state is MappingState.IDLE


def test_builder_skips_noise_and_writes_project(tmp_path):
    frames = make_synthetic_translation_frames(count=8, bad_index=3)
    builder = VideoMosaicBuilder(overlap_hint=0.85, motion="feature")
    assert builder.controller.state is MappingState.IDLE
    statuses = [builder.push_frame(frame) for frame in frames]
    assert builder.controller.state is MappingState.RECORDING
    assert statuses[0].accepted is True
    assert statuses[3].accepted is False
    out = tmp_path / "assets" / "maps" / "VIDEO-MAP-001.png"
    result = builder.finalize(out, board_id="VIDEO-MAP-001", repo_root=tmp_path)
    assert result.ok is True
    assert result.state == MappingState.MOSAIC_READY.value
    assert result.frames_total == 8
    assert result.frames_accepted >= 6
    assert result.frames_skipped >= 1
    assert result.canvas_size[0] > frames[0].shape[1]
    assert result.map_source == MAP_SOURCE
    image = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape[1] == result.canvas_size[0]
    assert image.shape[0] == result.canvas_size[1]
    project = BoardProject.from_json((tmp_path / "assets" / "projects" / "VIDEO-MAP-001.json").read_text())
    assert project.map_source == "video_mosaic"
    assert project.global_map_path.endswith("maps/VIDEO-MAP-001.png")
    mapping = project.meta["mapping"]
    assert mapping["frames_used"] == result.frames_accepted
    assert mapping["canvas_size"] == [result.canvas_size[0], result.canvas_size[1]]
    assert mapping["state"] == "MosaicReady"
    assert mapping["motion"] == "feature"


def test_optical_flow_accepts_translation_and_skips_noise():
    frames = make_synthetic_translation_frames(count=6, bad_index=2)
    builder = VideoMosaicBuilder(overlap_hint=0.85, motion="optical_flow")
    for frame in frames:
        builder.push_frame(frame)
    assert builder.frames[0].accepted is True
    assert builder.frames[2].accepted is False
    assert sum(1 for item in builder.frames if item.accepted) >= 4


def test_cancel_blocks_finalize(tmp_path):
    frame = make_synthetic_translation_frames(count=1, bad_index=None)[0]
    builder = VideoMosaicBuilder()
    builder.push_frame(frame)
    assert builder.cancel() is MappingState.CANCELLED
    result = builder.finalize(tmp_path / "mosaic.png")
    assert result.ok is False
    assert result.state == MappingState.CANCELLED.value
    assert not (tmp_path / "mosaic.png").exists()


def test_serpentine_plan_reverses_rows_and_keeps_overlap():
    from vision.mosaic import plan_serpentine_scan

    scan = plan_serpentine_scan(200, 160, overlap=0.85, window_size=(80, 60))
    assert scan.pattern == "serpentine"
    assert scan.frame_count == scan.rows * scan.cols
    assert scan.rows >= 2 and scan.cols >= 2
    assert 0.80 <= scan.overlap_x <= 0.95
    assert 0.80 <= scan.overlap_y <= 0.95
    row0 = [origin for origin in scan.origins if origin[1] == scan.origins[0][1]]
    row1_y = next(origin[1] for origin in scan.origins if origin[1] != scan.origins[0][1])
    row1 = [origin for origin in scan.origins if origin[1] == row1_y]
    assert [origin[0] for origin in row0] == sorted(origin[0] for origin in row0)
    assert [origin[0] for origin in row1] == sorted((origin[0] for origin in row1), reverse=True)
    assert row0[-1][0] == row1[0][0]
    payload = scan.to_dict()
    assert payload["step"]["dx"] == scan.step[0]
    assert payload["step"]["dy"] == scan.step[1]
    assert payload["window_size"] == [80, 60]


def test_serpentine_crops_rebuild_and_match_source(tmp_path):
    from vision.mosaic import crop_serpentine_frames, plan_serpentine_scan, run_from_global

    image = _textured_board(120, 160)
    source = tmp_path / "global_ref.png"
    assert cv2.imwrite(str(source), image)
    scan = plan_serpentine_scan(image.shape[1], image.shape[0], overlap=0.85, window_size=(80, 64))
    frames = crop_serpentine_frames(image, scan)
    assert len(frames) == scan.frame_count
    assert frames[0].shape[1] == 80 and frames[0].shape[0] == 64
    artifact_dir = tmp_path / "artifacts"
    result = run_from_global(
        source,
        tmp_path / "assets" / "maps" / "BOARD-SCAN.png",
        board_id="BOARD-SCAN",
        repo_root=tmp_path,
        overlap=0.85,
        window_size=(80, 64),
        artifact_dir=artifact_dir,
    )
    assert result.ok is True
    assert result.frames_accepted >= result.frames_total - 1
    assert result.scan["pattern"] == "serpentine"
    assert result.compare["ncc"] is not None and result.compare["ncc"] > 0.7
    project = BoardProject.from_json((tmp_path / "assets" / "projects" / "BOARD-SCAN.json").read_text())
    assert project.map_source == "video_mosaic"
    assert project.meta["mapping"]["scan"]["frame_count"] == scan.frame_count
    assert (artifact_dir / "summary.json").is_file()
    assert (artifact_dir / "overlay.png").is_file()


def _textured_board(height: int, width: int) -> np.ndarray:
    board = np.zeros((height, width, 3), dtype=np.uint8)
    board[:, :] = (36, 110, 48)
    rng = np.random.default_rng(5)
    for y in range(6, height - 6, 14):
        for x in range(6, width - 6, 16):
            color = tuple(int(value) for value in rng.integers(15, 245, size=3))
            cv2.rectangle(board, (x, y), (x + 9, y + 7), color, -1)
            cv2.circle(board, (x + 4, y + 3), 2, (250, 250, 250), -1)
    return board


def test_reset_returns_to_idle_and_rejects_unrelated_frame():
    frames = make_synthetic_translation_frames(count=2, bad_index=None)
    builder = VideoMosaicBuilder(motion="auto")
    builder.push_frame(frames[0])
    builder.reset()
    assert builder.controller.state is MappingState.IDLE
    assert builder.frames == []
    noise = np.random.default_rng(1).integers(0, 255, size=frames[0].shape, dtype=np.uint8)
    builder.push_frame(frames[0])
    skipped = builder.push_frame(noise)
    assert skipped.accepted is False
    progress = builder.progress()
    payload = json.loads(json.dumps(progress.to_dict()))
    assert payload["frames_total"] == 2
    assert payload["frames_skipped"] == 1
    assert payload["state"] == MappingState.RECORDING.value
