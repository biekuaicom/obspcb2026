"""Tile import rules and fail-closed compare. Full 15-tile run is the CLI."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from model.schema import BoardProject
from vision.stitch import (
    GLOBAL_REF_SOURCE_NAME,
    SampleAssetImporter,
    SemiAutoStitchPipeline,
    TileSet,
    classify_sample_name,
    discover_raw_assets,
)


def test_names_not_sizes_decide_tile_versus_global(tmp_path: Path):
    assert classify_sample_name(GLOBAL_REF_SOURCE_NAME) == "global"
    assert classify_sample_name("20261109-184132-411.jpg") == "tile"
    assert classify_sample_name("20261109-184559-39.jpeg") == "tile"
    assert classify_sample_name("notes.jpg") == "ignore"
    assert classify_sample_name("global_ref.png") == "global"

    big_ignored = tmp_path / "poster.jpg"
    small_tile = tmp_path / "20261109-1.jpg"
    small_global = tmp_path / GLOBAL_REF_SOURCE_NAME
    _write(big_ignored, np.zeros((400, 400, 3), np.uint8))
    _write(small_tile, np.zeros((8, 8, 3), np.uint8))
    _write(small_global, np.zeros((9, 7, 3), np.uint8))
    tiles, global_ref = discover_raw_assets(tmp_path)
    assert [path.name for path in tiles] == ["20261109-1.jpg"]
    assert global_ref is not None and global_ref.name == GLOBAL_REF_SOURCE_NAME


def test_import_writes_samples_project_and_manifest(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw / "20261109-184132-411.jpg", _texture(80, 60))
    _write(raw / "20261109-184159-641.jpg", _texture(80, 60, shift=3))
    _write(raw / GLOBAL_REF_SOURCE_NAME, _texture(90, 120))
    _write(raw / "ignore-me.jpg", _texture(30, 30))
    repo = tmp_path / "repo"
    (repo / "assets" / "projects").mkdir(parents=True)
    importer = SampleAssetImporter(repo)
    tile_set = importer.import_board_dir(raw, "SAMPLE-PARTIAL-001")
    project_path = importer.ensure_project_stub("SAMPLE-PARTIAL-001", tile_set)

    assert len(tile_set.tile_paths) == 2
    assert tile_set.global_ref_path is not None
    assert tile_set.global_ref_path.name == "global_ref.png"
    assert tile_set.global_ref_path.is_file()
    manifest = tile_set.manifest_path.read_text(encoding="utf-8")
    assert "20261109-184132-411.jpg" in manifest
    assert "ignore-me" not in manifest
    project = BoardProject.from_json(project_path.read_text(encoding="utf-8"))
    assert project.board_id == "SAMPLE-PARTIAL-001"
    assert project.global_ref_path == "../samples/SAMPLE-PARTIAL-001/global_ref.png"
    assert project.global_map_path == "../maps/SAMPLE-PARTIAL-001.png"
    assert (repo / "assets" / "projects").resolve() in project_path.resolve().parents


def test_imported_sample_board_counts_when_present():
    sample = Path(__file__).resolve().parents[1] / "assets" / "samples" / "SAMPLE-PARTIAL-001"
    if not (sample / "global_ref.png").is_file():
        pytest.skip("real sample has not been imported")
    tiles, _global_ref = discover_raw_assets(sample / "tiles")
    # normalized tiles keep the 20261109 names; global lives beside tiles/
    assert len(list((sample / "tiles").glob("20261109-*.jpg"))) == 15
    assert (sample / "global_ref.png").is_file()
    assert (sample / "tiles_manifest.json").is_file()
    assert tiles  # filename rule still sees the copied tiles


def test_compare_fails_closed_on_unrelated_images(tmp_path: Path):
    mosaic = tmp_path / "mosaic.png"
    reference = tmp_path / "global_ref.png"
    _write(mosaic, np.full((40, 50, 3), 255, np.uint8))
    _write(reference, _texture(50, 40))
    report = SemiAutoStitchPipeline().compare_to_ref(mosaic, reference)
    assert report["ok"] is False
    assert "notes" in report and report["notes"]


def test_pipeline_places_a_crop_of_the_reference(tmp_path: Path):
    reference = _texture(240, 320)
    crop = reference[40:180, 30:200].copy()
    tile = tmp_path / "20261109-1.jpg"
    global_ref = tmp_path / "global_ref.png"
    _write(tile, crop)
    _write(global_ref, reference)
    out = tmp_path / "mosaic.png"
    result = SemiAutoStitchPipeline(min_inliers=8).run(
        TileSet("T", [tile], global_ref, None),
        out,
    )
    assert result.mosaic_path == out
    assert out.is_file()
    assert result.compare is not None
    assert "ok" in result.compare and "notes" in result.compare
    assert result.coverage_hint is not None and result.coverage_hint > 0
    if not result.ok:
        pytest.fail(result.message)


def _texture(height: int, width: int, shift: int = 0) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (30, 110, 40)
    rng = np.random.default_rng(1 + shift)
    step = 18
    for y in range(0, height, step):
        for x in range(0, width, step):
            color = tuple(int(value) for value in rng.integers(0, 255, size=3))
            cv2.rectangle(
                image,
                (x + 1, y + 1),
                (min(width - 1, x + step - 4), min(height - 1, y + step - 4)),
                color,
                -1,
            )
    return image


def _write(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image)
    assert ok
