"""Headless stitch-and-compare entry.

From the repository root::

    PYTHONPATH=src python -m vision.stitch_compare \\
        --zip /path/to/局部图片.zip \\
        --board-id SAMPLE-PARTIAL-001 \\
        --out artifacts/stitch_compare
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

from vision.stitch import (
    GLOBAL_REF_SOURCE_NAME,
    SampleAssetImporter,
    SemiAutoStitchPipeline,
    discover_raw_assets,
    write_compare_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导入局部瓦片，拼到全局对照图上并打分")
    parser.add_argument("--zip", type=Path, default=None, help="局部图片.zip")
    parser.add_argument("--board-dir", type=Path, default=None, help="已解压目录，或 assets/samples/<board_id>")
    parser.add_argument("--board-id", default="SAMPLE-PARTIAL-001")
    parser.add_argument("--out", type=Path, default=Path("artifacts/stitch_compare"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--try-stitcher", action="store_true", help="额外试一次 OpenCV Stitcher，失败只记日志")
    args = parser.parse_args(argv)
    if args.zip is None and args.board_dir is None:
        parser.error("需要 --zip 或 --board-dir")

    repo_root = args.repo_root.resolve()
    importer = SampleAssetImporter(repo_root)
    source = _source_dir(args.zip, args.board_dir)
    tile_set = importer.import_board_dir(source, args.board_id)
    project_path = importer.ensure_project_stub(args.board_id, tile_set)
    pipeline = SemiAutoStitchPipeline(try_stitcher=args.try_stitcher)
    mosaic_path = args.out / "mosaic.png"
    result = pipeline.run(tile_set, mosaic_path)
    if result.mosaic_path is not None and result.mosaic_path.is_file():
        maps = repo_root / "assets" / "maps"
        maps.mkdir(parents=True, exist_ok=True)
        target = maps / f"{args.board_id}.png"
        target.write_bytes(result.mosaic_path.read_bytes())
    summary = write_compare_artifacts(result, tile_set.global_ref_path, args.out)
    summary["board_id"] = args.board_id
    summary["tiles"] = [path.name for path in tile_set.tile_paths]
    summary["global_ref"] = str(tile_set.global_ref_path)
    summary["global_ref_rule"] = GLOBAL_REF_SOURCE_NAME
    summary["project"] = str(project_path)
    summary["manifest"] = None if tile_set.manifest_path is None else str(tile_set.manifest_path)
    summary_path = args.out / "summary.json"
    public = {key: value for key, value in summary.items() if key != "summary_path"}
    summary_path.write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": result.ok, "coverage_hint": result.coverage_hint, "summary": str(summary_path)}, ensure_ascii=False))
    print(result.message)
    return 0 if result.mosaic_path is not None or result.ok else 1


def _source_dir(zip_path: Path | None, board_dir: Path | None) -> Path:
    if board_dir is not None:
        return board_dir
    assert zip_path is not None
    extract_root = Path(tempfile.mkdtemp(prefix="pcb-tiles-"))
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_root)
    _tiles, global_ref = discover_raw_assets(extract_root)
    if global_ref is None:
        raise SystemExit(f"zip 里没有 {GLOBAL_REF_SOURCE_NAME}")
    return global_ref.parent


if __name__ == "__main__":
    raise SystemExit(main())
