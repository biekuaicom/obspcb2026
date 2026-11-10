"""Structure-tree data model and the first cheap shape extractor."""

import cv2
import numpy as np

from vision.structure_tree import (
    KIND_CIRCLE,
    KIND_LINE,
    KIND_RECT,
    StructureSnapshot,
    extract_structure_tree,
)


def _synthetic_shapes():
    image = np.full((220, 320, 3), 255, np.uint8)
    # Rectangle, circle, and a long line; plus tiny clutter that must be ignored.
    cv2.rectangle(image, (24, 36), (124, 96), (0, 0, 0), -1)
    cv2.circle(image, (220, 80), 28, (0, 0, 0), -1)
    cv2.line(image, (30, 160), (260, 172), (0, 0, 0), 5)
    cv2.circle(image, (16, 16), 1, (0, 0, 0), -1)
    cv2.rectangle(image, (300, 8), (303, 11), (0, 0, 0), -1)
    return image


def test_extractor_classifies_rect_circle_line_and_records_relative_positions():
    snapshot = extract_structure_tree(_synthetic_shapes())
    kinds = set(snapshot.kinds())
    assert KIND_RECT in kinds
    assert KIND_CIRCLE in kinds
    assert KIND_LINE in kinds
    # Tiny clutter is dropped: only the three intended shapes remain.
    assert len(snapshot.primitives) == 3
    by_kind = {item.kind: item for item in snapshot.primitives}
    rect = by_kind[KIND_RECT]
    circle = by_kind[KIND_CIRCLE]
    line = by_kind[KIND_LINE]
    assert rect.center[0] < circle.center[0]
    assert line.center[1] > rect.center[1]
    assert snapshot.relations
    pair = next(
        item
        for item in snapshot.relations
        if {snapshot.primitives[item.src].kind, snapshot.primitives[item.dst].kind} == {KIND_RECT, KIND_CIRCLE}
    )
    assert pair.distance > 50
    assert snapshot.units
    assert snapshot.combinations
    payload = snapshot.to_dict()
    assert payload["image_size"] == [320, 220]
    assert {item["kind"] for item in payload["primitives"]} == kinds


def test_empty_or_blank_image_returns_an_empty_snapshot():
    blank = np.full((80, 80, 3), 255, np.uint8)
    snapshot = extract_structure_tree(blank)
    assert isinstance(snapshot, StructureSnapshot)
    assert snapshot.primitives == []
    assert snapshot.relations == []
    assert extract_structure_tree(np.zeros((0, 0), np.uint8)).primitives == []
