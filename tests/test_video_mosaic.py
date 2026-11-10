"""Synthetic video mosaic: translating views are kept, a noise frame is not."""

import numpy as np

from vision.video_mosaic import VideoMosaicMapper, make_synthetic_translation_frames


def test_translation_sequence_skips_noise_and_expands_canvas():
    frames = make_synthetic_translation_frames(count=8, bad_index=3)
    mapper = VideoMosaicMapper(min_inliers=20)
    result = mapper.run_frames(frames)
    assert result.frames_total == 8
    assert result.frames_accepted >= 6
    assert result.frames_skipped >= 1
    assert result.frames[0]["accepted"] is True
    assert result.frames[3]["accepted"] is False
    assert result.canvas_size[0] > frames[0].shape[1]
    assert result.coverage_hint is not None and result.coverage_hint > 0
    mosaic = mapper.render()
    assert mosaic is not None
    assert mosaic.shape[1] == result.canvas_size[0]
    assert mosaic.shape[0] == result.canvas_size[1]
    assert result.ok is True


def test_unrelated_second_frame_is_not_accepted():
    first = make_synthetic_translation_frames(count=1, bad_index=None)[0]
    noise = np.random.default_rng(1).integers(0, 255, size=first.shape, dtype=np.uint8)
    result = VideoMosaicMapper(min_inliers=20).run_frames([first, noise])
    assert result.frames_accepted == 1
    assert result.frames_skipped == 1
    assert result.frames[1]["accepted"] is False
    assert result.ok is False
