"""Fiducial affine tracker stub (D3 fallback, not the primary path).

``M`` is 2×3 and maps **global-map pixels → current-frame pixels** (D1),
matching ``cv2.estimateAffinePartial2D(src_global, dst_frame)``. Use this when
``LocalGlobalTracker`` fails. This slice does not search templates. ``track``
fails closed with an identity-like matrix. See ``vision.policy``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_STUB_MESSAGE = "桩：未做 Fiducial 匹配，返回单位仿射（全局像素→当前帧像素）"


def identity_affine() -> np.ndarray:
    """2×3 partial affine with no translation, rotation, or scale."""
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


@dataclass
class AffineResult:
    """One affine registration attempt.

    ``M`` always has shape ``(2, 3)`` and dtype float64. ``ok`` is False when
    the matrix must not move annotations (identity, or the last good matrix
    once a real tracker exists).
    """

    M: np.ndarray
    confidence: float
    ok: bool
    message: str = ""


class PCBAffineTracker:
    """Fallback affine path (D3). Fiducial match is not implemented.

    Parameters
    ----------
    roi_size:
        Search window stored for a later slice. Default 200×200. Unused by the
        stub beyond validation and ``set_templates`` bookkeeping.
    """

    def __init__(self, roi_size: tuple[int, int] = (200, 200)):
        if len(roi_size) != 2 or int(roi_size[0]) <= 0 or int(roi_size[1]) <= 0:
            raise ValueError("roi_size must be a positive (width, height)")
        self.roi_size = (int(roi_size[0]), int(roi_size[1]))
        self.templates: list[dict] = []
        self.last_result: AffineResult | None = None

    def set_templates(self, templates: list[dict]) -> None:
        """Store fiducial templates. No matching is performed.

        Each item is a dict ``{id, image|path, global_pixel:(x, y)}``.
        The dicts are shallow-copied. Images are not read from disk here.
        """
        self.templates = [dict(item) for item in templates]

    def reset(self) -> None:
        """Drop the last result. Templates already set are kept."""
        self.last_result = None

    def track(self, frame_bgr: np.ndarray) -> AffineResult:
        """Return a failed registration with an identity matrix.

        A later slice will match each template inside ``roi_size`` and call
        ``estimateAffinePartial2D``. Callers can still pass an ``AffineResult``
        they built themselves into the projection helpers.
        """
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            result = AffineResult(
                M=identity_affine(),
                confidence=0.0,
                ok=False,
                message="空帧，无法配准",
            )
        else:
            result = AffineResult(
                M=identity_affine(),
                confidence=0.0,
                ok=False,
                message=_STUB_MESSAGE,
            )
        self.last_result = result
        return result
