"""Primary local–global tracker stub (D3).

``H`` is 3×3 and maps **global-map pixels → current-frame pixels** (D1).
``project_boxes`` applies that direction with ``cv2.perspectiveTransform``.
``PCBAffineTracker`` is the degrade path when this matcher fails. ``track``
does not run ORB. It fails closed with an identity matrix. Lost-lock timing
is ``vision.policy.TRACKING_LOST_TTL_S`` and is not started here.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_STUB_MESSAGE = "桩：未做 ORB 局部–全局匹配，返回单位单应（全局像素→当前帧像素）"


def identity_homography() -> np.ndarray:
    """3×3 identity. Global pixels and current-frame pixels coincide."""
    return np.eye(3, dtype=np.float64)


@dataclass
class HomographyResult:
    """One local–global registration attempt.

    ``H`` has shape ``(3, 3)`` and dtype float64. ``inliers`` is 0 while the
    matcher is a stub. ``ok`` False means callers must not treat ``H`` as a
    fresh lock (the stub ``H`` is identity).
    """

    H: np.ndarray
    inliers: int
    confidence: float
    ok: bool
    message: str = ""


class LocalGlobalTracker:
    """Primary registration path (D3). ORB matching is not implemented.

    Parameters are stored for the later matcher and are not used to detect
    features in this slice.
    """

    def __init__(
        self,
        n_features: int = 5000,
        ratio: float = 0.75,
        ransac_thresh: float = 3.0,
    ):
        if int(n_features) <= 0:
            raise ValueError("n_features must be positive")
        if not 0.0 < float(ratio) < 1.0:
            raise ValueError("ratio must be between 0 and 1")
        if float(ransac_thresh) <= 0:
            raise ValueError("ransac_thresh must be positive")
        self.n_features = int(n_features)
        self.ratio = float(ratio)
        self.ransac_thresh = float(ransac_thresh)
        self.map_shape: tuple[int, ...] | None = None
        self.last_result: HomographyResult | None = None

    def set_global_map(self, global_bgr: np.ndarray) -> None:
        """Remember the map shape. Descriptors are not computed and pixels are not kept."""
        if not isinstance(global_bgr, np.ndarray) or global_bgr.size == 0:
            raise ValueError("global_bgr must be a non-empty ndarray")
        self.map_shape = tuple(int(value) for value in global_bgr.shape)

    def track(self, frame_bgr: np.ndarray) -> HomographyResult:
        """Return a failed registration with an identity homography."""
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            result = HomographyResult(
                H=identity_homography(),
                inliers=0,
                confidence=0.0,
                ok=False,
                message="空帧，无法配准",
            )
        else:
            result = HomographyResult(
                H=identity_homography(),
                inliers=0,
                confidence=0.0,
                ok=False,
                message=_STUB_MESSAGE,
            )
        self.last_result = result
        return result

    def project_boxes(
        self,
        boxes_xywh: list[tuple[float, float, float, float]],
        H: np.ndarray,
    ) -> list[np.ndarray]:
        """Project global ``xywh`` boxes through ``H`` into the current frame.

        Each result is a ``(4, 2)`` float64 array of corners in this order:
        top-left, top-right, bottom-right, bottom-left. This is geometry only.
        It does not decide visibility and it does not run ``track``.

        ``H`` must be 3×3, global pixel → current frame pixel.
        """
        matrix = np.asarray(H, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("H must have shape (3, 3)")
        projected: list[np.ndarray] = []
        for box in boxes_xywh:
            if len(box) != 4:
                raise ValueError("each box must be (x, y, w, h)")
            x, y, w, h = (float(value) for value in box)
            corners = np.array(
                [
                    [[x, y]],
                    [[x + w, y]],
                    [[x + w, y + h]],
                    [[x, y + h]],
                ],
                dtype=np.float64,
            )
            transformed = cv2.perspectiveTransform(corners, matrix)
            projected.append(np.asarray(transformed, dtype=np.float64).reshape(4, 2))
        return projected
