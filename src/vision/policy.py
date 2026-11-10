"""Confirmed registration policy. Constants only — the live loop is not wired.

D1. ``M`` and ``H`` map global-map pixels to the current frame.
D3. ``LocalGlobalTracker`` is the primary path. ``PCBAffineTracker`` runs only
    as fallback when local-global does not lock.
D4. After ``TRACKING_LOST_TTL_S`` seconds without a good fix, freeze the last
    overlay and show a UI hint. This slice stores the timeout and does not
    start that timer.
"""

TRANSFORM_DIRECTION = "global_to_frame"
PRIMARY_TRACKER = "local_global"
FALLBACK_TRACKER = "affine"
TRACKING_LOST_TTL_S = 0.8
