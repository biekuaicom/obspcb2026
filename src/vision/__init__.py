"""Vision workers.

Capture, tracker stubs, still-tile stitch, optional video mosaic, topology
live match, and stable interval capture live here. Default mapping is
stable capture with red coverage. Video mosaic does not grow a full-board
canvas on that path. Live lock uses geometric board features in the first-tile frame, not a drifted color mosaic.
structure_tree holds the planned component-combination model for later
single-component recognition; it is not the live pose matcher.
YOLO detection is not part of this tree.
"""
