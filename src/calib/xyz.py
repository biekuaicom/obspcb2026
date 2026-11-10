"""XYZ stage calibration placeholder.

The board stores millimetres and pixels separately so a later machine path can
replace the vision matrix. This module does not implement that path.

Product decisions around the physical origin, Scale(Z), and the serial
protocol are still open. Do not treat this file as a stable calibration API.
"""


def screen_position_px(pcb_xy_mm, machine_xy_mm, scale_px_per_mm):
    """Map a PCB point into screen pixels using encoder pose.

    Intended formula, not implemented::

        Screen_Pos = (PCB_Phys_Pos - Machine_Current_Pos) × Scale(Z)

    Raises:
        NotImplementedError: always, in this slice.
    """
    del pcb_xy_mm, machine_xy_mm, scale_px_per_mm
    raise NotImplementedError(
        "XYZ 机台标定本期不实现。"
        "约定公式：Screen_Pos = (PCB_Phys_Pos − Machine_Current_Pos) × Scale(Z)。"
    )
