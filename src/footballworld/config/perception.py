"""Configuration for optional view-limited observations."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Perception:
    """Agent-facing visibility and bounded torso-relative gaze settings.

    Gaze defaults are explicit design priors rather than measured biological
    constants.  The relative interval is strictly smaller than a full turn,
    so its normalized scalar action has no circular seam.
    """

    limit_by_view_angle: bool = False
    horizontal_fov_degrees: float = 160.0
    gaze_yaw_limit_degrees: float = 90.0
    gaze_slew_rate_degrees_s: float = 540.0
