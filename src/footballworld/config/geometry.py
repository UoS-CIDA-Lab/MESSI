"""Static ball and pitch geometry."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Ball:
    """Ball dimensions in metres."""

    radius: float = 0.11


@dataclass(frozen=True, slots=True)
class Stadium:
    """Pitch and goal dimensions in metres."""

    width: float = 68.0
    length: float = 105.0
    goal_width: float = 7.32
    goal_height: float = 2.44
    penalty_area_length: float = 16.5
    penalty_area_width: float = 40.32
    goal_area_length: float = 5.5
    goal_area_width: float = 18.32
    center_circle_radius: float = 9.15
    penalty_arc_radius: float = 9.15
    corner_arc_radius: float = 1.0

    @property
    def half_length(self) -> float:
        return self.length / 2.0

    @property
    def half_width(self) -> float:
        return self.width / 2.0
