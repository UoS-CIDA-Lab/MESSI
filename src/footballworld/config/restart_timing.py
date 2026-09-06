"""Environment-owned timing for releasing a prepared restart."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RestartTiming:
    """Release deadline and static ordinary-restart approach selection.

    Continuous approach is initially limited to the four ordinary restart
    families whose release pose can lie far from the selected taker. Kickoffs,
    penalties, goal kicks, and goalkeeper holds retain exact positioning.
    """

    forced_release_delay_s: float = 3.0
    continuous_freekick_approach: bool = True
    continuous_offside_approach: bool = True
    continuous_corner_approach: bool = True
    continuous_throwin_approach: bool = True


__all__ = ["RestartTiming"]
