"""Law-level goalkeeper hand-control duration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GoalkeeperHolding:
    """SI duration fixed by IFAB Law 12.3 (2026/27)."""

    hand_control_limit_s: float = 8.0
