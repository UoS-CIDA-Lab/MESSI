"""Configuration for player-ball reach eligibility."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Reach:
    """Physical reach envelopes before contact outcome resolution."""

    carry_radius_m: float = 1.1
    challenge_radius_m: float = 1.4
    goalkeeper_radius_m: float = 2.0
    block_speed_limit_mps: float = 31.0
    height_speed_penalty_mps_per_m: float = 3.5
    # Appended to preserve the positional constructor order of the original
    # five public fields.
    goalkeeper_standing_radius_m: float = 1.1
