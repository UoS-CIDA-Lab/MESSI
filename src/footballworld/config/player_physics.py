"""Configuration for player locomotion and collision separation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlayerPhysics:
    """Coefficients consumed by player locomotion and separation."""

    forward_acceleration_mps2: float = 7.95
    lateral_acceleration_mps2: float = 8.46
    braking_deceleration_mps2: float = 10.51
    body_turn_rate_max_radps: float = 8.0
    backward_speed_ratio: float = 0.65
    collision_normal_restitution: float = 0.05
    separation_iterations: int = 3
