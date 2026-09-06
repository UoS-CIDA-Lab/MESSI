"""Configuration for the two-timescale player stamina model.

The defaults define a latent workload model and are not claimed as directly
measured physiology. Their provenance is recorded in the coefficient audit.
"""

from dataclasses import dataclass

from footballworld.core.timebase import DEFAULT_MATCH_DURATION_SECONDS


@dataclass(frozen=True, slots=True)
class LongStamina:
    """Match-scale workload accumulation and sustained-speed response."""

    sprint_speed: float = 5.5
    sprint_mult: float = 3.0
    idle_load: float = 0.12
    speed_ref: float = 2.4
    speed_load: float = 0.90
    accel_ref: float = 4.0
    accel_load: float = 0.10
    vmax_floor: float = 0.99
    end_frac: float = 0.05
    tail_knee: float = 0.20
    reference_duration_s: float = DEFAULT_MATCH_DURATION_SECONDS
    reference_workload: float = 0.82


@dataclass(frozen=True, slots=True)
class ShortStamina:
    """Sprint depletion, recovery, and instantaneous speed headroom."""

    vmax_floor: float = 0.70
    headroom_knee: float = 0.25
    depletion_s: float = 10.0
    depletion_speed_frac: float = 0.70
    speed_exponent: float = 2.0
    accel_ref: float = 6.0
    accel_load: float = 0.20
    recovery_tau_s: float = 60.0
    recovery_speed_frac: float = 0.70
    recovery_exponent: float = 1.0
