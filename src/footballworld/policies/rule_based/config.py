"""Small, host-side configuration for the reproducible rule policy."""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Real

from footballworld.core.numeric import require_float32_representable
from footballworld.policies.rule_based.tactical_plan import (
    TacticalPlan,
    canonical_tactical_plan,
)


@dataclass(frozen=True, slots=True)
class RulePolicyConfig:
    """Balanced reference-policy choices, independent of JAX and environment state.

    These defaults define a readable seeded baseline rather than fitted
    football coefficients.  Only values that change a decision made by the
    baseline are exposed; physics and rule legality remain owned by the
    environment. ``default_seed`` is used only by compatibility calls that do
    not supply a match key; managed rollouts should supply their match key.
    """

    team_tactical_plans: tuple[TacticalPlan, TacticalPlan] = (
        TacticalPlan.JUEGO_DE_POSICION,
        TacticalPlan.JUEGO_DE_POSICION,
    )
    default_seed: int = 0

    # Movement command magnitudes. The five off-ball arrival fields retain a
    # useful distance-taper structure, but this policy owns the values because
    # its accelerated support-role decomposition is different.
    # They remain a transfer prior until a role-stratified tracking receipt is
    # available; stamina coefficients are not retuned to hide policy activity.
    approach_power: float = 0.90
    support_power: float = 0.55
    pressure_power: float = 1.00
    carrier_power: float = 0.75
    goalkeeper_power: float = 0.65
    offball_arrival_radius_m: float = 6.5
    offball_surge_span_radii: float = 3.5
    offball_walk_power: float = 0.06
    # The current policy has fewer separately accelerated support/marking roles,
    # so its ordinary cruise value is an independent transfer prior.
    offball_cruise_power: float = 0.28
    offball_surge_cap: float = 0.47

    # Spatial decision scales in metres.
    approach_slow_radius_m: float = 3.0
    pressure_distance_m: float = 5.0
    pass_min_progress_m: float = 4.0
    # Characteristic midpoint of the smooth range preference, not a hard gate.
    shoot_distance_m: float = 24.0

    # Readable tactical scoring.
    pass_lateral_penalty: float = 0.20
    backward_pass_penalty: float = 0.08
    pass_distance_penalty_per_m: float = 0.01
    receiver_choice_temperature: float = 0.20
    macro_choice_temperature: float = 0.30
    shot_portion_temperature: float = 0.18
    shot_value_gain: float = 1.15
    cross_value_gain: float = 1.55
    # Bounded continuation and event-cadence choices. These are explicit
    # policy design priors, not measured football constants. The distance
    # envelope is inherited from SoccerWorld's physically checked second-leg
    # graph; FootballWorld evaluates it only for the one visible carrier row.
    continuation_value_gain: float = 0.16
    continuation_min_distance_m: float = 5.0
    continuation_max_distance_m: float = 26.0
    continuation_backward_tolerance_m: float = 2.0
    continuation_support_lead_cap_s: float = 1.0
    service_opportunity_completion_floor: float = 0.58
    # Possession-episode attack patterns are seeded coordination priors.  They
    # never change candidate legality and are not tracking-data fits.
    attack_pattern_receiver_gain: float = 0.14
    attack_pattern_shape_shift_m: float = 3.0
    progressive_carry_min_commit_s: float = 0.8
    solo_carry_soft_limit_s: float = 2.2
    solo_carry_value_decay: float = 0.55
    immediate_return_penalty: float = 0.16
    # SoccerWorld measures early relays on a different policy/environment
    # contract. Its rounded 0.31 incidence is only a seeded transfer prior;
    # FootballWorld also requires pressure, body alignment, completion, and a
    # safe second leg before the draw can permit an early pass.
    quick_relay_probability: float = 0.31
    quick_relay_window_s: float = 2.0
    quick_relay_completion_floor: float = 0.60
    quick_relay_continuation_floor: float = 0.25
    quick_relay_pressure_floor: float = 0.15
    quick_relay_alignment_floor: float = 0.20
    # Policy-exposure design priors, not physical tackle/foul probabilities or
    # fitted event rates. The retained provider aggregate lacks a matching
    # attempt/opportunity denominator, so it cannot identify these controls.
    challenge_attempt_probability: float = 0.34
    challenge_proximity_gain: float = 0.30
    # A perfect prospective Law 11 oracle makes offside events impossible.
    # On a small seeded fraction of decisions, only a narrow band beyond the
    # observed line is treated as plausible. These are policy-exposure priors,
    # not changes to the environment's offside law or fitted constants.
    offside_timing_error_probability: float = 0.18
    offside_timing_error_margin_m: float = 1.00

    # Predicted runner marking is a tactical target generator. The 0.25 s lead
    # and 1.6 m goal-side gap inherit SoccerWorld design priors, not DFL fits.
    # FootballWorld's narrower 3 m runner/ball activation margins are its own
    # conservative design prior: it rejects SoccerWorld's 5 m/14 m early box
    # collapse. The environment remains authoritative for physical contests.
    box_mark_lead_s: float = 0.25
    box_mark_runner_margin_m: float = 3.0
    box_mark_ball_margin_m: float = 3.0
    box_mark_goal_side_distance_m: float = 1.6

    # Normalized force-to-ball magnitudes.
    dribble_power: float = 0.06
    pass_power: float = 0.52
    shoot_power: float = 1.00
    clear_power: float = 0.88
    restart_power: float = 0.45
    cross_power: float = 0.70

    # A controlled carry is a sequence of touches, not a control-step kick stream.
    dribble_touch_interval_s: float = 0.24
    challenge_attempt_interval_s: float = 0.67
    counterpress_window_s: float = 3.0

    # Normalized launch controls in the public action convention.
    pass_launch: float = -0.90
    shoot_launch: float = -0.55
    clear_launch: float = 0.10
    restart_launch: float = -0.85
    cross_launch: float = 0.05

    # These normalized spin/aim values are design priors. The player policy
    # recomputes direction against FootballWorld's own goal and ball physics.
    shoot_curl_start_m: float = 14.0
    shoot_curl_spin: float = 0.42
    shoot_curl_aim_compensation: float = 0.35
    cross_start_fraction: float = 0.05
    cross_wide_fraction: float = 0.55
    cross_target_central_fraction: float = 0.50
    cross_side_spin: float = 0.18
    cross_back_spin: float = 0.35
    # These angular shot-noise values are external compatibility priors. Their
    # source aggregates and fitting receipt are intentionally private.
    shot_noise_base_rad: float = 0.070
    shot_noise_quality_rad: float = 0.030
    shot_noise_pressure_rad: float = 0.010
    defensive_clear_depth_fraction: float = 0.72
    deep_clear_distance_m: float = 24.0
    deep_clear_lateral_ratio: float = 0.72
    deep_clear_touchline_margin_m: float = 0.50

    def __post_init__(self) -> None:
        """Reject malformed host configuration before policy tracing begins."""

        plans = self.team_tactical_plans
        if not isinstance(plans, tuple) or len(plans) != 2:
            raise TypeError("team_tactical_plans must be a length-two tuple")
        object.__setattr__(
            self,
            "team_tactical_plans",
            tuple(canonical_tactical_plan(plan) for plan in plans),
        )
        if isinstance(self.default_seed, bool) or not isinstance(
            self.default_seed, int
        ):
            raise TypeError("default_seed must be an integer")
        if not 0 <= self.default_seed <= 0xFFFFFFFF:
            raise ValueError("default_seed must be in [0, 2**32 - 1]")

        power_names = (
            "approach_power",
            "support_power",
            "pressure_power",
            "carrier_power",
            "goalkeeper_power",
            "dribble_power",
            "pass_power",
            "shoot_power",
            "clear_power",
            "restart_power",
            "cross_power",
            "offball_walk_power",
            "offball_cruise_power",
            "offball_surge_cap",
        )
        distance_names = (
            "approach_slow_radius_m",
            "pressure_distance_m",
            "pass_min_progress_m",
            "shoot_distance_m",
            "shoot_curl_start_m",
            "offball_arrival_radius_m",
            "deep_clear_distance_m",
            "deep_clear_touchline_margin_m",
            "offside_timing_error_margin_m",
            "attack_pattern_shape_shift_m",
            "continuation_min_distance_m",
            "continuation_max_distance_m",
            "continuation_backward_tolerance_m",
            "box_mark_runner_margin_m",
            "box_mark_ball_margin_m",
            "box_mark_goal_side_distance_m",
        )
        positive_names = (
            "dribble_touch_interval_s",
            "challenge_attempt_interval_s",
            "counterpress_window_s",
            "solo_carry_soft_limit_s",
            "receiver_choice_temperature",
            "macro_choice_temperature",
            "shot_portion_temperature",
            "shot_value_gain",
            "cross_value_gain",
            "offball_surge_span_radii",
            "progressive_carry_min_commit_s",
            "continuation_support_lead_cap_s",
            "box_mark_lead_s",
            "quick_relay_window_s",
        )
        nonnegative_names = (
            "pass_lateral_penalty",
            "pass_distance_penalty_per_m",
            "shot_noise_base_rad",
            "shot_noise_quality_rad",
            "shot_noise_pressure_rad",
        )
        unit_names = (
            "immediate_return_penalty",
            "backward_pass_penalty",
            "solo_carry_value_decay",
            "challenge_attempt_probability",
            "challenge_proximity_gain",
            "offside_timing_error_probability",
            "shoot_curl_spin",
            "shoot_curl_aim_compensation",
            "cross_start_fraction",
            "cross_wide_fraction",
            "cross_target_central_fraction",
            "cross_side_spin",
            "cross_back_spin",
            "defensive_clear_depth_fraction",
            "deep_clear_lateral_ratio",
            "attack_pattern_receiver_gain",
            "continuation_value_gain",
            "service_opportunity_completion_floor",
            "quick_relay_probability",
            "quick_relay_completion_floor",
            "quick_relay_continuation_floor",
            "quick_relay_pressure_floor",
        )
        launch_names = (
            "pass_launch",
            "shoot_launch",
            "clear_launch",
            "restart_launch",
            "cross_launch",
        )

        for field in fields(self):
            if field.name in ("team_tactical_plans", "default_seed"):
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field.name} must be a real number")
            value = require_float32_representable(field.name, value)
            object.__setattr__(self, field.name, value)

        for name in power_names:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in distance_names:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be greater than zero")
        for name in positive_names:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be greater than zero")
        for name in nonnegative_names:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in unit_names:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not -1.0 <= self.quick_relay_alignment_floor <= 1.0:
            raise ValueError("quick_relay_alignment_floor must be in [-1, 1]")
        for name in launch_names:
            value = getattr(self, name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")
        if self.shoot_curl_start_m >= self.shoot_distance_m:
            raise ValueError("shoot_curl_start_m must be less than shoot_distance_m")
        if self.progressive_carry_min_commit_s > self.solo_carry_soft_limit_s:
            raise ValueError(
                "progressive_carry_min_commit_s must not exceed solo_carry_soft_limit_s"
            )
        if self.continuation_min_distance_m >= self.continuation_max_distance_m:
            raise ValueError(
                "continuation_min_distance_m must be less than "
                "continuation_max_distance_m"
            )
        if not (
            self.offball_walk_power
            <= self.offball_cruise_power
            <= self.offball_surge_cap
        ):
            raise ValueError(
                "offball movement powers must satisfy walk <= cruise <= surge cap"
            )


__all__ = ["RulePolicyConfig"]
