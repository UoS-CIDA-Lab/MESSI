"""Host-only validation for public rollout construction."""

import math
import numbers
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike

from footballworld.config.action import ActionScale
from footballworld.config.ball_physics import BallPhysics
from footballworld.config.body_contact import BodyContact
from footballworld.config.body_foul import BodyFoul
from footballworld.config.contact_timing import ContactTiming
from footballworld.config.contest import Contest
from footballworld.config.geometry import Ball, Stadium
from footballworld.config.gk_holding import GoalkeeperHolding
from footballworld.config.perception import Perception
from footballworld.config.player_physics import PlayerPhysics
from footballworld.config.reach import Reach
from footballworld.config.restart_timing import RestartTiming
from footballworld.config.roster import (
    Player,
    PlayerProfile,
    player_profile_values_valid,
)
from footballworld.config.roster_sampling import RosterSampling
from footballworld.config.stamina import LongStamina, ShortStamina
from footballworld.core.constants import (
    GEOMETRY_EPS,
    IFAB_MAX_TEAM_PLAYERS,
    MAX_PLAYER_COLLISION_MICROSTEPS,
    TEAM_0,
    TEAM_1,
)
from footballworld.core.timebase import Timebase
from footballworld.rules.restart_positioning import (
    RestartPositionLaw,
    kickoff_packing_minimum_half_extents,
)

MAX_ROLLING_TABLE_KNOTS = 256
_FINITE = (None, None, False, False)
_NONNEGATIVE = (0.0, None, False, False)
_POSITIVE = (0.0, None, True, False)
_UNIT = (0.0, 1.0, False, False)
_OPEN_UNIT = (0.0, 1.0, True, True)
# Normalization divides by the configured maximum and restores with one
# float32 multiply. Two mantissa bits of headroom keep adjacent ticks distinct
# even when that maximum is not a power of two and XLA uses fast division.
_FLOAT32_EXACT_COUNTER_MAX = (1 << 22) - 1

# These compact field groups are constructor invariants, not model logic.
_FIELD_SPECS = {
    Ball: ((_POSITIVE, "radius"),),
    Stadium: (
        (
            _POSITIVE,
            "width length goal_width goal_height penalty_area_length penalty_area_width goal_area_length goal_area_width center_circle_radius penalty_arc_radius corner_arc_radius",
        ),
    ),
    BodyContact: (
        (
            _POSITIVE,
            "shoulder_width_m torso_depth_m leg_radius_m foot_forward_extent_m head_radius_m",
        ),
        (
            (0.0, 1.0, True, False),
            "leg_apex_height_factor torso_top_height_factor",
        ),
        (_UNIT, "restitution"),
    ),
    BodyFoul: (
        (_POSITIVE, "minimum_closing_speed_mps ball_near_distance_m"),
        (
            _FINITE,
            "base_logit closing_speed_logit_weight_per_mps behind_logit_weight shoulder_alignment_logit_discount ball_far_logit_weight possessed_victim_logit_weight goal_denial_logit_weight",
        ),
        (_UNIT, "probability_floor probability_ceiling"),
        ((1.0, None, False, False), "rare_case_coverage_multiplier"),
    ),
    ActionScale: (
        (
            _POSITIVE,
            "kick_speed_max_mps throw_speed_max_mps control_request_speed_max_mps spin_max_radps restart_min_ball_speed_mps",
        ),
        (
            _UNIT,
            "header_speed_retention chest_speed_retention challenge_speed_retention",
        ),
        (
            _NONNEGATIVE,
            "throw_release_height_addition_m ground_launch_down_max_radians",
        ),
        ((0.0, 1.0, True, False), "pelvis_height_factor"),
        ((0.0, 0.5 * math.pi, True, False), "launch_max_radians"),
    ),
    Reach: (
        (
            _POSITIVE,
            "carry_radius_m challenge_radius_m goalkeeper_standing_radius_m goalkeeper_radius_m block_speed_limit_mps",
        ),
        (_NONNEGATIVE, "height_speed_penalty_mps_per_m"),
    ),
    ContactTiming: (
        (_POSITIVE, "active_contact_interval_s"),
        (
            _NONNEGATIVE,
            "challenge_recovery_s max_lunge_extra_recovery_s aerial_attempt_recovery_s goalkeeper_dive_recovery_s possession_loss_lock_s",
        ),
    ),
    Contest: (
        (
            _FINITE,
            "distance_weight reach_time_weight height_fit_weight possession_weight ball_control_weight card_attack_progress_logit_weight card_elapsed_fraction_logit_weight",
        ),
        (_POSITIVE, "temperature goalkeeper_catch_speed_scale_mps"),
        (
            _NONNEGATIVE,
            "goalkeeper_catch_speed_midpoint_mps tackle_foul_context_logit_limit",
        ),
        (
            (1.0, None, False, False),
            "tackle_foul_rare_case_coverage_multiplier",
        ),
        (
            _UNIT,
            "tackle_success_probability tackle_foul_probability tackle_deflection_probability card_probability_midpoint direct_red_given_card_probability",
        ),
    ),
    GoalkeeperHolding: ((_POSITIVE, "hand_control_limit_s"),),
    RestartTiming: ((_POSITIVE, "forced_release_delay_s"),),
    PlayerPhysics: (
        (
            _POSITIVE,
            "forward_acceleration_mps2 lateral_acceleration_mps2 braking_deceleration_mps2 body_turn_rate_max_radps",
        ),
        ((0.0, 1.0, True, False), "backward_speed_ratio"),
        (_UNIT, "collision_normal_restitution"),
    ),
    LongStamina: (
        (
            _POSITIVE,
            "sprint_speed speed_ref accel_ref reference_duration_s reference_workload",
        ),
        (_NONNEGATIVE, "idle_load speed_load accel_load"),
        ((1.0, None, False, False), "sprint_mult"),
        (_UNIT, "vmax_floor"),
        ((0.0, 1.0, True, False), "end_frac tail_knee"),
    ),
    ShortStamina: (
        (
            _POSITIVE,
            "depletion_s speed_exponent accel_ref recovery_tau_s recovery_exponent",
        ),
        (_NONNEGATIVE, "accel_load"),
        ((0.0, 1.0, True, False), "vmax_floor headroom_knee recovery_speed_frac"),
        (_OPEN_UNIT, "depletion_speed_frac"),
    ),
    BallPhysics: (
        (
            _POSITIVE,
            "g ball_mass_kg drag_crisis_speed_mps drag_crisis_width_mps spin_drag_exponent lift_coefficient_limit ball_inertia_ratio",
        ),
        (
            _NONNEGATIVE,
            "ground_settle_vz air_density_kgpm3 drag_coefficient_high_re drag_crisis_drop spin_drag_scale spin_drag_min_parameter air_spin_decay c_ground_curl ground_slide_friction ground_spin_decay bounce_spin_vmin goal_frame_radius goal_frame_mu",
        ),
        (_UNIT, "e_rest bounce_tangential_e bounce_h_keep goal_frame_e_rest"),
    ),
    Perception: (
        ((0.0, 360.0, True, False), "horizontal_fov_degrees"),
        ((0.0, 180.0, True, True), "gaze_yaw_limit_degrees"),
        (_POSITIVE, "gaze_slew_rate_degrees_s"),
    ),
}


def _real(name, value, bounds=_FINITE) -> float:
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    with np.errstate(over="ignore"):
        represented = np.asarray(result, dtype=np.float32)
    if not np.isfinite(represented):
        raise ValueError(f"{name} must be finite when represented as float32")
    if result != 0.0 and float(represented) == 0.0:
        raise ValueError(f"{name} must not underflow to zero in float32")
    low, high, low_open, high_open = bounds
    if low is not None and low_open and float(represented) <= low:
        raise ValueError(f"{name} must remain greater than {low} in float32")
    if high is not None and high_open and float(represented) >= high:
        raise ValueError(f"{name} must remain less than {high} in float32")
    if low is not None and (result <= low if low_open else result < low):
        relation = "greater than" if low_open else "at least"
        raise ValueError(f"{name} must be {relation} {low}")
    if high is not None and (result >= high if high_open else result > high):
        relation = "less than" if high_open else "at most"
        raise ValueError(f"{name} must be {relation} {high}")
    return result


def _fields(prefix, owner, specs) -> None:
    for bounds, names in specs:
        for name in names.split():
            _real(f"{prefix}.{name}", getattr(owner, name), bounds)


def _integer(name, value, minimum, maximum) -> int:
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _f32_product(name: str, *factors: float) -> float:
    result = 1.0
    for factor in factors:
        result *= float(factor)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            represented = np.float32(result)
        if (
            not math.isfinite(result)
            or not np.isfinite(represented)
            or (result != 0.0 and float(represented) == 0.0)
        ):
            raise ValueError(f"{name} is outside the float32 runtime range")
    return result


def _f32_sum(name: str, *terms: float) -> float:
    """Mirror a non-negative float32 runtime sum and reject overflow."""

    represented = np.float32(0.0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for term in terms:
            represented = np.float32(represented + np.float32(term))
            if not np.isfinite(represented):
                raise ValueError(f"{name} is outside the float32 runtime range")
    return float(represented)


def _validate_stamina_float32_envelope(
    timebase: Timebase,
    long_stamina: LongStamina,
) -> None:
    """Validate coupled intermediates used by the long-stamina transition.

    Check the derived drain rate at construction and extend that host-only
    protection to the denominator and soft-tail
    intermediates, whose individually valid inputs can otherwise form Inf/NaN
    in the lean JAX step.
    """

    denominator = _f32_product(
        "long_stamina reference duration-workload denominator",
        long_stamina.reference_duration_s,
        long_stamina.reference_workload,
    )
    drain_base = (1.0 - long_stamina.end_frac) / denominator
    _real("long_stamina derived drain base", drain_base, _POSITIVE)

    tail_gap = long_stamina.tail_knee - long_stamina.end_frac
    _real("long_stamina tail gap", tail_gap, _POSITIVE)
    tail_decay = math.log(long_stamina.tail_knee / long_stamina.end_frac) / tail_gap
    _real("long_stamina derived tail decay", tail_decay, _POSITIVE)

    sprint_load = _f32_product(
        "long_stamina maximum sprint load",
        2.0,
        long_stamina.sprint_mult - 1.0,
    )
    speed_load = _f32_product(
        "long_stamina maximum speed load", 4.0, long_stamina.speed_load
    )
    acceleration_load = _f32_product(
        "long_stamina maximum acceleration load", 16.0, long_stamina.accel_load
    )
    maximum_workload = _f32_sum(
        "long_stamina maximum workload",
        long_stamina.idle_load,
        sprint_load,
        speed_load,
        acceleration_load,
    )
    maximum_drain_rate = _f32_product(
        "long_stamina maximum drain rate", drain_base, maximum_workload
    )
    _f32_product(
        "long_stamina maximum drain per physics tick",
        timebase.dt_phys,
        maximum_drain_rate,
    )


def _validate_bounce_energy_admissibility(physics: BallPhysics) -> None:
    """Reject ground-bounce maps that can create tangential kinetic energy.

    The configured horizontal response maps one translation/spin axis
    linearly. Its energy-normalized two-by-two operator must be contractive.
    This guard changes no coefficient and stays outside the rollout graph; a
    full slip-based replacement remains deferred until the bounce group is
    jointly calibrated.
    """

    inertia = float(physics.ball_inertia_ratio)
    tangential = float(physics.bounce_tangential_e)
    horizontal_keep = float(physics.bounce_h_keep)
    impulse_fraction = (1.0 + tangential) * inertia / (1.0 + inertia)
    residual_spin = 1.0 - impulse_fraction / inertia
    coupling = horizontal_keep * impulse_fraction / math.sqrt(inertia)
    operator = np.asarray(
        ((horizontal_keep, -coupling), (0.0, residual_spin)),
        dtype=np.float64,
    )
    maximum_energy_ratio = float(np.linalg.norm(operator, ord=2) ** 2)
    tolerance = 8.0 * float(np.finfo(np.float32).eps)
    if (
        not math.isfinite(maximum_energy_ratio)
        or maximum_energy_ratio > 1.0 + tolerance
    ):
        raise ValueError(
            "ground bounce translation-spin map can create kinetic energy; "
            "reduce bounce_tangential_e or bounce_h_keep"
        )


def _validate_player_collision_microstep_budget(
    *,
    timebase: Timebase,
    body: BodyContact,
    maximum_profile_speed_mps: float,
) -> None:
    """Bound the dynamic collision loop for every normalized roster state."""

    # Equal-or-dissipative player impacts cannot concentrate more total kinetic
    # energy into one slot than sqrt(N) times the per-slot speed domain. This is
    # deliberately an operational resource guard, not a football coefficient.
    active_slots = 2 * IFAB_MAX_TEAM_PLAYERS
    concentrated_speed = _f32_product(
        "player collision concentrated speed support",
        math.sqrt(active_slots),
        maximum_profile_speed_mps,
    )
    relative_speed = _f32_product(
        "player collision relative speed support", 2.0, concentrated_speed
    )
    safe_depth = max(body.torso_depth_m - GEOMETRY_EPS, GEOMETRY_EPS)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        relative_path = np.float32(relative_speed) * np.float32(timebase.dt_phys)
        microsteps = np.ceil(relative_path / np.float32(safe_depth))
    if not np.isfinite(microsteps) or microsteps > MAX_PLAYER_COLLISION_MICROSTEPS:
        raise ValueError(
            "player collision microstep budget exceeds the operational limit "
            f"of {MAX_PLAYER_COLLISION_MICROSTEPS}"
        )


def _edge_ulp(stadium: Stadium, boundary_margin_m: float) -> float:
    return max(
        float(np.spacing(np.float32(value)))
        for value in (
            stadium.half_length + boundary_margin_m,
            stadium.half_width + boundary_margin_m,
        )
    )


def _nearest_physics_ticks(
    name: str,
    seconds: float,
    timebase: Timebase,
    *,
    minimum: int = 1,
) -> int:
    """Return the host-rounded counter used by the contact transition."""

    try:
        ticks = timebase.physics_steps_for(
            seconds,
            rounding="nearest",
            minimum=minimum,
        )
    except (OverflowError, ValueError) as error:
        raise ValueError(
            f"{name} is outside the exact normalized-State counter horizon"
        ) from error
    if ticks > _FLOAT32_EXACT_COUNTER_MAX:
        raise ValueError(
            f"{name} produces {ticks} physics ticks, exceeding the exact "
            "normalized-State counter horizon "
            f"({_FLOAT32_EXACT_COUNTER_MAX})"
        )
    return ticks


def _f32_rint_physics_ticks(
    name: str,
    seconds: float,
    timebase: Timebase,
    *,
    configured_seconds: float | None = None,
    minimum: int = 1,
) -> int:
    """Return the float32-rint counter used by athletic contact recovery."""

    _nearest_physics_ticks(
        name,
        seconds if configured_seconds is None else configured_seconds,
        timebase,
        minimum=minimum,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw_ticks = np.float32(seconds) / np.float32(timebase.dt_phys)
        rounded_ticks = np.rint(raw_ticks)
    if not np.isfinite(rounded_ticks) or rounded_ticks > _FLOAT32_EXACT_COUNTER_MAX:
        raise ValueError(
            f"{name} is outside the exact normalized-State counter horizon "
            f"({_FLOAT32_EXACT_COUNTER_MAX})"
        )
    return max(minimum, int(rounded_ticks))


def _validate_state_timers(
    timebase: Timebase,
    contact_timing: ContactTiming,
    goalkeeper_holding: GoalkeeperHolding,
    restart_timing: RestartTiming,
) -> None:
    """Validate every public duration installed in a normalized State timer."""

    _nearest_physics_ticks(
        "contact_timing.active_contact_interval_s",
        contact_timing.active_contact_interval_s,
        timebase,
    )
    _nearest_physics_ticks(
        "contact_timing.possession_loss_lock_s",
        contact_timing.possession_loss_lock_s,
        timebase,
    )

    # These three counters are formed inside JAX with float32 arithmetic and
    # rint.  Reproduce that arithmetic on the host, including the reachable
    # full-lunge sum, rather than validating each coefficient independently.
    challenge_configured_s = (
        contact_timing.challenge_recovery_s + contact_timing.max_lunge_extra_recovery_s
    )
    with np.errstate(over="ignore", invalid="ignore"):
        challenge_max_s = float(
            np.float32(contact_timing.challenge_recovery_s)
            + np.float32(contact_timing.max_lunge_extra_recovery_s)
        )
    _f32_rint_physics_ticks(
        "contact_timing.challenge_recovery_s + max_lunge_extra_recovery_s",
        challenge_max_s,
        timebase,
        configured_seconds=challenge_configured_s,
    )
    _f32_rint_physics_ticks(
        "contact_timing.aerial_attempt_recovery_s",
        contact_timing.aerial_attempt_recovery_s,
        timebase,
    )
    _f32_rint_physics_ticks(
        "contact_timing.goalkeeper_dive_recovery_s",
        contact_timing.goalkeeper_dive_recovery_s,
        timebase,
    )

    try:
        release_ticks = timebase.physics_steps_for(
            restart_timing.forced_release_delay_s,
            rounding="ceil",
            minimum=1,
        )
        hold_ticks = timebase.physics_steps_for(
            goalkeeper_holding.hand_control_limit_s,
            rounding="floor",
        )
    except (OverflowError, ValueError) as error:
        raise ValueError(
            "restart timing is outside the exact normalized-State counter horizon"
        ) from error
    overflow = {
        name: ticks
        for name, ticks in (
            ("restart_timing.forced_release_delay_s", release_ticks),
            ("goalkeeper_holding.hand_control_limit_s", hold_ticks),
        )
        if ticks > _FLOAT32_EXACT_COUNTER_MAX
    }
    if overflow:
        raise ValueError(
            "restart timing exceeds the exact normalized-State counter "
            f"horizon ({_FLOAT32_EXACT_COUNTER_MAX}): {overflow}"
        )
    if release_ticks >= hold_ticks:
        raise ValueError(
            "forced restart release must precede the goalkeeper holding "
            "limit after physics-tick quantization"
        )


def _validate_float32_envelope(
    timebase, boundary_margin_m, ball, stadium, action, player, physics
) -> None:
    dt = timebase.dt_phys
    release = max(action.kick_speed_max_mps, action.throw_speed_max_mps)
    speed = release + _f32_product("gravity increment", physics.g, dt)
    candidate = speed + release
    candidate_squared = _f32_product("contact squared norm", candidate, candidate)
    maximum_height = ball.radius + candidate_squared / (2.0 * physics.g)
    path = _f32_product("ball substep path", speed, dt)
    world_span = math.hypot(
        stadium.length + boundary_margin_m + ball.radius,
        stadium.width + boundary_margin_m + ball.radius,
        maximum_height,
    )
    origin_squared = _f32_product("world-domain squared norm", world_span, world_span)
    path_squared = _f32_product("ball path squared norm", path, path)
    _f32_product("swept-contact quartic term", 4.0, origin_squared, path_squared)
    maximum_spin = max(action.spin_max_radps, speed / ball.radius)
    _f32_product("spin squared norm", maximum_spin, maximum_spin)
    surface_speed = _f32_product("radius-spin speed", ball.radius, maximum_spin)
    _f32_product("radius-spin squared norm", surface_speed, surface_speed)
    angular_impulse_denominator = _f32_product(
        "ground angular-impulse denominator",
        physics.ball_inertia_ratio,
        ball.radius,
    )
    _f32_product("ground angular-impulse reciprocal", 1.0 / angular_impulse_denominator)
    ground_curl_rate = _f32_product(
        "ground curl angular rate", physics.c_ground_curl, maximum_spin
    )
    _f32_product("ground curl angle per physics tick", ground_curl_rate, dt)
    _f32_product(
        "maximum tangential bounce impulse",
        1.0 + physics.bounce_tangential_e,
        physics.ball_inertia_ratio / (1.0 + physics.ball_inertia_ratio),
        surface_speed,
    )
    area = _f32_product("ball area", math.pi, ball.radius, ball.radius)
    force_scale = _f32_product(
        "aerodynamic scale",
        0.5,
        physics.air_density_kgpm3,
        area,
        1.0 / physics.ball_mass_kg,
    )
    spin_parameter = max(surface_speed / physics.drag_crisis_speed_mps, 1.0)
    try:
        spin_drag = physics.spin_drag_scale * (
            math.pow(spin_parameter, physics.spin_drag_exponent)
            if speed > physics.drag_crisis_speed_mps
            else 0.0
        )
    except OverflowError as error:
        raise ValueError("spinning drag is outside float32") from error
    drag = max(
        physics.drag_coefficient_high_re + physics.drag_crisis_drop,
        spin_drag,
    )
    _f32_product("aerodynamic drag step", force_scale, drag, speed, dt)
    _f32_product(
        "aerodynamic lift step",
        force_scale,
        physics.lift_coefficient_limit,
        speed,
        dt,
    )
    acceleration = min(
        player.forward_acceleration_mps2,
        player.lateral_acceleration_mps2,
    )
    player_step = _f32_product("player displacement", acceleration, dt, dt)
    ball_step = _f32_product(
        "restart-ball displacement", action.restart_min_ball_speed_mps, dt
    )
    if _edge_ulp(stadium, boundary_margin_m) >= min(player_step, ball_step):
        raise ValueError("stadium edge float32 ULP erases configured movement")


def validate_environment_configuration(
    *,
    timebase: Timebase,
    boundary_margin_m: float,
    ball: Ball,
    stadium: Stadium,
    reach: Reach,
    action_scale: ActionScale,
    contact_timing: ContactTiming,
    contest: Contest,
    body_foul: BodyFoul,
    goalkeeper_holding: GoalkeeperHolding,
    restart_timing: RestartTiming,
    player_physics: PlayerPhysics,
    body: BodyContact,
    long_stamina: LongStamina,
    short_stamina: ShortStamina,
    ball_physics: BallPhysics,
    perception: Perception,
) -> None:
    """Validate static public configuration once, outside traced rollout."""

    configs = {
        "timebase": (timebase, Timebase),
        "ball": (ball, Ball),
        "stadium": (stadium, Stadium),
        "reach": (reach, Reach),
        "action_scale": (action_scale, ActionScale),
        "contact_timing": (contact_timing, ContactTiming),
        "contest": (contest, Contest),
        "body_foul": (body_foul, BodyFoul),
        "goalkeeper_holding": (goalkeeper_holding, GoalkeeperHolding),
        "restart_timing": (restart_timing, RestartTiming),
        "player_physics": (player_physics, PlayerPhysics),
        "body": (body, BodyContact),
        "long_stamina": (long_stamina, LongStamina),
        "short_stamina": (short_stamina, ShortStamina),
        "ball_physics": (ball_physics, BallPhysics),
        "perception": (perception, Perception),
    }
    for name, (value, expected) in configs.items():
        if type(value) is not expected:
            raise TypeError(f"{name} must be exactly {expected.__name__}")
        _fields(name, value, _FIELD_SPECS.get(expected, ()))
    _real("boundary_margin_m", boundary_margin_m, _NONNEGATIVE)
    if body_foul.probability_floor > body_foul.probability_ceiling:
        raise ValueError(
            "body_foul.probability_floor must not exceed probability_ceiling"
        )
    if np.float32(ball.radius) < np.float32(GEOMETRY_EPS):
        raise ValueError(
            "ball.radius must be at least GEOMETRY_EPS so collision and "
            "bounce denominators remain representable"
        )
    if not GEOMETRY_EPS <= ball_physics.ball_inertia_ratio <= 1.0:
        raise ValueError(
            "ball_physics.ball_inertia_ratio must lie between GEOMETRY_EPS "
            "and 1 so angular-impulse denominators remain representable"
        )
    if body.torso_depth_m > body.shoulder_width_m:
        raise ValueError("body.torso_depth_m must not exceed body.shoulder_width_m")
    if body.leg_apex_height_factor >= body.torso_top_height_factor:
        raise ValueError(
            "body.leg_apex_height_factor must remain below torso_top_height_factor"
        )
    if 2.0 * body.leg_radius_m >= body.shoulder_width_m:
        raise ValueError("twice body.leg_radius_m must remain below shoulder_width_m")
    _real(
        "action_scale.ground_launch_down_reference_height_m",
        action_scale.ground_launch_down_reference_height_m,
        (ball.radius, None, True, False),
    )
    if type(perception.limit_by_view_angle) is not bool:
        raise TypeError("perception.limit_by_view_angle must be a bool")
    if reach.goalkeeper_standing_radius_m > reach.goalkeeper_radius_m:
        raise ValueError(
            "reach.goalkeeper_standing_radius_m must not exceed goalkeeper_radius_m"
        )
    _integer(
        "player_physics.separation_iterations",
        player_physics.separation_iterations,
        1,
        16,
    )

    if not (
        stadium.goal_width
        <= stadium.goal_area_width
        <= stadium.penalty_area_width
        <= stadium.width
        and stadium.goal_area_length
        <= stadium.penalty_area_length
        < 0.5 * stadium.length
    ):
        raise ValueError("stadium dimensions have an invalid nesting")
    ball_diameter_f32 = np.float32(2.0) * np.float32(ball.radius)
    if (
        np.float32(stadium.goal_width) <= ball_diameter_f32
        or np.float32(stadium.goal_height) <= ball_diameter_f32
    ):
        raise ValueError(
            "goal aperture must remain wider and taller than the ball "
            "diameter in float32"
        )
    restart_law = RestartPositionLaw()
    # The fixed corner liveness fallback packs at most 21 movable players in
    # three rings of eight around the pinned taker. Reproduce its maximum
    # centre radius and then reserve the largest possible capsule support.
    # This is a sufficient in-pitch layout bound, not a claim that every local
    # resolver candidate succeeds for every custom formation.
    with np.errstate(over="ignore", invalid="ignore"):
        corner_packing_base = np.float32(
            restart_law.ordinary_clearance_m
            + stadium.corner_arc_radius
            + body.shoulder_width_m
        )
        corner_packing_step = np.float32(body.shoulder_width_m + body.torso_depth_m)
        corner_outer_centre = np.float32(
            corner_packing_base + np.float32(2.0) * corner_packing_step
        )
        maximum_capsule_support = np.float32(0.5 * body.shoulder_width_m)
        corner_outer_body = np.float32(corner_outer_centre + maximum_capsule_support)
        minimum_corner_angle = np.float32(np.pi / 32.0)
        corner_near_boundary_inset = np.float32(
            corner_packing_base * np.sin(minimum_corner_angle)
        )
        corner_available_span = np.float32(min(stadium.length, stadium.width))
    if (
        not np.isfinite(corner_near_boundary_inset)
        or corner_near_boundary_inset <= maximum_capsule_support
    ):
        raise ValueError(
            "innermost corner-packing ring cannot contain the player body "
            "inside the adjacent goal and touch lines"
        )
    if not np.isfinite(corner_outer_body) or corner_outer_body >= corner_available_span:
        raise ValueError(
            "outermost corner-packing ring cannot contain the player body "
            "inside the opposite goal and touch lines"
        )
    if body.shoulder_width_m < body.torso_depth_m:
        raise ValueError("body shoulder width must not be below torso depth")
    required_half_length, required_half_width = kickoff_packing_minimum_half_extents(
        ball=ball,
        stadium=stadium,
        body=body,
    )
    if (
        stadium.half_length < required_half_length
        or stadium.half_width < required_half_width
    ):
        raise ValueError(
            "stadium is too small for the fixed maximum-roster kickoff packing "
            f"(requires half extents at least {required_half_length:.6g} x "
            f"{required_half_width:.6g} m)"
        )
    if action_scale.control_request_speed_max_mps > action_scale.kick_speed_max_mps:
        raise ValueError("control request speed must not exceed kick speed")
    if action_scale.ground_launch_down_max_radians > action_scale.launch_max_radians:
        raise ValueError("ground downward launch must not exceed launch maximum")
    if action_scale.restart_min_ball_speed_mps > min(
        action_scale.kick_speed_max_mps, action_scale.throw_speed_max_mps
    ):
        raise ValueError("restart minimum speed exceeds a release-speed maximum")
    if reach.carry_radius_m > reach.challenge_radius_m:
        raise ValueError("carry radius must not exceed challenge radius")
    if restart_timing.forced_release_delay_s >= goalkeeper_holding.hand_control_limit_s:
        raise ValueError(
            "forced restart release must precede the goalkeeper holding limit"
        )
    for name in (
        "continuous_freekick_approach",
        "continuous_offside_approach",
        "continuous_corner_approach",
        "continuous_throwin_approach",
    ):
        if type(getattr(restart_timing, name)) is not bool:
            raise TypeError(f"restart_timing.{name} must be a bool")
    if not 0.0 < long_stamina.end_frac < long_stamina.tail_knee < 1.0:
        raise ValueError(
            "long_stamina end_frac and tail_knee must satisfy "
            "0 < end_frac < tail_knee < 1"
        )
    if ball_physics.bounce_spin_vmin < ball_physics.ground_settle_vz:
        raise ValueError("bounce_spin_vmin must not be below ground_settle_vz")
    _validate_state_timers(
        timebase,
        contact_timing,
        goalkeeper_holding,
        restart_timing,
    )
    _validate_stamina_float32_envelope(timebase, long_stamina)
    _validate_bounce_energy_admissibility(ball_physics)
    _validate_roll_table(
        ball_physics,
        required_speed=max(
            action_scale.kick_speed_max_mps,
            action_scale.throw_speed_max_mps,
        ),
    )
    _validate_float32_envelope(
        timebase,
        boundary_margin_m,
        ball,
        stadium,
        action_scale,
        player_physics,
        ball_physics,
    )


def _validate_roll_table(ball_physics: BallPhysics, *, required_speed: float) -> None:
    speed, distance = ball_physics.roll_v_knots, ball_physics.roll_d_knots
    if type(speed) is not tuple or type(distance) is not tuple:
        raise TypeError("ball rolling knots must be tuples")
    if max(len(speed), len(distance)) > MAX_ROLLING_TABLE_KNOTS:
        raise ValueError("ball rolling knot tuples must contain at most 256 values")
    if len(speed) != len(distance) or len(speed) < 2:
        raise ValueError("ball rolling knot tuples need equal length of at least two")
    speed = tuple(
        _real(f"ball_physics.roll_v_knots[{i}]", value, _NONNEGATIVE)
        for i, value in enumerate(speed)
    )
    for i, value in enumerate(distance):
        _real(f"ball_physics.roll_d_knots[{i}]", value, _POSITIVE)
    if speed[0] != 0.0 or any(b <= a for a, b in pairwise(speed)):
        raise ValueError("roll_v_knots must start at zero and increase strictly")
    speed_f32 = np.asarray(speed, dtype=np.float32)
    if np.any(speed_f32[:-1] >= speed_f32[1:]):
        raise ValueError("roll_v_knots must remain strictly increasing in float32")
    if speed[-1] < required_speed:
        raise ValueError("rolling speed knots must cover generated release speeds")


def validate_formation_inside_pitch(
    position: ArrayLike, *, stadium: Stadium, name: str
):
    centres = np.asarray(position, dtype=np.float64)
    outside = (np.abs(centres[:, 0]) > stadium.half_length) | (
        np.abs(centres[:, 1]) > stadium.half_width
    )
    if np.any(outside):
        slots = tuple(int(i) for i in np.flatnonzero(outside))
        raise ValueError(
            f"{name} player centres must lie on or inside the pitch; "
            f"invalid slots: {slots}"
        )


def validate_public_rosters(
    team_0: tuple[Player, ...],
    team_1: tuple[Player, ...],
    *,
    minimum_team_players: tuple[int, int],
    body: BodyContact,
    timebase: Timebase,
    stadium: Stadium,
    boundary_margin_m: float,
    roster_sampling: RosterSampling,
) -> None:
    edge_ulp = _edge_ulp(stadium, boundary_margin_m)
    _validate_player_collision_microstep_budget(
        timebase=timebase,
        body=body,
        maximum_profile_speed_mps=roster_sampling.max_max_speed_mps,
    )
    seen: set[int] = set()
    for team, roster in enumerate((team_0, team_1)):
        minimum = minimum_team_players[team]
        if not minimum <= len(roster) <= IFAB_MAX_TEAM_PLAYERS:
            raise ValueError(
                f"team_{team} must contain between {minimum} and "
                f"{IFAB_MAX_TEAM_PLAYERS} players, got {len(roster)}"
            )
        goalkeepers = 0
        for slot, player in enumerate(roster):
            prefix = f"team_{team} slot {slot}"
            if type(player) is not Player:
                raise TypeError(f"{prefix} must be exactly Player")
            profile = player.profile
            if type(profile) is not PlayerProfile:
                raise TypeError(f"{prefix} profile must be exactly PlayerProfile")
            player_id = _integer(
                f"{prefix} player_id",
                profile.player_id,
                0,
                np.iinfo(np.int32).max,
            )
            if player_id in seen:
                raise ValueError(f"duplicate player_id across rosters: {player_id}")
            seen.add(player_id)
            if type(profile.is_goalkeeper) is not bool:
                raise TypeError(f"{prefix} is_goalkeeper must be a bool")
            goalkeepers += int(profile.is_goalkeeper)
            _fields(
                prefix,
                profile,
                (
                    (
                        _POSITIVE,
                        "max_speed_mps height_m max_reach_height_m endurance_factor",
                    ),
                    (_UNIT, "ball_control"),
                ),
            )
            _f32_product(
                f"{prefix} max-speed squared norm",
                profile.max_speed_mps,
                profile.max_speed_mps,
            )
            if profile.max_speed_mps * timebase.dt_phys <= edge_ulp:
                raise ValueError(f"{prefix} movement is below stadium edge ULP")
            if not bool(
                player_profile_values_valid(
                    player_id,
                    np.float32(profile.max_speed_mps),
                    np.float32(profile.height_m),
                    np.float32(profile.max_reach_height_m),
                    np.float32(profile.ball_control),
                    np.float32(profile.endurance_factor),
                    head_radius=np.float32(body.head_radius_m),
                )
            ):
                raise ValueError(f"{prefix} profile values are outside their domain")
            reach_margin = profile.max_reach_height_m - profile.height_m
            normalized_domain = (
                roster_sampling.min_max_speed_mps
                <= profile.max_speed_mps
                <= roster_sampling.max_max_speed_mps
                and roster_sampling.min_height_m
                <= profile.height_m
                <= roster_sampling.max_height_m
                and roster_sampling.min_reach_margin_m
                <= reach_margin
                <= roster_sampling.max_reach_margin_m
                and roster_sampling.min_ball_control
                <= profile.ball_control
                <= roster_sampling.max_ball_control
                and roster_sampling.min_endurance_factor
                <= profile.endurance_factor
                <= roster_sampling.max_endurance_factor
            )
            if not normalized_domain:
                raise ValueError(
                    f"{prefix} profile exceeds the configured model-normalization domain"
                )
            try:
                position = np.asarray(player.initial_position, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{prefix} initial_position must have shape (2,) and be finite"
                ) from error
            if position.shape != (2,):
                raise ValueError(f"{prefix} initial_position must have shape (2,)")
            if not np.all(np.isfinite(position)):
                raise ValueError(f"{prefix} initial_position must be finite")
            with np.errstate(over="ignore"):
                position_f32 = np.asarray(position, dtype=np.float32)
            if not np.all(np.isfinite(position_f32)):
                raise ValueError(f"{prefix} initial_position must be finite in float32")
        if goalkeepers != 1:
            raise ValueError(
                f"team_{team} must contain exactly one goalkeeper, got {goalkeepers}"
            )


def validate_kickoff_team(value: object) -> None:
    _integer("kickoff_team", value, TEAM_0, TEAM_1)


def validate_no_torso_overlap(
    position: ArrayLike,
    facing: ArrayLike,
    *,
    body: BodyContact,
    name: str,
) -> None:
    """Check the exact initial capsules, whose facings are always 0 or pi."""

    centres = np.asarray(position, dtype=np.float64)
    angles = np.asarray(facing, dtype=np.float64)
    if not np.allclose(np.sin(angles), 0.0, atol=1.0e-6):
        raise ValueError(f"{name} facing must align with a half's attack direction")
    separation = np.abs(centres[:, None, :] - centres[None, :, :])
    core_length = body.shoulder_width_m - body.torso_depth_m
    gap_x = separation[..., 0]
    gap_y = np.maximum(separation[..., 1] - core_length, 0.0)
    overlap = gap_x * gap_x + gap_y * gap_y < body.torso_depth_m**2 - 1.0e-12
    first, second = np.where(np.triu(overlap, k=1))
    pairs = tuple(zip(first.tolist(), second.tolist(), strict=True))
    if pairs:
        raise ValueError(f"{name} torso capsules overlap: {pairs}")
