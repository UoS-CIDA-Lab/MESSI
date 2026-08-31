"""Host-side validation for immutable soccer dynamics configuration."""

from __future__ import annotations

import math
import numbers
from dataclasses import fields

import numpy as np

from .config import (
    MAX_POSITION_SOLVER_ROUNDS,
    MAX_ROLLING_TABLE_KNOTS,
    Ball,
    Engine,
    Foul,
    Reward,
    Stadium,
)
from .constants import GEOMETRY_EPS, PROB_EPS

# Contact sweeps contain quartic intermediates such as ``(rel dot segment)^2``
# and ``|segment|^2 * |rel|^2``.  Merely keeping components or even their
# squared norms finite is insufficient.  Bound every public dynamics magnitude
# by the fourth root of float32_max with fourfold headroom for vector
# differences and one-tick composition.  The resulting ~1.07e9 SI-unit limit
# is still far beyond any football configuration while keeping those raw XLA
# products finite.
_FLOAT32_DYNAMICS_MAGNITUDE_LIMIT = (
    float(np.finfo(np.float32).max) ** 0.25 / 4.0
)

# Largest consecutive non-negative integer representable by the float32 State
# and compact-vector contract.  Several physical clocks are stored as int32 but
# exposed as ``counter / configured_window`` float32 ratios; challenge cooldown
# is itself stored as float32.  Above 2**24 adjacent ticks collapse, so a compact
# vector can hide a release boundary and ``cooldown - 1`` can become an exact
# no-op.  Keep one shared ceiling for every such counter rather than validating
# only the most visible restart timer.
_FLOAT32_EXACT_COUNTER_MAX = 1 << 24

def _float32_position_resolution(stadium, engine) -> float:
    """Four-ULP world-coordinate displacement required by public mechanics.

    Finite float32 arithmetic is not sufficient for a usable world.  At a
    coordinate around 5e7, adjacent float32 values are four metres apart: a
    normal player's velocity can change while ``position += dt * velocity``
    remains bitwise stationary forever.  Measure the worst active-player
    coordinate and reserve four representable steps for geometry/motion.
    """

    extent = max(
        float(stadium.half_length) + float(engine.player_boundary_margin),
        float(stadium.half_width) + float(engine.player_boundary_margin),
    )
    extent32 = np.float32(extent)
    ulp = float(
        np.nextafter(extent32, np.float32(np.inf), dtype=np.float32)
        - extent32
    )
    return 4.0 * ulp


def _require_float32_scalar(name: str, value) -> float:
    """Validate one public real scalar against the runtime float32 contract.

    Python/NumPy can represent finite values that become either infinity or
    signed zero when captured by JAX with x64 disabled.  Accepting them makes
    validation and fingerprints describe a value the transition never uses.
    Return the Python float only as a convenience for callers doing further
    host-side checks.
    """

    value_f64 = float(value)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value_f32 = np.float32(value_f64)
    if not np.isfinite(value_f32):
        raise ValueError(
            f"{name} must be finite in the float32 dynamics contract, got {value!r}"
        )
    if value_f64 != 0.0 and float(value_f32) == 0.0:
        raise ValueError(
            f"{name} underflows to zero in the float32 dynamics contract: {value!r}"
        )
    return value_f64


def _validate_rolling_tables_and_types(
    b: Ball,
    s: Stadium,
    e: Engine,
    f: Foul,
    r: Reward,
) -> None:
    """Validate bounded rolling tables and declared scalar/container types."""
    # Preflight rolling-table containers and cardinality before the generic
    # annotation walk visits a single element or NumPy/JAX materialises an
    # array.  These tuples become static interpolation constants in both
    # the per-physics-tick ball kernel and rule-policy graph, so merely
    # checking equal/nonempty tables after an O(length) scan leaves an
    # effectively unbounded public compile/runtime input.
    rolling_lengths = {}
    for table_name, table in (
        ("roll_v_knots", e.roll_v_knots),
        ("roll_d_knots", e.roll_d_knots),
    ):
        try:
            table_length = len(table)
        except TypeError:
            table_length = None
        rolling_lengths[table_name] = table_length
        if (table_length is not None
                and table_length > MAX_ROLLING_TABLE_KNOTS):
            raise ValueError(
                f"engine.{table_name} must contain at most "
                f"MAX_ROLLING_TABLE_KNOTS={MAX_ROLLING_TABLE_KNOTS} "
                f"knots, got {table_length}"
            )
    if not isinstance(e.roll_v_knots, tuple) or not isinstance(
        e.roll_d_knots, tuple
    ):
        raise ValueError(
            "engine.roll_v_knots and engine.roll_d_knots must be tuples"
        )
    rolling_knot_count = rolling_lengths["roll_v_knots"]
    if (rolling_knot_count != rolling_lengths["roll_d_knots"]
            or rolling_knot_count < 2):
        raise ValueError(
            "roll_v_knots and roll_d_knots must have the same length >= 2"
        )
    for group_name, cfg in (
        ("ball", b), ("stadium", s), ("engine", e),
        ("foul", f), ("reward", r),
    ):
        # Dataclass annotations are an executable public contract, not just
        # documentation.  Python ``bool`` is an ``int`` subclass, so range
        # checks alone silently accepted values such as
        # ``offside_margin=True`` (one metre) and ``advance=True`` (1.0).
        # Reject the wrong scalar kind before any coercion or geometry check
        # so callers get the actual cause and fingerprints cannot encode a
        # configuration with accidental boolean physics.
        for declared in fields(cfg):
            value = getattr(cfg, declared.name)
            expected = declared.type
            valid_type = True
            if expected is float:
                valid_type = (
                    isinstance(value, numbers.Real)
                    and not isinstance(value, (bool, np.bool_))
                )
            elif expected is int:
                valid_type = (
                    isinstance(value, numbers.Integral)
                    and not isinstance(value, (bool, np.bool_))
                )
            elif expected is bool:
                valid_type = isinstance(value, (bool, np.bool_))
            elif expected is str:
                valid_type = isinstance(value, str)
            elif expected is tuple:
                valid_type = (
                    isinstance(value, tuple)
                    and all(
                        isinstance(item, numbers.Real)
                        and not isinstance(item, (bool, np.bool_))
                        for item in value
                    )
                )
            elif declared.name == "norm_spin":
                # ``float | None`` is resolved to a concrete float by
                # Engine.__post_init__ before validation.
                valid_type = (
                    isinstance(value, numbers.Real)
                    and not isinstance(value, (bool, np.bool_))
                )
            if not valid_type:
                raise ValueError(
                    f"{group_name}.{declared.name} has the wrong type for "
                    f"{expected!r}: {value!r}"
                )

            # The dynamics contract is float32 even though public Python
            # configuration accepts Real scalar wrappers.  A value may be
            # finite as a Python float yet overflow (1e300 -> inf) or
            # underflow (1e-300 -> 0) when captured by a JAX computation.
            # Both cases previously passed construction and could turn
            # otherwise harmless expressions such as ``gain * 0`` into
            # NaN, or silently erase a positive physical scale.  Reject
            # those values at the public boundary before fingerprinting.
            float_values = ()
            if expected is float or declared.name == "norm_spin":
                float_values = (value,)
            elif expected is tuple:
                float_values = tuple(value)
            for item in float_values:
                _require_float32_scalar(
                    f"{group_name}.{declared.name}", item
                )
                if abs(float(item)) > _FLOAT32_DYNAMICS_MAGNITUDE_LIMIT:
                    raise ValueError(
                        f"{group_name}.{declared.name} exceeds the float32 "
                        "quartic/norm-safe dynamics magnitude limit"
                    )
        for name, value in vars(cfg).items():
            if isinstance(value, (int, float, np.number)) and not isinstance(
                value, (bool, np.bool_)
            ):
                if not math.isfinite(float(value)):
                    raise ValueError(f"{group_name}.{name} must be finite, got {value!r}")


def _validate_positive_and_geometry(
    b: Ball,
    s: Stadium,
    e: Engine,
    f: Foul,
    r: Reward,
) -> None:
    """Validate positive quantities and mutually constrained geometry."""
    positive = {
        "ball.radius": b.radius,
        "stadium.width": s.width,
        "stadium.length": s.length,
        "stadium.goal_width": s.goal_width,
        "stadium.goal_height": s.goal_height,
        "stadium.penalty_area_length": s.penalty_area_length,
        "stadium.penalty_area_width": s.penalty_area_width,
        "stadium.goal_area_length": s.goal_area_length,
        "stadium.goal_area_width": s.goal_area_width,
        "stadium.center_circle_radius": s.center_circle_radius,
        "stadium.penalty_arc_radius": s.penalty_arc_radius,
        "stadium.corner_arc_radius": s.corner_arc_radius,
        "engine.dt_phys": e.dt_phys,
        "engine.r_player": e.r_player,
        "engine.bench_first_x_offset": e.bench_first_x_offset,
        "engine.bench_spacing": e.bench_spacing,
        "engine.bench_boundary_inset": e.bench_boundary_inset,
        "engine.bench_touchline_inset": e.bench_touchline_inset,
        "engine.a_max": e.a_max,
        "engine.accel_norm_max": e.accel_norm_max,
        "engine.brake_decel_max": e.brake_decel_max,
        "engine.reach_xy_carry": e.reach_xy_carry,
        "engine.reach_xy_challenge": e.reach_xy_challenge,
        "engine.reach_height_factor": e.reach_height_factor,
        "engine.gk_reach_xy": e.gk_reach_xy,
        "engine.gk_backpass_target_radius": e.gk_backpass_target_radius,
        "engine.gk_catch_speed_cap": e.gk_catch_speed_cap,
        "engine.parry_punch_speed_cap": e.parry_punch_speed_cap,
        "engine.parry_tip_keep": e.parry_tip_keep,
        "engine.parry_tip_clearance": e.parry_tip_clearance,
        "engine.parry_tip_lift_max": e.parry_tip_lift_max,
        "engine.parry_tip_max_range": e.parry_tip_max_range,
        "engine.kicker_arrive_r": e.kicker_arrive_r,
        "engine.goalkick_depth": e.goalkick_depth,
        "engine.goalkick_lateral_offset": e.goalkick_lateral_offset,
        "engine.throwin_clear": e.throwin_clear,
        "engine.clear_dist": e.clear_dist,
        "engine.kicker_speed": e.kicker_speed,
        "engine.contest_temp": e.contest_temp,
        "engine.cooldown_s": e.cooldown_s,
        "engine.contact_interval_s": e.contact_interval_s,
        "engine.challenge_cooldown_extra_s": e.challenge_cooldown_extra_s,
        "engine.ctrl_lock_s": e.ctrl_lock_s,
        "engine.restart_s": e.restart_s,
        "engine.setup_hold_s": e.setup_hold_s,
        "engine.throwin_restart_delay_s": e.throwin_restart_delay_s,
        "engine.goalkick_restart_delay_s": e.goalkick_restart_delay_s,
        "engine.corner_restart_delay_s": e.corner_restart_delay_s,
        "engine.freekick_restart_delay_s": e.freekick_restart_delay_s,
        "engine.penalty_restart_delay_s": e.penalty_restart_delay_s,
        "engine.offside_restart_delay_s": e.offside_restart_delay_s,
        "engine.penalty_s": e.penalty_s,
        "engine.gk_hold_s": e.gk_hold_s,
        "engine.f2b_speed_max": e.f2b_speed_max,
        "engine.spin_max": e.spin_max,
        "engine.launch_max": e.launch_max,
        "engine.g": e.g,
        "engine.z_ground": e.z_ground,
        "engine.ground_settle_vz": e.ground_settle_vz,
        "engine.charge_speed": e.charge_speed,
        "engine.ball_inertia_ratio": e.ball_inertia_ratio,
        "engine.bounce_spin_vmin": e.bounce_spin_vmin,
        "engine.intercept_speed": e.intercept_speed,
        "engine.deflect_stationary_speed": e.deflect_stationary_speed,
        "engine.dribble_speed_max": e.dribble_speed_max,
        "engine.collide_speed_min": e.collide_speed_min,
        "engine.leg_top": e.leg_top,
        "engine.body_r": e.body_r,
        "engine.trap_speed_ref": e.trap_speed_ref,
        "engine.trap_drop_distance": e.trap_drop_distance,
        "engine.sprint_speed": e.sprint_speed,
        "engine.long_stamina_sprint_mult": e.long_stamina_sprint_mult,
        "engine.long_stamina_idle_load": e.long_stamina_idle_load,
        "engine.long_stamina_speed_ref": e.long_stamina_speed_ref,
        "engine.long_stamina_speed_load": e.long_stamina_speed_load,
        "engine.long_stamina_accel_ref": e.long_stamina_accel_ref,
        "engine.long_stamina_accel_load": e.long_stamina_accel_load,
        "engine.long_stamina_vmax_floor": e.long_stamina_vmax_floor,
        "engine.long_stamina_end_frac": e.long_stamina_end_frac,
        "engine.long_stamina_tail_knee": e.long_stamina_tail_knee,
        "engine.long_stamina_reference_duration_s": e.long_stamina_reference_duration_s,
        "engine.long_stamina_reference_workload": e.long_stamina_reference_workload,
        "engine.short_stamina_vmax_floor": e.short_stamina_vmax_floor,
        "engine.short_stamina_headroom_knee": e.short_stamina_headroom_knee,
        "engine.short_stamina_depletion_s": e.short_stamina_depletion_s,
        "engine.short_stamina_depletion_speed_frac": e.short_stamina_depletion_speed_frac,
        "engine.short_stamina_speed_exponent": e.short_stamina_speed_exponent,
        "engine.short_stamina_accel_ref": e.short_stamina_accel_ref,
        "engine.short_stamina_accel_load": e.short_stamina_accel_load,
        "engine.short_stamina_recovery_tau_s": e.short_stamina_recovery_tau_s,
        "engine.short_stamina_recovery_speed_frac": e.short_stamina_recovery_speed_frac,
        "engine.short_stamina_recovery_exponent": e.short_stamina_recovery_exponent,
        "engine.penalty_spot": e.penalty_spot,
        "engine.f2b_shoot_range": e.f2b_shoot_range,
        # 슛 라벨은 State(``touch``·``last_touch_code``)에 남으므로 그 폭도 동역학
        # 계약의 일부다. 종전 ``shot_aim_cos``는 지문에 빠져 있었다.
        "engine.shot_aim_mouth_scale": e.shot_aim_mouth_scale,
        "engine.throw_speed_max": e.throw_speed_max,
        "engine.restart_min_ball_speed": e.restart_min_ball_speed,
        "engine.throw_height": e.throw_height,
        "engine.goal_line_tolerance": e.goal_line_tolerance,
        "engine.goal_post_tolerance": e.goal_post_tolerance,
        "engine.legal_margin_floor": e.legal_margin_floor,
        "engine.unrestricted_margin": e.unrestricted_margin,
        "engine.norm_player_vel": e.norm_player_vel,
        "engine.norm_ball_vel": e.norm_ball_vel,
        "engine.norm_ball_z": e.norm_ball_z,
        "engine.norm_spin": e.norm_spin,
        "engine.norm_body_z": e.norm_body_z,
        "engine.norm_score": e.norm_score,
        "foul.clean_ball_dist": f.clean_ball_dist,
        "foul.clean_ball_speed": f.clean_ball_speed,
        "foul.balldist_ref": f.balldist_ref,
        "foul.tackle_contact_range": f.tackle_contact_range,
        "foul.charge_play_dist": f.charge_play_dist,
        "reward.goal": r.goal,
    }
    bad = [name for name, value in positive.items()
           if not math.isfinite(float(value)) or float(value) <= 0.0]
    if bad:
        raise ValueError(f"configuration values must be finite and positive: {', '.join(bad)}")
    if e.post_goal_kickoff_delay_s < 0.0:
        raise ValueError(
            "engine.post_goal_kickoff_delay_s must be finite and non-negative"
        )
    if b.radius < GEOMETRY_EPS:
        raise ValueError(
            "ball.radius must be at least GEOMETRY_EPS so contact and "
            "bounce denominators remain representable"
        )
    if e.r_player < GEOMETRY_EPS:
        raise ValueError(
            "engine.r_player must be at least GEOMETRY_EPS"
        )
    if e.ball_inertia_ratio < GEOMETRY_EPS:
        raise ValueError(
            "engine.ball_inertia_ratio must be at least GEOMETRY_EPS so "
            "the angular-impulse denominator cannot underflow"
        )
    if e.ball_inertia_ratio > 1.0:
        raise ValueError(
            "engine.ball_inertia_ratio=I/(m*r^2) must not exceed 1 for "
            "mass contained inside the ball radius"
        )
    if not (
        s.goal_width <= s.goal_area_width
        <= s.penalty_area_width <= s.width
    ):
        raise ValueError(
            "stadium widths must satisfy goal_width <= goal_area_width "
            "<= penalty_area_width <= width"
        )
    if s.goal_width <= 2.0 * b.radius or s.goal_height <= 2.0 * b.radius:
        raise ValueError("goal aperture must be wider and taller than the ball diameter")
    if not 0.0 <= e.player_boundary_margin < min(s.half_length, s.half_width):
        raise ValueError(
            "engine.player_boundary_margin must lie in [0, min(pitch half extents))"
        )
    if not (
        s.goal_area_length <= s.penalty_area_length < s.half_length
    ):
        raise ValueError(
            "stadium lengths must satisfy goal_area_length <= "
            "penalty_area_length < half_length"
        )
    if e.launch_down_ref <= b.radius:
        raise ValueError("engine.launch_down_ref must be greater than ball.radius")
    if e.z_ground < b.radius:
        raise ValueError("engine.z_ground must be at least ball.radius")
    if e.z_ground >= e.leg_top:
        raise ValueError(
            "engine.z_ground must lie below engine.leg_top; otherwise a "
            "torso-height ball is simultaneously treated as ground-supported"
        )
    if e.body_top_frac <= 0.0:
        raise ValueError("engine.body_top_frac must be positive")
    if e.trap_drop_distance < e.r_player + b.radius:
        raise ValueError(
            "engine.trap_drop_distance must be at least r_player + "
            "ball.radius so a trapped ball is not placed inside the player"
        )
    if e.penalty_spot >= s.penalty_area_length:
        raise ValueError(
            "engine.penalty_spot must lie strictly inside the penalty area"
        )
    if s.center_circle_radius >= min(s.half_length, s.half_width):
        raise ValueError(
            "stadium.center_circle_radius must fit strictly inside the pitch"
        )
    if s.penalty_arc_radius >= min(e.penalty_spot, s.half_width):
        raise ValueError(
            "stadium.penalty_arc_radius must fit between the penalty mark, "
            "goal line, and touchlines"
        )
    if s.corner_arc_radius >= min(s.half_length, s.half_width):
        raise ValueError(
            "stadium.corner_arc_radius must fit inside the pitch"
        )
    if math.sqrt(2.0) * e.restart_field_inset > s.corner_arc_radius:
        raise ValueError(
            "engine.restart_field_inset places the canonical corner ball "
            "outside stadium.corner_arc_radius"
        )
    if e.goalkick_depth > s.goal_area_length:
        raise ValueError(
            "engine.goalkick_depth must lie inside the goal area"
        )
    if e.goalkick_lateral_offset > s.goal_area_width / 2.0:
        raise ValueError(
            "engine.goalkick_lateral_offset must lie inside the goal area"
        )
    legal_radius = math.hypot(
        s.half_length + e.player_boundary_margin,
        s.half_width + e.player_boundary_margin,
    )
    player_domain_diameter = math.hypot(
        s.length + 2.0 * e.player_boundary_margin,
        s.width + 2.0 * e.player_boundary_margin,
    )
    if player_domain_diameter > _FLOAT32_DYNAMICS_MAGNITUDE_LIMIT:
        raise ValueError(
            "stadium player domain exceeds the float32 norm-safe geometry "
            "limit; squared distance calculations must remain representable"
        )
    position_resolution = _float32_position_resolution(s, e)
    resolution_critical = {
        "ball.radius": b.radius,
        "engine.r_player": e.r_player,
        "engine.legal_margin_floor": e.legal_margin_floor,
        "engine.a_max * dt_phys^2": e.a_max * e.dt_phys * e.dt_phys,
        "engine.kicker_speed * dt_phys": e.kicker_speed * e.dt_phys,
        "engine.restart_min_ball_speed * dt_phys": (
            e.restart_min_ball_speed * e.dt_phys
        ),
    }
    unresolved = {
        name: value for name, value in resolution_critical.items()
        if float(value) < position_resolution
    }
    if unresolved:
        raise ValueError(
            "configured geometry/motion is below the float32 spatial "
            f"resolution ({position_resolution:g} m at the world edge): "
            f"{unresolved}"
        )
    if max(
        e.clear_dist + s.corner_arc_radius,
        e.throwin_clear,
    ) + e.legal_margin_floor >= legal_radius:
        raise ValueError(
            "restart clearance radius leaves no guaranteed legal player "
            "position in the configured player domain"
        )
    if not e.reach_xy_carry <= e.reach_xy_challenge:
        raise ValueError("reach_xy_carry must not exceed reach_xy_challenge")
    if not 0.0 < e.reach_height_factor <= 1.0:
        raise ValueError("engine.reach_height_factor must lie in (0, 1]")
    if e.goal_line_tolerance >= min(e.clear_dist, s.penalty_area_length):
        raise ValueError(
            "engine.goal_line_tolerance must remain a local goal-line band "
            "smaller than clear_dist and penalty_area_length"
        )
    body_line_limit = e.r_player + e.legal_margin_floor
    if e.goal_line_tolerance > body_line_limit:
        raise ValueError(
            "engine.goal_line_tolerance cannot exceed r_player + "
            "legal_margin_floor; otherwise a player with no body part on "
            "the goal line is treated as legal"
        )
    if e.goal_post_tolerance > body_line_limit:
        raise ValueError(
            "engine.goal_post_tolerance cannot exceed r_player + "
            "legal_margin_floor; otherwise a player wholly outside the "
            "posts receives the goal-line exception"
        )
    if e.goal_post_tolerance >= (s.width - s.goal_width) / 2.0:
        raise ValueError(
            "engine.goal_post_tolerance must not extend the goal-line "
            "exemption across the whole pitch width"
        )
    if e.dt_phys > min(e.cooldown_s, e.ctrl_lock_s):
        raise ValueError(
            "engine.dt_phys must not exceed cooldown_s or ctrl_lock_s; "
            "their floor-rounded one-tick locks must remain upper bounds"
        )
    # ``f2b_dir`` is the horizontal bearing and ``launch`` is elevation
    # above that bearing.  At pi/2 horizontal motion vanishes; above it
    # ``cos(launch)`` reverses the requested bearing (including a Law-14
    # penalty direction already projected forward in contest.py).
    if e.launch_max >= math.pi / 2.0:
        raise ValueError("engine.launch_max must be strictly less than pi/2")
    if not 0.0 <= e.launch_down_ground <= e.launch_max:
        raise ValueError("launch_down_ground must lie in [0, launch_max]")


def _validate_float32_coupled_scales(
    b: Ball,
    e: Engine,
) -> None:
    """Validate float32 scales whose products appear in dynamics."""
    numerical_scales = {
        "engine.norm_player_vel": e.norm_player_vel,
        "engine.norm_ball_vel": e.norm_ball_vel,
        "engine.norm_ball_z": e.norm_ball_z,
        "engine.norm_spin": e.norm_spin,
        "engine.norm_body_z": e.norm_body_z,
        "engine.norm_score": e.norm_score,
        "engine.contest_temp": e.contest_temp,
    }
    too_small = {
        name: value for name, value in numerical_scales.items()
        if float(value) < PROB_EPS
    }
    if too_small:
        raise ValueError(
            "normalization and stochastic temperature scales must be at "
            f"least {PROB_EPS:g} in float32: {too_small}"
        )
    # Individual coefficients can all fit float32 while the exact product
    # used by a kernel does not (for example c_magnus=spin_max=1e30).
    # Validate the public coupled quantities before sin/cos or an impulse
    # sees infinity.  For positive per-tick effects the same check also
    # rejects a nonzero Python value that disappears to zero at runtime.
    coupled_float32 = {
        "engine.a_max * dt_phys": e.a_max * e.dt_phys,
        "engine.accel_norm_max * dt_phys": e.accel_norm_max * e.dt_phys,
        "engine.brake_decel_max * dt_phys": e.brake_decel_max * e.dt_phys,
        "engine.g * dt_phys": e.g * e.dt_phys,
        "engine.kicker_speed * dt_phys": e.kicker_speed * e.dt_phys,
        "engine.f2b_speed_max * dt_phys": e.f2b_speed_max * e.dt_phys,
        "engine.throw_speed_max * dt_phys": e.throw_speed_max * e.dt_phys,
        # Kick spin has one lateral/back-spin component and one side-spin
        # component, so its reachable vector norm is sqrt(2)*spin_max.
        # Check the multiplication *before* dt as well: XLA evaluates
        # ``c_magnus * spin`` first, and an overflowing intermediate does
        # not become finite again merely because dt < 1.
        "engine.c_magnus * spin_max": e.c_magnus * e.spin_max,
        "engine.c_magnus * spin_max * dt_phys": (
            math.sqrt(2.0) * e.c_magnus * e.spin_max * e.dt_phys
        ),
        "engine.c_ground_curl * spin_max": (
            e.c_ground_curl * e.spin_max
        ),
        "engine.c_ground_curl * spin_max * dt_phys": (
            e.c_ground_curl * e.spin_max * e.dt_phys
        ),
        "engine.spin_decay * dt_phys": e.spin_decay * e.dt_phys,
        "ball.radius * engine.spin_max": b.radius * e.spin_max,
        "engine.ball_inertia_ratio * ball.radius": (
            e.ball_inertia_ratio * b.radius
        ),
        "maximum tangential bounce impulse": (
            (1.0 + e.bounce_tangential_e)
            * (e.ball_inertia_ratio / (1.0 + e.ball_inertia_ratio))
            * b.radius
            * e.spin_max
        ),
    }
    for name, value in coupled_float32.items():
        _require_float32_scalar(name, value)
    bounded_coupled_magnitudes = {
        name: value for name, value in coupled_float32.items()
        if name != "engine.spin_decay * dt_phys"
    }
    # Gravity contributes both a velocity delta and a position displacement
    # during one public physics tick.
    bounded_coupled_magnitudes["engine.g * dt_phys^2"] = (
        e.g * e.dt_phys * e.dt_phys
    )
    oversized_coupled = {
        name: value for name, value in bounded_coupled_magnitudes.items()
        if abs(float(value)) > _FLOAT32_DYNAMICS_MAGNITUDE_LIMIT
    }
    if oversized_coupled:
        raise ValueError(
            "coupled physics quantities exceed the float32 quartic/norm-safe "
            f"dynamics magnitude limit: {oversized_coupled}"
        )
    if (
        max(e.f2b_speed_max, e.throw_speed_max, e.gk_catch_speed_cap)
        > _FLOAT32_DYNAMICS_MAGNITUDE_LIMIT
    ):
        raise ValueError(
            "configured ball speed caps exceed the float32 norm-safe limit; "
            "squared norms and opposite signed velocity differences must "
            "remain representable"
        )
    if e.launch_down_ref - b.radius < GEOMETRY_EPS:
        raise ValueError(
            "engine.launch_down_ref must exceed ball.radius by at least "
            "GEOMETRY_EPS"
        )


def _validate_timing_and_counters(
    e: Engine,
) -> None:
    """Validate derived timing windows, solver rounds, and counters."""
    windows = (e.restart_substeps, e.penalty_substeps, e.gk_hold_substeps)
    if any((not isinstance(v, int) or isinstance(v, bool) or v <= 0) for v in windows):
        raise ValueError("restart, penalty, and GK-hold windows must be positive integers")
    if not isinstance(e.setup_hold_substeps, int) or not 0 <= e.setup_hold_substeps < min(windows):
        raise ValueError("setup_hold_substeps must be an integer smaller than every restart window")
    restart_delays = {
        "throwin_restart_delay_substeps": e.throwin_restart_delay_substeps,
        "goalkick_restart_delay_substeps": e.goalkick_restart_delay_substeps,
        "corner_restart_delay_substeps": e.corner_restart_delay_substeps,
        "freekick_restart_delay_substeps": e.freekick_restart_delay_substeps,
        "offside_restart_delay_substeps": e.offside_restart_delay_substeps,
    }
    invalid_restart_delays = {
        name: value for name, value in restart_delays.items()
        if not isinstance(value, int) or isinstance(value, bool)
        or value <= 0 or value >= e.restart_substeps
    }
    if invalid_restart_delays:
        raise ValueError(
            "ordinary restart delays must be positive integer ticks "
            "strictly below restart_substeps: "
            f"{invalid_restart_delays}"
        )
    if (
        not isinstance(e.post_goal_kickoff_delay_substeps, int)
        or isinstance(e.post_goal_kickoff_delay_substeps, bool)
        or not 0 <= e.post_goal_kickoff_delay_substeps < e.restart_substeps
    ):
        raise ValueError(
            "post-goal kickoff delay must be non-negative integer ticks "
            "strictly below restart_substeps"
        )
    if (
        not isinstance(e.penalty_restart_delay_substeps, int)
        or isinstance(e.penalty_restart_delay_substeps, bool)
        or not 0 < e.penalty_restart_delay_substeps < e.penalty_substeps
    ):
        raise ValueError(
            "penalty restart delay must be positive integer ticks "
            "strictly below penalty_substeps"
        )
    if (not isinstance(e.sep_iters, numbers.Integral)
            or isinstance(e.sep_iters, (bool, np.bool_))
            or not 3 <= e.sep_iters <= MAX_POSITION_SOLVER_ROUNDS):
        raise ValueError(
            "engine.sep_iters must be an integer in "
            f"[3, {MAX_POSITION_SOLVER_ROUNDS}]; two Jacobi passes leave "
            "a measured 7.78 mm boundary-constrained overlap, while three reduce "
            "it to 2.40 mm at about 2.7% whole-step cost, and excessive "
            "Python-unrolled passes cause JIT graph blow-up without turning "
            "the local relaxation into an exact multi-body solver"
        )
    if (not isinstance(e.restart_slide_rounds, numbers.Integral)
            or isinstance(e.restart_slide_rounds, (bool, np.bool_))
            or not 6 <= e.restart_slide_rounds <= MAX_POSITION_SOLVER_ROUNDS):
        raise ValueError(
            "engine.restart_slide_rounds must be an integer in "
            f"[6, {MAX_POSITION_SOLVER_ROUNDS}]; fewer rounds leave "
            "projection-created player overlaps, while excessive O(N^2) "
            "sweeps cannot repair inconsistent multi-body geometry"
        )
    if (not math.isfinite(e.restart_separation_relaxation)
            or not 1.5 <= e.restart_separation_relaxation <= 2.0):
        raise ValueError(
            "engine.restart_separation_relaxation must lie in [1.5, 2]; "
            "lower relaxation does not converge within the six-round safety floor"
        )
    counters = {
        "cooldown_substeps": e.cooldown_substeps,
        "contact_lock_substeps": e.contact_lock_substeps,
        "ctrl_lock_substeps": e.ctrl_lock_substeps,
        "challenge_cooldown_extra": e.challenge_cooldown_extra,
    }
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in counters.values()):
        raise ValueError(f"engine counters must be positive integers: {counters}")
    int32_max = np.iinfo(np.int32).max
    all_state_counters = {
        **counters,
        "restart_substeps": e.restart_substeps,
        "setup_hold_substeps": e.setup_hold_substeps,
        **restart_delays,
        "penalty_substeps": e.penalty_substeps,
        "penalty_restart_delay_substeps": e.penalty_restart_delay_substeps,
        "gk_hold_substeps": e.gk_hold_substeps,
    }
    overflow = {
        name: value for name, value in all_state_counters.items()
        if value > int32_max
    }
    if overflow:
        raise ValueError(
            "engine tick counters exceed the int32 State contract: "
            f"{overflow}"
        )
    # These are not merely int32 implementation counters.  Each is either
    # represented directly in a float32 State leaf (cooldown), or encoded in
    # observation/state as a float32 ratio whose adjacent integer values must
    # remain distinguishable.  Letting a window exceed 2**24 produces two
    # compact vectors that are bitwise identical while one can release a
    # restart and the other cannot.  For cooldown it is worse: at 45,000,000
    # ticks, subtracting one is bitwise stationary and the actor remains
    # locked forever.  The maximum reachable cooldown includes the lunge
    # extension, so validate the sum rather than its two inputs separately.
    float32_exact_counters = {
        "restart_substeps": e.restart_substeps,
        "penalty_substeps": e.penalty_substeps,
        "gk_hold_substeps": e.gk_hold_substeps,
        "contact_lock_substeps": e.contact_lock_substeps,
        "ctrl_lock_substeps": e.ctrl_lock_substeps,
        "cooldown_substeps + challenge_cooldown_extra": (
            e.cooldown_substeps + e.challenge_cooldown_extra
        ),
    }
    inexact = {
        name: value for name, value in float32_exact_counters.items()
        if value > _FLOAT32_EXACT_COUNTER_MAX
    }
    if inexact:
        raise ValueError(
            "engine tick counters exceed the float32 exact-counter "
            f"contract ({_FLOAT32_EXACT_COUNTER_MAX}): {inexact}"
        )
    if not math.isclose(
        e.norm_spin, e.spin_max, rel_tol=GEOMETRY_EPS, abs_tol=GEOMETRY_EPS
    ):
        raise ValueError("engine.norm_spin and engine.spin_max must stay synchronized")


def _validate_probabilities_and_coefficients(
    e: Engine,
    f: Foul,
    r: Reward,
) -> None:
    """Validate probabilities, coefficients, and rolling-knot ordering."""
    probs = {
        "tackle_prob": e.tackle_prob,
        "deflect_prob": e.deflect_prob,
        "trap_base": e.trap_base,
        "spin_head_cap": e.spin_head_cap,
        "spin_chest_cap": e.spin_chest_cap,
        "tackle_out_cap": e.tackle_out_cap,
        "pelvis_frac": e.pelvis_frac,
        "chest_cap": e.chest_cap,
        "header_cap": e.header_cap,
        "body_spin_keep": e.body_spin_keep,
        "body_top_frac": e.body_top_frac,
        "e_body": e.e_body,
        "body_player_vel_transfer": e.body_player_vel_transfer,
        "trap_velocity_keep": e.trap_velocity_keep,
        "deflect_lift_frac": e.deflect_lift_frac,
        "parry_lateral_keep": e.parry_lateral_keep,
        "parry_lift_frac": e.parry_lift_frac,
        "parry_tip_keep": e.parry_tip_keep,
        "long_stamina_vmax_floor": e.long_stamina_vmax_floor,
        "long_stamina_end_frac": e.long_stamina_end_frac,
        "short_stamina_vmax_floor": e.short_stamina_vmax_floor,
        "short_stamina_depletion_speed_frac": e.short_stamina_depletion_speed_frac,
        "short_stamina_recovery_speed_frac": e.short_stamina_recovery_speed_frac,
        "short_stamina_long_recovery_penalty": e.short_stamina_long_recovery_penalty,
        "bounce_tangential_e": e.bounce_tangential_e,
        "bounce_h_keep": e.bounce_h_keep,
        "e_rest": e.e_rest,
        "card_per_foul": f.card_per_foul,
        "red_given_card": f.red_given_card,
    }
    if any(not math.isfinite(float(v)) or not 0.0 <= float(v) <= 1.0 for v in probs.values()):
        raise ValueError(f"probabilities must lie in [0, 1]: {probs}")
    if not (0.0 <= f.tackle_p_min <= f.tackle_p_max <= 1.0):
        raise ValueError("tackle foul probability bounds must satisfy 0 <= min <= max <= 1")
    if not (0.0 <= f.charge_p_min <= f.charge_p_max <= 1.0):
        raise ValueError("charge foul probability bounds must satisfy 0 <= min <= max <= 1")
    if not e.shot_aim_mouth_scale >= 1.0:
        raise ValueError(
            "shot_aim_mouth_scale must be >= 1.0 (the goal mouth itself)")
    nonnegative = {
        "engine.offside_margin": e.offside_margin,
        "engine.gk_catch_speed_scale": e.gk_catch_speed_scale,
        "engine.deflect_out_frac": e.deflect_out_frac,
        "engine.deflect_out_base": e.deflect_out_base,
        "engine.c_drag": e.c_drag,
        "engine.c_magnus": e.c_magnus,
        "engine.c_ground_curl": e.c_ground_curl,
        "engine.spin_decay": e.spin_decay,
        "reward.advance": r.advance,
        "reward.poss_gain": r.poss_gain,
    }
    if any(not math.isfinite(float(value)) for value in nonnegative.values()):
        raise ValueError(
            f"configuration values must be finite and non-negative: {nonnegative}"
        )
    if any(float(value) < 0.0 for value in nonnegative.values()):
        raise ValueError(f"configuration values must be non-negative: {nonnegative}")
    if f.charge_contact_padding < 0.0 or not math.isfinite(f.charge_contact_padding):
        raise ValueError("charge_contact_padding must be finite and non-negative")
    if any(not a < b for a, b in zip(e.roll_v_knots, e.roll_v_knots[1:])):
        raise ValueError("roll_v_knots must be strictly increasing")
    roll_v_f32 = np.asarray(e.roll_v_knots, dtype=np.float32)
    if np.any(roll_v_f32[:-1] >= roll_v_f32[1:]):
        raise ValueError(
            "roll_v_knots must remain strictly increasing after float32 "
            "runtime conversion"
        )
    if any((not math.isfinite(float(v)) or v < 0.0) for v in e.roll_d_knots):
        raise ValueError("roll_d_knots must be finite and non-negative")
    if not 0.0 < e.deflect_angle_max <= math.pi:
        raise ValueError("deflect_angle_max must lie in (0, pi]")


def _validate_insets_stamina_and_reward(
    s: Stadium,
    e: Engine,
    r: Reward,
) -> None:
    """Validate pitch insets, stamina constraints, and reward settings."""
    if e.restart_field_inset < 0.0 or e.restart_field_inset >= min(s.half_length, s.half_width):
        raise ValueError("restart_field_inset is outside the pitch")
    if e.throwin_line_inset < 0.0 or e.throwin_line_inset >= s.half_width:
        raise ValueError("throwin_line_inset is outside the pitch")
    if e.free_kick_boundary_inset < 0.0 or e.free_kick_boundary_inset >= min(
        s.half_length, s.half_width
    ):
        raise ValueError("free_kick_boundary_inset is outside the pitch")
    if e.bench_boundary_inset >= s.half_length or e.bench_touchline_inset >= s.half_width:
        raise ValueError("engine bench inset is outside the pitch")
    positive_stamina = {
        "long_stamina_speed_ref": e.long_stamina_speed_ref,
        "long_stamina_accel_ref": e.long_stamina_accel_ref,
        "long_stamina_reference_duration_s": e.long_stamina_reference_duration_s,
        "long_stamina_reference_workload": e.long_stamina_reference_workload,
        "short_stamina_headroom_knee": e.short_stamina_headroom_knee,
        "short_stamina_depletion_s": e.short_stamina_depletion_s,
        "short_stamina_speed_exponent": e.short_stamina_speed_exponent,
        "short_stamina_accel_ref": e.short_stamina_accel_ref,
        "short_stamina_recovery_tau_s": e.short_stamina_recovery_tau_s,
        "short_stamina_recovery_exponent": e.short_stamina_recovery_exponent,
    }
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in positive_stamina.values()
    ):
        raise ValueError(
            f"stamina configuration values must be finite and positive: {positive_stamina}"
        )
    nonnegative_stamina = {
        "long_stamina_idle_load": e.long_stamina_idle_load,
        "long_stamina_speed_load": e.long_stamina_speed_load,
        "long_stamina_accel_load": e.long_stamina_accel_load,
        "short_stamina_accel_load": e.short_stamina_accel_load,
    }
    if any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in nonnegative_stamina.values()
    ):
        raise ValueError(
            "stamina load coefficients must be finite and non-negative: "
            f"{nonnegative_stamina}"
        )
    if (
        not math.isfinite(float(e.long_stamina_sprint_mult))
        or e.long_stamina_sprint_mult < 1.0
    ):
        raise ValueError("long_stamina_sprint_mult must be finite and at least 1")
    if e.short_stamina_headroom_knee > 1.0:
        raise ValueError("short_stamina_headroom_knee must lie in (0, 1]")
    if not e.long_stamina_end_frac < e.long_stamina_tail_knee < 1.0:
        raise ValueError(
            "long_stamina_tail_knee must lie strictly between "
            "long_stamina_end_frac and 1"
        )
    if e.restart_min_ball_speed > min(e.f2b_speed_max, e.throw_speed_max):
        raise ValueError(
            "engine.restart_min_ball_speed must not exceed either kick or "
            "throw speed maximum"
        )
    if e.spin_decay * e.dt_phys > 1.0:
        raise ValueError("spin_decay * dt_phys must not exceed 1")
    if any(not math.isfinite(float(v)) for v in e.roll_v_knots):
        raise ValueError("roll_v_knots must be finite")
    if e.roll_v_knots[0] < 0.0:
        raise ValueError("roll_v_knots must be non-negative")
    if not 0.0 <= r.shaping_gamma <= 1.0:
        raise ValueError("reward.shaping_gamma must lie in [0, 1]")
    if r.mode not in {"sparse", "dense"}:
        raise ValueError(f"reward.mode must be 'sparse' or 'dense', got {r.mode!r}")


def validate_configuration(
    b: Ball,
    s: Stadium,
    e: Engine,
    f: Foul,
    r: Reward,
) -> None:
    """Validate one complete configuration in stable first-error order."""

    _validate_rolling_tables_and_types(b, s, e, f, r)
    _validate_positive_and_geometry(b, s, e, f, r)
    _validate_float32_coupled_scales(b, e)
    _validate_timing_and_counters(e)
    _validate_probabilities_and_coefficients(e, f, r)
    _validate_insets_stamina_and_reward(s, e, r)
