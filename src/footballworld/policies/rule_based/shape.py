"""Observation-only team-shape targets for the seeded rule policy.

Off-ball players preserve formation depth and width while the block follows the
ball, and only explicitly selected pressure players leave that block. The
constants below are deliberately named design priors until MESSI rollouts can
be calibrated against tracking data; no fitted position table from a different
observation contract is treated as directly transferable.

Every result row is computed from the matching observer row in
:class:`RulePolicyContext`.  Hidden opponents and a hidden ball never acquire a
value from another observer, and this module has no access to rollout ``State``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.policies.rule_based.attack_pattern import AttackPattern
from footballworld.policies.rule_based.context import RulePolicyContext
from footballworld.policies.rule_based.state import (
    ROLE_CENTRE_FORWARD,
    ROLE_CENTRE_MIDFIELDER,
    ROLE_COUNT,
    ROLE_FULL_BACK,
    ROLE_GOALKEEPER,
    ROLE_WIDE_FORWARD,
    RulePolicyState,
)
from footballworld.policies.rule_based.tactical_plan import TacticalProfile

# These are bounded tactical design priors, not measured FootballWorld
# coefficients.  Their names keep later tracking-data calibration auditable.
ATTACK_BALL_FOLLOW_X = 0.38
"""Fraction of longitudinal ball displacement followed in own possession."""

DEFEND_BALL_FOLLOW_X = 0.48
"""Fraction of longitudinal ball displacement followed by the defending block."""

ATTACK_BALL_FOLLOW_Y = 0.25
"""Lateral ball-side translation while retaining attacking width."""

DEFEND_BALL_FOLLOW_Y = 0.34
"""Lateral ball-side translation of the more compact defending block."""

ATTACK_DEPTH_SCALE = 1.08
"""Formation-depth spread in possession; one preserves the initial depth."""

DEFEND_DEPTH_SCALE = 0.86
"""Formation-depth contraction used by the defending block."""

ATTACK_WIDTH_SCALE = 1.08
"""Formation-width spread used to retain passing lanes in possession."""

DEFEND_WIDTH_SCALE = 0.88
"""Formation-width contraction used without collapsing the weak side."""

BALL_SHIFT_X_CAP_M = 18.0
"""Maximum longitudinal translation of the whole formation block."""

BALL_SHIFT_Y_CAP_M = 9.0
"""Maximum lateral translation; lower than half-width to preserve a weak side."""

COUNTERPRESS_SHIFT_X_CAP_M = 5.0
"""Maximum extra longitudinal squeeze during the caller-defined loss window."""

COUNTERPRESS_SHIFT_Y_CAP_M = 4.0
"""Maximum extra lateral squeeze during the caller-defined loss window."""

PRESSURE_GOAL_SIDE_GAP_M = 1.6
"""Goal-side containing gap used by the single nearest pressure override."""

PRESSURE_COVER_DISTANCE_M = 4.8
"""Goal-side depth held by the second defender during settled pressure."""

PRESSURE_COVER_BALL_LEAD_S = 0.18
"""Visible ball-velocity lead used by the pressure-cover target."""

SECONDARY_PRESSURE_BALL_LEAD_S = 0.30
"""Visible ball-velocity lead used by supporting direct pressers."""

SECONDARY_PRESSURE_GOAL_SIDE_GAP_M = 3.0
"""Goal-side spacing for direct pressers behind the primary actor."""

OPEN_FIELD_MARK_BASE_GAIN = 0.55
"""Zonal-to-goal-side blend for the sole open-field threat marker.

This is an explicit FootballWorld tactical design prior, not a fitted tracking
coefficient. ``ZONA_MISTA`` adds its existing mixed-mark gain to this baseline.
"""

COUNTERPRESS_OUTLET_MIN_DISTANCE_M = 2.0
"""Ignore the carrier when selecting the nearest transition outlet."""

COUNTERPRESS_LANE_MIN_STEP_M = 2.5
"""Minimum distance advanced from the ball into a transition pass lane."""

COUNTERPRESS_LANE_DISTANCE_FRACTION = 0.42
"""Fraction of the ball-to-outlet lane occupied by the cover defender."""

COUNTERPRESS_LANE_MAX_STEP_M = 6.5
"""Maximum distance advanced from the ball into a transition pass lane."""

COUNTERPRESS_COVER_GOAL_SIDE_M = 1.5
"""Goal-side insurance retained while the nearest outlet lane is closed."""

OFFSIDE_SAFETY_MARGIN_M = 0.30
"""Observable offside-line setback for a moving player's body extent."""

PITCH_X_INSET_M = 1.0
"""Target inset from either goal line."""

PITCH_Y_INSET_M = 1.0
"""Target inset from either touchline."""

DIRECTION_EPS_M = 1.0e-6
"""Distance below which a movement direction is exactly zero."""

# Role ordering follows state.py.  Goalkeepers remain anchored, defenders join
# conservatively, midfielders connect the block, and forwards supply depth.
_ATTACK_ROLE_ADVANCE_M = (
    0.0,  # goalkeeper
    0.5,  # centre back
    1.5,  # full back
    3.0,  # centre midfielder
    3.5,  # wide midfielder
    5.0,  # centre forward
    5.0,  # wide forward
)
"""Role-specific forward displacement in an own-possession shape."""

_ROLE_BALL_FOLLOW = (
    0.15,  # goalkeeper
    0.72,  # centre back
    0.86,  # full back
    1.00,  # centre midfielder
    1.00,  # wide midfielder
    1.05,  # centre forward
    1.05,  # wide forward
)
"""Role multiplier preventing a ball-following block from dragging its keeper."""

_COUNTERPRESS_ROLE_GAIN = (
    0.00,  # goalkeeper
    0.05,  # centre back: rest defence
    0.09,  # full back
    0.17,  # centre midfielder
    0.15,  # wide midfielder
    0.24,  # centre forward
    0.21,  # wide forward
)
"""Extra ball-side squeeze by role after possession is lost."""

_WIDE_ROLE = (
    False,
    False,
    True,
    False,
    True,
    False,
    True,
)
"""Roles whose formation anchors must retain explicit lateral width."""
_KICKOFF_ROLE_PATH = (
    (0.00, 0.00),  # goalkeeper
    (0.60, 0.45),  # centre back
    (0.85, 1.80),  # full back
    (1.10, 0.95),  # centre midfielder
    (1.25, 2.00),  # wide midfielder
    (1.50, 0.85),  # centre forward
    (1.40, 1.90),  # wide forward
)
"""Role-scaled 2-D opening waypoint DESIGN_PRIOR; not a tracking-data fit."""


class ShapeMovement(NamedTuple):
    """Per-observer target, direction, distance, and urgent role override."""

    target: jax.Array
    direction: jax.Array
    distance: jax.Array
    urgent: jax.Array
    direct_pressure: jax.Array


def _validate_shapes(
    context: RulePolicyContext,
    policy_state: RulePolicyState,
    own_possession: jax.Array,
    opponent_possession: jax.Array,
    nearest_pressure: jax.Array,
    counterpress_active: jax.Array,
) -> int:
    """Validate static axes before JAX traces the fixed-shape calculation."""

    player_count = context.self_position.shape[0]
    if context.self_position.shape != (player_count, 2):
        raise ValueError("context self_position must have shape (players, 2)")
    if context.player_position.shape != (player_count, player_count, 2):
        raise ValueError("context player_position must have observer and roster axes")
    if context.opponent.shape != (player_count, player_count):
        raise ValueError("context opponent mask must match both player axes")
    if context.ball_position.shape != (player_count, 3):
        raise ValueError("context ball_position must have shape (players, 3)")
    if policy_state.formation_anchor.shape != (player_count, 2):
        raise ValueError("formation anchors must match the policy roster")
    if policy_state.role.shape != (player_count,):
        raise ValueError("policy roles must match the policy roster")
    for name, value in (
        ("own_possession", own_possession),
        ("opponent_possession", opponent_possession),
        ("nearest_pressure", nearest_pressure),
        ("counterpress_active", counterpress_active),
    ):
        if value.shape != (player_count,):
            raise ValueError(f"{name} must have shape (players,)")
    return player_count


def _observable_offside_line(
    context: RulePolicyContext,
) -> tuple[jax.Array, jax.Array]:
    """Return an observable Law-11 line and whether enough evidence exists.

    The offside line is the farther-forward of the ball and the second-last
    visible opponent, bounded by halfway because a player in their own half is
    not offside. A visible ball alone is a conservative valid line: remaining
    behind it is sufficient regardless of hidden defenders. Two visible
    opponents alone also define a usable defender line. With neither source,
    the function reports no cap rather than inventing hidden positions.
    """

    opponent_x = jnp.where(
        context.opponent,
        context.player_position[..., 0],
        -jnp.inf,
    )
    visible_count = jnp.sum(context.opponent, axis=-1)
    # Only the two most advanced opponents are needed. top_k preserves the
    # same second-last value while avoiding a full roster sort per observer.
    second_last = jax.lax.top_k(opponent_x, 2)[0][:, 1]
    has_defender_line = visible_count >= 2

    ball_x = context.ball_position[:, 0]
    defender_line = jnp.where(has_defender_line, second_last, -jnp.inf)
    observable_line = jnp.maximum(
        jnp.where(context.ball_visible, ball_x, -jnp.inf),
        defender_line,
    )
    has_line = context.ball_visible | has_defender_line
    observable_line = jnp.maximum(observable_line, jnp.float32(0.0))
    return observable_line.astype(jnp.float32), has_line


_BOX_MARK_ASSIGNMENT_COUNT = 3


def _assign_box_runners(
    defender_position,
    defender_candidate,
    runner_position,
    runner_candidate,
    runner_threat,
    ball_position,
):
    """Assign near-post, far-post, and cutback runners without duplication.

    The three-step scan is a fixed graph bound, not a football coefficient.
    Rows use only their own visible positions. Channel preference prevents all
    markers collapsing onto the closest runner; when a channel is absent the
    next most threatening unassigned runner is selected instead.
    """

    defender_position = jnp.asarray(defender_position, dtype=jnp.float32)
    defender_candidate = jnp.asarray(defender_candidate, dtype=jnp.bool_)
    runner_position = jnp.asarray(runner_position, dtype=jnp.float32)
    runner_candidate = jnp.asarray(runner_candidate, dtype=jnp.bool_)
    runner_threat = jnp.asarray(runner_threat, dtype=jnp.float32)
    ball_position = jnp.asarray(ball_position, dtype=jnp.float32)
    observer_count, player_count = defender_candidate.shape
    if defender_position.shape != (observer_count, player_count, 2):
        raise ValueError("defender_position must have shape (observers, players, 2)")
    if runner_position.shape != defender_position.shape:
        raise ValueError("runner_position must match defender_position")
    if runner_candidate.shape != defender_candidate.shape:
        raise ValueError("runner_candidate must match defender_candidate")
    if runner_threat.shape != defender_candidate.shape:
        raise ValueError("runner_threat must match defender_candidate")
    if ball_position.shape != (observer_count, 2):
        raise ValueError("ball_position must have shape (observers, 2)")

    row = jnp.arange(observer_count, dtype=jnp.int32)
    assignment = jnp.full((observer_count, player_count), -1, dtype=jnp.int32)
    used_defender = jnp.zeros_like(defender_candidate)
    used_runner = jnp.zeros_like(runner_candidate)
    cutback = runner_position[..., 0] > ball_position[:, None, 0]
    same_side = (runner_position[..., 1] * ball_position[:, None, 1]) >= 0.0

    def assign_channel(channel_index, carry):
        current_assignment, current_defender, current_runner = carry
        unassigned_runner = runner_candidate & (~current_runner)
        preferred_runner = jax.lax.switch(
            channel_index,
            (
                lambda: unassigned_runner & same_side & (~cutback),
                lambda: unassigned_runner & (~same_side) & (~cutback),
                lambda: unassigned_runner & cutback,
            ),
        )
        use_preferred = jnp.any(preferred_runner, axis=-1)
        runner_pool = jnp.where(
            use_preferred[:, None], preferred_runner, unassigned_runner
        )
        has_runner = jnp.any(runner_pool, axis=-1)
        selected_runner = jnp.argmax(
            jnp.where(runner_pool, runner_threat, -jnp.inf), axis=-1
        )
        selected_position = runner_position[row, selected_runner]
        # The assignment consumes only nearest-defender rank.
        distance_squared = jnp.sum(
            jnp.square(defender_position - selected_position[:, None, :]), axis=-1
        )
        defender_pool = defender_candidate & (~current_defender)
        has_defender = jnp.any(defender_pool, axis=-1)
        selected_defender = jnp.argmin(
            jnp.where(defender_pool, distance_squared, jnp.inf), axis=-1
        )
        can_assign = has_runner & has_defender
        previous_assignment = current_assignment[row, selected_defender]
        current_assignment = current_assignment.at[row, selected_defender].set(
            jnp.where(can_assign, selected_runner, previous_assignment)
        )
        previous_defender = current_defender[row, selected_defender]
        current_defender = current_defender.at[row, selected_defender].set(
            jnp.where(can_assign, True, previous_defender)
        )
        previous_runner = current_runner[row, selected_runner]
        current_runner = current_runner.at[row, selected_runner].set(
            jnp.where(can_assign, True, previous_runner)
        )
        return current_assignment, current_defender, current_runner

    assignment, _, _ = jax.lax.fori_loop(
        0,
        _BOX_MARK_ASSIGNMENT_COUNT,
        assign_channel,
        (assignment, used_defender, used_runner),
    )
    return assignment


def _goal_side_mark_target(
    runner_position,
    *,
    own_goal_x,
    distance_m,
):
    """Place a marker on the runner-to-own-goal line at a bounded gap."""

    runner_position = jnp.asarray(runner_position, dtype=jnp.float32)
    if runner_position.shape[-1:] != (2,):
        raise ValueError("runner_position must have trailing shape (2,)")
    own_goal = jnp.stack(
        (
            jnp.full(runner_position.shape[:-1], own_goal_x, dtype=jnp.float32),
            jnp.zeros(runner_position.shape[:-1], dtype=jnp.float32),
        ),
        axis=-1,
    )
    runner_to_goal = own_goal - runner_position
    goal_distance = jnp.linalg.norm(runner_to_goal, axis=-1, keepdims=True)
    fallback = jnp.zeros_like(runner_position)
    direction = jnp.where(
        goal_distance > DIRECTION_EPS_M,
        runner_to_goal / jnp.maximum(goal_distance, DIRECTION_EPS_M),
        fallback,
    )
    return runner_position + direction * jnp.float32(distance_m)


def _forward_pocket_delta(attack_pattern, ball_side, shift_m):
    """Return one stable, pattern-specific pocket displacement."""

    attack_pattern = jnp.asarray(attack_pattern, dtype=jnp.int32)
    ball_side = jnp.where(
        jnp.asarray(ball_side, dtype=jnp.float32) >= 0.0,
        jnp.float32(1.0),
        jnp.float32(-1.0),
    )
    shift_m = jnp.maximum(jnp.asarray(shift_m, dtype=jnp.float32), 0.0)
    progress = attack_pattern == jnp.int32(AttackPattern.PROGRESSIVE_CARRY)
    third_man = attack_pattern == jnp.int32(AttackPattern.THIRD_MAN)
    overload = attack_pattern == jnp.int32(AttackPattern.WIDE_OVERLOAD)
    switch = attack_pattern == jnp.int32(AttackPattern.SWITCH_PLAY)
    depth = jnp.where(
        progress,
        jnp.float32(0.45),
        jnp.where(
            third_man,
            jnp.float32(0.72),
            jnp.where(overload, jnp.float32(0.58), jnp.float32(0.35)),
        ),
    )
    lateral = jnp.where(
        progress,
        -jnp.float32(0.45) * ball_side,
        jnp.where(
            third_man,
            -jnp.float32(0.50) * ball_side,
            jnp.where(
                overload,
                jnp.float32(0.72) * ball_side,
                jnp.where(switch, -jnp.float32(0.85) * ball_side, 0.0),
            ),
        ),
    )
    direct = attack_pattern == jnp.int32(AttackPattern.RUN_BEHIND_DIRECT)
    depth = jnp.where(direct, jnp.float32(1.0), depth)
    lateral = jnp.where(direct, jnp.float32(0.0), lateral)
    return shift_m * jnp.stack((depth, lateral), axis=-1)


def shape_movement(
    context: RulePolicyContext,
    policy_state: RulePolicyState,
    *,
    half_length: float,
    half_width: float,
    penalty_area_length: float,
    penalty_area_width: float,
    goal_width: float,
    cross_start_fraction: float,
    cross_wide_fraction: float,
    cross_target_central_fraction: float,
    box_mark_lead_s: float,
    box_mark_runner_margin_m: float,
    box_mark_ball_margin_m: float,
    box_mark_goal_side_distance_m: float,
    own_possession: jax.Array,
    opponent_possession: jax.Array,
    nearest_pressure: jax.Array,
    counterpress_active: jax.Array,
    tactical: TacticalProfile,
    attack_pattern: jax.Array,
    attack_phase: jax.Array,
    attack_pattern_shape_shift_m: float,
    forward_pocket_shift_m: float,
    run_behind_receiver: jax.Array,
    run_behind_release: jax.Array,
    run_behind_timing_error: jax.Array,
    offside_line_error_m: jax.Array,
    forward_run_min_gap_m: float,
    kickoff_path_phase: jax.Array = jnp.float32(0.0),
    kickoff_path_lateral_shift_m: float = 0.0,
) -> ShapeMovement:
    """Build one formation-respecting movement target per observation row.

    ``own_possession`` selects attacking support versus defending shape.
    ``opponent_possession`` distinguishes a controlled opposition attack from
    a loose ball or restart. ``nearest_pressure`` is the caller-selected
    primary presser. In settled defence a tactical profile may promote the
    next one or two visible outfielders into supporting direct pressure; the
    following player remains cover. ``counterpress_active`` is a caller-owned
    time-window mask (normally derived from ``RulePolicyState.counterpress_age``);
    during that window only the primary presses while the next player closes
    the nearest visible outlet lane. All inputs have one boolean per observer
    and may differ when view-limited observations differ.

    Coordinates and the returned target use each observer's attacking frame.
    The direction is a unit vector from that observer to its target and is zero
    at the target, ready for the public radial action encoder.
    """

    own_possession = jnp.asarray(own_possession, dtype=jnp.bool_)
    opponent_possession = jnp.asarray(opponent_possession, dtype=jnp.bool_)
    nearest_pressure = jnp.asarray(nearest_pressure, dtype=jnp.bool_)
    counterpress_active = jnp.asarray(counterpress_active, dtype=jnp.bool_)
    offside_line_error_m = jnp.asarray(offside_line_error_m, dtype=jnp.float32)
    if offside_line_error_m.shape != ():
        raise ValueError("offside_line_error_m must be scalar")
    offside_line_error_m = jnp.maximum(offside_line_error_m, 0.0)
    kickoff_path_phase = jnp.asarray(kickoff_path_phase, dtype=jnp.float32)
    kickoff_path_lateral_shift_m = jnp.maximum(
        jnp.asarray(kickoff_path_lateral_shift_m, dtype=jnp.float32),
        0.0,
    )
    expected_phase_shape = context.self_index.shape
    if kickoff_path_phase.shape not in ((), expected_phase_shape):
        raise ValueError("kickoff_path_phase must be scalar or per observer")
    if kickoff_path_lateral_shift_m.shape != ():
        raise ValueError("kickoff_path_lateral_shift_m must be scalar")
    kickoff_path_phase = jnp.broadcast_to(
        jnp.clip(kickoff_path_phase, 0.0, 1.0), expected_phase_shape
    )

    attack_pattern = jnp.asarray(attack_pattern, dtype=jnp.int32)
    if attack_pattern.shape != (context.self_index.shape[0],):
        raise ValueError("attack_pattern must have one code per observer")
    attack_phase = jnp.asarray(attack_phase, dtype=jnp.int32)
    if attack_phase.shape != (context.self_index.shape[0],):
        raise ValueError("attack_phase must have one code per observer")
    run_behind_receiver = jnp.asarray(run_behind_receiver, dtype=jnp.int32)
    run_behind_release = jnp.asarray(run_behind_release, dtype=jnp.bool_)
    run_behind_timing_error = jnp.asarray(run_behind_timing_error, dtype=jnp.bool_)
    for name, value in (
        ("run_behind_receiver", run_behind_receiver),
        ("run_behind_release", run_behind_release),
        ("run_behind_timing_error", run_behind_timing_error),
    ):
        if value.shape != ():
            raise ValueError(f"{name} must be scalar")
    pattern_shift_m = jnp.maximum(
        jnp.asarray(attack_pattern_shape_shift_m, dtype=jnp.float32),
        0.0,
    )
    forward_pocket_shift_m = jnp.maximum(
        jnp.asarray(forward_pocket_shift_m, dtype=jnp.float32),
        0.0,
    )
    forward_run_min_gap_m = jnp.maximum(
        jnp.asarray(forward_run_min_gap_m, dtype=jnp.float32), 0.0
    )
    _validate_shapes(
        context,
        policy_state,
        own_possession,
        opponent_possession,
        nearest_pressure,
        counterpress_active,
    )

    self_index = context.self_index.astype(jnp.int32)
    role = policy_state.role[self_index].astype(jnp.int32)
    role = jnp.clip(role, 0, ROLE_COUNT - 1)
    anchor = policy_state.formation_anchor[self_index].astype(jnp.float32)

    # The anchor centroid is formed from public same-team slot semantics, not
    # current hidden player positions.  Every teammate's initialization uses
    # the same attacking frame, so its anchor can be combined within this row.
    team_slot = context.same_team
    team_count = jnp.maximum(jnp.sum(team_slot, axis=-1), 1)
    anchor_sum = jnp.sum(
        jnp.where(
            team_slot[..., None],
            policy_state.formation_anchor[None, :, :],
            jnp.float32(0.0),
        ),
        axis=1,
    )
    anchor_center = anchor_sum / team_count[:, None]
    anchor_offset = anchor - anchor_center

    attack_advance = jnp.asarray(_ATTACK_ROLE_ADVANCE_M, dtype=jnp.float32)[role]
    role_follow = jnp.asarray(_ROLE_BALL_FOLLOW, dtype=jnp.float32)[role]
    wide_role = jnp.asarray(_WIDE_ROLE, dtype=jnp.bool_)[role]
    team_roles = policy_state.role[None, :]
    pivot_candidate = team_slot & (team_roles == ROLE_CENTRE_MIDFIELDER)
    pivot_index = jnp.argmin(
        jnp.where(
            pivot_candidate,
            jnp.abs(policy_state.formation_anchor[None, :, 1]),
            jnp.inf,
        ),
        axis=-1,
    )
    is_pivot = pivot_candidate[jnp.arange(self_index.shape[0]), pivot_index] & (
        self_index == pivot_index
    )
    attack_advance = attack_advance + jnp.where(
        role == ROLE_FULL_BACK, tactical.fullback_advance_m, 0.0
    )
    attack_advance = attack_advance - jnp.where(is_pivot, tactical.pivot_drop_m, 0.0)

    attack_scale_x = tactical.attack_depth_scale
    defend_scale_x = tactical.defend_depth_scale
    attack_scale_y = jnp.where(
        wide_role,
        tactical.attack_width_scale,
        jnp.float32(1.0),
    )
    defend_scale_y = jnp.where(
        wide_role,
        tactical.defend_width_scale,
        jnp.float32(0.94),
    )
    depth_scale = jnp.where(own_possession, attack_scale_x, defend_scale_x)
    width_scale = jnp.where(own_possession, attack_scale_y, defend_scale_y)

    base_target = anchor_center + jnp.stack(
        (
            anchor_offset[:, 0] * depth_scale
            + jnp.where(own_possession, 0.0, tactical.defend_line_shift_m)
            + jnp.where(own_possession, attack_advance, 0.0),
            anchor_offset[:, 1] * width_scale,
        ),
        axis=-1,
    )
    halfspace_side = jnp.where(anchor_offset[:, 1] >= 0.0, 1.0, -1.0)
    halfspace_y = halfspace_side * jnp.float32(0.5) * half_width
    halfspace_shift = jnp.where(
        own_possession & (role == ROLE_CENTRE_MIDFIELDER),
        tactical.halfspace_gain * (halfspace_y - base_target[:, 1]),
        0.0,
    )
    base_target = base_target.at[:, 1].add(halfspace_shift)

    ball_xy = context.ball_position[:, :2]
    advanced_wide_ball = (
        context.ball_visible
        & (ball_xy[:, 0] >= cross_start_fraction * half_length)
        & (jnp.abs(ball_xy[:, 1]) >= cross_wide_fraction * half_width)
    )
    ball_from_center = ball_xy - anchor_center
    follow_x = jnp.where(
        own_possession,
        jnp.float32(ATTACK_BALL_FOLLOW_X),
        jnp.float32(DEFEND_BALL_FOLLOW_X),
    )
    follow_y = jnp.where(
        own_possession,
        jnp.float32(ATTACK_BALL_FOLLOW_Y),
        jnp.float32(DEFEND_BALL_FOLLOW_Y),
    )
    # FootballWorld's fitted positional field cannot be transferred across this
    # observation contract. Retain FootballWorld's bounded ball-follow prior:
    # formation-relative roles advance together without turning ball depth into
    # an absolute target or dragging the goalkeeper through the block.
    attacking_shift_x = jnp.clip(
        ball_from_center[:, 0] * jnp.float32(ATTACK_BALL_FOLLOW_X) * role_follow,
        -BALL_SHIFT_X_CAP_M,
        BALL_SHIFT_X_CAP_M,
    )
    defensive_shift_x = jnp.clip(
        ball_from_center[:, 0] * follow_x * role_follow,
        -BALL_SHIFT_X_CAP_M,
        BALL_SHIFT_X_CAP_M,
    )
    ball_shift = jnp.stack(
        (
            jnp.where(own_possession, attacking_shift_x, defensive_shift_x),
            jnp.clip(
                ball_from_center[:, 1] * follow_y * role_follow,
                -BALL_SHIFT_Y_CAP_M,
                BALL_SHIFT_Y_CAP_M,
            ),
        ),
        axis=-1,
    )
    target = base_target + jnp.where(
        context.ball_visible[:, None], ball_shift, jnp.float32(0.0)
    )
    # The fixed role/roster-slot waypoint has longitudinal and lateral parts.
    # Its quadratic envelope is exactly zero at the final live-window tick, so
    # settled formation targets stay unchanged without random or recurrent state.
    formation_side = jnp.where(
        jnp.abs(anchor_offset[:, 1]) >= 1.0,
        jnp.where(anchor_offset[:, 1] >= 0.0, 1.0, -1.0),
        jnp.where((self_index & 1) == 0, 1.0, -1.0),
    )
    slot_side = jnp.where(((self_index + role) & 1) == 0, 1.0, -1.0)
    lateral_side = jnp.where(own_possession, formation_side, -formation_side)
    role_path = jnp.asarray(_KICKOFF_ROLE_PATH, dtype=jnp.float32)[role]
    waypoint = jnp.stack(
        (role_path[:, 0] * slot_side, role_path[:, 1] * lateral_side),
        axis=-1,
    )
    kickoff_curve = (
        4.0
        * kickoff_path_phase
        * (1.0 - kickoff_path_phase)
        * kickoff_path_lateral_shift_m
    )
    target = target + kickoff_curve[:, None] * waypoint

    # Select at most one public, active wide-role slot on either side. These
    # players preserve the formation-scaled width before a pass arrives, but
    # are never pinned to the touchline. Stable roster winners prevent an
    # entire position group from being widened by the same rule. The lateral
    # ball-follow term may move a ball-side outlet farther out; it may not drag
    # either selected outlet inside its formation-relative attacking lane.
    roster_slot = jnp.arange(self_index.shape[0], dtype=jnp.int32)[None, :]
    observed_carrier = policy_state.current_possessor[:, None]
    team_anchor_y = policy_state.formation_anchor[None, :, 1]
    anchor_side = jnp.where(team_anchor_y >= 0.0, 1.0, -1.0)
    self_anchor_side = jnp.where(anchor[:, 1] >= 0.0, 1.0, -1.0)
    public_wide_candidate = (
        team_slot
        & context.participating
        & ((team_roles == ROLE_FULL_BACK) | (team_roles == ROLE_WIDE_FORWARD))
    )
    side_wide_candidate = public_wide_candidate & (
        anchor_side == self_anchor_side[:, None]
    )
    side_wide_index = jnp.argmax(
        jnp.where(side_wide_candidate, jnp.abs(team_anchor_y), -jnp.inf),
        axis=-1,
    )
    selected_wide = jnp.any(side_wide_candidate, axis=-1) & (
        self_index == side_wide_index
    )
    is_observed_carrier = self_index == policy_state.current_possessor
    proactive_wide = (
        own_possession
        & context.ball_visible
        & context.self_active
        & selected_wide
        & (~is_observed_carrier)
    )
    formation_wide_y = anchor_center[:, 1] + (
        anchor_offset[:, 1] * tactical.attack_width_scale
    )
    retained_wide_y = self_anchor_side * jnp.maximum(
        self_anchor_side * target[:, 1],
        self_anchor_side * formation_wide_y,
    )
    target = target.at[:, 1].set(
        jnp.where(proactive_wide, retained_wide_y, target[:, 1])
    )
    ball_side = jnp.where(ball_xy[:, 1] >= 0.0, 1.0, -1.0)

    overlap_side = jnp.where(anchor[:, 1] >= 0.0, 1.0, -1.0)
    overlap_target = jnp.stack(
        (
            ball_xy[:, 0] + tactical.overlap_run_m,
            overlap_side * jnp.float32(0.82) * half_width,
        ),
        axis=-1,
    )
    overlap = (
        own_possession
        & advanced_wide_ball
        & (role == ROLE_FULL_BACK)
        & (anchor[:, 1] * ball_xy[:, 1] >= 0.0)
        & (tactical.overlap_run_m > 0.0)
    )
    target = jnp.where(overlap[:, None], overlap_target, target)

    # An advanced wide carrier needs three distinct receiving lanes: a centre
    # forward attacks the central corridor, one weak-side wide forward attacks
    # the far-post corridor, and one centre midfielder holds the cutback zone
    # once the ball is deep. These fixed targets use only public ball geometry
    # and MESSI's formation roles rather than a fitted position table from a
    # different observation contract. Each role has
    # one stable roster-slot winner, excluding the currently observed carrier,
    # so a whole position group cannot collapse into the box.
    signed_anchor_y = team_anchor_y * ball_side[:, None]
    support_candidate = (
        team_slot & context.participating & (roster_slot != observed_carrier)
    )
    central_forward_candidate = support_candidate & (team_roles == ROLE_CENTRE_FORWARD)
    central_forward_index = jnp.argmin(
        jnp.where(central_forward_candidate, roster_slot, self_index.shape[0]),
        axis=-1,
    )
    central_forward = jnp.any(central_forward_candidate, axis=-1) & (
        self_index == central_forward_index
    )
    far_wide_forward_candidate = (
        support_candidate & (team_roles == ROLE_WIDE_FORWARD) & (signed_anchor_y < 0.0)
    )
    far_wide_forward_index = jnp.argmin(
        jnp.where(far_wide_forward_candidate, signed_anchor_y, jnp.inf),
        axis=-1,
    )
    far_wide_forward = jnp.any(far_wide_forward_candidate, axis=-1) & (
        self_index == far_wide_forward_index
    )
    cutback_midfielder_candidate = support_candidate & (
        team_roles == ROLE_CENTRE_MIDFIELDER
    )
    cutback_midfielder_index = jnp.argmin(
        jnp.where(
            cutback_midfielder_candidate,
            jnp.abs(team_anchor_y),
            jnp.inf,
        ),
        axis=-1,
    )
    cutback_midfielder = jnp.any(cutback_midfielder_candidate, axis=-1) & (
        self_index == cutback_midfielder_index
    )
    box_support_start_x = jnp.maximum(
        jnp.float32(cross_start_fraction * half_length),
        jnp.float32(half_length - 2.0 * penalty_area_length),
    )
    # Wide attacks retain their earlier two-box-length trigger.  A central
    # attack previously had no corresponding final-third support at all, so
    # forwards could retreat toward their formation anchors after the carrier
    # advanced beyond the defensive line.  Activate the same two distinct box
    # lanes only once a central ball is within six metres of the penalty area;
    # the observable Law-11 cap below remains final authority.
    central_box_support = (
        context.ball_visible
        & (ball_xy[:, 0] >= half_length - penalty_area_length - jnp.float32(6.0))
        & (jnp.abs(ball_xy[:, 1]) < cross_wide_fraction * half_width)
    )
    wide_box_support = advanced_wide_ball & (ball_xy[:, 0] >= box_support_start_x)
    box_runner_support = wide_box_support | central_box_support
    central_y_limit = jnp.float32(cross_target_central_fraction * half_width)
    central_support = (
        own_possession
        & box_runner_support
        & context.self_active
        & central_forward
    )
    far_support = (
        own_possession
        & box_runner_support
        & context.self_active
        & far_wide_forward
    )
    deep_wide_ball = advanced_wide_ball & (
        ball_xy[:, 0] >= jnp.float32(half_length - penalty_area_length - 2.0)
    )
    cutback_support = (
        own_possession & deep_wide_ball & context.self_active & cutback_midfielder
    )
    central_support_target = jnp.stack(
        (
            jnp.full_like(ball_xy[:, 0], jnp.float32(half_length - 9.0)),
            jnp.clip(
                jnp.float32(0.22) * ball_xy[:, 1],
                -jnp.minimum(jnp.float32(5.0), central_y_limit),
                jnp.minimum(jnp.float32(5.0), central_y_limit),
            ),
        ),
        axis=-1,
    )
    far_post_y = -ball_side * jnp.minimum(
        jnp.float32(0.42 * goal_width),
        jnp.float32(0.5 * penalty_area_width - 2.0),
    )
    far_support_target = jnp.stack(
        (
            jnp.full_like(ball_xy[:, 0], jnp.float32(half_length - 7.0)),
            far_post_y,
        ),
        axis=-1,
    )
    cutback_support_target = jnp.stack(
        (
            jnp.full_like(
                ball_xy[:, 0],
                jnp.float32(half_length - penalty_area_length - 2.5),
            ),
            jnp.clip(
                -jnp.float32(0.25) * ball_xy[:, 1],
                -jnp.minimum(jnp.float32(9.0), central_y_limit),
                jnp.minimum(jnp.float32(9.0), central_y_limit),
            ),
        ),
        axis=-1,
    )
    target = jnp.where(central_support[:, None], central_support_target, target)
    target = jnp.where(far_support[:, None], far_support_target, target)
    target = jnp.where(cutback_support[:, None], cutback_support_target, target)

    # Possession-episode patterns perturb the established shape by at most one
    # configured displacement.  These are movement priors, not new roles or
    # candidate gates.  The pitch and observable offside caps below remain the
    # final authority, and rows without observed own possession ignore them.
    eligible_pattern_actor = own_possession & context.self_active
    setup_phase = attack_phase == jnp.int32(0)
    execution_phase = attack_phase == jnp.int32(1)
    active_pattern_phase = setup_phase | execution_phase
    progressive_support = (
        eligible_pattern_actor
        & (attack_pattern == jnp.int32(AttackPattern.PROGRESSIVE_CARRY))
        & setup_phase
        & cutback_midfielder
    )
    third_man_support = (
        eligible_pattern_actor
        & (attack_pattern == jnp.int32(AttackPattern.THIRD_MAN))
        & execution_phase
        & cutback_midfielder
    )
    same_side_wide = wide_role & (anchor[:, 1] * ball_xy[:, 1] >= 0.0)
    wide_pattern_support = (
        eligible_pattern_actor
        & (attack_pattern == jnp.int32(AttackPattern.WIDE_OVERLOAD))
        & active_pattern_phase
        & same_side_wide
    )
    weak_side_wide = wide_role & (anchor[:, 1] * ball_xy[:, 1] < 0.0)
    switch_setup_support = (
        eligible_pattern_actor
        & (attack_pattern == jnp.int32(AttackPattern.SWITCH_PLAY))
        & setup_phase
        & same_side_wide
    )
    switch_release_support = (
        eligible_pattern_actor
        & (attack_pattern == jnp.int32(AttackPattern.SWITCH_PLAY))
        & execution_phase
        & weak_side_wide
    )
    designated_forward_support = (
        eligible_pattern_actor
        & active_pattern_phase
        & (self_index == run_behind_receiver)
        & (context.self_position[:, 0] >= ball_xy[:, 0] + jnp.float32(1.0))
    )
    run_pattern_support = designated_forward_support & (
        attack_pattern == jnp.int32(AttackPattern.RUN_BEHIND_DIRECT)
    )
    pattern_delta = jnp.zeros_like(target)
    pattern_delta = jnp.where(
        progressive_support[:, None],
        pattern_shift_m * jnp.asarray((0.35, 0.0), dtype=jnp.float32),
        pattern_delta,
    )
    pattern_delta = jnp.where(
        third_man_support[:, None],
        pattern_shift_m
        * jnp.stack(
            (
                jnp.full_like(ball_side, jnp.float32(0.72)),
                -jnp.float32(0.45) * ball_side,
            ),
            axis=-1,
        ),
        pattern_delta,
    )
    pattern_delta = jnp.where(
        wide_pattern_support[:, None],
        pattern_shift_m
        * jnp.stack(
            (
                jnp.full_like(ball_side, jnp.float32(0.62)),
                jnp.float32(0.78) * ball_side,
            ),
            axis=-1,
        ),
        pattern_delta,
    )
    pattern_delta = jnp.where(
        switch_setup_support[:, None],
        pattern_shift_m
        * jnp.stack(
            (
                jnp.full_like(ball_side, jnp.float32(0.35)),
                jnp.float32(0.45) * ball_side,
            ),
            axis=-1,
        ),
        pattern_delta,
    )
    pattern_delta = jnp.where(
        switch_release_support[:, None],
        pattern_shift_m
        * jnp.stack(
            (
                jnp.full_like(ball_side, jnp.float32(0.35)),
                -jnp.float32(0.90) * ball_side,
            ),
            axis=-1,
        ),
        pattern_delta,
    )
    pattern_delta = jnp.where(
        run_pattern_support[:, None],
        pattern_shift_m * jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        pattern_delta,
    )
    # One episode-stable forward is already selected from the possession key.
    # Give that player a fixed pattern pocket instead of reacting frame by
    # frame to the nearest marker. Other players retain the team shape, so they
    # do not drag several defenders into the same receiving lane.
    forward_pocket_delta = _forward_pocket_delta(
        attack_pattern, ball_side, forward_pocket_shift_m
    )
    pattern_delta = jnp.where(
        designated_forward_support[:, None],
        forward_pocket_delta,
        pattern_delta,
    )
    target = target + pattern_delta

    # The loss-window squeeze is role selective.  Centre backs retain rest
    # defence, while midfielders and forwards close the ball-side escape area.
    counterpress_gain = (
        jnp.asarray(_COUNTERPRESS_ROLE_GAIN, dtype=jnp.float32)[role]
        * tactical.counterpress_gain
    )
    counterpress_delta = (ball_xy - target) * counterpress_gain[:, None]
    counterpress_delta = jnp.stack(
        (
            jnp.clip(
                counterpress_delta[:, 0],
                -COUNTERPRESS_SHIFT_X_CAP_M,
                COUNTERPRESS_SHIFT_X_CAP_M,
            ),
            jnp.clip(
                counterpress_delta[:, 1],
                -COUNTERPRESS_SHIFT_Y_CAP_M,
                COUNTERPRESS_SHIFT_Y_CAP_M,
            ),
        ),
        axis=-1,
    )
    squeeze = (
        (~own_possession)
        & counterpress_active
        & context.ball_visible
        & (~nearest_pressure)
    )
    target = target + jnp.where(squeeze[:, None], counterpress_delta, jnp.float32(0.0))

    # Rank visible outfielders once per public observer row. Tactical plans may
    # use one, two, or three settled direct pressers. Supporting pressers lead
    # the visible ball and remain goal-side instead of collapsing onto the same
    # point as the primary actor. Immediately after a loss the existing
    # counterpress contract remains 1+cover: the second player closes an outlet
    # rather than creating an indiscriminate swarm.
    team_outfielder = (
        context.same_team
        & context.participating
        & context.player_visible
        & (policy_state.role[None, :] != ROLE_GOALKEEPER)
    )
    player_ball_offset = context.player_position - ball_xy[:, None, :]
    player_ball_distance_sq = jnp.sum(player_ball_offset * player_ball_offset, axis=-1)
    self_ball_offset = context.self_position - ball_xy
    self_ball_distance_sq = jnp.sum(self_ball_offset * self_ball_offset, axis=-1)
    closer_to_ball = team_outfielder & (
        (player_ball_distance_sq < self_ball_distance_sq[:, None])
        | (
            (player_ball_distance_sq == self_ball_distance_sq[:, None])
            & (roster_slot < self_index[:, None])
        )
    )
    pressure_rank = jnp.sum(closer_to_ball, axis=-1)
    settled_pressure_count = jnp.clip(
        jnp.rint(tactical.settled_pressure_count).astype(jnp.int32), 1, 3
    )
    settled_pressure_count = jnp.where(
        (settled_pressure_count >= 3) & (ball_xy[:, 0] > 0.0),
        jnp.int32(2),
        settled_pressure_count,
    )
    active_pressure_count = jnp.where(
        counterpress_active, jnp.int32(1), settled_pressure_count
    )
    supporting_pressure = (
        opponent_possession
        & (~counterpress_active)
        & context.ball_visible
        & context.self_active
        & (role != ROLE_GOALKEEPER)
        & (pressure_rank > 0)
        & (pressure_rank < active_pressure_count)
    )
    direct_pressure = nearest_pressure | supporting_pressure
    pressure_cover = (
        opponent_possession
        & context.ball_visible
        & context.self_active
        & (role != ROLE_GOALKEEPER)
        & (pressure_rank == active_pressure_count)
        & (~direct_pressure)
    )

    # Outside the box, mark the single most dangerous visible runner with one
    # eligible defender.  The old wide-role rule independently followed each
    # player's nearest opponent, so two wide players could converge on the same
    # harmless outlet while a central runner advanced untracked.  FootballWorld's
    # sound part is the *assignment contract*: predict visible opponents, rank
    # one threat, and let only the nearest non-pressing outfielder claim it.
    # FootballWorld keeps its observer-local fixed-shape implementation and
    # reuses the existing box lead/goal-side distance rather than importing a
    # dense fitted table or another coefficient.
    predicted_opponent_position = (
        context.player_position + context.player_velocity * jnp.float32(box_mark_lead_s)
    )
    own_goal_position = jnp.asarray((-half_length, 0.0), dtype=jnp.float32)
    open_field_goal_delta = predicted_opponent_position - own_goal_position
    # Negative squared distance has the same threat order as negative distance
    # and is also reusable by the box assignment without a square root.
    open_field_threat = -jnp.sum(jnp.square(open_field_goal_delta), axis=-1)
    opponent_outfielder = context.opponent & (
        policy_state.role[None, :] != ROLE_GOALKEEPER
    )
    has_open_field_threat = jnp.any(opponent_outfielder, axis=-1)
    open_field_runner = jnp.argmax(
        jnp.where(opponent_outfielder, open_field_threat, -jnp.inf),
        axis=-1,
    )
    observer_row = jnp.arange(self_index.shape[0], dtype=jnp.int32)
    open_field_runner_position = predicted_opponent_position[
        observer_row, open_field_runner
    ]
    open_field_mark_target = _goal_side_mark_target(
        open_field_runner_position,
        own_goal_x=-half_length,
        distance_m=box_mark_goal_side_distance_m,
    )
    open_field_mark_distance_squared = jnp.sum(
        jnp.square(context.player_position - open_field_runner_position[:, None, :]),
        axis=-1,
    )

    # In the own box, preserve press/cover roles and assign up to three other
    # outfielders one-to-one to predicted near-post, far-post, and cutback
    # runners. Threat is the predicted distance to the own goal, so velocity
    # enters causally without inventing an xG model or pooling observer rows.
    candidate_i = team_outfielder[:, :, None]
    candidate_j = team_outfielder[:, None, :]
    distance_i = player_ball_distance_sq[:, :, None]
    distance_j = player_ball_distance_sq[:, None, :]
    slot_i = roster_slot[:, :, None]
    slot_j = roster_slot[:, None, :]
    closer_matrix = (
        candidate_i
        & candidate_j
        & ((distance_i < distance_j) | ((distance_i == distance_j) & (slot_i < slot_j)))
    )
    all_pressure_rank = jnp.sum(closer_matrix, axis=1)
    reserved_pressure_slots = active_pressure_count[:, None] + jnp.int32(1)
    open_field_defender = team_outfielder & (
        all_pressure_rank >= reserved_pressure_slots
    )
    has_open_field_defender = jnp.any(open_field_defender, axis=-1)
    open_field_marker = jnp.argmin(
        jnp.where(open_field_defender, open_field_mark_distance_squared, jnp.inf),
        axis=-1,
    )
    claims_open_field_mark = self_index == open_field_marker
    box_defender_candidate = team_outfielder & (
        all_pressure_rank >= reserved_pressure_slots
    )
    predicted_runner_position = predicted_opponent_position
    box_edge_x = jnp.float32(-half_length + penalty_area_length)
    runner_in_box = (
        context.opponent
        & (policy_state.role[None, :] != ROLE_GOALKEEPER)
        & (
            predicted_runner_position[..., 0]
            <= box_edge_x + jnp.float32(box_mark_runner_margin_m)
        )
        & (
            jnp.abs(predicted_runner_position[..., 1])
            <= 0.5 * penalty_area_width + jnp.float32(box_mark_runner_margin_m)
        )
    )
    runner_threat = open_field_threat
    box_assignment = _assign_box_runners(
        context.player_position,
        box_defender_candidate,
        predicted_runner_position,
        runner_in_box,
        runner_threat,
        ball_xy,
    )
    assigned_runner = box_assignment[observer_row, self_index]
    safe_runner = jnp.clip(assigned_runner, 0, self_index.shape[0] - 1)
    assigned_runner_position = predicted_runner_position[observer_row, safe_runner]
    box_mark_target = _goal_side_mark_target(
        assigned_runner_position,
        own_goal_x=-half_length,
        distance_m=box_mark_goal_side_distance_m,
    )
    ball_threatens_box = ball_xy[:, 0] <= box_edge_x + jnp.float32(
        box_mark_ball_margin_m
    )
    box_mark = (
        opponent_possession
        & context.ball_visible
        & context.self_active
        & ball_threatens_box
        & (assigned_runner >= 0)
        & (~direct_pressure)
        & (~pressure_cover)
    )
    open_field_mark = (
        opponent_possession
        & context.ball_visible
        & context.self_active
        & (~ball_threatens_box)
        & has_open_field_threat
        & has_open_field_defender
        & claims_open_field_mark
        & (~direct_pressure)
        & (~pressure_cover)
    )
    # Every tactical plan tracks the one primary open-field threat without
    # abandoning its zonal anchor. ZONA_MISTA's existing mixed-mark gain raises
    # that sole marker to full commitment; it never creates a second claimant.
    open_field_mark_gain = jnp.clip(
        jnp.float32(OPEN_FIELD_MARK_BASE_GAIN) + tactical.mixed_wide_mark_gain,
        0.0,
        1.0,
    )
    target = jnp.where(
        open_field_mark[:, None],
        target + open_field_mark_gain[:, None] * (open_field_mark_target - target),
        target,
    )
    target = jnp.where(box_mark[:, None], box_mark_target, target)

    # During settled defence the cover player leads visible ball motion while
    # preserving a goal-side buffer. Immediately after a loss, it closes the
    # nearest visible outlet lane instead. The values are MESSI policy design
    # priors; only the causal role separation is asserted here.
    own_goal_offset = jnp.asarray((-1.0, 0.0), dtype=jnp.float32)
    normal_cover_target = (
        ball_xy
        + context.ball_velocity[:, :2] * PRESSURE_COVER_BALL_LEAD_S
        + own_goal_offset * PRESSURE_COVER_DISTANCE_M
    )
    outlet_candidate = (
        context.opponent
        & (policy_state.role[None, :] != ROLE_GOALKEEPER)
        & (
            player_ball_distance_sq
            > COUNTERPRESS_OUTLET_MIN_DISTANCE_M * COUNTERPRESS_OUTLET_MIN_DISTANCE_M
        )
    )
    has_outlet = jnp.any(outlet_candidate, axis=-1)
    outlet_index = jnp.argmin(
        jnp.where(outlet_candidate, player_ball_distance_sq, jnp.inf), axis=-1
    )
    outlet_position = context.player_position[
        jnp.arange(self_index.shape[0]), outlet_index
    ]
    outlet_offset = outlet_position - ball_xy
    outlet_distance = jnp.linalg.norm(outlet_offset, axis=-1)
    outlet_direction = outlet_offset / jnp.maximum(
        outlet_distance[:, None], DIRECTION_EPS_M
    )
    lane_step = jnp.clip(
        COUNTERPRESS_LANE_DISTANCE_FRACTION * outlet_distance,
        COUNTERPRESS_LANE_MIN_STEP_M,
        COUNTERPRESS_LANE_MAX_STEP_M,
    )
    counterpress_cover_target = (
        ball_xy
        + outlet_direction * lane_step[:, None]
        + own_goal_offset * COUNTERPRESS_COVER_GOAL_SIDE_M
    )
    use_outlet_cover = counterpress_active & has_outlet
    cover_target = jnp.where(
        use_outlet_cover[:, None], counterpress_cover_target, normal_cover_target
    )
    target = jnp.where(pressure_cover[:, None], cover_target, target)

    supporting_pressure_target = (
        ball_xy
        + context.ball_velocity[:, :2] * SECONDARY_PRESSURE_BALL_LEAD_S
        + own_goal_offset * SECONDARY_PRESSURE_GOAL_SIDE_GAP_M
    )
    target = jnp.where(supporting_pressure[:, None], supporting_pressure_target, target)

    # One pressure actor contains from the goal side.  The caller decides who
    # is nearest using its own observation row; shape.py only applies the role.
    pressure_target = ball_xy + jnp.asarray(
        [-PRESSURE_GOAL_SIDE_GAP_M, 0.0], dtype=jnp.float32
    )
    pressure_override = (~own_possession) & nearest_pressure & context.ball_visible
    target = jnp.where(pressure_override[:, None], pressure_target, target)

    # Put only forwards on the shoulder when the visible line is meaningfully
    # ahead of the ball. On the carrier episode's existing timing-error draw,
    # the same 0.75 m uncertainty already used by prospective receiver gating
    # shifts both their perceived shoulder and cap. The exact environment Law
    # 11 line remains unchanged, so this creates a genuine timing mistake
    # rather than declaring a known-offside target eligible. Hidden-line rows
    # fail closed independently.
    observable_line, has_offside_cap = _observable_offside_line(context)
    forward_role = (role == ROLE_CENTRE_FORWARD) | (role == ROLE_WIDE_FORWARD)
    selected_run_behind = (
        own_possession
        & forward_role
        & run_behind_release
        & (self_index == run_behind_receiver)
    )
    forward_line_error = jnp.where(
        selected_run_behind & run_behind_timing_error,
        offside_line_error_m,
        jnp.float32(0.0),
    )
    perceived_line = observable_line + forward_line_error
    forward_shoulder = (
        own_possession
        & has_offside_cap
        & context.ball_visible
        & context.self_active
        & forward_role
        & (observable_line > ball_xy[:, 0] + forward_run_min_gap_m)
    )
    shoulder_x = perceived_line - jnp.float32(OFFSIDE_SAFETY_MARGIN_M)
    target_x = jnp.where(
        forward_shoulder,
        jnp.maximum(target[:, 0], shoulder_x),
        target[:, 0],
    )

    # The cap is applied after attacking support and local overrides.  It uses
    # only this row's visible evidence and never reconstructs a hidden line.
    offside_cap = jnp.where(
        perceived_line > OFFSIDE_SAFETY_MARGIN_M,
        perceived_line - jnp.float32(OFFSIDE_SAFETY_MARGIN_M),
        jnp.float32(0.0),
    )
    cap_attack = own_possession & has_offside_cap & (role != ROLE_GOALKEEPER)
    target_x = jnp.where(
        cap_attack,
        jnp.minimum(target_x, offside_cap),
        target_x,
    )

    x_limit = jnp.maximum(
        jnp.asarray(half_length, dtype=jnp.float32) - PITCH_X_INSET_M,
        jnp.float32(0.0),
    )
    y_limit = jnp.maximum(
        jnp.asarray(half_width, dtype=jnp.float32) - PITCH_Y_INSET_M,
        jnp.float32(0.0),
    )
    target = jnp.stack(
        (
            jnp.clip(target_x, -x_limit, x_limit),
            jnp.clip(target[:, 1], -y_limit, y_limit),
        ),
        axis=-1,
    ).astype(jnp.float32)

    displacement = target - context.self_position
    distance = jnp.linalg.norm(displacement, axis=-1)
    direction = jnp.where(
        (distance > DIRECTION_EPS_M)[:, None],
        displacement / jnp.maximum(distance[:, None], DIRECTION_EPS_M),
        jnp.float32(0.0),
    )

    target = jnp.where(context.self_active[:, None], target, context.self_position)
    direction = jnp.where(context.self_active[:, None], direction, jnp.float32(0.0))
    return ShapeMovement(
        target=target.astype(jnp.float32),
        direction=direction.astype(jnp.float32),
        distance=jnp.where(context.self_active, distance, 0.0).astype(jnp.float32),
        urgent=(
            context.self_active
            & (
                overlap
                | central_support
                | far_support
                | cutback_support
                | progressive_support
                | third_man_support
                | wide_pattern_support
                | switch_setup_support
                | switch_release_support
                | designated_forward_support
                | squeeze
                | open_field_mark
                | pressure_cover
                | supporting_pressure
                | box_mark
            )
        ),
        direct_pressure=direct_pressure.astype(jnp.bool_),
    )


__all__ = [
    "ShapeMovement",
    "_forward_pocket_delta",
    "shape_movement",
]
