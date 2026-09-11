"""Deliberate player-ball contact for one physics substep.

Policies use a six-way categorical intent plus eight continuous controls.
Contact mechanism and realized outcome are inferred from physical/rules state
and remain separate from requested intent.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.action import ActionScale
from footballworld.config.ball_physics import BallPhysics
from footballworld.config.body_contact import BodyContact
from footballworld.config.contact_timing import ContactTiming
from footballworld.config.contest import Contest
from footballworld.config.geometry import Ball, Stadium
from footballworld.config.reach import Reach
from footballworld.core.action import IntentAction
from footballworld.core.constants import (
    DIV_EPS,
    GEOMETRY_EPS,
    NO_PLAYER,
    NO_TEAM,
    RK_CORNER,
    RK_GK_HOLD,
    RK_GOALKICK,
    RK_NONE,
    RK_PENALTY,
    RK_THROWIN,
    SAFE_NORM_EPS,
    SQUARED_EPS,
    STATIONARY_SPEED_EPS,
)
from footballworld.core.contact import (
    ACTION_INTENT_COUNT,
    INTENT_CHALLENGE,
    INTENT_CLEAR,
    INTENT_CONTROL,
    INTENT_MOVE,
    INTENT_PASS,
    INTENT_SHOT,
    INTENT_SOURCE_NONE,
    LAW11_DEFLECTION_NO_RESET,
    LAW11_DELIBERATE_PLAY_RESET,
    LAW11_DELIBERATE_SAVE_NO_RESET,
    LAW11_DIRECT_RESTART_EXEMPTION,
    LAW11_NONE,
    MECHANISM_CHEST,
    MECHANISM_FOOT,
    MECHANISM_GOALKEEPER_HAND,
    MECHANISM_HEAD,
    MECHANISM_NONE,
    MECHANISM_THROW,
    OUTCOME_CATCH,
    OUTCOME_DEFLECTION,
    OUTCOME_INTERCEPTION,
    OUTCOME_MISCONTROL,
    OUTCOME_NONE,
    OUTCOME_PARRY,
    OUTCOME_RELEASE,
    OUTCOME_TACKLE_WON,
    OUTCOME_TRAP,
    ContactOccurrence,
    ContactResult,
)
from footballworld.core.state import (
    RestartReleaseProvenance,
    State,
)
from footballworld.dynamics.action import PhysicsAction, decode_physics_action
from footballworld.dynamics.ball import advance_smooth
from footballworld.dynamics.contact_predicates import (
    evaluate_contact_predicates,
    fresh_trap_control_grace,
    opponent_control_continuation,
)
from footballworld.dynamics.contact_response import resolve_control_response
from footballworld.dynamics.contest import (
    ContestOverride,
    ContestResult,
    resolve_contest,
)
from footballworld.dynamics.goal_threat import is_goal_threat
from footballworld.dynamics.passive_contact import _circle_interval, _slab_interval
from footballworld.rules.action_legality import restart_intent_allowed
from footballworld.rules.gk_handling_restriction import (
    goalkeeper_hand_restricted_mask,
    update_backpass_after_deliberate_attempt,
)
from footballworld.rules.restart_legality import restart_actor_mask


class _DecodedAction(NamedTuple):
    contact: jax.Array
    direction: jax.Array
    power: jax.Array
    launch: jax.Array
    spin: jax.Array
    requested_intent: jax.Array
    intent_source: jax.Array


class DeliberateContactStep(NamedTuple):
    """State plus the contact realized during this physics substep."""

    state: State
    occurred: jax.Array
    actor: jax.Array
    contact: ContactResult
    contest: ContestResult
    attempted: jax.Array
    occurrence: ContactOccurrence


class ActiveContactEvent(NamedTuple):
    """First deliberate reach-envelope entry on relative player/ball paths."""

    occurred: jax.Array
    time_fraction: jax.Array


def _intersect_intervals(
    first_valid: jax.Array,
    first_entry: jax.Array,
    first_exit: jax.Array,
    second_valid: jax.Array,
    second_entry: jax.Array,
    second_exit: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    entry = jnp.maximum(first_entry, second_entry)
    exit = jnp.minimum(first_exit, second_exit)
    return first_valid & second_valid & (entry <= exit), entry, exit


def _active_horizontal_interval(
    state: State,
    ball_path_delta: jax.Array,
    player_start_position: jax.Array,
    player_path_delta: jax.Array,
    radius: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the shared horizontal interval for one reach-radius family."""

    relative_origin = state.ball.position[:2] - player_start_position
    relative_delta = ball_path_delta[:2] - player_path_delta
    predicate_radius = jnp.sqrt(
        jnp.maximum(
            jnp.asarray(radius, dtype=relative_origin.dtype) ** 2 - SAFE_NORM_EPS, 0.0
        )
    )
    return _circle_interval(relative_origin, relative_delta, predicate_radius)


def _active_reach_interval(
    state: State,
    ball_path_delta: jax.Array,
    player_start_position: jax.Array,
    player_path_delta: jax.Array,
    radius: float,
    lower_height: jax.Array,
    upper_height: jax.Array,
    *,
    horizontal: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if horizontal is None:
        horizontal = _active_horizontal_interval(
            state,
            ball_path_delta,
            player_start_position,
            player_path_delta,
            radius,
        )
    player_count = player_start_position.shape[0]
    vertical = _slab_interval(
        jnp.broadcast_to(state.ball.position[2], (player_count, 1)),
        jnp.broadcast_to(ball_path_delta[2], (player_count, 1)),
        lower_height[:, None],
        upper_height[:, None],
    )
    return _intersect_intervals(*horizontal, *vertical)


def _detect_active_contact_requested(
    state: State,
    action: PhysicsAction,
    restart_release_allowed: jax.Array,
    contact_attempted: jax.Array,
    ball_path_delta: jax.Array,
    *,
    remaining_dt: jax.Array,
    player_start_position: jax.Array,
    player_path_delta: jax.Array,
    excluded_actor: jax.Array = NO_PLAYER,
    ball_geometry: Ball = Ball(),
    stadium: Stadium = Stadium(),
    reach: Reach = Reach(),
    scale: ActionScale = ActionScale(),
    body: BodyContact = BodyContact(),
    ball_physics: BallPhysics = BallPhysics(),
) -> ActiveContactEvent:
    """Return the first exact entry into an action-eligible reach volume.

    Ball and player horizontal motion are reduced to one relative segment.
    Foot, chest, head, and goalkeeper-hand volumes are intersected with their
    existing height, restart, recovery, speed, and penalty-area gates. The
    authoritative predicates are evaluated again by ``resolve_contact_step``
    at the returned impact state; this detector only schedules that one call.
    """

    players = state.players
    dtype = state.ball.position.dtype
    player_count = players.position.shape[0]
    player_index = jnp.arange(player_count, dtype=jnp.int32)
    restart_release_allowed = jnp.broadcast_to(
        jnp.asarray(restart_release_allowed, dtype=jnp.bool_), (player_count,)
    )
    contact_attempted = jnp.asarray(contact_attempted, dtype=jnp.bool_)
    safe_intent = jnp.clip(
        action.requested_intent, INTENT_MOVE, ACTION_INTENT_COUNT - 1
    )
    active_row = (
        players.active
        & (~contact_attempted)
        & (player_index != jnp.asarray(excluded_actor, dtype=jnp.int32))
    )
    # contact_lock_substeps debounces immediate repeated touches. In contrast,
    # aerial_recovery_substeps blocks every deliberate intervention while a
    # jumper or diving goalkeeper recovers; neither lock disables movement or
    # passive body contact.
    base_recovery = (
        (players.aerial_recovery_substeps <= 0)
        & (players.contact_lock_substeps <= 0)
        & (players.possession_loss_lock_substeps <= 0)
    )

    designated_restart = restart_actor_mask(state)
    restart_active = state.restart.kind != RK_NONE
    structural = restart_intent_allowed(state, restart_release_allowed)
    selected_structural = structural[player_index, safe_intent]
    ordinary_intent = (
        (safe_intent == INTENT_CONTROL)
        | (safe_intent == INTENT_PASS)
        | (safe_intent == INTENT_SHOT)
        | (safe_intent == INTENT_CLEAR)
    )
    ordinary_requested = selected_structural & ordinary_intent

    pelvis = players.height * scale.pelvis_height_factor
    torso_top = body.torso_top_height(players.height)
    minus_infinity = jnp.full(player_count, -jnp.inf, dtype=dtype)
    # Foot, chest, and head partition the same horizontal swept cylinder.
    # Reuse the quadratic roots; only the vertical slabs differ.
    ordinary_horizontal = _active_horizontal_interval(
        state,
        ball_path_delta,
        player_start_position,
        player_path_delta,
        reach.carry_radius_m + ball_geometry.radius,
    )
    foot = _active_reach_interval(
        state,
        ball_path_delta,
        player_start_position,
        player_path_delta,
        reach.carry_radius_m + ball_geometry.radius,
        minus_infinity,
        pelvis + ball_geometry.radius,
        horizontal=ordinary_horizontal,
    )
    chest = _active_reach_interval(
        state,
        ball_path_delta,
        player_start_position,
        player_path_delta,
        reach.carry_radius_m + ball_geometry.radius,
        pelvis + ball_geometry.radius,
        torso_top + ball_geometry.radius,
        horizontal=ordinary_horizontal,
    )
    head = _active_reach_interval(
        state,
        ball_path_delta,
        player_start_position,
        player_path_delta,
        reach.carry_radius_m + ball_geometry.radius,
        torso_top + ball_geometry.radius,
        players.reach_height + ball_geometry.radius,
        horizontal=ordinary_horizontal,
    )

    def foot_speed_allowed(interval, needed):
        def evaluate(_):
            safe_fraction = jnp.where(interval[0], interval[1], 0.0)
            velocity_at_entry = jax.vmap(
                lambda fraction: (
                    advance_smooth(
                        state.ball,
                        dt=remaining_dt * fraction,
                        geometry=ball_geometry,
                        physics=ball_physics,
                    ).velocity
                )
            )(safe_fraction)
            height_at_entry = state.ball.position[2] + interval[1] * ball_path_delta[2]
            return (
                _norm(velocity_at_entry[:, :2])
                + reach.height_speed_penalty_mps_per_m * height_at_entry
                <= reach.block_speed_limit_mps
            )

        return jax.lax.cond(
            jnp.any(needed & interval[0]),
            evaluate,
            lambda _: jnp.ones(player_count, dtype=jnp.bool_),
            operand=None,
        )

    carrier_in_range = (
        (state.possession.player >= 0)
        & (state.possession.player < player_count)
        & (state.possession.control_ticks > 0)
    )
    safe_carrier = jnp.clip(state.possession.player, 0, player_count - 1)
    retained_body_trap = (
        carrier_in_range
        & (state.possession.last_contact.actor == safe_carrier)
        & (state.possession.last_contact.intent == INTENT_CONTROL)
        & (state.possession.last_contact.outcome == OUTCOME_TRAP)
        & (state.possession.last_contact.mechanism == MECHANISM_CHEST)
    )
    carrier_top = jnp.where(
        retained_body_trap,
        players.reach_height[safe_carrier],
        pelvis[safe_carrier],
    )
    carrier_horizontal = _circle_interval(
        (state.ball.position[:2] - player_start_position[safe_carrier])[None, :],
        (ball_path_delta[:2] - player_path_delta[safe_carrier])[None, :],
        jnp.sqrt((reach.carry_radius_m + ball_geometry.radius) ** 2 - SAFE_NORM_EPS),
    )
    carrier_vertical = _slab_interval(
        state.ball.position[2][None, None],
        ball_path_delta[2][None, None],
        jnp.asarray([[-jnp.inf]], dtype=dtype),
        (carrier_top + ball_geometry.radius)[None, None],
    )
    carrier_interval = _intersect_intervals(*carrier_horizontal, *carrier_vertical)
    carrier_valid = (
        carrier_interval[0][0]
        & carrier_in_range
        & players.active[safe_carrier]
        & (state.possession.team == players.team_id[safe_carrier])
    )
    carrier_entry = carrier_interval[1][0]
    carrier_exit = carrier_interval[2][0]
    opposing_carrier = carrier_valid & (
        players.team_id != players.team_id[safe_carrier]
    )

    last_actor = state.possession.last_contact.actor
    last_actor_valid = (last_actor >= 0) & (last_actor < player_count)
    safe_last_actor = jnp.clip(last_actor, 0, player_count - 1)
    opponent_release = (
        last_actor_valid
        & (state.possession.last_contact.outcome == OUTCOME_RELEASE)
        & (
            (state.possession.last_contact.intent == INTENT_PASS)
            | (state.possession.last_contact.intent == INTENT_SHOT)
            | (state.possession.last_contact.intent == INTENT_CLEAR)
        )
        & state.ball.live
        & (_norm(state.ball.velocity) > STATIONARY_SPEED_EPS)
        & (players.team_id != players.team_id[safe_last_actor])
    )
    control_continuation = opponent_control_continuation(state, carrier_valid)
    opponent_interceptable_play = opponent_release | control_continuation
    challenge_requested = selected_structural & (safe_intent == INTENT_CHALLENGE)
    challenge_recovery = base_recovery & (players.challenge_recovery_substeps <= 0)

    challenge_horizontal = _active_horizontal_interval(
        state,
        ball_path_delta,
        player_start_position,
        player_path_delta,
        reach.challenge_radius_m + ball_geometry.radius,
    )
    challenge_foot = _active_reach_interval(
        state,
        ball_path_delta,
        player_start_position,
        player_path_delta,
        reach.challenge_radius_m + ball_geometry.radius,
        minus_infinity,
        pelvis + ball_geometry.radius,
        horizontal=challenge_horizontal,
    )
    challenge_chest = _active_reach_interval(
        state,
        ball_path_delta,
        player_start_position,
        player_path_delta,
        reach.challenge_radius_m + ball_geometry.radius,
        pelvis + ball_geometry.radius,
        torso_top + ball_geometry.radius,
        horizontal=challenge_horizontal,
    )
    challenge_head = _active_reach_interval(
        state,
        ball_path_delta,
        player_start_position,
        player_path_delta,
        reach.challenge_radius_m + ball_geometry.radius,
        torso_top + ball_geometry.radius,
        players.reach_height + ball_geometry.radius,
        horizontal=challenge_horizontal,
    )

    def challenge_time(interval, mechanism_recovery):
        carrier_overlap = _intersect_intervals(
            *interval,
            jnp.broadcast_to(carrier_valid, (player_count,)),
            jnp.broadcast_to(carrier_entry, (player_count,)),
            jnp.broadcast_to(carrier_exit, (player_count,)),
        )
        release_branch = opponent_interceptable_play & challenge_requested & interval[0]
        carrier_branch = challenge_requested & opposing_carrier & carrier_overlap[0]
        entry = jnp.minimum(
            jnp.where(release_branch, interval[1], jnp.inf),
            jnp.where(carrier_branch, carrier_overlap[1], jnp.inf),
        )
        valid = (
            active_row & challenge_recovery & mechanism_recovery & jnp.isfinite(entry)
        )
        return jnp.where(valid, entry, jnp.inf)

    challenge_speed_needed = (
        active_row
        & challenge_recovery
        & challenge_requested
        & (opponent_interceptable_play | opposing_carrier)
    )
    challenge_foot_time = challenge_time(
        challenge_foot,
        foot_speed_allowed(challenge_foot, challenge_speed_needed),
    )
    challenge_chest_time = challenge_time(
        challenge_chest, jnp.ones(player_count, dtype=jnp.bool_)
    )
    challenge_head_time = challenge_time(
        challenge_head, jnp.ones(player_count, dtype=jnp.bool_)
    )

    def ordinary_time(interval, mechanism_recovery):
        valid = (
            interval[0]
            & active_row
            & ordinary_requested
            & base_recovery
            & mechanism_recovery
        )
        return jnp.where(valid, interval[1], jnp.inf)

    ordinary_speed_needed = active_row & ordinary_requested & base_recovery
    foot_time = ordinary_time(
        foot,
        foot_speed_allowed(foot, ordinary_speed_needed),
    )
    chest_time = ordinary_time(chest, jnp.ones(player_count, dtype=jnp.bool_))
    head_time = ordinary_time(head, jnp.ones(player_count, dtype=jnp.bool_))

    team_direction = state.attack_direction[players.team_id]
    own_goal_x = -team_direction * stadium.half_length
    penalty_inner_x = own_goal_x + team_direction * stadium.penalty_area_length
    penalty_lower = jnp.stack(
        (
            jnp.minimum(own_goal_x, penalty_inner_x),
            jnp.full(player_count, -0.5 * stadium.penalty_area_width, dtype=dtype),
        ),
        axis=-1,
    )
    penalty_upper = jnp.stack(
        (
            jnp.maximum(own_goal_x, penalty_inner_x),
            jnp.full(player_count, 0.5 * stadium.penalty_area_width, dtype=dtype),
        ),
        axis=-1,
    )
    penalty_interval = _slab_interval(
        jnp.broadcast_to(state.ball.position[:2], (player_count, 2)),
        jnp.broadcast_to(ball_path_delta[:2], (player_count, 2)),
        penalty_lower,
        penalty_upper,
    )
    goalkeeper_reach = _active_reach_interval(
        state,
        ball_path_delta,
        player_start_position,
        player_path_delta,
        reach.goalkeeper_radius_m + ball_geometry.radius,
        minus_infinity,
        players.reach_height + ball_geometry.radius,
    )
    goalkeeper_interval = _intersect_intervals(*goalkeeper_reach, *penalty_interval)
    hand_restricted = goalkeeper_hand_restricted_mask(state)
    explicit_hand = (safe_intent == INTENT_CONTROL) | (safe_intent == INTENT_CLEAR)
    goalkeeper_requested = selected_structural & explicit_hand
    goalkeeper_valid = (
        goalkeeper_interval[0]
        & active_row
        & base_recovery
        & state.ball.live
        & (~restart_active)
        & players.is_goalkeeper
        & (~hand_restricted)
        & goalkeeper_requested
    )
    goalkeeper_time = jnp.where(goalkeeper_valid, goalkeeper_interval[1], jnp.inf)

    held_release = (
        active_row
        & base_recovery
        & designated_restart
        & restart_release_allowed
        & players.is_goalkeeper
        & (state.restart.kind == RK_GK_HOLD)
        & selected_structural
        & ordinary_intent
    )
    held_time = jnp.where(held_release, jnp.asarray(0.0, dtype=dtype), jnp.inf)

    earliest = jnp.min(
        jnp.stack(
            (
                foot_time,
                chest_time,
                head_time,
                challenge_foot_time,
                challenge_chest_time,
                challenge_head_time,
                goalkeeper_time,
                held_time,
            ),
            axis=0,
        )
    )
    occurred = jnp.isfinite(earliest)
    # Enter one representable instant into the closed predicate volume. This
    # is numerical progress, not a physical tolerance or sampled time axis.
    earliest = jnp.where(
        occurred & (earliest > 0.0) & (earliest < 1.0),
        jnp.nextafter(earliest, jnp.asarray(1.0, dtype=dtype)),
        earliest,
    )
    return ActiveContactEvent(
        occurred=occurred,
        time_fraction=jnp.where(occurred, earliest, 0.0),
    )


def active_contact_possible(
    state: State,
    action: PhysicsAction,
    restart_release_allowed: jax.Array,
    contact_attempted: jax.Array,
    ball_path_delta: jax.Array,
    player_start_position: jax.Array,
    player_path_delta: jax.Array,
    *,
    excluded_actor: jax.Array,
    search_enabled: jax.Array,
    ball_geometry: Ball,
    reach: Reach,
) -> jax.Array:
    """Conservatively reject requested reaches outside every swept AABB."""

    players = state.players
    dtype = state.ball.position.dtype
    player_count = players.position.shape[0]
    player_index = jnp.arange(player_count, dtype=jnp.int32)
    active_row = (
        players.active
        & (~jnp.asarray(contact_attempted, dtype=jnp.bool_))
        & (player_index != jnp.asarray(excluded_actor, dtype=jnp.int32))
    )
    explicit_request = active_row & (action.requested_intent != INTENT_MOVE)
    requested = jnp.asarray(search_enabled, dtype=jnp.bool_) & jnp.any(explicit_request)

    # A goalkeeper hold releases at time zero and therefore must not depend on
    # reconstructed spatial coherence. The exact detector remains authoritative
    # for designated actor, recovery, intent, and restart legality.
    release_allowed = jnp.broadcast_to(
        jnp.asarray(restart_release_allowed, dtype=jnp.bool_),
        (player_count,),
    )
    goalkeeper_hold_release = (state.restart.kind == RK_GK_HOLD) & jnp.any(
        explicit_request & release_allowed
    )

    maximum_radius = jnp.asarray(
        max(
            reach.carry_radius_m,
            reach.challenge_radius_m,
            reach.goalkeeper_radius_m,
        )
        + ball_geometry.radius,
        dtype=dtype,
    )
    ball_start = state.ball.position
    ball_end = ball_start + ball_path_delta
    ball_min_xy = jnp.minimum(ball_start[:2], ball_end[:2])
    ball_max_xy = jnp.maximum(ball_start[:2], ball_end[:2])
    player_end = player_start_position + player_path_delta
    player_min_xy = jnp.minimum(player_start_position, player_end) - maximum_radius
    player_max_xy = jnp.maximum(player_start_position, player_end) + maximum_radius
    horizontal_overlap = jnp.all(
        (ball_max_xy[None, :] >= player_min_xy)
        & (ball_min_xy[None, :] <= player_max_xy),
        axis=-1,
    )
    # Foot envelopes deliberately extend below the pitch, so only the upper
    # reach plane can safely reject a path vertically.
    ball_min_z = jnp.minimum(ball_start[2], ball_end[2])
    vertical_overlap = ball_min_z <= players.reach_height + ball_geometry.radius
    finite = (
        jnp.all(jnp.isfinite(ball_start))
        & jnp.all(jnp.isfinite(ball_path_delta))
        & jnp.all(jnp.isfinite(player_start_position))
        & jnp.all(jnp.isfinite(player_path_delta))
        & jnp.all(jnp.isfinite(players.reach_height))
    )
    spatial_candidate = jnp.any(
        explicit_request & horizontal_overlap & vertical_overlap
    )
    return requested & (goalkeeper_hold_release | (~finite) | spatial_candidate)


def detect_active_contact(
    state: State,
    action: PhysicsAction,
    restart_release_allowed: jax.Array,
    contact_attempted: jax.Array,
    ball_path_delta: jax.Array,
    *,
    remaining_dt: jax.Array,
    player_start_position: jax.Array,
    player_path_delta: jax.Array,
    excluded_actor: jax.Array = NO_PLAYER,
    search_enabled: jax.Array = True,
    ball_geometry: Ball = Ball(),
    stadium: Stadium = Stadium(),
    reach: Reach = Reach(),
    scale: ActionScale = ActionScale(),
    body: BodyContact = BodyContact(),
    ball_physics: BallPhysics = BallPhysics(),
) -> ActiveContactEvent:
    """Skip the exact swept detector when no requested reach can intersect."""

    requested = active_contact_possible(
        state,
        action,
        restart_release_allowed,
        contact_attempted,
        ball_path_delta,
        player_start_position,
        player_path_delta,
        excluded_actor=excluded_actor,
        search_enabled=search_enabled,
        ball_geometry=ball_geometry,
        reach=reach,
    )

    return jax.lax.cond(
        requested,
        lambda _: _detect_active_contact_requested(
            state,
            action,
            restart_release_allowed,
            contact_attempted,
            ball_path_delta,
            remaining_dt=remaining_dt,
            player_start_position=player_start_position,
            player_path_delta=player_path_delta,
            excluded_actor=excluded_actor,
            ball_geometry=ball_geometry,
            stadium=stadium,
            reach=reach,
            scale=scale,
            body=body,
            ball_physics=ball_physics,
        ),
        lambda _: ActiveContactEvent(
            occurred=jnp.bool_(False),
            time_fraction=jnp.asarray(0.0, dtype=state.ball.position.dtype),
        ),
        operand=None,
    )


def _norm(vector: jax.Array, *, axis: int = -1) -> jax.Array:
    return jnp.sqrt(jnp.sum(vector * vector, axis=axis) + SAFE_NORM_EPS)


def _decode_action(
    state: State,
    action: PhysicsAction,
    *,
    ball_geometry: Ball,
    scale: ActionScale,
) -> _DecodedAction:
    fallback = state.players.body_forward
    direction = jnp.where(
        (action.force_power > STATIONARY_SPEED_EPS)[:, None],
        action.force_direction,
        fallback,
    )

    foot_punt_height = jnp.maximum(
        ball_geometry.radius,
        state.players.height * scale.pelvis_height_factor,
    )
    held_foot_punt = (state.restart.kind == RK_GK_HOLD) & state.players.is_goalkeeper
    action_ball_height = jnp.where(
        held_foot_punt, foot_punt_height, state.ball.position[2]
    )
    height_fraction = jnp.clip(
        (action_ball_height - ball_geometry.radius)
        / (scale.ground_launch_down_reference_height_m - ball_geometry.radius),
        0.0,
        1.0,
    )
    launch_floor = -(
        scale.ground_launch_down_max_radians
        + (scale.launch_max_radians - scale.ground_launch_down_max_radians)
        * height_fraction
    )
    launch = launch_floor + action.launch * (scale.launch_max_radians - launch_floor)
    return _DecodedAction(
        contact=action.contact,
        direction=direction,
        power=action.force_power,
        launch=launch,
        spin=action.spin,
        requested_intent=action.requested_intent,
        intent_source=action.intent_source,
    )


def _limit_generated_state(
    incoming: jax.Array,
    candidate: jax.Array,
    generated_limit: jax.Array,
) -> jax.Array:
    incoming_size = _norm(incoming)
    candidate_size = _norm(candidate)
    limit = jnp.maximum(incoming_size, generated_limit)
    return candidate * jnp.minimum(1.0, limit / (candidate_size + DIV_EPS))


def _legal_restart_direction(
    state: State,
    winner: jax.Array,
    direction: jax.Array,
    launch: jax.Array,
    release_speed: jax.Array,
    *,
    dt: float,
    scale: ActionScale,
) -> jax.Array:
    """Project penalty kicks and throw-ins into their legal half-planes.

    The action remains continuous. Only a designated taker's actual restart
    release is projected, preserving the requested tangent component whenever
    possible. Four float32 ULPs of first-step movement prevent a mathematically
    positive component from rounding back to a stationary coordinate.
    """

    dtype = direction.dtype
    restart_take = (state.restart.kind != RK_NONE) & (state.restart.taker == winner)
    horizontal_speed = release_speed * jnp.maximum(jnp.cos(launch), 0.0)

    def project_half_plane(
        raw: jax.Array,
        axis: jax.Array,
        minimum_cosine: jax.Array,
        tangent_fallback: jax.Array,
    ) -> jax.Array:
        component = jnp.dot(raw, axis)
        tangent = raw - component * axis
        tangent_norm = _norm(tangent[None, :])[0]
        tangent_unit = jnp.where(
            tangent_norm > GEOMETRY_EPS,
            tangent / (tangent_norm + DIV_EPS),
            tangent_fallback,
        )
        projected = (
            minimum_cosine * axis
            + jnp.sqrt(jnp.maximum(0.0, 1.0 - minimum_cosine**2)) * tangent_unit
        )
        projected = jnp.where(tangent_norm > GEOMETRY_EPS, projected, axis)
        return jnp.where(component >= minimum_cosine, raw, projected)

    attack = state.attack_direction[state.players.team_id[winner]]
    forward_axis = jnp.asarray([attack, 0.0], dtype=dtype)
    forward_target = jnp.where(
        attack > 0.0,
        jnp.asarray(jnp.inf, dtype=dtype),
        jnp.asarray(-jnp.inf, dtype=dtype),
    )
    forward_ulp = jnp.abs(
        jnp.nextafter(state.ball.position[0], forward_target) - state.ball.position[0]
    )
    forward_cosine = jnp.clip(
        4.0 * forward_ulp / (horizontal_speed * jnp.asarray(dt, dtype=dtype) + DIV_EPS),
        GEOMETRY_EPS,
        1.0,
    )
    penalty_direction = project_half_plane(
        direction,
        forward_axis,
        forward_cosine,
        jnp.asarray([0.0, 1.0], dtype=dtype),
    )
    penalty_take = restart_take & (state.restart.kind == RK_PENALTY)
    legal = jnp.where(penalty_take, penalty_direction, direction)

    touchline_side = jnp.where(
        state.ball.position[1] != 0.0,
        jnp.sign(state.ball.position[1]),
        jnp.where(state.players.position[winner, 1] >= 0.0, 1.0, -1.0),
    )
    inward_axis = jnp.asarray([0.0, -touchline_side], dtype=dtype)
    inward_target = jnp.where(
        touchline_side > 0.0,
        jnp.asarray(-jnp.inf, dtype=dtype),
        jnp.asarray(jnp.inf, dtype=dtype),
    )
    inward_ulp = jnp.abs(
        jnp.nextafter(state.ball.position[1], inward_target) - state.ball.position[1]
    )
    representable_cosine = (
        4.0 * inward_ulp / (horizontal_speed * jnp.asarray(dt, dtype=dtype) + DIV_EPS)
    )
    clear_entry_cosine = jnp.minimum(
        jnp.asarray(scale.restart_min_ball_speed_mps, dtype=dtype),
        horizontal_speed,
    ) / (horizontal_speed + DIV_EPS)
    inward_cosine = jnp.clip(
        jnp.maximum(representable_cosine, clear_entry_cosine),
        GEOMETRY_EPS,
        1.0,
    )
    throw_direction = project_half_plane(
        legal,
        inward_axis,
        inward_cosine,
        forward_axis,
    )
    throw_take = restart_take & (state.restart.kind == RK_THROWIN)
    return jnp.where(throw_take, throw_direction, legal)


def resolve_contact_step(
    state: State,
    action: IntentAction | PhysicsAction,
    key: jax.Array,
    contest_override: ContestOverride,
    restart_release_allowed: jax.Array,
    *,
    dt: float,
    contact_attempted: jax.Array,
    ball_geometry: Ball = Ball(),
    stadium: Stadium = Stadium(),
    reach: Reach = Reach(),
    scale: ActionScale = ActionScale(),
    body: BodyContact = BodyContact(),
    timing: ContactTiming = ContactTiming(),
    ball_physics: BallPhysics = BallPhysics(),
    contest_config: Contest = Contest(),
    regulation_elapsed_fraction: jax.Array = 0.0,
) -> DeliberateContactStep:
    """Resolve one fixed-shape contest and any realized ball contact."""

    if isinstance(action, IntentAction):
        action = decode_physics_action(state, action)
    elif not isinstance(action, PhysicsAction):
        raise TypeError("action must be IntentAction or PhysicsAction")
    decoded = _decode_action(state, action, ball_geometry=ball_geometry, scale=scale)
    predicates = evaluate_contact_predicates(
        state,
        restart_release_allowed,
        requested_intent=decoded.requested_intent,
        ball_geometry=ball_geometry,
        stadium=stadium,
        reach=reach,
        scale=scale,
        body=body,
    )
    candidate = predicates.possible_now & predicates.intent_allowed
    contact_attempted = jnp.asarray(contact_attempted, dtype=jnp.bool_)
    candidate = candidate & (~contact_attempted)
    attempted = contact_attempted | candidate
    carry_edge = reach.carry_radius_m + ball_geometry.radius
    challenge_edge = reach.challenge_radius_m + ball_geometry.radius
    challenge_lunge_fraction = jnp.clip(
        (predicates.distance_xy - carry_edge)
        / jnp.maximum(challenge_edge - carry_edge, DIV_EPS),
        0.0,
        1.0,
    )

    contest = resolve_contest(
        state,
        candidate,
        predicates.distance_xy,
        predicates.goalkeeper_claim
        & (predicates.mechanism == MECHANISM_GOALKEEPER_HAND),
        predicates.verified_controlled_carrier,
        key,
        contest_override,
        goalkeeper_clear=predicates.goalkeeper_hand_clear,
        challenge_request=predicates.challenge_context,
        challenge_interception=predicates.challenge_interception,
        challenge_lunge_fraction=challenge_lunge_fraction,
        regulation_elapsed_fraction=regulation_elapsed_fraction,
        config=contest_config,
        body=body,
        stadium=stadium,
    )
    winner = jnp.clip(contest.actor, 0, state.players.position.shape[0] - 1)
    selected = contest.selected
    mechanism = predicates.mechanism[winner]
    challenge_context = predicates.challenge_context[winner]
    control_request = predicates.control_request[winner]
    intent = decoded.requested_intent[winner].astype(jnp.int32)

    raw_outcome = contest.outcome
    raw_ball_contact = selected & (
        (raw_outcome == OUTCOME_RELEASE)
        | (raw_outcome == OUTCOME_INTERCEPTION)
        | (raw_outcome == OUTCOME_TACKLE_WON)
        | (raw_outcome == OUTCOME_DEFLECTION)
        | (raw_outcome == OUTCOME_CATCH)
        | (raw_outcome == OUTCOME_PARRY)
    )
    restart_contact = (
        raw_ball_contact
        & (state.restart.kind != RK_NONE)
        & (state.restart.taker == winner)
    )
    foot = mechanism == MECHANISM_FOOT
    head = mechanism == MECHANISM_HEAD
    chest = mechanism == MECHANISM_CHEST
    challenge = challenge_context
    throw = mechanism == MECHANISM_THROW
    goalkeeper_hand = mechanism == MECHANISM_GOALKEEPER_HAND
    body_control_request = raw_ball_contact & (intent == INTENT_CONTROL) & chest
    head_control_failure = raw_ball_contact & (intent == INTENT_CONTROL) & head
    caught = raw_ball_contact & goalkeeper_hand & (raw_outcome == OUTCOME_CATCH)
    parried = raw_ball_contact & goalkeeper_hand & (raw_outcome == OUTCOME_PARRY)
    tackle_won = raw_ball_contact & challenge & (raw_outcome == OUTCOME_TACKLE_WON)
    deflected = raw_ball_contact & (raw_outcome == OUTCOME_DEFLECTION)
    hand_restricted = goalkeeper_hand_restricted_mask(state)
    restricted_goalkeeper_clear_foot_attempt = (
        selected
        & foot
        & decoded.contact[winner]
        & (~control_request)
        & hand_restricted[winner]
    )

    direction = decoded.direction[winner]
    launch = decoded.launch[winner]
    generated_speed_limit = jnp.where(
        foot,
        scale.kick_speed_max_mps,
        jnp.where(throw, scale.throw_speed_max_mps, 0.0),
    )
    requested_action_speed = decoded.power[winner] * jnp.where(
        (intent == INTENT_CONTROL) | (intent == INTENT_CHALLENGE),
        scale.control_request_speed_max_mps,
        scale.kick_speed_max_mps,
    )
    requested_release_speed = decoded.power[winner] * generated_speed_limit
    release_speed = jnp.where(
        restart_contact,
        jnp.maximum(requested_release_speed, scale.restart_min_ball_speed_mps),
        requested_release_speed,
    )
    direction = _legal_restart_direction(
        state,
        winner,
        direction,
        launch,
        release_speed,
        dt=dt,
        scale=scale,
    )
    impulse_direction = jnp.asarray(
        [
            jnp.cos(launch) * direction[0],
            jnp.cos(launch) * direction[1],
            jnp.sin(launch),
        ],
        dtype=state.ball.velocity.dtype,
    )
    released_velocity = _limit_generated_state(
        state.ball.velocity,
        state.ball.velocity + release_speed * impulse_direction,
        generated_speed_limit,
    )

    incoming_speed_squared = jnp.sum(state.ball.velocity * state.ball.velocity)
    incoming_speed = jnp.where(
        incoming_speed_squared > SQUARED_EPS,
        jnp.sqrt(incoming_speed_squared),
        0.0,
    )
    redirect_retention = jnp.where(
        head,
        scale.header_speed_retention,
        jnp.where(
            chest,
            scale.chest_speed_retention,
            scale.challenge_speed_retention,
        ),
    )
    redirect_target = redirect_retention * incoming_speed * impulse_direction
    redirected_velocity = state.ball.velocity + decoded.power[winner] * (
        redirect_target - state.ball.velocity
    )

    player_velocity = jnp.asarray(
        [
            state.players.velocity[winner, 0],
            state.players.velocity[winner, 1],
            0.0,
        ],
        dtype=state.ball.velocity.dtype,
    )
    # Chest CONTROL cushions player-relative motion. Its target follows the
    # requested redirect direction but may not exceed the incoming world-speed
    # energy budget; player motion can therefore never propel the ball through
    # this non-foot mechanism. A descending vertical component may be absorbed
    # passively down to the non-upward target before the existing foot-control
    # impulse envelope prices the remaining directional change. This represents
    # cushioning gravity-driven arrival without discounting an upward reversal
    # or introducing another coefficient. "Relationship between acceleration
    # of the foot and lower-leg movement in soccer ball trapping" and
    # "Kinematic factors associated with first-touch control and trap-to-pass
    # transition in football during a dynamic receive-to-pass task" provide
    # foot first-touch mechanics only, not a chest incidence-angle coefficient:
    # this is a mechanics-informed design prior, not a fitted chest-angle law.
    incoming_relative_velocity = state.ball.velocity - player_velocity
    incoming_relative_speed = _norm(incoming_relative_velocity)
    chest_target_relative = (
        redirect_retention * incoming_relative_speed * impulse_direction
    )
    raw_chest_target = player_velocity + chest_target_relative
    raw_chest_target_squared = jnp.sum(raw_chest_target * raw_chest_target)
    energy_feasible = raw_chest_target_squared <= (incoming_speed_squared + SQUARED_EPS)
    raw_chest_target_speed = jnp.sqrt(jnp.maximum(raw_chest_target_squared, 0.0))
    chest_target = raw_chest_target * jnp.minimum(
        1.0,
        incoming_speed / (raw_chest_target_speed + DIV_EPS),
    )
    chest_requested_relative = chest_target - player_velocity
    passive_relative_z = jnp.minimum(chest_requested_relative[2], 0.0)
    passive_descent = (
        body_control_request
        & (incoming_relative_velocity[2] < 0.0)
        & (passive_relative_z > incoming_relative_velocity[2])
    )
    response_incoming = state.ball.velocity.at[2].set(
        jnp.where(
            passive_descent,
            player_velocity[2] + passive_relative_z,
            state.ball.velocity[2],
        )
    )
    requested_relative_velocity = jnp.where(
        body_control_request,
        chest_requested_relative,
        requested_action_speed * impulse_direction,
    )
    control_response = resolve_control_response(
        response_incoming,
        player_velocity,
        requested_relative_velocity,
        impulse_speed_limit_mps=jnp.asarray(
            scale.kick_speed_max_mps, dtype=state.ball.velocity.dtype
        ),
    )
    control_eligible = (
        raw_ball_contact
        & control_request
        & (
            (raw_outcome == OUTCOME_RELEASE)
            | (raw_outcome == OUTCOME_INTERCEPTION)
            | tackle_won
        )
    )
    control_success = control_eligible & control_response.success
    control_failure = control_eligible & (~control_response.success)
    chest_control_success = (
        body_control_request & control_response.success & energy_feasible
    )
    chest_control_failure = body_control_request & (~chest_control_success)

    automatic_direction = jnp.asarray(
        [
            state.players.body_forward[winner, 0],
            state.players.body_forward[winner, 1],
            0.0,
        ],
        dtype=state.ball.velocity.dtype,
    )
    gated_parry = decoded.contact[winner] & (decoded.power[winner] > 0.0)
    parry_direction = jnp.where(
        gated_parry,
        impulse_direction,
        automatic_direction,
    )
    parry_target = incoming_speed * parry_direction
    parry_delta = parry_target - state.ball.velocity
    parry_strength = jnp.where(gated_parry, decoded.power[winner], 1.0)
    parry_fraction = jnp.minimum(
        parry_strength,
        incoming_speed / (_norm(parry_delta) + DIV_EPS),
    )
    parried_velocity = state.ball.velocity + parry_fraction * parry_delta

    deliberate_release = (
        raw_ball_contact
        & (~control_request)
        & (restart_contact | throw | (foot & (~challenge)))
    )
    challenge_redirect = (
        raw_ball_contact
        & challenge
        & (
            deflected
            | (tackle_won & (~control_request))
            | (raw_outcome == OUTCOME_INTERCEPTION)
        )
    )
    body_redirect = raw_ball_contact & (head | chest)
    new_velocity = jnp.where(
        caught,
        jnp.zeros_like(state.ball.velocity),
        jnp.where(
            parried,
            parried_velocity,
            jnp.where(
                control_eligible,
                control_response.velocity,
                jnp.where(
                    body_control_request,
                    control_response.velocity,
                    jnp.where(
                        challenge_redirect | body_redirect,
                        redirected_velocity,
                        released_velocity,
                    ),
                ),
            ),
        ),
    )

    # "The effect of surface geometry on soccer ball trajectories" anchors
    # the observed 146--147 rad/s range. The configured 150 rad/s maximum is a
    # rounded design envelope, not a direct fit to that paper.
    spin_generation_fraction = jnp.clip(decoded.power[winner], 0.0, 1.0)
    spin_limit = jnp.where(
        foot & deliberate_release,
        scale.spin_max_radps * spin_generation_fraction,
        0.0,
    )
    spin_control = decoded.spin[winner]
    lateral_axis = jnp.asarray(
        [-direction[1], direction[0], 0.0], dtype=state.ball.spin.dtype
    )
    delta_spin = spin_limit * (
        spin_control[0] * jnp.asarray([0.0, 0.0, 1.0], dtype=state.ball.spin.dtype)
        - spin_control[1] * lateral_axis
    )
    released_spin = _limit_generated_state(
        state.ball.spin,
        state.ball.spin + delta_spin,
        spin_limit,
    )
    new_spin = jnp.where(caught, jnp.zeros_like(state.ball.spin), released_spin)
    new_spin = jnp.where(
        parried | challenge_redirect | body_redirect | control_eligible,
        state.ball.spin,
        new_spin,
    )

    contact_velocity = jnp.where(raw_ball_contact, new_velocity, state.ball.velocity)
    contact_spin = jnp.where(raw_ball_contact, new_spin, state.ball.spin)
    throw_position = state.ball.position.at[2].set(
        body.torso_top_height(state.players.height[winner])
        + scale.throw_release_height_addition_m
    )
    foot_release_position = jnp.asarray(
        [
            state.players.position[winner, 0],
            state.players.position[winner, 1],
            jnp.maximum(
                ball_geometry.radius,
                state.players.height[winner] * scale.pelvis_height_factor,
            ),
        ],
        dtype=state.ball.position.dtype,
    )
    foot_punt = restart_contact & (state.restart.kind == RK_GK_HOLD) & foot
    release_position = jnp.where(
        raw_ball_contact & throw,
        throw_position,
        jnp.where(foot_punt, foot_release_position, state.ball.position),
    )
    gk_backpass_team = update_backpass_after_deliberate_attempt(
        state,
        actor=winner,
        actual_contact=raw_ball_contact,
        restricted_goalkeeper_clear_foot_attempt=(
            restricted_goalkeeper_clear_foot_attempt
        ),
        arm_targeted_foot_release=(
            raw_ball_contact & deliberate_release & foot & (intent == INTENT_PASS)
        ),
        release_position=release_position,
        outgoing_velocity=contact_velocity,
        outgoing_spin=contact_spin,
        ball=ball_geometry,
        reach=reach,
        physics=ball_physics,
    )
    ball = state.ball._replace(
        position=release_position,
        velocity=contact_velocity,
        spin=contact_spin,
        live=jnp.where(caught, False, state.ball.live | raw_ball_contact),
    )

    outcome = jnp.where(
        chest_control_success,
        OUTCOME_TRAP,
        jnp.where(
            chest_control_failure | head_control_failure,
            OUTCOME_MISCONTROL,
            jnp.where(
                control_success,
                jnp.where(
                    tackle_won,
                    OUTCOME_TACKLE_WON,
                    jnp.where(
                        raw_outcome == OUTCOME_INTERCEPTION,
                        OUTCOME_INTERCEPTION,
                        OUTCOME_TRAP,
                    ),
                ),
                jnp.where(control_failure, OUTCOME_MISCONTROL, raw_outcome),
            ),
        ),
    ).astype(jnp.int32)
    action_parameters_applied = raw_ball_contact & decoded.contact[winner] & (~caught)
    contest = contest._replace(parameters_applied=action_parameters_applied)

    goal_threat = is_goal_threat(
        state,
        winner,
        ball=ball_geometry,
        stadium=stadium,
        physics=ball_physics,
    )
    law11_effect = jnp.where(
        restart_contact,
        jnp.where(
            (state.restart.kind == RK_THROWIN)
            | (state.restart.kind == RK_GOALKICK)
            | (state.restart.kind == RK_CORNER),
            LAW11_DIRECT_RESTART_EXEMPTION,
            LAW11_DELIBERATE_PLAY_RESET,
        ),
        jnp.where(
            (outcome == OUTCOME_DEFLECTION)
            | chest_control_failure
            | ((outcome == OUTCOME_MISCONTROL) & challenge),
            LAW11_DEFLECTION_NO_RESET,
            jnp.where(
                goal_threat,
                LAW11_DELIBERATE_SAVE_NO_RESET,
                LAW11_DELIBERATE_PLAY_RESET,
            ),
        ),
    ).astype(jnp.int32)
    result = ContactResult(
        actor=jnp.where(raw_ball_contact, winner, NO_PLAYER).astype(jnp.int32),
        mechanism=jnp.where(raw_ball_contact, mechanism, MECHANISM_NONE).astype(
            jnp.int32
        ),
        intent=jnp.where(raw_ball_contact, intent, INTENT_MOVE).astype(jnp.int32),
        outcome=jnp.where(raw_ball_contact, outcome, OUTCOME_NONE).astype(jnp.int32),
        restart_kind=jnp.where(restart_contact, state.restart.kind, RK_NONE).astype(
            jnp.int32
        ),
        law11_effect=jnp.where(raw_ball_contact, law11_effect, LAW11_NONE).astype(
            jnp.int32
        ),
        kick_applied=deliberate_release & foot,
        intent_source=jnp.where(
            raw_ball_contact,
            decoded.intent_source[winner],
            INTENT_SOURCE_NONE,
        ).astype(jnp.int32),
    )
    last_contact = jax.tree_util.tree_map(
        lambda new, old: jnp.where(raw_ball_contact, new, old),
        result,
        state.possession.last_contact,
    )

    winner_team = state.players.team_id[winner].astype(jnp.int32)
    gains_control = control_success | chest_control_success | caught
    loses_control = raw_ball_contact & (~gains_control)
    gk_hold_control = (
        (state.restart.kind == RK_GK_HOLD)
        & (state.restart.taker == state.possession.player)
        & (state.restart.team == state.possession.team)
        & jnp.any(predicates.designated_restart)
    )
    keeps_control = (
        (predicates.verified_controlled_carrier != NO_PLAYER)
        | gk_hold_control
        | fresh_trap_control_grace(state)
    )
    possession = state.possession._replace(
        team=jnp.where(
            gains_control,
            winner_team,
            jnp.where(
                loses_control | (~keeps_control),
                NO_TEAM,
                state.possession.team,
            ),
        ).astype(jnp.int32),
        player=jnp.where(
            gains_control,
            winner,
            jnp.where(
                loses_control | (~keeps_control),
                NO_PLAYER,
                state.possession.player,
            ),
        ).astype(jnp.int32),
        previous_team=jnp.where(
            raw_ball_contact & (state.possession.team != NO_TEAM),
            state.possession.team,
            state.possession.previous_team,
        ).astype(jnp.int32),
        control_ticks=jnp.where(
            gains_control,
            1,
            jnp.where(
                loses_control | (~keeps_control),
                0,
                state.possession.control_ticks,
            ),
        ).astype(jnp.int32),
        last_contact=last_contact,
    )

    restart_cleared = caught | restart_contact
    restart = state.restart._replace(
        kind=jnp.where(
            caught,
            RK_GK_HOLD,
            jnp.where(restart_contact, RK_NONE, state.restart.kind),
        ).astype(jnp.int32),
        team=jnp.where(
            caught,
            winner_team,
            jnp.where(restart_contact, NO_TEAM, state.restart.team),
        ).astype(jnp.int32),
        substeps_remaining=jnp.where(
            restart_cleared, 0, state.restart.substeps_remaining
        ).astype(jnp.int32),
        taker=jnp.where(
            caught,
            winner,
            jnp.where(restart_contact, NO_PLAYER, state.restart.taker),
        ).astype(jnp.int32),
        indirect=jnp.where(restart_cleared, False, state.restart.indirect),
        opened_control_tick=jnp.where(
            caught,
            state.control_tick,
            jnp.where(
                restart_contact,
                jnp.int32(-1),
                state.restart.opened_control_tick,
            ),
        ).astype(jnp.int32),
    )

    restart_law11_direct_exempt = (
        (state.restart.kind == RK_THROWIN)
        | (state.restart.kind == RK_GOALKICK)
        | (state.restart.kind == RK_CORNER)
    )
    next_restart_release = RestartReleaseProvenance(
        active=jnp.bool_(True),
        untouched=jnp.bool_(True),
        kind=state.restart.kind,
        team=state.restart.team,
        taker=winner,
        indirect=state.restart.indirect,
        law11_direct_exempt=restart_law11_direct_exempt,
        release_mechanism=mechanism.astype(jnp.int32),
    )
    followup_contact = raw_ball_contact & (~restart_contact)
    other_player_contact = (
        followup_contact
        & state.restart_release.active
        & (winner != state.restart_release.taker)
        # Receiving a team-mate's throw-in with the foot, head, or chest does
        # not by itself remove the goalkeeper's handling restriction.  Only
        # the explicit clear-kick attempt below, or a different player's
        # subsequent contact, ends that cause.
        & (~((state.restart_release.kind == RK_THROWIN) & hand_restricted[winner]))
    )
    throw_clear_attempt = (
        restricted_goalkeeper_clear_foot_attempt
        & state.restart_release.active
        & (state.restart_release.kind == RK_THROWIN)
    )
    release_cleared = other_player_contact | throw_clear_attempt
    cleared_restart_release = RestartReleaseProvenance(
        active=jnp.bool_(False),
        untouched=jnp.bool_(False),
        kind=jnp.int32(RK_NONE),
        team=jnp.int32(NO_TEAM),
        taker=jnp.int32(NO_PLAYER),
        indirect=jnp.bool_(False),
        law11_direct_exempt=jnp.bool_(False),
        release_mechanism=jnp.int32(MECHANISM_NONE),
    )
    continued_restart_release = state.restart_release._replace(
        untouched=jnp.where(
            followup_contact,
            jnp.bool_(False),
            state.restart_release.untouched,
        )
    )
    restart_release = jax.tree_util.tree_map(
        lambda released, cleared, current: jnp.where(
            restart_contact,
            released,
            jnp.where(release_cleared, cleared, current),
        ),
        next_restart_release,
        cleared_restart_release,
        continued_restart_release,
    )

    contact_ticks = max(1, round(timing.active_contact_interval_s / dt))
    loss_ticks = max(1, round(timing.possession_loss_lock_s / dt))
    contact_lock = state.players.contact_lock_substeps.at[winner].set(
        jnp.where(
            raw_ball_contact,
            contact_ticks,
            state.players.contact_lock_substeps[winner],
        )
    )

    # Candidate is the complete reachable attempt set before the contest
    # chooses one actor.  Installing recovery from it therefore includes a
    # goalkeeper or jumper who reached the ball's swept contact envelope but
    # lost the contest or failed to realize contact.
    head_attempt = candidate & (predicates.mechanism == MECHANISM_HEAD)
    goalkeeper_attempt = candidate & (predicates.mechanism == MECHANISM_GOALKEEPER_HAND)
    athletic_attempt = (head_attempt | goalkeeper_attempt) & (
        predicates.athletic_reach_effort > 0.0
    )
    maximum_recovery_seconds = jnp.where(
        goalkeeper_attempt,
        timing.goalkeeper_dive_recovery_s,
        timing.aerial_attempt_recovery_s,
    )
    effort_recovery_ticks = jnp.where(
        athletic_attempt,
        jnp.maximum(
            1,
            jnp.rint(
                maximum_recovery_seconds * predicates.athletic_reach_effort / dt
            ).astype(jnp.int32),
        ),
        0,
    )
    aerial_recovery = jnp.where(
        athletic_attempt,
        jnp.maximum(
            state.players.aerial_recovery_substeps,
            effort_recovery_ticks,
        ),
        state.players.aerial_recovery_substeps,
    )
    challenge_attempt = candidate & predicates.challenge_context
    lunge_fraction = challenge_lunge_fraction
    challenge_seconds = (
        timing.challenge_recovery_s + lunge_fraction * timing.max_lunge_extra_recovery_s
    )
    challenge_ticks = jnp.maximum(
        1,
        jnp.rint(challenge_seconds / dt).astype(jnp.int32),
    )
    challenge_recovery = jnp.where(
        challenge_attempt,
        jnp.maximum(
            state.players.challenge_recovery_substeps,
            challenge_ticks,
        ),
        state.players.challenge_recovery_substeps,
    )

    old_player_valid = (state.possession.player >= 0) & (
        state.possession.player < state.players.position.shape[0]
    )
    old_player = jnp.clip(
        state.possession.player, 0, state.players.position.shape[0] - 1
    )
    changed_team = gains_control & (winner_team != state.possession.team)
    possession_loss_lock = state.players.possession_loss_lock_substeps.at[
        old_player
    ].set(
        jnp.where(
            old_player_valid & changed_team,
            loss_ticks,
            state.players.possession_loss_lock_substeps[old_player],
        )
    )
    players = state.players._replace(
        contact_lock_substeps=contact_lock,
        aerial_recovery_substeps=aerial_recovery,
        challenge_recovery_substeps=challenge_recovery,
        possession_loss_lock_substeps=possession_loss_lock,
    )
    next_state = state._replace(
        ball=ball,
        players=players,
        possession=possession,
        restart=restart,
        restart_release=restart_release,
        gk_backpass_team=gk_backpass_team,
    )
    return DeliberateContactStep(
        state=next_state,
        occurred=raw_ball_contact,
        actor=result.actor,
        contact=result,
        contest=contest,
        attempted=attempted,
        occurrence=ContactOccurrence(
            occurred=raw_ball_contact,
            actor=result.actor,
            mechanism=result.mechanism,
            law11_effect=result.law11_effect,
            position=jnp.where(
                raw_ball_contact,
                release_position,
                jnp.zeros_like(state.ball.position),
            ),
            time_fraction=jnp.asarray(0.0, dtype=state.ball.position.dtype),
        ),
    )


def resolve_contact(
    state: State,
    action: IntentAction,
    key: jax.Array,
    contest_override: ContestOverride,
    restart_release_allowed: jax.Array,
    *,
    dt: float,
    contact_attempted: jax.Array,
    ball_geometry: Ball = Ball(),
    stadium: Stadium = Stadium(),
    reach: Reach = Reach(),
    scale: ActionScale = ActionScale(),
    body: BodyContact = BodyContact(),
    timing: ContactTiming = ContactTiming(),
    ball_physics: BallPhysics = BallPhysics(),
    contest_config: Contest = Contest(),
    regulation_elapsed_fraction: jax.Array = 0.0,
) -> State:
    """Resolve at most one deliberate contact and return its next state."""

    return resolve_contact_step(
        state,
        action,
        key,
        contest_override,
        restart_release_allowed,
        dt=dt,
        ball_geometry=ball_geometry,
        stadium=stadium,
        reach=reach,
        scale=scale,
        body=body,
        timing=timing,
        ball_physics=ball_physics,
        contest_config=contest_config,
        regulation_elapsed_fraction=regulation_elapsed_fraction,
        contact_attempted=contact_attempted,
    ).state
