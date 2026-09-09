"""One ordered FootballWorld physics substep."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.action import ActionScale
from footballworld.config.ball_physics import BallPhysics
from footballworld.config.body_contact import BodyContact
from footballworld.config.contact_timing import ContactTiming
from footballworld.config.contest import Contest
from footballworld.config.geometry import Ball, Stadium
from footballworld.config.player_physics import PlayerPhysics
from footballworld.config.reach import Reach
from footballworld.config.stamina import LongStamina, ShortStamina
from footballworld.core.constants import (
    DISCIPLINE_NONE,
    DIV_EPS,
    GEOMETRY_EPS,
    NO_PLAYER,
    RK_GK_HOLD,
    RK_PENALTY,
    WOODWORK_NONE,
)
from footballworld.core.contact import (
    INTENT_MOVE,
    INTENT_SOURCE_NONE,
    LAW11_DEFLECTION_NO_RESET,
    LAW11_NONE,
    MECHANISM_NONE,
    MECHANISM_PASSIVE_BODY,
    OUTCOME_NONE,
    ContactOccurrence,
    ContactResult,
)
from footballworld.core.state import BallState, State, body_forward_from_angle
from footballworld.dynamics.action import PhysicsAction
from footballworld.dynamics.ball import advance_smooth, apply_ground_impact
from footballworld.dynamics.contact import (
    DeliberateContactStep,
    active_contact_possible,
    detect_active_contact,
    resolve_contact_step,
)
from footballworld.dynamics.contest import (
    SAMPLE_CONTEST,
    ContestOverride,
    ContestResult,
)
from footballworld.dynamics.goal_frame import (
    GoalFrameEvent,
    apply_goal_frame_contact,
    detect_goal_frame_contact,
)
from footballworld.dynamics.passive_contact import (
    PassiveContactEvent,
    apply_passive_contact,
    detect_passive_contact,
)
from footballworld.dynamics.player_step import step_players
from footballworld.dynamics.separation import PlayerImpactFacts
from footballworld.dynamics.stamina import effective_speed_limit
from footballworld.rules.ball_boundary import (
    CROSSING_NONE,
    BoundaryCrossing,
    detect_boundary_crossing,
)
from footballworld.rules.restart_legality import restart_actor_mask
from footballworld.rules.restart_positioning import (
    prepare_restart_positioning,
    restart_taker_release_pose,
)


class PhysicsSubstep(NamedTuple):
    """Next state and the control-frame contact-attempt mask."""

    state: State
    contact_attempted: jax.Array
    deliberate_occurred: jax.Array
    deliberate_actor: jax.Array
    deliberate_contact: ContactResult
    contest: ContestResult
    passive_occurred: jax.Array
    passive_actor: jax.Array
    boundary: BoundaryCrossing
    woodwork_occurred: jax.Array
    woodwork_kind: jax.Array
    woodwork_time_fraction: jax.Array
    event_budget_exhausted: jax.Array
    contact_occurrences: ContactOccurrence
    player_path_start: jax.Array
    player_path_delta: jax.Array
    deliberate_time_fraction: jax.Array
    penalty_settling_contact: jax.Array
    player_impact: PlayerImpactFacts
    restart_approach_moved: jax.Array
    restart_projection_moved: jax.Array


class _GroundEvent(NamedTuple):
    occurred: jax.Array
    time_fraction: jax.Array


class _BallEventStep(NamedTuple):
    state: State
    passive_occurred: jax.Array
    passive_actor: jax.Array
    boundary: BoundaryCrossing
    woodwork_occurred: jax.Array
    woodwork_kind: jax.Array
    woodwork_time_fraction: jax.Array
    event_budget_exhausted: jax.Array
    deliberate: DeliberateContactStep
    deliberate_time_fraction: jax.Array
    contact_occurrences: ContactOccurrence
    penalty_settling_contact: jax.Array


class _BallEventCarry(NamedTuple):
    event_index: jax.Array
    nonactive_index: jax.Array
    state: State
    remaining: jax.Array
    terminal: jax.Array
    boundary: BoundaryCrossing
    passive_occurred: jax.Array
    passive_actor: jax.Array
    woodwork_occurred: jax.Array
    woodwork_kind: jax.Array
    woodwork_time_fraction: jax.Array
    event_budget_exhausted: jax.Array
    previous_passive_actor: jax.Array
    deliberate_evaluated: jax.Array
    deliberate_occurred: jax.Array
    deliberate_actor: jax.Array
    deliberate_contact: ContactResult
    contest: ContestResult
    contact_attempted: jax.Array
    deliberate_occurrence: ContactOccurrence
    occurrence_occurred: jax.Array
    occurrence_actor: jax.Array
    occurrence_mechanism: jax.Array
    occurrence_law11_effect: jax.Array
    occurrence_position: jax.Array
    occurrence_time_fraction: jax.Array
    penalty_tracking: jax.Array
    penalty_team: jax.Array
    penalty_settling_contact: jax.Array


def _empty_deliberate_step(
    state: State,
    contact_attempted: jax.Array,
    contest_override: ContestOverride,
) -> DeliberateContactStep:
    dtype = state.ball.position.dtype
    no_candidate_override_valid = (
        (contest_override.winner == SAMPLE_CONTEST)
        | (contest_override.winner == NO_PLAYER)
    ) & (contest_override.outcome == SAMPLE_CONTEST)
    contact = ContactResult(
        actor=jnp.int32(NO_PLAYER),
        mechanism=jnp.int32(MECHANISM_NONE),
        intent=jnp.int32(INTENT_MOVE),
        outcome=jnp.int32(OUTCOME_NONE),
        restart_kind=jnp.int32(0),
        law11_effect=jnp.int32(LAW11_NONE),
        kick_applied=jnp.bool_(False),
        intent_source=jnp.int32(INTENT_SOURCE_NONE),
    )
    contest = ContestResult(
        selected=jnp.bool_(False),
        occurred=jnp.bool_(False),
        actor=jnp.int32(NO_PLAYER),
        challenger=jnp.int32(NO_PLAYER),
        outcome=jnp.int32(OUTCOME_NONE),
        parameters_applied=jnp.bool_(False),
        foul_actor=jnp.int32(NO_PLAYER),
        foul_victim=jnp.int32(NO_PLAYER),
        offence_position=jnp.zeros(2, dtype=dtype),
        discipline=jnp.int32(DISCIPLINE_NONE),
        override_valid=no_candidate_override_valid,
    )
    occurrence = ContactOccurrence(
        occurred=jnp.bool_(False),
        actor=jnp.int32(NO_PLAYER),
        mechanism=jnp.int32(MECHANISM_NONE),
        law11_effect=jnp.int32(LAW11_NONE),
        position=jnp.zeros(3, dtype=dtype),
        time_fraction=jnp.asarray(0.0, dtype=dtype),
    )
    return DeliberateContactStep(
        state=state,
        occurred=jnp.bool_(False),
        actor=jnp.int32(NO_PLAYER),
        contact=contact,
        contest=contest,
        attempted=jnp.asarray(contact_attempted, dtype=jnp.bool_),
        occurrence=occurrence,
    )


def _attach_held_ball(state: State, *, body: BodyContact) -> State:
    player_count = state.players.position.shape[0]
    taker = jnp.clip(state.restart.taker, 0, player_count - 1)
    valid_taker = restart_actor_mask(state)[taker]
    held = (state.restart.kind == RK_GK_HOLD) & valid_taker
    held_position = jnp.array(
        [
            state.players.position[taker, 0],
            state.players.position[taker, 1],
            body.torso_top_height(state.players.height[taker]),
        ],
        dtype=state.ball.position.dtype,
    )
    held_ball = BallState(
        position=jnp.where(held, held_position, state.ball.position),
        velocity=jnp.where(
            held, jnp.zeros_like(state.ball.velocity), state.ball.velocity
        ),
        spin=jnp.where(held, jnp.zeros_like(state.ball.spin), state.ball.spin),
        live=jnp.where(held, False, state.ball.live),
    )
    return state._replace(ball=held_ball)


def _passive_contact_broadphase(
    state: State,
    ball_path_delta: jax.Array,
    player_start_position: jax.Array,
    player_path_delta: jax.Array,
    *,
    ball_geometry: Ball,
    body: BodyContact,
) -> jax.Array:
    """Conservatively reject swept paths outside every player body AABB."""

    dtype = state.ball.position.dtype
    horizontal_margin = jnp.asarray(
        ball_geometry.radius
        + body.shoulder_width_m
        + max(body.head_radius_m, body.leg_radius_m),
        dtype=dtype,
    )
    ball_start = state.ball.position
    ball_end = ball_start + ball_path_delta
    ball_min_xy = jnp.minimum(ball_start[:2], ball_end[:2])
    ball_max_xy = jnp.maximum(ball_start[:2], ball_end[:2])
    player_end = player_start_position + player_path_delta
    player_min_xy = jnp.minimum(player_start_position, player_end) - horizontal_margin
    player_max_xy = jnp.maximum(player_start_position, player_end) + horizontal_margin
    horizontal_overlap = jnp.all(
        (ball_max_xy[None, :] >= player_min_xy)
        & (ball_min_xy[None, :] <= player_max_xy),
        axis=-1,
    )
    active_height = jnp.where(state.players.active, state.players.height, 0.0)
    vertical_margin = jnp.asarray(
        ball_geometry.radius + max(body.head_radius_m, body.leg_radius_m),
        dtype=dtype,
    )
    ball_min_z = jnp.minimum(ball_start[2], ball_end[2])
    ball_max_z = jnp.maximum(ball_start[2], ball_end[2])
    vertical_overlap = (ball_max_z >= -vertical_margin) & (
        ball_min_z <= jnp.max(active_height) + vertical_margin
    )
    finite = (
        jnp.all(jnp.isfinite(ball_start))
        & jnp.all(jnp.isfinite(ball_path_delta))
        & jnp.all(jnp.isfinite(player_start_position))
        & jnp.all(jnp.isfinite(player_path_delta))
        & jnp.all(jnp.isfinite(state.players.height))
    )
    candidate = vertical_overlap & jnp.any(state.players.active & horizontal_overlap)
    return state.ball.live & ((~finite) | candidate)


def _goal_frame_broadphase(
    position: jax.Array,
    path_delta: jax.Array,
    ball_live: jax.Array,
    *,
    ball: Ball,
    stadium: Stadium,
    physics: BallPhysics,
) -> jax.Array:
    """Conservatively reject paths outside the union AABB of both frames."""

    dtype = position.dtype
    end = position + path_delta
    lower = jnp.minimum(position, end)
    upper = jnp.maximum(position, end)
    frame_radius = jnp.asarray(physics.goal_frame_radius, dtype=dtype)
    collision_radius = jnp.asarray(ball.radius, dtype=dtype) + frame_radius
    half_length = jnp.asarray(stadium.half_length, dtype=dtype)
    post_y = jnp.asarray(0.5 * stadium.goal_width, dtype=dtype) + frame_radius
    bar_z = jnp.asarray(stadium.goal_height, dtype=dtype) + frame_radius
    negative_goal = (lower[0] <= -half_length + collision_radius) & (
        upper[0] >= -half_length - collision_radius
    )
    positive_goal = (lower[0] <= half_length + collision_radius) & (
        upper[0] >= half_length - collision_radius
    )
    y_overlap = (lower[1] <= post_y + collision_radius) & (
        upper[1] >= -post_y - collision_radius
    )
    z_overlap = (lower[2] <= bar_z + collision_radius) & (upper[2] >= -collision_radius)
    finite = jnp.all(jnp.isfinite(position)) & jnp.all(jnp.isfinite(path_delta))
    candidate = (negative_goal | positive_goal) & y_overlap & z_overlap
    return (
        jnp.asarray(ball_live, dtype=jnp.bool_)
        & (frame_radius > 0.0)
        & ((~finite) | candidate)
    )


def _detect_ground_event(
    ball: BallState,
    path_delta: jax.Array,
    *,
    geometry: Ball,
) -> _GroundEvent:
    ground_height = jnp.asarray(geometry.radius, dtype=ball.position.dtype)
    immediate = (
        ball.live
        & (ball.position[2] <= ground_height + GEOMETRY_EPS)
        & (ball.velocity[2] < 0.0)
    )
    crossed = (
        ball.live
        & (ball.position[2] > ground_height + GEOMETRY_EPS)
        & (ball.position[2] + path_delta[2] <= ground_height)
        & (path_delta[2] < 0.0)
    )
    # A semi-implicit step can leave the ground band upward and return below it
    # within one physics tick.  Its endpoint chord has a spurious root at zero;
    # treat the represented re-impact as an endpoint event so the chronological
    # loop makes progress and preserves the no-penetration invariant without a
    # broad post-step clamp.
    depart_reimpact = (
        ball.live
        & (ball.position[2] <= ground_height + GEOMETRY_EPS)
        & (ball.velocity[2] > 0.0)
        & (ball.position[2] + path_delta[2] <= ground_height)
        & (path_delta[2] < 0.0)
    )
    fraction = (ground_height - ball.position[2]) / jnp.where(
        jnp.abs(path_delta[2]) > DIV_EPS,
        path_delta[2],
        -1.0,
    )
    return _GroundEvent(
        occurred=immediate | crossed | depart_reimpact,
        time_fraction=jnp.where(
            immediate,
            0.0,
            jnp.where(depart_reimpact, 1.0, jnp.clip(fraction, 0.0, 1.0)),
        ),
    )


def _step_ball_events_chronological(
    state: State,
    action: PhysicsAction,
    key: jax.Array,
    contest_override: ContestOverride,
    restart_release_allowed: jax.Array,
    *,
    dt: float | jax.Array,
    contact_attempted: jax.Array,
    player_start_position: jax.Array,
    player_path_delta: jax.Array,
    player_start_velocity: jax.Array,
    player_velocity_delta: jax.Array,
    ball_geometry: Ball,
    stadium: Stadium,
    reach: Reach,
    action_scale: ActionScale,
    body: BodyContact,
    contact_timing: ContactTiming,
    ball_physics: BallPhysics,
    contest_config: Contest,
    regulation_elapsed_fraction: jax.Array,
    period_boundary_penalty_enforced: jax.Array,
) -> _BallEventStep:
    """Schedule deliberate and passive ball events on one common timeline.

    This uses bounded piecewise-remainder ordering on one shared swept
    player/ball timeline. Whole contact-lock exclusion would make a moving body
    temporarily transparent;
    the three fixed passive slots instead exclude only actors already hit in this
    physics substep. At most one deliberate resolution and three nonterminal
    physical events are applied. A fifth fixed iteration is detection-only for a
    genuine fourth physical event, while a terminal boundary still completes.
    """

    dtype = state.ball.position.dtype
    total_dt = jnp.asarray(dt, dtype=dtype)
    player_final_position = state.players.position
    player_final_velocity = state.players.velocity
    empty_boundary = BoundaryCrossing(
        occurred=jnp.bool_(False),
        axis=jnp.int32(CROSSING_NONE),
        time_fraction=jnp.asarray(0.0, dtype=dtype),
        position=jnp.zeros(3, dtype=dtype),
        through_goal=jnp.bool_(False),
    )
    empty_deliberate = _empty_deliberate_step(
        state, contact_attempted, contest_override
    )
    completion_team = state.penalty_completion_team.astype(jnp.int32)
    completion_team_valid = (completion_team == 0) | (completion_team == 1)
    carry = _BallEventCarry(
        event_index=jnp.int32(0),
        nonactive_index=jnp.int32(0),
        state=state,
        remaining=total_dt,
        terminal=jnp.bool_(False),
        boundary=empty_boundary,
        passive_occurred=jnp.zeros(3, dtype=jnp.bool_),
        passive_actor=jnp.full(3, NO_PLAYER, dtype=jnp.int32),
        woodwork_occurred=jnp.zeros(3, dtype=jnp.bool_),
        woodwork_kind=jnp.full(3, WOODWORK_NONE, dtype=jnp.int32),
        woodwork_time_fraction=jnp.zeros(3, dtype=dtype),
        event_budget_exhausted=jnp.bool_(False),
        previous_passive_actor=jnp.int32(NO_PLAYER),
        deliberate_evaluated=jnp.bool_(False),
        deliberate_occurred=empty_deliberate.occurred,
        deliberate_actor=empty_deliberate.actor,
        deliberate_contact=empty_deliberate.contact,
        contest=empty_deliberate.contest,
        contact_attempted=empty_deliberate.attempted,
        deliberate_occurrence=empty_deliberate.occurrence,
        occurrence_occurred=jnp.zeros(4, dtype=jnp.bool_),
        occurrence_actor=jnp.full(4, NO_PLAYER, dtype=jnp.int32),
        occurrence_mechanism=jnp.full(4, MECHANISM_NONE, dtype=jnp.int32),
        occurrence_law11_effect=jnp.full(4, LAW11_NONE, dtype=jnp.int32),
        occurrence_position=jnp.zeros((4, 3), dtype=dtype),
        occurrence_time_fraction=jnp.zeros(4, dtype=dtype),
        penalty_tracking=completion_team_valid,
        penalty_team=completion_team,
        penalty_settling_contact=jnp.bool_(False),
    )

    # The first chronological iteration normally proves that no ball event
    # exists and consumes the whole remainder. Reuse the exact smooth endpoint,
    # the existing conservative contact/frame gates, and the authoritative
    # ground/boundary predicates to prove that outcome before entering the
    # large bounded event loop. Non-finite geometry remains on the full path
    # because every broad-phase gate fails open.
    def advance_initial_carry(
        initial_carry: _BallEventCarry,
    ) -> _BallEventCarry:
        # Preserve the baseline while-loop arithmetic boundary. Moving this
        # smooth advance into a plain outer expression changes last-bit results
        # after repeated scans on XLA CPU.
        def light_pending(light_carry: _BallEventCarry) -> jax.Array:
            return light_carry.remaining > 0.0

        def light_iteration(light_carry: _BallEventCarry) -> _BallEventCarry:
            predicted_ball = advance_smooth(
                light_carry.state.ball,
                dt=light_carry.remaining,
                geometry=ball_geometry,
                physics=ball_physics,
            )
            return light_carry._replace(
                event_index=light_carry.event_index + jnp.int32(1),
                state=light_carry.state._replace(ball=predicted_ball),
                remaining=jnp.asarray(0.0, dtype=dtype),
            )

        return jax.lax.while_loop(
            light_pending,
            light_iteration,
            initial_carry,
        )

    initial_advanced_carry = advance_initial_carry(carry)
    initial_predicted_ball = initial_advanced_carry.state.ball
    initial_ball_path_delta = initial_predicted_ball.position - state.ball.position
    initial_active_possible = active_contact_possible(
        state,
        action,
        restart_release_allowed,
        contact_attempted,
        initial_ball_path_delta,
        player_start_position,
        player_path_delta,
        excluded_actor=jnp.int32(NO_PLAYER),
        search_enabled=jnp.bool_(True),
        ball_geometry=ball_geometry,
        reach=reach,
    )
    initial_passive_possible = _passive_contact_broadphase(
        state,
        initial_ball_path_delta,
        player_start_position,
        player_path_delta,
        ball_geometry=ball_geometry,
        body=body,
    )
    initial_frame_possible = _goal_frame_broadphase(
        state.ball.position,
        initial_ball_path_delta,
        state.ball.live,
        ball=ball_geometry,
        stadium=stadium,
        physics=ball_physics,
    )
    initial_ground = _detect_ground_event(
        state.ball,
        initial_ball_path_delta,
        geometry=ball_geometry,
    )
    initial_boundary = detect_boundary_crossing(
        state.ball.position,
        initial_predicted_ball.position,
        state.ball.live,
        stadium=stadium,
        ball=ball_geometry,
    )
    initial_no_event = (
        (total_dt > 0.0)
        & state.ball.live
        & (~initial_active_possible)
        & (~initial_passive_possible)
        & (~initial_frame_possible)
        & (~initial_ground.occurred)
        & (~initial_boundary.occurred)
    )

    def event_iteration(current_carry: _BallEventCarry) -> _BallEventCarry:
        current = current_carry.state
        predicted_ball = advance_smooth(
            current.ball,
            dt=current_carry.remaining,
            geometry=ball_geometry,
            physics=ball_physics,
        )
        ball_path_delta = predicted_ball.position - current.ball.position
        elapsed = total_dt - current_carry.remaining
        elapsed_fraction = jnp.where(total_dt > 0.0, elapsed / total_dt, 0.0)
        remaining_fraction = jnp.where(
            total_dt > 0.0, current_carry.remaining / total_dt, 0.0
        )
        current_player_start = (
            player_start_position + elapsed_fraction * player_path_delta
        )
        current_player_delta = remaining_fraction * player_path_delta
        event_live = current.ball.live & (~current_carry.terminal)
        query_state = current._replace(ball=current.ball._replace(live=event_live))

        active = detect_active_contact(
            current,
            action,
            restart_release_allowed,
            current_carry.contact_attempted,
            ball_path_delta,
            remaining_dt=current_carry.remaining,
            player_start_position=current_player_start,
            player_path_delta=current_player_delta,
            excluded_actor=current_carry.previous_passive_actor,
            search_enabled=~current_carry.deliberate_evaluated,
            ball_geometry=ball_geometry,
            stadium=stadium,
            reach=reach,
            scale=action_scale,
            body=body,
            ball_physics=ball_physics,
        )
        active_occurred = active.occurred & (~current_carry.deliberate_evaluated)
        budget_full = current_carry.nonactive_index >= 3
        nonactive_allowed = (~current_carry.event_budget_exhausted) & event_live
        player_indices = jnp.arange(state.players.position.shape[0])
        passive_seen = jnp.any(
            player_indices[:, None] == current_carry.passive_actor[None, :],
            axis=1,
        )
        passive_query_state = query_state._replace(
            players=query_state.players._replace(
                on_pitch=query_state.players.on_pitch & (~passive_seen)
            )
        )
        passive_possible = _passive_contact_broadphase(
            passive_query_state,
            ball_path_delta,
            current_player_start,
            current_player_delta,
            ball_geometry=ball_geometry,
            body=body,
        )

        def detect_passive(_):
            return detect_passive_contact(
                passive_query_state,
                ball_path_delta,
                # Keep one 1/90 s event integration internally coherent. A causal
                # reach/separation gate handles the following physics substeps.
                excluded_actor=current_carry.deliberate_actor,
                secondary_excluded_actor=current_carry.previous_passive_actor,
                player_start_position=current_player_start,
                player_path_delta=current_player_delta,
                ball_geometry=ball_geometry,
                body=body,
                reach=reach,
            )

        passive = jax.lax.cond(
            passive_possible,
            detect_passive,
            lambda _: PassiveContactEvent(
                occurred=jnp.bool_(False),
                actor=jnp.int32(NO_PLAYER),
                time_fraction=jnp.asarray(0.0, dtype=dtype),
                normal=jnp.zeros(3, dtype=dtype),
            ),
            operand=None,
        )
        frame_possible = _goal_frame_broadphase(
            current.ball.position,
            ball_path_delta,
            event_live,
            ball=ball_geometry,
            stadium=stadium,
            physics=ball_physics,
        )
        frame = jax.lax.cond(
            frame_possible,
            lambda _: detect_goal_frame_contact(
                current.ball.position,
                ball_path_delta,
                event_live,
                ball=ball_geometry,
                stadium=stadium,
                physics=ball_physics,
            ),
            lambda _: GoalFrameEvent(
                occurred=jnp.bool_(False),
                kind=jnp.int32(WOODWORK_NONE),
                time_fraction=jnp.asarray(0.0, dtype=dtype),
                position=jnp.zeros(3, dtype=dtype),
                normal=jnp.zeros(3, dtype=dtype),
            ),
            operand=None,
        )
        ground = _detect_ground_event(
            query_state.ball, ball_path_delta, geometry=ball_geometry
        )
        boundary = detect_boundary_crossing(
            current.ball.position,
            predicted_ball.position,
            event_live,
            stadium=stadium,
            ball=ball_geometry,
        )

        active_time = jnp.where(active_occurred, active.time_fraction, jnp.inf)
        passive_time = jnp.where(
            nonactive_allowed & passive.occurred, passive.time_fraction, jnp.inf
        )
        frame_time = jnp.where(
            nonactive_allowed & frame.occurred, frame.time_fraction, jnp.inf
        )
        ground_time = jnp.where(
            nonactive_allowed & ground.occurred, ground.time_fraction, jnp.inf
        )
        boundary_time = jnp.where(
            nonactive_allowed & boundary.occurred, boundary.time_fraction, jnp.inf
        )
        use_active = active_occurred & (
            (active_time <= passive_time)
            & (active_time <= frame_time)
            & (active_time <= ground_time)
            & (active_time <= boundary_time)
        )
        use_passive = (passive_time < active_time) & (
            (passive_time <= frame_time)
            & (passive_time <= ground_time)
            & (passive_time <= boundary_time)
        )
        use_frame = (
            (frame_time < active_time)
            & (frame_time < passive_time)
            & ((frame_time <= ground_time) & (frame_time <= boundary_time))
        )
        use_ground = (
            (ground_time < active_time)
            & (ground_time < passive_time)
            & (ground_time < frame_time)
            & (ground_time <= boundary_time)
        )
        use_boundary = (
            (boundary_time < active_time)
            & (boundary_time < passive_time)
            & (boundary_time < frame_time)
            & (boundary_time < ground_time)
        )
        nonactive_event = use_passive | use_frame | use_ground | use_boundary
        event_detected = use_active | nonactive_event
        # The fifth fixed iteration is detection-only when three nonterminal
        # physical events have already been applied. An exact endpoint turf
        # impact is also safe to close: its represented remainder is exactly
        # zero, it needs no telemetry slot, and it cannot hide unchecked time.
        # Earlier turf candidates and all further player/frame contacts remain
        # fail-closed. A terminal boundary still completes normally because it
        # needs no passive/woodwork output slot.
        endpoint_ground_closure = (
            budget_full
            & use_ground
            & (ground.time_fraction == jnp.asarray(1.0, dtype=dtype))
        )
        budget_hit = (
            budget_full
            & (use_passive | use_frame | use_ground)
            & (~endpoint_ground_closure)
        )
        apply_passive = use_passive & (~budget_hit)
        apply_frame = use_frame & (~budget_hit)
        apply_ground = use_ground & (~budget_hit)
        applied_nonactive_event = (
            apply_passive | apply_frame | apply_ground | use_boundary
        )
        occurred = use_active | applied_nonactive_event
        event_fraction = jnp.where(
            use_active,
            active.time_fraction,
            jnp.where(
                use_passive,
                passive.time_fraction,
                jnp.where(
                    use_frame,
                    frame.time_fraction,
                    jnp.where(use_ground, ground.time_fraction, boundary.time_fraction),
                ),
            ),
        )
        event_fraction = jnp.where(event_detected, event_fraction, 0.0)
        global_event_fraction = jnp.where(
            total_dt > 0.0,
            (elapsed + current_carry.remaining * event_fraction) / total_dt,
            0.0,
        )

        advanced_ball = jax.lax.cond(
            event_detected,
            lambda ball: advance_smooth(
                ball,
                dt=current_carry.remaining * event_fraction,
                geometry=ball_geometry,
                physics=ball_physics,
            ),
            # event_fraction is exactly zero on this path, so the former
            # smooth advance was an expensive identity whose result was later
            # discarded in favour of ``predicted_ball``.
            lambda ball: ball,
            current.ball,
        )
        impact_position = current.ball.position + event_fraction * ball_path_delta
        impact_position = impact_position.at[2].set(
            jnp.where(use_ground, ball_geometry.radius, impact_position[2])
        )
        impact_position = jnp.where(use_frame, frame.position, impact_position)
        impact_position = jnp.where(use_boundary, boundary.position, impact_position)
        at_impact = current._replace(
            ball=advanced_ball._replace(position=impact_position)
        )

        def resolve_active(active_state: State) -> DeliberateContactStep:
            contact_positions = (
                player_start_position + global_event_fraction * player_path_delta
            )
            contact_velocities = (
                player_start_velocity + global_event_fraction * player_velocity_delta
            )
            contact_state = active_state._replace(
                players=active_state.players._replace(
                    position=contact_positions,
                    velocity=contact_velocities,
                )
            )
            resolution = resolve_contact_step(
                contact_state,
                action,
                key,
                contest_override,
                restart_release_allowed,
                dt=dt,
                ball_geometry=ball_geometry,
                stadium=stadium,
                reach=reach,
                scale=action_scale,
                body=body,
                timing=contact_timing,
                ball_physics=ball_physics,
                contest_config=contest_config,
                regulation_elapsed_fraction=regulation_elapsed_fraction,
                contact_attempted=current_carry.contact_attempted,
            )
            restored_players = resolution.state.players._replace(
                position=player_final_position,
                velocity=player_final_velocity,
            )
            occurrence = resolution.occurrence._replace(
                time_fraction=global_event_fraction
            )
            return resolution._replace(
                state=resolution.state._replace(players=restored_players),
                occurrence=occurrence,
            )

        deliberate = jax.lax.cond(
            use_active,
            resolve_active,
            lambda inactive_state: _empty_deliberate_step(
                inactive_state, current_carry.contact_attempted, contest_override
            ),
            at_impact,
        )
        physical_response = jnp.where(
            apply_passive,
            jnp.int32(0),
            jnp.where(
                apply_frame,
                jnp.int32(1),
                jnp.where(apply_ground, jnp.int32(2), jnp.int32(3)),
            ),
        )

        def resolve_passive(impact_state: State) -> State:
            return apply_passive_contact(
                impact_state,
                passive,
                ball_geometry=ball_geometry,
                body=body,
            )

        def resolve_frame(impact_state: State) -> State:
            return impact_state._replace(
                ball=apply_goal_frame_contact(
                    impact_state.ball,
                    frame,
                    ball=ball_geometry,
                    physics=ball_physics,
                )
            )

        def resolve_ground(impact_state: State) -> State:
            return impact_state._replace(
                ball=apply_ground_impact(
                    impact_state.ball,
                    geometry=ball_geometry,
                    physics=ball_physics,
                )
            )

        physical_state = jax.lax.switch(
            physical_response,
            (
                resolve_passive,
                resolve_frame,
                resolve_ground,
                lambda impact_state: impact_state,
            ),
            at_impact,
        )
        nonterminal_state = jax.tree_util.tree_map(
            lambda active_value, physical_value: jnp.where(
                use_active, active_value, physical_value
            ),
            deliberate.state,
            physical_state,
        )
        event_state = jax.tree_util.tree_map(
            lambda physical, terminal_state: jnp.where(
                use_boundary, terminal_state, physical
            ),
            nonterminal_state,
            at_impact,
        )
        no_event_state = jax.tree_util.tree_map(
            lambda frozen, advanced: jnp.where(
                current_carry.event_budget_exhausted | budget_hit,
                frozen,
                advanced,
            ),
            current,
            current._replace(ball=predicted_ball),
        )
        next_state = jax.tree_util.tree_map(
            lambda event_value, no_event_value: jnp.where(
                occurred, event_value, no_event_value
            ),
            event_state,
            no_event_state,
        )

        nonactive_slot = jnp.minimum(current_carry.nonactive_index, 2)
        passive_occurred = current_carry.passive_occurred.at[nonactive_slot].set(
            current_carry.passive_occurred[nonactive_slot] | apply_passive
        )
        passive_actor = current_carry.passive_actor.at[nonactive_slot].set(
            jnp.where(
                apply_passive,
                passive.actor,
                current_carry.passive_actor[nonactive_slot],
            )
        )
        woodwork_occurred = current_carry.woodwork_occurred.at[nonactive_slot].set(
            current_carry.woodwork_occurred[nonactive_slot] | apply_frame
        )
        woodwork_kind = current_carry.woodwork_kind.at[nonactive_slot].set(
            jnp.where(
                apply_frame, frame.kind, current_carry.woodwork_kind[nonactive_slot]
            )
        )
        woodwork_time_fraction = current_carry.woodwork_time_fraction.at[
            nonactive_slot
        ].set(
            jnp.where(
                apply_frame,
                global_event_fraction,
                current_carry.woodwork_time_fraction[nonactive_slot],
            )
        )

        contact_occurred = apply_passive | (use_active & deliberate.occurrence.occurred)
        contact_actor = jnp.where(
            use_active, deliberate.occurrence.actor, passive.actor
        ).astype(jnp.int32)
        contact_mechanism = jnp.where(
            use_active, deliberate.occurrence.mechanism, MECHANISM_PASSIVE_BODY
        ).astype(jnp.int32)
        contact_law11 = jnp.where(
            use_active,
            deliberate.occurrence.law11_effect,
            LAW11_DEFLECTION_NO_RESET,
        ).astype(jnp.int32)
        contact_position = jnp.where(
            use_active, deliberate.occurrence.position, impact_position
        )
        occurrence_slot = jnp.minimum(current_carry.event_index, 3)
        recorded_contact = (current_carry.event_index < 4) & contact_occurred
        occurrence_occurred = current_carry.occurrence_occurred.at[occurrence_slot].set(
            current_carry.occurrence_occurred[occurrence_slot] | recorded_contact
        )
        occurrence_actor = current_carry.occurrence_actor.at[occurrence_slot].set(
            jnp.where(
                recorded_contact,
                contact_actor,
                current_carry.occurrence_actor[occurrence_slot],
            )
        )
        occurrence_mechanism = current_carry.occurrence_mechanism.at[
            occurrence_slot
        ].set(
            jnp.where(
                recorded_contact,
                contact_mechanism,
                current_carry.occurrence_mechanism[occurrence_slot],
            )
        )
        occurrence_law11_effect = current_carry.occurrence_law11_effect.at[
            occurrence_slot
        ].set(
            jnp.where(
                recorded_contact,
                contact_law11,
                current_carry.occurrence_law11_effect[occurrence_slot],
            )
        )
        occurrence_position = current_carry.occurrence_position.at[occurrence_slot].set(
            jnp.where(
                recorded_contact,
                contact_position,
                current_carry.occurrence_position[occurrence_slot],
            )
        )
        occurrence_time_fraction = current_carry.occurrence_time_fraction.at[
            occurrence_slot
        ].set(
            jnp.where(
                recorded_contact,
                global_event_fraction,
                current_carry.occurrence_time_fraction[occurrence_slot],
            )
        )

        penalty_release = (
            use_active
            & deliberate.occurrence.occurred
            & deliberate.contact.kick_applied
            & (deliberate.contact.restart_kind == jnp.int32(RK_PENALTY))
        )
        penalty_team = jnp.where(
            penalty_release,
            current.restart.team,
            current_carry.penalty_team,
        ).astype(jnp.int32)
        contact_actor_valid = (contact_actor >= 0) & (
            contact_actor < current.players.position.shape[0]
        )
        safe_contact_actor = jnp.clip(
            contact_actor, 0, current.players.position.shape[0] - 1
        )
        defending_goalkeeper_contact = (
            contact_actor_valid
            & current.players.is_goalkeeper[safe_contact_actor]
            & (
                current.players.team_id[safe_contact_actor]
                == (jnp.int32(1) - current_carry.penalty_team)
            )
        )
        settling_contact = (
            contact_occurred
            & current_carry.penalty_tracking
            & (~penalty_release)
            & (~defending_goalkeeper_contact)
        )
        penalty_settling_contact = (
            current_carry.penalty_settling_contact | settling_contact
        )
        stop_at_contact = period_boundary_penalty_enforced & settling_contact

        global_boundary_fraction = jnp.where(
            total_dt > 0.0,
            (elapsed + current_carry.remaining * boundary.time_fraction) / total_dt,
            0.0,
        )
        boundary_global = boundary._replace(time_fraction=global_boundary_fraction)
        boundary_latch = jax.tree_util.tree_map(
            lambda new, old: jnp.where(use_boundary, new, old),
            boundary_global,
            current_carry.boundary,
        )
        remaining_after_event = current_carry.remaining * (1.0 - event_fraction)
        next_nonactive_index = jnp.minimum(
            jnp.int32(3),
            current_carry.nonactive_index + applied_nonactive_event.astype(jnp.int32),
        )
        next_deliberate_evaluated = current_carry.deliberate_evaluated | use_active
        next_contact_attempted = jnp.where(
            use_active, deliberate.attempted, current_carry.contact_attempted
        )
        next_previous_passive_actor = jnp.where(
            apply_passive, passive.actor, current_carry.previous_passive_actor
        ).astype(jnp.int32)
        next_deliberate_actor = jnp.where(
            use_active, deliberate.actor, current_carry.deliberate_actor
        ).astype(jnp.int32)
        exhausted = current_carry.event_budget_exhausted | budget_hit
        can_continue = occurred & (~use_boundary) & (~stop_at_contact) & (~exhausted)
        next_remaining = jnp.where(
            can_continue, remaining_after_event, jnp.asarray(0.0, dtype=dtype)
        )
        deliberate_occurrence = jax.tree_util.tree_map(
            lambda new, old: jnp.where(use_active, new, old),
            deliberate.occurrence,
            current_carry.deliberate_occurrence,
        )
        return _BallEventCarry(
            event_index=current_carry.event_index + jnp.int32(1),
            nonactive_index=next_nonactive_index,
            state=next_state,
            remaining=next_remaining,
            terminal=current_carry.terminal | use_boundary | stop_at_contact,
            boundary=boundary_latch,
            passive_occurred=passive_occurred,
            passive_actor=passive_actor,
            woodwork_occurred=woodwork_occurred,
            woodwork_kind=woodwork_kind,
            woodwork_time_fraction=woodwork_time_fraction,
            event_budget_exhausted=exhausted,
            previous_passive_actor=next_previous_passive_actor,
            deliberate_evaluated=next_deliberate_evaluated,
            deliberate_occurred=jnp.where(
                use_active, deliberate.occurred, current_carry.deliberate_occurred
            ),
            deliberate_actor=next_deliberate_actor,
            deliberate_contact=jax.tree_util.tree_map(
                lambda new, old: jnp.where(use_active, new, old),
                deliberate.contact,
                current_carry.deliberate_contact,
            ),
            contest=jax.tree_util.tree_map(
                lambda new, old: jnp.where(use_active, new, old),
                deliberate.contest,
                current_carry.contest,
            ),
            contact_attempted=next_contact_attempted,
            deliberate_occurrence=deliberate_occurrence,
            occurrence_occurred=occurrence_occurred,
            occurrence_actor=occurrence_actor,
            occurrence_mechanism=occurrence_mechanism,
            occurrence_law11_effect=occurrence_law11_effect,
            occurrence_position=occurrence_position,
            occurrence_time_fraction=occurrence_time_fraction,
            penalty_tracking=current_carry.penalty_tracking | penalty_release,
            penalty_team=penalty_team,
            penalty_settling_contact=penalty_settling_contact,
        )

    def pending(current_carry: _BallEventCarry) -> jax.Array:
        active_search = (~current_carry.deliberate_evaluated) & (
            current_carry.state.ball.live | jnp.any(restart_release_allowed)
        )
        physical_search = (
            current_carry.state.ball.live
            & (current_carry.nonactive_index <= 3)
            & (~current_carry.event_budget_exhausted)
        )
        return (
            (current_carry.event_index < 5)
            & (current_carry.remaining > 0.0)
            & (~current_carry.terminal)
            & (active_search | physical_search)
        )

    def finish_initial_no_event(
        _initial_carry: _BallEventCarry,
    ) -> _BallEventCarry:
        return initial_advanced_carry

    def run_event_loop(initial_carry: _BallEventCarry) -> _BallEventCarry:
        return jax.lax.while_loop(pending, event_iteration, initial_carry)

    final = jax.lax.cond(
        initial_no_event,
        finish_initial_no_event,
        run_event_loop,
        carry,
    )
    # A live ball can leave the loop only after a no-event iteration consumes
    # the full remainder, a terminal event sets it to zero, or the event budget
    # fails closed. If no search is pending while time remains, the ball is
    # necessarily dead and ``advance_smooth`` is an identity. The former final
    # call therefore evaluated one full ground/air branch with either dt=0 or
    # live=False on every physics substep without changing any leaf.
    final_state = _attach_held_ball(final.state, body=body)
    deliberate = DeliberateContactStep(
        state=final_state,
        occurred=final.deliberate_occurred,
        actor=final.deliberate_actor,
        contact=final.deliberate_contact,
        contest=final.contest,
        attempted=final.contact_attempted,
        occurrence=final.deliberate_occurrence,
    )
    return _BallEventStep(
        state=final_state,
        passive_occurred=final.passive_occurred,
        passive_actor=final.passive_actor,
        boundary=final.boundary,
        woodwork_occurred=final.woodwork_occurred,
        woodwork_kind=final.woodwork_kind,
        woodwork_time_fraction=final.woodwork_time_fraction,
        event_budget_exhausted=final.event_budget_exhausted,
        deliberate=deliberate,
        deliberate_time_fraction=final.deliberate_occurrence.time_fraction,
        contact_occurrences=ContactOccurrence(
            occurred=final.occurrence_occurred,
            actor=final.occurrence_actor,
            mechanism=final.occurrence_mechanism,
            law11_effect=final.occurrence_law11_effect,
            position=final.occurrence_position,
            time_fraction=final.occurrence_time_fraction,
        ),
        penalty_settling_contact=final.penalty_settling_contact,
    )


def _decrement_timers(state: State) -> State:
    players = state.players._replace(
        challenge_recovery_substeps=jnp.maximum(
            0, state.players.challenge_recovery_substeps - 1
        ),
        contact_lock_substeps=jnp.maximum(0, state.players.contact_lock_substeps - 1),
        aerial_recovery_substeps=jnp.maximum(
            0, state.players.aerial_recovery_substeps - 1
        ),
        possession_loss_lock_substeps=jnp.maximum(
            0, state.players.possession_loss_lock_substeps - 1
        ),
    )
    return state._replace(players=players)


def step_physics_substep(
    state: State,
    action: PhysicsAction,
    key: jax.Array,
    contest_override: ContestOverride,
    restart_release_allowed: jax.Array,
    *,
    dt: float,
    boundary_margin_m: float,
    contact_attempted: jax.Array,
    locomotion_enabled: jax.Array,
    position_update_enabled: jax.Array,
    separation_pinned: jax.Array,
    restart_positioning_enabled: jax.Array = False,
    restart_approach_enabled: jax.Array = False,
    ball_geometry: Ball = Ball(),
    stadium: Stadium = Stadium(),
    reach: Reach = Reach(),
    action_scale: ActionScale = ActionScale(),
    contact_timing: ContactTiming = ContactTiming(),
    player_physics: PlayerPhysics = PlayerPhysics(),
    body: BodyContact = BodyContact(),
    long_stamina: LongStamina = LongStamina(),
    short_stamina: ShortStamina = ShortStamina(),
    ball_physics: BallPhysics = BallPhysics(),
    contest_config: Contest = Contest(),
    regulation_elapsed_fraction: jax.Array = 0.0,
    period_boundary_penalty_enforced: jax.Array = False,
) -> PhysicsSubstep:
    """Advance players, contacts, and the ball by one physics interval."""

    contact_attempted = jnp.asarray(contact_attempted, dtype=jnp.bool_)
    locomotion_enabled = jnp.asarray(locomotion_enabled, dtype=jnp.bool_)
    position_update_enabled = jnp.asarray(position_update_enabled, dtype=jnp.bool_)
    separation_pinned = jnp.asarray(separation_pinned, dtype=jnp.bool_)
    restart_positioning_enabled = jnp.asarray(
        restart_positioning_enabled, dtype=jnp.bool_
    )
    restart_approach_enabled = jnp.asarray(restart_approach_enabled, dtype=jnp.bool_)
    if restart_approach_enabled.shape != ():
        raise ValueError("restart_approach_enabled must be scalar")
    period_boundary_penalty_enforced = jnp.asarray(
        period_boundary_penalty_enforced, dtype=jnp.bool_
    )
    separation_pinned = separation_pinned | (~position_update_enabled)

    def advance_restart_approach(approach_state: State):
        """Move only a visible continuous-restart taker toward release pose."""

        approach_actor = restart_actor_mask(approach_state)
        taker = jnp.clip(
            approach_state.restart.taker,
            0,
            approach_state.players.position.shape[0] - 1,
        )
        target, target_facing = restart_taker_release_pose(
            approach_state,
            stadium=stadium,
            ball=ball_geometry,
            body=body,
        )
        displacement = target - approach_state.players.position[taker]
        distance = jnp.linalg.norm(displacement)
        direction = displacement / jnp.maximum(distance, DIV_EPS)
        speed_limit = effective_speed_limit(
            approach_state.players.max_speed[taker],
            approach_state.players.stamina_long[taker],
            approach_state.players.stamina_short[taker],
            long=long_stamina,
            short=short_stamina,
        )
        travel = jnp.minimum(
            distance,
            speed_limit * jnp.asarray(dt, distance.dtype),
        )
        approached_position = (
            approach_state.players.position[taker] + direction * travel
        )
        approach_moved = approach_actor & (distance > GEOMETRY_EPS)
        approach_velocity = (
            approached_position - approach_state.players.position[taker]
        ) / jnp.asarray(dt, distance.dtype)
        approached_players = approach_state.players._replace(
            position=jnp.where(
                approach_actor[:, None],
                approach_state.players.position.at[taker].set(approached_position),
                approach_state.players.position,
            ),
            velocity=jnp.where(
                approach_actor[:, None],
                approach_state.players.velocity.at[taker].set(approach_velocity),
                approach_state.players.velocity,
            ),
            body_forward=jnp.where(
                approach_actor[:, None],
                approach_state.players.body_forward.at[taker].set(
                    body_forward_from_angle(target_facing)
                ),
                approach_state.players.body_forward,
            ),
        )
        return (
            approach_state._replace(players=approached_players),
            approach_actor,
            approach_moved,
        )

    def skip_restart_approach(approach_state: State):
        inactive = jnp.zeros_like(approach_state.players.active)
        return approach_state, inactive, inactive

    approached, approach_actor, approach_moved = jax.lax.cond(
        restart_approach_enabled,
        advance_restart_approach,
        skip_restart_approach,
        state,
    )
    player_position_update_enabled = position_update_enabled & (~approach_actor)
    separation_pinned = separation_pinned | approach_actor

    field_half_extent = jnp.asarray(
        [stadium.half_length, stadium.half_width],
        dtype=state.players.position.dtype,
    )
    player_step = step_players(
        approached.players,
        action.desired_velocity,
        locomotion_enabled,
        state.attack_direction,
        field_half_extent,
        boundary_margin_m=boundary_margin_m,
        dt=dt,
        body_target=action.body_target,
        body_target_valid=action.body_target_valid,
        position_update_enabled=player_position_update_enabled,
        pinned=separation_pinned,
        physics=player_physics,
        body=body,
        long_stamina=long_stamina,
        short_stamina=short_stamina,
    )
    # The ordinary player separator clips every active position to the live
    # pitch boundary.  A throw-in release pose is intentionally just outside
    # that boundary, so clipping the environment-authored, pinned approach
    # actor makes its target unreachable and freezes the restart countdown
    # forever. Preserve only this authoritative actor, keeping the tighter
    # boundary for every policy-controlled player.
    approached_player_state = player_step.players._replace(
        position=jnp.where(
            approach_actor[:, None],
            approached.players.position,
            player_step.players.position,
        ),
        velocity=jnp.where(
            approach_actor[:, None],
            approached.players.velocity,
            player_step.players.velocity,
        ),
        body_forward=jnp.where(
            approach_actor[:, None],
            approached.players.body_forward,
            player_step.players.body_forward,
        ),
    )
    contact_player_position = jnp.where(
        approach_actor[:, None],
        approached.players.position,
        player_step.contact_position,
    )
    player_start_velocity = jnp.where(
        approach_actor[:, None],
        approached.players.velocity,
        state.players.velocity,
    )
    contact_player_velocity = jnp.where(
        approach_actor[:, None],
        approached.players.velocity,
        player_step.contact_velocity,
    )
    player_velocity_delta = contact_player_velocity - player_start_velocity
    moved = approached._replace(players=approached_player_state)

    def enforce_restart_positioning(restart_state):
        positioning = prepare_restart_positioning(
            restart_state,
            stadium=stadium,
            ball=ball_geometry,
            body=body,
            preserve_taker=restart_approach_enabled,
            _resolve_constraints=False,
        )
        projected = restart_state._replace(
            players=restart_state.players._replace(
                position=positioning.position,
                body_forward=body_forward_from_angle(positioning.facing),
            ),
            restart_layout_ready=(
                restart_state.restart_layout_ready & (~jnp.any(positioning.forced))
            ),
        )
        position_changed = jnp.any(
            jnp.abs(projected.players.position - restart_state.players.position)
            > GEOMETRY_EPS,
            axis=-1,
        )
        return projected, position_changed

    forced_positioning = restart_positioning_enabled & (
        state.restart.kind != RK_GK_HOLD
    )
    moved, projected_rows = jax.lax.cond(
        forced_positioning,
        enforce_restart_positioning,
        lambda unpositioned: (
            unpositioned,
            jnp.zeros(unpositioned.players.active.shape, dtype=jnp.bool_),
        ),
        moved,
    )
    # Administrative restart projection has no within-substep swept path.
    # Otherwise the ball scheduler sees collision-free locomotion, while the
    # authoritative output retains the separator's positions and impulses.
    contact_player_position = jnp.where(
        projected_rows[:, None], moved.players.position, contact_player_position
    )
    contact_player_velocity = jnp.where(
        projected_rows[:, None], moved.players.velocity, contact_player_velocity
    )
    player_start_velocity = jnp.where(
        projected_rows[:, None], moved.players.velocity, player_start_velocity
    )
    player_velocity_delta = contact_player_velocity - player_start_velocity
    player_start_position = jnp.where(
        projected_rows[:, None],
        moved.players.position,
        state.players.position,
    )
    player_path_delta = jnp.where(
        projected_rows[:, None],
        0.0,
        contact_player_position - state.players.position,
    )
    stepped = _step_ball_events_chronological(
        moved,
        action,
        key,
        contest_override,
        restart_release_allowed,
        dt=dt,
        contact_attempted=contact_attempted,
        player_start_position=player_start_position,
        player_path_delta=player_path_delta,
        player_start_velocity=player_start_velocity,
        player_velocity_delta=player_velocity_delta,
        ball_geometry=ball_geometry,
        stadium=stadium,
        reach=reach,
        action_scale=action_scale,
        body=body,
        contact_timing=contact_timing,
        ball_physics=ball_physics,
        contest_config=contest_config,
        regulation_elapsed_fraction=regulation_elapsed_fraction,
        period_boundary_penalty_enforced=period_boundary_penalty_enforced,
    )
    deliberate = stepped.deliberate
    return PhysicsSubstep(
        state=_decrement_timers(stepped.state),
        deliberate_occurred=deliberate.occurred,
        deliberate_actor=deliberate.actor,
        deliberate_contact=deliberate.contact,
        contest=deliberate.contest,
        passive_occurred=stepped.passive_occurred,
        passive_actor=stepped.passive_actor,
        boundary=stepped.boundary,
        woodwork_occurred=stepped.woodwork_occurred,
        woodwork_kind=stepped.woodwork_kind,
        woodwork_time_fraction=stepped.woodwork_time_fraction,
        event_budget_exhausted=stepped.event_budget_exhausted,
        contact_attempted=deliberate.attempted,
        contact_occurrences=stepped.contact_occurrences,
        player_path_start=player_start_position,
        player_path_delta=player_path_delta,
        deliberate_time_fraction=stepped.deliberate_time_fraction,
        penalty_settling_contact=stepped.penalty_settling_contact,
        player_impact=player_step.impact,
        restart_approach_moved=approach_moved,
        restart_projection_moved=projected_rows,
    )
