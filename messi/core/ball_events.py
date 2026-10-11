"""한 physics substep의 bounded chronological free-ball event solver."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from messi.core.ball import (
    BallGeometry,
    BallPhysics,
    advance_ball_smooth,
    step_free_ball,
)
from messi.core.boundary import (
    CROSSING_NONE,
    BoundaryCrossing,
    GoalGeometry,
    detect_boundary_crossing,
)
from messi.core.constants import NO_PLAYER
from messi.core.goal_frame import (
    GoalFrameEvent,
    GoalFramePhysics,
    apply_goal_frame_contact,
    detect_goal_frame_contact,
    empty_goal_frame_event,
)
from messi.core.passive_contact import (
    PassiveContactEvent,
    PassiveContactPhysics,
    apply_passive_contact,
    detect_passive_contact,
    empty_passive_contact_event,
)

MAX_NONTERMINAL_BALL_EVENTS = 3


class ChronologicalBallStep(NamedTuple):
    ball: object
    passive: PassiveContactEvent
    passive_occurred: jax.Array
    passive_actor_mask: jax.Array
    rule_passive: PassiveContactEvent
    last_passive_actor: jax.Array
    woodwork: GoalFrameEvent
    boundary: BoundaryCrossing
    event_budget_exhausted: jax.Array


class _BallEventCarry(NamedTuple):
    ball: object
    remaining_dt: jax.Array
    event_count: jax.Array
    iteration: jax.Array
    excluded_actor_mask: jax.Array
    passive: PassiveContactEvent
    rule_passive: PassiveContactEvent
    woodwork: GoalFrameEvent
    boundary: BoundaryCrossing
    event_budget_exhausted: jax.Array
    passive_occurred_any: jax.Array
    last_passive_actor: jax.Array


def _empty_boundary(dtype) -> BoundaryCrossing:
    return BoundaryCrossing(
        occurred=jnp.bool_(False),
        axis=jnp.int32(CROSSING_NONE),
        time_fraction=jnp.asarray(0.0, dtype=dtype),
        position=jnp.zeros(3, dtype=dtype),
        through_goal=jnp.bool_(False),
    )


def _passive_contact_possible(
    ball,
    predicted_ball,
    players,
    roster,
    *,
    ball_geometry,
    physics,
    player_start_position=None,
    player_path_delta=None,
):
    """Exact solver를 안전하게 생략하기 위한 conservative player broad-phase."""

    player_start_position = (
        players.position if player_start_position is None else player_start_position
    )
    player_path_delta = (
        jnp.zeros_like(players.position)
        if player_path_delta is None
        else player_path_delta
    )
    # Exact detector가 사용하는 모든 silhouette를 포함해야 한다. Leg envelope만
    # 쓰면 valid custom torso가 더 클 때 broad-phase가 실제 충돌을 거짓으로
    # 제외한다.
    body_radius = max(
        physics.torso_radius_m,
        physics.head_radius_m,
        0.5 * physics.shoulder_width_m + physics.leg_radius_m,
    )
    horizontal_radius = ball_geometry.radius_m + body_radius
    ball_low = jnp.minimum(ball.position[:2], predicted_ball.position[:2])
    ball_high = jnp.maximum(ball.position[:2], predicted_ball.position[:2])
    player_end_position = player_start_position + player_path_delta
    player_low = jnp.minimum(player_start_position, player_end_position)
    player_high = jnp.maximum(player_start_position, player_end_position)
    horizontal = jnp.all(
        (ball_high[None, :] >= player_low - horizontal_radius)
        & (ball_low[None, :] <= player_high + horizontal_radius),
        axis=-1,
    )
    start_z = ball.position[2]
    end_z = predicted_ball.position[2]
    vertical = (jnp.minimum(start_z, end_z) <= roster.height + ball_geometry.radius_m) & (
        jnp.maximum(start_z, end_z) >= -ball_geometry.radius_m
    )
    finite = (
        jnp.all(jnp.isfinite(ball.position))
        & jnp.all(jnp.isfinite(predicted_ball.position))
        & jnp.all(jnp.isfinite(player_start_position))
        & jnp.all(jnp.isfinite(player_path_delta))
    )
    possible = jnp.any(
        players.active & vertical & horizontal
    )
    return jnp.asarray(ball.live) & (possible | (~finite))


def _goal_frame_possible(
    ball,
    predicted_ball,
    field_half_extent,
    *,
    ball_geometry,
    goal_geometry,
    physics,
):
    """Post/crossbar exact roots 앞의 conservative swept AABB gate."""

    radius = ball_geometry.radius_m + physics.frame_radius_m
    low = jnp.minimum(ball.position, predicted_ball.position)
    high = jnp.maximum(ball.position, predicted_ball.position)
    reaches_goal_x = (low[0] <= -field_half_extent[0] + radius) | (
        high[0] >= field_half_extent[0] - radius
    )
    overlaps_mouth_y = (low[1] <= 0.5 * goal_geometry.width_m + radius) & (
        high[1] >= -0.5 * goal_geometry.width_m - radius
    )
    overlaps_height = (low[2] <= goal_geometry.height_m + radius) & (
        high[2] >= -radius
    )
    finite = jnp.all(jnp.isfinite(ball.position)) & jnp.all(
        jnp.isfinite(predicted_ball.position)
    )
    return jnp.asarray(ball.live) & (
        (reaches_goal_x & overlaps_mouth_y & overlaps_height) | (~finite)
    )


def step_ball_events_chronological(
    ball,
    players,
    roster,
    *,
    dt: float | jax.Array,
    field_half_extent: jax.Array,
    excluded_actor: jax.Array = NO_PLAYER,
    excluded_actor_mask: jax.Array | None = None,
    rule_actor_mask: jax.Array | None = None,
    player_start_position: jax.Array | None = None,
    player_path_delta: jax.Array | None = None,
    ball_geometry: BallGeometry = BallGeometry(),
    ball_physics: BallPhysics = BallPhysics(),
    goal_geometry: GoalGeometry = GoalGeometry(),
    goal_frame_physics: GoalFramePhysics = GoalFramePhysics(),
    passive_contact_physics: PassiveContactPhysics = PassiveContactPhysics(),
    collect_events: bool = False,
) -> ChronologicalBallStep:
    """Passive contact·woodwork·boundary를 하나의 bounded timeline으로 처리한다.

    Event가 없으면 ``while_loop`` body는 한 번만 실행된다. 최대 세 nonterminal event를
    적용하며, 네 번째가 검출되면 unchecked remainder를 전개하지 않고 fail-closed한다.
    Boundary는 terminal이므로 slot을 소비하지 않고 즉시 종료한다.
    """

    dtype = ball.position.dtype
    total_dt = jnp.asarray(dt, dtype=dtype)
    player_count = players.position.shape[0]
    player_start_position = (
        players.position
        if player_start_position is None
        else jnp.asarray(player_start_position, dtype=dtype)
    )
    player_path_delta = (
        jnp.zeros_like(players.position)
        if player_path_delta is None
        else jnp.asarray(player_path_delta, dtype=dtype)
    )
    if player_start_position.shape != players.position.shape:
        raise ValueError("player_start_position shape는 players.position과 같아야 한다")
    if player_path_delta.shape != players.position.shape:
        raise ValueError("player_path_delta shape는 players.position과 같아야 한다")
    excluded_actor_mask = (
        jnp.zeros(player_count, dtype=jnp.bool_)
        if excluded_actor_mask is None
        else jnp.asarray(excluded_actor_mask, dtype=jnp.bool_)
    )
    if excluded_actor_mask.shape != (player_count,):
        raise ValueError("excluded_actor_mask는 player축 bool mask여야 한다")
    rule_actor_mask = (
        jnp.zeros(player_count, dtype=jnp.bool_)
        if rule_actor_mask is None
        else jnp.asarray(rule_actor_mask, dtype=jnp.bool_)
    )
    if rule_actor_mask.shape != (player_count,):
        raise ValueError("rule_actor_mask는 player축 bool mask여야 한다")
    initial_predicted = step_free_ball(
        ball,
        dt=total_dt,
        geometry=ball_geometry,
        physics=ball_physics,
    )
    initial_passive_possible = _passive_contact_possible(
        ball,
        initial_predicted,
        players,
        roster,
        ball_geometry=ball_geometry,
        physics=passive_contact_physics,
        player_start_position=player_start_position,
        player_path_delta=player_path_delta,
    )
    initial_frame_possible = _goal_frame_possible(
        ball,
        initial_predicted,
        field_half_extent,
        ball_geometry=ball_geometry,
        goal_geometry=goal_geometry,
        physics=goal_frame_physics,
    )
    initial_boundary = detect_boundary_crossing(
        ball.position,
        initial_predicted.position,
        ball.live,
        field_half_extent,
        goal=goal_geometry,
        ball=ball_geometry,
    )
    initial_no_event = (~jnp.asarray(ball.live)) | (
        (~initial_passive_possible)
        & (~initial_frame_possible)
        & (~initial_boundary.occurred)
    )
    initial_ball = jax.tree.map(
        lambda predicted, current: jnp.where(initial_no_event, predicted, current),
        initial_predicted,
        ball,
    )
    initial = _BallEventCarry(
        ball=initial_ball,
        remaining_dt=jnp.where(initial_no_event, 0.0, total_dt),
        event_count=jnp.int32(0),
        iteration=jnp.int32(0),
        excluded_actor_mask=excluded_actor_mask,
        passive=empty_passive_contact_event(dtype),
        rule_passive=empty_passive_contact_event(dtype),
        woodwork=empty_goal_frame_event(dtype),
        boundary=_empty_boundary(dtype),
        event_budget_exhausted=jnp.bool_(False),
        passive_occurred_any=jnp.bool_(False),
        last_passive_actor=jnp.int32(NO_PLAYER),
    )

    def pending(carry: _BallEventCarry):
        return (
            (carry.remaining_dt > 0.0)
            & (~carry.boundary.occurred)
            & (~carry.event_budget_exhausted)
            & (carry.iteration <= MAX_NONTERMINAL_BALL_EVENTS)
        )

    def iteration(carry: _BallEventCarry):
        predicted = step_free_ball(
            carry.ball,
            dt=carry.remaining_dt,
            geometry=ball_geometry,
            physics=ball_physics,
        )
        path_delta = predicted.position - carry.ball.position
        elapsed_fraction = jnp.where(
            total_dt > 0.0, 1.0 - carry.remaining_dt / total_dt, 0.0
        )
        remaining_fraction = jnp.where(
            total_dt > 0.0, carry.remaining_dt / total_dt, 0.0
        )
        segment_player_start = (
            player_start_position + elapsed_fraction * player_path_delta
        )
        segment_player_delta = remaining_fraction * player_path_delta
        passive = detect_passive_contact(
            carry.ball.position,
            path_delta,
            carry.ball.live,
            players,
            roster,
            excluded_actor=excluded_actor,
            excluded_actor_mask=carry.excluded_actor_mask,
            player_start_position=segment_player_start,
            player_path_delta=segment_player_delta,
            ball_radius_m=ball_geometry.radius_m,
            physics=passive_contact_physics,
        )
        woodwork = detect_goal_frame_contact(
            carry.ball.position,
            path_delta,
            carry.ball.live,
            half_length=field_half_extent[0],
            goal_half_width=0.5 * goal_geometry.width_m,
            goal_height=goal_geometry.height_m,
            ball_radius=ball_geometry.radius_m,
            physics=goal_frame_physics,
        )
        boundary = detect_boundary_crossing(
            carry.ball.position,
            predicted.position,
            carry.ball.live,
            field_half_extent,
            goal=goal_geometry,
            ball=ball_geometry,
        )
        passive_time = jnp.where(passive.occurred, passive.time_fraction, jnp.inf)
        woodwork_time = jnp.where(
            woodwork.occurred, woodwork.time_fraction, jnp.inf
        )
        boundary_time = jnp.where(
            boundary.occurred, boundary.time_fraction, jnp.inf
        )
        use_passive = passive.occurred & (passive_time <= woodwork_time) & (
            passive_time <= boundary_time
        )
        use_woodwork = woodwork.occurred & (woodwork_time < passive_time) & (
            woodwork_time <= boundary_time
        )
        use_boundary = boundary.occurred & (boundary_time < passive_time) & (
            boundary_time < woodwork_time
        )
        nonterminal = use_passive | use_woodwork
        fraction = jnp.where(
            use_passive,
            passive.time_fraction,
            jnp.where(
                use_woodwork,
                woodwork.time_fraction,
                jnp.where(use_boundary, boundary.time_fraction, 1.0),
            ),
        )
        budget_full = carry.event_count >= MAX_NONTERMINAL_BALL_EVENTS
        budget_exhausted = budget_full & nonterminal
        applied_passive = use_passive & (~budget_exhausted)
        impact_position = jnp.where(
            use_passive,
            passive.position,
            jnp.where(
                use_woodwork,
                woodwork.position,
                jnp.where(use_boundary, boundary.position, predicted.position),
            ),
        )
        # Detector가 찾은 event 시각의 속도·spin을 response에 전달한다. 이전에는
        # substep 끝까지 적분한 ``predicted``의 속도를 event 위치에 붙여, drag와
        # gravity가 충돌 뒤의 시간까지 선반영되는 chronology 오류가 있었다.
        event_dt = carry.remaining_dt * jnp.clip(fraction, 0.0, 1.0)
        # Event detector의 한 segment는 smooth swept path로 정의된다. Turf impact는
        # ``step_free_ball``의 segment endpoint 생성 단계에서 처리되므로 여기서는
        # 같은 smooth force kernel만 event 시각까지 적용해 response state를 만든다.
        at_event = advance_ball_smooth(
            carry.ball,
            dt=event_dt,
            geometry=ball_geometry,
            physics=ball_physics,
        )._replace(position=impact_position)
        passive_ball = apply_passive_contact(
            at_event,
            passive,
            dt=carry.remaining_dt,
            advance_remaining=False,
            surface_velocity=jnp.concatenate(
                (
                    player_path_delta[jnp.clip(passive.actor, 0, player_count - 1)]
                    / jnp.maximum(total_dt, 1.0e-9),
                    jnp.zeros(1, dtype=dtype),
                )
            ),
            physics=passive_contact_physics,
        )
        woodwork_ball = apply_goal_frame_contact(
            at_event,
            woodwork,
            physics=goal_frame_physics,
            ball_geometry=ball_geometry,
            ball_physics=ball_physics,
        )
        response_ball = jax.tree.map(
            lambda passive_value, woodwork_value: jnp.where(
                use_passive, passive_value, woodwork_value
            ),
            passive_ball,
            woodwork_ball,
        )
        next_ball = jax.tree.map(
            lambda response_value, predicted_value, event_value: jnp.where(
                nonterminal & (~budget_exhausted),
                response_value,
                jnp.where(use_boundary | budget_exhausted, event_value, predicted_value),
            ),
            response_ball,
            predicted,
            at_event,
        )
        remaining_dt = jnp.where(
            nonterminal & (~budget_exhausted),
            carry.remaining_dt * jnp.maximum(1.0 - fraction, 0.0),
            0.0,
        )
        global_fraction = jnp.where(
            total_dt > 0.0,
            1.0
            - carry.remaining_dt / total_dt
            + (carry.remaining_dt / total_dt) * fraction,
            0.0,
        )
        passive = passive._replace(time_fraction=global_fraction)
        woodwork = woodwork._replace(time_fraction=global_fraction)
        boundary = boundary._replace(time_fraction=global_fraction)
        first_passive = applied_passive & (~carry.passive.occurred)
        safe_actor = jnp.clip(passive.actor, 0, player_count - 1)
        first_rule_passive = (
            applied_passive
            & rule_actor_mask[safe_actor]
            & (~carry.rule_passive.occurred)
        )
        first_woodwork = use_woodwork & (~carry.woodwork.occurred)
        first_boundary = use_boundary & (~carry.boundary.occurred)
        captured_passive = jax.tree.map(
            lambda new, old: jnp.where(first_passive, new, old),
            passive,
            carry.passive,
        )
        captured_rule_passive = jax.tree.map(
            lambda new, old: jnp.where(first_rule_passive, new, old),
            passive,
            carry.rule_passive,
        )
        if collect_events:
            captured_woodwork = jax.tree.map(
                lambda new, old: jnp.where(first_woodwork, new, old),
                woodwork,
                carry.woodwork,
            )
        else:
            captured_passive = carry.passive
            captured_woodwork = carry.woodwork
        captured_boundary = jax.tree.map(
            lambda new, old: jnp.where(first_boundary, new, old),
            boundary,
            carry.boundary,
        )
        excluded_mask = carry.excluded_actor_mask.at[safe_actor].set(
            carry.excluded_actor_mask[safe_actor] | applied_passive
        )
        return _BallEventCarry(
            ball=next_ball,
            remaining_dt=remaining_dt,
            event_count=carry.event_count + nonterminal.astype(jnp.int32),
            iteration=carry.iteration + jnp.int32(1),
            excluded_actor_mask=excluded_mask,
            passive=captured_passive,
            rule_passive=captured_rule_passive,
            woodwork=captured_woodwork,
            boundary=captured_boundary,
            event_budget_exhausted=(
                carry.event_budget_exhausted | budget_exhausted
            ),
            passive_occurred_any=carry.passive_occurred_any | applied_passive,
            last_passive_actor=jnp.where(
                applied_passive,
                passive.actor,
                carry.last_passive_actor,
            ).astype(jnp.int32),
        )

    final = lax.while_loop(pending, iteration, initial)
    passive = final.passive
    woodwork = final.woodwork
    event_budget_exhausted = final.event_budget_exhausted
    if not collect_events:
        woodwork = empty_goal_frame_event(dtype)
        event_budget_exhausted = jnp.bool_(False)
    return ChronologicalBallStep(
        ball=final.ball,
        passive=passive,
        passive_occurred=final.passive_occurred_any,
        passive_actor_mask=final.excluded_actor_mask,
        rule_passive=final.rule_passive,
        last_passive_actor=final.last_passive_actor,
        woodwork=woodwork,
        boundary=final.boundary,
        event_budget_exhausted=event_budget_exhausted,
    )


__all__ = [
    "MAX_NONTERMINAL_BALL_EVENTS",
    "ChronologicalBallStep",
    "step_ball_events_chronological",
]
