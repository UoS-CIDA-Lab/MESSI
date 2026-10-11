"""Restart taker의 금지된 두 번째 ball contact 판정."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.core.boundary import GoalGeometry
from messi.core.constants import (
    NO_PLAYER,
    NO_TEAM,
    RK_CORNER,
    RK_FREEKICK,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_NONE,
    RK_OFFSIDE,
    RK_PENALTY,
    RK_THROWIN,
    TEAM_0,
    TEAM_1,
)
from messi.core.contact import MECHANISM_NONE, OUTCOME_NONE, ContactResult
from messi.core.geometry import BallGeometry
from messi.core.restart import select_restart_taker
from messi.core.state import RestartReleaseProvenance

_DEFAULT_GOAL = GoalGeometry()
_DEFAULT_BALL = BallGeometry()


class RestartRetouchEvent(NamedTuple):
    """한 physics substep에서 발생한 prohibited second touch."""

    occurred: jax.Array
    offender: jax.Array
    source_restart_kind: jax.Array
    team: jax.Array
    restart_kind: jax.Array
    indirect: jax.Array
    time_fraction: jax.Array
    contact_position: jax.Array
    restart_position: jax.Array


class RestartRetouchResolution(NamedTuple):
    state: object
    event: RestartRetouchEvent


def empty_restart_retouch_event(dtype=jnp.float32) -> RestartRetouchEvent:
    return RestartRetouchEvent(
        occurred=jnp.bool_(False),
        offender=jnp.int32(NO_PLAYER),
        source_restart_kind=jnp.int32(RK_NONE),
        team=jnp.int32(NO_TEAM),
        restart_kind=jnp.int32(RK_NONE),
        indirect=jnp.bool_(False),
        time_fraction=jnp.asarray(0.0, dtype=dtype),
        contact_position=jnp.zeros(3, dtype=dtype),
        restart_position=jnp.zeros(3, dtype=dtype),
    )


def _set_piece(kind: jax.Array) -> jax.Array:
    return (
        (kind == RK_KICKOFF)
        | (kind == RK_THROWIN)
        | (kind == RK_GOALKICK)
        | (kind == RK_CORNER)
        | (kind == RK_FREEKICK)
        | (kind == RK_PENALTY)
        | (kind == RK_OFFSIDE)
    )


def detect_restart_retouch(
    pre_contact_state,
    contact: ContactResult,
    contact_position: jax.Array,
    *,
    time_fraction: jax.Array = 1.0,
) -> RestartRetouchEvent:
    """현재 realized contact가 taker의 금지된 두 번째 touch인지 판정한다."""

    release = pre_contact_state.restart_release
    position = jnp.asarray(
        contact_position, dtype=pre_contact_state.ball.position.dtype
    )
    valid_team = (release.team == TEAM_0) | (release.team == TEAM_1)
    realized = (contact.actor >= 0) & (contact.outcome != OUTCOME_NONE)
    occurred = (
        release.active
        & release.untouched
        & valid_team
        & _set_piece(release.kind)
        & realized
        & (contact.actor == release.taker)
    )
    restart_team = (TEAM_1 - release.team).astype(jnp.int32)
    return RestartRetouchEvent(
        occurred=occurred,
        offender=jnp.where(occurred, contact.actor, NO_PLAYER).astype(jnp.int32),
        source_restart_kind=jnp.where(occurred, release.kind, RK_NONE).astype(
            jnp.int32
        ),
        team=jnp.where(occurred, restart_team, NO_TEAM).astype(jnp.int32),
        restart_kind=jnp.where(occurred, RK_FREEKICK, RK_NONE).astype(jnp.int32),
        indirect=occurred,
        time_fraction=jnp.where(occurred, time_fraction, 0.0).astype(position.dtype),
        contact_position=jnp.where(occurred, position, jnp.zeros_like(position)),
        restart_position=jnp.zeros_like(position),
    )


def _indirect_restart_spot(
    incident: jax.Array,
    restart_team: jax.Array,
    attack_direction: jax.Array,
    field_half_extent: jax.Array,
    *,
    goal: GoalGeometry,
    ball: BallGeometry,
) -> jax.Array:
    dtype = incident.dtype
    extent = jnp.asarray(field_half_extent, dtype=dtype)
    radius = jnp.asarray(ball.radius_m, dtype=dtype)
    ordinary = jnp.asarray(
        (
            jnp.clip(incident[0], -extent[0], extent[0]),
            jnp.clip(incident[1], -extent[1], extent[1]),
            radius,
        ),
        dtype=dtype,
    )
    safe_team = jnp.clip(restart_team, TEAM_0, TEAM_1)
    direction = attack_direction[safe_team]
    depth_from_attacking_goal = extent[0] - direction * ordinary[0]
    goal_area_half_width = 0.5 * goal.width_m + goal.goal_area_length_m
    inside_goal_area = (
        (depth_from_attacking_goal >= 0.0)
        & (depth_from_attacking_goal <= goal.goal_area_length_m)
        & (jnp.abs(ordinary[1]) <= goal_area_half_width)
    )
    goal_area_line_x = direction * (extent[0] - goal.goal_area_length_m)
    return ordinary.at[0].set(
        jnp.where(inside_goal_area, goal_area_line_x, ordinary[0])
    )




def apply_indirect_restart(
    state,
    roster,
    event: RestartRetouchEvent,
    field_half_extent: jax.Array,
    *,
    goal: GoalGeometry = _DEFAULT_GOAL,
    ball: BallGeometry = _DEFAULT_BALL,
) -> RestartRetouchResolution:
    """이미 판정된 event를 공통 indirect free-kick state로 한 번만 적용한다."""

    spot = _indirect_restart_spot(
        event.contact_position,
        event.team,
        state.attack_direction,
        field_half_extent,
        goal=goal,
        ball=ball,
    )
    taker = select_restart_taker(
        state,
        roster,
        jnp.int32(RK_FREEKICK),
        event.team,
        spot,
        field_half_extent=field_half_extent,
    )
    candidate = state._replace(
        ball=state.ball._replace(
            position=spot,
            velocity=jnp.zeros_like(state.ball.velocity),
            spin=jnp.zeros_like(state.ball.spin),
            live=jnp.bool_(False),
        ),
        possession=state.possession._replace(
            team=jnp.int32(NO_TEAM),
            player=jnp.int32(NO_PLAYER),
            previous_team=state.possession.team.astype(jnp.int32),
            control_ticks=jnp.int32(0),
        ),
        restart=state.restart._replace(
            kind=event.restart_kind,
            team=event.team,
            substeps_remaining=jnp.int32(0),
            taker=taker,
            indirect=jnp.bool_(True),
            opened_control_tick=state.control_tick,
        ),
        restart_release=RestartReleaseProvenance(
            active=jnp.bool_(False),
            untouched=jnp.bool_(False),
            kind=jnp.int32(RK_NONE),
            team=jnp.int32(NO_TEAM),
            taker=jnp.int32(NO_PLAYER),
            indirect=jnp.bool_(False),
            law11_direct_exempt=jnp.bool_(False),
            release_mechanism=jnp.int32(MECHANISM_NONE),
        ),
        gk_backpass_team=jnp.int32(NO_TEAM),
        offside_flagged=jnp.zeros_like(state.offside_flagged),
        offside_direct_exempt_team=jnp.int32(NO_TEAM),
        restart_layout_ready=jnp.bool_(False),
        restart_taker_approach=jnp.bool_(False),
        penalty_completion_active=jnp.bool_(False),
        penalty_completion_team=jnp.int32(NO_TEAM),
    )
    next_state = jax.tree.map(
        lambda new, old: jnp.where(event.occurred, new, old), candidate, state
    )
    resolved_event = event._replace(
        restart_position=jnp.where(
            event.occurred,
            spot,
            jnp.zeros_like(spot),
        )
    )
    return RestartRetouchResolution(next_state, resolved_event)


def resolve_restart_retouch(
    pre_contact_state,
    post_contact_state,
    roster,
    contact: ContactResult,
    contact_position: jax.Array,
    field_half_extent: jax.Array,
    *,
    time_fraction: jax.Array = 1.0,
    goal: GoalGeometry = _DEFAULT_GOAL,
    ball: BallGeometry = _DEFAULT_BALL,
) -> RestartRetouchResolution:
    """Retouch를 상대 team의 indirect free kick state로 변환한다."""

    detected = detect_restart_retouch(
        pre_contact_state,
        contact,
        contact_position,
        time_fraction=time_fraction,
    )
    return apply_indirect_restart(
        post_contact_state,
        roster,
        detected,
        field_half_extent,
        goal=goal,
        ball=ball,
    )


def detect_indirect_offence(
    state,
    roster,
    occurred: jax.Array,
    offender: jax.Array,
    contact_position: jax.Array,
    *,
    source_restart_kind: jax.Array = RK_NONE,
    time_fraction: jax.Array = 1.0,
) -> RestartRetouchEvent:
    """이미 판정된 technical offence를 state 변경 없는 event로 만든다."""

    count = state.players.position.shape[0]
    safe_offender = jnp.clip(offender, 0, count - 1)
    valid = occurred & (offender >= 0) & (offender < count)
    offender_team = roster.team_id[safe_offender]
    valid &= (offender_team == TEAM_0) | (offender_team == TEAM_1)
    restart_team = (TEAM_1 - offender_team).astype(jnp.int32)
    position = jnp.asarray(contact_position, dtype=state.ball.position.dtype)
    return RestartRetouchEvent(
        occurred=valid,
        offender=jnp.where(valid, offender, NO_PLAYER).astype(jnp.int32),
        source_restart_kind=jnp.where(valid, source_restart_kind, RK_NONE).astype(jnp.int32),
        team=jnp.where(valid, restart_team, NO_TEAM).astype(jnp.int32),
        restart_kind=jnp.where(valid, RK_FREEKICK, RK_NONE).astype(jnp.int32),
        indirect=valid,
        time_fraction=jnp.where(valid, time_fraction, 0.0).astype(position.dtype),
        contact_position=jnp.where(valid, position, jnp.zeros_like(position)),
        restart_position=jnp.zeros_like(position),
    )


def resolve_indirect_offence(
    state,
    roster,
    occurred: jax.Array,
    offender: jax.Array,
    contact_position: jax.Array,
    field_half_extent: jax.Array,
    *,
    source_restart_kind: jax.Array = RK_NONE,
    time_fraction: jax.Array = 1.0,
    goal: GoalGeometry = _DEFAULT_GOAL,
    ball: BallGeometry = _DEFAULT_BALL,
) -> RestartRetouchResolution:
    """Technical offence event를 상대 indirect free kick으로 적용한다."""

    event = detect_indirect_offence(
        state,
        roster,
        occurred,
        offender,
        contact_position,
        source_restart_kind=source_restart_kind,
        time_fraction=time_fraction,
    )
    return apply_indirect_restart(
        state,
        roster,
        event,
        field_half_extent,
        goal=goal,
        ball=ball,
    )


__all__ = [
    "RestartRetouchEvent",
    "RestartRetouchResolution",
    "apply_indirect_restart",
    "detect_indirect_offence",
    "detect_restart_retouch",
    "empty_restart_retouch_event",
    "resolve_indirect_offence",
    "resolve_restart_retouch",
]
