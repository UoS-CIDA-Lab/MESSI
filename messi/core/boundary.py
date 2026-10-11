"""공 전체의 swept boundary crossing과 restart state transition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.core.constants import (
    BALL_EVENT_CORNER,
    BALL_EVENT_GOAL,
    BALL_EVENT_GOALKICK,
    BALL_EVENT_NONE,
    BALL_EVENT_THROWIN,
    NO_PLAYER,
    NO_TEAM,
    RK_CORNER,
    RK_GK_HOLD,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_NONE,
    RK_THROWIN,
    TEAM_0,
    TEAM_1,
)
from messi.core.contact import MECHANISM_NONE, MECHANISM_THROW
from messi.core.geometry import BallGeometry
from messi.core.numeric import require_float32_representable
from messi.core.restart import select_restart_taker
from messi.core.state import RestartReleaseProvenance

CROSSING_NONE = 0
CROSSING_GOAL_LINE = 1
CROSSING_TOUCH_LINE = 2

_GEOMETRY_EPS = 1.0e-6


@dataclass(frozen=True, slots=True)
class GoalGeometry:
    """IFAB pitch goal과 goal-area의 고정 치수."""

    width_m: float = 7.32
    height_m: float = 2.44
    goal_area_length_m: float = 5.5

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("width_m", 1.0, 10.0),
            ("height_m", 1.0, 4.0),
            ("goal_area_length_m", 1.0, 10.0),
        ):
            value = require_float32_representable(name, getattr(self, name))
            if not lower <= value <= upper:
                raise ValueError(f"{name}은 [{lower}, {upper}] 범위여야 한다")


_DEFAULT_GOAL = GoalGeometry()
_DEFAULT_BALL = BallGeometry()


class BoundaryCrossing(NamedTuple):
    """한 직선 segment에서 처음 발생한 complete-ball crossing."""

    occurred: jax.Array
    axis: jax.Array
    time_fraction: jax.Array
    position: jax.Array
    through_goal: jax.Array


class BoundaryEvent(NamedTuple):
    """Goal 또는 out-of-play로 분류된 fixed-shape event."""

    occurred: jax.Array
    kind: jax.Array
    team: jax.Array
    scoring_team: jax.Array
    time_fraction: jax.Array
    position: jax.Array


class BoundaryResolution(NamedTuple):
    """Boundary event가 반영된 authoritative state."""

    state: object
    event: BoundaryEvent
    goal_awarded: jax.Array


def empty_boundary_event(dtype=jnp.float32) -> BoundaryEvent:
    return BoundaryEvent(
        occurred=jnp.bool_(False),
        kind=jnp.int32(BALL_EVENT_NONE),
        team=jnp.int32(NO_TEAM),
        scoring_team=jnp.int32(NO_TEAM),
        time_fraction=jnp.asarray(0.0, dtype=dtype),
        position=jnp.zeros(3, dtype=dtype),
    )


def _crossing_fraction(
    before: jax.Array,
    after: jax.Array,
    boundary: jax.Array,
) -> jax.Array:
    delta = after - before
    target = jnp.where(after >= 0.0, boundary, -boundary)
    raw = (target - before) / jnp.where(jnp.abs(delta) > _GEOMETRY_EPS, delta, 1.0)
    return jnp.where(jnp.abs(before) > boundary, 0.0, jnp.clip(raw, 0.0, 1.0))


def detect_boundary_crossing(
    position_before: jax.Array,
    position_after: jax.Array,
    ball_live: jax.Array,
    field_half_extent: jax.Array,
    *,
    goal: GoalGeometry = _DEFAULT_GOAL,
    ball: BallGeometry = _DEFAULT_BALL,
) -> BoundaryCrossing:
    """공 전체가 가장 먼저 넘은 goal line 또는 touchline을 검출한다."""

    before = jnp.asarray(position_before)
    after = jnp.asarray(position_after)
    half_extent = jnp.asarray(field_half_extent, dtype=after.dtype)
    radius = jnp.asarray(ball.radius_m, dtype=after.dtype)
    line_x = half_extent[0] + radius
    line_y = half_extent[1] + radius
    x_out = jnp.abs(after[0]) > line_x
    y_out = jnp.abs(after[1]) > line_y
    x_fraction = _crossing_fraction(before[0], after[0], line_x)
    y_fraction = _crossing_fraction(before[1], after[1], line_y)
    goal_line_first = x_out & ((~y_out) | (x_fraction <= y_fraction))
    touch_line_first = y_out & ((~x_out) | (y_fraction < x_fraction))
    occurred = jnp.asarray(ball_live, dtype=jnp.bool_) & (
        goal_line_first | touch_line_first
    )
    fraction = jnp.where(
        goal_line_first,
        x_fraction,
        jnp.where(touch_line_first, y_fraction, 0.0),
    )
    position = before + fraction * (after - before)
    position = position.at[0].set(
        jnp.where(
            goal_line_first,
            jnp.where(after[0] >= 0.0, line_x, -line_x),
            position[0],
        )
    )
    position = position.at[1].set(
        jnp.where(
            touch_line_first,
            jnp.where(after[1] >= 0.0, line_y, -line_y),
            position[1],
        )
    )
    through_goal = (
        occurred
        & goal_line_first
        & (jnp.abs(position[1]) <= 0.5 * goal.width_m - radius)
        & (position[2] <= goal.height_m - radius)
    )
    axis = jnp.where(
        occurred,
        jnp.where(goal_line_first, CROSSING_GOAL_LINE, CROSSING_TOUCH_LINE),
        CROSSING_NONE,
    ).astype(jnp.int32)
    return BoundaryCrossing(
        occurred=occurred,
        axis=axis,
        time_fraction=jnp.where(occurred, fraction, 0.0).astype(after.dtype),
        position=jnp.where(occurred, position, jnp.zeros_like(position)),
        through_goal=through_goal,
    )


def classify_boundary_event(
    crossing: BoundaryCrossing,
    last_touch_team: jax.Array,
    possession_team: jax.Array,
    attack_direction: jax.Array,
    kickoff_team: jax.Array,
) -> BoundaryEvent:
    """마지막 접촉 provenance와 공격 방향으로 crossing을 분류한다."""

    team_plus = jnp.where(attack_direction[TEAM_0] > 0.0, TEAM_0, TEAM_1).astype(
        jnp.int32
    )
    team_minus = (TEAM_1 - team_plus).astype(jnp.int32)
    scoring_team = jnp.where(crossing.position[0] >= 0.0, team_plus, team_minus)
    defending_team = (TEAM_1 - scoring_team).astype(jnp.int32)
    valid_kickoff = (kickoff_team == TEAM_0) | (kickoff_team == TEAM_1)
    fallback = jnp.where(valid_kickoff, kickoff_team, TEAM_0)
    end_attacker = jnp.where(
        crossing.position[0] > _GEOMETRY_EPS,
        team_plus,
        jnp.where(
            crossing.position[0] < -_GEOMETRY_EPS,
            team_minus,
            fallback,
        ),
    )
    valid_last = (last_touch_team == TEAM_0) | (last_touch_team == TEAM_1)
    valid_possession = (possession_team == TEAM_0) | (possession_team == TEAM_1)
    effective_last = jnp.where(
        valid_last,
        last_touch_team,
        jnp.where(valid_possession, possession_team, end_attacker),
    ).astype(jnp.int32)
    restart_team = (TEAM_1 - effective_last).astype(jnp.int32)

    goal = crossing.occurred & crossing.through_goal
    goal_line_out = crossing.occurred & (crossing.axis == CROSSING_GOAL_LINE) & (~goal)
    touch_line_out = crossing.occurred & (crossing.axis == CROSSING_TOUCH_LINE)
    corner = goal_line_out & (effective_last == defending_team)
    goal_kick = goal_line_out & (~corner)
    kind = jnp.where(
        goal,
        BALL_EVENT_GOAL,
        jnp.where(
            corner,
            BALL_EVENT_CORNER,
            jnp.where(
                goal_kick,
                BALL_EVENT_GOALKICK,
                jnp.where(touch_line_out, BALL_EVENT_THROWIN, BALL_EVENT_NONE),
            ),
        ),
    ).astype(jnp.int32)
    event_team = jnp.where(
        goal,
        scoring_team,
        jnp.where(corner | goal_kick | touch_line_out, restart_team, NO_TEAM),
    ).astype(jnp.int32)
    return BoundaryEvent(
        occurred=crossing.occurred,
        kind=kind,
        team=event_team,
        scoring_team=jnp.where(goal, scoring_team, NO_TEAM).astype(jnp.int32),
        time_fraction=crossing.time_fraction,
        position=crossing.position,
    )


def _canonical_restart_spot(
    kind: jax.Array,
    team: jax.Array,
    incident: jax.Array,
    attack_direction: jax.Array,
    field_half_extent: jax.Array,
    radius: jax.Array,
    goal: GoalGeometry,
) -> jax.Array:
    safe_team = jnp.clip(team, TEAM_0, TEAM_1)
    direction = attack_direction[safe_team]
    x_sign = jnp.where(incident[0] >= 0.0, 1.0, -1.0)
    y_sign = jnp.where(incident[1] >= 0.0, 1.0, -1.0)
    half_length, half_width = field_half_extent
    centre = jnp.asarray((0.0, 0.0, radius), dtype=incident.dtype)
    throw_in = jnp.asarray(
        (
            jnp.clip(incident[0], -half_length, half_length),
            y_sign * half_width,
            radius,
        ),
        dtype=incident.dtype,
    )
    corner = jnp.asarray(
        (x_sign * half_length, y_sign * half_width, radius),
        dtype=incident.dtype,
    )
    goal_kick = jnp.asarray(
        (
            -direction * (half_length - 0.5 * goal.goal_area_length_m),
            0.0,
            radius,
        ),
        dtype=incident.dtype,
    )
    ordinary = jnp.asarray(
        (
            jnp.clip(incident[0], -half_length, half_length),
            jnp.clip(incident[1], -half_width, half_width),
            radius,
        ),
        dtype=incident.dtype,
    )
    return jnp.where(
        kind == RK_KICKOFF,
        centre,
        jnp.where(
            kind == RK_THROWIN,
            throw_in,
            jnp.where(
                kind == RK_CORNER,
                corner,
                jnp.where(kind == RK_GOALKICK, goal_kick, ordinary),
            ),
        ),
    )




def _last_touch_team(state, roster) -> jax.Array:
    actor = state.possession.last_contact.actor
    count = roster.team_id.shape[0]
    valid = (actor >= 0) & (actor < count)
    safe_actor = jnp.clip(actor, 0, count - 1)
    return jnp.where(valid, roster.team_id[safe_actor], NO_TEAM).astype(jnp.int32)


def resolve_boundary_crossing(
    state,
    roster,
    crossing: BoundaryCrossing,
    field_half_extent: jax.Array,
    *,
    goal: GoalGeometry = _DEFAULT_GOAL,
    ball: BallGeometry = _DEFAULT_BALL,
) -> BoundaryResolution:
    """Goal과 out-of-play를 score, dead ball과 restart state에 반영한다."""

    raw = classify_boundary_event(
        crossing,
        _last_touch_team(state, roster),
        state.possession.team,
        state.attack_direction,
        state.kickoff_team,
    )
    raw_goal = raw.occurred & (raw.kind == BALL_EVENT_GOAL)
    release = state.restart_release
    valid_release_team = (release.team == TEAM_0) | (release.team == TEAM_1)
    from_restart = release.active & release.untouched & valid_release_team
    gk_distribution = (release.kind == RK_GK_HOLD) & (
        release.release_mechanism == MECHANISM_THROW
    )
    direct_own_goal = (
        raw_goal
        & from_restart
        & (release.team != raw.scoring_team)
        & ((release.kind != RK_GK_HOLD) | gk_distribution)
    )
    direct_opponent_goal = raw_goal & from_restart & (release.team == raw.scoring_team)
    opponent_goal_forbidden = direct_opponent_goal & (
        (release.kind == RK_THROWIN) | gk_distribution | release.indirect
    )
    goal_awarded = raw_goal & (~direct_own_goal) & (~opponent_goal_forbidden)
    conceding_team = (TEAM_1 - raw.scoring_team).astype(jnp.int32)
    restart_team = jnp.where(
        goal_awarded,
        conceding_team,
        jnp.where(
            direct_own_goal,
            raw.scoring_team,
            jnp.where(opponent_goal_forbidden, conceding_team, raw.team),
        ),
    ).astype(jnp.int32)
    final_kind = jnp.where(
        direct_own_goal,
        BALL_EVENT_CORNER,
        jnp.where(opponent_goal_forbidden, BALL_EVENT_GOALKICK, raw.kind),
    ).astype(jnp.int32)
    restart_kind = jnp.where(
        goal_awarded,
        RK_KICKOFF,
        jnp.where(
            final_kind == BALL_EVENT_THROWIN,
            RK_THROWIN,
            jnp.where(
                final_kind == BALL_EVENT_CORNER,
                RK_CORNER,
                jnp.where(
                    final_kind == BALL_EVENT_GOALKICK,
                    RK_GOALKICK,
                    RK_NONE,
                ),
            ),
        ),
    ).astype(jnp.int32)
    radius = jnp.asarray(ball.radius_m, dtype=state.ball.position.dtype)
    spot = _canonical_restart_spot(
        restart_kind,
        restart_team,
        crossing.position,
        state.attack_direction,
        jnp.asarray(field_half_extent, dtype=state.ball.position.dtype),
        radius,
        goal,
    )
    taker = select_restart_taker(
        state,
        roster,
        restart_kind,
        restart_team,
        spot,
        field_half_extent=field_half_extent,
    )
    safe_scoring_team = jnp.clip(raw.scoring_team, TEAM_0, TEAM_1)
    score = state.score.at[safe_scoring_team].add(
        goal_awarded.astype(state.score.dtype)
    )
    candidate = state._replace(
        ball=state.ball._replace(
            position=spot,
            velocity=jnp.zeros_like(state.ball.velocity),
            spin=jnp.zeros_like(state.ball.spin),
            live=jnp.bool_(False),
        ),
        kickoff_team=jnp.where(goal_awarded, conceding_team, state.kickoff_team).astype(
            jnp.int32
        ),
        possession=state.possession._replace(
            team=jnp.int32(NO_TEAM),
            player=jnp.int32(NO_PLAYER),
            previous_team=state.possession.team.astype(jnp.int32),
            control_ticks=jnp.int32(0),
        ),
        restart=state.restart._replace(
            kind=restart_kind,
            team=restart_team,
            substeps_remaining=jnp.int32(0),
            taker=taker,
            indirect=jnp.bool_(False),
            opened_control_tick=state.control_tick,
        ),
        score=score,
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
        penalty_completion_active=jnp.bool_(False),
        penalty_completion_team=jnp.int32(NO_TEAM),
        restart_layout_ready=jnp.bool_(False),
        restart_taker_approach=jnp.bool_(False),
    )
    next_state = jax.tree.map(
        lambda new, old: jnp.where(crossing.occurred, new, old),
        candidate,
        state,
    )
    event = BoundaryEvent(
        occurred=raw.occurred,
        kind=jnp.where(raw.occurred, final_kind, BALL_EVENT_NONE).astype(jnp.int32),
        team=jnp.where(
            raw.occurred,
            jnp.where(goal_awarded, raw.scoring_team, restart_team),
            NO_TEAM,
        ).astype(jnp.int32),
        scoring_team=jnp.where(goal_awarded, raw.scoring_team, NO_TEAM).astype(
            jnp.int32
        ),
        time_fraction=raw.time_fraction,
        position=raw.position,
    )
    return BoundaryResolution(next_state, event, goal_awarded)


__all__ = [
    "CROSSING_GOAL_LINE",
    "CROSSING_NONE",
    "CROSSING_TOUCH_LINE",
    "BoundaryCrossing",
    "BoundaryEvent",
    "BoundaryResolution",
    "GoalGeometry",
    "classify_boundary_event",
    "detect_boundary_crossing",
    "empty_boundary_event",
    "resolve_boundary_crossing",
]
