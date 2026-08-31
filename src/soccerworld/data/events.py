"""Device-side canonical event batches for training and audit capture."""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from soccerworld._engine.constants import (
    BALL_EVENT_CORNER,
    BALL_EVENT_GOAL,
    BALL_EVENT_GOALKICK,
    BALL_EVENT_NONE,
    BALL_EVENT_THROWIN,
    DISCIPLINE_NONE,
    DISCIPLINE_RED,
    DISCIPLINE_SAMPLE,
    DISCIPLINE_YELLOW,
    FOUL_CHARGE,
    FOUL_NONE,
    FOUL_SETPIECE,
    FOUL_TACKLE,
    FOUL_THROW,
    NO_PLAYER,
    NO_TEAM,
    TOUCH_BODY_TRAP,
    TOUCH_DEFLECT,
    TOUCH_DRIBBLE,
    TOUCH_GK_CATCH,
    TOUCH_INTERCEPT,
    TOUCH_NONE,
    TOUCH_PARRY,
    TOUCH_PASS,
    TOUCH_PASS_HEAD,
    TOUCH_SHOOT,
    TOUCH_SHOOT_HEAD,
    TOUCH_TACKLE,
    WOODWORK_CROSSBAR,
    WOODWORK_NONE,
    WOODWORK_POST,
)


class EventDomain(IntEnum):
    """Stable namespace for :attr:`EventBatch.code`."""

    NONE = 0
    TOUCH = 1
    BALL_BOUNDARY = 2
    WOODWORK = 3
    FOUL = 4


class TouchEventCode(IntEnum):
    NONE = TOUCH_NONE
    PASS = TOUCH_PASS
    SHOT = TOUCH_SHOOT
    HEADED_PASS = TOUCH_PASS_HEAD
    HEADED_SHOT = TOUCH_SHOOT_HEAD
    DRIBBLE = TOUCH_DRIBBLE
    TACKLE = TOUCH_TACKLE
    GOALKEEPER_CATCH = TOUCH_GK_CATCH
    INTERCEPTION = TOUCH_INTERCEPT
    DEFLECTION = TOUCH_DEFLECT
    PARRY = TOUCH_PARRY
    BODY_TRAP = TOUCH_BODY_TRAP


class FoulEventCode(IntEnum):
    NONE = FOUL_NONE
    TACKLE = FOUL_TACKLE
    CHARGE = FOUL_CHARGE
    THROW_IN_RETOUCH = FOUL_THROW
    SET_PIECE_RETOUCH = FOUL_SETPIECE


class DisciplinaryOutcome(IntEnum):
    """Observed or sampled card result for foul reconstruction."""

    SAMPLE = DISCIPLINE_SAMPLE
    NONE = DISCIPLINE_NONE
    YELLOW = DISCIPLINE_YELLOW
    RED = DISCIPLINE_RED


class BallBoundaryEventCode(IntEnum):
    NONE = BALL_EVENT_NONE
    GOAL = BALL_EVENT_GOAL
    CORNER = BALL_EVENT_CORNER
    GOAL_KICK = BALL_EVENT_GOALKICK
    THROW_IN = BALL_EVENT_THROWIN


class WoodworkEventCode(IntEnum):
    NONE = WOODWORK_NONE
    POST = WOODWORK_POST
    CROSSBAR = WOODWORK_CROSSBAR


class EventBatch(NamedTuple):
    """Fixed-width events produced by one control frame.

    Empty cells have ``valid=False`` and all identifiers set to their public sentinels. Positions
    are the physical event positions when the engine exposes one; callers must consult ``valid`` and
    must not infer a position for events whose position is zero-filled.
    """

    valid: Array
    domain: Array
    code: Array
    actor: Array
    actor_id: Array
    target: Array
    target_id: Array
    team: Array
    position: Array
    velocity_before: Array
    velocity_after: Array
    time_of_impact: Array
    impulse: Array
    control_tick: Array
    substep: Array
    phase: Array

    @classmethod
    def empty(cls, dtype=jnp.float32) -> EventBatch:
        return cls(
            valid=jnp.empty((0,), dtype=jnp.bool_),
            domain=jnp.empty((0,), dtype=jnp.int32),
            code=jnp.empty((0,), dtype=jnp.int32),
            actor=jnp.empty((0,), dtype=jnp.int32),
            actor_id=jnp.empty((0,), dtype=jnp.int32),
            target=jnp.empty((0,), dtype=jnp.int32),
            target_id=jnp.empty((0,), dtype=jnp.int32),
            team=jnp.empty((0,), dtype=jnp.int32),
            position=jnp.empty((0, 3), dtype=dtype),
            velocity_before=jnp.empty((0, 3), dtype=dtype),
            velocity_after=jnp.empty((0, 3), dtype=dtype),
            time_of_impact=jnp.empty((0,), dtype=dtype),
            impulse=jnp.empty((0,), dtype=dtype),
            control_tick=jnp.empty((0,), dtype=jnp.int32),
            substep=jnp.empty((0,), dtype=jnp.int32),
            phase=jnp.empty((0,), dtype=jnp.int32),
        )


def _domain(value: EventDomain, valid: Array) -> Array:
    return jnp.where(valid, jnp.int32(value), jnp.int32(EventDomain.NONE))


def events_from_state(state, previous_state=None) -> EventBatch:
    """Project the engine's fixed-width frame telemetry without host synchronization.

    ``previous_state`` suppresses a foul latch that was already visible before this transition.
    Contact, boundary, and woodwork buffers are reset at each engine step and need no such edge
    detection. Fields that a domain does not produce are zero-filled; ``valid`` and ``domain`` are
    authoritative.
    """

    touch_actor = state.touch_event_actor.reshape(-1).astype(jnp.int32)
    touch_kind = state.touch_event_code.reshape(-1).astype(jnp.int32)
    touch_valid = (touch_actor >= 0) & (touch_kind != TOUCH_NONE)
    safe_actor = jnp.clip(touch_actor, 0, state.team_id.shape[0] - 1)
    touch_team = jnp.where(touch_valid, state.team_id[safe_actor], NO_TEAM)
    touch_count = touch_kind.shape[0]
    touch_steps = jnp.repeat(
        jnp.arange(state.touch_event_actor.shape[0], dtype=jnp.int32),
        state.touch_event_actor.shape[1],
    )
    touch_phases = jnp.tile(
        jnp.arange(state.touch_event_actor.shape[1], dtype=jnp.int32),
        state.touch_event_actor.shape[0],
    )

    ball_kind = state.ball_event_kind.astype(jnp.int32)
    ball_valid = ball_kind != BALL_EVENT_NONE
    ball_count = ball_kind.shape[0]

    woodwork_kind = state.woodwork_kind.astype(jnp.int32)
    woodwork_valid = woodwork_kind != WOODWORK_NONE
    woodwork_count = woodwork_kind.shape[0]

    foul_kind = jnp.reshape(state.foul_kind.astype(jnp.int32), (1,))
    foul_actor = jnp.reshape(state.foul_actor.astype(jnp.int32), (1,))
    foul_target = jnp.reshape(state.foul_victim.astype(jnp.int32), (1,))
    foul_valid = foul_kind != FOUL_NONE
    if previous_state is not None:
        foul_changed = (
            (state.foul_kind != previous_state.foul_kind)
            | (state.foul_actor != previous_state.foul_actor)
            | (state.foul_victim != previous_state.foul_victim)
        )
        foul_valid = foul_valid & foul_changed
    safe_foul_actor = jnp.clip(foul_actor, 0, state.team_id.shape[0] - 1)
    safe_foul_target = jnp.clip(foul_target, 0, state.team_id.shape[0] - 1)
    foul_actor_valid = foul_valid & (foul_actor >= 0)
    foul_target_valid = foul_valid & (foul_target >= 0)
    foul_team = jnp.where(
        foul_actor_valid, state.team_id[safe_foul_actor], NO_TEAM
    )
    foul_position = jnp.where(foul_valid[:, None], state.ball_pos[None, :], 0.0)

    zero_ball_vectors = jnp.zeros((ball_count, 3), dtype=state.ball_pos.dtype)
    zero_woodwork_vectors = jnp.zeros(
        (woodwork_count, 3), dtype=state.ball_pos.dtype
    )
    zero_foul_vectors = jnp.zeros((1, 3), dtype=state.ball_pos.dtype)
    zero_ball_scalars = jnp.zeros((ball_count,), dtype=state.ball_pos.dtype)
    zero_woodwork_scalars = jnp.zeros(
        (woodwork_count,), dtype=state.ball_pos.dtype
    )
    zero_foul_scalars = jnp.zeros((1,), dtype=state.ball_pos.dtype)

    valid = jnp.concatenate((touch_valid, ball_valid, woodwork_valid, foul_valid))
    return EventBatch(
        valid=valid,
        domain=jnp.concatenate(
            (
                _domain(EventDomain.TOUCH, touch_valid),
                _domain(EventDomain.BALL_BOUNDARY, ball_valid),
                _domain(EventDomain.WOODWORK, woodwork_valid),
                _domain(EventDomain.FOUL, foul_valid),
            )
        ),
        code=jnp.concatenate(
            (
                jnp.where(touch_valid, touch_kind, TOUCH_NONE),
                jnp.where(ball_valid, ball_kind, BALL_EVENT_NONE),
                jnp.where(woodwork_valid, woodwork_kind, WOODWORK_NONE),
                jnp.where(foul_valid, foul_kind, FOUL_NONE),
            )
        ),
        actor=jnp.concatenate(
            (
                jnp.where(touch_valid, touch_actor, NO_PLAYER),
                jnp.full((ball_count,), NO_PLAYER, dtype=jnp.int32),
                jnp.full((woodwork_count,), NO_PLAYER, dtype=jnp.int32),
                jnp.where(foul_actor_valid, foul_actor, NO_PLAYER),
            )
        ),
        actor_id=jnp.concatenate(
            (
                jnp.where(
                    touch_valid,
                    state.touch_event_player_id.reshape(-1),
                    NO_PLAYER,
                ).astype(jnp.int32),
                jnp.full((ball_count,), NO_PLAYER, dtype=jnp.int32),
                jnp.full((woodwork_count,), NO_PLAYER, dtype=jnp.int32),
                jnp.where(
                    foul_actor_valid, state.player_id[safe_foul_actor], NO_PLAYER
                ),
            )
        ),
        target=jnp.concatenate(
            (
                jnp.full_like(touch_actor, NO_PLAYER),
                jnp.full((ball_count,), NO_PLAYER, dtype=jnp.int32),
                jnp.full((woodwork_count,), NO_PLAYER, dtype=jnp.int32),
                jnp.where(foul_target_valid, foul_target, NO_PLAYER),
            )
        ),
        target_id=jnp.concatenate(
            (
                jnp.full((touch_count,), NO_PLAYER, dtype=jnp.int32),
                jnp.full((ball_count,), NO_PLAYER, dtype=jnp.int32),
                jnp.full((woodwork_count,), NO_PLAYER, dtype=jnp.int32),
                jnp.where(
                    foul_target_valid, state.player_id[safe_foul_target], NO_PLAYER
                ),
            )
        ),
        team=jnp.concatenate(
            (
                touch_team.astype(jnp.int32),
                jnp.where(ball_valid, state.ball_event_team, NO_TEAM).astype(jnp.int32),
                jnp.full((woodwork_count,), NO_TEAM, dtype=jnp.int32),
                foul_team.astype(jnp.int32),
            )
        ),
        position=jnp.concatenate(
            (
                state.touch_event_ball_pos.reshape(-1, 3),
                state.ball_event_pos,
                state.woodwork_pos,
                foul_position,
            )
        ),
        velocity_before=jnp.concatenate(
            (
                state.touch_event_ball_vel_before.reshape(-1, 3),
                state.ball_event_vel,
                state.woodwork_vel_in,
                zero_foul_vectors,
            )
        ),
        velocity_after=jnp.concatenate(
            (
                state.touch_event_ball_vel_after.reshape(-1, 3),
                zero_ball_vectors,
                zero_woodwork_vectors,
                zero_foul_vectors,
            )
        ),
        time_of_impact=jnp.concatenate(
            (
                state.touch_event_toi.reshape(-1),
                zero_ball_scalars,
                zero_woodwork_scalars,
                zero_foul_scalars,
            )
        ),
        impulse=jnp.concatenate(
            (
                state.touch_event_impulse.reshape(-1),
                zero_ball_scalars,
                zero_woodwork_scalars,
                zero_foul_scalars,
            )
        ),
        control_tick=jnp.concatenate(
            (
                state.touch_event_control_t.reshape(-1),
                state.ball_event_control_t,
                state.woodwork_control_t,
                jnp.where(foul_valid, state.t, jnp.int32(-1)),
            )
        ),
        substep=jnp.concatenate(
            (
                touch_steps,
                jnp.arange(ball_count, dtype=jnp.int32),
                jnp.arange(woodwork_count, dtype=jnp.int32),
                jnp.full((1,), -1, dtype=jnp.int32),
            )
        ),
        phase=jnp.concatenate(
            (
                touch_phases,
                jnp.full((ball_count,), -1, dtype=jnp.int32),
                jnp.full((woodwork_count,), -1, dtype=jnp.int32),
                jnp.full((1,), -1, dtype=jnp.int32),
            )
        ),
    )


__all__ = [
    "BallBoundaryEventCode",
    "EventBatch",
    "EventDomain",
    "FoulEventCode",
    "TouchEventCode",
    "WoodworkEventCode",
    "events_from_state",
]
