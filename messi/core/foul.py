"""이미 확정된 contest foul을 Law 12 restart로 변환한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.core.constants import (
    NO_PLAYER,
    NO_TEAM,
    RK_FREEKICK,
    RK_NONE,
    RK_PENALTY,
    TEAM_0,
    TEAM_1,
)
from messi.core.contact import OUTCOME_FOUL
from messi.core.numeric import require_float32_representable


@dataclass(frozen=True, slots=True)
class FoulLaw:
    """Direct-free-kick foul의 penalty-area 및 penalty-mark 치수."""

    penalty_area_length_m: float = 16.5
    penalty_area_half_width_m: float = 20.16
    penalty_mark_distance_m: float = 11.0

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("penalty_area_length_m", 10.0, 25.0),
            ("penalty_area_half_width_m", 10.0, 30.0),
            ("penalty_mark_distance_m", 8.0, 14.0),
        ):
            value = require_float32_representable(name, getattr(self, name))
            if not lower <= value <= upper:
                raise ValueError(f"{name}은 [{lower}, {upper}] 범위여야 한다")


DEFAULT_FOUL_LAW = FoulLaw()


class FoulEvent(NamedTuple):
    """보고서와 replay 검증에 필요한 compact foul 사실."""

    occurred: jax.Array
    offender: jax.Array
    victim: jax.Array
    position: jax.Array
    restart_kind: jax.Array
    restart_team: jax.Array
    discipline: jax.Array
    severity: jax.Array
    tactical_effect: jax.Array
    time_fraction: jax.Array


class FoulResolution(NamedTuple):
    state: object
    event: FoulEvent


def empty_foul_event(dtype=jnp.float32) -> FoulEvent:
    return FoulEvent(
        occurred=jnp.bool_(False),
        offender=jnp.int32(NO_PLAYER),
        victim=jnp.int32(NO_PLAYER),
        position=jnp.zeros(3, dtype=dtype),
        restart_kind=jnp.int32(RK_NONE),
        restart_team=jnp.int32(NO_TEAM),
        discipline=jnp.int32(0),
        severity=jnp.int32(0),
        tactical_effect=jnp.int32(0),
        time_fraction=jnp.asarray(0.0, dtype=dtype),
    )


def resolve_contest_foul(
    state,
    roster,
    contest,
    field_half_extent: jax.Array,
    *,
    time_fraction: jax.Array = 0.0,
    ball_radius_m: float = 0.11,
    law: FoulLaw = DEFAULT_FOUL_LAW,
) -> FoulResolution:
    """확정된 direct-contact foul을 free-kick 또는 penalty로 원자적으로 적용한다."""

    player_count = state.players.position.shape[0]
    offender = jnp.asarray(contest.foul_actor, dtype=jnp.int32)
    victim = jnp.asarray(contest.foul_victim, dtype=jnp.int32)
    valid_indices = (
        (offender >= 0)
        & (offender < player_count)
        & (victim >= 0)
        & (victim < player_count)
    )
    safe_offender = jnp.clip(offender, 0, player_count - 1)
    safe_victim = jnp.clip(victim, 0, player_count - 1)
    different_teams = roster.team_id[safe_offender] != roster.team_id[safe_victim]
    occurred = (
        contest.selected
        & (contest.outcome == OUTCOME_FOUL)
        & valid_indices
        & different_teams
    )
    restart_team = roster.team_id[safe_victim].astype(jnp.int32)
    offender_team = roster.team_id[safe_offender].astype(jnp.int32)
    offence_xy = jnp.asarray(
        contest.offence_position, dtype=state.ball.position.dtype
    )[:2]
    position = jnp.concatenate(
        (offence_xy, jnp.zeros((1,), dtype=state.ball.position.dtype))
    )
    half_extent = jnp.asarray(field_half_extent, dtype=position.dtype)
    safe_offender_team = jnp.clip(offender_team, TEAM_0, TEAM_1)
    own_goal_direction = -state.attack_direction[safe_offender_team]
    depth = half_extent[0] - own_goal_direction * position[0]
    inside_penalty = (
        (depth >= 0.0)
        & (depth <= law.penalty_area_length_m)
        & (jnp.abs(position[1]) <= law.penalty_area_half_width_m)
    )
    penalty = occurred & inside_penalty
    restart_kind = jnp.where(penalty, RK_PENALTY, RK_FREEKICK).astype(jnp.int32)
    clipped_spot = jnp.asarray(
        (
            jnp.clip(position[0], -half_extent[0], half_extent[0]),
            jnp.clip(position[1], -half_extent[1], half_extent[1]),
            ball_radius_m,
        ),
        dtype=position.dtype,
    )
    penalty_spot = jnp.asarray(
        (
            state.attack_direction[jnp.clip(restart_team, TEAM_0, TEAM_1)]
            * (half_extent[0] - law.penalty_mark_distance_m),
            0.0,
            ball_radius_m,
        ),
        dtype=position.dtype,
    )
    restart_spot = jnp.where(penalty, penalty_spot, clipped_spot)
    # Contest victim은 이미 active carrier로 검증됐다. 피해자를 taker로 유지하면
    # 별도 player-axis argmin 없이 causal identity도 보존된다.
    taker = victim
    coherent = occurred & state.players.active[safe_victim]
    candidate = state._replace(
        ball=state.ball._replace(
            position=restart_spot,
            velocity=jnp.zeros_like(state.ball.velocity),
            spin=jnp.zeros_like(state.ball.spin),
            live=jnp.bool_(False),
        ),
        possession=state.possession._replace(
            team=jnp.int32(NO_TEAM),
            player=jnp.int32(NO_PLAYER),
            previous_team=state.possession.team,
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
        restart_release=state.restart_release._replace(
            active=jnp.bool_(False), untouched=jnp.bool_(False)
        ),
        offside_flagged=jnp.zeros_like(state.offside_flagged),
        offside_direct_exempt_team=jnp.int32(NO_TEAM),
        gk_backpass_team=jnp.int32(NO_TEAM),
        penalty_completion_active=jnp.bool_(False),
        penalty_completion_team=jnp.int32(NO_TEAM),
        restart_layout_ready=jnp.bool_(False),
        restart_taker_approach=jnp.bool_(False),
    )
    def select(new, old):
        return jnp.where(coherent, new, old)
    next_state = state._replace(
        ball=jax.tree.map(select, candidate.ball, state.ball),
        possession=jax.tree.map(select, candidate.possession, state.possession),
        restart=jax.tree.map(select, candidate.restart, state.restart),
        restart_release=jax.tree.map(
            select, candidate.restart_release, state.restart_release
        ),
        offside_flagged=select(candidate.offside_flagged, state.offside_flagged),
        offside_direct_exempt_team=select(
            candidate.offside_direct_exempt_team,
            state.offside_direct_exempt_team,
        ),
        gk_backpass_team=select(
            candidate.gk_backpass_team, state.gk_backpass_team
        ),
        penalty_completion_active=select(
            candidate.penalty_completion_active,
            state.penalty_completion_active,
        ),
        penalty_completion_team=select(
            candidate.penalty_completion_team,
            state.penalty_completion_team,
        ),
        restart_layout_ready=select(
            candidate.restart_layout_ready, state.restart_layout_ready
        ),
        restart_taker_approach=select(
            candidate.restart_taker_approach, state.restart_taker_approach
        ),
    )
    event = FoulEvent(
        occurred=coherent,
        offender=jnp.where(coherent, offender, NO_PLAYER).astype(jnp.int32),
        victim=jnp.where(coherent, victim, NO_PLAYER).astype(jnp.int32),
        position=jnp.where(coherent, position, jnp.zeros_like(position)),
        restart_kind=jnp.where(coherent, restart_kind, RK_NONE).astype(jnp.int32),
        restart_team=jnp.where(coherent, restart_team, NO_TEAM).astype(jnp.int32),
        discipline=jnp.where(coherent, contest.discipline, 0).astype(jnp.int32),
        severity=jnp.where(coherent, 1, 0).astype(jnp.int32),
        tactical_effect=jnp.int32(0),
        time_fraction=jnp.where(coherent, time_fraction, 0.0).astype(position.dtype),
    )
    return FoulResolution(next_state, event)


__all__ = [
    "DEFAULT_FOUL_LAW",
    "FoulEvent",
    "FoulLaw",
    "FoulResolution",
    "empty_foul_event",
    "resolve_contest_foul",
]
