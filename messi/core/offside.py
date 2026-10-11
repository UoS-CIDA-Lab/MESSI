"""고정 shape Law 11 snapshot과 touch/challenge offence transition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.core.constants import NO_PLAYER, NO_TEAM, RK_OFFSIDE, TEAM_0, TEAM_1
from messi.core.contact import (
    LAW11_DEFLECTION_NO_RESET,
    LAW11_DELIBERATE_PLAY_RESET,
    LAW11_DELIBERATE_SAVE_NO_RESET,
    LAW11_DIRECT_RESTART_EXEMPTION,
    OUTCOME_FOUL,
)
from messi.core.numeric import require_float32_representable
from messi.core.restart import select_restart_taker


@dataclass(frozen=True, slots=True)
class OffsideGeometry:
    """Law 11 playable-body front support의 static envelope."""

    shoulder_width_m: float = 0.50
    torso_depth_m: float = 0.20
    head_radius_m: float = 0.10
    foot_forward_extent_m: float = 0.30

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("shoulder_width_m", 0.10, 0.80),
            ("torso_depth_m", 0.05, 0.50),
            ("head_radius_m", 0.05, 0.20),
            ("foot_forward_extent_m", 0.10, 0.50),
        ):
            value = require_float32_representable(name, getattr(self, name))
            if not lower <= value <= upper:
                raise ValueError(f"{name}은 [{lower}, {upper}] 범위여야 한다")


DEFAULT_OFFSIDE_GEOMETRY = OffsideGeometry()


class OffsideEvent(NamedTuple):
    occurred: jax.Array
    actor: jax.Array
    team: jax.Array
    position: jax.Array
    time_fraction: jax.Array


class OffsideResolution(NamedTuple):
    state: object
    event: OffsideEvent


def empty_offside_event(dtype=jnp.float32) -> OffsideEvent:
    return OffsideEvent(
        occurred=jnp.bool_(False),
        actor=jnp.int32(NO_PLAYER),
        team=jnp.int32(NO_TEAM),
        position=jnp.zeros(3, dtype=dtype),
        time_fraction=jnp.asarray(0.0, dtype=dtype),
    )


def detect_passive_offside(
    state,
    roster,
    contact,
    contact_position: jax.Array,
    *,
    time_fraction: jax.Array,
) -> OffsideResolution:
    """Passive deflection의 touch offence와 cheap Law 11 latch 갱신만 계산한다.

    Restart candidate와 nearest taker는 restart-retouch와 공통 적용기가 한 번만
    계산한다. Passive deflection은 새 snapshot을 만들지 않는다.
    """

    count = state.players.position.shape[0]
    actor = jnp.asarray(contact.actor, dtype=jnp.int32)
    valid_actor = (actor >= 0) & (actor < count)
    safe_actor = jnp.clip(actor, 0, count - 1)
    team = roster.team_id[safe_actor].astype(jnp.int32)
    exempt_contact = state.offside_direct_exempt_team == team
    occurred = (
        state.ball.live
        & valid_actor
        & state.players.active[safe_actor]
        & state.offside_flagged[safe_actor]
        & (~exempt_contact)
    )
    position = jnp.asarray(contact_position, dtype=state.ball.position.dtype)
    ordinary = state._replace(
        offside_direct_exempt_team=jnp.where(
            valid_actor & exempt_contact,
            jnp.int32(NO_TEAM),
            state.offside_direct_exempt_team,
        ).astype(jnp.int32)
    )
    return OffsideResolution(
        ordinary,
        OffsideEvent(
            occurred=occurred,
            actor=jnp.where(occurred, actor, NO_PLAYER).astype(jnp.int32),
            team=jnp.where(occurred, team, NO_TEAM).astype(jnp.int32),
            position=jnp.where(occurred, position, jnp.zeros_like(position)),
            time_fraction=jnp.where(occurred, time_fraction, 0.0).astype(
                position.dtype
            ),
        ),
    )


def clear_offside(state):
    """모든 stoppage가 Law 11 snapshot과 직접 restart exemption을 지운다."""

    return state._replace(
        offside_flagged=jnp.zeros_like(state.offside_flagged),
        offside_direct_exempt_team=jnp.int32(NO_TEAM),
    )


def player_front_support(
    state, *, geometry: OffsideGeometry = DEFAULT_OFFSIDE_GEOMETRY
) -> jax.Array:
    """두 공격 방향에서 팔을 제외한 playable-body front 좌표를 계산한다."""

    shoulder_axis_x = -state.players.body_forward[:, 1]
    half_core = 0.5 * (geometry.shoulder_width_m - geometry.torso_depth_m)
    torso_radius = 0.5 * geometry.torso_depth_m
    direction = state.attack_direction[:, None]
    centre = direction * state.players.position[None, :, 0]
    torso = centre + half_core * jnp.abs(shoulder_axis_x)[None, :] + torso_radius
    head = centre + geometry.head_radius_m
    foot = centre + (
        jnp.abs(state.players.body_forward[:, 0])[None, :]
        * geometry.foot_forward_extent_m
    )
    return jnp.maximum(jnp.maximum(torso, head), foot)


def offside_position_snapshot(
    state,
    roster,
    team: jax.Array,
    actor: jax.Array,
    ball_position: jax.Array,
    *,
    ball_radius_m: float = 0.11,
    geometry: OffsideGeometry = DEFAULT_OFFSIDE_GEOMETRY,
) -> jax.Array:
    """한 deliberate team touch 순간 offside-position player를 flag한다."""

    count = state.players.position.shape[0]
    indices = jnp.arange(count, dtype=jnp.int32)
    safe_team = jnp.clip(team, TEAM_0, TEAM_1)
    support = player_front_support(state, geometry=geometry)
    front = support[safe_team]
    opponent = state.players.active & (roster.team_id != safe_team)
    has_second_opponent = jnp.sum(opponent.astype(jnp.int32)) >= 2
    candidates = jnp.where(opponent, front, -jnp.inf)
    first = jnp.argmax(candidates)
    second_last = jnp.max(
        jnp.where(opponent & (indices != first), front, -jnp.inf)
    )
    ball_front = (
        state.attack_direction[safe_team] * ball_position[0] + ball_radius_m
    )
    valid_team = (team == TEAM_0) | (team == TEAM_1)
    return (
        valid_team
        & has_second_opponent
        & state.players.active
        & (roster.team_id == team)
        & (indices != actor)
        & (front > 0.0)
        & (front > ball_front)
        & (front > second_last)
    )


def prospective_offside_mask(
    state,
    roster,
    *,
    ball_position: jax.Array | None = None,
    ball_radius_m: float = 0.11,
    geometry: OffsideGeometry = DEFAULT_OFFSIDE_GEOMETRY,
) -> jax.Array:
    """현재 접촉을 가정했을 때 각 선수가 offside position인지 한 번 계산한다."""

    ball_position = state.ball.position if ball_position is None else ball_position
    support = player_front_support(state, geometry=geometry)
    indices = jnp.arange(state.players.position.shape[0], dtype=jnp.int32)
    team_indices = jnp.asarray((TEAM_0, TEAM_1), dtype=jnp.int32)
    opponent = state.players.active[None, :] & (
        roster.team_id[None, :] != team_indices[:, None]
    )
    has_second_opponent = jnp.sum(opponent.astype(jnp.int32), axis=1) >= 2
    candidates = jnp.where(opponent, support, -jnp.inf)
    first = jnp.argmax(candidates, axis=1)
    second_last = jnp.max(
        jnp.where(opponent & (indices[None, :] != first[:, None]), support, -jnp.inf),
        axis=1,
    )
    safe_player_team = jnp.clip(roster.team_id, TEAM_0, TEAM_1)
    front = support[safe_player_team, indices]
    ball_front = state.attack_direction * ball_position[0] + ball_radius_m
    return (
        state.players.active
        & has_second_opponent[safe_player_team]
        & (front > 0.0)
        & (front > ball_front[safe_player_team])
        & (front > second_last[safe_player_team])
    )


def resolve_offside(
    state,
    roster,
    contact,
    contest,
    field_half_extent: jax.Array,
    *,
    time_fraction: jax.Array = 0.0,
    ball_radius_m: float = 0.11,
    geometry: OffsideGeometry = DEFAULT_OFFSIDE_GEOMETRY,
) -> OffsideResolution:
    """한 realized contact에서 snapshot 갱신 또는 offside restart를 적용한다."""

    count = state.players.position.shape[0]
    contact_actor = jnp.asarray(contact.actor, dtype=jnp.int32)
    valid_contact = (contact_actor >= 0) & (contact_actor < count)
    safe_contact = jnp.clip(contact_actor, 0, count - 1)
    contact_team = roster.team_id[safe_contact].astype(jnp.int32)
    direct_exempt = contact.law11_effect == LAW11_DIRECT_RESTART_EXEMPTION
    exempt_contact = state.offside_direct_exempt_team == contact_team
    touch_offence = (
        state.ball.live
        & valid_contact
        & state.players.active[safe_contact]
        & state.offside_flagged[safe_contact]
        & (~exempt_contact)
        & (~direct_exempt)
    )

    challenger = jnp.asarray(contest.challenger, dtype=jnp.int32)
    valid_challenger = (challenger >= 0) & (challenger < count)
    safe_challenger = jnp.clip(challenger, 0, count - 1)
    challenger_team = roster.team_id[safe_challenger].astype(jnp.int32)
    challenge_offence = (
        state.ball.live
        & contest.selected
        & (contest.outcome != OUTCOME_FOUL)
        & valid_challenger
        & state.players.active[safe_challenger]
        & state.offside_flagged[safe_challenger]
        & (state.offside_direct_exempt_team != challenger_team)
    )
    offence = challenge_offence | touch_offence
    actor = jnp.where(challenge_offence, challenger, contact_actor).astype(jnp.int32)
    safe_actor = jnp.clip(actor, 0, count - 1)
    actor_team = roster.team_id[safe_actor].astype(jnp.int32)
    restart_team = (TEAM_1 - actor_team).astype(jnp.int32)
    position = jnp.concatenate(
        (
            state.players.position[safe_actor],
            jnp.asarray((ball_radius_m,), dtype=state.ball.position.dtype),
        )
    )
    half_extent = jnp.asarray(field_half_extent, dtype=position.dtype)
    restart_spot = position.at[:2].set(
        jnp.clip(position[:2], -half_extent, half_extent)
    )
    restart_candidates = state.players.active & (roster.team_id == restart_team)
    taker = select_restart_taker(
        state,
        roster,
        jnp.int32(RK_OFFSIDE),
        restart_team,
        restart_spot,
        field_half_extent=field_half_extent,
    )
    coherent_offence = offence & jnp.any(restart_candidates)

    snapshot = offside_position_snapshot(
        state,
        roster,
        contact_team,
        contact_actor,
        state.ball.position,
        ball_radius_m=ball_radius_m,
        geometry=geometry,
    )
    deliberate = contact.law11_effect == LAW11_DELIBERATE_PLAY_RESET
    no_reset = (contact.law11_effect == LAW11_DEFLECTION_NO_RESET) | (
        contact.law11_effect == LAW11_DELIBERATE_SAVE_NO_RESET
    )
    next_flagged = jnp.where(
        direct_exempt,
        jnp.zeros_like(state.offside_flagged),
        jnp.where(deliberate, snapshot, state.offside_flagged),
    )
    next_exempt = jnp.where(
        direct_exempt,
        contact_team,
        jnp.where(deliberate | (no_reset & exempt_contact), NO_TEAM, state.offside_direct_exempt_team),
    ).astype(jnp.int32)
    update_snapshot = valid_contact & (~coherent_offence)
    ordinary = state._replace(
        offside_flagged=jnp.where(
            update_snapshot, next_flagged, state.offside_flagged
        ),
        offside_direct_exempt_team=jnp.where(
            update_snapshot, next_exempt, state.offside_direct_exempt_team
        ),
    )
    stopped = ordinary._replace(
        ball=ordinary.ball._replace(
            position=restart_spot,
            velocity=jnp.zeros_like(ordinary.ball.velocity),
            spin=jnp.zeros_like(ordinary.ball.spin),
            live=jnp.bool_(False),
        ),
        possession=ordinary.possession._replace(
            team=jnp.int32(NO_TEAM),
            player=jnp.int32(NO_PLAYER),
            previous_team=ordinary.possession.team,
            control_ticks=jnp.int32(0),
        ),
        restart=ordinary.restart._replace(
            kind=jnp.int32(RK_OFFSIDE),
            team=restart_team,
            substeps_remaining=jnp.int32(0),
            taker=taker,
            indirect=jnp.bool_(True),
            opened_control_tick=ordinary.control_tick,
        ),
        restart_release=ordinary.restart_release._replace(
            active=jnp.bool_(False), untouched=jnp.bool_(False)
        ),
        offside_flagged=jnp.zeros_like(ordinary.offside_flagged),
        offside_direct_exempt_team=jnp.int32(NO_TEAM),
        gk_backpass_team=jnp.int32(NO_TEAM),
        penalty_completion_active=jnp.bool_(False),
        penalty_completion_team=jnp.int32(NO_TEAM),
        restart_layout_ready=jnp.bool_(False),
        restart_taker_approach=jnp.bool_(False),
    )
    select = lambda new, old: jnp.where(coherent_offence, new, old)
    next_state = ordinary._replace(
        ball=jax.tree.map(select, stopped.ball, ordinary.ball),
        possession=jax.tree.map(select, stopped.possession, ordinary.possession),
        restart=jax.tree.map(select, stopped.restart, ordinary.restart),
        restart_release=jax.tree.map(
            select, stopped.restart_release, ordinary.restart_release
        ),
        offside_flagged=select(stopped.offside_flagged, ordinary.offside_flagged),
        offside_direct_exempt_team=select(
            stopped.offside_direct_exempt_team,
            ordinary.offside_direct_exempt_team,
        ),
        gk_backpass_team=select(
            stopped.gk_backpass_team, ordinary.gk_backpass_team
        ),
        penalty_completion_active=select(
            stopped.penalty_completion_active,
            ordinary.penalty_completion_active,
        ),
        penalty_completion_team=select(
            stopped.penalty_completion_team,
            ordinary.penalty_completion_team,
        ),
        restart_layout_ready=select(
            stopped.restart_layout_ready, ordinary.restart_layout_ready
        ),
        restart_taker_approach=select(
            stopped.restart_taker_approach, ordinary.restart_taker_approach
        ),
    )
    event = OffsideEvent(
        occurred=coherent_offence,
        actor=jnp.where(coherent_offence, actor, NO_PLAYER).astype(jnp.int32),
        team=jnp.where(coherent_offence, actor_team, NO_TEAM).astype(jnp.int32),
        position=jnp.where(
            coherent_offence, restart_spot, jnp.zeros_like(restart_spot)
        ),
        time_fraction=jnp.where(coherent_offence, time_fraction, 0.0).astype(
            restart_spot.dtype
        ),
    )
    return OffsideResolution(next_state, event)


__all__ = [
    "DEFAULT_OFFSIDE_GEOMETRY",
    "OffsideEvent",
    "OffsideGeometry",
    "OffsideResolution",
    "clear_offside",
    "detect_passive_offside",
    "empty_offside_event",
    "offside_position_snapshot",
    "player_front_support",
    "prospective_offside_mask",
    "resolve_offside",
]
