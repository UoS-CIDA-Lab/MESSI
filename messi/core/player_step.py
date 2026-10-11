"""Locomotion과 원형 body collision을 결합한 physics substep."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.core.collision import PlayerImpactFacts, separate_players
from messi.core.collision_config import BodyCollision, CollisionPhysics
from messi.core.constants import MAX_SUPPORTED_PLAYER_SPEED_MPS
from messi.core.movement import PlayerPhysics, step_locomotion
from messi.core.stamina import LongStamina, ShortStamina
from messi.core.state import PlayerState, RosterState


class PlayerStep(NamedTuple):
    """갱신된 선수 상태와 substep 내 가장 강한 상대 team 충돌."""

    players: PlayerState
    impact: PlayerImpactFacts


def _stronger_impact(
    previous: PlayerImpactFacts,
    current: PlayerImpactFacts,
) -> PlayerImpactFacts:
    """동일 outer substep의 earliest strongest body impact를 유지한다."""

    select = current.occurred & (
        (~previous.occurred) | (current.impact_score > previous.impact_score)
    )
    return jax.tree.map(lambda new, old: jnp.where(select, new, old), current, previous)


def step_players(
    players: PlayerState,
    roster: RosterState,
    desired_velocity: jax.Array,
    movement_enabled: jax.Array,
    attack_direction: jax.Array,
    field_half_extent: jax.Array,
    *,
    dt: float,
    boundary_margin_m: float = 0.0,
    pinned: jax.Array | None = None,
    player_physics: PlayerPhysics = PlayerPhysics(),
    body: BodyCollision = BodyCollision(),
    collision_physics: CollisionPhysics = CollisionPhysics(),
    long_stamina: LongStamina = LongStamina(),
    short_stamina: ShortStamina = ShortStamina(),
) -> PlayerStep:
    """한 locomotion 뒤 body impulse와 separation을 한 번 적용한다.

    공개 roster 속도 상한과 80 Hz 기본 timebase에서 최대 상대 이동은 body diameter보다
    작으므로 별도 collision microstep 없이 지원 범위의 tunnelling을 방지한다.
    """

    # 두 선수가 지원 속도 상한으로 정면 접근해도 한 collision envelope를 한 번에
    # 완전히 건너뛰지 않게 한다. 기본 80 Hz에서는 1이므로 production graph는 기존과
    # 동일하고, 허용된 coarse Timebase에서만 static microstep variant가 생긴다.
    relative_travel_m = 2.0 * MAX_SUPPORTED_PLAYER_SPEED_MPS * float(dt)
    microsteps = max(1, math.ceil(relative_travel_m / body.effective_diameter_m))
    micro_dt = float(dt) / microsteps

    def step_once(current_players: PlayerState):
        moved = step_locomotion(
            current_players,
            roster,
            desired_velocity,
            movement_enabled,
            field_half_extent,
            dt=micro_dt,
            boundary_margin_m=boundary_margin_m,
            physics=player_physics,
            long_stamina=long_stamina,
            short_stamina=short_stamina,
        )
        return separate_players(
            moved,
            roster.team_id,
            attack_direction,
            field_half_extent,
            boundary_margin_m=boundary_margin_m,
            pinned=pinned,
            body=body,
            physics=collision_physics,
        )

    first = step_once(players)
    if microsteps == 1:
        return PlayerStep(players=first.players, impact=first.impact)

    first_impact = first.impact._replace(
        time_fraction=jnp.where(
            first.impact.occurred,
            jnp.asarray(1.0 / microsteps, dtype=players.position.dtype),
            first.impact.time_fraction,
        )
    )

    def advance(carry, microstep_index):
        current_players, strongest = carry
        separated = step_once(current_players)
        impact = separated.impact._replace(
            time_fraction=jnp.where(
                separated.impact.occurred,
                (microstep_index + 1).astype(players.position.dtype) / microsteps,
                separated.impact.time_fraction,
            )
        )
        return (
            separated.players,
            _stronger_impact(strongest, impact),
        ), None

    (final_players, strongest), _ = jax.lax.scan(
        advance,
        (first.players, first_impact),
        jnp.arange(1, microsteps, dtype=jnp.int32),
    )
    return PlayerStep(players=final_players, impact=strongest)


__all__ = ["PlayerStep", "step_players"]
