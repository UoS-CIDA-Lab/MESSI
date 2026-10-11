"""Free-ball segment와 선수 torso/head/leg의 bounded passive contact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.core.action import INTENT_MOVE
from messi.core.constants import NO_PLAYER, NO_TEAM, RK_NONE, RK_THROWIN
from messi.core.contact import (
    INTENT_SOURCE_NONE,
    LAW11_DEFLECTION_NO_RESET,
    MECHANISM_HEAD,
    MECHANISM_NONE,
    MECHANISM_PASSIVE_BODY,
    OUTCOME_DEFLECTION,
)
from messi.core.numeric import require_float32_representable


@dataclass(frozen=True, slots=True)
class PassiveContactPhysics:
    torso_radius_m: float = 0.16
    torso_top_height_factor: float = 0.85
    head_radius_m: float = 0.10
    shoulder_width_m: float = 0.50
    leg_apex_height_factor: float = 0.50
    leg_radius_m: float = 0.05
    restitution: float = 0.42
    tangential_retention: float = 0.78

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("torso_radius_m", 0.08, 0.35),
            ("torso_top_height_factor", 0.60, 0.95),
            ("head_radius_m", 0.05, 0.20),
            ("shoulder_width_m", 0.30, 0.80),
            ("leg_apex_height_factor", 0.35, 0.65),
            ("leg_radius_m", 0.03, 0.12),
            ("restitution", 0.0, 1.0),
            ("tangential_retention", 0.0, 1.0),
        ):
            value = require_float32_representable(name, getattr(self, name))
            if not lower <= value <= upper:
                raise ValueError(f"{name}은 [{lower}, {upper}] 범위여야 한다")
        if 2.0 * self.leg_radius_m >= self.shoulder_width_m:
            raise ValueError("두 leg radius 합은 shoulder_width_m보다 작아야 한다")


DEFAULT_PASSIVE_CONTACT_PHYSICS = PassiveContactPhysics()


class PassiveContactEvent(NamedTuple):
    occurred: jax.Array
    actor: jax.Array
    mechanism: jax.Array
    time_fraction: jax.Array
    position: jax.Array
    normal: jax.Array


class NutmegEvent(NamedTuple):
    """의도적 공격 접촉 뒤 상대 선수의 열린 leg gap을 통과한 관측 event."""

    occurred: jax.Array
    attacker: jax.Array
    defender: jax.Array
    time_fraction: jax.Array
    position: jax.Array


def empty_passive_contact_event(dtype=jnp.float32) -> PassiveContactEvent:
    return PassiveContactEvent(
        occurred=jnp.bool_(False),
        actor=jnp.int32(NO_PLAYER),
        mechanism=jnp.int32(MECHANISM_NONE),
        time_fraction=jnp.asarray(0.0, dtype=dtype),
        position=jnp.zeros(3, dtype=dtype),
        normal=jnp.zeros(3, dtype=dtype),
    )


def empty_nutmeg_event(dtype=jnp.float32) -> NutmegEvent:
    return NutmegEvent(
        occurred=jnp.bool_(False),
        attacker=jnp.int32(NO_PLAYER),
        defender=jnp.int32(NO_PLAYER),
        time_fraction=jnp.asarray(0.0, dtype=dtype),
        position=jnp.zeros(3, dtype=dtype),
    )


def _segment_sphere_entry(origin, delta, centres, radius):
    offset = origin[None, :] - centres
    a = jnp.sum(delta * delta)
    b = 2.0 * jnp.sum(offset * delta[None, :], axis=-1)
    c = jnp.sum(offset * offset, axis=-1) - radius * radius
    discriminant = b * b - 4.0 * a * c
    moving = a > 1.0e-12
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    entry = (-b - root) / jnp.maximum(2.0 * a, 1.0e-12)
    inside = c <= 0.0
    fraction = jnp.where(inside, 0.0, entry)
    approaching = jnp.sum(offset * delta[None, :], axis=-1) < 0.0
    hit = (
        moving
        & ((inside & approaching) | ((~inside) & (discriminant >= 0.0)))
        & (fraction >= 0.0)
        & (fraction <= 1.0)
    )
    offset_at = offset + fraction[:, None] * delta[None, :]
    return hit, jnp.clip(fraction, 0.0, 1.0), offset_at


def _relative_segment_sphere_entry(relative_origin, relative_delta, radius):
    """Player별 상대 좌표에서 swept sphere entry를 계산한다."""

    a = jnp.sum(relative_delta * relative_delta, axis=-1)
    b = 2.0 * jnp.sum(relative_origin * relative_delta, axis=-1)
    c = jnp.sum(relative_origin * relative_origin, axis=-1) - radius * radius
    discriminant = b * b - 4.0 * a * c
    moving = a > 1.0e-12
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    entry = (-b - root) / jnp.maximum(2.0 * a, 1.0e-12)
    inside = c <= 0.0
    fraction = jnp.where(inside, 0.0, entry)
    approaching = jnp.sum(relative_origin * relative_delta, axis=-1) < 0.0
    hit = (
        moving
        & ((inside & approaching) | ((~inside) & (discriminant >= 0.0)))
        & (fraction >= 0.0)
        & (fraction <= 1.0)
    )
    offset_at = relative_origin + fraction[:, None] * relative_delta
    return hit, jnp.clip(fraction, 0.0, 1.0), offset_at


def _segment_capsule_entry(origin, delta, core_start, core_end, radius):
    """3D capsule에 대한 swept point의 earliest entry와 normal을 계산한다.

    ``radius``에 공 반지름을 더해 별도 시간 반복 없이 swept-sphere 충돌로 사용한다.
    FootballWorld의 fixed isosceles leg silhouette와 같은 기하다.
    """

    axis = core_end - core_start
    axis_squared = jnp.sum(axis * axis, axis=-1)
    safe_axis_squared = jnp.maximum(axis_squared, 1.0e-12)
    relative_origin = origin - core_start
    origin_axis = jnp.sum(relative_origin * axis, axis=-1)
    delta_axis = jnp.sum(delta * axis, axis=-1)
    perpendicular_origin = (
        relative_origin - (origin_axis / safe_axis_squared)[..., None] * axis
    )
    perpendicular_delta = delta - (
        delta_axis / safe_axis_squared
    )[..., None] * axis

    quadratic = jnp.sum(perpendicular_delta * perpendicular_delta, axis=-1)
    linear = 2.0 * jnp.sum(perpendicular_origin * perpendicular_delta, axis=-1)
    constant = jnp.sum(perpendicular_origin * perpendicular_origin, axis=-1) - (
        radius * radius
    )
    discriminant = linear * linear - 4.0 * quadratic * constant
    moving = quadratic > 1.0e-12
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    denominator = jnp.where(moving, 2.0 * quadratic, 1.0)
    cylinder_entry_raw = (-linear - root) / denominator
    cylinder_exit_raw = (-linear + root) / denominator
    inside = (
        (constant <= 0.0) & (origin_axis >= 0.0) & (origin_axis <= axis_squared)
    )
    cylinder_entry = jnp.where(inside, 0.0, cylinder_entry_raw)
    cylinder_axis = origin_axis + cylinder_entry * delta_axis
    cylinder_valid = inside | (
        moving
        & (discriminant >= 0.0)
        & (cylinder_exit_raw >= 0.0)
        & (cylinder_entry_raw <= 1.0)
        & (cylinder_axis >= 0.0)
        & (cylinder_axis <= axis_squared)
    )
    cylinder_entry = jnp.clip(cylinder_entry, 0.0, 1.0)

    def sphere_entry(centre):
        offset = origin - centre
        quadratic = jnp.sum(delta * delta, axis=-1)
        linear = 2.0 * jnp.sum(offset * delta, axis=-1)
        constant = jnp.sum(offset * offset, axis=-1) - radius * radius
        discriminant = linear * linear - 4.0 * quadratic * constant
        moving = quadratic > 1.0e-12
        root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        denominator = jnp.where(moving, 2.0 * quadratic, 1.0)
        raw_entry = (-linear - root) / denominator
        raw_exit = (-linear + root) / denominator
        inside_sphere = constant <= 0.0
        sphere_valid = jnp.where(
            moving,
            (discriminant >= 0.0) & (raw_exit >= 0.0) & (raw_entry <= 1.0),
            inside_sphere,
        )
        sphere_entry_value = jnp.where(moving, raw_entry, 0.0)
        sphere_entry_value = jnp.where(
            inside_sphere, 0.0, sphere_entry_value
        )
        sphere_entry_value = jnp.clip(sphere_entry_value, 0.0, 1.0)
        sphere_offset = offset + sphere_entry_value[..., None] * delta
        return sphere_valid, sphere_entry_value, sphere_offset

    start_valid, start_entry, start_offset = sphere_entry(core_start)
    end_valid, end_entry, end_offset = sphere_entry(core_end)
    valid_parts = jnp.stack((cylinder_valid, start_valid, end_valid), axis=-1)
    entries = jnp.stack((cylinder_entry, start_entry, end_entry), axis=-1)
    winner = jnp.argmin(jnp.where(valid_parts, entries, jnp.inf), axis=-1)
    valid = jnp.any(valid_parts, axis=-1)
    entry = jnp.take_along_axis(entries, winner[..., None], axis=-1)[..., 0]

    point = origin + entry[..., None] * delta
    core_fraction = jnp.clip(
        jnp.sum((point - core_start) * axis, axis=-1) / safe_axis_squared,
        0.0,
        1.0,
    )
    cylinder_normal = point - (core_start + core_fraction[..., None] * axis)
    cap_normals = jnp.stack((start_offset, end_offset), axis=-2)
    cap_winner = jnp.clip(winner - 1, 0, 1)
    cap_normal = jnp.take_along_axis(
        cap_normals, cap_winner[..., None, None], axis=-2
    )[..., 0, :]
    normal = jnp.where((winner == 0)[..., None], cylinder_normal, cap_normal)
    return valid, entry, normal


def _leg_collision(
    position,
    path_delta,
    players,
    roster,
    ball_radius_m,
    physics,
    *,
    player_start_position=None,
    player_path_delta=None,
):
    """선수 facing frame의 고정 삼각 stance 양쪽 leg capsule 중 earliest hit."""

    count = players.position.shape[0]
    forward = players.body_forward
    shoulder = jnp.stack((-forward[:, 1], forward[:, 0]), axis=-1)
    player_start_position = (
        players.position
        if player_start_position is None
        else jnp.asarray(player_start_position, dtype=position.dtype)
    )
    player_path_delta = (
        jnp.zeros_like(players.position)
        if player_path_delta is None
        else jnp.asarray(player_path_delta, dtype=position.dtype)
    )
    relative_xy = position[:2] - player_start_position
    relative_path_xy = path_delta[:2] - player_path_delta
    local_origin_xy = jnp.stack(
        (
            jnp.sum(relative_xy * shoulder, axis=-1),
            jnp.sum(relative_xy * forward, axis=-1),
        ),
        axis=-1,
    )
    local_delta_xy = jnp.stack(
        (
            jnp.sum(relative_path_xy * shoulder, axis=-1),
            jnp.sum(relative_path_xy * forward, axis=-1),
        ),
        axis=-1,
    )
    local_origin = jnp.concatenate(
        (local_origin_xy, jnp.full((count, 1), position[2], dtype=position.dtype)),
        axis=-1,
    )
    local_delta = jnp.concatenate(
        (
            local_delta_xy,
            jnp.full((count, 1), path_delta[2], dtype=position.dtype),
        ),
        axis=-1,
    )
    lateral = jnp.asarray(
        (-0.5 * physics.shoulder_width_m, 0.5 * physics.shoulder_width_m),
        dtype=position.dtype,
    )
    foot = jnp.zeros((count, 2, 3), dtype=position.dtype)
    foot = foot.at[:, :, 0].set(lateral[None, :])
    apex = jnp.zeros_like(foot)
    apex = apex.at[:, :, 2].set(
        (roster.height * physics.leg_apex_height_factor)[:, None]
    )
    valid, entry, local_normal = _segment_capsule_entry(
        local_origin[:, None, :],
        local_delta[:, None, :],
        apex,
        foot,
        physics.leg_radius_m + ball_radius_m,
    )
    first = jnp.argmin(jnp.where(valid, entry, jnp.inf), axis=-1)
    selected_entry = jnp.take_along_axis(entry, first[:, None], axis=-1)[:, 0]
    selected_local_normal = jnp.take_along_axis(
        local_normal, first[:, None, None], axis=1
    )[:, 0, :]
    horizontal = (
        selected_local_normal[:, 0, None] * shoulder
        + selected_local_normal[:, 1, None] * forward
    )
    normal = jnp.concatenate((horizontal, selected_local_normal[:, 2, None]), axis=-1)
    relative_path = jnp.concatenate(
        (relative_path_xy, jnp.full((count, 1), path_delta[2], dtype=position.dtype)),
        axis=-1,
    )
    approaching = jnp.sum(relative_path * normal, axis=-1) < -1.0e-9
    return jnp.any(valid, axis=-1) & approaching, selected_entry, normal


def detect_between_legs_passage(
    position,
    path_delta,
    ball_live,
    players,
    roster,
    *,
    eligible_mask: jax.Array | None = None,
    ball_radius_m: float = 0.11,
    physics: PassiveContactPhysics = DEFAULT_PASSIVE_CONTACT_PHYSICS,
) -> PassiveContactEvent:
    """선수 facing frame의 leg gap 중앙면을 통과한 가장 이른 선수를 찾는다.

    양쪽 leg capsule과 충돌한 경로는 nutmeg 후보에서 제외한다. 반환형은 내부
    geometry 재사용을 위해 ``PassiveContactEvent``이며 mechanism과 normal은 비운다.
    """

    count = players.position.shape[0]
    indices = jnp.arange(count, dtype=jnp.int32)
    forward = players.body_forward
    shoulder = jnp.stack((-forward[:, 1], forward[:, 0]), axis=-1)
    relative_xy = position[:2] - players.position
    start_lateral = jnp.sum(relative_xy * shoulder, axis=-1)
    start_forward = jnp.sum(relative_xy * forward, axis=-1)
    delta_lateral = jnp.sum(path_delta[:2] * shoulder, axis=-1)
    delta_forward = jnp.sum(path_delta[:2] * forward, axis=-1)
    crossing = (start_forward * (start_forward + delta_forward) <= 0.0) & (
        jnp.abs(delta_forward) > 1.0e-9
    )
    fraction = jnp.clip(
        -start_forward / jnp.where(crossing, delta_forward, 1.0), 0.0, 1.0
    )
    passage_lateral = start_lateral + fraction * delta_lateral
    passage_height = position[2] + fraction * path_delta[2]
    apex_height = roster.height * physics.leg_apex_height_factor
    half_gap = 0.5 * physics.shoulder_width_m * (
        1.0 - passage_height / jnp.maximum(apex_height, 1.0e-9)
    )
    leg_hit, _, _ = _leg_collision(
        position, path_delta, players, roster, ball_radius_m, physics
    )
    eligible = players.active & jnp.asarray(ball_live)
    if eligible_mask is not None:
        eligible &= jnp.asarray(eligible_mask, dtype=jnp.bool_)
    valid = (
        eligible
        & crossing
        & (passage_height >= ball_radius_m)
        & (passage_height <= apex_height)
        & (jnp.abs(passage_lateral) + ball_radius_m < half_gap)
        & (~leg_hit)
    )
    winner = jnp.argmin(jnp.where(valid, fraction, jnp.inf))
    occurred = jnp.any(valid)
    passage_position = position + fraction[winner] * path_delta
    return PassiveContactEvent(
        occurred=occurred,
        actor=jnp.where(occurred, indices[winner], NO_PLAYER).astype(jnp.int32),
        mechanism=jnp.int32(MECHANISM_NONE),
        time_fraction=jnp.where(occurred, fraction[winner], 0.0),
        position=jnp.where(occurred, passage_position, jnp.zeros_like(position)),
        normal=jnp.zeros_like(position),
    )


def detect_nutmeg(
    position,
    path_delta,
    ball_live,
    players,
    roster,
    *,
    attacker,
    deliberate,
    uninterrupted,
    ball_radius_m: float = 0.11,
    physics: PassiveContactPhysics = DEFAULT_PASSIVE_CONTACT_PHYSICS,
) -> NutmegEvent:
    """의도적 마지막 접촉의 상대 선수에 한정해 nutmeg telemetry를 판정한다."""

    count = players.position.shape[0]
    valid_attacker = (attacker >= 0) & (attacker < count)
    safe_attacker = jnp.clip(attacker, 0, count - 1)
    defender_mask = (
        players.active
        & (roster.team_id != roster.team_id[safe_attacker])
        & (jnp.arange(count, dtype=jnp.int32) != safe_attacker)
    )
    passage = detect_between_legs_passage(
        position,
        path_delta,
        ball_live & valid_attacker & deliberate & uninterrupted,
        players,
        roster,
        eligible_mask=defender_mask,
        ball_radius_m=ball_radius_m,
        physics=physics,
    )
    return NutmegEvent(
        occurred=passage.occurred,
        attacker=jnp.where(passage.occurred, attacker, NO_PLAYER).astype(jnp.int32),
        defender=passage.actor,
        time_fraction=passage.time_fraction,
        position=passage.position,
    )


def detect_passive_contact(
    position,
    path_delta,
    ball_live,
    players,
    roster,
    *,
    excluded_actor: jax.Array = NO_PLAYER,
    excluded_actor_mask: jax.Array | None = None,
    player_start_position: jax.Array | None = None,
    player_path_delta: jax.Array | None = None,
    ball_radius_m: float = 0.11,
    physics: PassiveContactPhysics = DEFAULT_PASSIVE_CONTACT_PHYSICS,
) -> PassiveContactEvent:
    """한 free-ball segment에서 가장 이른 torso/head/leg contact를 검출한다."""

    count = players.position.shape[0]
    indices = jnp.arange(count, dtype=jnp.int32)
    eligible = players.active & (indices != excluded_actor) & jnp.asarray(ball_live)
    if excluded_actor_mask is not None:
        eligible &= ~jnp.asarray(excluded_actor_mask, dtype=jnp.bool_)
    player_start_position = (
        players.position
        if player_start_position is None
        else jnp.asarray(player_start_position, dtype=position.dtype)
    )
    player_path_delta = (
        jnp.zeros_like(players.position)
        if player_path_delta is None
        else jnp.asarray(player_path_delta, dtype=position.dtype)
    )
    if player_start_position.shape != players.position.shape:
        raise ValueError("player_start_position shape는 players.position과 같아야 한다")
    if player_path_delta.shape != players.position.shape:
        raise ValueError("player_path_delta shape는 players.position과 같아야 한다")
    relative_xy = position[:2] - player_start_position
    relative_path_xy = path_delta[:2] - player_path_delta
    torso_radius = ball_radius_m + physics.torso_radius_m
    torso_hit, torso_t, torso_offset = _relative_segment_sphere_entry(
        relative_xy, relative_path_xy, torso_radius
    )
    torso_z = position[2] + torso_t * path_delta[2]
    torso_bottom = roster.height * physics.leg_apex_height_factor - ball_radius_m
    torso_hit &= eligible & (torso_z >= torso_bottom) & (
        torso_z <= roster.height * physics.torso_top_height_factor
    )
    torso_normal = jnp.concatenate(
        (torso_offset, jnp.zeros((count, 1), dtype=position.dtype)), axis=-1
    )

    head_relative_origin = jnp.concatenate(
        (relative_xy, (position[2] - roster.height + physics.head_radius_m)[:, None]),
        axis=-1,
    )
    head_relative_delta = jnp.concatenate(
        (
            relative_path_xy,
            jnp.full((count, 1), path_delta[2], dtype=position.dtype),
        ),
        axis=-1,
    )
    head_hit, head_t, head_normal = _relative_segment_sphere_entry(
        head_relative_origin,
        head_relative_delta,
        ball_radius_m + physics.head_radius_m,
    )
    head_hit &= eligible
    leg_hit, leg_t, leg_normal = _leg_collision(
        position,
        path_delta,
        players,
        roster,
        ball_radius_m,
        physics,
        player_start_position=player_start_position,
        player_path_delta=player_path_delta,
    )
    leg_hit &= eligible
    hit = jnp.concatenate((torso_hit, head_hit, leg_hit))
    fraction = jnp.concatenate((torso_t, head_t, leg_t))
    normals = jnp.concatenate((torso_normal, head_normal, leg_normal), axis=0)
    actors = jnp.concatenate((indices, indices, indices))
    mechanisms = jnp.concatenate(
        (
            jnp.full(count, MECHANISM_PASSIVE_BODY, dtype=jnp.int32),
            jnp.full(count, MECHANISM_HEAD, dtype=jnp.int32),
            jnp.full(count, MECHANISM_PASSIVE_BODY, dtype=jnp.int32),
        )
    )
    winner = jnp.argmin(jnp.where(hit, fraction, jnp.inf))
    occurred = jnp.any(hit)
    normal = normals[winner]
    normal = normal / jnp.maximum(jnp.linalg.norm(normal), 1.0e-6)
    contact_position = position + fraction[winner] * path_delta
    return PassiveContactEvent(
        occurred=occurred,
        actor=jnp.where(occurred, actors[winner], NO_PLAYER).astype(jnp.int32),
        mechanism=jnp.where(
            occurred, mechanisms[winner], MECHANISM_NONE
        ).astype(jnp.int32),
        time_fraction=jnp.where(occurred, fraction[winner], 0.0),
        position=jnp.where(
            occurred, contact_position, jnp.zeros_like(contact_position)
        ),
        normal=jnp.where(occurred, normal, jnp.zeros_like(normal)),
    )


def apply_passive_contact(
    ball,
    event: PassiveContactEvent,
    *,
    dt: float,
    advance_remaining: bool = True,
    surface_velocity: jax.Array | None = None,
    physics: PassiveContactPhysics = DEFAULT_PASSIVE_CONTACT_PHYSICS,
):
    """Earliest contact에서 반사한 뒤 substep의 남은 시간을 전개한다."""

    surface_velocity = (
        jnp.zeros_like(ball.velocity)
        if surface_velocity is None
        else jnp.asarray(surface_velocity, dtype=ball.velocity.dtype)
    )
    relative_velocity = ball.velocity - surface_velocity
    normal_speed = jnp.dot(relative_velocity, event.normal)
    normal_component = normal_speed * event.normal
    tangent = relative_velocity - normal_component
    reflected = (
        physics.tangential_retention * tangent
        - physics.restitution * normal_component
    ) + surface_velocity
    remaining = jnp.maximum(1.0 - event.time_fraction, 0.0)
    if advance_remaining:
        position = event.position + 1.0e-4 * event.normal + reflected * dt * remaining
    else:
        position = event.position + 1.0e-4 * event.normal
    return ball._replace(
        position=jnp.where(event.occurred, position, ball.position),
        velocity=jnp.where(event.occurred, reflected, ball.velocity),
        spin=jnp.where(
            event.occurred, physics.tangential_retention * ball.spin, ball.spin
        ),
    )


def apply_passive_contact_provenance(
    state,
    roster,
    actor,
    occurred,
    *,
    possession_loss_lock_substeps=0,
):
    """마지막으로 적용된 passive contact를 possession provenance에 반영한다."""

    count = state.players.position.shape[0]
    actor = jnp.asarray(actor, dtype=jnp.int32)
    applied = jnp.asarray(occurred) & (actor >= 0) & (actor < count)
    safe_actor = jnp.clip(actor, 0, count - 1)
    old_player = state.possession.player
    old_player_valid = (old_player >= 0) & (old_player < count)
    safe_old_player = jnp.clip(old_player, 0, count - 1)
    opponent_loss = (
        applied
        & old_player_valid
        & (roster.team_id[safe_actor] != roster.team_id[safe_old_player])
    )
    loss_lock = state.players.possession_loss_lock_substeps.at[safe_old_player].set(
        jnp.where(
            opponent_loss,
            jnp.maximum(
                state.players.possession_loss_lock_substeps[safe_old_player],
                jnp.asarray(possession_loss_lock_substeps, dtype=jnp.int32),
            ),
            state.players.possession_loss_lock_substeps[safe_old_player],
        )
    )
    contact = state.possession.last_contact._replace(
        actor=jnp.where(applied, actor, state.possession.last_contact.actor).astype(
            jnp.int32
        ),
        mechanism=jnp.where(
            applied,
            MECHANISM_PASSIVE_BODY,
            state.possession.last_contact.mechanism,
        ).astype(jnp.int32),
        intent=jnp.where(
            applied, INTENT_MOVE, state.possession.last_contact.intent
        ).astype(jnp.int32),
        outcome=jnp.where(
            applied, OUTCOME_DEFLECTION, state.possession.last_contact.outcome
        ).astype(jnp.int32),
        restart_kind=jnp.where(
            applied, RK_NONE, state.possession.last_contact.restart_kind
        ).astype(jnp.int32),
        law11_effect=jnp.where(
            applied,
            LAW11_DEFLECTION_NO_RESET,
            state.possession.last_contact.law11_effect,
        ).astype(jnp.int32),
        kick_applied=jnp.where(
            applied, jnp.bool_(False), state.possession.last_contact.kick_applied
        ),
        intent_source=jnp.where(
            applied, INTENT_SOURCE_NONE, state.possession.last_contact.intent_source
        ).astype(jnp.int32),
    )
    possession = state.possession._replace(
        team=jnp.where(applied, NO_TEAM, state.possession.team).astype(jnp.int32),
        player=jnp.where(applied, NO_PLAYER, state.possession.player).astype(jnp.int32),
        previous_team=jnp.where(
            applied & (state.possession.team != NO_TEAM),
            state.possession.team,
            state.possession.previous_team,
        ).astype(jnp.int32),
        control_ticks=jnp.where(applied, 0, state.possession.control_ticks).astype(
            jnp.int32
        ),
        last_contact=contact,
    )
    release = state.restart_release
    restricted_throw_goalkeeper = (
        applied
        & release.active
        & (release.kind == RK_THROWIN)
        & roster.is_goalkeeper[safe_actor]
        & (roster.team_id[safe_actor] == release.team)
    )
    other_player_contact = (
        applied
        & release.active
        & (actor != release.taker)
        & (~restricted_throw_goalkeeper)
    )
    restart_release = release._replace(
        active=jnp.where(other_player_contact, jnp.bool_(False), release.active),
        untouched=jnp.where(applied, jnp.bool_(False), release.untouched),
    )
    return state._replace(
        players=state.players._replace(
            possession_loss_lock_substeps=loss_lock,
        ),
        possession=possession,
        restart_release=restart_release,
    )


__all__ = [
    "DEFAULT_PASSIVE_CONTACT_PHYSICS",
    "NutmegEvent",
    "PassiveContactEvent",
    "PassiveContactPhysics",
    "apply_passive_contact",
    "apply_passive_contact_provenance",
    "detect_between_legs_passage",
    "detect_nutmeg",
    "detect_passive_contact",
    "empty_nutmeg_event",
    "empty_passive_contact_event",
]
