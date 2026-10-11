"""기존 direct-contact card의 Law 12 severity·tactical label 분류."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.core.constants import (
    DISCIPLINE_RED,
    DISCIPLINE_YELLOW,
    TEAM_0,
    TEAM_1,
)
from messi.core.foul import FoulLaw

SEVERITY_NONE = 0
SEVERITY_CARELESS = 1
SEVERITY_RECKLESS = 2
SEVERITY_EXCESSIVE_FORCE = 3

TACTICAL_NONE = 0
TACTICAL_SPA = 1
TACTICAL_DOGSO = 2


class DisciplineClassification(NamedTuple):
    severity: jax.Array
    tactical_effect: jax.Array


def classify_existing_contact_discipline(
    state,
    roster,
    *,
    valid: jax.Array,
    offender: jax.Array,
    victim: jax.Array,
    contact_position: jax.Array,
    administrative_discipline: jax.Array,
    attempt_to_play_ball: jax.Array,
    closing_speed: jax.Array,
    approach_direction: jax.Array,
    velocity_alignment: jax.Array,
    field_half_extent: jax.Array = (52.5, 34.0),
    law: FoulLaw = FoulLaw(),
) -> DisciplineClassification:
    """이미 정해진 card 강도를 올리지 않는 strict post-classifier.

    FootballWorld의 contact discipline classifier를 MESSI의 분리된 roster/state
    contract에 맞춘 것이다. No-card contact를 card로 올리지 않으며 yellow/red label도
    기존 administrative discipline과 양립할 때만 부여한다.
    """

    dtype = state.players.position.dtype
    count = state.players.position.shape[0]
    offender = jnp.asarray(offender, dtype=jnp.int32)
    victim = jnp.asarray(victim, dtype=jnp.int32)
    indices_valid = (
        (offender >= 0)
        & (offender < count)
        & (victim >= 0)
        & (victim < count)
    )
    safe_offender = jnp.clip(offender, 0, count - 1)
    safe_victim = jnp.clip(victim, 0, count - 1)
    offender_team = roster.team_id[safe_offender].astype(jnp.int32)
    victim_team = roster.team_id[safe_victim].astype(jnp.int32)
    safe_victim_team = jnp.clip(victim_team, TEAM_0, TEAM_1)

    approach = jnp.asarray(approach_direction, dtype=dtype)
    approach = approach / jnp.maximum(jnp.linalg.norm(approach), 1.0e-9)
    victim_body = state.players.body_forward[safe_victim]
    end_on = jnp.abs(jnp.clip(jnp.dot(approach, victim_body), -1.0, 1.0))
    shoulder = jnp.clip(velocity_alignment, 0.0, 1.0) * (1.0 - end_on)
    unsafe_approach = end_on > shoulder
    closing_speed = jnp.maximum(jnp.asarray(closing_speed, dtype=dtype), 0.0)
    combined_maximum = roster.max_speed[safe_offender] + roster.max_speed[safe_victim]
    individual_maximum = jnp.maximum(
        roster.max_speed[safe_offender], roster.max_speed[safe_victim]
    )
    reckless_geometry = unsafe_approach & (2.0 * closing_speed >= combined_maximum)
    excessive_geometry = unsafe_approach & (closing_speed >= individual_maximum)

    half_extent = jnp.asarray(field_half_extent, dtype=dtype)
    contact_xy = jnp.asarray(contact_position, dtype=dtype)[:2]
    victim_direction = state.attack_direction[safe_victim_team]
    victim_goal = jnp.asarray((victim_direction * half_extent[0], 0.0), dtype=dtype)
    goal_vector = victim_goal - contact_xy
    goal_unit = goal_vector / jnp.maximum(jnp.linalg.norm(goal_vector), 1.0e-9)
    goalward_attack = (
        (state.possession.player == victim)
        & (state.possession.team == victim_team)
        & (jnp.dot(victim_body, goal_unit) > 0.0)
        & (jnp.dot(state.players.velocity[safe_victim], goal_unit) >= 0.0)
    )
    progress = contact_xy[0] * victim_direction
    distance_to_goal = half_extent[0] - progress
    near_goal = (distance_to_goal >= 0.0) & (
        distance_to_goal <= law.penalty_area_length_m
    )
    goalward_defender = (
        state.players.active
        & (roster.team_id == offender_team)
        & (jnp.arange(count, dtype=jnp.int32) != safe_offender)
        & (state.players.position[:, 0] * victim_direction > progress)
    )
    dogso = goalward_attack & near_goal & (
        jnp.sum(goalward_defender.astype(jnp.int32)) <= 1
    )
    spa = goalward_attack & (progress >= 0.0) & (~dogso)

    offender_direction = state.attack_direction[jnp.clip(offender_team, TEAM_0, TEAM_1)]
    own_goal_direction = -offender_direction
    depth = half_extent[0] - own_goal_direction * contact_xy[0]
    inside_penalty = (
        (depth >= 0.0)
        & (depth <= law.penalty_area_length_m)
        & (jnp.abs(contact_xy[1]) <= law.penalty_area_half_width_m)
    )
    valid = jnp.asarray(valid, dtype=jnp.bool_) & indices_valid
    discipline = jnp.asarray(administrative_discipline, dtype=jnp.int32)
    yellow = valid & (discipline == DISCIPLINE_YELLOW)
    red = valid & (discipline == DISCIPLINE_RED)
    yellow_dogso = yellow & dogso & inside_penalty & attempt_to_play_ball
    red_dogso = red & dogso & ((~inside_penalty) | (~attempt_to_play_ball))
    yellow_reckless = yellow & (~yellow_dogso) & reckless_geometry
    red_excessive = red & (~red_dogso) & excessive_geometry
    yellow_spa = (
        yellow
        & (~yellow_dogso)
        & (~yellow_reckless)
        & spa
        & (~(inside_penalty & attempt_to_play_ball))
    )
    severity = jnp.where(valid, SEVERITY_CARELESS, SEVERITY_NONE)
    severity = jnp.where(yellow_reckless, SEVERITY_RECKLESS, severity)
    severity = jnp.where(red_excessive, SEVERITY_EXCESSIVE_FORCE, severity)
    tactical = jnp.where(
        yellow_dogso | red_dogso,
        TACTICAL_DOGSO,
        jnp.where(yellow_spa, TACTICAL_SPA, TACTICAL_NONE),
    )
    return DisciplineClassification(
        severity=severity.astype(jnp.int32),
        tactical_effect=tactical.astype(jnp.int32),
    )


__all__ = [
    "SEVERITY_CARELESS",
    "SEVERITY_EXCESSIVE_FORCE",
    "SEVERITY_NONE",
    "SEVERITY_RECKLESS",
    "TACTICAL_DOGSO",
    "TACTICAL_NONE",
    "TACTICAL_SPA",
    "DisciplineClassification",
    "classify_existing_contact_discipline",
]
