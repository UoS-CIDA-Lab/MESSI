"""Law 12 eligibility gates for automatic goalkeeper hand contact.

Only a targeted team-mate foot play needs a new scalar latch. Goalkeeper
hand-release and direct throw-in restrictions reuse ``restart_release``.
"""

import jax
import jax.numpy as jnp

from footballworld.config.ball_physics import BallPhysics
from footballworld.config.geometry import Ball
from footballworld.config.reach import Reach
from footballworld.core.constants import (
    DIV_EPS,
    GEOMETRY_EPS,
    NO_TEAM,
    RK_GK_HOLD,
    RK_THROWIN,
    TEAM_0,
    TEAM_1,
)
from footballworld.core.state import State


def _valid_team(team: jax.Array) -> jax.Array:
    return (team == TEAM_0) | (team == TEAM_1)


def goalkeeper_hand_restricted_mask(state: State) -> jax.Array:
    """Return active goalkeepers forbidden from automatic hand contact."""

    players = state.players
    indices = jnp.arange(players.position.shape[0], dtype=jnp.int32)
    release = state.restart_release
    valid_release = release.active & _valid_team(release.team)
    after_own_hand_release = (
        valid_release
        & (release.kind == RK_GK_HOLD)
        & (indices == release.taker)
        & (players.team_id == release.team)
    )
    direct_team_throw = (
        valid_release & (release.kind == RK_THROWIN) & (players.team_id == release.team)
    )
    targeted_team_kick = _valid_team(state.gk_backpass_team) & (
        players.team_id == state.gk_backpass_team
    )
    return (
        players.active
        & players.is_goalkeeper
        & (after_own_hand_release | direct_team_throw | targeted_team_kick)
    )


def goalkeeper_hand_restricted_team(state: State) -> jax.Array:
    """Return the restricted active goalkeeper's team, or ``NO_TEAM``."""

    restricted = goalkeeper_hand_restricted_mask(state)
    return jnp.max(jnp.where(restricted, state.players.team_id, NO_TEAM)).astype(
        jnp.int32
    )


def _conservative_path_length(
    position: jax.Array,
    velocity: jax.Array,
    *,
    ball: Ball,
    physics: BallPhysics,
) -> jax.Array:
    """Bound a straight unopposed path without integrating a trajectory.

    Ground range uses the rollout model's local rolling deceleration. Flight
    uses a no-drag closed form with finitely many configured bounces and then
    rolls at the attenuated horizontal speed. Magnus curvature is deliberately
    outside this intent proxy.
    """

    horizontal_speed = jnp.linalg.norm(velocity[:2])
    gravity = jnp.maximum(jnp.asarray(physics.g, dtype=velocity.dtype), DIV_EPS)
    height = jnp.maximum(position[2] - ball.radius, 0.0)
    vertical = velocity[2]
    impact_speed = jnp.sqrt(
        jnp.maximum(vertical * vertical + 2.0 * gravity * height, 0.0)
    )
    first_flight_s = jnp.maximum((vertical + impact_speed) / gravity, 0.0)
    restitution = jnp.clip(jnp.asarray(physics.e_rest, dtype=velocity.dtype), 0.0, 1.0)
    horizontal_keep = jnp.clip(
        jnp.asarray(physics.bounce_h_keep, dtype=velocity.dtype), 0.0, 1.0
    )
    settle_speed = jnp.maximum(
        jnp.asarray(physics.ground_settle_vz, dtype=velocity.dtype),
        DIV_EPS,
    )
    can_bounce = (impact_speed > settle_speed) & (restitution > DIV_EPS)
    safe_restitution = jnp.clip(restitution, DIV_EPS, 1.0 - DIV_EPS)
    bounce_count = jnp.where(
        can_bounce,
        jnp.ceil(
            jnp.log(settle_speed / jnp.maximum(impact_speed, DIV_EPS))
            / jnp.log(safe_restitution)
        ),
        0.0,
    )
    bounce_count = jnp.maximum(bounce_count, 0.0)
    ratio = restitution * horizontal_keep
    finite_ratio_sum = jnp.where(
        bounce_count > 0.0,
        ratio
        * (1.0 - jnp.power(ratio, bounce_count))
        / jnp.maximum(1.0 - ratio, DIV_EPS),
        0.0,
    )
    later_bounce_s = 2.0 * impact_speed / gravity * finite_ratio_sum
    post_bounce_speed = horizontal_speed * jnp.power(horizontal_keep, bounce_count)
    post_bounce_deceleration = jnp.interp(
        post_bounce_speed,
        jnp.asarray(physics.roll_v_knots, dtype=velocity.dtype),
        jnp.asarray(physics.roll_d_knots, dtype=velocity.dtype),
    )
    rolling_distance = (
        post_bounce_speed
        * post_bounce_speed
        / (2.0 * jnp.maximum(post_bounce_deceleration, DIV_EPS))
    )
    return horizontal_speed * (first_flight_s + later_bounce_s) + rolling_distance


def targeted_own_goalkeeper_team(
    state: State,
    actor: jax.Array,
    release_position: jax.Array,
    outgoing_velocity: jax.Array,
    *,
    ball: Ball = Ball(),
    reach: Reach = Reach(),
    physics: BallPhysics = BallPhysics(),
) -> jax.Array:
    """Infer a foot play whose finite outgoing ray enters own-GK reach."""

    players = state.players
    player_count = players.position.shape[0]
    actor = jnp.asarray(actor, dtype=jnp.int32)
    actor_valid = (actor >= 0) & (actor < player_count)
    safe_actor = jnp.clip(actor, 0, player_count - 1)
    actor_team = players.team_id[safe_actor]
    valid_actor = actor_valid & players.active[safe_actor] & _valid_team(actor_team)
    own_goalkeeper = (
        players.active
        & players.is_goalkeeper
        & (players.team_id == actor_team)
        & (jnp.arange(player_count, dtype=jnp.int32) != actor)
    )
    horizontal_velocity = outgoing_velocity[:2]
    horizontal_speed = jnp.linalg.norm(horizontal_velocity)
    direction = horizontal_velocity / jnp.maximum(horizontal_speed, DIV_EPS)
    path_length = _conservative_path_length(
        release_position,
        outgoing_velocity,
        ball=ball,
        physics=physics,
    )
    to_goalkeeper = players.position - release_position[:2]
    along = jnp.sum(to_goalkeeper * direction, axis=-1)
    closest_along = jnp.clip(along, 0.0, path_length)
    closest = to_goalkeeper - closest_along[:, None] * direction
    claim_radius = jnp.asarray(
        reach.goalkeeper_radius_m + ball.radius,
        dtype=players.position.dtype,
    )
    intersects = (
        own_goalkeeper
        & (horizontal_speed > GEOMETRY_EPS)
        & (along > GEOMETRY_EPS)
        & (jnp.sum(closest * closest, axis=-1) <= claim_radius * claim_radius)
    )
    targeted = valid_actor & jnp.any(intersects)
    return jnp.where(targeted, actor_team, NO_TEAM).astype(jnp.int32)


def update_backpass_after_deliberate_attempt(
    state: State,
    *,
    actor: jax.Array,
    actual_contact: jax.Array,
    restricted_goalkeeper_clear_foot_attempt: jax.Array,
    arm_targeted_foot_release: jax.Array,
    release_position: jax.Array,
    outgoing_velocity: jax.Array,
    ball: Ball = Ball(),
    reach: Reach = Reach(),
    physics: BallPhysics = BallPhysics(),
) -> jax.Array:
    """Clear or arm the team-kick latch after one selected contest."""

    restricted = goalkeeper_hand_restricted_mask(state) & (
        state.players.team_id == state.gk_backpass_team
    )
    actor = jnp.asarray(actor, dtype=jnp.int32)
    actor_valid = (actor >= 0) & (actor < state.players.position.shape[0])
    safe_actor = jnp.clip(actor, 0, state.players.position.shape[0] - 1)
    restricted_goalkeeper = actor_valid & restricted[safe_actor]
    clear = (actual_contact & (~restricted_goalkeeper)) | jnp.asarray(
        restricted_goalkeeper_clear_foot_attempt, dtype=jnp.bool_
    )
    cleared = jnp.where(clear, NO_TEAM, state.gk_backpass_team).astype(jnp.int32)

    def targeted(_: None) -> jax.Array:
        return targeted_own_goalkeeper_team(
            state,
            actor,
            release_position,
            outgoing_velocity,
            ball=ball,
            reach=reach,
            physics=physics,
        )

    targeted_team = jax.lax.cond(
        arm_targeted_foot_release,
        targeted,
        lambda _: jnp.int32(NO_TEAM),
        operand=None,
    )
    return jnp.where(_valid_team(targeted_team), targeted_team, cleared).astype(
        jnp.int32
    )


def clear_backpass_after_passive_contact(
    state: State,
    actor: jax.Array,
    occurred: jax.Array,
) -> jax.Array:
    """Clear on another player's hit; preserve the restricted GK's body hit."""

    restricted = goalkeeper_hand_restricted_mask(state) & (
        state.players.team_id == state.gk_backpass_team
    )
    actor = jnp.asarray(actor, dtype=jnp.int32)
    valid = (actor >= 0) & (actor < state.players.position.shape[0])
    safe_actor = jnp.clip(actor, 0, state.players.position.shape[0] - 1)
    restricted_goalkeeper = valid & restricted[safe_actor]
    return jnp.where(
        occurred & (~restricted_goalkeeper), NO_TEAM, state.gk_backpass_team
    ).astype(jnp.int32)


__all__ = [
    "clear_backpass_after_passive_contact",
    "goalkeeper_hand_restricted_mask",
    "goalkeeper_hand_restricted_team",
    "targeted_own_goalkeeper_team",
    "update_backpass_after_deliberate_attempt",
]
