"""Per-observer geometric context derived from public observations only."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.environment.observation import Observation, RosterMetadata


class RulePolicyContext(NamedTuple):
    """Absolute attack-local geometry and lawful roster masks for each actor.

    The leading axis is the observer.  ``player_position[row, slot]`` and all
    matching fields are reconstructed exclusively from ``row``; rows are
    never pooled to fill a visibility gap.  Hidden kinematics are zero and
    remain distinguished by their explicit visibility masks.
    """

    self_index: jax.Array
    self_team: jax.Array
    self_active: jax.Array
    self_goalkeeper: jax.Array
    self_position: jax.Array
    self_velocity: jax.Array
    player_position: jax.Array
    player_velocity: jax.Array
    player_visible: jax.Array
    participating: jax.Array
    same_team: jax.Array
    teammate: jax.Array
    opponent: jax.Array
    ball_position: jax.Array
    ball_velocity: jax.Array
    ball_spin: jax.Array
    ball_visible: jax.Array


def build_rule_policy_context(
    observations: Observation,
    roster: RosterMetadata,
) -> RulePolicyContext:
    """Reconstruct observer-local absolute geometry without cross-row leakage.

    ``observations`` must be the fixed-shape result of ``observe_all_si``.  Pitch
    x/y coordinates and velocities are in each observer team's attacking
    frame.  Height, vertical velocity, and spin retain the observation's
    physical units and axis convention.
    """

    player_count = observations.self_state.player_index.shape[0]
    if observations.self_state.position.shape != (player_count, 2):
        raise ValueError("observations must have one self row per roster slot")
    if observations.self_state.velocity.shape != (player_count, 2):
        raise ValueError("self velocity must have one row per roster slot")
    if observations.players.relative_position.shape != (
        player_count,
        player_count,
        2,
    ):
        raise ValueError("observations must have leading observer and roster axes")
    if observations.players.relative_velocity.shape != (
        player_count,
        player_count,
        2,
    ):
        raise ValueError("player velocity axes must match player positions")
    if observations.players.visible.shape != (player_count, player_count):
        raise ValueError("player visibility must match both player axes")
    if observations.ball.relative_state.shape != (player_count, 9):
        raise ValueError("ball relative_state must have shape (players, 9)")
    if roster.team_id.shape != (player_count,):
        raise ValueError("roster must describe the observation roster axis")
    if roster.is_goalkeeper.shape != (player_count,):
        raise ValueError("roster goalkeeper flags must match the roster axis")

    row = jnp.arange(player_count, dtype=jnp.int32)
    self_index = observations.self_state.player_index.astype(jnp.int32)
    self_position = jnp.asarray(observations.self_state.position, dtype=jnp.float32)
    self_velocity = jnp.asarray(observations.self_state.velocity, dtype=jnp.float32)
    self_team = roster.team_id[self_index]
    self_goalkeeper = roster.is_goalkeeper[self_index]

    visible = jnp.asarray(observations.players.visible, dtype=jnp.bool_)
    participating = observations.players.on_pitch & (~observations.players.sent_off)
    self_active = participating[row, self_index]
    same_team = roster.team_id[None, :] == self_team[:, None]
    is_self = row[None, :] == self_index[:, None]
    teammate = same_team & participating & visible & (~is_self)
    opponent = (~same_team) & participating & visible

    player_position_raw = (
        self_position[:, None, :] + observations.players.relative_position
    )
    player_velocity_raw = (
        self_velocity[:, None, :] + observations.players.relative_velocity
    )
    player_position = jnp.where(
        visible[..., None], player_position_raw, jnp.float32(0.0)
    ).astype(jnp.float32)
    player_velocity = jnp.where(
        visible[..., None], player_velocity_raw, jnp.float32(0.0)
    ).astype(jnp.float32)

    ball_state = observations.ball.relative_state
    ball_position_raw = jnp.concatenate(
        (
            self_position + ball_state[:, 0:2],
            ball_state[:, 2:3],
        ),
        axis=-1,
    )
    ball_velocity_raw = jnp.concatenate(
        (
            self_velocity + ball_state[:, 3:5],
            ball_state[:, 5:6],
        ),
        axis=-1,
    )
    ball_visible = jnp.asarray(observations.ball.visible, dtype=jnp.bool_)
    ball_position = jnp.where(
        ball_visible[:, None], ball_position_raw, jnp.float32(0.0)
    ).astype(jnp.float32)
    ball_velocity = jnp.where(
        ball_visible[:, None], ball_velocity_raw, jnp.float32(0.0)
    ).astype(jnp.float32)
    ball_spin = jnp.where(
        ball_visible[:, None], ball_state[:, 6:9], jnp.float32(0.0)
    ).astype(jnp.float32)

    return RulePolicyContext(
        self_index=self_index,
        self_team=self_team.astype(jnp.int32),
        self_active=self_active,
        self_goalkeeper=self_goalkeeper,
        self_position=self_position,
        self_velocity=self_velocity,
        player_position=player_position,
        player_velocity=player_velocity,
        player_visible=visible,
        participating=participating,
        same_team=same_team,
        teammate=teammate,
        opponent=opponent,
        ball_position=ball_position,
        ball_velocity=ball_velocity,
        ball_spin=ball_spin,
        ball_visible=ball_visible,
    )


__all__ = ["RulePolicyContext", "build_rule_policy_context"]
