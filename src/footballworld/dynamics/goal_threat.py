"""Conservative pre-contact goal-threat classification."""

import jax
import jax.numpy as jnp

from footballworld.config.ball_physics import BallPhysics
from footballworld.config.geometry import Ball, Stadium
from footballworld.core.constants import GEOMETRY_EPS
from footballworld.core.state import State
from footballworld.dynamics.goal_frame import detect_goal_frame_contact


def is_goal_threat(
    state: State,
    actor: jax.Array,
    *,
    ball: Ball = Ball(),
    stadium: Stadium = Stadium(),
    physics: BallPhysics = BallPhysics(),
) -> jax.Array:
    """Return whether the incoming ball is headed at the actor's own goal.

    This is a conservative O(1) kinematic classifier, not a future trajectory
    forecast with drag or Magnus forces. It projects the current velocity ray
    to the whole-ball scoring plane, applies gravity while airborne, clamps
    height at the turf after ground arrival, and includes swept goal-frame
    contact along the same chord.
    """

    player_count = state.players.position.shape[0]
    safe_actor = jnp.clip(jnp.asarray(actor, dtype=jnp.int32), 0, player_count - 1)
    actor_valid = (
        (actor >= 0) & (actor < player_count) & state.players.active[safe_actor]
    )
    actor_team = state.players.team_id[safe_actor]
    own_goal_sign = -state.attack_direction[actor_team]
    scoring_plane_x = own_goal_sign * (stadium.half_length + ball.radius)

    velocity_x = state.ball.velocity[0]
    toward_own_goal = own_goal_sign * velocity_x > GEOMETRY_EPS
    safe_velocity_x = jnp.where(toward_own_goal, velocity_x, own_goal_sign)
    time_to_plane = (scoring_plane_x - state.ball.position[0]) / safe_velocity_x
    reaches_plane = toward_own_goal & (time_to_plane >= 0.0)
    time_to_plane = jnp.maximum(time_to_plane, 0.0)

    plane_y = state.ball.position[1] + state.ball.velocity[1] * time_to_plane
    ballistic_z = (
        state.ball.position[2]
        + state.ball.velocity[2] * time_to_plane
        - 0.5 * physics.g * time_to_plane * time_to_plane
    )
    plane_z = jnp.maximum(ballistic_z, ball.radius)
    plane_position = jnp.stack((scoring_plane_x, plane_y, plane_z))
    path_delta = plane_position - state.ball.position

    aperture_clear = (jnp.abs(plane_y) <= 0.5 * stadium.goal_width - ball.radius) & (
        plane_z <= stadium.goal_height - ball.radius
    )
    frame_contact = detect_goal_frame_contact(
        state.ball.position,
        path_delta,
        state.ball.live & actor_valid & reaches_plane,
        ball=ball,
        stadium=stadium,
        physics=physics,
    )
    return (
        state.ball.live
        & actor_valid
        & reaches_plane
        & (aperture_clear | frame_contact.occurred)
    )
