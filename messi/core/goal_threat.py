"""Goalkeeper contact의 Law 11 deliberate-save 분류용 O(1) 기하."""

import jax
import jax.numpy as jnp

from messi.core.ball import BallPhysics
from messi.core.boundary import GoalGeometry
from messi.core.geometry import BallGeometry
from messi.core.goal_frame import GoalFramePhysics, detect_goal_frame_contact

_GEOMETRY_EPS = 1.0e-6


def is_goal_threat(
    state,
    roster,
    actor: jax.Array,
    field_half_extent: jax.Array,
    *,
    ball: BallGeometry = BallGeometry(),
    goal: GoalGeometry = GoalGeometry(),
    physics: BallPhysics = BallPhysics(),
    frame_physics: GoalFramePhysics = GoalFramePhysics(),
) -> jax.Array:
    """현재 velocity ray가 actor team의 goal을 향하는지 보수적으로 판정한다.

    Drag·Magnus의 미래 적분을 복제하지 않고 scoring plane까지의 직선 운동과 중력만
    사용한다. 같은 chord가 post/crossbar에 닿는 경우도 deliberate save로 보존한다.
    FootballWorld ``dynamics/goal_threat.py``의 Law 11 contract를 따른다.
    """

    count = state.players.position.shape[0]
    actor = jnp.asarray(actor, dtype=jnp.int32)
    safe_actor = jnp.clip(actor, 0, count - 1)
    actor_valid = (
        (actor >= 0) & (actor < count) & state.players.active[safe_actor]
    )
    team = roster.team_id[safe_actor]
    own_goal_sign = -state.attack_direction[team]
    half_length = jnp.asarray(field_half_extent, state.ball.position.dtype)[0]
    scoring_plane_x = own_goal_sign * (half_length + ball.radius_m)

    velocity_x = state.ball.velocity[0]
    toward_goal = own_goal_sign * velocity_x > _GEOMETRY_EPS
    safe_velocity_x = jnp.where(toward_goal, velocity_x, own_goal_sign)
    time_to_plane = (scoring_plane_x - state.ball.position[0]) / safe_velocity_x
    reaches_plane = toward_goal & (time_to_plane >= 0.0)
    time_to_plane = jnp.maximum(time_to_plane, 0.0)
    plane_y = state.ball.position[1] + state.ball.velocity[1] * time_to_plane
    plane_z = jnp.maximum(
        state.ball.position[2]
        + state.ball.velocity[2] * time_to_plane
        - 0.5 * physics.gravity_mps2 * time_to_plane * time_to_plane,
        ball.radius_m,
    )
    plane = jnp.stack((scoring_plane_x, plane_y, plane_z))
    path_delta = plane - state.ball.position
    aperture_clear = (
        jnp.abs(plane_y) <= 0.5 * goal.width_m - ball.radius_m
    ) & (plane_z <= goal.height_m - ball.radius_m)
    frame_contact = detect_goal_frame_contact(
        state.ball.position,
        path_delta,
        state.ball.live & actor_valid & reaches_plane,
        half_length=half_length,
        goal_half_width=0.5 * goal.width_m,
        goal_height=goal.height_m,
        ball_radius=ball.radius_m,
        physics=frame_physics,
    )
    return (
        state.ball.live
        & actor_valid
        & reaches_plane
        & (aperture_clear | frame_contact.occurred)
    )


__all__ = ["is_goal_threat"]
