"""Physical goalkeeper-held ball pose contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from footballworld import FootballWorld, Player, PlayerProfile
from footballworld.config.geometry import Ball
from footballworld.core.constants import RK_GK_HOLD
from footballworld.dynamics.substep import _attach_held_ball
from footballworld.rules.goalkeeper_hold_position import (
    goalkeeper_held_ball_position,
)


def _team(identity_base: int) -> tuple[Player, ...]:
    return tuple(
        Player(
            PlayerProfile(
                player_id=identity_base + index,
                is_goalkeeper=index == 0,
            ),
            (-48.0 + 4.0 * index, -20.0 + 4.0 * index),
        )
        for index in range(11)
    )


def test_held_ball_is_one_configured_radius_ahead_and_within_reach():
    ball = Ball(radius=0.2)
    env = FootballWorld(ball=ball)
    reset = env.reset(_team(1_000), _team(2_000), key=jax.random.key(17))
    goalkeeper = jnp.int32(0)
    state = reset.rollout.state._replace(
        players=reset.rollout.state.players._replace(
            position=reset.rollout.state.players.position.at[goalkeeper].set(
                jnp.asarray((-40.0, 3.0), jnp.float32)
            ),
            body_forward=reset.rollout.state.players.body_forward.at[goalkeeper].set(
                jnp.asarray((0.0, 1.0), jnp.float32)
            ),
        )
    )

    eager = goalkeeper_held_ball_position(state, goalkeeper, ball=ball, body=env.body)
    compiled = jax.jit(
        lambda value: goalkeeper_held_ball_position(
            value, goalkeeper, ball=ball, body=env.body
        )
    )(state)

    expected = np.asarray(
        (
            -40.0,
            3.0 + ball.radius,
            env.body.torso_top_height(float(state.players.height[goalkeeper])),
        ),
        np.float32,
    )
    np.testing.assert_allclose(eager, expected, rtol=0.0, atol=1.0e-6)
    np.testing.assert_array_equal(compiled, eager)
    distance_xy = np.linalg.norm(
        np.asarray(eager[:2]) - np.asarray(state.players.position[goalkeeper])
    )
    assert distance_xy <= env.reach.carry_radius_m + ball.radius


def test_substep_attachment_uses_the_shared_forward_pose_under_jit():
    ball = Ball(radius=0.2)
    env = FootballWorld(ball=ball)
    reset = env.reset(_team(1_000), _team(2_000), key=jax.random.key(18))
    goalkeeper = jnp.int32(0)
    state = reset.rollout.state._replace(
        ball=reset.rollout.state.ball._replace(
            position=jnp.asarray((20.0, 20.0, 5.0), jnp.float32)
        ),
        restart=reset.rollout.state.restart._replace(
            kind=jnp.int32(RK_GK_HOLD),
            team=reset.rollout.state.players.team_id[goalkeeper],
            taker=goalkeeper,
        ),
    )
    expected = goalkeeper_held_ball_position(
        state, goalkeeper, ball=ball, body=env.body
    )

    attached = jax.jit(
        lambda value: _attach_held_ball(value, ball_geometry=ball, body=env.body)
    )(state)

    np.testing.assert_array_equal(attached.ball.position, expected)
    np.testing.assert_array_equal(attached.ball.velocity, np.zeros(3, np.float32))
    assert not bool(attached.ball.live)
