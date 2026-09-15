"""Focused contracts for angular-only partial observation visibility."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from footballworld import FootballWorld, Player, PlayerProfile
from footballworld.config.perception import Perception

_FORMATION = (
    (-50.0, 0.0),
    (-35.0, -24.0),
    (-35.0, -8.0),
    (-35.0, 8.0),
    (-35.0, 24.0),
    (-20.0, -18.0),
    (-20.0, 0.0),
    (-20.0, 18.0),
    (-8.0, -24.0),
    (-8.0, 0.0),
    (-8.0, 24.0),
)


def _team(identity_base: int) -> tuple[Player, ...]:
    return tuple(
        Player(
            PlayerProfile(
                player_id=identity_base + index,
                is_goalkeeper=index == 0,
            ),
            position,
        )
        for index, position in enumerate(_FORMATION)
    )


def _view_limited_rollout():
    env = FootballWorld(perception=Perception(limit_by_view_angle=True))
    reset = env.reset(_team(1_000), _team(2_000), key=jax.random.key(101))
    observer = 1
    near_target = 2
    far_target = 3
    positions = reset.rollout.state.players.position
    positions = positions.at[observer].set(jnp.asarray((0.0, 0.0)))
    positions = positions.at[near_target].set(jnp.asarray((10.0, 0.0)))
    positions = positions.at[far_target].set(jnp.asarray((50.0, 0.0)))
    players = reset.rollout.state.players._replace(
        position=positions,
        body_forward=reset.rollout.state.players.body_forward.at[observer].set(
            jnp.asarray((1.0, 0.0))
        ),
        gaze_yaw=reset.rollout.state.players.gaze_yaw.at[observer].set(
            jnp.float32(0.0)
        ),
    )
    state = reset.rollout.state._replace(
        players=players,
        ball=reset.rollout.state.ball._replace(
            position=jnp.asarray((50.0, 0.0, env.ball.radius))
        ),
    )
    return env, reset.rollout._replace(state=state), observer, near_target, far_target


def test_partial_observation_has_no_ten_or_fifty_metre_range_cutoff() -> None:
    env, rollout, observer, near_target, far_target = _view_limited_rollout()

    eager = env.observe_all_si(rollout)
    compiled = jax.jit(env.observe_all_si)(rollout)

    for observation in (eager, compiled):
        assert bool(observation.players.visible[observer, near_target])
        assert bool(observation.players.visible[observer, far_target])
        assert bool(observation.ball.visible[observer])


def test_near_target_behind_view_direction_is_invisible() -> None:
    env, rollout, observer, near_target, _ = _view_limited_rollout()
    state = rollout.state._replace(
        players=rollout.state.players._replace(
            position=rollout.state.players.position.at[near_target].set(
                jnp.asarray((-1.0, 0.0))
            )
        ),
        ball=rollout.state.ball._replace(
            position=jnp.asarray((-1.0, 0.0, env.ball.radius))
        ),
    )
    rollout = rollout._replace(state=state)

    eager = env.observe_all_si(rollout)
    compiled = jax.jit(env.observe_all_si)(rollout)

    for observation in (eager, compiled):
        assert not bool(observation.players.visible[observer, near_target])
        assert not bool(observation.ball.visible[observer])

    np.testing.assert_array_equal(compiled.players.visible, eager.players.visible)
    np.testing.assert_array_equal(compiled.ball.visible, eager.ball.visible)
