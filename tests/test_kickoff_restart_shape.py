"""Kick-off layout realism and collision-liveness contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from footballworld import FootballWorld, Player, PlayerProfile
from footballworld.environment.validation import validate_no_torso_overlap
from footballworld.rules.restart_positioning import prepare_restart_positioning

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


def _prepare(env: FootballWorld, state):
    return prepare_restart_positioning(
        state,
        stadium=env.stadium,
        ball=env.ball,
        body=env.body,
    )


def test_legal_authored_kickoff_shape_is_retained_exactly() -> None:
    env = FootballWorld()
    state = env.reset(_team(1_000), _team(2_000), key=jax.random.key(29)).rollout.state

    positioned = _prepare(env, state)

    assert bool(np.asarray(positioned.taker_ready))
    np.testing.assert_array_equal(
        np.asarray(positioned.position),
        np.asarray(state.players.position),
    )


def test_colliding_post_goal_projection_uses_spread_team_shape_eager_and_jit() -> None:
    env = FootballWorld()
    state = env.reset(_team(1_000), _team(2_000), key=jax.random.key(29)).rollout.state
    clustered = jnp.broadcast_to(
        jnp.asarray((45.0, 0.0), dtype=jnp.float32),
        state.players.position.shape,
    )
    state = state._replace(
        players=state.players._replace(position=clustered),
        restart_layout_ready=jnp.bool_(False),
    )

    eager = _prepare(env, state)
    compiled = jax.jit(lambda candidate: _prepare(env, candidate))(state)
    eager, compiled = jax.block_until_ready((eager, compiled))

    np.testing.assert_allclose(
        np.asarray(compiled.position),
        np.asarray(eager.position),
        rtol=0.0,
        atol=2.0e-6,
    )
    assert bool(np.asarray(eager.taker_ready))

    active = np.asarray(state.players.active)
    team_id = np.asarray(state.players.team_id)
    position = np.asarray(eager.position)
    facing = np.asarray(eager.facing)
    validate_no_torso_overlap(
        position[active],
        facing[active],
        body=env.body,
        name="post-goal kick-off emergency shape",
    )

    for team in (0, 1):
        team_position = position[active & (team_id == team)]
        # This rejects the old centre-line packing while leaving the exact
        # geometry-derived slot coordinates private to the implementation.
        assert np.ptp(team_position[:, 0]) > env.stadium.penalty_area_length
        assert np.ptp(team_position[:, 1]) > env.stadium.center_circle_radius
        goalkeeper = np.asarray(state.players.is_goalkeeper) & (team_id == team)
        assert np.max(np.abs(position[goalkeeper, 0])) > (
            env.stadium.half_length - env.stadium.penalty_area_length
        )
