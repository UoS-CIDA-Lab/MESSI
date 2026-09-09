"""Focused semantics for the chronological no-ball-event fast path."""

import jax
import jax.numpy as jnp
import numpy as np

from footballworld.config.geometry import Ball
from footballworld.config.reach import Reach
from footballworld.config.roster import Player, PlayerProfile
from footballworld.core.action import IntentAction
from footballworld.core.constants import NO_PLAYER, NO_TEAM, RK_NONE
from footballworld.core.contact import INTENT_PASS
from footballworld.dynamics.action import decode_physics_action
from footballworld.dynamics.ball import advance_smooth
from footballworld.dynamics.contact import active_contact_possible
from footballworld.dynamics.contest import sample_contest_override
from footballworld.dynamics.substep import step_physics_substep
from footballworld.environment.initialization import initialize_state

_PLAYER_COUNT = 22
_DT = 1.0 / 90.0


def _clear_fixture():
    team_0 = tuple(
        Player(
            PlayerProfile(index, is_goalkeeper=index == 0),
            (-47.0 + 2.0 * (index % 6), -25.0 + 4.0 * (index % 3)),
        )
        for index in range(11)
    )
    team_1 = tuple(
        Player(
            PlayerProfile(100 + index, is_goalkeeper=index == 0),
            (-47.0 + 2.0 * (index % 6), -25.0 + 4.0 * (index % 3)),
        )
        for index in range(11)
    )
    state = initialize_state(team_0, team_1).state
    state = state._replace(
        restart=state.restart._replace(
            kind=jnp.int32(RK_NONE),
            team=jnp.int32(NO_TEAM),
            taker=jnp.int32(NO_PLAYER),
        ),
        ball=state.ball._replace(
            position=jnp.asarray([0.0, 0.0, 1.2], dtype=jnp.float32),
            velocity=jnp.asarray([4.5, 0.7, 0.5], dtype=jnp.float32),
            spin=jnp.asarray([0.0, 7.0, -3.0], dtype=jnp.float32),
            live=jnp.bool_(True),
        ),
    )
    public_action = IntentAction.neutral(_PLAYER_COUNT)
    return state, decode_physics_action(state, public_action)


def _step(state, action):
    inactive = jnp.zeros(_PLAYER_COUNT, dtype=jnp.bool_)
    enabled = jnp.ones(_PLAYER_COUNT, dtype=jnp.bool_)
    return step_physics_substep(
        state,
        action,
        jax.random.key(20260909),
        sample_contest_override(),
        inactive,
        dt=_DT,
        boundary_margin_m=0.0,
        contact_attempted=inactive,
        locomotion_enabled=enabled,
        position_update_enabled=enabled,
        separation_pinned=inactive,
    )


def test_clear_substep_reuses_authoritative_smooth_endpoint_exactly():
    state, action = _clear_fixture()
    expected = advance_smooth(state.ball, dt=_DT)
    result = jax.jit(_step)(state, action)

    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(result.state.ball),
        jax.tree.leaves(expected),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(actual_leaf), np.asarray(expected_leaf)
        )
    assert not bool(np.asarray(result.deliberate_occurred))
    assert not bool(np.asarray(result.passive_occurred).any())
    assert not bool(np.asarray(result.woodwork_occurred).any())
    assert not bool(np.asarray(result.boundary.occurred))
    assert not bool(np.asarray(result.event_budget_exhausted))


def test_active_broadphase_is_conservative_for_nonfinite_requested_path():
    state, _ = _clear_fixture()
    intents = jnp.zeros(_PLAYER_COUNT, dtype=jnp.int32).at[9].set(INTENT_PASS)
    continuous = jnp.zeros((_PLAYER_COUNT, 8), dtype=jnp.float32)
    action = decode_physics_action(state, IntentAction.from_array(intents, continuous))
    possible = active_contact_possible(
        state,
        action,
        jnp.zeros(_PLAYER_COUNT, dtype=jnp.bool_),
        jnp.zeros(_PLAYER_COUNT, dtype=jnp.bool_),
        jnp.asarray([jnp.nan, 0.0, 0.0], dtype=jnp.float32),
        state.players.position,
        jnp.zeros_like(state.players.position),
        excluded_actor=jnp.int32(NO_PLAYER),
        search_enabled=jnp.bool_(True),
        ball_geometry=Ball(),
        reach=Reach(),
    )
    assert bool(np.asarray(possible))
