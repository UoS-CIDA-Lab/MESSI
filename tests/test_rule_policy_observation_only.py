"""Contracts for strict current-observation rule-policy decisions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from footballworld import FootballWorld, Player, PlayerProfile, initialize_policy_state
from footballworld.policies import make_rule_based_policy
from footballworld.policies.rule_based.manager import (
    RuleManagerState,
    make_rule_based_manager,
)

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


def _assert_tree_equivalent(left, right, *, exact: bool = True) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        if exact or not np.issubdtype(left_array.dtype, np.inexact):
            np.testing.assert_array_equal(left_array, right_array)
        else:
            np.testing.assert_allclose(
                left_array, right_array, rtol=1.0e-6, atol=1.0e-7
            )


def test_previous_dynamic_policy_state_cannot_change_eager_or_jit_decision():
    """Only explicit static tactics may survive between policy decisions."""

    env = FootballWorld()
    reset = env.reset(_team(1_000), _team(2_000), key=jax.random.key(72))
    roster = env.roster_metadata_si(reset.rollout)
    observations = env.observe_all_si(reset.rollout)
    policy = make_rule_based_policy(env)
    clean = initialize_policy_state(env, policy, reset.rollout, roster)
    count = roster.player_id.shape[0]
    index = jnp.arange(count, dtype=jnp.int32)
    stale = clean._replace(
        restart_kind=index + jnp.int32(100),
        restart_age=index + jnp.int32(200),
        possession_team=index % jnp.int32(2),
        possession_age=index + jnp.int32(300),
        carrier_age=index + jnp.int32(400),
        attack_phase=index % jnp.int32(3),
        current_possessor=(index + jnp.int32(1)) % jnp.int32(count),
        previous_possessor=(index + jnp.int32(2)) % jnp.int32(count),
        counterpress_age=index + jnp.int32(500),
        loose_chaser=(index + jnp.int32(3)) % jnp.int32(count),
        planned_receiver=(index + jnp.int32(4)) % jnp.int32(count),
        planned_receiver_id=index + jnp.int32(10_000),
        planned_arrival=jnp.stack(
            (index.astype(jnp.float32), -index.astype(jnp.float32)), axis=-1
        ),
        planned_eta_ticks=index + jnp.int32(600),
        service_opportunity=jnp.ones((count,), dtype=jnp.bool_),
        secure_control_age=index + jnp.int32(700),
        last_control_tick=index - jnp.int32(800),
    )
    match_key = jax.random.key(73)

    eager_clean = policy.step(observations, roster, clean, match_key)
    eager_stale = policy.step(observations, roster, stale, match_key)
    _assert_tree_equivalent(eager_clean, eager_stale)

    compiled_step = jax.jit(policy.step)
    jit_clean = compiled_step(observations, roster, clean, match_key)
    jit_stale = compiled_step(observations, roster, stale, match_key)
    _assert_tree_equivalent(jit_clean, jit_stale)
    _assert_tree_equivalent(eager_clean, jit_clean, exact=False)


def test_previous_manager_state_cannot_change_eager_or_jit_command():
    env = FootballWorld()
    key = jax.random.key(74)
    reset = env.reset(_team(3_000), _team(4_000), key=key)
    initialized = env.initialize_management(reset.rollout, (), ())
    observations = env.observe_managers(
        reset.rollout, initialized.squad, initialized.state
    )
    manager = make_rule_based_manager(env)
    clean = manager.initialize(observations)
    stale = RuleManagerState(
        processed_restart_tick=jnp.asarray([123, 456], dtype=jnp.int32),
        processed_restart_kind=jnp.asarray([7, 8], dtype=jnp.int32),
        formation_change_tick=jnp.asarray([789, 987], dtype=jnp.int32),
    )

    eager_clean = manager.step(observations, key, clean)
    eager_stale = manager.step(observations, key, stale)
    _assert_tree_equivalent(eager_clean.command, eager_stale.command)

    compiled = jax.jit(manager.step)
    jit_clean = compiled(observations, key, clean)
    jit_stale = compiled(observations, key, stale)
    _assert_tree_equivalent(jit_clean.command, jit_stale.command)
    _assert_tree_equivalent(eager_clean.command, jit_clean.command, exact=False)
