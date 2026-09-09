import jax
import numpy as np

from footballworld import (
    FootballWorld,
    Player,
    PlayerProfile,
    initialize_policy_state,
)
from footballworld.policies import make_rule_based_policy

FORMATION = (
    (-50.0, 0.0),
    (-35.0, 25.0),
    (-35.0, -25.0),
    (-35.0, 15.0),
    (-35.0, -15.0),
    (-20.0, 20.0),
    (-20.0, -20.0),
    (-20.0, 0.0),
    (-10.0, 25.0),
    (-10.0, -25.0),
    (-12.0, 0.0),
)


def _team(offset):
    return tuple(
        Player(
            PlayerProfile(player_id=offset + slot, is_goalkeeper=slot == 0),
            initial_position=position,
        )
        for slot, position in enumerate(FORMATION)
    )


def _assert_tree_equivalent(left, right):
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        if np.issubdtype(left_array.dtype, np.inexact):
            np.testing.assert_allclose(left_array, right_array, rtol=1e-6, atol=1e-6)
        else:
            np.testing.assert_array_equal(left_array, right_array)


def test_pass_diagnostic_path_preserves_lean_action_and_state():
    env = FootballWorld()
    reset = env.reset(_team(0), _team(100), key=jax.random.key(19))
    roster = env.roster_metadata_si(reset.rollout)
    policy = make_rule_based_policy(env)
    state = initialize_policy_state(env, policy, reset.rollout, roster)
    observations = env.observe_all_si(reset.rollout)
    match_key = jax.random.key(90210)

    lean = policy.step(observations, roster, state, match_key)
    event = policy.step_with_event_receipt(observations, roster, state, match_key)
    diagnostic = policy.step_with_pass_diagnostic(
        observations, roster, state, match_key
    )

    _assert_tree_equivalent(lean.action, event.action)
    _assert_tree_equivalent(lean.action, diagnostic.action)
    _assert_tree_equivalent(lean.state, event.state)
    _assert_tree_equivalent(lean.state, diagnostic.state)
    assert not hasattr(lean, "intended_receiver_ids")
    assert not hasattr(event, "pass_candidates")

    player_count = roster.player_id.shape[0]
    candidates = diagnostic.pass_candidates
    assert candidates.receiver_slot.shape == (2 * player_count,)
    assert candidates.target_xy.shape == (2 * player_count, 2)
    assert candidates.eligible.shape == (2 * player_count,)
    assert candidates.selected.shape == (2 * player_count,)
    assert np.asarray(candidates.decision_due).shape == ()
    assert np.asarray(diagnostic.carrier_slot).shape == ()
