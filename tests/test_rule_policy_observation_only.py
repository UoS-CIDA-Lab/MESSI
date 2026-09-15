"""Contracts for strict current-observation rule-policy decisions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from footballworld import FootballWorld, Player, PlayerProfile, initialize_policy_state
from footballworld.core.constants import NO_PLAYER, NO_TEAM, RK_KICKOFF, RK_NONE
from footballworld.policies import make_rule_based_policy
from footballworld.policies.rule_based.context import build_rule_policy_context
from footballworld.policies.rule_based.keeper_aerial import (
    AerialContestDecision,
    aerial_contest_decision,
)
from footballworld.policies.rule_based.manager import (
    RuleManagerState,
    make_rule_based_manager,
)
from footballworld.policies.rule_based.policy import _aerial_contest_if_possible

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


def _aerial_fast_path_inputs():
    env = FootballWorld()
    reset = env.reset(_team(5_000), _team(6_000), key=jax.random.key(75))
    roster = env.roster_metadata_si(reset.rollout)
    observations = env.observe_all_si(reset.rollout)
    context = build_rule_policy_context(observations, roster)._replace(
        self_active=jnp.ones_like(observations.valid),
        ball_visible=jnp.ones_like(observations.valid),
    )
    observers = roster.player_id.shape[0]
    target = jnp.zeros((observers, 2), dtype=jnp.float32)
    flight = jnp.ones((observers,), dtype=jnp.float32)
    excluded = jnp.zeros((observers, observers), dtype=jnp.bool_)
    kwargs = {
        "half_length_m": env.stadium.half_length,
        "excluded_player": excluded,
        "long_stamina_vmax_floor": env.long_stamina.vmax_floor,
        "short_stamina_vmax_floor": env.short_stamina.vmax_floor,
        "short_stamina_headroom_knee": env.short_stamina.headroom_knee,
    }
    return context, observations, roster, target, flight, kwargs


def _with_aerial_gate_state(context, observations, *, height, live, restart_kind):
    ball_position = context.ball_position.at[:, 2].set(jnp.float32(height))
    context = context._replace(ball_position=ball_position)
    observations = observations._replace(
        ball=observations.ball._replace(
            live=jnp.full_like(observations.ball.live, live)
        ),
        restart=observations.restart._replace(
            kind=jnp.full_like(observations.restart.kind, restart_kind)
        ),
    )
    return context, observations


def test_aerial_fast_path_skips_ground_boundary_dead_and_restart_rows(monkeypatch):
    import footballworld.policies.rule_based.policy as policy_module

    context, observations, roster, target, flight, kwargs = _aerial_fast_path_inputs()
    calls = []

    def should_not_execute(
        candidate_context,
        *_args,
        **_kwargs,
    ):
        jax.debug.callback(lambda _value: calls.append(1), jnp.int32(0))
        observers = candidate_context.self_index.shape[0]
        return AerialContestDecision(
            target=jnp.full((observers, 2), 99.0, dtype=jnp.float32),
            direct_runner=jnp.ones((observers,), dtype=jnp.bool_),
            cover_runner=jnp.ones((observers,), dtype=jnp.bool_),
            team_best_slot=jnp.zeros((observers,), dtype=jnp.int32),
            team_best_eta_s=jnp.zeros((observers,), dtype=jnp.float32),
            opponent_best_eta_s=jnp.zeros((observers,), dtype=jnp.float32),
            arrival_score=jnp.ones((observers,), dtype=jnp.float32),
        )

    monkeypatch.setattr(policy_module, "aerial_contest_decision", should_not_execute)

    def decide(candidate_context, candidate_observations):
        return _aerial_contest_if_possible(
            candidate_context,
            candidate_observations,
            roster,
            target,
            flight,
            **kwargs,
        )

    compiled = jax.jit(decide)
    cases = (
        (0.11, True, RK_NONE),
        (0.19, True, RK_NONE),
        (2.0, False, RK_NONE),
        (2.0, True, RK_KICKOFF),
    )
    for height, live, restart_kind in cases:
        candidate_context, candidate_observations = _with_aerial_gate_state(
            context,
            observations,
            height=height,
            live=live,
            restart_kind=restart_kind,
        )
        eager = decide(candidate_context, candidate_observations)
        compiled_result = compiled(candidate_context, candidate_observations)
        jax.block_until_ready(compiled_result.target)
        assert calls == []
        np.testing.assert_array_equal(eager.target, candidate_context.self_position)
        assert not bool(jnp.any(eager.direct_runner))
        assert not bool(jnp.any(eager.cover_runner))
        np.testing.assert_array_equal(
            eager.team_best_slot,
            jnp.full_like(eager.team_best_slot, NO_PLAYER),
        )
        _assert_tree_equivalent(eager, compiled_result)


def test_aerial_fast_path_retains_airborne_decision_eager_and_jit():
    context, observations, roster, target, flight, kwargs = _aerial_fast_path_inputs()
    context, observations = _with_aerial_gate_state(
        context,
        observations,
        height=2.0,
        live=True,
        restart_kind=RK_NONE,
    )

    def baseline(candidate_context, candidate_observations):
        return aerial_contest_decision(
            candidate_context,
            candidate_observations,
            roster,
            target,
            flight,
            **kwargs,
        )

    def candidate(candidate_context, candidate_observations):
        return _aerial_contest_if_possible(
            candidate_context,
            candidate_observations,
            roster,
            target,
            flight,
            **kwargs,
        )

    eager_baseline = baseline(context, observations)
    eager_candidate = candidate(context, observations)
    jit_baseline = jax.jit(baseline)(context, observations)
    jit_candidate = jax.jit(candidate)(context, observations)
    _assert_tree_equivalent(eager_baseline, eager_candidate, exact=False)
    _assert_tree_equivalent(jit_baseline, jit_candidate, exact=False)
    _assert_tree_equivalent(eager_candidate, jit_candidate, exact=False)
    assert bool(jnp.any(eager_candidate.direct_runner))


def test_ground_aerial_skip_retains_goalkeeper_cover_eager_and_jit(monkeypatch):
    import footballworld.policies.rule_based.policy as policy_module

    env = FootballWorld()
    reset = env.reset(_team(7_000), _team(8_000), key=jax.random.key(76))
    state = reset.rollout.state
    state = state._replace(
        ball=state.ball._replace(
            position=jnp.asarray((-35.0, 0.4, env.ball.radius), dtype=jnp.float32),
            velocity=jnp.asarray((-13.0, -0.1, 0.0), dtype=jnp.float32),
            live=jnp.bool_(True),
        ),
        possession=state.possession._replace(
            team=jnp.int32(NO_TEAM),
            player=jnp.int32(NO_PLAYER),
            control_ticks=jnp.int32(0),
        ),
        restart=state.restart._replace(
            kind=jnp.int32(RK_NONE),
            team=jnp.int32(NO_TEAM),
            substeps_remaining=jnp.int32(0),
            taker=jnp.int32(NO_PLAYER),
            indirect=jnp.bool_(False),
        ),
        restart_layout_ready=jnp.bool_(False),
    )
    rollout = reset.rollout._replace(state=state)
    observations = env.observe_all_si(rollout)
    roster = env.roster_metadata_si(rollout)
    policy = make_rule_based_policy(env)
    policy_state = initialize_policy_state(env, policy, rollout, roster)
    match_key = jax.random.key(77)
    guarded_aerial = policy_module._aerial_contest_if_possible
    original_aerial = policy_module.aerial_contest_decision

    def direct_aerial(*args, **kwargs):
        return original_aerial(*args, **kwargs)

    monkeypatch.setattr(policy_module, "_aerial_contest_if_possible", direct_aerial)
    eager_baseline = policy.step(observations, roster, policy_state, match_key)
    jit_baseline = jax.jit(policy.step)(observations, roster, policy_state, match_key)

    monkeypatch.setattr(policy_module, "_aerial_contest_if_possible", guarded_aerial)
    eager_candidate = policy.step(observations, roster, policy_state, match_key)
    jit_candidate = jax.jit(policy.step)(observations, roster, policy_state, match_key)

    _assert_tree_equivalent(eager_baseline, eager_candidate)
    _assert_tree_equivalent(jit_baseline, jit_candidate)
    _assert_tree_equivalent(eager_candidate, jit_candidate, exact=False)
    goalkeeper_rows = np.asarray(roster.is_goalkeeper, dtype=bool)
    goalkeeper_move = np.asarray(eager_candidate.action.move)[goalkeeper_rows]
    assert np.any(np.linalg.norm(goalkeeper_move, axis=-1) > 0.0)
