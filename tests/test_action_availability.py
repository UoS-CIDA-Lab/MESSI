"""Strict observation-side intent availability contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from footballworld import FootballWorld, Player, PlayerProfile, intent_availability_hint
from footballworld.core.action import IntentAction
from footballworld.core.constants import NO_PLAYER, NO_TEAM, RK_KICKOFF, RK_NONE
from footballworld.core.contact import (
    INTENT_CHALLENGE,
    INTENT_CLEAR,
    INTENT_CONTROL,
    INTENT_MOVE,
    INTENT_PASS,
    INTENT_SHOT,
    MECHANISM_FOOT,
    OUTCOME_INTERCEPTION,
    OUTCOME_RELEASE,
)
from footballworld.dynamics.contact import resolve_contact_step
from footballworld.dynamics.contact_predicates import evaluate_contact_predicates
from footballworld.dynamics.contest import sample_contest_override

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


def _assert_tree_equal(left, right):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(left_leaf, right_leaf)


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


def _open_play_at_distance(distance_m: float):
    env = FootballWorld()
    reset = env.reset(_team(1_000), _team(2_000), key=jax.random.key(23))
    state = reset.rollout.state
    actor = int(np.flatnonzero(np.asarray(state.players.team_id) == 0)[1])
    positions = state.players.position.at[actor].set(jnp.asarray((0.0, 0.0)))
    state = state._replace(
        players=state.players._replace(
            position=positions,
            velocity=jnp.zeros_like(state.players.velocity),
            contact_lock_substeps=jnp.zeros_like(state.players.contact_lock_substeps),
            aerial_recovery_substeps=jnp.zeros_like(
                state.players.aerial_recovery_substeps
            ),
            possession_loss_lock_substeps=jnp.zeros_like(
                state.players.possession_loss_lock_substeps
            ),
        ),
        ball=state.ball._replace(
            position=jnp.asarray((distance_m, 0.0, env.ball.radius)),
            velocity=jnp.zeros_like(state.ball.velocity),
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
            taker=jnp.int32(NO_PLAYER),
            substeps_remaining=jnp.int32(0),
        ),
    )
    rollout = reset.rollout._replace(state=state)
    return env, rollout, actor


def test_control_is_unavailable_ten_metres_from_ball_eager_and_jit():
    env, rollout, actor = _open_play_at_distance(10.0)
    observations = env.observe_all_si(rollout)
    roster = env.roster_metadata_si(rollout)

    eager = intent_availability_hint(observations, roster)
    compiled = jax.jit(intent_availability_hint)(observations, roster)

    assert not bool(eager[actor, INTENT_CONTROL])
    np.testing.assert_array_equal(
        eager[actor], np.arange(eager.shape[-1]) == INTENT_MOVE
    )
    np.testing.assert_array_equal(compiled, eager)


def test_near_loose_ball_allows_control_but_not_challenge():
    env, rollout, actor = _open_play_at_distance(1.0)
    availability = intent_availability_hint(
        env.observe_all_si(rollout), env.roster_metadata_si(rollout)
    )

    assert bool(availability[actor, INTENT_CONTROL])
    assert not bool(availability[actor, INTENT_CHALLENGE])


def test_near_verified_opponent_carrier_enables_challenge_alongside_move():
    env, rollout, actor = _open_play_at_distance(1.0)
    state = rollout.state
    actor_team = int(state.players.team_id[actor])
    opponent = int(np.flatnonzero(np.asarray(state.players.team_id) != actor_team)[1])
    players = state.players._replace(
        position=state.players.position.at[opponent].set(state.ball.position[:2])
    )
    state = state._replace(
        players=players,
        possession=state.possession._replace(
            team=state.players.team_id[opponent],
            player=jnp.int32(opponent),
            control_ticks=jnp.int32(2),
        ),
    )
    rollout = rollout._replace(state=state)

    availability = intent_availability_hint(
        env.observe_all_si(rollout), env.roster_metadata_si(rollout)
    )

    assert bool(availability[actor, INTENT_MOVE])
    assert bool(availability[actor, INTENT_CHALLENGE])
    for blocked_intent in (INTENT_CONTROL, INTENT_PASS, INTENT_SHOT, INTENT_CLEAR):
        assert not bool(availability[actor, blocked_intent])

    for blocked_intent in (INTENT_CONTROL, INTENT_PASS, INTENT_SHOT, INTENT_CLEAR):
        intents = (
            jnp.full(state.players.position.shape[0], INTENT_MOVE, dtype=jnp.int32)
            .at[actor]
            .set(blocked_intent)
        )
        predicates = evaluate_contact_predicates(
            state,
            jnp.zeros(state.players.position.shape[0], dtype=jnp.bool_),
            requested_intent=intents,
            ball_geometry=env.ball,
            stadium=env.stadium,
            reach=env.reach,
            scale=env.action_scale,
            body=env.body,
        )
        assert not bool(predicates.intent_allowed[actor])
        assert not bool(predicates.possible_now[actor])

    without_roster = intent_availability_hint(env.observe_all_si(rollout))
    np.testing.assert_array_equal(
        without_roster[actor],
        np.arange(without_roster.shape[-1]) == INTENT_MOVE,
    )

    recovering_state = state._replace(
        players=state.players._replace(
            challenge_recovery_substeps=(
                state.players.challenge_recovery_substeps.at[actor].set(1)
            )
        )
    )
    recovering_rollout = rollout._replace(state=recovering_state)
    recovering_availability = intent_availability_hint(
        env.observe_all_si(recovering_rollout),
        env.roster_metadata_si(recovering_rollout),
    )
    assert not bool(recovering_availability[actor, INTENT_CHALLENGE])

    player_count = state.players.position.shape[0]
    pass_action = IntentAction.neutral(player_count)._replace(
        intent=jnp.full(player_count, INTENT_MOVE, dtype=jnp.int32)
        .at[actor]
        .set(INTENT_PASS),
        force_to_ball=jnp.zeros((player_count, 2), dtype=jnp.float32)
        .at[actor]
        .set(jnp.asarray((1.0, 0.0), jnp.float32)),
    )

    def resolve_bypass(request):
        return resolve_contact_step(
            state,
            request,
            jax.random.key(26),
            sample_contest_override(),
            jnp.zeros(player_count, dtype=jnp.bool_),
            dt=0.0125,
            contact_attempted=jnp.zeros(player_count, dtype=jnp.bool_),
            ball_geometry=env.ball,
            stadium=env.stadium,
            reach=env.reach,
            scale=env.action_scale,
            body=env.body,
            timing=env.contact_timing,
            ball_physics=env.ball_physics,
            contest_config=env.contest,
        )

    eager_bypass = resolve_bypass(pass_action)
    compiled_bypass = jax.jit(resolve_bypass)(pass_action)
    assert int(eager_bypass.contact.actor) == NO_PLAYER
    _assert_tree_equal(compiled_bypass, eager_bypass)

    challenge_action = pass_action._replace(
        intent=pass_action.intent.at[actor].set(INTENT_CHALLENGE)
    )
    eager_tackle = resolve_bypass(challenge_action)
    compiled_tackle = jax.jit(resolve_bypass)(challenge_action)
    assert bool(eager_tackle.contest.selected)
    assert int(eager_tackle.contest.challenger) == actor
    _assert_tree_equal(compiled_tackle, eager_tackle)


def test_opponent_release_is_control_interception_not_challenge():
    env, rollout, actor = _open_play_at_distance(1.0)
    state = rollout.state
    actor_team = int(state.players.team_id[actor])
    opponent = int(np.flatnonzero(np.asarray(state.players.team_id) != actor_team)[1])
    state = state._replace(
        ball=state.ball._replace(
            velocity=jnp.asarray((-4.0, 0.0, 0.0), jnp.float32),
        ),
        possession=state.possession._replace(
            last_contact=state.possession.last_contact._replace(
                actor=jnp.int32(opponent),
                intent=jnp.int32(INTENT_PASS),
                mechanism=jnp.int32(MECHANISM_FOOT),
                outcome=jnp.int32(OUTCOME_RELEASE),
            ),
            previous_team=state.players.team_id[opponent],
        ),
    )
    rollout = rollout._replace(state=state)
    observation = env.observe_all_si(rollout)
    availability = intent_availability_hint(
        observation, env.roster_metadata_si(rollout)
    )

    assert bool(availability[actor, INTENT_MOVE])
    assert bool(availability[actor, INTENT_CONTROL])
    assert not bool(availability[actor, INTENT_CHALLENGE])

    control_intents = (
        jnp.full(state.players.position.shape[0], INTENT_MOVE, dtype=jnp.int32)
        .at[actor]
        .set(INTENT_CONTROL)
    )
    control_predicates = evaluate_contact_predicates(
        state,
        jnp.zeros(state.players.position.shape[0], dtype=jnp.bool_),
        requested_intent=control_intents,
        ball_geometry=env.ball,
        stadium=env.stadium,
        reach=env.reach,
        scale=env.action_scale,
        body=env.body,
    )
    assert bool(control_predicates.control_interception[actor])
    assert not bool(control_predicates.challenge_context[actor])

    challenge_intents = control_intents.at[actor].set(INTENT_CHALLENGE)
    challenge_predicates = evaluate_contact_predicates(
        state,
        jnp.zeros(state.players.position.shape[0], dtype=jnp.bool_),
        requested_intent=challenge_intents,
        ball_geometry=env.ball,
        stadium=env.stadium,
        reach=env.reach,
        scale=env.action_scale,
        body=env.body,
    )
    assert not bool(challenge_predicates.control_interception[actor])
    assert not bool(challenge_predicates.challenge_context[actor])
    assert not bool(challenge_predicates.possible_now[actor])

    player_count = state.players.position.shape[0]
    control_action = IntentAction.neutral(player_count)._replace(
        intent=jnp.full(player_count, INTENT_MOVE, dtype=jnp.int32)
        .at[actor]
        .set(INTENT_CONTROL),
        force_to_ball=jnp.zeros((player_count, 2), dtype=jnp.float32)
        .at[actor]
        .set(jnp.asarray((1.0, 0.0), jnp.float32)),
    )

    def resolve(request):
        return resolve_contact_step(
            state,
            request,
            jax.random.key(25),
            sample_contest_override(),
            jnp.zeros(player_count, dtype=jnp.bool_),
            dt=0.0125,
            contact_attempted=jnp.zeros(player_count, dtype=jnp.bool_),
            ball_geometry=env.ball,
            stadium=env.stadium,
            reach=env.reach,
            scale=env.action_scale,
            body=env.body,
            timing=env.contact_timing,
            ball_physics=env.ball_physics,
            contest_config=env.contest,
        )

    eager_control = resolve(control_action)
    compiled_control = jax.jit(resolve)(control_action)
    assert int(eager_control.contact.actor) == actor
    assert int(eager_control.contact.intent) == INTENT_CONTROL
    assert int(eager_control.contact.outcome) == OUTCOME_INTERCEPTION
    _assert_tree_equal(compiled_control, eager_control)

    chest_height = (
        state.players.height[actor] * env.action_scale.pelvis_height_factor
        + 2.0 * env.ball.radius
    )
    chest_state = state._replace(
        ball=state.ball._replace(position=state.ball.position.at[2].set(chest_height))
    )
    chest_action = control_action._replace(
        force_to_ball=jnp.zeros_like(control_action.force_to_ball)
    )

    def resolve_chest(request):
        return resolve_contact_step(
            chest_state,
            request,
            jax.random.key(27),
            sample_contest_override(),
            jnp.zeros(player_count, dtype=jnp.bool_),
            dt=0.0125,
            contact_attempted=jnp.zeros(player_count, dtype=jnp.bool_),
            ball_geometry=env.ball,
            stadium=env.stadium,
            reach=env.reach,
            scale=env.action_scale,
            body=env.body,
            timing=env.contact_timing,
            ball_physics=env.ball_physics,
            contest_config=env.contest,
        )

    eager_chest = resolve_chest(chest_action)
    compiled_chest = jax.jit(resolve_chest)(chest_action)
    assert int(eager_chest.contact.outcome) == OUTCOME_INTERCEPTION
    _assert_tree_equal(compiled_chest, eager_chest)

    challenge_action = control_action._replace(
        intent=control_action.intent.at[actor].set(INTENT_CHALLENGE)
    )
    eager_challenge = resolve(challenge_action)
    compiled_challenge = jax.jit(resolve)(challenge_action)
    assert int(eager_challenge.contact.actor) == NO_PLAYER
    _assert_tree_equal(compiled_challenge, eager_challenge)


def test_goalkeeper_hand_envelope_only_opens_control_and_clear_beyond_carry_reach():
    env = FootballWorld()
    reset = env.reset(_team(1_000), _team(2_000), key=jax.random.key(24))
    state = reset.rollout.state
    goalkeeper = int(
        np.flatnonzero(
            (np.asarray(state.players.team_id) == 0)
            & np.asarray(state.players.is_goalkeeper)
        )[0]
    )
    goalkeeper_xy = jnp.asarray((-45.0, 0.0), jnp.float32)
    state = state._replace(
        players=state.players._replace(
            position=state.players.position.at[goalkeeper].set(goalkeeper_xy),
            contact_lock_substeps=jnp.zeros_like(state.players.contact_lock_substeps),
            aerial_recovery_substeps=jnp.zeros_like(
                state.players.aerial_recovery_substeps
            ),
            possession_loss_lock_substeps=jnp.zeros_like(
                state.players.possession_loss_lock_substeps
            ),
        ),
        ball=state.ball._replace(
            position=jnp.asarray((-43.2, 0.0, 1.0), jnp.float32),
            velocity=jnp.zeros_like(state.ball.velocity),
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
            taker=jnp.int32(NO_PLAYER),
            substeps_remaining=jnp.int32(0),
        ),
    )
    rollout = reset.rollout._replace(state=state)

    availability = intent_availability_hint(
        env.observe_all_si(rollout), env.roster_metadata_si(rollout)
    )[goalkeeper]

    assert bool(availability[INTENT_MOVE])
    assert bool(availability[INTENT_CONTROL])
    assert bool(availability[INTENT_CLEAR])
    assert not bool(availability[INTENT_PASS])
    assert not bool(availability[INTENT_SHOT])
    assert not bool(availability[INTENT_CHALLENGE])


def test_release_taker_cannot_request_prohibited_second_contact_eager_and_jit():
    env, rollout, actor = _open_play_at_distance(1.0)
    state = rollout.state._replace(
        restart_release=rollout.state.restart_release._replace(
            active=jnp.bool_(True),
            untouched=jnp.bool_(True),
            kind=jnp.int32(RK_KICKOFF),
            team=rollout.state.players.team_id[actor],
            taker=jnp.int32(actor),
        )
    )
    rollout = rollout._replace(state=state)
    observations = env.observe_all_si(rollout)
    roster = env.roster_metadata_si(rollout)

    eager = intent_availability_hint(observations, roster)
    compiled = jax.jit(intent_availability_hint)(observations, roster)

    np.testing.assert_array_equal(
        eager[actor], np.arange(eager.shape[-1]) == INTENT_MOVE
    )
    np.testing.assert_array_equal(compiled, eager)

    touched = rollout._replace(
        state=state._replace(
            restart_release=state.restart_release._replace(untouched=jnp.bool_(False))
        )
    )
    after_touch = intent_availability_hint(
        env.observe_all_si(touched), env.roster_metadata_si(touched)
    )
    assert bool(after_touch[actor, INTENT_CONTROL])


def test_batched_environment_rosters_broadcast_before_observer_axis_under_jit():
    env, rollout, actor = _open_play_at_distance(1.0)
    observations = env.observe_all_si(rollout)
    roster = env.roster_metadata_si(rollout)
    batched_observations = jax.tree.map(
        lambda value: jnp.stack((value, value, value), axis=0), observations
    )
    batched_roster = jax.tree.map(
        lambda value: jnp.stack((value, value, value), axis=0), roster
    )

    eager = intent_availability_hint(batched_observations, batched_roster)
    compiled = jax.jit(intent_availability_hint)(batched_observations, batched_roster)

    assert eager.shape == (3, roster.team_id.shape[0], 6)
    assert bool(eager[2, actor, INTENT_CONTROL])
    assert not bool(eager[2, actor, INTENT_CHALLENGE])
    np.testing.assert_array_equal(compiled, eager)


def test_full_view_observation_omits_redundant_public_leaves() -> None:
    env = FootballWorld()
    reset = env.reset(_team(1_000), _team(2_000), key=jax.random.key(31))
    observation = env.observe(reset.rollout, jnp.int32(0))
    spec = env.player_observation_spec(observation)

    assert "visible" not in observation.players._fields
    assert "visible" not in observation.ball._fields
    assert "known" not in observation.possession._fields
    assert "known" not in observation.possession.last_contact._fields
    assert "known" not in observation.restart_release._fields
    assert "gk_handling_restriction_known" not in observation.match._fields
    assert "team" not in observation.possession._fields
    assert "team" not in observation.restart_release._fields
    assert "attack_direction" not in observation.match._fields
    assert "kickoff_team" not in observation.match._fields
    assert spec.semantic_version == 10
    assert spec.float32_size == 288
    assert spec.int32_size == 39
    assert spec.boolean_size == 185
