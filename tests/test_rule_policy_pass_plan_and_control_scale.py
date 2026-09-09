"""Focused contracts for receiver-plan handoff and CONTROL power units.

SoccerWorld keeps one deterministic trajectory runner for an observed pass and
tests that its action and trace agree.  FootballWorld extends that sound
single-runner principle with causal, observer-row policy state: the receiver
selected by a submitted PASS must be installed before the next observation can
fall back to the generic loose-ball interceptor.

SoccerWorld's absolute fast-ball dribble gate is intentionally not copied here.
FootballWorld's CONTROL request is player-relative and has its own public speed
scale, so this file instead protects the normalized CONTROL-unit contract.
Neither policy coefficient is asserted to be a measured football constant.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from footballworld import FootballWorld, Player, PlayerProfile, initialize_policy_state
from footballworld.core.constants import NO_PLAYER, NO_TEAM, RK_NONE
from footballworld.core.contact import (
    INTENT_CONTROL,
    INTENT_MOVE,
    INTENT_PASS,
    INTENT_SOURCE_POLICY,
    LAW11_DELIBERATE_PLAY_RESET,
    MECHANISM_FOOT,
    OUTCOME_PARRY,
    OUTCOME_TRAP,
)
from footballworld.policies import make_rule_based_policy
from footballworld.policies.rule_based.config import RulePolicyConfig
from footballworld.policies.rule_based.contact_timing import dribble_recontact_ready
from footballworld.policies.rule_based.possession import (
    POSSESSION_DRIBBLE,
    POSSESSION_PASS,
    PossessionDecision,
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


def _controlled_open_play(env: FootballWorld):
    """Return a public-observation scene with one stationary team-0 carrier."""

    reset = env.reset(_team(1_000), _team(2_000), key=jax.random.key(6100))
    state = reset.rollout.state
    team = 0
    team_mask = np.asarray(state.players.team_id) == team
    outfield = team_mask & (~np.asarray(state.players.is_goalkeeper, dtype=bool))
    actor, receiver = (int(slot) for slot in np.flatnonzero(outfield)[-2:])
    attack = float(np.asarray(state.attack_direction[team]))

    positions = np.array(state.players.position, copy=True)
    positions[actor] = (0.0, 0.0)
    positions[receiver] = (12.0 * attack, 0.0)
    # Keep every other player away from the selected pass and contact radius.
    for ordinal, slot in enumerate(
        index for index in range(positions.shape[0]) if index not in (actor, receiver)
    ):
        positions[slot] = (
            (-38.0 + 2.0 * (ordinal % 5)) * attack,
            -28.0 + 5.5 * (ordinal % 10),
        )

    players = state.players._replace(
        position=jnp.asarray(positions, dtype=jnp.float32),
        velocity=jnp.zeros_like(state.players.velocity),
    )
    last_contact = state.possession.last_contact._replace(
        actor=jnp.int32(actor),
        mechanism=jnp.int32(MECHANISM_FOOT),
        intent=jnp.int32(INTENT_CONTROL),
        outcome=jnp.int32(OUTCOME_TRAP),
        restart_kind=jnp.int32(RK_NONE),
        law11_effect=jnp.int32(LAW11_DELIBERATE_PLAY_RESET),
        kick_applied=jnp.bool_(False),
        intent_source=jnp.int32(INTENT_SOURCE_POLICY),
    )
    possession = state.possession._replace(
        team=jnp.int32(team),
        player=jnp.int32(actor),
        previous_team=jnp.int32(team),
        control_ticks=jnp.int32(2),
        last_contact=last_contact,
    )
    restart = state.restart._replace(
        kind=jnp.int32(RK_NONE),
        team=jnp.int32(NO_TEAM),
        substeps_remaining=jnp.int32(0),
        taker=jnp.int32(NO_PLAYER),
        indirect=jnp.bool_(False),
    )
    state = state._replace(
        ball=state.ball._replace(
            position=jnp.asarray(
                (positions[actor, 0], positions[actor, 1], env.ball.radius),
                dtype=jnp.float32,
            ),
            velocity=jnp.zeros(3, dtype=jnp.float32),
            spin=jnp.zeros(3, dtype=jnp.float32),
            live=jnp.bool_(True),
        ),
        players=players,
        possession=possession,
        restart=restart,
        restart_layout_ready=jnp.bool_(False),
    )
    return reset._replace(rollout=reset.rollout._replace(state=state)), actor, receiver


def _forced_decision(kind: int, receiver: int = NO_PLAYER):
    def decide(*_args, **_kwargs):
        return PossessionDecision(
            direction=jnp.asarray((1.0, 0.0), dtype=jnp.float32),
            power=jnp.float32(1.0),
            launch=jnp.float32(-1.0),
            spin=jnp.zeros(2, dtype=jnp.float32),
            cross=jnp.bool_(False),
            kind=jnp.int32(kind),
            target=jnp.int32(receiver),
        )

    return decide


def test_submitted_pass_seeds_team_receiver_plan_and_retains_it_in_flight(
    monkeypatch,
):
    """The current PASS receiver wins over next-frame generic interception."""

    import footballworld.policies.rule_based.policy as policy_module

    env = FootballWorld()
    reset, actor, receiver = _controlled_open_play(env)
    monkeypatch.setattr(
        policy_module,
        "decide_possession",
        _forced_decision(POSSESSION_PASS, receiver),
    )
    roster = env.roster_metadata_si(reset.rollout)
    policy = make_rule_based_policy(env)
    observations = env.observe_all_si(reset.rollout)
    state = initialize_policy_state(env, policy, reset.rollout, roster)

    submitted = policy.step_with_event_receipt(
        observations,
        roster,
        state,
        jax.random.key(6101),
    )
    player_team = np.asarray(roster.team_id)
    valid = np.asarray(observations.valid, dtype=bool)
    passer_team = int(player_team[actor])
    teammate_observers = valid & (player_team == passer_team)
    opponent_observers = valid & (player_team != passer_team)
    receiver_id = int(np.asarray(roster.player_id[receiver]))

    assert int(np.asarray(submitted.action.intent[actor])) == INTENT_PASS
    assert int(np.asarray(submitted.intended_receiver_ids[actor])) == receiver_id
    np.testing.assert_array_equal(
        np.asarray(submitted.state.planned_receiver)[teammate_observers],
        np.full(np.count_nonzero(teammate_observers), receiver, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(submitted.state.planned_receiver_id)[teammate_observers],
        np.full(np.count_nonzero(teammate_observers), receiver_id, dtype=np.int32),
    )
    assert np.all(np.asarray(submitted.state.planned_eta_ticks)[teammate_observers] > 0)
    np.testing.assert_array_equal(
        np.asarray(submitted.state.planned_receiver)[opponent_observers],
        np.full(np.count_nonzero(opponent_observers), NO_PLAYER, dtype=np.int32),
    )

    released = env.step(
        reset.rollout,
        reset.setup,
        submitted.action,
        jax.random.key(6102),
    )
    assert bool(np.asarray(released.kick_applied[actor]))
    assert int(np.asarray(released.rollout.state.possession.team)) == NO_TEAM

    flight_observations = env.observe_all_si(released.rollout)
    continued = policy.step(
        flight_observations,
        roster,
        submitted.state,
        jax.random.key(6103),
    )
    flight_teammates = np.asarray(flight_observations.valid, dtype=bool) & (
        player_team == passer_team
    )
    np.testing.assert_array_equal(
        np.asarray(continued.state.planned_receiver)[flight_teammates],
        np.full(np.count_nonzero(flight_teammates), receiver, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(continued.state.planned_receiver_id)[flight_teammates],
        np.full(np.count_nonzero(flight_teammates), receiver_id, dtype=np.int32),
    )

    # Retain the declared receiver, not the release-time endpoint.  On a live
    # ground pass the receiver must attack the causal trajectory and meet the
    # ball earlier instead of waiting for an opponent to step into the lane.
    released_state = released.rollout.state
    receiver_position = np.asarray(released_state.players.position[receiver])
    ball_delta = np.asarray(released_state.ball.position[:2]) - receiver_position
    release_target = np.asarray(submitted.state.planned_arrival[receiver])
    live_target = np.asarray(continued.state.planned_arrival[receiver])
    decoded = continued.action.decode()
    receiver_move = np.asarray(decoded.move.direction[receiver])
    assert np.linalg.norm(live_target - release_target) > 1.0
    assert float(np.dot(live_target - receiver_position, ball_delta)) > 0.0
    assert float(np.dot(receiver_move, ball_delta)) > 0.0
    assert float(np.asarray(decoded.move.power[receiver])) > 0.0


def test_carrier_control_power_uses_the_normalized_control_scale(monkeypatch):
    """A dribble CONTROL must not be rescaled through the kick-speed limit."""

    import footballworld.policies.rule_based.policy as policy_module

    env = FootballWorld()
    reset, actor, _ = _controlled_open_play(env)
    monkeypatch.setattr(
        policy_module,
        "decide_possession",
        _forced_decision(POSSESSION_DRIBBLE),
    )
    config = RulePolicyConfig(dribble_power=0.037, dribble_touch_interval_s=0.1)
    roster = env.roster_metadata_si(reset.rollout)
    policy = make_rule_based_policy(env, config)
    state = initialize_policy_state(env, policy, reset.rollout, roster)
    result = policy.step(
        env.observe_all_si(reset.rollout),
        roster,
        state,
        jax.random.key(6104),
    )
    decoded = result.action.decode()
    actual_power = float(np.asarray(decoded.force_to_ball.power[actor]))

    assert int(np.asarray(result.action.intent[actor])) == INTENT_CONTROL
    np.testing.assert_allclose(actual_power, config.dribble_power, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(
        actual_power * env.action_scale.control_request_speed_max_mps,
        config.dribble_power * env.action_scale.control_request_speed_max_mps,
        rtol=0.0,
        atol=1e-6,
    )
    kick_rescaled_power = (
        config.dribble_power
        * env.action_scale.kick_speed_max_mps
        / env.action_scale.control_request_speed_max_mps
    )
    assert not np.isclose(actual_power, kick_rescaled_power, rtol=0.0, atol=1e-7)


def test_misaligned_carrier_recovers_toward_ball_without_another_control(
    monkeypatch,
):
    """A reachable caught ball behind the dribble axis triggers recovery MOVE."""

    import footballworld.policies.rule_based.policy as policy_module

    env = FootballWorld()
    reset, actor, _ = _controlled_open_play(env)
    state = reset.rollout.state
    team = int(np.asarray(state.players.team_id[actor]))
    attack = float(np.asarray(state.attack_direction[team]))
    player_position = np.asarray(state.players.position[actor])
    behind_offset_world = np.asarray((-0.45 * attack, 0.0), dtype=np.float32)
    state = state._replace(
        ball=state.ball._replace(
            position=jnp.asarray(
                (
                    player_position[0] + behind_offset_world[0],
                    player_position[1] + behind_offset_world[1],
                    env.ball.radius,
                ),
                dtype=jnp.float32,
            )
        )
    )
    reset = reset._replace(rollout=reset.rollout._replace(state=state))
    monkeypatch.setattr(
        policy_module,
        "decide_possession",
        _forced_decision(POSSESSION_DRIBBLE),
    )
    roster = env.roster_metadata_si(reset.rollout)
    policy = make_rule_based_policy(env)
    observations = env.observe_all_si(reset.rollout)
    policy_state = initialize_policy_state(env, policy, reset.rollout, roster)

    ball_offset = observations.ball.relative_state[actor, :2]
    relative_velocity = observations.ball.relative_state[actor, 3:5]
    ball_velocity = relative_velocity + observations.self_state.velocity[actor]
    ball_speed = jnp.linalg.norm(ball_velocity)
    self_touched_last = observations.players.last_actor[actor, actor]
    ready = dribble_recontact_ready(
        ball_offset,
        relative_velocity,
        ball_speed,
        self_touched_last,
        observations.possession.last_contact.known[actor],
        observations.possession.last_contact.intent[actor],
        observations.possession.last_contact.outcome[actor],
    )
    distance = float(np.linalg.norm(np.asarray(ball_offset)))

    assert distance < env.reach.carry_radius_m + env.ball.radius
    assert bool(np.asarray(ready))
    # The forced final dribble direction is positive attack-local x.
    assert float(np.asarray(ball_offset[0])) < 0.0

    result = policy.step(
        observations,
        roster,
        policy_state,
        jax.random.key(6105),
    )
    decoded = result.action.decode()
    move_direction = np.asarray(decoded.move.direction[actor])
    ball_direction = np.asarray(ball_offset) / distance

    assert int(np.asarray(result.action.intent[actor])) == INTENT_MOVE
    assert not bool(np.asarray(decoded.contact[actor]))
    assert float(np.dot(move_direction, ball_direction)) > 0.999


def test_settling_control_is_not_a_pass_runner_or_goalkeeper_sweep():
    """One same-actor settling touch has one recovery runner, not a swarm."""

    env = FootballWorld()
    reset, actor, _ = _controlled_open_play(env)
    controlled = reset.rollout.state
    team = int(np.asarray(controlled.players.team_id[actor]))
    goalkeeper = int(
        np.flatnonzero(
            (np.asarray(controlled.players.team_id) == team)
            & np.asarray(controlled.players.is_goalkeeper, dtype=bool)
        )[0]
    )
    positions = np.array(controlled.players.position, copy=True)
    positions[goalkeeper] = (-40.0, -6.0)
    positions[actor] = (-36.5, -8.0)
    players = controlled.players._replace(
        position=jnp.asarray(positions, dtype=jnp.float32),
        velocity=jnp.zeros_like(controlled.players.velocity),
    )
    controlled = controlled._replace(
        players=players,
        ball=controlled.ball._replace(
            position=jnp.asarray((-37.9, -8.2, env.ball.radius), dtype=jnp.float32),
            velocity=jnp.asarray((-2.8, -0.6, 0.0), dtype=jnp.float32),
        ),
    )
    controlled_reset = reset._replace(rollout=reset.rollout._replace(state=controlled))
    roster = env.roster_metadata_si(controlled_reset.rollout)
    policy = make_rule_based_policy(env)
    policy_state = initialize_policy_state(
        env, policy, controlled_reset.rollout, roster
    )
    player_team = np.asarray(roster.team_id)
    team_observers = jnp.asarray(player_team == team, dtype=jnp.bool_)
    stale_receiver = int(
        np.flatnonzero(
            (player_team == team)
            & (~np.asarray(roster.is_goalkeeper, dtype=bool))
            & (np.arange(player_team.size) != actor)
        )[0]
    )
    policy_state = policy_state._replace(
        planned_receiver=jnp.where(
            team_observers, jnp.int32(stale_receiver), policy_state.planned_receiver
        ),
        planned_receiver_id=jnp.where(
            team_observers,
            roster.player_id[stale_receiver],
            policy_state.planned_receiver_id,
        ),
        planned_arrival=jnp.where(
            team_observers[:, None],
            jnp.asarray((12.0, 8.0), dtype=jnp.float32),
            policy_state.planned_arrival,
        ),
        planned_eta_ticks=jnp.where(
            team_observers, jnp.int32(20), policy_state.planned_eta_ticks
        ),
    )

    loose = controlled._replace(
        possession=controlled.possession._replace(
            team=jnp.int32(NO_TEAM),
            player=jnp.int32(NO_PLAYER),
            previous_team=jnp.int32(team),
            control_ticks=jnp.int32(0),
        )
    )
    loose_rollout = controlled_reset.rollout._replace(state=loose)
    observations = env.observe_all_si(loose_rollout)
    result = policy.step(
        observations,
        roster,
        policy_state,
        jax.random.key(6106),
    )
    decoded = result.action.decode()

    # CONTROL lineage must not exclude the original actor as though it were a
    # PASS release. The nearby actor, rather than a second teammate, recovers.
    assert int(np.asarray(result.state.loose_chaser[actor])) == actor
    np.testing.assert_array_equal(
        np.asarray(result.state.planned_receiver)[player_team == team],
        np.full(np.count_nonzero(player_team == team), NO_PLAYER, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(result.state.planned_receiver_id)[player_team == team],
        np.full(np.count_nonzero(player_team == team), NO_PLAYER, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(result.state.planned_eta_ticks)[player_team == team],
        np.zeros(np.count_nonzero(player_team == team), dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(result.state.planned_arrival)[player_team == team],
        0.0,
        rtol=0.0,
        atol=0.0,
    )
    # The ball is moving toward the own half but outside the projected goal
    # mouth. The keeper returns goalward instead of joining that recovery.
    assert int(np.asarray(result.action.intent[goalkeeper])) == INTENT_MOVE
    assert float(np.asarray(decoded.move.direction[goalkeeper, 0])) < 0.0

    # After a keeper parry the control lineage is gone, but the visibly nearer
    # outfielder still owns the recovery. The keeper must not rejoin the same
    # point merely because the ball remains physically loose.
    parry_contact = loose.possession.last_contact._replace(
        actor=jnp.int32(goalkeeper),
        outcome=jnp.int32(OUTCOME_PARRY),
    )
    parried = loose._replace(
        possession=loose.possession._replace(last_contact=parry_contact)
    )
    parried_rollout = loose_rollout._replace(state=parried)
    parried_result = policy.step(
        env.observe_all_si(parried_rollout),
        roster,
        policy_state,
        jax.random.key(6107),
    )
    parried_move = parried_result.action.decode().move.direction
    assert int(np.asarray(parried_result.state.loose_chaser[actor])) == actor
    assert int(np.asarray(parried_result.action.intent[goalkeeper])) == INTENT_MOVE
    assert float(np.asarray(parried_move[goalkeeper, 0])) < 0.0
