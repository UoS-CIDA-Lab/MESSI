"""Focused Law 12 contracts for a team-mate pass to the goalkeeper."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from footballworld import FootballWorld, Player, PlayerProfile
from footballworld.config.ball_physics import BallPhysics
from footballworld.config.geometry import Ball
from footballworld.core.constants import NO_PLAYER, NO_TEAM, RK_NONE
from footballworld.core.contact import (
    INTENT_CLEAR,
    INTENT_CONTROL,
    INTENT_MOVE,
    MECHANISM_FOOT,
    MECHANISM_GOALKEEPER_HAND,
)
from footballworld.dynamics.contact_predicates import evaluate_contact_predicates
from footballworld.rules.gk_handling_restriction import (
    _conservative_path_length,
    targeted_own_goalkeeper_team,
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


def _open_play_state():
    env = FootballWorld()
    reset = env.reset(_team(1_000), _team(2_000), key=jax.random.key(912))
    state = reset.rollout.state
    state = state._replace(
        ball=state.ball._replace(live=jnp.bool_(True)),
        possession=state.possession._replace(
            team=jnp.int32(NO_TEAM),
            player=jnp.int32(NO_PLAYER),
        ),
        restart=state.restart._replace(
            kind=jnp.int32(RK_NONE),
            team=jnp.int32(NO_TEAM),
            substeps_remaining=jnp.int32(0),
            taker=jnp.int32(NO_PLAYER),
            indirect=jnp.bool_(False),
        ),
        restart_release=state.restart_release._replace(active=jnp.bool_(False)),
    )
    return env, reset, state


def test_fast_ground_pass_path_uses_the_whole_rolling_table_and_arms_law_12():
    """Protect the seed-29 56.6 s pass geometry that exposed the short proxy."""

    _, _, state = _open_play_state()
    team = 0
    team_slots = np.flatnonzero(np.asarray(state.players.team_id) == team)
    goalkeeper = int(team_slots[np.asarray(state.players.is_goalkeeper)[team_slots]][0])
    actor = int(team_slots[team_slots != goalkeeper][0])
    positions = np.asarray(state.players.position).copy()
    positions[actor] = (-29.72137, 15.72850)
    positions[goalkeeper] = (-50.94918, 0.86372)
    state = state._replace(
        players=state.players._replace(
            position=jnp.asarray(positions, dtype=jnp.float32)
        )
    )
    release = jnp.asarray((-29.72137, 15.72850, 0.11), dtype=jnp.float32)
    velocity = jnp.asarray((-12.393, -8.463, 0.0), dtype=jnp.float32)

    # The former single-deceleration estimate was about 12.2 m even though
    # this configured rolling law carries the pass beyond the goalkeeper ray.
    assert (
        float(
            _conservative_path_length(
                release,
                velocity,
                jnp.zeros(3, dtype=jnp.float32),
                ball=Ball(),
                physics=BallPhysics(),
            )
        )
        > 25.0
    )
    spin = jnp.zeros(3, dtype=jnp.float32)
    eager = targeted_own_goalkeeper_team(state, actor, release, velocity, spin)
    compiled = jax.jit(targeted_own_goalkeeper_team)(
        state,
        jnp.int32(actor),
        release,
        velocity,
        spin,
    )
    assert int(eager) == team
    assert int(compiled) == team


def test_supported_ground_slip_extends_fast_pass_path_before_rolling():
    """Protect the zero-spin slide phase seen in the second 25-match audit."""

    _, _, state = _open_play_state()
    team = 0
    team_slots = np.flatnonzero(np.asarray(state.players.team_id) == team)
    goalkeeper = int(team_slots[np.asarray(state.players.is_goalkeeper)[team_slots]][0])
    actor = int(team_slots[team_slots != goalkeeper][0])
    positions = np.asarray(state.players.position).copy()
    positions[actor] = (-5.998, 3.245)
    positions[goalkeeper] = (-49.640, 0.260)
    state = state._replace(
        players=state.players._replace(
            position=jnp.asarray(positions, dtype=jnp.float32)
        )
    )
    release = jnp.asarray((-5.998, 3.245, 0.11), dtype=jnp.float32)
    velocity = jnp.asarray((-19.744, -1.438, 0.0), dtype=jnp.float32)

    # A rolling-only estimate is about 38.5 m. The environment first slides a
    # zero-spin release, carrying this pass about 62.5 m in total.
    path_length = float(
        _conservative_path_length(
            release,
            velocity,
            jnp.zeros(3, dtype=jnp.float32),
            ball=Ball(),
            physics=BallPhysics(),
        )
    )
    assert 62.0 < path_length < 63.0
    assert (
        int(
            targeted_own_goalkeeper_team(
                state,
                actor,
                release,
                velocity,
                jnp.zeros(3, dtype=jnp.float32),
            )
        )
        == team
    )


def test_goalkeeper_movement_can_reach_the_end_of_a_targeted_pass_ray():
    """A moving receiver must not make a deliberate back-pass hand-legal."""

    _, _, state = _open_play_state()
    team = 0
    team_slots = np.flatnonzero(np.asarray(state.players.team_id) == team)
    goalkeeper = int(team_slots[np.asarray(state.players.is_goalkeeper)[team_slots]][0])
    actor = int(team_slots[team_slots != goalkeeper][0])
    positions = np.asarray(state.players.position).copy()
    positions[actor] = (0.0, 0.0)
    positions[goalkeeper] = (-46.0, 0.0)
    state = state._replace(
        players=state.players._replace(
            position=jnp.asarray(positions, dtype=jnp.float32)
        )
    )
    release = jnp.asarray((0.0, 0.0, 0.11), dtype=jnp.float32)
    velocity = jnp.asarray((-15.0, 0.0, 0.0), dtype=jnp.float32)
    spin = jnp.zeros(3, dtype=jnp.float32)

    path_length = float(
        _conservative_path_length(
            release,
            velocity,
            spin,
            ball=Ball(),
            physics=BallPhysics(),
        )
    )
    assert 42.0 < path_length < 43.0
    assert 46.0 - path_length > Ball().radius
    assert (
        int(targeted_own_goalkeeper_team(state, actor, release, velocity, spin)) == team
    )


def test_short_or_laterally_missed_team_kick_does_not_arm_backpass_restriction():
    _, _, state = _open_play_state()
    team = 0
    team_slots = np.flatnonzero(np.asarray(state.players.team_id) == team)
    goalkeeper = int(team_slots[np.asarray(state.players.is_goalkeeper)[team_slots]][0])
    actor = int(team_slots[team_slots != goalkeeper][0])
    positions = np.asarray(state.players.position).copy()
    positions[actor] = (-29.72137, 15.72850)
    positions[goalkeeper] = (-50.94918, 0.86372)
    state = state._replace(
        players=state.players._replace(
            position=jnp.asarray(positions, dtype=jnp.float32)
        )
    )
    release = jnp.asarray((-29.72137, 15.72850, 0.11), dtype=jnp.float32)

    short = targeted_own_goalkeeper_team(
        state,
        actor,
        release,
        jnp.asarray((-2.0, -1.36, 0.0), dtype=jnp.float32),
        jnp.zeros(3, dtype=jnp.float32),
    )
    lateral_miss = targeted_own_goalkeeper_team(
        state,
        actor,
        release,
        jnp.asarray((-12.393, 8.463, 0.0), dtype=jnp.float32),
        jnp.zeros(3, dtype=jnp.float32),
    )
    assert int(short) == NO_TEAM
    assert int(lateral_miss) == NO_TEAM


def test_restricted_goalkeeper_clear_and_control_stay_foot_only():
    env, reset, state = _open_play_state()
    team = 0
    team_slots = np.flatnonzero(np.asarray(state.players.team_id) == team)
    goalkeeper = int(team_slots[np.asarray(state.players.is_goalkeeper)[team_slots]][0])
    attack = float(np.asarray(state.attack_direction[team]))
    goalkeeper_position = np.asarray(
        (-attack * env.stadium.half_length + attack * 6.0, 2.0),
        dtype=np.float32,
    )
    positions = np.asarray(state.players.position).copy()
    positions[goalkeeper] = goalkeeper_position
    state = state._replace(
        ball=state.ball._replace(
            position=jnp.asarray(
                (*goalkeeper_position, env.ball.radius), dtype=jnp.float32
            ),
            velocity=jnp.zeros(3, dtype=jnp.float32),
        ),
        players=state.players._replace(
            position=jnp.asarray(positions, dtype=jnp.float32)
        ),
        gk_backpass_team=jnp.int32(team),
    )

    for intent in (INTENT_CONTROL, INTENT_CLEAR):
        requested = jnp.full(state.players.position.shape[0], INTENT_MOVE, jnp.int32)
        requested = requested.at[goalkeeper].set(intent)
        predicates = evaluate_contact_predicates(
            state,
            jnp.bool_(True),
            requested_intent=requested,
        )
        assert int(predicates.mechanism[goalkeeper]) == MECHANISM_FOOT
        assert not bool(predicates.goalkeeper_claim[goalkeeper])
        assert bool(predicates.possible_now[goalkeeper])

    unrestricted = state._replace(gk_backpass_team=jnp.int32(NO_TEAM))
    requested = jnp.full(state.players.position.shape[0], INTENT_MOVE, jnp.int32)
    requested = requested.at[goalkeeper].set(INTENT_CLEAR)
    legal_hand_clear = evaluate_contact_predicates(
        unrestricted,
        jnp.bool_(True),
        requested_intent=requested,
    )
    assert int(legal_hand_clear.mechanism[goalkeeper]) == MECHANISM_GOALKEEPER_HAND
    assert bool(legal_hand_clear.goalkeeper_hand_clear[goalkeeper])

    observations = env.observe_all_si(reset.rollout._replace(state=state))
    own_observer = int(team_slots[1])
    assert int(observations.match.gk_handling_restricted_team[goalkeeper]) == team
    assert int(observations.match.gk_handling_restricted_team[own_observer]) == team
