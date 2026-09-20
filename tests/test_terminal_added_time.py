"""A match that ends when added time runs out must say so.

``_terminal_classification`` names why a match ended. The environment truncates
on ``regulation_complete | added_time_limit_reached | wall_clock_exhausted``
(``episode.py::_status``); the classifier covered the first and the third and
fell through to ``unknown_environment_done`` for the second. That is not a
cosmetic label. A period may run past its regulation length only by the
statutory added time, so a match that exhausts it ended exactly as the laws
require, and recording that as an unidentified cause asserts an ignorance the
environment does not have.

This matters in practice rather than in principle. A policy that cannot
accumulate ninety minutes of live play inside the added-time ceiling reaches
this ending in every match: measured on the first DAgger collection, 33 of 33.

The completion flag is deliberately unchanged. Such a match did not play a full
ninety minutes of live football, so a corpus release gate should still see
``full_duration_complete=False``; only the recorded cause changes.
"""

from __future__ import annotations

import jax.numpy as jnp

from footballworld import FootballWorld, Player, PlayerProfile
from footballworld.rendering.capture import _terminal_classification

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


def _team(identity_base):
    return tuple(
        Player(
            PlayerProfile(player_id=identity_base + index, is_goalkeeper=index == 0),
            position,
        )
        for index, position in enumerate(_FORMATION)
    )


def _environment():
    """Return the env and a rollout whose state the tests rewrite."""

    env = FootballWorld()
    return env, env.reset(_team(1_000), _team(2_000)).rollout


def _classify(env, rollout, state):
    """Classify a finished match. The cause only exists once ``done``."""

    return _terminal_classification(
        env,
        rollout._replace(state=state),
        done=True,
        maximum_steps=None,
        steps_executed=int(state.control_tick),
    )


def _clock(env):
    fulltime, halftime = env.match.clock_ticks(env.timebase)
    return fulltime, halftime, env.match.maximum_added_time_ticks(env.timebase)


def test_added_time_exhaustion_is_named_rather_than_unknown():
    """The ending every learner match actually reached."""

    env, rollout = _environment()
    state = rollout.state
    fulltime, halftime, added = _clock(env)
    # Both periods ran their regulation length plus all of their added time,
    # and live play still never reached full time because the dead-ball tail
    # absorbed the difference. The dead-ball figure is the measured mean of the
    # first DAgger self-play collection (15,873 of 66,000 ticks, 24.1%), so the
    # state this asserts on is the one that actually occurs rather than a
    # constructed edge case: regulation_elapsed lands at 50,127, short of the
    # 54,000 live ticks `regulation_complete` requires.
    first_half_wall = halftime + added
    second_half_wall = (fulltime - halftime) + added
    exhausted = state._replace(
        control_tick=jnp.int32(first_half_wall + second_half_wall),
        first_half_wall_end_tick=jnp.int32(first_half_wall),
        dead_ball_control_ticks=jnp.int32(15_873),
        first_half_live_extension_ticks=jnp.int32(0),
    )
    basis, complete = _classify(env, rollout, exhausted)
    assert basis == "added_time_limit_reached"
    assert complete is False


def test_a_match_that_played_its_ninety_minutes_is_not_an_added_time_ending():
    env, rollout = _environment()
    state = rollout.state
    fulltime, halftime, _ = _clock(env)
    played = state._replace(
        control_tick=jnp.int32(fulltime + 500),
        first_half_wall_end_tick=jnp.int32(halftime + 100),
        dead_ball_control_ticks=jnp.int32(500),
        first_half_live_extension_ticks=jnp.int32(0),
    )
    basis, complete = _classify(env, rollout, played)
    assert basis == "regulation_complete"
    assert complete is True


def test_a_second_half_short_of_its_ceiling_is_not_classified_as_ended():
    """One tick below the limit must not trip the new branch."""

    env, rollout = _environment()
    state = rollout.state
    fulltime, halftime, added = _clock(env)
    first_half_wall = halftime + added
    nearly = state._replace(
        control_tick=jnp.int32(first_half_wall + (fulltime - halftime) + added - 1),
        first_half_wall_end_tick=jnp.int32(first_half_wall),
        dead_ball_control_ticks=jnp.int32(2 * added),
        first_half_live_extension_ticks=jnp.int32(0),
    )
    basis, complete = _classify(env, rollout, nearly)
    assert basis == "unknown_environment_done"
    assert complete is False


def test_a_first_half_still_in_progress_is_not_an_added_time_ending():
    """Before the halftime boundary there is no second-half wall to measure."""

    env, rollout = _environment()
    state = rollout.state
    _, halftime, _ = _clock(env)
    midway = state._replace(
        control_tick=jnp.int32(halftime // 2),
        first_half_wall_end_tick=jnp.int32(-1),
        dead_ball_control_ticks=jnp.int32(50),
        first_half_live_extension_ticks=jnp.int32(0),
    )
    basis, complete = _classify(env, rollout, midway)
    assert basis == "unknown_environment_done"
    assert complete is False
