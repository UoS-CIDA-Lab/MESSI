from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from footballworld.core.constants import NO_TEAM, RK_NONE
from footballworld.environment.api import _goal_reward
from footballworld.environment.episode import RuleOutcome


def _outcome(score_delta: tuple[int, int]) -> RuleOutcome:
    return RuleOutcome(
        score_delta=jnp.asarray(score_delta, dtype=jnp.int32),
        restart_opened=jnp.bool_(False),
        restart_kind=jnp.int32(RK_NONE),
        restart_team=jnp.int32(NO_TEAM),
    )


@pytest.mark.parametrize(
    ("score_delta", "expected"),
    (
        ((1, 0), (1.0, -1.0)),
        ((0, 1), (-1.0, 1.0)),
        ((0, 0), (0.0, 0.0)),
    ),
)
def test_default_goal_reward_is_zero_sum_eager_and_jit(
    score_delta: tuple[int, int], expected: tuple[float, float]
) -> None:
    outcome = _outcome(score_delta)

    eager = _goal_reward(outcome)
    compiled = jax.jit(_goal_reward)(outcome)

    for reward in (eager, compiled):
        assert reward.shape == (2,)
        assert reward.dtype == jnp.float32
        np.testing.assert_array_equal(np.asarray(reward), np.asarray(expected))
        assert float(jnp.sum(reward)) == 0.0


def test_default_goal_reward_tracks_multiple_score_delta_without_clipping() -> None:
    reward = _goal_reward(_outcome((2, 0)))
    np.testing.assert_array_equal(np.asarray(reward), np.asarray((2.0, -2.0)))
