import jax
import jax.numpy as jnp
import numpy as np

from footballworld.policies import (
    TacticalPlan,
    select_tactical_plans_from_abilities,
    tactical_plan_code,
)


def _selection(profile, *, prior=None, seed=7):
    ability = jnp.broadcast_to(jnp.asarray(profile, dtype=jnp.float32), (2, 11, 5))
    valid = jnp.ones((2, 11), dtype=jnp.bool_)
    goalkeeper = jnp.zeros((2, 11), dtype=jnp.bool_).at[:, 0].set(True)
    return select_tactical_plans_from_abilities(
        ability,
        valid,
        goalkeeper,
        jax.random.key(seed),
        prior=prior,
    )


def test_roster_abilities_shift_tactical_softmax_before_sampling():
    control = _selection([0.0, 0.0, 0.0, 1.0, 0.0]).probability
    running = _selection([1.0, 0.0, 0.0, 0.0, 1.0]).probability
    juego = tactical_plan_code(TacticalPlan.JUEGO_DE_POSICION)
    gegenpress = tactical_plan_code(TacticalPlan.GEGENPRESS)

    assert np.all(np.asarray(control[:, juego]) > np.asarray(control[:, gegenpress]))
    assert np.all(np.asarray(running[:, gegenpress]) > np.asarray(running[:, juego]))


def test_zero_prior_is_a_hard_exclusion_and_same_seed_is_reproducible():
    prior = (
        jnp.zeros(5, dtype=jnp.float32)
        .at[tactical_plan_code(TacticalPlan.CATENACCIO)]
        .set(1.0)
    )
    first = _selection([0.5] * 5, prior=prior, seed=29)
    repeated = _selection([0.5] * 5, prior=prior, seed=29)

    np.testing.assert_array_equal(first.plan_code, repeated.plan_code)
    np.testing.assert_array_equal(
        first.plan_code,
        np.full(2, tactical_plan_code(TacticalPlan.CATENACCIO)),
    )
    assert np.count_nonzero(np.asarray(first.probability)) == 2
