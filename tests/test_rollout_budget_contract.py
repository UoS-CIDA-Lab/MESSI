import jax
import jax.numpy as jnp
import numpy as np
import pytest

from footballworld import FootballWorld
from footballworld.policies.player import FunctionalPlayerPolicy
from footballworld.rollout import (
    _runtime_step_budget,
    make_advance,
    make_event_rollout,
    make_interruptible_advance,
    make_managed_advance,
    make_rollout,
)

INT32_MAX = np.iinfo(np.int32).max


def _policy() -> FunctionalPlayerPolicy:
    return FunctionalPlayerPolicy(
        initialize_fn=lambda *_args: jnp.int32(0),
        step_fn=lambda *_args: None,
    )


@pytest.mark.parametrize(
    "factory",
    [
        make_advance,
        make_interruptible_advance,
        make_managed_advance,
        make_rollout,
        make_event_rollout,
    ],
)
def test_rollout_factories_reject_static_horizon_outside_int32(factory):
    with pytest.raises(ValueError, match="supported int32 scan horizon"):
        factory(FootballWorld(), _policy(), INT32_MAX + 1)


@pytest.mark.parametrize("value", [np.uint64(2**32), np.int64(-1), 5])
def test_eager_runtime_budget_rejects_before_jax_narrowing(value):
    with pytest.raises(ValueError, match=r"must lie in \[0, 4\]"):
        _runtime_step_budget(value, 4)


def test_eager_runtime_budget_canonicalizes_safe_wide_integer():
    budget = _runtime_step_budget(np.int64(4), 4)

    assert budget.dtype == jnp.int32
    assert int(budget) == 4


def test_runtime_budget_rejects_boolean_dtype():
    with pytest.raises(TypeError, match="non-boolean integer dtype"):
        _runtime_step_budget(np.bool_(True), 4)


def test_traced_runtime_budget_remains_fail_closed():
    sanitize = jax.jit(lambda value: _runtime_step_budget(value, 4))

    assert int(sanitize(jnp.int32(4))) == 4
    assert int(sanitize(jnp.int32(-1))) == 0
    assert int(sanitize(jnp.int32(5))) == 0
