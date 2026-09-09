from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from footballworld import batch_rollout, batching


def _single_rollout(values):
    def step(carry, value):
        updated = jax.lax.cond(
            jnp.sum(value) > 0.0,
            lambda pair: pair[0] + pair[1],
            lambda pair: pair[0] - pair[1],
            (carry, value),
        )
        return updated, {"value": updated, "positive": updated > 0.0}

    return jax.lax.scan(
        step,
        jnp.zeros(values.shape[1], dtype=values.dtype),
        values,
    )


def test_batch_rollout_preserves_result_contract():
    batch_size, steps, width = 8, 7, 4
    values = jnp.linspace(
        -1.0,
        1.0,
        batch_size * steps * width,
        dtype=jnp.float32,
    ).reshape(batch_size, steps, width)

    result = jax.jit(batch_rollout(_single_rollout))(values)
    expected = jax.tree.map(
        lambda *leaves: jnp.stack(leaves),
        *[_single_rollout(value) for value in values],
    )

    assert jax.tree.structure(result) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(result),
        jax.tree.leaves(expected),
        strict=True,
    ):
        assert actual_leaf.shape == expected_leaf.shape
        assert actual_leaf.dtype == expected_leaf.dtype
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
    assert result[0].shape == (batch_size, width)
    assert result[1]["value"].shape == (batch_size, steps, width)


def test_batch_rollout_rejects_malformed_leading_axes():
    unary = batch_rollout(lambda value: value)
    with pytest.raises(ValueError, match="non-empty"):
        unary(jnp.empty((0, 2), dtype=jnp.float32))
    with pytest.raises(ValueError, match="is scalar"):
        unary(jnp.asarray(1.0, dtype=jnp.float32))

    binary = batch_rollout(lambda left, right: left + right)
    with pytest.raises(ValueError, match="share one leading match axis"):
        binary(
            jnp.zeros((2, 3), dtype=jnp.float32),
            jnp.zeros((3, 3), dtype=jnp.float32),
        )


def test_batch_rollout_rejects_non_callable():
    with pytest.raises(TypeError, match="one_match must be callable"):
        batch_rollout(None)


def test_batching_module_does_not_import_jax_when_loaded_in_isolation():
    module_path = Path(batching.__file__).resolve()
    code = f"""
import importlib.util
import sys
assert 'jax' not in sys.modules
spec = importlib.util.spec_from_file_location('_footballworld_batching_isolated', {str(module_path)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'jax' not in sys.modules
assert module.batch_rollout.__kwdefaults__ is None
"""
    subprocess.run([sys.executable, "-I", "-c", code], check=True)
