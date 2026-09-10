from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from footballworld.core.action import IntentAction
from footballworld.core.constants import INTENT_ACTION_CONTINUOUS_DIM
from footballworld.core.contact import INTENT_CONTROL, INTENT_MOVE
from footballworld.dynamics.action import (
    ACTION_FLAG_INVALID_INTENT,
    ACTION_REASON_INPUT_SANITIZED,
    trace_action,
    trace_action_receipt,
)


def _continuous(player_count: int = 1) -> np.ndarray:
    return np.zeros(
        (player_count, INTENT_ACTION_CONTINUOUS_DIM),
        dtype=np.float32,
    )


def test_host_uint64_intent_is_checked_before_jax_narrowing() -> None:
    raw = np.asarray([2**32 + INTENT_CONTROL], dtype=np.uint64)

    action = IntentAction.from_array(raw, _continuous())

    np.testing.assert_array_equal(action.intent, [INTENT_MOVE])


def test_unrepresentable_host_intent_uses_explicit_trace_sentinel() -> None:
    raw = np.asarray([2**32 + INTENT_CONTROL], dtype=np.uint64)
    action = IntentAction.neutral(1)._replace(intent=raw)

    trace = trace_action(action)
    receipt = trace_action_receipt(action)

    np.testing.assert_array_equal(trace.requested_intent, [-1])
    np.testing.assert_array_equal(receipt.requested_intent, [-1])
    np.testing.assert_array_equal(receipt.effective_intent, [INTENT_MOVE])
    assert int(receipt.flags[0]) & ACTION_FLAG_INVALID_INTENT
    assert int(receipt.primary_reason[0]) == ACTION_REASON_INPUT_SANITIZED


def test_representable_host_intent_preserves_value_and_meaning() -> None:
    raw = np.asarray([INTENT_CONTROL], dtype=np.int64)
    action = IntentAction.neutral(1)._replace(intent=raw)

    sanitized = IntentAction.from_array(raw, _continuous())
    trace = trace_action(action)
    receipt = trace_action_receipt(action)

    np.testing.assert_array_equal(sanitized.intent, [INTENT_CONTROL])
    np.testing.assert_array_equal(trace.requested_intent, [INTENT_CONTROL])
    np.testing.assert_array_equal(receipt.effective_intent, [INTENT_CONTROL])
    assert not int(receipt.flags[0]) & ACTION_FLAG_INVALID_INTENT


def test_python_integer_beyond_numpy_fixed_width_fails_closed() -> None:
    action = IntentAction.from_array([2**100], _continuous())

    np.testing.assert_array_equal(action.intent, [INTENT_MOVE])


@pytest.mark.parametrize(
    "raw",
    (
        np.asarray([True], dtype=np.bool_),
        np.asarray([1.0], dtype=np.float32),
        [False],
    ),
)
def test_non_integer_or_boolean_intent_is_rejected(raw: object) -> None:
    with pytest.raises(TypeError, match="non-boolean integer dtype"):
        IntentAction.from_array(raw, _continuous())


def test_traced_uint32_intent_fails_closed_without_aliasing() -> None:
    continuous = jnp.zeros((1, INTENT_ACTION_CONTINUOUS_DIM), dtype=jnp.float32)
    sanitize = jax.jit(
        lambda intent: IntentAction.from_array(intent, continuous).intent
    )
    trace = jax.jit(
        lambda intent: (
            trace_action(
                IntentAction.neutral(1)._replace(intent=intent)
            ).requested_intent
        )
    )
    raw = jnp.asarray([np.iinfo(np.uint32).max], dtype=jnp.uint32)

    np.testing.assert_array_equal(sanitize(raw), [INTENT_MOVE])
    np.testing.assert_array_equal(trace(raw), [-1])
