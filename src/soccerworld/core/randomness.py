"""Privileged, named control of SoccerWorld's environment random streams.

The control is deliberately separate from :class:`~soccerworld.core.commands.StepCommand`.
Player and manager policies submit decisions; dataset reconstruction and controlled experiments
may replace individual environment PRNG keys.  A disabled cell leaves the original key untouched,
so supplying an empty control is transition-identical to supplying no control.

Each event identifier is append-only.  The first axis is the physical substep within one control
frame.  Frame-boundary and reset events use row zero.  Raw two-word Threefry keys make the PyTree
fixed-shape and portable across JAX's legacy and typed key representations.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from soccerworld._schema_versions import RANDOMNESS_CONTROL_SCHEMA

__all__ = [
    "RANDOM_EVENT_COUNT",
    "RANDOM_EVENT_NAMES",
    "RANDOMNESS_CONTROL_SCHEMA",
    "RandomEvent",
    "RandomnessControl",
    "select_random_key",
    "validate_randomness_control",
]


class RandomEvent(IntEnum):
    """Stable addresses for independently replaceable environment draws.

    Values are a wire contract and must never be reordered or reused.  A root-key event controls a
    configured stochastic component whose internal number of draws is user-defined (manager or
    set-piece-taker); the remaining events each name one engine-native draw.
    """

    RESET_KICKOFF_TEAM = 0
    RESET_EPISODE_SEED = 1
    CONTEST_WINNER = 2
    CHARGE_FOUL = 3
    CHARGE_CARD = 4
    CHARGE_CARD_COLOR = 5
    TACKLE_FOUL = 6
    TACKLE_SUCCESS = 7
    TACKLE_CARD = 8
    TACKLE_CARD_COLOR = 9
    DEFLECTION = 10
    DEFLECTION_DIRECTION = 11
    GOALKEEPER_CATCH = 12
    BODY_TRAP = 13
    SET_PIECE_TAKER = 14
    MANAGER_DECISION = 15


RANDOM_EVENT_NAMES = tuple(event.name.lower() for event in RandomEvent)
RANDOM_EVENT_COUNT = len(RandomEvent)
_EVENT_BY_NAME = {name: RandomEvent(index) for index, name in enumerate(RANDOM_EVENT_NAMES)}


def _host_index(name: str, value: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-boolean integer")
    result = int(value)
    if not 0 <= result < upper:
        raise ValueError(f"{name} must lie in [0, {upper}), got {result}")
    return result


def _event_index(event: RandomEvent | str | int) -> int:
    if isinstance(event, str):
        try:
            return int(_EVENT_BY_NAME[event.lower()])
        except KeyError as exc:
            raise ValueError(
                f"unknown random event {event!r}; expected one of {RANDOM_EVENT_NAMES}"
            ) from exc
    if isinstance(event, RandomEvent):
        return int(event)
    return _host_index("event", event, RANDOM_EVENT_COUNT)


def _raw_threefry_key(key: Array) -> Array:
    try:
        implementation = jax.random.key_impl(key)
        raw = jax.random.key_data(key)
    except (TypeError, ValueError) as exc:
        raise TypeError("key must be one scalar threefry2x32 JAX PRNG key") from exc
    if str(implementation) != "threefry2x32" or raw.shape != (2,):
        raise ValueError("key must be one scalar threefry2x32 JAX PRNG key")
    return jnp.asarray(raw, jnp.uint32)


class RandomnessControl(NamedTuple):
    """Fixed-shape event-key overrides for one environment control frame.

    ``key_override_mask`` has shape ``(D, E)`` and ``key_data`` has shape
    ``(D, E, 2)``, where ``D`` is the environment's physical decimation and ``E`` is
    :data:`RANDOM_EVENT_COUNT`.  Disabled key words are inert and conventionally zero.
    """

    key_override_mask: Array
    key_data: Array

    @classmethod
    def empty(cls, decimation: int) -> RandomnessControl:
        decimation = _host_index("decimation", decimation, np.iinfo(np.int32).max)
        if decimation == 0:
            raise ValueError("decimation must be positive")
        return cls(
            key_override_mask=jnp.zeros(
                (decimation, RANDOM_EVENT_COUNT), dtype=jnp.bool_
            ),
            key_data=jnp.zeros(
                (decimation, RANDOM_EVENT_COUNT, 2), dtype=jnp.uint32
            ),
        )

    def with_event_key(
        self,
        event: RandomEvent | str | int,
        key: Array,
        *,
        substep: int = 0,
    ) -> RandomnessControl:
        """Return a copy that replaces one named event key."""

        event_index = _event_index(event)
        substep_index = _host_index(
            "substep", substep, int(self.key_override_mask.shape[0])
        )
        raw = _raw_threefry_key(key)
        return RandomnessControl(
            key_override_mask=self.key_override_mask.at[
                substep_index, event_index
            ].set(True),
            key_data=self.key_data.at[substep_index, event_index].set(raw),
        )

    def with_event_seed(
        self,
        event: RandomEvent | str | int,
        seed: int,
        *,
        substep: int = 0,
    ) -> RandomnessControl:
        """Convenience builder using one unsigned 32-bit seed."""

        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be a non-boolean integer")
        integer = int(seed)
        if not 0 <= integer <= np.iinfo(np.uint32).max:
            raise ValueError("seed must lie in the uint32 range")
        return self.with_event_key(
            event, jax.random.PRNGKey(integer), substep=substep
        )

    def without_event(
        self, event: RandomEvent | str | int, *, substep: int = 0
    ) -> RandomnessControl:
        """Return a copy with one override disabled and its inactive words zeroed."""

        event_index = _event_index(event)
        substep_index = _host_index(
            "substep", substep, int(self.key_override_mask.shape[0])
        )
        return RandomnessControl(
            key_override_mask=self.key_override_mask.at[
                substep_index, event_index
            ].set(False),
            key_data=self.key_data.at[substep_index, event_index].set(
                jnp.zeros((2,), jnp.uint32)
            ),
        )


def validate_randomness_control(
    control: RandomnessControl, *, decimation: int
) -> RandomnessControl:
    """Validate static PyTree structure, shape, and dtype without reading traced values."""

    if not isinstance(control, RandomnessControl):
        raise TypeError(
            f"randomness must be RandomnessControl, got {type(control).__name__}"
        )
    expected_mask = (decimation, RANDOM_EVENT_COUNT)
    expected_keys = (*expected_mask, 2)
    mask = jnp.asarray(control.key_override_mask)
    keys = jnp.asarray(control.key_data)
    if mask.shape != expected_mask:
        raise ValueError(
            f"randomness.key_override_mask must have shape {expected_mask}, got {mask.shape}"
        )
    if mask.dtype != jnp.dtype(jnp.bool_):
        raise TypeError(
            "randomness.key_override_mask must have bool dtype, "
            f"got {mask.dtype}"
        )
    if keys.shape != expected_keys:
        raise ValueError(
            f"randomness.key_data must have shape {expected_keys}, got {keys.shape}"
        )
    if keys.dtype != jnp.dtype(jnp.uint32):
        raise TypeError(
            f"randomness.key_data must have uint32 dtype, got {keys.dtype}"
        )
    return RandomnessControl(mask, keys)


def select_random_key(
    control: RandomnessControl | None,
    event: RandomEvent | int,
    substep: int | Array,
    default_key: Array,
) -> Array:
    """Select an event override while preserving the caller's key representation.

    ``control=None`` is a zero-operation fast path.  Legacy ``uint32[2]`` keys stay legacy and
    typed Threefry keys stay typed, avoiding representation-driven retracing in callers.
    """

    if control is None:
        return default_key
    event_index = int(event)
    use = control.key_override_mask[substep, event_index]
    raw_default = jax.random.key_data(default_key)
    raw = jnp.where(use, control.key_data[substep, event_index], raw_default)
    if jnp.asarray(default_key).dtype == jnp.dtype(jnp.uint32):
        return raw
    return jax.random.wrap_key_data(raw, impl="threefry2x32")
