"""Stable names for FootballWorld environment random streams.

Append-only event addresses keep stochastic mechanisms reproducible without a
fixed override tensor in the lean transition. FootballWorld uses a ``fold_in``
key hierarchy and publishes only its stable names here. Controlled event keys,
if needed later, belong to a separate scenario API.
"""

from __future__ import annotations

from enum import IntEnum
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

RANDOM_EVENT_SCHEMA = "footballworld.random-event/1"
_MAX_FOLD_IN_ADDRESS = int(np.iinfo(np.uint32).max)


class RandomEvent(IntEnum):
    """Append-only numeric addresses for stochastic environment mechanisms."""

    ACTIVE_ROSTER_SAMPLING = 0x41435456  # ASCII "ACTV"
    BENCH_ROSTER_SAMPLING = 0x42454E43  # ASCII "BENC"
    RESTART_TAKER = 0x54414B52  # ASCII "TAKR"
    CONTEST_FOUL = 0x464F554C  # ASCII "FOUL"
    CONTEST_SUCCESS = 0x0057494E  # ASCII "WIN"
    CONTEST_DEFLECTION = 0x4445464C  # ASCII "DEFL"
    CONTEST_CARD = 0x43415244  # ASCII "CARD"
    CONTEST_CARD_COLOUR = 0x434F4C52  # ASCII "COLR"
    CONTEST_WINNER = 0x57494E52  # ASCII "WINR"
    CONTEST_OUTCOME = 0x4F555443  # ASCII "OUTC"
    BODY_FOUL = 0x424F4459  # ASCII "BODY"
    BODY_CARD = 0x42434152  # ASCII "BCAR"
    BODY_CARD_COLOUR = 0x42434F4C  # ASCII "BCOL"
    OPENING_CANDIDATE = 0x4F50454E  # ASCII "OPEN"
    OPENING_FORMATION = 0x464F524D  # ASCII "FORM"
    OPENING_PLAYER_LINE = 0x4C494E45  # ASCII "LINE"
    OPENING_REGION = 0x52454749  # ASCII "REGI"
    MANAGER_SUBSTITUTION = 0x53554253  # ASCII "SUBS"
    MANAGER_SUBSTITUTION_TIMING = 0x54494D45  # ASCII "TIME"
    MANAGER_SUBSTITUTION_OUTGOING = 0x4F555447  # ASCII "OUTG"
    MANAGER_SUBSTITUTION_INCOMING = 0x494E434D  # ASCII "INCM"
    POLICY_ATTACK_PATTERN = 0x4154504E  # ASCII "ATPN"
    POLICY_CARRY_COMMIT = 0x434D4954  # ASCII "CMIT"
    POLICY_CARRIER_DECISION = 0x43415252  # ASCII "CARR"
    POLICY_ATTACK_EPISODE = 0x41544550  # ASCII "ATEP"
    POLICY_RUN_BEHIND = 0x52554E52  # ASCII "RUNR"
    POLICY_CHALLENGE = 0x5441434B  # ASCII "TACK"
    POLICY_DEEP_CLEAR = 0x434C4541  # ASCII "CLEA"
    POLICY_OFFSIDE_TIMING = 0x4F465344  # ASCII "OFSD"
    POLICY_RESTART_DECISION = 0x52535452  # ASCII "RSTR"
    POLICY_REBOUND_SHOT = 0x52425348  # ASCII "RBSH"
    POLICY_REBOUND_CHOICE = 0x52424348  # ASCII "RBCH"
    POLICY_RECEIVER = 0x52454356  # ASCII "RECV"
    POLICY_MACRO = 0x4D414352  # ASCII "MACR"
    POLICY_DRIBBLE_DIRECTION = 0x44524942  # ASCII "DRIB"
    POLICY_CLEAR_DIRECTION = 0x434C5244  # ASCII "CLRD"
    POLICY_SHOT_PORTION = 0x53484F54  # ASCII "SHOT"
    POLICY_CURVE = 0x43555256  # ASCII "CURV"
    POLICY_SHOT_DIRECTION = 0x53484452  # ASCII "SHDR"
    POLICY_SHOT_LAUNCH = 0x53484C4E  # ASCII "SHLN"
    POLICY_GOALKEEPER_DISTRIBUTION = 0x474B4449  # ASCII "GKDI"
    POLICY_GROUND_RESTART = 0x47524E44  # ASCII "GRND"
    POLICY_AERIAL_RESTART = 0x4145524C  # ASCII "AERL"
    POLICY_GOALKEEPER_KICK = 0x474B4943  # ASCII "GKIC"
    POLICY_GOAL_TARGET = 0x474F414C  # ASCII "GOAL"


RANDOM_EVENT_NAMES = tuple(event.name.lower() for event in RandomEvent)
RANDOM_EVENT_ADDRESSES = MappingProxyType(
    {name.lower(): int(event) for name, event in RandomEvent.__members__.items()}
)


def validate_prng_key(
    key: jax.Array,
    *,
    name: str = "key",
    batch_size: int | None = None,
) -> jax.Array:
    """Require FootballWorld's reproducible scalar or fixed-batch key contract.

    Typed JAX keys encode their implementation in the dtype. Legacy uint32
    keys inherit the process default. Inspecting the typed dtype and the
    legacy-key shape/config separately avoids materializing ``key_data`` and
    therefore adds no wrap/unwrap operations to traced transition graphs.
    FootballWorld receipts and named fold-in streams are pinned to
    non-partitionable Threefry; accepting another implementation would let one
    recorded seed produce a different roster or trajectory without changing
    the environment fingerprint.
    """

    if bool(jax.config.jax_threefry_partitionable):
        raise RuntimeError(
            "FootballWorld requires JAX_THREEFRY_PARTITIONABLE=0; enabling it "
            "changes seeded environment trajectories"
        )
    if not isinstance(key, (jax.Array, jax.core.Tracer, np.ndarray)):
        raise TypeError(f"{name} must be a JAX PRNG key")
    try:
        key_dtype = key.dtype
        key_shape = key.shape
        typed_key = jax.dtypes.issubdtype(key_dtype, jax.dtypes.prng_key)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a JAX PRNG key") from exc
    if typed_key:
        try:
            implementation = jax.random.key_impl(key)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a JAX PRNG key") from exc
        expected_shape = () if batch_size is None else (batch_size,)
    else:
        if key_dtype != jnp.uint32:
            raise TypeError(f"{name} must be a JAX PRNG key")
        implementation = jax.config.jax_default_prng_impl
        expected_shape = (2,) if batch_size is None else (batch_size, 2)
    if str(implementation) != "threefry2x32":
        raise ValueError(
            f"{name} must use threefry2x32 for reproducible FootballWorld "
            f"randomness, got {implementation}"
        )
    if key_shape != expected_shape:
        description = "one unbatched" if batch_size is None else f"exactly {batch_size}"
        suffix = "" if batch_size is None else "s"
        raise TypeError(
            f"{name} must contain {description} threefry2x32 PRNG key{suffix}"
        )
    return key


def _validate_host_random_address(value: object, *, name: str) -> object:
    """Validate and canonicalize a host index before JAX can narrow it."""

    if isinstance(value, (jax.Array, jax.core.Tracer)):
        return value
    try:
        host = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a non-boolean integer scalar") from exc
    if host.shape != ():
        return value
    if not np.issubdtype(host.dtype, np.integer) or np.issubdtype(host.dtype, np.bool_):
        return value
    integer = int(host)
    if not 0 <= integer <= _MAX_FOLD_IN_ADDRESS:
        raise ValueError(
            f"{name} must lie in [0, {_MAX_FOLD_IN_ADDRESS}], got {integer}"
        )
    return np.uint32(integer)


def frame_random_key(match_key: jax.Array, control_tick: jax.Array) -> jax.Array:
    """Address one control frame without consuming or mutating ``match_key``.

    This publishes the rollout runner's frame-addressed key contract. Reusing
    one immutable match key across chunks is intentional;
    absolute control ticks make chunk boundaries irrelevant.
    """

    validate_prng_key(match_key, name="match_key")
    control_tick = _validate_host_random_address(control_tick, name="control_tick")
    return _frame_random_key_unchecked(match_key, control_tick)


def _frame_random_key_unchecked(
    match_key: jax.Array, control_tick: jax.Array
) -> jax.Array:
    """Derive a frame key after the owning public API validated ``match_key``."""

    tick = jnp.asarray(control_tick)
    if tick.shape != ():
        raise ValueError("control_tick must be scalar")
    if not jnp.issubdtype(tick.dtype, jnp.integer) or tick.dtype == jnp.bool_:
        raise TypeError("control_tick must have non-boolean integer dtype")
    if tick.dtype.itemsize > np.dtype(np.uint32).itemsize:
        raise TypeError("control_tick must have at most 32-bit integer dtype")
    return jax.random.fold_in(match_key, tick.astype(jnp.uint32))


def event_random_key(
    base_key: jax.Array,
    event: RandomEvent,
    *addresses: jax.Array,
) -> jax.Array:
    """Fold a named event and scalar causal addresses into ``base_key``."""

    if not isinstance(event, RandomEvent):
        raise TypeError("event must be RandomEvent")
    validate_prng_key(base_key, name="base_key")
    addresses = tuple(
        _validate_host_random_address(address, name=f"addresses[{index}]")
        for index, address in enumerate(addresses)
    )
    return _event_random_key_unchecked(base_key, event, *addresses)


def _event_random_key_unchecked(
    base_key: jax.Array,
    event: RandomEvent,
    *addresses: jax.Array,
) -> jax.Array:
    """Fold an event after the owning public API validated ``base_key``."""

    if not isinstance(event, RandomEvent):
        raise TypeError("event must be RandomEvent")
    key = jax.random.fold_in(base_key, jnp.uint32(int(event)))
    for address in addresses:
        value = jnp.asarray(address)
        if value.shape != ():
            raise ValueError("random event addresses must be scalar")
        if not jnp.issubdtype(value.dtype, jnp.integer) or value.dtype == jnp.bool_:
            raise TypeError("random event addresses must have integer dtype")
        if value.dtype.itemsize > np.dtype(np.uint32).itemsize:
            raise TypeError(
                "random event addresses must have at most 32-bit integer dtype"
            )
        key = jax.random.fold_in(key, value.astype(jnp.uint32))
    return key


__all__ = [
    "RANDOM_EVENT_ADDRESSES",
    "RANDOM_EVENT_NAMES",
    "RANDOM_EVENT_SCHEMA",
    "RandomEvent",
    "event_random_key",
    "frame_random_key",
    "validate_prng_key",
]
