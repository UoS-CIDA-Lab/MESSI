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

RANDOM_EVENT_SCHEMA = "footballworld.random-event/1"


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


def frame_random_key(match_key: jax.Array, control_tick: jax.Array) -> jax.Array:
    """Address one control frame without consuming or mutating ``match_key``.

    This publishes the rollout runner's frame-addressed key contract. Reusing
    one immutable match key across chunks is intentional;
    absolute control ticks make chunk boundaries irrelevant.
    """

    try:
        key_data = jax.random.key_data(match_key)
    except (TypeError, ValueError) as exc:
        raise TypeError("match_key must be one JAX PRNG key") from exc
    if key_data.shape != (2,):
        raise ValueError("match_key must be one unbatched Threefry PRNG key")
    tick = jnp.asarray(control_tick)
    if tick.shape != ():
        raise ValueError("control_tick must be scalar")
    if not jnp.issubdtype(tick.dtype, jnp.integer) or tick.dtype == jnp.bool_:
        raise TypeError("control_tick must have non-boolean integer dtype")
    return jax.random.fold_in(match_key, tick.astype(jnp.uint32))


def event_random_key(
    base_key: jax.Array,
    event: RandomEvent,
    *addresses: jax.Array,
) -> jax.Array:
    """Fold a named event and scalar causal addresses into ``base_key``."""

    if not isinstance(event, RandomEvent):
        raise TypeError("event must be RandomEvent")
    key = jax.random.fold_in(base_key, jnp.uint32(int(event)))
    for address in addresses:
        value = jnp.asarray(address)
        if value.shape != ():
            raise ValueError("random event addresses must be scalar")
        if not jnp.issubdtype(value.dtype, jnp.integer) or value.dtype == jnp.bool_:
            raise TypeError("random event addresses must have integer dtype")
        key = jax.random.fold_in(key, value.astype(jnp.uint32))
    return key


__all__ = [
    "RANDOM_EVENT_ADDRESSES",
    "RANDOM_EVENT_NAMES",
    "RANDOM_EVENT_SCHEMA",
    "RandomEvent",
    "event_random_key",
    "frame_random_key",
]
