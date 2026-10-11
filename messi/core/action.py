"""한 control frame에 사용하는 고정 shape 5-intent player action."""

from __future__ import annotations

from enum import IntEnum
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from messi.core.action_mapping import (
    RadialControl,
    linf_radial_decode,
    signed_to_unit,
)

ACTION_MIN = -1.0
ACTION_MAX = 1.0

ACTION_CONTINUOUS_DIM = 7
ACTION_MOVE = slice(0, 2)
ACTION_FORCE_TO_BALL = slice(2, 4)
ACTION_LAUNCH = 4
ACTION_SPIN_SIDE = 5
ACTION_SPIN_BACK = 6

PARAMETER_MOVE = 1 << 0
PARAMETER_FORCE_DIRECTION = 1 << 1
PARAMETER_FORCE_POWER = 1 << 2
PARAMETER_LAUNCH = 1 << 3
PARAMETER_SPIN = 1 << 4
PARAMETER_ALL = (
    PARAMETER_MOVE
    | PARAMETER_FORCE_DIRECTION
    | PARAMETER_FORCE_POWER
    | PARAMETER_LAUNCH
    | PARAMETER_SPIN
)

INTENT_MOVE = 0
INTENT_CONTROL = 1
INTENT_PASS = 2
INTENT_SHOOT = 3
INTENT_CHALLENGE = 4
ACTION_INTENT_COUNT = 5

ACTION_SCHEMA_VERSION = 1
ACTION_SCHEMA = f"messi.intent-action/{ACTION_SCHEMA_VERSION}"


def _canonicalize_intent(
    intent: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """submitted, effective, invalid intent를 int32 narrowing 전에 분리한다."""

    int32_limits = np.iinfo(np.int32)
    if isinstance(intent, (jax.Array, jax.core.Tracer)):
        source = jnp.asarray(intent)
        if jnp.issubdtype(source.dtype, jnp.bool_) or not jnp.issubdtype(
            source.dtype, jnp.integer
        ):
            raise TypeError("intent는 bool이 아닌 integer dtype이어야 한다")
        source_limits = np.iinfo(np.dtype(source.dtype))
        if (
            source_limits.min >= int32_limits.min
            and source_limits.max <= int32_limits.max
        ):
            submitted = source.astype(jnp.int32)
        else:
            if jnp.issubdtype(source.dtype, jnp.unsignedinteger):
                representable = source <= np.asarray(
                    int32_limits.max, dtype=np.dtype(source.dtype)
                )
            else:
                representable = (source >= int32_limits.min) & (
                    source <= int32_limits.max
                )
            narrowed = jnp.where(representable, source, jnp.zeros_like(source)).astype(
                jnp.int32
            )
            submitted = jnp.where(representable, narrowed, jnp.int32(-1))
    else:
        try:
            source = np.asarray(intent)
        except (TypeError, ValueError) as exc:
            raise TypeError("intent는 integer array여야 한다") from exc
        if source.dtype == np.dtype(object):
            flat_values = tuple(source.flat)
            if any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                for value in flat_values
            ):
                raise TypeError("intent는 bool이 아닌 integer dtype이어야 한다")
            submitted_host = np.asarray(
                [
                    int(value)
                    if int32_limits.min <= int(value) <= int32_limits.max
                    else -1
                    for value in flat_values
                ],
                dtype=np.int32,
            ).reshape(source.shape)
        else:
            if np.issubdtype(source.dtype, np.bool_) or not np.issubdtype(
                source.dtype, np.integer
            ):
                raise TypeError("intent는 bool이 아닌 integer dtype이어야 한다")
            representable = source <= int32_limits.max
            if np.issubdtype(source.dtype, np.signedinteger):
                representable &= source >= int32_limits.min
            narrowed = np.where(
                representable,
                source,
                np.zeros((), dtype=source.dtype),
            )
            submitted_host = narrowed.astype(np.int32)
            submitted_host = np.where(
                representable,
                submitted_host,
                np.int32(-1),
            ).astype(np.int32)
        submitted = jnp.asarray(submitted_host, dtype=jnp.int32)

    valid = (submitted >= INTENT_MOVE) & (submitted < ACTION_INTENT_COUNT)
    effective = jnp.where(valid, submitted, jnp.int32(INTENT_MOVE))
    return submitted, effective, ~valid


class ActionIntent(IntEnum):
    """정책이 선택할 수 있는 전체 categorical intent."""

    MOVE = INTENT_MOVE
    CONTROL = INTENT_CONTROL
    PASS = INTENT_PASS
    SHOOT = INTENT_SHOOT
    CHALLENGE = INTENT_CHALLENGE


class DecodedIntentAction(NamedTuple):
    """5개 intent와 물리 단위로 decode된 continuous control."""

    intent: jax.Array
    contact: jax.Array
    move: RadialControl
    force_to_ball: RadialControl
    launch: jax.Array
    spin: jax.Array


class IntentAction(NamedTuple):
    """5-way intent와 일곱 개의 정규화된 continuous control.

    `SHOOT`는 득점 시도와 defensive clearance를 모두 포함한다. 두 행동의 물리적
    차이는 방향, power, launch, spin과 경기 맥락에서 결정하며 별도 `CLEAR`
    category는 제공하지 않는다.
    """

    intent: jax.Array
    move: jax.Array
    force_to_ball: jax.Array
    launch: jax.Array
    spin: jax.Array

    @classmethod
    def from_array(
        cls,
        intent: jax.Array,
        continuous: jax.Array,
    ) -> IntentAction:
        """categorical intent와 trailing 7-vector를 검증하고 정규화한다."""

        continuous = jnp.asarray(continuous, dtype=jnp.float32)
        if continuous.shape[-1:] != (ACTION_CONTINUOUS_DIM,):
            raise ValueError(
                "continuous의 마지막 dimension은 "
                f"{ACTION_CONTINUOUS_DIM}이어야 한다: {continuous.shape}"
            )
        _, effective_intent, _ = _canonicalize_intent(intent)
        if effective_intent.shape != continuous.shape[:-1]:
            raise ValueError(
                "intent shape는 continuous prefix와 같아야 한다: "
                f"{effective_intent.shape} != {continuous.shape[:-1]}"
            )
        controls = jnp.clip(
            jnp.nan_to_num(
                continuous,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            ACTION_MIN,
            ACTION_MAX,
        )
        spin = jnp.stack(
            (
                controls[..., ACTION_SPIN_SIDE],
                controls[..., ACTION_SPIN_BACK],
            ),
            axis=-1,
        )
        return cls(
            intent=effective_intent,
            move=controls[..., ACTION_MOVE],
            force_to_ball=controls[..., ACTION_FORCE_TO_BALL],
            launch=controls[..., ACTION_LAUNCH],
            spin=spin,
        )

    def as_continuous_array(self) -> jax.Array:
        """categorical intent를 제외한 일곱 continuous control을 반환한다."""

        return jnp.concatenate(
            (
                self.move,
                self.force_to_ball,
                self.launch[..., None],
                self.spin,
            ),
            axis=-1,
        ).astype(jnp.float32)

    def decode(self) -> DecodedIntentAction:
        """임의 생성된 NamedTuple 입력까지 sanitize한 뒤 물리 control로 decode한다."""

        sanitized = self.from_array(self.intent, self.as_continuous_array())
        return DecodedIntentAction(
            intent=sanitized.intent,
            contact=sanitized.intent != INTENT_MOVE,
            move=linf_radial_decode(sanitized.move),
            force_to_ball=linf_radial_decode(sanitized.force_to_ball),
            launch=signed_to_unit(sanitized.launch),
            spin=sanitized.spin,
        )

    @classmethod
    def neutral(cls, player_count: int) -> IntentAction:
        """지정한 선수 수에 맞는 MOVE intent와 0 control을 반환한다."""

        if (
            not isinstance(player_count, Integral)
            or isinstance(player_count, bool)
            or player_count < 0
        ):
            raise ValueError("player_count는 음이 아닌 정수여야 한다")
        count = int(player_count)
        return cls.from_array(
            jnp.full(count, INTENT_MOVE, dtype=jnp.int32),
            jnp.zeros((count, ACTION_CONTINUOUS_DIM), dtype=jnp.float32),
        )

    @classmethod
    def move_only(cls, move: jax.Array) -> IntentAction:
        """입력 movement만 보존한 MOVE action을 반환한다."""

        move = jnp.asarray(move, dtype=jnp.float32)
        if move.shape[-1:] != (2,):
            raise ValueError(f"move의 마지막 dimension은 2여야 한다: {move.shape}")
        prefix = move.shape[:-1]
        continuous = jnp.zeros((*prefix, ACTION_CONTINUOUS_DIM), dtype=jnp.float32)
        continuous = continuous.at[..., ACTION_MOVE].set(move)
        return cls.from_array(
            jnp.full(prefix, INTENT_MOVE, dtype=jnp.int32),
            continuous,
        )


def neutral_action(player_count: int) -> IntentAction:
    """표준 neutral player action을 반환한다."""

    return IntentAction.neutral(player_count)


__all__ = [
    "ACTION_CONTINUOUS_DIM",
    "ACTION_INTENT_COUNT",
    "ACTION_MAX",
    "ACTION_MIN",
    "ACTION_SCHEMA",
    "ACTION_SCHEMA_VERSION",
    "INTENT_CHALLENGE",
    "INTENT_CONTROL",
    "INTENT_MOVE",
    "INTENT_PASS",
    "INTENT_SHOOT",
    "PARAMETER_ALL",
    "PARAMETER_FORCE_DIRECTION",
    "PARAMETER_FORCE_POWER",
    "PARAMETER_LAUNCH",
    "PARAMETER_MOVE",
    "PARAMETER_SPIN",
    "ActionIntent",
    "DecodedIntentAction",
    "IntentAction",
    "neutral_action",
]
