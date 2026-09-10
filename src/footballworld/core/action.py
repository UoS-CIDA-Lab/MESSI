"""Authoritative fixed-shape player actions for one control frame."""

from enum import IntEnum
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from footballworld.core.action_mapping import (
    RadialControl,
    linf_radial_decode,
    signed_to_unit,
)
from footballworld.core.constants import (
    ACTION_MAX,
    ACTION_MIN,
    INTENT_ACTION_CONTINUOUS_DIM,
    INTENT_ACTION_FORCE_TO_BALL,
    INTENT_ACTION_GAZE_CENTER,
    INTENT_ACTION_LAUNCH,
    INTENT_ACTION_MOVE,
    INTENT_ACTION_SPIN_BACK,
    INTENT_ACTION_SPIN_SIDE,
)
from footballworld.core.contact import (
    ACTION_INTENT_COUNT,
    INTENT_CHALLENGE,
    INTENT_CLEAR,
    INTENT_CONTROL,
    INTENT_MOVE,
    INTENT_PASS,
    INTENT_SHOT,
)

ACTION_SCHEMA_VERSION = 2
ACTION_SCHEMA = f"footballworld.intent-action/{ACTION_SCHEMA_VERSION}"


def _canonicalize_intent(
    intent: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return submitted, effective, and invalid intent without narrowing aliases.

    Host integers are range-checked before conversion to JAX's canonical int32
    wire dtype. Values that cannot be represented use ``-1`` as the submitted
    telemetry sentinel and fail closed to MOVE. Traced arrays follow the same
    rule using dtype-aware JAX operations.
    """

    int32_limits = np.iinfo(np.int32)
    if isinstance(intent, (jax.Array, jax.core.Tracer)):
        source = jnp.asarray(intent)
        if jnp.issubdtype(source.dtype, jnp.bool_) or not jnp.issubdtype(
            source.dtype, jnp.integer
        ):
            raise TypeError("intent must have a non-boolean integer dtype")
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
            raise TypeError("intent must be an integer array") from exc
        if source.dtype == np.dtype(object):
            flat_values = tuple(source.flat)
            if any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                for value in flat_values
            ):
                raise TypeError("intent must have a non-boolean integer dtype")
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
                raise TypeError("intent must have a non-boolean integer dtype")
            representable = source <= int32_limits.max
            if np.issubdtype(source.dtype, np.signedinteger):
                representable &= source >= int32_limits.min
            narrowed_source = np.where(
                representable,
                source,
                np.zeros((), dtype=source.dtype),
            )
            submitted_host = narrowed_source.astype(np.int32)
            submitted_host = np.where(
                representable, submitted_host, np.int32(-1)
            ).astype(np.int32)
        submitted = jnp.asarray(submitted_host, dtype=jnp.int32)

    valid = (submitted >= INTENT_MOVE) & (submitted < ACTION_INTENT_COUNT)
    effective = jnp.where(valid, submitted, jnp.int32(INTENT_MOVE))
    return submitted, effective, ~valid


class ActionIntent(IntEnum):
    """The complete public categorical vocabulary for player intent."""

    MOVE = INTENT_MOVE
    CONTROL = INTENT_CONTROL
    PASS = INTENT_PASS
    SHOT = INTENT_SHOT
    CLEAR = INTENT_CLEAR
    CHALLENGE = INTENT_CHALLENGE


class DecodedIntentAction(NamedTuple):
    """Normalized controls paired with one of six explicit action intents."""

    intent: jax.Array
    contact: jax.Array
    move: RadialControl
    force_to_ball: RadialControl
    launch: jax.Array
    spin: jax.Array
    gaze_center: jax.Array


class IntentAction(NamedTuple):
    """Six-way categorical intent plus eight continuous physical controls.

    The intent is never packed into the float array. Valid codes are exactly
    MOVE, CONTROL, PASS, SHOT, CLEAR, and CHALLENGE. Invalid integer values
    fail closed to MOVE, including under JIT; non-integer intent arrays are
    rejected before tracing.
    """

    intent: jax.Array
    move: jax.Array
    force_to_ball: jax.Array
    launch: jax.Array
    spin: jax.Array
    gaze_center: jax.Array

    @classmethod
    def from_array(
        cls,
        intent: jax.Array,
        continuous: jax.Array,
    ) -> "IntentAction":
        """Sanitize a categorical intent and a separate trailing 8-vector."""

        continuous = jnp.asarray(continuous, dtype=jnp.float32)
        if continuous.shape[-1:] != (INTENT_ACTION_CONTINUOUS_DIM,):
            raise ValueError(
                "continuous trailing dimension must be "
                f"{INTENT_ACTION_CONTINUOUS_DIM}, got {continuous.shape}"
            )
        _, effective_intent, _ = _canonicalize_intent(intent)
        if effective_intent.shape != continuous.shape[:-1]:
            raise ValueError(
                "intent shape must equal the continuous prefix, got "
                f"{effective_intent.shape} and {continuous.shape[:-1]}"
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
                controls[..., INTENT_ACTION_SPIN_SIDE],
                controls[..., INTENT_ACTION_SPIN_BACK],
            ),
            axis=-1,
        )
        return cls(
            intent=effective_intent,
            move=controls[..., INTENT_ACTION_MOVE],
            force_to_ball=controls[..., INTENT_ACTION_FORCE_TO_BALL],
            launch=controls[..., INTENT_ACTION_LAUNCH],
            spin=spin,
            gaze_center=controls[..., INTENT_ACTION_GAZE_CENTER],
        )

    def as_continuous_array(self) -> jax.Array:
        """Return only the eight continuous controls; intent stays categorical."""

        return jnp.concatenate(
            (
                self.move,
                self.force_to_ball,
                self.launch[..., None],
                self.spin,
                self.gaze_center[..., None],
            ),
            axis=-1,
        ).astype(jnp.float32)

    def decode(self) -> DecodedIntentAction:
        """Decode continuous controls without weakening categorical intent."""

        sanitized = self.from_array(self.intent, self.as_continuous_array())
        return DecodedIntentAction(
            intent=sanitized.intent,
            contact=sanitized.intent != INTENT_MOVE,
            move=linf_radial_decode(sanitized.move),
            force_to_ball=linf_radial_decode(sanitized.force_to_ball),
            launch=signed_to_unit(sanitized.launch),
            spin=sanitized.spin,
            gaze_center=sanitized.gaze_center,
        )

    @classmethod
    def neutral(cls, player_count: int) -> "IntentAction":
        """Return MOVE intent with zero continuous controls."""

        return cls.from_array(
            jnp.full(player_count, INTENT_MOVE, dtype=jnp.int32),
            jnp.zeros(
                (player_count, INTENT_ACTION_CONTINUOUS_DIM),
                dtype=jnp.float32,
            ),
        )

    @classmethod
    def move_only(cls, move: jax.Array) -> "IntentAction":
        """Return MOVE intent with the supplied normalized movement."""

        move = jnp.asarray(move, dtype=jnp.float32)
        if move.shape[-1:] != (2,):
            raise ValueError(f"move trailing dimension must be 2, got {move.shape}")
        prefix = move.shape[:-1]
        continuous = jnp.zeros(
            (*prefix, INTENT_ACTION_CONTINUOUS_DIM), dtype=jnp.float32
        )
        continuous = continuous.at[..., INTENT_ACTION_MOVE].set(move)
        return cls.from_array(
            jnp.full(prefix, INTENT_MOVE, dtype=jnp.int32),
            continuous,
        )


def neutral_action(player_count: int) -> IntentAction:
    """Return the canonical hierarchical MOVE action batch."""

    return IntentAction.neutral(player_count)
