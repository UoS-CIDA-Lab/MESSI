"""Authoritative fixed-shape player actions for one control frame."""

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

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
        intent = jnp.asarray(intent)
        if not jnp.issubdtype(intent.dtype, jnp.integer):
            raise TypeError("intent must have an integer dtype")
        if intent.shape != continuous.shape[:-1]:
            raise ValueError(
                "intent shape must equal the continuous prefix, got "
                f"{intent.shape} and {continuous.shape[:-1]}"
            )
        valid = (intent >= INTENT_MOVE) & (intent < ACTION_INTENT_COUNT)
        intent = jnp.where(valid, intent, INTENT_MOVE).astype(jnp.int32)
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
            intent=intent,
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
