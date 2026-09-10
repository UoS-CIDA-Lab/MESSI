"""One control-frame decoding of the public continuous action."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.core.action import IntentAction, _canonicalize_intent
from footballworld.core.constants import ACTION_MAX, ACTION_MIN, SAFE_NORM_EPS
from footballworld.core.contact import INTENT_SOURCE_POLICY
from footballworld.core.state import State


class PhysicsAction(NamedTuple):
    """Compact controls held fixed across every physics substep in a frame."""

    contact: jax.Array
    desired_velocity: jax.Array
    force_direction: jax.Array
    force_power: jax.Array
    launch: jax.Array
    spin: jax.Array
    gaze_center: jax.Array
    body_target: jax.Array
    body_target_valid: jax.Array
    requested_intent: jax.Array
    intent_source: jax.Array


class ActionTrace(NamedTuple):
    """Compact submitted-intent provenance retained once per eventful frame."""

    requested_intent: jax.Array
    intent_source: jax.Array
    executed: jax.Array


# Fixed wire meanings for the eventful-only action receipt.  These are causal
# telemetry bits, not selectable actions or football coefficients.
ACTION_TRACE_SCHEMA = "footballworld.action-trace/1"
ACTION_RECEIPT_SCHEMA = "footballworld.action-receipt/1"

ACTION_FLAG_INVALID_INTENT = 1 << 0
ACTION_FLAG_NONFINITE_CONTINUOUS = 1 << 1
ACTION_FLAG_CONTINUOUS_CLIPPED = 1 << 2
ACTION_FLAG_INTENT_AVAILABLE = 1 << 3
ACTION_FLAG_CONTACT_ATTEMPTED = 1 << 4
ACTION_FLAG_ACTIVE_CONTACT = 1 << 5
ACTION_FLAG_PASSIVE_CONTACT = 1 << 6
ACTION_FLAG_PARAMETERS_APPLIED = 1 << 7
ACTION_FLAG_KICK_APPLIED = 1 << 8
ACTION_FLAG_BODY_ONLY_DEFLECTION = 1 << 9
ACTION_FLAG_FORCED_RELEASE = 1 << 10
ACTION_FLAG_REFEREE_PROJECTION = 1 << 11
ACTION_FLAG_ENVIRONMENT_OVERWRITE = 1 << 12
ACTION_FLAG_TERMINAL_SUPPRESSED = 1 << 13
ACTION_FLAG_NAMES = (
    "INVALID_INTENT",
    "NONFINITE_CONTINUOUS",
    "CONTINUOUS_CLIPPED",
    "INTENT_AVAILABLE",
    "CONTACT_ATTEMPTED",
    "ACTIVE_CONTACT",
    "PASSIVE_CONTACT",
    "PARAMETERS_APPLIED",
    "KICK_APPLIED",
    "BODY_ONLY_DEFLECTION",
    "FORCED_RELEASE",
    "REFEREE_PROJECTION",
    "ENVIRONMENT_OVERWRITE",
    "TERMINAL_SUPPRESSED",
)

ELIGIBILITY_PHASE_RULE = 1 << 0
ELIGIBILITY_HORIZONTAL_REACH = 1 << 1
ELIGIBILITY_HEIGHT = 1 << 2
ELIGIBILITY_SPEED = 1 << 3
ELIGIBILITY_RECOVERY = 1 << 4
ELIGIBILITY_PHYSICAL_CANDIDATE = 1 << 5
ELIGIBILITY_NAMES = (
    "PHASE_RULE",
    "HORIZONTAL_REACH",
    "HEIGHT",
    "SPEED",
    "RECOVERY",
    "PHYSICAL_CANDIDATE",
)

PARAMETER_MOVE = 1 << 0
PARAMETER_FORCE_DIRECTION = 1 << 1
PARAMETER_FORCE_POWER = 1 << 2
PARAMETER_LAUNCH = 1 << 3
PARAMETER_SPIN = 1 << 4
PARAMETER_GAZE = 1 << 5
PARAMETER_NAMES = (
    "MOVE",
    "FORCE_DIRECTION",
    "FORCE_POWER",
    "LAUNCH",
    "SPIN",
    "GAZE",
)

DISPLACEMENT_SELF_MOTION = 1 << 0
DISPLACEMENT_COLLISION_VELOCITY = 1 << 1
DISPLACEMENT_SEPARATION_POSITION = 1 << 2
DISPLACEMENT_REFEREE_PROJECTION = 1 << 3
DISPLACEMENT_HALFTIME_RESET = 1 << 4
DISPLACEMENT_RESTART_APPROACH = 1 << 5
DISPLACEMENT_SOURCE_NAMES = (
    "SELF_MOTION",
    "COLLISION_VELOCITY",
    "SEPARATION_POSITION",
    "REFEREE_PROJECTION",
    "HALFTIME_RESET",
    "RESTART_APPROACH",
)

ACTION_REASON_NONE = 0
ACTION_REASON_MOVE_CONSUMED = 1
ACTION_REASON_INTENT_UNAVAILABLE = 2
ACTION_REASON_CONTACT_NOT_REACHED = 3
ACTION_REASON_CONTACT_ATTEMPTED_NO_REALIZATION = 4
ACTION_REASON_CONTACT_REALIZED = 5
ACTION_REASON_PARAMETERS_APPLIED = 6
ACTION_REASON_PASSIVE_CONTACT = 7
ACTION_REASON_FORCED_RELEASE = 8
ACTION_REASON_REFEREE_PROJECTION = 9
ACTION_REASON_ENVIRONMENT_OVERWRITE = 10
ACTION_REASON_TERMINAL_SUPPRESSED = 11
ACTION_REASON_INPUT_SANITIZED = 12

ACTION_REASON_NAMES = (
    "NONE",
    "MOVE_CONSUMED",
    "INTENT_UNAVAILABLE",
    "CONTACT_NOT_REACHED",
    "CONTACT_ATTEMPTED_NO_REALIZATION",
    "CONTACT_REALIZED",
    "PARAMETERS_APPLIED",
    "PASSIVE_CONTACT",
    "FORCED_RELEASE",
    "REFEREE_PROJECTION",
    "ENVIRONMENT_OVERWRITE",
    "TERMINAL_SUPPRESSED",
    "INPUT_SANITIZED",
)


class ActionReceipt(NamedTuple):
    """Per-player causal explanation emitted only by eventful transitions.

    Independent bit sets retain facts that a single UI-oriented
    ``primary_reason`` cannot.  The receipt is privileged telemetry and never
    becomes a player observation.
    """

    requested_intent: jax.Array
    effective_intent: jax.Array
    flags: jax.Array
    eligibility_seen: jax.Array
    primary_reason: jax.Array
    parameter_consumed: jax.Array
    displacement_source: jax.Array


def trace_action_receipt(
    action: IntentAction,
    *,
    executed: jax.Array = True,
    _effective_intent: jax.Array | None = None,
) -> ActionReceipt:
    """Return input-sanitization and terminal facts for one eventful frame.

    Keep input repair, structural availability, reach, realization, and
    parameter use as independent causal sidecar facts rather than collapsing
    them into a flat agency mask. This preserves
    those facts independently.  This function intentionally sees the action
    before :func:`decode_physics_action` sanitizes it. The transition may pass
    the already-sanitized intent privately so event capture does not repeat
    the full continuous-control normalization performed by physics.
    """

    if not isinstance(action, IntentAction):
        raise TypeError("action must be IntentAction")
    submitted_intent, canonical_effective, invalid_intent = _canonicalize_intent(
        action.intent
    )
    submitted_continuous = action.as_continuous_array()
    if _effective_intent is None:
        effective_intent = canonical_effective
    else:
        effective_intent = jnp.asarray(_effective_intent, dtype=jnp.int32)
        if effective_intent.shape != submitted_intent.shape:
            raise ValueError(
                "_effective_intent must match action.intent shape, got "
                f"{effective_intent.shape} and {submitted_intent.shape}"
            )
    nonfinite = ~jnp.all(jnp.isfinite(submitted_continuous), axis=-1)
    clipped = jnp.any(
        jnp.isfinite(submitted_continuous)
        & ((submitted_continuous < ACTION_MIN) | (submitted_continuous > ACTION_MAX)),
        axis=-1,
    )
    flags = (
        invalid_intent.astype(jnp.uint32) * jnp.uint32(ACTION_FLAG_INVALID_INTENT)
        | nonfinite.astype(jnp.uint32) * jnp.uint32(ACTION_FLAG_NONFINITE_CONTINUOUS)
        | clipped.astype(jnp.uint32) * jnp.uint32(ACTION_FLAG_CONTINUOUS_CLIPPED)
    )
    executed = jnp.asarray(executed, dtype=jnp.bool_)
    terminal = jnp.broadcast_to(~executed, submitted_intent.shape)
    flags = flags | terminal.astype(jnp.uint32) * jnp.uint32(
        ACTION_FLAG_TERMINAL_SUPPRESSED
    )
    sanitized_input = invalid_intent | nonfinite | clipped
    primary_reason = jnp.where(
        terminal,
        jnp.int16(ACTION_REASON_TERMINAL_SUPPRESSED),
        jnp.where(
            sanitized_input,
            jnp.int16(ACTION_REASON_INPUT_SANITIZED),
            jnp.int16(ACTION_REASON_NONE),
        ),
    )
    zeros_u16 = jnp.zeros(submitted_intent.shape, dtype=jnp.uint16)
    return ActionReceipt(
        requested_intent=submitted_intent.astype(jnp.int32),
        effective_intent=effective_intent,
        flags=flags,
        eligibility_seen=zeros_u16,
        primary_reason=primary_reason,
        parameter_consumed=zeros_u16,
        displacement_source=zeros_u16,
    )


def trace_action(
    action: IntentAction,
    *,
    executed: jax.Array = True,
) -> ActionTrace:
    """Trace the submitted categorical intent for one control frame."""

    if not isinstance(action, IntentAction):
        raise TypeError("action must be IntentAction")
    submitted_intent, _, _ = _canonicalize_intent(action.intent)
    return ActionTrace(
        requested_intent=submitted_intent.astype(jnp.int32),
        intent_source=jnp.full_like(
            submitted_intent, INTENT_SOURCE_POLICY, dtype=jnp.int32
        ),
        executed=jnp.asarray(executed, dtype=jnp.bool_),
    )


def decode_physics_action(
    state: State,
    action: IntentAction,
) -> PhysicsAction:
    """Decode all state-invariant action transforms exactly once."""

    if not isinstance(action, IntentAction):
        raise TypeError("action must be IntentAction")
    decoded = action.decode()
    team_rotation = state.attack_direction[state.players.team_id, None]
    move_planar = decoded.move.direction * decoded.move.power[:, None]
    desired_velocity = move_planar * team_rotation * state.players.max_speed[:, None]
    force_direction = decoded.force_to_ball.direction * team_rotation
    move_direction = decoded.move.direction * team_rotation
    body_aim_valid = (~decoded.contact) & (
        decoded.force_to_ball.power > jnp.finfo(jnp.float32).eps
    )
    move_aim_valid = (
        (~decoded.contact)
        & (~body_aim_valid)
        & (decoded.move.power > jnp.finfo(jnp.float32).eps)
    )
    body_target = jnp.where(
        body_aim_valid[:, None],
        force_direction,
        jnp.where(move_aim_valid[:, None], move_direction, state.players.body_forward),
    )

    spin = jnp.clip(jnp.asarray(decoded.spin), -1.0, 1.0)
    spin_norm = jnp.sqrt(jnp.sum(spin * spin, axis=-1) + SAFE_NORM_EPS)
    spin = spin / jnp.maximum(1.0, spin_norm)[:, None]
    return PhysicsAction(
        contact=decoded.contact,
        desired_velocity=desired_velocity,
        force_direction=force_direction,
        force_power=decoded.force_to_ball.power,
        launch=decoded.launch,
        spin=spin,
        gaze_center=decoded.gaze_center,
        body_target=body_target,
        body_target_valid=body_aim_valid | move_aim_valid,
        requested_intent=decoded.intent,
        intent_source=jnp.full_like(
            decoded.contact, INTENT_SOURCE_POLICY, dtype=jnp.int32
        ),
    )


__all__ = [
    "ACTION_FLAG_NAMES",
    "ACTION_REASON_NAMES",
    "ACTION_RECEIPT_SCHEMA",
    "ACTION_TRACE_SCHEMA",
    "DISPLACEMENT_SOURCE_NAMES",
    "ELIGIBILITY_NAMES",
    "PARAMETER_NAMES",
    "ActionReceipt",
    "ActionTrace",
    "PhysicsAction",
    "decode_physics_action",
    "trace_action",
    "trace_action_receipt",
]
