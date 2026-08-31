"""Canonical causal records returned by compiled SoccerWorld rollouts."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
from jax import Array

from soccerworld._schema_versions import TRANSITION_SCHEMA
from soccerworld.core.commands import StepCommand
from soccerworld.core.results import CommandResult
from soccerworld.data.events import EventBatch

__all__ = [
    "ActionProvenance",
    "DatasetMetadata",
    "FrameIdentity",
    "ManagerProvenance",
    "TRANSITION_SCHEMA",
    "TransitionRecord",
]


class FrameIdentity(NamedTuple):
    """Stable episode, frame, slot, and roster identity."""

    episode_seed: Array
    control_tick: Array
    player_id: Array
    slot_generation: Array


class ActionProvenance(NamedTuple):
    """Per-dimension causal masks needed by imitation-learning materializers."""

    parameter_consumed: Array
    kick_applied: Array
    move_forced: Array
    kick_gated: Array
    kick_forced: Array
    halftime_reset: Array

    @classmethod
    def empty(cls) -> ActionProvenance:
        return cls(
            parameter_consumed=jnp.empty((0, 0), dtype=jnp.bool_),
            kick_applied=jnp.empty((0,), dtype=jnp.bool_),
            move_forced=jnp.empty((0,), dtype=jnp.bool_),
            kick_gated=jnp.empty((0,), dtype=jnp.bool_),
            kick_forced=jnp.empty((0,), dtype=jnp.bool_),
            halftime_reset=jnp.empty((0,), dtype=jnp.bool_),
        )


class ManagerProvenance(NamedTuple):
    """Whether the internal manager ran and the observation-bounded view it consumed."""

    called: Array
    view: Any

    @classmethod
    def empty(cls) -> ManagerProvenance:
        return cls(called=jnp.bool_(False), view=None)


class TransitionRecord(NamedTuple):
    """One explicitly ordered ``pre -> command -> post`` transition.

    A static :class:`~soccerworld.data.capture.CaptureSpec` may represent omitted dense fields as
    zero-width arrays or select a smaller internal result type.  Storage adapters must preserve this
    causal ordering and the schema metadata rather than infer it from filenames or row position.
    """

    pre: FrameIdentity
    observation: Array
    command: StepCommand
    command_result: CommandResult
    reward: Array
    terminated: Array
    truncated: Array
    events: EventBatch
    action_provenance: ActionProvenance
    manager_provenance: ManagerProvenance
    substeps: Any
    post: FrameIdentity


class DatasetMetadata(NamedTuple):
    """Array-friendly reproducibility metadata stored once per dataset shard."""

    schema_version: Array
    dynamics_fingerprint: Array
    policy_fingerprint: Array
    config_digest: Array
    timebase: Array
