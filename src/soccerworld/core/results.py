"""Stable results for submitted and environment-applied commands."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, NamedTuple

from jax import Array

from soccerworld._schema_versions import COMMAND_RESULT_SCHEMA

__all__ = [
    "COMMAND_RESULT_SCHEMA",
    "ActionResult",
    "CommandReason",
    "CommandResult",
    "DecisionSource",
    "FormationResult",
    "SetPieceTakerResult",
    "SubstitutionResult",
    "TransitionResult",
]


class CommandReason(IntEnum):
    """Domain-neutral reason codes stored as ``int32`` arrays.

    Codes are append-only after the first public schema release. Invalid array shapes and dtypes are
    host API errors; these values describe well-formed proposals adjudicated by the environment.
    """

    NOT_REQUESTED = 0
    APPLIED = 1
    ACCEPTED_PENDING = 2
    INTERNAL_FALLBACK = 3
    ACTION_MASKED = 4
    INVALID_SLOT = 5
    INACTIVE_PLAYER = 6
    BENCH_UNAVAILABLE = 7
    NOT_DEAD_BALL = 8
    SUBSTITUTION_WINDOW_EXHAUSTED = 9
    SUBSTITUTION_LIMIT_EXHAUSTED = 10
    GOALKEEPER_CONSTRAINT = 11
    INVALID_FORMATION = 12
    NO_MATCHING_RESTART = 13
    WRONG_TEAM = 14
    INELIGIBLE_TAKER = 15
    TERMINAL = 16
    ALREADY_CHANGED = 17
    PLACEMENT_FAILED = 18
    # Deprecated reserved wire value. Kept so historical traces retain their
    # numeric meaning; current formation-command producers never emit it.
    FORMATION_IN_TRANSITION = 19
    UNCHANGED = 20


class DecisionSource(IntEnum):
    """Origin of a proposal independently selected for each control axis."""

    INTERNAL = 0
    EXTERNAL = 1


class ActionResult(NamedTuple):
    source: Array
    submitted: Array
    normalized: Array
    applied: Array
    agency: Array
    consumed: Array
    reason: Array


class SubstitutionResult(NamedTuple):
    source: Array
    requested: Array
    accepted: Array
    applied: Array
    out_slot: Array
    bench_index: Array
    reason: Array


class FormationResult(NamedTuple):
    source: Array
    requested: Array
    accepted: Array
    applied: Array
    layout_index: Array
    reason: Array


class SetPieceTakerResult(NamedTuple):
    source: Array
    requested: Array
    accepted: Array
    applied: Array
    player_slot: Array
    reason: Array


class CommandResult(NamedTuple):
    """What the environment consumed, accepted, applied, deferred, or rejected."""

    actions: ActionResult
    substitutions: SubstitutionResult
    formations: FormationResult
    set_piece_takers: SetPieceTakerResult


class TransitionResult(NamedTuple):
    """Typed result of one public command transition.

    ``record`` and ``info`` are selected by static output options and may be ``None``. Keeping them
    in the PyTree lets callers JIT a complete rollout without converting device arrays to host data.
    """

    observation: Array
    state: Any
    reward: Array
    terminated: Array
    truncated: Array
    done: Array
    command_result: CommandResult
    record: Any
    info: Any
