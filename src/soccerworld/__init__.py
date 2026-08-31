"""Lazy public package boundary for SoccerWorld.

The top-level package imports no numerical or rendering stack. This lets command-line bootstrap
modules select a JAX backend before JAX is imported, while documented exports remain available on
demand.
"""

from __future__ import annotations

from importlib import import_module

from soccerworld._version import __version__

_LAZY_EXPORTS = {
    "Agent": ("soccerworld.api", "Agent"),
    "Ball": ("soccerworld.api", "Ball"),
    "BallState": ("soccerworld.tactics", "BallState"),
    "BallBoundaryEventCode": (
        "soccerworld.data.events",
        "BallBoundaryEventCode",
    ),
    "BenchPlayer": ("soccerworld.api", "BenchPlayer"),
    "DEFAULT_MAX_SUBSTITUTIONS": (
        "soccerworld.api",
        "DEFAULT_MAX_SUBSTITUTIONS",
    ),
    "batch_rollout": ("soccerworld.rollout", "batch_rollout"),
    "CaptureSpec": ("soccerworld.data.capture", "CaptureSpec"),
    "CommandReason": ("soccerworld.core.results", "CommandReason"),
    "CommandResult": ("soccerworld.core.results", "CommandResult"),
    "DEFAULT_TIMEBASE": ("soccerworld.timebase", "DEFAULT_TIMEBASE"),
    "DecisionSource": ("soccerworld.core.results", "DecisionSource"),
    "DisciplinaryOutcome": (
        "soccerworld.data.events",
        "DisciplinaryOutcome",
    ),
    "enable_compilation_cache": (
        "soccerworld.runtime",
        "enable_compilation_cache",
    ),
    "Engine": ("soccerworld.api", "Engine"),
    "EventBatch": ("soccerworld.data.events", "EventBatch"),
    "EventDomain": ("soccerworld.data.events", "EventDomain"),
    "FORMATION_LAYOUT_NAMES": (
        "soccerworld.tactics",
        "FORMATION_LAYOUT_NAMES",
    ),
    "Foul": ("soccerworld.api", "Foul"),
    "FoulEventCode": ("soccerworld.data.events", "FoulEventCode"),
    "FormationCommand": ("soccerworld.core.commands", "FormationCommand"),
    "Reward": ("soccerworld.api", "Reward"),
    "RandomEvent": ("soccerworld.api", "RandomEvent"),
    "RandomnessControl": ("soccerworld.api", "RandomnessControl"),
    "RestartKind": ("soccerworld.tactics", "RestartKind"),
    "RulePolicy": ("soccerworld.api", "RulePolicy"),
    "SetPieceTakerCommand": (
        "soccerworld.core.commands",
        "SetPieceTakerCommand",
    ),
    "SoccerEnv": ("soccerworld.api", "SoccerEnv"),
    "Stadium": ("soccerworld.api", "Stadium"),
    "StepCommand": ("soccerworld.core.commands", "StepCommand"),
    "Substitution": ("soccerworld.api", "Substitution"),
    "SubstitutionCommand": (
        "soccerworld.core.commands",
        "SubstitutionCommand",
    ),
    "Timebase": ("soccerworld.timebase", "Timebase"),
    "Team": ("soccerworld.tactics", "Team"),
    "TouchEventCode": ("soccerworld.data.events", "TouchEventCode"),
    "TransitionRecord": ("soccerworld.data.records", "TransitionRecord"),
    "TransitionResult": ("soccerworld.core.results", "TransitionResult"),
    "WoodworkEventCode": ("soccerworld.data.events", "WoodworkEventCode"),
}

__all__ = [*_LAZY_EXPORTS, "__version__"]


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
