"""Lazy public helpers backing the shipped single-match demo."""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "build_env": ("soccerworld.demo.roster", "build_env"),
    "main": ("soccerworld.demo.__main__", "main"),
    "make_bench": ("soccerworld.demo.roster", "make_bench"),
    "rollout_single_match": (
        "soccerworld.demo.rollout",
        "rollout_single_match",
    ),
    "summarize": ("soccerworld.demo.reporting", "summarize"),
    "TeamPolicyAdapter": ("soccerworld.demo.policy", "TeamPolicyAdapter"),
}

__all__ = list(_LAZY_EXPORTS)


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
