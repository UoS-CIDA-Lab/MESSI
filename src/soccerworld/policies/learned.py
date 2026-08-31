"""Lazy adapter boundary for externally supplied learned policies.

SoccerWorld deliberately does not own a checkpoint format. A project can pass
``package.module:factory::checkpoint`` to the demo; the factory is imported only
when selected and must return :class:`LearnedPolicy`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any

NORMALIZED_PLAYER_ACTION_OUTPUT = "soccerworld.normalized-player-action/1"


@dataclass(frozen=True)
class LearnedPolicy:
    """JAX-compatible learned policy supplied by an external adapter factory."""

    parameters: Any
    apply_fn: Callable[[Any, Any, Any, Any], Any]
    fingerprint: str
    metadata: Mapping[str, Any]
    parameter_count: int | None = None
    checkpoint_step: int | None = None

    def __post_init__(self) -> None:
        if not callable(self.apply_fn):
            raise TypeError("apply_fn must be callable")
        if not isinstance(self.fingerprint, str) or not self.fingerprint:
            raise ValueError("fingerprint must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

    def apply_with_parameters(self, parameters, observation, key, affordance):
        return self.apply_fn(parameters, observation, key, affordance)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "learned",
            "fingerprint": self.fingerprint,
            "checkpoint_step": self.checkpoint_step,
            "parameter_count": self.parameter_count,
            "adapter_metadata": dict(self.metadata),
        }


def load_learned_policy(
    spec: str,
    *,
    observation_spec: Mapping[str, Any],
    deterministic: bool,
) -> LearnedPolicy:
    """Resolve an external factory without coupling SoccerWorld to its framework.

    The factory is called as ``factory(checkpoint, observation_spec=...,
    deterministic=...)``. Import paths and checkpoint payloads stay outside the
    compiled environment and may refer to any framework chosen by the caller.
    """

    adapter, separator, checkpoint = spec.partition("::")
    module_name, callable_separator, factory_name = adapter.partition(":")
    if not separator or not checkpoint or not callable_separator or not factory_name:
        raise ValueError(
            "learned policy must use "
            "'package.module:factory::checkpoint'; SoccerWorld does not bundle "
            "a legacy checkpoint loader"
        )
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"learned-policy adapter module {module_name!r} is not installed"
        ) from exc
    try:
        factory = getattr(module, factory_name)
    except AttributeError as exc:
        raise ImportError(
            f"learned-policy adapter {module_name!r} has no factory {factory_name!r}"
        ) from exc
    if not callable(factory):
        raise TypeError(f"learned-policy adapter {adapter!r} is not callable")
    loaded = factory(
        checkpoint,
        observation_spec=observation_spec,
        deterministic=bool(deterministic),
    )
    if not isinstance(loaded, LearnedPolicy):
        raise TypeError(
            f"learned-policy adapter {adapter!r} must return LearnedPolicy, "
            f"got {type(loaded).__name__}"
        )
    return loaded


__all__ = (
    "NORMALIZED_PLAYER_ACTION_OUTPUT",
    "LearnedPolicy",
    "load_learned_policy",
)
