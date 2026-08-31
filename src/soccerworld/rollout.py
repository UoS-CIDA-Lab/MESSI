"""Controlled batching for compiled SoccerWorld rollouts.

JAX is imported only when :func:`batch_rollout` is called. Importing this module cannot initialize
a backend, so applications may configure device visibility and compilation caching first.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Literal, TypeVar, cast

BatchStrategy = Literal["lax_map", "vmap"]

_Function = TypeVar("_Function", bound=Callable[..., Any])
_STRATEGIES = frozenset({"lax_map", "vmap"})

__all__ = ["BatchStrategy", "batch_rollout"]


def batch_rollout(
    rollout: _Function,
    *,
    strategy: BatchStrategy = "lax_map",
) -> _Function:
    """Batch one-match rollout inputs without changing its output PyTree.

    Every input is an array or PyTree whose leaves share a leading match axis. The return value has
    exactly the single-match output tree with that axis stacked in front of every leaf.

    The conservative default is an outer sequential ``lax.map`` on every backend. It preserves the
    scalar ``lax.cond`` kernels that ``vmap`` can turn into dense selects in the current 11v11
    physics graph. ``strategy="vmap"`` remains an explicit override for workloads whose controlled
    receipt demonstrates a benefit and satisfies that workload's numerical gate. Both transforms
    preserve the output PyTree and independent-match contract, but compiler lowering can change
    float32 results inside a policy; do not assume byte-exact interchangeability without a receipt.

    JAX is loaded at this factory boundary, never when :mod:`soccerworld.rollout` is imported.
    """

    if not callable(rollout):
        raise TypeError(f"rollout must be callable, got {type(rollout).__name__}")
    if strategy not in _STRATEGIES:
        raise ValueError(f"unsupported rollout batch strategy: {strategy!r}")

    import jax  # Imported deliberately at the factory boundary.

    if strategy == "vmap":
        transformed = jax.vmap(rollout)

        @wraps(rollout)
        def batched(*batched_inputs: Any) -> Any:
            return transformed(*batched_inputs)

    else:

        @wraps(rollout)
        def batched(*batched_inputs: Any) -> Any:
            if not batched_inputs:
                raise TypeError("a batched rollout requires at least one positional input")
            return jax.lax.map(
                lambda inputs: rollout(*inputs),
                batched_inputs,
            )

    batched.batch_strategy = strategy  # type: ignore[attr-defined]
    return cast(_Function, batched)
