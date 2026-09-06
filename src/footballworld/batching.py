"""Controlled batching for scalar whole-match rollout kernels.

JAX is imported only when :func:`batch_rollout` is called. The wrapper does
not add a compilation boundary: callers remain responsible for applying and
reusing :func:`jax.jit` around the fully composed rollout.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

_Function = TypeVar("_Function", bound=Callable[..., Any])

__all__ = ["batch_rollout"]


def _validate_batched_inputs(jax: Any, batched_inputs: tuple[Any, ...]) -> None:
    """Reject malformed leading match axes before the batch transform."""

    if not batched_inputs:
        raise TypeError("a batched rollout requires at least one positional input")

    leaves = jax.tree_util.tree_leaves(batched_inputs)
    if not leaves:
        raise TypeError("batched rollout inputs must contain at least one array leaf")

    batch_size: int | None = None
    for leaf_index, leaf in enumerate(leaves):
        shape = getattr(leaf, "shape", None)
        if shape is None:
            raise TypeError(
                f"batched rollout leaf {leaf_index} must be an array with a leading match axis"
            )
        if len(shape) == 0:
            raise ValueError(
                f"batched rollout leaf {leaf_index} is scalar; every leaf needs a leading match axis"
            )

        leaf_batch_size = shape[0]
        if leaf_batch_size == 0:
            raise ValueError("batched rollout match axis must be non-empty")
        if batch_size is None:
            batch_size = leaf_batch_size
        elif leaf_batch_size != batch_size:
            raise ValueError(
                "all batched rollout leaves must share one leading match axis; "
                f"expected {batch_size}, got {leaf_batch_size} at leaf {leaf_index}"
            )


def batch_rollout(one_match: _Function) -> _Function:
    """Batch independent inputs to a scalar whole-match rollout.

    Every positional input must be an array or PyTree whose leaves share the
    same non-empty leading match axis. Every returned leaf has the scalar
    rollout's time-major result stacked behind that match axis, yielding
    ``[B, T, ...]`` for trajectories.

    ``lax.map`` is the only whole-match transform. It preserves scalar control
    flow and avoids the compile-time and runtime expansion measured for dense
    vectorization of FootballWorld's branch-heavy transition.

    This factory never calls :func:`jax.jit`. Compile only after composing the
    scalar rollout and this outer match transform, and reuse that executable.
    """

    if not callable(one_match):
        raise TypeError(f"one_match must be callable, got {type(one_match).__name__}")

    import jax  # Imported deliberately at the public factory boundary.

    @wraps(one_match)
    def batched(*batched_inputs: Any) -> Any:
        _validate_batched_inputs(jax, batched_inputs)
        return jax.lax.map(
            lambda inputs: one_match(*inputs),
            batched_inputs,
        )

    return cast(_Function, batched)
