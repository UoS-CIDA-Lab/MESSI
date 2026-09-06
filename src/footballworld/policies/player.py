"""Framework-neutral contract for high-frequency player policies.

Player policies are pure functions of the current SI observation, immutable
roster metadata, explicit recurrent state, and the match key.  A rollout
factory captures only the policy implementation; every numerical parameter
that callers want to replace without retracing should live in the explicit
state PyTree.

The state may have any JAX-compatible PyTree structure, but one compiled scan
requires its tree, leaf shapes, and leaf dtypes to remain fixed.  FootballWorld
checks that contract while the scan body is traced, before backend execution.
Environment legality remains authoritative: a policy emits an
:class:`~footballworld.core.action.IntentAction`, and the environment still
sanitizes and adjudicates that action.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, NamedTuple, Protocol, TypeVar, runtime_checkable

import jax
import jax.numpy as jnp

from footballworld.core.action import IntentAction
from footballworld.environment.observation import Observation, RosterMetadata

PlayerPolicyStateT = TypeVar("PlayerPolicyStateT")


class PlayerPolicyStep(NamedTuple):
    """One fixed-shape action and the policy memory for the next frame."""

    action: IntentAction
    state: PlayerPolicyStateT


@runtime_checkable
class PlayerPolicy(Protocol[PlayerPolicyStateT]):
    """Structural contract accepted by FootballWorld rollout factories.

    Both methods must be pure and JAX-traceable.  ``match_key`` is immutable;
    policies should derive stable event keys from observed causal identities
    instead of consuming a Python or host-side random stream.
    """

    def initialize(
        self,
        observations: Observation,
        roster: RosterMetadata,
    ) -> PlayerPolicyStateT: ...

    def step(
        self,
        observations: Observation,
        roster: RosterMetadata,
        state: PlayerPolicyStateT,
        match_key: jax.Array,
    ) -> PlayerPolicyStep[PlayerPolicyStateT]: ...


@dataclass(frozen=True, slots=True)
class FunctionalPlayerPolicy(Generic[PlayerPolicyStateT]):
    """Small adapter for plain-JAX player-policy functions.

    Learned parameters can be leaves of ``PlayerPolicyStateT`` so they remain
    dynamic rollout inputs.  Keeping them out of this frozen adapter avoids
    embedding replaceable weights as constants in the compiled executable.
    """

    initialize_fn: Callable[[Observation, RosterMetadata], PlayerPolicyStateT]
    step_fn: Callable[
        [Observation, RosterMetadata, PlayerPolicyStateT, jax.Array],
        PlayerPolicyStep[PlayerPolicyStateT],
    ]

    def __post_init__(self) -> None:
        if not callable(self.initialize_fn):
            raise TypeError("initialize_fn must be callable")
        if not callable(self.step_fn):
            raise TypeError("step_fn must be callable")

    def initialize(
        self,
        observations: Observation,
        roster: RosterMetadata,
    ) -> PlayerPolicyStateT:
        return self.initialize_fn(observations, roster)

    def step(
        self,
        observations: Observation,
        roster: RosterMetadata,
        state: PlayerPolicyStateT,
        match_key: jax.Array,
    ) -> PlayerPolicyStep[PlayerPolicyStateT]:
        return self.step_fn(observations, roster, state, match_key)


def validate_player_policy(policy: object) -> None:
    """Validate the host-side structural policy boundary before tracing."""

    if not isinstance(policy, PlayerPolicy):
        raise TypeError(
            "policy must implement PlayerPolicy.initialize and PlayerPolicy.step"
        )
    if not callable(policy.initialize) or not callable(policy.step):
        raise TypeError("player policy initialize and step attributes must be callable")


def validate_player_policy_step(
    step: object,
    previous_state: PlayerPolicyStateT,
) -> None:
    """Reject an invalid policy result while the scan body is being traced.

    Shape and dtype comparisons are Python-static for both concrete arrays and
    JAX tracers.  They therefore improve errors without adding operations to
    the compiled rollout graph.
    """

    action = getattr(step, "action", None)
    if not isinstance(action, IntentAction):
        raise TypeError("player policy step.action must be IntentAction")
    if not hasattr(step, "state"):
        raise TypeError("player policy step result must expose a state field")
    next_state = step.state
    previous_structure = jax.tree_util.tree_structure(previous_state)
    next_structure = jax.tree_util.tree_structure(next_state)
    if previous_structure != next_structure:
        raise TypeError(
            "player policy state PyTree structure must remain fixed across a step"
        )
    previous_leaves = jax.tree_util.tree_leaves(previous_state)
    next_leaves = jax.tree_util.tree_leaves(next_state)
    for index, (previous, current) in enumerate(
        zip(previous_leaves, next_leaves, strict=True)
    ):
        try:
            previous_array = jnp.asarray(previous)
            current_array = jnp.asarray(current)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"player policy state leaf {index} must be JAX-array compatible"
            ) from exc
        previous_shape = previous_array.shape
        current_shape = current_array.shape
        previous_dtype = previous_array.dtype
        current_dtype = current_array.dtype
        if previous_shape != current_shape:
            raise ValueError(
                f"player policy state leaf {index} changed shape from "
                f"{previous_shape} to {current_shape}"
            )
        if previous_dtype != current_dtype:
            raise TypeError(
                f"player policy state leaf {index} changed dtype from "
                f"{previous_dtype} to {current_dtype}"
            )


__all__ = [
    "FunctionalPlayerPolicy",
    "PlayerPolicy",
    "PlayerPolicyStateT",
    "PlayerPolicyStep",
    "validate_player_policy",
    "validate_player_policy_step",
]
