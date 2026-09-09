"""Learnable policy contracts for low-frequency football management.

The contracts in this module are intentionally outside the environment
transition. A host runner observes and evaluates a manager only at an opening
or management boundary, then submits the returned proposal to the
authoritative environment transaction. Candidate rosters, benches, model
parameters, and policy memory therefore never enter the high-frequency player
``step`` carry.

Two boundaries are distinct:

* :class:`ManagerPolicy` proposes substitutions, formation changes, acting
  goalkeepers, and/or set-piece takers during a match.
* :class:`OpeningManagerPolicy` selects the registered squad, starting lineup,
  formation, and lineup-to-formation assignment. It is eligible only at the
  coherent initial boundary of a new match. Starting a runner from a later
  checkpoint must preserve the checkpoint roster and tactics.

Both definitions keep parameters as explicit dynamic PyTrees. Replacing a
checkpoint with the same leaf shapes and dtypes can therefore reuse the
compiled rare-boundary executable instead of making the weights part of a new
Python closure.

One manager jointly owns fixed-shape substitution and formation proposals,
observes only causal facts, and leaves legality to the environment. Policy and
parameter dispatch stay outside the main transition, and restart scheduling is
not coupled to a concrete rule-manager state. Putting either inside would add
rare bench/model work to every frame or prevent a
learned recurrent state from being substituted without changing the rollout
carry. :class:`ManagerBoundaryState` is therefore a separate four-integer
scheduler tree.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, NamedTuple, Protocol, TypeVar, runtime_checkable

import jax
import jax.numpy as jnp

from footballworld.core.constants import NO_PLAYER, RESTART_COUNT, RK_NONE
from footballworld.environment.management import ManagerCommand, ManagerFormationCommand
from footballworld.environment.normalization import NormalizedManagerObservation
from footballworld.policies.opening_formation import (
    NormalizedOpeningFormationObservation,
)

ParametersT = TypeVar("ParametersT")
StateT = TypeVar("StateT")
OpeningParametersT = TypeVar("OpeningParametersT")
OpeningStateT = TypeVar("OpeningStateT")


class NoPolicyParameters(NamedTuple):
    """Empty dynamic PyTree used by a rule policy with only static config."""


NO_POLICY_PARAMETERS = NoPolicyParameters()


class ManagerPolicyStep(NamedTuple):
    """One in-match proposal and the policy memory for the next boundary."""

    command: ManagerCommand
    state: StateT


class ManagerBoundaryState(NamedTuple):
    """Policy-independent identity of restart boundaries already handled.

    Only this four-int tree belongs beside ``make_managed_advance``. A learned
    manager's recurrent state can be arbitrarily shaped and must stay in the
    separately compiled decision path instead of entering every player frame.
    """

    processed_restart_tick: jax.Array
    processed_restart_kind: jax.Array


class ManagedManagerState(NamedTuple):
    """Host-side composition of scheduler memory and policy memory.

    Runners pass only ``boundary`` into the high-frequency managed scan and
    pass only ``policy`` into the low-frequency manager callable. Keeping the
    two leaves conceptually separate prevents the rollout scheduler from
    depending on a particular rule or learned model state layout.
    """

    boundary: ManagerBoundaryState
    policy: StateT


def initialize_manager_boundary_state() -> ManagerBoundaryState:
    """Return the fixed scheduler state before any restart was handled."""

    return ManagerBoundaryState(
        processed_restart_tick=jnp.full(2, -1, dtype=jnp.int32),
        processed_restart_kind=jnp.full(2, RK_NONE, dtype=jnp.int32),
    )


def manager_restart_unseen(
    state: ManagerBoundaryState,
    opened_control_tick: jax.Array,
    restart_kind: jax.Array,
) -> jax.Array:
    """Return a ``[2]`` exact-identity mask for the managed scan."""

    if type(state) is not ManagerBoundaryState:
        raise TypeError("state must be ManagerBoundaryState")
    for name in state._fields:
        value = getattr(state, name)
        if value.shape != (2,):
            raise ValueError(f"state.{name} must have shape [2]")
        if not jnp.issubdtype(value.dtype, jnp.integer):
            raise TypeError(f"state.{name} must have an integer dtype")
    tick = jnp.asarray(opened_control_tick)
    kind = jnp.asarray(restart_kind)
    if tick.shape != () or kind.shape != ():
        raise ValueError("restart identity must contain scalar tick and kind")
    if not jnp.issubdtype(tick.dtype, jnp.integer):
        raise TypeError("opened_control_tick must have an integer dtype")
    if not jnp.issubdtype(kind.dtype, jnp.integer):
        raise TypeError("restart_kind must have an integer dtype")
    valid = (tick >= 0) & (kind > RK_NONE) & (kind < RESTART_COUNT)
    return valid & (
        (state.processed_restart_tick != tick) | (state.processed_restart_kind != kind)
    )


def acknowledge_manager_boundary(
    state: ManagerBoundaryState,
    opened_control_tick: jax.Array,
    restart_kind: jax.Array,
    team_mask: jax.Array,
) -> ManagerBoundaryState:
    """Mark an authoritative boundary handled independently of policy state.

    Call this after a manager proposal has been adjudicated, including a
    legal no-op or rejected proposal. Otherwise the unchanged restart would
    immediately pause the next chunk again. Invalid or non-restart identities
    are ignored, while malformed shapes and dtypes fail before tracing.
    """

    unseen = manager_restart_unseen(state, opened_control_tick, restart_kind)
    mask = jnp.asarray(team_mask)
    if mask.shape != (2,):
        raise ValueError("team_mask must have shape [2]")
    if mask.dtype != jnp.dtype(jnp.bool_):
        raise TypeError("team_mask must have dtype bool")
    apply = unseen & mask
    tick = jnp.asarray(opened_control_tick, dtype=jnp.int32)
    kind = jnp.asarray(restart_kind, dtype=jnp.int32)
    return ManagerBoundaryState(
        processed_restart_tick=jnp.where(
            apply, tick, state.processed_restart_tick
        ).astype(jnp.int32),
        processed_restart_kind=jnp.where(
            apply, kind, state.processed_restart_kind
        ).astype(jnp.int32),
    )


@runtime_checkable
class ManagerPolicy(Protocol[ParametersT, StateT]):
    """Structural contract for a rule-based or learned in-match manager.

    ``initialize`` and ``step`` must be pure and traceable. The policy may
    propose an invalid command; command legality and application always remain
    the environment's responsibility.
    """

    def initialize(
        self,
        observations: NormalizedManagerObservation,
        parameters: ParametersT,
    ) -> StateT: ...

    def step(
        self,
        observations: NormalizedManagerObservation,
        match_key: jax.Array,
        state: StateT,
        parameters: ParametersT,
    ) -> ManagerPolicyStep[StateT]: ...


@dataclass(frozen=True, slots=True)
class FunctionalManagerPolicy(Generic[ParametersT, StateT]):
    """Small adapter for framework-defined manager functions.

    A Flax, Equinox, Haiku, or plain JAX caller can expose its own parameter
    and recurrent-state PyTrees without FootballWorld importing that
    framework. This definition is static; ``parameters`` and ``state`` are
    dynamic arguments to the methods.
    """

    initialize_fn: Callable[
        [NormalizedManagerObservation, ParametersT],
        StateT,
    ]
    step_fn: Callable[
        [NormalizedManagerObservation, jax.Array, StateT, ParametersT],
        ManagerPolicyStep[StateT],
    ]

    def __post_init__(self) -> None:
        if not callable(self.initialize_fn):
            raise TypeError("initialize_fn must be callable")
        if not callable(self.step_fn):
            raise TypeError("step_fn must be callable")

    def initialize(
        self,
        observations: NormalizedManagerObservation,
        parameters: ParametersT,
    ) -> StateT:
        return self.initialize_fn(observations, parameters)

    def step(
        self,
        observations: NormalizedManagerObservation,
        match_key: jax.Array,
        state: StateT,
        parameters: ParametersT,
    ) -> ManagerPolicyStep[StateT]:
        return self.step_fn(observations, match_key, state, parameters)


def validate_manager_policy(policy: object) -> None:
    """Validate the host-side structural manager boundary before tracing."""

    if not isinstance(policy, ManagerPolicy):
        raise TypeError("manager_policy must implement initialize and step")
    if not callable(policy.initialize) or not callable(policy.step):
        raise TypeError("manager policy initialize and step must be callable")


class OpeningPlayerPool(NamedTuple):
    """Normalized candidate players padded on one fixed pool axis.

    Every leaf starts with ``[2, candidates]``. ``preferred_position`` has one
    additional planar coordinate axis. ``player_id`` is carried for identity,
    output adaptation, and audit only; learned policies must not treat its
    numeric value or the candidate array order as an ability.
    """

    valid: jax.Array
    player_id: jax.Array
    is_goalkeeper: jax.Array
    preferred_position: jax.Array
    max_speed: jax.Array
    height: jax.Array
    reach_height: jax.Array
    ball_control: jax.Array
    endurance_factor: jax.Array


class OpeningManagerObservation(NamedTuple):
    """Causal model view available before the opening whistle only.

    ``formation.valid`` is computed by the existing exact start-only formation
    observer. It is false when a managed rollout begins from a mid-match
    checkpoint, so the opening policy produces a no-op and the existing roster,
    positions, and formation carry through unchanged.

    The candidate-player pool extends that contract to registered-squad,
    lineup, and placement decisions. The host-only
    ``prepare_opening_match_inputs`` transaction validates and materializes
    those leaves before reset; none enters the recurrent environment step.
    """

    players: OpeningPlayerPool
    formation: NormalizedOpeningFormationObservation
    max_registered_players: jax.Array
    starter_count: jax.Array


class OpeningManagerDecision(NamedTuple):
    """Fixed-shape proposal for roster, lineup, formation, and placement.

    The three player leaves are ``[2, candidates]``. ``registered`` selects
    the match squad, ``starter`` must be a subset of it, and
    ``placement_slot`` assigns each starter to a unique registered formation
    slot (``NO_PLAYER`` for a non-starter). ``formation`` reuses the existing
    public start-only formation command rather than inventing another action
    type.

    Selection is represented as masks rather than an ordered pointer list so
    candidate permutation can be handled equivariantly and a training loss
    need not learn arbitrary roster order. The environment remains the source
    of truth for counts, uniqueness, goalkeeper requirements, pitch geometry,
    and application.
    """

    registered: jax.Array
    starter: jax.Array
    placement_slot: jax.Array
    formation: ManagerFormationCommand

    @classmethod
    def empty(cls, candidate_count: int) -> OpeningManagerDecision:
        if (
            not isinstance(candidate_count, int)
            or isinstance(candidate_count, bool)
            or candidate_count < 0
        ):
            raise ValueError("candidate_count must be a non-negative integer")
        player_shape = (2, candidate_count)
        return cls(
            registered=jnp.zeros(player_shape, dtype=jnp.bool_),
            starter=jnp.zeros(player_shape, dtype=jnp.bool_),
            placement_slot=jnp.full(player_shape, NO_PLAYER, dtype=jnp.int32),
            formation=ManagerFormationCommand.empty(),
        )


class OpeningManagerPolicyStep(NamedTuple):
    """One start-only proposal and idempotence/recurrent policy memory."""

    decision: OpeningManagerDecision
    state: OpeningStateT


@runtime_checkable
class OpeningManagerPolicy(Protocol[OpeningParametersT, OpeningStateT]):
    """Structural contract for a rule-based or learned opening manager."""

    def initialize(
        self,
        observations: OpeningManagerObservation,
        parameters: OpeningParametersT,
    ) -> OpeningStateT: ...

    def step(
        self,
        observations: OpeningManagerObservation,
        match_key: jax.Array,
        state: OpeningStateT,
        parameters: OpeningParametersT,
    ) -> OpeningManagerPolicyStep[OpeningStateT]: ...


@dataclass(frozen=True, slots=True)
class FunctionalOpeningManagerPolicy(Generic[OpeningParametersT, OpeningStateT]):
    """Framework-neutral adapter for a learned start-only manager."""

    initialize_fn: Callable[
        [OpeningManagerObservation, OpeningParametersT],
        OpeningStateT,
    ]
    step_fn: Callable[
        [OpeningManagerObservation, jax.Array, OpeningStateT, OpeningParametersT],
        OpeningManagerPolicyStep[OpeningStateT],
    ]

    def __post_init__(self) -> None:
        if not callable(self.initialize_fn):
            raise TypeError("initialize_fn must be callable")
        if not callable(self.step_fn):
            raise TypeError("step_fn must be callable")

    def initialize(
        self,
        observations: OpeningManagerObservation,
        parameters: OpeningParametersT,
    ) -> OpeningStateT:
        return self.initialize_fn(observations, parameters)

    def step(
        self,
        observations: OpeningManagerObservation,
        match_key: jax.Array,
        state: OpeningStateT,
        parameters: OpeningParametersT,
    ) -> OpeningManagerPolicyStep[OpeningStateT]:
        return self.step_fn(observations, match_key, state, parameters)


def validate_opening_manager_policy(policy: object) -> None:
    """Validate the host-side structural opening boundary before tracing."""

    if not isinstance(policy, OpeningManagerPolicy):
        raise TypeError("policy must implement the OpeningManagerPolicy contract")
    if not callable(policy.initialize) or not callable(policy.step):
        raise TypeError("opening policy initialize and step must be callable")


def validate_opening_policy_shapes(
    observations: OpeningManagerObservation,
    decision: OpeningManagerDecision | None = None,
) -> tuple[int, int, int]:
    """Validate only the static array contract and return ``(C, L, S)``.

    Value legality deliberately stays in the authoritative opening
    transaction. Shape checks are Python-static and can run while tracing;
    they neither transfer arrays to the host nor add value checks to a JAX
    graph.
    """

    if type(observations) is not OpeningManagerObservation:
        raise TypeError("observations must be OpeningManagerObservation")
    if observations.max_registered_players.shape != (2,):
        raise ValueError("max_registered_players must have shape [2]")
    if observations.starter_count.shape != (2,):
        raise ValueError("starter_count must have shape [2]")

    if observations.players.valid.ndim != 2:
        raise ValueError("players.valid must have shape [2, C]")
    if observations.players.valid.shape[0] != 2:
        raise ValueError("players.valid must contain exactly two team rows")
    candidates = observations.players.valid.shape[1]
    candidate_shape = (2, candidates)
    for name in (
        "valid",
        "player_id",
        "is_goalkeeper",
        "max_speed",
        "height",
        "reach_height",
        "ball_control",
        "endurance_factor",
    ):
        if getattr(observations.players, name).shape != candidate_shape:
            raise ValueError(f"players.{name} must have shape {candidate_shape}")
    preferred_shape = (*candidate_shape, 2)
    if observations.players.preferred_position.shape != preferred_shape:
        raise ValueError(
            f"players.preferred_position must have shape {preferred_shape}"
        )

    formation = observations.formation
    if type(formation) is not NormalizedOpeningFormationObservation:
        raise TypeError("formation must be NormalizedOpeningFormationObservation")
    if formation.valid.shape != (2,):
        raise ValueError("formation.valid must have shape [2]")
    if formation.candidate_probability.ndim != 2:
        raise ValueError("formation candidate_probability must have shape [2, L]")
    if formation.candidate_probability.shape[0] != 2:
        raise ValueError("formation candidate_probability must have shape [2, L]")
    layouts = formation.candidate_probability.shape[1]
    anchor = formation.candidate_anchor
    if anchor.ndim != 4 or anchor.shape[:2] != (2, layouts) or anchor.shape[-1] != 2:
        raise ValueError("formation.candidate_anchor must have shape [2, L, N, 2]")
    starter_slots = anchor.shape[2]
    if formation.candidate_role.shape != (2, layouts, starter_slots):
        raise ValueError("formation.candidate_role must have shape [2, L, N]")

    expected_dtype = (
        ("players.valid", observations.players.valid, jnp.bool_),
        ("players.player_id", observations.players.player_id, jnp.int32),
        ("players.is_goalkeeper", observations.players.is_goalkeeper, jnp.bool_),
        ("formation.valid", formation.valid, jnp.bool_),
        ("formation.team", formation.team, jnp.int32),
        ("formation.ours_kickoff", formation.ours_kickoff, jnp.bool_),
        ("formation.candidate_role", formation.candidate_role, jnp.int32),
        (
            "max_registered_players",
            observations.max_registered_players,
            jnp.int32,
        ),
        ("starter_count", observations.starter_count, jnp.int32),
    )
    for name, value, dtype in expected_dtype:
        if value.dtype != jnp.dtype(dtype):
            raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}")
    for name in (
        "preferred_position",
        "max_speed",
        "height",
        "reach_height",
        "ball_control",
        "endurance_factor",
    ):
        if getattr(observations.players, name).dtype != jnp.dtype(jnp.float32):
            raise TypeError(f"players.{name} must have dtype float32")
    if anchor.dtype != jnp.dtype(jnp.float32):
        raise TypeError("formation.candidate_anchor must have dtype float32")
    if formation.candidate_probability.dtype != jnp.dtype(jnp.float32):
        raise TypeError("formation.candidate_probability must have dtype float32")

    if decision is not None:
        if type(decision) is not OpeningManagerDecision:
            raise TypeError("decision must be OpeningManagerDecision or None")
        for name in ("registered", "starter"):
            value = getattr(decision, name)
            if value.shape != candidate_shape or value.dtype != jnp.dtype(jnp.bool_):
                raise TypeError(
                    f"decision.{name} must have bool dtype and shape {candidate_shape}"
                )
        if (
            decision.placement_slot.shape != candidate_shape
            or decision.placement_slot.dtype != jnp.dtype(jnp.int32)
        ):
            raise TypeError(
                "decision.placement_slot must have int32 dtype and shape "
                f"{candidate_shape}"
            )
        if type(decision.formation) is not ManagerFormationCommand:
            raise TypeError("decision.formation must be ManagerFormationCommand")
        if decision.formation.requested.shape != (
            2,
        ) or decision.formation.requested.dtype != jnp.dtype(jnp.bool_):
            raise TypeError(
                "decision.formation.requested must have bool dtype and shape [2]"
            )
        if decision.formation.layout_index.shape != (
            2,
        ) or decision.formation.layout_index.dtype != jnp.dtype(jnp.int32):
            raise TypeError(
                "decision.formation.layout_index must have int32 dtype and shape [2]"
            )
    return candidates, layouts, starter_slots


__all__ = [
    "NO_POLICY_PARAMETERS",
    "FunctionalManagerPolicy",
    "FunctionalOpeningManagerPolicy",
    "ManagedManagerState",
    "ManagerBoundaryState",
    "ManagerPolicy",
    "ManagerPolicyStep",
    "NoPolicyParameters",
    "OpeningManagerDecision",
    "OpeningManagerObservation",
    "OpeningManagerPolicy",
    "OpeningManagerPolicyStep",
    "OpeningPlayerPool",
    "acknowledge_manager_boundary",
    "initialize_manager_boundary_state",
    "manager_restart_unseen",
    "validate_manager_policy",
    "validate_opening_manager_policy",
    "validate_opening_policy_shapes",
]
