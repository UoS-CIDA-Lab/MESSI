"""Batch orchestration for independently interrupted managed matches.

A conservative outer ``lax.map`` is used because dense match vectorization can
expand branch-heavy football graphs. Per-match step budgets and manager masks
do not change ``FootballWorld.step``.
Manager models, benches, opening placement, and roster refresh stay in separate
low-frequency executables.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from footballworld.batching import batch_rollout
from footballworld.environment.api import (
    FootballWorld,
    ManagerState,
    Rollout,
    SquadSetup,
)
from footballworld.environment.episode import MatchSetup
from footballworld.environment.management import ManagerCommand
from footballworld.environment.observation import RosterMetadata
from footballworld.policies.manager import (
    NO_POLICY_PARAMETERS,
    ManagedManagerState,
    ManagerPolicy,
    acknowledge_manager_boundary,
    initialize_manager_boundary_state,
)
from footballworld.policies.opening_formation import (
    OpeningFormationPolicy,
    RuleBasedOpeningFormationPolicy,
    observe_opening_formations,
)
from footballworld.policies.rule_based.manager import make_rule_based_manager
from footballworld.policies.rule_based.policy import (
    RuleBasedPolicy,
    make_rule_based_policy,
)
from footballworld.policies.rule_based.state import RulePolicyState
from footballworld.rollout import (
    apply_management_tactics,
    make_managed_advance,
    refresh_policy_state,
)


class ManagedBatchState(NamedTuple):
    """Batch-major recurrent values owned by the host between chunks."""

    rollout: Rollout
    setup: MatchSetup
    management: ManagerState
    roster: RosterMetadata
    player_policy_state: RulePolicyState
    manager: ManagedManagerState[Any]


class ManagedBatchRunResult(NamedTuple):
    """Per-match exact progress and low-frequency decision counts."""

    state: ManagedBatchState
    steps_executed: np.ndarray
    manager_decisions: np.ndarray
    opening_formation_applied: np.ndarray


class _ManagerDecisionResult(NamedTuple):
    rollout: Rollout
    management: ManagerState
    policy_state: Any
    boundary_state: Any
    roster_metadata_changed: jax.Array


class _RosterRefreshResult(NamedTuple):
    roster: RosterMetadata
    player_policy_state: RulePolicyState


class _OpeningDecisionResult(NamedTuple):
    rollout: Rollout
    setup: MatchSetup
    management: ManagerState
    applied: jax.Array


def _batch_size(rollout: Rollout) -> int:
    if not isinstance(rollout, Rollout):
        raise TypeError("rollout must be a batch-major Rollout")
    position = rollout.state.players.position
    if position.ndim != 3 or position.shape[-1] != 2:
        raise ValueError("batched player positions must have shape [B, N, 2]")
    if position.shape[0] < 1:
        raise ValueError("managed batch must contain at least one match")
    return position.shape[0]


def _validate_batch_tree(name: str, value: Any, batch_size: int) -> None:
    for index, leaf in enumerate(jax.tree_util.tree_leaves(value)):
        shape = getattr(leaf, "shape", None)
        if shape is None or len(shape) == 0 or shape[0] != batch_size:
            raise ValueError(
                f"{name} leaf {index} must have leading match axis {batch_size}"
            )


def _validate_keys(keys: jax.Array, batch_size: int) -> None:
    try:
        data = jax.random.key_data(keys)
    except (TypeError, ValueError) as exc:
        raise TypeError("match_keys must be a batch of JAX PRNG keys") from exc
    if data.shape != (batch_size, 2):
        raise ValueError(f"match_keys must contain exactly {batch_size} keys")


def _horizons(value: int | np.ndarray | jax.Array, batch_size: int) -> np.ndarray:
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        if int(value) < 0:
            raise ValueError("num_steps must be non-negative")
        result = np.full(batch_size, int(value), dtype=np.int64)
    else:
        raw = np.asarray(jax.device_get(value))
        if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(
            raw.dtype, np.bool_
        ):
            raise TypeError("num_steps array must use a non-boolean integer dtype")
        if raw.shape == ():
            result = np.full(batch_size, int(raw), dtype=np.int64)
        elif raw.shape == (batch_size,):
            result = raw.astype(np.int64, copy=True)
        else:
            raise ValueError(f"num_steps must be scalar or have shape [{batch_size}]")
        if np.any(result < 0):
            raise ValueError("num_steps must be non-negative")
    if np.any(result > np.iinfo(np.int32).max):
        raise ValueError("num_steps exceeds the supported int32 step budget")
    return result


def _map_shared(one_match):
    """Map match inputs while sharing one dynamic parameter PyTree."""

    def mapped(match_inputs, parameters):
        return jax.lax.map(lambda inputs: one_match(inputs, parameters), match_inputs)

    return mapped


@dataclass(frozen=True, slots=True)
class ManagedBatchRunner:
    """Reusable scheduler for a fixed-shape batch of managed matches."""

    env: FootballWorld
    player_policy: RuleBasedPolicy
    chunk_steps: int
    manager_policy: ManagerPolicy | None
    opening_policy: OpeningFormationPolicy | None
    _advance: Any
    _manager_initialize: Any
    _manager_decide: Any
    _acknowledge: Any
    _refresh_roster: Any
    _apply_tactics: Any
    _opening_decide: Any
    _can_handle_goalkeeper: bool

    def initialize(
        self,
        rollout: Rollout,
        setup: MatchSetup,
        squad: SquadSetup,
        management: ManagerState,
        roster: RosterMetadata,
        player_policy_state: RulePolicyState,
        manager_parameters: Any = NO_POLICY_PARAMETERS,
    ) -> ManagedBatchState:
        """Initialize scheduler and manager memory without advancing."""

        batch_size = _batch_size(rollout)
        for name, value in (
            ("setup", setup),
            ("squad", squad),
            ("management", management),
            ("roster", roster),
            ("player_policy_state", player_policy_state),
        ):
            _validate_batch_tree(name, value, batch_size)
        base = initialize_manager_boundary_state()
        boundary = jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, (batch_size, *leaf.shape)), base
        )
        policy_state = (
            NO_POLICY_PARAMETERS
            if self.manager_policy is None
            else self._manager_initialize(
                (rollout, squad, management), manager_parameters
            )
        )
        return ManagedBatchState(
            rollout=rollout,
            setup=setup,
            management=management,
            roster=roster,
            player_policy_state=player_policy_state,
            manager=ManagedManagerState(boundary, policy_state),
        )

    def run(
        self,
        state: ManagedBatchState,
        squad: SquadSetup,
        match_keys: jax.Array,
        num_steps: int | np.ndarray | jax.Array,
        *,
        manager_parameters: Any = NO_POLICY_PARAMETERS,
        apply_opening_formation: bool = True,
    ) -> ManagedBatchRunResult:
        """Advance each row by its exact requested transition horizon."""

        if type(state) is not ManagedBatchState:
            raise TypeError("state must be ManagedBatchState")
        if type(apply_opening_formation) is not bool:
            raise TypeError("apply_opening_formation must be bool")
        batch_size = _batch_size(state.rollout)
        for name, value in (
            ("squad", squad),
            ("setup", state.setup),
            ("management", state.management),
            ("roster", state.roster),
            ("player_policy_state", state.player_policy_state),
            ("manager.boundary", state.manager.boundary),
        ):
            _validate_batch_tree(name, value, batch_size)
        if self.manager_policy is not None:
            _validate_batch_tree("manager.policy", state.manager.policy, batch_size)
        _validate_keys(match_keys, batch_size)

        remaining = _horizons(num_steps, batch_size)
        current = state
        opening_applied = np.zeros((batch_size, 2), dtype=np.bool_)
        if apply_opening_formation and self._opening_decide is not None:
            opening = self._opening_decide(
                current.rollout,
                current.setup,
                squad,
                current.management,
                match_keys,
            )
            opening_applied = np.asarray(
                jax.device_get(opening.applied), dtype=np.bool_
            )
            current = current._replace(
                rollout=opening.rollout,
                setup=opening.setup,
                management=opening.management,
            )
            opening_rows = np.any(opening_applied, axis=1)
            if np.any(opening_rows):
                current = current._replace(
                    player_policy_state=self._apply_tactics(
                        current.rollout,
                        current.management,
                        current.roster,
                        current.player_policy_state,
                        jnp.asarray(opening_rows),
                    )
                )

        executed = np.zeros(batch_size, dtype=np.int64)
        decisions = np.zeros(batch_size, dtype=np.int64)
        repeated_boundaries = np.zeros(batch_size, dtype=np.int32)
        while np.any(remaining > 0):
            was_active = remaining > 0
            budget = np.minimum(remaining, self.chunk_steps).astype(np.int32)
            advance = self._advance(
                current.rollout,
                current.setup,
                current.roster,
                current.player_policy_state,
                current.manager.boundary,
                jnp.asarray(budget),
                match_keys,
            )
            progressed = np.asarray(
                jax.device_get(advance.steps_executed), dtype=np.int64
            )
            if np.any(progressed < 0) or np.any(progressed > budget):
                raise RuntimeError("managed batch returned an invalid step count")
            executed += progressed
            remaining -= progressed
            current = current._replace(
                rollout=advance.final_rollout,
                player_policy_state=advance.final_policy_state,
            )

            manager_rows = (
                np.asarray(jax.device_get(advance.manager_required), dtype=np.bool_)
                & was_active
            )
            unexpected = was_active & (progressed == 0) & (~manager_rows)
            if np.any(unexpected):
                raise RuntimeError(
                    "managed batch made no progress without a boundary: "
                    f"rows={np.flatnonzero(unexpected).tolist()}"
                )
            if not np.any(manager_rows):
                repeated_boundaries[:] = 0
                continue

            goalkeeper_rows = manager_rows & np.any(
                np.asarray(
                    jax.device_get(advance.goalkeeper_team_mask), dtype=np.bool_
                ),
                axis=1,
            )
            if np.any(goalkeeper_rows) and not self._can_handle_goalkeeper:
                raise RuntimeError(
                    "missing-goalkeeper management is unavailable for batch rows "
                    f"{np.flatnonzero(goalkeeper_rows).tolist()}"
                )

            active = jnp.asarray(manager_rows)
            if self._manager_decide is None:
                boundary = self._acknowledge(
                    current.manager.boundary,
                    current.rollout,
                    advance.manager_team_mask,
                    active,
                )
                current = current._replace(
                    manager=current.manager._replace(boundary=boundary)
                )
            else:
                decision = self._manager_decide(
                    (
                        current.rollout,
                        squad,
                        current.management,
                        current.manager.boundary,
                        advance.manager_team_mask,
                        current.manager.policy,
                        match_keys,
                        active,
                    ),
                    manager_parameters,
                )
                decisions += manager_rows.astype(np.int64)
                previous_roster = current.roster
                current = current._replace(
                    rollout=decision.rollout,
                    management=decision.management,
                    manager=ManagedManagerState(
                        decision.boundary_state, decision.policy_state
                    ),
                )
                changed = manager_rows & np.asarray(
                    jax.device_get(decision.roster_metadata_changed), dtype=np.bool_
                )
                if np.any(changed):
                    refreshed = self._refresh_roster(
                        current.rollout,
                        current.management,
                        previous_roster,
                        current.player_policy_state,
                        jnp.asarray(changed),
                    )
                    current = current._replace(
                        roster=refreshed.roster,
                        player_policy_state=refreshed.player_policy_state,
                    )
                current = current._replace(
                    player_policy_state=self._apply_tactics(
                        current.rollout,
                        current.management,
                        current.roster,
                        current.player_policy_state,
                        active,
                    )
                )

            stalled = manager_rows & (progressed == 0)
            repeated_boundaries = np.where(stalled, repeated_boundaries + 1, 0)
            if np.any(repeated_boundaries > 2):
                raise RuntimeError(
                    "managed batch repeated a zero-progress boundary: "
                    f"rows={np.flatnonzero(repeated_boundaries > 2).tolist()}"
                )

        return ManagedBatchRunResult(
            state=current,
            steps_executed=executed,
            manager_decisions=decisions,
            opening_formation_applied=opening_applied,
        )


def make_managed_batch_runner(
    env: FootballWorld,
    player_policy: RuleBasedPolicy | None = None,
    chunk_steps: int = 256,
    *,
    manager_policy: ManagerPolicy | None = None,
    opening_policy: OpeningFormationPolicy | None = None,
) -> ManagedBatchRunner:
    """Build a batch runner whose low-frequency graphs remain separate."""

    if not isinstance(env, FootballWorld):
        raise TypeError("env must be FootballWorld")
    selected_player = player_policy
    if selected_player is None:
        if not env.policies.rule_based_player:
            raise ValueError(
                "player_policy is required when the built-in player policy is disabled"
            )
        selected_player = make_rule_based_policy(env)
    if not isinstance(selected_player, RuleBasedPolicy):
        raise TypeError("player_policy must be RuleBasedPolicy or None")
    if not isinstance(chunk_steps, int) or isinstance(chunk_steps, bool):
        raise TypeError("chunk_steps must be an integer")
    if chunk_steps < 1:
        raise ValueError("chunk_steps must be positive")

    selected_manager = manager_policy
    manager_default = env.policies.rule_based_match_manager
    taker_default = env.policies.rule_based_set_piece_taker
    using_reference_manager = selected_manager is None and (
        manager_default or taker_default
    )
    if using_reference_manager:
        selected_manager = make_rule_based_manager(env)
    selected_opening = opening_policy
    if selected_opening is None and env.policies.rule_based_opening_formation_adapter:
        selected_opening = RuleBasedOpeningFormationPolicy()

    advance = jax.jit(
        batch_rollout(make_managed_advance(env, selected_player, chunk_steps))
    )

    manager_initialize = None
    manager_decide = None
    if selected_manager is not None:

        def initialize_one(inputs, parameters):
            rollout, squad, management = inputs
            observations = env.observe_managers(rollout, squad, management)
            return selected_manager.initialize(observations, parameters)

        def decide_one(inputs, parameters):
            (
                rollout,
                squad,
                management,
                boundary,
                team_mask,
                policy_state,
                match_key,
                active,
            ) = inputs

            def decide(_):
                observations = env.observe_managers(rollout, squad, management)
                proposal = selected_manager.step(
                    observations, match_key, policy_state, parameters
                )
                command = proposal.command
                if using_reference_manager:
                    empty = ManagerCommand.empty(
                        command.substitutions.requested.shape[1]
                    )
                    if not manager_default:
                        command = command._replace(
                            substitutions=empty.substitutions,
                            formations=empty.formations,
                            acting_goalkeepers=empty.acting_goalkeepers,
                        )
                    if not taker_default:
                        command = command._replace(
                            set_piece_takers=empty.set_piece_takers
                        )
                result = env.manager_command(rollout, squad, management, command)
                acknowledged = acknowledge_manager_boundary(
                    boundary,
                    rollout.state.restart.opened_control_tick,
                    rollout.state.restart.kind,
                    team_mask,
                )
                return _ManagerDecisionResult(
                    result.rollout,
                    result.management,
                    proposal.state,
                    acknowledged,
                    result.roster_metadata_changed,
                )

            return jax.lax.cond(
                active,
                decide,
                lambda _: _ManagerDecisionResult(
                    rollout,
                    management,
                    policy_state,
                    boundary,
                    jnp.bool_(False),
                ),
                operand=None,
            )

        manager_initialize = jax.jit(_map_shared(initialize_one))
        manager_decide = jax.jit(_map_shared(decide_one))

    def acknowledge_one(boundary, rollout, team_mask, active):
        return jax.lax.cond(
            active,
            lambda _: acknowledge_manager_boundary(
                boundary,
                rollout.state.restart.opened_control_tick,
                rollout.state.restart.kind,
                team_mask,
            ),
            lambda _: boundary,
            operand=None,
        )

    acknowledge = jax.jit(batch_rollout(acknowledge_one))

    def refresh_one(rollout, management, old_roster, policy_state, active):
        def refresh(_):
            roster = env.roster_metadata_si(rollout, management)
            return _RosterRefreshResult(
                roster,
                refresh_policy_state(
                    env,
                    selected_player,
                    rollout,
                    old_roster,
                    roster,
                    policy_state,
                ),
            )

        return jax.lax.cond(
            active,
            refresh,
            lambda _: _RosterRefreshResult(old_roster, policy_state),
            operand=None,
        )

    refresh_roster = jax.jit(batch_rollout(refresh_one))

    def apply_tactics_one(rollout, management, roster, policy_state, active):
        return jax.lax.cond(
            active,
            lambda _: apply_management_tactics(
                env, selected_player, rollout, management, roster, policy_state
            ),
            lambda _: policy_state,
            operand=None,
        )

    apply_tactics = jax.jit(batch_rollout(apply_tactics_one))

    opening_decide = None
    if selected_opening is not None:

        def opening_one(rollout, setup, squad, management, match_key):
            observations = observe_opening_formations(
                rollout.state, squad, management, env.normalization_context()
            )
            command = selected_opening(observations, match_key)
            result = env.opening_formation_command(
                rollout, setup, squad, management, command
            )
            return _OpeningDecisionResult(
                result.rollout, result.setup, result.management, result.applied
            )

        opening_decide = jax.jit(batch_rollout(opening_one))

    return ManagedBatchRunner(
        env=env,
        player_policy=selected_player,
        chunk_steps=chunk_steps,
        manager_policy=selected_manager,
        opening_policy=selected_opening,
        _advance=advance,
        _manager_initialize=manager_initialize,
        _manager_decide=manager_decide,
        _acknowledge=acknowledge,
        _refresh_roster=refresh_roster,
        _apply_tactics=apply_tactics,
        _opening_decide=opening_decide,
        _can_handle_goalkeeper=(manager_policy is not None) or manager_default,
    )


__all__ = [
    "ManagedBatchRunResult",
    "ManagedBatchRunner",
    "ManagedBatchState",
    "make_managed_batch_runner",
]
