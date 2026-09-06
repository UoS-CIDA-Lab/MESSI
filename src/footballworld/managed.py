"""Host orchestration for exact-horizon managed football rollouts.

The player controller remains a fixed-size compiled scan.  Opening formation,
manager observations, bench decisions, roster refresh, and tactical-anchor
updates are separate rare-boundary executables dispatched by the host.  This
keeps every manager tensor out of :meth:`FootballWorld.step` and out of the
high-frequency scan carry.

The host loop may synchronize once per player chunk and once per management
boundary.  It always reuses the immutable match key and asks the scan for the
remaining runtime budget, so splitting a match at management boundaries does
not change transition randomness or overshoot the requested number of steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

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
from footballworld.policies.rule_based.manager import (
    make_rule_based_manager,
)
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


class ManagedMatchState(NamedTuple):
    """All recurrent values owned by the host between managed chunks."""

    rollout: Rollout
    setup: MatchSetup
    management: ManagerState
    roster: RosterMetadata
    player_policy_state: RulePolicyState
    manager: ManagedManagerState[Any]


class ManagedRunResult(NamedTuple):
    """Exact requested progress plus low-frequency decision counts."""

    state: ManagedMatchState
    steps_executed: int
    manager_decisions: int
    opening_formation_applied: np.ndarray


class _GenericManagerDecisionResult(NamedTuple):
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


def _one_key(key: jax.Array) -> None:
    try:
        data = jax.random.key_data(key)
    except (TypeError, ValueError) as exc:
        raise TypeError("match_key must be one JAX PRNG key") from exc
    if data.shape != (2,):
        raise ValueError("match_key must be one unbatched JAX PRNG key")


def _host_bool(value: jax.Array) -> bool:
    array = np.asarray(jax.device_get(value))
    if array.shape != ():
        raise ValueError("expected one scalar boundary flag")
    return bool(array)


def _host_int(value: jax.Array) -> int:
    array = np.asarray(jax.device_get(value))
    if array.shape != ():
        raise ValueError("expected one scalar step count")
    return int(array)


@dataclass(frozen=True, slots=True)
class ManagedRunner:
    """Reusable scalar host runner composed from independent JIT kernels.

    A caller-provided ``manager_policy`` takes precedence over environment
    reference-policy switches.  Its parameter and recurrent-state PyTrees are
    dynamic inputs to the low-frequency executable, so replacing learned
    weights with identical shapes does not retrace the player scan.
    """

    env: FootballWorld
    player_policy: RuleBasedPolicy
    chunk_steps: int
    manager_policy: ManagerPolicy | None
    opening_policy: OpeningFormationPolicy | None
    _advance: Any
    _manager_initialize: Any
    _manager_decide: Any
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
    ) -> ManagedMatchState:
        """Build scheduler and optional manager memory without advancing."""

        boundary = initialize_manager_boundary_state()
        if self.manager_policy is None:
            manager_state = NO_POLICY_PARAMETERS
        else:
            manager_state = self._manager_initialize(
                rollout,
                squad,
                management,
                manager_parameters,
            )
        return ManagedMatchState(
            rollout=rollout,
            setup=setup,
            management=management,
            roster=roster,
            player_policy_state=player_policy_state,
            manager=ManagedManagerState(boundary=boundary, policy=manager_state),
        )

    def run(
        self,
        state: ManagedMatchState,
        squad: SquadSetup,
        match_key: jax.Array,
        num_steps: int,
        *,
        manager_parameters: Any = NO_POLICY_PARAMETERS,
        apply_opening_formation: bool = True,
    ) -> ManagedRunResult:
        """Advance one scalar match by exactly ``num_steps`` transitions.

        Opening placement is queried before the first manager boundary only;
        its environment predicate makes a mid-match checkpoint a no-op.  A
        manager proposal, including a rejected or legal no-op proposal,
        acknowledges the exact restart identity so it cannot be re-issued on
        the next chunk.
        """

        if type(state) is not ManagedMatchState:
            raise TypeError("state must be ManagedMatchState")
        if not isinstance(squad, SquadSetup):
            raise TypeError("squad must be SquadSetup")
        if not isinstance(num_steps, int) or isinstance(num_steps, bool):
            raise TypeError("num_steps must be an integer")
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if type(apply_opening_formation) is not bool:
            raise TypeError("apply_opening_formation must be a bool")
        _one_key(match_key)

        current = state
        opening_applied = np.zeros(2, dtype=np.bool_)
        if apply_opening_formation and self._opening_decide is not None:
            opening = self._opening_decide(
                current.rollout,
                current.setup,
                squad,
                current.management,
                match_key,
            )
            opening_applied = np.asarray(
                jax.device_get(opening.applied), dtype=np.bool_
            )
            current = current._replace(
                rollout=opening.rollout,
                setup=opening.setup,
                management=opening.management,
            )
            if bool(np.any(opening_applied)):
                player_state = self._apply_tactics(
                    current.rollout,
                    current.management,
                    current.roster,
                    current.player_policy_state,
                )
                current = current._replace(player_policy_state=player_state)

        executed = 0
        decisions = 0
        # A valid decision must either acknowledge a restart or change the
        # goalkeeper condition. The small guard turns accidental non-progress
        # into an explicit host error instead of an infinite loop.
        zero_progress_boundaries = 0
        while executed < num_steps:
            budget = min(self.chunk_steps, num_steps - executed)
            advance = self._advance(
                current.rollout,
                current.setup,
                current.roster,
                current.player_policy_state,
                current.manager.boundary,
                jnp.int32(budget),
                match_key,
            )
            progressed = _host_int(advance.steps_executed)
            executed += progressed
            current = current._replace(
                rollout=advance.final_rollout,
                player_policy_state=advance.final_policy_state,
            )

            if not _host_bool(advance.manager_required):
                zero_progress_boundaries = 0
                continue

            goalkeeper_required = bool(
                np.any(jax.device_get(advance.goalkeeper_team_mask))
            )
            if goalkeeper_required and not self._can_handle_goalkeeper:
                raise RuntimeError(
                    "management is required for a missing goalkeeper, but "
                    "the selected policy configuration cannot handle it"
                )
            if self._manager_decide is None:
                boundary = acknowledge_manager_boundary(
                    current.manager.boundary,
                    current.rollout.state.restart.opened_control_tick,
                    current.rollout.state.restart.kind,
                    advance.manager_team_mask,
                )
                current = current._replace(
                    manager=current.manager._replace(boundary=boundary)
                )
            else:
                decision = self._manager_decide(
                    current.rollout,
                    squad,
                    current.management,
                    current.manager.boundary,
                    advance.manager_team_mask,
                    current.manager.policy,
                    manager_parameters,
                    match_key,
                )
                decisions += 1
                previous_roster = current.roster
                current = current._replace(
                    rollout=decision.rollout,
                    management=decision.management,
                    manager=ManagedManagerState(
                        boundary=decision.boundary_state,
                        policy=decision.policy_state,
                    ),
                )
                if _host_bool(decision.roster_metadata_changed):
                    refreshed = self._refresh_roster(
                        current.rollout,
                        current.management,
                        previous_roster,
                        current.player_policy_state,
                    )
                    current = current._replace(
                        roster=refreshed.roster,
                        player_policy_state=refreshed.player_policy_state,
                    )
                player_state = self._apply_tactics(
                    current.rollout,
                    current.management,
                    current.roster,
                    current.player_policy_state,
                )
                current = current._replace(player_policy_state=player_state)

            if progressed == 0:
                zero_progress_boundaries += 1
                if zero_progress_boundaries > 2:
                    raise RuntimeError(
                        "managed runner made no progress across repeated boundaries"
                    )
            else:
                zero_progress_boundaries = 0

        return ManagedRunResult(
            state=current,
            steps_executed=executed,
            manager_decisions=decisions,
            opening_formation_applied=opening_applied,
        )


def make_managed_runner(
    env: FootballWorld,
    player_policy: RuleBasedPolicy | None = None,
    chunk_steps: int = 256,
    *,
    manager_policy: ManagerPolicy | None = None,
    opening_policy: OpeningFormationPolicy | None = None,
) -> ManagedRunner:
    """Build and compile-independent scalar kernels for managed execution.

    When a policy is omitted, the matching static ``env.policies`` switch
    decides whether FootballWorld installs its reference implementation.
    Supplying a learned manager or explicit player policy takes precedence
    over its reference-policy switch.  With both manager axes disabled, the
    runner only acknowledges ordinary restart boundaries; a missing
    goalkeeper still requires a caller manager. The two rule-manager axes are
    statically masked, so a disabled command component adds no runtime branch.
    """

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
    if (
        selected_opening is None
        and env.policies.rule_based_opening_formation_adapter
    ):
        selected_opening = RuleBasedOpeningFormationPolicy()

    advance = jax.jit(make_managed_advance(env, selected_player, chunk_steps))

    manager_initialize = None
    manager_decide = None
    if selected_manager is not None:

        def initialize_manager_with_squad(rollout, squad, management, parameters):
            observations = env.observe_managers(rollout, squad, management)
            return selected_manager.initialize(observations, parameters)

        def decide_manager(
            rollout,
            squad,
            management,
            boundary_state,
            team_mask,
            policy_state,
            parameters,
            match_key,
        ):
            observations = env.observe_managers(rollout, squad, management)
            proposal = selected_manager.step(
                observations,
                match_key,
                policy_state,
                parameters,
            )
            command = proposal.command
            if using_reference_manager:
                empty = ManagerCommand.empty(command.substitutions.requested.shape[1])
                if not manager_default:
                    command = command._replace(
                        substitutions=empty.substitutions,
                        formations=empty.formations,
                        acting_goalkeepers=empty.acting_goalkeepers,
                    )
                if not taker_default:
                    command = command._replace(set_piece_takers=empty.set_piece_takers)
            result = env.manager_command(rollout, squad, management, command)
            boundary = acknowledge_manager_boundary(
                boundary_state,
                rollout.state.restart.opened_control_tick,
                rollout.state.restart.kind,
                team_mask,
            )
            return _GenericManagerDecisionResult(
                rollout=result.rollout,
                management=result.management,
                policy_state=proposal.state,
                boundary_state=boundary,
                roster_metadata_changed=result.roster_metadata_changed,
            )

        manager_initialize = jax.jit(initialize_manager_with_squad)
        manager_decide = jax.jit(decide_manager)

    def refresh_roster(rollout, management, previous_roster, policy_state):
        roster = env.roster_metadata_si(rollout, management)
        return _RosterRefreshResult(
            roster=roster,
            player_policy_state=refresh_policy_state(
                env,
                selected_player,
                rollout,
                previous_roster,
                roster,
                policy_state,
            ),
        )

    def apply_tactics(rollout, management, roster, policy_state):
        return apply_management_tactics(
            env,
            selected_player,
            rollout,
            management,
            roster,
            policy_state,
        )

    opening_decide = None
    if selected_opening is not None:

        def decide_opening(rollout, setup, squad, management, match_key):
            observations = observe_opening_formations(
                rollout.state,
                squad,
                management,
                env.normalization_context(),
            )
            command = selected_opening(observations, match_key)
            result = env.opening_formation_command(
                rollout,
                setup,
                squad,
                management,
                command,
            )
            return _OpeningDecisionResult(
                rollout=result.rollout,
                setup=result.setup,
                management=result.management,
                applied=result.applied,
            )

        opening_decide = jax.jit(decide_opening)

    runner = ManagedRunner(
        env=env,
        player_policy=selected_player,
        chunk_steps=chunk_steps,
        manager_policy=selected_manager,
        opening_policy=selected_opening,
        _advance=advance,
        _manager_initialize=manager_initialize,
        _manager_decide=manager_decide,
        _refresh_roster=jax.jit(refresh_roster),
        _apply_tactics=jax.jit(apply_tactics),
        _opening_decide=opening_decide,
        _can_handle_goalkeeper=(manager_policy is not None) or manager_default,
    )
    return runner


__all__ = [
    "ManagedMatchState",
    "ManagedRunResult",
    "ManagedRunner",
    "make_managed_runner",
]
