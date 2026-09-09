"""Traceable fixed-length rollouts for fixed-shape player policies.

The factories in this module deliberately do not call :func:`jax.jit`.
Their scalar one-match kernels can therefore be composed in either order with
``jit`` and a match-batching transform without hiding a nested compilation.
Create and reuse one kernel for each fixed chunk length; changing the length
creates a different ``lax.scan`` program.

Low-frequency manager transitions belong between chunks.  A scan carry holds
only the physical/rule rollout and the policy's observation-derived memory;
bench, formation-command, and substitution-ledger tensors never enter the
per-control-frame loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.core.action import IntentAction
from footballworld.core.constants import (
    NO_TEAM,
    RESTART_COUNT,
    RK_GK_HOLD,
    RK_NONE,
    TEAM_0,
    TEAM_1,
)
from footballworld.core.randomness import frame_random_key
from footballworld.dynamics.action import trace_action, trace_action_receipt
from footballworld.environment.api import (
    FootballWorld,
    ManagerCommandStepResult,
    ManagerState,
    Rollout,
    SquadSetup,
    StepResult,
    StepWithEventsResult,
)
from footballworld.environment.episode import (
    MatchSetup,
    RuleOutcome,
    _empty_frame_events,
    _status,
)
from footballworld.environment.observation import RosterMetadata
from footballworld.environment.substitution import _management_stoppage_open
from footballworld.policies.manager import ManagerBoundaryState
from footballworld.policies.player import (
    PlayerPolicy,
    PlayerPolicyStateT,
    validate_player_policy,
    validate_player_policy_step,
)
from footballworld.policies.rule_based.manager import (
    ManagerPolicyStep,
    RuleBasedManager,
    RuleManagerState,
)
from footballworld.policies.rule_based.policy import RuleBasedPolicy
from footballworld.policies.rule_based.state import RulePolicyState


class AdvanceResult(NamedTuple):
    """Only final recurrent values from a capture-free training scan."""

    final_rollout: Rollout
    final_policy_state: PlayerPolicyStateT


class InterruptibleAdvanceResult(NamedTuple):
    """Final carry and the reason an interruptible chunk stopped advancing."""

    final_rollout: Rollout
    final_policy_state: PlayerPolicyStateT
    steps_executed: jax.Array
    manager_required: jax.Array
    manager_team_mask: jax.Array


class ManagedAdvanceResult(NamedTuple):
    """Capture-free progress ending at a manager or requested-step boundary."""

    final_rollout: Rollout
    final_policy_state: PlayerPolicyStateT
    steps_executed: jax.Array
    manager_required: jax.Array
    restart_required: jax.Array
    goalkeeper_team_mask: jax.Array
    manager_team_mask: jax.Array
    budget_exhausted: jax.Array
    terminated: jax.Array
    truncated: jax.Array


class ManagementDecisionResult(NamedTuple):
    """A rule-manager decision and its authoritative environment result."""

    step: ManagerCommandStepResult
    policy_state: RuleManagerState


class RolloutResult(NamedTuple):
    """Final recurrent values and a time-major event-free trajectory."""

    final_rollout: Rollout
    final_policy_state: PlayerPolicyStateT
    actions: IntentAction
    steps: StepResult


class EventRolloutResult(NamedTuple):
    """Final recurrent values and a time-major eventful trajectory."""

    final_rollout: Rollout
    final_policy_state: PlayerPolicyStateT
    actions: IntentAction
    steps: StepWithEventsResult


AdvanceKernel = Callable[
    [Rollout, MatchSetup, RosterMetadata, PlayerPolicyStateT, jax.Array],
    AdvanceResult[PlayerPolicyStateT],
]
InterruptibleAdvanceKernel = Callable[
    [Rollout, MatchSetup, RosterMetadata, PlayerPolicyStateT, jax.Array],
    InterruptibleAdvanceResult[PlayerPolicyStateT],
]
ManagedAdvanceKernel = Callable[
    [
        Rollout,
        MatchSetup,
        RosterMetadata,
        PlayerPolicyStateT,
        ManagerBoundaryState,
        jax.Array,
        jax.Array,
    ],
    ManagedAdvanceResult[PlayerPolicyStateT],
]
ManagementDecisionKernel = Callable[
    [Rollout, SquadSetup, ManagerState, RuleManagerState, jax.Array],
    ManagementDecisionResult,
]
RolloutKernel = Callable[
    [Rollout, MatchSetup, RosterMetadata, PlayerPolicyStateT, jax.Array],
    RolloutResult[PlayerPolicyStateT],
]
EventRolloutKernel = Callable[
    [
        Rollout,
        MatchSetup,
        RosterMetadata,
        PlayerPolicyStateT,
        jax.Array,
        jax.Array | None,
    ],
    EventRolloutResult[PlayerPolicyStateT],
]


def _validate_factory_inputs(
    env: FootballWorld,
    policy: PlayerPolicy[PlayerPolicyStateT],
    num_steps: int,
) -> None:
    """Reject invalid static inputs before a trace or backend dispatch."""

    if not isinstance(env, FootballWorld):
        raise TypeError("env must be FootballWorld")
    validate_player_policy(policy)
    if not isinstance(num_steps, int) or isinstance(num_steps, bool):
        raise TypeError("num_steps must be an integer")
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")


def _validate_scalar_inputs(
    rollout: Rollout,
    setup: MatchSetup,
    roster: RosterMetadata,
    policy_state: PlayerPolicyStateT,
    key: jax.Array,
) -> None:
    """Perform shape-only validation in Python, never in the XLA graph."""

    if not isinstance(rollout, Rollout):
        raise TypeError("rollout must be Rollout")
    if not isinstance(setup, MatchSetup):
        raise TypeError("setup must be MatchSetup")
    if not isinstance(roster, RosterMetadata):
        raise TypeError(
            "roster must be SI RosterMetadata; use env.roster_metadata_si()"
        )
    position = rollout.state.players.position
    if position.ndim != 2 or position.shape[-1:] != (2,):
        raise ValueError("rollout must be one unbatched match with [N, 2] positions")
    player_count = position.shape[0]
    if setup.second_half_positions.shape != (player_count, 2):
        raise ValueError("setup positions must have shape [N, 2]")
    if setup.second_half_kickoff_team.shape != ():
        raise ValueError("setup kickoff team must be scalar")

    for field in roster._fields:
        if getattr(roster, field).shape != (player_count,):
            raise ValueError(f"roster.{field} must have shape [N]")

    try:
        key_data = jax.random.key_data(key)
    except (TypeError, ValueError) as exc:
        raise TypeError("key must be one JAX PRNG key") from exc
    if key_data.shape != (2,):
        raise ValueError("key must be one unbatched JAX PRNG key")


def _runtime_step_budget(step_budget: jax.Array, num_steps: int) -> jax.Array:
    """Validate a scalar budget and fail closed for invalid traced values."""

    budget = jnp.asarray(step_budget)
    if budget.shape != ():
        raise ValueError("step_budget must be scalar")
    if not jnp.issubdtype(budget.dtype, jnp.integer):
        raise TypeError("step_budget must have an integer dtype")
    if not isinstance(budget, jax.core.Tracer):
        budget_value = int(budget)
        if not 0 <= budget_value <= num_steps:
            raise ValueError(f"step_budget must lie in [0, {num_steps}]")
    valid = (budget >= 0) & (budget <= num_steps)
    return jnp.where(valid, budget, jnp.int32(0)).astype(jnp.int32)


def initialize_policy_state(
    env: FootballWorld,
    policy: PlayerPolicy[PlayerPolicyStateT],
    rollout: Rollout,
    roster: RosterMetadata,
) -> PlayerPolicyStateT:
    """Initialize player-policy memory from public observations only.

    This helper is itself traceable and does not compile or transfer values to
    the host.  Callers may place it inside their own reset/JIT/batch boundary.
    """

    if not isinstance(env, FootballWorld):
        raise TypeError("env must be FootballWorld")
    validate_player_policy(policy)
    if not isinstance(rollout, Rollout):
        raise TypeError("rollout must be Rollout")
    if not isinstance(roster, RosterMetadata):
        raise TypeError(
            "roster must be SI RosterMetadata; use env.roster_metadata_si()"
        )
    return policy.initialize(env.observe_all_si(rollout), roster)


def refresh_policy_state(
    env: FootballWorld,
    policy: RuleBasedPolicy,
    rollout: Rollout,
    previous_roster: RosterMetadata,
    roster: RosterMetadata,
    policy_state: RulePolicyState,
) -> RulePolicyState:
    """Refresh only identity-scoped policy memory after a roster change.

    previous_roster identifies the players that own policy_state, while roster
    identifies the players in rollout. A slot is refreshed when either its
    managed generation or player id changes. Formation semantics and team-slot
    tables belong to the physical slot, so they are preserved.

    The helper is traceable, performs no host transfer, and is intended for the
    manager-command boundary between fixed-length rollout chunks.
    """

    if not isinstance(env, FootballWorld):
        raise TypeError("env must be FootballWorld")
    if not isinstance(policy, RuleBasedPolicy):
        raise TypeError("policy must be RuleBasedPolicy")
    if not isinstance(rollout, Rollout):
        raise TypeError("rollout must be Rollout")
    if not isinstance(previous_roster, RosterMetadata):
        raise TypeError(
            "previous_roster must be SI RosterMetadata; use env.roster_metadata_si()"
        )
    if not isinstance(roster, RosterMetadata):
        raise TypeError(
            "roster must be SI RosterMetadata; use env.roster_metadata_si()"
        )
    if not isinstance(policy_state, RulePolicyState):
        raise TypeError("policy_state must be RulePolicyState")

    position = rollout.state.players.position
    if position.ndim != 2 or position.shape[-1:] != (2,):
        raise ValueError("rollout must be one unbatched match with [N, 2] positions")
    player_count = position.shape[0]
    for name, metadata in (
        ("previous_roster", previous_roster),
        ("roster", roster),
    ):
        if metadata.player_id.shape != (player_count,):
            raise ValueError(f"{name}.player_id must have shape [N]")
        if metadata.slot_generation.shape != (player_count,):
            raise ValueError(f"{name}.slot_generation must have shape [N]")
        if metadata.is_goalkeeper.shape != (player_count,):
            raise ValueError(f"{name}.is_goalkeeper must have shape [N]")

    if policy_state.formation_anchor.shape != (player_count, 2):
        raise ValueError("policy formation anchors must have shape [N, 2]")
    for field in (
        "role",
        "restart_kind",
        "restart_age",
        "possession_team",
        "possession_age",
        "attack_phase",
        "current_possessor",
        "previous_possessor",
        "counterpress_age",
        "loose_chaser",
        "secure_control_age",
        "last_control_tick",
    ):
        if getattr(policy_state, field).shape != (player_count,):
            raise ValueError(f"policy_state.{field} must have shape [N]")
    if policy_state.team_slot_index.shape != (2, 11):
        raise ValueError("policy team-slot indices must have shape [2, 11]")
    if policy_state.team_slot_valid.shape != (2, 11):
        raise ValueError("policy team-slot validity must have shape [2, 11]")

    fresh = policy.initialize(env.observe_all_si(rollout), roster)
    changed = (previous_roster.slot_generation != roster.slot_generation) | (
        previous_roster.player_id != roster.player_id
    )
    relational_identity_changed = jnp.any(changed)
    goalkeeper_role_changed = previous_roster.is_goalkeeper != roster.is_goalkeeper

    return policy_state._replace(
        role=jnp.where(goalkeeper_role_changed, fresh.role, policy_state.role),
        restart_kind=jnp.where(changed, fresh.restart_kind, policy_state.restart_kind),
        restart_age=jnp.where(changed, fresh.restart_age, policy_state.restart_age),
        possession_team=jnp.where(
            changed, fresh.possession_team, policy_state.possession_team
        ),
        possession_age=jnp.where(
            changed, fresh.possession_age, policy_state.possession_age
        ),
        attack_phase=jnp.where(changed, fresh.attack_phase, policy_state.attack_phase),
        current_possessor=jnp.where(
            changed, fresh.current_possessor, policy_state.current_possessor
        ),
        previous_possessor=jnp.where(
            changed, fresh.previous_possessor, policy_state.previous_possessor
        ),
        counterpress_age=jnp.where(
            changed, fresh.counterpress_age, policy_state.counterpress_age
        ),
        loose_chaser=jnp.where(
            relational_identity_changed,
            fresh.loose_chaser,
            policy_state.loose_chaser,
        ),
        planned_receiver=jnp.where(
            relational_identity_changed,
            fresh.planned_receiver,
            policy_state.planned_receiver,
        ),
        planned_receiver_id=jnp.where(
            relational_identity_changed,
            fresh.planned_receiver_id,
            policy_state.planned_receiver_id,
        ),
        planned_arrival=jnp.where(
            relational_identity_changed,
            fresh.planned_arrival,
            policy_state.planned_arrival,
        ),
        planned_eta_ticks=jnp.where(
            relational_identity_changed,
            fresh.planned_eta_ticks,
            policy_state.planned_eta_ticks,
        ),
        service_opportunity=jnp.where(
            relational_identity_changed,
            fresh.service_opportunity,
            policy_state.service_opportunity,
        ),
        secure_control_age=jnp.where(
            relational_identity_changed,
            fresh.secure_control_age,
            policy_state.secure_control_age,
        ),
        last_control_tick=jnp.where(
            changed, fresh.last_control_tick, policy_state.last_control_tick
        ),
    )


def apply_management_tactics(
    env: FootballWorld,
    policy: RuleBasedPolicy,
    rollout: Rollout,
    management: ManagerState,
    roster: RosterMetadata,
    policy_state: RulePolicyState,
) -> RulePolicyState:
    """Apply both teams' current manager anchors without rebuilding observations.

    This helper is intentionally separate from :func:`refresh_policy_state`.
    Call the latter only after ``roster_metadata_changed``; formation-only and
    taker-only decisions therefore avoid tracing the much larger all-player
    observation refresh into their low-frequency graph.
    """

    if not isinstance(env, FootballWorld):
        raise TypeError("env must be FootballWorld")
    if not isinstance(policy, RuleBasedPolicy):
        raise TypeError("policy must be RuleBasedPolicy")
    if not isinstance(rollout, Rollout):
        raise TypeError("rollout must be Rollout")
    if not isinstance(management, ManagerState):
        raise TypeError("management must be ManagerState")
    if not isinstance(roster, RosterMetadata):
        raise TypeError(
            "roster must be SI RosterMetadata; use env.roster_metadata_si()"
        )
    if not isinstance(policy_state, RulePolicyState):
        raise TypeError("policy_state must be RulePolicyState")

    next_state = policy_state
    player_index = jnp.arange(roster.team_id.shape[0], dtype=jnp.int32)
    for team in (TEAM_0, TEAM_1):
        belongs = roster.team_id == jnp.int32(team)
        observer = jnp.min(jnp.where(belongs, player_index, player_index.shape[0]))
        safe_observer = jnp.minimum(observer, player_index.shape[0] - 1)
        tactics = env.observe_player_tactics_si(
            rollout,
            management,
            safe_observer,
        )
        tactics = tactics._replace(valid=tactics.valid & jnp.any(belongs))
        next_state = policy.apply_tactics(tactics, roster, next_state)
    return next_state


def _transition_key(match_key: jax.Array, rollout: Rollout) -> jax.Array:
    """Derive a chunk-invariant frame key from the absolute control tick."""

    return frame_random_key(match_key, rollout.state.control_tick)


def _terminal_status(
    env: FootballWorld, rollout: Rollout
) -> tuple[jax.Array, jax.Array]:
    """Return the authoritative entry-state terminal flags."""

    fulltime_tick, halftime_tick = env.match.clock_ticks(env.timebase)
    return _status(
        rollout.state,
        fulltime_tick=fulltime_tick,
        minimum_team_players=env.match.minimum_team_players,
        halftime_tick=halftime_tick,
        halftime_enabled=env.match.halftime_enabled,
        maximum_added_time_ticks=env.match.maximum_added_time_ticks(env.timebase),
    )


def _zero_step_output(
    step_shape: StepResult | StepWithEventsResult,
    rollout: Rollout,
    terminated: jax.Array,
    truncated: jax.Array,
    action: IntentAction,
    decimation: int,
):
    """Build a canonical non-executed scan row without running physics."""

    step = jax.tree.map(
        lambda leaf: jnp.zeros(leaf.shape, leaf.dtype),
        step_shape,
    )
    done = terminated | truncated
    updates = {
        "rollout": rollout,
        "outcome": RuleOutcome(
            score_delta=jnp.zeros_like(rollout.state.score),
            restart_opened=jnp.bool_(False),
            restart_kind=jnp.int32(RK_NONE),
            restart_team=jnp.int32(NO_TEAM),
        ),
        "contest_override_valid": jnp.bool_(True),
        "terminated": terminated,
        "truncated": truncated,
        "done": done,
        "terminal_frozen": jnp.bool_(True),
    }
    if hasattr(step, "events"):
        updates.update(
            action_trace=trace_action(action, executed=jnp.bool_(False)),
            action_receipt=trace_action_receipt(action, executed=jnp.bool_(False)),
            events=_empty_frame_events(decimation, rollout.state.ball.position.dtype),
        )
    if hasattr(step, "render_samples"):
        sample_count = step.render_samples.state.control_tick.shape[0]
        updates["render_samples"] = jax.tree.map(
            lambda value: jnp.broadcast_to(value, (sample_count,) + value.shape),
            rollout,
        )
    return step._replace(**updates)


def _manager_requirement(
    rollout: Rollout,
    *,
    fulltime_tick: int,
    minimum_team_players: tuple[int, int],
    stoppage_open: jax.Array | None = None,
) -> jax.Array:
    """Return teams missing a goalkeeper at an authoritative manager boundary."""

    players = rollout.state.players
    has_goalkeeper = jnp.stack(
        [
            jnp.any(
                players.active
                & players.is_goalkeeper
                & (players.team_id == jnp.int32(team))
            )
            for team in (TEAM_0, TEAM_1)
        ]
    )
    if stoppage_open is None:
        stoppage_open = _management_stoppage_open(
            rollout.state,
            rollout.offside,
            fulltime_tick=fulltime_tick,
            minimum_team_players=minimum_team_players,
        )
    return (stoppage_open & (~has_goalkeeper)).astype(jnp.bool_)


class _ManagementBoundary(NamedTuple):
    required: jax.Array
    restart_required: jax.Array
    goalkeeper_team_mask: jax.Array
    team_mask: jax.Array


def _management_boundary(
    rollout: Rollout,
    manager_state: ManagerBoundaryState,
    *,
    fulltime_tick: int,
    minimum_team_players: tuple[int, int],
) -> _ManagementBoundary:
    """Return a fixed-shape reason for pausing before a manager decision."""

    state = rollout.state
    stoppage_open = _management_stoppage_open(
        state,
        rollout.offside,
        fulltime_tick=fulltime_tick,
        minimum_team_players=minimum_team_players,
    )
    restart_active = (
        (state.restart.kind > RK_NONE)
        & (state.restart.kind < RESTART_COUNT)
        & (state.restart.kind != RK_GK_HOLD)
        & (state.restart.opened_control_tick >= 0)
    )
    unseen = (
        manager_state.processed_restart_tick != state.restart.opened_control_tick
    ) | (manager_state.processed_restart_kind != state.restart.kind)
    restart_required = stoppage_open & restart_active & jnp.any(unseen)
    goalkeeper_team_mask = _manager_requirement(
        rollout,
        fulltime_tick=fulltime_tick,
        minimum_team_players=minimum_team_players,
        stoppage_open=stoppage_open,
    )
    team_mask = goalkeeper_team_mask | jnp.full((2,), restart_required, dtype=jnp.bool_)
    return _ManagementBoundary(
        required=jnp.any(team_mask),
        restart_required=restart_required,
        goalkeeper_team_mask=goalkeeper_team_mask,
        team_mask=team_mask,
    )


_ORIGINAL_FOOTBALLWORLD_STEP = FootballWorld.step


def _step_assuming_live(
    env: FootballWorld,
    rollout: Rollout,
    setup: MatchSetup,
    action: IntentAction,
    key: jax.Array,
) -> StepResult:
    """Skip redundant entry status only for the unmodified environment step."""

    step = env.step
    if getattr(step, "__func__", None) is _ORIGINAL_FOOTBALLWORLD_STEP:
        return step(rollout, setup, action, key, _entry_live=True)
    return step(rollout, setup, action, key)


def make_advance(
    env: FootballWorld,
    policy: PlayerPolicy[PlayerPolicyStateT],
    num_steps: int,
) -> AdvanceKernel[PlayerPolicyStateT]:
    """Build a capture-free scalar training rollout.

    The returned callable has the same scalar input contract as
    :func:`make_rollout`, but ``lax.scan`` emits no per-frame values.  Only the
    final rollout and player-policy memory survive, allowing XLA to avoid the
    ``O(T)`` trajectory buffers when a learner or simulator does not need
    them.  This is a separate factory rather than a runtime collection flag,
    so importing or using another rollout mode cannot enlarge this graph.
    """

    _validate_factory_inputs(env, policy, num_steps)

    def advance_kernel(
        initial_rollout: Rollout,
        setup: MatchSetup,
        roster: RosterMetadata,
        initial_policy_state: PlayerPolicyStateT,
        match_key: jax.Array,
    ) -> AdvanceResult:
        _validate_scalar_inputs(
            initial_rollout,
            setup,
            roster,
            initial_policy_state,
            match_key,
        )

        initial_terminated, initial_truncated = _terminal_status(env, initial_rollout)
        initial_done = initial_terminated | initial_truncated

        def scan_step(carry, _):
            current_rollout, policy_state, done = carry

            def advance(_):
                observations = env.observe_all_si(current_rollout)
                policy_step = policy.step(observations, roster, policy_state, match_key)
                validate_player_policy_step(policy_step, policy_state)
                step = _step_assuming_live(
                    env,
                    current_rollout,
                    setup,
                    policy_step.action,
                    _transition_key(match_key, current_rollout),
                )
                return step.rollout, policy_step.state, step.done

            next_carry = jax.lax.cond(done, lambda _: carry, advance, None)
            return next_carry, None

        (final_rollout, final_policy_state, _), _ = jax.lax.scan(
            scan_step,
            (initial_rollout, initial_policy_state, initial_done),
            xs=None,
            length=num_steps,
        )
        return AdvanceResult(
            final_rollout=final_rollout,
            final_policy_state=final_policy_state,
        )

    advance_kernel.num_steps = num_steps  # type: ignore[attr-defined]
    advance_kernel.capture_mode = "none"  # type: ignore[attr-defined]
    return advance_kernel


def make_interruptible_advance(
    env: FootballWorld,
    policy: PlayerPolicy[PlayerPolicyStateT],
    num_steps: int,
) -> InterruptibleAdvanceKernel[PlayerPolicyStateT]:
    """Build a capture-free chunk that pauses for emergency GK management.

    The scan has the same static maximum length and scalar inputs as
    :func:`make_advance`. It advances until a team without an active
    goalkeeper reaches the authoritative environment management stoppage,
    then carries identity values through the remaining cells. Live-ball
    advantage and goalkeeper holds are not management stoppages, and a match
    below its configured minimum player count remains terminal rather than a
    manager request.

    No squad, bench, command, or manager-ledger tensor enters this graph. The
    host may apply an emergency command between chunks, refresh roster and
    policy metadata, then resume with the same match key. Transition keys
    continue to fold in the absolute control tick, so a pause does not alter
    the match random stream.
    """

    _validate_factory_inputs(env, policy, num_steps)
    fulltime_tick, _ = env.match.clock_ticks(env.timebase)
    minimum_team_players = env.match.minimum_team_players

    def interruptible_advance_kernel(
        initial_rollout: Rollout,
        setup: MatchSetup,
        roster: RosterMetadata,
        initial_policy_state: PlayerPolicyStateT,
        match_key: jax.Array,
    ) -> InterruptibleAdvanceResult:
        _validate_scalar_inputs(
            initial_rollout,
            setup,
            roster,
            initial_policy_state,
            match_key,
        )
        initial_manager_team_mask = _manager_requirement(
            initial_rollout,
            fulltime_tick=fulltime_tick,
            minimum_team_players=minimum_team_players,
        )
        initial_terminated, initial_truncated = _terminal_status(env, initial_rollout)
        initial_done = initial_terminated | initial_truncated

        def scan_step(carry, _):
            (
                current_rollout,
                policy_state,
                steps_executed,
                manager_team_mask,
                done,
            ) = carry

            def pause(_):
                return carry

            def advance(_):
                observations = env.observe_all_si(current_rollout)
                policy_step = policy.step(observations, roster, policy_state, match_key)
                validate_player_policy_step(policy_step, policy_state)
                step = _step_assuming_live(
                    env,
                    current_rollout,
                    setup,
                    policy_step.action,
                    _transition_key(match_key, current_rollout),
                )
                next_manager_team_mask = _manager_requirement(
                    step.rollout,
                    fulltime_tick=fulltime_tick,
                    minimum_team_players=minimum_team_players,
                )
                return (
                    step.rollout,
                    policy_step.state,
                    steps_executed + jnp.int32(1),
                    next_manager_team_mask,
                    step.done,
                )

            next_carry = jax.lax.cond(
                jnp.any(manager_team_mask) | done,
                pause,
                advance,
                operand=None,
            )
            return next_carry, None

        (
            (
                final_rollout,
                final_policy_state,
                steps_executed,
                manager_team_mask,
                _,
            ),
            _,
        ) = jax.lax.scan(
            scan_step,
            (
                initial_rollout,
                initial_policy_state,
                jnp.int32(0),
                initial_manager_team_mask,
                initial_done,
            ),
            xs=None,
            length=num_steps,
        )
        return InterruptibleAdvanceResult(
            final_rollout=final_rollout,
            final_policy_state=final_policy_state,
            steps_executed=steps_executed,
            manager_required=jnp.any(manager_team_mask),
            manager_team_mask=manager_team_mask,
        )

    interruptible_advance_kernel.num_steps = num_steps  # type: ignore[attr-defined]
    interruptible_advance_kernel.capture_mode = "interruptible"  # type: ignore[attr-defined]
    interruptible_advance_kernel.manager_aware = True  # type: ignore[attr-defined]
    return interruptible_advance_kernel


def make_managed_advance(
    env: FootballWorld,
    policy: PlayerPolicy[PlayerPolicyStateT],
    num_steps: int,
) -> ManagedAdvanceKernel[PlayerPolicyStateT]:
    """Build a capture-free chunk that pauses at every new manager boundary.

    ``num_steps`` is the static scan maximum. ``step_budget`` is a scalar
    runtime cap in ``[0, num_steps]``; it lets independently paused matches in
    one batch finish the same requested horizon without recompiling shorter
    scans or allowing already-complete rows to overshoot.

    The only manager value entering the scan is the four-int restart identity
    memory. No bench, manager observation, random draw, or command tensor is
    carried through player-control frames. The initial kickoff is returned as
    a zero-step boundary and every later newly opened restart is returned after
    the environment step that created it.
    """

    _validate_factory_inputs(env, policy, num_steps)
    fulltime_tick, _ = env.match.clock_ticks(env.timebase)
    minimum_team_players = env.match.minimum_team_players

    def managed_advance_kernel(
        initial_rollout: Rollout,
        setup: MatchSetup,
        roster: RosterMetadata,
        initial_policy_state: PlayerPolicyStateT,
        manager_state: ManagerBoundaryState,
        step_budget: jax.Array,
        match_key: jax.Array,
    ) -> ManagedAdvanceResult:
        _validate_scalar_inputs(
            initial_rollout,
            setup,
            roster,
            initial_policy_state,
            match_key,
        )
        if not isinstance(manager_state, ManagerBoundaryState):
            raise TypeError("manager_state must be ManagerBoundaryState")
        for field in manager_state._fields:
            if getattr(manager_state, field).shape != (2,):
                raise ValueError(f"manager_state.{field} must have shape [2]")
        safe_budget = _runtime_step_budget(step_budget, num_steps)
        initial_boundary = _management_boundary(
            initial_rollout,
            manager_state,
            fulltime_tick=fulltime_tick,
            minimum_team_players=minimum_team_players,
        )
        initial_terminated, initial_truncated = _terminal_status(env, initial_rollout)

        def continue_running(carry):
            _, _, steps_executed, boundary, terminated, truncated = carry
            return (
                (~boundary.required)
                & (steps_executed < safe_budget)
                & (~terminated)
                & (~truncated)
            )

        def advance(carry):
            current_rollout, policy_state, steps_executed, _, _, _ = carry
            observations = env.observe_all_si(current_rollout)
            policy_step = policy.step(observations, roster, policy_state, match_key)
            validate_player_policy_step(policy_step, policy_state)
            step = _step_assuming_live(
                env,
                current_rollout,
                setup,
                policy_step.action,
                _transition_key(match_key, current_rollout),
            )
            next_boundary = _management_boundary(
                step.rollout,
                manager_state,
                fulltime_tick=fulltime_tick,
                minimum_team_players=minimum_team_players,
            )
            return (
                step.rollout,
                policy_step.state,
                steps_executed + jnp.int32(1),
                next_boundary,
                step.terminated,
                step.truncated,
            )

        (
            final_rollout,
            final_policy_state,
            steps_executed,
            boundary,
            terminated,
            truncated,
        ) = jax.lax.while_loop(
            continue_running,
            advance,
            (
                initial_rollout,
                initial_policy_state,
                jnp.int32(0),
                initial_boundary,
                initial_terminated,
                initial_truncated,
            ),
        )
        return ManagedAdvanceResult(
            final_rollout=final_rollout,
            final_policy_state=final_policy_state,
            steps_executed=steps_executed,
            manager_required=boundary.required,
            restart_required=boundary.restart_required,
            goalkeeper_team_mask=boundary.goalkeeper_team_mask,
            manager_team_mask=boundary.team_mask,
            budget_exhausted=steps_executed >= safe_budget,
            terminated=terminated,
            truncated=truncated,
        )

    managed_advance_kernel.num_steps = num_steps  # type: ignore[attr-defined]
    managed_advance_kernel.capture_mode = "managed-interruptible"  # type: ignore[attr-defined]
    managed_advance_kernel.manager_aware = True  # type: ignore[attr-defined]
    return managed_advance_kernel


def make_management_decision(
    env: FootballWorld,
    manager: RuleBasedManager,
) -> ManagementDecisionKernel:
    """Build the separate low-frequency manager-decision transaction.

    This factory deliberately excludes player-policy refresh. A caller reads
    ``step.roster_metadata_changed`` and invokes :func:`refresh_policy_state`
    only on that rare path, keeping its all-player observation graph out of
    taker-only and formation-only decisions.
    """

    if not isinstance(env, FootballWorld):
        raise TypeError("env must be FootballWorld")
    if not isinstance(manager, RuleBasedManager):
        raise TypeError("manager must be RuleBasedManager")

    def management_decision_kernel(
        rollout: Rollout,
        squad: SquadSetup,
        management: ManagerState,
        policy_state: RuleManagerState,
        match_key: jax.Array,
    ) -> ManagementDecisionResult:
        if not isinstance(rollout, Rollout):
            raise TypeError("rollout must be Rollout")
        if not isinstance(squad, SquadSetup):
            raise TypeError("squad must be SquadSetup")
        if not isinstance(management, ManagerState):
            raise TypeError("management must be ManagerState")
        if not isinstance(policy_state, RuleManagerState):
            raise TypeError("policy_state must be RuleManagerState")
        try:
            key_data = jax.random.key_data(match_key)
        except (TypeError, ValueError) as exc:
            raise TypeError("match_key must be one JAX PRNG key") from exc
        if key_data.shape != (2,):
            raise ValueError("match_key must be one unbatched JAX PRNG key")

        observations = env.observe_managers(rollout, squad, management)
        proposal: ManagerPolicyStep = manager.step(
            observations,
            match_key,
            policy_state,
        )
        result = env.manager_command(
            rollout,
            squad,
            management,
            proposal.command,
        )
        return ManagementDecisionResult(
            step=result,
            policy_state=proposal.state,
        )

    management_decision_kernel.capture_mode = "manager-decision"  # type: ignore[attr-defined]
    management_decision_kernel.manager_aware = True  # type: ignore[attr-defined]
    return management_decision_kernel


def make_rollout(
    env: FootballWorld,
    policy: PlayerPolicy[PlayerPolicyStateT],
    num_steps: int,
) -> RolloutKernel[PlayerPolicyStateT]:
    """Build a pure scalar event-free rollout with a fixed scan length.

    Unlike :func:`make_advance`, this captures every action and complete
    :class:`StepResult`, including the post-step rollout state.  Its memory is
    therefore ``O(T)`` even though it excludes the much larger event tree.

    The returned callable is not jitted.  Its signature is ``(rollout, setup,
    roster, policy_state, match_key)`` and every trajectory leaf is time-major
    ``[T, ...]``.  It may be wrapped by ``jax.jit`` or by a match batching
    transform, which then produces ``[B, T, ...]`` leaves.

    ``match_key`` is immutable.  Every environment transition folds in the
    pre-step absolute control tick, so splitting an otherwise identical match
    into chunks preserves its random stream when the same match key is reused.
    The scan never exits early: FootballWorld's own terminal transition keeps
    completed matches absorbing without a data-dependent loop shape.

    Observations are derived in the body rather than retained in the scan
    carry.  A freshly initialized policy consequently derives the first view
    once for initialization and once for its first action, but the recurrent
    carry and every stored transition avoid the much larger [N, N, ...]
    observation tree.  This bounded one-view cost is preferable for long
    training chunks and large match batches.

    A match batching transform stacks the scalar time-major result outside
    this kernel, giving [B, T, ...] rather than the renderer's transfer
    convention [T, B, ...].  Transpose/select explicitly at that boundary;
    the rollout kernel does not silently reorder axes.
    """

    _validate_factory_inputs(env, policy, num_steps)

    def rollout_kernel(
        initial_rollout: Rollout,
        setup: MatchSetup,
        roster: RosterMetadata,
        initial_policy_state: PlayerPolicyStateT,
        match_key: jax.Array,
    ) -> RolloutResult:
        _validate_scalar_inputs(
            initial_rollout,
            setup,
            roster,
            initial_policy_state,
            match_key,
        )

        initial_terminated, initial_truncated = _terminal_status(env, initial_rollout)

        def scan_step(carry, _):
            current_rollout, policy_state, terminated, truncated = carry
            neutral = IntentAction.neutral(
                current_rollout.state.players.player_id.shape[0]
            )
            step_shape = jax.eval_shape(
                lambda value, match_setup, key: _step_assuming_live(
                    env, value, match_setup, neutral, key
                ),
                current_rollout,
                setup,
                _transition_key(match_key, current_rollout),
            )

            def pause(_):
                step = _zero_step_output(
                    step_shape,
                    current_rollout,
                    terminated,
                    truncated,
                    neutral,
                    env.timebase.decimation,
                )
                return (
                    current_rollout,
                    policy_state,
                    terminated,
                    truncated,
                ), (neutral, step)

            def advance(_):
                observations = env.observe_all_si(current_rollout)
                policy_step = policy.step(observations, roster, policy_state, match_key)
                validate_player_policy_step(policy_step, policy_state)
                step = _step_assuming_live(
                    env,
                    current_rollout,
                    setup,
                    policy_step.action,
                    _transition_key(match_key, current_rollout),
                )
                return (
                    step.rollout,
                    policy_step.state,
                    step.terminated,
                    step.truncated,
                ), (
                    policy_step.action,
                    step,
                )

            return jax.lax.cond(terminated | truncated, pause, advance, None)

        (
            (
                final_rollout,
                final_policy_state,
                _,
                _,
            ),
            (actions, steps),
        ) = jax.lax.scan(
            scan_step,
            (
                initial_rollout,
                initial_policy_state,
                initial_terminated,
                initial_truncated,
            ),
            xs=None,
            length=num_steps,
        )
        return RolloutResult(
            final_rollout=final_rollout,
            final_policy_state=final_policy_state,
            actions=actions,
            steps=steps,
        )

    rollout_kernel.num_steps = num_steps  # type: ignore[attr-defined]
    rollout_kernel.capture_mode = "trajectory"  # type: ignore[attr-defined]
    rollout_kernel.collects_events = False  # type: ignore[attr-defined]
    return rollout_kernel


def make_event_rollout(
    env: FootballWorld,
    policy: PlayerPolicy[PlayerPolicyStateT],
    num_steps: int,
    *,
    render_fps: float | None = None,
) -> EventRolloutKernel[PlayerPolicyStateT]:
    """Build a separate event-retaining scalar evaluation rollout.

    This factory intentionally traces :meth:`FootballWorld.step_with_events`
    rather than selecting event collection with a runtime boolean.  As a
    result, training kernels never acquire the large event-output tree or its
    compilation and memory cost.  The returned function follows the same
    scalar, time-major, externally-jitted contract as :func:`make_rollout`.
    """

    _validate_factory_inputs(env, policy, num_steps)

    def event_rollout_kernel(
        initial_rollout: Rollout,
        setup: MatchSetup,
        roster: RosterMetadata,
        initial_policy_state: PlayerPolicyStateT,
        match_key: jax.Array,
        step_budget: jax.Array | None = None,
    ) -> EventRolloutResult:
        _validate_scalar_inputs(
            initial_rollout,
            setup,
            roster,
            initial_policy_state,
            match_key,
        )
        if step_budget is None:
            safe_budget = jnp.int32(num_steps)
        else:
            safe_budget = _runtime_step_budget(step_budget, num_steps)

        initial_terminated, initial_truncated = _terminal_status(env, initial_rollout)

        def scan_step(carry, _):
            current_rollout, policy_state, executed, terminated, truncated = carry
            neutral = IntentAction.neutral(
                current_rollout.state.players.player_id.shape[0]
            )
            step_shape = jax.eval_shape(
                lambda value, match_setup, key: env._step_with_events_single(
                    value,
                    match_setup,
                    neutral,
                    key,
                    _render_fps=render_fps,
                    _entry_live=True,
                ),
                current_rollout,
                setup,
                _transition_key(match_key, current_rollout),
            )

            def pause(_):
                step = _zero_step_output(
                    step_shape,
                    current_rollout,
                    terminated,
                    truncated,
                    neutral,
                    env.timebase.decimation,
                )
                return (
                    current_rollout,
                    policy_state,
                    executed,
                    terminated,
                    truncated,
                ), (neutral, step)

            def advance(_):
                observations = env.observe_all_si(current_rollout)
                policy_step = policy.step(observations, roster, policy_state, match_key)
                validate_player_policy_step(policy_step, policy_state)
                step = env._step_with_events_single(
                    current_rollout,
                    setup,
                    policy_step.action,
                    _transition_key(match_key, current_rollout),
                    _render_fps=render_fps,
                    _entry_live=True,
                )
                return (
                    step.rollout,
                    policy_step.state,
                    executed + jnp.int32(1),
                    step.terminated,
                    step.truncated,
                ), (
                    policy_step.action,
                    step,
                )

            should_pause = (executed >= safe_budget) | terminated | truncated
            return jax.lax.cond(should_pause, pause, advance, None)

        (
            (final_rollout, final_policy_state, _, _, _),
            (actions, steps),
        ) = jax.lax.scan(
            scan_step,
            (
                initial_rollout,
                initial_policy_state,
                jnp.int32(0),
                initial_terminated,
                initial_truncated,
            ),
            xs=None,
            length=num_steps,
        )
        return EventRolloutResult(
            final_rollout=final_rollout,
            final_policy_state=final_policy_state,
            actions=actions,
            steps=steps,
        )

    event_rollout_kernel.num_steps = num_steps  # type: ignore[attr-defined]
    event_rollout_kernel.capture_mode = "events"  # type: ignore[attr-defined]
    event_rollout_kernel.collects_events = True  # type: ignore[attr-defined]
    event_rollout_kernel.render_fps = render_fps  # type: ignore[attr-defined]
    return event_rollout_kernel


__all__ = [
    "AdvanceKernel",
    "AdvanceResult",
    "EventRolloutKernel",
    "EventRolloutResult",
    "InterruptibleAdvanceKernel",
    "InterruptibleAdvanceResult",
    "ManagedAdvanceKernel",
    "ManagedAdvanceResult",
    "ManagementDecisionKernel",
    "ManagementDecisionResult",
    "RolloutKernel",
    "RolloutResult",
    "apply_management_tactics",
    "initialize_policy_state",
    "make_advance",
    "make_event_rollout",
    "make_interruptible_advance",
    "make_managed_advance",
    "make_management_decision",
    "make_rollout",
    "refresh_policy_state",
]
