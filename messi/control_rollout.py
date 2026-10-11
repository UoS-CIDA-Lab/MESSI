"""고정 horizon environment rollout과 final-state advance."""

from __future__ import annotations

from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from messi.core.collision_config import BodyCollision
from messi.core.constants import NO_PLAYER, NO_TEAM, RK_GK_HOLD, RK_NONE
from messi.core.contest import SAMPLE_CONTEST
from messi.core.foul import DEFAULT_FOUL_LAW
from messi.core.numeric import (
    require_bounded_integer,
    require_float32_field_half_extent,
)
from messi.core.randomness import validate_prng_key
from messi.core.restart import RestartPhysics, restart_active
from messi.core.timebase import DEFAULT_TIMEBASE, Timebase
from messi.core.transition import (
    ControlFrameReceipt,
    _validate_static_restart_pitch_compatibility,
    step_control,
    step_control_replay,
    step_control_with_events,
)
from messi.environment.episode import (
    DEFAULT_MATCH_CONFIG,
    MatchConfig,
    MatchSetup,
    match_status,
    transition_match,
)


class EventReceiptTrajectory(NamedTuple):
    """평가·분석 전용 compact event trajectory."""

    final_state: object
    receipts: ControlFrameReceipt


class ReplayResult(NamedTuple):
    """Trajectory를 저장하지 않는 strict replay 결과."""

    final_state: object
    override_valid: jax.Array


def _require_max_steps(max_steps: int) -> int:
    """Static rollout 길이를 int32 loop/index contract로 검증한다."""

    try:
        return require_bounded_integer("max_steps", max_steps, minimum=1)
    except TypeError as error:
        raise ValueError("max_steps는 양의 정수여야 한다") from error


def _bounded_step_budget(step_budget, *, max_steps: int) -> jax.Array:
    """큰 host integer도 int32 cast 전에 static chunk 범위로 clamp한다."""

    if isinstance(step_budget, Integral):
        step_budget = min(max(int(step_budget), 0), max_steps)
    budget = jnp.asarray(step_budget)
    if budget.shape != ():
        raise ValueError("step_budget은 scalar여야 한다")
    return jnp.clip(budget, 0, max_steps).astype(jnp.int32)


class ManagerBoundaryState(NamedTuple):
    """Caller가 처리했다고 확인한 마지막 restart identity."""

    kind: jax.Array
    team: jax.Array
    taker: jax.Array
    opened_control_tick: jax.Array


class ManagedAdvanceResult(NamedTuple):
    """고정 최대 chunk에서 실제로 진행한 match-aware final carry."""

    final_state: object
    steps_executed: jax.Array
    manager_required: jax.Array
    boundary: ManagerBoundaryState
    budget_exhausted: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    halftime_reset: jax.Array
    termination_reason: jax.Array


def empty_manager_boundary() -> ManagerBoundaryState:
    return ManagerBoundaryState(
        kind=jnp.int32(RK_NONE),
        team=jnp.int32(NO_TEAM),
        taker=jnp.int32(NO_PLAYER),
        opened_control_tick=jnp.int32(-1),
    )


def acknowledge_manager_boundary(state) -> ManagerBoundaryState:
    """현재 restart를 처리 완료한 boundary identity로 기록한다."""

    active = restart_active(state.restart.kind)
    return ManagerBoundaryState(
        kind=jnp.where(active, state.restart.kind, RK_NONE).astype(jnp.int32),
        team=jnp.where(active, state.restart.team, NO_TEAM).astype(jnp.int32),
        taker=jnp.where(active, state.restart.taker, NO_PLAYER).astype(jnp.int32),
        opened_control_tick=jnp.where(
            active, state.restart.opened_control_tick, jnp.int32(-1)
        ).astype(jnp.int32),
    )


def _manager_boundary(state, acknowledged: ManagerBoundaryState):
    current = acknowledge_manager_boundary(state)
    changed = (
        (current.kind != acknowledged.kind)
        | (current.team != acknowledged.team)
        | (current.taker != acknowledged.taker)
        | (current.opened_control_tick != acknowledged.opened_control_tick)
    )
    management_stoppage = restart_active(state.restart.kind) & (
        state.restart.kind != RK_GK_HOLD
    )
    return current, management_stoppage & changed


def make_managed_advance(
    max_steps: int,
    *,
    config: MatchConfig = DEFAULT_MATCH_CONFIG,
    timebase: Timebase = DEFAULT_TIMEBASE,
    **transition_kwargs,
):
    """고정 최대 길이의 resumable match-aware final-state kernel을 만든다.

    ``step_budget``은 runtime scalar이며 ``[0, max_steps]``로 clamp된다. 새 restart가
    열리면 그 step까지 반영한 뒤 반환한다. Caller는 management transition을 적용하고
    :func:`acknowledge_manager_boundary`로 새 identity를 확인한 뒤 같은 ``match_key``로
    재개한다.
    """

    max_steps = _require_max_steps(max_steps)
    if not isinstance(config, MatchConfig):
        raise TypeError("config는 MatchConfig여야 한다")
    if not isinstance(timebase, Timebase):
        raise TypeError("timebase는 Timebase여야 한다")
    transition_kwargs = dict(transition_kwargs)
    transition_kwargs["field_half_extent"] = require_float32_field_half_extent(
        transition_kwargs.get("field_half_extent", (52.5, 34.0))
    )
    _validate_static_restart_pitch_compatibility(
        transition_kwargs["field_half_extent"],
        transition_kwargs.get("restart_physics", RestartPhysics()),
        transition_kwargs.get("body_collision", BodyCollision()),
        transition_kwargs.get("foul_law", DEFAULT_FOUL_LAW),
    )

    def managed_advance(
        initial_state,
        roster,
        actions,
        match_key: jax.Array,
        setup: MatchSetup,
        acknowledged_boundary: ManagerBoundaryState,
        step_budget: jax.Array,
    ) -> ManagedAdvanceResult:
        validate_prng_key(match_key, name="match_key")
        if not isinstance(setup, MatchSetup):
            raise TypeError("setup은 MatchSetup이어야 한다")
        if not isinstance(acknowledged_boundary, ManagerBoundaryState):
            raise TypeError("acknowledged_boundary는 ManagerBoundaryState여야 한다")
        for leaf in jax.tree.leaves(actions):
            if leaf.ndim < 1 or leaf.shape[0] != max_steps:
                raise ValueError("actions의 time축은 max_steps와 같아야 한다")
        safe_budget = _bounded_step_budget(step_budget, max_steps=max_steps)
        initial_status = match_status(
            initial_state, roster, config=config, timebase=timebase
        )
        initial_boundary, initial_manager_required = _manager_boundary(
            initial_state, acknowledged_boundary
        )

        def continue_running(carry):
            _, steps, manager_required, terminated, truncated, _, _ = carry
            return (
                (steps < safe_budget)
                & (~manager_required)
                & (~terminated)
                & (~truncated)
            )

        def advance(carry):
            state, steps, _, _, _, halftime_reset, _ = carry
            action = jax.tree.map(lambda leaf: leaf[steps], actions)
            result = transition_match(
                state,
                roster,
                action,
                match_key,
                setup,
                config=config,
                timebase=timebase,
                **transition_kwargs,
            )
            boundary, manager_required = _manager_boundary(
                result.state, acknowledged_boundary
            )
            return (
                result.state,
                steps + jnp.int32(1),
                manager_required,
                result.terminated,
                result.truncated,
                halftime_reset | result.halftime_reset,
                boundary,
            )

        (
            final_state,
            steps_executed,
            manager_required,
            terminated,
            truncated,
            halftime_reset,
            boundary,
        ) = lax.while_loop(
            continue_running,
            advance,
            (
                initial_state,
                jnp.int32(0),
                initial_manager_required,
                initial_status.terminated,
                initial_status.truncated,
                jnp.bool_(False),
                initial_boundary,
            ),
        )
        return ManagedAdvanceResult(
            final_state=final_state,
            steps_executed=steps_executed,
            manager_required=manager_required,
            boundary=boundary,
            budget_exhausted=steps_executed >= safe_budget,
            terminated=terminated,
            truncated=truncated,
            halftime_reset=halftime_reset,
            termination_reason=match_status(
                final_state, roster, config=config, timebase=timebase
            ).reason,
        )

    managed_advance.max_steps = max_steps
    managed_advance.output_profile = "managed-final-state"
    return managed_advance


def _fully_specified(*overrides) -> jax.Array:
    """Override PyTree에 stochastic sampling sentinel이 없는지 고정 shape로 검사한다."""

    valid = jnp.bool_(True)
    for override in overrides:
        if override is None:
            return jnp.bool_(False)
        for leaf in jax.tree.leaves(override):
            valid &= jnp.all(jnp.asarray(leaf) != SAMPLE_CONTEST)
    return valid


def _validate_rollout_time_axes(actions, **trajectories) -> int:
    """Rollout PyTree의 고정 time축을 Python shape metadata로 검증한다."""

    action_leaves = jax.tree.leaves(actions)
    if not action_leaves or action_leaves[0].ndim < 1:
        raise ValueError("actions는 선두 time축을 가져야 한다")
    horizon = action_leaves[0].shape[0]

    def validate(name, tree):
        for leaf in jax.tree.leaves(tree):
            if leaf.ndim < 1 or leaf.shape[0] != horizon:
                raise ValueError(f"{name}의 선두 time축은 actions와 같아야 한다")

    validate("actions", actions)
    for name, tree in trajectories.items():
        if tree is not None:
            validate(name, tree)
    return horizon


def advance_control(
    initial_state,
    roster,
    actions,
    match_key: jax.Array,
    **kwargs,
):
    """Override와 receipt가 없는 일반 학습용 final-state rollout."""

    _validate_rollout_time_axes(actions)

    def scan_step(state, action):
        return step_control(state, roster, action, match_key, **kwargs), None

    final_state, _ = lax.scan(scan_step, initial_state, actions)
    return final_state


def collect_event_receipts(
    initial_state,
    roster,
    actions,
    match_key: jax.Array,
    **kwargs,
) -> EventReceiptTrajectory:
    """Sampling 경기를 별도 실행해 compact event trajectory를 수집한다."""

    _validate_rollout_time_axes(actions)

    def scan_step(state, action):
        result = step_control_with_events(
            state,
            roster,
            action,
            match_key,
            **kwargs,
        )
        return result.state, result.receipt

    final_state, receipts = lax.scan(scan_step, initial_state, actions)
    return EventReceiptTrajectory(final_state=final_state, receipts=receipts)


def evaluate_control(initial_state, roster, actions, match_key: jax.Array, **kwargs):
    """Event를 요청하지 않는 일반 Evaluation; Training과 같은 graph를 사용한다."""
    return advance_control(initial_state, roster, actions, match_key, **kwargs)


def advance_control_replay(
    initial_state,
    roster,
    actions,
    match_key: jax.Array,
    contest_overrides,
    body_foul_overrides=None,
    **kwargs,
):
    """Override를 적용하고 전체 rollout의 strict validity만 보존한다.

    Body-foul trajectory를 생략하거나 어느 field든 ``SAMPLE_CONTEST``를 포함하면
    sampling 실행은 가능하지만 결과 validity는 반드시 ``False``다. Strict 성공에는 두
    override trajectory의 모든 확률 출력을 명시해야 한다.
    """

    _validate_rollout_time_axes(
        actions,
        contest_overrides=contest_overrides,
        body_foul_overrides=body_foul_overrides,
    )

    fully_specified = _fully_specified(contest_overrides, body_foul_overrides)

    def scan_step(carry, inputs):
        state, override_valid = carry
        if body_foul_overrides is None:
            action, override = inputs
            body_override = None
        else:
            action, override, body_override = inputs
        result = step_control_replay(
            state,
            roster,
            action,
            match_key,
            override,
            body_override,
            **kwargs,
        )
        return (
            result.state,
            override_valid & result.receipt.override_valid,
        ), None

    scan_inputs = (
        (actions, contest_overrides)
        if body_foul_overrides is None
        else (actions, contest_overrides, body_foul_overrides)
    )
    (final_state, override_valid), _ = lax.scan(
        scan_step,
        (initial_state, fully_specified),
        scan_inputs,
    )
    return ReplayResult(final_state=final_state, override_valid=override_valid)


def evaluate_control_replay(
    initial_state,
    roster,
    actions,
    match_key: jax.Array,
    contest_overrides,
    body_foul_overrides=None,
    **kwargs,
):
    """Event trajectory 없이 strict validity scalar만 유지하는 Replay Evaluation."""
    return advance_control_replay(
        initial_state,
        roster,
        actions,
        match_key,
        contest_overrides,
        body_foul_overrides,
        **kwargs,
    )


def collect_replay_event_receipts(
    initial_state,
    roster,
    actions,
    match_key: jax.Array,
    contest_overrides,
    body_foul_overrides=None,
    **kwargs,
) -> EventReceiptTrajectory:
    """실제 경기 override를 적용하며 compact event trajectory를 수집한다."""

    _validate_rollout_time_axes(
        actions,
        contest_overrides=contest_overrides,
        body_foul_overrides=body_foul_overrides,
    )

    def scan_step(state, inputs):
        if body_foul_overrides is None:
            action, override = inputs
            body_override = None
        else:
            action, override, body_override = inputs
        result = step_control_replay(
            state,
            roster,
            action,
            match_key,
            override,
            body_override,
            **kwargs,
        )
        return result.state, result.receipt

    scan_inputs = (
        (actions, contest_overrides)
        if body_foul_overrides is None
        else (actions, contest_overrides, body_foul_overrides)
    )
    final_state, receipts = lax.scan(
        scan_step,
        initial_state,
        scan_inputs,
    )
    return EventReceiptTrajectory(final_state=final_state, receipts=receipts)


def _validate_batch_inputs(initial_states, match_keys: jax.Array, **trees) -> None:
    """Public batch 경계에서 모든 PyTree의 선두 match축을 trace-time 검증한다."""

    control_tick = initial_states.control_tick
    if control_tick.ndim != 1:
        raise ValueError("initial_states는 선두 match batch축을 가져야 한다")
    batch_size = control_tick.shape[0]

    def validate_tree(name, tree):
        for leaf in jax.tree.leaves(tree):
            if leaf.ndim < 1 or leaf.shape[0] != batch_size:
                raise ValueError(f"{name}의 선두 batch축은 initial_states와 같아야 한다")

    validate_tree("initial_states", initial_states)
    validate_tree("match_keys", match_keys)
    for name, tree in trees.items():
        if tree is not None:
            validate_tree(name, tree)


def _map_independent_matches(function, *trees, device_mapper=None):
    """경기 batch를 단일 device에서는 `vmap`, 다중 device에서는 `pmap`한다."""

    batch_size = trees[0].control_tick.shape[0]
    local_devices = jax.local_devices()
    device_count = len(local_devices)
    if device_count <= 1 or batch_size == 0:
        return jax.vmap(function)(*trees)

    # Pad the independent-match axis so every local device participates even when
    # the caller's batch is not evenly divisible. Padded results are discarded.
    local_batch = (batch_size + device_count - 1) // device_count
    padded_batch = local_batch * device_count

    def pad_batch(tree):
        padding = padded_batch - batch_size
        if padding == 0:
            return tree
        return jax.tree.map(
            lambda leaf: jnp.concatenate(
                (
                    leaf,
                    jnp.broadcast_to(
                        leaf[-1:], (padding, *leaf.shape[1:])
                    ),
                ),
                axis=0,
            ),
            tree,
        )

    def add_device_axis(tree):
        return jax.tree.map(
            lambda leaf: leaf.reshape((device_count, local_batch, *leaf.shape[1:])),
            pad_batch(tree),
        )

    sharded_trees = tuple(add_device_axis(tree) for tree in trees)
    mapper = device_mapper
    if mapper is None:
        mapper = jax.pmap(lambda *local_trees: jax.vmap(function)(*local_trees))
    mapped = mapper(*sharded_trees)
    return jax.tree.map(
        lambda leaf: leaf.reshape((padded_batch, *leaf.shape[2:]))[:batch_size], mapped
    )


def advance_control_batch(
    initial_states,
    rosters,
    actions,
    match_keys: jax.Array,
    **kwargs,
):
    """Training 경기를 단일 device에서는 `vmap`, 다중 device에서는 `pmap`한다."""

    _validate_batch_inputs(
        initial_states,
        match_keys,
        rosters=rosters,
        actions=actions,
    )
    return _map_independent_matches(
        lambda state, roster, match_actions, match_key: advance_control(
            state,
            roster,
            match_actions,
            match_key,
            **kwargs,
        ),
        initial_states,
        rosters,
        actions,
        match_keys,
    )


def advance_control_replay_batch(
    initial_states,
    rosters,
    actions,
    match_keys: jax.Array,
    contest_overrides,
    body_foul_overrides=None,
    **kwargs,
):
    """Replay 경기를 단일 device에서는 `vmap`, 다중 device에서는 `pmap`한다."""

    _validate_batch_inputs(
        initial_states,
        match_keys,
        rosters=rosters,
        actions=actions,
        contest_overrides=contest_overrides,
        body_foul_overrides=body_foul_overrides,
    )
    if body_foul_overrides is None:
        return _map_independent_matches(
            lambda state, roster, match_actions, match_key, overrides: (
                advance_control_replay(
                    state,
                    roster,
                    match_actions,
                    match_key,
                    overrides,
                    **kwargs,
                )
            ),
            initial_states,
            rosters,
            actions,
            match_keys,
            contest_overrides,
        )
    return _map_independent_matches(
        lambda state, roster, match_actions, match_key, overrides, body_overrides: (
            advance_control_replay(
                state,
                roster,
                match_actions,
                match_key,
                overrides,
                body_overrides,
                **kwargs,
            )
        ),
        initial_states,
        rosters,
        actions,
        match_keys,
        contest_overrides,
        body_foul_overrides,
    )


__all__ = [
    "EventReceiptTrajectory",
    "ManagedAdvanceResult",
    "ManagerBoundaryState",
    "ReplayResult",
    "acknowledge_manager_boundary",
    "advance_control",
    "advance_control_batch",
    "advance_control_replay",
    "advance_control_replay_batch",
    "collect_event_receipts",
    "collect_replay_event_receipts",
    "empty_manager_boundary",
    "evaluate_control",
    "evaluate_control_replay",
    "make_managed_advance",
]
