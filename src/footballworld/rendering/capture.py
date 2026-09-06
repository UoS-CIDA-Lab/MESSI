"""Fixed-chunk exact-event rollout and bounded-memory replay rendering."""

from __future__ import annotations

import json
import math
import multiprocessing
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from footballworld.core.constants import (
    RESTART_COUNT,
    RK_GK_HOLD,
    RK_KICKOFF,
    RK_NONE,
    RK_PENALTY,
    TEAM_0,
    TEAM_1,
)
from footballworld.core.timebase import _exact_render_grid
from footballworld.environment.api import FootballWorld, Rollout, SquadSetup
from footballworld.environment.episode import MAX_WALL_CONTROL_TICKS, MatchSetup
from footballworld.environment.management import (
    ActingGoalkeeperEvent,
    ManagerCommand,
    SubstitutionEvent,
)
from footballworld.environment.observation import RosterMetadata
from footballworld.managed import ManagedMatchState, ManagedRunner
from footballworld.policies.manager import (
    NO_POLICY_PARAMETERS,
    ManagedManagerState,
    acknowledge_manager_boundary,
)
from footballworld.policies.opening_formation import (
    RULE_OPENING_FORMATION_VERSION,
    RuleBasedOpeningFormationPolicy,
)
from footballworld.policies.rule_based.manager import (
    RULE_MANAGER_VERSION,
    RuleBasedManager,
)
from footballworld.policies.rule_based.policy import (
    RuleBasedPolicy,
    make_rule_based_policy,
)
from footballworld.policies.rule_based.provenance import (
    RULE_POLICY_VERSION,
    policy_config_fingerprint,
)
from footballworld.policies.rule_based.state import RulePolicyState
from footballworld.rendering.defaults import DEFAULT_RENDER_FPS, RenderStyle
from footballworld.rendering.integrity import (
    _source_sha256,
    artifact_receipt,
    replay_provenance,
    stable_json_sha256,
    staged_output_directory,
    validate_authoritative_completion,
    verify_video_decode,
    write_completion_manifest,
)
from footballworld.rendering.receipt import render_settings_receipt
from footballworld.rendering.renderer import (
    _ASYNC_FRAME_BUFFER_COUNT,
    _FOV_FAN_ALPHA,
    _FOV_FAN_SAMPLES,
    RenderResult,
    _carry_adjudication,
    _concat_segments,
    _interaction_overlay_geometry,
    _render_segment,
    _renderer_spawn_environment,
    _visual_frame,
)
from footballworld.rendering.spool import ReplaySidecarSpool
from footballworld.rendering.transfer import HostFrame, prepare_host_frames
from footballworld.rendering.window import ReplayWindow
from footballworld.rollout import (
    _management_boundary,
    _transition_key,
    initialize_policy_state,
    make_event_rollout,
)
from footballworld.rules.restart_timing import forced_release_delay_substeps


def _tree_at(tree: Any, index: int) -> Any:
    return jax.tree.map(lambda value: value[index], tree)


def _event_render_grid(env: FootballWorld, fps: float) -> tuple[float, int]:
    """Validate a bounded physics-backed render grid for exact capture."""

    render_fps, samples_per_control, _ = _exact_render_grid(
        env.timebase,
        fps,
        context="exact-event",
    )
    return render_fps, samples_per_control


def _video_sample_time_s(
    control_tick: int,
    sample_index: int,
    *,
    control_fps: float,
    render_fps: float,
) -> float:
    """Absolute time of one uniform end-of-substep render sample."""

    return (control_tick - 1) / control_fps + (sample_index + 1) / render_fps


def _render_sample_groups(
    steps: Any,
    control_frames: list[HostFrame],
    *,
    control_fps: float,
    render_fps: float,
    slot_generations: Any = None,
) -> list[list[HostFrame]]:
    """Build renderer-only groups from bounded physical substep samples."""

    samples = getattr(steps, "render_samples", None)
    if samples is None:
        raise ValueError("event rollout did not retain renderer physics samples")
    shape = samples.state.players.position.shape
    if len(shape) != 4:
        raise ValueError("render samples must have [T, K, N, 2] positions")
    control_count, samples_per_control = int(shape[0]), int(shape[1])
    if control_count != len(control_frames):
        raise ValueError("render and control sample counts disagree")
    if not np.isfinite(control_fps) or control_fps <= 0.0:
        raise ValueError("control_fps must be finite and positive")
    if not np.isfinite(render_fps) or render_fps <= 0.0:
        raise ValueError("render_fps must be finite and positive")
    flattened = jax.tree.map(
        lambda value: value.reshape(
            (control_count * samples_per_control,) + value.shape[2:]
        ),
        samples,
    )
    flat_generations = None
    if slot_generations is not None:
        flat_generations = np.repeat(
            np.asarray(slot_generations), samples_per_control, axis=0
        )
    physical_frames = prepare_host_frames(
        flattened,
        slot_generations=flat_generations,
    )
    groups: list[list[HostFrame]] = []
    for control_index, control in enumerate(control_frames):
        group: list[HostFrame] = []
        for sample_index in range(samples_per_control):
            endpoint = sample_index == samples_per_control - 1
            physical = physical_frames[
                control_index * samples_per_control + sample_index
            ]
            group.append(
                replace(
                    physical,
                    control_tick=control.control_tick,
                    submitted_action=control.submitted_action,
                    action_trace=control.action_trace,
                    video_time_s=_video_sample_time_s(
                        control.control_tick,
                        sample_index,
                        control_fps=control_fps,
                        render_fps=render_fps,
                    ),
                    frame_events=control.frame_events if endpoint else None,
                    substitution_events=(
                        control.substitution_events if endpoint else None
                    ),
                    acting_goalkeeper_events=(
                        control.acting_goalkeeper_events if endpoint else None
                    ),
                    observation=control.observation if endpoint else None,
                    telemetry=control.telemetry if endpoint else None,
                    pre_management_identity=(
                        control.pre_management_identity if endpoint else None
                    ),
                )
            )
        groups.append(group)
    return groups


def _append_event_budget_records(
    records: list[dict[str, Any]],
    steps: Any,
    local_indices: np.ndarray,
    *,
    chunk_start: int,
    limit: int = 32,
) -> None:
    """Retain a bounded host diagnostic for rare physical-budget exhaustion."""

    remaining = max(limit - len(records), 0)
    if remaining == 0 or local_indices.size == 0:
        return
    indices = np.asarray(local_indices[:remaining], dtype=np.int32)
    (
        ticks,
        ball_position,
        ball_velocity,
        restart_kind,
        contact_occurrences,
        boundary_occurrences,
        woodwork_occurrences,
    ) = jax.device_get(
        (
            steps.rollout.state.control_tick[indices],
            steps.rollout.state.ball.position[indices],
            steps.rollout.state.ball.velocity[indices],
            steps.rollout.state.restart.kind[indices],
            steps.events.contacts.occurred[indices],
            steps.events.boundary.occurred[indices],
            steps.events.woodwork_occurred[indices],
        )
    )
    for row, local_index in enumerate(indices.tolist()):
        records.append(
            {
                "frame_index": int(chunk_start + local_index),
                "control_tick": int(np.asarray(ticks[row])),
                "ball_position_m": np.asarray(
                    ball_position[row], dtype=np.float64
                ).tolist(),
                "ball_velocity_mps": np.asarray(
                    ball_velocity[row], dtype=np.float64
                ).tolist(),
                "restart_kind": int(np.asarray(restart_kind[row])),
                "contact_occurrence_count": int(
                    np.count_nonzero(contact_occurrences[row])
                ),
                "boundary_occurrence_count": int(
                    np.count_nonzero(boundary_occurrences[row])
                ),
                "woodwork_occurrence_count": int(
                    np.count_nonzero(woodwork_occurrences[row])
                ),
            }
        )


def _qualified_name(value: Any) -> str | None:
    if value is None:
        return None
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _policy_identity(value: Any) -> dict[str, Any]:
    record: dict[str, Any] = {"class": _qualified_name(value)}
    if value is None:
        return record
    if isinstance(value, RuleBasedPolicy):
        record["version"] = RULE_POLICY_VERSION
        record["config_sha256"] = policy_config_fingerprint(value.config)
        return record
    if isinstance(value, RuleBasedManager):
        record["version"] = RULE_MANAGER_VERSION
    if isinstance(value, RuleBasedOpeningFormationPolicy):
        record["version"] = RULE_OPENING_FORMATION_VERSION
    config = getattr(value, "config", None)
    if is_dataclass(config) and not isinstance(config, type):
        record["config_sha256"] = stable_json_sha256(asdict(config))
    return record


def _automatic_provenance(
    env: FootballWorld,
    match_key: jax.Array,
    *,
    player_policy: Any,
    manager_policy: Any = None,
    opening_policy: Any = None,
) -> dict[str, Any]:
    return replay_provenance(
        env,
        match_key,
        policies={
            "player": _policy_identity(player_policy),
            "manager": _policy_identity(manager_policy),
            "opening": _policy_identity(opening_policy),
        },
    )


def _source_stability_receipt(provenance: Mapping[str, Any]) -> dict[str, Any]:
    start = provenance.get("source_sha256")
    end = _source_sha256()
    if not isinstance(start, str) or end != start:
        raise RuntimeError(
            "FootballWorld source changed during replay capture; staged output "
            "will not be published"
        )
    return {
        "start_sha256": start,
        "end_sha256": end,
        "stable_during_capture": True,
    }


def _host_control_tick(rollout: Rollout) -> int:
    return int(np.asarray(jax.device_get(rollout.state.control_tick)))


def _terminal_classification(
    env: FootballWorld,
    rollout: Rollout,
    *,
    done: bool,
    maximum_steps: int | None,
    steps_executed: int,
) -> tuple[str, bool]:
    """Classify a final host state without adding leaves to the event graph."""

    if not done:
        reached_budget = maximum_steps is not None and steps_executed >= maximum_steps
        return ("maximum_steps" if reached_budget else "incomplete"), False

    state = jax.device_get(rollout.state)
    active = np.asarray(state.players.active, dtype=np.bool_)
    team_id = np.asarray(state.players.team_id, dtype=np.int32)
    active_per_team = np.asarray(
        [np.count_nonzero(active & (team_id == team)) for team in (TEAM_0, TEAM_1)],
        dtype=np.int32,
    )
    below_minimum = bool(
        np.any(active_per_team < np.asarray(env.match.minimum_team_players))
    )
    kind = int(np.asarray(state.restart.kind))
    team = int(np.asarray(state.restart.team))
    ordinary_restart = RK_NONE < kind < RESTART_COUNT and kind != RK_GK_HOLD
    valid_restart_team = team in (TEAM_0, TEAM_1)
    has_restart_candidate = valid_restart_team and active_per_team[team] > 0
    invalid_restart_team = ordinary_restart and not has_restart_candidate

    control = int(np.asarray(state.control_tick))
    dead = int(np.asarray(state.dead_ball_control_ticks))
    first_half_extension = int(np.asarray(state.first_half_live_extension_ticks))
    regulation_elapsed = max(control - dead - first_half_extension, 0)
    fulltime_tick, _ = env.match.clock_ticks(env.timebase)
    penalty_incomplete = kind == RK_PENALTY or bool(
        np.asarray(state.penalty_completion_active)
    )
    regulation_complete = regulation_elapsed >= fulltime_tick and not penalty_incomplete
    wall_clock_exhausted = control >= MAX_WALL_CONTROL_TICKS

    # Match the environment status families, but prefer fail-closed causes if
    # more than one predicate becomes true on the same terminal frame.
    if below_minimum:
        basis = "below_minimum"
    elif invalid_restart_team:
        basis = "invalid_restart_team"
    elif wall_clock_exhausted:
        basis = "wall_clock_exhausted"
    elif regulation_complete:
        basis = "regulation_complete"
    else:
        basis = "unknown_environment_done"
    return basis, basis == "regulation_complete"


def _completion_record(
    *,
    done: bool,
    terminal_basis: str,
    full_duration_complete: bool,
    maximum_steps: int | None,
    start_control_tick: int,
    final_control_tick: int,
    steps_executed: int,
) -> dict[str, Any]:
    if final_control_tick - start_control_tick != steps_executed:
        raise RuntimeError("capture control ticks do not match executed steps")
    return {
        "done": bool(done),
        "complete": bool(done),
        "termination_reason": "environment_done" if done else "maximum_steps",
        "terminal_basis": terminal_basis,
        "full_duration_complete": bool(full_duration_complete),
        "maximum_steps": maximum_steps,
        "start_control_tick": int(start_control_tick),
        "final_control_tick": int(final_control_tick),
        "steps_executed": int(steps_executed),
    }


def _published_result(
    result: RenderResult, staging: Path, destination: Path
) -> RenderResult:
    def published(path: Path) -> Path:
        return destination / path.relative_to(staging)

    return replace(
        result,
        video=published(result.video),
        event=published(result.event),
        tracking=published(result.tracking),
        metadata=published(result.metadata),
    )


def _manifest_outputs(
    outputs: tuple[RenderResult, ...], root: Path, *, verify_video: bool
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for output in outputs:
        with output.metadata.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        counts = metadata.get("verified_counts")
        if not isinstance(counts, dict):
            raise TypeError("replay metadata has no verified frame counts")
        tracking_storage = metadata.get("tracking_storage")
        if not isinstance(tracking_storage, dict):
            raise TypeError("replay metadata has no tracking storage receipt")
        required = ("source", "tracking", "event", "video")
        if not all(name in counts for name in required):
            raise ValueError("replay metadata has incomplete verified frame counts")
        normalized = {name: int(counts[name]) for name in required}
        if normalized["video"] != output.frames:
            raise ValueError("render result and verified video counts disagree")
        render = metadata.get("render")
        if not isinstance(render, dict):
            raise TypeError("replay metadata has no render receipt")
        video_verification = (
            verify_video_decode(
                output.video,
                expected_frame_count=normalized["video"],
                expected_fps=float(render["video_fps"]),
                expected_width_px=int(render["width_px"]),
                expected_height_px=int(render["height_px"]),
                expected_codec=str(render["codec"]),
                expected_pixel_format=str(render["pixel_format"]),
            )
            if verify_video
            else {"enabled": False, "method": "not_requested"}
        )
        artifacts = {
            "video": artifact_receipt(output.video),
            "event": artifact_receipt(output.event),
            "tracking": artifact_receipt(output.tracking),
            "metadata": artifact_receipt(output.metadata),
        }
        records.append(
            {
                "video": output.video.relative_to(root).as_posix(),
                "event": output.event.relative_to(root).as_posix(),
                "tracking": output.tracking.relative_to(root).as_posix(),
                "metadata": output.metadata.relative_to(root).as_posix(),
                "source_frame_count": normalized["source"],
                "tracking_frame_count": normalized["tracking"],
                "event_frame_count": normalized["event"],
                "video_frame_count": normalized["video"],
                "tracking_storage": tracking_storage,
                "artifacts": artifacts,
                "video_decode_verification": video_verification,
            }
        )
    return records


@dataclass(frozen=True, slots=True)
class _RestartWatchdog:
    """Fail closed on a restart that outlives its environment-owned release gate."""

    ordinary_control_steps: int

    @classmethod
    def from_environment(cls, env: FootballWorld) -> _RestartWatchdog:
        delay = forced_release_delay_substeps(
            timebase=env.timebase,
            config=env.restart_timing,
        )
        controls = (delay + env.timebase.decimation - 1) // env.timebase.decimation
        return cls(ordinary_control_steps=max(1, controls) + 1)

    def check(self, rollout: Rollout, *, done: bool) -> None:
        if done:
            return
        state = rollout.state
        tick, kind, team, taker, opened, remaining, ready = jax.device_get(
            (
                state.control_tick,
                state.restart.kind,
                state.restart.team,
                state.restart.taker,
                state.restart.opened_control_tick,
                state.restart.substeps_remaining,
                state.restart_layout_ready,
            )
        )
        kind = int(np.asarray(kind))
        if kind <= RK_NONE or kind >= RESTART_COUNT:
            return
        tick = int(np.asarray(tick))
        opened = int(np.asarray(opened))
        remaining = int(np.asarray(remaining))
        ready = bool(np.asarray(ready))
        detail = (
            f"tick={tick}, kind={kind}, team={int(np.asarray(team))}, "
            f"taker={int(np.asarray(taker))}, opened_tick={opened}, "
            f"remaining={remaining}, layout_ready={ready}"
        )
        if remaining <= 0 and not ready:
            raise RuntimeError(f"expired restart has no legal layout: {detail}")
        if opened < 0:
            raise RuntimeError(f"active restart has no opening tick: {detail}")
        limit = 1 if kind == RK_KICKOFF else self.ordinary_control_steps
        if tick - opened > limit:
            raise RuntimeError(
                f"restart did not release by its configured gate: {detail}"
            )


@dataclass(frozen=True, slots=True)
class EventMatchRenderResult:
    """Terminal-or-budget replay products from one scalar match."""

    outputs: tuple[RenderResult, ...]
    final_rollout: Rollout
    steps_executed: int
    done: bool
    capture_and_render_seconds: float
    event_chunk_steps: int
    terminal_basis: str = "unknown"
    full_duration_complete: bool = False


@dataclass(frozen=True, slots=True)
class ManagedEventMatchRenderResult:
    """Exact-event replay plus the complete final managed host state."""

    outputs: tuple[RenderResult, ...]
    final_state: ManagedMatchState
    steps_executed: int
    manager_decisions: int
    opening_formation_applied: np.ndarray
    done: bool
    capture_and_render_seconds: float
    event_chunk_steps: int
    terminal_basis: str = "unknown"
    full_duration_complete: bool = False


class _ManagedEventChunkResult(NamedTuple):
    final_rollout: Rollout
    final_policy_state: RulePolicyState
    actions: Any
    steps: Any
    valid: jax.Array
    steps_executed: jax.Array
    manager_required: jax.Array
    manager_team_mask: jax.Array
    goalkeeper_team_mask: jax.Array
    done: jax.Array


class _ExactManagerDecisionResult(NamedTuple):
    rollout: Rollout
    management: Any
    policy_state: Any
    boundary_state: Any
    substitution_events: Any
    acting_goalkeeper_events: Any
    formation_requested: jax.Array
    formation_layout_index: jax.Array
    formations_applied: jax.Array
    roster_metadata_changed: jax.Array


class _ManagedBoundaryResult(NamedTuple):
    state: ManagedMatchState
    substitution_events: Any
    acting_goalkeeper_events: Any
    formation_requested: Any
    formation_layout_index: Any
    formations_applied: Any
    decided: bool


def _make_managed_event_chunk(
    runner: ManagedRunner,
    num_steps: int,
    render_fps: float,
):
    """Build an exact-event scan that pauses at the same manager boundaries."""

    env = runner.env
    policy = runner.player_policy
    fulltime_tick, _ = env.match.clock_ticks(env.timebase)
    minimum_team_players = env.match.minimum_team_players

    def kernel(
        initial_rollout,
        setup,
        roster,
        initial_policy_state,
        boundary_state,
        step_budget,
        match_key,
    ):
        boundary = _management_boundary(
            initial_rollout,
            boundary_state,
            fulltime_tick=fulltime_tick,
            minimum_team_players=minimum_team_players,
        )

        def scan_step(carry, _):
            current, policy_state, executed, current_boundary, terminal = carry
            observations = env.observe_all_si(current)
            policy_step = policy.step(observations, roster, policy_state, match_key)
            candidate = env._step_with_events_single(
                current,
                setup,
                policy_step.action,
                _transition_key(match_key, current),
                _render_fps=render_fps,
            )
            candidate_boundary = _management_boundary(
                candidate.rollout,
                boundary_state,
                fulltime_tick=fulltime_tick,
                minimum_team_players=minimum_team_players,
            )
            paused = current_boundary.required | terminal | (executed >= step_budget)
            valid = ~paused
            next_rollout = jax.tree.map(
                lambda new, old: jnp.where(valid, new, old),
                candidate.rollout,
                current,
            )
            next_policy_state = jax.tree.map(
                lambda new, old: jnp.where(valid, new, old),
                policy_step.state,
                policy_state,
            )
            next_boundary = jax.tree.map(
                lambda new, old: jnp.where(valid, new, old),
                candidate_boundary,
                current_boundary,
            )
            next_terminal = jnp.where(valid, candidate.done, terminal)
            return (
                next_rollout,
                next_policy_state,
                executed + valid.astype(jnp.int32),
                next_boundary,
                next_terminal,
            ), (policy_step.action, candidate, valid)

        (
            (
                final_rollout,
                final_policy_state,
                steps_executed,
                final_boundary,
                done,
            ),
            (actions, steps, valid),
        ) = jax.lax.scan(
            scan_step,
            (
                initial_rollout,
                initial_policy_state,
                jnp.int32(0),
                boundary,
                jnp.bool_(False),
            ),
            xs=None,
            length=num_steps,
        )
        return _ManagedEventChunkResult(
            final_rollout=final_rollout,
            final_policy_state=final_policy_state,
            actions=actions,
            steps=steps,
            valid=valid,
            steps_executed=steps_executed,
            manager_required=final_boundary.required,
            manager_team_mask=final_boundary.team_mask,
            goalkeeper_team_mask=final_boundary.goalkeeper_team_mask,
            done=done,
        )

    return kernel


def _make_exact_manager_decision(
    runner: ManagedRunner,
    *,
    respect_environment_switches: bool,
):
    """Compile the rare manager transaction while retaining its exact events."""

    policy = runner.manager_policy
    if policy is None:
        return None
    env = runner.env
    mask_reference = respect_environment_switches and isinstance(
        policy, RuleBasedManager
    )

    def decide(
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
        proposal = policy.step(
            observations,
            match_key,
            policy_state,
            parameters,
        )
        command = proposal.command
        if mask_reference:
            empty = ManagerCommand.empty(command.substitutions.requested.shape[1])
            if not env.policies.rule_based_match_manager:
                command = command._replace(
                    substitutions=empty.substitutions,
                    formations=empty.formations,
                    acting_goalkeepers=empty.acting_goalkeepers,
                )
            if not env.policies.rule_based_set_piece_taker:
                command = command._replace(set_piece_takers=empty.set_piece_takers)
        step = env.manager_command(rollout, squad, management, command)
        boundary = acknowledge_manager_boundary(
            boundary_state,
            rollout.state.restart.opened_control_tick,
            rollout.state.restart.kind,
            team_mask,
        )
        return _ExactManagerDecisionResult(
            rollout=step.rollout,
            management=step.management,
            policy_state=proposal.state,
            boundary_state=boundary,
            substitution_events=step.substitution_events,
            acting_goalkeeper_events=step.acting_goalkeeper_events,
            formation_requested=command.formations.requested,
            formation_layout_index=command.formations.layout_index,
            formations_applied=step.formations_applied,
            roster_metadata_changed=step.roster_metadata_changed,
        )

    return jax.jit(decide)


def _handle_managed_boundary(
    runner: ManagedRunner,
    decision_kernel: Any,
    state: ManagedMatchState,
    squad: SquadSetup,
    manager_team_mask: jax.Array,
    goalkeeper_team_mask: jax.Array,
    manager_parameters: Any,
    match_key: jax.Array,
) -> _ManagedBoundaryResult:
    goalkeeper_required = bool(np.any(jax.device_get(goalkeeper_team_mask)))
    if goalkeeper_required and not runner._can_handle_goalkeeper:
        raise RuntimeError(
            "management is required for a missing goalkeeper, but the "
            "selected policy configuration cannot handle it"
        )
    if decision_kernel is None:
        boundary = acknowledge_manager_boundary(
            state.manager.boundary,
            state.rollout.state.restart.opened_control_tick,
            state.rollout.state.restart.kind,
            manager_team_mask,
        )
        return _ManagedBoundaryResult(
            state=state._replace(manager=state.manager._replace(boundary=boundary)),
            substitution_events=None,
            acting_goalkeeper_events=None,
            formation_requested=None,
            formation_layout_index=None,
            formations_applied=None,
            decided=False,
        )

    decision = decision_kernel(
        state.rollout,
        squad,
        state.management,
        state.manager.boundary,
        manager_team_mask,
        state.manager.policy,
        manager_parameters,
        match_key,
    )
    previous_roster = state.roster
    state = state._replace(
        rollout=decision.rollout,
        management=decision.management,
        manager=ManagedManagerState(
            boundary=decision.boundary_state,
            policy=decision.policy_state,
        ),
    )
    if bool(np.asarray(jax.device_get(decision.roster_metadata_changed))):
        refreshed = runner._refresh_roster(
            state.rollout,
            state.management,
            previous_roster,
            state.player_policy_state,
        )
        state = state._replace(
            roster=refreshed.roster,
            player_policy_state=refreshed.player_policy_state,
        )
    state = state._replace(
        player_policy_state=runner._apply_tactics(
            state.rollout,
            state.management,
            state.roster,
            state.player_policy_state,
        )
    )
    formation_requested, formation_layout_index, formations_applied = jax.device_get(
        (
            decision.formation_requested,
            decision.formation_layout_index,
            decision.formations_applied,
        )
    )
    return _ManagedBoundaryResult(
        state=state,
        substitution_events=decision.substitution_events,
        acting_goalkeeper_events=decision.acting_goalkeeper_events,
        formation_requested=np.asarray(formation_requested, dtype=bool),
        formation_layout_index=np.asarray(formation_layout_index, dtype=np.int32),
        formations_applied=np.asarray(formations_applied, dtype=bool),
        decided=True,
    )


class _RenderScheduler:
    def __init__(
        self,
        *,
        workers: int,
        style: RenderStyle,
        stadium: Any,
        reach: Any,
        ball_radius_m: float,
        control_fps: float,
        physics_fps: float,
        halftime_seconds: float,
        fulltime_seconds: float,
        halftime_enabled: bool,
        environment_view_limited: bool,
        environment_horizontal_fov_degrees: float,
        gaze_yaw_limit_degrees: float,
        gaze_slew_rate_degrees_s: float,
        horizontal_fov_degrees: float,
        render_fps: float,
    ) -> None:
        self.workers = workers
        self.requested_style = style
        self.style = replace(
            style, encoder_threads=max(1, style.encoder_threads // workers)
        )
        self.stadium = stadium
        self.reach = reach
        self.ball_radius_m = float(ball_radius_m)
        self.overlay = _interaction_overlay_geometry(reach, self.ball_radius_m)
        self.control_fps = control_fps
        self.physics_fps = physics_fps
        self.halftime_seconds = halftime_seconds
        self.fulltime_seconds = fulltime_seconds
        self.halftime_enabled = halftime_enabled
        self.environment_view_limited = environment_view_limited
        self.environment_horizontal_fov_degrees = environment_horizontal_fov_degrees
        self.gaze_yaw_limit_degrees = gaze_yaw_limit_degrees
        self.gaze_slew_rate_degrees_s = gaze_slew_rate_degrees_s
        self.horizontal_fov_degrees = horizontal_fov_degrees
        self.render_fps = render_fps
        self.samples_per_control = round(render_fps / control_fps)
        self.decimation = round(physics_fps / control_fps)
        self.render_substep_indices = tuple(
            max(0, index * self.decimation // self.samples_per_control - 1)
            for index in range(1, self.samples_per_control + 1)
        )
        self.pool: ProcessPoolExecutor | None = None
        if workers > 1:
            self.pool = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
        self.pending: dict[Future[tuple[str, int]], tuple[Path, int]] = {}

    def submit(self, frames: list[Any], path: Path) -> None:
        if self.pool is None:
            result = _render_segment(
                frames,
                str(path),
                self.stadium,
                self.reach,
                self.ball_radius_m,
                self.control_fps,
                self.halftime_seconds,
                self.fulltime_seconds,
                self.halftime_enabled,
                self.horizontal_fov_degrees,
                self.style,
                self.render_fps,
            )
            self._validate_segment(result, path, len(frames))
            return
        # One queued segment per worker is enough to keep the pool saturated.
        # A second full pool-width queue only retains and pickles another set
        # of large frame payloads while providing no additional concurrency.
        while len(self.pending) >= self.workers:
            self._drain_one()
        with _renderer_spawn_environment():
            future = self.pool.submit(
                _render_segment,
                frames,
                str(path),
                self.stadium,
                self.reach,
                self.ball_radius_m,
                self.control_fps,
                self.halftime_seconds,
                self.fulltime_seconds,
                self.halftime_enabled,
                self.horizontal_fov_degrees,
                self.style,
                self.render_fps,
            )
        self.pending[future] = (path, len(frames))

    @staticmethod
    def _validate_segment(
        result: tuple[str, int], expected_path: Path, expected_count: int
    ) -> None:
        actual_path, actual_count = result
        if Path(actual_path) != expected_path or actual_count != expected_count:
            raise RuntimeError("render worker returned inconsistent segment metadata")
        if not expected_path.is_file() or expected_path.stat().st_size <= 0:
            raise RuntimeError(f"render worker produced no segment: {expected_path}")

    def _drain_one(self) -> None:
        completed, _ = wait(set(self.pending), return_when=FIRST_COMPLETED)
        for future in completed:
            expected_path, expected_count = self.pending.pop(future)
            self._validate_segment(future.result(), expected_path, expected_count)

    def finish(self) -> None:
        while self.pending:
            self._drain_one()
        self.shutdown()

    def shutdown(self, *, cancel_futures: bool = False) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=cancel_futures)
            self.pool = None


class _WindowSink:
    def __init__(
        self,
        *,
        window: ReplayWindow,
        root: Path,
        temp: Path,
        multiple: bool,
        index: int,
        every: int,
        chunk_frames: int,
        scheduler: _RenderScheduler,
        control_fps: float,
        match_index: int,
        env: FootballWorld,
        metadata: Any,
        max_outfield_aerial_recovery_substeps: int,
        max_goalkeeper_aerial_recovery_substeps: int,
        ball_radius_m: float,
        event_chunk_steps: int,
        include_all_action_controls: bool,
    ) -> None:
        self.window = window
        self.start, self.end = window.step_bounds(control_fps)
        output_dir = root / window.name if multiple else root
        output_dir.mkdir(parents=True, exist_ok=True)
        self.video = output_dir / "match.mp4"
        self.every = every
        self.chunk_frames = chunk_frames
        self.scheduler = scheduler
        self.control_fps = control_fps
        self.previous_adjudication = None
        self.max_outfield_aerial_recovery_substeps = (
            max_outfield_aerial_recovery_substeps
        )
        self.max_goalkeeper_aerial_recovery_substeps = (
            max_goalkeeper_aerial_recovery_substeps
        )
        self.ball_radius_m = ball_radius_m
        self.event_chunk_steps = event_chunk_steps
        self.adjudication_lookback_steps = math.ceil(
            scheduler.requested_style.adjudication_seconds * control_fps
        )
        self.visual: list[Any] = []
        self.segments: list[Path] = []
        fulltime_tick, halftime_tick = env.match.clock_ticks(env.timebase)
        self.spool = ReplaySidecarSpool(
            temp / f"sidecars-{index:03d}",
            self.video,
            control_fps=control_fps,
            match_index=match_index,
            stadium=env.stadium,
            halftime_seconds=halftime_tick / control_fps,
            fulltime_seconds=fulltime_tick / control_fps,
            halftime_enabled=env.match.halftime_enabled,
            metadata=metadata,
            include_all_action_controls=include_all_action_controls,
        )
        self.segment_root = temp / f"video-{index:03d}"
        self.segment_root.mkdir()
        self.encoded_frames = 0

    def intersects(self, first: int, last: int) -> bool:
        end = self.end if self.end is not None else last
        return first < end and last > self.start

    def visually_intersects(self, first: int, last: int) -> bool:
        end = self.end if self.end is not None else last
        lookback_start = max(0, self.start - self.adjudication_lookback_steps)
        return first < end and last > lookback_start

    def _prepare_visual(self, frame: HostFrame) -> Any:
        visual = _visual_frame(
            frame,
            max_outfield_aerial_recovery_substeps=(
                self.max_outfield_aerial_recovery_substeps
            ),
            max_goalkeeper_aerial_recovery_substeps=(
                self.max_goalkeeper_aerial_recovery_substeps
            ),
            ball_radius_m=self.ball_radius_m,
        )
        visual, self.previous_adjudication = _carry_adjudication(
            visual,
            self.previous_adjudication,
            control_fps=self.control_fps,
            duration_seconds=(self.scheduler.requested_style.adjudication_seconds),
        )
        return visual

    def append(
        self,
        frames: list[HostFrame],
        global_start: int,
        *,
        render_groups: list[list[HostFrame]],
    ) -> None:
        if len(render_groups) != len(frames):
            raise ValueError("render groups must match control frames")
        selected = []
        for local, frame in enumerate(frames):
            position = global_start + local
            if position < self.start:
                if position >= self.start - self.adjudication_lookback_steps:
                    for render_frame in render_groups[local]:
                        self._prepare_visual(render_frame)
                continue
            if self.end is not None and position >= self.end:
                break
            selected.append(frame)
            for render_frame in render_groups[local]:
                self.visual.append(self._prepare_visual(render_frame))
                if len(self.visual) >= self.chunk_frames:
                    self._flush_visual()
        self.spool.append(selected)

    def _flush_visual(self) -> None:
        if not self.visual:
            return
        segment = self.segment_root / f"{len(self.segments):06d}.mp4"
        payload = self.visual
        self.visual = []
        self.segments.append(segment)
        self.encoded_frames += len(payload)
        self.scheduler.submit(payload, segment)

    def finish(
        self, *, elapsed: float, workers: int, completion: dict[str, Any]
    ) -> RenderResult:
        if not self.segments or self.spool.frame_count == 0:
            raise ValueError(f"replay window {self.window.name!r} contains no frames")
        expected_encoded = self.spool.frame_count * self.scheduler.samples_per_control
        if self.encoded_frames != expected_encoded:
            raise RuntimeError(
                "video frame count does not match exact sidecar decimation"
            )
        _concat_segments(self.segments, self.video)
        render_fps = self.scheduler.render_fps
        segment_count = len(self.segments)
        effective_workers = min(workers, segment_count)
        render_metadata = render_settings_receipt(
            style=self.scheduler.requested_style,
            video_fps=render_fps,
            workers_requested=workers,
            workers_effective=effective_workers,
            encoder_threads_per_segment=self.scheduler.style.encoder_threads,
            render_chunk_frame_cap=self.chunk_frames,
            segment_count=segment_count,
            process_start_method="spawn" if workers > 1 else None,
            environment_view_limited=(self.scheduler.environment_view_limited),
            environment_horizontal_fov_degrees=(
                self.scheduler.environment_horizontal_fov_degrees
            ),
            gaze_yaw_limit_degrees=self.scheduler.gaze_yaw_limit_degrees,
            gaze_slew_rate_degrees_s=(self.scheduler.gaze_slew_rate_degrees_s),
            rendered_fov_degrees=self.scheduler.horizontal_fov_degrees,
            fov_fan_inner_m=self.scheduler.overlay.fov_inner_radius_m,
            fov_fan_outer_m=self.scheduler.overlay.fov_outer_radius_m,
            fov_fan_alpha=_FOV_FAN_ALPHA,
            fov_fan_samples=_FOV_FAN_SAMPLES,
            intent_ring_ordinary_radius_m=(
                self.scheduler.overlay.ordinary_ring_radius_m
            ),
            intent_ring_challenge_radius_m=(
                self.scheduler.overlay.challenge_ring_radius_m
            ),
            intent_ring_goalkeeper_control_radius_m=(
                self.scheduler.overlay.goalkeeper_control_ring_radius_m
            ),
            event_chunk_steps=self.event_chunk_steps,
            async_rgba_buffers_per_worker=_ASYNC_FRAME_BUFFER_COUNT,
        )
        render_metadata["physics_sampling"] = {
            "source": "authoritative_post_physics_substep_state",
            "physics_fps": self.scheduler.physics_fps,
            "control_fps": self.scheduler.control_fps,
            "samples_per_control": self.scheduler.samples_per_control,
            "selected_zero_based_substeps": list(self.scheduler.render_substep_indices),
            "control_endpoint_preserved": True,
            "ordinary_training_step_unchanged": True,
        }
        event, tracking, metadata = self.spool.finalize(
            video_sample_frame_count=self.encoded_frames,
            video_sample_fps=render_fps,
            video_frame_count=self.encoded_frames,
            video_fps=render_fps,
            sample_every=self.every,
            render_metadata=render_metadata,
            completion=completion,
        )
        return RenderResult(
            video=self.video,
            event=event,
            tracking=tracking,
            metadata=metadata,
            frames=self.encoded_frames,
            seconds=elapsed,
            throughput_fps=self.encoded_frames / max(elapsed, 1e-9),
            workers=effective_workers,
        )


def render_event_match(
    env: FootballWorld,
    initial_rollout: Rollout,
    setup: MatchSetup,
    match_key: jax.Array,
    output_dir: str | Path,
    *,
    policy: RuleBasedPolicy | None = None,
    roster: RosterMetadata | None = None,
    policy_state: RulePolicyState | None = None,
    windows: Sequence[ReplayWindow] | None = None,
    event_chunk_steps: int = 256,
    maximum_steps: int | None = None,
    fps: float = DEFAULT_RENDER_FPS,
    every: int = 1,
    workers: int = 4,
    chunk_frames: int = 3750,
    style: RenderStyle | None = None,
    metadata: Any = None,
    publication_guard: Callable[[], Mapping[str, Any] | None] | None = None,
    verify_video: bool = False,
    exact_actions: bool = False,
    match_index: int = 0,
) -> EventMatchRenderResult:
    """Roll out one match with exact events and render selected time windows.

    One separately compiled fixed-size step_with_events scan is reused until
    the environment's authoritative done flag. The terminal chunk's absorbing
    suffix is discarded before host serialization. maximum_steps is an
    optional wall-frame safety budget; None means continue to done and
    therefore includes dead-ball added time.

    This API is intentionally scalar and host-orchestrated. It never changes
    the lean FootballWorld.step graph used by training.
    """

    if not isinstance(env, FootballWorld):
        raise TypeError("env must be FootballWorld")
    if not isinstance(initial_rollout, Rollout):
        raise TypeError("initial_rollout must be Rollout")
    if not isinstance(setup, MatchSetup):
        raise TypeError("setup must be MatchSetup")
    for name, value in (
        ("event_chunk_steps", event_chunk_steps),
        ("every", every),
        ("workers", workers),
        ("chunk_frames", chunk_frames),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if maximum_steps is not None and (
        not isinstance(maximum_steps, int)
        or isinstance(maximum_steps, bool)
        or maximum_steps < 1
    ):
        raise ValueError("maximum_steps must be None or a positive integer")
    if match_index != 0:
        raise ValueError("event match capture is scalar; match_index must be zero")
    if style is not None and type(style) is not RenderStyle:
        raise TypeError("style must be RenderStyle or None")
    if publication_guard is not None and not callable(publication_guard):
        raise TypeError("publication_guard must be callable or None")
    if type(verify_video) is not bool:
        raise TypeError("verify_video must be bool")
    if type(exact_actions) is not bool:
        raise TypeError("exact_actions must be bool")

    if every != 1:
        raise ValueError("exact-event capture requires every=1 for an exact time axis")
    render_fps, _ = _event_render_grid(env, fps)
    if policy is None:
        if not env.policies.rule_based_player:
            raise ValueError(
                "rule_based_player is disabled; provide an explicit player policy"
            )
        policy = make_rule_based_policy(env)
    if not isinstance(policy, RuleBasedPolicy):
        raise TypeError("policy must be RuleBasedPolicy")
    roster = env.roster_metadata_si(initial_rollout) if roster is None else roster
    if policy_state is None:
        policy_state = initialize_policy_state(env, policy, initial_rollout, roster)

    selected_windows = (ReplayWindow("match"),) if windows is None else tuple(windows)
    if not selected_windows:
        raise ValueError("windows must contain at least one ReplayWindow")
    if not all(isinstance(window, ReplayWindow) for window in selected_windows):
        raise TypeError("windows must contain only ReplayWindow values")
    names = [window.name for window in selected_windows]
    if len(set(names)) != len(names):
        raise ValueError("replay window names must be unique")

    style = RenderStyle() if style is None else style
    control_fps = float(env.timebase.control_fps)
    fulltime_tick, halftime_tick = env.match.clock_ticks(env.timebase)
    max_outfield_aerial_recovery_substeps = max(
        1,
        round(env.contact_timing.aerial_attempt_recovery_s / env.timebase.dt_phys),
    )
    max_goalkeeper_aerial_recovery_substeps = max(
        1,
        round(env.contact_timing.goalkeeper_dive_recovery_s / env.timebase.dt_phys),
    )
    destination = Path(output_dir)
    provenance = _automatic_provenance(env, match_key, player_policy=policy)
    watchdog = _RestartWatchdog.from_environment(env)
    start_control_tick = _host_control_tick(initial_rollout)
    started = time.perf_counter()
    kernel = jax.jit(
        make_event_rollout(
            env,
            policy,
            event_chunk_steps,
            render_fps=render_fps,
        )
    )
    current = initial_rollout
    current_policy_state = policy_state
    executed = 0
    event_budget_exhausted_count = 0
    event_budget_exhausted_records: list[dict[str, Any]] = []
    done = False
    final_rollout = initial_rollout

    with (
        staged_output_directory(destination) as staging,
        TemporaryDirectory(prefix=".work-", dir=staging) as temp_name,
    ):
        temp = Path(temp_name)
        scheduler = _RenderScheduler(
            workers=workers,
            style=style,
            stadium=env.stadium,
            reach=env.reach,
            ball_radius_m=float(env.ball.radius),
            control_fps=control_fps,
            physics_fps=float(env.timebase.physics_fps),
            halftime_seconds=halftime_tick / control_fps,
            fulltime_seconds=fulltime_tick / control_fps,
            halftime_enabled=env.match.halftime_enabled,
            environment_view_limited=env.perception.limit_by_view_angle,
            environment_horizontal_fov_degrees=(env.perception.horizontal_fov_degrees),
            gaze_yaw_limit_degrees=env.perception.gaze_yaw_limit_degrees,
            gaze_slew_rate_degrees_s=(env.perception.gaze_slew_rate_degrees_s),
            horizontal_fov_degrees=(
                env.perception.horizontal_fov_degrees
                if env.perception.limit_by_view_angle
                else style.gaze_cue_degrees
            ),
            render_fps=render_fps,
        )
        sinks = [
            _WindowSink(
                window=window,
                root=staging,
                temp=temp,
                multiple=len(selected_windows) > 1,
                index=index,
                every=every,
                chunk_frames=chunk_frames,
                scheduler=scheduler,
                control_fps=control_fps,
                match_index=match_index,
                env=env,
                metadata={
                    "capture": {
                        "transition": "step_with_events",
                        "event_chunk_steps": event_chunk_steps,
                        "terminal_suffix": "absorbing_frames_discarded",
                    },
                    "provenance": provenance,
                    "user": metadata,
                },
                max_outfield_aerial_recovery_substeps=(
                    max_outfield_aerial_recovery_substeps
                ),
                max_goalkeeper_aerial_recovery_substeps=(
                    max_goalkeeper_aerial_recovery_substeps
                ),
                ball_radius_m=float(env.ball.radius),
                event_chunk_steps=event_chunk_steps,
                include_all_action_controls=exact_actions,
            )
            for index, window in enumerate(selected_windows)
        ]
        try:
            while not done:
                result = kernel(
                    current,
                    setup,
                    roster,
                    current_policy_state,
                    match_key,
                )
                done_flags = np.asarray(jax.device_get(result.steps.done), dtype=bool)
                valid = event_chunk_steps
                terminal = np.flatnonzero(done_flags)
                terminal_valid = int(terminal[0]) + 1 if terminal.size else None
                if maximum_steps is not None:
                    remaining = maximum_steps - executed
                    if remaining <= 0:
                        break
                    valid = min(valid, remaining)
                if terminal_valid is not None and terminal_valid <= valid:
                    valid = terminal_valid
                    done = True
                budget_flags = np.asarray(
                    jax.device_get(result.steps.event_budget_exhausted[:valid]),
                    dtype=bool,
                )
                event_budget_exhausted_count += int(np.count_nonzero(budget_flags))
                _append_event_budget_records(
                    event_budget_exhausted_records,
                    result.steps,
                    np.flatnonzero(budget_flags),
                    chunk_start=executed,
                )
                chunk_end = executed + valid
                interested = [
                    sink
                    for sink in sinks
                    if sink.visually_intersects(executed, chunk_end)
                ]
                if interested:
                    valid_steps = jax.tree.map(
                        lambda value, count=valid: value[:count], result.steps
                    )
                    valid_actions = jax.tree.map(
                        lambda value, count=valid: value[:count], result.actions
                    )
                    host = prepare_host_frames(
                        valid_steps,
                        submitted_actions=valid_actions,
                    )
                    render_groups = _render_sample_groups(
                        valid_steps,
                        host,
                        control_fps=control_fps,
                        render_fps=render_fps,
                    )
                    for sink in interested:
                        sink.append(host, executed, render_groups=render_groups)
                final_rollout = _tree_at(result.steps.rollout, valid - 1)
                executed = chunk_end
                watchdog.check(final_rollout, done=done)
                if done or (maximum_steps is not None and executed >= maximum_steps):
                    break
                current = result.final_rollout
                current_policy_state = result.final_policy_state
            for sink in sinks:
                sink._flush_visual()
            scheduler.finish()
            elapsed = time.perf_counter() - started
            terminal_basis, full_duration_complete = _terminal_classification(
                env,
                final_rollout,
                done=done,
                maximum_steps=maximum_steps,
                steps_executed=executed,
            )
            completion = _completion_record(
                done=done,
                terminal_basis=terminal_basis,
                full_duration_complete=full_duration_complete,
                maximum_steps=maximum_steps,
                start_control_tick=start_control_tick,
                final_control_tick=_host_control_tick(final_rollout),
                steps_executed=executed,
            )
            completion["event_budget_exhausted_count"] = event_budget_exhausted_count
            completion["event_budget_exhausted_records"] = (
                event_budget_exhausted_records
            )
            source_receipt = _source_stability_receipt(provenance)
            completion["production_source"] = source_receipt
            external_receipt: Mapping[str, Any] | None = None
            if publication_guard is not None:
                external_receipt = publication_guard()
                if external_receipt is not None:
                    if not isinstance(external_receipt, Mapping):
                        raise TypeError(
                            "publication_guard must return a mapping or None"
                        )
                    completion["publication_guard"] = dict(external_receipt)
            validate_authoritative_completion(completion, external_receipt)
            if (
                external_receipt is not None
                and external_receipt.get("authoritative") is True
                and not verify_video
            ):
                raise RuntimeError(
                    "authoritative publication requires verify_video=True"
                )
            staged_outputs = tuple(
                sink.finish(elapsed=elapsed, workers=workers, completion=completion)
                for sink in sinks
            )
            if _source_stability_receipt(provenance) != source_receipt:
                raise RuntimeError(
                    "FootballWorld source receipt changed while finalizing replay"
                )
            if publication_guard is not None:
                final_external_receipt = publication_guard()
                if final_external_receipt != external_receipt:
                    raise RuntimeError(
                        "publication guard receipt changed while finalizing replay"
                    )
            write_completion_manifest(
                staging,
                completion=completion,
                outputs=_manifest_outputs(
                    staged_outputs, staging, verify_video=verify_video
                ),
            )
        finally:
            scheduler.shutdown(cancel_futures=True)

    outputs = tuple(
        _published_result(output, staging, destination) for output in staged_outputs
    )

    return EventMatchRenderResult(
        outputs=outputs,
        final_rollout=final_rollout,
        steps_executed=executed,
        done=done,
        capture_and_render_seconds=time.perf_counter() - started,
        event_chunk_steps=event_chunk_steps,
        terminal_basis=terminal_basis,
        full_duration_complete=full_duration_complete,
    )


def render_managed_event_match(
    runner: ManagedRunner,
    initial_state: ManagedMatchState,
    squad: SquadSetup,
    match_key: jax.Array,
    output_dir: str | Path,
    *,
    manager_parameters: Any = NO_POLICY_PARAMETERS,
    apply_opening_formation: bool = True,
    respect_environment_manager_switches: bool = True,
    windows: Sequence[ReplayWindow] | None = None,
    maximum_steps: int | None = None,
    fps: float = DEFAULT_RENDER_FPS,
    every: int = 1,
    workers: int = 4,
    chunk_frames: int = 3750,
    style: RenderStyle | None = None,
    metadata: Any = None,
    publication_guard: Callable[[], Mapping[str, Any] | None] | None = None,
    verify_video: bool = False,
    exact_actions: bool = False,
    match_index: int = 0,
) -> ManagedEventMatchRenderResult:
    """Render one managed match from fixed exact-event chunks until done.

    The eventful player scan pauses on every manager boundary before its
    three-second restart can elapse. Manager parameters, recurrent memory,
    bench observations, and commands enter only a separate rare executable.
    The ordinary managed runner and training step graphs are unchanged.
    """

    if not isinstance(runner, ManagedRunner):
        raise TypeError("runner must be ManagedRunner")
    if type(initial_state) is not ManagedMatchState:
        raise TypeError("initial_state must be ManagedMatchState")
    if not isinstance(squad, SquadSetup):
        raise TypeError("squad must be SquadSetup")
    for name, value in (
        ("every", every),
        ("workers", workers),
        ("chunk_frames", chunk_frames),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if maximum_steps is not None and (
        not isinstance(maximum_steps, int)
        or isinstance(maximum_steps, bool)
        or maximum_steps < 1
    ):
        raise ValueError("maximum_steps must be None or a positive integer")
    if match_index != 0:
        raise ValueError("managed event capture is scalar; match_index must be zero")
    if type(apply_opening_formation) is not bool:
        raise TypeError("apply_opening_formation must be a bool")
    if type(respect_environment_manager_switches) is not bool:
        raise TypeError("respect_environment_manager_switches must be a bool")
    if publication_guard is not None and not callable(publication_guard):
        raise TypeError("publication_guard must be callable or None")
    if type(verify_video) is not bool:
        raise TypeError("verify_video must be bool")
    if type(exact_actions) is not bool:
        raise TypeError("exact_actions must be bool")
    if style is not None and type(style) is not RenderStyle:
        raise TypeError("style must be RenderStyle or None")

    env = runner.env
    render_fps, _ = _event_render_grid(env, fps)
    event_chunk_steps = runner.chunk_steps
    selected_windows = (ReplayWindow("match"),) if windows is None else tuple(windows)
    if not selected_windows:
        raise ValueError("windows must contain at least one ReplayWindow")
    if not all(isinstance(window, ReplayWindow) for window in selected_windows):
        raise TypeError("windows must contain only ReplayWindow values")
    names = [window.name for window in selected_windows]
    if len(set(names)) != len(names):
        raise ValueError("replay window names must be unique")

    if every != 1:
        raise ValueError(
            "managed exact-event capture requires every=1 for an exact time axis"
        )
    opening = runner.run(
        initial_state,
        squad,
        match_key,
        0,
        manager_parameters=manager_parameters,
        apply_opening_formation=apply_opening_formation,
    )
    current = opening.state
    opening_applied = opening.opening_formation_applied
    style = RenderStyle() if style is None else style
    control_fps = float(env.timebase.control_fps)
    fulltime_tick, halftime_tick = env.match.clock_ticks(env.timebase)
    max_outfield_aerial_recovery_substeps = max(
        1,
        round(env.contact_timing.aerial_attempt_recovery_s / env.timebase.dt_phys),
    )
    max_goalkeeper_aerial_recovery_substeps = max(
        1,
        round(env.contact_timing.goalkeeper_dive_recovery_s / env.timebase.dt_phys),
    )
    destination = Path(output_dir)
    provenance = _automatic_provenance(
        env,
        match_key,
        player_policy=runner.player_policy,
        manager_policy=runner.manager_policy,
        opening_policy=runner.opening_policy,
    )
    watchdog = _RestartWatchdog.from_environment(env)
    start_control_tick = _host_control_tick(current.rollout)
    started = time.perf_counter()
    chunk_kernel = jax.jit(
        _make_managed_event_chunk(runner, event_chunk_steps, render_fps)
    )
    manager_kernel = _make_exact_manager_decision(
        runner,
        respect_environment_switches=respect_environment_manager_switches,
    )
    manager_event_width = int(getattr(runner.manager_policy, "max_simultaneous", 0))
    empty_substitution_events = jax.device_get(
        SubstitutionEvent.empty((event_chunk_steps, 2, manager_event_width))
    )
    empty_acting_goalkeeper_events = jax.device_get(
        ActingGoalkeeperEvent.empty((event_chunk_steps, 2))
    )
    executed = 0
    event_budget_exhausted_count = 0
    event_budget_exhausted_records: list[dict[str, Any]] = []
    decisions = 0
    done = False
    zero_progress_boundaries = 0

    with (
        staged_output_directory(destination) as staging,
        TemporaryDirectory(prefix=".work-", dir=staging) as temp_name,
    ):
        temp = Path(temp_name)
        scheduler = _RenderScheduler(
            workers=workers,
            style=style,
            stadium=env.stadium,
            reach=env.reach,
            ball_radius_m=float(env.ball.radius),
            control_fps=control_fps,
            physics_fps=float(env.timebase.physics_fps),
            halftime_seconds=halftime_tick / control_fps,
            fulltime_seconds=fulltime_tick / control_fps,
            halftime_enabled=env.match.halftime_enabled,
            environment_view_limited=env.perception.limit_by_view_angle,
            environment_horizontal_fov_degrees=(env.perception.horizontal_fov_degrees),
            gaze_yaw_limit_degrees=env.perception.gaze_yaw_limit_degrees,
            gaze_slew_rate_degrees_s=(env.perception.gaze_slew_rate_degrees_s),
            horizontal_fov_degrees=(
                env.perception.horizontal_fov_degrees
                if env.perception.limit_by_view_angle
                else style.gaze_cue_degrees
            ),
            render_fps=render_fps,
        )
        sinks = [
            _WindowSink(
                window=window,
                root=staging,
                temp=temp,
                multiple=len(selected_windows) > 1,
                index=index,
                every=every,
                chunk_frames=chunk_frames,
                scheduler=scheduler,
                control_fps=control_fps,
                match_index=match_index,
                env=env,
                metadata={
                    "capture": {
                        "transition": "managed_step_with_events",
                        "event_chunk_steps": event_chunk_steps,
                        "terminal_suffix": "absorbing_frames_discarded",
                        "manager_boundary": "interrupt_before_restart_progress",
                    },
                    "provenance": provenance,
                    "user": metadata,
                },
                max_outfield_aerial_recovery_substeps=(
                    max_outfield_aerial_recovery_substeps
                ),
                max_goalkeeper_aerial_recovery_substeps=(
                    max_goalkeeper_aerial_recovery_substeps
                ),
                ball_radius_m=float(env.ball.radius),
                event_chunk_steps=event_chunk_steps,
                include_all_action_controls=exact_actions,
            )
            for index, window in enumerate(selected_windows)
        ]
        try:
            while not done:
                if maximum_steps is not None:
                    remaining = maximum_steps - executed
                    if remaining <= 0:
                        break
                    budget = min(event_chunk_steps, remaining)
                else:
                    budget = event_chunk_steps
                chunk_generation = current.management.slot_generation
                result = chunk_kernel(
                    current.rollout,
                    current.setup,
                    current.roster,
                    current.player_policy_state,
                    current.manager.boundary,
                    jnp.int32(budget),
                    match_key,
                )
                progressed, valid_mask = jax.device_get(
                    (result.steps_executed, result.valid)
                )
                progressed = int(np.asarray(progressed))
                valid_mask = np.asarray(valid_mask, dtype=bool)
                if progressed != int(np.count_nonzero(valid_mask)):
                    raise RuntimeError("managed event chunk valid mask is inconsistent")
                if np.any(valid_mask[progressed:]) or not np.all(
                    valid_mask[:progressed]
                ):
                    raise RuntimeError(
                        "managed event chunk valid rows are not a prefix"
                    )
                budget_flags = np.asarray(
                    jax.device_get(result.steps.event_budget_exhausted[:progressed]),
                    dtype=bool,
                )
                event_budget_exhausted_count += int(np.count_nonzero(budget_flags))
                _append_event_budget_records(
                    event_budget_exhausted_records,
                    result.steps,
                    np.flatnonzero(budget_flags),
                    chunk_start=executed,
                )

                chunk_end = executed + progressed
                interested = [
                    sink
                    for sink in sinks
                    if sink.visually_intersects(executed, chunk_end)
                ]
                host: list[HostFrame] = []
                render_groups: list[list[HostFrame]] = []
                if interested and progressed:
                    full_steps, full_actions, host_generation = jax.device_get(
                        (result.steps, result.actions, chunk_generation)
                    )
                    valid_steps = jax.tree.map(
                        lambda value, count=progressed: value[:count],
                        full_steps,
                    )
                    valid_actions = jax.tree.map(
                        lambda value, count=progressed: value[:count],
                        full_actions,
                    )
                    generations = np.broadcast_to(
                        np.asarray(host_generation),
                        (progressed,) + host_generation.shape,
                    )
                    substitution_events = jax.tree.map(
                        lambda value, count=progressed: value[:count],
                        empty_substitution_events,
                    )
                    acting_goalkeeper_events = jax.tree.map(
                        lambda value, count=progressed: value[:count],
                        empty_acting_goalkeeper_events,
                    )
                    host = prepare_host_frames(
                        valid_steps,
                        submitted_actions=valid_actions,
                        slot_generations=generations,
                        substitution_events=substitution_events,
                        acting_goalkeeper_events=acting_goalkeeper_events,
                    )
                    render_groups = _render_sample_groups(
                        valid_steps,
                        host,
                        control_fps=control_fps,
                        render_fps=render_fps,
                        slot_generations=generations,
                    )

                current = current._replace(
                    rollout=result.final_rollout,
                    player_policy_state=result.final_policy_state,
                )
                done = bool(np.asarray(jax.device_get(result.done)))
                manager_required = bool(
                    np.asarray(jax.device_get(result.manager_required))
                )
                if manager_required and not done:
                    boundary = _handle_managed_boundary(
                        runner,
                        manager_kernel,
                        current,
                        squad,
                        result.manager_team_mask,
                        result.goalkeeper_team_mask,
                        manager_parameters,
                        match_key,
                    )
                    current = boundary.state
                    decisions += int(boundary.decided)
                    boundary_tick = int(
                        np.asarray(jax.device_get(current.rollout.state.control_tick))
                    )
                    if progressed == 0 and (
                        boundary.decided
                        or boundary.substitution_events is not None
                        or boundary.acting_goalkeeper_events is not None
                    ):
                        for sink in sinks:
                            if sink.intersects(executed, executed + 1):
                                sink.spool.append_pre_frame_management(
                                    control_tick=boundary_tick,
                                    substitution_events=(boundary.substitution_events),
                                    acting_goalkeeper_events=(
                                        boundary.acting_goalkeeper_events
                                    ),
                                    formation_requested=(boundary.formation_requested),
                                    formation_layout_index=(
                                        boundary.formation_layout_index
                                    ),
                                    formations_applied=boundary.formations_applied,
                                )
                    if host and (
                        boundary.substitution_events is not None
                        or boundary.acting_goalkeeper_events is not None
                    ):
                        post = prepare_host_frames(
                            current.rollout,
                            slot_generations=current.management.slot_generation,
                            substitution_events=boundary.substitution_events,
                            acting_goalkeeper_events=(
                                boundary.acting_goalkeeper_events
                            ),
                        )[0]
                        previous = host[-1]
                        host[-1] = replace(
                            post,
                            submitted_action=previous.submitted_action,
                            action_trace=previous.action_trace,
                            frame_events=previous.frame_events,
                            observation=previous.observation,
                            telemetry=previous.telemetry,
                            pre_management_identity={
                                "player_id": previous.player_id,
                                "slot_generation": previous.slot_generation,
                                "team_id": previous.team_id,
                                "is_goalkeeper": previous.is_goalkeeper,
                                "player_height": previous.player_height,
                            },
                        )
                    if host and boundary.decided:
                        for sink in sinks:
                            if sink.intersects(chunk_end - 1, chunk_end):
                                sink.spool.append_frame_formations(
                                    control_tick=boundary_tick,
                                    requested=boundary.formation_requested,
                                    layout_index=boundary.formation_layout_index,
                                    applied=boundary.formations_applied,
                                )

                    if render_groups:
                        render_groups[-1][-1] = host[-1]

                if host:
                    for sink in interested:
                        sink.append(
                            host,
                            executed,
                            render_groups=render_groups,
                        )
                executed = chunk_end
                watchdog.check(current.rollout, done=done)
                if done or (maximum_steps is not None and executed >= maximum_steps):
                    break
                if progressed == 0:
                    zero_progress_boundaries += 1
                    if zero_progress_boundaries > 2:
                        raise RuntimeError(
                            "managed event capture made no progress across "
                            "repeated boundaries"
                        )
                else:
                    zero_progress_boundaries = 0

            for sink in sinks:
                sink._flush_visual()
            scheduler.finish()
            elapsed = time.perf_counter() - started
            terminal_basis, full_duration_complete = _terminal_classification(
                env,
                current.rollout,
                done=done,
                maximum_steps=maximum_steps,
                steps_executed=executed,
            )
            completion = _completion_record(
                done=done,
                terminal_basis=terminal_basis,
                full_duration_complete=full_duration_complete,
                maximum_steps=maximum_steps,
                start_control_tick=start_control_tick,
                final_control_tick=_host_control_tick(current.rollout),
                steps_executed=executed,
            )
            completion["event_budget_exhausted_count"] = event_budget_exhausted_count
            completion["event_budget_exhausted_records"] = (
                event_budget_exhausted_records
            )
            source_receipt = _source_stability_receipt(provenance)
            completion["production_source"] = source_receipt
            external_receipt: Mapping[str, Any] | None = None
            if publication_guard is not None:
                external_receipt = publication_guard()
                if external_receipt is not None:
                    if not isinstance(external_receipt, Mapping):
                        raise TypeError(
                            "publication_guard must return a mapping or None"
                        )
                    completion["publication_guard"] = dict(external_receipt)
            validate_authoritative_completion(completion, external_receipt)
            if (
                external_receipt is not None
                and external_receipt.get("authoritative") is True
                and not verify_video
            ):
                raise RuntimeError(
                    "authoritative publication requires verify_video=True"
                )
            staged_outputs = tuple(
                sink.finish(elapsed=elapsed, workers=workers, completion=completion)
                for sink in sinks
            )
            if _source_stability_receipt(provenance) != source_receipt:
                raise RuntimeError(
                    "FootballWorld source receipt changed while finalizing replay"
                )
            if publication_guard is not None:
                final_external_receipt = publication_guard()
                if final_external_receipt != external_receipt:
                    raise RuntimeError(
                        "publication guard receipt changed while finalizing replay"
                    )
            write_completion_manifest(
                staging,
                completion=completion,
                outputs=_manifest_outputs(
                    staged_outputs, staging, verify_video=verify_video
                ),
            )
        finally:
            scheduler.shutdown(cancel_futures=True)

    outputs = tuple(
        _published_result(output, staging, destination) for output in staged_outputs
    )
    return ManagedEventMatchRenderResult(
        outputs=outputs,
        final_state=current,
        steps_executed=executed,
        manager_decisions=decisions,
        opening_formation_applied=opening_applied,
        done=done,
        capture_and_render_seconds=time.perf_counter() - started,
        event_chunk_steps=event_chunk_steps,
        terminal_basis=terminal_basis,
        full_duration_complete=full_duration_complete,
    )


__all__ = [
    "EventMatchRenderResult",
    "ManagedEventMatchRenderResult",
    "render_event_match",
    "render_managed_event_match",
]
