"""전체 경기 trajectory를 보존하지 않는 chunked 기록·보고서 집계."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from messi.analysis.causal import CausalEventResolver
from messi.analysis.events import EVENT_NAMES, EventStream, extract_events
from messi.analysis.goal_attribution import extract_goal_last_contacts
from messi.analysis.identity import PlayerGenerationTracker, build_player_action_credit
from messi.analysis.invariants import MatchInvariantAuditor, MatchInvariantReport
from messi.analysis.report import (
    DFL_ATTACKING_THIRD_REFERENCE,
    MATCH_REPORT_SCHEMA,
    MatchReport,
    _challenge_contact_counts,
    _has_policy_decision_telemetry,
    build_tactical_summary,
    initial_roster_summary,
    roster_player_profiles,
)
from messi.analysis.spatial import (
    PlayerSpatialAccumulator,
    SpatialAccumulator,
    TeamShapeAccumulator,
)
from messi.analysis.timeline import PossessionTimelineAccumulator
from messi.analysis.release import classify_applied_releases
from messi.control_rollout import (
    _bounded_step_budget,
    _map_independent_matches,
    _require_max_steps,
)
from messi.core.action import (
    ACTION_INTENT_COUNT,
    INTENT_CONTROL,
    INTENT_PASS,
)
from messi.core.collision_config import BodyCollision
from messi.core.foul import DEFAULT_FOUL_LAW
from messi.core.numeric import require_float32_field_half_extent
from messi.core.randomness import frame_random_key, validate_prng_key
from messi.core.restart import RestartPhysics
from messi.core.timebase import DEFAULT_TIMEBASE, Timebase
from messi.core.transition import _validate_static_restart_pitch_compatibility
from messi.environment.episode import (
    DEFAULT_MATCH_CONFIG,
    MatchConfig,
    MatchSetup,
    match_status,
    transition_match_with_events,
)


class MatchRecordingChunk(NamedTuple):
    """한정된 device trajectory와 다음 chunk의 recurrent carry."""

    final_state: object
    states: object
    actions: object
    policy_telemetry: object
    receipts: object
    valid: jax.Array
    steps_executed: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    termination_reason: jax.Array


@dataclass(frozen=True, slots=True)
class MatchRecording:
    """전체 trajectory 없이 완성한 경기 기록."""

    final_state: object
    events: EventStream
    report: MatchReport
    chunks: int
    termination_reason: int
    invariants: MatchInvariantReport


@dataclass(frozen=True, slots=True)
class MatchRecordingBatch:
    """독립 경기 batch의 bounded-memory 기록 결과."""

    matches: tuple[MatchRecording, ...]
    chunks: int


@dataclass(frozen=True, slots=True)
class StoredMatchChunk:
    """GPU에서 회수한 연속 경기 구간과 전체 경기 frame offset."""

    match_index: int
    frame_offset: int
    chunk: MatchRecordingChunk


class InMemoryMatchStore:
    """전체 host chunk를 보존하는 기본 sink; GPU tensor를 유지하지 않는다."""

    def __init__(self) -> None:
        self.chunks: list[StoredMatchChunk] = []

    def __call__(
        self, match_index: int, frame_offset: int, chunk: MatchRecordingChunk
    ) -> None:
        self.chunks.append(StoredMatchChunk(match_index, frame_offset, chunk))


def make_match_recording_chunk(
    max_steps: int,
    policy: Callable,
    *,
    match_config: MatchConfig = DEFAULT_MATCH_CONFIG,
    timebase: Timebase = DEFAULT_TIMEBASE,
    policy_returns_telemetry: bool = False,
    policy_takes_key: bool = False,
    policy_takes_context: bool = False,
    policy_context=None,
    **transition_kwargs,
):
    """고정 최대 길이 capture kernel을 만들고 선택적으로 policy scalar를 보존한다.

    `policy_takes_key=True`면 frame마다 match key와 절대 control tick에서 만든
    frame-local PRNG key를 policy에 전달한다. 정책이 stochastic 판단을 쓸 때만 필요하며
    key를 상태로 보존하지 않는다.
    """
    max_steps = _require_max_steps(max_steps)
    if not callable(policy):
        raise TypeError("policy는 호출 가능해야 한다")
    if not isinstance(policy_takes_key, bool):
        raise TypeError("policy_takes_key는 bool이어야 한다")
    if not isinstance(policy_takes_context, bool):
        raise TypeError("policy_takes_context는 bool이어야 한다")
    if policy_takes_context and policy_context is None:
        raise ValueError("policy_takes_context=True이면 기본 policy_context가 필요하다")
    if not isinstance(policy_returns_telemetry, bool):
        raise TypeError("policy_returns_telemetry는 bool이어야 한다")
    transition_kwargs = dict(transition_kwargs)
    field_half_extent = require_float32_field_half_extent(
        transition_kwargs.get("field_half_extent", (52.5, 34.0))
    )
    transition_kwargs["field_half_extent"] = field_half_extent
    _validate_static_restart_pitch_compatibility(
        field_half_extent,
        transition_kwargs.get("restart_physics", RestartPhysics()),
        transition_kwargs.get("body_collision", BodyCollision()),
        transition_kwargs.get("foul_law", DEFAULT_FOUL_LAW),
    )

    def capture_chunk(
        initial_state,
        roster,
        match_key,
        setup: MatchSetup,
        step_budget,
        runtime_policy_context=None,
    ) -> MatchRecordingChunk:
        validate_prng_key(match_key, name="match_key")
        if not isinstance(setup, MatchSetup):
            raise TypeError("setup은 MatchSetup이어야 한다")
        safe_budget = _bounded_step_budget(step_budget, max_steps=max_steps)
        initial_status = match_status(
            initial_state, roster, config=match_config, timebase=timebase
        )
        initial_done = initial_status.terminated | initial_status.truncated
        active_policy_context = (
            policy_context if runtime_policy_context is None else runtime_policy_context
        )

        def step(carry, index):
            state, done = carry
            active = (index < safe_budget) & (~done)
            if policy_takes_context:
                policy_output = (
                    policy(
                        state,
                        roster,
                        active_policy_context,
                        frame_random_key(match_key, state.control_tick),
                    )
                    if policy_takes_key
                    else policy(state, roster, active_policy_context)
                )
            else:
                policy_output = (
                    policy(
                        state,
                        roster,
                        frame_random_key(match_key, state.control_tick),
                    )
                    if policy_takes_key
                    else policy(state, roster)
                )
            if policy_returns_telemetry:
                action, policy_telemetry = policy_output
            else:
                action = policy_output
                policy_telemetry = jnp.int8(0)
            transition, control = transition_match_with_events(
                state,
                roster,
                action,
                match_key,
                setup,
                already_done=~active,
                config=match_config,
                timebase=timebase,
                **transition_kwargs,
            )
            next_done = done | (active & transition.done)
            return (transition.state, next_done), (
                transition.state,
                action,
                policy_telemetry,
                control.receipt,
                active,
            )

        (final_state, _), (states, actions, policy_telemetry, receipts, valid) = (
            lax.scan(
                step,
                (initial_state, initial_done),
                jnp.arange(max_steps, dtype=jnp.int32),
            )
        )
        status = match_status(
            final_state, roster, config=match_config, timebase=timebase
        )
        return MatchRecordingChunk(
            final_state=final_state,
            states=states,
            actions=actions,
            policy_telemetry=policy_telemetry,
            receipts=receipts,
            valid=valid,
            steps_executed=jnp.sum(valid, dtype=jnp.int32),
            terminated=status.terminated,
            truncated=status.truncated,
            termination_reason=status.reason,
        )

    capture_chunk.max_steps = max_steps
    capture_chunk.output_profile = "bounded-match-recording"
    capture_chunk.match_config = match_config
    capture_chunk.timebase = timebase
    capture_chunk.field_half_extent = field_half_extent
    capture_chunk.policy_takes_context = policy_takes_context
    return capture_chunk


class MatchReportAccumulator:
    """Chunk output을 sparse event와 O(1) report 통계로 축약한다."""

    def __init__(
        self,
        roster,
        *,
        timebase: Timebase = DEFAULT_TIMEBASE,
        field_half_extent: tuple[float, float] = (52.5, 34.0),
    ) -> None:
        team, player_id, is_goalkeeper = jax.device_get(
            (roster.team_id, roster.player_id, roster.is_goalkeeper)
        )
        self._team = np.asarray(team, dtype=np.int32)
        self._goalkeeper = np.asarray(is_goalkeeper, dtype=bool)
        self._player_id = np.asarray(player_id, dtype=np.int32).copy()
        host_roster = roster._replace(
            team_id=team,
            player_id=player_id,
            is_goalkeeper=is_goalkeeper,
        )
        self._player_generation = np.zeros(self._team.shape, dtype=np.int32)
        self._roster_summary = initial_roster_summary(host_roster)
        self._player_generation_distance: dict[tuple[int, int], float] = {
            (slot, 0): 0.0 for slot in range(self._team.size)
        }
        self._control_dt_s = float(timebase.control_dt)
        extent = np.asarray(field_half_extent, dtype=np.float64)
        if extent.shape != (2,) or not np.all(np.isfinite(extent)):
            raise ValueError("field_half_extent는 유한한 (2,) 값이어야 한다")
        if np.any(extent <= 0.0):
            raise ValueError("field_half_extent는 양수여야 한다")
        self._goal_line_x = float(extent[0])
        self._pass_window_frames = timebase.control_steps_for(15.0 * 60.0, minimum=1)
        self._frames = 0
        self._goal_last_contacts = []
        self._event_parts: list[EventStream] = []
        self._intent = np.zeros(ACTION_INTENT_COUNT, dtype=np.int64)
        self._team_intent = np.zeros((2, ACTION_INTENT_COUNT), dtype=np.int64)
        self._possession = np.zeros(2, dtype=np.int64)
        self._team_distance = np.zeros(2, dtype=np.float64)
        self._player_distance = np.zeros(self._team.size, dtype=np.float64)
        self._ball_distance = 0.0
        self._ball_speed_sum = 0.0
        self._ball_speed_count = 0
        self._ball_speed_max = 0.0
        self._live_ball_frames = 0
        self._challenge_contact_attempts = 0
        self._challenge_fouls = 0
        self._previous_player_position = None
        self._previous_player_active = None
        self._previous_player_generation = None
        self._previous_ball_position = None
        self._previous_restart_layout_ready = None
        self._previous_first_half_wall_end_tick = None
        self._causal = CausalEventResolver(
            self._team, is_goalkeeper=self._goalkeeper, timebase=timebase
        )
        self._invariants = MatchInvariantAuditor(host_roster)
        self._spatial = SpatialAccumulator(
            self._team,
            control_dt=timebase.control_dt,
            field_half_extent=field_half_extent,
        )
        self._player_spatial = PlayerSpatialAccumulator(
            self._team, control_dt=timebase.control_dt
        )
        self._team_shape = TeamShapeAccumulator(self._team)
        self._identity = PlayerGenerationTracker(host_roster)
        self._possession_timeline = PossessionTimelineAccumulator(
            control_dt=timebase.control_dt
        )
        self._policy_overload = np.zeros(2, dtype=np.int64)
        self._policy_intent = np.zeros((2, ACTION_INTENT_COUNT), dtype=np.int64)
        self._policy_direction = np.zeros((2, 4), dtype=np.int64)
        self._policy_completion_sum = np.zeros(2, dtype=np.float64)
        self._policy_completion_observed = np.zeros(2, dtype=np.int64)
        self._policy_completion_head_unavailable = np.zeros(2, dtype=np.int64)
        self._policy_gap_sum = np.zeros(2, dtype=np.float64)
        self._policy_next_outlet_sum = np.zeros(2, dtype=np.float64)
        self._policy_control_retained = np.zeros(2, dtype=np.int64)
        self._policy_overload_pass_team: dict[int, int] = {}

    def update_roster(self, roster) -> None:
        """다음 chunk 전에 fixed slot의 새 identity generation을 연다."""
        player_id, team, is_goalkeeper = jax.device_get(
            (roster.player_id, roster.team_id, roster.is_goalkeeper)
        )
        player_id = np.asarray(player_id, dtype=np.int32)
        team = np.asarray(team, dtype=np.int32)
        if player_id.shape != self._player_id.shape or team.shape != self._team.shape:
            raise ValueError("roster player축은 recording 중 바뀔 수 없다")
        if not np.array_equal(team, self._team):
            raise ValueError("roster team_id는 recording 중 바뀔 수 없다")
        changed = player_id != self._player_id
        host_roster = roster._replace(
            team_id=team,
            player_id=player_id,
            is_goalkeeper=is_goalkeeper,
        )
        self._identity.update(host_roster, frame=self._frames)
        self._invariants.update_roster(host_roster)
        self._player_generation[changed] += 1
        self._player_id = player_id.copy()
        if np.any(changed):
            self._roster_summary["players"] += roster_player_profiles(
                host_roster,
                self._player_generation,
                self._frames,
                np.flatnonzero(changed),
            )
        for slot in np.flatnonzero(changed):
            self._player_generation_distance[
                (int(slot), int(self._player_generation[slot]))
            ] = 0.0

    def seed_state(self, state) -> None:
        """첫 post-step chunk 전의 초기 위치를 host distance carry에 넣는다."""

        position, active, ball_position, layout_ready, half_end = jax.device_get(
            (
                state.players.position,
                state.players.active,
                state.ball.position,
                state.restart_layout_ready,
                state.first_half_wall_end_tick,
            )
        )
        self._seed_positions(position, active, ball_position, layout_ready, half_end)

    def _seed_positions(
        self, position, active, ball_position, layout_ready=False, half_end=-1
    ) -> None:
        """Device에서 한 번 회수한 batch position view를 seed한다."""

        if self._frames != 0 or self._previous_player_position is not None:
            raise RuntimeError("seed_state는 첫 chunk 전에 한 번만 호출할 수 있다")
        position = np.asarray(position)
        active = np.asarray(active, dtype=bool)
        ball_position = np.asarray(ball_position)
        if position.shape != (self._team.size, 2):
            raise ValueError("initial state player축은 roster와 같아야 한다")
        if active.shape != self._team.shape or ball_position.shape != (3,):
            raise ValueError("initial state의 active/ball position shape가 잘못됐다")
        self._previous_player_position = position.copy()
        self._previous_player_active = active.copy()
        self._previous_player_generation = self._player_generation.copy()
        self._previous_ball_position = ball_position.copy()
        self._previous_restart_layout_ready = bool(np.asarray(layout_ready))
        self._previous_first_half_wall_end_tick = int(np.asarray(half_end))
        self._causal.seed_period(half_end)

    def _add_policy_telemetry(self, telemetry, owner_team: np.ndarray, receipts=None, actions=None) -> None:
        carrier = np.asarray(telemetry.carrier, dtype=np.int32)
        carrier_valid = np.asarray(telemetry.carrier_valid, dtype=bool)
        safe_carrier = np.clip(carrier, 0, self._team.size - 1)
        carrier_team = np.where(carrier_valid, self._team[safe_carrier], -1)
        overload = carrier_valid & (
            np.asarray(telemetry.local_opponents)
            > np.asarray(telemetry.local_teammates)
        )
        selected_intent = np.asarray(telemetry.selected_intent, dtype=np.int32)
        direction = np.asarray(telemetry.target_direction_family, dtype=np.int32)
        head_model_unavailable = np.zeros(carrier.shape, dtype=bool)
        if receipts is not None:
            release = classify_applied_releases(receipts.applied_contact, actions)
            actual_actor = np.asarray(receipts.applied_contact.actor)
            head_model_unavailable = release.head_candidate & (actual_actor == carrier)
        for team_index in range(2):
            selected = overload & (carrier_team == team_index)
            selected_count = int(np.count_nonzero(selected))
            self._policy_overload[team_index] += selected_count
            self._policy_intent[team_index] += np.bincount(
                selected_intent[selected], minlength=ACTION_INTENT_COUNT
            )[:ACTION_INTENT_COUNT]
            self._policy_direction[team_index] += np.asarray(
                [
                    np.count_nonzero(selected & (direction == value))
                    for value in (-1, 0, 1, 2)
                ],
                dtype=np.int64,
            )
            self._policy_completion_sum[team_index] += float(
                np.sum(np.asarray(telemetry.target_completion)[selected & ~head_model_unavailable])
            )
            self._policy_completion_observed[team_index] += int(np.count_nonzero(selected & ~head_model_unavailable))
            self._policy_completion_head_unavailable[team_index] += int(np.count_nonzero(selected & head_model_unavailable))
            self._policy_gap_sum[team_index] += float(
                np.sum(np.asarray(telemetry.target_score_gap)[selected & ~head_model_unavailable])
            )
            self._policy_next_outlet_sum[team_index] += float(
                np.sum(np.asarray(telemetry.target_next_outlet_quality)[selected])
            )
            self._policy_control_retained[team_index] += int(
                np.count_nonzero(
                    selected
                    & (selected_intent == INTENT_CONTROL)
                    & (owner_team == team_index)
                )
            )
            for local_frame in np.flatnonzero(
                selected & (selected_intent == INTENT_PASS)
            ):
                self._policy_overload_pass_team[self._frames + int(local_frame)] = (
                    team_index
                )

    def add(self, chunk: MatchRecordingChunk) -> None:
        self._add_host(jax.device_get(chunk))

    def _add_host(self, host: MatchRecordingChunk) -> None:
        """이미 회수한 NumPy chunk의 PyTree를 다시 device_get하지 않는다."""
        count = int(np.asarray(host.steps_executed))
        if count <= 0:
            return
        states = jax.tree.map(lambda value: np.asarray(value)[:count], host.states)
        actions = jax.tree.map(lambda value: np.asarray(value)[:count], host.actions)
        receipts = jax.tree.map(lambda value: np.asarray(value)[:count], host.receipts)
        challenge_attempts, challenge_fouls = _challenge_contact_counts(
            actions, receipts
        )
        self._challenge_contact_attempts += challenge_attempts
        self._challenge_fouls += challenge_fouls
        self._invariants.add(states)
        self._spatial.add(states)
        self._player_spatial.add(states, self._player_generation)
        self._team_shape.add(states)
        self._causal.add(states, actions, receipts, frame_offset=self._frames)
        self._goal_last_contacts.extend(extract_goal_last_contacts(
            states, receipts, self._team, frame_offset=self._frames))

        stream = extract_events(receipts, team_id=self._team)
        self._event_parts.append(
            EventStream(
                frame=stream.frame.astype(np.int64) + self._frames,
                kind=stream.kind,
                team=stream.team,
                actor=stream.actor,
                secondary_actor=stream.secondary_actor,
                severity=stream.severity,
                tactical_effect=stream.tactical_effect,
            )
        )
        intents = np.asarray(actions.intent, dtype=np.int32)
        self._intent += np.bincount(intents.reshape(-1), minlength=ACTION_INTENT_COUNT)[
            :ACTION_INTENT_COUNT
        ]
        for team_index in range(2):
            self._team_intent[team_index] += np.bincount(
                intents[:, self._team == team_index].reshape(-1),
                minlength=ACTION_INTENT_COUNT,
            )[:ACTION_INTENT_COUNT]

        possession = np.asarray(states.possession.player, dtype=np.int32)
        safe_owner = np.clip(possession, 0, self._team.size - 1)
        owner_team = np.where(
            (possession >= 0) & (possession < self._team.size),
            self._team[safe_owner],
            -1,
        )
        for team_index in range(2):
            self._possession[team_index] += np.count_nonzero(owner_team == team_index)
        self._possession_timeline.add(owner_team)

        telemetry = jax.tree.map(
            lambda value: np.asarray(value)[:count], host.policy_telemetry
        )
        if _has_policy_decision_telemetry(telemetry):
            self._add_policy_telemetry(telemetry, owner_team, receipts, actions)

        player_position = np.asarray(states.players.position, dtype=np.float64)
        player_active = np.asarray(states.players.active, dtype=bool)
        player_generation = np.broadcast_to(
            self._player_generation, player_active.shape
        )
        ball_position = np.asarray(states.ball.position, dtype=np.float64)
        layout_ready = np.asarray(states.restart_layout_ready, dtype=bool)
        half_end = np.asarray(states.first_half_wall_end_tick, dtype=np.int64)
        has_previous = self._previous_player_position is not None
        if self._previous_player_position is not None:
            player_position = np.concatenate(
                (self._previous_player_position[None], player_position), axis=0
            )
            player_active = np.concatenate(
                (self._previous_player_active[None], player_active), axis=0
            )
            player_generation = np.concatenate(
                (self._previous_player_generation[None], player_generation), axis=0
            )
            ball_position = np.concatenate(
                (self._previous_ball_position[None], ball_position), axis=0
            )
            layout_ready = np.concatenate(
                ((self._previous_restart_layout_ready,), layout_ready)
            )
            half_end = np.concatenate(
                ((self._previous_first_half_wall_end_tick,), half_end)
            )
        if player_position.shape[0] > 1:
            player_distance = np.linalg.norm(np.diff(player_position, axis=0), axis=-1)
            nonphysical_player_reset = (layout_ready[1:] & (~layout_ready[:-1])) | (
                (half_end[1:] >= 0) & (half_end[:-1] < 0)
            )
            player_distance *= (
                player_active[1:]
                & player_active[:-1]
                & (player_generation[1:] == player_generation[:-1])
                & (~nonphysical_player_reset[:, None])
            )
            distance_by_slot = np.sum(player_distance, axis=0)
            self._player_distance += distance_by_slot
            for slot, distance in enumerate(distance_by_slot):
                key = (slot, int(self._player_generation[slot]))
                self._player_generation_distance[key] += float(distance)
            for team_index in range(2):
                self._team_distance[team_index] += np.sum(
                    player_distance[:, self._team == team_index]
                )
            ball_event_reset = (
                np.asarray(receipts.boundary.occurred, dtype=bool)
                | np.asarray(receipts.foul.occurred, dtype=bool)
                | np.asarray(receipts.offside.occurred, dtype=bool)
                | np.asarray(receipts.restart_retouch.occurred, dtype=bool)
                | np.asarray(receipts.goalkeeper_holding.expired, dtype=bool)
            )
            if not has_previous:
                ball_event_reset = ball_event_reset[1:]
            ball_reset = ball_event_reset | ((half_end[1:] >= 0) & (half_end[:-1] < 0))
            ball_step_distance = np.linalg.norm(np.diff(ball_position, axis=0), axis=-1)
            self._ball_distance += float(np.sum(ball_step_distance[~ball_reset]))
        # 작은 carry view가 이전 batch chunk의 원본 buffer를 붙잡지 않게 한다.
        self._previous_player_position = np.asarray(states.players.position[-1]).copy()
        self._previous_player_active = np.asarray(states.players.active[-1])
        self._previous_player_generation = self._player_generation.copy()
        self._previous_ball_position = np.asarray(states.ball.position[-1]).copy()
        self._previous_restart_layout_ready = bool(states.restart_layout_ready[-1])
        self._previous_first_half_wall_end_tick = int(
            states.first_half_wall_end_tick[-1]
        )

        ball_speed = np.linalg.norm(
            np.asarray(states.ball.velocity, dtype=np.float64), axis=-1
        )
        self._ball_speed_sum += float(np.sum(ball_speed))
        self._ball_speed_count += int(ball_speed.size)
        self._ball_speed_max = max(self._ball_speed_max, float(np.max(ball_speed)))
        self._live_ball_frames += int(
            np.count_nonzero(np.asarray(states.ball.live, dtype=bool))
        )
        self._frames += count

    def events(self) -> EventStream:
        if not self._event_parts:
            return EventStream(
                frame=np.empty(0, dtype=np.int64),
                kind=np.empty(0, dtype=np.int8),
                team=np.empty(0, dtype=np.int32),
                actor=np.empty(0, dtype=np.int32),
                secondary_actor=np.empty(0, dtype=np.int32),
                severity=np.empty(0, dtype=np.int32),
                tactical_effect=np.empty(0, dtype=np.int32),
            )
        return EventStream(
            **{
                field: np.concatenate(
                    [getattr(part, field) for part in self._event_parts]
                )
                for field in EventStream.__dataclass_fields__
            }
        )

    def invariant_report(self) -> MatchInvariantReport:
        """현재까지 누적한 long-match invariant 결과를 반환한다."""
        return self._invariants.finish()

    def finish(self, final_state) -> MatchReport:
        stream = self.events()
        counts = {
            name: int(np.count_nonzero(stream.kind == index))
            for index, name in enumerate(EVENT_NAMES)
        }
        foul = stream.kind == 5
        severity = np.bincount(stream.severity[foul], minlength=4)[:4]
        tactical = np.bincount(stream.tactical_effect[foul], minlength=3)[:3]
        controlled = max(int(np.sum(self._possession)), 1)
        score = np.asarray(jax.device_get(final_state.score), dtype=np.int32)
        causal = self._causal.finish(self._frames)
        generations = self._identity.finish(self._frames)
        overload_pass_attempts = np.zeros(2, dtype=np.int64)
        overload_pass_completions = np.zeros(2, dtype=np.int64)
        for row in causal.pass_map:
            team = self._policy_overload_pass_team.get(int(row["source_frame"]))
            if team is None:
                continue
            overload_pass_attempts[team] += 1
            overload_pass_completions[team] += int(bool(row["completed"]))
        policy_summary = tuple(
            {
                "overload_frames": int(self._policy_overload[team]),
                "selected_intents": tuple(
                    int(value) for value in self._policy_intent[team]
                ),
                "target_directions": tuple(
                    int(value) for value in self._policy_direction[team]
                ),
                "mean_target_completion": float(
                    self._policy_completion_sum[team]
                    / self._policy_completion_observed[team]
                ) if self._policy_completion_observed[team] else (
                    None if self._policy_completion_head_unavailable[team] else 0.0
                ),
                "target_completion_observed_frames": int(self._policy_completion_observed[team]),
                "target_completion_head_unavailable_frames": int(self._policy_completion_head_unavailable[team]),
                "target_completion_coverage": (
                    float(self._policy_completion_observed[team]/self._policy_overload[team])
                    if self._policy_overload[team] else None
                ),
                "mean_target_score_gap": (
                    float(self._policy_gap_sum[team] / self._policy_completion_observed[team])
                    if self._policy_completion_observed[team]
                    else None if self._policy_completion_head_unavailable[team] else 0.0
                ),
                "mean_target_next_outlet_quality": float(
                    self._policy_next_outlet_sum[team]
                    / max(self._policy_overload[team], 1)
                ),
                "control_possession_retained": int(self._policy_control_retained[team]),
                "pass_attempts": int(overload_pass_attempts[team]),
                "pass_completions": int(overload_pass_completions[team]),
            }
            for team in range(2)
        )
        return MatchReport(
            schema=MATCH_REPORT_SCHEMA,
            deliberate_release_summary=causal.release_observation,
            frames=self._frames,
            control_dt_s=self._control_dt_s,
            final_score=(int(score[0]), int(score[1])),
            event_counts=counts,
            foul_severity_counts=tuple(int(value) for value in severity),
            foul_tactical_counts=tuple(int(value) for value in tactical),
            challenge_contact_attempts=self._challenge_contact_attempts,
            challenge_fouls=self._challenge_fouls,
            intent_counts=tuple(int(value) for value in self._intent),
            team_intent_counts=tuple(
                tuple(int(value) for value in row) for row in self._team_intent
            ),
            possession_frames=tuple(int(value) for value in self._possession),
            possession_share=tuple(
                float(value) / controlled for value in self._possession
            ),
            team_distance_m=tuple(float(value) for value in self._team_distance),
            player_distance_m=tuple(float(value) for value in self._player_distance),
            ball_distance_m=self._ball_distance,
            mean_ball_speed_mps=(self._ball_speed_sum / max(self._ball_speed_count, 1)),
            maximum_ball_speed_mps=self._ball_speed_max,
            live_ball_frames=self._live_ball_frames,
            pass_attempts=causal.pass_attempts,
            pass_completions=causal.pass_completions,
            shoot_releases=causal.shoot_releases,
            shot_outcomes=causal.shot_outcomes,
            pass_map=causal.pass_map,
            shot_map=causal.shot_map,
            spatial=self._spatial.finish(),
            tactical_summary=build_tactical_summary(
                causal,
                pass_window_frames=self._pass_window_frames,
                goal_line_x=self._goal_line_x,
                live_ball_minutes=self._live_ball_frames * self._control_dt_s / 60.0,
            ),
            team_shape_summary=self._team_shape.finish(),
            policy_decision_summary=policy_summary,
            player_generations=generations,
            goal_last_contacts=tuple(self._goal_last_contacts),
            player_action_credit=build_player_action_credit(
                causal,
                generations,
                distance_by_generation_m=self._player_generation_distance,
                spatial_by_generation=self._player_spatial.finish(),
                goal_line_x=self._goal_line_x,
            ),
            roster_summary=self._roster_summary,
            possession_timeline=self._possession_timeline.finish(),
            reference_context={
                "dfl_attacking_third_pass": DFL_ATTACKING_THIRD_REFERENCE
            },
        )


def _trim_host_chunk(chunk: MatchRecordingChunk) -> MatchRecordingChunk:
    """Padding frame을 제거하되 다음 chunk carry인 final state는 보존한다."""
    count = int(np.asarray(chunk.steps_executed))
    return MatchRecordingChunk(
        final_state=chunk.final_state,
        states=jax.tree.map(lambda value: np.asarray(value)[:count], chunk.states),
        actions=jax.tree.map(lambda value: np.asarray(value)[:count], chunk.actions),
        policy_telemetry=jax.tree.map(
            lambda value: np.asarray(value)[:count], chunk.policy_telemetry
        ),
        receipts=jax.tree.map(lambda value: np.asarray(value)[:count], chunk.receipts),
        valid=np.asarray(chunk.valid)[:count],
        steps_executed=np.asarray(chunk.steps_executed),
        terminated=np.asarray(chunk.terminated),
        truncated=np.asarray(chunk.truncated),
        termination_reason=np.asarray(chunk.termination_reason),
    )


def record_match(
    chunk_kernel: Callable,
    initial_state,
    roster,
    match_key,
    setup: MatchSetup,
    maximum_steps: int,
    *,
    chunk_sink: Callable[[int, int, MatchRecordingChunk], None] | None = None,
    policy_context=None,
) -> MatchRecording:
    """Compiled chunk kernel을 반복해 bounded-memory 경기 기록을 완성한다."""
    if isinstance(maximum_steps, bool) or not isinstance(maximum_steps, int):
        raise TypeError("maximum_steps는 정수여야 한다")
    if maximum_steps <= 0:
        raise ValueError("maximum_steps는 양수여야 한다")
    chunk_size = getattr(chunk_kernel, "max_steps", None)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise TypeError("chunk_kernel은 make_match_recording_chunk 결과여야 한다")
    if policy_context is not None and not getattr(
        chunk_kernel, "policy_takes_context", False
    ):
        raise TypeError("이 chunk_kernel은 policy_context를 받지 않는다")
    compiled_chunk = jax.jit(chunk_kernel)
    accumulator = MatchReportAccumulator(
        roster,
        timebase=getattr(chunk_kernel, "timebase", DEFAULT_TIMEBASE),
        field_half_extent=getattr(chunk_kernel, "field_half_extent", (52.5, 34.0)),
    )
    accumulator.seed_state(initial_state)
    state = initial_state
    remaining = maximum_steps
    chunks = 0
    frame_offset = 0
    while remaining > 0:
        arguments = (
            state,
            roster,
            match_key,
            setup,
            jnp.int32(min(remaining, chunk_size)),
        )
        result = (
            compiled_chunk(*arguments)
            if policy_context is None
            else compiled_chunk(*arguments, policy_context)
        )
        host_result = _trim_host_chunk(jax.device_get(result))
        if chunk_sink is not None:
            chunk_sink(0, frame_offset, host_result)
        accumulator._add_host(host_result)
        steps = int(np.asarray(host_result.steps_executed))
        state = result.final_state
        chunks += int(steps > 0)
        remaining -= steps
        frame_offset += steps
        done = bool(host_result.terminated | host_result.truncated)
        if done or steps == 0:
            break
    return MatchRecording(
        final_state=state,
        events=accumulator.events(),
        report=accumulator.finish(state),
        chunks=chunks,
        termination_reason=int(np.asarray(host_result.termination_reason)),
        invariants=accumulator.invariant_report(),
    )


def _validate_recording_batch_inputs(initial_states, match_keys, **trees) -> int:
    """모든 recording 입력 PyTree가 같은 leading match축을 갖는지 확인한다."""

    control_tick = initial_states.control_tick
    if control_tick.ndim != 1 or control_tick.shape[0] <= 0:
        raise ValueError(
            "initial_states는 비어 있지 않은 선두 match batch축을 가져야 한다"
        )
    batch_size = int(control_tick.shape[0])

    def validate(name, tree) -> None:
        for leaf in jax.tree.leaves(tree):
            if leaf.ndim < 1 or leaf.shape[0] != batch_size:
                raise ValueError(
                    f"{name}의 모든 leaf 선두 batch축은 initial_states와 같아야 한다"
                )

    validate("initial_states", initial_states)
    validate("match_keys", match_keys)
    for name, tree in trees.items():
        validate(name, tree)
    return batch_size


def _make_batch_recording_kernel(chunk_kernel):
    """Reuse the per-device executable across recording chunks."""
    if jax.local_device_count() <= 1:
        return jax.jit(jax.vmap(chunk_kernel))
    device_mapper = jax.pmap(
        lambda *local_trees: jax.vmap(chunk_kernel)(*local_trees)
    )
    return lambda *trees: _map_independent_matches(
        chunk_kernel, *trees, device_mapper=device_mapper
    )


@lru_cache(maxsize=16)
def _cached_batch_recording_kernel(chunk_kernel):
    return _make_batch_recording_kernel(chunk_kernel)


class _ParallelMatchProcessor:
    """CPU worker에서 한 경기의 report와 선택적 추가 집계 carry를 유지한다."""

    def __init__(self, index, payload):
        self.accumulator = MatchReportAccumulator(
            payload["roster"], timebase=payload["timebase"],
            field_half_extent=payload["field_half_extent"],
        )
        self.accumulator.seed_state(payload["initial_state"])
        self.final_state = payload["initial_state"]
        self.chunks = 0
        self.reason = 0
        factory = payload["extra_factory"]
        self.extra = None if factory is None else factory(index, payload)

    def process(self, offset, chunk):
        self.final_state = chunk.final_state
        self.reason = int(np.asarray(chunk.termination_reason))
        if int(np.asarray(chunk.steps_executed)) > 0:
            if self.extra is not None:
                self.extra.process(offset, chunk)
            self.accumulator._add_host(chunk)
            self.chunks += 1

    def finish(self):
        recording = MatchRecording(
            final_state=self.final_state, events=self.accumulator.events(),
            report=self.accumulator.finish(self.final_state), chunks=self.chunks,
            termination_reason=self.reason, invariants=self.accumulator.invariant_report(),
        )
        extra = None if self.extra is None else self.extra.finish(recording)
        return recording, extra


def _make_parallel_match_processor(index, payload):
    return _ParallelMatchProcessor(index, payload)


def record_match_batch(
    chunk_kernel: Callable,
    initial_states,
    rosters,
    match_keys,
    setups,
    maximum_steps: int,
    *,
    chunk_sink: Callable[[int, int, MatchRecordingChunk], None] | None = None,
    policy_contexts=None,
    prefetch_next_chunk: bool = False,
    host_workers: int = 0,
    host_sink_factory: Callable | None = None,
    host_sink_result: Callable | None = None,
) -> MatchRecordingBatch:
    """독립 경기 batch를 기록하고 선택적으로 CPU process별로 host 집계를 분산한다.

    ``host_workers=0``은 기존 순차 경로다. 양수이면 경기별 고정 worker를 사용하고
    요청 수를 batch와 허용 CPU core 수로 제한한다. 한 chunk의 집계가 끝나기 전
    다음 host chunk를 전달하지 않는다. 기존 ``chunk_sink``는 parent에서 순차 호출한다.
    무거운 추가 집계는 import 가능한 ``host_sink_factory(index, payload)``로 전달한다.
    Payload의 state/roster는 NumPy PyTree이며 timebase와 field_half_extent를 포함한다.
    추가 processor의 ``process(offset, chunk)``와 ``finish(recording)``를 worker에서
    호출하고, 최종 결과는 parent의 ``host_sink_result(index, value)``에 순서대로 전달한다.
    """
    if isinstance(host_workers, bool) or not isinstance(host_workers, int):
        raise TypeError("host_workers는 정수여야 한다")
    if host_workers < 0:
        raise ValueError("host_workers는 음수일 수 없다")
    if (host_sink_factory is None) != (host_sink_result is None):
        raise ValueError("host_sink_factory와 host_sink_result를 함께 제공해야 한다")
    if host_sink_factory is not None and (
        not callable(host_sink_factory) or not callable(host_sink_result)
    ):
        raise TypeError("host sink factory와 result는 callable이어야 한다")
    kwargs = dict(chunk_sink=chunk_sink, policy_contexts=policy_contexts,
                  prefetch_next_chunk=prefetch_next_chunk)
    if host_workers == 0:
        if host_sink_factory is not None:
            raise ValueError("병렬 host sink에는 양수 host_workers가 필요하다")
        return _record_match_batch_impl(chunk_kernel, initial_states, rosters, match_keys,
                                       setups, maximum_steps, **kwargs)
    if os.name != "posix":
        raise RuntimeError("병렬 host 집계는 현재 POSIX에서만 지원한다")
    if isinstance(maximum_steps, bool) or not isinstance(maximum_steps, int):
        raise TypeError("maximum_steps는 정수여야 한다")
    if maximum_steps <= 0:
        raise ValueError("maximum_steps는 양수여야 한다")
    if type(prefetch_next_chunk) is not bool:
        raise TypeError("prefetch_next_chunk는 bool이어야 한다")
    chunk_size = getattr(chunk_kernel, "max_steps", None)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise TypeError("chunk_kernel은 make_match_recording_chunk 결과여야 한다")
    if policy_contexts is not None and not getattr(chunk_kernel, "policy_takes_context", False):
        raise TypeError("이 chunk_kernel은 policy_contexts를 받지 않는다")
    contexts = {} if policy_contexts is None else {"policy_contexts": policy_contexts}
    batch_size = _validate_recording_batch_inputs(
        initial_states, match_keys, rosters=rosters, setups=setups, **contexts,
    )
    from messi.analysis.host_parallel import HostEvaluationPool, _factory_reference

    if host_sink_factory is not None:
        _factory_reference(host_sink_factory)

    host_states, host_rosters = jax.device_get((initial_states, rosters))
    payloads = {
        index: {
            "initial_state": jax.tree.map(lambda value: value[index], host_states),
            "roster": jax.tree.map(lambda value: value[index], host_rosters),
            "timebase": getattr(chunk_kernel, "timebase", DEFAULT_TIMEBASE),
            "field_half_extent": getattr(chunk_kernel, "field_half_extent", (52.5, 34.0)),
            "extra_factory": host_sink_factory,
        }
        for index in range(batch_size)
    }
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    with HostEvaluationPool(min(host_workers, batch_size, available),
                            _make_parallel_match_processor, payloads) as pool:
        return _record_match_batch_impl(
            chunk_kernel, initial_states, rosters, match_keys, setups, maximum_steps,
            _host_pool=pool, _host_rosters=host_rosters,
            _host_sink_result=host_sink_result, **kwargs,
        )


def _record_match_batch_impl(
    chunk_kernel: Callable,
    initial_states,
    rosters,
    match_keys,
    setups,
    maximum_steps: int,
    *,
    chunk_sink: Callable[[int, int, MatchRecordingChunk], None] | None = None,
    policy_contexts=None,
    prefetch_next_chunk: bool = False,
    _host_pool=None,
    _host_rosters=None,
    _host_sink_result=None,
) -> MatchRecordingBatch:
    """Repeat batched chunks and keep per-match statistics on the host.

    ``prefetch_next_chunk``는 현재 host chunk의 sink 처리를 시작하기 전에 다음 device
    chunk를 enqueue한다. JAX dispatch의 비동기 실행만 이용하며 host chunk를 추가로
    보관하거나 sink 호출 순서를 바꾸지 않는다.
    """
    if isinstance(maximum_steps, bool) or not isinstance(maximum_steps, int):
        raise TypeError("maximum_steps는 정수여야 한다")
    if maximum_steps <= 0:
        raise ValueError("maximum_steps는 양수여야 한다")
    if type(prefetch_next_chunk) is not bool:
        raise TypeError("prefetch_next_chunk는 bool이어야 한다")
    chunk_size = getattr(chunk_kernel, "max_steps", None)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise TypeError("chunk_kernel은 make_match_recording_chunk 결과여야 한다")
    if policy_contexts is not None and not getattr(
        chunk_kernel, "policy_takes_context", False
    ):
        raise TypeError("이 chunk_kernel은 policy_contexts를 받지 않는다")
    context_trees = (
        {} if policy_contexts is None else {"policy_contexts": policy_contexts}
    )
    batch_size = _validate_recording_batch_inputs(
        initial_states,
        match_keys,
        rosters=rosters,
        setups=setups,
        **context_trees,
    )
    host_rosters = jax.device_get(rosters) if _host_rosters is None else _host_rosters
    accumulators = [
        MatchReportAccumulator(
            jax.tree.map(lambda value, i=index: value[i], host_rosters),
            timebase=getattr(chunk_kernel, "timebase", DEFAULT_TIMEBASE),
            field_half_extent=getattr(chunk_kernel, "field_half_extent", (52.5, 34.0)),
        )
        for index in range(batch_size) if _host_pool is None
    ]
    (
        initial_position,
        initial_active,
        initial_ball_position,
        initial_layout_ready,
        initial_half_end,
    ) = jax.device_get(
        (
            initial_states.players.position,
            initial_states.players.active,
            initial_states.ball.position,
            initial_states.restart_layout_ready,
            initial_states.first_half_wall_end_tick,
        )
    )
    for index, accumulator in enumerate(accumulators):
        accumulator._seed_positions(
            initial_position[index],
            initial_active[index],
            initial_ball_position[index],
            initial_layout_ready[index],
            initial_half_end[index],
        )
    try:
        hash(chunk_kernel)
    except TypeError:
        compiled_chunk = _make_batch_recording_kernel(chunk_kernel)
    else:
        compiled_chunk = _cached_batch_recording_kernel(chunk_kernel)
    states = initial_states
    elapsed = 0
    chunks = 0
    match_chunks = np.zeros(batch_size, dtype=np.int32)
    done = np.zeros(batch_size, dtype=bool)
    frame_offsets = np.zeros(batch_size, dtype=np.int64)
    pending_result = None
    while elapsed < maximum_steps and not bool(np.all(done)):
        budget = min(maximum_steps - elapsed, chunk_size)
        if pending_result is None:
            budgets = jnp.full((batch_size,), budget, dtype=jnp.int32)
            arguments = (states, rosters, match_keys, setups, budgets)
            result = (
                compiled_chunk(*arguments)
                if policy_contexts is None
                else compiled_chunk(*arguments, policy_contexts)
            )
        else:
            result = pending_result
            pending_result = None
        host_result = jax.device_get(result)
        host_steps = np.asarray(host_result.steps_executed, dtype=np.int32)
        next_done = np.asarray(
            host_result.terminated | host_result.truncated, dtype=bool
        )
        next_elapsed = elapsed + budget
        if (
            prefetch_next_chunk
            and next_elapsed < maximum_steps
            and not bool(np.all(next_done))
            and np.any(host_steps > 0)
        ):
            next_budget = min(maximum_steps - next_elapsed, chunk_size)
            next_budgets = jnp.full(
                (batch_size,), next_budget, dtype=jnp.int32
            )
            next_arguments = (
                result.final_state,
                rosters,
                match_keys,
                setups,
                next_budgets,
            )
            pending_result = (
                compiled_chunk(*next_arguments)
                if policy_contexts is None
                else compiled_chunk(*next_arguments, policy_contexts)
            )
        match_chunks += (host_steps > 0).astype(np.int32)
        host_items = []
        for index in range(batch_size):
            if host_steps[index] > 0 or (_host_pool is not None and chunks == 0):
                match_chunk = _trim_host_chunk(
                    jax.tree.map(lambda value, i=index: value[i], host_result)
                )
                if chunk_sink is not None and host_steps[index] > 0:
                    chunk_sink(index, int(frame_offsets[index]), match_chunk)
                if _host_pool is None:
                    accumulators[index]._add_host(match_chunk)
                else:
                    host_items.append((index, int(frame_offsets[index]), match_chunk))
                frame_offsets[index] += int(host_steps[index])
        if _host_pool is not None:
            _host_pool.process_batch(host_items)
        states = result.final_state
        done = next_done
        elapsed = next_elapsed
        chunks += 1
        if not np.any(host_steps > 0):
            break
    # 마지막 chunk를 위에서 이미 host로 한 번 복사했다. 같은 batched final State를
    # 다시 device_get하지 않고 그 결과를 보고서와 최종 반환값에 재사용한다.
    host_states = host_result.final_state
    if _host_pool is not None:
        finished = _host_pool.finish()
        matches = tuple(finished[index][0] for index in range(batch_size))
        if _host_sink_result is not None:
            for index in range(batch_size):
                _host_sink_result(index, finished[index][1])
        return MatchRecordingBatch(matches=matches, chunks=chunks)
    matches = tuple(
        MatchRecording(
            final_state=jax.tree.map(lambda value, i=index: value[i], host_states),
            events=accumulator.events(),
            report=accumulator.finish(
                jax.tree.map(lambda value, i=index: value[i], host_states)
            ),
            chunks=int(match_chunks[index]),
            termination_reason=int(np.asarray(host_result.termination_reason[index])),
            invariants=accumulator.invariant_report(),
        )
        for index, accumulator in enumerate(accumulators)
    )
    return MatchRecordingBatch(matches=matches, chunks=chunks)
