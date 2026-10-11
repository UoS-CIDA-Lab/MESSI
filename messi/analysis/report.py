"""Evaluation output에서 재현 가능한 JSON·Markdown match report를 생성한다."""

from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np

from messi._json import strict_json_dumps
from messi.analysis.causal import (
    SHOT_OUTCOMES,
    CausalEventResolver,
    build_shot_on_target_proxy,
    build_shot_release_geometry,
    is_goal_directed_shot,
    is_shot_attempt,
)
from messi.analysis.events import EVENT_NAMES, extract_events
from messi.analysis.goal_attribution import (
    extract_goal_last_contacts,
    summarize_goal_last_contacts,
)
from messi.analysis.identity import PlayerGenerationTracker, build_player_action_credit
from messi.analysis.release import classify_applied_releases
from messi.analysis.shot_context import summarize_shot_context
from messi.analysis.shot_rule_stop import summarize_shot_rule_stops
from messi.analysis.spatial import (
    PlayerSpatialAccumulator,
    SpatialAccumulator,
    SpatialSummary,
    TeamShapeAccumulator,
)
from messi.analysis.timeline import PossessionTimelineAccumulator
from messi.core.action import (
    ACTION_INTENT_COUNT,
    INTENT_CHALLENGE,
    INTENT_CONTROL,
    INTENT_PASS,
)
from messi.core.timebase import DEFAULT_TIMEBASE, Timebase

MATCH_REPORT_SCHEMA = "messi.match-report/28"

_FORMATION_NAMES = ("4-3-3", "4-4-2", "3-5-2", "4-2-3-1", "3-4-3")
_ROLE_NAMES = (
    "Goalkeeper", "Centre back", "Full back", "Centre midfielder",
    "Wide midfielder", "Centre forward", "Wide forward",
)


def initial_roster_summary(roster) -> dict[str, object]:
    """Capture the actual reset roster before later substitutions replace slots."""
    host = jax.device_get(roster)
    team = np.asarray(host.team_id, dtype=np.int32)
    formation_id = np.asarray(host.formation_id, dtype=np.int32)
    formations = []
    for side in (0, 1):
        ids = np.unique(formation_id[team == side])
        chosen = int(ids[0]) if ids.size == 1 else -1
        formations.append({
            "team": side,
            "formation_id": chosen,
            "formation": _FORMATION_NAMES[chosen] if 0 <= chosen < len(_FORMATION_NAMES) else "unspecified",
        })
    return {
        "source": "recorded_roster",
        "formations": tuple(formations),
        "players": roster_player_profiles(host, np.zeros(team.size, dtype=np.int32), 0),
    }


def roster_player_profiles(
    roster, generations: np.ndarray, start_frame: int, slots=None,
) -> tuple[dict[str, object], ...]:
    """Snapshot realized player attributes for the given fixed slots."""
    host = jax.device_get(roster)
    team = np.asarray(host.team_id, dtype=np.int32)
    generations = np.asarray(generations, dtype=np.int32)
    selected = range(team.size) if slots is None else slots
    role = np.asarray(host.role, dtype=np.int32)
    anchor = np.asarray(host.formation_anchor, dtype=np.float32)
    ids = np.asarray(host.player_id, dtype=np.int32)
    goalkeeper = np.asarray(host.is_goalkeeper, dtype=bool)
    speed = np.asarray(host.max_speed, dtype=np.float32)
    height = np.asarray(host.height, dtype=np.float32)
    reach = np.asarray(host.reach_height, dtype=np.float32)
    control = np.asarray(host.ball_control, dtype=np.float32)
    endurance = np.asarray(host.endurance_factor, dtype=np.float32)
    return tuple({
        "slot": int(slot),
        "generation": int(generations[slot]),
        "start_frame": int(start_frame),
        "player_id": int(ids[slot]),
        "team": int(team[slot]),
        "is_goalkeeper": bool(goalkeeper[slot]),
        "role_id": int(role[slot]),
        "role": _ROLE_NAMES[int(role[slot])] if 0 <= int(role[slot]) < len(_ROLE_NAMES) else "Unspecified",
        "formation_anchor_reference_m": [float(anchor[slot, 0]), float(anchor[slot, 1])],
        "max_speed_mps": float(speed[slot]),
        "height_m": float(height[slot]),
        "reach_height_m": float(reach[slot]),
        "ball_control": float(control[slot]),
        "endurance_factor": float(endurance[slot]),
    } for slot in selected)

_DIRECTION_FIVE = (
    "direct_backward",
    "backward_diagonal",
    "lateral",
    "forward_diagonal",
    "direct_forward",
)

DFL_ATTACKING_THIRD_REFERENCE = {
    "source": "DFL seven-match PlayAngle extraction",
    "matches": 7,
    "attempts": 921,
    "completion": 607 / 921,
    "direction_share": {
        "backward": 335 / 921,
        "lateral": 259 / 921,
        "forward": 327 / 921,
    },
    "comparability": (
        "DFL provider evaluation and MESSI next-distinct-contact measure "
        "different quantities; use this as a descriptive reference only."
    ),
}

_POLICY_DECISION_TELEMETRY_FIELDS = (
    "carrier_valid",
    "carrier",
    "local_teammates",
    "local_opponents",
    "target_direction_family",
    "target_completion",
    "target_score_gap",
    "target_next_outlet_quality",
    "selected_intent",
)


def _challenge_contact_counts(actions, receipts) -> tuple[int, int]:
    """실제 challenge contest 시도와 그중 foul을 분리해 센다.

    Contest foul은 contact response 적용 전에 stoppage가 되어
    ``applied_contact.occurred``에 남지 않는다. Transition contract상 이 foul은
    substep 시작(``time_fraction == 0``)에 기록되고, swept body-impact foul은 양의
    time fraction을 가진다. Offender action까지 CHALLENGE인지 확인해 분류한다.
    """

    applied = receipts.applied_contact
    challenge_contact = np.asarray(applied.occurred, dtype=bool) & (
        np.asarray(applied.intent, dtype=np.int32) == INTENT_CHALLENGE
    )
    foul = receipts.foul
    foul_occurred = np.asarray(foul.occurred, dtype=bool)
    offender = np.asarray(foul.offender, dtype=np.int32)
    intents = np.asarray(actions.intent, dtype=np.int32)
    valid_offender = (offender >= 0) & (offender < intents.shape[1])
    safe_offender = np.clip(offender, 0, intents.shape[1] - 1)
    frame = np.arange(intents.shape[0], dtype=np.int32)
    challenge_foul = (
        foul_occurred
        & (np.asarray(foul.time_fraction) == 0.0)
        & valid_offender
        & (intents[frame, safe_offender] == INTENT_CHALLENGE)
    )
    # 현재 transition에서는 두 mask가 배타적이다. Union으로 세어 future receipt가
    # foul contact도 applied로 보존하더라도 같은 frame을 중복 집계하지 않는다.
    if hasattr(receipts, "challenge_attempt_count"):
        attempts = int(
            np.asarray(receipts.challenge_attempt_count, dtype=np.int64).sum()
        )
    else:
        attempts = int(np.count_nonzero(challenge_contact | challenge_foul))
    return attempts, int(np.count_nonzero(challenge_foul))


def _has_policy_decision_telemetry(value) -> bool:
    return all(hasattr(value, field) for field in _POLICY_DECISION_TELEMETRY_FIELDS)


def _direction_five(row) -> str | None:
    direction = row.get("attack_normalized_direction")
    if direction is None:
        start = row.get("attack_normalized_source_position")
        end = row.get("attack_normalized_terminal_position")
        if start is None or end is None:
            return None
        direction = np.asarray(end[:2], dtype=np.float64) - np.asarray(
            start[:2], dtype=np.float64
        )
    direction = np.asarray(direction[:2], dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length <= 1.0e-9:
        return None
    cosine = float(direction[0]) / length
    direct = float(np.cos(np.deg2rad(30.0)))
    if cosine <= -direct:
        return "direct_backward"
    if cosine <= -0.34:
        return "backward_diagonal"
    if cosine < 0.34:
        return "lateral"
    if cosine < direct:
        return "forward_diagonal"
    return "direct_forward"


def build_tactical_summary(
    causal, *, pass_window_frames: int = 9000, goal_line_x: float = 52.5,
    live_ball_minutes: float | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Attack-normalized causal row를 team별 전술 count로 축약한다."""

    match_ticks = causal.pass_source_match_ticks
    use_match_ticks = len(match_ticks) == len(causal.pass_map)
    source_ticks = (
        match_ticks
        if use_match_ticks
        else tuple(int(row.get("source_frame", 0)) for row in causal.pass_map)
    )
    summaries = []
    for team in range(2):
        team_passes = [
            (row, source_tick)
            for row, source_tick in zip(causal.pass_map, source_ticks, strict=True)
            if int(row["team"]) == team
        ]
        passes = [row for row, _ in team_passes]
        shots = [row for row in causal.shot_map if int(row["team"]) == team]

        def terminal(row):
            return row.get("attack_normalized_terminal_position")

        def source(row):
            return row.get("attack_normalized_source_position")

        forward = [
            row
            for row in passes
            if terminal(row) is not None
            and source(row) is not None
            and float(terminal(row)[0]) - float(source(row)[0]) >= 4.0
        ]
        attacking_third = [
            row
            for row in passes
            if source(row) is not None and float(source(row)[0]) >= 17.5
        ]

        def direction_family(row):
            start = source(row)
            end = terminal(row)
            if start is None or end is None:
                return None
            delta = np.asarray(end[:2], dtype=np.float64) - np.asarray(
                start[:2], dtype=np.float64
            )
            length = float(np.linalg.norm(delta))
            if length <= 1.0e-9:
                return None
            cosine = float(delta[0]) / length
            return (
                "backward"
                if cosine <= -0.34
                else ("lateral" if cosine < 0.34 else "forward")
            )

        attacking_third_direction = tuple(
            direction_family(row) for row in attacking_third
        )
        box_entries = [
            row
            for row in passes
            if source(row) is not None
            and terminal(row) is not None
            and not (
                float(source(row)[0]) >= 36.0 and abs(float(source(row)[1])) <= 20.16
            )
            and float(terminal(row)[0]) >= 36.0
            and abs(float(terminal(row)[1])) <= 20.16
        ]
        # Shot attempt는 goal 근처 forward release(population)이면서 release 탄도가
        # goal mouth를 폭과 crossbar 높이 모두에서 통과해야 한다. Population 자체는
        # 유효 슛 비율의 분모로 따로 보고한다.
        releases_near_goal = [
            row for row in shots if is_shot_attempt(row, goal_line_x=goal_line_x)
        ]
        attempts = [
            row
            for row in releases_near_goal
            if is_goal_directed_shot(row, goal_line_x=goal_line_x)
        ]
        shots_in_box = [
            row
            for row in attempts
            if source(row) is not None
            and float(source(row)[0]) >= 36.0
            and abs(float(source(row)[1])) <= 20.16
        ]
        release_box_known = [row for row in releases_near_goal if source(row) is not None]
        releases_in_box = [
            row for row in release_box_known
            if float(source(row)[0]) >= goal_line_x - 16.5
            and abs(float(source(row)[1])) <= 20.16
        ]
        open_play = [row for row in passes if int(row.get("restart_kind", 0)) == 0]
        restart_passes = [row for row in passes if int(row.get("restart_kind", 0)) != 0]
        terminal_kinds = {
            name: sum(row.get("terminal_kind") == name for row in passes)
            for name in (
                "contact",
                "same_actor_retouch",
                "boundary",
                "goal",
                "capture_end",
            )
        }
        direction_five = {
            name: sum(_direction_five(row) == name for row in open_play)
            for name in _DIRECTION_FIVE
        }
        def is_on_target(row):
            """Goal, 또는 상대 goalkeeper가 막은 shot. Attempt는 이미 조준을 통과했다."""

            return row.get("outcome") in ("goal", "saved")

        attempt_outcomes = {
            name: sum(row.get("outcome") == name for row in attempts)
            for name in SHOT_OUTCOMES
        }
        release_outcomes = {
            name: sum(row.get("outcome") == name for row in releases_near_goal)
            for name in SHOT_OUTCOMES
        }
        shot_distances = [
            float(
                np.linalg.norm(
                    np.asarray((goal_line_x, 0.0), dtype=np.float64)
                    - np.asarray(row["attack_normalized_source_position"][:2])
                )
            )
            for row in attempts
            if row.get("attack_normalized_source_position") is not None
        ]
        release_distances = [
            float(np.hypot(goal_line_x - float(source(row)[0]), float(source(row)[1])))
            for row in release_box_known
        ]
        pass_windows = []
        for window in range(6):
            selected = [
                row
                for row, source_tick in team_passes
                if min(int(source_tick) // max(pass_window_frames, 1), 5) == window
            ]
            pass_windows.append(
                {
                    "window": window,
                    "attempts": len(selected),
                    "completions": sum(bool(row["completed"]) for row in selected),
                }
            )
        summaries.append(
            {
                "forward_pass_attempts": len(forward),
                "forward_pass_completions": sum(
                    bool(row["completed"]) for row in forward
                ),
                "attacking_third_pass_attempts": len(attacking_third),
                "attacking_third_pass_completions": sum(
                    bool(row["completed"]) for row in attacking_third
                ),
                "attacking_third_backward_attempts": attacking_third_direction.count(
                    "backward"
                ),
                "attacking_third_lateral_attempts": attacking_third_direction.count(
                    "lateral"
                ),
                "attacking_third_forward_attempts": attacking_third_direction.count(
                    "forward"
                ),
                "box_entry_pass_attempts": len(box_entries),
                "box_entry_pass_completions": sum(
                    bool(row["completed"]) for row in box_entries
                ),
                "shots_in_box": len(shots_in_box),
                "shot_attempts": len(attempts),
                "shots_on_target": sum(is_on_target(row) for row in attempts),
                "open_play_pass_attempts": len(open_play),
                "open_play_pass_completions": sum(
                    bool(row["completed"]) for row in open_play
                ),
                "restart_pass_attempts": len(restart_passes),
                "restart_pass_completions": sum(
                    bool(row["completed"]) for row in restart_passes
                ),
                "pass_terminal_kinds": terminal_kinds,
                "open_play_direction_five": direction_five,
                "pass_completion_by_15m": tuple(pass_windows),
                "submitted_shoot_releases": len(shots),
                "shot_releases_near_goal": len(releases_near_goal),
                "shot_release_context": summarize_shot_context(
                    shots, goal_line_x=goal_line_x, live_ball_minutes=live_ball_minutes
                ),
                "shot_release_rule_stop": summarize_shot_rule_stops(releases_near_goal, goal_line_x=goal_line_x),
                "shot_release_box_known": len(release_box_known),
                "shot_releases_in_box": len(releases_in_box),
                "shot_releases_outside_box": len(release_box_known) - len(releases_in_box),
                "shot_release_box_unknown": len(releases_near_goal) - len(release_box_known),
                "shot_attempt_outcomes": attempt_outcomes,
                "shot_release_outcomes": release_outcomes,
                "shot_on_target_proxy": build_shot_on_target_proxy(
                    len(releases_near_goal), attempt_outcomes, release_outcomes
                ),
                "shot_release_geometry": build_shot_release_geometry(
                    len(releases_near_goal),
                    distance_known_count=len(release_distances),
                    distance_sum_m=sum(release_distances),
                    box_known_count=len(release_box_known),
                    in_box_count=len(releases_in_box),
                ),
                "shots_on_target_in_box": sum(
                    is_on_target(row) for row in shots_in_box
                ),
                "mean_shot_attempt_distance_m": (
                    float(np.mean(shot_distances)) if shot_distances else 0.0
                ),
            }
        )
    return summaries[0], summaries[1]


@dataclass(frozen=True, slots=True)
class MatchReport:
    schema: str
    frames: int
    control_dt_s: float
    final_score: tuple[int, int]
    event_counts: dict[str, int]
    foul_severity_counts: tuple[int, int, int, int]
    foul_tactical_counts: tuple[int, int, int]
    challenge_contact_attempts: int
    challenge_fouls: int
    intent_counts: tuple[int, ...]
    team_intent_counts: tuple[tuple[int, ...], tuple[int, ...]]
    possession_frames: tuple[int, int]
    possession_share: tuple[float, float]
    team_distance_m: tuple[float, float]
    player_distance_m: tuple[float, ...]
    ball_distance_m: float
    mean_ball_speed_mps: float
    maximum_ball_speed_mps: float
    live_ball_frames: int
    pass_attempts: tuple[int, int]
    pass_completions: tuple[int, int]
    shoot_releases: tuple[int, int]
    shot_outcomes: dict[str, int]
    pass_map: tuple[dict[str, object], ...]
    shot_map: tuple[dict[str, object], ...]
    spatial: SpatialSummary
    tactical_summary: tuple[dict[str, int], dict[str, int]]
    team_shape_summary: tuple[dict[str, object], dict[str, object]]
    policy_decision_summary: tuple[dict[str, object], dict[str, object]]
    player_generations: tuple[dict[str, object], ...]
    player_action_credit: tuple[dict[str, object], ...]
    roster_summary: dict[str, object]
    possession_timeline: tuple[dict[str, object], ...]
    reference_context: dict[str, object]
    goal_last_contacts: tuple[dict[str, object], ...] = ()
    deliberate_release_summary: dict[str, object] | None = None

    @property
    def last_contact_goal_provenance(self):
        return summarize_goal_last_contacts(self.goal_last_contacts, self.final_score)

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "last_contact_goal_provenance": self.last_contact_goal_provenance}


def build_match_report(
    evaluation,
    roster,
    *,
    timebase: Timebase = DEFAULT_TIMEBASE,
    initial_state=None,
    field_half_extent: tuple[float, float] = (52.5, 34.0),
) -> MatchReport:
    initial_half_end = (
        np.int32(-1)
        if initial_state is None
        else initial_state.first_half_wall_end_tick
    )
    host, team, player_id, goalkeeper, initial_half_end = jax.device_get(
        (
            evaluation,
            roster.team_id,
            roster.player_id,
            roster.is_goalkeeper,
            initial_half_end,
        )
    )
    valid = getattr(host, "valid", None)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.ndim != 1:
            raise ValueError("evaluation.valid는 rank-1 time mask여야 한다")
        frame_count = int(np.count_nonzero(valid))
        expected = np.arange(valid.size) < frame_count
        if not np.array_equal(valid, expected):
            raise ValueError("evaluation.valid는 연속된 prefix mask여야 한다")
        if frame_count == 0:
            raise ValueError("dense match report에는 유효 frame이 하나 이상 필요하다")
        host = host._replace(
            states=jax.tree.map(
                lambda value: np.asarray(value)[:frame_count], host.states
            ),
            actions=jax.tree.map(
                lambda value: np.asarray(value)[:frame_count], host.actions
            ),
            policy_telemetry=jax.tree.map(
                lambda value: np.asarray(value)[:frame_count], host.policy_telemetry
            ),
            receipts=jax.tree.map(
                lambda value: np.asarray(value)[:frame_count], host.receipts
            ),
            valid=valid[:frame_count],
        )
    team = np.asarray(team, dtype=np.int32)
    host_roster = roster._replace(team_id=team, player_id=player_id)
    intents = np.asarray(host.actions.intent, dtype=np.int32)
    possession = np.asarray(host.states.possession.player, dtype=np.int32)
    safe = np.clip(possession, 0, team.size - 1)
    valid = (possession >= 0) & (possession < team.size)
    owner_team = np.where(valid, team[safe], -1)
    stream = extract_events(host.receipts, team_id=team)
    challenge_contact_attempts, challenge_fouls = _challenge_contact_counts(
        host.actions, host.receipts
    )
    causal = CausalEventResolver(
        team, is_goalkeeper=np.asarray(goalkeeper, dtype=bool), timebase=timebase
    )
    if initial_state is not None:
        causal.seed_period(initial_half_end)
    causal.add(host.states, host.actions, host.receipts, frame_offset=0)
    causal_summary = causal.finish(int(intents.shape[0]))
    spatial_accumulator = SpatialAccumulator(
        team,
        control_dt=timebase.control_dt,
        field_half_extent=field_half_extent,
    )
    spatial_accumulator.add(host.states)
    player_spatial = PlayerSpatialAccumulator(team, control_dt=timebase.control_dt)
    player_spatial.add(host.states)
    team_shape = TeamShapeAccumulator(team)
    team_shape.add(host.states)
    timeline = PossessionTimelineAccumulator(control_dt=timebase.control_dt)
    timeline.add(owner_team)
    identity = PlayerGenerationTracker(host_roster)
    generations = identity.finish(int(intents.shape[0]))
    rows = stream.rows()
    # EVENT_NAMES를 재선언하면 새 event kind가 count에서 조용히 빠진다.
    counts = {name: sum(row["kind"] == name for row in rows) for name in EVENT_NAMES}
    foul_mask = stream.kind == 5
    foul_severity_counts = np.bincount(stream.severity[foul_mask], minlength=4)[:4]
    foul_tactical_counts = np.bincount(stream.tactical_effect[foul_mask], minlength=3)[
        :3
    ]
    intent_counts = np.bincount(intents.reshape(-1), minlength=ACTION_INTENT_COUNT)[
        :ACTION_INTENT_COUNT
    ]
    score = np.asarray(host.final_state.score, dtype=np.int32)
    state_position = np.asarray(host.states.players.position, dtype=np.float64)
    state_active = np.asarray(host.states.players.active, dtype=bool)
    ball_position = np.asarray(host.states.ball.position, dtype=np.float64)
    ball_velocity = np.asarray(host.states.ball.velocity, dtype=np.float64)
    ball_live = np.asarray(host.states.ball.live, dtype=bool)
    distance_player_position = state_position
    distance_player_active = state_active
    distance_ball_position = ball_position
    distance_layout_ready = np.asarray(host.states.restart_layout_ready, dtype=bool)
    distance_half_end = np.asarray(host.states.first_half_wall_end_tick, dtype=np.int64)
    has_initial_state = initial_state is not None
    if initial_state is not None:
        (
            initial_position,
            initial_active,
            initial_ball_position,
            initial_layout_ready,
            initial_half_end,
        ) = jax.device_get(
            (
                initial_state.players.position,
                initial_state.players.active,
                initial_state.ball.position,
                initial_state.restart_layout_ready,
                initial_state.first_half_wall_end_tick,
            )
        )
        initial_position = np.asarray(initial_position, dtype=np.float64)
        initial_active = np.asarray(initial_active, dtype=bool)
        initial_ball_position = np.asarray(initial_ball_position, dtype=np.float64)
        if initial_position.shape != state_position.shape[1:]:
            raise ValueError(
                "initial_state player position shape가 evaluation과 다르다"
            )
        if initial_active.shape != state_active.shape[1:]:
            raise ValueError("initial_state active shape가 evaluation과 다르다")
        if initial_ball_position.shape != ball_position.shape[1:]:
            raise ValueError("initial_state ball position shape가 evaluation과 다르다")
        distance_player_position = np.concatenate(
            (initial_position[None, ...], state_position), axis=0
        )
        distance_player_active = np.concatenate(
            (initial_active[None, ...], state_active), axis=0
        )
        distance_ball_position = np.concatenate(
            (initial_ball_position[None, ...], ball_position), axis=0
        )
        distance_layout_ready = np.concatenate(
            ((bool(np.asarray(initial_layout_ready)),), distance_layout_ready)
        )
        distance_half_end = np.concatenate(
            ((int(np.asarray(initial_half_end)),), distance_half_end)
        )
    player_step_distance = np.linalg.norm(
        np.diff(distance_player_position, axis=0), axis=-1
    )
    if distance_player_position.shape[0] > 1:
        nonphysical_player_reset = (
            distance_layout_ready[1:] & (~distance_layout_ready[:-1])
        ) | ((distance_half_end[1:] >= 0) & (distance_half_end[:-1] < 0))
        player_step_distance *= (
            distance_player_active[1:] & distance_player_active[:-1]
        ) & (~nonphysical_player_reset[:, None])
    ball_step_distance = np.linalg.norm(
        np.diff(distance_ball_position, axis=0), axis=-1
    )
    ball_event_reset = (
        np.asarray(host.receipts.boundary.occurred, dtype=bool)
        | np.asarray(host.receipts.foul.occurred, dtype=bool)
        | np.asarray(host.receipts.offside.occurred, dtype=bool)
        | np.asarray(host.receipts.restart_retouch.occurred, dtype=bool)
        | np.asarray(host.receipts.goalkeeper_holding.expired, dtype=bool)
    )
    if not has_initial_state:
        ball_event_reset = ball_event_reset[1:]
    ball_reset = ball_event_reset | (
        (distance_half_end[1:] >= 0) & (distance_half_end[:-1] < 0)
    )
    ball_step_distance = np.where(ball_reset, 0.0, ball_step_distance)
    ball_speed = np.linalg.norm(ball_velocity, axis=-1)
    team_intent_counts = tuple(
        tuple(
            int(value)
            for value in np.bincount(
                intents[:, team == team_index].reshape(-1),
                minlength=ACTION_INTENT_COUNT,
            )[:ACTION_INTENT_COUNT]
        )
        for team_index in range(2)
    )
    team_distance = tuple(
        float(np.sum(player_step_distance[:, team == team_index]))
        for team_index in range(2)
    )
    player_distance = tuple(
        float(value) for value in np.sum(player_step_distance, axis=0)
    )
    possession_frames = (
        int(np.count_nonzero(owner_team == 0)),
        int(np.count_nonzero(owner_team == 1)),
    )
    controlled_frames = max(sum(possession_frames), 1)
    telemetry = host.policy_telemetry
    if _has_policy_decision_telemetry(telemetry):
        carrier = np.asarray(telemetry.carrier, dtype=np.int32)
        carrier_valid = np.asarray(telemetry.carrier_valid, dtype=bool)
        safe_carrier = np.clip(carrier, 0, team.size - 1)
        carrier_team = np.where(carrier_valid, team[safe_carrier], -1)
        local_teammates = np.asarray(telemetry.local_teammates)
        local_opponents = np.asarray(telemetry.local_opponents)
        overload = carrier_valid & (local_opponents > local_teammates)
        selected_intent = np.asarray(telemetry.selected_intent, dtype=np.int32)
        direction = np.asarray(telemetry.target_direction_family, dtype=np.int32)
        completion = np.asarray(telemetry.target_completion, dtype=np.float64)
        score_gap = np.asarray(telemetry.target_score_gap, dtype=np.float64)
        next_outlet = np.asarray(telemetry.target_next_outlet_quality, dtype=np.float64)
        releases = classify_applied_releases(host.receipts.applied_contact, host.actions)
        head_model_unavailable = releases.head_candidate & (
            np.asarray(host.receipts.applied_contact.actor) == carrier
        )
    else:
        frame_count = int(intents.shape[0])
        carrier_team = np.full(frame_count, -1, dtype=np.int32)
        overload = np.zeros(frame_count, dtype=bool)
        selected_intent = np.zeros(frame_count, dtype=np.int32)
        direction = np.full(frame_count, -1, dtype=np.int32)
        completion = np.zeros(frame_count, dtype=np.float64)
        score_gap = np.zeros(frame_count, dtype=np.float64)
        next_outlet = np.zeros(frame_count, dtype=np.float64)
        head_model_unavailable = np.zeros(frame_count, dtype=bool)
    policy_summary = []
    overload_pass_team = {
        int(frame): int(carrier_team[frame])
        for frame in np.flatnonzero(overload & (selected_intent == INTENT_PASS))
    }
    overload_pass_attempts = np.zeros(2, dtype=np.int64)
    overload_pass_completions = np.zeros(2, dtype=np.int64)
    for row in causal_summary.pass_map:
        pass_team = overload_pass_team.get(int(row["source_frame"]))
        if pass_team is None:
            continue
        overload_pass_attempts[pass_team] += 1
        overload_pass_completions[pass_team] += int(bool(row["completed"]))
    for team_index in range(2):
        selected = overload & (carrier_team == team_index)
        count = int(np.count_nonzero(selected))
        completion_selected = selected & ~head_model_unavailable
        completion_count = int(np.count_nonzero(completion_selected))
        head_count = int(np.count_nonzero(selected & head_model_unavailable))
        control = selected & (selected_intent == INTENT_CONTROL)
        policy_summary.append(
            {
                "overload_frames": count,
                "selected_intents": tuple(
                    int(value)
                    for value in np.bincount(
                        selected_intent[selected], minlength=ACTION_INTENT_COUNT
                    )[:ACTION_INTENT_COUNT]
                ),
                "target_directions": tuple(
                    int(np.count_nonzero(selected & (direction == value)))
                    for value in (-1, 0, 1, 2)
                ),
                "mean_target_completion": (
                    float(np.mean(completion[completion_selected])) if completion_count
                    else None if head_count else 0.0
                ),
                "target_completion_observed_frames": completion_count,
                "target_completion_head_unavailable_frames": head_count,
                "target_completion_coverage": completion_count / count if count else None,
                "mean_target_score_gap": (
                    float(np.mean(score_gap[completion_selected])) if completion_count
                    else None if head_count else 0.0
                ),
                "mean_target_next_outlet_quality": (
                    float(np.mean(next_outlet[selected])) if count else 0.0
                ),
                "control_possession_retained": int(
                    np.count_nonzero(control & (owner_team == team_index))
                ),
                "pass_attempts": int(overload_pass_attempts[team_index]),
                "pass_completions": int(overload_pass_completions[team_index]),
            }
        )
    return MatchReport(
        schema=MATCH_REPORT_SCHEMA,
        frames=int(intents.shape[0]),
        control_dt_s=float(timebase.control_dt),
        final_score=(int(score[0]), int(score[1])),
        event_counts=counts,
        foul_severity_counts=tuple(int(value) for value in foul_severity_counts),
        foul_tactical_counts=tuple(int(value) for value in foul_tactical_counts),
        challenge_contact_attempts=challenge_contact_attempts,
        challenge_fouls=challenge_fouls,
        intent_counts=tuple(int(value) for value in intent_counts),
        team_intent_counts=team_intent_counts,
        goal_last_contacts=extract_goal_last_contacts(host.states, host.receipts, team),
        possession_frames=possession_frames,
        possession_share=(
            possession_frames[0] / controlled_frames,
            possession_frames[1] / controlled_frames,
        ),
        team_distance_m=team_distance,
        player_distance_m=player_distance,
        ball_distance_m=float(np.sum(ball_step_distance)),
        mean_ball_speed_mps=float(np.mean(ball_speed)) if ball_speed.size else 0.0,
        maximum_ball_speed_mps=float(np.max(ball_speed)) if ball_speed.size else 0.0,
        live_ball_frames=int(np.count_nonzero(ball_live)),
        pass_attempts=causal_summary.pass_attempts,
        pass_completions=causal_summary.pass_completions,
        shoot_releases=causal_summary.shoot_releases,
        shot_outcomes=causal_summary.shot_outcomes,
        pass_map=causal_summary.pass_map,
        deliberate_release_summary=causal_summary.release_observation,
        shot_map=causal_summary.shot_map,
        spatial=spatial_accumulator.finish(),
        tactical_summary=build_tactical_summary(
            causal_summary,
            pass_window_frames=timebase.control_steps_for(15.0 * 60.0, minimum=1),
            goal_line_x=float(field_half_extent[0]),
            live_ball_minutes=float(np.count_nonzero(ball_live)) * float(timebase.control_dt) / 60.0,
        ),
        team_shape_summary=team_shape.finish(),
        policy_decision_summary=(policy_summary[0], policy_summary[1]),
        player_generations=generations,
        player_action_credit=build_player_action_credit(
            causal_summary,
            generations,
            distance_by_generation_m={
                (slot, 0): distance for slot, distance in enumerate(player_distance)
            },
            spatial_by_generation=player_spatial.finish(),
            goal_line_x=float(field_half_extent[0]),
        ),
        roster_summary=initial_roster_summary(roster),
        possession_timeline=timeline.finish(),
        reference_context={"dfl_attacking_third_pass": DFL_ATTACKING_THIRD_REFERENCE},
    )


def write_match_report(
    report: MatchReport,
    directory: str | Path,
    *,
    title: str = "Match evaluation report",
) -> tuple[Path, Path]:
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "match-report.json"
    md_path = output / "match-report.md"
    json_path.write_text(
        strict_json_dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ball_density_seconds = sum(sum(row) for row in report.spatial.ball_density_seconds)
    team_occupancy_seconds = tuple(
        sum(sum(row) for row in matrix)
        for matrix in report.spatial.team_occupancy_seconds
    )
    lines = [
        f"# {title.strip()}",
        "",
        f"- Frames: {report.frames}",
        (
            "- Control frame / captured time: "
            f"{report.control_dt_s:.6g} s / "
            f"{report.frames * report.control_dt_s:.2f} s"
        ),
        f"- Final score: {report.final_score[0]}–{report.final_score[1]}",
        (
            "- Team possession frame: "
            f"{report.possession_frames[0]} / {report.possession_frames[1]}"
        ),
        (
            "- Team possession share: "
            f"{report.possession_share[0]:.3f} / {report.possession_share[1]:.3f}"
        ),
        (
            "- Intent count (MOVE, CONTROL, PASS, SHOOT, CHALLENGE): "
            f"{report.intent_counts}"
        ),
        (
            "- Team intent count: "
            f"{report.team_intent_counts[0]} / {report.team_intent_counts[1]}"
        ),
        (
            "- Team distance across retained frames: "
            f"{report.team_distance_m[0]:.2f} m / "
            f"{report.team_distance_m[1]:.2f} m"
        ),
        f"- Ball distance across retained frames: {report.ball_distance_m:.2f} m",
        (
            "- Mean/maximum ball speed: "
            f"{report.mean_ball_speed_mps:.2f} / "
            f"{report.maximum_ball_speed_mps:.2f} m/s"
        ),
        f"- Live-ball frame: {report.live_ball_frames}",
        f"- Live-ball density within the pitch: {ball_density_seconds:.2f} s",
        (
            "- Team active-player occupancy: "
            f"{team_occupancy_seconds[0]:.2f} / "
            f"{team_occupancy_seconds[1]:.2f} player-s"
        ),
        (
            "- Live-ball time outside the pitch: "
            f"{report.spatial.ball_out_of_pitch_seconds:.2f} s"
        ),
        (
            "- Pass attempts/completions: "
            f"{report.pass_attempts} / {report.pass_completions}"
        ),
        f"- Raw SHOOT releases (including clearances): {report.shoot_releases}",
        f"- Raw SHOOT release outcomes: {report.shot_outcomes}",
        (
            "- Shot attempts/on-target: "
            f"{report.tactical_summary[0]['shot_attempts']} / "
            f"{report.tactical_summary[0]['shots_on_target']} vs "
            f"{report.tactical_summary[1]['shot_attempts']} / "
            f"{report.tactical_summary[1]['shots_on_target']}"
        ),
        (
            "- SHOOT releases near goal (shot-rate denominator): "
            f"{report.tactical_summary[0]['shot_releases_near_goal']} vs "
            f"{report.tactical_summary[1]['shot_releases_near_goal']}"
        ),
        (
            "- Open-play pass attempts/completions: "
            f"{report.tactical_summary[0]['open_play_pass_attempts']} / "
            f"{report.tactical_summary[0]['open_play_pass_completions']} vs "
            f"{report.tactical_summary[1]['open_play_pass_attempts']} / "
            f"{report.tactical_summary[1]['open_play_pass_completions']}"
        ),
        f"- Team tactical pass/shot summary: {report.tactical_summary}",
        f"- Policy overload decision summary: {report.policy_decision_summary}",
        "",
        "## Events",
        "",
    ]
    lines.extend(f"- {name}: {count}" for name, count in report.event_counts.items())
    lines.extend(
        (
            "",
            "## Foul categories",
            "",
            (
                "- Severity (none, careless, reckless, excessive force): "
                f"{report.foul_severity_counts}"
            ),
            (f"- Tactical (none, SPA, DOGSO): {report.foul_tactical_counts}"),
            (
                "- Challenge fouls / actual challenge contact attempts: "
                f"{report.challenge_fouls} / {report.challenge_contact_attempts} "
                f"({report.challenge_fouls / report.challenge_contact_attempts:.2%})"
                if report.challenge_contact_attempts
                else "- Challenge fouls / actual challenge contact attempts: 0 / 0 (N/A)"
            ),
        )
    )
    lines.extend(
        (
            "",
            "## Player possession, position, and spacing",
            "",
            (
                "| Team | Player | Gen | Active (s) | Possession (s) | "
                "Mean absolute position (m) | Mean teammate distance (m) | Nearest teammate (m) |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in report.player_action_credit:
        position = row.get("mean_absolute_position_m")
        position_text = (
            f"({float(position[0]):.2f}, {float(position[1]):.2f})"
            if position is not None
            else "—"
        )
        teammate = row.get("mean_teammate_distance_m")
        nearest = row.get("mean_nearest_teammate_distance_m")
        teammate_text = f"{float(teammate):.2f}" if teammate is not None else "—"
        nearest_text = f"{float(nearest):.2f}" if nearest is not None else "—"
        lines.append(
            "| "
            f"{row['team']} | {row['player_id']} | {row['generation']} | "
            f"{float(row.get('active_seconds', 0.0)):.2f} | "
            f"{float(row.get('controlled_possession_seconds', 0.0)):.2f} | "
            f"{position_text} | {teammate_text} | {nearest_text} |"
        )
    lines.extend(("", "### Mean player position by possession phase", ""))
    lines.append("| Team | Player | Gen | Team in possession (m) | Opponent in possession (m) |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in report.player_action_credit:
        in_position = row.get("mean_in_possession_position_m")
        out_position = row.get("mean_out_of_possession_position_m")

        def position_text(value) -> str:
            return (
                f"({float(value[0]):.2f}, {float(value[1]):.2f})"
                if value is not None
                else "—"
            )

        lines.append(
            f"| {row['team']} | {row['player_id']} | {row['generation']} | "
            f"{position_text(in_position)} | {position_text(out_position)} |"
        )
    lines.extend(("", "## Team shape", ""))
    lines.append(
        "| Team | Mean centroid (m) | Mean width (m) | Mean depth (m) | Mean pairwise distance (m) | "
        "Width/depth in possession (m) | Width/depth out of possession (m) |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for team, shape in enumerate(report.team_shape_summary):
        optional = lambda value: "—" if value is None else f"{float(value):.2f}"
        centroid = shape["mean_centroid_m"]
        centroid_text = (
            f"({float(centroid[0]):.2f}, {float(centroid[1]):.2f})"
            if centroid is not None
            else "—"
        )
        lines.append(
            f"| {team} | {centroid_text} | "
            f"{float(shape['mean_width_m']):.2f} | "
            f"{float(shape['mean_depth_m']):.2f} | "
            f"{float(shape['mean_pairwise_distance_m']):.2f} | "
            f"{optional(shape['in_possession_mean_width_m'])} / "
            f"{optional(shape['in_possession_mean_depth_m'])} | "
            f"{optional(shape['out_of_possession_mean_width_m'])} / "
            f"{optional(shape['out_of_possession_mean_depth_m'])} |"
        )
    lines.extend(("", "### Team spatial structure", ""))
    lines.append(
        "| Team | Stretch index (m) | Maximum longitudinal gap (m) | "
        "Pairwise distance P10/P50/P90 (m) | Opponent centroid distance (m) | Ball–centroid distance (m) |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for team, shape in enumerate(report.team_shape_summary):
        optional = lambda value: "—" if value is None else f"{float(value):.2f}"
        lines.append(
            f"| {team} | {optional(shape.get('mean_stretch_index_m'))} | "
            f"{optional(shape.get('mean_longitudinal_max_gap_m'))} | "
            f"{optional(shape.get('pairwise_distance_p10_m'))} / "
            f"{optional(shape.get('pairwise_distance_p50_m'))} / "
            f"{optional(shape.get('pairwise_distance_p90_m'))} | "
            f"{optional(shape.get('mean_opponent_centroid_distance_m'))} | "
            f"{optional(shape.get('mean_ball_centroid_distance_m'))} |"
        )
    roster = report.roster_summary
    lines.extend((
        "", "## Starting formations and realized player attributes", "",
        f"Source: `{roster['source']}`. Formations are the initial setup; attributes belong to the listed player generation.",
        roster.get("provenance_note", "These values are realized model inputs, not observed player ratings."),
        "", "| Team | Starting formation | Formation ID |", "|---:|---|---:|",
    ))
    for formation in roster["formations"]:
        lines.append(
            f"| {formation['team']} | {formation['formation']} | {formation['formation_id']} |"
        )
    lines.extend((
        "", "| Team | Player | Slot | Gen | Start frame | Role | Anchor, reference m | "
        "Max speed (m/s) | Height (m) | Reach height (m) | Ball control | Endurance factor |",
        "|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ))
    for player in roster["players"]:
        anchor = player["formation_anchor_reference_m"]
        lines.append(
            f"| {player['team']} | {player['player_id']} | {player['slot']} | "
            f"{player['generation']} | {player['start_frame']} | {player['role']} | "
            f"({anchor[0]:.1f}, {anchor[1]:.1f}) | "
            f"{player['max_speed_mps']:.2f} | {player['height_m']:.2f} | "
            f"{player['reach_height_m']:.2f} | {player['ball_control']:.3f} | "
            f"{player['endurance_factor']:.3f} |"
        )
    lines.extend(("", "## Last-contact goal provenance", "",
                  f"Goal-boundary last-contact classification, independent of release origin: {report.last_contact_goal_provenance}"))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
