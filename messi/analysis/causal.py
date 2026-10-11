"""Chunk 경계를 넘는 pass·shot causal flight 분석."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from messi.core.action import INTENT_PASS, INTENT_SHOOT
from messi.core.constants import BALL_EVENT_GOAL
from messi.core.contact import (
    LAW11_DEFLECTION_NO_RESET,
    LAW11_DELIBERATE_SAVE_NO_RESET,
    LAW11_NONE,
    OUTCOME_CATCH,
)
from messi.core.contact_response import ContactResponsePhysics
from messi.core.timebase import DEFAULT_TIMEBASE, Timebase
from messi.analysis.release import classify_applied_releases

SHOT_ATTEMPT_MAX_DISTANCE_M = 40.0
GOAL_MOUTH_HALF_WIDTH_M = 3.66
GOAL_MOUTH_HEIGHT_M = 2.44
GRAVITY_MPS2 = 9.81
SHOT_OUTCOMES = (
    "goal",
    "own_goal",
    "saved",
    "blocked",
    "deflected",
    "woodwork",
    "off_target",
    "stopped",
    "unresolved",
)


def build_shot_on_target_proxy(releases, attempt_outcomes, release_outcomes):
    """동일 release 모집단에서 실제 goal과 조준 확인/미확인 GK 접촉을 분리한다.

    Lower/upper는 saved 분류의 모호성만 확장한 proxy이며 실제 provider SOT의
    통계적 신뢰구간 또는 수학적 상하한이 아니다.
    """

    counts = (releases, *attempt_outcomes.values(), *release_outcomes.values())
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 0
        for value in counts
    ):
        raise ValueError("shot on-target proxy count는 non-negative 정수여야 한다")
    if set(attempt_outcomes) != set(SHOT_OUTCOMES) or set(release_outcomes) != set(
        SHOT_OUTCOMES
    ):
        raise ValueError("shot on-target proxy outcome key가 일치해야 한다")
    if sum(release_outcomes.values()) != releases or any(
        attempt_outcomes[name] > release_outcomes[name] for name in SHOT_OUTCOMES
    ):
        raise ValueError("shot on-target proxy outcome 모집단이 일치하지 않는다")
    releases = int(releases)
    goals = int(release_outcomes["goal"])
    saves = int(release_outcomes["saved"])
    confirmed = int(attempt_outcomes["saved"])
    unresolved = int(release_outcomes["unresolved"])
    if not 0 <= confirmed <= saves or min(releases, goals, saves, unresolved) < 0:
        raise ValueError("shot on-target proxy count가 유효하지 않다")
    lower = goals + confirmed
    upper = goals + saves
    if upper + unresolved > releases:
        raise ValueError("shot on-target proxy count가 release 모집단을 초과한다")
    return {
        "basis": "near_goal_forward_release_actual_goal_or_geometry_confirmed_save",
        "comparability": "provider_proxy_not_equivalent",
        "release_count": releases,
        "causal_goal_count": goals,
        "geometry_confirmed_save_count": confirmed,
        "geometry_unconfirmed_save_count": saves - confirmed,
        "unresolved_release_count": unresolved,
        "lower_count": lower,
        "upper_count": upper,
        "lower_share": lower / releases if releases else None,
        "upper_share": upper / releases if releases else None,
    }


def build_shot_release_geometry(
    releases,
    *,
    distance_known_count=0,
    distance_sum_m=0.0,
    box_known_count=0,
    in_box_count=0,
):
    """Near-goal release의 거리·box를 coverage와 함께 같은 모집단으로 반환한다."""

    counts = (releases, distance_known_count, box_known_count, in_box_count)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 0
        for value in counts
    ):
        raise ValueError("shot release geometry count는 non-negative 정수여야 한다")
    if (
        distance_known_count > releases
        or not in_box_count <= box_known_count <= releases
    ):
        raise ValueError("shot release geometry의 분자·분모가 일치하지 않는다")
    total_distance = float(distance_sum_m)
    if (
        not np.isfinite(total_distance)
        or total_distance < 0
        or (not distance_known_count and total_distance != 0.0)
    ):
        raise ValueError("shot release 거리 합계가 유효하지 않다")
    return {
        "basis": "near_goal_forward_release_geometry_not_DFL_provider_shots",
        "release_count": int(releases),
        "distance_known_count": int(distance_known_count),
        "distance_unknown_count": int(releases - distance_known_count),
        "distance_sum_m": total_distance,
        "mean_distance_m": total_distance / distance_known_count
        if distance_known_count
        else None,
        "distance_coverage": distance_known_count / releases if releases else None,
        "box_known_count": int(box_known_count),
        "box_unknown_count": int(releases - box_known_count),
        "in_box_count": int(in_box_count),
        "outside_box_count": int(box_known_count - in_box_count),
        "in_box_share_of_known": in_box_count / box_known_count
        if box_known_count
        else None,
        "box_coverage": box_known_count / releases if releases else None,
    }


def validate_shot_release_geometry(value, releases):
    """Legacy 미측정은 unknown으로 남기고 제공된 파생 값의 불일치는 거부한다."""

    if value is None:
        return build_shot_release_geometry(releases)
    if not isinstance(value, dict):
        raise TypeError("shot_release_geometry는 mapping이어야 한다")
    expected = build_shot_release_geometry(
        releases,
        distance_known_count=value.get("distance_known_count"),
        distance_sum_m=value.get("distance_sum_m"),
        box_known_count=value.get("box_known_count"),
        in_box_count=value.get("in_box_count"),
    )
    if set(value) != set(expected):
        raise ValueError("shot_release_geometry key contract가 일치해야 한다")
    for key, expected_value in expected.items():
        observed = value[key]
        if isinstance(expected_value, float):
            valid = (
                not isinstance(observed, bool)
                and isinstance(observed, (int, float))
                and np.isfinite(observed)
                and np.isclose(observed, expected_value, rtol=1.0e-9, atol=1.0e-9)
            )
        else:
            valid = (
                type(observed) is type(expected_value) and observed == expected_value
            )
        if not valid:
            raise ValueError(f"shot_release_geometry의 {key}가 원시 count와 다르다")
    return expected


def merge_shot_release_geometry(rows):
    values = [
        validate_shot_release_geometry(value, releases) for releases, value in rows
    ]
    return build_shot_release_geometry(
        sum(value["release_count"] for value in values),
        distance_known_count=sum(value["distance_known_count"] for value in values),
        distance_sum_m=sum(value["distance_sum_m"] for value in values),
        box_known_count=sum(value["box_known_count"] for value in values),
        in_box_count=sum(value["in_box_count"] for value in values),
    )


def _shot_geometry(row, goal_line_x: float):
    """Attack-normalized release에서 goal까지의 남은 거리와 방향을 꺼낸다."""

    source = row.get("attack_normalized_source_position")
    direction = row.get("attack_normalized_direction")
    if source is None or direction is None:
        return None
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(direction)):
        return None
    goal_line_x = float(goal_line_x)
    if not np.isfinite(goal_line_x) or goal_line_x <= 0.0:
        raise ValueError("goal_line_x는 유한한 양수여야 한다")
    remaining = goal_line_x - float(source[0])
    return (
        remaining,
        float(np.hypot(remaining, float(source[1]))),
        float(source[1]),
        float(direction[0]),
        float(direction[1]),
    )


def is_shot_attempt(row, *, goal_line_x: float = 52.5) -> bool:
    """Release 방향의 정확도와 무관하게 shot attempt인지 판별한다.

    SHOOT intent는 clearance와 goalkeeper distribution도 함께 쓰므로 goal 근처
    forward release만 attempt로 센다. 조준이 mouth를 벗어난 release도 attempt이며
    on-target 여부는 :func:`is_goal_directed_shot`이 따로 판정한다.
    """

    geometry = _shot_geometry(row, goal_line_x)
    if geometry is None:
        return False
    remaining, goal_distance, _, dx, _ = geometry
    if dx <= 1.0e-6 or remaining < 0.0:
        return False
    return goal_distance <= SHOT_ATTEMPT_MAX_DISTANCE_M


def is_goal_directed_shot(row, *, goal_line_x: float = 52.5) -> bool:
    """Release가 goal mouth를 폭과 crossbar 높이 모두에서 통과하는지 판별한다.

    ``release_velocity``(contact response가 공에 실제로 준 3D 속도)가 있으면 진공
    탄도로 goal line 도달 시각의 y와 z를 본다. 높이는 **상한만** 본다. Goal line 전에
    착지하는 shot은 같은 수평 방향으로 굴러 들어가므로 off target이 아니고, 판정에서
    빠져야 하는 것은 crossbar를 넘는 shot뿐이다. 항력과 Magnus, rolling 감속은
    무시하므로 굴러서 goal 앞에 멈추는 shot도 on target으로 센다. 속도가 없는 legacy
    row는 요청 방향의 2D ray만 보는 이전 판정으로 되돌아간다.
    """

    geometry = _shot_geometry(row, goal_line_x)
    if geometry is None:
        return False
    remaining, goal_distance, source_y, dx, dy = geometry
    if dx <= 1.0e-6 or remaining < 0.0 or goal_distance > SHOT_ATTEMPT_MAX_DISTANCE_M:
        return False
    ray_on_target = abs(source_y + remaining * dy / dx) <= GOAL_MOUTH_HALF_WIDTH_M
    velocity = row.get("attack_normalized_release_velocity")
    source = row.get("attack_normalized_source_position")
    if velocity is None or source is None or len(source) < 3:
        return ray_on_target
    vx, vy, vz = (float(value) for value in velocity[:3])
    if not np.all(np.isfinite((vx, vy, vz))) or vx <= 1.0e-6:
        return False
    flight = remaining / vx
    if not np.isfinite(flight) or flight <= 0.0:
        return False
    mouth_y = float(source[1]) + vy * flight
    mouth_z = float(source[2]) + vz * flight - 0.5 * GRAVITY_MPS2 * flight * flight
    return abs(mouth_y) <= GOAL_MOUTH_HALF_WIDTH_M and mouth_z <= GOAL_MOUTH_HEIGHT_M


@dataclass(frozen=True, slots=True)
class CausalSummary:
    pass_attempts: tuple[int, int]
    pass_completions: tuple[int, int]
    shoot_releases: tuple[int, int]
    shot_outcomes: dict[str, int]
    pass_map: tuple[dict[str, object], ...]
    shot_map: tuple[dict[str, object], ...]
    # pass_map과 같은 순서의 regulation match tick. 빈 tuple은 이 field가 없던
    # legacy/in-memory summary이며 report가 source_frame으로 fallback한다.
    pass_source_match_ticks: tuple[int, ...] = ()
    release_observation: dict = field(default_factory=dict)


class CausalEventResolver:
    """Dense trajectory 대신 하나의 pending ball flight만 유지한다."""

    def __init__(
        self,
        team_id,
        *,
        is_goalkeeper=None,
        timebase: Timebase = DEFAULT_TIMEBASE,
        contact_response: ContactResponsePhysics = ContactResponsePhysics(),
    ) -> None:
        self._team = np.asarray(team_id, dtype=np.int32)
        if is_goalkeeper is None:
            self._goalkeeper = np.zeros(self._team.shape, dtype=bool)
        else:
            self._goalkeeper = np.asarray(is_goalkeeper, dtype=bool)
            if self._goalkeeper.shape != self._team.shape:
                raise ValueError("is_goalkeeper는 team_id와 같은 shape여야 한다")
        self._physics_substeps_per_frame = int(timebase.decimation)
        self._same_actor_lock_substeps = max(
            1,
            round(contact_response.deliberate_kick_contact_lock_s / timebase.dt_phys),
        )
        self._pending: dict[str, object] | None = None
        self._passes: list[dict[str, object]] = []
        self._shots: list[dict[str, object]] = []
        self._head_power_observation = np.zeros((2, 3), dtype=np.int64)
        self._pass_source_clock: list[tuple[int, int]] = []
        self._first_half_regulation_end_tick: int | None = None
        self._previous_first_half_wall_end_tick: int | None = None

    def seed_period(self, first_half_wall_end_tick) -> None:
        """첫 post-step frame 전에 period marker를 host carry에 넣는다."""

        if self._previous_first_half_wall_end_tick is not None:
            raise RuntimeError("period marker는 첫 chunk 전에 한 번만 seed할 수 있다")
        self._previous_first_half_wall_end_tick = int(
            np.asarray(first_half_wall_end_tick)
        )

    def _actor_team(self, actor: int) -> int:
        return int(self._team[actor]) if 0 <= actor < self._team.size else -1

    def _actor_is_goalkeeper(self, actor: int) -> bool:
        if not 0 <= actor < self._goalkeeper.size:
            return False
        return bool(self._goalkeeper[actor])

    def _same_actor_contact_locked(
        self,
        *,
        frame: int,
        physics_substep: int,
        time_fraction: float,
        actor: int,
    ) -> bool:
        if self._pending is None or actor != int(self._pending["actor"]):
            return False
        source_time = (
            int(self._pending["source_frame"]) * self._physics_substeps_per_frame
            + int(self._pending["physics_substep"])
            + float(self._pending["time_fraction"])
        )
        terminal_time = (
            int(frame) * self._physics_substeps_per_frame
            + int(physics_substep)
            + float(time_fraction)
        )
        return terminal_time - source_time < self._same_actor_lock_substeps

    def _resolve(
        self,
        *,
        frame: int,
        kind: str,
        actor: int = -1,
        position=None,
        scoring_team: int = -1,
    ) -> None:
        if self._pending is None:
            return
        source = self._pending
        source_team = int(source["team"])
        intent = int(source["intent"])
        terminal_team = self._actor_team(actor)
        row = dict(source)
        row.update(
            terminal_frame=int(frame),
            elapsed_frames=int(frame) - int(source["source_frame"]),
            terminal_kind=kind,
            terminal_actor=int(actor),
            terminal_team=terminal_team,
            terminal_position=(
                None
                if position is None
                else tuple(float(value) for value in np.asarray(position))
            ),
        )
        if position is not None:
            terminal_position = np.asarray(position)
            attack_direction = float(source["attack_direction"])
            row["attack_normalized_terminal_position"] = (
                float(terminal_position[0] * attack_direction),
                float(terminal_position[1]),
                float(terminal_position[2]),
            )
        else:
            row["attack_normalized_terminal_position"] = None
        if intent == INTENT_PASS:
            # actor가 source와 같으면 호출자가 kind를 same_actor_retouch로 넘긴다.
            completed = kind == "contact" and terminal_team == source_team
            row["completed"] = completed
            if kind == "goal":
                row["outcome"] = (
                    "goal" if scoring_team == source_team else "own_goal"
                )
            else:
                row["outcome"] = "completed" if completed else kind
            self._pass_source_clock.append(
                (
                    int(row.pop("_source_regulation_tick")),
                    int(row.pop("_source_period")),
                )
            )
            self._passes.append(row)
        else:
            row.pop("_source_regulation_tick")
            row.pop("_source_period")
            if kind == "goal":
                outcome = "goal" if scoring_team == source_team else "own_goal"
            elif bool(source.get("woodwork")):
                outcome = "woodwork"
            elif (
                kind in ("contact", "same_actor_retouch", "boundary")
                and source.get("first_opponent_deflection") is not None
            ):
                defender = int(source["first_opponent_deflection"]["actor"])
                outcome = "saved" if self._actor_is_goalkeeper(defender) else "blocked"
            elif kind in ("contact", "same_actor_retouch"):
                if terminal_team < 0:
                    outcome = "off_target"
                elif terminal_team == source_team:
                    outcome = "deflected"
                elif self._actor_is_goalkeeper(actor):
                    outcome = "saved"
                else:
                    outcome = "blocked"
            elif kind == "boundary":
                outcome = "off_target"
            elif kind == "capture_end":
                outcome = "unresolved"
            else:
                outcome = "stopped"
            row["outcome"] = outcome
            self._shots.append(row)
        self._pending = None

    def _remember_opponent_deflection(self, *, frame: int, actor: int, kind: str):
        """Flight를 계속 추적하되 첫 상대 저지를 최종 아웃과 구분해 보존한다."""

        if self._pending is None or "first_opponent_deflection" in self._pending:
            return
        self._pending["first_opponent_deflection"] = {
            "frame": int(frame),
            "actor": int(actor),
            "team": self._actor_team(actor),
            "kind": kind,
        }

    def add(self, states, actions, receipts, *, frame_offset: int) -> None:
        frame_count = int(np.asarray(states.control_tick).shape[0])
        applied = receipts.applied_contact
        releases = classify_applied_releases(applied, actions)
        source_teams = np.asarray(applied.team)
        for team in range(2):
            selected = source_teams == team
            self._head_power_observation[team] += np.asarray([
                np.count_nonzero(selected & releases.head_candidate),
                np.count_nonzero(selected & releases.head_power_known),
                np.count_nonzero(selected & releases.head_zero_power),
            ])
        passive = receipts.passive_contact
        boundary = receipts.boundary
        woodwork = receipts.woodwork
        foul = getattr(receipts, "foul", None)
        restart_retouch = getattr(receipts, "restart_retouch", None)
        offside = getattr(receipts, "offside", None)
        chronology = getattr(receipts, "event_chronology", None)
        first_half_wall_end_tick = np.asarray(
            getattr(
                states,
                "first_half_wall_end_tick",
                np.full(frame_count, -1, dtype=np.int32),
            ),
            dtype=np.int64,
        )
        regulation_elapsed_tick = np.maximum(
            np.asarray(states.control_tick, dtype=np.int64)
            - np.asarray(
                getattr(
                    states,
                    "dead_ball_control_ticks",
                    np.zeros(frame_count, dtype=np.int32),
                ),
                dtype=np.int64,
            )
            - np.asarray(
                getattr(
                    states,
                    "first_half_live_extension_ticks",
                    np.zeros(frame_count, dtype=np.int32),
                ),
                dtype=np.int64,
            ),
            0,
        )
        applied_occurred = np.asarray(applied.occurred, dtype=bool)
        applied_substep = np.asarray(applied.physics_substep)
        applied_time = np.asarray(applied.time_fraction)
        applied_actor = np.asarray(applied.actor)
        applied_position = np.asarray(applied.position)
        applied_intent = np.asarray(applied.intent)
        applied_kick = np.asarray(applied.kick_applied, dtype=bool)
        applied_law11 = np.asarray(
            getattr(
                applied,
                "law11_effect",
                np.full(frame_count, LAW11_NONE, dtype=np.int32),
            )
        )
        applied_direction = np.asarray(applied.direction)
        applied_attack_direction = np.asarray(applied.attack_direction)
        applied_team = np.asarray(applied.team)
        applied_mechanism = np.asarray(applied.mechanism)
        applied_outcome = np.asarray(applied.outcome)
        applied_launch = np.asarray(applied.launch)
        applied_release_velocity = np.asarray(
            getattr(
                applied,
                "release_velocity",
                np.zeros((frame_count, 3), dtype=np.float32),
            )
        )
        applied_spin = np.asarray(applied.spin)
        applied_restart = np.asarray(applied.restart_kind)
        applied_possession_changed = np.asarray(applied.possession_changed, dtype=bool)
        passive_occurred = np.asarray(passive.occurred, dtype=bool)
        passive_time = np.asarray(passive.time_fraction)
        passive_actor = np.asarray(passive.actor)
        passive_position = np.asarray(passive.position)
        woodwork_occurred = np.asarray(woodwork.occurred, dtype=bool)
        woodwork_time = np.asarray(woodwork.time_fraction)
        boundary_occurred = np.asarray(boundary.occurred, dtype=bool)
        boundary_time = np.asarray(boundary.time_fraction)
        boundary_kind = np.asarray(boundary.kind)
        boundary_position = np.asarray(boundary.position)
        boundary_scoring_team = np.asarray(boundary.scoring_team)
        false_frames = np.zeros(frame_count, dtype=bool)
        zero_time = np.zeros(frame_count, dtype=np.float32)
        zero_position = np.zeros((frame_count, 3), dtype=np.float32)
        foul_occurred = (
            np.asarray(foul.occurred, dtype=bool) if foul is not None else false_frames
        )
        foul_time = np.asarray(foul.time_fraction) if foul is not None else zero_time
        foul_position = np.asarray(foul.position) if foul is not None else zero_position
        retouch_occurred = (
            np.asarray(restart_retouch.occurred, dtype=bool)
            if restart_retouch is not None
            else false_frames
        )
        retouch_time = (
            np.asarray(restart_retouch.time_fraction)
            if restart_retouch is not None
            else zero_time
        )
        retouch_position = (
            np.asarray(restart_retouch.contact_position)
            if restart_retouch is not None
            else zero_position
        )
        offside_occurred = (
            np.asarray(offside.occurred, dtype=bool)
            if offside is not None
            else false_frames
        )
        offside_time = (
            np.asarray(offside.time_fraction) if offside is not None else zero_time
        )
        offside_position = (
            np.asarray(offside.position) if offside is not None else zero_position
        )
        if chronology is None:
            chronology_arrays = None
        else:
            default_substep = np.zeros(frame_count, dtype=np.int32)
            rule_stoppage_substep = np.asarray(
                getattr(chronology, "rule_stoppage_substep", default_substep)
            )
            chronology_arrays = {
                "foul_substep": rule_stoppage_substep,
                "restart_retouch_substep": rule_stoppage_substep,
                "offside_substep": rule_stoppage_substep,
                "woodwork_substep": np.asarray(
                    getattr(chronology, "woodwork_substep", default_substep)
                ),
                "passive_contact_substep": np.asarray(
                    getattr(chronology, "passive_contact_substep", default_substep)
                ),
                "boundary_substep": np.asarray(
                    getattr(chronology, "boundary_substep", default_substep)
                ),
            }
        for local_frame in range(frame_count):
            frame = frame_offset + local_frame
            current_half_end = int(first_half_wall_end_tick[local_frame])
            halftime = (
                self._previous_first_half_wall_end_tick is not None
                and self._previous_first_half_wall_end_tick < 0
                and current_half_end >= 0
            )
            if halftime:
                self._first_half_regulation_end_tick = int(
                    regulation_elapsed_tick[local_frame]
                )
            events: list[tuple[tuple[int, float, int], str]] = []
            if bool(applied_occurred[local_frame]):
                events.append(
                    (
                        (
                            int(applied_substep[local_frame]),
                            float(applied_time[local_frame]),
                            0,
                        ),
                        "applied",
                    )
                )
            for order, name, occurred, time_fraction, substep_name in (
                (-3, "foul", foul_occurred, foul_time, "foul_substep"),
                (
                    -2,
                    "restart_retouch",
                    retouch_occurred,
                    retouch_time,
                    "restart_retouch_substep",
                ),
                (-1, "offside", offside_occurred, offside_time, "offside_substep"),
            ):
                if bool(occurred[local_frame]):
                    substep = (
                        int(chronology_arrays[substep_name][local_frame])
                        if chronology_arrays is not None
                        else 0
                    )
                    events.append(
                        ((substep, float(time_fraction[local_frame]), order), name)
                    )
            for order, name, occurred, time_fraction, substep_name in (
                (1, "woodwork", woodwork_occurred, woodwork_time, "woodwork_substep"),
                (
                    2,
                    "passive",
                    passive_occurred,
                    passive_time,
                    "passive_contact_substep",
                ),
                (3, "boundary", boundary_occurred, boundary_time, "boundary_substep"),
            ):
                if bool(occurred[local_frame]):
                    substep = (
                        int(chronology_arrays[substep_name][local_frame])
                        if chronology_arrays is not None
                        else 0
                    )
                    events.append(
                        (
                            (
                                substep,
                                float(time_fraction[local_frame]),
                                order,
                            ),
                            name,
                        )
                    )
            frame_stopped = False
            for _, event_name in sorted(events):
                if frame_stopped:
                    continue
                if event_name in ("foul", "restart_retouch", "offside"):
                    terminal_position = {
                        "foul": foul_position,
                        "restart_retouch": retouch_position,
                        "offside": offside_position,
                    }[event_name][local_frame]
                    self._resolve(
                        frame=frame,
                        kind=event_name,
                        position=terminal_position,
                    )
                    frame_stopped = True
                    continue
                if event_name == "applied":
                    actor = int(applied_actor[local_frame])
                    same_source_contact = False
                    continuing_deflection = False
                    if self._pending is not None:
                        same_source_contact = self._same_actor_contact_locked(
                            frame=frame,
                            physics_substep=int(applied_substep[local_frame]),
                            time_fraction=float(applied_time[local_frame]),
                            actor=actor,
                        )
                        continuing_deflection = (
                            not same_source_contact
                            and int(applied_team[local_frame])
                            != int(self._pending["team"])
                            and int(applied_law11[local_frame])
                            in (
                                LAW11_DEFLECTION_NO_RESET,
                                LAW11_DELIBERATE_SAVE_NO_RESET,
                            )
                            and int(applied_outcome[local_frame]) != OUTCOME_CATCH
                        )
                        if continuing_deflection:
                            self._remember_opponent_deflection(
                                frame=frame, actor=actor, kind="applied"
                            )
                        if not same_source_contact and not continuing_deflection:
                            self._resolve(
                                frame=frame,
                                kind=(
                                    "same_actor_retouch"
                                    if actor == int(self._pending["actor"])
                                    else "contact"
                                ),
                                actor=actor,
                                position=applied_position[local_frame],
                            )
                    intent = int(applied_intent[local_frame])
                    if (
                        not same_source_contact
                        and not continuing_deflection
                        and bool(releases.release[local_frame])
                        and intent
                        in (
                            INTENT_PASS,
                            INTENT_SHOOT,
                        )
                    ):
                        source_position = applied_position[local_frame]
                        source_direction = applied_direction[local_frame]
                        attack_direction = float(applied_attack_direction[local_frame])
                        self._pending = {
                            "source_frame": frame,
                            "_source_regulation_tick": int(
                                regulation_elapsed_tick[local_frame]
                            ),
                            "_source_period": 2 if current_half_end >= 0 else 1,
                            "actor": actor,
                            "team": int(applied_team[local_frame]),
                            "intent": intent,
                            "mechanism": int(applied_mechanism[local_frame]),
                            "release_subtype": "head" if releases.head_release[local_frame] else "foot_kick",
                            "release_power_observed": bool(np.isfinite(releases.actor_power[local_frame])),
                            "release_power": float(releases.actor_power[local_frame])
                            if np.isfinite(releases.actor_power[local_frame]) else None,
                            "outcome_at_source": int(applied_outcome[local_frame]),
                            "source_position": tuple(
                                float(value) for value in source_position
                            ),
                            "attack_normalized_source_position": (
                                float(source_position[0] * attack_direction),
                                float(source_position[1]),
                                float(source_position[2]),
                            ),
                            "attack_direction": attack_direction,
                            "physics_substep": int(applied_substep[local_frame]),
                            "time_fraction": float(applied_time[local_frame]),
                            "direction": tuple(
                                float(value) for value in source_direction
                            ),
                            "attack_normalized_direction": (
                                float(source_direction[0] * attack_direction),
                                float(source_direction[1]),
                            ),
                            "launch": float(applied_launch[local_frame]),
                            "release_velocity": tuple(
                                float(value)
                                for value in applied_release_velocity[local_frame]
                            ),
                            "attack_normalized_release_velocity": (
                                float(
                                    applied_release_velocity[local_frame][0]
                                    * attack_direction
                                ),
                                float(applied_release_velocity[local_frame][1]),
                                float(applied_release_velocity[local_frame][2]),
                            ),
                            "spin": tuple(
                                float(value) for value in applied_spin[local_frame]
                            ),
                            "restart_kind": int(applied_restart[local_frame]),
                            "possession_changed": bool(
                                applied_possession_changed[local_frame]
                            ),
                            "woodwork": False,
                        }
                elif event_name == "woodwork" and self._pending is not None:
                    self._pending["woodwork"] = True
                elif event_name == "passive" and self._pending is not None:
                    actor = int(passive_actor[local_frame])
                    passive_substep = (
                        int(chronology_arrays["passive_contact_substep"][local_frame])
                        if chronology_arrays is not None
                        else 0
                    )
                    same_source_contact = self._same_actor_contact_locked(
                        frame=frame,
                        physics_substep=passive_substep,
                        time_fraction=float(passive_time[local_frame]),
                        actor=actor,
                    )
                    if not same_source_contact:
                        # Passive body contact is a deflection, not controlled
                        # possession. Keep an opponent's original flight alive so
                        # a later goal/boundary receives the correct source.
                        if self._actor_team(actor) == int(self._pending["team"]):
                            self._resolve(
                                frame=frame,
                                kind=(
                                    "same_actor_retouch"
                                    if actor == int(self._pending["actor"])
                                    else "contact"
                                ),
                                actor=actor,
                                position=passive_position[local_frame],
                            )
                        elif self._actor_team(actor) >= 0:
                            self._remember_opponent_deflection(
                                frame=frame, actor=actor, kind="passive"
                            )
                elif event_name == "boundary" and self._pending is not None:
                    is_goal = int(boundary_kind[local_frame]) == BALL_EVENT_GOAL
                    self._resolve(
                        frame=frame,
                        kind="goal" if is_goal else "boundary",
                        position=boundary_position[local_frame],
                        scoring_team=int(boundary_scoring_team[local_frame]),
                    )
                    frame_stopped = True
            if halftime:
                self._resolve(frame=frame, kind="halftime")
            self._previous_first_half_wall_end_tick = current_half_end

    def finish(self, final_frame: int) -> CausalSummary:
        if self._pending is not None:
            self._resolve(frame=final_frame, kind="capture_end")
        pass_attempts = tuple(
            sum(int(row["team"]) == team for row in self._passes) for team in range(2)
        )
        pass_completions = tuple(
            sum(
                int(row["team"]) == team and bool(row["completed"])
                for row in self._passes
            )
            for team in range(2)
        )
        shoot_releases = tuple(
            sum(int(row["team"]) == team for row in self._shots) for team in range(2)
        )
        names = SHOT_OUTCOMES
        first_half_end = self._first_half_regulation_end_tick
        pass_source_match_ticks = tuple(
            (
                min(raw_tick, max(first_half_end - 1, 0))
                if period == 1 and first_half_end is not None
                else raw_tick
            )
            for raw_tick, period in self._pass_source_clock
        )
        return CausalSummary(
            pass_attempts=pass_attempts,
            pass_completions=pass_completions,
            shoot_releases=shoot_releases,
            shot_outcomes={
                name: sum(row["outcome"] == name for row in self._shots)
                for name in names
            },
            pass_map=tuple(self._passes),
            shot_map=tuple(self._shots),
            pass_source_match_ticks=pass_source_match_ticks,
            release_observation={
                "schema": "messi.deliberate-release-observation/1",
                "head_candidate_counts": tuple(int(x) for x in self._head_power_observation[:, 0]),
                "head_power_known_counts": tuple(int(x) for x in self._head_power_observation[:, 1]),
                "head_zero_power_counts": tuple(int(x) for x in self._head_power_observation[:, 2]),
                "head_power_unknown_counts": tuple(int(x) for x in self._head_power_observation[:, 0]-self._head_power_observation[:, 1]),
                "pass_subtypes": tuple({
                    subtype: {"attempts": sum(r['team']==team and r.get('release_subtype')==subtype for r in self._passes),
                              "completions": sum(r['team']==team and r.get('release_subtype')==subtype and r['completed'] for r in self._passes)}
                    for subtype in ('foot_kick','head')} for team in range(2)),
                "limitation": "Positive HEAD power is confirmed by joining action actor and frame; it is unobserved when legacy actions are unavailable.",
            },
        )


__all__ = [
    "SHOT_OUTCOMES",
    "CausalEventResolver",
    "CausalSummary",
    "is_goal_directed_shot",
    "is_shot_attempt",
]
