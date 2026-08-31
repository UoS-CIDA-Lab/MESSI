"""Deterministic tactical profiler for :mod:`policy`.

This is deliberately a rollout-only diagnostic: it does not write datasets or mutate the
environment.  Metrics use ``info['kick_applied']`` so passive body contacts and failed challenges
are never counted as policy kicks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..constants import (
    BALL_ALIVE,
    RK_CORNER,
    RK_FREEKICK,
    RK_GK_HOLD,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_NONE,
    RK_OFFSIDE,
    RK_PENALTY,
    RK_THROWIN,
    TOUCH_DRIBBLE,
    TOUCH_PASS,
    TOUCH_PASS_HEAD,
    TOUCH_GK_CATCH,
    TOUCH_PARRY,
    TOUCH_SHOOT,
    TOUCH_SHOOT_HEAD,
    TOUCH_TACKLE,
    TOUCH_INTERCEPT,
)
from ..env import SoccerEnv
from ..config import RulePolicy
from .policy import (
    RULE_POLICY_VERSION,
    make_rule_based_policy,
    policy_config_fingerprint,
    prefix_stable_keys,
)
from ..timebase import DEFAULT_TIMEBASE
from . import tactics as T


_RESTART_KIND_NAMES = {
    RK_KICKOFF: "kickoff",
    RK_THROWIN: "throw_in",
    RK_GOALKICK: "goal_kick",
    RK_CORNER: "corner_kick",
    RK_FREEKICK: "free_kick",
    RK_PENALTY: "penalty_kick",
    RK_OFFSIDE: "offside_free_kick",
    RK_GK_HOLD: "goalkeeper_hold",
}

_PLAYER_SPEED_BINS_MPS = np.asarray(
    [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 20.0],
    dtype=np.float64,
)
_PASS_LENGTH_BINS_M = np.asarray(
    [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 200.0],
    dtype=np.float64,
)
_LOW_PASS_LAUNCH_MAX_RAD = 0.12
_CHALLENGE_CONTROL_SPEED_MARGIN_MPS = 0.75
_CHALLENGE_CLEARANCE_SPEED_MIN_MPS = 12.0
_DFL_REFERENCE_PATH = Path(__file__).with_name("dfl_reference.json")


def _load_dfl_reference():
    """Load the frozen DFL target artifact without making rollout depend on data."""

    try:
        payload = json.loads(_DFL_REFERENCE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"available": False, "error": str(exc)}
    if payload.get("schema") not in {
        "soccer.rule-policy.dfl-reference/v1",
        "soccer.rule-policy.dfl-reference/v2",
        "soccer.rule-policy.dfl-reference/v3",
    }:
        return {
            "available": False,
            "error": f"unsupported DFL reference schema {payload.get('schema')!r}",
        }
    return {"available": True, **payload}


def _distribution_summary(values):
    """Return a compact, JSON-safe summary for a finite numeric sample."""

    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return {
            "p5": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "mean": None,
            "std": None,
            "n": 0,
        }
    p5, p25, p50, p75, p95 = np.quantile(
        sample, (0.05, 0.25, 0.50, 0.75, 0.95)
    )
    return {
        "p5": float(p5),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "mean": float(sample.mean()),
        "std": float(sample.std()),
        "n": int(sample.size),
    }


def _completed_low_pass_chain_metrics(records, max_gap_frames):
    """Summarise connected chains of completed low passes.

    A continuation must be kicked by the previous receiver, for the same team,
    after that reception and within ``max_gap_frames``. This is deliberately
    stricter than merely observing two team passes close together: the metric
    represents an actual ground combination. Net progression uses the first
    passer and final receiver's attack-folded x positions, so movement between
    receptions and releases remains part of the build-up phase.
    """

    chains = []
    current = []

    def finish_current():
        nonlocal current
        if current:
            chains.append(current)
            current = []

    for record in sorted(records, key=lambda item: item["pass_frame"]):
        if not record["low"]:
            finish_current()
            continue
        follows = bool(
            current
            and record["team"] == current[-1]["team"]
            and record["passer"] == current[-1]["receiver"]
            and 0 <= record["pass_frame"] - current[-1]["receive_frame"]
            <= max_gap_frames
        )
        if not follows:
            finish_current()
        current.append(record)
    finish_current()

    multi = [chain for chain in chains if len(chain) >= 2]
    net_progress = np.asarray(
        [
            chain[-1]["end_folded_x"] - chain[0]["start_folded_x"]
            for chain in multi
        ],
        dtype=np.float64,
    )
    low_records = [record for record in records if record["low"]]
    return {
        "completed_low_passes": len(low_records),
        "completed_low_progressive_passes": sum(
            record["progress_m"] > 5.0 for record in low_records
        ),
        "low_multi_pass_chains": len(multi),
        "low_positive_chains": int(np.sum(net_progress > 0.0)),
        "low_progressive_chains": int(np.sum(net_progress > 5.0)),
        "longest_low_pass_chain": max((len(chain) for chain in chains), default=0),
        "low_chain_net_progress_m": _distribution_summary(net_progress),
        "_low_chain_net_progress_values": net_progress,
    }


def _completed_restart_durations(restart_kind, control_fps):
    """Return uncensored restart-run durations grouped by public kind name.

    A rollout may end during a restart.  Including that right-censored tail as
    if it were complete biases slow restarts downwards, so only runs followed
    by another state are admitted.  A run beginning at reset is not
    left-censored: the reset contract opens the initial kickoff at frame zero.
    """

    restart_kind = np.asarray(restart_kind)
    if restart_kind.ndim != 1:
        raise ValueError(
            f"restart_kind must be rank 1, got {restart_kind.shape}"
        )
    durations = {name: [] for name in _RESTART_KIND_NAMES.values()}
    frame = 0
    while frame < restart_kind.size:
        kind = int(restart_kind[frame])
        stop = frame + 1
        while stop < restart_kind.size and int(restart_kind[stop]) == kind:
            stop += 1
        if kind != RK_NONE and stop < restart_kind.size:
            name = _RESTART_KIND_NAMES.get(kind, f"unknown_{kind}")
            durations.setdefault(name, []).append(
                float(stop - frame) / float(control_fps)
            )
        frame = stop
    return durations


def _restart_starts(restart_kind):
    """재개 **개시** 횟수. 지속 프레임이 아니라 인스턴스를 센다.

    ``_completed_restart_durations``는 지속시간만 보므로 "경기당 코너 몇 개"를 낼 수
    없다. 재개율은 축구다움을 재는 가장 굵은 지표라 따로 센다 — 코너가 0건인지
    8건인지는 지속시간 분포만 봐서는 드러나지 않는다.
    """

    restart_kind = np.asarray(restart_kind)
    if restart_kind.ndim != 1:
        raise ValueError(
            f"restart_kind must be rank 1, got {restart_kind.shape}"
        )
    counts = {name: 0 for name in _RESTART_KIND_NAMES.values()}
    previous = np.concatenate([[RK_NONE], restart_kind[:-1]])
    for kind, name in _RESTART_KIND_NAMES.items():
        counts[name] = int(np.count_nonzero(
            (restart_kind == kind) & (previous != kind)
        ))
    return counts


# env 재개 종류 → K리그 제공자 라벨. 킥오프·GK홀드는 대응 라벨이 없어 비율을 내지
# 않는다(개수는 그대로 보고한다). 오프사이드 재개는 제공자가 freeKick으로 찍는다.
_KLEAGUE_RESTART_LABEL = {
    "throw_in": ("setpiece_throwIn",),
    "goal_kick": ("setpiece_goalKick",),
    "corner_kick": ("setpiece_cornerKick",),
    "free_kick": ("setpiece_freeKick",),
    "offside_free_kick": ("setpiece_freeKick",),
    "penalty_kick": ("setpiece_penaltyKick",),
}
_EVENT_RATES_PATH = Path(__file__).with_name("event_rates.json")


def _load_event_rates():
    """경기당 실측 이벤트율. 없으면 None — 롤아웃이 자료에 의존하지 않게 한다."""

    try:
        with open(_EVENT_RATES_PATH, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    per_match = payload.get("per_match")
    return per_match if isinstance(per_match, dict) else None


def _duration_summary(samples):
    """JSON-safe compact summary for one uncensored duration sample."""

    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return {
            "completed_count": 0,
            "mean_s": None,
            "p10_s": None,
            "p50_s": None,
            "p90_s": None,
        }
    p10, p50, p90 = np.quantile(values, (0.10, 0.50, 0.90))
    return {
        "completed_count": int(values.size),
        "mean_s": float(values.mean()),
        "p10_s": float(p10),
        "p50_s": float(p50),
        "p90_s": float(p90),
    }


def _first_following_touch(touch, start, horizon):
    """다음으로 공을 건드린 control frame과 그 프레임의 **모든** 접촉자.

    한 control frame 안의 선후는 여기서 알 수 없다. env는 순서가 보존된
    ``touch_event_actor``/``touch_event_control_t``를 내주지만 이 프로파일러는 프레임
    단위 ``touch``만 수집한다. 그래서 같은 프레임에 동료와 상대가 함께 닿으면 호출부는
    **동료 우선**으로 성공 처리한다(``same_team.size`` 검사) — 성공률을 과대평가하는
    쪽이다. 순서를 복원할 수 없는 이상 어느 쪽을 골라도 편향이 생기고, 이 지표는 DFL
    성공률과의 비교로 캘리브됐으므로 그 편향을 유지한다. 여기서 나온 성공률을 정확한
    값으로 읽으면 안 된다.
    """

    stop = min(touch.shape[0], start + horizon + 1)
    for frame in range(start + 1, stop):
        actors = np.flatnonzero(touch[frame] != 0)
        if actors.size:
            return frame, actors
    return None, np.empty(0, np.int64)


def _classify_pass_direction(kick_direction_xy, attack_direction):
    """Classify an applied kick using DFL's folded ±60-degree sectors."""

    folded = (
        np.asarray(kick_direction_xy, dtype=np.float64)
        * float(attack_direction)
    )
    if folded.shape != (2,) or not np.all(np.isfinite(folded)):
        return None
    norm = float(np.linalg.norm(folded))
    if norm <= 1.0e-8:
        return None
    forward_component = float(folded[0]) / norm
    if forward_component > 0.5:
        return "forward"
    if forward_component < -0.5:
        return "backward"
    return "sideways"


def analyse_rollout(out, final_state, env, policy_config):
    (
        positions,
        player_velocity,
        attack_dir,
        team_id,
        poss_team,
        restart_kind,
        kick_dir,
        kick_power,
        launch,
        spin_side,
        spin_back,
        kick_applied,
        kick_intent,
        f2b_avail,
        player_xg,
        touch,
        scored,
        finite_state,
        finite_action,
        ball_position,
        ball_velocity,
        ball_state,
        active_player,
        challenge_cooldown,
        player_reach_z,
        restart_position_forced,
        move_power,
        stamina_long,
        stamina_short,
        stamina_long_workload,
        stamina_long_speed_load,
        stamina_long_acceleration_load,
        stamina_long_sprint_extra,
        stamina_short_load,
        stamina_short_recovery_factor,
        stamina_locomotion_seconds,
    ) = (np.asarray(x) for x in out)
    is_goalkeeper = np.asarray(final_state.gk_indices) > 0
    open_kick = (
        kick_applied
        & (restart_kind[:, None] == RK_NONE)
        & (poss_team[:, None] == team_id)
    )
    played_pass_touch = (
        open_kick & ((touch == TOUCH_PASS) | (touch == TOUCH_PASS_HEAD))
    )
    engine_shot_mask = open_kick & ((touch == TOUCH_SHOOT) | (touch == TOUCH_SHOOT_HEAD))
    dribble_mask = open_kick & (touch == TOUCH_DRIBBLE)
    forward_component = kick_dir[:, :, 0] * attack_dir
    folded_x = positions[:, :, 0] * attack_dir
    folded_y = positions[:, :, 1] * attack_dir
    cross_mask = (
        open_kick
        & (np.abs(folded_y) > env.hy * 0.38)
        & (folded_x > env.hx * 0.08)
        & (launch > 0.25)
        & (np.abs(spin_side) > 0.4)
        & (spin_back > 0.2)
    )
    # Env의 TOUCH_PASS는 물리 접촉 분류라서 수신자 없는 압박 해제/공중 클리어도 포함한다.
    # DFL의 Play/Pass는 Cross와 별개이며 의도한 동료가 있는 사건이다. 적용 킥 방향의
    # 전방 45도 cone(lead target 여유 포함)에 3~50m 활성 동료가 있어야 공통 pass 분모에
    # 넣는다. 이 필터가 없으면 3m/s 중립 제어 뒤 반복 클리어가 실패 패스로 누적된다.
    actor_to_player = (
        positions[:, None, :, :] - positions[:, :, None, :]
    )
    actor_to_player_distance = np.linalg.norm(actor_to_player, axis=3)
    kick_direction_norm = np.linalg.norm(kick_dir[:, :, :2], axis=2)
    intended_cos = np.sum(
        actor_to_player * kick_dir[:, :, None, :2], axis=3
    ) / (
        actor_to_player_distance
        * kick_direction_norm[:, :, None]
        + 1.0e-8
    )
    teammate_candidate = (
        (team_id[:, :, None] == team_id[:, None, :])
        & active_player[:, None, :]
        & (~np.eye(env.N, dtype=bool)[None, :, :])
        & (actor_to_player_distance >= 3.0)
        & (actor_to_player_distance <= 50.0)
        & (intended_cos >= np.cos(np.deg2rad(45.0)))
    )
    has_intended_teammate = np.any(teammate_candidate, axis=2)
    pass_mask = played_pass_touch & (~cross_mask) & has_intended_teammate
    clearance_mask = played_pass_touch & (~cross_mask) & (~has_intended_teammate)
    shot_mask = engine_shot_mask & (spin_back < -0.1)
    # "슛 뒤 4초" 창의 단일 진실원천. 프레임 수로 박으면 control_fps를 바꾼 순간
    # 이름과 값이 어긋난다.
    shot_horizon = int(round(4.0 * env.control_fps))
    shot_xg_values = player_xg[shot_mask]
    shot_frames, shot_actors = np.where(shot_mask)
    shot_folded_positions = (
        ball_position[shot_frames, :2]
        * attack_dir[shot_frames, shot_actors, None]
    )
    shot_distance_values = np.sqrt(
        (env.hx - shot_folded_positions[:, 0]) ** 2
        + shot_folded_positions[:, 1] ** 2
    )
    low_pass_mask = pass_mask & (launch <= _LOW_PASS_LAUNCH_MAX_RAD)
    pass_frames, pass_actors = np.where(pass_mask)
    pass_post_frame = np.minimum(pass_frames + 1, ball_velocity.shape[0] - 1)
    pass_post_kick_speed_3d = np.linalg.norm(
        ball_velocity[pass_post_frame], axis=1
    ) if pass_frames.size else np.empty(0, dtype=np.float64)
    open_intent = (
        kick_intent
        & (restart_kind[:, None] == RK_NONE)
        & (poss_team[:, None] == team_id)
    )
    shot_intent = open_intent & (spin_back < -0.1)

    # 소유팀 공격 방향으로 공 좌표를 접어, 정책이 실제로 슈팅존에 머물렀는지 구분한다.
    # 기회 자체가 없으면 마무리 가중치 문제가 아니라 전진/수신 문제다.
    poss_attack = np.zeros(poss_team.shape, np.float32)
    for team in (0, 1):
        player = int(np.flatnonzero(team_id[0] == team)[0])
        rows = poss_team == team
        poss_attack[rows] = attack_dir[rows, player]
    folded_ball_x = ball_position[:, 0] * poss_attack
    folded_ball_y = ball_position[:, 1] * poss_attack
    ball_goal_distance = np.sqrt(
        (env.hx - folded_ball_x) ** 2 + folded_ball_y ** 2
    )
    controlled_live = (
        (poss_team >= 0)
        & (ball_state == BALL_ALIVE)
        & (restart_kind == RK_NONE)
    )
    shooting_zone = controlled_live & (
        ball_goal_distance < env.e_cfg.f2b_shoot_range
    )

    # Use the same open-play denominator and >7 m/s sprint-start definition as
    # the original realism analysis.  This keeps movement and event tuning
    # in one configuration-stamped profile receipt.
    live_open_play = (ball_state == BALL_ALIVE) & (restart_kind == RK_NONE)
    live_active = active_player & live_open_play[:, None]
    player_speed = np.linalg.norm(player_velocity, axis=2)
    live_player_speed = player_speed[live_active]
    live_move_power = move_power[live_active]

    # Challenge output taxonomy is inferred from the applied contact and the
    # next-frame physical ball speed, not from a submitted action.  This makes
    # the receipt robust to failed gates and directly exposes the v41 contract:
    # a soft control, a low intended outlet, or an emergency forward clearance.
    challenge_touch_mask = (
        kick_applied
        & live_open_play[:, None]
        & (~is_goalkeeper[None, :])
        & np.isin(touch, (TOUCH_TACKLE, TOUCH_INTERCEPT))
    )
    challenge_frames, challenge_actors = np.where(challenge_touch_mask)
    challenge_post_frames = np.minimum(
        challenge_frames + 1, ball_velocity.shape[0] - 1
    )
    challenge_post_kick_speed_3d = (
        np.linalg.norm(ball_velocity[challenge_post_frames], axis=1)
        if challenge_frames.size
        else np.empty(0, dtype=np.float64)
    )
    challenge_post_speed_grid = np.zeros(
        challenge_touch_mask.shape, dtype=np.float64
    )
    challenge_post_speed_grid[
        challenge_frames, challenge_actors
    ] = challenge_post_kick_speed_3d
    challenge_actor_ball_x = ball_position[:, 0, None] * attack_dir
    challenge_forward = kick_dir[:, :, 0] * attack_dir > 0.25
    challenge_clearance_like_mask = (
        challenge_touch_mask
        & (challenge_actor_ball_x < 0.0)
        & challenge_forward
        & (launch > _LOW_PASS_LAUNCH_MAX_RAD)
        & (
            challenge_post_speed_grid
            >= _CHALLENGE_CLEARANCE_SPEED_MIN_MPS
        )
    )
    challenge_control_like_mask = (
        challenge_touch_mask
        & (
            challenge_post_speed_grid
            <= policy_config.challenge_control_touch_speed_mps
            + _CHALLENGE_CONTROL_SPEED_MARGIN_MPS
        )
    )
    challenge_outlet_like_mask = (
        challenge_touch_mask
        & (~challenge_control_like_mask)
        & (~challenge_clearance_like_mask)
        & (launch <= _LOW_PASS_LAUNCH_MAX_RAD)
        & has_intended_teammate
    )
    challenge_other_like_mask = (
        challenge_touch_mask
        & (~challenge_control_like_mask)
        & (~challenge_outlet_like_mask)
        & (~challenge_clearance_like_mask)
    )
    challenge_touch_by_team = np.asarray([
        np.sum(challenge_touch_mask & (team_id == team))
        for team in (0, 1)
    ], dtype=np.int64)
    challenge_control_by_team = np.asarray([
        np.sum(challenge_control_like_mask & (team_id == team))
        for team in (0, 1)
    ], dtype=np.int64)
    challenge_outlet_by_team = np.asarray([
        np.sum(challenge_outlet_like_mask & (team_id == team))
        for team in (0, 1)
    ], dtype=np.int64)
    challenge_clearance_by_team = np.asarray([
        np.sum(challenge_clearance_like_mask & (team_id == team))
        for team in (0, 1)
    ], dtype=np.int64)

    # Retention is evaluated only for a challenge whose next state awards
    # controlled possession to the actor's team.  Neutral ball is allowed
    # during the following two seconds; any opponent control breaks retention.
    challenge_retained_2s_eligible = 0
    challenge_retained_for_2s = 0
    challenge_retention_eligible_by_team = np.zeros(2, np.int64)
    challenge_retained_by_team = np.zeros(2, np.int64)
    challenge_retention_horizon = int(round(2.0 * env.control_fps))
    for frame, actor in zip(challenge_frames, challenge_actors):
        start = int(frame) + 1
        stop = start + challenge_retention_horizon
        team = int(team_id[int(frame), int(actor)])
        if stop >= poss_team.shape[0] or int(poss_team[start]) != team:
            continue
        challenge_retained_2s_eligible += 1
        challenge_retention_eligible_by_team[team] += 1
        retained = not np.any(poss_team[start:stop + 1] == 1 - team)
        challenge_retained_for_2s += int(retained)
        challenge_retained_by_team[team] += int(retained)

    # 실제 challenge gate 소비는 정책의 거리나 ``want_f2b`` 제출을 세지 않는다. 상대/중립
    # 후보가 경합기에 들어가면 contest가 해당 actor의 cooldown을 0보다 크게 설치하고,
    # cooldown 중에는 재시도가 불가능하므로 0→양수 상승 에지가 시도와 일대일 대응한다.
    # state[t]의 상승은 직전 control frame에서 생겼으므로 문맥도 t-1에서 읽는다.
    cooldown_rise = (
        challenge_cooldown[1:] > challenge_cooldown[:-1] + 1.0e-6
    )
    challenge_context_live = live_open_play[:-1, None]
    challenge_gate = cooldown_rise & challenge_context_live
    outfield_challenge_gate = challenge_gate & (~is_goalkeeper[None, :])
    prior_possession = poss_team[:-1]
    opposing_challenger = (
        np.isin(prior_possession, (0, 1))[:, None]
        & (team_id[:-1] != prior_possession[:, None])
    )
    loose_challenge_gate = (
        outfield_challenge_gate & (prior_possession[:, None] < 0)
    )

    possessor_mask = (
        (team_id[:-1] == prior_possession[:, None])
        & active_player[:-1]
        & np.isin(prior_possession, (0, 1))[:, None]
    )
    possessor_ball_distance = np.linalg.norm(
        positions[:-1] - ball_position[:-1, None, :2], axis=2
    )
    carrier_index = np.argmin(
        np.where(possessor_mask, possessor_ball_distance, np.inf), axis=1
    )
    frame_index = np.arange(carrier_index.shape[0])
    carrier_exists = np.any(possessor_mask, axis=1)
    carrier_distance = possessor_ball_distance[frame_index, carrier_index]
    carrier_height_reachable = (
        ball_position[:-1, 2]
        <= player_reach_z[:-1][frame_index, carrier_index] + env.r_ball
    )
    carrier_controls = (
        carrier_exists
        & (carrier_distance <= env.e_cfg.reach_xy_carry + env.r_ball)
        & carrier_height_reachable
    )
    carrier_duel_gate = (
        outfield_challenge_gate
        & opposing_challenger
        & carrier_controls[:, None]
    )
    travelling_interception_gate = (
        outfield_challenge_gate
        & opposing_challenger
        & (~carrier_controls[:, None])
    )
    challenge_gate_by_team = np.asarray([
        np.sum(
            outfield_challenge_gate
            & (team_id[:-1] == team)
        )
        for team in (0, 1)
    ], dtype=np.int64)
    carrier_duel_gate_by_team = np.asarray([
        np.sum(carrier_duel_gate & (team_id[:-1] == team))
        for team in (0, 1)
    ], dtype=np.int64)

    # DFL tracking과 같은 캐리어 중심 오프더볼 기하. 공 소유 슬롯은 provider에 없으므로
    # 양쪽 모두 해당 소유팀의 공 최근접 활성 선수로 정의한다. 분포를 raw sample로 남겨
    # 여러 seed의 quantile을 평균하지 않고 한 번에 다시 계산한다.
    support_rank_distance_values = [[], [], []]
    nearest_three_support_speed_values = []
    nearby_teammate_values = {10: [], 15: [], 20: []}
    final_third_box_attacker_values = []
    final_third_ahead_runner_values = []
    final_third_ball_x_values = []
    final_third_defensive_line_x_values = []
    final_third_front_gap_values = []
    final_third_legal_box_opportunity = []
    for frame in np.flatnonzero(controlled_live):
        team = int(poss_team[frame])
        members = np.flatnonzero(
            active_player[frame] & (team_id[frame] == team)
        )
        if members.size < 2:
            continue
        carrier = int(members[np.argmin(np.linalg.norm(
            positions[frame, members] - ball_position[frame, :2], axis=1
        ))])
        teammates = members[members != carrier]
        distances = np.linalg.norm(
            positions[frame, teammates] - positions[frame, carrier], axis=1
        )
        order = np.argsort(distances, kind="stable")
        for rank in range(min(3, order.size)):
            support_rank_distance_values[rank].append(
                float(distances[order[rank]])
            )
            nearest_three_support_speed_values.append(
                float(player_speed[frame, teammates[order[rank]]])
            )
        for radius in nearby_teammate_values:
            nearby_teammate_values[radius].append(
                int(np.sum(distances <= float(radius)))
            )

        if folded_ball_x[frame] <= env.hx / 3.0:
            continue
        attacking_outfield = (
            active_player[frame]
            & (team_id[frame] == team)
            & (~is_goalkeeper)
        )
        final_third_box_attacker_values.append(int(np.sum(
            attacking_outfield
            & (folded_x[frame] >= env.hx - env.pen_len)
            & (np.abs(folded_y[frame]) <= env.pen_hw)
        )))
        final_third_ahead_runner_values.append(int(np.sum(
            attacking_outfield
            & (folded_x[frame] > folded_ball_x[frame] + 3.0)
        )))
        # 빈 박스를 '안 뛴 공격수'와 '아직 들어갈 수 없는 온사이드 경계'로 분해한다.
        # Law 11 상한은 소유팀 공격 프레임에서 공과 두 번째 최종 수비수 중 더 앞선 값이다.
        team_frame_x = positions[frame, :, 0] * poss_attack[frame]
        defending_players = (
            active_player[frame] & (team_id[frame] != team)
        )
        defender_x = np.sort(team_frame_x[defending_players])
        defensive_line_x = (
            float(defender_x[-2]) if defender_x.size >= 2 else env.hx
        )
        legal_line_x = max(float(folded_ball_x[frame]), defensive_line_x)
        front_attacker_x = float(np.max(team_frame_x[attacking_outfield]))
        final_third_ball_x_values.append(float(folded_ball_x[frame]))
        final_third_defensive_line_x_values.append(defensive_line_x)
        final_third_front_gap_values.append(
            max(0.0, legal_line_x - front_attacker_x)
        )
        final_third_legal_box_opportunity.append(
            legal_line_x >= env.hx - env.pen_len + 0.25
        )

    # 소유 라벨 사이의 중립 비행은 건너뛰고, 다음 비중립 팀이 바뀔 때 한 번의 손실로 센다.
    # 회수는 3초 안에 이전 팀이 다시 소유하는가, 안정 유지는 2초 안에 상대 팀 소유가
    # 나타나지 않는가다. 후자는 의도적 패스의 짧은 neutral flight를 실패로 오인하지 않는다.
    transition_counts = {
        "events": 0,
        "regain_3s_eligible": 0,
        "regains_within_3s": 0,
        "stable_2s_eligible": 0,
        "stable_for_2s": 0,
    }
    transition_by_zone = {
        name: dict(transition_counts)
        for name in ("own_half", "attacking_half", "final_third")
    }
    last_control_team = None
    last_control_frame = None
    regain_horizon = int(round(3.0 * env.control_fps))
    stable_horizon = int(round(2.0 * env.control_fps))
    for frame in range(poss_team.shape[0]):
        if not live_open_play[frame] or int(poss_team[frame]) not in (0, 1):
            continue
        current_team = int(poss_team[frame])
        if last_control_team is not None and current_team != last_control_team:
            transition_counts["events"] += 1
            loss_player = int(np.flatnonzero(
                team_id[last_control_frame] == last_control_team
            )[0])
            loss_x = float(
                ball_position[last_control_frame, 0]
                * attack_dir[last_control_frame, loss_player]
            )
            zones = ["own_half"] if loss_x < 0.0 else ["attacking_half"]
            if loss_x > env.hx / 3.0:
                zones.append("final_third")
            for zone in zones:
                transition_by_zone[zone]["events"] += 1

            regain_stop = frame + regain_horizon + 1
            if regain_stop <= poss_team.shape[0]:
                regained = bool(np.any(
                    poss_team[frame + 1:regain_stop] == last_control_team
                ))
                transition_counts["regain_3s_eligible"] += 1
                transition_counts["regains_within_3s"] += int(regained)
                for zone in zones:
                    transition_by_zone[zone]["regain_3s_eligible"] += 1
                    transition_by_zone[zone]["regains_within_3s"] += int(regained)

            stable_stop = frame + stable_horizon + 1
            if stable_stop <= poss_team.shape[0]:
                opponent_control = np.isin(
                    poss_team[frame + 1:stable_stop], (0, 1)
                ) & (poss_team[frame + 1:stable_stop] != current_team)
                stable = not bool(np.any(opponent_control))
                transition_counts["stable_2s_eligible"] += 1
                transition_counts["stable_for_2s"] += int(stable)
                for zone in zones:
                    transition_by_zone[zone]["stable_2s_eligible"] += 1
                    transition_by_zone[zone]["stable_for_2s"] += int(stable)

            last_control_team = current_team
        elif last_control_team is None:
            last_control_team = current_team
        last_control_frame = frame
    per_player_seconds = live_active.sum(axis=0) / env.control_fps
    eligible_player = per_player_seconds > 0.0
    per_player_distance_m = (
        (player_speed * live_active).sum(axis=0) / env.control_fps
    )
    distance_per_90_m = (
        per_player_distance_m[eligible_player]
        / per_player_seconds[eligible_player]
        * 5400.0
    )
    sprinting = (player_speed > 7.0) & live_active
    sprint_starts = sprinting[1:] & (~sprinting[:-1])
    sprints_per_90 = (
        sprint_starts.sum(axis=0)[eligible_player]
        / per_player_seconds[eligible_player]
        * 5400.0
    )
    eligible_is_gk = np.asarray(final_state.gk_indices)[eligible_player] > 0

    # ── 장기/단기 stamina와 workload ────────────────────────────────────────
    # workload는 control frame당 평균 부하(무차원)다. 90분 환산 소모는 정의상
    # 장기 drop은 config의 drain base와 기준시간을 그대로 써서 90분으로 환산한다.
    # 분모는 clip 전체의 active 선수 프레임이다 — stamina는 재개 중에도 흐르므로
    # 이동 지표의 in-play 분모를 그대로 쓰면 안 된다.
    e_cfg = env.e_cfg
    charged = active_player.astype(bool)
    workload_sample = stamina_long_workload[charged]
    per_player_charged_seconds = charged.sum(axis=0) / env.control_fps
    charged_player = per_player_charged_seconds > 0.0
    workload_per_player = (
        (stamina_long_workload * charged).sum(axis=0)[charged_player]
        / (charged.sum(axis=0)[charged_player])
    )
    short_load_per_player = (
        (stamina_short_load * charged).sum(axis=0)[charged_player]
        / charged.sum(axis=0)[charged_player]
    )
    short_recovery_factor_per_player = (
        (stamina_short_recovery_factor * charged).sum(axis=0)[charged_player]
        / charged.sum(axis=0)[charged_player]
    )
    nominal_long_drain_per_90 = (
        workload_per_player
        * env.long_stamina_drain_base
        * e_cfg.long_stamina_reference_duration_s
    )
    linear_room = 1.0 - e_cfg.long_stamina_tail_knee
    stamina_end_per_90 = np.where(
        nominal_long_drain_per_90 <= linear_room,
        1.0 - nominal_long_drain_per_90,
        e_cfg.long_stamina_tail_knee * np.exp(
            -env.long_stamina_tail_decay
            * (nominal_long_drain_per_90 - linear_room)
        ),
    )
    stamina_long_drop_per_90 = 1.0 - np.clip(stamina_end_per_90, 0.0, 1.0)
    final_stamina_long = stamina_long[-1]
    final_stamina_short = stamina_short[-1]
    active_final = np.asarray(final_state.active_player).astype(bool)
    locomotion_share = (
        stamina_locomotion_seconds.sum(axis=0)[charged_player]
        / per_player_charged_seconds[charged_player]
    )
    charged_is_gk = np.asarray(final_state.gk_indices)[charged_player] > 0
    # idle은 config 상수를 베끼지 않고 telemetry의 잔차로 되돌린다. 그래야 계수를 바꾸거나
    # 다른 energy 버전으로 만든 receipt를 나란히 놓아도 분해가 총합과 항상 일치한다.
    _term_means = {
        name: float(channel[charged].mean()) if workload_sample.size else 0.0
        for name, channel in (
            ("speed", stamina_long_speed_load),
            ("acceleration", stamina_long_acceleration_load),
            ("sprint", stamina_long_sprint_extra),
        )
    }
    workload_terms = {
        "idle": (
            float(workload_sample.mean()) - sum(_term_means.values())
            if workload_sample.size else 0.0
        ),
        **_term_means,
    }

    ball_speed_xy = np.linalg.norm(ball_velocity[:, :2], axis=1)
    ball_speed_3d = np.linalg.norm(ball_velocity, axis=1)
    live_ball_speed = ball_speed_xy[live_open_play]
    live_ball_speed_3d = ball_speed_3d[live_open_play]
    controlled_ball_speed = ball_speed_xy[
        live_open_play & np.isin(poss_team, (0, 1))
    ]
    neutral_ball_speed = ball_speed_xy[
        live_open_play & (~np.isin(poss_team, (0, 1)))
    ]

    complete = 0
    progressive_complete = 0
    kill_complete = 0
    complete_by_team = np.zeros(2, np.int64)
    progressive_by_team = np.zeros(2, np.int64)
    kill_by_team = np.zeros(2, np.int64)
    cross_complete = 0
    cross_opponent_first = 0
    cross_self_first = 0
    cross_dead_before_touch = 0
    cross_no_touch = 0
    cross_goals = 0
    cross_followups = []
    next_shot_after_kill = 0
    shot_goals = 0
    completed_events = []
    completed_pass_records = []
    completed_progress = []
    reception_x = []
    pass_no_following_touch = 0
    pass_opponent_first = 0
    pass_passer_retouch = 0
    pass_failure_events = []
    kill_reception_xg = []
    kill_reception_goal_distance = []
    kill_followups = []
    for frame, passer in zip(*np.where(pass_mask)):
        receive_frame, receivers = _first_following_touch(touch, int(frame), 45)
        if receive_frame is None:
            pass_no_following_touch += 1
            pass_failure_events.append({
                "outcome": "no_following_touch",
                "second": round(float(frame) / env.control_fps, 3),
                "passer": int(passer),
                "passer_is_gk": bool(is_goalkeeper[passer]),
                "direction": _classify_pass_direction(
                    kick_dir[frame, passer, :2], attack_dir[frame, passer]
                ),
                "launch_speed_mps": round(float(np.linalg.norm(
                    ball_velocity[min(frame + 1, ball_velocity.shape[0] - 1)]
                )), 3),
            })
            continue
        passer_team = int(team_id[frame, passer])
        same_team = receivers[team_id[receive_frame, receivers] == passer_team]
        if same_team.size == 0:
            pass_opponent_first += 1
            pass_failure_events.append({
                "outcome": "opponent_first",
                "second": round(float(frame) / env.control_fps, 3),
                "touch_second": round(float(receive_frame) / env.control_fps, 3),
                "passer": int(passer),
                "passer_is_gk": bool(is_goalkeeper[passer]),
                "direction": _classify_pass_direction(
                    kick_dir[frame, passer, :2], attack_dir[frame, passer]
                ),
                "travel_m": round(float(np.linalg.norm(
                    ball_position[receive_frame, :2] - ball_position[frame, :2]
                )), 3),
                "next_actors": [int(actor) for actor in receivers],
                "next_touch_codes": [int(touch[receive_frame, actor]) for actor in receivers],
            })
            continue
        # Several contacts can occur in one control frame.  Prefer a different teammate;
        # a self-retouch is not a completed pass.
        different = same_team[same_team != passer]
        if different.size == 0:
            pass_passer_retouch += 1
            pass_failure_events.append({
                "outcome": "passer_retouch",
                "second": round(float(frame) / env.control_fps, 3),
                "touch_second": round(float(receive_frame) / env.control_fps, 3),
                "passer": int(passer),
                "passer_is_gk": bool(is_goalkeeper[passer]),
                "direction": _classify_pass_direction(
                    kick_dir[frame, passer, :2], attack_dir[frame, passer]
                ),
                "travel_m": round(float(np.linalg.norm(
                    ball_position[receive_frame, :2] - ball_position[frame, :2]
                )), 3),
                "next_touch_code": int(touch[receive_frame, passer]),
                "passer_to_ball_m": round(float(np.linalg.norm(
                    positions[receive_frame, passer] - ball_position[receive_frame, :2]
                )), 3),
            })
            continue
        receiver = int(different[0])
        complete += 1
        complete_by_team[passer_team] += 1
        progress = (
            positions[receive_frame, receiver, 0]
            - positions[frame, passer, 0]
        ) * attack_dir[frame, passer]
        completed_progress.append(float(progress))
        reception_x.append(float(folded_x[receive_frame, receiver]))
        progressive = progress > 5.0
        kill = progress > 10.0 and folded_x[receive_frame, receiver] > env.hx / 3.0
        progressive_complete += int(progressive)
        kill_complete += int(kill)
        progressive_by_team[passer_team] += int(progressive)
        kill_by_team[passer_team] += int(kill)
        completed_events.append((int(frame), passer_team))
        completed_pass_records.append({
            "pass_frame": int(frame),
            "receive_frame": int(receive_frame),
            "passer": int(passer),
            "receiver": receiver,
            "team": passer_team,
            "low": bool(low_pass_mask[frame, passer]),
            "progress_m": float(progress),
            "start_folded_x": float(folded_x[frame, passer]),
            "end_folded_x": float(folded_x[receive_frame, receiver]),
        })
        if kill:
            receiver_xg = float(player_xg[receive_frame, receiver])
            receiver_goal_distance = float(np.linalg.norm(
                np.array([env.hx, 0.0])
                - positions[receive_frame, receiver] * attack_dir[receive_frame, receiver]
            ))
            kill_reception_xg.append(receiver_xg)
            kill_reception_goal_distance.append(receiver_goal_distance)
            # A through ball can be followed by one layoff before the shot.  Four seconds
            # captures that short combination without attributing a later possession spell.
            stop = min(touch.shape[0], receive_frame + shot_horizon + 1)
            team_shot = np.any(
                shot_mask[receive_frame:stop]
                & (team_id[receive_frame:stop] == passer_team)
            )
            next_shot_after_kill += int(team_shot)
            follow_start = min(receive_frame + 1, stop)
            follow_frames, follow_players = np.where(
                open_kick[follow_start:stop]
                & (team_id[follow_start:stop] == passer_team)
            )
            next_team_kick = None
            if follow_frames.size:
                next_team_kick = {
                    "delay_s": round(
                        float(follow_start + follow_frames[0] - receive_frame)
                        / env.control_fps,
                        3,
                    ),
                    "touch_code": int(touch[
                        follow_start + int(follow_frames[0]), int(follow_players[0])
                    ]),
                    "actor": int(follow_players[0]),
                }
            following_team_kicks = []
            for rel_frame, actor in zip(follow_frames[:8], follow_players[:8]):
                kick_frame = follow_start + int(rel_frame)
                folded_actor = (
                    positions[kick_frame, int(actor)]
                    * attack_dir[kick_frame, int(actor)]
                )
                following_team_kicks.append({
                    "delay_s": round(
                        float(kick_frame - receive_frame) / env.control_fps, 3
                    ),
                    "touch_code": int(touch[kick_frame, int(actor)]),
                    "actor": int(actor),
                    "xg": round(float(player_xg[kick_frame, int(actor)]), 4),
                    "goal_distance_m": round(float(np.linalg.norm(
                        np.array([env.hx, 0.0]) - folded_actor
                    )), 3),
                })
            post_frame = min(receive_frame + 1, poss_team.shape[0] - 1)
            kill_followups.append({
                "pass_second": round(float(frame) / env.control_fps, 3),
                "receive_second": round(float(receive_frame) / env.control_fps, 3),
                "passer": int(passer),
                "receiver": receiver,
                "first_touch_code": int(touch[receive_frame, receiver]),
                "post_touch_possession": int(poss_team[post_frame]),
                "receiver_xg": round(receiver_xg, 4),
                "receiver_goal_distance_m": round(receiver_goal_distance, 3),
                "team_shot_within_4s": bool(team_shot),
                "next_team_kick": next_team_kick,
                "following_team_kicks": following_team_kicks,
            })

    # DFL ``PlayAngle``은 패스 결과 지점이 아니라 플레이가 시작될 때의 의도 방향이다.
    # 따라서 시뮬도 다음 터치(가로채기 위치)에 의해 방향이 뒤집히지 않도록 실제 적용된
    # kick direction을 공격 프레임으로 접어 분류한다. 성공 여부는 별도로 다음 실제 터치의
    # 다른 팀 동료 여부를 사용한다. 패스 길이는 아직 관측 결과 거리이므로 3 m 미만 제어
    # 터치와 비정상적으로 긴 미완결 궤적만 그 분포에서 제외한다.
    direction_counts = {"forward": 0, "sideways": 0, "backward": 0}
    classified_pass_distances = []
    reference_success_counts = {"forward": 0, "sideways": 0, "backward": 0}
    direction_by_team = {
        team: {"forward": 0, "sideways": 0, "backward": 0}
        for team in (0, 1)
    }
    reference_success_by_team = np.zeros(2, np.int64)
    for frame, passer in zip(*np.where(pass_mask)):
        next_frame, next_actors = _first_following_touch(touch, int(frame), 120)
        team = int(team_id[int(frame), int(passer)])
        reference_success = bool(
            next_frame is not None
            and np.any(
                (team_id[int(next_frame), next_actors] == team)
                & (next_actors != int(passer))
            )
        )
        direction = _classify_pass_direction(
            kick_dir[int(frame), int(passer), :2],
            attack_dir[int(frame), int(passer)],
        )
        if direction is not None:
            direction_counts[direction] += 1
            direction_by_team[team][direction] += 1
            reference_success_counts[direction] += int(reference_success)
            reference_success_by_team[team] += int(reference_success)

        end_frame = (
            min(touch.shape[0] - 1, int(frame) + 60)
            if next_frame is None else int(next_frame)
        )
        delta = (
            ball_position[end_frame, :2] - ball_position[int(frame), :2]
        ) * attack_dir[int(frame), int(passer)]
        distance = float(np.linalg.norm(delta))
        if distance < 3.0 or distance > 100.0:
            continue
        classified_pass_distances.append(distance)

    # 수신 직전 1초의 이동은 '선수가 먼저 공간을 만들고 그곳에 공이 왔는가'를 직접 본다.
    # 수신 프레임의 공격방향으로 접어 전진/횡 이동을 DFL tracking과 같은 좌표로 기록한다.
    receiver_prepass_movement = []
    receiver_prepass_lateral = []
    receiver_prepass_forward = []
    movement_horizon = int(round(env.control_fps))
    for record in completed_pass_records:
        receive_frame = int(record["receive_frame"])
        receiver = int(record["receiver"])
        before_frame = max(0, receive_frame - movement_horizon)
        delta = (
            positions[receive_frame, receiver]
            - positions[before_frame, receiver]
        ) * attack_dir[receive_frame, receiver]
        receiver_prepass_movement.append(float(np.linalg.norm(delta)))
        receiver_prepass_lateral.append(float(abs(delta[1])))
        receiver_prepass_forward.append(float(delta[0]))

    # 완료 패스를 받은 선수가 2초 안에 첫 팀 킥으로 원 패서에게 다시 패스하면 즉시 반환이다.
    # 정책 내부에 숨은 기억을 넣지 않고도 rollout 결과에서 A→B→A 순환을 직접 감시한다.
    immediate_return_passes = 0
    receiver_next_action_delays = []
    return_horizon = int(round(2.0 * env.control_fps))
    for record in completed_pass_records:
        start = record["receive_frame"] + 1
        stop = min(open_kick.shape[0], start + return_horizon)
        if start >= stop:
            continue
        team_kicks = np.argwhere(
            open_kick[start:stop]
            & (team_id[start:stop] == record["team"])
        )
        if team_kicks.size == 0:
            continue
        rel_frame, actor = map(int, team_kicks[0])
        kick_frame = start + rel_frame
        if actor != record["receiver"]:
            continue
        receiver_next_action_delays.append(
            float(kick_frame - record["receive_frame"]) / env.control_fps
        )
        if not pass_mask[kick_frame, actor]:
            continue
        return_frame, return_receivers = _first_following_touch(
            touch, kick_frame, 45
        )
        if return_frame is None:
            continue
        same_team = return_receivers[
            team_id[return_frame, return_receivers] == record["team"]
        ]
        immediate_return_passes += int(record["passer"] in same_team)

    # DFL exposes both ``Height=flat`` and the event actor/recipient, allowing
    # the same two-second low-relay and goalkeeper build-up definitions to be
    # evaluated on rollout outcomes.  A relay is a pass/cross by the actual
    # receiver, not merely any subsequent kick by the same team.
    open_play_distribution = pass_mask | cross_mask
    quick_relays = 0
    quick_flat_relays = 0
    quick_relay_opportunities = len(completed_pass_records)
    gk_received_passes = 0
    gk_buildup_cycles = 0
    quick_relay_by_team = np.zeros(2, np.int64)
    quick_flat_relay_by_team = np.zeros(2, np.int64)
    quick_opportunity_by_team = np.zeros(2, np.int64)
    gk_received_by_team = np.zeros(2, np.int64)
    gk_buildup_by_team = np.zeros(2, np.int64)
    relay_horizon = int(round(2.0 * env.control_fps))
    gk_horizon = int(round(8.0 * env.control_fps))
    for record in completed_pass_records:
        team = int(record["team"])
        quick_opportunity_by_team[team] += 1
        start = record["receive_frame"] + 1
        stop = min(open_play_distribution.shape[0], start + relay_horizon)
        plays = np.argwhere(open_play_distribution[start:stop])
        if plays.size:
            rel_frame, actor = map(int, plays[0])
            play_frame = start + rel_frame
            if (
                actor == record["receiver"]
                and int(team_id[play_frame, actor]) == team
            ):
                quick_relays += 1
                quick_relay_by_team[team] += 1
                is_flat_relay = bool(
                    record["low"]
                    and launch[play_frame, actor] <= _LOW_PASS_LAUNCH_MAX_RAD
                )
                quick_flat_relays += int(is_flat_relay)
                quick_flat_relay_by_team[team] += int(is_flat_relay)

        receiver = int(record["receiver"])
        if not is_goalkeeper[receiver]:
            continue
        gk_received_passes += 1
        gk_received_by_team[team] += 1
        stop = min(open_play_distribution.shape[0], start + gk_horizon)
        plays = np.argwhere(open_play_distribution[start:stop])
        if plays.size == 0:
            continue
        rel_frame, actor = map(int, plays[0])
        play_frame = start + rel_frame
        if (
            actor == receiver
            and int(team_id[play_frame, actor]) == team
            and pass_mask[play_frame, actor]
        ):
            gk_buildup_cycles += 1
            gk_buildup_by_team[team] += 1

    # Possession-transition frequency is taxonomy independent; the subset whose
    # acquisition touch is TACKLE/INTERCEPT is the closest simulator analogue
    # of DFL TacklingGame possession wins under pressure.
    possession_gains = 0
    challenge_possession_gains = 0
    ground_challenge_possession_gains = 0
    possession_gain_by_team = np.zeros(2, np.int64)
    challenge_gain_by_team = np.zeros(2, np.int64)
    ground_challenge_gain_by_team = np.zeros(2, np.int64)
    for frame in np.flatnonzero(
        np.isin(poss_team[:-1], (0, 1))
        & np.isin(poss_team[1:], (0, 1))
        & (poss_team[:-1] != poss_team[1:])
        & live_open_play[:-1]
        & live_open_play[1:]
    ):
        gaining_team = int(poss_team[frame + 1])
        acquisition_codes = touch[frame, team_id[frame] == gaining_team]
        tackle_gain = bool(np.any(acquisition_codes == TOUCH_TACKLE))
        pressured_gain = tackle_gain or bool(
            np.any(acquisition_codes == TOUCH_INTERCEPT)
        )
        possession_gains += 1
        possession_gain_by_team[gaining_team] += 1
        challenge_possession_gains += int(pressured_gain)
        challenge_gain_by_team[gaining_team] += int(pressured_gain)
        ground_challenge_possession_gains += int(tackle_gain)
        ground_challenge_gain_by_team[gaining_team] += int(tackle_gain)

    for frame, crosser in zip(*np.where(cross_mask)):
        stop = min(touch.shape[0], int(frame) + 46)
        # ``scored``는 **득점 팀 번호**(없으면 -1)다. ``>= 0``만 보면 상대 팀 골도 우리
        # 크로스의 성공으로 귀속된다 — 굴절 자책골이나 곧바로 이어진 역습이 그렇다.
        if np.any(scored[int(frame):stop] == int(team_id[int(frame), int(crosser)])):
            cross_goals += 1
            cross_followups.append({
                "cross_second": round(float(frame) / env.control_fps, 3),
                "crosser": int(crosser),
                "outcome": "goal",
            })
            continue
        receive_frame, receivers = _first_following_touch(touch, int(frame), 45)
        if receive_frame is None:
            outcome = (
                "dead_before_touch"
                if np.any(restart_kind[int(frame) + 1:stop] != RK_NONE)
                else "no_touch"
            )
            if np.any(restart_kind[int(frame) + 1:stop] != RK_NONE):
                cross_dead_before_touch += 1
            else:
                cross_no_touch += 1
            cross_followups.append({
                "cross_second": round(float(frame) / env.control_fps, 3),
                "crosser": int(crosser),
                "outcome": outcome,
                "last_ball_folded_xy_m": np.round(
                    ball_position[stop - 1, :2] * attack_dir[frame, crosser], 3
                ).tolist(),
            })
            continue
        crosser_team = int(team_id[frame, crosser])
        if np.any(team_id[receive_frame, receivers] != crosser_team):
            cross_opponent_first += 1
            outcome = "opponent_first"
        elif np.any(receivers == crosser):
            cross_self_first += 1
            outcome = "crosser_retouch"
        else:
            cross_complete += 1
            outcome = "teammate_first"
        cross_followups.append({
            "cross_second": round(float(frame) / env.control_fps, 3),
            "crosser": int(crosser),
            "outcome": outcome,
            "touch_delay_s": round(
                float(receive_frame - frame) / env.control_fps, 3
            ),
            "receivers": [int(player) for player in receivers],
            "touch_codes": [int(touch[receive_frame, player]) for player in receivers],
            "ball_folded_xy_m": np.round(
                ball_position[receive_frame, :2] * attack_dir[frame, crosser], 3
            ).tolist(),
        })

    # 슛 뒤 4초 이내 득점. 같은 슛을 여러 터치로 중복 계수하지 않도록 실제 kick_applied
    # SHOT 행만 순회한다. 전체 score와 분리해 GK/조준 현실성을 감시한다.
    for frame, shooter in zip(*np.where(shot_mask)):
        # 4초 창을 프레임 수로 박으면 control_fps를 바꾼 순간 창 길이가 달라진다.
        stop = min(scored.shape[0], int(frame) + shot_horizon + 1)
        # 크로스와 같은 이유로 득점 팀을 확인한다. 확인하지 않으면 슛 전환율이
        # 상대 팀 골만큼 부풀고, 그 지표로 GK·조준 현실성을 감시할 수 없다.
        shot_goals += int(np.any(
            scored[int(frame):stop] == int(team_id[int(frame), int(shooter)])))

    # 슛 결말 분해 — 골/선방/차단/골대밖. "슛이 몇 개냐"만으로는 축구다움을 못 잰다:
    # 실측으로 슛이 골 아니면 GK 처리로만 끝나 골킥·코너가 0건이었는데, 슛 개수만
    # 보면 그 상태가 정상으로 보인다. K리그 실측 비율은 골 12%·선방 24%·차단 24%·
    # 골대밖 40%다(event_rates.json: shot 22.25 / save 5.30 / blockShot 5.41).
    shot_outcomes = {
        "goal": 0, "saved": 0, "blocked": 0, "off_target": 0, "unresolved": 0,
    }
    for frame, shooter in zip(*np.where(shot_mask)):
        frame = int(frame)
        team = int(team_id[frame, int(shooter)])
        stop = min(scored.shape[0], frame + shot_horizon + 1)
        label = "unresolved"
        # 킥이 적용된 **그 프레임**에 이미 물리가 돌아 골·접촉이 끝날 수 있다.
        # frame+1부터 훑으면 근거리 슛의 결말을 통째로 놓친다.
        for step in range(frame, stop):
            if int(scored[step]) == team:
                label = "goal"
                break
            # 수비 접촉을 재개보다 **먼저** 본다. 차단·선방이 같은 프레임에 코너를
            # 만들면 재개를 먼저 검사한 순간 그 슛이 "골대밖"으로 잘못 접힌다 —
            # 제공자 라벨로는 shotBlocked다.
            actors = np.flatnonzero(touch[step] != 0)
            other = [
                index for index in actors
                if int(team_id[step, index]) != team
            ]
            if other:
                codes = {int(touch[step, index]) for index in other}
                label = ("saved"
                         if codes & {TOUCH_GK_CATCH, TOUCH_PARRY}
                         else "blocked")
                break
            kind = int(restart_kind[step])
            if kind in (RK_GOALKICK, RK_CORNER, RK_THROWIN):
                # 골킥·코너·스로인은 전부 "골문을 지나쳤다"는 뜻이다.
                label = "off_target"
                break
        shot_outcomes[label] += 1

    # Longest sequence of completed passes by one team without a >3s discontinuity.
    longest_chain = 0
    chain = 0
    previous_frame = -10_000
    previous_team = -1
    for frame, team in completed_events:
        if team == previous_team and frame - previous_frame <= 45:
            chain += 1
        else:
            chain = 1
        longest_chain = max(longest_chain, chain)
        previous_frame, previous_team = frame, team

    low_chain_metrics = _completed_low_pass_chain_metrics(
        completed_pass_records,
        max_gap_frames=int(round(3.0 * env.control_fps)),
    )

    transition_by_zone_public = {
        zone: {
            **counts,
            "regain_within_3s_rate": (
                counts["regains_within_3s"] / counts["regain_3s_eligible"]
                if counts["regain_3s_eligible"] else None
            ),
            "stable_2s_rate": (
                counts["stable_for_2s"] / counts["stable_2s_eligible"]
                if counts["stable_2s_eligible"] else None
            ),
        }
        for zone, counts in transition_by_zone.items()
    }

    passes = int(pass_mask.sum())
    forward_passes = int((pass_mask & (forward_component > 0.25)).sum())
    restart_duration_samples = _completed_restart_durations(
        restart_kind, env.control_fps
    )
    restart_start_counts = _restart_starts(restart_kind)
    metrics = {
        "rollout_seconds": float(restart_kind.size) / float(env.control_fps),
        "restart_starts_by_kind": restart_start_counts,
        "causal_open_kicks": int(open_kick.sum()),
        "passes": passes,
        "clearances": int(clearance_mask.sum()),
        "completed_passes": complete,
        "pass_completion_rate": complete / passes if passes else 0.0,
        "low_passes": int(low_pass_mask.sum()),
        "pass_post_kick_speed_3d_mps": _distribution_summary(
            pass_post_kick_speed_3d
        ),
        "_pass_post_kick_speed_3d_values": np.asarray(
            pass_post_kick_speed_3d, dtype=np.float64
        ),
        "forward_passes": forward_passes,
        "forward_pass_rate": forward_passes / passes if passes else 0.0,
        "direction_classified_passes": int(sum(direction_counts.values())),
        "direction_forward_passes": int(direction_counts["forward"]),
        "direction_sideways_passes": int(direction_counts["sideways"]),
        "direction_backward_passes": int(direction_counts["backward"]),
        "reference_successful_passes": int(sum(reference_success_counts.values())),
        "pass_no_following_touch": int(pass_no_following_touch),
        "pass_opponent_first": int(pass_opponent_first),
        "pass_passer_retouch": int(pass_passer_retouch),
        "pass_failure_events": pass_failure_events,
        "reference_pass_success_rate": (
            sum(reference_success_counts.values()) / passes
            if passes else 0.0
        ),
        "pass_length_m": _distribution_summary(classified_pass_distances),
        "pass_length_histogram": {
            "bins_m": _PASS_LENGTH_BINS_M.tolist(),
            "counts": np.histogram(
                classified_pass_distances, bins=_PASS_LENGTH_BINS_M
            )[0].astype(np.int64).tolist(),
        },
        "_pass_length_values": np.asarray(
            classified_pass_distances, dtype=np.float64
        ),
        "reference_forward_successful_passes": int(reference_success_counts["forward"]),
        "reference_sideways_successful_passes": int(reference_success_counts["sideways"]),
        "reference_backward_successful_passes": int(reference_success_counts["backward"]),
        "immediate_return_passes": int(immediate_return_passes),
        "receiver_next_action_samples": len(receiver_next_action_delays),
        "receiver_next_action_delay_seconds_sum": float(sum(receiver_next_action_delays)),
        "receiver_next_action_delay_median_s": (
            float(np.median(receiver_next_action_delays))
            if receiver_next_action_delays else None
        ),
        "quick_relay_opportunities": int(quick_relay_opportunities),
        "quick_relays": int(quick_relays),
        "quick_flat_relays": int(quick_flat_relays),
        "gk_passes": int((pass_mask & is_goalkeeper[None, :]).sum()),
        "gk_flat_passes": int((low_pass_mask & is_goalkeeper[None, :]).sum()),
        "gk_received_passes": int(gk_received_passes),
        "gk_buildup_cycles": int(gk_buildup_cycles),
        "possession_gains": int(possession_gains),
        "challenge_possession_gains": int(challenge_possession_gains),
        "ground_challenge_possession_gains": int(
            ground_challenge_possession_gains
        ),
        "challenge_touches": int(challenge_touch_mask.sum()),
        "challenge_control_like_touches": int(
            challenge_control_like_mask.sum()
        ),
        "challenge_outlet_like_touches": int(
            challenge_outlet_like_mask.sum()
        ),
        "defensive_clearance_like_touches": int(
            challenge_clearance_like_mask.sum()
        ),
        "challenge_other_like_touches": int(
            challenge_other_like_mask.sum()
        ),
        "challenge_retained_2s_eligible": int(
            challenge_retained_2s_eligible
        ),
        "challenge_retained_for_2s": int(challenge_retained_for_2s),
        "challenge_post_touch_speed_3d_mps": _distribution_summary(
            challenge_post_kick_speed_3d
        ),
        "_challenge_post_touch_speed_3d_values": np.asarray(
            challenge_post_kick_speed_3d, dtype=np.float64
        ),
        "challenge_gate_activations": int(challenge_gate.sum()),
        "outfield_challenge_gate_activations": int(
            outfield_challenge_gate.sum()
        ),
        "carrier_duel_gate_activations": int(carrier_duel_gate.sum()),
        "travelling_interception_gate_activations": int(
            travelling_interception_gate.sum()
        ),
        "loose_ball_gate_activations": int(loose_challenge_gate.sum()),
        "possession_transitions": int(transition_counts["events"]),
        "transition_regain_3s_eligible": int(
            transition_counts["regain_3s_eligible"]
        ),
        "transition_regains_within_3s": int(
            transition_counts["regains_within_3s"]
        ),
        "transition_stable_2s_eligible": int(
            transition_counts["stable_2s_eligible"]
        ),
        "transition_stable_for_2s": int(
            transition_counts["stable_for_2s"]
        ),
        "transition_by_zone": transition_by_zone_public,
        "carrier_support_first_distance_m": _distribution_summary(
            support_rank_distance_values[0]
        ),
        "carrier_support_second_distance_m": _distribution_summary(
            support_rank_distance_values[1]
        ),
        "carrier_support_third_distance_m": _distribution_summary(
            support_rank_distance_values[2]
        ),
        "nearest_three_supporter_speed_mps": _distribution_summary(
            nearest_three_support_speed_values
        ),
        "carrier_teammates_within_10m": _distribution_summary(
            nearby_teammate_values[10]
        ),
        "carrier_teammates_within_15m": _distribution_summary(
            nearby_teammate_values[15]
        ),
        "carrier_teammates_within_20m": _distribution_summary(
            nearby_teammate_values[20]
        ),
        "final_third_box_attackers": _distribution_summary(
            final_third_box_attacker_values
        ),
        "final_third_ahead_runners": _distribution_summary(
            final_third_ahead_runner_values
        ),
        "final_third_ball_x_m": _distribution_summary(
            final_third_ball_x_values
        ),
        "final_third_defensive_line_x_m": _distribution_summary(
            final_third_defensive_line_x_values
        ),
        "final_third_front_gap_to_onside_line_m": _distribution_summary(
            final_third_front_gap_values
        ),
        "final_third_legal_box_opportunity_frames": int(sum(
            final_third_legal_box_opportunity
        )),
        "receiver_prepass_movement_m": _distribution_summary(
            receiver_prepass_movement
        ),
        "receiver_prepass_lateral_m": _distribution_summary(
            receiver_prepass_lateral
        ),
        "receiver_prepass_forward_m": _distribution_summary(
            receiver_prepass_forward
        ),
        "_carrier_support_first_distance_values": np.asarray(
            support_rank_distance_values[0], np.float64
        ),
        "_carrier_support_second_distance_values": np.asarray(
            support_rank_distance_values[1], np.float64
        ),
        "_carrier_support_third_distance_values": np.asarray(
            support_rank_distance_values[2], np.float64
        ),
        "_nearest_three_supporter_speed_values": np.asarray(
            nearest_three_support_speed_values, np.float64
        ),
        "_carrier_teammates_within_10m_values": np.asarray(
            nearby_teammate_values[10], np.float64
        ),
        "_carrier_teammates_within_15m_values": np.asarray(
            nearby_teammate_values[15], np.float64
        ),
        "_carrier_teammates_within_20m_values": np.asarray(
            nearby_teammate_values[20], np.float64
        ),
        "_final_third_box_attacker_values": np.asarray(
            final_third_box_attacker_values, np.float64
        ),
        "_final_third_ahead_runner_values": np.asarray(
            final_third_ahead_runner_values, np.float64
        ),
        "_final_third_ball_x_values": np.asarray(
            final_third_ball_x_values, np.float64
        ),
        "_final_third_defensive_line_x_values": np.asarray(
            final_third_defensive_line_x_values, np.float64
        ),
        "_final_third_front_gap_values": np.asarray(
            final_third_front_gap_values, np.float64
        ),
        "_receiver_prepass_movement_values": np.asarray(
            receiver_prepass_movement, np.float64
        ),
        "_receiver_prepass_lateral_values": np.asarray(
            receiver_prepass_lateral, np.float64
        ),
        "_receiver_prepass_forward_values": np.asarray(
            receiver_prepass_forward, np.float64
        ),
        "completed_progressive_passes": progressive_complete,
        **low_chain_metrics,
        "completed_kill_passes": kill_complete,
        "kill_pass_to_shot": next_shot_after_kill,
        "crosses": int(cross_mask.sum()),
        "completed_crosses": cross_complete,
        "cross_opponent_first": cross_opponent_first,
        "cross_self_first": cross_self_first,
        "cross_dead_before_touch": cross_dead_before_touch,
        "cross_no_touch": cross_no_touch,
        "cross_goals": cross_goals,
        "cross_followups": cross_followups,
        "shots": int(shot_mask.sum()),
        "shot_xg": _distribution_summary(shot_xg_values),
        "_shot_xg_values": np.asarray(shot_xg_values, dtype=np.float64),
        "shot_distance_m": _distribution_summary(shot_distance_values),
        "_shot_distance_values": np.asarray(
            shot_distance_values, dtype=np.float64
        ),
        "shot_goals": shot_goals,
        "shot_outcomes": shot_outcomes,
        "engine_shot_labels": int(engine_shot_mask.sum()),
        "shot_intents": int(shot_intent.sum()),
        "shot_intents_not_applied": int((shot_intent & (~kick_applied)).sum()),
        "shot_intents_gate_blocked": int((shot_intent & (~f2b_avail)).sum()),
        "shooting_zone_frames": int(shooting_zone.sum()),
        "shooting_zone_seconds": float(shooting_zone.sum() / env.control_fps),
        "final_third_possession_frames": int(
            (controlled_live & (folded_ball_x > env.hx / 3.0)).sum()
        ),
        "curl_shots": int((shot_mask & (np.abs(spin_side) >= 0.25)).sum()),
        "dribble_touches": int(dribble_mask.sum()),
        "alive_frames": int((ball_state == BALL_ALIVE).sum()),
        "open_play_frames": int(live_open_play.sum()),
        "player_speed_mps": _distribution_summary(live_player_speed),
        "player_speed_histogram": {
            "bins_mps": _PLAYER_SPEED_BINS_MPS.tolist(),
            "counts": np.histogram(
                live_player_speed, bins=_PLAYER_SPEED_BINS_MPS
            )[0].astype(np.int64).tolist(),
        },
        "distance_per_90_m": _distribution_summary(distance_per_90_m),
        "stamina_long_workload": _distribution_summary(workload_per_player),
        "stamina_long_workload_terms": workload_terms,
        "stamina_long_drop_per_90": _distribution_summary(stamina_long_drop_per_90),
        "stamina_long_final": _distribution_summary(final_stamina_long[active_final]),
        "stamina_short_final": _distribution_summary(final_stamina_short[active_final]),
        "stamina_short_load": _distribution_summary(short_load_per_player),
        "stamina_short_recovery_factor": _distribution_summary(
            short_recovery_factor_per_player
        ),
        "stamina_locomotion_share": _distribution_summary(locomotion_share),
        "stamina_long_workload_by_role": {
            "goalkeeper": _distribution_summary(workload_per_player[charged_is_gk]),
            "outfield": _distribution_summary(workload_per_player[~charged_is_gk]),
        },
        "stamina_long_final_by_role": {
            "goalkeeper": _distribution_summary(
                final_stamina_long[active_final & (np.asarray(final_state.gk_indices) > 0)]
            ),
            "outfield": _distribution_summary(
                final_stamina_long[active_final & (np.asarray(final_state.gk_indices) == 0)]
            ),
        },
        "_stamina_long_workload_values": np.asarray(workload_per_player, np.float64),
        "_stamina_long_drop_per_90_values": np.asarray(
            stamina_long_drop_per_90, np.float64
        ),
        "_stamina_long_final_values": np.asarray(
            final_stamina_long[active_final], np.float64
        ),
        "_stamina_short_final_values": np.asarray(
            final_stamina_short[active_final], np.float64
        ),
        "_stamina_short_load_values": np.asarray(
            short_load_per_player, np.float64
        ),
        "_stamina_short_recovery_factor_values": np.asarray(
            short_recovery_factor_per_player, np.float64
        ),
        "_stamina_locomotion_share_values": np.asarray(
            locomotion_share, np.float64
        ),
        "_stamina_long_workload_gk_values": np.asarray(
            workload_per_player[charged_is_gk], np.float64
        ),
        "_stamina_long_workload_outfield_values": np.asarray(
            workload_per_player[~charged_is_gk], np.float64
        ),
        "_stamina_long_final_gk_values": np.asarray(
            final_stamina_long[active_final & (np.asarray(final_state.gk_indices) > 0)],
            np.float64,
        ),
        "_stamina_long_final_outfield_values": np.asarray(
            final_stamina_long[active_final & (np.asarray(final_state.gk_indices) == 0)],
            np.float64,
        ),
        "_stamina_charged_player_frames": int(charged.sum()),
        "_stamina_long_term_sums": {
            "idle": float(
                stamina_long_workload[charged].sum()
                - stamina_long_speed_load[charged].sum()
                - stamina_long_acceleration_load[charged].sum()
                - stamina_long_sprint_extra[charged].sum()
            ),
            "speed": float(stamina_long_speed_load[charged].sum()),
            "acceleration": float(stamina_long_acceleration_load[charged].sum()),
            "sprint": float(stamina_long_sprint_extra[charged].sum()),
        },
        "sprints_per_90": _distribution_summary(sprints_per_90),
        "sprints_per_90_by_role": {
            "goalkeeper": _distribution_summary(
                sprints_per_90[eligible_is_gk]
            ),
            "outfield": _distribution_summary(
                sprints_per_90[~eligible_is_gk]
            ),
        },
        "ball_speed_mps": _distribution_summary(live_ball_speed),
        "ball_speed_3d_mps": _distribution_summary(live_ball_speed_3d),
        "ball_speed_controlled_mps": _distribution_summary(
            controlled_ball_speed
        ),
        "ball_speed_neutral_mps": _distribution_summary(neutral_ball_speed),
        "move_power": _distribution_summary(live_move_power),
        # Exact cross-seed quantiles are built in ``main``; these arrays are
        # removed before either compact or full JSON is serialised.
        "_player_speed_values": live_player_speed,
        "_distance_per_90_values": distance_per_90_m,
        "_sprints_per_90_values": sprints_per_90,
        "_sprints_per_90_gk_values": sprints_per_90[eligible_is_gk],
        "_sprints_per_90_outfield_values": sprints_per_90[~eligible_is_gk],
        "_ball_speed_values": live_ball_speed,
        "_ball_speed_3d_values": live_ball_speed_3d,
        "_ball_speed_controlled_values": controlled_ball_speed,
        "_ball_speed_neutral_values": neutral_ball_speed,
        "_move_power_values": live_move_power,
        "restart_duration_by_kind": {
            name: _duration_summary(samples)
            for name, samples in restart_duration_samples.items()
        },
        # Kept only until the multi-seed aggregate is built in ``main``.
        # The public JSON retains compact summaries, not every raw sample.
        "_restart_duration_samples_by_kind": restart_duration_samples,
        "longest_pass_chain": longest_chain,
        "max_passer_x": float(np.max(folded_x[pass_mask])) if passes else None,
        "max_completed_progress": max(completed_progress, default=None),
        "max_reception_x": max(reception_x, default=None),
        "final_third_receptions": int(sum(x > env.hx / 3.0 for x in reception_x)),
        "kill_reception_xg_max": max(kill_reception_xg, default=None),
        "kill_reception_xg_mean": (
            float(np.mean(kill_reception_xg)) if kill_reception_xg else None
        ),
        "kill_reception_goal_distance_min": min(
            kill_reception_goal_distance, default=None
        ),
        "kill_receptions_first_time_eligible": int(sum(
            xg >= policy_config.first_time_shot_xg
            and distance < policy_config.first_time_shot_distance
            for xg, distance in zip(
                kill_reception_xg, kill_reception_goal_distance
            )
        )),
        "kill_followups": kill_followups,
        "by_team": {
            str(team): {
                "passes": int((pass_mask & (team_id == team)).sum()),
                "completed_passes": int(complete_by_team[team]),
                "low_passes": int((low_pass_mask & (team_id == team)).sum()),
                "quick_relay_opportunities": int(quick_opportunity_by_team[team]),
                "quick_relays": int(quick_relay_by_team[team]),
                "quick_flat_relays": int(quick_flat_relay_by_team[team]),
                "gk_passes": int((
                    pass_mask & (team_id == team) & is_goalkeeper[None, :]
                ).sum()),
                "gk_flat_passes": int((
                    low_pass_mask & (team_id == team) & is_goalkeeper[None, :]
                ).sum()),
                "gk_received_passes": int(gk_received_by_team[team]),
                "gk_buildup_cycles": int(gk_buildup_by_team[team]),
                "possession_gains": int(possession_gain_by_team[team]),
                "challenge_possession_gains": int(challenge_gain_by_team[team]),
                "ground_challenge_possession_gains": int(
                    ground_challenge_gain_by_team[team]
                ),
                "challenge_touches": int(challenge_touch_by_team[team]),
                "challenge_control_like_touches": int(
                    challenge_control_by_team[team]
                ),
                "challenge_outlet_like_touches": int(
                    challenge_outlet_by_team[team]
                ),
                "defensive_clearance_like_touches": int(
                    challenge_clearance_by_team[team]
                ),
                "challenge_retained_2s_eligible": int(
                    challenge_retention_eligible_by_team[team]
                ),
                "challenge_retained_for_2s": int(
                    challenge_retained_by_team[team]
                ),
                "outfield_challenge_gate_activations": int(
                    challenge_gate_by_team[team]
                ),
                "carrier_duel_gate_activations": int(
                    carrier_duel_gate_by_team[team]
                ),
                "forward_passes": int((
                    pass_mask & (team_id == team) & (forward_component > 0.25)
                ).sum()),
                "direction_forward_passes": int(direction_by_team[team]["forward"]),
                "direction_sideways_passes": int(direction_by_team[team]["sideways"]),
                "direction_backward_passes": int(direction_by_team[team]["backward"]),
                "reference_successful_passes": int(reference_success_by_team[team]),
                "completed_progressive_passes": int(progressive_by_team[team]),
                "completed_kill_passes": int(kill_by_team[team]),
                "crosses": int((cross_mask & (team_id == team)).sum()),
                "shots": int((shot_mask & (team_id == team)).sum()),
                # 팀별 집계도 득점 팀을 확인한다 — 상대 골을 우리 슛의 결과로 세면
                # 두 팀 전환율이 동시에 부풀어 비교 자체가 무의미해진다.
                "shot_goals": int(sum(
                    np.any(
                        scored[frame:min(scored.shape[0], frame + shot_horizon + 1)]
                        == team)
                    for frame, actor in zip(*np.where(
                        shot_mask & (team_id == team)
                    ))
                )),
            }
            for team in (0, 1)
        },
        "score_team_0": int(np.asarray(final_state.score)[0]),
        "score_team_1": int(np.asarray(final_state.score)[1]),
        "nonfinite_state_frames": int((~finite_state).sum()),
        "nonfinite_action_frames": int((~finite_action).sum()),
    }
    # 렌더에서 보이는 기하 이상을 수치로 대조한다. 선수는 정책과 무관하게 공유 5m margin까지
    # 허용되지만 그 밖은 위반이며, 라이브 공은 control-frame 경계에선 이미 아웃 이벤트가 처리돼야 한다.
    player_limit = np.array([
        env.hx + env.e_cfg.player_boundary_margin,
        env.hy + env.e_cfg.player_boundary_margin,
    ])
    player_violation = active_player & np.any(
        np.abs(positions) > player_limit[None, None, :] + 1e-5, axis=2
    )
    player_outside_pitch = active_player & np.any(
        np.abs(positions) > np.array([env.hx, env.hy])[None, None, :], axis=2
    )
    live_player_outside_pitch = player_outside_pitch & live_open_play[:, None]
    player_excursion = np.maximum(
        np.abs(positions) - np.array([env.hx, env.hy])[None, None, :],
        0.0,
    )
    ball_outside = (
        (np.abs(ball_position[:, 0]) > env.hx + env.r_ball + 1e-5)
        | (np.abs(ball_position[:, 1]) > env.hy + env.r_ball + 1e-5)
    )
    live_ball_outside = (ball_state == BALL_ALIVE) & ball_outside
    delta = positions[:, :, None, :] - positions[:, None, :, :]
    pair_distance = np.linalg.norm(delta, axis=3)
    pair_active = (
        active_player[:, :, None]
        & active_player[:, None, :]
        & (~np.eye(env.N, dtype=bool)[None, :, :])
    )
    active_pair_distance = np.where(pair_active, pair_distance, np.inf)
    min_pair_flat = int(np.argmin(active_pair_distance))
    min_frame, min_player_a, min_player_b = np.unravel_index(
        min_pair_flat, active_pair_distance.shape
    )
    active_excursion = np.where(
        active_player[:, :, None], player_excursion, -np.inf
    )
    max_excursion_flat = int(np.argmax(active_excursion))
    max_frame, max_player, max_axis = np.unravel_index(
        max_excursion_flat, active_excursion.shape
    )
    metrics.update({
        "player_boundary_violation_frames": int(np.any(player_violation, axis=1).sum()),
        "active_player_outside_pitch_player_frames": int(
            player_outside_pitch.sum()
        ),
        "live_active_player_outside_pitch_player_frames": int(
            live_player_outside_pitch.sum()
        ),
        "maximum_active_player_pitch_excursion_m": float(
            np.max(np.where(active_player[:, :, None], player_excursion, 0.0))
        ),
        "live_ball_outside_frames": int(live_ball_outside.sum()),
        "minimum_active_player_separation_m": float(
            active_pair_distance[min_frame, min_player_a, min_player_b]
        ),
        "minimum_separation_event": {
            "second": round(float(min_frame) / env.control_fps, 3),
            "players": [int(min_player_a), int(min_player_b)],
            "positions_m": [
                np.round(positions[min_frame, min_player_a], 4).tolist(),
                np.round(positions[min_frame, min_player_b], 4).tolist(),
            ],
            "restart_kind": int(restart_kind[min_frame]),
            "ball_alive": bool(ball_state[min_frame] == BALL_ALIVE),
        },
        "maximum_excursion_event": {
            "second": round(float(max_frame) / env.control_fps, 3),
            "player": int(max_player),
            "axis": "x" if max_axis == 0 else "y",
            "position_m": np.round(positions[max_frame, max_player], 4).tolist(),
            "restart_kind": int(restart_kind[max_frame]),
            "ball_alive": bool(ball_state[max_frame] == BALL_ALIVE),
        },
        "restart_projection_frames": int(
            np.any(restart_position_forced, axis=1).sum()
        ),
        "cross_event_seconds": [
            round(float(frame) / env.control_fps, 3)
            for frame in np.flatnonzero(np.any(cross_mask, axis=1))
        ],
        "cross_events": [
            {
                "second": round(float(frame) / env.control_fps, 3),
                "actor": int(actor),
                "folded_x_m": round(float(folded_x[frame, actor]), 3),
                "folded_y_m": round(float(folded_y[frame, actor]), 3),
                "power": round(float(kick_power[frame, actor]), 3),
                "launch_rad": round(float(launch[frame, actor]), 3),
            }
            for frame, actor in zip(*np.where(cross_mask))
        ],
        "curl_shot_event_seconds": [
            round(float(frame) / env.control_fps, 3)
            for frame in np.flatnonzero(np.any(shot_mask, axis=1))
        ],
        "curl_shot_events": [
            {
                "second": round(float(frame) / env.control_fps, 3),
                "actor": int(actor),
                "folded_x_m": round(float(folded_x[frame, actor]), 3),
                "folded_y_m": round(float(folded_y[frame, actor]), 3),
                "folded_dir_x": round(
                    float(kick_dir[frame, actor, 0] * attack_dir[frame, actor]), 4
                ),
                "folded_dir_y": round(
                    float(kick_dir[frame, actor, 1] * attack_dir[frame, actor]), 4
                ),
                "power": round(float(kick_power[frame, actor]), 3),
                "launch_rad": round(float(launch[frame, actor]), 3),
                "spin_side": round(float(spin_side[frame, actor]), 3),
                "spin_back": round(float(spin_back[frame, actor]), 3),
                "xg": round(float(player_xg[frame, actor]), 4),
                "goal_distance_m": round(float(np.linalg.norm(
                    np.array([env.hx, 0.0])
                    - positions[frame, actor] * attack_dir[frame, actor]
                )), 3),
                # 이름이 4초인 창을 프레임 수로 박으면 control_fps를 바꾼 순간
                # 이름과 값이 어긋난다.
                "goal_within_4s": bool(np.any(
                    scored[frame:min(scored.shape[0], frame + shot_horizon + 1)]
                    == int(team_id[frame, actor])
                )),
            }
            for frame, actor in zip(*np.where(shot_mask))
        ],
    })
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Profile causal rule-policy tactics")
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4300)
    parser.add_argument("--style-a", default="balanced")
    parser.add_argument("--style-b", default="balanced")
    parser.add_argument(
        "--pass-completion-floor", type=float, default=None,
        help="override RulePolicy.pass_completion_floor for a paired calibration run",
    )
    parser.add_argument(
        "--through-completion-floor", type=float, default=None,
        help="override RulePolicy.through_completion_floor for a paired calibration run",
    )
    parser.add_argument(
        "--pass-release-base-probability", type=float, default=None,
        help="override the ordinary per-frame pass-release hazard",
    )
    parser.add_argument(
        "--pass-release-pressure-gain", type=float, default=None,
        help="override the pressure-conditioned pass-release hazard gain",
    )
    parser.add_argument(
        "--contact-release-probability", type=float, default=None,
        help="override the per-frame safe outlet probability under close contact",
    )
    parser.add_argument(
        "--quick-relay-probability", type=float, default=None,
        help="override the safe first-time low-relay probability",
    )
    parser.add_argument(
        "--quick-relay-max-incoming-speed-mps", type=float, default=None,
        help="override the fastest pass eligible for a first-time relay",
    )
    parser.add_argument(
        "--quick-relay-low-skill-speed-fraction", type=float, default=None,
        help="override the lower-pro share of the first-time-pass speed band",
    )
    parser.add_argument(
        "--professional-control-floor", type=float, default=None,
        help="override the lower-pro saturation point for ball control",
    )
    parser.add_argument(
        "--professional-control-ceiling", type=float, default=None,
        help="override the elite-pro saturation point for ball control",
    )
    parser.add_argument(
        "--receiver-control-value-gain", type=float, default=None,
        help="override receiver-control influence on pass selection",
    )
    parser.add_argument(
        "--cross-receiver-ability-gain", type=float, default=None,
        help="override control/reach influence on cross-target selection",
    )
    parser.add_argument(
        "--dribble-control-retention-gain", type=float, default=None,
        help="override control influence on predicted dribble retention",
    )
    parser.add_argument(
        "--execution-control-noise-gain", type=float, default=None,
        help="override control influence on horizontal/vertical kick error",
    )
    parser.add_argument(
        "--pass-target-distance-penalty", type=float, default=None,
        help="override the receiver-selection-only long target penalty",
    )
    parser.add_argument(
        "--pass-progress-base", type=float, default=None,
        help="override the receiver-value reward for forward progress",
    )
    parser.add_argument(
        "--pass-progress-direct-gain", type=float, default=None,
        help="override the directness-dependent forward-progress reward",
    )
    parser.add_argument(
        "--pass-short-forward-bonus", type=float, default=None,
        help="override the possession-style short forward pass bonus",
    )
    parser.add_argument(
        "--shot-min-xg", type=float, default=None,
        help="override the minimum xG required for an open-play shot",
    )
    parser.add_argument(
        "--shot-commit-xg", type=float, default=None,
        help="override the xG above which a shot commits over a pass",
    )
    parser.add_argument(
        "--first-time-shot-xg", type=float, default=None,
        help="override the minimum xG for a first-time shot or volley",
    )
    parser.add_argument(
        "--arrival-radius", type=float, default=None,
        help="override the off-ball arrival slowdown radius",
    )
    parser.add_argument(
        "--offball-cruise", type=float, default=None,
        help="override ordinary off-ball cruise power",
    )
    parser.add_argument(
        "--offball-surge-cap", type=float, default=None,
        help="override recovery/run-ahead off-ball power cap",
    )
    parser.add_argument(
        "--support-progression-blend", type=float, default=None,
        help="override lateral-to-forward combination-support geometry blend",
    )
    parser.add_argument(
        "--support-progression-directness-start", type=float, default=None,
        help="override directness where forward support blending begins",
    )
    parser.add_argument(
        "--support-progression-directness-full", type=float, default=None,
        help="override directness where forward support blending reaches its maximum",
    )
    parser.add_argument(
        "--support-run-power", type=float, default=None,
        help="override active combination-support movement power",
    )
    parser.add_argument(
        "--carrier-chase-power", type=float, default=None,
        help="override the far-ball carrier recovery power",
    )
    parser.add_argument(
        "--press-run-power", type=float, default=None,
        help="override the primary pressure runner power",
    )
    parser.add_argument(
        "--receive-run-power", type=float, default=None,
        help="override the predictive receiver run power",
    )
    parser.add_argument(
        "--loose-run-power", type=float, default=None,
        help="override the nearest neutral-ball chaser power",
    )
    parser.add_argument(
        "--aerial-run-power", type=float, default=None,
        help="override the nearest aerial-ball chaser power",
    )
    parser.add_argument(
        "--aerial-cover-run-power", type=float, default=None,
        help="override the late defender's second-ball cover power",
    )
    parser.add_argument(
        "--aerial-duel-eta-window-s", type=float, default=None,
        help="override the maximum defender ETA deficit for a direct aerial duel",
    )
    parser.add_argument(
        "--aerial-defender-cover-distance", type=float, default=None,
        help="override the goal-side offset for a late aerial defender",
    )
    parser.add_argument(
        "--aerial-pass-min-distance", type=float, default=None,
        help="override the minimum headed team-pass distance",
    )
    parser.add_argument(
        "--aerial-pass-max-distance", type=float, default=None,
        help="override the maximum headed team-pass distance",
    )
    parser.add_argument(
        "--aerial-pass-completion-floor", type=float, default=None,
        help="override the headed team-pass lane/ETA quality floor",
    )
    parser.add_argument(
        "--aerial-pass-min-speed-mps", type=float, default=None,
        help="override the shortest headed team-pass exit speed",
    )
    parser.add_argument(
        "--aerial-pass-max-speed-mps", type=float, default=None,
        help="override the longest headed team-pass exit speed",
    )
    parser.add_argument(
        "--aerial-shot-speed-mps", type=float, default=None,
        help="override the attacking header exit speed",
    )
    parser.add_argument(
        "--aerial-clear-speed-mps", type=float, default=None,
        help="override the aerial clearance exit speed",
    )
    parser.add_argument(
        "--pressured-aerial-control-max-height", type=float, default=None,
        help="override the low-bounce height still controlled under pressure",
    )
    parser.add_argument(
        "--tackle-hazard-contact-gain-per-s", type=float, default=None,
        help="override the close-contact-only carrier tackle hazard gain",
    )
    parser.add_argument(
        "--drive-arrive-speed-mps", type=float, default=None,
        help="override the ground-pass solver's target arrival speed",
    )
    parser.add_argument(
        "--ground-control-touch-speed-mps", type=float, default=None,
        help="override the ground trap/loose-ball control touch speed",
    )
    parser.add_argument(
        "--receive-control-lead-speed-mps", type=float, default=None,
        help="override the running receiver's first-touch lead over player speed",
    )
    parser.add_argument(
        "--challenge-control-touch-speed-mps", type=float, default=None,
        help="override the soft first-touch speed after a ground challenge",
    )
    parser.add_argument(
        "--challenge-outlet-max-distance", type=float, default=None,
        help="override the maximum immediate low outlet distance after a challenge",
    )
    parser.add_argument(
        "--challenge-outlet-completion-floor", type=float, default=None,
        help="override the predicted completion floor for a challenge outlet",
    )
    parser.add_argument(
        "--challenge-outlet-max-incoming-speed-mps", type=float, default=None,
        help="override the incoming-speed ceiling for a one-touch challenge outlet",
    )
    parser.add_argument(
        "--challenge-clearance-depth-fraction", type=float, default=None,
        help="override the own-end depth threshold for emergency clearances",
    )
    parser.add_argument(
        "--challenge-clearance-pressure-min", type=float, default=None,
        help="override the pressure threshold for emergency clearances",
    )
    parser.add_argument(
        "--gk-buildup-gain", type=float, default=None,
        help="override the own-half goalkeeper build-up value bonus",
    )
    parser.add_argument(
        "--gk-long-directness-threshold", type=float, default=None,
        help="override the directness threshold for long GK distribution",
    )
    parser.add_argument(
        "--loft-distance-base", type=float, default=None,
        help="override the possession-style loft-pass distance threshold",
    )
    parser.add_argument(
        "--loft-force-directness", type=float, default=None,
        help="override directness threshold that deliberately lofts a safe long route",
    )
    parser.add_argument(
        "--cross-preference-ratio", type=float, default=None,
        help="override cross value required relative to an ordinary pass",
    )
    parser.add_argument(
        "--dribble-evade-pressure", type=float, default=None,
        help="override pressure that activates multidirectional dribbling",
    )
    parser.add_argument(
        "--box-mark-max-runners", type=int, default=None,
        help="override the maximum distinct runners marked around the box",
    )
    parser.add_argument(
        "--box-mark-runner-margin", type=float, default=None,
        help="override the pre-box runner recognition margin in metres",
    )
    parser.add_argument(
        "--box-mark-ball-margin", type=float, default=None,
        help="override the ball-distance margin activating multi-runner marking",
    )
    parser.add_argument(
        "--box-mark-goal-side-distance", type=float, default=None,
        help="override each marker's goal-side offset in metres",
    )
    parser.add_argument(
        "--total-only", action="store_true",
        help="omit per-seed event details and print only aggregate metrics",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="atomically write the same profile JSON to this path",
    )
    args = parser.parse_args()
    if args.seconds <= 0 or args.seeds <= 0:
        parser.error("--seconds and --seeds must be positive")

    steps = DEFAULT_TIMEBASE.control_steps_for(args.seconds, minimum=1)
    env = SoccerEnv(game_duration=steps, halftime=False)
    overrides = {
        name: value
        for name, value in (
            ("pass_completion_floor", args.pass_completion_floor),
            ("through_completion_floor", args.through_completion_floor),
            ("pass_release_base_probability", args.pass_release_base_probability),
            ("pass_release_pressure_gain", args.pass_release_pressure_gain),
            ("contact_release_probability", args.contact_release_probability),
            ("quick_relay_probability", args.quick_relay_probability),
            (
                "quick_relay_max_incoming_speed_mps",
                args.quick_relay_max_incoming_speed_mps,
            ),
            (
                "quick_relay_low_skill_speed_fraction",
                args.quick_relay_low_skill_speed_fraction,
            ),
            ("professional_control_floor", args.professional_control_floor),
            (
                "professional_control_ceiling",
                args.professional_control_ceiling,
            ),
            (
                "receiver_control_value_gain",
                args.receiver_control_value_gain,
            ),
            (
                "cross_receiver_ability_gain",
                args.cross_receiver_ability_gain,
            ),
            (
                "dribble_control_retention_gain",
                args.dribble_control_retention_gain,
            ),
            (
                "execution_control_noise_gain",
                args.execution_control_noise_gain,
            ),
            ("pass_target_distance_penalty", args.pass_target_distance_penalty),
            ("pass_progress_base", args.pass_progress_base),
            ("pass_progress_direct_gain", args.pass_progress_direct_gain),
            ("pass_short_forward_bonus", args.pass_short_forward_bonus),
            ("shot_min_xg", args.shot_min_xg),
            ("shot_commit_xg", args.shot_commit_xg),
            ("first_time_shot_xg", args.first_time_shot_xg),
            ("arrival_radius", args.arrival_radius),
            ("offball_cruise", args.offball_cruise),
            ("offball_surge_cap", args.offball_surge_cap),
            ("support_progression_blend", args.support_progression_blend),
            (
                "support_progression_directness_start",
                args.support_progression_directness_start,
            ),
            (
                "support_progression_directness_full",
                args.support_progression_directness_full,
            ),
            ("support_run_power", args.support_run_power),
            ("carrier_chase_power", args.carrier_chase_power),
            ("press_run_power", args.press_run_power),
            ("receive_run_power", args.receive_run_power),
            ("loose_run_power", args.loose_run_power),
            ("aerial_run_power", args.aerial_run_power),
            ("aerial_cover_run_power", args.aerial_cover_run_power),
            ("aerial_duel_eta_window_s", args.aerial_duel_eta_window_s),
            (
                "aerial_defender_cover_distance",
                args.aerial_defender_cover_distance,
            ),
            ("aerial_pass_min_distance", args.aerial_pass_min_distance),
            ("aerial_pass_max_distance", args.aerial_pass_max_distance),
            (
                "aerial_pass_completion_floor",
                args.aerial_pass_completion_floor,
            ),
            ("aerial_pass_min_speed_mps", args.aerial_pass_min_speed_mps),
            ("aerial_pass_max_speed_mps", args.aerial_pass_max_speed_mps),
            ("aerial_shot_speed_mps", args.aerial_shot_speed_mps),
            ("aerial_clear_speed_mps", args.aerial_clear_speed_mps),
            (
                "pressured_aerial_control_max_height",
                args.pressured_aerial_control_max_height,
            ),
            (
                "tackle_hazard_contact_gain_per_s",
                args.tackle_hazard_contact_gain_per_s,
            ),
            ("drive_arrive_speed_mps", args.drive_arrive_speed_mps),
            (
                "ground_control_touch_speed_mps",
                args.ground_control_touch_speed_mps,
            ),
            (
                "receive_control_lead_speed_mps",
                args.receive_control_lead_speed_mps,
            ),
            (
                "challenge_control_touch_speed_mps",
                args.challenge_control_touch_speed_mps,
            ),
            (
                "challenge_outlet_max_distance",
                args.challenge_outlet_max_distance,
            ),
            (
                "challenge_outlet_completion_floor",
                args.challenge_outlet_completion_floor,
            ),
            (
                "challenge_outlet_max_incoming_speed_mps",
                args.challenge_outlet_max_incoming_speed_mps,
            ),
            (
                "challenge_clearance_depth_fraction",
                args.challenge_clearance_depth_fraction,
            ),
            (
                "challenge_clearance_pressure_min",
                args.challenge_clearance_pressure_min,
            ),
            ("gk_buildup_gain", args.gk_buildup_gain),
            (
                "gk_long_directness_threshold",
                args.gk_long_directness_threshold,
            ),
            ("loft_distance_base", args.loft_distance_base),
            ("loft_force_directness", args.loft_force_directness),
            ("cross_preference_ratio", args.cross_preference_ratio),
            ("dribble_evade_pressure", args.dribble_evade_pressure),
            ("box_mark_max_runners", args.box_mark_max_runners),
            ("box_mark_runner_margin", args.box_mark_runner_margin),
            ("box_mark_ball_margin", args.box_mark_ball_margin),
            (
                "box_mark_goal_side_distance",
                args.box_mark_goal_side_distance,
            ),
        )
        if value is not None
    }
    try:
        policy_config = replace(RulePolicy(), **overrides)
    except ValueError as exc:
        parser.error(str(exc))
    policy = make_rule_based_policy(
        env,
        match_key=jax.random.PRNGKey(args.seed),
        team_styles=(args.style_a, args.style_b),
        policy_config=policy_config,
    )

    def body(carry, key):
        obs, state = carry
        policy_key, env_key = jax.random.split(key)
        aff = env.affordance_view(state)
        action = policy(obs, policy_key, aff)
        (
            want_f2b,
            _,
            move_power,
            kick_dir,
            kick_power,
            launch,
            spin_side,
            spin_back,
        ) = env._decode(action, state.attack_dir)
        # `_decode`의 launch는 [0, launch_max] action parameter다. 실제 물리각은 접촉 높이에
        # 따른 launch_lo(z) 재매핑 뒤 결정되므로, rad라는 이름으로 전자를 보고하면 낮게 뜬 공의
        # 원터치 슛과 지면 킥을 잘못 비교하게 된다.
        launch_floor = env.launch_lo(state.ball_pos[2])
        physical_launch = launch_floor + (
            launch / env.e_cfg.launch_max
        ) * (env.e_cfg.launch_max - launch_floor)
        folded_players = state.player_pos * state.attack_dir[:, None]
        folded_all = (
            state.player_pos[None, :, :]
            * state.attack_dir[:, None, None]
        )
        opponent = (
            (state.team_id[:, None] != state.team_id[None, :])
            & state.active_player[None, :]
        )
        goalkeeper = jnp.broadcast_to(
            state.gk_indices[None, :] > 0, opponent.shape
        )
        player_xg = T.shot_xg(
            folded_players,
            folded_all,
            opponent,
            goalkeeper,
            env.hx,
            env.goal_w,
        )
        obs2, state2, _, _, info = env.step_env_array(env_key, state, action)
        out = (
            state.player_pos,
            state.player_vel,
            state.attack_dir,
            state.team_id,
            state.poss_team,
            state.restart_kind,
            kick_dir,
            kick_power,
            physical_launch,
            spin_side,
            spin_back,
            info["kick_applied"],
            want_f2b,
            aff["f2b_avail"] > 0.5,
            player_xg,
            info["touch"],
            info["scored"],
            (
                jnp.all(jnp.isfinite(state.player_pos))
                & jnp.all(jnp.isfinite(state.player_vel))
                & jnp.all(jnp.isfinite(state.ball_pos))
                & jnp.all(jnp.isfinite(state.ball_vel))
            ),
            jnp.all(jnp.isfinite(action)),
            state.ball_pos,
            state.ball_vel,
            state.ball_state,
            state.active_player,
            state.cooldown,
            state.reach_z,
            info["restart_position_forced"],
            move_power,
            state2.stamina_long,
            state2.stamina_short,
            info["stamina_long_workload"],
            info["stamina_long_speed_load"],
            info["stamina_long_acceleration_load"],
            info["stamina_long_sprint_extra"],
            info["stamina_short_load"],
            info["stamina_short_recovery_factor"],
            info["stamina_locomotion_seconds"],
        )
        return (obs2, state2), out

    rollout = jax.jit(
        lambda obs, state, keys: jax.lax.scan(body, (obs, state), keys)
    )
    runs = []
    started = time.perf_counter()
    for offset in range(args.seeds):
        seed = args.seed + 10 + offset
        reset_key, rollout_key = jax.random.split(jax.random.PRNGKey(seed))
        obs, state = env.reset_array(reset_key)
        (_, final_state), out = rollout(
            obs, state, prefix_stable_keys(rollout_key, steps)
        )
        final_state.ball_pos.block_until_ready()
        metrics = analyse_rollout(out, final_state, env, policy.policy_config)
        metrics["seed"] = seed
        runs.append(metrics)

    total = {}
    count_keys = (
        "causal_open_kicks", "passes", "clearances", "completed_passes", "low_passes",
        "forward_passes",
        "direction_classified_passes", "direction_forward_passes",
        "direction_sideways_passes", "direction_backward_passes",
        "reference_successful_passes", "reference_forward_successful_passes",
        "reference_sideways_successful_passes", "reference_backward_successful_passes",
        "pass_no_following_touch", "pass_opponent_first", "pass_passer_retouch",
        "immediate_return_passes", "receiver_next_action_samples",
        "quick_relay_opportunities", "quick_relays", "quick_flat_relays",
        "gk_passes", "gk_flat_passes", "gk_received_passes",
        "gk_buildup_cycles", "possession_gains",
        "challenge_possession_gains", "ground_challenge_possession_gains",
        "challenge_touches", "challenge_control_like_touches",
        "challenge_outlet_like_touches", "defensive_clearance_like_touches",
        "challenge_other_like_touches", "challenge_retained_2s_eligible",
        "challenge_retained_for_2s",
        "challenge_gate_activations", "outfield_challenge_gate_activations",
        "carrier_duel_gate_activations",
        "travelling_interception_gate_activations",
        "loose_ball_gate_activations",
        "possession_transitions", "transition_regain_3s_eligible",
        "transition_regains_within_3s", "transition_stable_2s_eligible",
        "transition_stable_for_2s",
        "completed_progressive_passes", "completed_low_passes",
        "completed_low_progressive_passes", "low_multi_pass_chains",
        "low_positive_chains", "low_progressive_chains",
        "completed_kill_passes", "kill_pass_to_shot",
        "crosses", "completed_crosses", "cross_opponent_first", "cross_self_first",
        "cross_dead_before_touch", "cross_no_touch", "cross_goals",
        "shots", "shot_goals", "curl_shots", "engine_shot_labels", "shot_intents",
        "shot_intents_not_applied", "shot_intents_gate_blocked", "shooting_zone_frames",
        "final_third_possession_frames", "dribble_touches", "alive_frames",
        "open_play_frames",
        "final_third_legal_box_opportunity_frames",
        "final_third_receptions",
        "score_team_0", "score_team_1",
        "nonfinite_state_frames", "nonfinite_action_frames",
        "player_boundary_violation_frames",
        "active_player_outside_pitch_player_frames",
        "live_active_player_outside_pitch_player_frames",
        "live_ball_outside_frames", "restart_projection_frames",
    )
    for key in count_keys:
        total[key] = int(sum(run[key] for run in runs))
    team_count_keys = (
        "passes", "completed_passes", "low_passes", "forward_passes",
        "direction_forward_passes", "direction_sideways_passes",
        "direction_backward_passes",
        "reference_successful_passes",
        "completed_progressive_passes", "completed_kill_passes",
        "quick_relay_opportunities", "quick_relays", "quick_flat_relays",
        "gk_passes", "gk_flat_passes", "gk_received_passes",
        "gk_buildup_cycles", "possession_gains",
        "challenge_possession_gains", "ground_challenge_possession_gains",
        "challenge_touches", "challenge_control_like_touches",
        "challenge_outlet_like_touches", "defensive_clearance_like_touches",
        "challenge_retained_2s_eligible", "challenge_retained_for_2s",
        "outfield_challenge_gate_activations",
        "carrier_duel_gate_activations",
        "crosses", "shots", "shot_goals",
    )
    total["by_team"] = {
        str(team): {
            key: int(sum(run["by_team"][str(team)][key] for run in runs))
            for key in team_count_keys
        }
        for team in (0, 1)
    }
    total["by_team"]["0"]["style"] = args.style_a
    total["by_team"]["1"]["style"] = args.style_b
    total["pass_completion_rate"] = (
        total["completed_passes"] / total["passes"] if total["passes"] else 0.0
    )
    total["low_pass_share"] = (
        total["low_passes"] / total["passes"] if total["passes"] else 0.0
    )
    total["quick_relay_rate"] = (
        total["quick_relays"] / total["quick_relay_opportunities"]
        if total["quick_relay_opportunities"] else 0.0
    )
    total["quick_flat_relay_rate"] = (
        total["quick_flat_relays"] / total["quick_relay_opportunities"]
        if total["quick_relay_opportunities"] else 0.0
    )
    total["completed_low_pass_rate"] = (
        total["completed_low_passes"] / total["low_passes"]
        if total["low_passes"] else 0.0
    )
    total["low_positive_chain_rate"] = (
        total["low_positive_chains"] / total["low_multi_pass_chains"]
        if total["low_multi_pass_chains"] else 0.0
    )
    total["low_progressive_chain_rate"] = (
        total["low_progressive_chains"] / total["low_multi_pass_chains"]
        if total["low_multi_pass_chains"] else 0.0
    )
    total["gk_pass_share"] = (
        total["gk_passes"] / total["passes"] if total["passes"] else 0.0
    )
    total["gk_flat_pass_share"] = (
        total["gk_flat_passes"] / total["gk_passes"]
        if total["gk_passes"] else 0.0
    )
    total["forward_pass_rate"] = (
        total["forward_passes"] / total["passes"] if total["passes"] else 0.0
    )
    total["direction_forward_pass_rate"] = (
        total["direction_forward_passes"] / total["direction_classified_passes"]
        if total["direction_classified_passes"] else 0.0
    )
    total["direction_sideways_pass_rate"] = (
        total["direction_sideways_passes"] / total["direction_classified_passes"]
        if total["direction_classified_passes"] else 0.0
    )
    total["direction_backward_pass_rate"] = (
        total["direction_backward_passes"] / total["direction_classified_passes"]
        if total["direction_classified_passes"] else 0.0
    )
    total["reference_pass_success_rate"] = (
        total["reference_successful_passes"] / total["passes"]
        if total["passes"] else 0.0
    )
    total["final_third_legal_box_opportunity_rate"] = (
        total["final_third_legal_box_opportunity_frames"]
        / total["final_third_possession_frames"]
        if total["final_third_possession_frames"] else None
    )
    total["transition_regain_within_3s_rate"] = (
        total["transition_regains_within_3s"]
        / total["transition_regain_3s_eligible"]
        if total["transition_regain_3s_eligible"] else None
    )
    total["transition_stable_2s_rate"] = (
        total["transition_stable_for_2s"]
        / total["transition_stable_2s_eligible"]
        if total["transition_stable_2s_eligible"] else None
    )
    transition_zone_count_keys = (
        "events", "regain_3s_eligible", "regains_within_3s",
        "stable_2s_eligible", "stable_for_2s",
    )
    total["transition_by_zone"] = {}
    for zone in ("own_half", "attacking_half", "final_third"):
        counts = {
            key: int(sum(
                run["transition_by_zone"][zone][key] for run in runs
            ))
            for key in transition_zone_count_keys
        }
        total["transition_by_zone"][zone] = {
            **counts,
            "regain_within_3s_rate": (
                counts["regains_within_3s"] / counts["regain_3s_eligible"]
                if counts["regain_3s_eligible"] else None
            ),
            "stable_2s_rate": (
                counts["stable_for_2s"] / counts["stable_2s_eligible"]
                if counts["stable_2s_eligible"] else None
            ),
        }
    total["immediate_return_rate"] = (
        total["immediate_return_passes"] / total["completed_passes"]
        if total["completed_passes"] else 0.0
    )
    total["receiver_next_action_delay_mean_s"] = (
        sum(run["receiver_next_action_delay_seconds_sum"] for run in runs)
        / total["receiver_next_action_samples"]
        if total["receiver_next_action_samples"] else None
    )
    total["max_passer_x"] = max(
        (run["max_passer_x"] for run in runs if run["max_passer_x"] is not None),
        default=None,
    )
    total["max_completed_progress"] = max(
        (run["max_completed_progress"] for run in runs if run["max_completed_progress"] is not None),
        default=None,
    )
    total["max_reception_x"] = max(
        (run["max_reception_x"] for run in runs if run["max_reception_x"] is not None),
        default=None,
    )
    total["longest_pass_chain"] = max(run["longest_pass_chain"] for run in runs)
    total["longest_low_pass_chain"] = max(
        run["longest_low_pass_chain"] for run in runs
    )
    kill_xg = [
        run["kill_reception_xg_max"] for run in runs
        if run["kill_reception_xg_max"] is not None
    ]
    total["kill_reception_xg_max"] = max(kill_xg, default=None)
    weighted_kill_xg = [
        (run["kill_reception_xg_mean"], run["completed_kill_passes"])
        for run in runs if run["kill_reception_xg_mean"] is not None
    ]
    total["kill_reception_xg_mean"] = (
        sum(mean * count for mean, count in weighted_kill_xg)
        / sum(count for _, count in weighted_kill_xg)
        if weighted_kill_xg else None
    )
    kill_distance = [
        run["kill_reception_goal_distance_min"] for run in runs
        if run["kill_reception_goal_distance_min"] is not None
    ]
    total["kill_reception_goal_distance_min"] = min(
        kill_distance, default=None
    )
    total["kill_receptions_first_time_eligible"] = sum(
        run["kill_receptions_first_time_eligible"] for run in runs
    )
    minimum_run = min(
        runs, key=lambda run: run["minimum_active_player_separation_m"]
    )
    maximum_excursion_run = max(
        runs, key=lambda run: run["maximum_active_player_pitch_excursion_m"]
    )
    total["minimum_active_player_separation_m"] = minimum_run[
        "minimum_active_player_separation_m"
    ]
    total["minimum_separation_event"] = {
        "seed": minimum_run["seed"],
        **minimum_run["minimum_separation_event"],
    }
    total["maximum_active_player_pitch_excursion_m"] = maximum_excursion_run[
        "maximum_active_player_pitch_excursion_m"
    ]
    total["maximum_excursion_event"] = {
        "seed": maximum_excursion_run["seed"],
        **maximum_excursion_run["maximum_excursion_event"],
    }
    aggregate_restart_samples = {
        name: [
            duration
            for run in runs
            for duration in run["_restart_duration_samples_by_kind"].get(
                name, ()
            )
        ]
        for name in _RESTART_KIND_NAMES.values()
    }
    total["restart_duration_by_kind"] = {
        name: _duration_summary(samples)
        for name, samples in aggregate_restart_samples.items()
    }
    # 재개율(90분 환산)과 K리그 실측 대비 배율. 롤아웃 길이를 시드마다 합쳐서
    # 정규화하므로 짧은 시드가 섞여도 가중이 맞는다.
    event_rates = _load_event_rates()
    outcome_totals = {
        name: sum(int(run["shot_outcomes"].get(name, 0)) for run in runs)
        for name in ("goal", "saved", "blocked", "off_target", "unresolved")
    }
    resolved = sum(outcome_totals.values())
    total["shot_outcomes"] = outcome_totals
    total["shot_outcome_share"] = {
        name: (count / resolved if resolved else 0.0)
        for name, count in outcome_totals.items()
    }
    # K리그 실측 비율은 event_rates.json이 단일 진실원천이다 — 여기에 숫자를 박으면
    # 자료를 다시 적합해도 비교 기준이 낡은 채로 남는다. 제공자 라벨을 env 결말로
    # 접는다: 유효슛=선방, 골대맞음·빗나감=골대밖.
    reference_shares = None
    if event_rates is not None:
        provider = {
            name: float(event_rates[f"shot_{name}"]["mean"])
            for name in ("goal", "shotOnTarget", "shotOffTarget", "shotBlocked",
                         "shotMissed", "shotHitGoalpost")
            if f"shot_{name}" in event_rates
        }
        folded = {
            "goal": provider.get("goal", 0.0),
            "saved": provider.get("shotOnTarget", 0.0),
            "blocked": provider.get("shotBlocked", 0.0),
            "off_target": (provider.get("shotOffTarget", 0.0)
                           + provider.get("shotMissed", 0.0)
                           + provider.get("shotHitGoalpost", 0.0)),
        }
        grand = sum(folded.values())
        if grand > 0.0:
            reference_shares = {k: v / grand for k, v in folded.items()}
    total["shot_outcome_share_kleague"] = reference_shares
    rollout_seconds = sum(float(run["rollout_seconds"]) for run in runs)
    restart_totals = {
        name: sum(int(run["restart_starts_by_kind"].get(name, 0)) for run in runs)
        for name in _RESTART_KIND_NAMES.values()
    }
    scale = 5400.0 / rollout_seconds if rollout_seconds > 0.0 else 0.0
    total["restarts_per_90_by_kind"] = {
        name: count * scale for name, count in restart_totals.items()
    }
    if event_rates is not None:
        ratios = {}
        for name, labels in _KLEAGUE_RESTART_LABEL.items():
            reference = sum(
                float(event_rates[label]["mean"])
                for label in labels if label in event_rates
            )
            if reference > 0.0:
                ratios[name] = total["restarts_per_90_by_kind"][name] / reference
        total["restarts_per_90_ratio_to_kleague"] = ratios
        total["restarts_per_90_kleague_reference"] = {
            name: sum(
                float(event_rates[label]["mean"])
                for label in labels if label in event_rates
            )
            for name, labels in _KLEAGUE_RESTART_LABEL.items()
        }
    raw_distribution_keys = {
        "pass_length_m": "_pass_length_values",
        "pass_post_kick_speed_3d_mps": "_pass_post_kick_speed_3d_values",
        "low_chain_net_progress_m": "_low_chain_net_progress_values",
        "shot_xg": "_shot_xg_values",
        "shot_distance_m": "_shot_distance_values",
        "player_speed_mps": "_player_speed_values",
        "distance_per_90_m": "_distance_per_90_values",
        "sprints_per_90": "_sprints_per_90_values",
        "ball_speed_mps": "_ball_speed_values",
        "ball_speed_3d_mps": "_ball_speed_3d_values",
        "ball_speed_controlled_mps": "_ball_speed_controlled_values",
        "ball_speed_neutral_mps": "_ball_speed_neutral_values",
        "move_power": "_move_power_values",
        "stamina_long_workload": "_stamina_long_workload_values",
        "stamina_long_drop_per_90": "_stamina_long_drop_per_90_values",
        "stamina_long_final": "_stamina_long_final_values",
        "stamina_short_final": "_stamina_short_final_values",
        "stamina_short_load": "_stamina_short_load_values",
        "stamina_short_recovery_factor": "_stamina_short_recovery_factor_values",
        "stamina_locomotion_share": "_stamina_locomotion_share_values",
        "carrier_support_first_distance_m":
            "_carrier_support_first_distance_values",
        "carrier_support_second_distance_m":
            "_carrier_support_second_distance_values",
        "carrier_support_third_distance_m":
            "_carrier_support_third_distance_values",
        "nearest_three_supporter_speed_mps":
            "_nearest_three_supporter_speed_values",
        "carrier_teammates_within_10m":
            "_carrier_teammates_within_10m_values",
        "carrier_teammates_within_15m":
            "_carrier_teammates_within_15m_values",
        "carrier_teammates_within_20m":
            "_carrier_teammates_within_20m_values",
        "final_third_box_attackers": "_final_third_box_attacker_values",
        "final_third_ahead_runners": "_final_third_ahead_runner_values",
        "final_third_ball_x_m": "_final_third_ball_x_values",
        "final_third_defensive_line_x_m":
            "_final_third_defensive_line_x_values",
        "final_third_front_gap_to_onside_line_m":
            "_final_third_front_gap_values",
        "receiver_prepass_movement_m": "_receiver_prepass_movement_values",
        "receiver_prepass_lateral_m": "_receiver_prepass_lateral_values",
        "receiver_prepass_forward_m": "_receiver_prepass_forward_values",
        "challenge_post_touch_speed_3d_mps":
            "_challenge_post_touch_speed_3d_values",
    }
    for public_name, private_name in raw_distribution_keys.items():
        values = np.concatenate([run[private_name] for run in runs])
        total[public_name] = _distribution_summary(values)
    charged_frames = sum(run["_stamina_charged_player_frames"] for run in runs)
    total["stamina_long_workload_terms"] = {
        term: (
            sum(run["_stamina_long_term_sums"][term] for run in runs) / charged_frames
            if charged_frames else 0.0
        )
        for term in ("idle", "speed", "acceleration", "sprint")
    }
    total["stamina_long_workload_by_role"] = {
        "goalkeeper": _distribution_summary(np.concatenate([
            run["_stamina_long_workload_gk_values"] for run in runs
        ])),
        "outfield": _distribution_summary(np.concatenate([
            run["_stamina_long_workload_outfield_values"] for run in runs
        ])),
    }
    total["stamina_long_final_by_role"] = {
        "goalkeeper": _distribution_summary(np.concatenate([
            run["_stamina_long_final_gk_values"] for run in runs
        ])),
        "outfield": _distribution_summary(np.concatenate([
            run["_stamina_long_final_outfield_values"] for run in runs
        ])),
    }
    total["sprints_per_90_by_role"] = {
        "goalkeeper": _distribution_summary(np.concatenate([
            run["_sprints_per_90_gk_values"] for run in runs
        ])),
        "outfield": _distribution_summary(np.concatenate([
            run["_sprints_per_90_outfield_values"] for run in runs
        ])),
    }
    histogram_counts = np.sum(
        np.asarray(
            [run["player_speed_histogram"]["counts"] for run in runs],
            dtype=np.int64,
        ),
        axis=0,
    )
    total["player_speed_histogram"] = {
        "bins_mps": _PLAYER_SPEED_BINS_MPS.tolist(),
        "counts": histogram_counts.tolist(),
        "fractions": (
            histogram_counts / max(int(histogram_counts.sum()), 1)
        ).tolist(),
    }
    pass_length_counts = np.sum(
        np.asarray(
            [run["pass_length_histogram"]["counts"] for run in runs],
            dtype=np.int64,
        ),
        axis=0,
    )
    total["pass_length_histogram"] = {
        "bins_m": _PASS_LENGTH_BINS_M.tolist(),
        "counts": pass_length_counts.tolist(),
        "fractions": (
            pass_length_counts / max(int(pass_length_counts.sum()), 1)
        ).tolist(),
    }
    for run in runs:
        run.pop("_restart_duration_samples_by_kind", None)
        for private_name in raw_distribution_keys.values():
            run.pop(private_name, None)
        run.pop("_sprints_per_90_gk_values", None)
        run.pop("_sprints_per_90_outfield_values", None)
        for private_name in (
            "_stamina_long_workload_gk_values",
            "_stamina_long_workload_outfield_values",
            "_stamina_long_final_gk_values",
            "_stamina_long_final_outfield_values",
            "_stamina_charged_player_frames",
            "_stamina_long_term_sums",
        ):
            run.pop(private_name, None)
    total["wall_seconds"] = time.perf_counter() - started
    total["simulated_minutes"] = args.seconds * args.seeds / 60.0
    total["passes_per_minute"] = total["passes"] / total["simulated_minutes"]
    alive_minutes = total["alive_frames"] / env.control_fps / 60.0
    total["alive_minutes"] = alive_minutes
    total["ball_alive_fraction"] = (
        alive_minutes / total["simulated_minutes"]
        if total["simulated_minutes"] else 0.0
    )
    total["passes_per_clock_minute_per_team"] = (
        total["passes"] / total["simulated_minutes"] / 2.0
        if total["simulated_minutes"] else 0.0
    )
    total["passes_per_alive_minute_per_team"] = (
        total["passes"] / alive_minutes / 2.0 if alive_minutes else 0.0
    )
    total["dribble_touches_per_alive_minute_per_team"] = (
        total["dribble_touches"] / alive_minutes / 2.0 if alive_minutes else 0.0
    )
    total["shots_per_alive_minute_per_team"] = (
        total["shots"] / alive_minutes / 2.0 if alive_minutes else 0.0
    )
    total["crosses_per_alive_minute_per_team"] = (
        total["crosses"] / alive_minutes / 2.0 if alive_minutes else 0.0
    )
    for name in (
        "possession_gains",
        "challenge_possession_gains",
        "ground_challenge_possession_gains",
        "challenge_touches",
        "challenge_control_like_touches",
        "challenge_outlet_like_touches",
        "defensive_clearance_like_touches",
        "challenge_gate_activations",
        "outfield_challenge_gate_activations",
        "carrier_duel_gate_activations",
        "travelling_interception_gate_activations",
        "loose_ball_gate_activations",
        "gk_passes",
        "gk_received_passes",
        "gk_buildup_cycles",
    ):
        total[f"{name}_per_alive_minute_per_team"] = (
            total[name] / alive_minutes / 2.0 if alive_minutes else 0.0
        )
    # DFL의 ground TacklingGame은 공 없는 참가자가 통제된 캐리어에게 건 시도다.
    # 시뮬에서는 같은 의미의 cooldown 상승 subset을 별칭으로 노출해 직접 비교한다.
    total["ground_challenge_attempts_per_alive_minute_per_team"] = total[
        "carrier_duel_gate_activations_per_alive_minute_per_team"
    ]
    total["defensive_clearances_per_alive_minute_per_team"] = total[
        "defensive_clearance_like_touches_per_alive_minute_per_team"
    ]
    total["challenge_retained_2s_rate"] = (
        total["challenge_retained_for_2s"]
        / total["challenge_retained_2s_eligible"]
        if total["challenge_retained_2s_eligible"]
        else None
    )
    total["kill_passes_per_minute"] = (
        total["completed_kill_passes"] / total["simulated_minutes"]
    )
    total["crosses_per_minute"] = total["crosses"] / total["simulated_minutes"]
    total["shots_per_minute"] = total["shots"] / total["simulated_minutes"]
    total["shooting_zone_seconds"] = (
        total["shooting_zone_frames"] / env.control_fps
    )
    total["rule_policy_version"] = RULE_POLICY_VERSION
    # 구성값을 문서에 복사해 두면 config를 바꾼 뒤 보고서 숫자가 조용히 낡는다. 실행에 실제로
    # 사용한 frozen dataclass를 그대로 직렬화해 모든 profile JSON이 자기 설정을 증명하게 한다.
    total["policy_config"] = asdict(policy.policy_config)
    total["policy_config_sha256"] = policy_config_fingerprint(
        policy.policy_config
    )
    dfl_reference = _load_dfl_reference()
    if dfl_reference.get("available"):
        total["dfl_reference"] = {
            key: dfl_reference[key]
            for key in (
                "available",
                "schema",
                "source",
                "definitions",
                "pooled",
                "team_match_distribution",
                "tracking",
            )
        }
        dfl = dfl_reference["pooled"]
        comparisons = {
            "ball_alive_fraction": total["ball_alive_fraction"],
            "passes_per_alive_minute_per_team": total[
                "passes_per_alive_minute_per_team"
            ],
            "pass_success_rate": total["reference_pass_success_rate"],
            "flat_pass_share": total["low_pass_share"],
            "direction_forward_rate": total["direction_forward_pass_rate"],
            "direction_sideways_rate": total["direction_sideways_pass_rate"],
            "direction_backward_rate": total["direction_backward_pass_rate"],
            "shots_per_alive_minute_per_team": total[
                "shots_per_alive_minute_per_team"
            ],
            "crosses_per_alive_minute_per_team": total[
                "crosses_per_alive_minute_per_team"
            ],
            "ground_challenge_possession_gains_per_alive_minute_per_team": total[
                "ground_challenge_possession_gains_per_alive_minute_per_team"
            ],
            "ground_challenge_attempts_per_alive_minute_per_team": total[
                "ground_challenge_attempts_per_alive_minute_per_team"
            ],
            "defensive_clearances_per_alive_minute_per_team": total[
                "defensive_clearances_per_alive_minute_per_team"
            ],
            "gk_passes_per_alive_minute_per_team": total[
                "gk_passes_per_alive_minute_per_team"
            ],
            "gk_received_passes_per_alive_minute_per_team": total[
                "gk_received_passes_per_alive_minute_per_team"
            ],
            "gk_buildup_cycles_per_alive_minute_per_team": total[
                "gk_buildup_cycles_per_alive_minute_per_team"
            ],
            "gk_pass_share": total["gk_pass_share"],
            "gk_flat_pass_share": total["gk_flat_pass_share"],
            "quick_relay_rate": total["quick_relay_rate"],
            "quick_flat_relay_rate": total["quick_flat_relay_rate"],
        }
        total["dfl_comparison"] = {}
        total["dfl_comparison_notes"] = {
            "challenge_possession_gains_per_alive_minute_per_team": (
                "diagnostic only: env TOUCH_INTERCEPT includes ordinary pass-lane "
                "interceptions, whereas DFL challenge gains are TacklingGame rows; "
                "the ground/tackle-only row below is the calibrated common subset"
            ),
            "ground_challenge_attempts_per_alive_minute_per_team": (
                "simulated value is an outfield challenge-cooldown rising edge while "
                "the prior possession team's nearest player physically controlled the "
                "ball; DFL target is every ground TacklingGame attributed to its "
                "withoutBallControl participant"
            ),
            "defensive_clearances_per_alive_minute_per_team": (
                "simulated value is a successful outfield TACKLE/INTERCEPT kick "
                "from the actor's own half with >=12 m/s post-touch speed, a "
                "lofted launch, and a forward direction; DFL target is the "
                "explicit DefensiveClearance event flag"
            ),
        }
        for name, simulated in comparisons.items():
            target = dfl.get(name)
            if target is None:
                continue
            total["dfl_comparison"][name] = {
                "simulated": float(simulated),
                "target": float(target),
                "delta": float(simulated - target),
                "ratio": float(simulated / target) if target else None,
            }
        tracking_reference = dfl_reference["tracking"]
        for name, simulated in (
            ("player_speed_mean_mps", total["player_speed_mps"]["mean"]),
            ("ball_speed_3d_mean_mps", total["ball_speed_3d_mps"]["mean"]),
        ):
            reference_name = (
                "player_speed_mps" if name.startswith("player")
                else "ball_speed_mps"
            )
            target = tracking_reference[reference_name]["mean"]
            total["dfl_comparison"][name] = {
                "simulated": float(simulated),
                "target": float(target),
                "delta": float(simulated - target),
                "ratio": float(simulated / target) if target else None,
            }
    else:
        total["dfl_reference"] = dfl_reference
    total["kleague_reference"] = {
        # Event and tracking references have different corpus cardinalities.
        # A single historical ``matches=120`` label incorrectly implied that
        # player/ball kinematics were also measured on all 120 event matches.
        "sources": {
            "events": "real_event_reference.json",
            "tracking": "real_tracking_reference.json",
        },
        "event_matches": 120,
        "tracking_matches": 30,
        "ball_alive_fraction": 0.522,
        "passes_per_alive_minute_per_team": 9.45,
        "passes_per_alive_minute_per_team_iqr": [7.72, 10.92],
        "pass_success_rate": 0.8454,
        "pass_success_rate_iqr": [0.80, 0.8765],
        "shots_per_alive_minute_per_team": 0.23,
        "shots_per_alive_minute_per_team_iqr": [0.17, 0.30],
        "shot_distance_mean_m": 16.965032977050587,
        "shot_distance_p50_m": 16.686965353202147,
        "shot_distance_p75_m": 22.11965384383409,
        "shot_distance_p90_m": 26.446197866915558,
        "crosses_per_alive_minute_per_team": 19.0 / (90.0 * 0.522),
        "direction_forward_rate": 183.0 / 443.0,
        "direction_sideways_rate": 157.0 / 443.0,
        "direction_backward_rate": 103.0 / 443.0,
        "pass_length_mean_m": 17.6974,
        "pass_length_p50_m": 14.5610,
        "pass_length_histogram_fractions": [
            0.0541106241,
            0.2209759623,
            0.2441407533,
            0.1822490872,
            0.1141156926,
            0.0638439662,
            0.0545799270,
            0.0287964258,
            0.0371875616,
        ],
        "player_speed_mean_mps": 2.3668,
        "player_speed_p50_mps": 2.0531,
        "distance_per_90_mean_m": 12766.2,
        "sprints_per_90_mean": 21.708,
        "ball_speed_mean_mps": 6.6226,
        # Same 30-match/3 Hz tracking sample, split by provider possession.
        # The total above is not an average of these two match medians.
        "ball_speed_controlled_mean_mps": 5.9492,
        "ball_speed_neutral_mean_mps": 10.0800,
    }
    total["pass_rate_ratio_to_kleague"] = (
        total["passes_per_alive_minute_per_team"] / 9.45
    )
    total["ball_alive_fraction_delta_to_kleague"] = (
        total["ball_alive_fraction"] - 0.522
    )
    total["pass_success_rate_delta_to_kleague"] = (
        total["reference_pass_success_rate"] - 0.8454
    )
    total["shot_rate_ratio_to_kleague"] = (
        total["shots_per_alive_minute_per_team"] / 0.23
    )
    total["shot_distance_mean_ratio_to_kleague"] = (
        total["shot_distance_m"]["mean"] / 16.965032977050587
        if total["shot_distance_m"]["mean"] is not None else None
    )
    total["shot_distance_p50_delta_to_kleague_m"] = (
        total["shot_distance_m"]["p50"] - 16.686965353202147
        if total["shot_distance_m"]["p50"] is not None else None
    )
    total["cross_rate_ratio_to_kleague"] = (
        total["crosses_per_alive_minute_per_team"]
        / (19.0 / (90.0 * 0.522))
    )
    total["pass_length_mean_ratio_to_kleague"] = (
        total["pass_length_m"]["mean"] / 17.6974
        if total["pass_length_m"]["mean"] is not None else None
    )
    total["pass_length_p50_delta_to_kleague_m"] = (
        total["pass_length_m"]["p50"] - 14.5610
        if total["pass_length_m"]["p50"] is not None else None
    )
    total["player_speed_mean_delta_to_kleague_mps"] = (
        total["player_speed_mps"]["mean"] - 2.3668
    )
    total["distance_per_90_mean_ratio_to_kleague"] = (
        total["distance_per_90_m"]["mean"] / 12766.2
    )
    total["sprints_per_90_mean_ratio_to_kleague"] = (
        total["sprints_per_90"]["mean"] / 21.708
    )
    total["ball_speed_mean_ratio_to_kleague"] = (
        total["ball_speed_mps"]["mean"] / 6.6226
    )
    total["ball_speed_controlled_mean_ratio_to_kleague"] = (
        total["ball_speed_controlled_mps"]["mean"] / 5.9492
        if total["ball_speed_controlled_mps"]["mean"] is not None else None
    )
    total["ball_speed_neutral_mean_ratio_to_kleague"] = (
        total["ball_speed_neutral_mps"]["mean"] / 10.0800
        if total["ball_speed_neutral_mps"]["mean"] is not None else None
    )
    payload = {"total": total} if args.total_only else {"runs": runs, "total": total}
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        if not output_path.name:
            parser.error("--output-json must name a file")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        temporary.write_text(encoded + "\n", encoding="utf-8")
        os.replace(temporary, output_path)
    print(encoded)


if __name__ == "__main__":
    main()
