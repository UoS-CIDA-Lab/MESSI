"""Goal boundary 시점의 authoritative last-contact와 release-origin을 분리한다."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from messi.core.action import (
    INTENT_CHALLENGE,
    INTENT_CONTROL,
    INTENT_PASS,
    INTENT_SHOOT,
)
from messi.core.constants import BALL_EVENT_GOAL


def extract_goal_last_contacts(states, receipts, team_id, *, frame_offset=0):
    """Host post-step state/receipt에서 득점 사건별 마지막 접촉을 반환한다.

    좌우 방향 대신 boundary.scoring_team과 roster team을 사용한다. 동일 frame의
    여러 접촉은 환경이 보존한 possession.last_contact를 그대로 사용한다. 선수
    identity generation의 credit이나 provider 슛 의도를 추정하지 않는다.
    """
    team = np.asarray(team_id)
    boundary = receipts.boundary
    goals = np.flatnonzero(np.asarray(boundary.occurred, dtype=bool)
                           & (np.asarray(boundary.kind) == BALL_EVENT_GOAL))
    contact = states.possession.last_contact
    rows = []
    for frame in goals:
        scoring_team = int(boundary.scoring_team[frame])
        if scoring_team not in (0, 1):
            raise ValueError("goal boundary scoring_team은 0 또는 1이어야 한다")
        actor = int(contact.actor[frame])
        mechanism = int(contact.mechanism[frame])
        known = 0 <= actor < len(team) and int(team[actor]) in (0, 1) and mechanism != 0
        rows.append({
            "frame": int(frame_offset + frame), "scoring_team": scoring_team,
            "actor": actor, "contact_team": int(team[actor]) if known else -1,
            "contact_known": known, "intent": int(contact.intent[frame]),
            "mechanism": mechanism, "outcome": int(contact.outcome[frame]),
            "kick_applied": bool(contact.kick_applied[frame]),
            "intent_source": int(contact.intent_source[frame]),
            "own_goal_last_touch": int(team[actor]) != scoring_team if known else None,
            "basis": "post_step_authoritative_last_contact_at_goal_boundary",
        })
    return tuple(rows)


def summarize_goal_last_contacts(rows, scores):
    """단일 경기 scoreboard를 분모로 접촉 intent와 미측정 골을 반환한다.

    두 팀의 비음수 정수 score, 0/1 scoring_team과 중복 없는 goal frame이 필요하다.
    잘못된 입력은 ValueError로 거부하며 음수 index를 상대 팀으로 해석하지 않는다.
    """
    scores = tuple(scores)
    if len(scores) != 2 or any(isinstance(score, bool) or not isinstance(score, Integral) or score < 0 for score in scores):
        raise ValueError("scoreboard는 두 팀의 비음수 정수여야 한다")
    categories = {INTENT_CONTROL: "control_goals", INTENT_PASS: "pass_goals",
                  INTENT_SHOOT: "shoot_goals", INTENT_CHALLENGE: "challenge_goals"}
    result = [{"scoreboard_goals": int(score), "observed_goal_events": 0,
               "control_goals": 0, "pass_goals": 0, "shoot_goals": 0,
               "challenge_goals": 0, "other_goals": 0, "unknown_contact_goals": 0,
               "own_goal_last_touch": 0, "missing_goal_events": 0} for score in scores]
    frames = set()
    for row in rows:
        scoring_team = row["scoring_team"]
        if isinstance(scoring_team, bool) or not isinstance(scoring_team, Integral) or scoring_team not in (0, 1):
            raise ValueError("goal row scoring_team은 정수 0 또는 1이어야 한다")
        if "frame" in row:
            frame = row["frame"]
            if isinstance(frame, bool) or not isinstance(frame, Integral) or frame < 0 or frame in frames:
                raise ValueError("goal row frame은 중복 없는 비음수 정수여야 한다")
            frames.add(frame)
        target = result[int(scoring_team)]
        target["observed_goal_events"] += 1
        category = categories.get(int(row["intent"]), "other_goals") if row["contact_known"] else "unknown_contact_goals"
        target[category] += 1
        target["own_goal_last_touch"] += int(row["own_goal_last_touch"] is True)
    for target in result:
        missing = target["scoreboard_goals"] - target["observed_goal_events"]
        if missing < 0:
            raise ValueError("관측 goal boundary 수가 scoreboard보다 많다")
        target["missing_goal_events"] = missing
    return tuple(result)


def normalize_goal_contact_summary(value, score):
    empty = summarize_goal_last_contacts((), (score, 0))[0]
    if value is None:
        return empty
    if set(value) != set(empty) or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in value.values()):
        raise ValueError("last-contact goal summary는 정해진 비음수 정수 필드여야 한다")
    result = {k: int(v) for k, v in value.items()}
    categories = ('control_goals', 'pass_goals', 'shoot_goals', 'challenge_goals', 'other_goals', 'unknown_contact_goals')
    if (result['scoreboard_goals'] != score
            or sum(result[k] for k in categories) != result['observed_goal_events']
            or result['observed_goal_events'] + result['missing_goal_events'] != score
            or result['own_goal_last_touch'] > result['observed_goal_events'] - result['unknown_contact_goals']):
        raise ValueError("last-contact goal summary 분자·분모가 일치하지 않는다")
    return result


def merge_goal_contact_summaries(values):
    result = summarize_goal_last_contacts((), (0, 0))[0]
    for value, score in values:
        for key, count in normalize_goal_contact_summary(value, score).items():
            result[key] += count
    return result
