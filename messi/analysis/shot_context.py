"""이미 기록된 SHOOT의 host-only 공간·결과 진단. 정책 quality 추정이 아니다."""

from __future__ import annotations

import math

import numpy as np

from messi.analysis.causal import (
    GOAL_MOUTH_HALF_WIDTH_M,
    SHOT_OUTCOMES,
    is_goal_directed_shot,
    is_shot_attempt,
)

SHOT_CONTEXT_SCHEMA = "messi.shot-context/1"


def summarize_shot_context(rows, *, goal_line_x=52.5, live_ball_minutes=None):
    """한 team의 causal SHOOT rows에서 raw near/GD를 별도로 요약한다.

    좌표는 attack-normalized m, angle은 source에서 두 goalpost가 벌어지는
    수평각 degree다. 거리/각도 bin은 왼쪽 닫힘·오른쪽 열림이며 마지막은
    상한이 없다. Goalpost와 source가 같으면 angle만 unknown이다.
    live_ball_minutes는 team 경기의 전체 live 노출(점유 시간 아님)이다.
    None/0 노출은 rate=None이다. 미관측 source/direction은 near 모집단
    제외가 아니라 classification_unknown으로 별도 보고한다.
    """
    goal_line_x = float(goal_line_x)
    if not math.isfinite(goal_line_x) or goal_line_x <= 0:
        raise ValueError("goal_line_x는 finite 양수여야 한다")
    if live_ball_minutes is not None:
        live_ball_minutes = float(live_ball_minutes)
        if not math.isfinite(live_ball_minutes) or live_ball_minutes < 0:
            raise ValueError("live_ball_minutes는 finite non-negative 값이어야 한다")
    rows = tuple(rows)
    classified = []
    unknown = 0
    for row in rows:
        source = row.get("attack_normalized_source_position")
        direction = row.get("attack_normalized_direction")
        if source is None or direction is None:
            unknown += 1
            continue
        source, direction = np.asarray(source), np.asarray(direction)
        if source.ndim != 1 or direction.ndim != 1 or source.size < 2 or direction.size < 2:
            raise ValueError("source/direction은 최소 2성분 vector여야 한다")
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(direction)):
            unknown += 1
            continue
        if is_shot_attempt(row, goal_line_x=goal_line_x):
            classified.append(row)

    def summarize(selected):
        count = len(selected)
        outcomes = {name: sum(row.get("outcome") == name for row in selected)
                    for name in SHOT_OUTCOMES}
        outcome_known = sum(outcomes.values())
        goals = outcomes["goal"]
        return {
            "count": count,
            "outcomes": outcomes,
            "outcome_known_count": outcome_known,
            "outcome_unknown_count": count - outcome_known,
            "outcome_coverage": outcome_known / count if count else None,
            "causal_goal_conversion": goals / count if count else None,
            "releases_per_team_live_minute": count / live_ball_minutes
            if live_ball_minutes else None,
            "causal_goals_per_team_live_minute": goals / live_ball_minutes
            if live_ball_minutes else None,
        }

    def population(selected):
        distances = []
        angles = []
        boxes = []
        for row in selected:
            x, y = (float(v) for v in row["attack_normalized_source_position"][:2])
            dx = goal_line_x - x
            distances.append(math.hypot(dx, y))
            left_y, right_y = -GOAL_MOUTH_HALF_WIDTH_M - y, GOAL_MOUTH_HALF_WIDTH_M - y
            if math.hypot(dx, left_y) == 0 or math.hypot(dx, right_y) == 0:
                angles.append(None)
            else:
                angles.append(math.degrees(math.atan2(
                    abs(dx * (right_y - left_y)), dx * dx + left_y * right_y)))
            boxes.append(x >= goal_line_x - 16.5 and abs(y) <= 20.16)

        def bins(values, edges):
            result = []
            for low, high in zip(edges[:-1], edges[1:], strict=True):
                result.append({
                    "lower_inclusive": low, "upper_exclusive": high,
                    **summarize([row for row, value in zip(selected, values, strict=True)
                                 if value is not None and low <= value
                                 and (high is None or value < high)]),
                })
            return result

        return {
            **summarize(selected),
            "distance_known_count": len(distances),
            "distance_unknown_count": 0,
            "distance_coverage": 1.0 if selected else None,
            "mean_distance_m": sum(distances) / len(distances) if distances else None,
            "angle_known_count": sum(value is not None for value in angles),
            "angle_unknown_count": sum(value is None for value in angles),
            "angle_coverage": sum(value is not None for value in angles) / len(angles)
            if angles else None,
            "distance_bins_m": bins(distances, (0, 10, 20, 30, None)),
            "goal_opening_angle_bins_deg": bins(angles, (0, 15, 30, 45, None)),
            "box": {name: summarize([row for row, inside in zip(selected, boxes, strict=True)
                                     if inside == expected])
                    for name, expected in (("inside", True), ("outside", False))},
        }

    directed = [row for row in classified if is_goal_directed_shot(row, goal_line_x=goal_line_x)]
    return {
        "schema": SHOT_CONTEXT_SCHEMA,
        "basis": "causal_near_goal_forward_release_not_provider_shots_or_xg",
        "angle_basis": "horizontal_goalpost_opening_angle_degrees",
        "live_ball_minutes": live_ball_minutes,
        "submitted_shoot_count": len(rows),
        "classification_unknown_count": unknown,
        "classification_coverage": (len(rows) - unknown) / len(rows) if rows else None,
        "classified_other_shoot_count": len(rows) - unknown - len(classified),
        "policy_shot_quality": "not_observed",
        "raw_near_goal": population(classified),
        "goal_directed": population(directed),
    }
