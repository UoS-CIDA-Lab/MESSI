"""Resolution and lifetime contracts for the goal celebration overlay."""

from __future__ import annotations

import numpy as np

from footballworld.rendering.defaults import RenderStyle
from footballworld.rendering.renderer import (
    _Adjudication,
    _carry_adjudication,
    _goal_overlay_font_sizes,
    _insert_goal_presentation_holds,
    _VisualFrame,
)


def _frame(control_tick: int, adjudication: _Adjudication | None) -> _VisualFrame:
    return _VisualFrame(
        control_tick=control_tick,
        video_time_s=None,
        first_half_wall_end_tick=27_000,
        ball_position=np.zeros(3, dtype=np.float32),
        ball_live=True,
        player_position=np.zeros((2, 2), dtype=np.float32),
        player_body_forward=np.zeros((2, 2), dtype=np.float32),
        player_gaze_yaw=np.zeros(2, dtype=np.float32),
        aerial_progress=np.zeros(2, dtype=np.float32),
        high_head_contact=np.zeros(2, dtype=bool),
        team_id=np.asarray((0, 1), dtype=np.int32),
        is_goalkeeper=np.asarray((True, True), dtype=bool),
        active=np.asarray((True, True), dtype=bool),
        sent_off=np.zeros(2, dtype=bool),
        stamina_long=np.ones(2, dtype=np.float32),
        stamina_short=np.ones(2, dtype=np.float32),
        yellow_cards=np.zeros(2, dtype=np.int32),
        possession_player=-1,
        score=np.asarray((1, 0), dtype=np.int32),
        requested_intent=np.zeros(2, dtype=np.int32),
        action_executed=True,
        adjudication=adjudication,
    )


def test_goal_overlay_typography_preserves_relative_size_at_1080p() -> None:
    assert _goal_overlay_font_sizes(RenderStyle(width_px=960, height_px=540)) == (
        30.0,
        10.0,
    )
    assert _goal_overlay_font_sizes(RenderStyle(width_px=1920, height_px=1080)) == (
        60.0,
        20.0,
    )


def test_goal_presentation_survives_adjudication_hold() -> None:
    goal = _Adjudication(
        origin_control_tick=100,
        title="GOAL — HOME",
        detail="SCORE 1 : 0",
        accent="#ef476f",
        goal=True,
    )
    held, previous = _carry_adjudication(
        _frame(124, None),
        goal,
        control_fps=10.0,
        duration_seconds=2.5,
    )
    assert held.adjudication is goal
    assert previous is goal
    assert held.adjudication.goal

    expired, previous = _carry_adjudication(
        _frame(125, None),
        goal,
        control_fps=10.0,
        duration_seconds=2.5,
    )
    assert expired.adjudication is None
    assert previous is None


def test_goal_presentation_freezes_video_for_exactly_two_seconds() -> None:
    goal = _Adjudication(
        origin_control_tick=100,
        title="GOAL — HOME",
        detail="SCORE 1 : 0",
        accent="#ef476f",
        goal=True,
    )
    origin = _frame(100, goal)
    duplicate = _frame(100, goal)
    carried = _frame(101, goal)
    resumed = _frame(102, None)
    presented = _insert_goal_presentation_holds(
        [origin, duplicate, carried, resumed],
        video_fps=20.0,
        duration_seconds=2.0,
    )
    assert len(presented) == 42
    assert all(frame is origin for frame in presented[:40])
    assert presented[40].control_tick == 101
    assert presented[40].adjudication is None
    assert presented[41] is resumed
