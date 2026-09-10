from __future__ import annotations

import jax

from footballworld import (
    FootballWorld,
    IntentAction,
    Player,
    PlayerProfile,
    StepWithActionReceiptResult,
)

_FORMATION = (
    (-50.0, 0.0),
    (-35.0, -24.0),
    (-35.0, -8.0),
    (-35.0, 8.0),
    (-35.0, 24.0),
    (-20.0, -18.0),
    (-20.0, 0.0),
    (-20.0, 18.0),
    (-8.0, -24.0),
    (-8.0, 0.0),
    (-8.0, 24.0),
)


def _team(team: int) -> tuple[Player, ...]:
    return tuple(
        Player(
            PlayerProfile(
                player_id=1_000 * (team + 1) + slot,
                is_goalkeeper=slot == 0,
            ),
            position,
        )
        for slot, position in enumerate(_FORMATION)
    )


def test_compact_action_receipt_matches_eventful_result_without_events_field() -> None:
    env = FootballWorld()
    reset = env.reset(_team(0), _team(1), key=jax.random.key(3))
    action = IntentAction.neutral(22)
    key = jax.random.key(9)
    compact = env.step_with_action_receipt(reset.rollout, reset.setup, action, key)
    eventful = env.step_with_events(reset.rollout, reset.setup, action, key)

    assert isinstance(compact, StepWithActionReceiptResult)
    assert "events" not in compact._fields
    assert compact.action_trace._fields == eventful.action_trace._fields
    assert compact.action_receipt._fields == eventful.action_receipt._fields
    for left, right in zip(
        jax.tree.leaves(compact.action_trace),
        jax.tree.leaves(eventful.action_trace),
        strict=True,
    ):
        assert (left == right).all()
    for left, right in zip(
        jax.tree.leaves(compact.action_receipt),
        jax.tree.leaves(eventful.action_receipt),
        strict=True,
    ):
        assert (left == right).all()
