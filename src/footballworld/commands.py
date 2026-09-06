"""Host façade for one player transition followed by one manager transaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from footballworld.core.action import IntentAction
from footballworld.environment.api import (
    FootballWorld,
    ManagerCommandStepResult,
    Rollout,
    StepResult,
    StepWithEventsResult,
)
from footballworld.environment.episode import MatchSetup
from footballworld.environment.management import (
    ManagerCommand,
    ManagerState,
    SquadSetup,
)


@dataclass(frozen=True, slots=True)
class StepCommand:
    """One public transaction; the player action is always applied first."""

    player_action: IntentAction
    manager_command: ManagerCommand


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """Both causal stages, retaining their independent receipts."""

    player: StepResult | StepWithEventsResult
    manager: ManagerCommandStepResult

    @property
    def rollout(self) -> Rollout:
        """Return the authoritative post-management rollout."""

        return self.manager.rollout


def execute_step_command(
    env: FootballWorld,
    rollout: Rollout,
    setup: MatchSetup,
    squad: SquadSetup,
    management: ManagerState,
    command: StepCommand,
    key: jax.Array,
    *,
    with_events: bool = False,
) -> TransitionResult:
    """Apply a player frame, then the rare manager command at that boundary.

    This convenience boundary is intentionally not used by rollout scans. It
    avoids carrying benches and manager ledgers through the physics graph while
    giving interactive callers one transactional entry point.
    """

    if not isinstance(env, FootballWorld):
        raise TypeError("env must be FootballWorld")
    if not isinstance(command, StepCommand):
        raise TypeError("command must be StepCommand")
    if type(with_events) is not bool:
        raise TypeError("with_events must be bool")
    player: Any = (
        env.step_with_events(rollout, setup, command.player_action, key)
        if with_events
        else env.step(rollout, setup, command.player_action, key)
    )
    manager = env.manager_command(
        player.rollout,
        squad,
        management,
        command.manager_command,
    )
    return TransitionResult(player=player, manager=manager)


__all__ = ["StepCommand", "TransitionResult", "execute_step_command"]
