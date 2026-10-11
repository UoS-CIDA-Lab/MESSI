"""긴 경기 chunk를 device graph 밖에서 점검하는 invariant 집계기."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np

from messi.core.constants import RK_NONE
from messi.core.contact import MECHANISM_GOALKEEPER_HAND, OUTCOME_CATCH


@dataclass(frozen=True, slots=True)
class MatchInvariantReport:
    """경기 전체에서 발견한 상태 이상과 최장 정체 구간."""

    nonfinite_frames: int
    first_nonfinite_frame: int
    possession_mismatch_frames: int
    first_possession_mismatch_frame: int
    longest_restart_frames: int
    longest_goalkeeper_hold_frames: int

    @property
    def valid(self) -> bool:
        return self.nonfinite_frames == 0 and self.possession_mismatch_frames == 0


class MatchInvariantAuditor:
    """연속 host chunk 사이의 streak를 보존하는 O(1) 집계기."""

    def __init__(self, roster) -> None:
        team, is_goalkeeper = jax.device_get(
            (roster.team_id, roster.is_goalkeeper)
        )
        self._team = np.asarray(team, dtype=np.int32)
        self._is_goalkeeper = np.asarray(is_goalkeeper, dtype=bool)
        self._frames = 0
        self._nonfinite_frames = 0
        self._first_nonfinite = -1
        self._possession_mismatch_frames = 0
        self._first_possession_mismatch = -1
        self._restart_streak = 0
        self._longest_restart = 0
        self._hold_streak = 0
        self._longest_hold = 0

    def update_roster(self, roster) -> None:
        """Management 경계 뒤의 authoritative goalkeeper metadata를 갱신한다."""

        team, is_goalkeeper = jax.device_get(
            (roster.team_id, roster.is_goalkeeper)
        )
        team = np.asarray(team, dtype=np.int32)
        is_goalkeeper = np.asarray(is_goalkeeper, dtype=bool)
        if team.shape != self._team.shape or is_goalkeeper.shape != self._team.shape:
            raise ValueError("roster player축은 invariant audit 중 바뀔 수 없다")
        if not np.array_equal(team, self._team):
            raise ValueError("roster team_id는 invariant audit 중 바뀔 수 없다")
        self._is_goalkeeper = is_goalkeeper.copy()

    @staticmethod
    def _update_streak(mask: np.ndarray, current: int, longest: int) -> tuple[int, int]:
        for enabled in mask:
            current = current + 1 if bool(enabled) else 0
            longest = max(longest, current)
        return current, longest

    def add(self, states) -> None:
        # Public auditor에 device trajectory가 직접 들어와도 leaf별 implicit
        # synchronization을 만들지 않는다. Recording accumulator에서 이미 host인
        # PyTree가 들어오면 이 호출은 구조만 보존하는 no-op이다.
        states = jax.device_get(states)
        leaves = [np.asarray(value) for value in jax.tree.leaves(states)]
        if not leaves:
            return
        count = int(leaves[0].shape[0])
        nonfinite = np.zeros(count, dtype=bool)
        for value in leaves:
            if np.issubdtype(value.dtype, np.inexact):
                nonfinite |= ~np.isfinite(value).reshape(count, -1).all(axis=1)
        indices = np.flatnonzero(nonfinite)
        if indices.size and self._first_nonfinite < 0:
            self._first_nonfinite = self._frames + int(indices[0])
        self._nonfinite_frames += int(indices.size)

        owner = np.asarray(states.possession.player, dtype=np.int32)
        owner_team = np.asarray(states.possession.team, dtype=np.int32)
        control_ticks = np.asarray(states.possession.control_ticks, dtype=np.int32)
        valid_owner = (owner >= 0) & (owner < self._team.size)
        absent_owner = owner == -1
        valid_team = (owner_team == 0) | (owner_team == 1)
        absent_team = owner_team == -1
        safe_owner = np.clip(owner, 0, self._team.size - 1)
        active = np.asarray(states.players.active, dtype=bool)
        last_contact = states.possession.last_contact
        catch_actor = np.asarray(last_contact.actor, dtype=np.int32)
        valid_catch_actor = (catch_actor >= 0) & (catch_actor < self._team.size)
        safe_catch_actor = np.clip(catch_actor, 0, self._team.size - 1)
        catch_outcome = np.asarray(last_contact.outcome) == OUTCOME_CATCH
        hand_catch_provenance = (
            catch_outcome
            & valid_catch_actor
            & (np.asarray(last_contact.mechanism) == MECHANISM_GOALKEEPER_HAND)
            & active[np.arange(owner.size), safe_catch_actor]
            & self._is_goalkeeper[safe_catch_actor]
        )
        catch_claim = (
            valid_owner
            & hand_catch_provenance
            & (~np.asarray(states.ball.live, dtype=bool))
            & (catch_actor == owner)
            & valid_team
            & (self._team[safe_owner] == owner_team)
            & active[np.arange(owner.size), safe_owner]
            & self._is_goalkeeper[safe_owner]
        )
        historical_restart_catch = (
            hand_catch_provenance
            & absent_owner
            & absent_team
            & (control_ticks == 0)
            & (~np.asarray(states.ball.live, dtype=bool))
            & (np.asarray(states.restart.kind) != RK_NONE)
        )
        mismatch = (
            (~(valid_owner | absent_owner))
            | (~(valid_team | absent_team))
            | (valid_owner != valid_team)
            | (valid_owner & (self._team[safe_owner] != owner_team))
            | (valid_owner & (~active[np.arange(owner.size), safe_owner]))
            | (catch_outcome & (~(catch_claim | historical_restart_catch)))
            | (control_ticks < 0)
            | ((control_ticks > 0) != valid_owner)
        )
        mismatch_indices = np.flatnonzero(mismatch)
        if mismatch_indices.size and self._first_possession_mismatch < 0:
            self._first_possession_mismatch = self._frames + int(mismatch_indices[0])
        self._possession_mismatch_frames += int(mismatch_indices.size)

        restart = np.asarray(states.restart.kind) != RK_NONE
        self._restart_streak, self._longest_restart = self._update_streak(
            restart, self._restart_streak, self._longest_restart
        )
        held = (
            catch_claim
        )
        self._hold_streak, self._longest_hold = self._update_streak(
            held, self._hold_streak, self._longest_hold
        )
        self._frames += count

    def finish(self) -> MatchInvariantReport:
        return MatchInvariantReport(
            nonfinite_frames=self._nonfinite_frames,
            first_nonfinite_frame=self._first_nonfinite,
            possession_mismatch_frames=self._possession_mismatch_frames,
            first_possession_mismatch_frame=self._first_possession_mismatch,
            longest_restart_frames=self._longest_restart,
            longest_goalkeeper_hold_frames=self._longest_hold,
        )
