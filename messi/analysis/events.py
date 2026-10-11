"""compact receipt trajectory에서 sparse event stream을 추출한다."""

from dataclasses import dataclass

import jax
import numpy as np

EVENT_NAMES = (
    "contact",
    "boundary",
    "restart_release",
    "restart_retouch",
    "woodwork",
    "foul",
    "offside",
    "passive_contact",
    "goalkeeper_holding_expiry",
    "nutmeg",
    "event_budget_exhausted",
)


@dataclass(frozen=True, slots=True)
class EventStream:
    frame: np.ndarray
    kind: np.ndarray
    team: np.ndarray
    actor: np.ndarray
    secondary_actor: np.ndarray
    severity: np.ndarray
    tactical_effect: np.ndarray

    def __len__(self) -> int:
        return int(self.frame.size)

    def rows(self) -> list[dict[str, int | str]]:
        return [
            {
                "frame": int(frame),
                "kind": EVENT_NAMES[int(kind)],
                "team": int(team),
                "actor": int(actor),
                "secondary_actor": int(secondary_actor),
                "severity": int(severity),
                "tactical_effect": int(tactical_effect),
            }
            for frame, kind, team, actor, secondary_actor, severity, tactical_effect in zip(
                self.frame,
                self.kind,
                self.team,
                self.actor,
                self.secondary_actor,
                self.severity,
                self.tactical_effect,
                strict=True,
            )
        ]


def extract_events(receipts, *, team_id=None) -> EventStream:
    """device synchronization 한 번으로 compact event를 sparse host table로 만든다.

    ``team_id``를 주면 receipt에 team column이 없는 event(passive contact, nutmeg)의
    team을 actor로부터 채운다. 주지 않으면 그 event의 team은 -1로 남는다.
    """
    host = jax.device_get(receipts)
    applied = getattr(host, "applied_contact", None)
    deliberate_occurred = (
        np.asarray(applied.occurred, dtype=bool)
        if applied is not None
        else np.asarray(host.contact_occurred, dtype=bool)
    )
    release_team = np.full(deliberate_occurred.shape, -1, dtype=np.int32)
    release_actor = np.full(deliberate_occurred.shape, -1, dtype=np.int32)
    if applied is not None:
        release_team = np.where(
            deliberate_occurred,
            np.asarray(applied.team, dtype=np.int32),
            release_team,
        )
        release_actor = np.where(
            deliberate_occurred,
            np.asarray(applied.actor, dtype=np.int32),
            release_actor,
        )
    masks = np.stack(
        (
            deliberate_occurred,
            np.asarray(host.boundary.occurred, dtype=bool),
            np.asarray(host.restart_released, dtype=bool),
            np.asarray(host.restart_retouch.occurred, dtype=bool),
            np.asarray(host.woodwork.occurred, dtype=bool),
            np.asarray(host.foul.occurred, dtype=bool),
            np.asarray(host.offside.occurred, dtype=bool),
            np.asarray(host.passive_contact.occurred, dtype=bool),
            np.asarray(host.goalkeeper_holding.expired, dtype=bool),
            np.asarray(host.nutmeg.occurred, dtype=bool),
            np.asarray(host.event_budget_exhausted, dtype=bool),
        ),
        axis=-1,
    )
    frame, kind = np.nonzero(masks)
    team_dense = np.stack(
        (
            (
                np.asarray(applied.team, dtype=np.int32)
                if applied is not None
                else np.full(masks.shape[0], -1, dtype=np.int32)
            ),
            np.asarray(host.boundary.team, dtype=np.int32),
            release_team,
            np.asarray(host.restart_retouch.team, dtype=np.int32),
            np.full(masks.shape[0], -1, dtype=np.int32),
            np.asarray(host.foul.restart_team, dtype=np.int32),
            np.asarray(host.offside.team, dtype=np.int32),
            np.full(masks.shape[0], -1, dtype=np.int32),
            np.asarray(host.goalkeeper_holding.restart_team, dtype=np.int32),
            np.full(masks.shape[0], -1, dtype=np.int32),
            np.full(masks.shape[0], -1, dtype=np.int32),
        ),
        axis=-1,
    )
    actor_dense = np.stack(
        (
            (
                np.asarray(applied.actor, dtype=np.int32)
                if applied is not None
                else np.full(masks.shape[0], -1, dtype=np.int32)
            ),
            np.full(masks.shape[0], -1, dtype=np.int32),
            release_actor,
            np.asarray(host.restart_retouch.offender, dtype=np.int32),
            np.full(masks.shape[0], -1, dtype=np.int32),
            np.asarray(host.foul.offender, dtype=np.int32),
            np.asarray(host.offside.actor, dtype=np.int32),
            np.asarray(host.passive_contact.actor, dtype=np.int32),
            np.asarray(host.goalkeeper_holding.goalkeeper, dtype=np.int32),
            np.asarray(host.nutmeg.attacker, dtype=np.int32),
            np.full(masks.shape[0], -1, dtype=np.int32),
        ),
        axis=-1,
    )
    empty_label = np.full(masks.shape[0], -1, dtype=np.int32)
    foul_severity = np.asarray(
        getattr(host.foul, "severity", np.zeros(masks.shape[0])), dtype=np.int32
    )
    foul_tactical = np.asarray(
        getattr(host.foul, "tactical_effect", np.zeros(masks.shape[0])),
        dtype=np.int32,
    )
    severity_dense = np.stack(
        tuple(foul_severity if index == 5 else empty_label for index in range(11)),
        axis=-1,
    )
    tactical_dense = np.stack(
        tuple(foul_tactical if index == 5 else empty_label for index in range(11)),
        axis=-1,
    )
    secondary_actor_dense = np.stack(
        tuple(
            np.asarray(host.nutmeg.defender, dtype=np.int32)
            if index == 9
            else empty_label
            for index in range(11)
        ),
        axis=-1,
    )
    team = team_dense[frame, kind]
    actor = actor_dense[frame, kind]
    if team_id is not None:
        roster_team = np.asarray(team_id, dtype=np.int32)
        derivable = (team < 0) & (actor >= 0) & (actor < roster_team.size)
        safe = np.clip(actor, 0, max(roster_team.size - 1, 0))
        team = np.where(derivable, roster_team[safe], team)
    return EventStream(
        frame=frame.astype(np.int32),
        kind=kind.astype(np.int8),
        team=team.astype(np.int32),
        actor=actor,
        secondary_actor=secondary_actor_dense[frame, kind],
        severity=severity_dense[frame, kind],
        tactical_effect=tactical_dense[frame, kind],
    )
