"""물리 kick flag와 구분한 실제 의도적 release의 host 분류."""

from dataclasses import dataclass

import numpy as np

from messi.core.action import INTENT_PASS, INTENT_SHOOT
from messi.core.contact import MECHANISM_HEAD, OUTCOME_RELEASE


@dataclass(frozen=True, slots=True)
class AppliedReleaseClassification:
    release: np.ndarray
    head_candidate: np.ndarray
    head_release: np.ndarray
    head_power_known: np.ndarray
    head_zero_power: np.ndarray
    actor_power: np.ndarray


def classify_applied_releases(event, actions=None):
    """[frame] receipt와 [frame,player] action을 actor로 exact 조인한다.

    Recording의 IntentAction force_to_ball[...,2]는 공개 decode와 같은 nonfinite→0,
    [-1,1] clip 및 L∞ radial power로 해석한다. decoded force_power도 받을 수 있다.
    actions가 없으면 HEAD의 positive/noop는 미관측이며 추측하여 release로 세지 않는다.
    kick_applied의 foot 물리 의미는 변경하지 않는다.
    """
    occurred = np.asarray(event.occurred, dtype=bool)
    intent = np.asarray(event.intent)
    kick = np.asarray(event.kick_applied, dtype=bool)
    if occurred.ndim != 1 or intent.shape != occurred.shape or kick.shape != occurred.shape:
        raise ValueError('release receipt는 같은 [frame] shape여야 한다')
    mechanism = np.asarray(getattr(event, 'mechanism', np.zeros_like(intent)))
    outcome = np.asarray(getattr(event, 'outcome', np.zeros_like(intent)))
    actor = np.asarray(event.actor)
    if any(v.shape != occurred.shape for v in (mechanism, outcome, actor)):
        raise ValueError('release actor/mechanism/outcome shape 불일치')
    if actor.dtype.kind not in 'iu':
        raise ValueError('release actor는 integer slot index여야 한다')
    requested = np.isin(intent, (INTENT_PASS, INTENT_SHOOT))
    candidate = occurred & requested & (mechanism == MECHANISM_HEAD) & (outcome == OUTCOME_RELEASE)
    power = np.full(occurred.shape, np.nan, dtype=np.float64)
    if actions is not None:
        values = getattr(actions, 'force_power', None)
        if values is not None:
            values = np.asarray(values)
            if values.ndim != 2 or values.shape[0] != len(occurred):
                raise ValueError('decoded force_power는 [frame,player]여야 한다')
        elif hasattr(actions, 'force_to_ball'):
            raw = np.asarray(actions.force_to_ball)
            if raw.ndim != 3 or raw.shape[0] != len(occurred) or raw.shape[-1] != 2:
                raise ValueError('IntentAction.force_to_ball은 [frame,player,2]여야 한다')
            sanitized = np.clip(np.nan_to_num(raw, nan=0., posinf=0., neginf=0.), -1., 1.)
            values = np.max(np.abs(sanitized), axis=-1)
        if values is not None:
            valid_actor = (actor >= 0) & (actor < values.shape[1])
            frames = np.flatnonzero(valid_actor)
            power[frames] = values[frames, actor[frames].astype(int)]
    known = candidate & np.isfinite(power)
    head = known & (power > 0.)
    return AppliedReleaseClassification(
        release=occurred & requested & (kick | head), head_candidate=candidate,
        head_release=head, head_power_known=known, head_zero_power=known & (power <= 0.),
        actor_power=power,
    )
