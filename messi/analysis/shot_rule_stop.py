"""명시적 offside 종결을 제외한 민감도 분석. Provider-valid 판정이 아니다."""

from numbers import Integral

from messi.analysis.causal import (
    SHOT_OUTCOMES,
    build_shot_on_target_proxy,
    is_goal_directed_shot,
)

_COUNTS = ('metadata_known_count', 'offside_same_frame_count', 'offside_later_count',
           'offside_timing_unknown_count', 'other_rule_stopped_count')
_TERMINALS = frozenset(('contact', 'same_actor_retouch', 'boundary', 'goal', 'capture_end',
                        'offside', 'foul', 'restart_retouch', 'halftime'))


def _finish(proxy, counts):
    raw = proxy['release_count']
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in counts.values()):
        raise ValueError('rule-stop count는 비음수 정수여야 한다')
    offside = sum(counts[k] for k in _COUNTS if k.startswith('offside_'))
    known = counts['metadata_known_count']
    if known > raw or offside + counts['other_rule_stopped_count'] > known:
        raise ValueError('rule-stop count와 raw release 분모가 일치하지 않는다')
    remaining = raw - offside
    if proxy['upper_count'] > remaining:
        raise ValueError('offside 종결을 SOT 분자와 중복 계산할 수 없다')
    return {
        'basis': 'direct_causal_stopped_offside_sensitivity_not_provider_valid_shots',
        'release_count': raw, **{k: int(counts[k]) for k in _COUNTS},
        'metadata_unknown_count': raw - known,
        'metadata_coverage': known / raw if raw else None,
        'offside_linked_count': offside,
        'offside_excluded_proxy_release_count': remaining,
        'raw_on_target_proxy_lower_count': proxy['lower_count'],
        'raw_on_target_proxy_upper_count': proxy['upper_count'],
        'offside_excluded_on_target_proxy_lower_share': proxy['lower_count'] / remaining if remaining else None,
        'offside_excluded_on_target_proxy_upper_share': proxy['upper_count'] / remaining if remaining else None,
    }


def summarize_shot_rule_stops(rows, *, goal_line_x=52.5):
    """이미 near-goal로 선택된 causal rows만 받는다. Missing timing은 제외를 막지 않는다.

    outcome=stopped와 terminal_kind=offside의 동시 증거만 제외한다. Terminal 정보가
    누락/불일치하면 unknown이며 임의로 제외하지 않는다. Late offside의 관여 actor나
    provider 취소 여부는 추정하지 않는다.
    """
    counts = dict.fromkeys(_COUNTS, 0)
    raw = {k: 0 for k in SHOT_OUTCOMES}
    directed = {k: 0 for k in SHOT_OUTCOMES}
    for row in rows:
        outcome, terminal = row.get('outcome'), row.get('terminal_kind')
        raw[outcome] += 1
        if is_goal_directed_shot(row, goal_line_x=goal_line_x):
            directed[outcome] += 1
        if terminal not in _TERMINALS:
            continue
        if terminal in ('offside', 'foul', 'restart_retouch', 'halftime') and outcome != 'stopped':
            continue
        if outcome == 'stopped' and terminal not in ('offside', 'foul', 'restart_retouch', 'halftime'):
            continue
        counts['metadata_known_count'] += 1
        if outcome != 'stopped':
            continue
        if terminal != 'offside':
            counts['other_rule_stopped_count'] += 1
            continue
        source, end = row.get('source_frame'), row.get('terminal_frame')
        valid = all(isinstance(v, Integral) and not isinstance(v, bool) and v >= 0 for v in (source, end))
        timing = 'same_frame' if valid and source == end else 'later' if valid and end > source else 'timing_unknown'
        counts[f'offside_{timing}_count'] += 1
    return _finish(build_shot_on_target_proxy(len(rows), directed, raw), counts)


def normalize_shot_rule_stops(value, proxy, *, stopped_count=None):
    counts = dict.fromkeys(_COUNTS, 0) if value is None else {key: value[key] for key in _COUNTS}
    expected = _finish(proxy, counts)
    if stopped_count is not None and expected['offside_linked_count'] + expected['other_rule_stopped_count'] > stopped_count:
        raise ValueError('직접 rule-stop 분자가 raw stopped outcome보다 많다')
    if value is not None and value != expected:
        raise ValueError('rule-stop summary의 파생 값과 raw proxy가 일치하지 않는다')
    return expected


def merge_shot_rule_stops(values):
    counts = dict.fromkeys(_COUNTS, 0)
    proxy = dict.fromkeys(('release_count', 'lower_count', 'upper_count'), 0)
    for value in values:
        for key in _COUNTS:
            counts[key] += value[key]
        proxy['release_count'] += value['release_count']
        proxy['lower_count'] += value['raw_on_target_proxy_lower_count']
        proxy['upper_count'] += value['raw_on_target_proxy_upper_count']
    return _finish(proxy, counts)
