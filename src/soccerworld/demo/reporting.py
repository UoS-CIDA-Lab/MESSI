"""Human-readable match summaries for the shipped demo."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from soccerworld.data import FoulEventCode, TouchEventCode
from soccerworld.rendering import RESTART_LABEL
from soccerworld.tactics import (
    FORMATION_LAYOUT_NAMES,
    PLAYER_ROLE_NAMES,
    classify_player_roles,
)

__all__ = ["summarize"]


def _shape_error(env, states, team_id: np.ndarray) -> tuple[float, float]:
    """Measure live outfield shape against each frame's instructed formation."""

    # Positions and anchors must share the kickoff attack frame across halftime.
    attack_dir = np.asarray(states.attack_dir, np.float64)
    flip = 1.0 if attack_dir[0, 0] > 0 else -1.0
    positions = np.asarray(states.player_pos) * (flip * attack_dir)[..., None]
    is_gk = np.asarray(states.gk_indices[0], dtype=bool)
    on_pitch = np.asarray(states.on_pitch, dtype=bool)
    layout = np.asarray(states.layout_index, dtype=np.int32)

    anchors: dict[bytes, np.ndarray] = {}

    def anchor_of(frame: int) -> np.ndarray:
        key = layout[frame].tobytes() + on_pitch[frame].tobytes()
        cached = anchors.get(key)
        if cached is None:
            cached = np.asarray(
                env.formation_layout_anchors(
                    jnp.asarray(layout[frame], jnp.int32),
                    jnp.asarray(on_pitch[frame], bool),
                )
            )
            anchors[key] = cached
        return cached

    frames = positions.shape[0]
    target = np.stack([anchor_of(frame) for frame in range(frames)])
    errors: list[float] = []
    for team in (0, 1):
        mask = (team_id == team) & (~is_gk)
        live = on_pitch[:, mask].astype(np.float64)
        count = live.sum(axis=1)
        usable = count > 0
        if not usable.any():
            errors.append(float("nan"))
            continue
        here = positions[:, mask]
        want = target[:, mask]
        centre_here = (here * live[..., None]).sum(
            axis=1,
            keepdims=True,
        ) / np.maximum(count, 1.0)[:, None, None]
        centre_want = (want * live[..., None]).sum(
            axis=1,
            keepdims=True,
        ) / np.maximum(count, 1.0)[:, None, None]
        distance = np.linalg.norm(
            (here - centre_here) - (want - centre_want),
            axis=2,
        )
        per_frame = (distance * live).sum(axis=1) / np.maximum(count, 1.0)
        errors.append(float(per_frame[usable].mean()))
    return errors[0], errors[1]


def _policy_report(env, states) -> None:
    """Report whether substitution, formation, and taker policies acted."""

    team_id = np.asarray(states.team_id[0], dtype=int)
    roles = np.asarray(
        classify_player_roles(
            env.formation_home(jax.tree_util.tree_map(lambda x: x[0], states)),
            jnp.asarray(states.gk_indices[0]) == 1,
            jnp.asarray(states.team_id[0]),
            jnp.asarray(states.active_player[0]),
        )
    )

    generation = np.asarray(states.slot_generation)
    changed = generation[1:] != generation[:-1]
    print("\n[정책] 교체 · 포메이션 · 세트피스 키커")
    if env.bench_size == 0:
        print("  교체    벤치 없음(--bench 0) — 교체 정책이 꺼져 있다")
    else:
        remaining = np.asarray(states.subs_remaining)
        windows = np.asarray(states.sub_windows_used)
        for team in (0, 1):
            mine = changed[:, team_id == team].any(axis=1)
            when = np.flatnonzero(mine)
            used = int(remaining[0, team]) - int(remaining[-1, team])
            stamp = ", ".join(
                f"{tick / env.control_fps / 60.0:.1f}분" for tick in when[:8]
            )
            print(
                f"  교체    {['HOME', 'AWAY'][team]:<4s} {used}명 / "
                f"정지 {int(windows[-1, team])}회"
                + (f"  [{stamp}]" if len(when) else "  (없음)")
            )

    layout = np.asarray(states.layout_index)
    for team in (0, 1):
        sequence = layout[:, team]
        switch = np.flatnonzero(sequence[1:] != sequence[:-1])
        shapes = " → ".join(
            FORMATION_LAYOUT_NAMES[int(sequence[index])]
            for index in [0, *(switch + 1)]
        )
        print(
            f"  포메이션 {['HOME', 'AWAY'][team]:<4s} "
            f"{len(switch)}회 전환  {shapes}"
        )

    kind = np.asarray(states.restart_kind)
    onset = np.flatnonzero((kind[1:] != kind[:-1]) & (kind[1:] > 0)) + 1
    taker = np.asarray(states.pending_taker)
    tally: dict[str, dict[str, int]] = {}
    for frame in onset:
        slot = int(taker[frame])
        if not 0 <= slot < roles.shape[0]:
            continue
        label = RESTART_LABEL.get(int(kind[frame]), str(int(kind[frame])))
        tally.setdefault(label, {})
        name = PLAYER_ROLE_NAMES[int(roles[slot])]
        tally[label][name] = tally[label].get(name, 0) + 1
    if not tally:
        print("  키커    재개가 한 번도 없었다 — 표본 없음")
    for label, counts in sorted(
        tally.items(),
        key=lambda item: -sum(item[1].values()),
    ):
        total = sum(counts.values())
        detail = "  ".join(
            f"{role}:{count}"
            for role, count in sorted(counts.items(), key=lambda item: -item[1])
        )
        print(f"  키커    {label:<10s} n={total:3d}   {detail}")


def summarize(env, states, names: tuple[str, str]) -> None:
    """Print score, possession, events, discipline, and formation error."""

    team_id = np.asarray(states.team_id[0], dtype=int)
    touch = np.asarray(states.touch)
    kick_touch = (touch != TouchEventCode.NONE) & (touch != TouchEventCode.DRIBBLE)
    foul_kind = np.asarray(states.foul_kind)
    foul_active = foul_kind != FoulEventCode.NONE
    foul_onset = foul_active & np.concatenate(
        (np.array([True]), ~foul_active[:-1])
    )
    foul_actors = np.asarray(states.foul_actor, dtype=int)[foul_onset]
    foul_actors = foul_actors[
        (foul_actors >= 0) & (foul_actors < team_id.shape[0])
    ]
    possession = np.asarray(states.poss_team)
    score = np.asarray(states.score[-1], dtype=int)
    yellow = np.asarray(states.yellow_cards[-1], dtype=int)
    sent_off = np.asarray(states.sent_off[-1], dtype=bool)
    shape = _shape_error(env, states, team_id)

    print(
        f"\n[match] {names[0]}(HOME, 적) {score[0]} - "
        f"{score[1]} {names[1]}(AWAY, 청)"
    )
    for team in (0, 1):
        mask = team_id == team
        possession_pct = float((possession == team).mean()) * 100.0
        foul_count = int((team_id[foul_actors] == team).sum())
        print(
            f"  {['HOME', 'AWAY'][team]:<4s} {names[team]:>14s} | "
            f"점유 {possession_pct:4.1f}%  "
            f"킥성터치 {int(kick_touch[:, mask].sum()):4d}  "
            f"드리블 {int((touch[:, mask] == TouchEventCode.DRIBBLE).sum()):4d}  "
            f"파울 {foul_count:2d}  "
            f"경고 {int(yellow[mask].sum()):2d}  "
            f"퇴장 {int(sent_off[mask].sum()):2d}  "
            f"대형오차 {shape[team]:.1f}m"
        )
    _policy_report(env, states)
