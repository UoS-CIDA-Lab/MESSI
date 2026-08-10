#!/usr/bin/env python3
"""SoccerBC 환경 데모 — 룰 정책 두 팀을 롤아웃하고 선택한 모드로 렌더링한다.

환경 밖에서는 공개 인터페이스인 ``obs -> action -> step_env_array``만 사용한다. 롤아웃은
``jax.lax.scan`` 하나로 JIT 컴파일하며, 렌더링은 환경의 단일 진입점 ``render_mp4``에 맡긴다.

예시:
  python demo_match.py
  python demo_match.py --home tiki_taka --away long_ball --seconds 120
  python demo_match.py --mode light
  python demo_match.py --mode rich
  python demo_match.py --render-fps 100 --mode rich
  python demo_match.py --no-render
  python demo_match.py --list

기본 출력은 실행 디렉터리 기준 ``./replays/<unix초>/match.mp4``이다. ``--dump``를 주면 같은
폴더에 ``state.jsonl``과 ``events.txt``도 저장한다.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_ROOT = os.path.join(PROJECT_ROOT, "env")
if ENV_ROOT not in sys.path:
    sys.path.insert(0, ENV_ROOT)

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from constants import (
    DEFAULT_CONTROL_FPS,
    FOUL_NONE,
    STYLE_PRESETS,
    TOUCH_DRIBBLE,
    TOUCH_NONE,
)
from env import SoccerEnv
from policy import make_rule_based_policy


def build_env(n_steps: int) -> SoccerEnv:
    """환경 기본 roster/config를 그대로 쓰는 데모 인스턴스를 만든다."""
    return SoccerEnv(
        game_duration=n_steps,
        control_fps=DEFAULT_CONTROL_FPS,
    )


def rollout(env, policy, key, n_steps: int, *, collect_substeps: bool = False):
    """룰 정책을 JIT 롤아웃하고 초기 상태를 포함한 궤적과 선택적 물리 서브스텝을 반환한다."""
    key, reset_key = jax.random.split(key)
    obs0, state0 = env.reset_array(reset_key)

    def step_fn(carry, _):
        obs, state, rng = carry
        rng, action_key, step_key = jax.random.split(rng, 3)
        action = policy(obs, action_key)
        obs_next, state_next, _, _, info = env.step_env_array(
            step_key,
            state,
            action,
            collect_substeps=collect_substeps,
            include_bc_info=False,
        )
        substeps = info["substeps"] if collect_substeps else None
        return (obs_next, state_next, rng), (state_next, substeps)

    run = jax.jit(
        lambda carry: lax.scan(step_fn, carry, xs=None, length=n_steps)
    )
    (_, state_last, _), (trajectory, substeps) = run((obs0, state0, key))
    states = jax.tree_util.tree_map(
        lambda initial, stacked: jnp.concatenate(
            (initial[None], stacked), axis=0
        ),
        state0,
        trajectory,
    )
    return (
        jax.device_get(states),
        jax.device_get(state_last),
        jax.device_get(substeps) if collect_substeps else None,
    )


def _shape_error(env, states, team_id: np.ndarray) -> tuple[float, float]:
    """각 팀의 중심을 제거한 뒤 시작 포메이션에서 벗어난 평균 거리(m)를 계산한다."""
    positions = np.asarray(states.player_pos)
    is_gk = np.asarray(states.gk_indices[0], dtype=bool)
    home = np.asarray(env.base_formation)
    errors: list[float] = []

    for team in (0, 1):
        mask = (team_id == team) & (~is_gk)
        team_positions = positions[:, mask]
        centered = team_positions - team_positions.mean(axis=1, keepdims=True)
        candidates = []
        for orientation in (1.0, -1.0):
            formation = home[mask] * orientation
            formation -= formation.mean(axis=0)
            candidates.append(
                np.linalg.norm(centered - formation, axis=2).mean(axis=1)
            )
        errors.append(float(np.minimum(candidates[0], candidates[1]).mean()))

    return errors[0], errors[1]


def summarize(env, states, names: tuple[str, str]) -> None:
    """스코어, 점유, 터치, 반칙, 카드, 퇴장과 포메이션 오차를 출력한다."""
    team_id = np.asarray(states.team_id[0], dtype=int)
    touch = np.asarray(states.touch)
    kick_touch = (touch != TOUCH_NONE) & (touch != TOUCH_DRIBBLE)
    foul_kind = np.asarray(states.foul_kind)
    foul_active = foul_kind != FOUL_NONE
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
    active = np.asarray(states.active_player[-1], dtype=bool)
    shape = _shape_error(env, states, team_id)

    print(
        f"\n[match] {names[0]}(T0, 적) {score[0]} - "
        f"{score[1]} {names[1]}(T1, 청)"
    )
    for team in (0, 1):
        mask = team_id == team
        possession_pct = float((possession == team).mean()) * 100.0
        foul_count = int((team_id[foul_actors] == team).sum())
        print(
            f"  T{team} {names[team]:>14s} | "
            f"점유 {possession_pct:4.1f}%  "
            f"킥성터치 {int(kick_touch[:, mask].sum()):4d}  "
            f"드리블 {int((touch[:, mask] == TOUCH_DRIBBLE).sum()):4d}  "
            f"파울 {foul_count:2d}  "
            f"경고 {int(yellow[mask].sum()):2d}  "
            f"퇴장 {int((~active[mask]).sum()):2d}  "
            f"대형오차 {shape[team]:.1f}m"
        )


def _style_spec(name: str):
    return None if name == "random" else name


def _positive_finite(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        parser.error(f"{name} must be a finite positive number")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SoccerBC 룰 정책 경기를 실행하고 light/rich 모드로 렌더링합니다."
    )
    parser.add_argument(
        "--home",
        default="random",
        help="T0(적) 스타일 프리셋 또는 random",
    )
    parser.add_argument(
        "--away",
        default="random",
        help="T1(청) 스타일 프리셋 또는 random",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=60.0,
        help="시뮬레이션 경기 길이(초, 기본 60)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_CONTROL_FPS,
        help="출력 영상 fps(기본 환경 control fps)",
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        default=None,
        help=(
            "렌더 상태 샘플링 fps. control fps보다 높으면 물리 서브스텝을 "
            "수집하며, 생략하면 --fps를 사용"
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="출력 mp4 경로(생략 시 ./replays/<unix초>/match.mp4)",
    )
    parser.add_argument(
        "--mode",
        "--render-mode",
        dest="mode",
        default="light",
        choices=("light", "rich"),
        help="light=빠른 탑다운 / rich=방송형 3D(느림)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="롤아웃과 경기 요약만 실행",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="영상 옆에 state.jsonl과 events.txt도 저장",
    )
    parser.add_argument(
        "--state-stride",
        type=int,
        default=1,
        help="state.jsonl 저장 프레임 간격(기본 1)",
    )
    parser.add_argument(
        "--dynamic-camera",
        action="store_true",
        help="rich 모드에서 공을 따라가는 동적 카메라 사용",
    )
    parser.add_argument(
        "--camera-zoom",
        type=float,
        default=0.66,
        help="rich 동적 카메라 화면 범위 비율(0 초과 1 이하)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="스타일 프리셋 목록을 출력하고 종료",
    )
    args = parser.parse_args(argv)

    if args.list:
        print("스타일 프리셋 (line_height, tempo, width, aggression, directness):")
        for name, values in STYLE_PRESETS.items():
            print(f"  {name:>14s} = {values}")
        return 0

    _positive_finite(parser, "--seconds", args.seconds)
    _positive_finite(parser, "--fps", args.fps)
    if args.render_fps is not None:
        _positive_finite(parser, "--render-fps", args.render_fps)
    if args.state_stride < 1:
        parser.error("--state-stride must be at least 1")
    if not 0.0 < args.camera_zoom <= 1.0:
        parser.error("--camera-zoom must lie in (0, 1]")

    for option, style in (("--home", args.home), ("--away", args.away)):
        if style != "random" and style not in STYLE_PRESETS:
            parser.error(f"{option}: unknown preset {style!r}; use --list")

    n_steps = int(round(args.seconds * DEFAULT_CONTROL_FPS))
    if n_steps < 1:
        parser.error(
            f"--seconds must produce at least one {DEFAULT_CONTROL_FPS:g} Hz step"
        )

    env = build_env(n_steps)
    key = jax.random.PRNGKey(args.seed)
    key, style_key = jax.random.split(key)
    print("[demo] 룰 정책의 물리 기반 킥 솔버를 준비합니다.")
    policy = make_rule_based_policy(
        env,
        match_key=style_key,
        team_styles=(_style_spec(args.home), _style_spec(args.away)),
    )
    actual_seconds = n_steps / env.control_fps
    print(
        f"[demo] {args.home}(T0) vs {args.away}(T1), "
        f"{actual_seconds:.2f}s ({n_steps}스텝), seed={args.seed}"
    )
    print(f"[demo] team_styles=\n{np.round(np.asarray(policy.team_styles), 3)}")

    render_fps = args.render_fps if args.render_fps is not None else args.fps
    physics_fps = 1.0 / env.e_cfg.dt_phys
    if render_fps > physics_fps:
        print(
            f"[demo] render-fps {render_fps:g} > 물리 {physics_fps:g} Hz; "
            f"{physics_fps:g} Hz로 제한합니다."
        )
        render_fps = physics_fps
    densify = (not args.no_render) and render_fps > env.control_fps

    started = time.perf_counter()
    states, _, substeps = rollout(
        env,
        policy,
        key,
        n_steps,
        collect_substeps=densify,
    )
    elapsed = time.perf_counter() - started
    print(f"[rollout] {n_steps}스텝 {elapsed:.1f}s (첫 JIT 컴파일 포함)")
    summarize(env, states, (args.home, args.away))

    if args.no_render:
        print("[demo] --no-render: 영상 생성을 건너뜁니다.")
        return 0

    render_states = (
        env.substep_trajectory(substeps, render_fps=render_fps)
        if densify
        else states
    )
    rich_options = (
        {
            "dynamic_cam": args.dynamic_camera,
            "cam_zoom": args.camera_zoom,
        }
        if args.mode == "rich"
        else {}
    )
    output = env.render_mp4(
        render_states,
        out_path=args.out,
        fps=render_fps,
        mode=args.mode,
        title=f"{args.home} vs {args.away}",
        dump_state=args.dump,
        dump_events=args.dump,
        state_stride=args.state_stride,
        **rich_options,
    )
    if isinstance(output, dict):
        print(f"[demo] video:  {os.path.abspath(output['mp4'])}")
        print(f"[demo] events: {os.path.abspath(output['events'])}")
        print(f"[demo] state:  {os.path.abspath(output['state'])}")
    else:
        print(f"[demo] video: {os.path.abspath(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
