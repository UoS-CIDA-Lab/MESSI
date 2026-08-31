"""Command-line orchestration for ``examples/demo_match.py``."""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict
from typing import Any

import jax
import numpy as np

from soccerworld import DEFAULT_TIMEBASE, Timebase
from soccerworld.demo.arguments import build_parser
from soccerworld.demo.policy import (
    TEAM_POLICY_ADAPTER_VERSION,
    TeamPolicyAdapter,
    load_external_policy,
    policy_label,
    policy_metadata,
    style_spec,
)
from soccerworld.demo.reporting import summarize
from soccerworld.demo.rollout import rollout_single_match
from soccerworld.demo.roster import build_env
from soccerworld.policies.rule_based import (
    STYLE_PRESETS,
    make_rule_based_policy,
    policy_config_fingerprint,
)

__all__ = ["build_parser", "main", "run"]


def _positive_finite(
    parser: argparse.ArgumentParser,
    name: str,
    value: float,
) -> None:
    if not math.isfinite(value) or value <= 0.0:
        parser.error(f"{name} must be a finite positive number")


def _validate_team_options(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    for option, style in (("--home", args.home), ("--away", args.away)):
        if style != "random" and style not in STYLE_PRESETS:
            parser.error(f"{option}: unknown preset {style!r}; use --list")
    for option, policy_spec in (
        ("--home-policy", args.home_policy),
        ("--away-policy", args.away_policy),
    ):
        if not policy_spec:
            parser.error(
                f"{option} must be 'rule' or package.module:factory::checkpoint"
            )


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    _positive_finite(parser, "--seconds", args.seconds)
    _positive_finite(parser, "--control-fps", args.control_fps)
    _positive_finite(parser, "--render-fps", args.render_fps)
    if args.state_stride < 1:
        parser.error("--state-stride must be at least 1")
    if not 0.0 < args.camera_zoom <= 1.0:
        parser.error("--camera-zoom must lie in (0, 1]")

    _validate_team_options(parser, args)

    try:
        demo_timebase = Timebase(
            dt_phys=DEFAULT_TIMEBASE.dt_phys,
            control_fps=args.control_fps,
        )
    except ValueError as exc:
        parser.error(str(exc))
    n_steps = demo_timebase.control_steps_for(args.seconds)
    if n_steps < 1:
        parser.error(
            "--seconds must produce at least one "
            f"{args.control_fps:g} Hz step"
        )
    if args.bench < 0:
        parser.error("--bench must not be negative")
    if not 0.0 < args.start_stamina <= 1.0:
        parser.error("--start-stamina must lie in (0, 1]")
    return n_steps


def _load_team_policies(parser, args, env) -> dict[int, Any]:
    learned_by_team: dict[int, Any] = {}
    loaded_by_spec: dict[str, Any] = {}
    for team, (side, policy_spec) in enumerate(
        (
            ("HOME", args.home_policy),
            ("AWAY", args.away_policy),
        )
    ):
        if policy_spec == "rule":
            continue
        cache_key = policy_spec
        learned = loaded_by_spec.get(cache_key)
        if learned is None:
            print(f"[demo] {side} 외부 학습 정책 adapter를 로드합니다: {cache_key}")
            try:
                learned = load_external_policy(
                    policy_spec,
                    observation_spec=env.obs_spec(),
                    deterministic=args.learned_deterministic,
                )
            except (ImportError, OSError, TypeError, ValueError) as exc:
                parser.error(f"{side} learned policy: {exc}")
            loaded_by_spec[cache_key] = learned
            print(
                f"[demo] {side} policy={learned.fingerprint[:12]} "
                f"step={learned.checkpoint_step} params={learned.parameter_count}"
            )
        learned_by_team[team] = learned
    return learned_by_team


def _render(
    env,
    states,
    substeps,
    *,
    args,
    render_fps: float,
    resample: bool,
    policy_labels: tuple[str, str],
    learned_for_team: tuple[Any | None, Any | None],
    rule_policy,
    actual_seconds: float,
) -> None:
    # Event detection needs the pre-kickoff reset state; video frames do not.
    event_states = states
    render_states = (
        env.substep_trajectory(substeps, render_fps=render_fps)
        if resample
        else jax.tree_util.tree_map(lambda value: value[1:], states)
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
        event_states=event_states,
        out_path=args.out,
        fps=render_fps,
        mode=args.mode,
        title=f"{policy_labels[0]} vs {policy_labels[1]}",
        dump_replay=args.replay_data,
        state_stride=args.state_stride,
        replay_metadata={
            "seed": int(args.seed),
            "home_style": args.home,
            "away_style": args.away,
            "home_policy": policy_metadata(
                args.home_policy,
                args.home,
                learned_for_team[0],
            ),
            "away_policy": policy_metadata(
                args.away_policy,
                args.away,
                learned_for_team[1],
            ),
            "team_policy_adapter_version": TEAM_POLICY_ADAPTER_VERSION,
            "team_slot_masks": {
                "home": [0, int(env.n_agents)],
                "away": [int(env.n_agents), int(env.N)],
            },
            "team_styles": np.asarray(rule_policy.team_styles, dtype=float).tolist(),
            "policy_config": asdict(rule_policy.policy_config),
            "policy_config_sha256": policy_config_fingerprint(
                rule_policy.policy_config
            ),
            "rule_policy_version": int(rule_policy.rule_policy_version),
            "positional_field_version": int(rule_policy.positional_field_version),
            "deadball_positioning_version": int(
                rule_policy.deadball_positioning_version
            ),
            "compressed_match": bool(args.compressed_match),
            "manager_mode": env.manager_mode,
            "formation_mode": env.formation_mode,
            "restart_taker_mode": env.restart_taker_mode,
            "bench_size": int(env.bench_size),
            "requested_seconds": float(args.seconds),
            "simulated_seconds": float(actual_seconds),
        },
        **rich_options,
    )
    if isinstance(output, dict):
        print(f"[demo] video:    {os.path.abspath(output['mp4'])}")
        for artefact in ("events", "tracking", "metadata"):
            if output.get(artefact):
                print(
                    f"[demo] {artefact + ':':9s} "
                    f"{os.path.abspath(output[artefact])}"
                )
    else:
        print(f"[demo] video: {os.path.abspath(output)}")


def run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run an already-parsed demo command after runtime initialization."""

    if args.list:
        print("스타일 프리셋 (line_height, tempo, width, aggression, directness):")
        for name, values in STYLE_PRESETS.items():
            print(f"  {name:>14s} = {values}")
        return 0

    n_steps = _validate_args(parser, args)
    env = build_env(
        n_steps,
        control_fps=args.control_fps,
        compressed_match=args.compressed_match,
        bench_size=args.bench,
        manager=args.manager,
        taker=args.taker,
    )
    key = jax.random.PRNGKey(args.seed)
    key, style_key = jax.random.split(key)
    learned_by_team = _load_team_policies(parser, args, env)

    print("[demo] 룰 정책의 물리 기반 킥 솔버를 준비합니다.")
    rule_policy = make_rule_based_policy(
        env,
        match_key=style_key,
        team_styles=(style_spec(args.home), style_spec(args.away)),
    )
    policy = TeamPolicyAdapter(
        rule_policy,
        learned_by_team,
        players=env.N,
        home_slots=env.n_agents,
        action_dim=env.action_dim,
    )
    learned_for_team = (
        learned_by_team.get(0),
        learned_by_team.get(1),
    )
    policy_specs = (args.home_policy, args.away_policy)
    styles = (args.home, args.away)
    policy_labels = tuple(
        policy_label(spec, style, learned)
        for spec, style, learned in zip(
            policy_specs,
            styles,
            learned_for_team,
            strict=True,
        )
    )
    actual_seconds = env.timebase.seconds_for_control_steps(n_steps)
    print(
        f"[demo] {policy_labels[0]}(HOME) vs {policy_labels[1]}(AWAY), "
        f"{actual_seconds:.2f}s ({n_steps}스텝), control={env.control_fps:g}Hz, "
        f"mode={'compressed-match' if args.compressed_match else 'physical-clip'}, "
        f"seed={args.seed}"
    )
    print(
        "[demo] long-stamina reference="
        f"{env.long_stamina_reference_duration_s:.2f}s, "
        f"dynamics={env.dynamics_fingerprint[:12]}"
    )
    owner = "감독" if env.manager_mode != "composed" else env.formation_mode
    print(
        f"[demo] 감독={env.manager_mode} 벤치={env.bench_size}명/팀 "
        f"교체상한={env.max_substitutions} 포메이션지휘={owner} "
        f"키커={env.restart_taker_mode}"
    )
    print(
        f"[demo] rule_team_styles=\n"
        f"{np.round(np.asarray(rule_policy.team_styles), 3)}"
    )

    render_fps = args.render_fps
    physics_fps = env.timebase.physics_fps
    if render_fps > physics_fps:
        print(
            f"[demo] render-fps {render_fps:g} > 물리 {physics_fps:g} Hz; "
            f"{physics_fps:g} Hz로 제한합니다."
        )
        render_fps = physics_fps
    resample = (not args.no_render) and not math.isclose(
        render_fps,
        env.control_fps,
        rel_tol=0.0,
        abs_tol=1e-9,
    )

    started = time.perf_counter()
    states, _, substeps = rollout_single_match(
        env,
        policy,
        key,
        n_steps,
        collect_substeps=resample,
        start_stamina=args.start_stamina,
    )
    elapsed = time.perf_counter() - started
    print(f"[rollout] {n_steps}스텝 {elapsed:.1f}s (첫 JIT 컴파일 포함)")
    summarize(env, states, policy_labels)

    if args.no_render:
        print("[demo] --no-render: 영상 생성을 건너뜁니다.")
        return 0

    _render(
        env,
        states,
        substeps,
        args=args,
        render_fps=render_fps,
        resample=resample,
        policy_labels=policy_labels,
        learned_for_team=learned_for_team,
        rule_policy=rule_policy,
        actual_seconds=actual_seconds,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse and run the demo for direct internal callers."""

    parser = build_parser()
    return run(parser, parser.parse_args(argv))
