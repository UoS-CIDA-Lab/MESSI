"""Short, deterministic render scenes for validating public control surfaces.

The module stays numerical-stack lazy so ``--dry-run`` works without importing JAX, NumPy, or the
optional renderer. Runtime imports happen only after the command line has been validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from soccerworld.demo.artifact_receipts import (
    assert_git_source_unchanged,
    git_source_receipt,
    media_receipt,
    rule_policy_receipt,
    state_tree_sha256,
)

SHOWCASE_SCHEMA = "soccerworld.render-showcase.v2"
DEFAULT_SCENE_SECONDS = 10.0
SMOKE_SCENE_SECONDS = 0.4
MIN_POST_RESTART_LIVE_SECONDS = 1.0


@dataclass(frozen=True)
class SceneSpec:
    """One referee boundary plus optional first-frame management command."""

    slug: str
    title: str
    restart_kind: str
    observed_xy_fraction: tuple[float, float]
    formation: str | None = None
    substitution: bool = False
    indirect: bool = False
    taker_start_offset_m: tuple[float, float] | None = None
    kick_vector_override: tuple[float, float] | None = None


SCENES = (
    SceneSpec(
        "kickoff",
        "Kick-off · external formation and taker",
        "KICKOFF",
        (0.0, 0.0),
        formation="4-3-3 mid normal",
    ),
    SceneSpec(
        "throw-in",
        "Throw-in · external substitution and taker",
        "THROW_IN",
        (-0.15, 1.0),
        substitution=True,
    ),
    SceneSpec(
        "corner",
        "Corner · external taker",
        "CORNER",
        (1.0, 1.0),
        # A kickoff-reset striker needs ~10.4 s merely to reach this corner,
        # before the configured 3 s hold. Stage the observed player inside
        # the attacking third so a ~10 s verification clip includes release.
        taker_start_offset_m=(-24.0, -12.0),
    ),
    SceneSpec(
        "free-kick",
        "Direct free kick · external taker",
        "FREE_KICK",
        (0.2, -0.18),
    ),
    SceneSpec(
        "penalty",
        "Penalty · external taker",
        "PENALTY",
        (0.8, 0.0),
        # Preserve the run-up while leaving enough of a ten-second clip to
        # show the resulting ball trajectory after the configured hold.
        taker_start_offset_m=(-18.0, 0.0),
        # A normal penalty shot reaches the goal line in under one second.
        # This lawful, forward-and-lateral external action keeps the ball in
        # view long enough for the verification clip to show its trajectory.
        kick_vector_override=(0.08, 0.22),
    ),
    SceneSpec(
        "goal-kick",
        "Goal kick · external goalkeeper taker",
        "GOAL_KICK",
        (-1.0, -0.3),
    ),
)
SCENES_BY_SLUG = {scene.slug: scene for scene in SCENES}

__all__ = [
    "DEFAULT_SCENE_SECONDS",
    "SCENES",
    "SHOWCASE_SCHEMA",
    "SceneSpec",
    "build_parser",
    "main",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render short SoccerWorld restart scenes while exercising external StepCommand "
            "actions, substitutions, formations, and set-piece takers."
        )
    )
    parser.add_argument(
        "--scene",
        action="append",
        choices=tuple(SCENES_BY_SLUG),
        help="scene to run; repeat for several (default: all six)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_SCENE_SECONDS,
        help=f"simulated seconds per scene (default: {DEFAULT_SCENE_SECONDS:g})",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: a new replays/showcase/<UTC timestamp> directory)",
    )
    parser.add_argument("--mode", choices=("light", "rich"), default="light")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="run real transitions and write the manifest, but do not encode MP4/replay files",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"cap every selected scene at {SMOKE_SCENE_SECONDS:g}s",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the scene/control plan without importing numerical or rendering dependencies",
    )
    return parser


def _selected_scenes(names: Sequence[str] | None) -> tuple[SceneSpec, ...]:
    if not names:
        return SCENES
    return tuple(SCENES_BY_SLUG[name] for name in dict.fromkeys(names))


def _effective_seconds(args: argparse.Namespace) -> float:
    return min(args.seconds, SMOKE_SCENE_SECONDS) if args.smoke else args.seconds


def _plan(args: argparse.Namespace, scenes: Sequence[SceneSpec]) -> dict[str, object]:
    seconds = _effective_seconds(args)
    return {
        "schema": SHOWCASE_SCHEMA,
        "dry_run": True,
        "seconds_per_scene": seconds,
        "mode": args.mode,
        "render": not args.no_render,
        "out_dir": args.out_dir or "<new replays/showcase/UTC-timestamp directory>",
        "boundary": {
            "scene_start_setup": (
                "SoccerEnv.prepare_observed_restart; pure setup, zero simulated transitions"
            ),
            "state_sequence": (
                "SoccerEnv.transition(StepCommand); rule-policy actions and external commands"
            ),
        },
        "scenes": [
            {
                "slug": scene.slug,
                "restart_kind": scene.restart_kind,
                "external_actions": True,
                "external_set_piece_taker": True,
                "external_substitution": scene.substitution,
                "external_formation": scene.formation,
                "scene_start_player_staging": scene.taker_start_offset_m,
                "external_kick_vector_override": scene.kick_vector_override,
            }
            for scene in scenes
        ],
    }


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("replays") / "showcase" / f"{stamp}-{os.getpid()}"


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative_render_outputs(outputs: object, output_root: Path) -> dict[str, str]:
    if isinstance(outputs, (str, os.PathLike)):
        outputs = {"mp4": outputs}
    if not isinstance(outputs, dict):
        raise TypeError(f"renderer returned unsupported output type: {type(outputs).__name__}")
    root = output_root.resolve()
    normalized = {}
    for name, value in outputs.items():
        if value is None:
            continue
        path = Path(value).resolve()
        try:
            normalized[str(name)] = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"renderer output escaped showcase directory: {path}") from exc
    return normalized


def _mp4_output(outputs: dict[str, str], output_root: Path) -> Path:
    matches = [output_root / value for value in outputs.values() if value.endswith(".mp4")]
    if len(matches) != 1:
        raise RuntimeError(f"renderer must return exactly one MP4 output, got {matches}")
    return matches[0]


def _compile_rollout(env, policy, n_steps, *, jax, jnp, kick_vector_slice):
    """Compile one scalar scan reused by every equal-duration scene."""

    empty_command = env.empty_command()

    def rollout(
        setup_state,
        opening_command,
        keys,
        action_override_mask,
        action_override_vector,
        action_override_restart_kind,
    ):
        observation = env.get_obs_array(setup_state)

        def step(carry, inputs):
            obs, state = carry
            frame, frame_key = inputs
            policy_key, transition_key = jax.random.split(frame_key)
            action = policy(obs, policy_key, env.affordance_view(state))
            override_active = action_override_mask & (
                state.restart_kind == action_override_restart_kind
            )
            action = action.at[:, kick_vector_slice].set(
                jnp.where(
                    override_active[:, None],
                    action_override_vector,
                    action[:, kick_vector_slice],
                )
            )
            command = jax.lax.cond(
                frame == 0,
                lambda _: opening_command,
                lambda _: empty_command,
                operand=None,
            ).with_player_actions(action)
            result = env.transition(transition_key, state, command)
            return (result.observation, result.state), (
                result.state,
                result.command_result,
            )

        (_, final_state), (trajectory, command_results) = jax.lax.scan(
            step,
            (observation, setup_state),
            (jnp.arange(n_steps, dtype=jnp.int32), keys),
        )
        return final_state, trajectory, command_results

    return jax.jit(rollout)


def _prepare_scene(
    env,
    scene: SceneSpec,
    reset_key,
    *,
    jnp,
    np,
    formation_layout_names,
    RestartKind,
    Team,
):
    """Create the non-transition setup state and the first real StepCommand."""

    base_state = env.reset_state(reset_key)
    restart_kind = int(RestartKind[scene.restart_kind])
    restart_team = int(Team.HOME)
    observed = jnp.asarray(
        [
            scene.observed_xy_fraction[0] * env.hx,
            scene.observed_xy_fraction[1] * env.hy,
            env.r_ball,
        ],
        dtype=jnp.float32,
    )
    canonical = env.canonical_restart_spot(
        base_state,
        restart_kind,
        restart_team,
        observed,
        scene.indirect,
    )

    active = np.asarray(base_state.active_player, dtype=bool)
    team_ids = np.asarray(base_state.team_id)
    keepers = np.asarray(base_state.gk_indices, dtype=bool)
    positions = np.asarray(base_state.player_pos)
    team_slots = np.flatnonzero(active & (team_ids == restart_team))
    distances = np.linalg.norm(
        positions[team_slots, :2] - np.asarray(canonical)[:2], axis=1
    )
    nearest = team_slots[np.argsort(distances)]
    if scene.restart_kind == "GOAL_KICK" and np.any(keepers[team_slots]):
        requested_taker = int(team_slots[keepers[team_slots]][0])
    else:
        outfield = nearest[~keepers[nearest]]
        requested_taker = int(outfield[0] if len(outfield) else nearest[0])
    initial_candidates = nearest[nearest != requested_taker]
    initial_taker = int(initial_candidates[0])

    player_staging = None
    if scene.taker_start_offset_m is not None:
        original = np.asarray(base_state.player_pos)[requested_taker]
        desired = canonical[:2] + jnp.asarray(
            scene.taker_start_offset_m, dtype=jnp.float32
        )
        blockers = jnp.asarray(active).at[requested_taker].set(False)
        staged = env.nearest_free_position(
            desired,
            base_state.player_pos,
            blockers,
            orientation=base_state.attack_dir[requested_taker],
        )
        # This is explicitly part of scene-start construction, before the
        # observed-restart adapter and before every simulated transition.
        base_state = base_state._replace(
            player_pos=base_state.player_pos.at[requested_taker].set(staged)
        )
        player_staging = {
            "method": "State scene-start placement before prepare_observed_restart",
            "resolver": "SoccerEnv.nearest_free_position",
            "counts_as_transition": False,
            "player_slot": requested_taker,
            "original_pos_m": [float(value) for value in original],
            "requested_pos_m": [float(value) for value in np.asarray(desired)],
            "resolved_pos_m": [float(value) for value in np.asarray(staged)],
            "reason": "include physical approach, configured hold, and release in an ~10 s clip",
        }

    engine_metadata = env.dynamics_metadata()["config"]["engine"]
    timer_name = "penalty_substeps" if scene.restart_kind == "PENALTY" else "restart_substeps"
    setup_state = env.prepare_observed_restart(
        base_state,
        restart_kind,
        restart_team,
        initial_taker,
        int(engine_metadata[timer_name]),
        observed,
        scene.indirect,
    )

    opening = env.empty_command()
    takers = opening.set_piece_takers.with_request(
        restart_team,
        restart_kind,
        player_slot=requested_taker,
    )
    opening = opening.with_set_piece_takers(takers)
    management = {}
    if scene.formation is not None:
        layout_index = formation_layout_names.index(scene.formation)
        formations = opening.formations.with_request(
            restart_team,
            layout_index=layout_index,
        )
        opening = opening.with_formations(formations)
        management["formation"] = {
            "team": "HOME",
            "layout_index": layout_index,
            "layout_name": scene.formation,
        }
    if scene.substitution:
        substitutes = team_slots[
            (~keepers[team_slots])
            & (team_slots != requested_taker)
            & (team_slots != initial_taker)
        ]
        out_slot = int(substitutes[0])
        substitutions = opening.substitutions.with_request(
            restart_team,
            0,
            out_slot=out_slot,
            bench_index=0,
        )
        opening = opening.with_substitutions(substitutions)
        management["substitution"] = {
            "team": "HOME",
            "out_slot": out_slot,
            "out_player_id": int(np.asarray(base_state.player_id)[out_slot]),
            "bench_index": 0,
        }

    start = {
        "adapter": "SoccerEnv.prepare_observed_restart",
        "counts_as_transition": False,
        "players_teleported": player_staging is not None,
        "player_staging": player_staging,
        "restart_kind": scene.restart_kind,
        "restart_kind_code": restart_kind,
        "restart_team": "HOME",
        "restart_team_code": restart_team,
        "restart_timer_source": f"dynamics_metadata.config.engine.{timer_name}",
        "restart_timer_substeps": int(engine_metadata[timer_name]),
        "observed_ball_pos_m": [float(value) for value in np.asarray(observed)],
        "canonical_ball_pos_m": [float(value) for value in np.asarray(canonical)],
        "initial_pending_taker_slot": initial_taker,
        "initial_pending_taker_player_id": int(
            np.asarray(base_state.player_id)[initial_taker]
        ),
    }
    requested = {
        "actions": {"source": "external rule-policy StepCommand on every transition"},
        "set_piece_taker": {
            "team": "HOME",
            "restart_kind": scene.restart_kind,
            "player_slot": requested_taker,
            "player_id": int(np.asarray(base_state.player_id)[requested_taker]),
        },
        **management,
    }
    action_override_mask = jnp.zeros((env.N,), dtype=jnp.bool_)
    action_override_vector = jnp.zeros((env.N, 2), dtype=jnp.float32)
    if scene.kick_vector_override is not None:
        action_override_mask = action_override_mask.at[requested_taker].set(True)
        action_override_vector = action_override_vector.at[requested_taker].set(
            jnp.asarray(scene.kick_vector_override, dtype=jnp.float32)
        )
        requested["player_action_override"] = {
            "scope": "selected taker while this restart remains active",
            "player_slot": requested_taker,
            "action_dimensions": "ACTION_KICK_VECTOR",
            "ego_frame_vector": list(scene.kick_vector_override),
            "reason": "retain at least one second of visible post-restart ball trajectory",
        }
    return (
        setup_state,
        opening,
        start,
        requested,
        action_override_mask,
        action_override_vector,
        jnp.int32(restart_kind),
    )


def _completion_receipt(
    trajectory,
    scene: SceneSpec,
    *,
    np,
    restart_none: int,
    ball_alive: int,
    minimum_live_tail_transitions: int,
    control_fps: float,
) -> dict[str, object]:
    restart_kind = np.asarray(trajectory.restart_kind)
    ball_state = np.asarray(trajectory.ball_state)
    completed = (restart_kind == restart_none) & (ball_state == ball_alive)
    if not np.any(completed):
        raise RuntimeError(
            f"{scene.slug}: restart did not clear to a live ball within the scene"
        )
    first = int(np.flatnonzero(completed)[0]) + 1
    live_run = 0
    for is_live in completed[first - 1 :]:
        if not bool(is_live):
            break
        live_run += 1
    post_restart_live_transitions = live_run - 1
    if post_restart_live_transitions < minimum_live_tail_transitions:
        raise RuntimeError(
            f"{scene.slug}: restart cleared without the required "
            f"{minimum_live_tail_transitions}-transition live tail "
            f"(got {post_restart_live_transitions})"
        )
    return {
        "restart_cleared": True,
        "ball_alive": True,
        "first_completed_transition": first,
        "minimum_post_restart_live_transitions": minimum_live_tail_transitions,
        "post_restart_live_transitions": post_restart_live_transitions,
        "post_restart_live_seconds": post_restart_live_transitions / control_fps,
        "final_restart_kind": int(restart_kind[-1]),
        "final_ball_state": int(ball_state[-1]),
    }


def _command_receipt(
    command_results,
    scene: SceneSpec,
    *,
    np,
    CommandReason,
    DecisionSource,
    RestartKind,
    Team,
) -> dict[str, object]:
    """Convert first-frame adjudication and all-frame action use to JSON."""

    team = int(Team.HOME)
    kind = int(RestartKind[scene.restart_kind])

    def reason_name(value) -> str:
        return CommandReason(int(value)).name

    action_source = np.asarray(command_results.actions.source)
    action_consumed = np.asarray(command_results.actions.consumed, dtype=bool)
    action_reasons = np.asarray(command_results.actions.reason)
    unique, counts = np.unique(action_reasons, return_counts=True)
    if not np.all(action_source == int(DecisionSource.EXTERNAL)):
        raise RuntimeError(f"{scene.slug}: an action row lost external provenance")
    if not np.any(action_consumed):
        raise RuntimeError(f"{scene.slug}: no externally submitted action parameter was consumed")
    receipt: dict[str, object] = {
        "transition_api": "SoccerEnv.transition",
        "transition_count": int(action_source.shape[0]),
        "actions": {
            "source": "EXTERNAL",
            "submitted_parameter_count": int(np.asarray(command_results.actions.submitted).size),
            "consumed_parameter_count": int(action_consumed.sum()),
            "reason_counts": {
                reason_name(code): int(count) for code, count in zip(unique, counts)
            },
        },
    }

    taker = command_results.set_piece_takers
    taker_index = (0, team, kind)
    taker_receipt = {
        "requested": bool(np.asarray(taker.requested)[taker_index]),
        "accepted": bool(np.asarray(taker.accepted)[taker_index]),
        "applied": bool(np.asarray(taker.applied)[taker_index]),
        "player_slot": int(np.asarray(taker.player_slot)[taker_index]),
        "reason": reason_name(np.asarray(taker.reason)[taker_index]),
    }
    if not taker_receipt["applied"] or taker_receipt["reason"] != "APPLIED":
        raise RuntimeError(f"{scene.slug}: external taker was not applied: {taker_receipt}")
    receipt["set_piece_taker"] = taker_receipt

    if scene.formation is not None:
        formation = command_results.formations
        index = (0, team)
        formation_receipt = {
            "requested": bool(np.asarray(formation.requested)[index]),
            "accepted": bool(np.asarray(formation.accepted)[index]),
            "applied": bool(np.asarray(formation.applied)[index]),
            "layout_index": int(np.asarray(formation.layout_index)[index]),
            "reason": reason_name(np.asarray(formation.reason)[index]),
        }
        if not formation_receipt["applied"] or formation_receipt["reason"] != "APPLIED":
            raise RuntimeError(
                f"{scene.slug}: external formation was not applied: {formation_receipt}"
            )
        receipt["formation"] = formation_receipt

    if scene.substitution:
        substitution = command_results.substitutions
        index = (0, team, 0)
        substitution_receipt = {
            "requested": bool(np.asarray(substitution.requested)[index]),
            "accepted": bool(np.asarray(substitution.accepted)[index]),
            "applied": bool(np.asarray(substitution.applied)[index]),
            "out_slot": int(np.asarray(substitution.out_slot)[index]),
            "bench_index": int(np.asarray(substitution.bench_index)[index]),
            "reason": reason_name(np.asarray(substitution.reason)[index]),
        }
        if (
            not substitution_receipt["applied"]
            or substitution_receipt["reason"] != "APPLIED"
        ):
            raise RuntimeError(
                f"{scene.slug}: external substitution was not applied: "
                f"{substitution_receipt}"
            )
        receipt["substitution"] = substitution_receipt
    return receipt


def _run_showcase(args: argparse.Namespace, scenes: Sequence[SceneSpec]) -> Path:
    """Load the numerical stack, execute each scene, render, and write one manifest."""

    repository = Path(__file__).resolve().parents[3]
    source_receipt = git_source_receipt(repository)

    from soccerworld.runtime import enable_compilation_cache

    enable_compilation_cache()

    import jax
    import jax.numpy as jnp
    import numpy as np

    from soccerworld import (
        DEFAULT_TIMEBASE,
        FORMATION_LAYOUT_NAMES,
        CommandReason,
        DecisionSource,
        RestartKind,
        Team,
    )
    from soccerworld._engine.constants import ACTION_KICK_VECTOR, BALL_ALIVE
    from soccerworld.demo.roster import build_env
    from soccerworld.policies.rule_based import (
        RULE_POLICY_VERSION,
        make_rule_based_policy,
        policy_config_fingerprint,
        prefix_stable_keys,
    )

    seconds = _effective_seconds(args)
    n_steps = DEFAULT_TIMEBASE.control_steps_for(seconds, minimum=1)
    minimum_live_tail_transitions = DEFAULT_TIMEBASE.control_steps_for(
        MIN_POST_RESTART_LIVE_SECONDS,
        minimum=1,
    )
    env = build_env(
        n_steps + 1,
        control_fps=DEFAULT_TIMEBASE.control_fps,
        bench_size=3,
        manager="idle",
        taker="nearest",
    )
    policy = make_rule_based_policy(
        env,
        match_key=jax.random.PRNGKey(args.seed),
        team_styles=("balanced", "balanced"),
    )
    policy_semantics = rule_policy_receipt(
        policy,
        expected_version=RULE_POLICY_VERSION,
        fingerprint=policy_config_fingerprint,
    )
    rollout = _compile_rollout(
        env,
        policy,
        n_steps,
        jax=jax,
        jnp=jnp,
        kick_vector_slice=ACTION_KICK_VECTOR,
    )
    output_root = Path(args.out_dir) if args.out_dir else _default_output_dir()
    output_root.mkdir(parents=True, exist_ok=True)

    dynamics = env.dynamics_metadata()
    dynamics_bytes = json.dumps(dynamics, sort_keys=True, separators=(",", ":")).encode()
    manifest: dict[str, object] = {
        "schema": SHOWCASE_SCHEMA,
        "purpose": "functional render verification; separate from long-form demonstration video",
        "seed": args.seed,
        "mode": args.mode,
        "rendered": not args.no_render,
        "requested_seconds_per_scene": args.seconds,
        "effective_seconds_per_scene": seconds,
        "control_fps": env.control_fps,
        "transition_count_per_scene": n_steps,
        "dynamics_sha256": hashlib.sha256(dynamics_bytes).hexdigest(),
        "source": source_receipt,
        "policy": {
            "kind": "rule_based",
            "team_styles": ["balanced", "balanced"],
            **policy_semantics,
        },
        "boundary": {
            "scene_start_setup": (
                "prepare_observed_restart is a pure boundary adapter and is not a transition"
            ),
            "state_sequence": (
                "every later frame is produced by SoccerEnv.transition with a StepCommand"
            ),
        },
        "scenes": [],
    }

    for scene_index, scene in enumerate(scenes):
        print(f"[{scene_index + 1}/{len(scenes)}] {scene.slug}: preparing", flush=True)
        scene_key = jax.random.fold_in(jax.random.PRNGKey(args.seed), scene_index)
        reset_key, rollout_key = jax.random.split(scene_key)
        (
            setup,
            opening,
            start_receipt,
            requested,
            action_override_mask,
            action_override_vector,
            action_override_restart_kind,
        ) = _prepare_scene(
            env,
            scene,
            reset_key,
            jnp=jnp,
            np=np,
            formation_layout_names=FORMATION_LAYOUT_NAMES,
            RestartKind=RestartKind,
            Team=Team,
        )
        keys = prefix_stable_keys(rollout_key, n_steps)
        final_state, trajectory, command_results = rollout(
            setup,
            opening,
            keys,
            action_override_mask,
            action_override_vector,
            action_override_restart_kind,
        )
        setup, final_state, trajectory, command_results = jax.device_get(
            (setup, final_state, trajectory, command_results)
        )
        states = jax.tree_util.tree_map(
            lambda initial, sequence: np.concatenate((initial[None], sequence), axis=0),
            setup,
            trajectory,
        )
        state_sequence_sha256 = state_tree_sha256(states, jax=jax, np=np)
        applied = _command_receipt(
            command_results,
            scene,
            np=np,
            CommandReason=CommandReason,
            DecisionSource=DecisionSource,
            RestartKind=RestartKind,
            Team=Team,
        )
        completion = _completion_receipt(
            trajectory,
            scene,
            np=np,
            restart_none=int(RestartKind.NONE),
            ball_alive=int(BALL_ALIVE),
            minimum_live_tail_transitions=minimum_live_tail_transitions,
            control_fps=env.control_fps,
        )
        scene_record: dict[str, object] = {
            "slug": scene.slug,
            "title": scene.title,
            "scene_start_setup": start_receipt,
            "external_commands_requested": requested,
            "transition_receipt": applied,
            "completion_receipt": completion,
            "state_sequence": {
                "sha256": state_sequence_sha256,
                "frame_count": n_steps + 1,
                "scope": "scene setup state plus every public transition result",
            },
            "rendered_frame_count": n_steps + 1,
            "rendered_seconds": (n_steps + 1) / env.control_fps,
            "final_state": {
                "control_step": int(np.asarray(final_state.t)),
                "score": [int(value) for value in np.asarray(final_state.score)],
                "ball_state": int(np.asarray(final_state.ball_state)),
                "restart_kind": int(np.asarray(final_state.restart_kind)),
                "restart_team": int(np.asarray(final_state.restart_team)),
            },
        }
        if args.no_render:
            scene_record["outputs"] = {}
            scene_record["media"] = None
        else:
            scene_dir = output_root / scene.slug
            scene_dir.mkdir(parents=True, exist_ok=True)
            rendered = env.render_mp4(
                states,
                event_states=states,
                out_path=str(scene_dir / f"{scene.slug}.mp4"),
                fps=env.control_fps,
                mode=args.mode,
                title=f"SoccerWorld showcase · {scene.title}",
                dump_replay=True,
                replay_metadata={
                    "showcase_schema": SHOWCASE_SCHEMA,
                    "scene": scene.slug,
                    "source": source_receipt,
                    "policy": policy_semantics,
                    "state_sequence_sha256": state_sequence_sha256,
                    "scene_start_setup": start_receipt,
                    "external_commands_requested": requested,
                    "transition_receipt": applied,
                },
            )
            outputs = _relative_render_outputs(rendered, output_root)
            scene_record["outputs"] = outputs
            scene_record["media"] = media_receipt(
                _mp4_output(outputs, output_root),
                expected_frames=n_steps + 1,
                expected_seconds=(n_steps + 1) / env.control_fps,
                expected_fps=env.control_fps,
            )
        manifest["scenes"].append(scene_record)
        print(f"[{scene_index + 1}/{len(scenes)}] {scene.slug}: verified", flush=True)

    assert_git_source_unchanged(repository, source_receipt)
    manifest_path = output_root / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0.0:
        parser.error("--seconds must be a finite positive number")
    scenes = _selected_scenes(args.scene)
    if args.dry_run:
        print(json.dumps(_plan(args, scenes), indent=2, sort_keys=True))
        return 0
    manifest_path = _run_showcase(args, scenes)
    print(f"showcase manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
