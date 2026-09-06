#!/usr/bin/env python3
"""Render one reproducible exact-event managed FootballWorld match.

This is host-only orchestration.  It deliberately uses the public opening
manager, managed runner, and exact-event renderer instead of introducing a
second rollout implementation or adding work to the environment step graph.

The roster means, formation anchors, and equal catalog probabilities below
are transparent design fixtures for long-run inspection.  They are not DFL
measurements or fitted football constants.  A keyed reset still applies
FootballWorld's identity-keyed clipped-Gaussian episode sampling exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

from footballworld import (
    RULE_OPENING_MANAGER_VERSION,
    FootballWorld,
    OpeningManagerDecision,
    OpeningManagerObservation,
    Player,
    PlayerProfile,
    RuleBasedOpeningManagerPolicy,
    build_opening_policy_inputs,
    create_opening_match_from_policy,
    initialize_policy_state,
    make_managed_runner,
)
from footballworld.rendering import (
    DEFAULT_RENDER_FPS,
    ReplayWindow,
    render_managed_event_match,
)
from footballworld.rendering.integrity import publication_authority

DEFAULT_CANDIDATE_COUNT = 20
MIN_CANDIDATE_COUNT = 18
MAX_CANDIDATE_COUNT = 23
ROOT = Path(__file__).resolve().parents[1]

FORMATION_NAMES = ("4-3-3", "4-2-3-1", "3-2-5-possession")
FORMATION_CATALOG = np.asarray(
    (
        (
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
        ),
        (
            (-50.0, 0.0),
            (-35.0, -24.0),
            (-35.0, -8.0),
            (-35.0, 8.0),
            (-35.0, 24.0),
            (-23.0, -10.0),
            (-23.0, 10.0),
            (-12.0, -24.0),
            (-12.0, 0.0),
            (-12.0, 24.0),
            (-5.0, 0.0),
        ),
        (
            (-50.0, 0.0),
            (-37.0, -18.0),
            (-37.0, 0.0),
            (-37.0, 18.0),
            (-25.0, -9.0),
            (-25.0, 9.0),
            (-10.0, -27.0),
            (-10.0, -13.5),
            (-10.0, 0.0),
            (-10.0, 13.5),
            (-10.0, 27.0),
        ),
    ),
    dtype=np.float32,
)

AUDIT_WINDOWS = (
    ReplayWindow("opening", 0.0, 5.0 * 60.0),
    ReplayWindow("middle-first-half", 20.0 * 60.0, 25.0 * 60.0),
    ReplayWindow("halftime-band", 42.0 * 60.0, 52.0 * 60.0),
    ReplayWindow("late", 65.0 * 60.0, 70.0 * 60.0),
    ReplayWindow("finish", 85.0 * 60.0, None),
)

# The first XI is a valid authored 4-3-3 reference.  The remaining entries are
# role-diverse candidates for the rule opening manager, including a second GK.
_CANDIDATE_FIXTURE = (
    # goalkeeper, preferred x/y, speed, height, reach, control, endurance
    (True, -50.0, 0.0, 7.00, 1.91, 2.86, 0.62, 1.02),
    (False, -35.0, -24.0, 8.45, 1.78, 2.67, 0.66, 1.07),
    (False, -35.0, -8.0, 7.78, 1.89, 2.80, 0.62, 1.04),
    (False, -35.0, 8.0, 7.74, 1.87, 2.78, 0.64, 1.05),
    (False, -35.0, 24.0, 8.38, 1.80, 2.69, 0.68, 1.08),
    (False, -20.0, -18.0, 8.09, 1.78, 2.68, 0.76, 1.08),
    (False, -20.0, 0.0, 7.76, 1.82, 2.72, 0.82, 1.10),
    (False, -20.0, 18.0, 8.15, 1.80, 2.70, 0.77, 1.07),
    (False, -8.0, -24.0, 8.76, 1.76, 2.65, 0.79, 1.03),
    (False, -8.0, 0.0, 8.03, 1.86, 2.77, 0.80, 1.01),
    (False, -8.0, 24.0, 8.70, 1.77, 2.66, 0.80, 1.03),
    (True, -49.0, 0.0, 6.92, 1.94, 2.90, 0.60, 1.00),
    (False, -34.0, -15.0, 7.82, 1.91, 2.83, 0.61, 1.03),
    (False, -34.0, 15.0, 8.20, 1.83, 2.73, 0.67, 1.06),
    (False, -22.0, -10.0, 7.73, 1.79, 2.68, 0.81, 1.09),
    (False, -22.0, 10.0, 8.18, 1.81, 2.71, 0.75, 1.08),
    (False, -12.0, -25.0, 8.60, 1.75, 2.64, 0.78, 1.02),
    (False, -10.0, 0.0, 8.04, 1.88, 2.80, 0.78, 1.01),
    (False, -12.0, 25.0, 8.56, 1.76, 2.65, 0.79, 1.04),
    (False, -24.0, 0.0, 7.69, 1.84, 2.75, 0.84, 1.07),
    (False, -36.0, -24.0, 8.31, 1.79, 2.68, 0.69, 1.09),
    (False, -36.0, 24.0, 8.34, 1.80, 2.69, 0.70, 1.08),
    (False, -7.0, 0.0, 8.17, 1.90, 2.82, 0.77, 1.00),
)


def _candidate_count(value: str) -> int:
    count = int(value)
    if not MIN_CANDIDATE_COUNT <= count <= MAX_CANDIDATE_COUNT:
        raise argparse.ArgumentTypeError(
            f"candidate count must be in [{MIN_CANDIDATE_COUNT}, {MAX_CANDIDATE_COUNT}]"
        )
    return count


def _positive_integer(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def _window(value: str) -> ReplayWindow:
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            "window must be NAME:START_SECONDS[:END_SECONDS]"
        )
    name, start_text = parts[:2]
    end_text = parts[2] if len(parts) == 3 else ""
    try:
        start = float(start_text)
        end = None if not end_text else float(end_text)
        return ReplayWindow(name, start, end)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _team_candidates(team: int, count: int) -> tuple[Player, ...]:
    identity_base = 1_000 + 1_000 * team
    players = []
    for slot, fixture in enumerate(_CANDIDATE_FIXTURE[:count]):
        goalkeeper, x, y, speed, height, reach, control, endurance = fixture
        players.append(
            Player(
                profile=PlayerProfile(
                    player_id=identity_base + slot,
                    max_speed_mps=speed,
                    height_m=height,
                    max_reach_height_m=reach,
                    ball_control=control,
                    endurance_factor=endurance,
                    is_goalkeeper=goalkeeper,
                ),
                initial_position=(x, y),
            )
        )
    return tuple(players)


def _selected_windows(args: argparse.Namespace) -> Sequence[ReplayWindow] | None:
    if args.window:
        return tuple(args.window)
    if args.windows == "audit":
        return AUDIT_WINDOWS
    return None


def _canonical_json_sha256(value: object) -> str:
    packed = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def _file_receipt(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _run_text(command: list[str]) -> str:
    process = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        detail = (process.stderr or process.stdout).strip()
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(command)}\n"
            f"{detail[-4000:]}"
        )
    return process.stdout.strip()


def _git_snapshot() -> dict[str, object]:
    revision = _run_text(["git", "rev-parse", "HEAD"])
    status = _run_text(["git", "status", "--porcelain=v1", "--untracked-files=all"])
    entries = status.splitlines() if status else []
    return {
        "revision": revision,
        "dirty": bool(entries),
        "porcelain": entries,
        "porcelain_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _opening_decision_receipt(
    observation: OpeningManagerObservation,
    decision: OpeningManagerDecision,
) -> dict[str, object]:
    """Return an explicit identity receipt for the one-shot opening decision."""

    player_id = np.asarray(observation.players.player_id, dtype=np.int64)
    valid = np.asarray(observation.players.valid, dtype=np.bool_)
    registered = np.asarray(decision.registered, dtype=np.bool_)
    starter = np.asarray(decision.starter, dtype=np.bool_)
    placement = np.asarray(decision.placement_slot, dtype=np.int64)
    formation = np.asarray(decision.formation.layout_index, dtype=np.int64)
    receipt: dict[str, object] = {}
    for team in (0, 1):
        candidate_rows: list[dict[str, object]] = [
            {
                "candidate_index": int(candidate),
                "player_id": int(player_id[team, candidate]),
                "registered": bool(registered[team, candidate]),
                "starter": bool(starter[team, candidate]),
                "placement_slot": int(placement[team, candidate]),
            }
            for candidate in np.flatnonzero(valid[team])
        ]
        starter_rows = sorted(
            (row for row in candidate_rows if row["starter"]),
            key=lambda row: int(row["placement_slot"]),
        )
        receipt[f"team_{team}"] = {
            "selected_formation_candidate_index": int(formation[team]),
            "selected_starting_xi_player_ids_by_slot": [
                row["player_id"] for row in starter_rows
            ],
            "registered_player_ids": [
                row["player_id"] for row in candidate_rows if row["registered"]
            ],
            "starter_player_ids_candidate_order": [
                row["player_id"] for row in candidate_rows if row["starter"]
            ],
            "registered_bench_player_ids": [
                row["player_id"]
                for row in candidate_rows
                if row["registered"] and not row["starter"]
            ],
            "candidate_decisions": candidate_rows,
        }
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render one scalar managed FootballWorld match through the exact "
            "step_with_events path."
        )
    )
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new output directory; existing paths are rejected",
    )
    parser.add_argument(
        "--maximum-steps",
        type=_positive_integer,
        default=None,
        help="optional smoke budget; omit for an authoritative full match",
    )
    parser.add_argument(
        "--windows",
        choices=("full", "audit"),
        default="full",
        help="render the full timeline or five long-run review intervals",
    )
    parser.add_argument(
        "--window",
        action="append",
        type=_window,
        help=(
            "custom NAME:START_SECONDS[:END_SECONDS] window; repeat to select "
            "multiple intervals and override --windows"
        ),
    )
    parser.add_argument(
        "--candidate-count",
        type=_candidate_count,
        default=DEFAULT_CANDIDATE_COUNT,
        help="candidate players per team, including two goalkeepers (18-23)",
    )
    parser.add_argument("--workers", type=_positive_integer, default=4)
    parser.add_argument(
        "--video-fps",
        type=float,
        default=DEFAULT_RENDER_FPS,
        help=f"physics-backed video rate (default {DEFAULT_RENDER_FPS:g} fps)",
    )
    parser.add_argument(
        "--event-chunk",
        type=_positive_integer,
        default=256,
        help="fixed managed exact-event scan length",
    )
    parser.add_argument(
        "--render-chunk",
        type=_positive_integer,
        default=3750,
        help="video segment frame length (measured default for four workers)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit a diagnostic replay from a dirty source tree",
    )
    parser.add_argument(
        "--verify-video",
        action="store_true",
        help=(
            "fully decode and probe every output before atomic publication; "
            "enabled automatically for an authoritative full match"
        ),
    )
    return parser


def _video_verification_enabled(
    *, requested: bool, publication_mode: Mapping[str, object]
) -> bool:
    """Require full decode for authoritative publication, allow opt-in otherwise."""

    return requested or publication_mode.get("authoritative") is True


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    git_start = _git_snapshot()
    if git_start["dirty"] and not args.allow_dirty:
        raise RuntimeError(
            "repository is dirty; commit changes before an authoritative replay "
            "(or use --allow-dirty for diagnostics only)"
        )
    cli_source_start = _file_receipt(Path(__file__).resolve())
    publication_receipt: dict[str, object] = {}
    publication_mode = publication_authority(
        dirty=bool(git_start["dirty"]),
        maximum_steps=args.maximum_steps,
    )

    verify_video = _video_verification_enabled(
        requested=args.verify_video,
        publication_mode=publication_mode,
    )

    def publication_guard() -> dict[str, object]:
        cli_source_end = _file_receipt(Path(__file__).resolve())
        git_end = _git_snapshot()
        if cli_source_end != cli_source_start:
            raise RuntimeError(
                "full-match fixture source changed during capture; staged output "
                "will not be published"
            )
        if git_end != git_start:
            raise RuntimeError(
                "repository revision or dirty state changed during capture; "
                "staged output will not be published"
            )
        receipt: dict[str, object] = {
            **publication_mode,
            "stable_during_capture": True,
            "git_start": git_start,
            "git_end": git_end,
            "git_stable_during_capture": True,
            "fixture_source_start": cli_source_start,
            "fixture_source_end": cli_source_end,
            "fixture_source_stable_during_capture": True,
        }
        publication_receipt.update(receipt)
        return receipt

    match_key = jax.random.key(args.seed)
    env = FootballWorld()
    team_0 = _team_candidates(0, args.candidate_count)
    team_1 = _team_candidates(1, args.candidate_count)
    equal_prior = np.full(
        len(FORMATION_NAMES), 1.0 / len(FORMATION_NAMES), dtype=np.float32
    )

    opening_inputs = build_opening_policy_inputs(
        env,
        team_0,
        team_1,
        FORMATION_CATALOG,
        FORMATION_CATALOG,
        max_registered_players=(args.candidate_count, args.candidate_count),
        formation_probabilities=equal_prior,
        key=match_key,
    )
    opening_policy = RuleBasedOpeningManagerPolicy()
    created = create_opening_match_from_policy(
        env,
        opening_inputs,
        match_key,
        policy=opening_policy,
    )
    runner = make_managed_runner(env, chunk_steps=args.event_chunk)
    match = created.match
    players_per_team = FORMATION_CATALOG.shape[1]
    selected_layout = np.asarray(
        match.inputs.selected_formation_layout, dtype=np.float32
    )
    selected_formation = {
        f"team_{team}": {
            "index": int(match.inputs.selected_formation_index[team]),
            "name": FORMATION_NAMES[match.inputs.selected_formation_index[team]],
            "layout": selected_layout[
                team * players_per_team : (team + 1) * players_per_team
            ].tolist(),
        }
        for team in (0, 1)
    }
    opening_config = asdict(opening_policy.config)
    opening_manager_receipt = {
        "class": (
            f"{type(opening_policy).__module__}.{type(opening_policy).__qualname__}"
        ),
        "version": RULE_OPENING_MANAGER_VERSION,
        "config": opening_config,
        "config_sha256": _canonical_json_sha256(opening_config),
        "config_hash_basis": "SHA-256 of sorted compact canonical JSON",
        "decision": _opening_decision_receipt(
            opening_inputs.observation,
            created.decision,
        ),
    }
    roster = env.roster_metadata_si(match.reset.rollout, match.management.state)
    player_state = initialize_policy_state(
        env,
        runner.player_policy,
        match.reset.rollout,
        roster,
    )
    managed = runner.initialize(
        match.reset.rollout,
        match.reset.setup,
        match.management.squad,
        match.management.state,
        roster,
        player_state,
    )
    result = render_managed_event_match(
        runner,
        managed,
        match.management.squad,
        match_key,
        args.output,
        windows=_selected_windows(args),
        maximum_steps=args.maximum_steps,
        fps=args.video_fps,
        every=1,
        workers=args.workers,
        chunk_frames=args.render_chunk,
        metadata={
            "long_run_fixture": {
                "seed": args.seed,
                "candidate_count_per_team": args.candidate_count,
                "formation_catalog": FORMATION_NAMES,
                "formation_probability_basis": (
                    "equal structural design prior; not fitted to DFL"
                ),
                "selected_formation": selected_formation,
                "opening_manager": opening_manager_receipt,
                "profile_sampling": (
                    "identity-keyed clipped Gaussian once before reset"
                ),
                "source_authority_at_start": {
                    **publication_mode,
                    "git": git_start,
                    "fixture_source": cli_source_start,
                },
            }
        },
        publication_guard=publication_guard,
        verify_video=verify_video,
    )

    full_duration_complete = bool(result.full_duration_complete)
    if full_duration_complete:
        status = "FULL_DURATION_COMPLETE"
    elif result.done:
        status = "TERMINAL_WITHOUT_FULL_DURATION"
    else:
        status = "INCOMPLETE_BUDGET"
    summary = {
        "status": status,
        "done": bool(result.done),
        "terminal_basis": result.terminal_basis,
        "full_duration_complete": full_duration_complete,
        "seed": args.seed,
        "output": str(args.output.resolve()),
        "steps_executed": result.steps_executed,
        "maximum_steps": args.maximum_steps,
        "manager_decisions": result.manager_decisions,
        "event_chunk_steps": result.event_chunk_steps,
        "control_fps": float(env.timebase.control_fps),
        "physics_fps": float(env.timebase.physics_fps),
        "video_fps": float(args.video_fps),
        "render_chunk_frames": args.render_chunk,
        "capture_and_render_seconds": result.capture_and_render_seconds,
        "selected_formation": selected_formation,
        "opening_manager": opening_manager_receipt,
        "publication_guard": publication_receipt,
        "video_decode_verified_before_publication": verify_video,
        "outputs": [str(output.video) for output in result.outputs],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if full_duration_complete or args.maximum_steps is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
