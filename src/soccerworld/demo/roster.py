"""Roster and environment construction for the shipped single-match demo."""

from __future__ import annotations

import numpy as np

from soccerworld import DEFAULT_TIMEBASE, BenchPlayer, SoccerEnv

__all__ = ["build_env", "make_bench"]


def make_bench(env: SoccerEnv, size: int) -> dict[int, list[BenchPlayer]]:
    """Mirror starter roles into a deterministic, globally identified bench."""

    if size <= 0:
        return {}
    anchors = np.asarray(env.base_formation, np.float32)
    teams = np.concatenate(
        (
            np.zeros(env.n_agents, dtype=np.int32),
            np.ones(env.n_opponents, dtype=np.int32),
        )
    )
    keepers = np.asarray([player.is_gk for player in env.players], dtype=np.bool_)
    next_player_id = max(int(player.id) for player in env.players) + 1
    bench: dict[int, list[BenchPlayer]] = {}
    for team in (0, 1):
        outfield = np.flatnonzero((teams == team) & (~keepers))
        goalkeeper = np.flatnonzero((teams == team) & keepers)
        rows: list[BenchPlayer] = []
        for seat in range(size):
            is_gk = seat == size - 1 and len(goalkeeper) > 0
            source = goalkeeper[0] if is_gk else outfield[seat % len(outfield)]
            rows.append(
                BenchPlayer(
                    player_id=next_player_id + team * size + seat,
                    role_pos=(
                        float(anchors[source, 0]),
                        float(anchors[source, 1]),
                    ),
                    speed=7.7 + 0.05 * (seat % 5),
                    ball_control=0.44 + 0.02 * (seat % 4),
                    is_gk=bool(is_gk),
                )
            )
        bench[team] = rows
    return bench


def build_env(
    n_steps: int,
    *,
    control_fps: float = DEFAULT_TIMEBASE.control_fps,
    compressed_match: bool = False,
    bench_size: int = 9,
    manager: str = "auto",
    taker: str = "auto",
) -> SoccerEnv:
    """Build the demo configuration while keeping the 90-minute stamina reference."""

    env = SoccerEnv(
        game_duration=n_steps,
        control_fps=control_fps,
        halftime=compressed_match,
        restart_taker_mode=taker,
    )
    if manager == "off" or bench_size <= 0:
        return env
    return SoccerEnv(
        game_duration=n_steps,
        control_fps=control_fps,
        halftime=compressed_match,
        restart_taker_mode=taker,
        bench=make_bench(env, bench_size),
        bench_size=bench_size,
        manager=manager,
    )
