"""Resolve network passes and shots to the player active at each action."""

from __future__ import annotations

from messi.analysis.causal import is_shot_attempt


def _active_player_lookup(report: dict):
    by_slot: dict[int, list[dict]] = {}
    for player in report.get("player_generations", ()):
        by_slot.setdefault(int(player["slot"]), []).append(player)
    for spans in by_slot.values():
        spans.sort(key=lambda player: int(player["start_frame"]))

    def active_player(slot: int, frame: int) -> dict | None:
        return next(
            (
                player
                for player in by_slot.get(slot, ())
                if int(player["start_frame"]) <= frame < int(player["end_frame"])
            ),
            None,
        )

    return active_player


def network_passes_with_generations(report: dict) -> list[dict]:
    """Add display-only endpoint generations without changing the saved report."""
    active_player = _active_player_lookup(report)
    rows = []
    for original in report["pass_map"]:
        row = dict(original)
        if row.get("completed") and row.get("terminal_actor", -1) >= 0:
            source = active_player(int(row["actor"]), int(row["source_frame"]))
            target = active_player(
                int(row["terminal_actor"]), int(row["terminal_frame"])
            )
            if (
                source is not None
                and target is not None
                and int(source["team"]) == int(target["team"]) == int(row["team"])
            ):
                row["_network_source_generation"] = int(source["generation"])
                row["_network_target_generation"] = int(target["generation"])
        rows.append(row)
    return rows


def network_shots_with_generations(report: dict) -> list[dict]:
    """Mark actual shot attempts with a known trajectory and shooter identity."""
    active_player = _active_player_lookup(report)
    goal_line_x = float(report["spatial"]["x_edges_m"][-1])
    rows = []
    for original in report["shot_map"]:
        row = dict(original)
        if (
            row.get("attack_normalized_source_position") is not None
            and row.get("attack_normalized_terminal_position") is not None
            and is_shot_attempt(row, goal_line_x=goal_line_x)
        ):
            shooter = active_player(int(row["actor"]), int(row["source_frame"]))
            if shooter is not None and int(shooter["team"]) == int(row["team"]):
                row["_network_shooter_generation"] = int(shooter["generation"])
        rows.append(row)
    return rows
