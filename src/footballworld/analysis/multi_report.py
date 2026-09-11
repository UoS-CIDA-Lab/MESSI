"""Host-only aggregation for a tactical matrix of verified match reports."""

from __future__ import annotations

import html
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from filelock import FileLock

MATRIX_SCHEMA = "footballworld.tactical-plan-matrix/1"
MULTI_REPORT_SCHEMA = "footballworld.tactical-matrix-report/2"
MATCH_METRICS_SCHEMA = "footballworld.match-metrics/14"


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_tactical_matrix_report(
    matrix_summary: str | Path,
) -> dict[str, object]:
    """Validate every child report and aggregate the complete matchup set."""

    summary_path = Path(matrix_summary).resolve()
    summary = _read_json(summary_path)
    _require(
        summary.get("schema") == MATRIX_SCHEMA, "unsupported matrix summary schema"
    )
    _require(summary.get("status") == "complete", "matrix summary is not complete")
    raw_matches = summary.get("matches")
    _require(isinstance(raw_matches, list) and raw_matches, "matrix has no matches")

    failures: list[str] = []
    matches: list[dict[str, object]] = []
    aggregates: defaultdict[str, dict[str, int | float]] = defaultdict(
        lambda: {
            "appearances": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "realized_shots": 0,
            "shots_on_target": 0,
            "shots_inside_penalty_area": 0,
            "shot_distance_sum_m": 0.0,
            "pass_attempts": 0,
            "passes_completed": 0,
            "cross_signatures": 0,
            "completed_cross_signatures": 0,
            "line_break_proxies": 0,
            "completed_line_break_proxies": 0,
            "forward_passes": 0,
            "completed_forward_passes": 0,
            "attacking_third_passes": 0,
            "attacking_third_direction_known_passes": 0,
            "completed_attacking_third_passes": 0,
            "attacking_third_backward_passes": 0,
            "attacking_third_backward_without_forward_support": 0,
            "completed_attacking_third_backward_without_forward_support": 0,
            "attacking_third_backward_without_forward_support_distance_sum_m": 0.0,
            "possession_s": 0.0,
            "penalty_area_entries": 0,
            "corners": 0,
            "fouls_committed": 0,
            "offsides": 0,
        }
    )
    ordered_cells: set[tuple[str, str]] = set()
    league: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "points": 0,
        }
    )
    authoritative_count = 0
    team_0_goals = 0
    team_1_goals = 0
    team_0_wins = 0
    team_1_wins = 0
    draws = 0
    policy_anomalies: list[dict[str, object]] = []

    for index, raw in enumerate(raw_matches):
        _require(isinstance(raw, dict), f"match {index} is not an object")
        team_0_plan = str(raw.get("team_0_plan", ""))
        team_1_plan = str(raw.get("team_1_plan", ""))
        _require(team_0_plan and team_1_plan, f"match {index} has no plan labels")
        cell = (team_0_plan, team_1_plan)
        _require(cell not in ordered_cells, f"duplicate ordered cell: {cell}")
        ordered_cells.add(cell)
        output = Path(str(raw.get("output", ""))).resolve()
        completion_path = output / "completion.json"
        status_path = output / "match-report-status.json"
        report_path = output / "report" / "report.json"
        html_path = output / "report" / "report.html"
        for required in (completion_path, status_path, report_path, html_path):
            if not required.is_file():
                failures.append(f"missing artifact: {required}")
        if failures and any(str(output) in failure for failure in failures):
            continue

        completion = _read_json(completion_path)
        status = _read_json(status_path)
        report = _read_json(report_path)
        _require(
            report.get("metrics_schema") == MATCH_METRICS_SCHEMA,
            f"unsupported child metrics schema: {report_path}",
        )
        checks = {
            "child return code": raw.get("returncode") == 0,
            "report status": status.get("status") == "complete",
            "environment done": completion.get("done") is True,
            "capture complete": completion.get("complete") is True,
            "full duration": completion.get("full_duration_complete") is True,
            "regulation terminal": completion.get("terminal_basis")
            == "regulation_complete",
            "unbounded rollout": completion.get("maximum_steps") is None,
            "event budget": completion.get("event_budget_exhausted_count") == 0,
            "stable capture source": (
                isinstance(completion.get("production_source"), dict)
                and completion["production_source"].get("stable_during_capture") is True
            ),
            "report hashes": (
                isinstance(report.get("quality"), dict)
                and report["quality"].get("hashes_verified") is True
            ),
            "report full duration": (
                isinstance(report.get("quality"), dict)
                and report["quality"].get("full_duration_complete") is True
            ),
        }
        failures.extend(
            f"{team_0_plan} vs {team_1_plan}: {name}"
            for name, passed in checks.items()
            if not passed
        )
        summary_block = report.get("summary")
        teams = report.get("teams")
        _require(
            isinstance(summary_block, dict), f"missing report summary: {report_path}"
        )
        _require(
            isinstance(teams, list) and len(teams) == 2, f"invalid teams: {report_path}"
        )
        score = summary_block.get("score")
        _require(
            isinstance(score, list)
            and len(score) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in score
            ),
            f"invalid score: {report_path}",
        )
        team_0_goals += score[0]
        team_1_goals += score[1]
        team_0_wins += int(score[0] > score[1])
        team_1_wins += int(score[1] > score[0])
        draws += int(score[0] == score[1])
        quality = report["quality"]
        authoritative_count += int(quality.get("authoritative") is True)
        policy_audit = quality.get("policy_audit", {})
        child_anomalies = (
            policy_audit.get("anomalies", []) if isinstance(policy_audit, dict) else []
        )
        _require(
            isinstance(child_anomalies, list),
            f"invalid policy anomaly list: {report_path}",
        )
        for anomaly in child_anomalies:
            _require(
                isinstance(anomaly, dict),
                f"invalid policy anomaly record: {report_path}",
            )
            policy_anomalies.append(
                {
                    "team_0_plan": team_0_plan,
                    "team_1_plan": team_1_plan,
                    "report_html": str(html_path),
                    **anomaly,
                }
            )
        match_row = {
            "team_0_plan": team_0_plan,
            "team_1_plan": team_1_plan,
            "score": score,
            "winner": (
                None
                if score[0] == score[1]
                else team_0_plan
                if score[0] > score[1]
                else team_1_plan
            ),
            "captured_duration_s": summary_block.get("captured_duration_s"),
            "tracking_frames": summary_block.get("tracking_frames"),
            "steps_executed": completion.get("steps_executed"),
            "authoritative": quality.get("authoritative") is True,
            "report_json": str(report_path),
            "report_html": str(html_path),
            "completion_json": str(completion_path),
        }
        matches.append(match_row)
        if team_0_plan != team_1_plan:
            for team_index, plan in enumerate(cell):
                mine = score[team_index]
                other = score[1 - team_index]
                standing = league[plan]
                standing["played"] += 1
                standing["wins"] += int(mine > other)
                standing["draws"] += int(mine == other)
                standing["losses"] += int(mine < other)
                standing["goals_for"] += mine
                standing["goals_against"] += other
                standing["points"] += 3 * int(mine > other) + int(mine == other)
        for team_index, plan in enumerate(cell):
            mine = score[team_index]
            other = score[1 - team_index]
            team = teams[team_index]
            _require(isinstance(team, dict), f"invalid team row: {report_path}")
            for metric_name in (
                "rule_policy_cross_control_signatures",
                "completed_rule_policy_cross_control_signatures",
                "defensive_line_breaking_pass_proxies",
                "completed_defensive_line_breaking_pass_proxies",
                "shots_inside_penalty_area",
                "forward_pass_attempts",
                "completed_forward_passes",
                "attacking_third_pass_attempts",
                "attacking_third_direction_known_pass_attempts",
                "completed_attacking_third_passes",
                "attacking_third_backward_pass_attempts",
                "attacking_third_backward_without_forward_support",
                "completed_attacking_third_backward_without_forward_support",
            ):
                metric_value = team.get(metric_name)
                _require(
                    isinstance(metric_value, int)
                    and not isinstance(metric_value, bool),
                    f"unavailable {metric_name}: {report_path}",
                )
            aggregate = aggregates[plan]
            aggregate["appearances"] += 1
            aggregate["wins"] += int(mine > other)
            aggregate["draws"] += int(mine == other)
            aggregate["losses"] += int(mine < other)
            aggregate["goals_for"] += mine
            aggregate["goals_against"] += other
            aggregate["realized_shots"] += int(team["realized_shots"])
            aggregate["shots_on_target"] += int(team["shots_on_target"])
            aggregate["shots_inside_penalty_area"] += int(
                team["shots_inside_penalty_area"]
            )
            if team["mean_shot_distance_m"] is not None:
                aggregate["shot_distance_sum_m"] += float(
                    team["mean_shot_distance_m"]
                ) * int(team["realized_shots"])
            aggregate["pass_attempts"] += int(team["open_play_pass_attempts"])
            aggregate["passes_completed"] += int(team["open_play_completed_passes"])
            aggregate["cross_signatures"] += int(
                team["rule_policy_cross_control_signatures"]
            )
            aggregate["completed_cross_signatures"] += int(
                team["completed_rule_policy_cross_control_signatures"]
            )
            aggregate["line_break_proxies"] += int(
                team["defensive_line_breaking_pass_proxies"]
            )
            aggregate["completed_line_break_proxies"] += int(
                team["completed_defensive_line_breaking_pass_proxies"]
            )
            aggregate["forward_passes"] += int(team["forward_pass_attempts"])
            aggregate["completed_forward_passes"] += int(
                team["completed_forward_passes"]
            )
            aggregate["attacking_third_passes"] += int(
                team["attacking_third_pass_attempts"]
            )
            aggregate["attacking_third_direction_known_passes"] += int(
                team["attacking_third_direction_known_pass_attempts"]
            )
            aggregate["completed_attacking_third_passes"] += int(
                team["completed_attacking_third_passes"]
            )
            aggregate["attacking_third_backward_passes"] += int(
                team["attacking_third_backward_pass_attempts"]
            )
            aggregate["attacking_third_backward_without_forward_support"] += int(
                team["attacking_third_backward_without_forward_support"]
            )
            aggregate[
                "completed_attacking_third_backward_without_forward_support"
            ] += int(
                team["completed_attacking_third_backward_without_forward_support"]
            )
            unsupported_mean_distance = team.get(
                "mean_attacking_third_backward_without_forward_support_distance_m"
            )
            unsupported_count = int(
                team["attacking_third_backward_without_forward_support"]
            )
            _require(
                unsupported_count == 0 or unsupported_mean_distance is not None,
                f"missing unsupported backward-pass mean distance: {report_path}",
            )
            if unsupported_mean_distance is not None:
                _require(
                    isinstance(unsupported_mean_distance, (int, float))
                    and not isinstance(unsupported_mean_distance, bool),
                    f"invalid unsupported backward-pass mean distance: {report_path}",
                )
                aggregate[
                    "attacking_third_backward_without_forward_support_distance_sum_m"
                ] += float(unsupported_mean_distance) * unsupported_count
            aggregate["possession_s"] += float(team["possession_s"])
            for name in (
                "penalty_area_entries",
                "corners",
                "fouls_committed",
                "offsides",
            ):
                aggregate[name] += int(team[name])

    _require(not failures, "; ".join(failures))
    plans = sorted(aggregates)
    expected_cells = {(first, second) for first in plans for second in plans}
    complete_ordered_matrix = ordered_cells == expected_cells
    policy_rows: list[dict[str, object]] = []
    for plan in plans:
        aggregate = dict(aggregates[plan])
        appearances = int(aggregate["appearances"])
        attempts = int(aggregate.pop("pass_attempts"))
        possession_s = float(aggregate.pop("possession_s"))
        realized_shots = int(aggregate["realized_shots"])
        shot_distance_sum_m = float(aggregate.pop("shot_distance_sum_m"))
        unsupported_backward_count = int(
            aggregate["attacking_third_backward_without_forward_support"]
        )
        unsupported_backward_distance_sum_m = float(
            aggregate.pop(
                "attacking_third_backward_without_forward_support_distance_sum_m"
            )
        )
        aggregate.update(
            {
                "plan": plan,
                "goal_difference": (
                    int(aggregate["goals_for"]) - int(aggregate["goals_against"])
                ),
                "open_play_pass_attempts": attempts,
                "open_play_pass_completion": (
                    int(aggregate["passes_completed"]) / attempts if attempts else None
                ),
                "mean_shot_distance_m": (
                    shot_distance_sum_m / realized_shots if realized_shots else None
                ),
                "forward_pass_attempts": int(aggregate.pop("forward_passes")),
                "completed_forward_passes": int(
                    aggregate.pop("completed_forward_passes")
                ),
                "attacking_third_pass_attempts": int(
                    aggregate.pop("attacking_third_passes")
                ),
                "attacking_third_direction_known_pass_attempts": int(
                    aggregate.pop("attacking_third_direction_known_passes")
                ),
                "completed_attacking_third_passes": int(
                    aggregate.pop("completed_attacking_third_passes")
                ),
                "attacking_third_backward_pass_attempts": int(
                    aggregate.pop("attacking_third_backward_passes")
                ),
                "attacking_third_backward_without_forward_support": int(
                    aggregate.pop("attacking_third_backward_without_forward_support")
                ),
                "completed_attacking_third_backward_without_forward_support": int(
                    aggregate.pop(
                        "completed_attacking_third_backward_without_forward_support"
                    )
                ),
                "mean_attacking_third_backward_without_forward_support_distance_m": (
                    unsupported_backward_distance_sum_m / unsupported_backward_count
                    if unsupported_backward_count
                    else None
                ),
                "rule_policy_cross_control_signatures": int(
                    aggregate.pop("cross_signatures")
                ),
                "completed_rule_policy_cross_control_signatures": int(
                    aggregate.pop("completed_cross_signatures")
                ),
                "defensive_line_breaking_pass_proxies": int(
                    aggregate.pop("line_break_proxies")
                ),
                "completed_defensive_line_breaking_pass_proxies": int(
                    aggregate.pop("completed_line_break_proxies")
                ),
                "average_controlled_possession_s": possession_s / appearances,
            }
        )
        policy_rows.append(aggregate)

    standings = []
    for plan in plans:
        standing = dict(league[plan])
        standing["plan"] = plan
        standing["goal_difference"] = standing["goals_for"] - standing["goals_against"]
        standings.append(standing)
    standings.sort(
        key=lambda row: (
            -row["points"],
            -row["goal_difference"],
            -row["goals_for"],
            row["plan"],
        )
    )
    for rank, standing in enumerate(standings, start=1):
        standing["rank"] = rank

    total_frames = sum(int(row["tracking_frames"]) for row in matches)
    total_duration = sum(float(row["captured_duration_s"]) for row in matches)
    return {
        "schema": MULTI_REPORT_SCHEMA,
        "source_matrix": str(summary_path),
        "seed": summary.get("seed"),
        "platform": summary.get("platform"),
        "quality": {
            "complete": complete_ordered_matrix,
            "complete_ordered_matrix": complete_ordered_matrix,
            "full_duration_match_count": len(matches),
            "authoritative_match_count": authoritative_count,
            "diagnostic_match_count": len(matches) - authoritative_count,
            "failure_count": 0,
            "policy_anomaly_count": len(policy_anomalies),
            "warnings": (
                (
                    []
                    if complete_ordered_matrix
                    else ["Matrix does not contain every ordered policy cell."]
                )
                + (
                    []
                    if authoritative_count == len(matches)
                    else [
                        "One or more full-duration reports are diagnostic because their source worktree was not authoritative."
                    ]
                )
                + (
                    []
                    if not policy_anomalies
                    else [
                        f"Policy audit found {len(policy_anomalies)} suspicious match condition(s); inspect the anomaly table before interpreting policy rankings."
                    ]
                )
            ),
        },
        "summary": {
            "plan_count": len(plans),
            "match_count": len(matches),
            "self_play_count": sum(
                row["team_0_plan"] == row["team_1_plan"] for row in matches
            ),
            "total_tracking_frames": total_frames,
            "total_captured_duration_s": total_duration,
        },
        "team_slot_diagnostic": {
            "team_0_wins": team_0_wins,
            "draws": draws,
            "team_1_wins": team_1_wins,
            "team_0_goals": team_0_goals,
            "team_1_goals": team_1_goals,
        },
        "league": {
            "points_for_win": 3,
            "points_for_draw": 1,
            "self_play_excluded": True,
            "tiebreakers": ["points", "goal_difference", "goals_for", "plan"],
            "standings": standings,
        },
        "policy_aggregates": policy_rows,
        "policy_anomalies": policy_anomalies,
        "matches": matches,
    }


def render_tactical_matrix_html(
    report: Mapping[str, object], *, output_dir: str | Path
) -> str:
    """Render a dependency-free matrix dashboard linked to match reports."""

    target = Path(output_dir).resolve()
    quality = report["quality"]
    summary = report["summary"]
    slot = report["team_slot_diagnostic"]
    warning = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in quality["warnings"]
    )
    warning_band = f"<aside><ul>{warning}</ul></aside>" if warning else ""
    standings = []
    for row in report["league"]["standings"]:
        standings.append(
            "<tr>"
            f"<td>{row['rank']}</td>"
            f"<td>{html.escape(str(row['plan']))}</td>"
            f"<td>{row['played']}</td>"
            f"<td>{row['wins']}-{row['draws']}-{row['losses']}</td>"
            f"<td>{row['goals_for']}-{row['goals_against']}</td>"
            f"<td>{row['goal_difference']:+d}</td>"
            f"<td><strong>{row['points']}</strong></td>"
            "</tr>"
        )
    policies = []
    for row in report["policy_aggregates"]:
        completion = row["open_play_pass_completion"]
        completion_label = "n/a" if completion is None else f"{completion:.1%}"
        shot_distance = row["mean_shot_distance_m"]
        shot_distance_label = (
            "n/a" if shot_distance is None else f"{float(shot_distance):.1f} m"
        )
        attacking_third_attempts = int(
            row["attacking_third_direction_known_pass_attempts"]
        )
        attacking_third_backward_share = (
            None
            if not attacking_third_attempts
            else float(row["attacking_third_backward_pass_attempts"])
            / attacking_third_attempts
        )
        attacking_third_backward_label = (
            "n/a"
            if attacking_third_backward_share is None
            else f"{attacking_third_backward_share:.1%} back"
        )
        backward_attempts = int(row["attacking_third_backward_pass_attempts"])
        no_support_share = (
            None
            if not backward_attempts
            else float(row["attacking_third_backward_without_forward_support"])
            / backward_attempts
        )
        no_support_label = (
            "n/a" if no_support_share is None else f"{no_support_share:.1%} unsupported"
        )
        unsupported_distance = row[
            "mean_attacking_third_backward_without_forward_support_distance_m"
        ]
        unsupported_detail_label = (
            "n/a"
            if unsupported_distance is None
            else f"{row['completed_attacking_third_backward_without_forward_support']} received; {float(unsupported_distance):.1f} m mean"
        )
        policies.append(
            "<tr>"
            f"<td>{html.escape(str(row['plan']))}</td>"
            f"<td>{row['appearances']}</td>"
            f"<td>{row['wins']}-{row['draws']}-{row['losses']}</td>"
            f"<td>{row['goals_for']}-{row['goals_against']}</td>"
            f"<td>{row['realized_shots']} ({row['shots_on_target']} OT; {row['shots_inside_penalty_area']} box; {shot_distance_label})</td>"
            f"<td>{completion_label}</td>"
            f"<td>{row['forward_pass_attempts']} ({row['completed_forward_passes']})</td>"
            f"<td>{row['attacking_third_pass_attempts']} ({row['completed_attacking_third_passes']}; {attacking_third_backward_label}; {no_support_label}; {unsupported_detail_label})</td>"
            f"<td>{row['rule_policy_cross_control_signatures']} ({row['completed_rule_policy_cross_control_signatures']})</td>"
            f"<td>{row['defensive_line_breaking_pass_proxies']} ({row['completed_defensive_line_breaking_pass_proxies']})</td>"
            f"<td>{row['average_controlled_possession_s']:.1f}</td>"
            "</tr>"
        )
    matches = []
    for row in report["matches"]:
        href = os.path.relpath(str(row["report_html"]), target)
        matches.append(
            "<tr>"
            f"<td>{html.escape(str(row['team_0_plan']))}</td>"
            f"<td>{html.escape(str(row['team_1_plan']))}</td>"
            f"<td>{row['score'][0]}-{row['score'][1]}</td>"
            f"<td>{float(row['captured_duration_s']):.1f}</td>"
            f"<td>{int(row['tracking_frames']):,}</td>"
            f"<td><a href='{html.escape(href)}'>open match report</a></td>"
            "</tr>"
        )
    anomalies = []
    for row in report.get("policy_anomalies", []):
        href = os.path.relpath(str(row["report_html"]), target)
        observed = row.get("observed", {})
        anomalies.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('severity', 'unknown')))}</td>"
            f"<td>{html.escape(str(row.get('code', 'unknown')))}</td>"
            f"<td>{html.escape(str(row['team_0_plan']))} vs {html.escape(str(row['team_1_plan']))}</td>"
            f"<td>{html.escape(str(row.get('message', '')))}</td>"
            f"<td><code>{html.escape(json.dumps(observed, ensure_ascii=False, sort_keys=True))}</code></td>"
            f"<td><a href='{html.escape(href)}'>inspect match</a></td>"
            "</tr>"
        )
    anomaly_section = (
        "<h2>Policy anomaly audit</h2>"
        "<p>No configured liveness or spatial-concentration anomaly was detected.</p>"
        if not anomalies
        else (
            "<h2>Policy anomaly audit</h2>"
            "<p>Host-only diagnostic priors; each item is a review lead, not proof of a policy defect.</p>"
            "<table><thead><tr><th>Severity</th><th>Code</th><th>Match</th><th>Reason</th><th>Observed</th><th>Report</th></tr></thead>"
            f"<tbody>{''.join(anomalies)}</tbody></table>"
        )
    )
    json_href = "report.json"
    matrix_label = (
        "complete ordered policy cells"
        if quality["complete_ordered_matrix"]
        else "incomplete ordered policy cells"
    )
    matrix_shape = (
        f"{summary['plan_count']}×{summary['plan_count']}"
        if quality["complete_ordered_matrix"]
        else f"{summary['match_count']} / {summary['plan_count']}×{summary['plan_count']}"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FootballWorld tactical matrix report</title>
<style>
:root{{--ink:#18201c;--muted:#66716a;--paper:#f3f5f1;--white:#fff;--line:#d7ddd8;--green:#236746;--gold:#b57d13}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
.wrap{{width:min(1320px,calc(100% - 28px));margin:auto}}header{{background:var(--white);border-bottom:1px solid var(--line);padding:20px 0}}
h1{{margin:0;font-size:27px}}h2{{font-size:18px;margin:24px 0 8px}}p{{margin:4px 0;color:var(--muted)}}
.facts{{display:grid;grid-template-columns:repeat(4,1fr);background:var(--white);border:1px solid var(--line);margin-top:16px}}
.fact{{padding:12px;border-right:1px solid var(--line)}}.fact:last-child{{border:0}}.fact strong{{display:block;font-size:20px}}.fact span{{color:var(--muted);font-size:11px}}
aside{{border-left:4px solid var(--gold);background:#fff8e8;padding:7px 12px;margin-top:14px}}main{{padding-bottom:36px}}
table{{width:100%;border-collapse:collapse;background:var(--white);border:1px solid var(--line)}}th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}}th:first-child,td:first-child,td:nth-child(2){{text-align:left}}th{{background:#e9eeea;color:#415047;font-size:11px;text-transform:uppercase}}a{{color:var(--green)}}footer{{padding:24px 0;color:var(--muted)}}
@media(max-width:700px){{.facts{{grid-template-columns:repeat(2,1fr)}}th,td{{padding:7px 5px;font-size:11px}}}}
</style></head><body>
<header><div class="wrap"><h1>FootballWorld tactical matrix report</h1><p>Seed {report["seed"]} · {html.escape(str(report["platform"]))} · {matrix_label}</p></div></header>
<main class="wrap">{warning_band}
<div class="facts">
<div class="fact"><strong>{summary["match_count"]}</strong><span>full-duration matches</span></div>
<div class="fact"><strong>{matrix_shape}</strong><span>ordered tactical cells</span></div>
<div class="fact"><strong>{float(summary["total_captured_duration_s"]) / 3600:.2f} h</strong><span>captured match time</span></div>
<div class="fact"><strong>{slot["team_0_wins"]}-{slot["draws"]}-{slot["team_1_wins"]}</strong><span>Team 0 wins-draws-Team 1 wins</span></div>
</div>
<h2>Policy league standings</h2>
<p>Distinct-plan fixtures only; 3 points for a win, 1 for a draw. Ties: goal difference, goals for, plan name.</p>
<table><thead><tr><th>Rank</th><th>Plan</th><th>P</th><th>W-D-L</th><th>GF-GA</th><th>GD</th><th>Pts</th></tr></thead><tbody>{"".join(standings)}</tbody></table>
<h2>Policy aggregates</h2>
<table><thead><tr><th>Plan</th><th>Apps</th><th>W-D-L</th><th>GF-GA</th><th>Shots (OT; box; mean distance)</th><th>Pass completion</th><th>Forward passes (received)</th><th>Att. third passes (received; back/unsupported)</th><th>Cross signatures (received)</th><th>Line breaks (received)</th><th>Avg controlled possession s</th></tr></thead><tbody>{"".join(policies)}</tbody></table>
{anomaly_section}
<h2>Every match</h2>
<table><thead><tr><th>Team 0</th><th>Team 1</th><th>Score</th><th>Duration s</th><th>Frames</th><th>Individual report</th></tr></thead><tbody>{"".join(matches)}</tbody></table>
<footer>Host-only aggregation from verified single-match reports. <a href="{json_href}">Machine-readable JSON</a>.</footer>
</main></body></html>"""


def write_tactical_matrix_report(
    matrix_summary: str | Path, output_dir: str | Path | None = None
) -> tuple[Path, Path]:
    """Build and atomically publish JSON and HTML multi-match reports."""

    summary_path = Path(matrix_summary).resolve()
    target = (
        Path(output_dir).resolve()
        if output_dir is not None
        else summary_path.parent / "multi-report"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    report = build_tactical_matrix_report(summary_path)
    json_path = target / "report.json"
    html_path = target / "report.html"
    lock = FileLock(str(target.parent / f".{target.name}.report.lock"))
    with lock:
        if target.exists():
            raise FileExistsError(f"multi-report output already exists: {target}")
        staging = Path(
            tempfile.mkdtemp(prefix=f".{target.name}-staging-", dir=target.parent)
        )
        try:
            (staging / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            (staging / "report.html").write_text(
                render_tactical_matrix_html(report, output_dir=target), encoding="utf-8"
            )
            staging.replace(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return html_path, json_path


__all__ = [
    "MULTI_REPORT_SCHEMA",
    "build_tactical_matrix_report",
    "render_tactical_matrix_html",
    "write_tactical_matrix_report",
]
