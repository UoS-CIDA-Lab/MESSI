import json
from pathlib import Path

import pytest

from footballworld.analysis.multi_report import (
    build_tactical_matrix_report,
    render_tactical_matrix_html,
    write_tactical_matrix_report,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _cell(root: Path, team_0: str, team_1: str, score: list[int]) -> dict:
    output = root / f"{team_0}__{team_1}"
    _write_json(
        output / "completion.json",
        {
            "done": True,
            "complete": True,
            "full_duration_complete": True,
            "terminal_basis": "regulation_complete",
            "maximum_steps": None,
            "event_budget_exhausted_count": 0,
            "steps_executed": 55_000,
            "production_source": {"stable_during_capture": True},
        },
    )
    _write_json(output / "match-report-status.json", {"status": "complete"})
    teams = []
    for index in range(2):
        teams.append(
            {
                "realized_shots": 4 + index,
                "shots_on_target": 2 + index,
                "open_play_pass_attempts": 100,
                "open_play_completed_passes": 90,
                "rule_policy_cross_control_signatures": 8 + index,
                "completed_rule_policy_cross_control_signatures": 5 + index,
                "defensive_line_breaking_pass_proxies": 6 + index,
                "completed_defensive_line_breaking_pass_proxies": 4 + index,
                "possession_s": 1_000.0 + index,
                "penalty_area_entries": 3,
                "corners": 2,
                "fouls_committed": 5,
                "offsides": 1,
            }
        )
    _write_json(
        output / "report" / "report.json",
        {
            "metrics_schema": "footballworld.match-metrics/12",
            "quality": {
                "hashes_verified": True,
                "full_duration_complete": True,
                "authoritative": False,
            },
            "summary": {
                "score": score,
                "captured_duration_s": 5_600.0,
                "tracking_frames": 56_000,
            },
            "teams": teams,
        },
    )
    (output / "report" / "report.html").write_text("<html></html>", encoding="utf-8")
    return {
        "team_0_plan": team_0,
        "team_1_plan": team_1,
        "output": str(output),
        "returncode": 0,
    }


def _matrix(tmp_path: Path) -> Path:
    cells = [
        _cell(tmp_path, "alpha", "alpha", [0, 5]),
        _cell(tmp_path, "alpha", "beta", [2, 0]),
        _cell(tmp_path, "beta", "alpha", [1, 1]),
        _cell(tmp_path, "beta", "beta", [7, 0]),
    ]
    summary = tmp_path / "matrix-summary.json"
    _write_json(
        summary,
        {
            "schema": "footballworld.tactical-plan-matrix/1",
            "status": "complete",
            "seed": 29,
            "platform": "cpu",
            "matches": cells,
        },
    )
    return summary


def test_multi_report_builds_complete_matrix_and_league_excludes_self_play(tmp_path):
    report = build_tactical_matrix_report(_matrix(tmp_path))

    assert report["quality"]["complete"]
    assert report["quality"]["complete_ordered_matrix"]
    assert report["summary"]["match_count"] == 4
    assert report["summary"]["self_play_count"] == 2
    assert report["summary"]["total_tracking_frames"] == 224_000
    standings = report["league"]["standings"]
    assert [(row["rank"], row["plan"], row["points"]) for row in standings] == [
        (1, "alpha", 4),
        (2, "beta", 1),
    ]
    assert all(row["played"] == 2 for row in standings)
    assert report["league"]["self_play_excluded"] is True


def test_multi_report_writes_json_and_html_with_individual_links(tmp_path):
    summary = _matrix(tmp_path)
    html_path, json_path = write_tactical_matrix_report(
        summary, tmp_path / "multi-report"
    )

    assert html_path.is_file()
    assert json_path.is_file()
    assert "Policy league standings" in html_path.read_text(encoding="utf-8")
    assert "open match report" in html_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text())["schema"] == (
        "footballworld.tactical-matrix-report/2"
    )


def test_multi_report_aggregates_cross_and_line_break_receipts(tmp_path):
    report = build_tactical_matrix_report(_matrix(tmp_path))

    alpha = next(row for row in report["policy_aggregates"] if row["plan"] == "alpha")
    assert alpha["rule_policy_cross_control_signatures"] == 34
    assert alpha["completed_rule_policy_cross_control_signatures"] == 22
    assert alpha["defensive_line_breaking_pass_proxies"] == 26
    assert alpha["completed_defensive_line_breaking_pass_proxies"] == 18
    rendered = render_tactical_matrix_html(report, output_dir=tmp_path / "report")
    assert "Cross signatures (received)" in rendered
    assert "Line breaks (received)" in rendered


def test_multi_report_surfaces_child_policy_anomalies(tmp_path):
    summary = _matrix(tmp_path)
    matrix = json.loads(summary.read_text())
    child = Path(matrix["matches"][1]["output"]) / "report" / "report.json"
    report = json.loads(child.read_text())
    report["quality"]["policy_audit"] = {
        "anomalies": [
            {
                "code": "stationary_live_loose_ball",
                "severity": "high",
                "message": "A stationary loose ball needs review.",
                "observed": {"duration_s": 42.0},
            }
        ]
    }
    _write_json(child, report)

    aggregate = build_tactical_matrix_report(summary)
    rendered = render_tactical_matrix_html(
        aggregate, output_dir=tmp_path / "multi-report"
    )

    assert aggregate["quality"]["policy_anomaly_count"] == 1
    assert aggregate["policy_anomalies"][0]["team_0_plan"] == "alpha"
    assert aggregate["policy_anomalies"][0]["team_1_plan"] == "beta"
    assert "Policy anomaly audit" in rendered
    assert "stationary_live_loose_ball" in rendered
    assert "inspect match" in rendered


def test_multi_report_rejects_duplicate_ordered_cells(tmp_path):
    summary = _matrix(tmp_path)
    value = json.loads(summary.read_text())
    value["matches"].append(value["matches"][0])
    _write_json(summary, value)

    with pytest.raises(ValueError, match="duplicate ordered cell"):
        build_tactical_matrix_report(summary)


def test_multi_report_rejects_old_child_metrics_instead_of_defaulting_to_zero(
    tmp_path,
):
    summary = _matrix(tmp_path)
    value = json.loads(summary.read_text())
    child = Path(value["matches"][0]["output"]) / "report" / "report.json"
    report = json.loads(child.read_text())
    report["metrics_schema"] = "footballworld.match-metrics/11"
    _write_json(child, report)

    with pytest.raises(ValueError, match="unsupported child metrics schema"):
        build_tactical_matrix_report(summary)


def test_multi_report_marks_and_labels_incomplete_ordered_matrix(tmp_path):
    summary = _matrix(tmp_path)
    value = json.loads(summary.read_text())
    value["matches"].pop()
    _write_json(summary, value)

    report = build_tactical_matrix_report(summary)
    rendered = render_tactical_matrix_html(report, output_dir=tmp_path / "report")

    assert report["quality"]["complete"] is False
    assert report["quality"]["complete_ordered_matrix"] is False
    assert (
        "Matrix does not contain every ordered policy cell."
        in report["quality"]["warnings"]
    )
    assert "incomplete ordered policy cells" in rendered
    assert "3 / 2×2" in rendered


def test_multi_report_does_not_replace_an_existing_report_generation(tmp_path):
    summary = _matrix(tmp_path)
    target = tmp_path / "multi-report"
    target.mkdir()
    sentinel = target / "existing.txt"
    sentinel.write_text("prior generation", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output already exists"):
        write_tactical_matrix_report(summary, target)

    assert sentinel.read_text(encoding="utf-8") == "prior generation"
    assert not (target / "report.json").exists()
    assert not (target / "report.html").exists()
