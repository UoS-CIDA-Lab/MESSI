import importlib.util
from pathlib import Path

import jax
import pytest

from footballworld import FootballWorld, build_opening_policy_inputs
from footballworld.policies import TacticalPlan

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render_full_match = _load_module(
    "footballworld_render_full_match_demo",
    ROOT / "examples" / "render_full_match.py",
)


def test_root_demo_match_delegates_to_the_canonical_renderer():
    demo_match = _load_module("footballworld_demo_match", ROOT / "demo_match.py")

    assert Path(demo_match.main.__code__.co_filename).resolve() == (
        ROOT / "examples" / "render_full_match.py"
    )


def test_demo_parser_preserves_the_existing_balanced_plan_defaults(tmp_path):
    args = render_full_match._parser().parse_args(["--output", str(tmp_path / "out")])

    assert args.team_0_plan == TacticalPlan.JUEGO_DE_POSICION.value
    assert args.team_1_plan == TacticalPlan.JUEGO_DE_POSICION.value


def test_demo_parser_accepts_explicit_and_random_team_plans(tmp_path):
    args = render_full_match._parser().parse_args(
        [
            "--output",
            str(tmp_path / "out"),
            "--team-0-plan",
            TacticalPlan.GEGENPRESS.value,
            "--team-1-plan",
            "random",
        ]
    )

    assert args.team_0_plan == TacticalPlan.GEGENPRESS.value
    assert args.team_1_plan == "random"


def test_demo_explicit_team_plans_are_seed_independent():
    requested = (
        TacticalPlan.CATENACCIO.value,
        TacticalPlan.SALIDA_LAVOLPIANA.value,
    )

    assert render_full_match._resolve_team_tactical_plans(
        requested, jax.random.key(1)
    ) == (TacticalPlan.CATENACCIO, TacticalPlan.SALIDA_LAVOLPIANA)
    assert render_full_match._resolve_team_tactical_plans(
        requested, jax.random.key(999)
    ) == (TacticalPlan.CATENACCIO, TacticalPlan.SALIDA_LAVOLPIANA)


def test_demo_random_team_plans_are_seed_reproducible_and_in_catalog():
    requested = ("random", "random")
    first = render_full_match._resolve_team_tactical_plans(
        requested, jax.random.key(2027)
    )
    repeated = render_full_match._resolve_team_tactical_plans(
        requested, jax.random.key(2027)
    )

    assert first == repeated
    assert all(plan in tuple(TacticalPlan) for plan in first)

    observed = {
        render_full_match._resolve_team_tactical_plans(requested, jax.random.key(seed))
        for seed in range(16)
    }
    assert len(observed) > 1


@pytest.mark.parametrize("requested", [("random",), ("random", "random", "random")])
def test_demo_plan_resolution_rejects_the_wrong_team_count(requested):
    with pytest.raises(ValueError, match="exactly two"):
        render_full_match._resolve_team_tactical_plans(requested, jax.random.key(0))


def test_demo_plan_resolution_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown demo tactical plan"):
        render_full_match._resolve_team_tactical_plans(
            ("unknown", "random"), jax.random.key(0)
        )


def test_demo_matrix_defaults_to_cpu_bounded_parallel_two_leg_runs(tmp_path):
    args = render_full_match._parser().parse_args(
        ["--output", str(tmp_path / "matrix"), "--plan-matrix"]
    )

    assert args.plan_matrix
    assert args.matrix_platform == "cpu"
    assert args.matrix_workers == 2
    assert args.matrix_legs == 2
    assert args.matrix_include_self_play


def test_demo_matrix_enumerates_all_unordered_pairs_and_slot_reversals():
    one_leg = render_full_match._matrix_matchups(1, include_self_play=False)
    two_legs = render_full_match._matrix_matchups(2, include_self_play=False)

    assert len(one_leg) == 10
    assert len(two_legs) == 20
    assert len({frozenset((team_0, team_1)) for team_0, team_1, _ in one_leg}) == 10
    assert all(team_0 != team_1 and leg == 1 for team_0, team_1, leg in one_leg)
    for team_0, team_1, _ in one_leg:
        assert (team_0, team_1, 1) in two_legs
        assert (team_1, team_0, 2) in two_legs


def test_demo_matrix_can_enumerate_complete_ordered_plan_space():
    matchups = render_full_match._matrix_matchups(2, include_self_play=True)

    assert len(matchups) == 25
    assert len({(team_0, team_1) for team_0, team_1, _ in matchups}) == 25
    assert sum(team_0 == team_1 for team_0, team_1, _ in matchups) == 5
    assert all(leg == 1 for team_0, team_1, leg in matchups if team_0 == team_1)


def test_demo_matrix_child_arguments_remove_parent_and_pairing_options():
    retained = render_full_match._matrix_child_arguments(
        [
            "--plan-matrix",
            "--matrix-workers=4",
            "--matrix-legs",
            "1",
            "--matrix-platform",
            "cpu",
            "--no-matrix-include-self-play",
            "--team-0-plan",
            "gegenpress",
            "--team-1-plan=juego_de_posicion",
            "--output",
            "matrix",
            "--maximum-steps",
            "12",
            "--report-only",
        ]
    )

    assert retained == ["--maximum-steps", "12", "--report-only"]


def test_demo_matrix_runs_each_pair_in_an_isolated_child_and_writes_summary(
    tmp_path, monkeypatch
):
    output = tmp_path / "matrix"
    argv = [
        "--output",
        str(output),
        "--plan-matrix",
        "--matrix-legs",
        "1",
        "--no-matrix-include-self-play",
        "--maximum-steps",
        "3",
        "--report-only",
    ]
    args = render_full_match._parser().parse_args(argv)
    commands = []
    multi_reports = []

    def write_multi(summary_path):
        multi_reports.append(summary_path)
        target = output / "multi-report"
        return target / "report.html", target / "report.json"

    monkeypatch.setattr(render_full_match, "write_tactical_matrix_report", write_multi)

    def completed(command, **kwargs):
        commands.append((command, kwargs))
        return render_full_match.subprocess.CompletedProcess(command, 0, "{}\n", "")

    monkeypatch.setattr(render_full_match.subprocess, "run", completed)

    assert render_full_match._run_plan_matrix(args, argv) == 0
    summary = render_full_match.json.loads(
        (output / "matrix-summary.json").read_text(encoding="utf-8")
    )
    assert summary["pair_count"] == 10
    assert summary["match_count"] == 10
    assert summary["failure_count"] == 0
    assert summary["multi_report"] == {
        "html": str(output / "multi-report" / "report.html"),
        "json": str(output / "multi-report" / "report.json"),
    }
    assert multi_reports == [output / "matrix-summary.json"]
    assert len(commands) == 10
    assert all(call[1]["env"]["JAX_PLATFORMS"] == "cpu" for call in commands)
    assert all("--plan-matrix" not in call[0] for call in commands)
    assert all("--match-report" in call[0] for call in commands)
    assert all("--allow-diagnostic-report" in call[0] for call in commands)


def test_opening_rejects_registration_maximum_before_int32_narrowing():
    env = FootballWorld()
    team_0 = render_full_match._team_candidates(0, 20)
    team_1 = render_full_match._team_candidates(1, 20)

    with pytest.raises(ValueError, match="non-negative int32 domain"):
        build_opening_policy_inputs(
            env,
            team_0,
            team_1,
            render_full_match.FORMATION_CATALOG,
            render_full_match.FORMATION_CATALOG,
            max_registered_players=(2**32 + 20, 20),
        )
