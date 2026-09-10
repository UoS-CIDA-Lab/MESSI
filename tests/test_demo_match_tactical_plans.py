import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest

from footballworld import FootballWorld, build_opening_policy_inputs
from footballworld.config.match_fixture import load_match_fixture
from footballworld.policies import (
    RuleBasedOpeningManagerPolicy,
    RuleOpeningManagerConfig,
    TacticalPlan,
)
from footballworld.policies.rule_based.opening_manager import _formation_tactical_fit

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


def _opening_players(key):
    inputs = build_opening_policy_inputs(
        FootballWorld(),
        render_full_match._team_candidates(0, 20),
        render_full_match._team_candidates(1, 20),
        render_full_match.FORMATION_CATALOG,
        render_full_match.FORMATION_CATALOG,
        max_registered_players=(20, 20),
        key=key,
    )
    return inputs.observation.players


def test_demo_parser_defaults_to_roster_conditioned_auto_plans(tmp_path):
    args = render_full_match._parser().parse_args(["--output", str(tmp_path / "out")])

    assert args.team_0_plan == "auto"
    assert args.team_1_plan == "auto"


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


def test_demo_detects_both_cli_assignment_forms():
    assert render_full_match._option_present(["--seed", "3"], "--seed")
    assert render_full_match._option_present(["--seed=3"], "--seed")
    assert not render_full_match._option_present(["--other=3"], "--seed")


def test_fixture_exact_fields_bypass_sampling_and_selection():
    loaded = load_match_fixture(ROOT / "examples/fixtures/recorded_match.json")
    arguments, names = render_full_match._fixture_opening_arguments(loaded)
    inputs = build_opening_policy_inputs(
        FootballWorld(), key=jax.random.key(29), **arguments
    )

    assert names[0][0] == "4-3-3"
    np.testing.assert_array_equal(
        np.asarray(inputs.observation.formation.candidate_probability[0]),
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    # The authored home goalkeeper is exact despite a supplied sampling key.
    assert float(inputs.observation.players.max_speed[0, 0]) == pytest.approx(
        (7.2 - 3.0) / (11.0 - 3.0)
    )
    # An omitted ability bundle is sampled rather than silently treated exact.
    assert not np.isclose(
        float(inputs.observation.players.max_speed[0, 1]),
        (7.96 - 3.0) / (11.0 - 3.0),
    )
    policy = RuleBasedOpeningManagerPolicy(
        RuleOpeningManagerConfig(
            team_tactical_plans=(
                TacticalPlan.JUEGO_DE_POSICION,
                TacticalPlan.GEGENPRESS,
            )
        )
    )
    decision = policy.step(
        inputs.observation,
        jax.random.key(29),
        policy.initialize(inputs.observation),
    ).decision
    decision = render_full_match._preserve_exact_lineups(
        decision, inputs.authored_selection, (True, False)
    )
    assert int(decision.formation.layout_index[0]) == 0
    np.testing.assert_array_equal(
        decision.starter[0], inputs.authored_selection.starter[0]
    )


def test_demo_explicit_team_plans_are_seed_independent():
    requested = (
        TacticalPlan.CATENACCIO.value,
        TacticalPlan.SALIDA_LAVOLPIANA.value,
    )
    players = _opening_players(jax.random.key(1))

    assert render_full_match._resolve_team_tactical_plans(
        requested, jax.random.key(1), players
    )[0] == (TacticalPlan.CATENACCIO, TacticalPlan.SALIDA_LAVOLPIANA)
    assert render_full_match._resolve_team_tactical_plans(
        requested, jax.random.key(999), players
    )[0] == (TacticalPlan.CATENACCIO, TacticalPlan.SALIDA_LAVOLPIANA)


def test_demo_random_team_plans_are_seed_reproducible_and_in_catalog():
    requested = ("random", "random")
    players = _opening_players(jax.random.key(2027))
    first = render_full_match._resolve_team_tactical_plans(
        requested, jax.random.key(2027), players
    )[0]
    repeated = render_full_match._resolve_team_tactical_plans(
        requested, jax.random.key(2027), players
    )[0]

    assert first == repeated
    assert all(plan in tuple(TacticalPlan) for plan in first)

    observed = {
        render_full_match._resolve_team_tactical_plans(
            requested, jax.random.key(seed), players
        )[0]
        for seed in range(16)
    }
    assert len(observed) > 1


def test_opening_formation_softmax_runs_after_tactical_plan_is_fixed():
    key = jax.random.key(29)
    inputs = build_opening_policy_inputs(
        FootballWorld(),
        render_full_match._team_candidates(0, 20),
        render_full_match._team_candidates(1, 20),
        render_full_match.FORMATION_CATALOG,
        render_full_match.FORMATION_CATALOG,
        max_registered_players=(20, 20),
        formation_probabilities=render_full_match.STARTING_FORMATION_PRIOR,
        key=key,
    )

    def selected(plan):
        policy = RuleBasedOpeningManagerPolicy(
            RuleOpeningManagerConfig(
                team_tactical_plans=(plan, plan),
                formation_fit_weight=100.0,
                lineup_noise_scale=0.0,
            )
        )
        return np.asarray(
            policy.step(
                inputs.observation,
                key,
                policy.initialize(inputs.observation),
            ).decision.formation.layout_index
        )

    np.testing.assert_array_equal(selected(TacticalPlan.JUEGO_DE_POSICION), [0, 0])
    np.testing.assert_array_equal(selected(TacticalPlan.SALIDA_LAVOLPIANA), [1, 1])


def test_opening_formation_mode_is_tactical_and_slot_equivalent_across_seeds():
    candidate_count = 20
    fixed = tuple(False for _ in range(candidate_count))
    inputs = build_opening_policy_inputs(
        FootballWorld(),
        render_full_match._team_candidates(0, candidate_count),
        render_full_match._team_candidates(1, candidate_count),
        render_full_match.FORMATION_CATALOG,
        render_full_match.FORMATION_CATALOG,
        max_registered_players=(candidate_count, candidate_count),
        formation_probabilities=render_full_match.STARTING_FORMATION_PRIOR,
        sample_abilities=(fixed, fixed),
        key=jax.random.key(0),
    )
    keys = jax.vmap(jax.random.key)(np.arange(64, dtype=np.uint32))
    expected_mode = {
        TacticalPlan.SALIDA_LAVOLPIANA: 1,
        TacticalPlan.JUEGO_DE_POSICION: 0,
        TacticalPlan.GEGENPRESS: 1,
        TacticalPlan.CATENACCIO: 5,
        TacticalPlan.ZONA_MISTA: 6,
    }

    for plan, expected in expected_mode.items():
        policy = RuleBasedOpeningManagerPolicy(
            RuleOpeningManagerConfig(team_tactical_plans=(plan, plan))
        )
        state = policy.initialize(inputs.observation)

        def select(key, selected_policy=policy, selected_state=state):
            return selected_policy.step(
                inputs.observation, key, selected_state
            ).decision.formation.layout_index

        selected = np.asarray(jax.jit(jax.vmap(select))(keys))
        for team in (0, 1):
            counts = np.bincount(
                selected[:, team], minlength=len(render_full_match.FORMATION_NAMES)
            )
            assert int(np.argmax(counts)) == expected
            assert counts[expected] > max(np.delete(counts, expected))
            assert counts[2] == 0
        assert np.unique(selected).size > 1


def test_opening_formation_temperature_must_be_positive():
    with pytest.raises(ValueError, match="formation_choice_temperature must be positive"):
        RuleOpeningManagerConfig(formation_choice_temperature=0.0)


def test_possession_shape_is_explicit_only_and_zona_asymmetry_is_mirror_safe():
    assert render_full_match.STARTING_FORMATION_PRIOR[2] == 0.0
    inputs = build_opening_policy_inputs(
        FootballWorld(),
        render_full_match._team_candidates(0, 20),
        render_full_match._team_candidates(1, 20),
        render_full_match.FORMATION_CATALOG,
        render_full_match.FORMATION_CATALOG,
        max_registered_players=(20, 20),
        formation_probabilities=render_full_match.STARTING_FORMATION_PRIOR,
        key=jax.random.key(0),
    )
    formation = inputs.observation.formation
    asymmetric = formation.candidate_anchor[0, 6]
    role = formation.candidate_role[0, 6]
    player_mask = formation.player_mask[0]
    baseline = _formation_tactical_fit(
        asymmetric[None, ...],
        role[None, ...],
        player_mask,
        TacticalPlan.ZONA_MISTA,
    )
    mirrored = _formation_tactical_fit(
        asymmetric.at[:, 1].multiply(-1.0)[None, ...],
        role[None, ...],
        player_mask,
        TacticalPlan.ZONA_MISTA,
    )

    np.testing.assert_allclose(baseline, mirrored, rtol=0.0, atol=1e-6)


def test_global_pair_lineup_is_formation_slot_permutation_equivariant():
    key = jax.random.key(17)
    candidates = (
        render_full_match._team_candidates(0, 20),
        render_full_match._team_candidates(1, 20),
    )

    def assigned_positions(layout):
        inputs = build_opening_policy_inputs(
            FootballWorld(),
            candidates[0],
            candidates[1],
            layout[None, ...],
            layout[None, ...],
            max_registered_players=(20, 20),
            formation_probabilities=np.ones(1, dtype=np.float32),
            key=key,
        )
        policy = RuleBasedOpeningManagerPolicy()
        decision = policy.step(
            inputs.observation, key, policy.initialize(inputs.observation)
        ).decision
        placement = np.asarray(decision.placement_slot[0])
        player_id = np.asarray(inputs.observation.players.player_id[0])
        return {
            int(identity): tuple(layout[int(slot)])
            for identity, slot in zip(player_id, placement, strict=True)
            if slot >= 0
        }

    layout = render_full_match.FORMATION_CATALOG[0]
    permutation = np.asarray([5, 2, 9, 0, 7, 1, 10, 4, 8, 3, 6])

    assert assigned_positions(layout) == assigned_positions(layout[permutation])


def test_joint_lineup_formation_choice_is_catalog_permutation_equivariant():
    key = jax.random.key(73)
    env = FootballWorld()
    probability = render_full_match.STARTING_FORMATION_PRIOR

    def decide(catalog, prior):
        inputs = build_opening_policy_inputs(
            env,
            render_full_match._team_candidates(0, 20),
            render_full_match._team_candidates(1, 20),
            catalog,
            catalog,
            max_registered_players=(20, 20),
            formation_probabilities=prior,
            key=key,
        )
        policy = RuleBasedOpeningManagerPolicy(
            RuleOpeningManagerConfig(
                team_tactical_plans=(
                    TacticalPlan.GEGENPRESS,
                    TacticalPlan.CATENACCIO,
                )
            )
        )
        decision = policy.step(
            inputs.observation, key, policy.initialize(inputs.observation)
        ).decision
        indices = np.asarray(decision.formation.layout_index, dtype=np.int64)
        return catalog[indices]

    baseline = decide(render_full_match.FORMATION_CATALOG, probability)
    permutation = np.asarray([6, 2, 4, 0, 5, 1, 3])
    permuted = decide(
        render_full_match.FORMATION_CATALOG[permutation], probability[permutation]
    )
    np.testing.assert_array_equal(baseline, permuted)


@pytest.mark.parametrize("requested", [("random",), ("random", "random", "random")])
def test_demo_plan_resolution_rejects_the_wrong_team_count(requested):
    with pytest.raises(ValueError, match="exactly two"):
        render_full_match._resolve_team_tactical_plans(
            requested, jax.random.key(0), _opening_players(jax.random.key(0))
        )


def test_demo_plan_resolution_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown demo tactical plan"):
        render_full_match._resolve_team_tactical_plans(
            ("unknown", "random"),
            jax.random.key(0),
            _opening_players(jax.random.key(0)),
        )


def test_demo_matrix_defaults_to_cpu_full_parallel_two_leg_runs(tmp_path):
    args = render_full_match._parser().parse_args(
        ["--output", str(tmp_path / "matrix"), "--plan-matrix"]
    )

    assert args.plan_matrix
    assert args.matrix_platform == "cpu"
    assert args.matrix_workers == 25
    assert args.matrix_legs == 2
    assert args.matrix_include_self_play
    assert args.matrix_equal_roster_abilities


def test_equal_roster_control_removes_identity_keyed_ability_noise():
    candidate_count = 20
    fixed = tuple(False for _ in range(candidate_count))
    inputs = build_opening_policy_inputs(
        FootballWorld(),
        render_full_match._team_candidates(0, candidate_count),
        render_full_match._team_candidates(1, candidate_count),
        render_full_match.FORMATION_CATALOG,
        render_full_match.FORMATION_CATALOG,
        max_registered_players=(candidate_count, candidate_count),
        sample_abilities=(fixed, fixed),
        key=jax.random.key(29),
    )

    players = inputs.observation.players
    for field in (
        "is_goalkeeper",
        "preferred_position",
        "max_speed",
        "height",
        "reach_height",
        "ball_control",
        "endurance_factor",
    ):
        values = np.asarray(getattr(players, field))
        np.testing.assert_array_equal(values[0], values[1])


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
    assert all("--equal-roster-abilities" in call[0] for call in commands)
    assert summary["equal_roster_abilities"] is True


def test_demo_matrix_can_restore_independent_roster_sampling(tmp_path, monkeypatch):
    output = tmp_path / "matrix"
    argv = [
        "--output",
        str(output),
        "--plan-matrix",
        "--no-matrix-equal-roster-abilities",
        "--matrix-legs",
        "1",
        "--no-matrix-include-self-play",
        "--maximum-steps",
        "1",
        "--report-only",
    ]
    args = render_full_match._parser().parse_args(argv)
    commands = []

    def completed(command, **kwargs):
        commands.append(command)
        return render_full_match.subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(render_full_match.subprocess, "run", completed)

    assert render_full_match._run_plan_matrix(args, argv) == 2
    summary = render_full_match.json.loads(
        (output / "matrix-summary.json").read_text(encoding="utf-8")
    )
    assert summary["equal_roster_abilities"] is False
    assert all("--equal-roster-abilities" not in command for command in commands)


def test_demo_matrix_normalizes_legacy_gpu_platform_to_cuda(tmp_path, monkeypatch):
    output = tmp_path / "matrix"
    argv = [
        "--output",
        str(output),
        "--plan-matrix",
        "--matrix-platform",
        "gpu",
        "--matrix-legs",
        "1",
        "--no-matrix-include-self-play",
        "--maximum-steps",
        "1",
        "--report-only",
    ]
    args = render_full_match._parser().parse_args(argv)
    environments = []

    def completed(command, **kwargs):
        environments.append(kwargs["env"])
        return render_full_match.subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(render_full_match.subprocess, "run", completed)

    assert render_full_match._run_plan_matrix(args, argv) == 2
    summary = render_full_match.json.loads(
        (output / "matrix-summary.json").read_text(encoding="utf-8")
    )
    assert summary["requested_platform"] == "gpu"
    assert summary["platform"] == "cuda"
    assert environments
    assert all(environment["JAX_PLATFORMS"] == "cuda" for environment in environments)


def test_demo_parser_accepts_explicit_cuda_matrix_platform(tmp_path):
    args = render_full_match._parser().parse_args(
        [
            "--output",
            str(tmp_path / "matrix"),
            "--plan-matrix",
            "--matrix-platform",
            "cuda",
        ]
    )

    assert args.matrix_platform == "cuda"


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
