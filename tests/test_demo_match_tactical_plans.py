import importlib.util
from pathlib import Path

import jax
import pytest

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
