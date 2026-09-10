import jax.numpy as jnp
import numpy as np

from footballworld.core.constants import RK_CORNER, RK_FREEKICK, RK_GOALKICK, RK_KICKOFF
from footballworld.policies import RuleManagerConfig, TacticalPlan
from footballworld.policies.rule_based.manager import (
    _formation_change_boundary,
    _formation_phase_fit,
)


def test_manager_tactical_plans_are_canonical_and_two_team():
    config = RuleManagerConfig(
        team_tactical_plans=("juego_de_posicion", TacticalPlan.CATENACCIO)
    )

    assert config.team_tactical_plans == (
        TacticalPlan.JUEGO_DE_POSICION,
        TacticalPlan.CATENACCIO,
    )


def test_phase_fit_prefers_expansion_in_possession_and_compact_cover_out():
    # Candidate 0 is higher and wider; candidate 1 has more defenders and is narrow.
    depth = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    width = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    defenders = jnp.asarray([0.0, 1.0], dtype=jnp.float32)

    juego_in = _formation_phase_fit(
        depth,
        width,
        defenders,
        jnp.bool_(True),
        jnp.bool_(False),
        TacticalPlan.JUEGO_DE_POSICION,
    )
    catenaccio_out = _formation_phase_fit(
        depth,
        width,
        defenders,
        jnp.bool_(False),
        jnp.bool_(True),
        TacticalPlan.CATENACCIO,
    )

    assert int(np.argmax(juego_in)) == 0
    assert int(np.argmax(catenaccio_out)) == 1


def test_phase_fit_is_inactive_when_restart_has_no_team():
    score = _formation_phase_fit(
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([1.0, 0.0]),
        jnp.asarray([0.0, 1.0]),
        jnp.bool_(False),
        jnp.bool_(False),
        TacticalPlan.GEGENPRESS,
    )

    np.testing.assert_array_equal(score, np.zeros(2, dtype=np.float32))


def test_formation_change_uses_only_full_reposition_restart_boundaries():
    assert bool(_formation_change_boundary(jnp.int32(RK_KICKOFF)))
    assert bool(_formation_change_boundary(jnp.int32(RK_GOALKICK)))
    assert not bool(_formation_change_boundary(jnp.int32(RK_FREEKICK)))
    assert not bool(_formation_change_boundary(jnp.int32(RK_CORNER)))
