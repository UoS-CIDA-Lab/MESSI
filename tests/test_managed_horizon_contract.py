from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from footballworld import FootballWorld, managed, managed_batch
from footballworld.managed_batch import _horizons

INT32_MAX = np.iinfo(np.int32).max


@pytest.mark.parametrize(
    "value",
    [INT32_MAX + 1, 2**100, np.uint64(2**63)],
)
def test_batch_horizon_rejects_wide_scalars_before_narrowing(value):
    with pytest.raises(ValueError, match="exceeds the supported int32 step budget"):
        _horizons(value, 2)


def test_batch_horizon_rejects_wide_vector_before_narrowing():
    value = np.asarray([1, 2**63], dtype=np.uint64)

    with pytest.raises(ValueError, match="exceeds the supported int32 step budget"):
        _horizons(value, 2)


def test_batch_horizon_preserves_safe_unsigned_values():
    value = np.asarray([0, INT32_MAX], dtype=np.uint64)

    result = _horizons(value, 2)

    assert result.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result, [0, INT32_MAX])


@pytest.mark.parametrize("module", [managed, managed_batch])
def test_managed_factory_rejects_wide_chunk_before_policy_construction(
    monkeypatch,
    module,
):
    monkeypatch.setattr(
        module,
        "make_rule_based_policy",
        lambda _env: pytest.fail("invalid chunk reached policy construction"),
    )
    factory = (
        module.make_managed_runner
        if module is managed
        else module.make_managed_batch_runner
    )

    with pytest.raises(ValueError, match="exceeds the supported int32 step budget"):
        factory(FootballWorld(), chunk_steps=INT32_MAX + 1)


def test_scalar_runner_rejects_wide_horizon_before_jax_narrowing(monkeypatch):
    class FakeSquad:
        pass

    monkeypatch.setattr(managed, "SquadSetup", FakeSquad)
    state = managed.ManagedMatchState(
        rollout=object(),
        setup=object(),
        management=object(),
        roster=object(),
        player_policy_state=object(),
        manager=SimpleNamespace(boundary=object(), policy=object()),
        opening_formation_checked=True,
    )
    runner = managed.ManagedRunner(
        env=None,
        player_policy=None,
        chunk_steps=1,
        manager_policy=None,
        opening_policy=None,
        _advance=None,
        _manager_initialize=None,
        _manager_decide=None,
        _refresh_roster=None,
        _apply_tactics=None,
        _opening_decide=None,
        _can_handle_goalkeeper=False,
    )

    with pytest.raises(ValueError, match="exceeds the supported int32 step budget"):
        runner.run(
            state,
            FakeSquad(),
            jnp.asarray([0, 1], dtype=jnp.uint32),
            INT32_MAX + 1,
        )
