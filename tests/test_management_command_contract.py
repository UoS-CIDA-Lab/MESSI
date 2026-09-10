import jax
import numpy as np
import pytest

from footballworld import (
    FootballWorld,
    ManagerActingGoalkeeperCommand,
    ManagerCommand,
    ManagerFormationCommand,
    ManagerSetPieceTakerCommand,
    ManagerSubstitutionCommand,
    Player,
    PlayerProfile,
)
from footballworld.environment.management import _validate_manager_command

FORMATION = (
    (-50.0, 0.0),
    (-35.0, 25.0),
    (-35.0, -25.0),
    (-35.0, 15.0),
    (-35.0, -15.0),
    (-20.0, 20.0),
    (-20.0, -20.0),
    (-20.0, 0.0),
    (-10.0, 25.0),
    (-10.0, -25.0),
    (-12.0, 0.0),
)


def _team(identity_base: int) -> tuple[Player, ...]:
    return tuple(
        Player(
            PlayerProfile(
                player_id=identity_base + slot,
                is_goalkeeper=slot == 0,
            ),
            initial_position=position,
        )
        for slot, position in enumerate(FORMATION)
    )


def _fixture():
    env = FootballWorld()
    reset = env.reset(_team(0), _team(100))
    initialized = env.initialize_management(reset.rollout, (), ())
    return env, reset.rollout, initialized


def _assert_tree_equal(left, right) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_command_builders_reject_host_values_that_would_alias_int32():
    bad = np.int64(2**32)

    with pytest.raises(ValueError, match="not representable as int32"):
        ManagerSubstitutionCommand.empty(1).with_request(
            0,
            0,
            outgoing_index=bad,
            incoming_bench_index=0,
        )
    with pytest.raises(ValueError, match="not representable as int32"):
        ManagerFormationCommand.empty().with_request(0, layout_index=bad)
    with pytest.raises(ValueError, match="not representable as int32"):
        ManagerActingGoalkeeperCommand.empty().with_request(0, player_slot=bad)
    with pytest.raises(ValueError, match="not representable as int32"):
        ManagerSetPieceTakerCommand.empty().with_request(
            0,
            1,
            player_slot=bad,
        )


def test_direct_host_command_rejects_int32_alias_before_jax_conversion():
    env, rollout, initialized = _fixture()
    command = ManagerCommand.empty(1)
    command = command._replace(
        formations=command.formations._replace(
            layout_index=np.asarray([2**32, -1], dtype=np.int64)
        )
    )

    with pytest.raises(ValueError, match="not representable as int32"):
        env.manager_command(
            rollout,
            initialized.squad,
            initialized.state,
            command,
        )


def test_safe_host_command_arrays_are_independent_of_global_x64_default():
    env, rollout, initialized = _fixture()
    command = jax.tree.map(
        lambda value: np.asarray(
            value,
            dtype=np.bool_ if value.dtype == np.dtype(np.bool_) else np.int64,
        ),
        ManagerCommand.empty(1),
    )
    previous = bool(jax.config.jax_enable_x64)
    try:
        jax.config.update("jax_enable_x64", False)
        ordinary = env.manager_command(
            rollout,
            initialized.squad,
            initialized.state,
            command,
        )
        jax.config.update("jax_enable_x64", True)
        wide_default = env.manager_command(
            rollout,
            initialized.squad,
            initialized.state,
            command,
        )
    finally:
        jax.config.update("jax_enable_x64", previous)

    _assert_tree_equal(ordinary, wide_default)


def test_traced_command_validation_adds_no_jaxpr_equations():
    _, rollout, initialized = _fixture()
    command = ManagerCommand.empty(1)

    traced = jax.make_jaxpr(
        lambda value: _validate_manager_command(
            rollout.state,
            initialized.squad,
            value,
        )[1]
    )(command)

    assert traced.jaxpr.eqns == []
