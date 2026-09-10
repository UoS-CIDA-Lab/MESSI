import jax
import jax.numpy as jnp
import numpy as np
import pytest

from footballworld import (
    FootballWorld,
    Player,
    PlayerProfile,
    SubstitutionRequest,
    neutral_action,
)
from footballworld.core.randomness import (
    RandomEvent,
    event_random_key,
    frame_random_key,
    validate_prng_key,
)
from footballworld.rendering.integrity import replay_provenance
from footballworld.rules.restart import restart_taker_random_key

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


def _assert_tree_equal(left, right) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _substitution_request() -> SubstitutionRequest:
    return SubstitutionRequest(
        enabled=jnp.bool_(False),
        team=jnp.int32(0),
        outgoing_index=jnp.int32(1),
        incoming_player_id=jnp.int32(1_000),
        incoming_is_goalkeeper=jnp.bool_(False),
        incoming_max_speed=jnp.float32(7.5),
        incoming_height=jnp.float32(1.78),
        incoming_reach_height=jnp.float32(2.05),
        incoming_ball_control=jnp.float32(0.5),
        incoming_endurance_factor=jnp.float32(1.0),
        incoming_yellow_cards=jnp.int32(0),
    )


def _python_substitution_request() -> SubstitutionRequest:
    return SubstitutionRequest(
        enabled=False,
        team=0,
        outgoing_index=1,
        incoming_player_id=1_000,
        incoming_is_goalkeeper=False,
        incoming_max_speed=7.5,
        incoming_height=1.78,
        incoming_reach_height=2.05,
        incoming_ball_control=0.5,
        incoming_endurance_factor=1.0,
        incoming_yellow_cards=0,
    )


def test_public_roster_sampling_accepts_equivalent_threefry_keys():
    env = FootballWorld()
    team_0 = _team(0)
    team_1 = _team(100)

    legacy = env.reset(team_0, team_1, key=jax.random.PRNGKey(73))
    typed = env.reset(
        team_0,
        team_1,
        key=jax.random.key(73, impl="threefry2x32"),
    )

    _assert_tree_equal(legacy, typed)


def test_public_bench_sampling_accepts_equivalent_threefry_keys():
    env = FootballWorld()
    reset = env.reset(_team(0), _team(100))
    bench_0 = tuple(PlayerProfile(player_id=1_000 + index) for index in range(3))
    bench_1 = tuple(PlayerProfile(player_id=2_000 + index) for index in range(3))

    legacy = env.initialize_management(
        reset.rollout,
        bench_0,
        bench_1,
        key=jax.random.PRNGKey(73),
    )
    typed = env.initialize_management(
        reset.rollout,
        bench_0,
        bench_1,
        key=jax.random.key(73, impl="threefry2x32"),
    )

    _assert_tree_equal(legacy, typed)


@pytest.mark.parametrize("implementation", ["rbg", "unsafe_rbg"])
def test_public_roster_sampling_rejects_non_threefry_keys(implementation):
    env = FootballWorld()
    team_0 = _team(0)
    team_1 = _team(100)
    bad = jax.random.key(73, impl=implementation)

    with pytest.raises(ValueError, match="threefry2x32"):
        env.reset(team_0, team_1, key=bad)

    reset = env.reset(team_0, team_1)
    bench_0 = (PlayerProfile(player_id=1_000),)
    bench_1 = (PlayerProfile(player_id=2_000),)
    with pytest.raises(ValueError, match="threefry2x32"):
        env.initialize_management(
            reset.rollout,
            bench_0,
            bench_1,
            key=bad,
        )


@pytest.mark.parametrize("implementation", ["rbg", "unsafe_rbg"])
def test_public_environment_step_rejects_non_threefry_keys(implementation):
    env = FootballWorld()
    reset = env.reset(_team(0), _team(100))
    action = neutral_action(reset.rollout.state.players.position.shape[0])
    bad = jax.random.key(81, impl=implementation)

    with pytest.raises(ValueError, match="threefry2x32"):
        env.step(reset.rollout, reset.setup, action, bad)
    with pytest.raises(ValueError, match="threefry2x32"):
        env.step_with_events(reset.rollout, reset.setup, action, bad)
    with pytest.raises(ValueError, match="threefry2x32"):
        event_random_key(bad, RandomEvent.CONTEST_WINNER)


def test_event_type_error_precedes_key_validation_for_compatibility():
    with pytest.raises(TypeError, match="event must be RandomEvent"):
        event_random_key(object(), 7)


@pytest.mark.parametrize("address", [-1, 2**32, np.int64(2**40)])
def test_public_random_addresses_reject_values_that_would_wrap_uint32(address):
    key = jax.random.key(7)

    with pytest.raises(ValueError, match=r"\[0, 4294967295\]"):
        frame_random_key(key, address)
    with pytest.raises(ValueError, match=r"\[0, 4294967295\]"):
        event_random_key(key, RandomEvent.CONTEST_WINNER, address)


def test_public_random_address_accepts_the_largest_distinct_uint32_value():
    key = jax.random.key(7)

    expected = jax.random.fold_in(
        jax.random.fold_in(key, jnp.uint32(int(RandomEvent.CONTEST_WINNER))),
        jnp.uint32(2**32 - 1),
    )
    actual = event_random_key(key, RandomEvent.CONTEST_WINNER, 2**32 - 1)

    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(actual)),
        np.asarray(jax.random.key_data(expected)),
    )


def test_traced_random_addresses_fail_closed_for_wide_integer_dtypes():
    previous = bool(jax.config.jax_enable_x64)
    try:
        jax.config.update("jax_enable_x64", True)
        key = jax.random.key(7)
        wide = jnp.asarray(3, dtype=jnp.int64)

        with pytest.raises(TypeError, match="at most 32-bit"):
            jax.make_jaxpr(frame_random_key)(key, wide)
        with pytest.raises(TypeError, match="at most 32-bit"):
            jax.make_jaxpr(
                lambda address: event_random_key(
                    key, RandomEvent.CONTEST_WINNER, address
                )
            )(wide)
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize(
    "field",
    ["team", "outgoing_index", "incoming_player_id", "incoming_yellow_cards"],
)
@pytest.mark.parametrize("value", [np.int64(2**32 + 1), np.int64(-(2**40))])
def test_substitution_host_integers_cannot_alias_int32_fields(field, value):
    env = FootballWorld()
    rollout = env.reset(_team(0), _team(100)).rollout
    request = _substitution_request()._replace(**{field: value})

    with pytest.raises(ValueError, match="not representable as int32"):
        env.substitute(rollout, request)


def test_substitution_python_scalars_are_independent_of_global_x64_default():
    env = FootballWorld()
    rollout = env.reset(_team(0), _team(100)).rollout
    request = _python_substitution_request()
    previous = bool(jax.config.jax_enable_x64)
    try:
        jax.config.update("jax_enable_x64", False)
        ordinary = env.substitute(rollout, request)
        jax.config.update("jax_enable_x64", True)
        wide_default = env.substitute(rollout, request)
    finally:
        jax.config.update("jax_enable_x64", previous)

    _assert_tree_equal(ordinary, wide_default)


def test_substitution_domain_uses_the_canonical_float32_profile_value():
    env = FootballWorld()
    rollout = env.reset(_team(0), _team(100)).rollout
    request = _python_substitution_request()._replace(
        enabled=True,
        incoming_max_speed=np.nextafter(
            np.float64(env.roster_sampling.max_max_speed_mps),
            np.float64(np.inf),
        ),
    )

    result = env.substitute(rollout, request)

    assert bool(result.applied)
    assert float(result.rollout.state.players.max_speed[1]) == np.float32(
        env.roster_sampling.max_max_speed_mps
    )


def test_substitution_canonicalization_preserves_compiled_fixed_shape_path():
    env = FootballWorld()
    rollout = env.reset(_team(0), _team(100)).rollout
    request = _substitution_request()

    eager = env.substitute(rollout, request)
    compiled = jax.jit(env.substitute)(rollout, request)

    _assert_tree_equal(eager, compiled)


def test_public_restart_key_derivation_preserves_key_validation():
    with pytest.raises(ValueError, match="threefry2x32"):
        restart_taker_random_key(
            jax.random.key(7, impl="rbg"),
            jnp.int32(4),
            jnp.int32(0),
            jnp.int32(1),
        )


@pytest.mark.parametrize("implementation", ["rbg", "unsafe_rbg"])
def test_replay_provenance_rejects_unreplayable_key_implementations(
    implementation,
):
    with pytest.raises(ValueError, match="threefry2x32"):
        replay_provenance(
            FootballWorld(),
            jax.random.key(7, impl=implementation),
            policies={},
        )


@pytest.mark.parametrize(
    "malformed",
    [
        7,
        np.uint32(3),
        np.zeros((3,), dtype=np.uint32),
        jax.random.split(jax.random.key(5), 3),
    ],
)
def test_scalar_key_contract_rejects_malformed_or_batched_values(malformed):
    with pytest.raises(TypeError):
        validate_prng_key(malformed)


def test_scalar_key_contract_rejects_objects_that_only_mimic_array_metadata():
    class FakeKey:
        dtype = np.dtype(np.uint32)
        shape = (2,)

    with pytest.raises(TypeError, match="JAX PRNG key"):
        validate_prng_key(FakeKey())


def test_scalar_key_contract_accepts_jax_compatible_legacy_numpy_storage():
    key = np.asarray(jax.random.PRNGKey(13))

    assert validate_prng_key(key) is key


def test_batch_key_contract_is_exact_and_rejects_other_implementations():
    keys = jax.random.split(jax.random.key(9), 3)
    assert validate_prng_key(keys, batch_size=3) is keys

    with pytest.raises(TypeError):
        validate_prng_key(keys, batch_size=2)
    with pytest.raises(ValueError, match="threefry2x32"):
        validate_prng_key(
            jax.random.split(jax.random.key(9, impl="rbg"), 3),
            batch_size=3,
        )


def test_legacy_key_rejects_a_non_threefry_process_default():
    previous = str(jax.config.jax_default_prng_impl)
    try:
        jax.config.update("jax_default_prng_impl", "rbg")
        with pytest.raises(ValueError, match="threefry2x32"):
            validate_prng_key(jax.random.PRNGKey(17))
    finally:
        jax.config.update("jax_default_prng_impl", previous)


@pytest.mark.parametrize(
    "key",
    [jax.random.PRNGKey(17), jax.random.key(17, impl="threefry2x32")],
)
def test_key_validation_adds_no_jaxpr_equations(key):
    traced = jax.make_jaxpr(validate_prng_key)(key)

    assert traced.jaxpr.eqns == []


def test_partitionable_threefry_fails_at_the_next_random_boundary():
    previous = bool(jax.config.jax_threefry_partitionable)
    try:
        jax.config.update("jax_threefry_partitionable", True)
        with pytest.raises(RuntimeError, match="THREEFRY_PARTITIONABLE=0"):
            validate_prng_key(jax.random.key(11, impl="threefry2x32"))
    finally:
        jax.config.update("jax_threefry_partitionable", previous)
