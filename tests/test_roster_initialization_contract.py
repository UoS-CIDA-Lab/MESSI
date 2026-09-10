import numpy as np
import pytest

from footballworld import Player, PlayerProfile
from footballworld.environment.episode import make_match_setup
from footballworld.environment.initialization import initialize_state


def _player(player_id, *, goalkeeper: bool = False) -> Player:
    return Player(
        PlayerProfile(player_id=player_id, is_goalkeeper=goalkeeper),
        initial_position=(-10.0, 0.0),
    )


@pytest.mark.parametrize("player_id", [-1, 2**31, 2**32 + 7])
def test_starter_identity_rejects_values_outside_int32_domain(player_id):
    with pytest.raises(ValueError, match="int32 identity domain"):
        initialize_state((_player(player_id),), (_player(100),))


@pytest.mark.parametrize("player_id", [True, 1.0, "1"])
def test_starter_identity_requires_non_boolean_integer(player_id):
    with pytest.raises(TypeError, match="non-boolean integers"):
        initialize_state((_player(player_id),), (_player(100),))


def test_starter_identity_checks_uniqueness_before_narrowing():
    with pytest.raises(ValueError, match="unique across both teams"):
        initialize_state((_player(7),), (_player(np.int64(7)),))


def test_safe_numpy_integer_identity_is_canonicalized_to_int32():
    result = initialize_state(
        (_player(np.int64(7)),),
        (_player(np.int64(100)),),
    )

    assert result.state.players.player_id.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(result.state.players.player_id, [7, 100])


def test_starter_goalkeeper_flag_requires_bool():
    with pytest.raises(TypeError, match="is_goalkeeper values must be bool"):
        initialize_state(
            (_player(7, goalkeeper=np.bool_(True)),),
            (_player(100),),
        )


@pytest.mark.parametrize("kickoff_team", [True, np.bool_(False), 1.0])
def test_direct_initialization_rejects_non_integer_kickoff_team(kickoff_team):
    with pytest.raises(TypeError, match="non-boolean integer"):
        initialize_state(
            (_player(7),),
            (_player(100),),
            kickoff_team=kickoff_team,
        )


def test_direct_initialization_accepts_safe_numpy_kickoff_team():
    result = initialize_state(
        (_player(7),),
        (_player(100),),
        kickoff_team=np.int64(1),
    )

    assert int(result.state.kickoff_team) == 1


def test_direct_initialization_rejects_zero_max_speed():
    stationary = Player(
        PlayerProfile(player_id=7, max_speed_mps=0.0),
        initial_position=(-10.0, 0.0),
    )

    with pytest.raises(ValueError, match="physical values"):
        initialize_state((stationary,), (_player(100),))


@pytest.mark.parametrize("kickoff_team", [True, np.bool_(False), 1.5])
def test_direct_match_setup_rejects_non_integer_kickoff_team(kickoff_team):
    initialized = initialize_state((_player(7),), (_player(100),))
    malformed = initialized.state._replace(kickoff_team=kickoff_team)

    with pytest.raises(TypeError, match="non-boolean integer"):
        make_match_setup(malformed)


def test_direct_match_setup_rejects_nonscalar_kickoff_team():
    initialized = initialize_state((_player(7),), (_player(100),))
    malformed = initialized.state._replace(kickoff_team=np.asarray([0], dtype=np.int32))

    with pytest.raises(ValueError, match="must be scalar"):
        make_match_setup(malformed)
