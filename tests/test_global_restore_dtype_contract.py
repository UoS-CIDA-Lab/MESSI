import numpy as np
import pytest

from footballworld import FootballWorld, MatchConfig, Player, PlayerProfile
from footballworld.core.contact import MECHANISM_COUNT


def _view():
    env = FootballWorld(match=MatchConfig(minimum_team_players=(1, 1)))
    reset = env.reset(
        (
            Player(
                PlayerProfile(player_id=7, is_goalkeeper=True),
                initial_position=(-10.0, 0.0),
            ),
        ),
        (
            Player(
                PlayerProfile(player_id=100, is_goalkeeper=True),
                initial_position=(-10.0, 0.0),
            ),
        ),
    )
    return env, env.global_state_view(reset.rollout)


def test_global_restore_accepts_canonical_view():
    env, view = _view()

    restored = env.restore_global_state_view(view)

    assert restored.state.players.team_id.dtype == np.dtype(np.int32)
    assert restored.state.players.position.dtype == np.dtype(np.float32)


def test_global_restore_preserves_finite_physics_outside_normalizer_envelope():
    env, view = _view()
    normalized_velocity = np.asarray([2.0, -1.5, 1.25], dtype=np.float32)
    outside = view._replace(
        state=view.state._replace(
            ball=view.state.ball._replace(velocity=normalized_velocity)
        )
    )

    restored = env.restore_global_state_view(outside)

    expected = normalized_velocity * np.float32(
        env.normalization_context().ball_speed_scale_mps
    )
    np.testing.assert_array_equal(restored.state.ball.velocity, expected)


def test_global_restore_rejects_float_team_ids_before_indexing():
    env, view = _view()
    damaged = view._replace(
        state=view.state._replace(
            players=view.state.players._replace(
                team_id=np.asarray(view.state.players.team_id, dtype=np.float32)
            )
        )
    )

    with pytest.raises(TypeError, match="players.team_id must have dtype int32"):
        env.restore_global_state_view(damaged)


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
def test_global_restore_rejects_non_float32_position(dtype):
    env, view = _view()
    damaged = view._replace(
        state=view.state._replace(
            players=view.state.players._replace(
                position=np.asarray(view.state.players.position, dtype=dtype)
            )
        )
    )

    with pytest.raises(TypeError, match="players.position must have dtype float32"):
        env.restore_global_state_view(damaged)


def test_global_restore_rejects_non_boolean_ball_live():
    env, view = _view()
    damaged = view._replace(
        state=view.state._replace(ball=view.state.ball._replace(live=np.int32(1)))
    )

    with pytest.raises(TypeError, match="ball.live must have dtype bool"):
        env.restore_global_state_view(damaged)


def test_global_restore_rejects_float_nested_contact_code():
    env, view = _view()
    damaged_contact = view.state.possession.last_contact._replace(
        actor=np.float32(-1.0)
    )
    damaged = view._replace(
        state=view.state._replace(
            possession=view.state.possession._replace(last_contact=damaged_contact)
        )
    )

    with pytest.raises(TypeError, match="possession.last_contact.actor.*int32"):
        env.restore_global_state_view(damaged)


def test_global_restore_rejects_counter_overflow_without_bounding_physics():
    env, view = _view()
    recovery = np.asarray(
        view.state.players.challenge_recovery_substeps,
        dtype=np.float32,
    ).copy()
    recovery[0] = np.finfo(np.float32).max
    damaged = view._replace(
        state=view.state._replace(
            players=view.state.players._replace(challenge_recovery_substeps=recovery)
        )
    )

    with pytest.raises(
        ValueError,
        match="players.challenge_recovery_substeps cannot be represented as int32 ticks",
    ):
        env.restore_global_state_view(damaged)


def test_global_restore_rejects_out_of_domain_normalized_ability():
    env, view = _view()
    max_speed = np.asarray(view.state.players.max_speed).copy()
    max_speed[0] = np.float32(-0.25)
    damaged = view._replace(
        state=view.state._replace(
            players=view.state.players._replace(max_speed=max_speed)
        )
    )

    with pytest.raises(ValueError, match="players.max_speed must lie in \\[0, 1\\]"):
        env.restore_global_state_view(damaged)


def test_global_restore_rejects_nonunit_body_forward():
    env, view = _view()
    body_forward = np.asarray(view.state.players.body_forward).copy()
    body_forward[0] = np.asarray([0.0, 0.0], dtype=np.float32)
    damaged = view._replace(
        state=view.state._replace(
            players=view.state.players._replace(body_forward=body_forward)
        )
    )

    with pytest.raises(
        ValueError, match="players.body_forward must contain unit vectors"
    ):
        env.restore_global_state_view(damaged)


def test_global_restore_rejects_out_of_range_nested_contact_enum():
    env, view = _view()
    damaged = view._replace(
        state=view.state._replace(
            possession=view.state.possession._replace(
                last_contact=view.state.possession.last_contact._replace(
                    mechanism=np.int32(MECHANISM_COUNT)
                )
            )
        )
    )

    with pytest.raises(ValueError, match="possession.last_contact.mechanism"):
        env.restore_global_state_view(damaged)


def test_global_restore_rejects_out_of_range_release_address():
    env, view = _view()
    player_count = view.state.players.position.shape[0]
    damaged = view._replace(
        state=view.state._replace(
            restart_release=view.state.restart_release._replace(
                taker=np.int32(player_count)
            )
        )
    )

    with pytest.raises(ValueError, match="restart_release.taker"):
        env.restore_global_state_view(damaged)
