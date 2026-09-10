import jax
import jax.numpy as jnp
import numpy as np

from footballworld import FootballWorld, Player, PlayerProfile
from footballworld.core.action import IntentAction
from footballworld.core.constants import RK_THROWIN
from footballworld.core.contact import INTENT_MOVE
from footballworld.environment.management import ManagerCommand
from footballworld.rules.restart_positioning import restart_taker_release_pose

_FORMATION = (
    (-50.0, 0.0),
    (-35.0, -24.0),
    (-35.0, -8.0),
    (-35.0, 8.0),
    (-35.0, 24.0),
    (-20.0, -18.0),
    (-20.0, 0.0),
    (-20.0, 18.0),
    (-8.0, -24.0),
    (-8.0, 0.0),
    (-8.0, 24.0),
)


def _team(identity_base):
    return tuple(
        Player(
            PlayerProfile(
                player_id=identity_base + index,
                is_goalkeeper=index == 0,
            ),
            position,
        )
        for index, position in enumerate(_FORMATION)
    )


def _zero_action(player_count):
    return IntentAction(
        intent=jnp.full((player_count,), INTENT_MOVE, dtype=jnp.int32),
        move=jnp.zeros((player_count, 2), dtype=jnp.float32),
        force_to_ball=jnp.zeros((player_count, 2), dtype=jnp.float32),
        launch=jnp.full((player_count,), -1.0, dtype=jnp.float32),
        spin=jnp.zeros((player_count, 2), dtype=jnp.float32),
        gaze_center=jnp.zeros((player_count,), dtype=jnp.float32),
    )


def test_manager_selected_throwin_taker_keeps_position_then_walks_at_speed_limit():
    env = FootballWorld()
    reset = env.reset(_team(1_000), _team(2_000))
    management = env.initialize_management(reset.rollout, (), ())
    state = reset.rollout.state
    previous_taker = 1
    selected_taker = 8
    throw_ball = state.ball._replace(
        position=jnp.asarray(
            [12.0, env.stadium.half_width, env.ball.radius], dtype=jnp.float32
        ),
        velocity=jnp.zeros(3, dtype=jnp.float32),
        spin=jnp.zeros(3, dtype=jnp.float32),
        live=jnp.bool_(False),
    )
    throw_restart = state.restart._replace(
        kind=jnp.int32(RK_THROWIN),
        team=jnp.int32(0),
        substeps_remaining=jnp.int32(240),
        taker=jnp.int32(previous_taker),
        indirect=jnp.bool_(False),
        opened_control_tick=state.control_tick,
    )
    provisional = state._replace(
        ball=throw_ball,
        restart=throw_restart,
        restart_layout_ready=jnp.bool_(True),
    )
    previous_release_pose, _ = restart_taker_release_pose(
        provisional,
        stadium=env.stadium,
        ball=env.ball,
        body=env.body,
    )
    provisional = provisional._replace(
        players=provisional.players._replace(
            position=provisional.players.position.at[previous_taker].set(
                previous_release_pose
            )
        )
    )
    rollout = reset.rollout._replace(state=provisional)
    position_before = np.asarray(provisional.players.position)
    velocity_before = np.asarray(provisional.players.velocity)

    command = ManagerCommand.empty(0)
    command = command._replace(
        set_piece_takers=command.set_piece_takers.with_request(
            0,
            RK_THROWIN,
            player_slot=selected_taker,
        )
    )
    decision = env.manager_command(
        rollout,
        management.squad,
        management.state,
        command,
    )
    designated = decision.rollout.state
    position_after_designation = np.asarray(designated.players.position)
    velocity_after_designation = np.asarray(designated.players.velocity)

    assert bool(np.asarray(decision.set_piece_takers_applied[0, RK_THROWIN]))
    assert int(np.asarray(designated.restart.taker)) == selected_taker
    assert bool(np.asarray(designated.restart_layout_ready))
    np.testing.assert_array_equal(
        position_after_designation[selected_taker],
        position_before[selected_taker],
    )
    np.testing.assert_array_equal(
        velocity_after_designation[selected_taker],
        velocity_before[selected_taker],
    )
    assert not np.array_equal(
        position_after_designation[previous_taker],
        position_before[selected_taker],
    )

    target, _ = restart_taker_release_pose(
        designated,
        stadium=env.stadium,
        ball=env.ball,
        body=env.body,
    )
    distance_before = float(
        np.linalg.norm(position_after_designation[selected_taker] - np.asarray(target))
    )
    stepped = env.step(
        decision.rollout,
        reset.setup,
        _zero_action(designated.players.position.shape[0]),
        jax.random.key(91),
    )
    position_after_step = np.asarray(stepped.rollout.state.players.position)
    distance_after = float(
        np.linalg.norm(position_after_step[selected_taker] - np.asarray(target))
    )
    travel = float(
        np.linalg.norm(
            position_after_step[selected_taker]
            - position_after_designation[selected_taker]
        )
    )
    maximum_travel = float(
        np.asarray(designated.players.max_speed[selected_taker])
    ) / float(env.timebase.control_fps)

    assert 0.0 < travel <= maximum_travel + 1e-5
    assert distance_after < distance_before
    assert int(np.asarray(stepped.rollout.state.restart.kind)) == RK_THROWIN


def test_kickoff_release_pose_is_invariant_to_roster_storage_order():
    env = FootballWorld()
    reset = env.reset(_team(1_000), _team(2_000))
    state = reset.rollout.state
    team = int(np.asarray(state.restart.team))
    taker = int(np.asarray(state.restart.taker))
    members = [
        index
        for index in np.flatnonzero(np.asarray(state.players.team_id) == team)
        if index != taker
    ][:3]
    assert len(members) == 3

    ball_y = state.ball.position[1]
    position = state.players.position.at[:, 1].set(ball_y)
    position = position.at[jnp.asarray(members), 1].set(
        ball_y + jnp.asarray([2.0, -1.0, -1.0], dtype=position.dtype)
    )
    state = state._replace(players=state.players._replace(position=position))
    target, facing = restart_taker_release_pose(
        state, stadium=env.stadium, ball=env.ball, body=env.body
    )

    permutation = np.arange(state.players.position.shape[0])
    permutation[members[0]], permutation[members[2]] = (
        permutation[members[2]],
        permutation[members[0]],
    )
    permutation = jnp.asarray(permutation)
    permuted_players = jax.tree_util.tree_map(
        lambda value: value[permutation], state.players
    )
    permuted_taker = int(np.flatnonzero(np.asarray(permutation) == taker)[0])
    permuted = state._replace(
        players=permuted_players,
        restart=state.restart._replace(taker=jnp.int32(permuted_taker)),
    )
    permuted_target, permuted_facing = restart_taker_release_pose(
        permuted, stadium=env.stadium, ball=env.ball, body=env.body
    )

    np.testing.assert_allclose(
        np.asarray(permuted_target), np.asarray(target), atol=2e-6
    )
    np.testing.assert_allclose(
        [np.cos(float(permuted_facing)), np.sin(float(permuted_facing))],
        [np.cos(float(facing)), np.sin(float(facing))],
        atol=2e-6,
    )
