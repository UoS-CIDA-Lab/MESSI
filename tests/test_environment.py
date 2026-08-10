"""SoccerBC 환경 계약 회귀 테스트.

AAMAS2027의 ``recon/tests/test_contracts.py``처럼 문서의 핵심 계약을 실행 가능한
반례로 고정한다. 특히 관측이 같은데 다음 규칙/물리가 달라지는 상태 alias는 단순 형상
검사로 잡히지 않으므로, 두 상태의 관측과 전이 결과를 함께 비교한다.

실행(저장소 루트에서)::

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
      PYTHONPATH=env python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import unittest
from dataclasses import FrozenInstanceError

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax
import jax.numpy as jnp
import numpy as np

from constants import *
from env import SoccerEnv


class EnvContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = SoccerEnv(game_duration=300)

    def test_default_shapes_and_schema(self):
        env = self.env
        obs, state = env.reset_array(jax.random.PRNGKey(0))
        self.assertEqual(obs.shape, (22, 470))
        self.assertEqual(env.get_state(state).shape, (538,))
        self.assertEqual((env.obs_dim, env.state_dim, env.action_dim), (470, 538, 8))
        self.assertEqual(env.schema_version, {"action": 1, "observation": 6, "state": 3})
        self.assertEqual(env.obs_spec()["context"]["end"], env.obs_dim)

    def test_custom_size_requires_explicit_rosters(self):
        with self.assertRaisesRegex(ValueError, "omitted rosters"):
            SoccerEnv(n_agents=5, n_opponents=5)

    def test_invalid_action_shape_is_rejected(self):
        state = self.env.reset_state(jax.random.PRNGKey(1))
        with self.assertRaisesRegex(ValueError, "act_arr must have shape"):
            self.env.step_env_array(
                jax.random.PRNGKey(2), state, jnp.zeros((self.env.N, ACTION_DIM - 1))
            )

    def test_control_fps_must_match_physics_decimation(self):
        for fps in (100.0, 50.0, 25.0, 20.0):
            self.assertAlmostEqual(SoccerEnv(control_fps=fps).control_fps, fps)
        with self.assertRaisesRegex(ValueError, "not exactly representable"):
            SoccerEnv(control_fps=30.0)

    def test_configuration_is_immutable_and_isolated(self):
        a, b = SoccerEnv(), SoccerEnv()
        self.assertIsNot(a.e_cfg, b.e_cfg)
        with self.assertRaises(FrozenInstanceError):
            a.e_cfg.decimation = 99

    def test_nan_action_is_sanitized(self):
        env = self.env
        state = env.reset_state(jax.random.PRNGKey(3))
        action = jnp.zeros((env.N, ACTION_DIM)).at[0, ACTION_MOVE.start].set(jnp.nan)
        obs, next_state, *_ = env.step_env_array(jax.random.PRNGKey(4), state, action)
        self.assertTrue(bool(jnp.all(jnp.isfinite(obs))))
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(x)))
                            for x in jax.tree_util.tree_leaves(next_state)))

    def test_lightweight_step_skips_optional_outputs(self):
        env = self.env
        state = env.reset_state(jax.random.PRNGKey(5))
        obs, _, _, _, info = env.step_env_array(
            jax.random.PRNGKey(6), state, jnp.zeros((env.N, ACTION_DIM)),
            include_bc_info=False, compute_observation=False,
        )
        self.assertEqual(obs.shape, (env.N, 0))
        self.assertNotIn("bc_action_mask", info)


class ObservationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = SoccerEnv(game_duration=300)

    def open_state(self, seed=10):
        env = self.env
        return env.reset_state(jax.random.PRNGKey(seed))._replace(
            ball_state=jnp.int32(BALL_ALIVE), restart_t=jnp.int32(0),
            restart_kind=jnp.int32(RK_NONE), pending_taker=jnp.int32(NO_PLAYER),
            poss_team=jnp.int32(TEAM_0), last_touch_team=jnp.int32(TEAM_0),
            restart_team=jnp.int32(TEAM_0), throw_taker=jnp.int32(NO_PLAYER),
            setpiece_taker=jnp.int32(NO_PLAYER), restart_indirect=jnp.bool_(False),
        )

    def test_self_velocity_is_observed_and_facing_is_derivable(self):
        env = self.env
        vel = jnp.zeros((env.N, 2)).at[0].set(jnp.array([3.0, -4.0]))
        state = self.open_state()._replace(
            player_vel=vel, player_facing=env.facing_from_velocity(vel, self.open_state().attack_dir)
        )
        obs = env.get_obs_array(state)
        start, end = env.obs_spec()["self"]["features"]["abs_vel"]
        np.testing.assert_allclose(
            np.asarray(obs[0, start:end]),
            np.asarray(vel[0] * state.attack_dir[0] / env.e_cfg.norm_player_vel), atol=1e-7,
        )
        expected = env.facing_from_velocity(state.player_vel, state.attack_dir)
        np.testing.assert_allclose(np.asarray(state.player_facing), np.asarray(expected), atol=1e-7)

    def test_live_indirect_free_kick_latch_is_observable(self):
        """★ restart_t=0 뒤에도 간접 직접골 무효 규칙은 살아 있으므로 비트를 숨기면 안 된다."""
        env = self.env
        direct = self.open_state(11)._replace(setpiece_taker=jnp.int32(0))
        indirect = direct._replace(restart_indirect=jnp.bool_(True))
        obs_diff = jnp.max(jnp.abs(env.get_obs_array(direct) - env.get_obs_array(indirect)))
        state_diff = jnp.max(jnp.abs(env.get_state(direct) - env.get_state(indirect)))
        self.assertEqual(float(obs_diff), 1.0)
        self.assertEqual(float(state_diff), 1.0)

    def test_retouch_origin_is_observable(self):
        """★ 같은 taker라도 스로인은 상대 골 직접 득점까지 무효라 출처를 구분해야 한다."""
        env = self.env
        base = self.open_state(12)
        throw = base._replace(throw_taker=jnp.int32(0))
        setpiece = base._replace(setpiece_taker=jnp.int32(0))
        spec = env.obs_spec()["context"]["features"]["retouch_is_throw"]
        start, end = spec
        self.assertTrue(bool(jnp.all(env.get_obs_array(throw)[:, start:end] == 1.0)))
        self.assertTrue(bool(jnp.all(env.get_obs_array(setpiece)[:, start:end] == 0.0)))

    def test_penalty_encroachment_latches_are_observable(self):
        env = self.env
        base = self.open_state(13)._replace(penalty_flight_team=jnp.int32(TEAM_0))
        clean = env.get_obs_array(base)
        attack = env.get_obs_array(base._replace(
            penalty_encroach_mask=jnp.zeros(env.N, bool).at[1].set(True)))
        defend = env.get_obs_array(base._replace(
            penalty_encroach_mask=jnp.zeros(env.N, bool).at[env.n_agents].set(True)))
        self.assertGreater(float(jnp.max(jnp.abs(clean - attack))), 0.0)
        self.assertGreater(float(jnp.max(jnp.abs(attack - defend))), 0.0)

    def test_second_half_kickoff_owner_is_observable(self):
        env = self.env
        a = self.open_state(14)._replace(kickoff_team=jnp.int32(TEAM_0))
        b = a._replace(kickoff_team=jnp.int32(TEAM_1))
        self.assertEqual(float(jnp.max(jnp.abs(env.get_obs_array(a) - env.get_obs_array(b)))), 2.0)

    def test_inactive_rows_and_other_slots_are_zero(self):
        env = self.env
        state = self.open_state(15)._replace(
            active_player=jnp.ones(env.N, bool).at[1].set(False))
        obs = env.get_obs_array(state)
        self.assertTrue(bool(jnp.all(obs[1] == 0.0)))
        spec = env.obs_spec()["others"]
        observer = 0
        slot = int(np.flatnonzero(np.asarray(env.others_idx[observer]) == 1)[0])
        start = spec["start"] + slot * spec["size"]
        self.assertTrue(bool(jnp.all(obs[observer, start:start + spec["size"]] == 0.0)))


class MovementContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = SoccerEnv(game_duration=300)

    def open_state(self, seed=20):
        env = self.env
        state = env.reset_state(jax.random.PRNGKey(seed))
        return state._replace(
            ball_state=jnp.int32(BALL_ALIVE), restart_t=jnp.int32(0),
            restart_kind=jnp.int32(RK_NONE), pending_taker=jnp.int32(NO_PLAYER),
        )

    def run_move(self, state, directions=None, powers=None, effective=None):
        env = self.env
        return env._move(
            state,
            jnp.ones(env.N, bool) if effective is None else effective,
            jnp.zeros((env.N, 2)) if directions is None else directions,
            jnp.zeros(env.N) if powers is None else powers,
        )

    def test_direct_boundary_clip_removes_outward_velocity(self):
        env = self.env
        state = self.open_state()._replace(
            player_pos=self.open_state().player_pos.at[0].set(jnp.array([env.hx, 0.0])),
            player_vel=jnp.zeros((env.N, 2)),
        )
        direction = jnp.zeros((env.N, 2)).at[0, 0].set(1.0)
        power = jnp.zeros(env.N).at[0].set(1.0)
        next_state = self.run_move(state, direction, power)
        self.assertEqual(float(next_state.player_pos[0, 0]), env.hx)
        self.assertEqual(float(next_state.player_vel[0, 0]), 0.0)

    def test_separation_boundary_clip_removes_outward_velocity(self):
        """★ 적분 뒤가 아니라 선수 분리로 경계에 닿아도 바깥 법선 속도는 남지 않는다."""
        env = self.env
        state = self.open_state(21)
        pos = state.player_pos.at[0].set(jnp.array([env.hx - 0.01, 0.0]))
        pos = pos.at[1].set(jnp.array([env.hx - 0.20, 0.0]))
        vel = jnp.zeros((env.N, 2)).at[0].set(jnp.array([0.5, 0.0]))
        state = state._replace(player_pos=pos, player_vel=vel)
        next_state = self.run_move(state)
        self.assertEqual(float(next_state.player_pos[0, 0]), env.hx)
        self.assertEqual(float(next_state.player_vel[0, 0]), 0.0)

    def test_boundary_projection_preserves_inward_and_tangent_velocity(self):
        env = self.env
        state = self.open_state(22)
        pos = state.player_pos.at[0].set(jnp.array([env.hx, 0.0]))
        vel = jnp.zeros((env.N, 2)).at[0].set(jnp.array([-2.0, 1.0]))
        state = state._replace(player_pos=pos, player_vel=vel)
        next_state = self.run_move(state, effective=jnp.zeros(env.N, bool))
        np.testing.assert_allclose(np.asarray(next_state.player_vel[0]), [-2.0, 1.0], atol=1e-6)

    def test_pinned_player_is_not_displaced_by_separation(self):
        env = self.env
        p = jnp.array([[0.0, 0.0], [0.1, 0.0]])
        v = jnp.zeros((2, 2))
        out, _ = env._separate(p, v, active=jnp.ones(2, bool), pinned=jnp.array([True, False]))
        np.testing.assert_allclose(np.asarray(out[0]), np.asarray(p[0]), atol=1e-7)
        self.assertGreaterEqual(float(jnp.linalg.norm(out[1] - out[0])), 2 * env.r_player - 1e-5)

    def test_move_inverse_round_trip(self):
        env = self.env
        state = self.open_state(23)
        action = jax.random.uniform(jax.random.PRNGKey(24), (env.N, ACTION_DIM), minval=-0.8, maxval=0.8)
        _, direction, power, *_ = env._decode(action, state.attack_dir)
        target = self.run_move(state, direction, power)
        inverse = env.infer_move_action(state, target.player_vel)
        _, inv_direction, inv_power, *_ = env._decode(inverse, state.attack_dir)
        replay = self.run_move(state, inv_direction, inv_power)
        np.testing.assert_allclose(np.asarray(replay.player_vel), np.asarray(target.player_vel), atol=3e-5)
        np.testing.assert_allclose(np.asarray(replay.player_pos), np.asarray(target.player_pos), atol=3e-5)

    def test_kick_inverse_round_trip(self):
        """AAMAS2027의 킥 역산 계약을 SoccerBC의 현재 L∞ action codec까지 포함해 검증한다."""
        env = self.env
        state = self.open_state(25)
        kicker = 0
        direction = jnp.array([0.6, 0.8])
        speed, launch, side, back = 18.0, 0.2, 0.35, -0.4
        velocity = speed * jnp.array([
            jnp.cos(launch) * direction[0], jnp.cos(launch) * direction[1], jnp.sin(launch)])
        lateral = jnp.array([-direction[1], direction[0], 0.0])
        spin = env.e_cfg.spin_max * (
            (-back) * lateral + side * jnp.array([0.0, 0.0, 1.0]))
        inferred = env.infer_kick_action(state, kicker, velocity, spin, state.ball_pos[DIM_Z])
        action = jnp.zeros((env.N, ACTION_DIM)).at[kicker].set(inferred)
        _, _, _, decoded_dir, decoded_power, decoded_launch, decoded_side, decoded_back = (
            env._decode(action, state.attack_dir))
        floor = env.launch_lo(state.ball_pos[DIM_Z])
        angle = floor + (decoded_launch[kicker] / env.e_cfg.launch_max) * (
            env.e_cfg.launch_max - floor)
        replay_velocity = decoded_power[kicker] * env.e_cfg.f2b_speed_max * jnp.array([
            jnp.cos(angle) * decoded_dir[kicker, 0],
            jnp.cos(angle) * decoded_dir[kicker, 1], jnp.sin(angle)])
        replay_lateral = jnp.array([
            -decoded_dir[kicker, 1], decoded_dir[kicker, 0], 0.0])
        replay_spin = env.e_cfg.spin_max * (
            (-decoded_back[kicker]) * replay_lateral
            + decoded_side[kicker] * jnp.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(np.asarray(replay_velocity), np.asarray(velocity), atol=1e-4)
        np.testing.assert_allclose(np.asarray(replay_spin), np.asarray(spin), atol=1e-4)


class RuleAndAgencyContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = SoccerEnv(game_duration=300)

    def open_state(self, seed=30):
        env = self.env
        return env.reset_state(jax.random.PRNGKey(seed))._replace(
            ball_state=jnp.int32(BALL_ALIVE), restart_t=jnp.int32(0),
            restart_kind=jnp.int32(RK_NONE), pending_taker=jnp.int32(NO_PLAYER),
            poss_team=jnp.int32(TEAM_0), last_touch_team=jnp.int32(TEAM_0),
            last_touch_code=jnp.int32(TOUCH_PASS), restart_team=jnp.int32(TEAM_0),
            throw_taker=jnp.int32(NO_PLAYER), setpiece_taker=jnp.int32(NO_PLAYER),
            restart_indirect=jnp.bool_(False),
        )

    def test_whole_ball_must_cross_goal_line(self):
        env = self.env
        line = env.hx + env.r_ball
        base = self.open_state()._replace(ball_vel=jnp.array([8.0, 0.0, 0.0]))
        before = base._replace(ball_pos=jnp.array([line - 0.01, 0.0, env.r_ball]))
        after = base._replace(ball_pos=jnp.array([line + 0.01, 0.0, env.r_ball]))
        _, score_before = env._events(before, jax.random.PRNGKey(31))
        _, score_after = env._events(after, jax.random.PRNGKey(31))
        self.assertEqual((int(score_before), int(score_after)), (NO_TEAM, TEAM_0))

    def test_direct_goal_rules_distinguish_throw_setpiece_and_indirect(self):
        env = self.env
        line = env.hx + env.r_ball
        goal = self.open_state(32)._replace(
            ball_pos=jnp.array([line + 0.05, 0.0, env.r_ball]),
            ball_vel=jnp.array([8.0, 0.0, 0.0]))
        throw, st = env._events(goal._replace(throw_taker=jnp.int32(0)), jax.random.PRNGKey(32))
        direct, sd = env._events(goal._replace(setpiece_taker=jnp.int32(0)), jax.random.PRNGKey(32))
        indirect, si = env._events(goal._replace(
            setpiece_taker=jnp.int32(0), restart_indirect=jnp.bool_(True)), jax.random.PRNGKey(32))
        self.assertEqual((int(st), int(throw.restart_kind)), (NO_TEAM, RK_GOALKICK))
        self.assertEqual((int(sd), int(direct.score[0])), (TEAM_0, 1))
        self.assertEqual((int(si), int(indirect.restart_kind)), (NO_TEAM, RK_GOALKICK))

    def test_penalty_defender_encroachment_restarts_penalty(self):
        env = self.env
        line = env.hx + env.r_ball
        state = self.open_state(33)._replace(
            ball_pos=jnp.array([line + 0.05, env.goal_w, env.r_ball]),
            ball_vel=jnp.array([8.0, 0.0, 0.0]), penalty_flight_team=jnp.int32(TEAM_0),
            penalty_encroach_mask=jnp.zeros(env.N, bool).at[env.n_agents].set(True))
        next_state, _ = env._events(state, jax.random.PRNGKey(33))
        self.assertEqual((int(next_state.restart_kind), int(next_state.restart_team)),
                         (RK_PENALTY, TEAM_0))

    def test_offside_requires_a_new_touch(self):
        env = self.env
        state = self.open_state(34)._replace(
            pass_team=jnp.int32(TEAM_0), pass_t=jnp.int32(10),
            offside_flag=jnp.zeros(env.N, bool).at[1].set(True),
            touch=jnp.zeros(env.N, jnp.int32).at[1].set(TOUCH_PASS))
        before = jnp.zeros(env.N, jnp.int32)
        called = env._offside_check(state, before)
        stale = env._offside_check(state, state.touch)
        self.assertEqual(int(called.restart_kind), RK_OFFSIDE)
        self.assertTrue(bool(called.restart_indirect))
        self.assertEqual(int(stale.restart_kind), RK_NONE)

    def test_retouch_law_calls_taker_and_clears_after_other_touch(self):
        env = self.env
        before = jnp.zeros(env.N, jnp.int32)
        base = self.open_state(35)._replace(throw_taker=jnp.int32(0))
        self_touch = base._replace(touch=before.at[0].set(TOUCH_PASS))
        foul = env._throwin_restriction(self_touch, jnp.int32(0), jnp.int32(NO_PLAYER), before)
        other_touch = base._replace(touch=before.at[1].set(TOUCH_PASS))
        clear = env._throwin_restriction(other_touch, jnp.int32(0), jnp.int32(NO_PLAYER), before)
        self.assertEqual((int(foul.restart_kind), int(foul.restart_team), bool(foul.restart_indirect)),
                         (RK_FREEKICK, TEAM_1, True))
        self.assertEqual((int(clear.throw_taker), int(clear.restart_kind)), (NO_PLAYER, RK_NONE))

    def test_no_active_players_remains_finite_and_foul_free(self):
        env = self.env
        state = self.open_state(36)._replace(active_player=jnp.zeros(env.N, bool))
        obs, _, _, _, info = env.step_env_array(
            jax.random.PRNGKey(36), state, jnp.zeros((env.N, ACTION_DIM)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(obs))))
        self.assertEqual(int(info["foul_kind"]), FOUL_NONE)

    def test_deadball_agency_keeps_non_kicker_movement(self):
        env = self.env
        state = env.reset_state(jax.random.PRNGKey(37))
        agency = env.action_agency(state)
        taker = int(state.pending_taker)
        non_kicker = np.arange(env.N) != taker
        self.assertTrue(bool(np.asarray(~agency["move_forced"])[non_kicker].all()))
        self.assertTrue(bool(np.asarray(agency["kick_gated"])[non_kicker].all()))

    def test_realized_kick_opens_bc_kick_dimensions(self):
        env = self.env
        info = {
            "move_forced": jnp.zeros(env.N, bool),
            "kick_gated": jnp.ones(env.N, bool),
            "kick_forced": jnp.zeros(env.N, bool),
        }
        applied = jnp.zeros(env.N, bool).at[0].set(True)
        mask = env.bc_action_mask(info, kick_applied=applied)
        kick_dims = [0, 3, 4, 5, 6, 7]
        self.assertTrue(bool(jnp.all(mask[0, jnp.array(kick_dims)])))
        self.assertFalse(bool(jnp.any(mask[1, jnp.array(kick_dims)])))


class SymmetryContract(unittest.TestCase):
    def test_one_step_is_equivariant_under_180_degree_rotation(self):
        """정책 프레임 행동은 그대로 두고 월드 상태만 180° 돌리면 관측·다음 상태가 같아야 한다."""
        env = SoccerEnv(game_duration=300)
        state = env.reset_state(jax.random.PRNGKey(40))._replace(
            ball_state=jnp.int32(BALL_ALIVE), restart_t=jnp.int32(0),
            restart_kind=jnp.int32(RK_NONE), pending_taker=jnp.int32(NO_PLAYER),
            ball_pos=jnp.array([4.0, -3.0, env.r_ball]),
            ball_vel=jnp.array([2.0, -1.0, 0.0]), ball_spin=jnp.array([1.0, -2.0, 3.0]))

        def rotate(s):
            ball_pos = s.ball_pos.at[:2].set(-s.ball_pos[:2])
            ball_vel = s.ball_vel.at[:2].set(-s.ball_vel[:2])
            ball_spin = s.ball_spin.at[:2].set(-s.ball_spin[:2])
            player_vel = -s.player_vel
            attack_dir = -s.attack_dir
            return s._replace(
                ball_pos=ball_pos, ball_vel=ball_vel, ball_spin=ball_spin,
                player_pos=-s.player_pos, player_vel=player_vel, attack_dir=attack_dir,
                player_facing=env.facing_from_velocity(player_vel, attack_dir))

        rotated = rotate(state)
        action = jax.random.uniform(
            jax.random.PRNGKey(41), (env.N, ACTION_DIM), minval=-1.0, maxval=1.0)
        kwargs = dict(suppress_charge=True, suppress_body=True,
                      suppress_retake=True, suppress_restart=True)
        obs_a, next_a, *_ = env.step_env_array(jax.random.PRNGKey(42), state, action, **kwargs)
        obs_b, next_b, *_ = env.step_env_array(jax.random.PRNGKey(42), rotated, action, **kwargs)
        np.testing.assert_allclose(np.asarray(obs_a), np.asarray(obs_b), atol=1e-5)
        rotated_next = rotate(next_a)
        for name in next_a._fields:
            a, b = np.asarray(getattr(rotated_next, name)), np.asarray(getattr(next_b, name))
            if a.dtype.kind in "biu":
                np.testing.assert_array_equal(a, b, err_msg=name)
            else:
                np.testing.assert_allclose(a, b, atol=1e-5, err_msg=name)


if __name__ == "__main__":
    unittest.main()
