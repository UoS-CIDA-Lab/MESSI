"""배치 무작위 롤아웃에서 환경의 장기 불변식을 검사한다.

``test_environment.py``가 희귀 반례를 결정적으로 고정한다면, 이 스크립트는 AAMAS2027의
rollout 계약 방식을 발전시켜 여러 환경·시간축에서 수치 오염과 단조성 위반을 찾는다.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax
import jax.numpy as jnp

from constants import *
from env import SoccerEnv


LABELS = (
    "nonfinite", "player_bounds", "ball_below_ground", "speed_cap", "stamina_range",
    "facing_cache", "inactive_observation", "inactive_velocity", "enum_range",
    "negative_counters", "score_decrease", "card_decrease", "player_reactivated",
    "kick_applied_but_bc_mask_closed", "outward_boundary_velocity",
)


def audit(env: SoccerEnv, envs: int, steps: int, seed: int):
    roots = jax.random.split(jax.random.PRNGKey(seed), envs)
    states = jax.vmap(env.reset_state)(roots)
    vstep = jax.vmap(lambda k, s, a: env.step_env_array(k, s, a))
    vfacing = jax.vmap(env.facing_from_velocity)

    def body(carry, step_index):
        state, keys, counts, maxima = carry
        split = jax.vmap(jax.random.split)(keys)
        keys, action_keys = split[:, 0], split[:, 1]
        action = jax.vmap(lambda k: jax.random.uniform(
            k, (env.N, ACTION_DIM), minval=-1.25, maxval=1.25))(action_keys)
        action = action.at[:, 0, ACTION_MOVE.start].set(
            jnp.where(step_index % 79 == 0, jnp.nan, action[:, 0, ACTION_MOVE.start]))
        obs, next_state, _, _, info = vstep(keys, state, action)

        finite = jnp.all(jnp.isfinite(obs)) & jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(next_state)]))
        x_over = jnp.max(jnp.abs(next_state.player_pos[..., DIM_X]) - env.hx)
        y_over = jnp.max(jnp.abs(next_state.player_pos[..., DIM_Y]) - env.hy)
        pos_over = jnp.maximum(x_over, y_over)
        ground_under = jnp.max(env.r_ball - next_state.ball_pos[..., DIM_Z])
        speed_over = jnp.max(jnp.linalg.norm(next_state.player_vel, axis=-1) - next_state.vmax)
        stamina_bad = jnp.any((next_state.stamina < -1e-6) | (next_state.stamina > 1.0 + 1e-6))
        face_expected = vfacing(next_state.player_vel, next_state.attack_dir)
        face_error = jnp.max(jnp.abs(jnp.arctan2(
            jnp.sin(next_state.player_facing - face_expected),
            jnp.cos(next_state.player_facing - face_expected))))
        inactive_obs = jnp.max(jnp.abs(jnp.where(
            (~next_state.active_player)[..., None], obs, 0.0)))
        inactive_vel = jnp.max(jnp.abs(jnp.where(
            (~next_state.active_player)[..., None], next_state.player_vel, 0.0)))
        enum_bad = (
            jnp.any((next_state.ball_state < BALL_DEAD) | (next_state.ball_state > BALL_ALIVE))
            | jnp.any((next_state.restart_kind < RK_NONE) | (next_state.restart_kind >= RESTART_COUNT))
            | jnp.any((next_state.touch < TOUCH_NONE) | (next_state.touch >= TOUCH_COUNT))
            | jnp.any((next_state.foul_kind < FOUL_NONE) | (next_state.foul_kind > FOUL_THROW))
            | jnp.any((next_state.poss_team < NO_TEAM) | (next_state.poss_team > TEAM_1))
            | jnp.any((next_state.pending_taker < NO_PLAYER) | (next_state.pending_taker >= env.N)))
        negative_counter = (jnp.any(next_state.cooldown < 0) | jnp.any(next_state.ctrl_lock_t < 0)
                            | jnp.any(next_state.restart_t < 0))
        kick_mask_bad = jnp.any(info["kick_applied"] & (~info["bc_action_mask"][..., 0]))
        px, py = next_state.player_pos[..., 0], next_state.player_pos[..., 1]
        vx, vy = next_state.player_vel[..., 0], next_state.player_vel[..., 1]
        outward = (((px >= env.hx - GEOMETRY_EPS) & (vx > 1e-6))
                   | ((px <= -env.hx + GEOMETRY_EPS) & (vx < -1e-6))
                   | ((py >= env.hy - GEOMETRY_EPS) & (vy > 1e-6))
                   | ((py <= -env.hy + GEOMETRY_EPS) & (vy < -1e-6)))

        flags = jnp.asarray([
            ~finite, pos_over > 1e-5, ground_under > 1e-5, speed_over > 1e-4,
            stamina_bad, face_error > 1e-5, inactive_obs > 1e-6, inactive_vel > 1e-6,
            enum_bad, negative_counter, jnp.any(next_state.score < state.score),
            jnp.any(next_state.yellow_cards < state.yellow_cards),
            jnp.any(next_state.active_player & (~state.active_player)), kick_mask_bad,
            jnp.any(outward),
        ], dtype=jnp.int32)
        maxima = jnp.maximum(maxima, jnp.asarray([
            jnp.maximum(pos_over, 0.0), jnp.maximum(ground_under, 0.0),
            jnp.maximum(speed_over, 0.0), face_error,
        ]))
        return (next_state, keys, counts + flags, maxima), None

    (states, _, counts, maxima), _ = jax.lax.scan(
        body, (states, roots, jnp.zeros(len(LABELS), jnp.int32), jnp.zeros(4)),
        jnp.arange(steps))
    violations = {name: int(value) for name, value in zip(LABELS, counts) if int(value)}
    return {
        "envs": envs, "steps": steps, "transitions": envs * steps,
        "violations": violations,
        "maxima": {
            "position_overrun_m": float(maxima[0]),
            "ball_ground_underrun_m": float(maxima[1]),
            "speed_over_vmax_mps": float(maxima[2]),
            "facing_error_rad": float(maxima[3]),
        },
        "final_score_sum": [int(x) for x in jnp.sum(states.score, axis=0)],
        "passed": not violations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    if args.envs <= 0 or args.steps <= 0:
        parser.error("--envs and --steps must be positive")
    env = SoccerEnv(game_duration=max(args.steps + 1, 1000))
    result = audit(env, args.envs, args.steps, args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
