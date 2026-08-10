"""보상 — sparse/dense 전환(Reward.mode, jit 정적). dense는 sparse(골)를 포함하고 전진 셰이핑·
소유획득 보너스를 더한다. 전진 셰이핑은 potential-based F=γ·Φ(s′)−Φ(s)(Ng 1999)로 shaping_gamma가
트레이너 할인율과 일치하고 Φ 연속일 때 무편향. 인플레이(재개·득점 스텝 제외) 게이팅은 Φ 불연속
구간을 잘라내는 실용적 근사다.
"""
import jax.numpy as jnp

from constants import *


class Rewards:
    def _reward_array(self, state, scored, ball_x0, poss0):
        """(N,) per-player 보상. state=스텝 후, scored=스캔 누적 득점팀(-1=없음),
        ball_x0/poss0=스텝 '전' 공 x·소유팀(dense 차분 기준)."""
        r_cfg = self.r_cfg
        scored_any = scored >= 0
        goal_r = jnp.where(state.team_id == scored, r_cfg.goal, -r_cfg.goal) * scored_any

        if r_cfg.mode == "sparse":
            return goal_r
        elif r_cfg.mode == "dense":
            inplay = (~scored_any) & (state.restart_t == 0)
            attack_dir = state.attack_dir
            phi_after = state.ball_pos[DIM_X] * attack_dir / self.hx
            phi_before = ball_x0 * attack_dir / self.hx
            advance_r = r_cfg.advance * (r_cfg.shaping_gamma * phi_after - phi_before) * inplay
            changed = (state.poss_team != poss0) & (state.poss_team >= 0)
            mine = changed & (state.team_id == state.poss_team)
            theirs = changed & (state.team_id != state.poss_team)
            poss_r = r_cfg.poss_gain * (mine.astype(jnp.float32) - theirs.astype(jnp.float32)) * inplay
            return goal_r + advance_r + poss_r
        else:
            raise ValueError(f"reward.mode must be 'sparse' or 'dense', got {r_cfg.mode!r}")
