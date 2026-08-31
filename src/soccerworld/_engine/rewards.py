"""보상 — sparse/dense 전환(Reward.mode, jit 정적). dense는 sparse(골)를 포함하고 전진 셰이핑·
소유획득 보너스를 더한다. 전진 셰이핑은 potential-based F=γ·Φ(s′)−Φ(s)(Ng 1999)로 shaping_gamma가
트레이너 할인율과 일치하고 Φ 연속일 때 무편향. 인플레이(재개·득점 스텝 제외) 게이팅은 Φ 불연속
구간을 잘라내는 실용적 근사다.
"""
import jax.numpy as jnp

from .constants import BALL_ALIVE, DIM_X
from .restart import restart_timer_active


class Rewards:
    def _reward_array(self, state, scored, ball_x0, poss0, entry_inplay,
                      transition_done):
        """(N,) per-player 보상. state=스텝 후, scored=스캔 누적 득점팀(-1=없음),
        ball_x0/poss0=스텝 '전' 공 x·소유팀(dense 차분 기준),
        entry_inplay=스텝 진입 시 공이 라이브이고 재개 타이머가 비활성이었는지 여부,
        transition_done=이번 전이로 시간제한/인원미달 종료에 도달했는지 여부.

        dense 셰이핑은 보통 **진입과 종료가 모두 인플레이인 프레임**에만 준다. 단, 라이브
        진입에서 episode가 끝난 전이는 후계 potential을 0으로 두고 ``-Phi(entry)``를 내어
        합을 닫는다. 종료 상태의 ``restart_t``만 보면, 진입 때 데드볼이던 재개가 같은 control
        frame에서 소비된 경우 ``restart_t=0``이 되어 재개 킥의 행정적 공 이동에 전진·소유
        보상이 새기 때문이다.
        """
        r_cfg = self.r_cfg
        scored_any = scored >= 0
        goal_r = jnp.where(state.team_id == scored, r_cfg.goal, -r_cfg.goal) * scored_any

        if r_cfg.mode == "sparse":
            return goal_r
        elif r_cfg.mode == "dense":
            post_inplay = ((state.ball_state == BALL_ALIVE)
                           & (~restart_timer_active(state.restart_t)))
            # A normal referee stoppage is excluded because its administrative
            # placement is not policy progress.  A terminal stoppage is
            # different: the episodic successor potential is defined as zero,
            # so an entry-live transition must still emit ``-Phi(entry)`` even
            # when the terminating red-card foul leaves a dead ball.  Requiring
            # ``post_inplay`` here used to erase that closure on the real 7->6
            # abandonment path while keeping it on a live time-limit path.
            inplay = (
                (~scored_any)
                & entry_inplay
                & (post_inplay | transition_done)
            )
            attack_dir = state.attack_dir
            phi_after = state.ball_pos[DIM_X] * attack_dir / self.hx
            phi_before = ball_x0 * attack_dir / self.hx
            # Public ``done`` is an episodic terminal contract: the returned state is
            # subsequently frozen and generic consumers do not bootstrap through it.
            # Therefore the terminal potential is exactly zero.  Merely suppressing
            # the whole term would leave ``Phi(s_before)`` uncancelled; keeping the
            # subtraction closes the potential sum at the episode boundary.
            phi_after = jnp.where(transition_done, 0.0, phi_after)
            advance_r = r_cfg.advance * (r_cfg.shaping_gamma * phi_after - phi_before) * inplay
            # A possession bonus is a turnover reward, so both sides of the
            # transition must name a team.  In particular, A -> neutral -> A
            # must not pay A again on the neutral -> A self-recovery step.
            changed = ((state.poss_team != poss0)
                       & (state.poss_team >= 0)
                       & (poss0 >= 0))
            mine = changed & (state.team_id == state.poss_team)
            theirs = changed & (state.team_id != state.poss_team)
            # Possession gain is an event bonus, not a potential.  Do not emit it on
            # the terminal transition, where there is no subsequent possession to
            # exploit and no bootstrap continuation.
            poss_inplay = inplay & (~transition_done)
            poss_r = (
                r_cfg.poss_gain
                * (mine.astype(jnp.float32) - theirs.astype(jnp.float32))
                * poss_inplay
            )
            return goal_r + advance_r + poss_r
        else:
            raise ValueError(f"reward.mode must be 'sparse' or 'dense', got {r_cfg.mode!r}")
