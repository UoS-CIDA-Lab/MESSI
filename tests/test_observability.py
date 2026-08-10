"""관측 aliasing 자동 검출 — "같은 obs / 다른 전이"를 기계적으로 찾는다.

지금까지 발견된 관측 결함(own_vel · facing · pass_t · penalty encroach latch · kickoff_team ·
restart_indirect · 스로인 vs 일반 세트피스 재터치)은 **전부 같은 패턴**이었다.

    두 State의 obs가 완전히 같은데, 같은 액션·같은 RNG로 한 스텝 굴리면 물리 결과가 갈린다.

정책은 obs만 보므로 그 차이를 원리상 설명할 수 없고, BC는 그것을 줄일 수 없는 라벨 노이즈로 받는다.
데이터를 늘려서 해결되지 않는다 — 관측 계약을 고쳐야 한다.

이 테스트는 State 필드를 하나씩 흔들어(perturb) 세 갈래로 분류한다.

    OBSERVED — obs가 달라진다                      → 정상(관측되고 있음)
    INERT    — obs도 전이도 같다                    → 정상(전이에 무관)
    ALIASED  — obs는 같은데 전이가 갈린다            → **결함**

`../AUDIT.md` 참조. 새 State 필드나 규칙을 추가하면 여기에 probe를 추가한다.
"""
from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

from env import SoccerEnv
from constants import *


# 전이가 "갈렸다"고 볼 물리·규칙 결과. foul_actor처럼 자기 자신만 다른 provenance 필드는
# 제외한다 — 그건 aliasing이 아니라 그냥 그 값이 흘러간 것이다.
CONSEQUENTIAL = (
    "ball_pos", "ball_vel", "ball_spin", "ball_state",
    "player_pos", "player_vel", "stamina",
    "poss_team", "score", "active_player", "yellow_cards",
    "restart_kind", "restart_t", "restart_team", "pending_taker",
)

# 누적 카운터는 **절대값이 아니라 이 스텝의 증분**으로 비교한다. perturbation이 상수 오프셋을
# 주입하면(예: 양 팀 score +3, 득실차 동일) 그 오프셋이 다음 상태로 그대로 흘러가는데, 그건
# aliasing이 아니라 주입값의 전달이다. 실제로 물어야 할 것은 "이 스텝에 골이 났는가"다.
# ※득실차가 같은 절대 스코어가 obs에서 동일한 것은 **설계**다(obs는 score_diff만 준다).
DELTA_COMPARED = frozenset({"score", "yellow_cards"})


def _perturbations():
    """(이름, State→State) — 물리적으로 성립 가능한 대안 상태를 만든다."""
    def swap_taker_kind(s):
        """재터치 제한의 출처만 스로인 ↔ 일반 세트피스로 바꾼다(직접골 규칙이 다르다)."""
        who = jnp.maximum(s.throw_taker, s.setpiece_taker)
        return s._replace(throw_taker=s.setpiece_taker.astype(jnp.int32) * 0 + who,
                          setpiece_taker=jnp.int32(NO_PLAYER))

    def swap_taker_kind_back(s):
        who = jnp.maximum(s.throw_taker, s.setpiece_taker)
        return s._replace(setpiece_taker=s.throw_taker.astype(jnp.int32) * 0 + who,
                          throw_taker=jnp.int32(NO_PLAYER))

    return [
        ("kickoff_team", lambda s: s._replace(kickoff_team=(TEAM_1 - s.kickoff_team).astype(jnp.int32))),
        ("restart_indirect", lambda s: s._replace(restart_indirect=~s.restart_indirect)),
        ("pass_t", lambda s: s._replace(pass_t=jnp.int32(120), pass_team=jnp.int32(TEAM_0))),
        ("pass_team", lambda s: s._replace(pass_team=jnp.int32(TEAM_1), pass_t=jnp.int32(120))),
        ("offside_flag", lambda s: s._replace(
            offside_flag=jnp.zeros_like(s.offside_flag).at[8].set(True),
            pass_t=jnp.int32(120), pass_team=jnp.int32(TEAM_0))),
        ("penalty_encroach_mask", lambda s: s._replace(
            penalty_encroach_mask=jnp.zeros_like(s.penalty_encroach_mask).at[3].set(True))),
        ("penalty_flight_team", lambda s: s._replace(penalty_flight_team=jnp.int32(TEAM_0))),
        ("player_vel", lambda s: s._replace(player_vel=s.player_vel + 1.5)),
        ("player_facing", lambda s: s._replace(player_facing=s.player_facing + 1.7)),
        ("cooldown", lambda s: s._replace(cooldown=s.cooldown + 5.0)),
        ("ctrl_lock_t", lambda s: s._replace(ctrl_lock_t=s.ctrl_lock_t + 5)),
        ("stamina", lambda s: s._replace(stamina=s.stamina * 0.5)),
        ("yellow_cards", lambda s: s._replace(yellow_cards=s.yellow_cards.at[4].set(1))),
        ("last_touch_team", lambda s: s._replace(last_touch_team=(TEAM_1 - s.last_touch_team).astype(jnp.int32))),
        ("last_touch_code", lambda s: s._replace(last_touch_code=jnp.int32(TOUCH_PASS))),
        ("touch", lambda s: s._replace(touch=jnp.zeros_like(s.touch).at[6].set(TOUCH_PASS))),
        ("foul_actor", lambda s: s._replace(foul_actor=jnp.int32(7))),
        ("setpiece_taker→throw_taker", swap_taker_kind),
        ("throw_taker→setpiece_taker", swap_taker_kind_back),
        ("score(동일 득실차)", lambda s: s._replace(score=s.score + 3)),
    ]


class TestObservationAliasing(unittest.TestCase):
    """State 필드를 흔들어 '같은 obs / 다른 전이'를 찾는다."""

    @classmethod
    def setUpClass(cls):
        cls.env = SoccerEnv(game_duration=3000, control_fps=25)
        cls.N = cls.env.N
        # 국면이 섞이도록 룰 정책으로 굴려 여러 시점의 State를 모은다.
        from policy import make_rule_based_policy
        pol = make_rule_based_policy(cls.env, match_key=jax.random.PRNGKey(0),
                                     team_styles=("gegenpress", "tiki_taka"))
        step = jax.jit(lambda st, o, k: (lambda kp, ke:
                                         cls.env.step_env_array(ke, st, pol(o, kp)))(*jax.random.split(k)))
        obs, s = cls.env.reset_array(jax.random.PRNGKey(1))
        cls.bases = []
        for i in range(240):
            o2, s2, *_ = step(s, obs, jax.random.PRNGKey(i))
            obs, s = o2, s2
            if i % 20 == 0:
                cls.bases.append(s)
        cls.act = jnp.zeros((cls.N, ACTION_DIM)).at[:, 1].set(0.7).at[:, 0].set(1.0)
        # staticmethod로 감싸지 않으면 self가 첫 인자로 묶여 jit이 클래스 인스턴스를 배열로 해석한다.
        cls.jstep = staticmethod(jax.jit(lambda st, a, k: cls.env.step_env_array(k, st, a)[1]))

    def _classify(self, base, mutate):
        """한 base State에서 perturbation을 분류한다."""
        try:
            var = mutate(base)
        except Exception as exc:                                  # 성립 불가한 조합은 건너뛴다
            return "SKIP", str(exc)
        o0, o1 = self.env.get_obs_array(base), self.env.get_obs_array(var)
        obs_same = bool(jnp.array_equal(o0, o1))
        if not obs_same:
            return "OBSERVED", None
        k = jax.random.PRNGKey(12345)
        n0 = self.jstep(base, self.act, k)
        n1 = self.jstep(var, self.act, k)
        for f in CONSEQUENTIAL:
            a, b = getattr(n0, f), getattr(n1, f)
            if f in DELTA_COMPARED:                               # 증분 비교(위 주석 참조)
                a = a - getattr(base, f)
                b = b - getattr(var, f)
            if not bool(jnp.array_equal(a, b)):
                return "ALIASED", f
        return "INERT", None

    def test_no_aliasing(self):
        rows, aliased = [], []
        for name, mut in _perturbations():
            verdicts = [self._classify(b, mut) for b in self.bases]
            kinds = [v[0] for v in verdicts]
            bad = [v for v in verdicts if v[0] == "ALIASED"]
            verdict = ("ALIASED" if bad else
                       "OBSERVED" if "OBSERVED" in kinds else
                       "INERT" if "INERT" in kinds else "SKIP")
            detail = f" (전이가 갈린 필드: {sorted({v[1] for v in bad})})" if bad else ""
            rows.append(f"  {verdict:<9} {name}{detail}"
                        f"   [OBSERVED {kinds.count('OBSERVED')} / INERT {kinds.count('INERT')}"
                        f" / ALIASED {len(bad)} of {len(kinds)}]")
            if bad:
                aliased.append(f"{name}{detail}")
        print(f"\n관측 aliasing 스윕 — base State {len(self.bases)}개 × probe {len(rows)}종")
        print("\n".join(rows))
        self.assertFalse(
            aliased,
            "같은 obs인데 전이가 갈리는 필드가 있다(관측 계약 결함):\n  " + "\n  ".join(aliased))


if __name__ == "__main__":
    unittest.main(verbosity=2)
