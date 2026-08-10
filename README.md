# UOS-FootballMARL-Env

**JAX 위에서 도는 11대11 축구 멀티에이전트 강화학습 환경.** 순수 함수형이라 `jit`/`vmap`으로
수천 경기를 동시에 굴릴 수 있고, IFAB 경기 규칙(오프사이드·세트피스·페널티·카드/퇴장)을
결정적 상태 기계로 구현한다.

일반적인 축구 시뮬레이터와 다른 점은 **두 가지를 1급 요구사항으로 두고 설계했다는 것**이다.

1. **BC(모방학습) 데이터 생성** — 어떤 프레임이 정책의 자유 결정이고 어떤 프레임이 환경이
   강제한 것인지를 환경이 스스로 라벨로 내보낸다(§ [행동 강제성 라벨](#5-bc-라벨-계약)).
   강제 프레임을 정책 결정처럼 학습하면 라벨 노이즈가 되기 때문이다.
2. **실경기 트래킹 복원(reconstruct)** — 모든 접촉 분기가 '결정적 propensity + 단일 난수 draw'
   구조라, 관측된 궤적에서 그것을 만든 액션·난수를 **역산**할 수 있다(§ [역동역학](#6-역동역학)).

---

## 1. 저장소 구성

```
.
├── demo_match.py          # 룰 정책 두 팀 경기 → 요약 + MP4 렌더 (진입점)
├── env/                   # 환경 본체 (19개 모듈, flat import)
│   ├── README.md          # ★ obs/action/state·API·규칙의 전체 스펙
│   ├── guide.md           # 설계 근거 — 왜 이렇게 만들었는가
│   ├── AUDIT.md           # 감사 이력 — 수정된 결함·성능 측정·남은 한계
│   ├── env.py             # SoccerEnv (mixin 합성) · step 본체 · BC 라벨 계약
│   ├── state.py config.py constants.py setup.py spatial.py
│   ├── observation.py     # obs/central-state 조립 · 스펙(단일 진실원천)
│   ├── movement.py ball.py contest.py            # 물리
│   ├── restart.py events.py offside.py fouls.py  # 규칙
│   ├── rewards.py inverse.py                     # 보상 · 역동역학
│   ├── policy.py tactics.py                      # 룰 정책(학습 정책 아님) · 전술 지표
│   └── render.py          # MP4 렌더 (light: numpy / rich: matplotlib 3D)
└── tests/                 # 계약 회귀 28개 · 관측 aliasing 스윕 · 장기 불변식 15개
```

문서는 역할이 나뉘어 있다. **무엇이 어떤 값인지**는 `env/README.md`, **왜 그렇게 정했는지**는
`env/guide.md`, **무엇이 틀렸었고 어떻게 고쳤는지**는 `env/AUDIT.md`.

> 하위 문서·주석에 나오는 **SoccerBC**는 이 환경의 내부 개발 코드명이고, **AAMAS2027**은
> DFL 트래킹 캘리브 값을 물려준 부모 프로젝트다. 두 이름이 가리키는 `recon/`·`bc/`·`calib/`
> 디렉터리는 본 저장소에 포함되지 않는다(원본 트래킹 데이터에 결합돼 있다).

---

## 2. 설치와 실행

Python 3.11 · JAX 0.4.38 기준으로 개발·검증했다.

```bash
pip install jax jaxlib jaxmarl jaxtyping numpy      # 코어
pip install imageio imageio-ffmpeg pillow           # light 렌더(MP4)
pip install matplotlib                              # rich 렌더(3D, 선택)
```

`env/`의 모듈들은 **flat import**(`from constants import *`)를 쓰므로 `env` 디렉터리가
`sys.path`에 있어야 한다. 아래 명령은 전부 **저장소 루트**에서 실행한다.

```bash
python demo_match.py                                       # 60초 경기 → ./replays/<unix초>/match.mp4
python demo_match.py --home tiki_taka --away long_ball --seconds 120
python demo_match.py --mode rich --render-fps 100          # 방송형 3D, 물리 100 Hz 밀도
python demo_match.py --no-render                           # 롤아웃·요약만
python demo_match.py --list                                # 팀 스타일 프리셋 목록
```

`demo_match.py`는 환경 밖에서 공개 인터페이스(`obs → action → step_env_array`)만 쓰고
롤아웃 전체를 `lax.scan` 하나로 JIT 컴파일한다 — 학습 코드가 참고할 최소 예제이기도 하다.

### 최소 사용 예

```python
import jax, jax.numpy as jnp
from env import SoccerEnv                    # PYTHONPATH=env

env = SoccerEnv(game_duration=3000, control_fps=25)      # 11v11, 25 Hz 컨트롤
key = jax.random.PRNGKey(0)
obs, state = env.reset_array(key)                        # obs (22, 470)

action = jnp.zeros((env.N, env.action_dim))              # 전 차원 [-1, 1]
action = action.at[:, 1].set(0.7)                        # 이동: 자기 공격 방향 +x
obs, state, reward, done, info = env.step_env_array(key, state, action)

print(env.obs_dim, env.state_dim, env.action_dim)        # 470 538 8
print(info["bc_action_mask"].shape)                      # (22, 8) — BC 손실 마스크
```

배열 경로(`reset_array`/`step_env_array`/`get_obs_array`)가 단일 진실원천이고, JaxMARL 규약의
dict 어댑터(`reset`/`step_env`/`get_obs`)는 편의용이다. **핫루프에서는 배열 경로 + `jit`/`vmap`을
쓸 것** — dict 조립은 순수 파이썬 오버헤드다.

---

## 3. 환경 한눈에

| 항목 | 값 |
|---|---|
| 선수 수 `N` | 22 (11 vs 11, 팀당 GK 1) |
| 관측 `obs_dim` | **470** — self 18 / others 19×21 / ball 14 / context 39 |
| 중앙 상태 `state_dim` | **538** — 중앙집중 크리틱용 절대 프레임 |
| 행동 `action_dim` | **8** — `[kick_gate, move(2), f2b(2), launch, spin_side, spin_back]` |
| 물리 스텝 | 100 Hz 고정 (`dt_phys = 0.01 s`) |
| 컨트롤 스텝 | 25 Hz 기본 (`decimation = 4`, `control_dt = 0.04 s`) |
| 필드 | 105 × 68 m, 원점 중심. 골 7.32 × 2.44 m |
| 종료 조건 | 시간제한뿐 (`t ≥ game_duration`) |
| 스키마 버전 | `{action: 1, observation: 6, state: 3}` |

**관측·행동은 자기 공격 프레임으로 접힌다**(`× attack_dir`). 회전 규약은 거울이 아니라
**180° 점대칭**이라, 스핀 z성분이나 터치 코드 같은 유사벡터·범주값은 접지 않는다.
인덱스의 단일 진실원천은 `env.obs_spec()` / `env.state_spec()`이며 **하드코딩 금지**다.

### 스텝 파이프라인

```
액션 디코드 (컨트롤 스텝당 1회)
└─ lax.scan × decimation(=4):
     키커 강제이동 → 이동 적분 → 차징 파울 → 킥 게이트 → 경합 승자(Gumbel argmax)
     → 킥/탈취/파울/굴절/GK 클레임 → 공-몸통 충돌 → 재터치 금지 → 오프사이드
     → 공 자유물리 → 이벤트(득점/아웃/페널티 해소) → 쿨다운 감쇠
t += 1 → 하프타임 전환 검사 → 보상 → 관측
```

성능 스위치: `include_bc_info=False`(BC 라벨 생략, RL 핫루프) ·
`compute_observation=False`(O(N²) 관측 조립 생략, 물리 검증·렌더) · `collect_substeps=True`
(100 Hz 서브스텝 스택, 렌더 밀도용).

---

## 4. 핵심 설계 결정

전체 근거는 `env/guide.md`에 있고, 요약하면 네 가지다.

**① 이동은 가속이 아니라 목표 속도 명령이다.** 트래킹 복원 라벨에서 속도 타깃은 매끄럽지만
(방향 지터 ~2°) 가속 타깃은 2차 미분 노이즈 증폭으로 ~69°까지 튀어 BC 라벨의 SNR이 무너진다.
정책은 '가고 싶은 속도'만 내고, 환경이 **마찰 타원**(종 가속 8.68 / 종 감속 10.51 / 횡 가속
8.46 m/s², DFL 실측 p99.9) 안으로 방사 투영해 도달 가능성을 강제한다.

**② facing은 상태가 아니라 속도의 결정함수다.** 독립 적분하면 관측으로 복원할 수 없는 은닉
상태가 되고, 그 은닉분이 파울 로짓을 통해 실제 전이를 바꿔 BC가 설명할 수 없는 라벨 노이즈가
된다. `state.player_facing`은 캐시일 뿐이며, 회전율 상한은 별도 계수가 아니라 마찰 타원의
횡가속 캡이 만든다.

**③ 킥 '타이밍'은 환경이, 킥 '파라미터'는 정책이 소유한다.** 세트피스는 세트업 완료 즉시
결정론적으로 발사된다 — '언제 찰지'라는 은닉 결정이 사라져 BC가 킥을 인과 라벨로 학습할 수
있고, 방향·파워·발사각·스핀은 여전히 정책 소유라 역산 가능하다.

**④ 확률 이벤트는 전부 '결정적 propensity + 단일 uniform draw'다.** 태클·파울·굴절·몸통 충돌·
카드가 모두 이 구조라 관측 결과에서 어느 분기·어떤 draw였는지 역산된다.

---

## 5. BC 라벨 계약

이 환경의 핵심 부가 계약. 목적은 **강제 프레임을 정책 결정처럼 학습하지 않게** 하는 것이다.

`env.action_agency(state)`가 스텝 **진입** 상태에서 결정론적으로 네 신호를 낸다.

| 신호 | 의미 |
|---|---|
| `move_forced` | 이동이 환경에 대체됨 — 세트피스 키커 강제 워킹 / 퇴장자 |
| `kick_gated` | 자발적 킥이 이번 스텝 효력 불가 — 미도달·쿨다운·비지정키커·데드볼 |
| `kick_forced` | 카운트다운 소진으로 킥이 강제 발사되는 프레임 |
| `halftime_reset` | 하프타임 전환 프레임 — 전 차원 하드 마스크 |

`info["bc_action_mask"]`는 이 신호를 8차원 액션 레이아웃에 매핑한 `(N, 8)` 손실 마스크다.
다만 **킥 차원은 진입 게이트만으로는 필요조건**일 뿐이다. 킥의 인과성은 스텝 내부의 확률적
경합에서 갈리므로(승자만 파라미터가 적용되고, 굴절·GK 캐치는 무효) 실현 신호와 AND 해야 한다.

```python
kick_label = info["bc_action_mask"][:, kick_dims] & info["kick_applied"][:, None]
```

vmax·스태미나·마찰 타원 같은 **물리 클립은 '강제'가 아니다** — 방향 의도가 반영되기 때문이다.

---

## 6. 역동역학

`inverse.py`가 관측된 접촉 결과에서 그것을 만든 액션 또는 난수를 되찾는다.

| 분기 | 자유도 | 함수 |
|---|---|---|
| 킥 (free play · 탈취) | 방향·파워·발사각·스핀 | `infer_kick_action` — 정확한 역함수 |
| 스로인 테이크 | 방향·파워·발사각 | `infer_throwin_action` |
| 굴절 | 랜덤 각도 draw 1개 | `infer_deflect` |
| GK parry | **0** (입사속도의 결정함수) | `predict_parry` (정방향 대조용) |
| 이동 | 목표속도 = 다음 속도 그 자체 | `infer_move_action` |

복원 시에는 관측된 경합 승자·분기를 주입하는 pin 인자들(`forced_winner`·`forced_freeplay`·
`suppress_*`)을 쓴다. **모두 기본값에서 포워드 불변**이고, 봉쇄된 추첨도 uniform은 그대로
소비해 RNG 열이 바뀌지 않는다.

---

## 7. 검증

```bash
PYTHONPATH=env python -m unittest discover -s tests -v              # 전체
PYTHONPATH=env python tests/audit_rollout.py --envs 64 --steps 500  # 장기 불변식
```

| 스위트 | 무엇을 지키나 |
|---|---|
| `tests/test_environment.py` | 계약 회귀 28개 — API 형상·스키마·FPS·config 격리·NaN 방어·facing 파생·관측 노출·경계 속도·이동/킥 역함수 왕복·IFAB 규칙·BC agency·**180° 회전 대칭** |
| `tests/test_observability.py` | **관측 aliasing 스윕** — State 필드를 흔들어 "obs는 같은데 전이가 갈리는" 조합을 기계적으로 찾는다 |
| `tests/audit_rollout.py` | 무작위 배치 롤아웃의 장기 불변식 15개 — 유한성·경계·속도 캡·enum 범위·단조성 |

`test_observability.py`가 특히 이 환경의 성격을 보여준다. 지금까지 발견된 관측 결함은 전부
같은 패턴이었다 — *두 State의 obs가 완전히 같은데 같은 액션·같은 RNG로 굴리면 결과가 갈린다.*
정책은 그 차이를 원리상 설명할 수 없고 BC는 줄일 수 없는 라벨 노이즈로 받는다. 데이터를 늘려서
해결되지 않으므로 관측 계약 자체를 고쳐야 한다. `obs_dim`이 442 → 470으로 커진 이력이 그 기록이다.

---

## 8. 알려진 한계

- 골대·크로스바 강체 충돌과 네트 물리가 없다(골라인 교차 기하로만 판정).
- 선수-선수 충돌은 100 Hz 원형 분리 근사이지 강체 impulse가 아니다.
- IFAB 최소 7명 미만 경기 중단은 미구현(단 전원 퇴장 극단에서도 유령 키커·파울은 방어된다).
- 개별 관측은 **의도적 부분관측**이다. 전이-완결을 목표로 하는 것은 `get_state()`뿐이다.
- `Engine` 상수의 `[DFL calib]` 표기는 **재검증 가능한 부분집합에 대한 주장**이다. 상속값
  (`launch_max`·`spin_*`·`gk_catch_speed_cap` 등)과 문헌 앵커·추정값(`c_drag`·`c_magnus`·
  스태미나 계열)은 그 한계를 `config.py` 필드 주석에 명시했다.

`env/AUDIT.md`에 수정된 결함과 그 재현 절차가 남아 있다.

---

## 9. 더 읽을 것

| 문서 | 내용 |
|---|---|
| [`env/README.md`](env/README.md) | obs/action/state·API·파이프라인·규칙의 **전체 스펙 표** |
| [`env/guide.md`](env/guide.md) | 설계 **근거** — 속도-명령 이동, facing 파생, 룰 정책 지표 설계 |
| [`env/AUDIT.md`](env/AUDIT.md) | 감사 이력 — 수정된 결함·성능 측정·남은 모델링 한계 |

---

University of Seoul · CIDA Lab
