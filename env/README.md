# SoccerBC `env` 명세 (spec)

11대11 축구 시뮬레이터. JAX(jit/vmap) 위에서 도는 순수 함수형 멀티에이전트 환경이며, **BC(모방학습)
데이터 생성과 실경기 트래킹 복원(reconstruct)**을 1급 요구사항으로 설계됐다.

문서 역할 분담:

| 문서 | 내용 |
|---|---|
| **`README.md`(이 문서)** | obs/action/state·API·파이프라인·규칙의 **스펙 표**(무엇이 어떤 값인가) |
| `guide.md` | 설계 **근거**(왜 속도-명령인가, facing을 왜 속도 파생으로 두나, 룰 정책 지표 설계) |
| `AUDIT.md` | 감사 이력 — 수정된 결함, 성능 측정, 남은 모델링 한계 |

---

## 1. 구성과 실행 계약

### 1.1 파일 구성

| 파일 | LoC | 역할 |
|---|---:|---|
| `env.py` | 858 | `SoccerEnv` — 생성자·검증·`reset`·`step_env_array`(스텝 본체)·BC 라벨 계약 |
| `state.py` | 53 | `State` NamedTuple(pytree) — 전 물리·규칙 필드 |
| `constants.py` | 307 | 불변 코드·sentinel·행동/관측/state 스키마·스타일 프리셋 |
| `config.py` | 711 | 조정 가능한 계수 — `Ball`/`Agent`/`Stadium`/`Engine`/`DeadBall`/`Foul`/`Reward`/`RulePolicy` |
| `setup.py` | 60 | 초기 배치·능력치 산출(`_meta`, 생성 시 1회) |
| `spatial.py` | 54 | 벡터 헬퍼 — `_safe_norm`/`_unit`/**L∞ radial stretch** 인코드·디코드 |
| `observation.py` | 523 | 액션 디코드·`get_obs_array`·`get_state`·`obs_spec`/`state_spec` |
| `movement.py` | 516 | reach 판정·키커 강제이동·데드볼 엔진·속도-명령 이동 적분·선수 분리 |
| `ball.py` | 206 | 공 자유물리(비행/굴림/바운스)·공-몸통 충돌·트래핑 |
| `contest.py` | 447 | 경합 승자 선정·킥/탈취/파울/굴절/GK 클레임/세트피스 소비·오프사이드 플래그 |
| `restart.py` | 356 | 침범 기하·세트피스 킥 락·재터치 금지·관측 재개 주입(복원용) |
| `events.py` | 306 | 득점/아웃 판정·재개 라우팅·페널티 해소·킥오프 배치·하프타임 전환 |
| `offside.py` | 81 | 오프사이드 콜(플래그 선수의 창 내 터치) |
| `fouls.py` | 138 | 카드/퇴장 추첨·차징 파울 |
| `rewards.py` | 33 | sparse/dense 보상 |
| `inverse.py` | 256 | 역동역학 — 관측 결과 → 액션/난수 draw 복원 |
| `policy.py` | 960 | 관측 기반 **룰 정책**(학습 정책 아님, 검증·데이터 생성용) |
| `tactics.py` | 174 | 정량 전술 지표(pressure/openness/lane_completion/xG/pitch_value) |
| `render.py` | 1531 | MP4 렌더 — `light`(numpy) / `rich`(matplotlib 3D) |

### 1.2 import 규약

모듈들은 **flat import**(`from constants import *`)를 쓴다. `env` 디렉터리가 `sys.path`에 있어야 한다.
아래 명령은 전부 **저장소 루트**에서 실행한다.

```bash
PYTHONPATH=env python -m unittest discover -s tests -v
PYTHONPATH=env python tests/audit_rollout.py --envs 64 --steps 500
```

### 1.3 클래스 합성

```python
class SoccerEnv(Events, Observation, Movement, Restart, BallPhysics,
                Contest, Fouls, Offside, Rewards, Inverse, Render, MultiAgentEnv)
```

기능별 mixin을 하나의 환경 클래스로 합친다. 상태는 전부 인자로 전달되는 `State`에 있고 인스턴스는
설정·파생 기하만 보유하므로 jit/vmap 안전하다.

### 1.4 생성자

```python
SoccerEnv(n_agents=11, n_opponents=11,
          agent_team=None, opponent_team=None,          # list[Agent]; 둘 다 생략 시 기본 11인
          game_duration=135_000, control_fps=25.0,
          *, ball_config=None, stadium_config=None, deadball_config=None,
          engine_config=None, foul_config=None, reward_config=None)
```

- 로스터는 **둘 다 주거나 둘 다 생략**해야 한다. 생략은 11대11에서만 허용(`DEFAULT_FORMATION`).
- 모든 config dataclass는 `frozen=True`이고 환경이 `replace()`로 **자기 인스턴스를 소유**한다
  (한 config 객체를 여러 환경이 공유해도 교차 변형 없음).
- `control_fps`는 `dt_phys`로 정확히 표현 가능해야 한다 — `1/(fps·dt_phys)`가 정수가 아니면
  `ValueError`(조용한 반올림 금지). 기본 `dt_phys=0.01`에서 지원 FPS = `100 / 양의정수`.
- `_validate_configuration()` / `_validate_rosters()`가 JIT 전에 물리·규칙·로스터 모순을 거부한다.

### 1.5 파생 상수 (11대11 · 25 fps 기준, 검증됨)

| 값 | 크기 |
|---|---|
| `N` (선수 수) | 22 |
| `obs_dim` | **470** |
| `state_dim` | **538** |
| `action_dim` | **8** |
| `decimation` (컨트롤 스텝당 물리 서브스텝) | 4 |
| `control_dt` | 0.04 s |
| `schema_version` | `{action: 1, observation: 6, state: 3}` |

### 1.6 하류 계약 상태 (2026-08-03)

> 이 절과 §15.1이 말하는 `recon/`·`calib/`·`bc/`는 이 env를 소비하는 **부모 프로젝트(SoccerBC)**의
> 디렉터리이며 본 저장소에는 포함되지 않는다. 여기 남기는 이유는 env 계약이 하류에서 어떻게
> 소비되는지가 계약 자체의 일부이기 때문이다.

env 내부 명세의 상수명은 `ACTION_SCHEMA_VERSION`, **`OBS_SCHEMA_VERSION`**,
`STATE_SCHEMA_VERSION`이다. 현재 recon의 `env_contract_hash()`는 관측 상수를
`OBSERVATION_SCHEMA_VERSION`이라는 잘못된 이름으로 조회하고 `observation.py`도 구현 해시에
포함하지 않아, obs 변경을 놓칠 수 있다. 자세한 재현은 [환경 감사](AUDIT.md)에 있다.

따라서 현 라벨의 env 해시 일치는 물리·행동 계약의 중요한 검사이지만, 기존 BC/MARL
체크포인트와의 obs 의미 호환성까지 증명하지 않는다. 하류 모델은 obs/action/architecture 지문을
별도로 저장·검사해야 한다.

---

## 2. 시공간·좌표 규약

- **물리 서브스텝** `dt_phys = 0.01 s`(100 Hz 고정). **컨트롤 스텝** = `decimation × dt_phys`.
  액션은 컨트롤 스텝마다 1회 제출되고 그 스텝의 모든 서브스텝에 **동일하게** 적용된다.
- 필드는 원점 중심. x∈[−52.5, 52.5], y∈[−34, 34] (기본 105×68 m). z는 위쪽 양수.
- `attack_dir[i] ∈ {+1, −1}` — 선수 i의 공격 x부호. 팀0은 +1로 시작하고 하프타임에 양 팀 반전.
- **정책 프레임 폴딩**: 액션 방향·관측 좌표는 `× attack_dir`로 자기 공격 프레임에 접힌다. 회전 규약은
  **180° 점대칭(x·y 동시 반전)**이지 거울이 아니다 — 스핀 z성분·`last_touch_code` 같은 유사벡터·
  범주값은 폴딩하지 않는다.
- 팀1 초기 포메이션은 `ROTATE_180`으로 team0 포메이션을 점대칭한 것이다.

---

## 3. 행동(Action) 스펙

`ACTION_DIM = 8`. 전 차원 `[-1, 1]`(tanh 정책 가정), env가 클립·디코드한다.
NaN 액션은 **0으로 무해화**된 뒤 클립된다(폭주 정책이 상태를 오염시키지 못하게).

| idx | 이름 | 범위 | 의미 |
|---:|---|---|---|
| 0 | `kick_gate` | [0,1] 유효, 임계 **0.5 초과** | 킥 의사(`want_f2b`) |
| 1:3 | `move` | L∞ stretch 2D | 이동 **목표 속도** 방향+크기 |
| 3:5 | `f2b` | L∞ stretch 2D | 킥 방향+파워 |
| 5 | `f2b_launch` | [-1,1] → `_u01` → [0,1]×`launch_max` | 발사각 커맨드 |
| 6 | `spin_side` | [-1,1] | 사이드스핀(수직축, 프레임 불변) |
| 7 | `spin_back` | [-1,1] | 백/톱스핀(진행방향 lateral축) |

### 3.1 L∞ radial stretch (`spatial.stretch_decode/encode`)

box `[-1,1]²` → 반지름 1 원판의 **방향보존·전단사** 사영: `stretch(v) = v·‖v‖∞/‖v‖₂`.
분해하면 방향 `unit(v)`, 크기 `‖v‖∞ ∈ [0,1]`. 결과적으로 **등방(모든 방향 최대치 도달 가능)**이고
잉여 표현이 없다(극좌표의 각도 잉여, 카르테시안의 비등방을 동시 해소).

### 3.2 디코드 (`Observation._decode`)

```
want_f2b   = clip(a)[0] > 0.5
mv_dir     = unit(a[1:3]) × attack_dir       # 정책 프레임 → 월드
mv_pow     = ‖a[1:3]‖∞
f2b_dir    = unit(a[3:5]) × attack_dir
f2b_pow    = ‖a[3:5]‖∞
f2b_launch = u01(a[5]) · launch_max          # 실제 각도는 contest에서 재매핑
spin_side, spin_back = a[6], a[7]            # 폴딩 없음
```

킥 발사각은 접촉 시 공 높이에 따라 `[launch_lo(ball_z), launch_max]`로 **재매핑**된다
(`Contest.launch_lo`): 지면공은 최대 −0.21 rad 하향, `launch_down_ref=1.0 m` 이상에서 −`launch_max`까지.

### 3.3 킥 파워·스핀 상한 (접촉 높이 의존)

| 접촉 밴드 | 조건 | 파워 상한 | 스핀 상한 |
|---|---|---|---|
| 발 | `z ≤ pelvis_frac·head_z` | 1.00 × `f2b_speed_max` (34.76 m/s) | 1.00 × `spin_max` |
| 가슴 | `pelvis_z < z ≤ head_z` | 0.87 | 0.55 |
| 머리 | `z > head_z` | 0.71 | 0.40 |

스로인은 별도 스케일 — `throw_speed_max = 21.5 m/s`, 파워캡·스핀 없음, 릴리즈 위치 = `head_z + 0.47 m`.

---

## 4. 상태(State) 스펙

`state.py`의 `State` NamedTuple. 전 필드가 JAX 배열이라 그대로 pytree/vmap 대상이다.

**공/시간**: `t` · `ball_pos(3)` · `ball_vel(3)` · `ball_spin(3)` · `ball_state`(`BALL_DEAD=0`/`BALL_ALIVE=1`)

**선수(N,·)**: `player_pos(N,2)` · `player_vel(N,2)` · `player_facing(N)` · `attack_dir(N)` · `team_id(N)` ·
`gk_indices(N)` · `vmax(N)` · `reach_z(N)` · `head_z(N)` · `player_ctrl(N)` · `stamina(N)` ·
`cooldown(N)` · `ctrl_lock_t(N)` · `yellow_cards(N)` · `active_player(N)` · `touch(N)` ·
`offside_flag(N)` · `penalty_encroach_mask(N)`

**경기/규칙(스칼라)**: `poss_team` · `last_touch_team` · `last_touch_code` · `kickoff_team` · `score(2)` ·
`restart_team` · `restart_t` · `restart_kind` · `restart_indirect` · `pending_taker` ·
`setpiece_taker` · `throw_taker` · `pass_team` · `pass_t` · `foul_kind` · `foul_actor` ·
`foul_victim` · `penalty_flight_team`

### 4.1 범주 코드 (`constants.py`)

**재개 종류** `RK_*`: `NONE=0` `KICKOFF=1` `THROWIN=2` `GOALKICK=3` `CORNER=4` `FREEKICK=5`
`PENALTY=6` `OFFSIDE=7` `GK_HOLD=8` (`RESTART_COUNT=9`)

**터치 코드** `TOUCH_*`: `NONE=0` `PASS=1` `SHOOT=2` `PASS_HEAD=3` `SHOOT_HEAD=4` `DRIBBLE=5`
`TACKLE=6` `GK_CATCH=7` `INTERCEPT=8` `DEFLECT=9` `PARRY=10` (`TOUCH_COUNT=11`)

**파울** `FOUL_*`: `NONE=0` `TACKLE=1` `CHARGE=2` `THROW=3`(스로인/세트피스 재터치)

**sentinel**: `NO_TEAM = NO_PLAYER = NO_EVENT = -1`, `SAMPLED_WINNER = -2`

---

## 5. 관측(Observation) 스펙 — `obs_dim = 470`

`get_obs_array(state) → (N, 470)`. 인덱스의 단일 진실원천은 **`obs_spec()`**(하드코딩 금지).
퇴장 선수(`active_player=False`)의 행은 전체가 0이고, 타 선수 블록도 퇴장자 열이 0 마스킹된다.

| 블록 | 슬라이스 | 폭 |
|---|---|---:|
| self | `[0:18]` | 18 |
| others | `[18:417]` | 19 × (N−1)=21 |
| ball | `[417:431]` | 14 |
| context | `[431:470]` | 39 |

### 5.1 self (18)

`abs_pos(2, ×attack_dir/half)` · **`abs_vel(2, ×attack_dir)`** · `stamina` · `vmax` · `reach_z` ·
`head_z` · `is_gk` · `yellow` · `cooldown` · `ball_ctrl` · `in_reach` · `f2b_avail` · `own_offside` ·
`pass_signal` · `retouch` · `ctrl_lock`

> **`facing` 성분은 없다 — 필요 없기 때문이다.** facing은 `movement.facing_from_velocity`가 정의하는
> **속도 방향의 결정함수**라 self `abs_vel`과 others `abs_vel`에 이미 전부 담겨 있다(§8). 별도 성분으로
> 넣으면 중복이다.

### 5.2 others (19 × 21) — 관측자 attack_dir로 접힘

`rel_pos(2)` · `abs_pos(2)` · **`abs_vel(2)`** · `stamina` · `vmax` · `reach_z` · `head_z` · `is_gk` ·
`yellow` · `offside` · **`pen_encroach`** · `ball_ctrl` · `ctrl_lock` · `cooldown` · `is_taker` ·
`team_flag(±1)`

> 타 선수 속도는 **절대 속도**로 준다. 위치는 절대·상대 둘 다 준다(국소 상호작용 + 필드 맥락).
> 절대속도는 **anticipation(선점)**을 가능케 해 룰 정책의 패스 차단·마킹 예측에 쓰인다
> (`tactics.lane_completion`·`pressure`). self `abs_vel`과 짝지으면 상대속도도 정책이 직접 만들 수 있다.

### 5.3 ball (14)

`rel_pos(3)` · `abs_pos(3)` · `abs_vel(3)` · `abs_spin(3)` · `ball_alive` · `poss_ours(∈{−1,0,+1})`

### 5.4 context (39)

`setpiece(21)` + **`retouch_is_throw`** + `off_line` + **`pass_t`** + `time_left` + `last_touch(±1/0)` +
`last_touch_code(one-hot 11)` + `score_diff` + **`second_half_kickoff_ours(±1)`**

`retouch_is_throw`는 현재 `retouch`/`is_taker` 래치가 스로인에서 생겼는지를 나타낸다. 스로인은
상대 골 직접 득점도 무효지만 일반 세트피스는 유효하므로, 이 비트가 없으면 같은 관측에서 득점과
골킥이 갈린다. `is_fk_indirect`도 재개 중에만 켜지는 신호가 아니라 **두 번째 터치 전까지 유지되는
라이브 래치**다. 재개 타이머가 0이 된 뒤에도 간접 프리킥의 직접골 무효 판정을 예측할 수 있다.

`second_half_kickoff_ours`는 **후반 킥오프가 우리 것인가**(±1)다. `kickoff_team`은 *전반* 킥오프
팀이고 후반은 그 반대인데(`events._halftime_switch`), 이 비트가 없으면 두 상태의 obs가 완전히
같은데 하프타임 전이 결과가 갈린다. 경기 내내 상수라 짧은 history로도 복원할 수 없다.

`pass_t`는 오프사이드 창의 **잔여 시간**을 `pass_protect`로 정규화한 값이다(창이 닫혀 있으면 0).
self 블록의 `pass_signal`이 '창이 열렸는가·누구 것인가'(±1/0)만 주므로, 이 값이 없으면 창이 곧
닫히는지를 관측만으로 알 수 없다 — 수신자가 언제까지 플래그에 걸리는지, 수비가 언제까지 기다리면
되는지가 빠진다. `get_state`의 game 블록은 이미 같은 값을 갖고 있었다(§6).

**setpiece(21)** = `is_sp_ours` · `rk_onehot(9)` · `rt_norm` · `is_kicker_locked` · `enc_margin` ·
`any_encroacher` · `is_fk_indirect` · `self_is_taker` · `self_kicker_ready` · `pen_flight` ·
`self_pen_encroach` · **`pen_encroach_ours_any`** · **`pen_encroach_theirs_any`**

`enc_margin`은 **규정 준수까지의 서명 여유거리**(음수=침범 깊이)를 `clear_r`로 정규화한 값이다.
페널티에서는 '이격 의무자 전원'(키커·수비GK 제외)이 subject이므로 공격수 쇄도도 신호에 잡힌다.

### 5.5 정규화

| 축 | 분모 |
|---|---|
| 위치 | `half_length`/`half_width`(self·abs) 또는 `length`/`width`(rel) |
| 선수 속도 | `norm_player_vel = 20 m/s` |
| 공 속도 | `norm_ball_vel = 50 m/s` |
| 공 높이 | `norm_ball_z = 20 m` |
| 스핀 | `norm_spin = spin_max = 78.4 rad/s`(동기화 강제) |
| 신체 높이 | `norm_body_z = 3 m` |
| 점수차 | `norm_score = 5` |

목표는 대략 `[-1,1]`이지만 **엄격한 클립이 아니다** — 그래서 `observation_space`는 비유계 `Box`로
정직하게 선언한다.

---

## 6. 중앙 상태(central state) 스펙 — `state_dim = 538`

`get_state(state) → (538,)`. 중앙집중 크리틱용 **절대 프레임**(폴딩 없음) 벡터. 레이아웃의
단일 진실원천은 `state_spec()`.

| 블록 | 슬라이스 | 구성 |
|---|---|---|
| players | `[0:484]` | 22 × 22 |
| ball | `[484:493]` | `pos(3)` `vel(3)` `spin(3)` |
| game | `[493:538]` | 45 |

**player(22)**: `pos(2)` `vel(2)` `face(2, cos/sin)` `vmax` `ball_ctrl` `reach_z` `head_z` `stamina`
`cooldown` `ctrl_lock` `on_pitch` `yellow` `pending_taker` `setpiece_taker` `throw_taker`
`penalty_encroach` `offside` `team` `is_gk`

**game(45)**: `poss_onehot(3)` `last_touch_team(3)` `attack_dir0` `ball_alive` `restart_t` `pass_t`
`time_left` `is_fk_indirect` `restart_team(3)` `pass_team(3)` `score(2)` `rk_onehot(9)`
`last_touch_code(11)` `kickoff_team(2)` `penalty_flight_team(3)`

> `STATE_SCHEMA_VERSION = 3`. v2에서 세 종류 taker·penalty latch·정확한 pass 잔여창을 넣었고,
> v3에서 라이브 간접FK 래치를 재개 타이머로 가리지 않게 했다. 차원은 538로 같아도 의미가 달라
> v2 중앙 크리틱 체크포인트와 호환되지 않으며, 구 405차원 체크포인트도 물론 호환되지 않는다.

---

## 7. 스텝 파이프라인

```python
obs, state, reward, done_all, info = env.step_env_array(key, state, act_arr, ...)
```

`act_arr`는 정확히 `(N, 8)`이어야 한다(형상 불일치는 `ValueError`).

### 7.1 스텝 구조

```
액션 디코드 (1회)
└─ lax.scan × decimation:
     1. _apply_kicker_move        세트피스 키커 강제 워킹(도착 전)
     2. (deadball_engine=True만)   재개 중 비-키커 이동을 엔진 타깃으로 대체 — 기본은 OFF(§16.1)
     3. _move                      속도-명령 적분 → 위치 → 분리 → facing → 스태미나
     4. _charge_foul               차징 파울 추첨(공이 데드될 수 있음)
     5. _kick_gate 재계산           차징 후 상태 기준
     6. _contest_winner             Gumbel argmax 승자 1명
     7. _apply_force2ball           킥/탈취/파울/굴절/GK클레임/세트피스 소비
     8. _ball_body                  공-몸통 충돌·트래핑
     9. _throwin_restriction        세트피스 재터치 금지 판정
    10. _offside_check              오프사이드 콜
    11. _ball_step                  공 자유물리 1서브스텝
    12. _events                     득점/아웃/페널티 해소 → 재개 라우팅
    13. ctrl_lock/cooldown 감쇠
t += 1 → 하프타임 전환 검사 → 보상 → 관측
```

**종료 조건은 시간제한뿐**(`t ≥ game_duration`)이라 `done`은 스칼라 하나로 충분하다.

### 7.2 반환 `info`

| 키 | 의미 |
|---|---|
| `scored` | 이 스텝 득점팀(−1=없음) |
| `poss_team` / `score` / `last_touch_team` / `active_player` | 스텝 후 상태 |
| `touch` | per-player 누적 터치 코드 |
| `foul_kind` / `foul_actor` / `foul_victim` | 파울 신호 |
| `truncated` | 시간제한 종료 |
| `kick_applied`* | **실현 인과킥** bool[N] |
| `move_forced`/`kick_gated`/`kick_forced`/`halftime_reset`* | 행동 강제성 원신호 |
| `bc_action_mask`* | per-dim 손실 마스크 (N,8) |
| `substeps`† | 서브스텝별 State 스택(렌더용) |

\* `include_bc_info=True`(기본)일 때. † `collect_substeps=True`일 때.

### 7.3 성능 스위치 (정적 bool)

- `include_bc_info=False` — BC 라벨 산출 생략(RL 핫루프).
- `compute_observation=False` — 관측 대신 `(N,0)` 반환. O(N²) 조립 생략(물리 검증·렌더).
- dict 어댑터(`reset`/`step_env`/`get_obs`)는 편의용이며, 핫루프는 **배열 경로 + jit/vmap**을 쓸 것.

---

## 8. 이동 모델 — 속도 명령 + 마찰 타원

정책은 **가속이 아니라 목표 속도**를 낸다. 트래킹 복원 라벨에서 속도 타깃은 매끄럽지만(방향 지터 ~2°)
가속 타깃은 2차 미분 노이즈 증폭으로 ~69°까지 튀어 BC 라벨의 SNR이 무너지기 때문이다(`guide.md §1.1`).

```
vmax_eff = vmax · (vmax_floor + (1 − vmax_floor)·stamina)      # vmax_floor = 0.91
v_cmd    = mv_pow · mv_dir · vmax_eff
```

`_vel_substep`이 Δv = `v_cmd − v`를 **현재 진행방향 기준 종·횡으로 분해**하고 **마찰 타원**
(friction ellipse / g-g diagram) 안으로 방사 투영한다:

| 캡 | 값 (m/s²) | 출처 |
|---|---:|---|
| 종 가속 `a_max` | 8.68 | DFL 실측 p99.9 |
| 종 감속 `brake_decel_max` | 10.51 | DFL 실측 p99.9 (감속 > 가속) |
| 횡 가속 `accel_norm_max` | 8.46 | DFL 실측 p99.9 |
| facing 회전 `turn_rate` | 4.76 rad/s | DFL 실측 p99.9 — **현재 미사용**(아래 facing 규약 참조) |

성분별 독립 박스 클립은 코너에서 √(종²+횡²)로 단일축 캡을 40~55% 초과했다. 타원 클립은 어느
방향으로도 합성가속 ≤ max(종캡, 횡캡)을 보장하며 균일 스케일이라 **명령 방향을 보존**한다.

부수 처리:
- 위치 적분 후 필드 경계 클립, `_separate`로 선수 겹침 해소(최소거리 `2·r_player`, `sep_iters=2`회).
  완전 겹침은 황금각 분산 방향으로 밀어 데드락을 막고, **강제이동 중인 키커는 pin**되어 밀리지 않는다.
- 퇴장 선수는 터치라인 안쪽 벤치 라인에 고정·정지(피치 밖 배치는 좌표 클립상 불가).
- 위치 적분 후 **경계에서 잘린 축은 속도 성분도 0으로** 만든다 — 위치만 클립하면 상태 속도가
  바깥으로 전속인 '유령 속도'가 남아 obs의 `abs_vel`이 거짓을 말하고 facing·스태미나·복귀 지연까지
  오염된다(실측 복귀 0.8 s → 0.08 s).
- **facing = 현재 속도 방향**(`movement.facing_from_velocity` — 단일 진실원천). 독립 적분 상태가
  아니라 파생값이며, `state.player_facing`은 그 캐시다. 정지(|v| ≤ 1e-3)면 자기 공격 방향으로 둔다 —
  '직전 방향 유지' 같은 은닉 래치를 두면 한 프레임 관측으로 복원할 수 없는 상태가 되살아난다.
  회전율 상한은 별도 계수가 아니라 **마찰 타원의 횡가속 캡(`accel_norm_max`)**이 만든다: 속도 방향이
  물리적으로 꺾이는 만큼만 facing도 꺾인다. 그래서 `turn_rate`는 더 이상 쓰이지 않는다.
  facing이 속도의 결정함수이므로 **obs에 facing 성분이 없어도 정책이 잃는 정보가 없다**(§5.1).
- 스태미나: 필드 선수만 소모. 기저 소모율은 `game_duration`에서 역산되어 **경기 길이와 무관하게**
  종료 시 `stamina_end_frac = 0.35`에 수렴한다. `sprint_speed = 5.5 m/s` 초과분에 배율 가산.
  **재개(데드볼) 구간은 스태미나 회계에서 통째로 제외**된다 — 소모도 회복도 없다. 실제 데드볼은
  p50 19.8 s라 선수가 걸어서 자리를 잡지만(실측 속력 p50 1.03 m/s) env 재개창은 5 s 안팎이라 같은
  거리를 뛰어야 하므로, 속도 비례로 두면 시간 스케일 차이가 그대로 벌점이 된다. 반대로 무소모면
  일부러 공을 내보내는 이득이 생긴다(`movement._move`).

---

## 9. 공 접촉 규칙

### 9.1 reach (`_in_reach`)

```
수평: ‖ball_xy − player_xy‖ ≤ Rxy + r_ball,  Rxy = gk_reach_xy(2.0) if 자기 박스 안 GK else reach_xy(1.6)
수직: ball_z ≤ reach_z + r_ball
```

속도-의존 reach 확장은 **폐지**됐다(필드 선수는 고정 반경).

> **reach_xy 재캘리브(2026-07-29): 1.2 → 1.6.** 구 근거였던 '접촉거리 p90≈1.25m'는 트래킹 임펄스에
> 스냅되지 않은 이벤트(액터-공 7~23m)가 섞인 오염 분포에서 나온 값이었다. J03WMX 전·후반 3,322접촉을
> 라벨 출처로 분해해, 임펄스 증거가 있는 깨끗한 집합(n=2,703)의 **p95=1.73m** 기준으로 재설정했다.
> 분위수 정책이 p90 컷 → p95 컷으로 한 단계 완화된 것이며, 실측 접촉 미표현율은 2.7% → 1.6%로 준다.
> `kicker_arrive_r`는 이 값에서 파생되므로 세트피스 도착 판정 반경도 함께 넓어진다(결합 유지).

### 9.2 킥 게이트 (`_kick_gate`)

```
allowed  = (재개 중) ? (지정 키커 && setup_done) : ball_alive
voluntary = want_f2b & in_reach & (cooldown ≤ 0) & allowed
forced    = 재개 중 & setup_done & 지정 키커 & 온피치       # [B] 킥 타이밍 강제
do_kick   = voluntary | forced
```

**킥 타이밍은 env가 결정한다** — 세트업 완료(키커 도착·정렬) 즉시 결정론적 발사. '언제 찰지'라는
은닉 결정이 사라져 BC가 킥을 인과 라벨로 학습할 수 있고, **킥 파라미터(방향·파워·발사각·스핀)는
여전히 정책 소유**라 역산 가능하다.

### 9.3 경합 승자 (`_contest_winner`)

후보 중 Gumbel argmax 1명:

```
score = −w_dist·d − w_time·(d/vmax) + w_height·height_fit + w_poss·poss + w_ctrl·ctrl
winner = argmax(score/contest_temp + Gumbel)      # contest_temp = 0.66
```

후보에서 제외되는 것: 퇴장자, **재탈취 지연 중인 상대**(`ctrl_lock_t > 0`), 쿨다운 중.
`w_poss = 0`(중립)이며, 균일 능력치 로스터에서는 `w_time`이 `w_dist`와 공선, `w_height`/`w_ctrl`은
공통상수라 **실효 지렛대는 거리뿐**이다(이질 로스터에서만 발현 — `config.py`에 정직 표기).

### 9.4 접촉 분기 (`_apply_force2ball`)

| 분기 | 조건 | 결과 |
|---|---|---|
| `free_play` | 승자가 비-상대소유 & GK클레임 아님 | 커맨드대로 발사 |
| `tackle_ok` | 상대 소유 경합 & `u < tackle_prob(0.28)` | 커맨드 적용 + 출구속도 캡(0.79×max) |
| `foul` | `u < p_foul`(로지스틱) | 공 정지 + FK/페널티 + 확률 카드 |
| `deflect` | 탈취 실패 & `u < deflect_prob(0.5)` | 소유 불변 루즈볼, 각도 랜덤(±1.2 rad), 속도 `0.61·v + 0.05` |
| `gk_claim` | 자기 박스 GK가 무의도 승리 | 저속(≤26.4)=캐치 홀드 / 고속=parry / 백패스=간접FK |

각 분기는 **결정적 propensity + 단일 uniform draw** 구조라 관측 결과에서 어느 분기·어떤 draw였는지
역산된다(§13).

터치 코드 라벨링: 골 방향 조준(`cos > 0.5`) + 26 m 안 = `SHOOT`, 자기팀 자기터치 & 저속(<9 m/s) =
`DRIBBLE`, 탈취 시 공속 ≥ 6 m/s = `INTERCEPT` 아니면 `TACKLE`, 헤더는 `*_HEAD`.

`TOUCH_TACKLE` 파울 라벨은 **경합 승자가 파울러인 경우에만** 붙는다. loser-foul(§10.4 `retained` —
파울러가 경합 패자)에서는 승자가 파울을 *당한* 쪽이므로 터치를 남기지 않고(`TOUCH_NONE`,
공도 정지), `last_touch_*`도 갱신하지 않는다(`AUDIT.md` 18).

### 9.5 공 자유물리 (`ball_step_only`) — 완전 결정적

- **공중**(`z > z_ground = 0.15`): 중력 + 드래그(`−c_drag·|v|·v`) + 마그누스(`c_magnus·(ω×v)`).
- **지면**: 드래그·마그누스 끄고 굴림 감속 테이블(속도-감속 10 knot 선형보간) 적용.
  지상 컬(`c_ground_curl`)은 **기본 비활성(0.0)** — 트래킹만으론 공중 컬과 통계적으로 구분
  불가해 방어 불가능한 물리항을 기본 탑재하지 않는다(코드 경로는 유지, 값만 0).
- **바운스**: 하강속도 > `ground_settle_vz(0.5)`만 실제 임팩트로 보고 `e_rest = 0.61` 반발,
  수평 `bounce_h_keep = 0.89`. 그 미만은 **정착**(z→r_ball, vz→0)시켜 미세진동 제거.
- **스핀↔병진 결합**: 접촉점 각운동량 보존 모델 `Δv = −(1+e_t)·α/(1+α)·u_spin`,
  `Δω = [Δv_y, −Δv_x]/(α·r)`. 에너지를 창출하지 않는다(구 독립전달 모델 폐기).
- 스핀은 `spin_decay = 0.31 /s`로 감쇠.

### 9.6 공-몸통 충돌 (`_ball_body`)

`leg_top(0.5) ≤ z < body_top_frac·head_z` 밴드의 접근 중인 공만 대상. **스윕 판정**(선분 최근접)으로
풀파워 킥의 터널링을 막는다. 분기: 트래핑(소유 획득·발밑 정착) / 바운스(반사) / 통과.

```
p_hit  = 1 − (1 − body_hit_prob)^(path/gate_r)          # 관통 경로 비례 hazard
p_trap = trap_base · ball_ctrl · clip(1 − v/trap_speed_ref, 0, 1)
```

지면공(z < `leg_top`)은 다리 아래로 통과시켜 능동 경합에만 맡긴다.

**이번 서브스텝에 이미 접촉한 선수**(`_apply_force2ball`의 승자)는 몸통 충돌 후보에서 빠진다 —
같은 서브스텝 이중 상호작용 방지. 배제는 `touch_before` 스냅샷과 대조해 판정하므로 창 길이가
`decimation`(=`control_fps`)에 종속되지 않는다(`AUDIT.md` 17).

---

## 10. 규칙 엔진

### 10.1 재개(restart) 상태 기계

| 종류 | 창(서브스텝) | 트리거 | 키커 |
|---|---:|---|---|
| `KICKOFF` | 500 | 경기 시작·득점 후·후반 | 센터스팟 최근접 |
| `THROWIN` | 500 | 터치라인 아웃 | 스폿 최근접 |
| `GOALKICK` | 500 | 공격팀 최종터치 후 골라인 아웃 | **GK 우선** |
| `CORNER` | 500 | 수비팀 최종터치 후 골라인 아웃 | 스폿 최근접 |
| `FREEKICK` | 500 | 태클·차징 파울, 재터치 위반, 백패스 | 스폿 최근접 |
| `PENALTY` | 540 | 자기 박스 안 파울 | 스폿 최근접 |
| `OFFSIDE` | 500 | 오프사이드 콜(통계 분리용 별도 코드) | 수비팀 최근접 |
| `GK_HOLD` | 800 | GK 캐치 | 그 GK |

- **카운트다운은 키커가 도착한 뒤에만 흐른다**. `setup_hold_substeps = 300`(3.0 s) 남으면
  `setup_done` → 결정론적 발사. `kickoff_instant = True`면 킥오프는 홀드를 건너뛰고 즉시 발사한다
  (키커를 스폿에 pre-snap하여 관측 시퀀스의 1프레임 점프와 BC 라벨 드롭을 함께 제거).
- **침범(`_encroach_geometry`)** — 단일 진실원천. 킥오프=센터서클, 스로인=2 m, 그 외 9.15 m.
  골킥은 **박스 밖만** 요구(IFAB Law 16, 반경 조항 없음), 페널티는 박스+아크 밖을 **양 팀 전원**에게
  요구(키커·수비GK 면제, 단 수비GK는 골라인 이탈 시 침범). 자기 골라인 위 수비수는 Law 13 면제.
- **퀵 프리킥**: FK/오프사이드FK는 '차는 행위 = 이격 요구 포기'로 보고 retake를 면제한다(Law 13).
  킥오프·코너·스로인은 침범 시 재실행.
- **페널티는 물리 플레이**다 — xG 주사위 없이 키커가 실제로 차고 GK가 필드선수로 경합한다.
  침범-재실행은 궤적 터미널(골/아웃/세이브/정지/GK캐치)이 관측된 뒤 IFAB 매트릭스로 판정한다:
  공격팀 침범 & 골 → 취소·재실행 / 수비팀 침범 & 무득점 → 재실행 / 그 외 결과 인정.
- **재터치 금지**: 세트피스 키커는 타인 접촉 전 재터치 시 상대 **간접** FK. 스로인은 `throw_taker`,
  그 외는 `setpiece_taker`로 분리 추적(스로인 직접골은 양방향 무효, Law 15).

### 10.2 득점·아웃 판정

IFAB 정합으로 **공 전체가 라인을 넘어야** 한다 → 판정선 = 라인 + `r_ball`. 골문 통과 여부(y·z)는
서브스텝 말 샘플이 아니라 **교차 시점으로 역내삽**해서 본다(40 m/s 공은 샘플까지 0.4 m를 지나쳐
포스트·크로스바 근처 오분류 밴드를 만든다).

무효 골: 스로인 직접골(양방향) · 세트피스 직접 자책골 · 간접FK 2차 터치 전 직접골.

### 10.3 오프사이드

1. 소유팀이 공을 **플레이**(PASS/SHOOT/DRIBBLE)한 순간, 2번째 최종수비 라인(+`offside_margin=0.5 m`)
   보다 앞서고 공보다 앞서고 상대 진영에 있는 동료에게 플래그를 세운다. 창 `pass_protect = 240`(2.4 s).
2. 창 안에 **플래그된 선수가 새로 터치**하면 콜 → 수비팀 **간접** FK.
3. 수비의 '의도적 플레이'(TACKLE/INTERCEPT)는 플래그를 리셋한다. **굴절(DEFLECT)은 리셋하지 않는다**
   (IFAB상 의도적 플레이가 아님).
4. Law 11 예외는 골킥/스로인/코너뿐 — GK 홀드 배급·킥오프도 플래그를 무장해 수비라인 뒤 상주
   익스플로잇을 막는다.

> 콜 트리거는 **이번 서브스텝에 새로 생긴 접촉만** 쓴다(`touch_before` 스냅샷). `state.touch`는 컨트롤
> 스텝 단위로만 초기화되므로 스냅샷 없이 보면 앞 서브스텝의 stale 터치로 즉시 오검된다(`AUDIT.md` 15).

### 10.4 파울·카드

| 채널 | 위치 | 성립 조건 | 확률 |
|---|---|---|---|
| 태클 | `contest.py` | 경합 승리(탈취) 또는 **loser-foul**(공 못 따고 돌진) & 접촉거리 2.5 m | `sigmoid(logit)` ∈ [0.008, 0.35] |
| 차징 | `fouls.py` | 보유자에게 `charge_speed(5.5 m/s)` 이상 접근 & 접촉 | ∈ [0.003, 0.10] |
| 재터치 | `restart.py` | 세트피스 키커의 2차 터치 | 결정적 |

로짓 특징: 등 뒤 접근, 접근속도, 깨끗한 태클(억제 −2.8), 박스 안(엄격 −0.67), 공중 경합(억제 −0.87).
페널티/FK 구분은 **파울러 위치**로 판정한다(공 위치 무관, IFAB Law 14).

카드: 파울당 `card_per_foul = 0.23125`(=37/160), 카드 중 레드 `0.054054`(=2/37).
**레드 1장 또는 옐로 2장 = 퇴장**. 보상엔 관여하지 않는 순수 규율(수적 열세는 물리로만 불리).

### 10.5 GK 특수 규칙

- 자기 박스 안 GK는 `want_f2b` 없이도 경합 후보가 된다(리액티브 클레임). 단 **합법 캐치일 때만** —
  자기 재터치 대상이거나 동료 **발 패스(백패스)**인 공은 제외한다. 이 게이트가 없으면
  '배급 → 재캐치 → 간접FK → …' 무한 루프가 생긴다.
- `gk_catch_speed_cap = 26.4 m/s` 이하 = 캐치 홀드(`RK_GK_HOLD`, 8초 창), 초과 = parry(쳐냄).
- 백패스를 손으로 잡으면 상대 **간접 FK**(카드 없는 기술 반칙).

---

## 11. 보상

```python
Reward(mode="sparse"|"dense", goal=1.0, advance=0.10, shaping_gamma=0.99, poss_gain=0.05)
```

- `sparse`: 득점팀 `+goal` / 실점팀 `−goal` (per-player).
- `dense`: sparse + **potential-based 전진 셰이핑** `advance·(γΦ′ − Φ)`, `Φ = ball_x·attack_dir/hx`
  (Ng 1999 — `shaping_gamma`가 트레이너 할인율과 일치하면 무편향) + 소유권 획득 1회 보너스.
  둘 다 인플레이(재개·득점 스텝 제외) 게이팅.

`mode`는 jit 정적 분기다.

---

## 12. BC 라벨 계약

이 환경의 핵심 부가 계약. **강제 프레임을 정책 결정처럼 학습하지 않게** 하는 것이 목적이다.

### 12.1 `action_agency(state) → dict[bool[N]]`

스텝 **진입 상태**에서 결정론적으로 계산(제출 액션값과 무관).

| 신호 | 의미 |
|---|---|
| `move_forced` | 이동이 env에 대체됨 — 세트피스 키커 강제 워킹 / 퇴장자 / (`deadball_engine=True`인 경우에만) 재개 프레임 전원 |
| `halftime_reset` | 이 스텝 끝에 `_halftime_switch`가 발동하는 프레임(`(t+1) == game_duration//2`) — **전 차원 라벨 마스크**의 근거이자 시퀀스 hard boundary |
| `kick_gated` | 자발적 킥 서브액션이 이번 스텝 효력 불가 — 미도달·쿨다운·비지정키커·데드볼·퇴장 |
| `kick_forced` | 카운트다운 소진으로 킥이 강제 발사되는 프레임 |

세 신호는 상호배타적이지 않다. vmax·스태미나·타원 클립 같은 **물리 클립은 '강제'가 아니다**
(방향 의도가 반영되므로).

### 12.2 `bc_action_mask(info) → bool[N, 8]`

```
dim[1:3] (move)        ← ~move_forced
dim[0,3:8] (kick)      ← (~kick_gated) ∨ kick_applied   # 강제킥·스텝내 도달킥 모두 열어 둔다
전 차원              ← ∧ ~halftime_reset                # 하프타임 전이 프레임은 하드 마스크
```

강제킥은 실제로 공에 적용되는 **인과 이벤트**이고 키커는 그 세트피스를 실제로 차는 주체라
'여기서 찬다'도 진짜 라벨이다.

★진입 게이트(`kick_gated`)는 스텝 **진입** 거리로 reach를 보는데 실제 전이는 `_move`로 선수를
옮긴 뒤 킥 게이트를 다시 계산한다. 그래서 진입엔 reach 밖이었지만 같은 0.04 s 안에 도달해 **실제로
찬** 킥이 존재한다(실측: 진입 1.815 m > 임계 1.71 m인데 `kick_applied=True`). 그 라벨을 버리지
않도록 킥 dim은 `kick_applied`로 게이트를 연다. `info["kick_gated"]` 자체는 문서화된 진입 시점
신호 그대로 나간다(RL의 사전 판단용) — 마스크만 사후 실현을 반영한다.

★**하프타임 전이 프레임은 전 차원 하드 마스크**이며 `kick_applied`보다 우선한다. 이 스텝이
끝난 뒤 `_halftime_switch`가 위치·공격방향·속도·공·재개를 통째로 덮어 **제출 액션이 post-state를
전혀 설명하지 못하기** 때문이다. 스텝 안에서 킥이 실제로 적용됐더라도 그 결과가 사라지므로 자유
행동 라벨로 세지 않는다. 진입 시점 마스크만으로는 잡히지 않지만(하프타임은 스텝 **끝**에 온다)
`t`는 스텝당 정확히 1 증가하므로 `(t+1) == game_duration//2`로 결정적으로 판정한다.
단 `get_avail_actions_array`가 투영하는 **사전 행동 권한은 그대로 열려 있다** — 이 프레임에도
에이전트는 액션을 제출해야 하고 그 액션은 전환 전 서브스텝 물리에 실제로 작용한다. 막는 것은
권한이 아니라 라벨이다.

오프볼은 여전히 `kick_gated=True`로 킥 dim이 막히고 이동은 살아난다
(통째 마스킹하면 필드 전원 이동 감독의 99.8%가 소실된다).

### 12.3 ★순수 킥 라벨 = 진입 마스크 ∧ `kick_applied`

`bc_action_mask`의 킥 dim은 **진입 결정론 게이트 = 필요조건**일 뿐이다. 킥의 인과성은 스텝 내부의
확률적 경합에서 갈리므로(승자만 파라미터 적용, 굴절·GK 캐치는 무효) 진입 마스크만으론
**경합 패자·굴절·GK 자동클레임을 과포함**한다.

```python
kick_label = info["bc_action_mask"][:, kick_dims] & info["kick_applied"][:, None]
```

`_kick_applied`는 per-player 터치 코드가 `{PASS*, SHOOT*, DRIBBLE, TACKLE, INTERCEPT}`인지로
판정하되, **파울 태클을 제외**한다(파울은 공을 정지시키면서 `TOUCH_TACKLE`로 라벨되므로 제출
파라미터가 공 속도를 결정하지 않은 **거짓 인과킥**이 된다).

또한 이 신호는 **컨트롤 스텝 끝의 누적 `touch`가 아니라 서브스텝별로**, `_apply_force2ball` 직후
`_ball_body` 이전에 집계된다. 위 인과 집합은 "이 코드는 contest가 params를 공에 적용한 결과"라는
전제 위에 있는데 **몸통 트랩(`ball._ball_body`)도 `DRIBBLE`/`INTERCEPT`를 기록**하기 때문이다 —
트랩 출구속도는 `trap_velocity_keep·v`인 순수 수동 물리라 제출 params와 무관하다(`AUDIT.md` 19).

이동은 진입 마스크 하나로 충분하다.

### 12.4 `get_avail_actions_array(state) → (N, 2)`

`[MOVE, F2B]` 마스크. 실제 전이에 쓰는 `action_agency`를 그대로 투영한다(마스크와 실권한 불일치 제거).

---

## 13. 복원(reconstruct) 인터페이스

실경기 트래킹으로부터 궤적을 재생하기 위한 **pin(주입)** 인자들. 모두 기본값에서 **포워드 불변**이다.

| 인자 | 형상 | 의미 |
|---|---|---|
| `forced_winner` | `(decimation,) int` | 서브스텝별 경합 승자 pin. `≥0`=그 선수 / `−1`=무승자 / `−2`=정상 샘플. **합법 후보 게이트(도달·쿨다운)는 통과해야 성립** — 게이트 밖 pin은 불발 |
| `forced_freeplay` | `(decimation,) bool` | `opp_poss`를 젖혀 터치를 결정론 킥으로 강제(태클/파울/굴절 추첨 무력화) |
| `suppress_charge` | 스칼라 bool | 차징 파울 추첨 봉쇄 |
| `suppress_body` | 스칼라 bool | 공-몸통 hit 추첨 봉쇄 |
| `suppress_retake` | 스칼라 bool | 침범 retake·페널티 재실행 봉쇄(근거: 심판이 실제로 진행시킴) |
| `suppress_restart` | 스칼라 bool | 드리프트로 라인을 넘은 sim 공이 재개를 유발하지 않게 |

봉쇄된 추첨도 **uniform은 그대로 소비**해 RNG 열이 불변이다.

`Restart`의 순수 어댑터:
- `canonical_restart_spot(...)` — 관측된 재개 종류/팀에 대한 env의 합법 스폿(스폿 기하 SSOT).
- `prepare_observed_restart(...)` — 데드볼 개시를 물리 진행 없이 1회 주입(선수 텔레포트 없음).
- `synchronize_observed_restart(...)` — 심판 페이즈·타이머만 pin.

---

## 14. 역동역학 (`inverse.py`)

`_apply_force2ball`의 모든 접촉 분기가 '결정적 propensity + 단일 draw'라 전부 역산된다.

| 분기 | 자유도 | 함수 |
|---|---|---|
| 킥(free_play·tackle_ok) | 방향·파워·발사각·스핀 | `infer_kick_action` — 정확한 역함수 |
| 스로인 테이크 | 방향·파워·발사각 | `infer_throwin_action` |
| 굴절 | **랜덤 각도 draw 1개** | `infer_deflect` → `{defl_ang, u, defl_speed}` |
| GK parry | **0** (입사속도·attack_dir의 결정함수) | `predict_parry`(정방향 대조용) |
| GK 캐치/백패스 | 0 (데드볼) | — |
| 이동 | 목표속도 = 다음 속도 그 자체 | `infer_move_action` |

`infer_contact(...)`가 `touch_code`·`restart_kind`로 디스패치하며 `kind ∈ {action, draw,
deterministic, dead}`를 반환한다. 관측점은 **`force2ball` 직후**(같은 서브스텝의 `_ball_body`·
`ball_step` 이전)이므로 트래킹의 컨트롤스텝 속도를 접촉 순간으로 역적분한 값을 넣어야 정확하다.

> 이동 역산이 자명해진 것이 속도-명령 모델의 직접적 이득이다 — 구 가속-명령의
> plant&cut·drag 근사 오차와 데드비트 과응답이 원천 소멸했다.

---

## 15. 설정 요약 (`config.py`)

| dataclass | 소유 |
|---|---|
| `Ball` | 반지름 0.11 m · 질량 0.43 kg · 단면적 0.038 m² |
| `Agent` | id · speed(vmax 7.96) · tall(1.8) · reach_z_max(2.7) · ball_control(0.5) · is_gk · init_pos |
| `Stadium` | 105×68 m · 골 7.32×2.44 · 페널티 16.5×40.32 · 골에어리어 5.5×18.32 · 서클/아크 9.15 |
| `Engine` | 물리·규칙 계수 전체(시간·이동·reach·킥·경합·파울창·재개창·공물리·정규화) |
| `DeadBall` | 데드볼 내장 배치 정책의 거리·혼합비 |
| `Foul` | 태클/차징 로짓 계수·확률 범위·카드율 |
| `Reward` | mode·goal·advance·shaping_gamma·poss_gain |
| `RulePolicy` | 룰 정책 튜너(역할 경계·shoot_gain·서포트·안티클럼프 등) |

### 15.1 캘리브 provenance (논문 정직성)

`Engine` 상수의 `[DFL calib/fit_*]` 표기는 **부모 프로젝트 AAMAS2027의 캘리브 파이프라인**에서
적합된 값을 가리킨다. SoccerBC 자체 `calib/`는 **프레임 동기가 불필요한 부분집합만** 독립 재검증한다
(두 파이프라인 모두 DFL 트래킹 원본에 결합돼 있어 본 저장소에는 포함되지 않는다 — §1.6).

| 등급 | 항목 |
|---|---|
| **재검증 가능** | vmax · a_max · brake · accel_norm · turn_rate(트래킹 유한차분 p99.9), tackle_prob·파울/카드율, **reach_xy(2026-07-29 SoccerBC 자체 재측정 — 임펄스 증거 있는 접촉 n=2,703의 p95)**, c_ground_curl |
| **재검증 불가 → 상속** | launch_max · spin_* · gk_catch_speed_cap · throw_speed_max · throw_height (이벤트 시각 지터 ±0.85 s로 단독 재측정 불가 — report가 N/A로 명시) |
| **미적합(문헌 앵커/추정)** | c_drag · c_magnus · ball_inertia_ratio(2/3) · bounce_tangential_e · body_spin_keep · 스태미나 계열 |

즉 "DFL-캘리브"는 재검증 가능 부분집합에 대한 주장이며, 상속·추정값은 그 한계를 필드 주석에 명시한다.

---

## 16. 부속 모듈

### 16.1 데드볼 내장 엔진 (`Engine.deadball_engine`, **기본 `False`**)

> **2026-08-03부터 기본 OFF — 데드볼 이동도 학습 대상이다.** 실측 대조에서 내장 휴리스틱이
> 체계적으로 틀렸다(골킥 수비 전진도는 현실 +2.5 vs 엔진 −34.9로 **부호가 반대**, 수비↔공 거리
> 42.6 vs 77.7 m). 이제 실데이터 데드볼 라벨(`recon/deadball.py`)로 배우고, 끄면서
> `move_forced`가 13.8 % → 5.0 %로 줄어 그만큼이 학습 범주에 들어왔다(남은 5 %는 키커 강제 +
> 퇴장). 코드는 **어블레이션용으로 유지** — `Engine(deadball_engine=True)`로 되돌리면 아래 동작
> 그대로다. 근거·측정표 전문은 `config.py`의 필드 주석 참조.

켠 경우: 재개 중 **비-키커 전원의 이동을 env 내장 컨트롤러가 강제**한다(`_deadball_target` =
교체 가능한 '두뇌'). 정책은 `ball_alive`에서만 이동을 몰고, 그 프레임은 전원 `move_forced=True`로
자동 마스킹되어 골/하프타임 텔레포트도 함께 흡수된다. 킥 파라미터는 여전히 정책 소유.
배치 규칙: 재개팀은 전진도 비례 침투, 수비팀은 컴팩트 블록, 코너=박스 침투/마킹, 페널티=아크 밖 정렬,
GK=골라인 앞. 마지막에 수비팀을 `clear_r` 밖으로 밀어 침범을 방지한다.

### 16.2 룰 정책 (`policy.py` + `tactics.py`)

**학습 정책이 아니다** — "obs만으로 그럴듯한 축구가 나오는가"를 검증하고 BC 데이터를 만드는 참조 정책.
각 매크로 행동의 가치를 **팀 득점 기여 기대값(≈xT)이라는 한 통화**로 재고 캐리어는 argmax만 한다:

```
V_shoot  = xg(self)·shoot_gain
V_pass_j = lane_completion_j · (max(xg(mate_j)·gain, pitch(mate_j)) + prog_w·전진량)
V_drib   = max(pitch(step), xg(step)) · 유지확률
V_clear  = 자기진영·강압박·무옵션 안전판
```

핵심은 슛을 'range 진입'이 아니라 **'지금 슛 vs 패스·드리블로 더 좋은 슛'**으로 결정하게 바꾼 것이다.
난사를 없앴는데 더 좋은 위치에서 쏘게 되어 **슛 시도가 오히려 ~3.5배 증가**했다(`guide.md §3` 측정표).

```python
policy_fn = make_rule_based_policy(env, match_key=..., team_styles=..., policy_config=...)
```

스타일 프리셋: `balanced` · `gegenpress` · `park_the_bus` · `tiki_taka` · `long_ball`
(5축: line/tempo/width/aggression/directness). kick/throw/shot 물리 solver는 환경별로 캐시된다(~226×).

### 16.3 렌더 (`render.py`)

```python
env.render_mp4(states, out_path=..., fps=25, mode="light"|"rich", ...)
env.substep_trajectory(info["substeps"])     # 100 Hz 물리 밀도 렌더용
```

`light`는 numpy 직접 래스터, `rich`는 matplotlib 3D 카메라·스켈레톤·미니맵·이벤트 오버레이.
제한구역 오버레이는 `_encroach_geometry`의 기하 규약을 따른다.

---

## 17. 불변식과 검증

```bash
PYTHONPATH=env python -m unittest discover -s tests -p "test_environment.py" -v   # env 회귀 28개
PYTHONPATH=env python -m unittest discover -s tests -p "test_observability.py" -v # 관측 aliasing 스윕
PYTHONPATH=env python -m unittest discover -s tests -v                            # 전체 로컬 테스트
PYTHONPATH=env python tests/audit_rollout.py --envs 64 --steps 500                # 무작위 불변식 15개
```

`tests/test_environment.py`는 AAMAS2027 `recon/tests/test_contracts.py`의 계약 테스트 구성을 참고하되
독립 실행되도록 발전시킨 것이다. 검사 항목은 API 형상·스키마 · FPS · config 격리 ·
NaN/경량 경로 · 자기속도/facing 파생 · 간접FK/재터치 출처/페널티/후반 킥오프 관측 · 비활성 마스크 ·
직접/분리 경계속도 · pinned 분리 · 이동/킥 역함수 · 직접골/오프사이드/재터치/페널티 규칙 · deadball/BC
agency · **180° 회전 대칭**이다. `audit_rollout.py`는 15개 장기 수치·범위·단조성 불변식을 별도로 센다.

`tests/test_observability.py`는 **관측 aliasing 자동 검출**이다 — State 필드를 하나씩 흔들어
"obs는 같은데 전이가 갈리는" 조합(=관측 계약 결함)을 기계적으로 찾는다. §5·§6이 열거하는 관측
성분들(`abs_vel`·`pass_t`·페널티 침범 래치·`kickoff_team`·`restart_indirect`·`retouch_is_throw`)이
전부 이 스윕이 잡아낸 결함을 메우며 추가된 것이다.

참고 처리량(동일 CPU): 32 env × 128 step jit/vmap ≈ **33 k env-step/s**,
관측 없는 state-only 경로 ≈ 33.9 k env-step/s.

### 남은 모델링 한계

- 골대/크로스바 강체 충돌·네트 물리 없음(골라인 교차 기하로만 판정).
- IFAB 최소 7명 미만 경기 중단 미구현(단 전원 퇴장 극단에서도 유령 키커/파울은 방어됨).
- 선수-선수 충돌은 100 Hz 원형 분리 근사(강체 impulse 아님).
- 개별 관측은 **의도적 부분관측**. 전이-완결을 목표로 하는 것은 `get_state()`뿐이다.
> **2026-08-06 4차 감사**에서 4건을 수정했다(`AUDIT.md` 16–19): 2차 감사가 남겼던 미수정 관찰 2건
> (retake된 킥의 `last_touch_*` 갱신, `_ball_body` 배제 창의 `decimation` 종속)과, 새로 확인한
> BC 라벨 계약 결함 2건(loser-foul이 피파울자에게 `TOUCH_TACKLE`+거짓 인과킥, 몸통 트랩이
> `kick_applied`로 오탐). `ball.py`가 바뀌었으므로 **env 계약 해시가 달라진다**(기존 recon 라벨은
> recon DESIGN §6.6 절차로 재검증 필요). `contest.py`·`env.py` 변경은 obs·BC 라벨을 바꾸지만
> 현 해시 구현이 그 파일들을 포함하지 않아 **해시로는 잡히지 않는다** — §1.6 참조.
