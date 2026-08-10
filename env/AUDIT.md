# SoccerBC environment audit

최초 검사일: 2026-07-27, 최신 갱신: 2026-08-06  
대상: 이 저장소의 `env/` (본문의 `SoccerBC/env`·`AAMAS2027/env` 표기는 감사 당시의 구 프로젝트명)

> 본문이 인용하는 `recon/`·`bc/`·`calib/` 경로는 이 env를 소비하는 **부모 프로젝트**의 문서이며
> 본 저장소에는 포함되지 않는다. 하류 계약이 어떻게 깨졌는지가 결함 기록의 일부라 그대로 남긴다
> (`env/README.md` §1.6 참조).

## 수정한 결함

1. `SoccerEnv()` 기본 인자가 `None + None`에서 즉시 실패하던 문제
   - 표준 11대11 기본 로스터를 `constants.py`에 정의했다.
   - 한 팀만 생략하거나 커스텀 인원에서 로스터를 생략하면 명시적인 `ValueError`를 낸다.

2. 표현 불가능한 control FPS의 조용한 반올림
   - 예전에는 60 Hz 요청이 `dt_phys=0.01`에서 decimation 2, 즉 실제 50 Hz로 실행되면서
     `control_fps`만 60으로 남았다.
   - 물리 서브스텝으로 정확히 표현할 수 없는 FPS는 초기화 시 거부한다.

3. 잘못된 행동/복원 pin 형상 묵인
   - `(N, 10)` 행동을 8차원 환경이 앞 8개만 읽던 경로를 차단했다.
   - `act_arr`, `forced_winner`, `forced_freeplay`의 정확한 형상을 검사한다.
   - `env.py` 데모에 남아 있던 10차원 행동도 8차원으로 고쳤다.

4. 세트피스 재터치 판정의 stale snapshot
   - 예전에는 control-step 입구의 taker만 4개 물리 서브스텝 전체에 재사용했다.
   - 재개 직후 같은 control-step 안에서 제3자가 먼저 공을 만져도 제한이 해제되지 않아,
     다음 프레임의 원 키커 터치를 반칙으로 오심할 수 있었다.
   - 매 서브스텝 접촉 직전의 taker와 touch를 snapshot하고, 그 서브스텝에서 새로 생긴 접촉만
     비교한다. 최초 재개 킥 자체는 재터치로 세지 않는다.

5. 실제 행동 권한과 available-action 마스크 불일치
   - `deadball_engine=True`이면 재개 중 전원의 이동을 환경이 덮어쓰지만 비키커 MOVE가 1이었다.
   - `get_avail_actions_array`가 실제 전이에 쓰는 `action_agency`를 그대로 투영하도록 통합했다.

6. 온피치 선수가 없는 팀의 키커가 상대 선수 0번으로 지정되는 문제
   - `argmin(inf)` 결과 0을 반환하지 않고 `NO_PLAYER=-1`을 반환한다.
   - 재개 카운트다운은 안전하게 소진되어 deadlock도 만들지 않는다.

7. 고정 키커와 다른 선수의 겹침을 통째로 무시하던 분리
   - pinned 선수는 움직이지 않되 상대 선수가 전체 겹침 보정을 받도록 수정했다.

8. 중앙 critic state의 Markov 누락
   - 합쳐진 taker 비트와 단순 pass-team 신호만으로는 서로 다른 다음 전이를 구분할 수 없었다.
   - pending/setpiece/throw taker, penalty encroachment, 정확한 pass 남은 시간, 마지막 터치 코드,
     penalty flight team, kickoff team, 팀/GK 표지를 추가했다.
   - 11대11 `state_dim`은 405에서 538로 바뀌며 `STATE_SCHEMA_VERSION=2`이다. 구 critic
     체크포인트는 그대로 로드하면 안 된다.

9. 소유팀 온피치 선수가 전무할 때의 유령 캐리어
   - `poss_team` 값만 남은 극단 상태에서 `argmin(all inf)=0` 선수를 상대로 태클/차징 파울이
     생길 수 있었다.
   - 실제 활성 possessor가 없으면 loose ball로 처리한다.

10. 범주/수치 안정성
    - `team_id`를 float32에서 int32로 바꿨다.
    - float32에서 무효인 `+1e-12` Gumbel 방식을 확률 clip으로 교체했다.
    - sentinel, 팀 코드, 행동 slice, epsilon, 카드 누적 기준, 스키마를 `constants.py`로 모았다.
    - 정방향과 역방향에 중복되던 굴절각·parry·몸통 반사 수치를 `Engine` 설정 하나로 합쳤다.

11. 파생 설정의 초기화 시점 오류
    - `kicker_arrive_r = reach_xy`, `norm_spin = spin_max`는 클래스 정의 시 복사되어 override를
      따라가지 못했다.
    - `Engine.__post_init__`에서 인스턴스 값으로 파생하고, 모순된 명시값은 환경 검증에서 거부한다.

12. 경기장 기하 중복
   - goal-area 규격, 재개 inset, 골라인 허용 밴드를 config로 옮겼다.
   - 이벤트 판정과 rich renderer가 같은 Stadium/Engine 값을 사용한다.

13. 여러 환경에서 공유한 config 객체의 교차 변형
   - `control_fps`를 적용하며 전달받은 `Engine.decimation`을 직접 바꿔, 동일 config를 재사용한
     다른 환경의 물리 서브스텝 수도 뒤늦게 변할 수 있었다.
   - 모든 config dataclass를 immutable로 만들고 환경별 `replace()` 인스턴스를 소유한다.
     FPS 파생 `decimation`도 대입하지 않고 새 `Engine` 인스턴스로 확정한다.
   - 초기화 후 config를 수정해 캐시된 `hx`·`goal_w` 등과 서로 다른 진실값을 만드는 경로도 차단했다.

14. 단일 진실원천(SSOT) 잔여 위반
   - 사용되지 않으면서 `Engine.g`와 중력을 중복 보유하던 `Physics`/`physics_config` API를 제거했다.
   - light renderer의 하드코딩 센터서클 `9.15`를 `Stadium.center_circle_radius` 참조로 바꿨다.
   - 팀 배열의 리터럴 크기 `2`는 `TEAM_COUNT`, rich 렌더의 복제 HEX 팔레트는 light RGB
     원본에서 파생하도록 바꿨다.
   - 룰 정책의 불변 역할/스타일 레이아웃과 이름 프리셋은 `constants.py`, 성능 튜너는
     `RulePolicy`, 데드볼 배치 튜너는 `DeadBall` 설정이 소유한다.
   - 공개 계약 이름을 `constants.py` 밖에서 재대입하면 실패하는 AST 회귀 검사를 추가했다.

## 2차 감사 (2026-07-28) — 추가 수정

15. 오프사이드 판정의 stale touch 오검 (`offside.py`)
    - `_offside_check`가 `state.touch`(컨트롤 스텝 단위로만 0 초기화)를 그대로 읽어
      "플래그된 동료가 터치했다"를 판정했다. 같은 컨트롤 스텝의 **앞 서브스텝**에서 찍힌 터치가
      남아 있으면, 뒤 서브스텝에서 패스가 나가 `set_flags`로 플래그가 서는 **바로 그 순간**
      조건이 성립해 오프사이드가 즉시 오검된다(수신자는 패스 이후 공을 만진 적이 없다).
    - 항목 4에서 `restart._throwin_restriction`에 도입한 `touch_before` 스냅샷 규약을
      `_offside_check`에도 적용해, 이번 서브스텝에 **새로 생긴 접촉만** 콜/리셋 트리거로 쓴다.
      수비 '의도적 플레이'(TACKLE/INTERCEPT) 창 리셋도 같은 규약으로 통일했다.
    - 재현: 서브스텝0에 X가 터치 → 서브스텝1에 동료 Y가 전방 패스(X는 오프사이드 위치).
      수정 전 `RK_OFFSIDE` 즉시 콜, 수정 후 플래그만 서고 정상 진행. 대조군(X 선행 터치 없음)은
      수정 전후 모두 콜 없음.
    - 노출도: 한 컨트롤 스텝에 2명 이상이 터치하는 프레임 비율 — 25 Hz 0.023%, 12.5 Hz 0.115%
      (decimation에 비례해 증가). 룰 정책 롤아웃에서는 드물지만 decimation이 큰 설정·실데이터
      복원에서 커진다.
    - 회귀 테스트 2개 추가(`test_flagged_receiver_touch_is_an_offside_call`,
      `test_stale_touch_from_earlier_substep_is_not_an_offside_call`).

미수정 관찰 2건(저위험, 결정 필요)을 남겼고 **4차 감사(2026-08-06)에서 둘 다 수정**했다 — 아래 16·17 참조.

## 4차 감사 (2026-08-06) — 미수정 관찰 2건 해소 + BC 라벨 계약 결함 2건

16. 세트피스 `retake`로 무효화된 킥이 `last_touch_team`/`last_touch_code`를 갱신 (`contest.py`)
    - `retake`는 `consume ⊂ free_play`라 `touched_real`에 포함됐다. 그래서 킥이 무효화되어
      공이 스폿에 정지(`new_vel=0`)하고 per-player `touch`도 `code=TOUCH_NONE`이라 기록되지
      않는데도, `last_touch_team`만 키커 팀으로 넘어가고 `last_touch_code`는 `TOUCH_NONE`이 됐다.
      obs로는 "방금 누가 찼는데 터치 종류는 없음"이라는 모순된 상태가 나간다.
    - 재현(코너킥, 팀1이 직전 SHOOT, 수비 1명이 스폿 3.5 m 침범 → retake):
      수정 전 `last_touch_team` 1→0 · `last_touch_code` SHOOT→NONE(obs one-hot argmax=0).
      수정 후 둘 다 보존. 대조군(침범 없음 → 정상 consume)은 수정 전후 모두 키커 팀·PASS로 갱신.
    - 수정: `touched_real = any_cand & (free_play | tackle_ok | foul | deflect) & (~retake)`.
    - 동역학 영향: 재개가 유지되는 동안 `_gk_reactive_claim`·`gk_backpass`는 `~restart_active`
      게이트라 두 필드를 읽지 않는다. 다만 **obs 값이 바뀌므로** 이 값을 읽는 정책(룰 정책의
      `i_last_touch` 게이트 등)의 롤아웃 궤적은 수치적으로 달라진다.

17. `_ball_body`의 몸통 배제 창이 `decimation`에 종속 (`ball.py`, `env.py`)
    - `excluded = state.touch > 0`인데 `state.touch`는 **컨트롤 스텝 단위로만** 0 초기화된다
      (`env.step_env_array`). 따라서 서브스텝 k에 접촉한 선수는 그 컨트롤 스텝의 남은 서브스텝
      전체에서 몸통 충돌 후보에서 빠지고, 배제 창 길이가 `decimation`(=`control_fps`)을 따라간다 —
      컨트롤 레이트가 물리를 바꾸는 결함이다(25 Hz 최대 0.04 s, 12.5 Hz 최대 0.08 s).
    - 수정: 항목 4·15에서 `_throwin_restriction`·`_offside_check`에 도입한 `touch_before`
      스냅샷 규약을 `_ball_body`에도 적용해 **이번 서브스텝에 새로 생긴 접촉만** 배제한다.
      `_ball_body(..., touch_before=None)`이면 종전 동작(호환용)이고, `env.step_env_array`가
      `_apply_force2ball` 직전 스냅샷을 넘긴다.
    - 재현: 몸통 밴드로 접근하는 공 + 선수가 앞 서브스텝에 접촉(`touch=PASS`)한 상태 —
      수정 전 상호작용 없음(배제), 수정 후 상호작용 발생. 같은 서브스텝 접촉자는 수정 후에도 배제.

18. loser-foul이 **파울 당한 선수**에게 `TOUCH_TACKLE`을 찍어 거짓 인과킥을 만든다 (`contest.py`)
    - 파울 채널은 두 갈래다. `contest`(도전자가 경합을 이기고 파울)에서는 승자 = 파울러라
      `code = where(foul, TOUCH_TACKLE, code)`가 맞고, `_kick_applied`의 `foul_actor` 제외 가드가
      거짓 인과킥을 막는다. 그러나 `retained`(loser-foul — 파울러가 경합 **패자**)에서는
      승자가 파울을 *당한* 쪽인데도 같은 줄이 그 승자에게 `TOUCH_TACKLE`을 찍는다.
    - 결과 3중 오류: ①피파울자가 '태클했다'는 거짓 터치 라벨 ②`last_touch_*`가 그 팀/TACKLE로 갱신
      ③`stop_ball`로 공이 정지(`new_vel=0`)했는데도 `TOUCH_TACKLE`이 인과 집합에 속해
      **`kick_applied=True`** — `foul_actor` 가드는 파울러만 막으므로 이 경로를 못 잡는다.
    - 재현(팀0 P5가 캐리어·경합 승자, 팀1 P16이 7 m/s로 돌진 → retained 파울, 400시드 중 22회):
      수정 전 `touch[P5]=TACKLE` · `last_touch=팀0/TACKLE` · `kick_applied=[P5]`(공속 0).
      수정 후 `touch[P5]=NONE` · `last_touch` 입력값 보존 · `kick_applied` 없음.
      대조군 contest-foul(승자=파울러)은 수정 전후 모두 `TACKLE` 라벨 + `kick_applied=False`.
    - 수정: `loser_foul = foul & (~opp_poss)`를 도입해 `TOUCH_TACKLE`은 `foul & ~loser_foul`에만,
      loser-foul은 `TOUCH_NONE`으로 두고 `touched_real`에서도 제외(항목 16과 동일 사유).

19. 몸통 트랩이 남긴 터치가 `kick_applied`(인과킥)로 오탐된다 (`env.py`)
    - `_kick_applied`는 "인과 집합 코드 = contest가 params를 적용한 결과"라는 전제로 컨트롤 스텝
      끝의 누적 `state.touch`를 읽는다. 그런데 `ball._ball_body`의 몸통 트랩도 `DRIBBLE`/
      `INTERCEPT`를 기록한다 — 출구속도가 `trap_velocity_keep · v`인 **순수 수동 물리**라
      제출 킥 params와 무관한데도 인과킥으로 라벨된다.
    - 재현: 몸통 밴드로 6 m/s 접근하는 공 → 트랩 성립 시 `touch=INTERCEPT`,
      공속 −6.0 → −0.6(=0.1×v), 그런데 `kick_applied=True`.
    - 수정: 인과킥 집계를 **서브스텝 단위**로 옮겨 `_apply_force2ball` 직후·`_ball_body` 이전에
      그 서브스텝의 새 contest 접촉만 뽑아 OR 누적한다(`_kick_applied(state, touch_before)`,
      스캔 캐리에 `kick_acc` 추가). `touch_before=None`이면 종전 동작(하위호환).
      덤으로 **누락 케이스도 복구**된다 — contest 킥 뒤 같은 컨트롤 스텝에 몸통 굴절이 덮어써
      최종 `touch`가 `DEFLECT`가 되면 종전 공식은 그 인과킥을 놓쳤다.
    - 노출도: 룰 정책 2,500스텝·랜덤 정책 3,000스텝 롤아웃에서는 차이 0건이었다(정책이 오는 공을
      능동적으로 차서 몸통 트랩이 드물다). 즉 **현재 라벨 데이터의 실측 오염은 관측되지 않았고**,
      계약 자체의 결함을 닫은 것이다 — 다른 정책·복원 경로에서는 발현할 수 있다.

부수 정리(동작 무관 또는 참조 정책 한정):
- `policy.py`: 세트피스 키커의 `spin_b`가 중화되지 않아 슛/크로스 분기의 백스핀(∓0.4)이
  새어 들던 비대칭을 `spin_s`와 맞춰 0으로 정리.
- `policy.py`: `_calibrate_shot_solver`의 사문(死文) `e.f2b_speed_max * 0.0` 항 제거(수치 동일).
- `render.py`: rich 렌더러가 재개 종류를 리터럴 정수(1~8)로 비교하던 곳을 `RK_*` 상수로 교체
  (항목 14의 SSOT 규약 위반 잔여분).

건전성 확인: 룰 정책(gegenpress vs tiki_taka) 1,500스텝(60 s) 롤아웃에서 공 위치·obs 전부 finite,
공이 필드 밖으로 이탈하지 않음, ball-alive 0.90. 추가로 **무작위 불변식 스윕 48 env × 400 step
(19,200 env-step, 범위를 벗어난 액션 포함) 34종 전부 통과** — finite/bounds, 카테고리 코드 범위,
sentinel 범위, 스코어·카드 단조성, 퇴장 흡수성, 퇴장자 obs 0-마스킹, 유령 소유자 부재,
NaN 액션 무해화, `avail_actions == action_agency` 투영. light·rich 렌더와 state/event 덤프도 정상.

**★하류 영향 — 라벨 재검증 필요**: `ball.py`는 `recon.runtime.env_contract_hash`가 해시하는
구현 파일 목록(`ball.py`·`movement.py`·`inverse.py`·`spatial.py`)에 들어 있어 **env 계약 해시가
바뀐다**(동일 레시피·동일 생성 파라미터로 대조: `fd870232…` → `30a7ec5b…`). `ReconBatch.load`는
불일치 시 경고가 아니라 **예외**를 던지므로, 기존 recon 라벨을 계속 쓰려면 recon DESIGN §6.6
절차를 먼저 수행해야 한다.

반면 항목 16·18이 건드린 `contest.py`와 항목 19가 건드린 `env.py`는 그 목록에 **없다** —
obs 의미(`last_touch_code`)와 BC 라벨(`kick_applied`)을 바꾸는데도 해시는 그대로다. 이는 3차 감사가
지적한 `env_contract_hash` 결함(관측·라벨 계약 변경 미검출)이 아직 살아 있다는 뜻이며, 해당 수정
전까지는 **하류 모델이 obs/action 지문을 별도로 저장·검사해야 한다**. 확인된 구현 위치는
`recon/runtime.py:98-105` — 존재하지 않는 `OBSERVATION_SCHEMA_VERSION`을 조회하고(실제 상수는
`OBS_SCHEMA_VERSION`) 해시 파일 목록에 `observation.py`·`contest.py`·`env.py`가 빠져 있다.

## 계약 변경 (2026-08-06) — facing을 속도 파생값으로, obs에 자기 절대속도 추가

결함 수정이 아니라 **의도된 설계 변경**이다. `OBS_SCHEMA_VERSION`이 1 → 3으로 올라간다
(2 = self `abs_vel` 추가, 3 = context `pass_t` 추가).

- **facing = 현재 속도 방향**(`movement.facing_from_velocity` 신설, 단일 진실원천).
  종전에는 `turn_rate` 상한으로 킥/이동 방향을 향해 따로 적분되는 상태였다. 그러면 facing이
  파울 로짓(`behind`·`shoulder`)을 통해 전이에 영향을 주면서도 obs로는 복원할 수 없는
  **은닉 상태**가 된다. 이제 `state.player_facing`은 속도의 결정함수를 담는 캐시일 뿐이다.
  - 정지(|v| ≤ `STATIONARY_SPEED_EPS`)면 자기 공격 방향. '직전 방향 유지' 래치를 쓰지 않는다
    (쓰면 한 프레임 관측으로 복원 불가능한 상태가 되살아난다).
  - 회전율 상한은 마찰 타원의 횡가속 캡(`accel_norm_max`)이 물리적으로 만든다.
    `Engine.turn_rate`는 **더 이상 참조되지 않는다**(캘리브 값이라 config에는 남겨 둠).
  - `_apply_kicker_move`의 '공을 바라보는 facing' 지정과 `_move`의 facing 적분·`eff_move`
    게이트를 제거했다. `reset_state`·`_halftime_switch`의 초기 facing도 같은 파생 규칙으로
    통일했고 **종전 값과 정확히 일치**한다(정지 + 공격 방향 → team0=0, team1=π).
  - `_move`에서 facing 전용이던 인자 `f2b_dir`·`do_kick`이 죽어 시그니처에서 제거됐다
    (`_move(state, eff_move, mv_dir, mv_pow)`). `env.step_env_array`의 중복 `_kick_gate`
    선계산도 함께 사라졌다.
- **obs self 블록에 `abs_vel(2)` 추가**(관측자 attack_dir로 접힘). facing이 속도의 결정함수이므로
  이 값이 있어야 facing이 관측 가능한 양만으로 완전히 결정된다(타 선수 facing은 others `abs_vel`로
  동일하게 복원). 결과적으로 **obs가 facing 성분을 따로 담지 않아도 정보 손실이 없다**.
  11대11 `obs_dim` 442 → **444**, 블록 오프셋 self `[0:18]` · others `[18:396]` · ball `[396:410]` ·
  context `[410:444]`. 구 442차원 체크포인트와 호환되지 않는다.
- 룰 정책은 `drib_pow`를 명목 상수 대신 실제 자기 속도 기반으로 되돌렸다(`i_self_vel`).
- **obs context 블록에 `pass_t`(1) 추가** — 오프사이드 창의 잔여 시간(`pass_protect` 정규화).
  self `pass_signal`은 창의 유무·소속(±1/0)만 주므로 창이 곧 닫히는지를 관측만으로 알 수 없었다
  (`get_state`의 game 블록은 이미 보유 — 크리틱만 알고 정책은 모르던 비대칭). `obs_dim` 444 → **445**,
  context `[410:445]`(34 → 35), `OBS_SCHEMA_VERSION` 2 → **3**.
- **obs에 페널티 침범 래치 노출** — others마다 `pen_encroach` 1비트(+21)와 setpiece 블록의
  팀 요약 `pen_encroach_ours_any`/`pen_encroach_theirs_any`(+2). 종전에는 `self_pen_encroach`만
  보여서, `events._events`의 재실행 매트릭스가 `penalty_encroach_mask`를 공격/수비로 갈라 쓰는데도
  **관측자는 남의 침범을 알 수 없었다** — 같은 obs에서 '골 인정'과 '골 취소 + 재실행 + 침범자 카드'가
  갈리는 aliasing이다. 팀 요약은 재실행 판정을, 선수별 비트는 카드·퇴장 결과를 각각 식별 가능하게
  한다(요약만으로는 후자를 못 푼다). 래치는 관측자 팀 기준 ours/theirs이고 재개 활성으로 게이팅하지
  않는다(페널티 해소 시 env가 마스크를 지운다). 퇴장자 열은 others 블록의 active 마스크가 0으로 만든다.
  `obs_dim` 445 → **468**, others `[18:417]`(18 → 19/선수), context `[431:468]`(35 → 37,
  setpiece 19 → 21), `OBS_SCHEMA_VERSION` 3 → **4**.
  ※이 21차원은 페널티 밖에서 상시 0인 희소 차원이다. 하류 감사의 '사실상 상수' 검사에 면제 등록하고,
  페널티 국면을 별도 stratum으로 표집해 실제로 변하는 표본이 배치에 들어가게 해야 한다
  (`../bc/DATASET_PLAN.md` §1.3·§5).

검증: `obs_spec()` 444/슬라이스 일치, obs의 self `abs_vel`이 `player_vel × attack_dir / norm`과
정확히 일치(오차 0), 전 프레임 `facing == f(vel, attack_dir)`(32 env × 400 step 최대오차 0 rad),
**obs만으로 재구성한 facing이 `state.player_facing`과 일치**(최대오차 2.4e-7 rad).
불변식 스윕 34종 재통과, 룰 정책 1,500스텝 정상, light·rich 렌더와 덤프 정상.

## 5차 감사 (2026-08-06) — 계약 변경이 드러낸 결함 4건

facing 파생·obs 확장 뒤 외부 리뷰가 제기한 4건을 전부 재현하고 수정했다.

20. **스텝 안에서 도달해 실제로 찬 킥이 BC 마스크에서 누락** (`env.py`) — 우선순위 높음
    - `action_agency`는 **스텝 진입 시점 거리**로 reach를 판정하는데(`_in_reach(state)`), 실제 전이는
      `_apply_kicker_move` → `_move`로 선수를 옮긴 **뒤** `_kick_gate`를 다시 계산한다. 두 시간 기준이
      다르다.
    - 재현: 진입 거리 **1.815 m**(임계 `reach_xy + r_ball` = 1.71 m) → `in_reach=False`,
      `kick_gated=True`. 같은 0.04 s 안에 선수 전진 + 공 접근으로 도달해 `touch=PASS`,
      `kick_applied=True`. 그런데 `bc_action_mask` 킥 dim은 False — 문서화된 계약
      `bc_action_mask ∧ kick_applied`를 따르면 **실제 적용된 킥 라벨이 버려진다**. vmap 3,000 시드
      스윕에서 이런 프레임이 **867건** 나왔다.
    - 수정: `bc_action_mask(info, kick_applied=None)`으로 확장해 킥 dim을
      `(~kick_gated) ∨ kick_applied`로 만든다. `bc_mask` 산출을 스캔 뒤로 옮겨 실현값을 받는다.
      **물리는 그대로**이고 마스크만 실현 결과와 정렬한다(리뷰의 1안 — 물리 의미를 덜 바꾸는 쪽).
      `info["kick_gated"]`는 문서화된 **진입 시점** 신호 그대로 내보낸다(RL의 사전 판단용).
    - 수정 후 867건 전부 마스크에 살아 있고, `kick_gated`는 867건 모두 진입값 True를 유지한다.

21. **터치라인 경계에 유령 속도가 남음** (`movement.py`)
    - 위치는 `clip`하지만 속도의 바깥 방향 성분을 제거하지 않았다. 경계에서 1초간 바깥 명령을 주면
      위치는 34.000 m로 멈추는데 저장 속도는 **+7.955 m/s**로 남는다.
    - 파급이 넓다: ①obs의 self/others `abs_vel`이 거짓을 말하고(정지한 선수가 전속으로 보고됨)
      ②facing이 속도 파생이라 바깥을 향하며 ③스태미나가 계속 소모되고 ④반대 명령을 줘도 마찰 타원
      캡 때문에 복귀가 지연된다(실측 **0.8 s**).
    - 수정: `V = where(P != P_free, 0.0, V)` — 경계에서 잘린 축의 속도 성분을 0으로. 클립은 그 축으로
      **바깥으로** 밀 때만 일어나므로 축 성분 0 = 바깥 법선 성분 제거다(안쪽 이동은 클립되지 않음).
    - 수정 후 저장 속도 0.0000, 복귀 **0.08 s**(2스텝).

22. **`kickoff_instant=False`에서 골 직후 facing 캐시 불일치** (`events.py`)
    - 득점 시 위치·속도는 리셋하면서 `player_facing`은 갱신하지 않았다. 기본값
      `kickoff_instant=True`에서는 뒤따르는 킥오프 pre-snap(`_apply_kicker_move`)이 우연히 고쳐 주지만,
      `False`로 두면 그대로 남아 **facing = f(velocity) 계약이 깨진다**.
    - 재현: 정상 스텝으로 facing 오차 0을 만든 뒤 득점 → `instant=True` 0.000 rad /
      **`instant=False` 1.571 rad**.
    - 수정: 속도를 건드린 뒤 `player_facing = facing_from_velocity(player_vel, attack_dir)`를
      **무조건** 재계산하고 `_replace`에 넣는다. 비이벤트 프레임에서는 속도가 그대로라 항등이므로
      포워드 불변이다(파생 계약을 코드로 강제하는 형태).

23. **전반 킥오프 팀이 관측에 없음** (`constants.py`, `observation.py`)
    - `kickoff_team`은 후반 킥오프 팀을 결정하는데(`_halftime_switch`의 `second_kick`) obs에 없었다.
      두 상태의 obs가 **완전히 동일**한데 하프타임 전이 결과가 A→팀1, B→팀0으로 갈린다.
      경기 전체에 걸친 상수라 짧은 history로도 복원 불가 — 관측이 답이다.
    - 수정: context에 `second_half_kickoff_ours`(±1) 1차원 추가. `obs_dim` 468 → **469**,
      context `[431:469]`(37 → 38), `OBS_SCHEMA_VERSION` 4 → **5**.
    - ※`/data/AAMAS2027/bc/DATASET_V2_PROPOSAL.md` §10.3은 이 항목을 "reset transition 전 action-head
      mask + halftime sequence hard boundary"로 처리하자고 판정했었다. 마스킹 규율에 의존하는 대신
      1비트로 aliasing 자체를 없애는 쪽을 택했다(둘은 배타적이지 않다 — 하프타임 경계 처리는
      `../bc/DATASET_PLAN.md` §4.2에 그대로 남는다).

검증: 4건 각각 수정 전 재현 → 수정 후 PASS. 무작위 불변식 스윕 34종 재통과(48 env × 400 step),
facing 파생 불변식·obs 복원(2.4e-7 rad)·`pass_t`·페널티 래치 전부 재통과, 룰 정책 1,500스텝,
`demo_match` 20초, light·rich 렌더와 state/event 덤프 정상.

## 6차 감사 (2026-08-06) — 하프타임 전이 프레임 라벨 마스크

24. **하프타임 전이 프레임의 BC 마스크가 열려 있다** (`env.py`)
    - `_halftime_switch`는 컨트롤 스텝 **끝**에서 발동해 위치·공격방향·속도·공·재개를 통째로 덮는다.
      그래서 그 프레임의 제출 액션은 post-state를 전혀 설명하지 못한다. 그런데 `action_agency`는
      스텝 **진입** 시점에 계산되므로 이를 알 수 없었고, 실측에서 전이 프레임(`t == game_duration//2`)의
      `bc_action_mask` 이동 dim이 **22명 전부 열려** 있었다. 항목 20과 같은 계열(진입 마스크 vs 실제 전이)이다.
    - `/data/AAMAS2027/bc/DATASET_V2_PROPOSAL.md` §10.3과 `../bc/DATASET_PLAN.md` §4.2는 이 프레임을
      "전 차원 마스킹"으로 규정했지만 **env가 강제하지 않아 수집기 규율에 의존**하고 있었다.
    - 수정: `t`는 스텝당 정확히 1 증가하고 전환 판정이 `state.t == game_duration//2`(증가 후)이므로,
      진입 시점에 `(t + 1) == game_duration//2`로 **결정적으로** 알 수 있다. `action_agency`가
      `halftime_reset`(bool[N], 프레임 단위 균일)을 추가로 반환하고, `bc_action_mask`가 전 차원을
      하드 마스크한다. 이 마스크는 **`kick_applied`보다 우선**한다 — 스텝 안에서 킥이 실제로
      적용됐더라도 전환이 그 결과를 지우기 때문이다.
    - `move_forced`/`kick_gated`에 섞지 않고 **별도 신호**로 둔 이유: 그 둘은
      `get_avail_actions_array`가 그대로 투영하는 **사전 행동 권한**인데, 하프타임 프레임에도
      에이전트는 액션을 제출해야 하고 그 액션은 전환 전 서브스텝 물리에 실제로 작용한다.
      막을 것은 권한이 아니라 라벨이다.
    - 검증(`game_duration=100`, 하프타임 `t=50`): 전이 프레임 이동 감독 **22 → 0**, 킥 dim 0,
      `halftime_reset=22`. 사전 권한 `avail_move`는 22로 유지. 인접 프레임(t=48·49·52·53)은
      플래그 0에 이동 감독 22로 무변화. **후반 킥오프 킥 라벨(t=51: `kick_forced=1`,
      `kick_applied=1`, 킥 dim 1)은 그대로 살아 있다** — 부수 피해 없음.
      우선순위 검사: `kick_applied=True ∧ halftime=True` → 열린 dim 0.
      항목 20 회귀(`kick_gated=True ∧ kick_applied=True` → 킥 dim 열림) 유지.
    - `info["halftime_reset"]`으로도 노출된다. 데이터셋의 동명 필드(시퀀스 hard boundary·
      GRU hidden 리셋 근거)가 이 값을 그대로 쓴다.

> 남은 관련 사안: **새 시퀀스 시작부 `H-1` 프레임의 history 창 구성**이 아직 규정돼 있지 않다.
> 후반 킥오프 킥은 새 시퀀스의 첫 전이라 가용 history가 1프레임뿐이므로, 고정 `H`를 요구하면
> 희소한 킥오프 라벨을 잃는다. `../bc/DATASET_PLAN.md` §4.2(W6 신설)와 `SEQUENCE_PLAN.md` §5의
> 과제로 남긴다.

## 구조와 성능

- 불변 코드·sentinel·행동/관측/state/룰 정책 범주 스키마: `constants.py`
- 조정 가능한 물리·규칙·데드볼·룰 정책 계수: `config.py`
- 로스터 메타데이터는 환경 생성 시 한 번 계산해 reset마다 재생성하지 않는다.
- `reset_state()`를 추가해 관측이 필요 없는 물리/렌더/정책 팩토리 경로에서 O(N²) 관측 조립을 생략한다.
- 룰 정책의 kick/throw/shot 물리 solver는 환경 인스턴스별로 캐시한다.
- 오프사이드 2번째 최종수비 계산은 전체 sort 대신 `lax.top_k(..., 2)`를 쓴다.
- 타 선수 관측의 active mask는 특징별 중복 곱 대신 블록 마지막에 한 번 적용한다.
- `step_env_array(..., include_bc_info=False, compute_observation=False)` 경량 경로를 제공한다.
  각각 BC 전용 정보와 관측이 필요 없는 호출에서 사용한다.

동일 CPU/JAX 프로세스의 수정 전후 참고 측정:

- 32 env × 128 step, JIT/vmap: 수정 전 약 33,639, 수정 후 반복 측정 약
  33,191~34,271 env-step/s. CPU 변동폭 안의 동급 처리량이며 유의한 회귀는 관측되지 않았다.
- 같은 환경에서 룰 정책 팩토리: 최초 약 9.22 s, solver cache hit 약 0.041 s (`약 226×`)
- 관측 없는 state-only 배치 경로: 약 33,864 env-step/s, compile+첫 실행 17.59 s
- 64 env × 500 random step 감사: 32,000 env-step, 최종 compile 포함 20.55 s
  (반복 범위 20.28~24.57 s), steady-state 처리량은 동급이며 모든 불변식 통과

작은 단일 환경 dense 커널을 `lax.cond`로 잘게 나누는 시도와 upper-triangle scatter 선수 분리는
CPU에서 각각 느려져 채택하지 않았다. 현재 구현은 N=22에서 dense vectorization과 batched `vmap`을
우선한다. 즉 일반 학습 스텝의 수치 의미를 바꿔 얻는 속도 향상은 택하지 않았고, 실질적인 큰 이득은
반복 팩토리의 solver 캐시와 관측/BC 정보가 필요 없는 호출의 경량 API에서 얻는다.

## 검증

빠른 회귀:

```bash
PYTHONPATH=env python -m unittest discover -s tests -v
```

대규모 무작위 불변식:

```bash
PYTHONPATH=env python tests/audit_rollout.py --envs 64 --steps 500
```

28개 회귀 테스트는 `tests/test_environment.py`에 있으며 API·스키마, 관측 alias 반례, 경계 물리,
룰·agency, 이동/킥 역함수와 180도 회전 대칭을 고정한다. `tests/audit_rollout.py`는 15개 장기
finite/bounds/range/단조성 불변식을 배치 롤아웃으로 검사한다. AAMAS2027의 계약 테스트 구성은
참고했지만 테스트와 import는 SoccerBC 내부에서 독립 실행된다.

## 5차 감사 (2026-08-06) — 관측 alias·분리 경계속도

다음 세 반례를 수정하고 `tests/`에 회귀 계약으로 고정했다.

1. **라이브 간접 프리킥 래치 은닉**: `restart_t=0` 뒤에도 `restart_indirect`가 직접골을 무효화하지만
   관측과 중앙 상태는 이를 0으로 가렸다. 두 벡터 모두 두 번째 터치 전까지 실제 래치를 노출한다.
2. **스로인/일반 세트피스 재터치 출처 alias**: 기존 `retouch`·`is_taker`는 taker만 같으면 동일했지만
   스로인 직접골과 일반 세트피스 직접골의 판정은 다르다. context에 `retouch_is_throw` 1비트를 추가했다.
3. **분리 뒤 경계 유령속도**: 적분 클립의 속도는 정리했지만 `_separate`가 선수를 경계로 밀어낸 뒤
   바깥 법선 속도가 남았다. 분리 후 바깥 성분만 제거하고 안쪽·접선 성분은 보존한다.

계약 변화: `obs_dim 469 → 470`, `OBS_SCHEMA_VERSION 5 → 6`, 중앙 상태 차원은 538 그대로지만
간접FK 비트 의미가 바뀌어 `STATE_SCHEMA_VERSION 2 → 3`이다. 구 체크포인트와 데이터 manifest는
스키마가 다르므로 혼용하지 않는다.

## 남은 모델링 한계

- 골대/크로스바의 강체 충돌과 네트 물리는 없다. 득점/아웃은 골라인 교차 기하로 판정한다.
- IFAB의 최소 7명 미만 경기 중단은 구현하지 않았다. 다만 전원 퇴장 극단에서도 잘못된 키커나
  유령 파울이 생기지 않게 방어한다.
- 선수-선수 충돌은 100 Hz 원형 분리 근사이며 강체 impulse 모델은 아니다.
- 개별 정책 관측은 의도적으로 부분관측이다. `get_state()`만 중앙 critic용 전이-완결 상태를 목표로 한다.
- DFL에서 직접 재식별하지 못한 상속/추정 계수의 통계적 한계는 `config.py` 각 필드 설명대로 남아 있다.

## 3차 감사 (2026-08-03) — recon → BC 계약

기능 코드는 수정하지 않고 하류 소비 경로를 검사했다. 현재 14개 recon v5 라벨은 저장된 env hash
`bcab4b17990de295…`와 현 환경이 일치하며, BC 캐시의 DFL 위치·활성·GK·인플레이 대조는 통과했다.

다만 계약 해시 구현에서 한 가지 높은 우선순위 결함을 확인했다.

- 실제 상수: `OBS_SCHEMA_VERSION`
- `recon.runtime.env_contract_hash`가 조회하는 이름: `OBSERVATION_SCHEMA_VERSION`(존재하지 않음)
- 구현 파일 해시: `ball.py`, `movement.py`, `inverse.py`, `spatial.py`만 포함하며
  `observation.py`는 제외

프로세스 안에서 선언된 `OBS_SCHEMA_VERSION`을 1→2로 바꿔도 해시가 변하지 않는 것을 재현했다.
해시 키가 없으면 즉시 실패하게 하고, 명시적 env contract manifest와 모델용 obs/action 지문을
도입해야 한다. 수정하면 저장 해시가 달라지므로 recon DESIGN §6.6 절차를 먼저 수행한다.

하프 문맥 감사에는 별도 false negative가 있었다. `bc.audit`는 전반 끝과 후반 시작의 stamina
**평균**, 카드 보유자 **수**만 비교하므로, 슬롯 간 사람 재배치와 이후 교체 상속을 보지 못한다.
person ID 기준으로 전수 대조하면 현재 인플레이 cache의 active 선수·프레임 중 stamina 9.58%
(최대 오차 0.359), yellow 0.777%가 다르다. 이는 env 구현 문제가 아니라 `bc.context`가 후반
carry를 슬롯 배열로 적용한 하류 복원 결함이다. 현재 `v6`도 이 입력으로 학습됐다.

`bc.audit`가 보고한 절대 위치 정규화의 `[-1,1]` 이탈은 0.13–0.18%, 최대 1.09였다. 이 문서의
환경 명세처럼 관측은 엄격히 클립하지 않고 observation space도 비유계이므로 **현재는 오류가 아닌
정보성 관찰**이다. 감사 도구가 이 구분을 출력에 명시하면 오해를 줄일 수 있다.
