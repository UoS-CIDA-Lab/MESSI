# 규칙 정책 현실성 개선 기록 — 2026-09-09

## 결정과 범위

이번 변경은 다음 현상을 대상으로 한다.

- 킥오프 직후 여러 선수가 같은 방향으로 동기화되는 이동
- 안전한 패스가 있는데도 한 선수가 오래 운반하는 행동
- 선수의 formation 측면을 과도하게 벗어나는 드리블
- 탈취 직후 낮은 품질의 슛이 과도하게 선택되는 행동
- 연결된 패스 수신자를 현재 프레임의 가시성·오프사이드로 다시 판정하던 인과 오류

환경의 접촉, 득점, 오프사이드, 재시작 합법성은 바꾸지 않았다.
드리블 재접촉의 정책 의미를 보존하기 위해 고정형
`RulePolicyState.carrier_age` `(P,)` int32 leaf 하나를 추가했다. 새 난수
스트림, 동적 반복, 선수 쌍 텐서는 추가하지 않았다.
`policy_config_fingerprint`는 dataclass 전체를 canonical JSON으로 직렬화하므로
새 public config 필드도 자동으로 정책 식별자에 포함된다.

## FootballWorld 정책 원리


1. 패스 후보의 합법성과 completion은 전술 선호보다 먼저 보존한다.
2. SHOT/PASS/DRIBBLE/CLEAR는 같은 bounded utility 공간에서 확률적으로 선택한다.
3. 역할과 폭·깊이는 formation anchor에 상대적으로 정의한다.
4. 외부 학습 policy는 callable을 고정하고 parameters/state를 동적 PyTree로
   전달해 값 변경 때문에 재컴파일하지 않게 한다.
5. 모든 episode draw는 명시 match key와 public clock/identity에서 유도한다.

### 변경하거나 거부한 기준 동작

- 안전한 패스가 있어도 soft limit 뒤 dribble utility가 일정한 비율로만
  남던 plateau를 거부했다. 강제 PASS나 숨은 timer 대신 안전한 lane,
  개인 control age, 관측 기반 team episode age를 결합한 연속 utility를 쓴다.
- 물리적 제어가 잠시 풀릴 때마다 개인 운반 시간과 decision random bucket이
  새로 시작되는 동작을 거부했다. 같은 visible actor의 `CONTROL/TRAP`
  provenance만 carry lineage로 인정하고, handoff·손실·다른 loose event는
  fail-closed로 끊는다.
- 현재 프레임의 `visible` 또는 prospective offside가 과거 연결 수신의
  provenance를 지우는 동작을 거부했다. 과거 행위자 identity와 현재 패스
  합법성은 서로 다른 estimand다.
- 킥오프에서 모든 선수가 현재 formation target까지 직선으로만 향하는
  동작을 거부했다. 역할·roster slot 기반의 짧은 2-D waypoint를 사용하되
  opening window가 끝나면 원래 target으로 정확히 돌아간다.
- 탈취 직후 낮은 품질 shot이 다른 macro utility를 압도하는 동작을
  거부했다. 짧은 settle 할인만 적용하고 높은 품질 기회는 그대로 둔다.

## FootballWorld 구현

### Solo possession과 연결 수신

`decide_possession`은 caller가 주는 public-causal 입력을 사용한다.

- `possession_episode_seconds`: observer-local `RulePolicyState.possession_age`
- `formation_anchor_y`: 현재 carrier의 시작 formation anchor
- `possession_seconds`: observer-local `RulePolicyState.carrier_age`에서
  유도한 현재 선수의 연속 전술 운반 시간

`carrier_age`는 uninterrupted physical control 시간이 아니다. 같은
visible actor가 계속 소유하거나, 공이 live인 동안 그 actor의
`CONTROL/TRAP` last-contact provenance가 확인되면 증가한다. 보이는 handoff나
소유 손실에서는 초기화되고, possession이 hidden이면 새 제어 시간을
추정하지 않고 기존 값을 동결한다. 따라서 반복 self-recontact가 soft limit,
decision cadence, episode-keyed 난수 bucket을 다시 시작하지 못한다.

`previous_actor`가 없는 동일 팀 episode에서는 team episode age도 solo
tenure의 보수적 fallback이다. 연결된 수신자가 있으면 긴 team build-up age를
새 carrier 개인 운반 시간으로 상속하지 않는다.

연결 판정은 다음 identity mask만 사용한다.

`previous_actor & context.same_team & context.participating & not_current_self`

현재 visibility와 prospective offside는 이 과거 identity 판정에 들어가지
않는다. 따라서 보이지 않거나 현재 offside인 이전 동료도 causal pass
연결은 보존한다. 반대로 상대, 비활성 슬롯, 현재 carrier 자신을 가리키는
malformed provenance는 fail-closed다.

안전한 pass completion이 있을 때만 solo soft limit 이후 pass utility를
올리고 dribble utility를 더 낮춘다. 충분히 높은 전진 dribble utility가
best pass보다 좋은 경우에는 추가 release pressure를 적용하지 않는다.
formation anchor로부터의 y 이탈 비용도 eligibility ban이 아닌 soft cost이며,
압박이 커지면 약해져 긴급 탈출을 막지 않는다.

### 탈취 직후 shot

연결된 이전 동료가 없고 episode age가 settle window보다 짧으며 shot
quality가 기존 safe-completion 경계보다 낮을 때만 shot macro utility를
연속적으로 할인한다. 슛 방향, 힘, 물리 적용, 득점 판정은 바꾸지 않는다.
높은 품질 chance와 연결된 수신 뒤의 슛은 이 할인에서 제외된다.

### 킥오프 역할·슬롯 2-D 경로

`_KICKOFF_ROLE_PATH`는 GK/CB/FB/CM/WM/CF/WF별 longitudinal/lateral
waypoint 크기를 갖는 고정 2-D 표다. longitudinal 부호는 stable roster-slot
parity로 분산하고, lateral 부호는 공격 시 formation 폭을 유지하며 수비 시
compact 방향을 강화한다. 이는 역할별 경로를 만들기 위한 choreography
`DESIGN_PRIOR`이며 tracking-data fit이 아니다.

caller는 첫 live action에서
`(absolute_tick + 1) / kickoff_path_window_ticks`를 전달한다. active 조건은
`absolute_tick < kickoff_path_window_ticks`다. quadratic envelope
`4 * phase * (1 - phase)`는 window 마지막 tick의 phase 1과 window 이후의
phase 0에서 모두 0이므로 settled formation target은 기존과 같다. 초기
선수 위치와 환경의 킥오프 합법성은 변경하지 않는다.

## 새 public config의 증거 분류

다음 다섯 값은 모두 `DESIGN_PRIOR`다. 현재 정책/관측/행동 계약에서 선택한
bounded tuning controls일 뿐, DFL 또는 다른 제공자에서 측정된 보편적 축구
상수가 아니다.

| 필드 | 개선 후 기본값 | 검증 범위 | 용도와 비측정 지위 |
|---|---:|---|---|
| `dribble_shape_drift_penalty` | 0.20 | `[0, 1]` | 오래된 비압박 운반의 formation-y 이탈 soft cost. 측정된 위치 복귀율이 아니다. |
| `turnover_shot_settle_s` | 0.8 s | `> 0` | 새 possession의 저품질 shot 할인 시간축. 측정된 프로 평균 시간이 아니다. |
| `turnover_shot_value_scale` | 0.35 | `[0, 1]` | settle 시작점의 shot utility 배율. 슛 성공 확률이 아니다. |
| `kickoff_path_window_s` | 3.0 s | `> 0` | 역할별 opening waypoint가 사라지는 정책 창. 경기 규칙 시간이 아니다. |
| `kickoff_path_lateral_shift_m` | 2.4 m | `> 0` | 2-D 역할표의 base displacement scale. tracking 이동량 추정치가 아니다. |

`_KICKOFF_ROLE_PATH`의 역할별 2-D multiplier도 같은 `DESIGN_PRIOR`
분류다. 다섯 public knob와 별도로 구성된 고정 choreography bundle이며,
측정 계수로 제시하지 않는다.

## 외부 학습·관리 경계 확인

- `ManagerPolicy`/`FunctionalManagerPolicy`와
  `OpeningManagerPolicy`/`FunctionalOpeningManagerPolicy`는 외부 학습
  parameters/state를 dynamic PyTree로 받을 수 있다.
- `RuleBasedManager`의 교체, formation candidate 선택, restart taker
  scoring/sampling은 명시 key를 사용하며 custom manager로 교체할 수 있다.
- opening API는 외부 formation layout과 `[L]` 또는 `[2, L]`
  `formation_probabilities`를 검증해 받는다.
- 외부에서 작성한 ability profile은 그대로 보존된다. 무작위 초기화 경로는
  명시 key와 bounded symmetric compatibility prior를 사용한다. 그 범위를
  선수 능력의 측정 분포라고 부르지 않는다.
- manager, taker, opening formation, ability 준비는 저빈도/host 경계에
  남아 있으며 lean player transition에 bench나 dataset adapter를 넣지 않았다.

## 킥오프 측정

측정 fixture는 seed 19, 11명씩 두 팀, 11 control-step actual rollout이다.
각 시점의 public encoded movement direction을 decode하고 팀 내 55개 선수
쌍 cosine의 median과 `cosine > 0.98` 개수를 계산했다.

| 1초 시점 | 이전 y-only waypoint | 개선된 역할·슬롯 2-D waypoint | 변화 |
|---|---:|---:|---:|
| Team 0 cosine median | 0.723 | 0.679 | -0.044 |
| Team 0 `> 0.98` pairs | 7/55 | 6/55 | -1 |
| Team 1 cosine median | 0.992 | 0.909 | -0.083 |
| Team 1 `> 0.98` pairs | 39/55 | 11/55 | -28 |

전용 shape 검증은 다음 의미 계약도 확인한다.

- phase 0, phase 1, disabled shift, 범위 밖 clamp가 같은 settled target
- 첫 command phase `1/30`에서 적어도 네 역할의 nonzero waypoint
- GK waypoint 0
- midpoint x 절댓값은 `1.5 * 2.4 m` 이하
- midpoint y 절댓값은 `2.0 * 2.4 m` 이하

## 검증

- `PYTHONPATH=src pytest -q tests/test_rule_policy_realism_validation.py`
  — **13 passed**
- previous teammate invisible/offside adversarial subset와 인접 settle/carry
  검증 — **4 passed**
- `tests/test_rule_policy_pass_diagnostics.py` — **1 passed**
- 전체 CPU suite — **82 passed in 206.64 s**
- 변경 Python 파일 `py_compile` — passed
- Ruff lint — passed
- `git diff --check` — passed

invisible/offside adversarial case는 두 조건에서 모두 오래된 team episode를
새 carrier solo tenure로 상속하지 않고, 연결된 수신이 fresh-shot settle
할인에서 면제됨을 검증한다.

추가된 전용 의미 검증은 eager/JIT 양쪽에서 같은 actor의 controlled 및
loose `CONTROL/TRAP` 재접촉이 carry age를 이어 가고, handoff·unrelated
loose event가 이를 초기화하며, hidden row는 추정 증가 없이 동결함을
확인한다. 연결 수신자도 자신의 개인 soft limit에 도달하면 안전한 pass를
release하는 것도 별도로 확인한다.

### README seed 3 장면 회귀

사용자가 지적한 기존 10초 영상과 같은 seed 3,
`salida_lavolpiana` 대 `gegenpress`의 첫 두 접촉을 tracking/event
sidecar로 대조했다.

| 두 번째 toucher의 첫 carrier episode | 수정 전 영상 | 최종 절충안 | 변화 |
|---|---:|---:|---:|
| 첫 control부터 본인 PASS/다른 actor touch까지 | 4.6 s | 2.5 s | -2.1 s (-45.7%) |
| 같은 선수 control 횟수 | 9 | 3 | -6 (-66.7%) |
| tracking 공 누적 경로 | 24.44 m | 10.95 m | -13.48 m (-55.2%) |
| 해당 선수 누적 경로 | 23.63 m | 6.70 m | -16.93 m (-71.7%) |
| 해당 선수 y 순이동 | 22.60 m | 3.49 m | -19.11 m (-84.6%) |

수정 전에는 2.3–2.5초, 3.9–4.2초, 5.3–5.4초의 짧은 loose 구간마다
물리 `control_ticks`가 다시 시작되어 최대 연속 값이 1.3초에 머물렀다.
정책 soft limit 2.2초에 도달할 수 없었던 직접 원인이다. 최종 절충안에서는
동일 actor lineage가 개인 시간을 이어 세 번의 control 뒤 3.9초에 같은 팀의
다른 미드필더가 공을 이어받았다. 중간안의 0.4초 연쇄 패스도 재현되지 않았다.
이 한 장면은 회귀 증거이며 전체 경기 분포나 측정 축구 상수의 근거로
해석하지 않는다.

## 성능·컴파일 주장 경계

구조적으로 추가된 hot-path 작업은 carrier row의 scalar utility 산술과
선수별 고정 길이 waypoint vector 연산, `(P,)` int32 `carrier_age`
갱신이다. 22인 fixture에서 이 leaf는 88 byte이며, 전체 정책 state는
1,900 byte다. 새 RNG, data-dependent branch, 선수 쌍 materialization은 없다.
`kickoff_path_window_ticks`는 policy factory에서 host scalar로 미리 계산한다.

CPU public 11v11, seed 29, 32-step `make_advance` fixture를 새 compilation
cache에서 측정했다. lower+compile+첫 실행은 28.1949초, 7회 warm median은
0.0437602초(731.258 step/s), StableHLO text는 3,868,278 byte/42,615 line이다.
compiled memory analysis는 argument 4,747 byte, output 4,697 byte, temporary
155,384 byte, alias 0 byte를 보고했다. 동일 source/backend의 엄밀한 변경 전
A/B는 없으므로 성능 개선이나 무퇴행을 주장하지 않으며 GPU 수치도 주장하지 않는다.

## whole-match vmap 옵션 폐기

사용자 지시에 따라 whole-match public batching에서 `vmap`을 선택하는
옵션·flag·실행 경로는 완전히 폐기했다. public `batch_rollout`은
`jax.lax.map` 한 경로만 제공한다. 이는 현재 정책의 재현 가능한 실행 경계를
단일화한다.

이 문장은 엔진 내부의 고정 크기 지역 계산에서 `jax.vmap` 연산 자체를
금했다는 뜻이 아니다. restart 후보, 관측 행, lookup 작성 같은 내부
벡터화는 public whole-match backend 선택 옵션과 별개다.

## 전체 90분 정책 진단

seed 29 개선 후 산출물은 57,642 frame을 기록했고
`full_duration_complete`와 `regulation_complete`를 모두 만족했다. event
budget exhausted는 0이며, 실행 시작과 종료의 production source hash는
동일했다. 다만 공유 worktree가 dirty인 상태에서 얻은 결과이므로 이 절의
수치는 release-authoritative calibration이 아니라 source-stable diagnostic다.

아래는 같은 seed의 구 산출물(report v6/v7)과 새 산출물(report v7/v8)을
비교한 단일 경기 기술 통계다. 표본 한 경기의 차이를 계수 보정 성공이나
인과 효과로 해석하지 않는다.

| 진단 | 개선 전 산출물 | 개선 후 산출물 | 기술적 변화 |
|---|---:|---:|---:|
| quick-after-regain shots | 18/25 (72.00%) | 16/27 (59.26%) | 비중 -12.74%p |
| quick-after-regain goals | 3/4 (75.00%) | 2/3 (66.67%) | 비중 -8.33%p |
| sustained + open-play buildup shots | 1/25 (4.00%) | 3/27 (11.11%) | 비중 +7.11%p |
| regain 뒤 0.8초 이하 quick shots | 6 | 5 | -1 |
| regain 뒤 1.5초 이하 quick shots | 8 | 12 | +4 |
| same-actor contacts skipped | 345 | 230 | -33.3% |
| same-team pass receipt | 0.8968 | 0.9119 | +0.0151 |
| continuous control 최댓값 | 3.3 s | 3.0 s | -0.3 s |
| continuous control 3초 초과 episode | 2 | 0 | -2 |
| lateral range 10 m 초과 episode | 4 | 1 | -3 |
| lateral range p95 | 4.95 m | 5.11 m | +0.16 m |
| lateral range 평균 | 1.395 m | 1.439 m | +0.044 m |

0.8초 이하 quick shot은 하나 줄었지만 1.5초 이하는 8회에서 12회로
늘었다. 현재 settle prior의 직접 범위가 0.8초뿐이라는 점과 함께 명시적인
잔여 위험으로 남긴다. same-actor skipped contact의 33.3% 감소도 원인 분리가
없는 진단 수치일 뿐 성능 또는 현실성 개선률로 주장하지 않는다.

lateral range 10 m 초과 episode는 줄었지만 p95와 평균은 각각 4.95 m에서
5.11 m, 1.395 m에서 1.439 m로 올랐다. 따라서 lateral movement가 전 구간에서
일괄 개선됐다고 주장하지 않는다. 최종 score도 3-1에서 0-3으로 달라졌으나
단일 경기 결과를 정책 변경의 인과 효과로 해석하지 않는다.

실제 manager/report event count는 substitution 10회, formation 변경 2회,
set-piece taker 변경 54회다. 이 수치는 당시 검증 입력에 한정된 진단값이며,
삭제된 로컬 output을 현재 배포 증거로 참조하지 않는다.

## 최종 동결 검증·성능·패키징

이번 carrier-lineage 변경 직전 동결 source에서 CPU public 11v11, seed 29,
32-step fixture를 격리된 새 cache로 측정한 값은 cold 29.2999초, 7회 warm
median 0.0507851초, 630.106 step/s였다. fixture 작성 명령과 backend 상태가
완전히 보존된 paired benchmark가 아니므로 위의 현재 측정치와 산술 비교해
성능 개선률을 주장하지 않는다.

직전 source의 CPU test suite는 204.74초에 80 passed였다. 현재 source의
검증은 위 검증 절에 별도로 기록했다.

직전 동결 source에서 sdist와 wheel을 build했고 wheel metadata의 배포
버전이 `0.1.0`임을 확인했다. wheel에는 `.orig`, test, `calib`, output
경로가 포함되지 않았다.

## PASS 수신 계획과 CONTROL 의사 패스 진단

README 선두 10초 경기의 tracking과 event sidecar를 영상과 함께 다시
대조했다. 0.1초 킥오프 PASS는 제출 방향과 이동 수신점 방향의 차이가
0.015도뿐이어서 방향 좌표계나 지상 패스 역산이 원인이 아니었다. 문제는
정책이 PASS를 제출한 프레임에는 의도 수신자를 recurrent state에 기록하지
않아, 다음 loose-ball 관측에서 일반 최근접 추격수가 원래 수신자를
대체했다는 것이다. 수정 뒤에는 같은 팀의 유효한 observer row에만 현재
PASS 수신자, 도착점, ETA를 즉시 저장하고, 다음 관측의 공개된
`PASS/RELEASE/kick_applied` 계보가 실제 릴리스를 증명할 때만 유지한다.

영상의 2.0초와 5.1초 접촉은 PASS가 아니라 파란 링의 `CONTROL/TRAP`이었다.
각 공은 loose 상태로 8.29 m와 10.16 m 이동해 동료 또는 상대에게 닿았으므로,
시각적으로는 표기되지 않은 패스가 되었다. `dribble_power`는 public CONTROL
속도 척도에 이미 정규화돼 있는데 정책이 이를 다시
`kick_speed_max/control_request_speed_max` 비율로 변환해 요청을 약 3.86배
키운 단위 오류였다. 이 변환을 제거하고 설정값을 CONTROL 척도에서 그대로
사용한다.

첫 수정 뒤 위의 장거리 CONTROL 두 건과 킥오프 수신자 교체는 사라졌다.
다만 3.4초에 수신자가 새 진행 방향으로 급회전하면서, 공이 최종 터치
방향축 기준 0.331 m 뒤에 있는데도 CONTROL을 제출하는 좁은 잔여 사례가
확인됐다. 이동 목표와 CONTROL 목표는 42.5도 달랐고, 선수와 공의 거리는
3.8초 0.459 m에서 4.4초 1.229 m로 벌어져 물리 carry edge 1.21 m를 넘었다.
정책은 이제 최종 경계 보정까지 적용된 터치 방향 앞에 공이 있을 때만
주기적 드리블 CONTROL을 허용한다. 공이 뒤에 있으면 별도 상태나 난수 없이
공 상대 위치로 먼저 이동해 다시 정렬한다. 재접촉 간격을 완전히 없애는
중간안은 3.1--3.5초에 매 프레임 CONTROL을 만들어 폐기했다.

### FootballWorld 설계 경계

- 유지: 공개된 상대 운동으로 재접촉을 재허용하고 캐리어가 공 위치를
  회수하는 원칙은 오버런을 막으므로 sound하다. 패스 비행 중 한 명의
  trajectory receiver를 유지하는 원칙도 프레임별 수신자 교체를 막는다.
- 변경: 전역적인 공 직접 호밍은 전술 이동을 지우므로, 최종 터치 방향과
  공이 불정렬인 캐리어에게만 제한한다.
- 거부: 절대 공속 2.5 m/s 드리블 gate는 player-relative CONTROL 의미와
  직접 호환되지 않으며, 측정 상수로 채택할 근거도 없다.
- 유지: FootballWorld의 가산형 공 충격과 현재 공속을 보상하는 패스 솔버는
  두 실제 PASS에서 목표 이동점과 궤적 방향이 일치했으므로 변경하지 않는다.

구현은 기존 고정 shape PyTree와 observer-local 상태를 유지한다. 정렬 보정은
2차원 scalar 곱셈 두 번과 `where` 기반 회수 이동만 추가하며 새 동적 loop,
PRNG stream, 선수 쌍 행렬, 데이터 계수는 추가하지 않는다. PASS 계획도 이미
계산한 단일 선택 수신자와 도착 정보를 broadcast할 뿐 후보 계산을 반복하지
않는다.

### 동결 검증

전용 회귀 테스트 3개는 135.83초에 모두 통과했다. 각각 동일 프레임 PASS
계획과 실제 릴리스 뒤 유지, CONTROL 전용 정규화 척도, 불정렬 캐리어의
무접촉 공 회수를 검증한다. Ruff check와 format check, `git diff --check`도
통과했다.

최종 진단 영상은
`output/pass-control-alignment-fix-seed3/match.mp4`에 생성됐다. seed 3,
`salida_lavolpiana` 대 `gegenpress`, 100 control frame, 1080p, 20 fps이며
200 video frame을 완전 decode 검증했다. production Python source와 git
porcelain은 capture 시작부터 종료까지 동일했다. 실제 PASS는 0.1초, 6.3초,
9.8초에 발생했고, 직후 공 진행 방향과 해당 시점 의도 수신자 방향의 차이는
각각 0.02도, 0.65도, 1.27도였다. 앞의 두 패스는 window 안에서 지정 선수
1002와 1018이 직접 수신했다. 마지막 패스의 수신 시점은 10초 window 밖이다.
기존 8.29 m와 10.16 m CONTROL 의사 패스는 재현되지 않았다. 남은 CONTROL은
짧은 trap/recovery이고, 4.8초 골키퍼 접촉은 소유 전달이 아니라 `PARRY` 뒤
원 캐리어가 다시 회수한 장면이다.

같은 MP4를 `docs/assets/rendering/latest-kickoff-10s.mp4`로 복사해 README
최상단 링크를 교체했다. 원본과 README MP4의 SHA-256은 모두
`37799d4b6b344b3cc660c4d66f511d75dec75d8105e001a8a5466720593091ca`다.
README GIF는 이 MP4에서 960x540, 10 fps, 100 frame, 10초로 파생했으며
SHA-256은
`ac515ae25324fb70adbf20a71ec2f6b9e35043ad664600cafe4b256a9e84e763`다.

변경 전 HEAD와 현재 소스를 같은 CPU batch-1, 32-step outer `lax.map`,
7회 warm fixture로 진단했다. warm median은 0.051119초에서 0.050698초로,
처리량은 625.99에서 631.19 frame/s로 측정됐다. compiler temporary 추정은
161,272 byte에서 161,632 byte로 360 byte 늘었고 executable text는
33,870,737자에서 33,511,159자로 줄었다. cold compile+first는 32.71초와
35.51초였으나 단 한 번의 순차 측정이므로 증가를 정책의 인과 효과로
주장하지 않는다. warm 차이도 성능 개선률로 주장하지 않으며, 새 graph가
고정 shape이고 이 fixture에서 뚜렷한 rollout 손실을 보이지 않았다는
진단 경계로만 사용한다.

## CONTROL 군집과 4-2-3-1 역할선 수정

위 최종 영상의 5.5--6.8초를 lossless tracking으로 다시 검사하자, team 0의
골키퍼 1011, 센터백 1002, 중앙 미드필더 1014가 같은 공에 수렴했다. 6.3초
세 쌍의 거리는 0.948 m, 0.808 m, 1.000 m였고 6.7초 골키퍼--센터백 거리는
0.233 m까지 줄었다. 80 Hz authoritative physics substep을 20 Hz로 직접
표본화한 결과이므로 렌더 보간 착시가 아니다.

원인은 두 개의 역할 중복이었다. `state.py`가 동일 선수의 `CONTROL/TRAP`
뒤 짧은 loose 구간에도 팀 소유 lineage를 유지하는 것은 드리블 연속성을
위해 타당하다. 그러나 `policy.py`는 그 팀 latch 전체를
`reliable_pass_flight_attack`으로 취급해 마지막 CONTROL actor를 원 패서처럼
제외했다. 실제 3.9초 공 거리는 1002가 1.275 m, 1014가 13.277 m였지만 먼
1014가 수신 주자로 선택됐다. 이제 팀 소유 lineage는 공격 대형에만 넓게
사용하고, 수신자 배정과 계획 보존은 공개 last-contact가
`PASS/RELEASE/kick_applied`를 모두 증명할 때만 활성화한다. CONTROL은 기존
stale PASS 계획도 즉시 지운다.

FootballWorld의 골키퍼는 상대와의 인터셉트 거리만 비교하고, 자기 골 방향
속도가 있거나 물리 소유가 없으면 동료 회수자와 무관하게 스위프한다. 한
명의 goalkeeper과 한 명의 outfield receiver를 분리한다는 구조는 유지하지만,
이 상대 전용 경쟁과 단순 위협 gate는 동료 세 명 군집을 실제로 만들었으므로
거부했다. FootballWorld는 이미 계산한 인터셉트 거리 배열에서 직전 actor를
제외한 visible teammate 최소거리만 추가로 구한다. 더 가까운 합법 동료가
있으면 일반 sweep와 비긴급 rush를 양보하고, 관측된 팀 소유 또는 동일
CONTROL lineage 중에는 골문으로 실제 투영되는 `heading`만 위협으로 남긴다.
진짜 back-pass flight, 상대 loose ball, goal-mouth emergency는 계속 처리한다.

첫 수정 뒤 별도의 장기 2인 중첩도 확인됐다. 4-2-3-1의 -12 m 공격형
미드필드선과 -5 m 스트라이커선이 모두 forward로 분류돼 같은 offside
shoulder 목표를 받았고, 5.5--6.8초 두 선수 간 거리가 0.488--0.450 m로
유지됐다. 슬롯 비의존·앵커 상대 역할 분류는 sound하지만, 세 번째 이후의
모든 깊이선을 forward로 clip하는 동작은
4개 이상 outfield line에서 정확하지 않다. 최후방과 최전방 outfield line만
defender와 forward로 두고 모든 interior line을 midfielder로 분류했다.
포메이션 이름이나 슬롯 번호 특례는 추가하지 않았다.

수정 후 seed 3의 clean-source 최종 경기에서는 5.5--6.8초 양 팀 모두
1.5 m 안에 다른 동료가 한 명도 없었다. team 0 최소 동료 거리는 같은
구간에서 10.450 m에서 9.536 m 범위였다. PASS는 별도 검증했다. 0.1초
1014→1002는 17.817 m/s의 실제 `PASS/RELEASE`였고 2.8초 최초 후속 접촉자가
지정 수신자 1002였다. 4.4초 1002→1010도 18.402 m/s의 실제 PASS였으나,
6.9초 상대 2008이 공에서 0.31 m로 먼저 도달해 차단했다. 둘 모두 즉시
공속과 현재 수신자 방향 오차는 float 출력 한계 안에서 0도였다. 성공 패스와
정상 인터셉트를 구별하며, CONTROL을 패스 완료로 세지 않는다.

관련 5개 테스트 파일의 27개 테스트는 80.86초에 통과했고, stale PASS
계획을 주입한 CONTROL/패리 회귀도 별도로 통과했다. 최종 전체 suite는
85개 테스트가 112.97초에 통과했다. Ruff와
`git diff --check`도 통과했다. 수정 전 HEAD와 현재를 같은 CPU batch-1 정책 step,
31회 warm 조건으로 측정했을 때 compiler temporary는 모두 120,608 byte로
같았다. cost-analysis FLOP은 1,072,780에서 1,073,310으로 0.049% 늘었고,
StableHLO text는 1,314,085자에서 1,316,027자로 0.148% 늘었지만 optimized
executable text는 8,471,766자에서 8,065,795자로 줄었다. one-shot compile은
5.332초와 5.476초, warm median은 2.949 ms와 1.833 ms였다. 순차 단일-host
측정이므로 속도 개선을 인과 주장하지 않으며, temporary 증가가 없고 graph
증가가 작다는 배포 guard로만 사용한다.

최종 공개 replay는 commit `db06af54a45198af0bed952e41fc4d858b78690e`
상태에서 `output/player-cluster-pass-fix-seed3/match.mp4`로 생성했다. seed 3,
100 control frame, 200 video frame, 1920x1080, 20 fps, 10초이며 전 프레임
decode 검증을 통과했다. source authority와 git porcelain은 capture 전후 모두
clean이었다. MP4 SHA-256은
`1932b3a7b385f0b276f160623fcff3b12690001f53bdc249636f18defe9183df`다.
같은 byte를 `docs/assets/rendering/latest-kickoff-10s.mp4`에 배치했고 README
GIF는 그 MP4에서 960x540, 10 fps, 100 frame으로만 파생했다. GIF SHA-256은
`17ca713ddfc4f86b98d09a2999c74e36b94c5ee9bb4ab615a2abebf0335e4d77`다.

## 지정 수신자의 동적 패스 궤적 선취

seed 3의 이전 clean replay에서 3번(slot 2, id 1002)이 11번(slot 10,
id 1010)을 지정해 패스했지만, 11번은 release-time 종착점
근처에서 기다렸고 18번(slot 17, id 2008)이 공을 먼저 차단했다. 원인은
수신자 identity뿐 아니라 최초 예상 종착 좌표와 ETA까지 비행 중 보존한
것이다. 상태 ETA는 감소했지만 예측 비행시간 전체가 지나기 전에는 더 이른
접촉점을 사용하지 못했다.

FootballWorld가 live pass에서 공의 현재 궤적을 매 프레임 다시 계산해
`receive_runner`를 실제 접촉점으로 보내는 원칙은 타당하므로 상속했다. 다만
FootballWorld는 제출된 PASS의 명시적 intended receiver receipt가 있으므로,
매 프레임 가장 가까운 팀원으로 수신자 identity까지 바꾸는 동작은 거부했다.
공개 `PASS/RELEASE/kick_applied` provenance가 유지되는 동안 지정 선수는
고정하고, 그 선수가 현재 공 궤적에서 처음 도달할 수 있는 위치와 ETA만
매 프레임 재계산한다. 기존 10-sample, 2.5초 고정-shape forecast를 그대로
재사용하며 새 계수, 상태, loop, 선수 쌍 tensor는 추가하지 않았다.

같은 seed의 수정 후 진단에서는 경기 인과가 바뀌어 해당 패스가 4.4초가
아니라 3.3초에 발생했다. 3번의 실제 `PASS/RELEASE` 뒤 11번은 공에서
33.046 m 떨어진 위치에서 전진해 4.4초 속도 8.045 m/s, 현재 공 방향
cosine 0.995를 기록했고 5.0초 최초 후속 접촉에서 `CONTROL/TRAP`했다.
release 이후 이동은 9.293 m였다. 같은 시점 18번은 공에서 8.337 m,
11번에서 7.373 m 떨어져 접촉하지 못했다. 비행 중 90도 이상 방향 반전은
없었고, 4.4초 이후 양 팀 최소 동료 거리는 각각 6.512 m와 6.199 m로 새
군집도 없었다.

수신자 identity 유지와 궤적 선취를 함께 고정한 focused policy 테스트 4개,
평면 intent-ring과 기존 renderer/font 테스트 4개, 전체 86개 테스트가 모두
통과했다. 직전 commit과 같은 CPU batch-1 policy step을 31회 warm 조건으로
비교했을 때 StableHLO text는 1,316,027자에서 1,314,086자로, cost-analysis
FLOP은 1,073,310에서 1,072,848로, compiler temporary는 120,608 byte에서
120,096 byte로 줄었다. one-shot compile은 5.590초와 5.277초, warm median은
4.705 ms와 2.748 ms였지만 순차 단일-host 측정이므로 속도 개선률은 인과
주장하지 않는다. 도착점 보존 연산 제거가 graph나 memory를 늘리지 않았다는
배포 guard로만 사용한다.


최종 clean-source replay는 commit
`54b4d91571216612236c5f07be2087a88b32cf70`에서
`output/pass-receiver-ring-fix-seed3/match.mp4`로 생성했다. 100 control
frame, 200 video frame, 1920x1080, 20 fps, 10초이며 전체 decode 검증과
capture 전후 source/git clean 검증을 통과했다. MP4 SHA-256은
`e54ea746d813e037863d5bcd95fc871de1f527f7bbc3a368cdfe8cb92461fe6b`,
event는 `0c697944598fb7d362c9178437bbaf08685a4f57d5b9b13c3a3b68c5ad8b2666`,
tracking은 `92dd85c06061c26c9df2fb40740172579d41aa070f542e43a35726f363c24261`다.
README MP4는 같은 byte이며 960x540, 10 fps GIF SHA-256은
`df218478767101b5ceb357a9489ca5f1ef2a90c08b7009255205e6ebf7062dec`다.


## 낮은 바운스 loose-ball claimant 유지 (2026-09-10)

seed 3 전체 경기의 54:01.2--54:01.3에서 Team 0의 기존 claimant slot 10, 새로 선택된 slot 9, Team 1 압박자 slot 19가 공 0.35 m 안에 동시에 들어왔다. 세 역할은 달랐지만 같은 팀 claimant가 0.1초 사이 바뀐 것은 전술적 인계가 아니었다. 공은 반지름 위 5.7 mm, 수직속도 0.202 m/s인 미세 바운스여서 CONTROL 가능한 `ground_ball`이면서 supported-ground forecast 대상은 아니었다. 이 분기에서 과거 claimant hysteresis가 사라지고 raw nearest player가 slot 9를 불러들였다.

유지한 sound contract는 관측된 공에 팀별 primary claimant 하나만 배정하는 것이다. 그러나 supported-ground forecast 밖의 저고도 미세 바운스에서 raw-nearest fallback은 기존 claimant의 안정성을 깨뜨렸다. FootballWorld는 기존 claimant가 공개 candidate이고 환경의 실제 `carry_radius_m + ball.radius` 접촉 범위 안에 있을 때 그 identity를 유지한다. 이 반경은 새 정책 계수나 측정 축구 상수가 아니라 환경의 기존 물리 도달 envelope다. 목표점은 계속 관측 공 중심이므로 trajectory attack과 swept-contact authority는 바뀌지 않는다.

실제 상대 좌표, 속도, 체력과 5.7 mm 바운스를 고정한 회귀는 수정 전 Team 0 claimant가 10에서 9로 바뀌며 실패했고 수정 후 10을 유지했다. 또한 claimant가 이 물리 반경 안에 있을 때 같은 팀 비담당 선수가 반경 안으로 들어오면 공 반대 방향으로 공간을 비운다. 양 팀은 observer-local 후보 집합에서 각자 한 명을 선택하므로 상대 압박자는 그대로 경합한다. 관련 pass-plan/CONTROL 테스트 5개는 68.85초에 통과했고 별도 JIT 검증에서도 action과 recurrent state shape는 각각 `(22, 2)`, `(22,)`로 유지됐다.

같은 seed 3 전체 경기 report-only 비교에서 수정 전 loose-ball 29,271프레임 중 공 0.6 m 안에 같은 팀 두 명과 상대 한 명이 동시에 있던 프레임은 5개였고, 54:01.1--54:01.4에는 4프레임 연속이었다. 수정 후 loose-ball 29,233프레임에서는 해당 3인 군집이 0개였다. 같은 팀 두 명이 공 0.6 m 안에 있던 프레임은 16개에서 11개로 줄었으며 수정 후에는 모두 연속되지 않는 단일 프레임이었다. 전체 57,424프레임은 regulation complete로 종료됐고 `event_budget_exhausted_count`는 0이었다. 이는 물리 이벤트 한도를 늘리지 않고 정책 간격 조절로 실패를 제거한 결과다. 같은 CPU batch-1 recurrent policy step을 31회 warm 조건으로 측정한 배포 guard는 StableHLO text 1,324,671 byte, compiler temporary 121,888 byte, cost-analysis FLOP 1,890,072, one-shot compile 5.399초, warm median 3.157 ms였다. 순차 단일-host 측정이므로 속도 개선률을 인과 주장하지 않는다. 최종 authoritative 렌더 receipt는 clean revision 검증 뒤 기록한다.


## Tactical width, multi-presser roles, and kickoff variation (2026-09-10)

This phase retains FootballWorld's useful structural contracts: formation-relative
roles, ranked primary/secondary pressure with a distinct cover player, seeded
restart decisions, and conventional behind-ball poses for ordinary foot
restarts. It rejects three behaviours that do not fit the current contracts. A dense
externally calibrated positional field would require different observations and
coefficient evidence, so FootballWorld keeps its bounded formation-relative ball
follow. A circular-player behind-ball kickoff pose is rejected for kickoff only
because FootballWorld's oriented solid torso makes that pose one-directional.
Finally, an exact kickoff receiver slot is not a football law
or stable policy semantic, so it is not retained as a regression oracle.

Attacking width now keeps at most one stable public wide-role player on each
side at or beyond the formation-scaled lane. It does not pin players to the
touchline and does not select an arbitrary positive-side runner for a central
ball. Settled pressure is profile-specific: salida lavolpiana and catenaccio use
one direct presser, juego de posicion uses two, zona mista retains one presser plus its
marker-oriented hybrid cover, and gegenpress may use three in the defending half but is capped at two high upfield. The next
ranked player remains cover. The existing immediate counterpress window remains
one primary presser plus outlet cover rather than becoming a swarm. Supporting
pressers use a 0.30 s visible-ball lead and a 3.0 m goal-side gap. Both are
explicit DESIGN_PRIOR values, not measured football constants.

Kickoff takers now stand laterally at exact ball-plus-capsule clearance and face
the ball. The side is derived from y-reflection-equivariant public state; a
perfectly symmetric state keeps the previously validated forward-side fallback
because no non-zero deterministic side can preserve reflection equivariance.
The policy path accepts tangent trajectories while still rejecting directions
that head back through the taker's capsule. It makes a seed-keyed choice between
a stochastic teammate receiver and a receiver-free territorial aerial kick.
The default territorial probability 0.18 and target distance 30.0 m are explicit
DESIGN_PRIOR controls and are not presented as measured professional kickoff
rates.

The environment itself does not reduce kick power by body-relative angle.
Foot-release speed remains normalized power times the existing 34.76 m/s cap,
and restart direction projection applies only to penalty kicks and throw-ins,
not kickoffs. An adversarial test executes otherwise identical maximum-power
face-on and 90-degree side-on kickoff actions and requires equal post-step ball
speed. Backward-through-body trajectories are excluded by contact geometry, not
by a hidden power coefficient.

The seed-29 opening still selects roster slot 6 under the short-kick branch.
That change is intentional: the stable semantic is a seeded eligible receiver,
not one historical formation slot. The regression follows the selected player
ID and requires that player to acquire actual environment possession by control
tick 40; passive contact alone is explicitly insufficient. In the final trace
the player first contacts the loose ball before later controlling it, so the
earlier tick-25 loose-ball observation was an intermediate state, not a failed
pass. A separate mirrored opening requires the same receiver identity, equal
power and launch, equal x direction, and negated y direction.

FootballWorld preserves its reproducibility boundary: non-partitionable
Threefry is required for seed receipts. The opening candidate
builder now validates a supplied sampling key before its first `fold_in`, and
the policy-driven match creator replaces its shape-only check with the same
shared validator. RBG, unsafe-RBG, and partitionable Threefry therefore fail
closed before they can alter candidate abilities or lineup selection. This is
host-only validation and adds nothing to the lean transition graph.

All rule-policy random floating draws now specify causal `float32` output;
categorical logits and the one x64-sensitive chaser index are likewise made
explicit. This rejects JAX's process-global default dtype as an unrecorded source
of policy variation. An independent-process validation compares x64 off/on for
the same Threefry seed and requires exact equality of sampled opening ability
leaves, registration, starters, placement, formation, first-frame intents, and
kickoff receiver IDs.

The renderer supports isolated parallel policy matrices. Five tactical plans
produce ten unordered distinct-plan pairings; the default two-leg mode reverses
team slots for twenty matches and adds five self-play controls for the complete
25-cell ordered space. `--matrix-workers` bounds concurrent child processes.
Every child retains logs and its own match report, and the parent fails closed
if any child fails. The following historical run explicitly used a one-leg,
distinct-plan-only diagnostic: the final source-stable seed-29,
100-control-frame, one-leg CPU matrix completed all ten pairings and all ten
child reports with zero failures at
`output/rule-policy-pair-matrix-final-v2-20260910/matrix-summary.json`
(SHA-256
`74b54ed20cc4f493e52d23be71ee1f8cd31f1f7ffffa7442fbfdaa7a1ec13baf`).
It is a cross-style execution diagnostic, not evidence of tactical superiority
or full-match rate calibration.

Final focused CPU checks passed: ten kickoff/render-legibility tests, three
open-field marker tests, 21 public randomness-contract tests, one x64
independent-process selection test, and three managed-opening regressions. The
full public suite passed 119/119 on the final source in 171.83 s; the three
managed-opening regressions and expanded x64 probe also passed. Ruff and
`git diff --check` were clean. This session's default JAX runtime could not collect its requested GPU smoke
because it attempted the ROCm backend and lacked `GpuAllocatorConfig`; this is
a runtime-package limitation, not a policy failure. Independently, the bug-audit
session's latest notice reports that its focused marker/seed-29 set passed 3/3
on both CPU and free GPU 1.

## Rule-policy rollout efficiency pass (2026-09-10)

This pass preserves FootballWorld's use of squared distance for nearest-player
ranking. That behavior is sound because every affected value is consumed only
by an `argmin`/`argmax` order or by comparison with the square of the same
non-negative physical radius. FootballWorld applies it to the two observer-by-
player pressure/claimant matrices and to the fixed three-channel box-marker
assignment. It also reuses the already computed pass distance, predicted
opponent positions, and goal-threat rank instead of spelling the same
intermediate twice. Fixed shapes, observer-local information, tie order,
environment contact authority, and every policy coefficient are unchanged.

A causal `lax.cond` around the rare box assignment was tested
and rejected. On this FootballWorld graph it did not reproduce a warm-runtime
gain, while StableHLO grew by 1,156 characters and 22 lines, estimated FLOP by
308, and compiler temporary memory by 128 bytes. Leaving the small fixed loop
dense is therefore the better current compilation/memory tradeoff; the rejected
branch is not left dormant in production source.

For one actual 22-player policy step, a fresh-process structural comparison
against an exact temporary source baseline changed StableHLO from 1,343,410 to
1,343,836 characters (+0.032%), kept estimated FLOP at 1,916,002 and compiler
temporary memory at 119,968 bytes, and reduced transcendental operations from
38,784 to 36,364 (-2,420, -6.24%). The complete output-tree SHA-256 remained
`c7fb5251ec82736bab5d424c000c4e01c9977eef89197db2c1aabb23f96a54e5`.
Three alternating fresh processes per variant measured policy-step median
2.629 ms for the baseline and 2.349 ms for the candidate. This is a directional
microbenchmark, not a claimed production speedup.

A separate two-process-per-variant, nine-repeat 256-step rollout comparison
also produced one identical final-tree SHA-256,
`84a4195d052c451ea62c5216f26eb51c99866a2c00df158b477a2ebab94f1f07`.
Its median was 429.326 ms for the baseline and 433.173 ms for the candidate;
the opposite 0.90% sign is treated as host noise/environment dominance, so no
end-to-end speedup or slowdown is attributed. Compile medians were 17.464 s and
17.394 s respectively.

Final semantic validation passed 172/172 CPU tests in 403.90 s. A source-stable
seed-29 matrix then completed all ten unordered tactical-plan pairings and all
ten child reports with zero failures. Its summary is
`output/rule-policy-efficiency-matrix-v2-20260910/matrix-summary.json`
(SHA-256
`363e52da76adbc19e065f54ccc7c4571c105909491d221b4cff4b25dd68dfa08`).
These are implementation-efficiency diagnostics and do not introduce or
relabel any coefficient as a measured football constant.

## 2026-09-10 movement-aware service and off-ball activation

The 25-cell seed-29 diagnostic matrix showed that static team width was not
being converted into enough box entries. Ordinary passes also moved backward
in the opposition half too often. This change keeps fixed-shape candidate
arrays, physical lane/completion checks, prospective offside checks, stamina
limits, and the existing moving-receiver arrival calculation.

The policy now exposes arrival lead weight/time/distance in
`RulePolicyConfig`, increases ordinary off-ball cruise and urgent forward-run
power, treats the episode-stable designated forward as urgent, increases the
bounded receiver preference and shape displacement for coordinated attacks,
and opens cross consideration from a moderately less extreme wide position.
A score-only backward-pass cost gains extra weight beyond midfield but fades to
zero under full pressure, so it cannot remove the only safe outlet. No pass,
cross, or run gains eligibility from these values.

All new and changed coefficients are FootballWorld DESIGN_PRIOR values, not
measured football constants. They must be evaluated from paired, slot-reversed
report-only matches before calibration. The implementation adds three scalar
configuration leaves and scalar/vector arithmetic on the already-materialized
carrier row; it adds no player-pair tensor, dynamic loop, host callback, or
render/report work to the policy step.

Focused validation receipts:

- 38 public policy/pass/matrix/render tests passed with
  `PYTHONPATH=src JAX_PLATFORMS=cpu`.
- 7 adversarial kickoff/mirror/power/receiver/open-field tests passed.
- The roster-refresh validation now supplies the required `(P,) int32`
  `carrier_age` fixture and verifies that a changed identity resets it; the
  focused test passed.
- Ruff passed on the changed policy and focused test files.

## 2026-09-11 integration blocker resolution

The currently approved repository contract treats FootballWorld's own tested
implementation and evidence as authoritative; that contract change is retained
by explicit user approval. Existing fixed-shape transition semantics, physical
kick power, mirrored kickoff geometry, match verification, distinct-fixture
league scoring, and host-only report aggregation remain unchanged.

Four correctness boundaries changed. The default two-leg plan matrix now
includes the five self-play controls and therefore emits all 25 ordered cells.
An explicit 20-cell opt-out remains available, but its JSON quality is false and
its HTML shows the observed/expected cell count. Multi-report JSON and HTML are
completed in a sibling staging directory and published with one directory
rename; an existing generation is never overwritten. The kickoff lateral stance
now derives handedness from the permutation-invariant cubic lateral moment of
the restart team rather than roster storage indices, while retaining y-mirror
equivariance and the symmetric fallback. Finally, opening registration maxima
are range-checked in the host builder before int32 conversion, preventing
`2**32 + n` from aliasing to `n`.

The report and opening changes are host-only. The kickoff change replaces one
fixed-size weighted reduction with one fixed-size cubic reduction and adds no
pairwise tensor, callback, dynamic shape, or data-dependent Python branch to the
JAX step. These changes introduce no football coefficient and make no measured
football claim. Focused CPU validation passed 31 tests covering default and
opt-out matrix construction, complete/incomplete report labels, immutable
report publication, opening narrowing, kickoff roster permutation and mirror
equivariance, seed-29 reception, and restart execution. Ruff and
`git diff --check` also passed before the broad integration suite.
The broad integration suite then passed 241/241 CPU tests in 417.58 s with
`PYTHONPATH=src:. JAX_PLATFORMS=cpu`; the repository-root path component is
required only for the separate `research` namespace.

## 2026-09-11 seed-29 action-mix and matrix-control revision

The completed 25-cell seed-29 report exposed two confounders: independently
sampled identity-keyed abilities could be read as a team-slot/tactical effect,
and abundant safe outlets dominated the macro action distribution. The matrix
now defaults to identical fixed demo-candidate ability bundles for both teams.
This control changes neither explicit match fixtures nor player physics, and
`--no-matrix-equal-roster-abilities` retains the former independent-sampling
experiment. Formation and lineup selection remain tactical-policy decisions.

Pass legality, completion estimates, receiver ranking, cross ranking, offside
checks, and the fixed-shape service arrays are retained. A new scalar
`pass_macro_value_scale` acts only after service ranking, at the existing
PASS/SHOT/DRIBBLE/CLEAR categorical choice. `shot_value_gain` changes from
0.72 to 0.92. Salida Lavolpiana's progressive-pass profile changes from 0.08
to 0.14 and its attack-depth scale from 1.04 to 1.08, preserving its pivot
drop, width, fullback, and defensive profile. These values are DESIGN_PRIOR
tuning responses, not measured football constants.

A fixed-ability seed-29, 1,200-second diagnostic between Salida Lavolpiana and
Juego de Posicion produced 188 and 198 pass attempts, 90.4% and 91.9%
completion, two Salida shots, ten combined cross-control signatures, and a
1-0 score. This partial match establishes only direction and event integrity;
the complete 25-cell matrix is still required for policy-quality calibration.
The renderer separately moves score/team text to normalized y=0.965 and the
clock to y=0.915, a 27-pixel baseline separation at 540p, without changing the
replay time axis or event semantics.

The matrix launcher default subsequently changes from two to 25 independent
child processes at explicit user request, allowing every ordered policy cell
to start together. This is process concurrency rather than a JAX `vmap` and
does not change match semantics. Eight concurrent CPU children in the v1 run
used approximately 2.9 GB RSS each; extrapolating that observation gives about
73 GB for 25 children, excluding filesystem cache and aggregation overhead.
`--matrix-workers` remains available for hosts that cannot support that load.

## 2026-09-11 tactical formation categorical calibration

The opening order remains abilities, tactical-plan selection, formation
selection, then starting-XI placement. Formation feasibility, role-weighted
ability fit, preferred-position distance, caller probabilities, content-keyed
catalog equivariance, and exact authored formations are retained. The former
unit-scale Gumbel draw obscured the much smaller tactical/roster logit
differences: over 100 seeds, all five tactical plans selected the same
formation in 92% of team-0 rows and 88% of team-1 rows.

`RuleOpeningManagerConfig.formation_choice_temperature` is now an explicit
positive DESIGN_PRIOR with default 0.075. The deterministic logit is divided
by this temperature before the existing Gumbel-max categorical. This changes
only the start-only opening manager and adds no rollout-graph work. With equal
abilities and equal formation priors over 100 seeds, the modal selections were
4-2-3-1 for Salida Lavolpiana (57%/61%), 4-3-3 for Juego de Posicion
(55%/58%), 4-2-3-1 for Gegenpress (50%/51%), 4-2-3-1 for Catenaccio
(68%/73%), and 4-3-3 for Zona Mista (95%/91%), where each pair is team 0/team
1. Alternatives remained reachable for every plan; these frequencies describe
the current demo catalog and are not measured real-football rates.

## 2026-09-11 starting-formation catalog and structural fit

Explicit fixture formations, including authored coordinates, remain exact and
bypass automatic selection. Ability-aware tactical-plan selection still runs
before formation selection, and the formation categorical still combines the
caller's prior, role-weighted XI fit, structural tactical fit, and keyed Gumbel
variation. These retained behaviors make an omitted setup depend on the
available squad while preserving reproducible sampling and exact user input.

The demo starting catalog expands from 4-3-3, 4-2-3-1, and 3-2-5 to also
include 4-1-4-1, 3-4-2-1, 5-3-2, and asymmetric 4-4-2. Automatic selection
assigns zero prior probability to 3-2-5 because it is modeled as an
in-possession target shape, not a default starting shape; an explicit fixture
may still request it. Structural fit now additionally represents advanced
player share, depth span, mirror-invariant lateral asymmetry, and classified
forward share. This gives Gegenpress a direct high-line/front-number signal,
Catenaccio a five-defender candidate, and Zona Mista an asymmetric candidate.
The weights are transparent DESIGN_PRIORS used for ranking, not measured
football constants.

Starting-XI construction changes from slot-ordered greedy assignment to a
fixed-shape global-pair greedy assignment. Each iteration selects the highest
scoring remaining compatible slot/player pair, so permuting an otherwise
identical formation's slot array no longer changes the physical player-to-slot
assignment. Goalkeeper compatibility, preferred-position distance,
role-weighted ability, identity-keyed lineup variation, and fixed authored
starters remain intact.

With equal candidate abilities and the default starting prior over 100 seeds,
the team-0/team-1 modal selections are 4-2-3-1 for Salida Lavolpiana
(58%/57%), 4-3-3 for Juego de Posicion (92%/86%), 4-2-3-1 for Gegenpress
(91%/91%), 5-3-2 for Catenaccio (68%/75%), and asymmetric 4-4-2 for Zona
Mista (61%/63%). The possession 3-2-5 was selected zero times, as required by
its default zero prior, while every tactic retained at least one sampled
alternative. These are seed-distribution diagnostics for this catalog, not
claims about real-world formation frequencies.

On the same CPU process shape, compiling the start-only opening decision took
1.280 seconds for three formations and 1.267 seconds for seven; mean warm
execution over three measured calls increased from 1.548 ms to 2.093 ms.
Compiler cost analysis estimated 349,126 versus 798,934 FLOPs. The larger
catalog therefore increases one-time opening work but does not add operations
to the per-tick physics or rule-policy rollout graph.

## 2026-09-11 observable phase-responsive manager formations

The existing low-frequency manager boundary, environment-authoritative
formation command, five-minute formation hold, incumbent bonus, keyed
variation, and score/time/fitness response are retained. Formation changes
remain legal manager transactions at observable restart boundaries rather
than hidden per-tick state changes. This prevents formation oscillation and
keeps manager reasoning outside the lean physics/player-policy step.

The manager now receives the same fixed two-team tactical plans as the player
and opening policies. At each new non-goalkeeper-hold restart, the restart
owner is used as the observable projection of the next possession phase. An
own restart adds tactic-conditioned attack-depth and width fit; an opponent
restart adds defender-share and compact-width fit. The previous chase/protect
score, match progress, fitness, catalog prior, and hold terms remain active,
so phase is one input rather than a scripted formation schedule. No live
possession is invented when the restart has no owner.

The structural principles follow FIFA Training Centre's separation of
[in-possession](https://www.fifatrainingcentre.com/en/resources-tools/football-language/in-possession/index.php)
maintenance/progression from
[out-of-possession](https://www.fifatrainingcentre.com/en/resources-tools/football-language/out-of-possession/index.php)
pressure and team shape, and its description of defensive organisation through
[horizontal and vertical compactness](https://www.fifatrainingcentre.com/en/game/game-analysis/out-of-possession/team-organisation--out-of-possession-.php).
The FA's coaching overview likewise treats the regain/loss moment as a distinct
[transition phase](https://www.thefa.com/bootroom/resources/coaching/what-is-transition).
The resulting four-value tactic table and default phase gain are declared
DESIGN_PRIORS, not measured transition frequencies or universal football
constants.

The full-match example explicitly constructs this configured manager and
publishes its complete configuration and hash in replay metadata and the
summary. Explicit fixture tactics continue to remain fixed; omitted tactics
continue to use the existing roster-conditioned opening selection.

Full-match seed-29 Gegenpress self-play exposed a deterministic liveness
interaction at control tick 45,205: changing formation during a continuous
free-kick approach preserved the old taker's causal position while projecting
the remaining players into the new layout, and the restart never regained its
layout-ready gate. The manager therefore limits formation commands to kickoffs
and goal kicks, whose restart projectors fully reset the taker and team shape.
Free kicks, offside restarts, corners, and throw-ins retain their continuous
approach contract without an in-flight formation mutation. This is a
correctness restriction on command timing, not a change to formation scoring;
score, phase, fitness, tactic, hysteresis, and keyed choice remain active at
the safe boundaries.

The first complete v2 matrix also exposed a separate open-play liveness defect
in the 3-1 Gegenpress--Zona Mista cell. A single live, stationary loose ball
occupied the same 2 m density cell for 993.7 seconds from match clock 1783.4
to 2776.1. The assigned Team-0 claimant remained about 4 cm from the ball while
the nearest opponent stayed 3.7--5.0 m away. The policy nevertheless converted
CONTROL to CHALLENGE solely because the previous contact team was the
opponent; repeated passive deflections never established possession.

Previous-team provenance remains part of contested-loose-ball intent, but it
now selects CHALLENGE only when an observed participating opponent is also
within the environment's existing physical challenge envelope of the ball.
An uncontested claimant uses CONTROL, while a genuinely co-located opponent
retains CHALLENGE and the deep defensive clearance override remains unchanged.
This adds one reduction over an already-computed player-to-ball distance
matrix and no new fitted coefficient.

The host-only match report now audits every capture for policy-pathology
signals before a tactical result is interpreted. It records the longest live
loose-ball run, the longest nearly stationary live loose-ball run, the
dominant 2 m ball-density cell, dismissal transitions, consecutive repeated
contact signatures, and realized pass receipt in fixed 15-minute windows.
The multi-match report carries every child anomaly into a linked review table.

Thresholds for these warnings are deliberately labelled diagnostic design
priors rather than measured football constants: 5 seconds for a stationary
live loose ball, 15 seconds for any live loose ball, at least 60 seconds and
2% of live time in one density cell, 30 seconds for one consecutive event
signature, and three dismissals for one team. A 5 percentage-point fitted
pass-receipt decline over the six 15-minute windows is considered only when at
least four windows have ten realized attempts. A full 15-minute window with no more
than five combined realized passes is flagged only when both neighboring
windows contain at least twenty. All raw counts, times, coordinates, and
thresholds remain in the JSON so reviewers can reject or revise a warning.

Recomputing the original defective 3-1 capture produces stationary-loose,
prolonged-loose, spatial-overconcentration, and 30--45-minute pass-activity
collapse findings. Recomputing the same full-match fixture after the policy
fix produces none of those warnings; its longest loose and stationary-loose
runs are 6.6 and 0.2 seconds respectively. These comparisons validate anomaly
detection against a known simulation defect, not against a claim about normal
professional-match distributions.

The temporal pass guard is informed by the hash-verified seven-match DFL
receipt `calib/policy/artifacts/dfl-pass-completion-by-time-v1.json`. Pooled
DFL open-play `Pass`/`Cross` rows fell from 81.05% provider completion in the
first half (2,421/2,987) to 75.54% in the second (1,797/2,379), a 5.52
percentage-point fall; the median paired match fall was 5.58 points. Its
attempt-weighted six-window linear trend implies a 6.75-point first-to-last
decline. The user selected 5 percentage points as the FootballWorld
degradation allowance.
This only informs the temporal-delta guard: DFL `Evaluation` and the report's
same-team-next-distinct-contact receipt are different estimands, so their
absolute percentages are not equated or fitted.

## 2026-09-11 target-aligned pass and shot-quality calibration

Pass and cross legality, completion estimation, receiver sampling, moving
targets, offside checks, physical execution, and the independent receiver and
macro PRNG streams remain unchanged. The PASS macro utility now uses the
unboosted value of the service that the policy would actually execute, rather
than the maximum value of a possibly different eligible service. This prevents
a weak sampled outlet from borrowing a phantom best receiver's score. The
cross-selection multiplier still affects only which service is sampled. PASS
is also explicitly unavailable when no service is eligible; the former clipped
zero utility could otherwise leave a negligible but nonzero categorical path.

The default shot range preference midpoint changes from 24 m to 20 m and the
shot value multiplier from 0.92 to 1.15. These are DESIGN_PRIOR calibration
values, not measured shot-conversion constants. On four fixed CPU seed pairs
over 4,500 steps, the retained target-aligned rule plus these shot controls
changed realized passes from 589 to 570, realized shots from 4 to 5, mean shot
distance from 18.52 m to 14.84 m, and the same-team next-contact pass proxy
from 92.15% to 94.16%. Forward pass share remained 37.2% versus 37.5%.
The small shot sample establishes only a candidate direction; the full
25-cell matrix is the acceptance test.

A rejected nonlinear pass-utility exponent candidate reduced passes to 569
but also reduced forward share to 36.0%, left shots at three, and increased
mean shot distance to 21.78 m on the same short horizon. It was removed rather
than accepting lower pass volume as sufficient evidence. To make this quality
criterion visible in subsequent reviews, match reports now publish team shot
distance and penalty-area origin counts plus forward and attacking-third pass
attempt/receipt counts. Tactical matrix reports aggregate those exact-event
and submitted-direction diagnostics alongside on-target shots, crosses, and
line-break proxies.

The clean `fb14e92` seed-29 matrix completed all 25 ordered 90-minute cells in
about six wall-clock minutes with 25 CPU child processes. Across 50 team
appearances it produced 368 realized shots, 216 on-target shots, and 49,081
passes. Mean shot distance was 17.55 m (realized-shot weighted), compared with
approximately 18.8 m in the preceding matrix; the on-target share increased
from approximately 56.5% to 58.7%. Total shot count nevertheless fell from 398,
so the change removed low-quality shots without yet increasing the absolute
high-quality-shot volume. Aggregate 15-minute pass-receipt trends ranged from
-1.83 to +0.81 percentage points by tactic and no dismissal occurred.

That matrix also made an attacking-third defect measurable: 69--80% of each
tactic's realized attacking-third releases were submitted backward. Reports
therefore add a host-only warning at at least 20 attacking-third passes and a
65% backward share. This review threshold is a diagnostic DESIGN_PRIOR, not a
provider-derived football constant, and raw attempts remain in the JSON.

An attempted central final-third box-support rule was removed after the full
25-cell matrix contradicted its short-profile signal. Although four 4,500-step
seed pairs increased shots from five to eight, the full matrix reduced total
shots from 368 to 364, on-target shots from 216 to 200, and penalty-area
entries from 104 to 99. Box-origin shots increased only from 195 to 202. The
existing wide-ball central/far-post/cutback support, formation shape, and
observable Law-11 cap therefore remain unchanged.

Three additional coefficient-only candidates were rejected. Increasing shot
gain from 1.15 to 1.35 left shots at five and worsened mean distance from
14.84 m to 16.72 m. Reducing the PASS macro scale to 0.65 reduced shots to one;
reducing it to 0.50 raised shots to six but removed 14.8% of forward passes and
worsened mean shot distance to 19.60 m. Increasing only the advanced backward
penalty reduced pass volume but did not create an attacking-third forward pass
in the short sample. None of those defaults was changed.

The DFL attacking-third check joins each open-play Pass/Cross leaf to its
event's `X-Source-Position`, actual `EventTime`, the latest chronological
GameSection kickoff, and that kickoff's TeamLeft/TeamRight attacking direction.
Each qualifying EventId is counted once. Across the same seven hash-verified
event files, 921 attacking-third passes were 335 backward (36.37%), 259 lateral
(28.12%), and 327 forward (35.50%); provider completion was 607/921 (65.91%).
FootballWorld's 72--82% backward share is therefore a material directional
departure, while its much higher receipt percentage remains a different
estimand. The ignored private-data extractor
`calib/policy/extract_dfl_attacking_third_passes.py` publishes the complete
per-match counts and source hashes to
`calib/policy/artifacts/dfl-attacking-third-pass-direction-v1.json`. It also
fails closed unless every file's 5,366 total open-play attempts and 4,218
provider completions agree with the independently generated 15-minute receipt.

A backward-service macro candidate retained 75% of PASS utility when an
unpressured carrier started in the attacking third, fading continuously to the
original value under maximum pressure. On four fixed 4,500-step seed pairs it
reduced attacking-third backward passes from 12 to 8 and all passes from 570
to 560, while retaining 206/210 forward passes and all five shots. A stronger
0.60 value produced the identical short trajectory.

The full 25-cell seed-29 matrix rejected the candidate. Relative to the prior
full matrix, attacking-third backward share improved only from 1,953/2,538
(76.95%) to 1,852/2,481 (74.65%). Total shots fell from 364 to 361, on-target
shots from 200 to 185, and box-origin shots from 202 to 198. Relative to the
last accepted pre-central-support matrix, the on-target deficit was 31
(185 versus 216). The result indicates suppressed attacking actions rather
than replacement with purposeful progression, so the macro control and its
hot-path operations were removed. The report produced no liveness, repeated
event, dismissal, spatial-concentration, pass-collapse, or temporal pass-trend
warnings; all 38 warnings were the still-unresolved attacking-third backward
concentration diagnostic.

Joining the same matrix's exact open-play PASS contacts to tracking rows found
that 76.3% of backward releases from the attacking third had no visible,
active, onside teammate at least one metre ahead of the carrier. For forward
releases the corresponding no-ahead-player share was 27.9%. This host-side
causal snapshot supports a missing-support diagnosis, but it is not a tracking
provider fit. Three movement candidates were rejected on the four-seed
4,500-step profile: one staggered forward link produced 579 passes and four
shots; activating the selected forward's pattern run before it was ahead
produced 578 and three; increasing attacking block follow from 0.38 to 0.50
produced 598 and two. The accepted baseline produced 570 and five. All three
also increased attacking-third backward releases, so none remains in source.

The next candidate changes only shot macro-choice selectivity. The existing
observation-only quality `q` is multiplied by `1 + 2q^4`, making the relative
increase 1.6% at `q=0.3`, 48.0% at `q=0.7`, and 131.2% at `q=0.9`. The gain is
a declared DESIGN_PRIOR, not xG calibration; direction, power, noise, shot
physics, and the quality ordering are retained. The four-seed profile retained
five shots and changed mean shot distance from 14.84 m to 14.47 m, with 571
passes versus 570. This neutral-to-positive short screen requires the complete
matrix to determine whether absolute on-target and box-origin shots improve.

The clean `c67d3c1` seed-29 matrix completed all 25 full-duration cells with no
child failures. Relative to the last accepted matrix, realized shots increased
from 368 to 411, on-target shots from 216 to 224, and box-origin shots from 195
to 214. Passes changed only from 49,081 to 49,275. The realized-shot-weighted
mean distance increased slightly from 17.55 m to 17.67 m, so the result is an
absolute high-quality-opportunity improvement rather than a claim that every
shot became better. A 1.0 selectivity gain reproduced the four-seed baseline
trajectory exactly and was not substituted for the matrix-tested 2.0 value.

Pooled 15-minute pass-receipt fitted changes remained within the user-selected
five-point allowance for every tactic: +0.47 points for Catenaccio, -0.56 for
Gegenpress, -0.24 for Juego de Posicion, -2.09 for Salida Lavolpiana, and -1.36
for Zona Mista. Five individual team-match trend warnings remain review items.
The other 45 warnings were attacking-third backward concentration; its pooled
share remained high at 1,940/2,578 (75.25%). No liveness, repeated-event,
dismissal, spatial-concentration, or pass-collapse warning occurred. The shot
selectivity change is retained, while advanced recycling remains an unresolved
policy-quality target rather than being hidden by pass suppression.

The next isolated candidate keeps the 0.20 receiver/service temperature in
build-up, restarts, dribbles, and clearances, but uses 0.10 after the carrier
enters the attacking third. It sharpens selection only among already legal,
completion-scored services; it does not create a forward receiver or alter the
macro PASS probability. The first short profiler attempt completed its rollout
but correctly withheld the receipt because the separate diagnostic replay had
zero categorical mismatches and one continuous element with a 1.52e-6 absolute
difference, exceeding its declared 1e-6 tolerance. That unpublished run is not
used as evidence and the profiler tolerance is not relaxed. The complete
matrix was therefore used as the fail-closed behavioral acceptance test. It
reduced pooled attacking-third backward share only from 75.25% to 74.18%, and
left 45 concentration warnings, while reducing shots from 411 to 393,
on-target shots from 224 to 212, and box-origin shots from 214 to 204. The
candidate and its extra hot-path conditional were removed. Pooled temporal
pass changes remained between -0.46 and +0.20 percentage points and no other
anomaly family appeared, so the rejection is specifically an attacking-output
tradeoff rather than a liveness or late-match-quality failure.

The host report now separates the consequence from one observable cause. For
each exact attacking-third backward PASS it reads the verified pre-contact
tracking snapshot and counts active, on-pitch, non-dismissed, currently onside
teammates at least one metre ahead of the carrier. At 20 backward attempts and
a 70% no-support share it emits a distinct medium-severity warning. Both
thresholds are diagnostic DESIGN_PRIORS; the DFL event feed does not expose an
equivalent synchronized onside-support opportunity denominator. Reprocessing
all 25 loop-06 captures on CPU produced 1,427 unsupported releases among 1,867
backward releases (76.43%) and 37 team-match warnings. Team-match shares had a
76.84% median and a 58.62--100% range. This variable-length join remains in the
host analysis path; the JAX policy and environment transition gain no output
leaf or runtime work.

Those first support figures used the submitted kick-force family. That is a
useful low-level execution diagnostic but not the realized pass direction:
ground-pass control can oppose incoming ball velocity while the resulting ball
still travels forward. Match-metrics schema 13 therefore retains the submitted
direction receipts unchanged in the detailed pass map and policy-alignment
section, but defines team forward, attacking-third backward, and unsupported
backward counts from the source contact to the next distinct-player contact.
Attempts ended by a boundary or censored at capture end remain in total
attacking-third attempts but are excluded from the explicit direction-share
denominator and are exposed as direction-unknown.

Reprocessing the same 25 loop-06 captures under this corrected host-only
definition found 1,814 realized backward passes among 2,501 direction-known
attacking-third passes (72.53%), plus 1,374 backward passes without a visible
active onside teammate at least one metre ahead (75.74% of backward passes).
The tactical totals were Catenaccio 262/352 (74.4%), Gegenpress 424/656
(64.6%), Juego de Posicion 333/476 (70.0%), Salida Lavolpiana 421/533
(79.0%), and Zona Mista 374/484 (77.3%). Thirty-two team-match support
warnings and 44 direction-concentration warnings remained, so the policy
defect is not an artifact of force-vector compensation. This realized-contact
geometry is the closest available FootballWorld neighbor to DFL PlayAngle,
not an identical provider statistic: interceptions determine the observed end
point and sixteen attacking-third attempts had no classifiable next-contact
direction. Policy selection, pass physics, and rollout trajectories are
retained exactly; only report meaning and denominators change.

The clean `1b13931` loop-07 rerun then regenerated all 25 full-duration
seed-29 ordered cells under schema 13 with no child failure. It reproduced the
accepted trajectory totals exactly: 411 realized shots, 224 on target, 214
from inside the penalty area, and 49,275 open-play passes. This confirms that
the report-only definition change did not alter policy behavior. Realized
attacking-third direction was known for 2,560/2,578 passes; 1,880 were
backward (73.44%) and 1,431 of those had no visible active onside teammate at
least one metre ahead (76.12%). The audit emitted 39 backward-concentration,
37 no-forward-support, and five individual team-match 15-minute pass-trend
warnings, with no liveness, repeated-event, dismissal, spatial-concentration,
or pass-collapse warning. The clean full-duration captures are still labelled
diagnostic because report-only publication deliberately does not claim video
authority.

Role reconstruction from each match's recorded opening formation shows where
the remaining issue concentrates, while remaining approximate after later
manager formation changes or substitutions. Centre forwards made 609 realized
attacking-third backward passes and wide forwards 415; respectively 78.7% and
82.7% lacked a teammate at least one metre ahead. Unsupported events occurred
throughout the match rather than emerging from fatigue: the seven 15-minute
or added-time buckets contained 229, 238, 215, 230, 235, 235, and 49. Of the
1,431 unsupported events, 1,386 had a submitted backward direction, 1,235
ended at a teammate, and 1,107 reached the intended receiver. The dominant
mechanism is therefore a deliberate, usually successful layoff by the most
advanced carrier, not a force-compensation artifact, interception artifact,
or late-match policy collapse. Any next intervention should be restricted to
this carrier/action context; pushing the entire formation forward already
failed the full-matrix shot-quality acceptance test.

A narrower forward-layoff macro candidate was also rejected before a matrix
run. It retained full pass eligibility and receiver ranking, but multiplied
only a low-pressure CF/WF's selected backward service after entering the
attacking third by a 0.75-to-1.0 pressure-dependent scale. On the fixed four
seed, 4,500-step profile it reduced all passes from 570 to 564 and submitted
attacking-third backward releases from 12 to 10, but forward realized passes
also fell from 210 to 207 and the same five shots moved from 14.84 m to
16.29 m mean distance. Removing two short-sample layoffs does not justify that
progression and shot-quality loss, so the coefficient, scalar branch, and
carrier-role argument were all removed. Existing pass macro behavior is
retained pending a candidate that creates a credible forward option rather
than suppressing a safe one.

The first behavior-preserving efficiency candidate conditionally skipped the
11-by-opponent aerial arrival race when the carrier was outside a legal cross
origin. It retained cross targets, masks, randomness, and all tested actions;
47 focused CPU policy and x64-randomness tests passed. Nevertheless the fixed
batch-8, 256-step CPU benchmark rejected it. Executable text fell from
36,552,278 to 36,467,879 characters and cold compile plus first execution from
39.43 s to 38.83 s, but warm median time increased from 2.656 s to 2.764 s and
throughput fell from 771.0 to 740.8 simulated match frames/s. Compiler
temporary bytes also rose from 936,032 to 936,224. The dynamic branch was
removed; the existing unconditional fixed-shape calculation is faster on this
CPU workload.

A second behavior-preserving candidate passed the already-computed carrier
pressure scalar from the quick-relay calculation into possession scoring
instead of spelling out the same reduction twice. Forty-seven focused tests,
including exact supplied-versus-local pressure output equality, passed. XLA
already removed nearly all of the duplicated expression: executable text fell
only from 36,552,278 to 36,525,240 characters and compiler temporary bytes
stayed at 936,032. Cold time changed from 39.43 s to 39.57 s, while the warm
median regressed from 2.656 s to 2.735 s (771.0 to 748.7 simulated match
frames/s). The new public argument and integration plumbing were removed
because they supplied no measured speed or memory benefit.

The retained efficiency change instead skips shot planning only when the
carrier's causal service cadence says no macro decision is due. Those frames
already force the returned action to DRIBBLE, so shot quality, goalkeeper
separation, target portion, shot noise, and shot controls cannot affect the
action. Decision frames execute the original `plan_shot` function and keyed
random streams unchanged; direct standalone calls default to a due decision.
The fixed four-seed, 4,500-step profile exactly retained the accepted 571
passes, five shots, 14.47 m mean shot distance, 211 realized forward passes,
and thirteen submitted attacking-third backward releases.

On the scalar CPU benchmark that models each of the 25 independent match
workers, warm median time fell from 0.384 s to 0.352 s per 256 frames and
throughput rose from 667.4 to 728.1 simulated match frames/s (9.1%). Cold
compile plus first execution fell from 36.24 s to 34.92 s, executable text from
33,973,399 to 33,906,991 characters, and compiler temporary bytes from 163,608
to 160,792. These are same-host synthetic benchmark results, not a universal
deployment speed constant. The change adds a scalar fail-closed
`decision_due` shape check and keeps the output PyTree fixed.

The clean `d2e39d3` loop-08 acceptance matrix completed all 25 full-duration
seed-29 cells with zero child failure. Relative to loop-07, every compared
aggregate was exactly unchanged: 411 shots, 224 on target, 214 box-origin,
49,275 passes with 45,899 same-team next contacts, 19,285 realized forward
passes, 2,578 attacking-third passes, 1,880 realized backward, 1,431 backward
without forward support, 818 rule-policy cross signatures, and 89 defensive
line-break proxies. All per-plan W-D-L, goals, and shot totals were also
unchanged. This is the full-matrix behavioral acceptance receipt for the
optimization rather than an inference from the microbenchmark alone.

Pooled six-window completion remained stable within the user-selected five
percentage-point allowance. A direct attempt-weighted fit from the six 15
minute bins changed first-to-last by +0.30 points for Catenaccio, -0.43 for
Gegenpress, -0.30 for Juego de Posicion, -2.10 for Salida Lavolpiana, and
-1.41 for Zona Mista. Five individual team-match trend warnings and the same
39 backward-concentration plus 37 no-forward-support warnings remain; no new
anomaly family appeared.

Applying the same non-decision-frame gate to the three-destination clearance
calculation was rejected. It passed 46 focused tests and reduced cold time from
the shot-only 34.92 s to 35.27 s within ordinary run variance, executable text
from 33,906,991 to 33,809,261 characters, but added a second dynamic branch.
Warm scalar median regressed from 0.352 s to 0.374 s and throughput from 728.1
to 685.1 simulated match frames/s; temporary bytes also rose from 160,792 to
160,984. The clearance path is therefore retained as unconditional fixed-shape
work, and only the independently beneficial shot gate remains.

The host-only match metrics contract then advanced from
`footballworld.match-metrics/13` to `/14` without changing a rollout. The
existing exact realized-contact definition of an attacking-third backward pass
without forward support is retained: the carrier is in the attacking third,
the next distinct actor contact is geometrically backward, and no onside
teammate is at least one metre farther forward at release. The report now also
records how many of those passes reached the same team and their mean realized
contact-to-contact distance. This replaces an approximate opening-formation
role attribution for automated reporting: substitutions and later formation
changes make that attribution unsafe, whereas the new outcome and geometry are
observed replay facts.

All 25 loop-08 reports were regenerated in parallel from the retained replay
artifacts; no simulation was rerun. The matrix remains complete with zero
failed cells and the same 81 warnings. Of 1,431 unsupported attacking-third
backward passes, 1,235 reached the same team. Their plan aggregates were:
Catenaccio 235/213 at 26.14 m mean, Gegenpress 323/282 at 20.27 m, Juego de
Posicion 263/213 at 22.73 m, Salida Lavolpiana 284/245 at 22.07 m, and Zona
Mista 326/282 at 24.15 m. The pooled mean is 22.93 m. High receipt success
therefore does not by itself clear the warning: a material portion of this
pattern resets the attack over a substantial distance. The behavioral policy
is unchanged because the narrower suppression candidate already reduced
forward progression and worsened shot distance; the safer next design target
is creating a viable forward option rather than deleting successful service.

One direct forward-support candidate was rejected on the fixed four-seed,
4,500-step profiler before a matrix run. The current episode-stable CF/WF
receives its attacking-pattern pocket only after it is already at least one
metre ahead of the carrier. Removing that precondition appeared to address the
missing-option mechanism while retaining the existing participant, non-carrier,
pitch, and offside caps. Instead, realized pass count stayed at 578 while shots
fell from five to three, same-team receipts fell from 545 to 537, realized
backward passes rose from 207 to 220, and realized forward passes stayed at
202 but their receipts fell from 188 to 180. Mean shot distance improved from
16.21 m to 15.60 m only because two shots disappeared. The precondition is
therefore retained: blindly moving the selected attacker from behind the ball
changes too much team spacing and does not create a usable forward lane.

The multi-match report contract advances from
`footballworld.tactical-matrix-report/2` to `/3` to retain the negative
evidence from every child policy audit, not only threshold crossings. Its
`policy_audit_envelope` records the match and complete observed child record
for the maximum live loose-ball duration, nearly stationary loose-ball
duration, dominant-cell seconds and share, and repeated-event duration, plus
the total dismissal count. This is host-only information compression; the JAX
transition, policy behavior, warning thresholds, and individual report facts
are unchanged.

For loop-08 the maxima are 9.0 s live loose ball, 2.4 s nearly stationary
loose ball, 24.2 s and 0.46% dominant-cell occupancy, and 1.4 s repeated event
signature. Two dismissals occurred across separate matches. These remain below
the documented 15 s, 5 s, joint 60 s/2%, 30 s, and three-dismissals-per-team
review priors respectively. The original 3-1 spatial spike therefore does not
recur anywhere in this matrix, and the only active anomaly families remain
attacking-third backward service and five individual pass-completion trends.

A final behavior-preserving efficiency candidate gated only the macro-action
categorical draw on `decision_due`. Non-decision frames already return DRIBBLE,
and all 29 focused policy tests passed, but two fresh scalar CPU measurements
were inconsistent at 735.9 and 681.6 frames/s. Two interleaved measurements of
the retained unconditional path were 693.5 and 712.2 frames/s. The pair-median
advantage was only 0.85%, while executable text grew from 33,906,991 to
33,913,894 characters and compiler temporary memory from 160,792 to 160,920
bytes. This is host noise rather than a defensible runtime improvement, so the
dynamic branch was removed and the simpler unconditional categorical retained.

A completion-conditioned progression-gain candidate advanced to a full matrix
check. Raising the existing DESIGN_PRIOR from 0.60 to 0.80 changes receiver
ranking only; pass eligibility, completion estimation, physical execution,
macro cadence, offside authority, and PRNG streams are retained. In the fixed
four-seed, 4,500-step profile, realized passes rose from 578 to 599, shots from
five to seven, and realized forward passes from 202 to 228. Same-team receipts
changed from 545/578 (94.29%) to 561/599 (93.66%), while mean shot distance
improved from 16.21 m to 15.88 m. Realized backward passes also rose from 207
to 214, so the candidate is not accepted as a direct backward-service fix; it
is promoted because it creates more completed forward service and more,
slightly closer shots without a material receipt decline. Full-matrix results
remain the acceptance authority.
