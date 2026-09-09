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

## 필수 SoccerWorld 기준 확인

구현 전에 `/data/SoccerWorld`의 다음 실제 자료를 확인했다.

| 확인 자료 | 유지한 장점 | 그대로 이식하지 않은 부분과 이유 |
|---|---|---|
| `src/soccerworld/_engine/rule_policy/guide.md` | 슛·패스·드리블·클리어를 하나의 bounded common currency에서 비교하고, 수치 지표와 경기 양상을 함께 평가한다. | 문서의 DFL 집계는 유용한 비교 기준이지만 일부 builder/동일 계약 재현 근거가 완결되지 않았다. 다른 관측·행동 계약의 계수를 측정 상수로 가져오지 않았다. |
| `src/soccerworld/_engine/rule_policy/policy.py` | 패스는 물리적 lane과 수신 경쟁을 함께 통과해야 하며, formation-relative role, 상황별 release hazard, 맥락적 빠른 릴레이를 사용한다. | SoccerWorld의 carrier cadence·계수·일반 패스에 더하는 릴레이 release 구조를 줄 단위로 복사하지 않았다. FootballWorld는 0.4초 cadence의 연쇄 패스를 막는 bounded 수신 창을 유지한다. 또한 물리 `control_ticks`는 공이 발에서 잠시 떨어질 때 합법적으로 초기화되므로 개인 전술 운반 시간으로 재사용하지 않고, public contact provenance로 별도 causal carry age를 유지한다. |
| `src/soccerworld/_engine/manager.py` 및 `src/soccerworld/_engine/formation.py` | static callable과 dynamic parameter PyTree를 분리하고, 감독 결정과 formation 의미를 저빈도 경계에 둔다. | bench·formation·host validation을 lean player step 안으로 옮기지 않았다. FootballWorld의 고정형 관리 경계를 유지했다. |
| `tests/unit/policies/test_rule_policy_tactics.py`, `test_rule_policy_formation_roles.py`, `test_rule_policy_dribble_recontact.py` | 역할이 절대 좌표가 아니라 formation에 상대적이어야 하고, loose touch가 정책 기억을 임의로 지우면 안 된다는 의미 계약을 유지했다. | FootballWorld의 public-observation 행 구조와 다른 fixture 수치를 복사하지 않았다. |
| `tests/unit/control/test_manager.py`, `tests/contracts/public_api/test_roster_configuration.py`, `experiments/phase_s/tests/test_ability_profiles.py` | 외부 작성 roster/ability는 그대로 보존하고, 무작위 profile은 명시 key와 bounded prior로 재현하며, 관리 policy는 외부 callable로 교체할 수 있어야 한다. | 다른 엔진의 roster 모양이나 비공개 상태를 FootballWorld transition에 추가하지 않았다. |
| `docs/performance/rule-policy-exact-gate-2026-08-29.md`, `docs/reproducibility/coefficient-provenance.md`, `docs/reproducibility/baseline-20260829.md`, `docs/reproducibility/release-audit-20260829.md` | cold compile, warm runtime, graph 크기, 경기 의미 지표를 구분해 기록한다. | 대응 A/B가 없는 상태에서 산술 감소만으로 성능 향상 또는 StableHLO 퇴행 부재를 주장하지 않는다. |

### 계승한 정책 원리

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

### SoccerWorld 상속 및 변경 경계

- 유지: 공개된 상대 운동으로 재접촉을 재허용하고 캐리어가 공 위치를
  회수하는 원칙은 오버런을 막으므로 sound하다. 패스 비행 중 한 명의
  trajectory receiver를 유지하는 원칙도 프레임별 수신자 교체를 막는다.
- 변경: SoccerWorld의 전역적인 공 직접 호밍은 FootballWorld의 전술 이동을
  지우므로, 최종 터치 방향과 공이 불정렬인 캐리어에게만 제한한다.
- 거부: SoccerWorld의 절대 공속 2.5 m/s 드리블 gate는 FootballWorld의
  player-relative CONTROL 의미와 직접 호환되지 않으며, 측정 상수로 이전할
  근거도 없으므로 복사하지 않는다.
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
