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

SoccerWorld의 골키퍼는 상대와의 인터셉트 거리만 비교하고, 자기 골 방향
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
유지됐다. SoccerWorld에서 계승한 슬롯 비의존·앵커 상대 역할 분류는
sound하지만, 세 번째 이후의 모든 깊이선을 forward로 clip하는 동작은
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

SoccerWorld가 live pass에서 공의 현재 궤적을 매 프레임 다시 계산해
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

SoccerWorld에서 유지한 sound contract는 관측된 공에 팀별 primary claimant 하나만 배정하는 것이다. 그러나 SoccerWorld에는 FootballWorld의 supported-ground forecast와 저고도 미세 바운스 분기가 없으므로 그 nearest fallback은 그대로 상속할 수 없다. FootballWorld는 기존 claimant가 공개 candidate이고 환경의 실제 `carry_radius_m + ball.radius` 접촉 범위 안에 있을 때 그 identity를 유지한다. 이 반경은 새 정책 계수나 측정 축구 상수가 아니라 환경의 기존 물리 도달 envelope다. 목표점은 계속 관측 공 중심이므로 trajectory attack과 swept-contact authority는 바뀌지 않는다.

실제 상대 좌표, 속도, 체력과 5.7 mm 바운스를 고정한 회귀는 수정 전 Team 0 claimant가 10에서 9로 바뀌며 실패했고 수정 후 10을 유지했다. 또한 claimant가 이 물리 반경 안에 있을 때 같은 팀 비담당 선수가 반경 안으로 들어오면 공 반대 방향으로 공간을 비운다. 양 팀은 observer-local 후보 집합에서 각자 한 명을 선택하므로 상대 압박자는 그대로 경합한다. 관련 pass-plan/CONTROL 테스트 5개는 68.85초에 통과했고 별도 JIT 검증에서도 action과 recurrent state shape는 각각 `(22, 2)`, `(22,)`로 유지됐다.

같은 seed 3 전체 경기 report-only 비교에서 수정 전 loose-ball 29,271프레임 중 공 0.6 m 안에 같은 팀 두 명과 상대 한 명이 동시에 있던 프레임은 5개였고, 54:01.1--54:01.4에는 4프레임 연속이었다. 수정 후 loose-ball 29,233프레임에서는 해당 3인 군집이 0개였다. 같은 팀 두 명이 공 0.6 m 안에 있던 프레임은 16개에서 11개로 줄었으며 수정 후에는 모두 연속되지 않는 단일 프레임이었다. 전체 57,424프레임은 regulation complete로 종료됐고 `event_budget_exhausted_count`는 0이었다. 이는 물리 이벤트 한도를 늘리지 않고 정책 간격 조절로 실패를 제거한 결과다. 같은 CPU batch-1 recurrent policy step을 31회 warm 조건으로 측정한 배포 guard는 StableHLO text 1,324,671 byte, compiler temporary 121,888 byte, cost-analysis FLOP 1,890,072, one-shot compile 5.399초, warm median 3.157 ms였다. 순차 단일-host 측정이므로 속도 개선률을 인과 주장하지 않는다. 최종 authoritative 렌더 receipt는 clean revision 검증 뒤 기록한다.
