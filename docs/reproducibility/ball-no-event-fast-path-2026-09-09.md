# 공 이벤트 없음 고속 경로 A/B — 2026-09-09

## 결정

채택한다. 공이 선수 접촉, 골 프레임, 지면, 경기장 경계를 만날 수 없음을 보수적으로 증명한 물리 서브스텝은 기존 5회 제한 chronological event loop를 실행하지 않고 동일한 smooth endpoint를 반환한다. 후보가 하나라도 가능하거나 입력 기하가 비정상이면 기존 전체 event loop를 그대로 실행한다.

이 변경은 축구 계수, 접촉 판정식, 이벤트 우선순위, 최대 이벤트 수를 바꾸지 않는다.

## SoccerWorld 비교

필수 기준으로 /data/SoccerWorld/src/soccerworld/_engine/ball.py의 _ball_step_after_body와 /data/SoccerWorld/docs/performance/contest-no-candidate-2026-08-30.md를 확인했다.

계승한 장점은 다음과 같다.

- scalar lax.cond로 후보가 없는 큰 계산을 건너뛴다.
- 후보가 있는 branch는 기존 exact 구현을 단일 진실 원천으로 유지한다.
- 같은 backend에서 전체 PyTree digest를 비교한다.
- warm 실행, cold compile, 임시 버퍼와 실행 코드 크기를 따로 보고한다.

그대로 이식하지 않은 부분은 SoccerWorld가 몸통 접촉 뒤 공/골 프레임 적분을 순차 적용하고, FootballWorld의 다중 chronological event loop와 같은 구조가 없기 때문이다. SoccerWorld의 물리식을 복사하지 않았다.

## 구현

contact.py의 기존 active-contact AABB gate를 active_contact_possible로 재사용했다. substep.py는 다음 조건을 모두 만족할 때만 fast branch를 선택한다.

- dt > 0이고 공이 live
- active contact broad-phase 후보 없음
- passive body broad-phase 후보 없음
- goal-frame broad-phase 후보 없음
- 기존 exact ground event 없음
- 기존 exact boundary crossing 없음

공의 smooth advance는 기존 while-loop 산술 경계 안에서 한 번만 실행한다. 처음에는 이 연산을 평범한 outer expression으로 옮겼지만, CPU 8-step scan에서 z 속도의 마지막 비트가 달라져 폐기했다. 현재 구조는 작은 light while의 결과를 후보 판정과 반환에 함께 사용해 중복 적분도 제거한다.

비정상 좌표는 기존 broad-phase가 fail-open하므로 fast branch에 들어가지 않는다.

## 동일성

동결 기준은 변경 직전 src snapshot이며, 순수 후보는 그 snapshot에 아래 두 파일만 교체했다.

- src/footballworld/dynamics/contact.py
- src/footballworld/dynamics/substep.py

전체 혼합 정책 GPU b8×256 결과 digest는 양쪽 모두 1766555d5412d0a2f5ef0feaaa1ea3b3e0104c4af760348736e11e1c16321df3였다.

추가로 clear CPU/GPU 256-step, active, passive, ground, boundary, goal-frame, dead-ball 시나리오의 전체 출력 digest가 각각 기준과 동일했다. CPU와 GPU digest는 backend 사이가 아니라 각 backend의 A/B 내부에서만 비교했다.

## 성능

| 셀 | 기준 | 후보 | 변화 |
|---|---:|---:|---:|
| CPU clear lean scan, 256 substeps | 14.943 ms | 11.316 ms | -24.27% |
| RTX A6000 clear lean scan, 256 substeps | 99.524 ms | 86.129 ms | -13.46% |
| RTX A6000 전체 혼합 정책, b8×256 | 17.212 s | 14.932 s | -13.25% |
| 혼합 정책 처리량 | 118.98 fps | 137.15 fps | +15.27% |
| cold compile+첫 실행 | 132.99 s | 138.12 s | +3.86% |
| compiler temp | 1,996,152 B | 2,001,656 B | +5,504 B |
| executable text | 37,234,615 chars | 38,089,367 chars | +2.30% |

혼합 정책 출력 payload는 양쪽 모두 6,113,928 B였다. GPU retained 셀 전후 nvidia-smi compute process 목록은 비어 있었다.

## 검증과 한계

- focused tests: 2 passed
- Ruff: passed
- git diff --check: passed
- 공유 worktree가 이미 dirty이므로 결과는 release-authoritative가 아닌 immutable-source diagnostic A/B다.
- imported snapshot package hash는 각 셀의 시작과 끝에 동일했다.
- 이 최적화는 이벤트 없는 구간에서 이득이 크고, 이벤트 후보가 있는 서브스텝에는 사전 gate 비용이 추가될 수 있다. 실제 혼합 정책 경로에서 순이득이 확인되어 유지한다.
- 새 계수나 측정 축구 상수는 추가하지 않았다.

기계 판독 영수증: calib/environment/artifacts/20260909-ball-no-event-fast-path/summary.json
