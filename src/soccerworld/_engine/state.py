"""
env 상태 컨테이너 — 전 물리·규칙 필드를 담는 불변 NamedTuple(pytree).

from __future__ import annotations: 이 파일의 모든 type hint는 forward reference로 해석. 타입 힌트에서 클래스 이름을 문자열로 감싸지 않아도 됨. (Python 3.7+)
"""

from __future__ import annotations

from typing import NamedTuple
from jaxtyping import Array, Float, Int, Bool

class State(NamedTuple):
    t: Int[Array, ""]
    ball_pos: Float[Array, "3"]
    ball_vel: Float[Array, "3"]
    ball_spin: Float[Array, "3"]
    player_ctrl: Float[Array, "N"]
    ball_state: Int[Array, ""]
    player_pos: Float[Array, "N 2"]
    player_vel: Float[Array, "N 2"]
    player_facing: Float[Array, "N"]
    attack_dir : Float[Array, "N"]
    team_id: Int[Array, "N"]
    gk_indices: Int[Array, "N"]
    vmax: Float[Array, "N"]
    reach_z: Float[Array, "N"]
    head_z: Float[Array, "N"]
    cooldown: Float[Array, "N"]  # 남은 재도전 잠금(substep). 현 소유팀 ball action은 우회한다.
    contact_lock_t: Int[Array, "N"]  # 마지막 도달 가능 f2b 시도 뒤 능동·자기몸 재접촉 잠금.
    stamina_long: Float[Array, "N"]
    stamina_short: Float[Array, "N"]
    endurance_factor: Float[Array, "N"]
    aerial_recovery_t: Int[Array, "N"]  # 키보다 높은 공 시도 뒤 능동 개입만 막는 회복 락.
    ctrl_lock_t: Int[Array, "N"]
    kickoff_team: Int[Array, ""]
    poss_team: Int[Array, ""]
    # Consecutive control frames under the current ``poss_team`` label and the
    # most recent non-neutral controller before that label changed.  Together
    # these make loss/win/loose transitions observable instead of forcing a
    # policy to infer them from a hidden recurrent state.
    possession_t: Int[Array, ""]
    previous_poss_team: Int[Array, ""]
    last_touch_team: Int[Array, ""]
    # 마지막으로 공을 건드린 **선수 슬롯**. 팀·코드만으로는 '내가 방금 찬 공인가'를
    # 알 수 없어, 정책이 기하(공 진행축 투영)로 원 패서를 추정해야 했다. 그 추정은 실측에서
    # 양쪽으로 틀렸다 — 원 패서를 막지 못하면서(1프레임 재접촉 38건/120초) 도착한 수신자를
    # 배제했다. 관측에는 슬롯이 아니라 관측자 기준 **관계**로만 나간다(identity 비노출 원칙).
    last_touch_actor: Int[Array, ""]
    # ── 교체 ──────────────────────────────────────────────────────────────
    # 벤치는 N개 출전 slot과 별개의 **투입 후보 데이터**다. slot으로 두지 않는 이유는 벤치 선수가
    # 관측 대상이 아니기 때문이다 — N을 늘리면 obs 차원과 모든 마스크가 연쇄로 바뀌는데
    # 얻는 것이 없다.
    #
    # 크기 ``S_b``는 env 생성 인자(``bench_size``)로 정한다. 환경별 교체 상한과 **다른
    # 값**이라는 점이 중요하다 — 허용 교체가 5명이어도 9명 중 어느 5명을 고르는가가 그
    # 자체로 결정이고, 선수단 전체를 벤치로 두는 실험도 가능해야 한다.
    # ``bench_player_id``가 NO_PLAYER면 그 자리는 비었거나 이미 투입됐다.
    bench_player_id: Int[Array, "2 S_b"]
    bench_vmax: Float[Array, "2 S_b"]
    bench_reach_z: Float[Array, "2 S_b"]
    bench_head_z: Float[Array, "2 S_b"]
    bench_player_ctrl: Float[Array, "2 S_b"]
    bench_endurance_factor: Float[Array, "2 S_b"]
    bench_is_gk: Bool[Array, "2 S_b"]
    bench_role_pos: Float[Array, "2 S_b 2"]
    # 교체로 **나간** 선수. 슬롯 identity는 덮이므로 여기 남기지 않으면 사라진다.
    # 렌더가 퇴장 선수와 다른 구역에 그리려면 신원이 필요하다.
    #
    # 깊이 ``S_r``은 벤치 크기 ``S_b``와 **다르다**. 스케줄 교체는 벤치를 쓰지 않으므로
    # 벤치 폭에 묶으면 스케줄 전용 env(``bench_size=0``)가 기록할 자리를 못 갖는다.
    # 나간 순서대로 앞에서부터 채운다 — 벤치 인덱스가 아니다.
    retired_player_id: Int[Array, "2 S_r"]
    subs_remaining: Int[Array, "2"]
    """팀별 잔여 교체 인원."""
    sub_windows_used: Int[Array, "2"]
    """팀별 사용한 교체 기회 수. 한 데드볼에 여러 명을 바꾸면 한 번으로 센다."""
    sub_window_open_t: Int[Array, "2"]
    """현재 열려 있는 교체 기회의 control tick. -1이면 닫혀 있다."""
    # ── 포메이션 ──────────────────────────────────────────────────────────
    # 팀의 **모양**. 교체가 '누가 뛰는가'라면 이쪽은 '어디에 서는가'다.
    #
    # 앵커 좌표 (N,2)를 직접 들고 있지 않는 이유는 그것이 파생값이기 때문이다. 레이아웃
    # 표는 생성 시점에 정해지는 정적 상수이고, 슬롯의 깊이·좌우 순위도 킥오프에서 한 번
    # 정해진다. 따라서 현재 규범 앵커는 ``layout_index``와 활성 roster에서 곧바로 복원한다.
    # 선수 좌표는 명령 경계에서 바뀌지 않고, 정책과 물리를 통해 목표를 향해 이동한다.
    layout_index: Int[Array, "2"]
    """현재 명령된 목표 레이아웃 인덱스."""
    layout_since_t: Int[Array, "2"]
    """마지막으로 모양을 명령한 control tick."""
    # ── 에피소드 신원 ─────────────────────────────────────────────────────
    episode_seed: Int[Array, ""]
    """이 에피소드의 시드 요약값.

    물리 scan 깊숙이에서 불리는 결정자(세트피스 키커)는 스텝 키를 받을 수 없다 — 그 경로에
    키를 꿰려면 ``_events``·``_fouls``·``_offside``·``_contest``를 전부 고쳐야 한다. 대신
    reset의 키를 한 칸에 접어 두고 결정자가 그것을 tick·종류와 함께 fold-in한다.

    이것이 없으면 확률적 키커 정책이 **에피소드 시드의 영향을 받지 않는다** — 서로 다른
    시드의 두 경기가 같은 시각·같은 재개에서 같은 난수를 받는다."""
    # Packed restriction code: team 0/1 means a direct deliberate team-mate
    # kick/throw; team+2 means that GK released hand possession before another
    # player touch.  Causes share current legality but differ after a GK kick.
    gk_handling_restricted_team: Int[Array, ""]
    restart_team: Int[Array, ""]
    restart_t: Int[Array, ""]
    restart_kind: Int[Array, ""]
    offside_flag: Bool[Array, "N"]
    pass_team: Int[Array, ""]
    pass_t: Int[Array, ""]
    foul_kind: Int[Array, ""]
    foul_actor: Int[Array, ""]
    foul_victim: Int[Array, ""]
    pending_taker: Int[Array, ""]
    setpiece_taker: Int[Array, ""]
    throw_taker: Int[Array, ""]
    touch: Int[Array, "N"]
    # control frame 안의 물리 접촉 스트림. 첫 축은 physics substep, 둘째 축은
    # force-to-ball -> passive body 순서다. 비어 있는 칸은 actor/player_id/control_t=-1,
    # code=TOUCH_NONE, 실수 telemetry=0이다. ``touch``가 선수별 최종 누적값인 것과 달리
    # 같은 선수의 반복 접촉과 서로 다른 선수의 실제 발생 순서·접촉 순간 물리량을 보존한다.
    touch_event_actor: Int[Array, "S 2"]
    touch_event_code: Int[Array, "S 2"]
    touch_event_player_id: Int[Array, "S 2"]
    touch_event_control_t: Int[Array, "S 2"]
    touch_event_toi: Float[Array, "S 2"]
    touch_event_ball_pos: Float[Array, "S 2 3"]
    touch_event_ball_vel_before: Float[Array, "S 2 3"]
    touch_event_ball_vel_after: Float[Array, "S 2 3"]
    touch_event_impulse: Float[Array, "S 2"]
    # control frame 안의 라인 통과 사건 스트림 — 축은 physics substep이다.
    # 골/아웃은 판정 즉시 공을 재개 스폿으로 옮기고 속도를 0으로 만들기 때문에,
    # 프레임 경계 표본에는 '어디서 어떤 속도로 라인을 넘었는가'가 남지 않는다.
    # 판정에 실제로 쓴 교차점을 여기 보존한다. 빈 칸은 kind=BALL_EVENT_NONE,
    # team=NO_TEAM, control_t=-1, 실수 telemetry=0이다.
    ball_event_kind: Int[Array, "S"]
    ball_event_team: Int[Array, "S"]
    ball_event_pos: Float[Array, "S 3"]
    ball_event_vel: Float[Array, "S 3"]
    ball_event_control_t: Int[Array, "S"]
    # control frame 안의 골 프레임(포스트·크로스바) 충돌 스트림 — 축은 physics substep이다.
    # ``ball_event_*``와 칸을 나눠 쓰지 않는 이유: 크로스바를 맞고 그대로 골라인을 넘는 공은
    # **한 substep 안에서** 프레임 충돌과 라인 통과를 둘 다 만든다. 한 칸이면 하필 그
    # 흥미로운 경우만 덮어써 사라진다. 빈 칸은 kind=WOODWORK_NONE, control_t=-1,
    # 실수 telemetry=0이다. ``vel_in``은 반사 **전** 속도라 충돌 강도를 그대로 준다.
    woodwork_kind: Int[Array, "S"]
    woodwork_pos: Float[Array, "S 3"]
    woodwork_vel_in: Float[Array, "S 3"]
    woodwork_control_t: Int[Array, "S"]
    score: Int[Array, "2"]
    yellow_cards: Int[Array, "N"]
    player_id: Int[Array, "N"]
    slot_generation: Int[Array, "N"]
    on_pitch: Bool[Array, "N"]
    sent_off: Bool[Array, "N"]
    restart_indirect: Bool[Array, ""]
    last_touch_code: Int[Array, ""]
    role_pos: Float[Array, "N 2"]   # 포지션 역할 벡터(공격 접힘 프레임, m) — 서술적 입력.
                                    # 현재 전술 epoch의 **라이브볼 프레임 인과적 누적평균**.
                                    # 표본이 아직 없으면 진입 prior(킥오프 포메이션/교체 시 지정값)를 유지한다.
                                    # 데이터셋도 이 전술 epoch 경계를 명시한다.
                                    # 재생/롤아웃은 같은 규약을 쓴다.
    role_pos_count: Float[Array, "N"]    # 표본 수. identity·하프·승인된 포메이션 경계에서 리셋.
                                         # 0이면 role_pos는 아직 prior다. 누적합은 따로 두지 않는다 —
                                         # 갱신이 증분식이라 (role_pos, role_pos_count)만으로 결정된다.
                                         # count 감소는 교체 증거가 아니다. identity SSOT는
                                         # player_id/slot_generation이다.

    @property
    def active_player(self):
        """물리·규칙 참여 projection. 저장 진실원천은 on_pitch와 sent_off다."""

        return self.on_pitch & (~self.sent_off)
