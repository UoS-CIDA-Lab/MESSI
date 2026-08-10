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
    cooldown: Float[Array, "N"]
    stamina: Float[Array, "N"]
    ctrl_lock_t: Int[Array, "N"]
    kickoff_team: Int[Array, ""]
    poss_team: Int[Array, ""]
    last_touch_team: Int[Array, ""]
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
    score: Int[Array, "2"]
    yellow_cards: Int[Array, "N"]
    active_player: Bool[Array, "N"]
    penalty_flight_team: Int[Array, ""]
    penalty_encroach_mask: Bool[Array, "N"]
    restart_indirect: Bool[Array, ""]
    last_touch_code: Int[Array, ""]
