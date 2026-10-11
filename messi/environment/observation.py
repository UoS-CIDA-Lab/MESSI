"""선수별 중복 계산을 공유하는 full-information observation."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.core.action import ACTION_INTENT_COUNT
from messi.core.ball import BallGeometry
from messi.core.constants import RK_NONE
from messi.core.contact import ContactPhysics
from messi.core.goalkeeper import held_ball_mask
from messi.core.timebase import DEFAULT_TIMEBASE, Timebase


class SharedScene(NamedTuple):
    valid: jax.Array
    team_id: jax.Array
    attack_direction: jax.Array
    position_team: jax.Array
    velocity_team: jax.Array
    body_forward_team: jax.Array
    relative_position: jax.Array
    relative_velocity: jax.Array
    distance_squared: jax.Array
    teammate: jax.Array
    opponent: jax.Array
    ball_relative_position: jax.Array
    ball_relative_velocity: jax.Array
    ball_position_team: jax.Array
    ball_velocity_team: jax.Array
    ball_spin_team: jax.Array
    ball_live: jax.Array
    stamina_long: jax.Array
    stamina_short: jax.Array
    possessor: jax.Array
    restart_taker: jax.Array
    score_for: jax.Array
    score_against: jax.Array
    control_tick: jax.Array
    # 접촉 recovery lock. 남은 substep 수 대신 다음 control frame 안에 physics가
    # 접촉을 시도할 수 있는지를 availability가 쓰는 형태로만 노출한다.
    contact_ready: jax.Array
    challenge_ready: jax.Array


class PolicyObservation(NamedTuple):
    """Rule policy와 neural policy가 공유하는 full-observable 입력."""

    scene: SharedScene
    player_id: jax.Array
    is_goalkeeper: jax.Array
    max_speed: jax.Array
    reach_height: jax.Array
    height: jax.Array
    ball_control: jax.Array
    endurance_factor: jax.Array
    role: jax.Array
    formation_anchor: jax.Array
    formation_id: jax.Array
    prospective_offside: jax.Array
    goalkeeper_claim_available: jax.Array
    goalkeeper_holding: jax.Array
    restart_retouch_blocked: jax.Array
    restart_kind: jax.Array
    restart_indirect: jax.Array
    ball_height: jax.Array
    ball_vertical_velocity: jax.Array
    possession_team: jax.Array
    possession_previous_team: jax.Array
    possession_control_ticks: jax.Array
    possession_episode_ticks: jax.Array
    last_contact_actor: jax.Array
    last_contact_intent: jax.Array
    last_contact_kick_applied: jax.Array


def _team_frame_2(vector: jax.Array, direction: jax.Array) -> jax.Array:
    return vector * jnp.stack((direction, jnp.ones_like(direction)), axis=-1)


def observe_all(
    state,
    roster,
    *,
    timebase: Timebase = DEFAULT_TIMEBASE,
) -> SharedScene:
    """모든 observer의 shared geometry를 한 번의 broadcasting으로 생성한다.

    Recovery readiness는 다음 control frame 전체를 기준으로 한다. Action은 frame 동안
    유지되고 transition은 각 substep의 contact 판정 전에 counter를 감소시키므로, 남은
    counter가 ``timebase.decimation`` 이하면 그 frame 안에 접촉을 시도할 수 있다.
    """

    if not isinstance(timebase, Timebase):
        raise TypeError("timebase는 Timebase여야 한다")
    players = state.players
    count = players.position.shape[0]
    indices = jnp.arange(count, dtype=jnp.int32)
    valid = players.active
    direction = state.attack_direction[roster.team_id]
    position_team = _team_frame_2(players.position, direction)
    velocity_team = _team_frame_2(players.velocity, direction)
    body_team = _team_frame_2(players.body_forward, direction)
    relative_position = _team_frame_2(
        players.position[None, :, :] - players.position[:, None, :],
        direction[:, None],
    )
    relative_velocity = _team_frame_2(
        players.velocity[None, :, :] - players.velocity[:, None, :],
        direction[:, None],
    )
    distance_squared = jnp.sum(relative_position * relative_position, axis=-1)
    same_team = roster.team_id[:, None] == roster.team_id[None, :]
    pair_valid = valid[:, None] & valid[None, :]
    ball_relative_position = _team_frame_2(
        state.ball.position[None, :2] - players.position, direction
    )
    ball_relative_velocity = _team_frame_2(
        state.ball.velocity[None, :2] - players.velocity, direction
    )
    ball_position_team = _team_frame_2(
        jnp.broadcast_to(state.ball.position[:2], (count, 2)), direction
    )
    ball_velocity_team = _team_frame_2(
        jnp.broadcast_to(state.ball.velocity[:2], (count, 2)), direction
    )
    ball_spin_team = (
        jnp.broadcast_to(state.ball.spin, (count, 3)).at[:, 0].multiply(direction)
    )
    safe_team = jnp.clip(roster.team_id, 0, state.score.shape[0] - 1)
    return SharedScene(
        valid=valid,
        team_id=roster.team_id,
        attack_direction=direction,
        position_team=position_team,
        velocity_team=velocity_team,
        body_forward_team=body_team,
        relative_position=relative_position,
        relative_velocity=relative_velocity,
        distance_squared=distance_squared,
        teammate=pair_valid & same_team & (indices[:, None] != indices[None, :]),
        opponent=pair_valid & (~same_team),
        ball_relative_position=ball_relative_position,
        ball_relative_velocity=ball_relative_velocity,
        ball_position_team=ball_position_team,
        ball_velocity_team=ball_velocity_team,
        ball_spin_team=ball_spin_team,
        ball_live=jnp.broadcast_to(state.ball.live, (count,)),
        stamina_long=players.stamina_long,
        stamina_short=players.stamina_short,
        possessor=indices == state.possession.player,
        restart_taker=(state.restart.kind != RK_NONE)
        & (indices == state.restart.taker),
        score_for=state.score[safe_team],
        score_against=state.score[1 - safe_team],
        control_tick=jnp.broadcast_to(state.control_tick, (count,)),
        contact_ready=(players.contact_lock_substeps <= timebase.decimation)
        & (players.aerial_recovery_substeps <= timebase.decimation)
        # Loss/challenge recovery는 MOVE·contain 추격만 허용한다. 남은 lock이
        # control frame 안에 만료된다는 이유로 contact를 미리 제출하면 frame 후반에
        # 즉시 재CONTROL해 소유권 ping-pong이 생긴다.
        & (players.possession_loss_lock_substeps <= 0)
        & (players.challenge_recovery_substeps <= 0),
        challenge_ready=(
            players.challenge_recovery_substeps <= 0
        ),
    )


def intent_availability(
    scene: SharedScene,
    *,
    goalkeeper_holding: jax.Array | None = None,
    restart_retouch_blocked: jax.Array | None = None,
    contact: ContactPhysics = ContactPhysics(),
    ball: BallGeometry = BallGeometry(),
) -> jax.Array:
    """실제 최대 contact reach를 상한으로 한 5-intent mask를 반환한다.

    ``goalkeeper_holding``은 authoritative state의 ``held_ball_mask``에서 만든
    player별 bool이다. Scene만으로 CATCH provenance를 추론하지 않는다.
    """

    maximum_contact_distance = max(
        contact.carry_radius_m,
        contact.challenge_radius_m,
        contact.goalkeeper_radius_m,
    ) + float(ball.radius_m)
    near_ball = (
        jnp.sum(scene.ball_relative_position**2, axis=-1)
        <= maximum_contact_distance**2
    )
    ownerless_or_self = ~jnp.any(scene.possessor) | scene.possessor
    opposing_owner = jnp.any(scene.opponent & scene.possessor[None, :], axis=-1)
    restart_open = jnp.any(scene.restart_taker)
    can_contact = scene.valid & scene.ball_live & near_ball & scene.contact_ready
    if goalkeeper_holding is None:
        goalkeeper_holding = jnp.zeros_like(scene.valid)
    else:
        goalkeeper_holding = jnp.broadcast_to(
            jnp.asarray(goalkeeper_holding, dtype=jnp.bool_), scene.valid.shape
        )
    if restart_retouch_blocked is None:
        restart_retouch_blocked = jnp.zeros_like(scene.valid)
    else:
        restart_retouch_blocked = jnp.broadcast_to(
            jnp.asarray(restart_retouch_blocked, dtype=jnp.bool_), scene.valid.shape
        )
    held_ball = (
        scene.valid & (~scene.ball_live) & scene.possessor & (~restart_open)
        & near_ball
        & goalkeeper_holding
    )
    normal = jnp.stack(
        (
            scene.valid,
            can_contact & ownerless_or_self,
            (can_contact & ownerless_or_self) | held_ball,
            (can_contact & ownerless_or_self) | held_ball,
            can_contact & opposing_owner & scene.challenge_ready,
        ),
        axis=-1,
    )
    restart = jnp.zeros((scene.valid.shape[0], ACTION_INTENT_COUNT), dtype=jnp.bool_)
    restart = restart.at[:, 0].set(scene.valid)
    # Held-ball과 restart release는 open-play contact geometry보다 먼저 처리되고
    # recovery counter로 게이팅되지 않는다. Availability도 같은 예외를 보존한다.
    restart_contact = scene.restart_taker & near_ball
    restart = restart.at[:, 2].set(restart_contact)
    restart = restart.at[:, 3].set(restart_contact)
    available = jnp.where(restart_open, restart, normal)
    move_only = jnp.zeros_like(available).at[:, 0].set(scene.valid)
    return jnp.where(restart_retouch_blocked[:, None], move_only, available)


def intent_availability_from_state(
    state,
    roster,
    *,
    contact: ContactPhysics = ContactPhysics(),
    ball: BallGeometry = BallGeometry(),
    timebase: Timebase = DEFAULT_TIMEBASE,
) -> jax.Array:
    """Authoritative State에서 held-ball legality까지 포함한 intent mask를 만든다."""

    held, holder = held_ball_mask(state, roster)
    indices = jnp.arange(state.players.position.shape[0], dtype=jnp.int32)
    goalkeeper_holding = held & (indices == holder)
    release = state.restart_release
    restart_retouch_blocked = (
        release.active & release.untouched & (indices == release.taker)
    )
    return intent_availability(
        observe_all(state, roster, timebase=timebase),
        goalkeeper_holding=goalkeeper_holding,
        restart_retouch_blocked=restart_retouch_blocked,
        contact=contact,
        ball=ball,
    )
