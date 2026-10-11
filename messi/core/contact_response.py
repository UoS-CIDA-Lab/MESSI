"""선택된 contest 결과를 deterministic ball·possession response로 변환한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from messi.core.action import INTENT_CHALLENGE, INTENT_CONTROL, INTENT_MOVE
from messi.core.ball import BallPhysics
from messi.core.boundary import GoalGeometry
from messi.core.constants import (
    DISCIPLINE_RED,
    DISCIPLINE_YELLOW,
    NO_PLAYER,
    NO_TEAM,
    RK_GK_HOLD,
    RK_NONE,
    RK_THROWIN,
)
from messi.core.contact import (
    INTENT_SOURCE_NONE,
    INTENT_SOURCE_POLICY,
    LAW11_DEFLECTION_NO_RESET,
    LAW11_DELIBERATE_PLAY_RESET,
    LAW11_DELIBERATE_SAVE_NO_RESET,
    LAW11_NONE,
    MECHANISM_CHEST,
    MECHANISM_FOOT,
    MECHANISM_GOALKEEPER_HAND,
    MECHANISM_HEAD,
    MECHANISM_NONE,
    OUTCOME_CATCH,
    OUTCOME_DEFLECTION,
    OUTCOME_INTERCEPTION,
    OUTCOME_MISCONTROL,
    OUTCOME_NONE,
    OUTCOME_PARRY,
    OUTCOME_RELEASE,
    OUTCOME_TACKLE_WON,
    OUTCOME_TRAP,
    ContactGeometry,
    ContactPhysics,
    ContactResult,
    PhysicsAction,
    apply_foot_actuation,
)
from messi.core.contest import ContestResult
from messi.core.geometry import BallGeometry
from messi.core.goal_frame import GoalFramePhysics
from messi.core.goal_threat import is_goal_threat
from messi.core.numeric import require_float32_representable
from messi.core.timebase import MAX_INT32_TICKS, duration_to_ticks

_DIV_EPS = 1.0e-9
_SQUARED_EPS = 1.0e-12
_MAX_DYNAMIC_COUNTER_TICKS = MAX_INT32_TICKS - 127


class ControlResponse(NamedTuple):
    velocity: jax.Array
    impulse: jax.Array
    required_impulse_speed: jax.Array
    applied_impulse_speed: jax.Array
    completion: jax.Array
    success: jax.Array


class ContactResolution(NamedTuple):
    state: object
    contact: ContactResult
    contest: ContestResult


@dataclass(frozen=True, slots=True)
class ContactResponsePhysics:
    """non-foot energy retention과 반복 접촉 lock의 design prior."""

    header_speed_retention: float = 0.90
    chest_speed_retention: float = 0.10
    challenge_speed_retention: float = 1.0
    active_contact_interval_s: float = 1.0 / 15.0
    deliberate_kick_contact_lock_s: float = 0.21
    challenge_recovery_s: float = 0.16
    max_lunge_extra_recovery_s: float = 0.84
    aerial_attempt_recovery_s: float = 0.5
    goalkeeper_dive_recovery_s: float = 1.0
    possession_loss_lock_s: float = 0.16

    def __post_init__(self) -> None:
        for name in (
            "header_speed_retention",
            "chest_speed_retention",
            "challenge_speed_retention",
        ):
            value = require_float32_representable(name, getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}은 [0, 1] 범위여야 한다")
        for name in (
            "active_contact_interval_s",
            "deliberate_kick_contact_lock_s",
            "challenge_recovery_s",
            "max_lunge_extra_recovery_s",
            "aerial_attempt_recovery_s",
            "goalkeeper_dive_recovery_s",
            "possession_loss_lock_s",
        ):
            value = require_float32_representable(name, getattr(self, name))
            if not 0.0 <= value <= 2.0:
                raise ValueError(f"{name}은 [0, 2] 범위여야 한다")
        if self.active_contact_interval_s < 0.001:
            raise ValueError("active_contact_interval_s는 0.001 이상이어야 한다")
        if self.deliberate_kick_contact_lock_s < self.active_contact_interval_s:
            raise ValueError(
                "deliberate_kick_contact_lock_s는 active_contact_interval_s 이상이어야 한다"
            )


_DEFAULT_CONTACT_PHYSICS = ContactPhysics()
_DEFAULT_RESPONSE_PHYSICS = ContactResponsePhysics()


def _validate_counter_tick_capacity(
    dt: float, response: ContactResponsePhysics
) -> None:
    """float duration이 dynamic int32 recovery counter에 안전하게 들어가는지 검사한다."""

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        durations = (
            ("active_contact_interval_s", np.float32(response.active_contact_interval_s)),
            (
                "deliberate_kick_contact_lock_s",
                np.float32(response.deliberate_kick_contact_lock_s),
            ),
            (
                "maximum challenge recovery",
                np.float32(response.challenge_recovery_s)
                + np.float32(response.max_lunge_extra_recovery_s),
            ),
            (
                "maximum aerial recovery",
                np.maximum(
                    np.float32(response.aerial_attempt_recovery_s),
                    np.float32(response.goalkeeper_dive_recovery_s),
                ),
            ),
            ("possession_loss_lock_s", np.float32(response.possession_loss_lock_s)),
        )
        represented_dt = np.float32(dt)
    for name, seconds in durations:
        if seconds <= 0.0:
            continue
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            represented_ticks = seconds / represented_dt
        if (
            not np.isfinite(represented_ticks)
            or represented_ticks > np.float32(_MAX_DYNAMIC_COUNTER_TICKS)
        ):
            raise ValueError(
                f"{name}의 physics tick 수는 int32 state counter 범위여야 한다"
            )
        try:
            duration_to_ticks(
                seconds,
                dt,
                rounding="ceil",
                minimum=1,
                maximum=_MAX_DYNAMIC_COUNTER_TICKS,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{name}의 physics tick 수는 int32 state counter 범위여야 한다"
            ) from error


def resolve_control_response(
    incoming_velocity: jax.Array,
    player_velocity: jax.Array,
    requested_relative_velocity: jax.Array,
    *,
    impulse_speed_limit_mps: jax.Array,
) -> ControlResponse:
    """player-relative exit velocity를 bounded impulse로 최대한 달성한다."""

    incoming_velocity = jnp.asarray(incoming_velocity)
    dtype = incoming_velocity.dtype
    player_velocity = jnp.asarray(player_velocity, dtype=dtype)
    requested_relative_velocity = jnp.asarray(requested_relative_velocity, dtype=dtype)
    target = player_velocity + requested_relative_velocity
    required = target - incoming_velocity
    required_squared = jnp.sum(required * required, axis=-1)
    required_speed = jnp.sqrt(jnp.maximum(required_squared, 0.0))
    budget = jnp.maximum(jnp.asarray(impulse_speed_limit_mps, dtype=dtype), 0.0)
    completion = jnp.where(
        required_squared > _SQUARED_EPS,
        jnp.minimum(1.0, budget / (required_speed + _DIV_EPS)),
        1.0,
    )
    impulse = required * completion[..., None]
    return ControlResponse(
        velocity=incoming_velocity + impulse,
        impulse=impulse,
        required_impulse_speed=required_speed,
        applied_impulse_speed=required_speed * completion,
        completion=completion,
        success=required_speed <= budget,
    )


def resolve_contact_response(
    state,
    roster,
    action: PhysicsAction,
    geometry: ContactGeometry,
    contest: ContestResult,
    *,
    dt: float,
    contact: ContactPhysics = _DEFAULT_CONTACT_PHYSICS,
    response: ContactResponsePhysics = _DEFAULT_RESPONSE_PHYSICS,
    field_half_extent: jax.Array = (52.5, 34.0),
    ball_geometry: BallGeometry = BallGeometry(),
    ball_physics: BallPhysics = BallPhysics(),
    goal_geometry: GoalGeometry = GoalGeometry(),
    goal_frame_physics: GoalFramePhysics = GoalFramePhysics(),
) -> ContactResolution:
    """한 contest의 물리 response와 최소 causal state를 갱신한다."""

    _validate_counter_tick_capacity(dt, response)
    player_count = state.players.position.shape[0]
    safe_actor = jnp.clip(contest.actor, 0, player_count - 1)
    selected = contest.selected
    mechanism = geometry.mechanism[safe_actor]
    intent = geometry.requested_intent[safe_actor]
    raw_outcome = contest.outcome
    raw_contact = selected & (
        (raw_outcome == OUTCOME_RELEASE)
        | (raw_outcome == OUTCOME_INTERCEPTION)
        | (raw_outcome == OUTCOME_TACKLE_WON)
        | (raw_outcome == OUTCOME_DEFLECTION)
        | (raw_outcome == OUTCOME_CATCH)
        | (raw_outcome == OUTCOME_PARRY)
    )
    foot = mechanism == MECHANISM_FOOT
    chest = mechanism == MECHANISM_CHEST
    head = mechanism == MECHANISM_HEAD
    goalkeeper = mechanism == MECHANISM_GOALKEEPER_HAND
    challenge = geometry.challenge_context[safe_actor]
    caught = raw_contact & goalkeeper & (raw_outcome == OUTCOME_CATCH)
    parried = raw_contact & goalkeeper & (raw_outcome == OUTCOME_PARRY)

    foot_actuation = apply_foot_actuation(
        state.ball,
        state.players,
        action,
        safe_actor,
        ball_geometry=ball_geometry,
        config=contact,
    )
    launch = foot_actuation.launch
    direction = foot_actuation.direction
    impulse_direction = jnp.asarray(
        (
            jnp.cos(launch) * direction[0],
            jnp.cos(launch) * direction[1],
            jnp.sin(launch),
        ),
        dtype=state.ball.velocity.dtype,
    )
    player_velocity = jnp.asarray(
        (
            state.players.velocity[safe_actor, 0],
            state.players.velocity[safe_actor, 1],
            0.0,
        ),
        dtype=state.ball.velocity.dtype,
    )
    incoming_speed = jnp.linalg.norm(state.ball.velocity)
    incoming_relative_speed = jnp.linalg.norm(state.ball.velocity - player_velocity)
    control_request = (intent == INTENT_CONTROL) | (intent == INTENT_CHALLENGE)
    chest_relative = (
        response.chest_speed_retention * incoming_relative_speed * impulse_direction
    )
    requested_relative = jnp.where(
        chest,
        chest_relative,
        action.force_power[safe_actor]
        * contact.control_request_speed_max_mps
        * impulse_direction,
    )
    controlled = resolve_control_response(
        state.ball.velocity,
        player_velocity,
        requested_relative,
        impulse_speed_limit_mps=jnp.asarray(
            contact.kick_speed_max_mps, dtype=state.ball.velocity.dtype
        ),
    )
    control_eligible = (
        raw_contact
        & control_request
        & (~head)
        & (~goalkeeper)
        & (
            (raw_outcome == OUTCOME_RELEASE)
            | (raw_outcome == OUTCOME_INTERCEPTION)
            | (raw_outcome == OUTCOME_TACKLE_WON)
        )
    )
    control_success = control_eligible & controlled.success
    control_failure = control_eligible & (~controlled.success)

    power = action.force_power[safe_actor]
    header_speed = incoming_speed * (
        1.0 - power * (1.0 - response.header_speed_retention)
    )
    header_redirected = jnp.where(
        power > 0.0,
        header_speed * impulse_direction,
        state.ball.velocity,
    )
    body_retention = jnp.where(
        chest,
        response.chest_speed_retention,
        response.challenge_speed_retention,
    )
    redirect_target = body_retention * incoming_speed * impulse_direction
    redirected = state.ball.velocity + power * (
        redirect_target - state.ball.velocity
    )
    redirected = jnp.where(head, header_redirected, redirected)
    gated_parry = action.force_power[safe_actor] > 0.0
    parry_direction = jnp.where(
        gated_parry,
        impulse_direction,
        jnp.asarray(
            (
                state.players.body_forward[safe_actor, 0],
                state.players.body_forward[safe_actor, 1],
                0.0,
            ),
            dtype=state.ball.velocity.dtype,
        ),
    )
    parry_target = incoming_speed * parry_direction
    parry_delta = parry_target - state.ball.velocity
    parry_fraction = jnp.minimum(
        jnp.where(gated_parry, action.force_power[safe_actor], 1.0),
        incoming_speed / (jnp.linalg.norm(parry_delta) + _DIV_EPS),
    )
    parried_velocity = state.ball.velocity + parry_fraction * parry_delta
    deliberate_kick = raw_contact & foot & (~challenge) & foot_actuation.applied
    body_redirect = raw_contact & (head | chest) & (~control_eligible)
    challenge_redirect = raw_contact & challenge & (~control_eligible)
    velocity = jnp.where(
        caught,
        jnp.zeros_like(state.ball.velocity),
        jnp.where(
            parried,
            parried_velocity,
            jnp.where(
                control_eligible,
                controlled.velocity,
                jnp.where(
                    body_redirect | challenge_redirect,
                    redirected,
                    jnp.where(
                        deliberate_kick,
                        foot_actuation.ball.velocity,
                        state.ball.velocity,
                    ),
                ),
            ),
        ),
    )
    spin = jnp.where(
        caught,
        jnp.zeros_like(state.ball.spin),
        jnp.where(deliberate_kick, foot_actuation.ball.spin, state.ball.spin),
    )
    ball = state.ball._replace(
        velocity=velocity,
        spin=spin,
        live=jnp.where(caught, False, state.ball.live | raw_contact),
    )

    outcome = jnp.where(
        control_success,
        jnp.where(
            raw_outcome == OUTCOME_INTERCEPTION,
            OUTCOME_INTERCEPTION,
            jnp.where(
                raw_outcome == OUTCOME_TACKLE_WON,
                OUTCOME_TACKLE_WON,
                OUTCOME_TRAP,
            ),
        ),
        jnp.where(
            control_failure | (raw_contact & control_request & head),
            OUTCOME_MISCONTROL,
            raw_outcome,
        ),
    ).astype(jnp.int32)
    deflection = (
        (outcome == OUTCOME_DEFLECTION)
        | (outcome == OUTCOME_MISCONTROL)
        | (outcome == OUTCOME_PARRY)
    )
    deliberate_save = is_goal_threat(
        state,
        roster,
        safe_actor,
        field_half_extent,
        ball=ball_geometry,
        goal=goal_geometry,
        physics=ball_physics,
        frame_physics=goal_frame_physics,
    )
    law11 = jnp.where(
        raw_contact,
        jnp.where(
            deflection,
            LAW11_DEFLECTION_NO_RESET,
            jnp.where(
                deliberate_save,
                LAW11_DELIBERATE_SAVE_NO_RESET,
                LAW11_DELIBERATE_PLAY_RESET,
            ),
        ),
        LAW11_NONE,
    ).astype(jnp.int32)
    result = ContactResult(
        actor=jnp.where(raw_contact, safe_actor, NO_PLAYER).astype(jnp.int32),
        mechanism=jnp.where(raw_contact, mechanism, MECHANISM_NONE).astype(jnp.int32),
        intent=jnp.where(raw_contact, intent, INTENT_MOVE).astype(jnp.int32),
        outcome=jnp.where(raw_contact, outcome, OUTCOME_NONE).astype(jnp.int32),
        restart_kind=jnp.int32(RK_NONE),
        law11_effect=law11,
        kick_applied=deliberate_kick,
        intent_source=jnp.where(
            raw_contact, INTENT_SOURCE_POLICY, INTENT_SOURCE_NONE
        ).astype(jnp.int32),
    )
    last_contact = jax.tree.map(
        lambda new, old: jnp.where(raw_contact, new, old),
        result,
        state.possession.last_contact,
    )
    gains_control = control_success | caught
    loses_control = raw_contact & (~gains_control)
    winner_team = roster.team_id[safe_actor]
    possession = state.possession._replace(
        team=jnp.where(
            gains_control,
            winner_team,
            jnp.where(loses_control, NO_TEAM, state.possession.team),
        ).astype(jnp.int32),
        player=jnp.where(
            gains_control,
            safe_actor,
            jnp.where(loses_control, NO_PLAYER, state.possession.player),
        ).astype(jnp.int32),
        previous_team=jnp.where(
            raw_contact & (state.possession.team != NO_TEAM),
            state.possession.team,
            state.possession.previous_team,
        ).astype(jnp.int32),
        control_ticks=jnp.where(
            gains_control,
            1,
            jnp.where(loses_control, 0, state.possession.control_ticks),
        ).astype(jnp.int32),
        last_contact=last_contact,
    )
    contact_ticks = duration_to_ticks(
        response.active_contact_interval_s,
        dt,
        rounding="nearest",
        minimum=1,
        maximum=MAX_INT32_TICKS,
    )
    deliberate_kick_ticks = max(
        contact_ticks,
        duration_to_ticks(
            response.deliberate_kick_contact_lock_s,
            dt,
            rounding="nearest",
            minimum=1,
            maximum=MAX_INT32_TICKS,
        ),
    )
    contact_lock = state.players.contact_lock_substeps.at[safe_actor].set(
        jnp.where(
            raw_contact,
            jnp.where(deliberate_kick, deliberate_kick_ticks, contact_ticks),
            state.players.contact_lock_substeps[safe_actor],
        )
    )
    challenge_attempt = geometry.possible & geometry.challenge_context
    lunge_fraction = jnp.clip(
        (geometry.horizontal_extension - contact.carry_radius_m)
        / jnp.maximum(
            contact.challenge_radius_m - contact.carry_radius_m,
            _DIV_EPS,
        ),
        0.0,
        1.0,
    )
    challenge_seconds = (
        response.challenge_recovery_s
        + lunge_fraction * response.max_lunge_extra_recovery_s
    )
    challenge_ticks = jnp.where(
        challenge_seconds > 0.0,
        jnp.maximum(1, jnp.rint(challenge_seconds / dt).astype(jnp.int32)),
        0,
    )
    challenge_recovery = jnp.where(
        challenge_attempt,
        jnp.maximum(
            state.players.challenge_recovery_substeps,
            challenge_ticks,
        ),
        state.players.challenge_recovery_substeps,
    )
    aerial_attempt = geometry.possible & (
        (geometry.mechanism == MECHANISM_HEAD) | geometry.goalkeeper_hand
    )
    standing_height = roster.height
    # Goalkeeper의 2 m hand envelope를 standing claim과 full-extension dive로
    # 구분한다. 공 중심이 아니라 가까운 표면까지의 extension을 사용하며, 낮은
    # lateral dive도 vertical effort가 0이라는 이유로 recovery를 우회하지 않는다.
    goalkeeper_horizontal_effort = jnp.clip(
        (
            geometry.horizontal_extension
            - jnp.asarray(contact.goalkeeper_standing_radius_m, jnp.float32)
        )
        / jnp.maximum(
            jnp.asarray(
                contact.goalkeeper_radius_m - contact.goalkeeper_standing_radius_m,
                jnp.float32,
            ),
            _DIV_EPS,
        ),
        0.0,
        1.0,
    )
    ball_bottom_height = state.ball.position[2] - jnp.asarray(
        ball_geometry.radius_m, state.ball.position.dtype
    )
    vertical_reach_effort = jnp.clip(
        (ball_bottom_height - standing_height)
        / jnp.maximum(roster.reach_height - standing_height, _DIV_EPS),
        0.0,
        1.0,
    )
    reach_effort = jnp.where(
        geometry.goalkeeper_hand,
        jnp.maximum(goalkeeper_horizontal_effort, vertical_reach_effort),
        vertical_reach_effort,
    )
    aerial_seconds = jnp.where(
        geometry.goalkeeper_hand,
        response.goalkeeper_dive_recovery_s,
        response.aerial_attempt_recovery_s,
    )
    aerial_duration = aerial_seconds * reach_effort
    aerial_ticks = jnp.where(
        aerial_duration > 0.0,
        jnp.maximum(1, jnp.rint(aerial_duration / dt).astype(jnp.int32)),
        0,
    )
    aerial_recovery = jnp.where(
        aerial_attempt & (reach_effort > 0.0),
        jnp.maximum(state.players.aerial_recovery_substeps, aerial_ticks),
        state.players.aerial_recovery_substeps,
    )
    old_player = state.possession.player
    old_valid = (old_player >= 0) & (old_player < player_count)
    safe_old = jnp.clip(old_player, 0, player_count - 1)
    # 상대 contact가 clean control 대신 deflection/miscontrol만 만들어도 직전
    # 소유자는 공을 잃었다. gains_control만 보면 이 loose-ball 경로가 lock을 우회해
    # 다음 decision에서 즉시 CONTROL을 다시 제출할 수 있다. Exit hysteresis owner를
    # 상대 ordinary CONTROL이 가져가는 경우도 같은 loss semantics를 쓴다.
    lost_to_opponent_contact = (
        raw_contact
        & old_valid
        & (winner_team != state.possession.team)
    )
    loss_ticks = (
        duration_to_ticks(
            response.possession_loss_lock_s,
            dt,
            rounding="nearest",
            minimum=1,
            maximum=MAX_INT32_TICKS,
        )
        if response.possession_loss_lock_s > 0.0
        else 0
    )
    possession_loss_lock = state.players.possession_loss_lock_substeps.at[safe_old].set(
        jnp.where(
            lost_to_opponent_contact,
            jnp.maximum(
                state.players.possession_loss_lock_substeps[safe_old],
                loss_ticks,
            ),
            state.players.possession_loss_lock_substeps[safe_old],
        )
    )
    yellow_cards = state.players.yellow_cards.at[safe_actor].add(
        jnp.where(
            contest.selected & (contest.discipline == DISCIPLINE_YELLOW), 1, 0
        ).astype(jnp.int32)
    )
    yellow = contest.selected & (contest.discipline == DISCIPLINE_YELLOW)
    direct_red = contest.selected & (contest.discipline == DISCIPLINE_RED)
    second_yellow = yellow & (state.players.yellow_cards[safe_actor] >= 1)
    sent_off = state.players.sent_off.at[safe_actor].set(
        state.players.sent_off[safe_actor] | direct_red | second_yellow
    )
    players = state.players._replace(
        contact_lock_substeps=contact_lock,
        challenge_recovery_substeps=challenge_recovery,
        aerial_recovery_substeps=aerial_recovery,
        possession_loss_lock_substeps=possession_loss_lock,
        yellow_cards=yellow_cards,
        sent_off=sent_off,
    )
    same_released_goalkeeper = (
        state.restart_release.active
        & (state.restart_release.kind == RK_GK_HOLD)
        & (safe_actor == state.restart_release.taker)
    )
    restricted_throw_goalkeeper = (
        state.restart_release.active
        & (state.restart_release.kind == RK_THROWIN)
        & roster.is_goalkeeper[safe_actor]
        & (roster.team_id[safe_actor] == state.restart_release.team)
    )
    preserve_release = same_released_goalkeeper | (
        restricted_throw_goalkeeper & (~deliberate_kick)
    )
    clear_release = raw_contact & (~preserve_release)
    restart_release = state.restart_release._replace(
        active=jnp.where(
            clear_release,
            jnp.bool_(False),
            state.restart_release.active,
        ),
        untouched=jnp.where(
            clear_release,
            jnp.bool_(False),
            state.restart_release.untouched,
        ),
    )
    return ContactResolution(
        state=state._replace(
            ball=ball,
            players=players,
            possession=possession,
            restart_release=restart_release,
        ),
        contact=result,
        contest=contest,
    )


__all__ = [
    "ContactResolution",
    "ContactResponsePhysics",
    "ControlResponse",
    "resolve_contact_response",
    "resolve_control_response",
]
