"""compact contact semantics, shared geometry와 bounded foot actuator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.core.action import (
    INTENT_CHALLENGE,
    INTENT_CONTROL,
    INTENT_MOVE,
    INTENT_PASS,
    INTENT_SHOOT,
    IntentAction,
)
from messi.core.constants import NO_PLAYER, RK_NONE
from messi.core.geometry import BallGeometry
from messi.core.numeric import require_float32_representable

_SAFE_NORM_EPS = 1.0e-18
_DIV_EPS = 1.0e-9
_STATIONARY_SPEED_EPS = 1.0e-3

MECHANISM_NONE = 0
MECHANISM_FOOT = 1
MECHANISM_HEAD = 2
MECHANISM_CHEST = 3
MECHANISM_PASSIVE_BODY = 4
MECHANISM_GOALKEEPER_HAND = 5
MECHANISM_THROW = 6
MECHANISM_COUNT = 7

INTENT_SOURCE_SCHEMA = "messi.intent-source/1"
INTENT_SOURCE_NONE = 0
INTENT_SOURCE_POLICY = 1
INTENT_SOURCE_ENVIRONMENT_FORCED = 2
INTENT_SOURCE_NAMES = ("NONE", "POLICY", "ENVIRONMENT_FORCED")
INTENT_SOURCE_COUNT = len(INTENT_SOURCE_NAMES)

OUTCOME_NONE = 0
OUTCOME_RELEASE = 1
OUTCOME_TRAP = 2
OUTCOME_INTERCEPTION = 3
OUTCOME_TACKLE_WON = 4
OUTCOME_DEFLECTION = 5
OUTCOME_CATCH = 6
OUTCOME_PARRY = 7
OUTCOME_FOUL = 8
OUTCOME_MISCONTROL = 9
OUTCOME_COUNT = 10

LAW11_NONE = 0
LAW11_DELIBERATE_PLAY_RESET = 1
LAW11_DEFLECTION_NO_RESET = 2
LAW11_DELIBERATE_SAVE_NO_RESET = 3
LAW11_DIRECT_RESTART_EXEMPTION = 4
LAW11_EFFECT_COUNT = 5


class ContactResult(NamedTuple):
    """possession provenance에 필요한 마지막 realized contact."""

    actor: jax.Array
    mechanism: jax.Array
    intent: jax.Array
    outcome: jax.Array
    restart_kind: jax.Array
    law11_effect: jax.Array
    kick_applied: jax.Array
    intent_source: jax.Array


class ContactOccurrence(NamedTuple):
    """state carry에 넣지 않는 fixed-shape contact event sidecar."""

    occurred: jax.Array
    actor: jax.Array
    mechanism: jax.Array
    law11_effect: jax.Array
    position: jax.Array
    time_fraction: jax.Array


class PhysicsAction(NamedTuple):
    """control frame에서 한 번 decode해 모든 physics substep이 공유하는 action."""

    contact: jax.Array
    desired_velocity: jax.Array
    force_direction: jax.Array
    force_power: jax.Array
    launch: jax.Array
    spin: jax.Array
    body_target: jax.Array
    body_target_valid: jax.Array
    requested_intent: jax.Array


class ContactGeometry(NamedTuple):
    """한 substep에서 eligibility와 이후 contest가 공유하는 player축 geometry."""

    possible: jax.Array
    requested_intent: jax.Array
    mechanism: jax.Array
    challenge_context: jax.Array
    goalkeeper_hand: jax.Array
    goalkeeper_hand_restricted_attempt: jax.Array
    distance_squared: jax.Array
    distance: jax.Array
    horizontal_extension: jax.Array
    verified_carrier: jax.Array


class FootActuation(NamedTuple):
    """선택된 foot contact가 만든 ball state와 적용된 impulse 정보."""

    ball: object
    applied: jax.Array
    direction: jax.Array
    launch: jax.Array
    release_speed: jax.Array
    delta_velocity: jax.Array
    delta_spin: jax.Array


@dataclass(frozen=True, slots=True)
class ContactPhysics:
    """open-play reach와 foot actuator의 bounded 고정 계수."""

    carry_radius_m: float = 1.1
    challenge_radius_m: float = 1.4
    goalkeeper_radius_m: float = 2.0
    goalkeeper_standing_radius_m: float = 1.1
    pelvis_height_factor: float = 0.55
    torso_top_height_factor: float = 0.85
    block_speed_limit_mps: float = 31.0
    height_speed_penalty_mps_per_m: float = 3.5

    # "The influence of speed of approach and accuracy constraint on the
    # maximal speed of the ball in soccer kicking"은 28.60--34.48 m/s를,
    # "Role of Vertical Jumps and Anthropometric Variables in Maximal Kicking
    # Ball Velocities in Elite Soccer Players"는 최대 34.5 m/s를 보고했다.
    # 34.76 m/s는 관측 최대에 작은 수치 여유를 둔 design envelope다.
    kick_speed_max_mps: float = 34.76
    throw_speed_max_mps: float = 21.5
    restart_min_ball_speed_mps: float = 0.5
    throw_release_height_addition_m: float = 0.47
    control_request_speed_max_mps: float = 9.0
    launch_max_radians: float = 1.0
    ground_launch_down_max_radians: float = 0.21
    ground_launch_down_reference_height_m: float = 1.0

    # "The effect of surface geometry on soccer ball trajectories"에서 관측한
    # high-spin launch 영역을 감싸는 rounded design envelope다.
    spin_max_radps: float = 150.0

    def __post_init__(self) -> None:
        limits = {
            "carry_radius_m": (0.1, 2.0),
            "challenge_radius_m": (0.1, 2.5),
            "goalkeeper_radius_m": (0.1, 3.0),
            "goalkeeper_standing_radius_m": (0.1, 2.0),
            "pelvis_height_factor": (0.1, 1.0),
            "torso_top_height_factor": (0.1, 1.0),
            "block_speed_limit_mps": (1.0, 40.0),
            "height_speed_penalty_mps_per_m": (0.0, 20.0),
            "kick_speed_max_mps": (1.0, 35.0),
            "throw_speed_max_mps": (1.0, 30.0),
            "restart_min_ball_speed_mps": (0.1, 3.0),
            "throw_release_height_addition_m": (0.0, 1.0),
            "control_request_speed_max_mps": (0.1, 20.0),
            "launch_max_radians": (0.01, 0.5 * math.pi),
            "ground_launch_down_max_radians": (0.0, 0.5 * math.pi),
            "ground_launch_down_reference_height_m": (0.12, 3.0),
            "spin_max_radps": (1.0, 160.0),
        }
        for name, (lower, upper) in limits.items():
            value = require_float32_representable(name, getattr(self, name))
            if not lower <= value <= upper:
                raise ValueError(f"{name}은 [{lower}, {upper}] 범위여야 한다")
        if self.challenge_radius_m < self.carry_radius_m:
            raise ValueError("challenge_radius_m은 carry_radius_m 이상이어야 한다")
        if self.goalkeeper_radius_m < self.goalkeeper_standing_radius_m:
            raise ValueError(
                "goalkeeper_radius_m은 goalkeeper_standing_radius_m 이상이어야 한다"
            )
        if self.torso_top_height_factor < self.pelvis_height_factor:
            raise ValueError(
                "torso_top_height_factor는 pelvis_height_factor 이상이어야 한다"
            )
        if self.ground_launch_down_max_radians > self.launch_max_radians:
            raise ValueError(
                "ground_launch_down_max_radians는 launch_max_radians 이하여야 한다"
            )
        if self.restart_min_ball_speed_mps > min(
            self.kick_speed_max_mps, self.throw_speed_max_mps
        ):
            raise ValueError(
                "restart_min_ball_speed_mps는 kick_speed_max_mps와 "
                "throw_speed_max_mps 이하여야 한다"
            )


_DEFAULT_BALL_GEOMETRY = BallGeometry()
_DEFAULT_CONTACT_PHYSICS = ContactPhysics()


def decode_physics_action(
    action: IntentAction,
    roster,
    attack_direction: jax.Array,
    body_forward: jax.Array,
) -> PhysicsAction:
    """state-invariant action 변환을 control frame마다 정확히 한 번 수행한다."""

    if not isinstance(action, IntentAction):
        raise TypeError("action은 IntentAction이어야 한다")
    decoded = action.decode()
    team_direction = jnp.asarray(attack_direction, dtype=jnp.float32)[roster.team_id]
    team_to_world = jnp.stack(
        (team_direction, jnp.ones_like(team_direction)), axis=-1
    )
    move_planar = decoded.move.direction * decoded.move.power[..., None]
    desired_velocity = (
        move_planar
        * team_to_world
        * jnp.asarray(roster.max_speed, dtype=jnp.float32)[..., None]
    )
    force_direction = decoded.force_to_ball.direction * team_to_world
    move_direction = decoded.move.direction * team_to_world
    body_aim_valid = (~decoded.contact) & (
        decoded.force_to_ball.power > jnp.finfo(jnp.float32).eps
    )
    move_aim_valid = (
        (~decoded.contact)
        & (~body_aim_valid)
        & (decoded.move.power > jnp.finfo(jnp.float32).eps)
    )
    body_target = jnp.where(
        body_aim_valid[..., None],
        force_direction,
        jnp.where(move_aim_valid[..., None], move_direction, body_forward),
    )
    spin = jnp.clip(jnp.asarray(decoded.spin, dtype=jnp.float32), -1.0, 1.0)
    spin_norm = jnp.sqrt(jnp.sum(spin * spin, axis=-1) + _SAFE_NORM_EPS)
    spin = spin / jnp.maximum(1.0, spin_norm)[..., None]
    return PhysicsAction(
        contact=decoded.contact,
        desired_velocity=desired_velocity.astype(jnp.float32),
        force_direction=force_direction.astype(jnp.float32),
        force_power=decoded.force_to_ball.power.astype(jnp.float32),
        launch=decoded.launch.astype(jnp.float32),
        spin=spin.astype(jnp.float32),
        body_target=body_target.astype(jnp.float32),
        body_target_valid=body_aim_valid | move_aim_valid,
        requested_intent=decoded.intent.astype(jnp.int32),
    )


def verified_controlled_carrier(
    state, roster, config: ContactPhysics, radius: float
) -> jax.Array:
    """저장된 possession actor가 strict carry 또는 exit hysteresis를 유지하는지 판정한다."""

    player_count = state.players.position.shape[0]
    carrier = state.possession.player
    valid = (
        (carrier >= 0) & (carrier < player_count) & (state.possession.control_ticks > 0)
    )
    safe = jnp.clip(carrier, 0, player_count - 1)
    delta = state.ball.position[:2] - state.players.position[safe]
    distance_squared = jnp.sum(delta * delta)
    retained_chest = (
        valid
        & (state.possession.last_contact.actor == safe)
        & (state.possession.last_contact.intent == INTENT_CONTROL)
        & (state.possession.last_contact.outcome == OUTCOME_TRAP)
        & (state.possession.last_contact.mechanism == MECHANISM_CHEST)
    )
    control_height = jnp.where(
        retained_chest,
        roster.reach_height[safe],
        roster.height[safe] * config.pelvis_height_factor,
    )
    common = (
        valid
        & state.ball.live
        & state.players.active[safe]
        & (state.possession.team == roster.team_id[safe])
        & (state.ball.position[2] <= control_height + radius)
    )
    strict_controls = common & (
        distance_squared <= (config.carry_radius_m + radius) ** 2
    )
    # 이미 저장된 owner가 strict carry를 아주 조금 벗어난 경우에만 attribution을
    # 안정화한다. 새 owner를 만들지 않고 ordinary contact reach도 넓히지 않는다.
    all_delta = state.ball.position[:2] - state.players.position
    all_distance_squared = jnp.sum(all_delta * all_delta, axis=-1)
    indices = jnp.arange(player_count, dtype=jnp.int32)
    nearest_other_squared = jnp.min(
        jnp.where(
            state.players.active & (indices != safe),
            all_distance_squared,
            jnp.inf,
        )
    )
    dominance_margin = jnp.asarray(
        config.challenge_radius_m - config.carry_radius_m,
        dtype=state.ball.position.dtype,
    )
    owner_distance = jnp.sqrt(distance_squared + _SAFE_NORM_EPS)
    ball_speed_squared = jnp.sum(state.ball.velocity[:2] ** 2)
    hysteresis_controls = (
        common
        & (~strict_controls)
        & (distance_squared <= (config.challenge_radius_m + radius) ** 2)
        & (ball_speed_squared <= roster.max_speed[safe] ** 2)
        & (
            (owner_distance + dominance_margin) ** 2
            <= nearest_other_squared
        )
    )
    controls = strict_controls | hysteresis_controls
    return jnp.where(controls, carrier, NO_PLAYER).astype(jnp.int32)


def fresh_trap_control_grace(state, roster) -> jax.Array:
    """성공한 trap이 다음 policy decision까지 한 번 관측되도록 보존한다."""

    player_count = state.players.position.shape[0]
    player = state.possession.player
    valid = (player >= 0) & (player < player_count)
    safe = jnp.clip(player, 0, player_count - 1)
    return (
        state.ball.live
        & (state.restart.kind == jnp.int32(RK_NONE))
        & valid
        & (state.possession.control_ticks == jnp.int32(1))
        & (state.possession.last_contact.actor == player)
        & (state.possession.last_contact.intent == jnp.int32(INTENT_CONTROL))
        & (state.possession.last_contact.outcome == jnp.int32(OUTCOME_TRAP))
        & state.players.active[safe]
        & (roster.team_id[safe] == state.possession.team)
    )


def evaluate_contact_geometry(
    state,
    roster,
    action: PhysicsAction,
    contact_attempted: jax.Array,
    *,
    goalkeeper_hand_allowed: jax.Array = False,
    goalkeeper_hand_restricted: jax.Array = False,
    ball_geometry: BallGeometry = _DEFAULT_BALL_GEOMETRY,
    config: ContactPhysics = _DEFAULT_CONTACT_PHYSICS,
) -> ContactGeometry:
    """현재 substep의 shared geometry를 한 번 계산해 open-play 후보를 반환한다."""

    player_count = state.players.position.shape[0]
    contact_attempted = jnp.broadcast_to(
        jnp.asarray(contact_attempted, dtype=jnp.bool_), (player_count,)
    )
    goalkeeper_hand_allowed = jnp.broadcast_to(
        jnp.asarray(goalkeeper_hand_allowed, dtype=jnp.bool_), (player_count,)
    )
    goalkeeper_hand_restricted = jnp.broadcast_to(
        jnp.asarray(goalkeeper_hand_restricted, dtype=jnp.bool_), (player_count,)
    )
    delta_xy = state.ball.position[:2] - state.players.position
    distance_squared = jnp.sum(delta_xy * delta_xy, axis=-1)
    distance = jnp.sqrt(distance_squared + _SAFE_NORM_EPS)
    radius = ball_geometry.radius_m
    ball_height = state.ball.position[2]
    pelvis = roster.height * config.pelvis_height_factor
    torso_top = roster.height * config.torso_top_height_factor
    foot_height = ball_height <= pelvis + radius
    chest_height = (ball_height >= pelvis - radius) & (
        ball_height <= torso_top + radius
    )
    head_height = (ball_height >= torso_top - radius) & (
        ball_height <= roster.reach_height + radius
    )
    height_mechanism = jnp.where(
        foot_height,
        MECHANISM_FOOT,
        jnp.where(chest_height, MECHANISM_CHEST, MECHANISM_HEAD),
    ).astype(jnp.int32)
    intent = action.requested_intent
    hand = (
        goalkeeper_hand_allowed
        & roster.is_goalkeeper
        & ((intent == INTENT_CONTROL) | (intent == INTENT_SHOOT))
    )
    mechanism = jnp.where(hand, MECHANISM_GOALKEEPER_HAND, height_mechanism).astype(
        jnp.int32
    )

    verified_carrier = verified_controlled_carrier(state, roster, config, radius)
    carrier_valid = verified_carrier != NO_PLAYER
    safe_carrier = jnp.clip(verified_carrier, 0, player_count - 1)
    opposing_carrier = carrier_valid & (roster.team_id != roster.team_id[safe_carrier])
    challenge_context = (intent == INTENT_CHALLENGE) & opposing_carrier
    ordinary_intent = (
        (intent == INTENT_CONTROL) | (intent == INTENT_PASS) | (intent == INTENT_SHOOT)
    )
    intent_allowed = (ordinary_intent & (~opposing_carrier)) | challenge_context
    contact_radius = jnp.where(
        hand,
        config.goalkeeper_radius_m,
        jnp.where(challenge_context, config.challenge_radius_m, config.carry_radius_m),
    )
    horizontal_reach = distance_squared <= (contact_radius + radius) ** 2
    height_allowed = jnp.where(
        mechanism == MECHANISM_FOOT,
        foot_height,
        jnp.where(
            mechanism == MECHANISM_CHEST,
            chest_height,
            jnp.where(
                mechanism == MECHANISM_HEAD,
                head_height,
                ball_height <= roster.reach_height + radius,
            ),
        ),
    )
    horizontal_ball_speed = jnp.sqrt(
        jnp.sum(state.ball.velocity[:2] * state.ball.velocity[:2]) + _SAFE_NORM_EPS
    )
    foot_speed_allowed = (
        horizontal_ball_speed + config.height_speed_penalty_mps_per_m * ball_height
        <= config.block_speed_limit_mps
    )
    speed_allowed = (mechanism != MECHANISM_FOOT) | foot_speed_allowed
    recovery_ready = (
        (state.players.contact_lock_substeps <= 0)
        & (state.players.challenge_recovery_substeps <= 0)
        & (state.players.aerial_recovery_substeps <= 0)
        & (state.players.possession_loss_lock_substeps <= 0)
    )
    # CONTROL/SHOOT는 hand intent가 아니다. 제한된 goalkeeper도 foot/chest/head로
    # 합법적으로 처리할 수 있으면 그 경로를 유지하고, goalkeeper hand reach만이
    # 접촉을 가능하게 하는 geometry에서만 hand offence attempt로 해석한다.
    ordinary_contact_possible = (
        intent_allowed
        & horizontal_reach
        & height_allowed
        & speed_allowed
    )
    restricted_hand_attempt = (
        state.ball.live
        & state.players.active
        & roster.is_goalkeeper
        & goalkeeper_hand_restricted
        & action.contact
        & (~contact_attempted)
        & ((intent == INTENT_CONTROL) | (intent == INTENT_SHOOT))
        & (distance_squared <= (config.goalkeeper_radius_m + radius) ** 2)
        & (ball_height <= roster.reach_height + radius)
        & (~ordinary_contact_possible)
        & recovery_ready
    )
    possible = (
        state.ball.live
        & state.players.active
        & action.contact
        & (~contact_attempted)
        & intent_allowed
        & horizontal_reach
        & height_allowed
        & speed_allowed
        & recovery_ready
    )
    return ContactGeometry(
        possible=possible,
        requested_intent=intent,
        mechanism=mechanism,
        challenge_context=challenge_context,
        goalkeeper_hand=hand,
        goalkeeper_hand_restricted_attempt=restricted_hand_attempt,
        distance_squared=distance_squared,
        distance=distance,
        horizontal_extension=jnp.maximum(distance - radius, 0.0),
        verified_carrier=verified_carrier,
    )


def _limit_generated_state(
    incoming: jax.Array,
    candidate: jax.Array,
    generated_limit: jax.Array,
) -> jax.Array:
    incoming_size = jnp.sqrt(jnp.sum(incoming * incoming) + _SAFE_NORM_EPS)
    candidate_size = jnp.sqrt(jnp.sum(candidate * candidate) + _SAFE_NORM_EPS)
    limit = jnp.maximum(incoming_size, generated_limit)
    return candidate * jnp.minimum(1.0, limit / (candidate_size + _DIV_EPS))


def apply_foot_actuation(
    ball,
    players,
    action: PhysicsAction,
    actor: jax.Array,
    *,
    ball_geometry: BallGeometry = _DEFAULT_BALL_GEOMETRY,
    config: ContactPhysics = _DEFAULT_CONTACT_PHYSICS,
) -> FootActuation:
    """이미 선택·검증된 actor의 PASS/SHOOT foot impulse를 적용한다."""

    player_count = players.position.shape[0]
    actor = jnp.asarray(actor, dtype=jnp.int32)
    actor_valid = (actor >= 0) & (actor < player_count)
    safe_actor = jnp.clip(actor, 0, player_count - 1)
    intent = action.requested_intent[safe_actor]
    applied = (
        actor_valid
        & ball.live
        & players.active[safe_actor]
        & ((intent == INTENT_PASS) | (intent == INTENT_SHOOT))
    )
    power = action.force_power[safe_actor]
    raw_direction = action.force_direction[safe_actor]
    direction = jnp.where(
        power > _STATIONARY_SPEED_EPS,
        raw_direction,
        players.body_forward[safe_actor],
    )
    direction_norm = jnp.sqrt(jnp.sum(direction * direction) + _SAFE_NORM_EPS)
    direction = direction / direction_norm
    height_fraction = jnp.clip(
        (ball.position[2] - ball_geometry.radius_m)
        / (config.ground_launch_down_reference_height_m - ball_geometry.radius_m),
        0.0,
        1.0,
    )
    launch_floor = -(
        config.ground_launch_down_max_radians
        + (config.launch_max_radians - config.ground_launch_down_max_radians)
        * height_fraction
    )
    launch = launch_floor + action.launch[safe_actor] * (
        config.launch_max_radians - launch_floor
    )
    impulse_direction = jnp.asarray(
        (
            jnp.cos(launch) * direction[0],
            jnp.cos(launch) * direction[1],
            jnp.sin(launch),
        ),
        dtype=ball.velocity.dtype,
    )
    release_speed = power * config.kick_speed_max_mps
    velocity = _limit_generated_state(
        ball.velocity,
        ball.velocity + release_speed * impulse_direction,
        jnp.asarray(config.kick_speed_max_mps, dtype=ball.velocity.dtype),
    )

    # "The effect of surface geometry on soccer ball trajectories" (Barber &
    # Carré, 2010, doi:10.1007/s12283-010-0048-x)의 launch spin 범위를 따른다.
    spin_limit = config.spin_max_radps * power
    lateral_axis = jnp.asarray(
        (-direction[1], direction[0], 0.0), dtype=ball.spin.dtype
    )
    spin_control = action.spin[safe_actor]
    delta_spin_request = spin_limit * (
        spin_control[0] * jnp.asarray((0.0, 0.0, 1.0), dtype=ball.spin.dtype)
        - spin_control[1] * lateral_axis
    )
    spin = _limit_generated_state(
        ball.spin,
        ball.spin + delta_spin_request,
        jnp.asarray(spin_limit, dtype=ball.spin.dtype),
    )
    next_ball = ball._replace(
        velocity=jnp.where(applied, velocity, ball.velocity),
        spin=jnp.where(applied, spin, ball.spin),
    )
    return FootActuation(
        ball=next_ball,
        applied=applied,
        direction=direction,
        launch=launch,
        release_speed=jnp.where(applied, release_speed, 0.0),
        delta_velocity=jnp.where(applied, velocity - ball.velocity, 0.0),
        delta_spin=jnp.where(applied, spin - ball.spin, 0.0),
    )


def empty_contact_result() -> ContactResult:
    """명시적인 int32·bool dtype의 contact 없음 값을 만든다."""

    return ContactResult(
        actor=jnp.int32(NO_PLAYER),
        mechanism=jnp.int32(MECHANISM_NONE),
        intent=jnp.int32(INTENT_MOVE),
        outcome=jnp.int32(OUTCOME_NONE),
        restart_kind=jnp.int32(RK_NONE),
        law11_effect=jnp.int32(LAW11_NONE),
        kick_applied=jnp.bool_(False),
        intent_source=jnp.int32(INTENT_SOURCE_NONE),
    )


__all__ = [
    "INTENT_SOURCE_COUNT",
    "INTENT_SOURCE_ENVIRONMENT_FORCED",
    "INTENT_SOURCE_NAMES",
    "INTENT_SOURCE_NONE",
    "INTENT_SOURCE_POLICY",
    "INTENT_SOURCE_SCHEMA",
    "LAW11_DEFLECTION_NO_RESET",
    "LAW11_DELIBERATE_PLAY_RESET",
    "LAW11_DELIBERATE_SAVE_NO_RESET",
    "LAW11_DIRECT_RESTART_EXEMPTION",
    "LAW11_EFFECT_COUNT",
    "LAW11_NONE",
    "MECHANISM_CHEST",
    "MECHANISM_COUNT",
    "MECHANISM_FOOT",
    "MECHANISM_GOALKEEPER_HAND",
    "MECHANISM_HEAD",
    "MECHANISM_NONE",
    "MECHANISM_PASSIVE_BODY",
    "MECHANISM_THROW",
    "OUTCOME_CATCH",
    "OUTCOME_COUNT",
    "OUTCOME_DEFLECTION",
    "OUTCOME_FOUL",
    "OUTCOME_INTERCEPTION",
    "OUTCOME_MISCONTROL",
    "OUTCOME_NONE",
    "OUTCOME_PARRY",
    "OUTCOME_RELEASE",
    "OUTCOME_TACKLE_WON",
    "OUTCOME_TRAP",
    "ContactGeometry",
    "ContactOccurrence",
    "ContactPhysics",
    "ContactResult",
    "FootActuation",
    "PhysicsAction",
    "apply_foot_actuation",
    "decode_physics_action",
    "empty_contact_result",
    "evaluate_contact_geometry",
    "fresh_trap_control_grace",
    "verified_controlled_carrier",
]
