"""goalkeeper catch 이후 held-ball constraint, release와 8초 위반."""

import numbers
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.core.action import INTENT_PASS, INTENT_SHOOT
from messi.core.ball import BallPhysics
from messi.core.constants import (
    NO_PLAYER,
    NO_TEAM,
    RK_CORNER,
    RK_GK_HOLD,
    RK_NONE,
    RK_THROWIN,
    TEAM_0,
    TEAM_1,
)
from messi.core.contact import (
    LAW11_DELIBERATE_PLAY_RESET,
    MECHANISM_FOOT,
    MECHANISM_GOALKEEPER_HAND,
    MECHANISM_NONE,
    OUTCOME_CATCH,
    OUTCOME_NONE,
    OUTCOME_RELEASE,
    ContactPhysics,
    ContactResult,
    apply_foot_actuation,
)
from messi.core.geometry import BallGeometry
from messi.core.numeric import require_float32_representable
from messi.core.state import RestartReleaseProvenance
from messi.core.timebase import DEFAULT_TIMEBASE, MAX_INT32_TICKS, Timebase


@dataclass(frozen=True, slots=True)
class GoalkeeperHoldingPhysics:
    """IFAB Law 12 goalkeeper hand-control 제한."""

    hand_control_limit_s: float = 8.0

    def __post_init__(self) -> None:
        value = require_float32_representable(
            "hand_control_limit_s", self.hand_control_limit_s
        )
        if not 1.0 <= value <= 20.0:
            raise ValueError("hand_control_limit_s는 [1, 20]초 범위여야 한다")


_DEFAULT_HOLDING = GoalkeeperHoldingPhysics()
_DEFAULT_CONTACT = ContactPhysics()
_DEFAULT_BALL = BallGeometry()
_DEFAULT_BALL_PHYSICS = BallPhysics()


class HeldBallRelease(NamedTuple):
    state: object
    held: jax.Array
    released: jax.Array
    forced: jax.Array


class GoalkeeperHoldingEvent(NamedTuple):
    expired: jax.Array
    goalkeeper: jax.Array
    goalkeeper_team: jax.Array
    restart_team: jax.Array
    corner_position: jax.Array


class GoalkeeperHoldingResolution(NamedTuple):
    state: object
    event: GoalkeeperHoldingEvent


def empty_goalkeeper_holding_event(dtype=jnp.float32) -> GoalkeeperHoldingEvent:
    return GoalkeeperHoldingEvent(
        expired=jnp.bool_(False),
        goalkeeper=jnp.int32(NO_PLAYER),
        goalkeeper_team=jnp.int32(NO_TEAM),
        restart_team=jnp.int32(NO_TEAM),
        corner_position=jnp.zeros(3, dtype=dtype),
    )


def holding_limit_control_ticks(
    *,
    timebase: Timebase = DEFAULT_TIMEBASE,
    config: GoalkeeperHoldingPhysics = _DEFAULT_HOLDING,
) -> int:
    """SI 제한을 graph 밖에서 고정 control tick 수로 변환한다."""

    return timebase.control_steps_for(
        config.hand_control_limit_s,
        minimum=1,
        maximum=MAX_INT32_TICKS,
    )


def held_ball_mask(state, roster) -> tuple[jax.Array, jax.Array]:
    """Authoritative goalkeeper hand possession과 안전한 owner index를 반환한다.

    Dead ball과 ``OUTCOME_CATCH``만으로는 충분하지 않다. 현재 possession owner가
    실제 hand catch actor·team과 일치해야만 held-ball constraint를 활성화한다.
    """

    count = state.players.position.shape[0]
    actor = state.possession.player.astype(jnp.int32)
    valid = (actor >= 0) & (actor < count)
    safe = jnp.clip(actor, 0, count - 1)
    held = (
        valid
        & (~state.ball.live)
        & (state.possession.team == roster.team_id[safe])
        & (state.possession.last_contact.actor == actor)
        & (state.possession.last_contact.mechanism == MECHANISM_GOALKEEPER_HAND)
        & (state.possession.last_contact.outcome == OUTCOME_CATCH)
        & roster.is_goalkeeper[safe]
        & state.players.active[safe]
    )
    return held, safe


def goalkeeper_hand_restricted_mask(state, roster) -> jax.Array:
    """Backpass, direct throw-in과 own hand-release의 goalkeeper hand 제한."""

    count = state.players.position.shape[0]
    indices = jnp.arange(count, dtype=jnp.int32)
    release = state.restart_release
    # Law 12 restriction은 공이 다른 선수에게 닿을 때까지 유지된다. 같은
    # goalkeeper의 foot/body touch로 ``untouched``가 내려가도 hand 재접촉은
    # 합법화되지 않으므로 release lifecycle의 ``active``를 authoritative gate로 쓴다.
    release_valid = release.active & (
        (release.team == TEAM_0) | (release.team == TEAM_1)
    )
    own_hand_release = (
        release_valid
        & (release.kind == RK_GK_HOLD)
        & (indices == release.taker)
        & (roster.team_id == release.team)
    )
    own_direct_throw = (
        release_valid
        & (release.kind == RK_THROWIN)
        & (roster.team_id == release.team)
    )
    targeted_backpass = (state.gk_backpass_team == TEAM_0) | (
        state.gk_backpass_team == TEAM_1
    )
    targeted_backpass &= roster.team_id == state.gk_backpass_team
    return (
        state.players.active
        & roster.is_goalkeeper
        & (own_hand_release | own_direct_throw | targeted_backpass)
    )


def goalkeeper_hand_allowed_mask(
    state,
    roster,
    field_half_extent: jax.Array,
    *,
    penalty_area_length_m: float = 16.5,
    penalty_area_half_width_m: float = 20.16,
) -> jax.Array:
    """현재 substep state에서 합법적인 goalkeeper hand-contact mask를 계산한다."""

    extent = jnp.asarray(field_half_extent, dtype=state.players.position.dtype)
    direction = state.attack_direction[roster.team_id]
    own_goal_depth = extent[0] + direction * state.players.position[:, 0]
    own_penalty_area = (own_goal_depth >= 0.0) & (
        own_goal_depth <= penalty_area_length_m
    ) & (jnp.abs(state.players.position[:, 1]) <= penalty_area_half_width_m)
    ball_own_goal_depth = extent[0] + direction * state.ball.position[0]
    ball_in_own_penalty_area = (ball_own_goal_depth >= 0.0) & (
        ball_own_goal_depth <= penalty_area_length_m
    ) & (jnp.abs(state.ball.position[1]) <= penalty_area_half_width_m)
    return (
        state.players.active
        & roster.is_goalkeeper
        & own_penalty_area
        & ball_in_own_penalty_area
        & (~goalkeeper_hand_restricted_mask(state, roster))
    )


def _rolling_stop_distance(speed, physics: BallPhysics):
    """Configured piecewise rolling deceleration의 직선 정지거리."""

    dtype = speed.dtype
    query = jnp.maximum(speed, 0.0)
    knots = jnp.asarray(physics.rolling_speed_knots_mps, dtype=dtype)
    deceleration = jnp.asarray(
        physics.rolling_deceleration_knots_mps2, dtype=dtype
    )
    lower_speed = knots[:-1]
    upper_speed = knots[1:]
    lower_d = deceleration[:-1]
    upper_d = deceleration[1:]
    segment_upper = jnp.clip(query, lower_speed, upper_speed)
    slope = (upper_d - lower_d) / (upper_speed - lower_speed)
    nonconstant = jnp.abs(slope) > 1.0e-9
    safe_slope = jnp.where(nonconstant, slope, 1.0)
    intercept = lower_d - slope * lower_speed
    segment_d = jnp.maximum(
        lower_d + slope * (segment_upper - lower_speed), 1.0e-9
    )
    linear = (
        segment_d
        - lower_d
        - intercept * jnp.log(segment_d / lower_d)
    ) / (safe_slope * safe_slope)
    constant = (segment_upper**2 - lower_speed**2) / (2.0 * lower_d)
    distance = jnp.where(
        segment_upper > lower_speed,
        jnp.where(nonconstant, linear, constant),
        0.0,
    )
    tail = jnp.where(
        query > knots[-1],
        (query**2 - knots[-1] ** 2) / (2.0 * deceleration[-1]),
        0.0,
    )
    return jnp.sum(distance) + tail


def _supported_ground_path_length(
    horizontal_velocity: jax.Array,
    spin: jax.Array,
    *,
    ball: BallGeometry,
    physics: BallPhysics,
) -> jax.Array:
    """현재 ground integrator와 같은 slip-to-roll 뒤의 직선 정지거리."""

    dtype = horizontal_velocity.dtype
    speed = jnp.linalg.norm(horizontal_velocity)
    direction = horizontal_velocity / jnp.maximum(speed, 1.0e-9)
    surface_velocity = jnp.asarray((-spin[1], spin[0]), dtype=dtype) * jnp.asarray(
        ball.radius_m, dtype=dtype
    )
    slip = horizontal_velocity + surface_velocity
    inertia = jnp.asarray(physics.inertia_ratio, dtype=dtype)
    sticking_delta = -(inertia / (1.0 + inertia)) * slip
    friction_acceleration = jnp.maximum(
        jnp.asarray(
            physics.ground_slide_friction * physics.gravity_mps2, dtype=dtype
        ),
        1.0e-9,
    )
    slip_time = jnp.linalg.norm(sticking_delta) / friction_acceleration
    slip_displacement = (horizontal_velocity + 0.5 * sticking_delta) * slip_time
    compatible_velocity = horizontal_velocity + sticking_delta
    compatible_speed = jnp.linalg.norm(compatible_velocity)
    rolling_direction_fraction = jnp.maximum(
        jnp.dot(compatible_velocity, direction)
        / jnp.maximum(compatible_speed, 1.0e-9),
        0.0,
    )
    return jnp.maximum(jnp.dot(slip_displacement, direction), 0.0) + (
        _rolling_stop_distance(compatible_speed, physics)
        * rolling_direction_fraction
    )


def _conservative_path_length(
    position: jax.Array,
    velocity: jax.Array,
    spin: jax.Array,
    *,
    ball: BallGeometry,
    physics: BallPhysics,
) -> jax.Array:
    """Slide, bounded bounce와 rolling을 포함한 O(1) unopposed 경로 상한."""

    dtype = velocity.dtype
    horizontal_speed = jnp.linalg.norm(velocity[:2])
    gravity = jnp.maximum(jnp.asarray(physics.gravity_mps2, dtype=dtype), 1.0e-9)
    height = jnp.maximum(position[2] - ball.radius_m, 0.0)
    vertical = velocity[2]
    impact_speed = jnp.sqrt(
        jnp.maximum(vertical * vertical + 2.0 * gravity * height, 0.0)
    )
    first_flight = jnp.maximum((vertical + impact_speed) / gravity, 0.0)
    restitution = jnp.clip(jnp.asarray(physics.restitution, dtype=dtype), 0.0, 1.0)
    horizontal_keep = jnp.clip(
        jnp.asarray(physics.bounce_horizontal_retention, dtype=dtype), 0.0, 1.0
    )
    settle_speed = jnp.maximum(
        jnp.asarray(physics.ground_settle_speed_mps, dtype=dtype), 1.0e-9
    )
    can_bounce = (impact_speed > settle_speed) & (restitution > 1.0e-9)
    safe_restitution = jnp.clip(restitution, 1.0e-9, 1.0 - 1.0e-9)
    bounce_count = jnp.maximum(
        jnp.where(
            can_bounce,
            jnp.ceil(
                jnp.log(settle_speed / jnp.maximum(impact_speed, 1.0e-9))
                / jnp.log(safe_restitution)
            ),
            0.0,
        ),
        0.0,
    )
    ratio = restitution * horizontal_keep
    ratio_sum = jnp.where(
        bounce_count > 0.0,
        ratio
        * (1.0 - jnp.power(ratio, bounce_count))
        / jnp.maximum(1.0 - ratio, 1.0e-9),
        0.0,
    )
    later_flight = 2.0 * impact_speed / gravity * ratio_sum
    post_bounce_speed = horizontal_speed * jnp.power(
        horizontal_keep, bounce_count
    )
    airborne_path = horizontal_speed * (first_flight + later_flight) + (
        _rolling_stop_distance(post_bounce_speed, physics)
    )
    supported_ground = (height <= 1.0e-6) & (
        jnp.abs(vertical) <= physics.ground_settle_speed_mps
    )
    ground_path = _supported_ground_path_length(
        velocity[:2], spin, ball=ball, physics=physics
    )
    return jnp.where(supported_ground, ground_path, airborne_path)


def targeted_own_goalkeeper_team(
    state,
    roster,
    actor,
    release_position,
    outgoing_velocity,
    *,
    ball: BallGeometry = _DEFAULT_BALL,
    physics: BallPhysics = _DEFAULT_BALL_PHYSICS,
    contact: ContactPhysics = _DEFAULT_CONTACT,
) -> jax.Array:
    """실제 foot release의 유한 직선 경로가 own goalkeeper reach와 만나는지 판정한다."""

    count = state.players.position.shape[0]
    actor = jnp.asarray(actor, dtype=jnp.int32)
    valid_actor = (actor >= 0) & (actor < count)
    safe_actor = jnp.clip(actor, 0, count - 1)
    actor_team = roster.team_id[safe_actor]
    valid_team = (actor_team == TEAM_0) | (actor_team == TEAM_1)
    horizontal = outgoing_velocity[:2]
    speed = jnp.linalg.norm(horizontal)
    direction = horizontal / jnp.maximum(speed, 1.0e-9)
    path_length = _conservative_path_length(
        release_position,
        outgoing_velocity,
        state.ball.spin,
        ball=ball,
        physics=physics,
    )
    to_keeper = state.players.position - release_position[:2]
    along = jnp.sum(to_keeper * direction, axis=-1)
    closest_along = jnp.clip(along, 0.0, path_length)
    lateral = to_keeper - closest_along[:, None] * direction
    arrival = closest_along / jnp.maximum(speed, 1.0e-9)
    claim_radius = jnp.asarray(
        contact.goalkeeper_radius_m + ball.radius_m, dtype=speed.dtype
    )
    reachable_radius = claim_radius + roster.max_speed * arrival
    indices = jnp.arange(count, dtype=jnp.int32)
    own_keeper = (
        state.players.active
        & roster.is_goalkeeper
        & (roster.team_id == actor_team)
        & (indices != actor)
    )
    targeted = (
        valid_actor
        & valid_team
        & state.players.active[safe_actor]
        & (speed > 1.0e-6)
        & jnp.any(
            own_keeper
            & (along > 1.0e-6)
            & (along <= path_length)
            & (jnp.sum(lateral * lateral, axis=-1) <= reachable_radius**2)
        )
    )
    return jnp.where(targeted, actor_team, NO_TEAM).astype(jnp.int32)


def update_backpass_latch(
    state,
    roster,
    contact: ContactResult,
    *,
    ball: BallGeometry = _DEFAULT_BALL,
    physics: BallPhysics = _DEFAULT_BALL_PHYSICS,
    contact_physics: ContactPhysics = _DEFAULT_CONTACT,
):
    """Realized contact로 기존 latch를 clear하고 targeted foot release면 새로 arm한다."""

    count = state.players.position.shape[0]
    actor = contact.actor
    valid_actor = (actor >= 0) & (actor < count)
    safe_actor = jnp.clip(actor, 0, count - 1)
    actual = valid_actor & (contact.outcome != OUTCOME_NONE)
    restricted_keeper = (
        valid_actor
        & roster.is_goalkeeper[safe_actor]
        & (roster.team_id[safe_actor] == state.gk_backpass_team)
    )
    restricted_keeper_kick = restricted_keeper & contact.kick_applied
    clear = actual & ((~restricted_keeper) | restricted_keeper_kick)
    cleared = jnp.where(clear, NO_TEAM, state.gk_backpass_team).astype(jnp.int32)
    targeted = targeted_own_goalkeeper_team(
        state,
        roster,
        actor,
        state.ball.position,
        state.ball.velocity,
        ball=ball,
        physics=physics,
        contact=contact_physics,
    )
    arm = actual & contact.kick_applied & (~roster.is_goalkeeper[safe_actor])
    return state._replace(
        gk_backpass_team=jnp.where(arm, targeted, cleared).astype(jnp.int32)
    )


def clear_backpass_after_passive_contact(state, roster, actor, occurred):
    """다른 선수의 passive contact에서 latch를 clear하고 제한된 GK의 body hit은 유지한다."""

    count = state.players.position.shape[0]
    actor = jnp.asarray(actor, dtype=jnp.int32)
    valid = (actor >= 0) & (actor < count)
    safe_actor = jnp.clip(actor, 0, count - 1)
    restricted_keeper = (
        valid
        & goalkeeper_hand_restricted_mask(state, roster)[safe_actor]
        & (roster.team_id[safe_actor] == state.gk_backpass_team)
    )
    clear = jnp.asarray(occurred) & valid & (~restricted_keeper)
    return state._replace(
        gk_backpass_team=jnp.where(clear, NO_TEAM, state.gk_backpass_team).astype(
            jnp.int32
        )
    )


def goalkeeper_hand_anchor(
    state,
    roster,
    *,
    ball: BallGeometry = _DEFAULT_BALL,
    contact: ContactPhysics = _DEFAULT_CONTACT,
) -> jax.Array:
    held, keeper = held_ball_mask(state, roster)
    forward = state.players.body_forward[keeper]
    height = jnp.minimum(
        roster.reach_height[keeper],
        roster.height[keeper] * contact.torso_top_height_factor,
    )
    anchor = jnp.asarray(
        (
            state.players.position[keeper, 0] + ball.radius_m * forward[0],
            state.players.position[keeper, 1] + ball.radius_m * forward[1],
            height,
        ),
        dtype=state.ball.position.dtype,
    )
    return jnp.where(held, anchor, state.ball.position)


def sync_held_ball(
    state,
    roster,
    *,
    ball: BallGeometry = _DEFAULT_BALL,
    contact: ContactPhysics = _DEFAULT_CONTACT,
):
    held, _ = held_ball_mask(state, roster)
    ball = state.ball._replace(
        position=goalkeeper_hand_anchor(state, roster, ball=ball, contact=contact),
        velocity=jnp.where(
            held, jnp.zeros_like(state.ball.velocity), state.ball.velocity
        ),
        spin=jnp.where(held, jnp.zeros_like(state.ball.spin), state.ball.spin),
        live=jnp.where(held, False, state.ball.live),
    )
    return state._replace(ball=ball)


def release_held_ball(
    state,
    roster,
    action,
    *,
    ball: BallGeometry = _DEFAULT_BALL,
    contact: ContactPhysics = _DEFAULT_CONTACT,
) -> HeldBallRelease:
    """PASS/SHOOT 요청으로 held ball을 원자적으로 방출한다."""
    state = sync_held_ball(state, roster, ball=ball, contact=contact)
    held, keeper = held_ball_mask(state, roster)
    intent = action.requested_intent[keeper]
    requested = (intent == INTENT_PASS) | (intent == INTENT_SHOOT)
    candidate_ball = state.ball._replace(live=jnp.bool_(True))
    actuation = apply_foot_actuation(
        candidate_ball,
        state.players,
        action,
        keeper,
        ball_geometry=ball,
        config=contact,
    )
    deliberate = held & requested & actuation.applied
    released = deliberate
    release_ball = actuation.ball._replace(live=jnp.bool_(True))
    ball = jax.tree.map(
        lambda new, old: jnp.where(released, new, old), release_ball, state.ball
    )
    result = ContactResult(
        actor=keeper,
        # 현재 action schema에는 hand throw가 없다. Held PASS/SHOOT는 실제
        # actuator와 동일하게 foot release provenance를 남긴다.
        mechanism=jnp.int32(MECHANISM_FOOT),
        intent=intent.astype(jnp.int32),
        outcome=jnp.int32(OUTCOME_RELEASE),
        restart_kind=state.possession.last_contact.restart_kind,
        law11_effect=jnp.int32(LAW11_DELIBERATE_PLAY_RESET),
        kick_applied=deliberate,
        intent_source=state.possession.last_contact.intent_source,
    )
    possession = state.possession._replace(
        team=jnp.where(released, NO_TEAM, state.possession.team).astype(jnp.int32),
        player=jnp.where(released, NO_PLAYER, state.possession.player).astype(
            jnp.int32
        ),
        previous_team=jnp.where(
            released, state.possession.team, state.possession.previous_team
        ).astype(jnp.int32),
        control_ticks=jnp.where(released, 0, state.possession.control_ticks).astype(
            jnp.int32
        ),
        last_contact=jax.tree.map(
            lambda new, old: jnp.where(released, new, old),
            result,
            state.possession.last_contact,
        ),
    )
    release = RestartReleaseProvenance(
        active=jnp.bool_(True),
        untouched=jnp.bool_(True),
        kind=jnp.int32(RK_GK_HOLD),
        team=state.possession.team.astype(jnp.int32),
        taker=keeper,
        indirect=jnp.bool_(False),
        law11_direct_exempt=jnp.bool_(False),
        release_mechanism=jnp.int32(MECHANISM_FOOT),
    )
    restart_release = jax.tree.map(
        lambda new, old: jnp.where(released, new, old),
        release,
        state.restart_release,
    )
    return HeldBallRelease(
        state._replace(
            ball=ball, possession=possession, restart_release=restart_release
        ),
        held,
        released,
        jnp.bool_(False),
    )


def resolve_goalkeeper_holding_expiry(
    state,
    roster,
    field_half_extent: jax.Array,
    *,
    limit_control_ticks: int,
    ball_radius_m: float = 0.11,
) -> GoalkeeperHoldingResolution:
    """8초를 넘겨 계속 보유한 goalkeeper에게 상대 corner를 부여한다."""

    if not isinstance(limit_control_ticks, numbers.Integral) or isinstance(
        limit_control_ticks, bool
    ):
        raise TypeError("limit_control_ticks는 bool이 아닌 정수여야 한다")
    limit_control_ticks = int(limit_control_ticks)
    if not 1 <= limit_control_ticks <= MAX_INT32_TICKS:
        raise ValueError("limit_control_ticks는 positive int32 범위여야 한다")
    held, goalkeeper = held_ball_mask(state, roster)
    goalkeeper_team = roster.team_id[goalkeeper]
    valid_team = (goalkeeper_team == TEAM_0) | (goalkeeper_team == TEAM_1)
    expired = (
        held
        & valid_team
        & (state.possession.control_ticks >= jnp.int32(limit_control_ticks))
    )
    restart_team = (TEAM_1 - goalkeeper_team).astype(jnp.int32)
    defending_end = -state.attack_direction[goalkeeper_team]
    side = jnp.where(state.players.position[goalkeeper, 1] >= 0.0, 1.0, -1.0)
    corner = jnp.asarray(
        (
            defending_end * field_half_extent[0],
            side * field_half_extent[1],
            jnp.asarray(ball_radius_m, dtype=state.ball.position.dtype),
        ),
        dtype=state.ball.position.dtype,
    )
    eligible = state.players.active & (roster.team_id == restart_team)
    distance_squared = jnp.sum(
        (state.players.position - corner[:2]) ** 2, axis=-1
    )
    nearest = jnp.argmin(jnp.where(eligible, distance_squared, jnp.inf))
    taker = jnp.where(jnp.any(eligible), nearest, NO_PLAYER).astype(jnp.int32)
    candidate = state._replace(
        ball=state.ball._replace(
            position=corner,
            velocity=jnp.zeros_like(state.ball.velocity),
            spin=jnp.zeros_like(state.ball.spin),
            live=jnp.bool_(False),
        ),
        possession=state.possession._replace(
            team=jnp.int32(NO_TEAM),
            player=jnp.int32(NO_PLAYER),
            previous_team=state.possession.team.astype(jnp.int32),
            control_ticks=jnp.int32(0),
        ),
        restart=state.restart._replace(
            kind=jnp.int32(RK_CORNER),
            team=restart_team,
            substeps_remaining=jnp.int32(0),
            taker=taker,
            indirect=jnp.bool_(False),
            opened_control_tick=state.control_tick,
        ),
        restart_release=RestartReleaseProvenance(
            active=jnp.bool_(False),
            untouched=jnp.bool_(False),
            kind=jnp.int32(RK_NONE),
            team=jnp.int32(NO_TEAM),
            taker=jnp.int32(NO_PLAYER),
            indirect=jnp.bool_(False),
            law11_direct_exempt=jnp.bool_(False),
            release_mechanism=jnp.int32(MECHANISM_NONE),
        ),
        gk_backpass_team=jnp.int32(NO_TEAM),
        offside_flagged=jnp.zeros_like(state.offside_flagged),
        offside_direct_exempt_team=jnp.int32(NO_TEAM),
        restart_layout_ready=jnp.bool_(False),
        restart_taker_approach=jnp.bool_(False),
        penalty_completion_active=jnp.bool_(False),
        penalty_completion_team=jnp.int32(NO_TEAM),
    )
    next_state = jax.tree.map(
        lambda new, old: jnp.where(expired, new, old), candidate, state
    )
    event = GoalkeeperHoldingEvent(
        expired=expired,
        goalkeeper=jnp.where(expired, goalkeeper, NO_PLAYER).astype(jnp.int32),
        goalkeeper_team=jnp.where(expired, goalkeeper_team, NO_TEAM).astype(jnp.int32),
        restart_team=jnp.where(expired, restart_team, NO_TEAM).astype(jnp.int32),
        corner_position=jnp.where(expired, corner, jnp.zeros_like(corner)),
    )
    return GoalkeeperHoldingResolution(next_state, event)


__all__ = [
    "GoalkeeperHoldingEvent",
    "GoalkeeperHoldingPhysics",
    "GoalkeeperHoldingResolution",
    "HeldBallRelease",
    "clear_backpass_after_passive_contact",
    "empty_goalkeeper_holding_event",
    "goalkeeper_hand_allowed_mask",
    "goalkeeper_hand_anchor",
    "goalkeeper_hand_restricted_mask",
    "held_ball_mask",
    "holding_limit_control_ticks",
    "release_held_ball",
    "resolve_goalkeeper_holding_expiry",
    "sync_held_ball",
    "targeted_own_goalkeeper_team",
    "update_backpass_latch",
]
