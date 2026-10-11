"""action decode 1회와 fixed-length scan으로 구성한 lean control transition."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from messi.core.action import (
    INTENT_CHALLENGE,
    INTENT_MOVE,
    PARAMETER_FORCE_DIRECTION,
    PARAMETER_FORCE_POWER,
    PARAMETER_LAUNCH,
    PARAMETER_MOVE,
    PARAMETER_SPIN,
    IntentAction,
)
from messi.core.ball import BallGeometry, BallPhysics
from messi.core.ball_events import step_ball_events_chronological
from messi.core.body_foul import (
    BodyFoulOverride,
    BodyFoulPhysics,
    resolve_body_impact_foul,
    sample_body_foul_draws,
)
from messi.core.boundary import (
    BoundaryEvent,
    GoalGeometry,
    empty_boundary_event,
    resolve_boundary_crossing,
)
from messi.core.collision_config import BodyCollision, CollisionPhysics
from messi.core.constants import (
    DISCIPLINE_RED,
    DISCIPLINE_YELLOW,
    NO_PLAYER,
    NO_TEAM,
    RK_GK_HOLD,
    RK_NONE,
    RK_OFFSIDE,
    RK_THROWIN,
)
from messi.core.contact import (
    INTENT_SOURCE_NONE,
    LAW11_DEFLECTION_NO_RESET,
    MECHANISM_GOALKEEPER_HAND,
    MECHANISM_NONE,
    MECHANISM_PASSIVE_BODY,
    OUTCOME_CATCH,
    OUTCOME_DEFLECTION,
    OUTCOME_FOUL,
    OUTCOME_NONE,
    ContactPhysics,
    ContactResult,
    decode_physics_action,
    evaluate_contact_geometry,
    fresh_trap_control_grace,
    verified_controlled_carrier,
)
from messi.core.contact_response import ContactResponsePhysics, resolve_contact_response
from messi.core.contest import (
    SAMPLE_CONTEST,
    ContestOverride,
    ContestPhysics,
    resolve_contest,
    resolve_contest_sample,
)
from messi.core.discipline import classify_existing_contact_discipline
from messi.core.foul import (
    DEFAULT_FOUL_LAW,
    FoulEvent,
    FoulLaw,
    empty_foul_event,
    resolve_contest_foul,
)
from messi.core.goal_frame import (
    GoalFrameEvent,
    GoalFramePhysics,
    empty_goal_frame_event,
)
from messi.core.goalkeeper import (
    GoalkeeperHoldingEvent,
    GoalkeeperHoldingPhysics,
    clear_backpass_after_passive_contact,
    empty_goalkeeper_holding_event,
    goalkeeper_hand_allowed_mask,
    goalkeeper_hand_restricted_mask,
    holding_limit_control_ticks,
    release_held_ball,
    resolve_goalkeeper_holding_expiry,
    sync_held_ball,
    update_backpass_latch,
)
from messi.core.movement import PlayerPhysics
from messi.core.numeric import require_float32_field_half_extent
from messi.core.offside import (
    OffsideEvent,
    OffsideGeometry,
    detect_passive_offside,
    empty_offside_event,
    resolve_offside,
)
from messi.core.passive_contact import (
    NutmegEvent,
    PassiveContactEvent,
    PassiveContactPhysics,
    apply_passive_contact_provenance,
    detect_nutmeg,
    empty_nutmeg_event,
    empty_passive_contact_event,
)
from messi.core.player_step import step_players
from messi.core.randomness import frame_random_key, validate_prng_key
from messi.core.restart import (
    RestartPhysics,
    RestartPreparation,
    _enforce_restart_countdown_positions_with_status,
    advance_restart_clock,
    continuous_restart_approach_enabled,
    ordinary_restart_non_takers_positioned,
    prepare_restart,
    release_restart,
    repair_restart_taker,
    restart_active,
    restart_taker_release_pose,
)
from messi.core.restart_retouch import (
    RestartRetouchEvent,
    apply_indirect_restart,
    detect_indirect_offence,
    detect_restart_retouch,
    empty_restart_retouch_event,
)
from messi.core.roster import validate_player_capacity
from messi.core.rule_arbitration import (
    STOPPAGE_BOUNDARY,
    STOPPAGE_FOUL,
    STOPPAGE_OFFSIDE,
    STOPPAGE_RETOUCH,
    select_first_stoppage,
)
from messi.core.stamina import LongStamina, ShortStamina
from messi.core.state import RosterState, State
from messi.core.timebase import (
    DEFAULT_TIMEBASE,
    MAX_INT32_TICKS,
    Timebase,
    duration_to_ticks,
)


class AppliedContactEvent(NamedTuple):
    """Event Capture 전용 첫 realized deliberate contact."""

    occurred: jax.Array
    actor: jax.Array
    team: jax.Array
    intent: jax.Array
    mechanism: jax.Array
    outcome: jax.Array
    position: jax.Array
    physics_substep: jax.Array
    time_fraction: jax.Array
    direction: jax.Array
    launch: jax.Array
    spin: jax.Array
    # direction·launch는 정책이 요청한 값이고, release_velocity는 contact response가
    # 실제로 공에 준 3D 속도다. 탄도 높이 복원은 후자만 써야 한다.
    release_velocity: jax.Array
    restart_kind: jax.Array
    possession_changed: jax.Array
    kick_applied: jax.Array
    law11_effect: jax.Array
    attack_direction: jax.Array


class FrameEventChronology(NamedTuple):
    """동일 control frame의 분석 event가 발생한 physics substep."""

    rule_stoppage_substep: jax.Array
    passive_contact_substep: jax.Array
    woodwork_substep: jax.Array
    boundary_substep: jax.Array


def empty_frame_event_chronology() -> FrameEventChronology:
    return FrameEventChronology(
        rule_stoppage_substep=jnp.int32(-1),
        passive_contact_substep=jnp.int32(-1),
        woodwork_substep=jnp.int32(-1),
        boundary_substep=jnp.int32(-1),
    )


def empty_applied_contact_event(dtype=jnp.float32) -> AppliedContactEvent:
    return AppliedContactEvent(
        occurred=jnp.bool_(False),
        actor=jnp.int32(NO_PLAYER),
        team=jnp.int32(NO_TEAM),
        intent=jnp.int32(INTENT_MOVE),
        mechanism=jnp.int32(MECHANISM_NONE),
        outcome=jnp.int32(OUTCOME_NONE),
        position=jnp.zeros(3, dtype=dtype),
        physics_substep=jnp.int32(-1),
        time_fraction=jnp.asarray(0.0, dtype=dtype),
        direction=jnp.zeros(2, dtype=dtype),
        launch=jnp.asarray(0.0, dtype=dtype),
        spin=jnp.zeros(2, dtype=dtype),
        release_velocity=jnp.zeros(3, dtype=dtype),
        restart_kind=jnp.int32(RK_NONE),
        possession_changed=jnp.bool_(False),
        kick_applied=jnp.bool_(False),
        law11_effect=jnp.int32(0),
        attack_direction=jnp.asarray(1.0, dtype=dtype),
    )


def _applied_contact_event(
    occurred,
    contact: ContactResult,
    roster,
    position,
    physics_substep,
    time_fraction,
    direction,
    launch,
    spin,
    release_velocity,
    possession_changed,
    attack_direction,
) -> AppliedContactEvent:
    player_count = roster.team_id.shape[0]
    valid_actor = (contact.actor >= 0) & (contact.actor < player_count)
    safe_actor = jnp.clip(contact.actor, 0, player_count - 1)
    valid = jnp.asarray(occurred, dtype=jnp.bool_) & valid_actor
    empty = empty_applied_contact_event(jnp.asarray(position).dtype)
    candidate = AppliedContactEvent(
        occurred=valid,
        actor=contact.actor.astype(jnp.int32),
        team=roster.team_id[safe_actor].astype(jnp.int32),
        intent=contact.intent.astype(jnp.int32),
        mechanism=contact.mechanism.astype(jnp.int32),
        outcome=contact.outcome.astype(jnp.int32),
        position=jnp.asarray(position),
        physics_substep=jnp.asarray(physics_substep, dtype=jnp.int32),
        time_fraction=jnp.asarray(time_fraction, dtype=jnp.asarray(position).dtype),
        direction=jnp.asarray(direction, dtype=jnp.asarray(position).dtype),
        launch=jnp.asarray(launch, dtype=jnp.asarray(position).dtype),
        spin=jnp.asarray(spin, dtype=jnp.asarray(position).dtype),
        release_velocity=jnp.asarray(
            release_velocity, dtype=jnp.asarray(position).dtype
        ),
        restart_kind=contact.restart_kind.astype(jnp.int32),
        possession_changed=jnp.asarray(possession_changed, dtype=jnp.bool_),
        kick_applied=contact.kick_applied.astype(jnp.bool_),
        law11_effect=contact.law11_effect.astype(jnp.int32),
        attack_direction=jnp.asarray(
            attack_direction, dtype=jnp.asarray(position).dtype
        ),
    )
    return jax.tree.map(
        lambda value, zero: jnp.where(valid, value, zero), candidate, empty
    )


class ControlFrameReceipt(NamedTuple):
    """실제 데이터 replay 검증에 필요한 scalar sidecar."""

    override_valid: jax.Array
    contact_occurred: jax.Array
    challenge_attempt_count: jax.Array
    parameter_consumed: jax.Array
    applied_contact: AppliedContactEvent
    event_chronology: FrameEventChronology
    boundary: BoundaryEvent
    restart_released: jax.Array
    restart_forced: jax.Array
    restart_retouch: RestartRetouchEvent
    woodwork: GoalFrameEvent
    foul: FoulEvent
    offside: OffsideEvent
    passive_contact: PassiveContactEvent
    nutmeg: NutmegEvent
    goalkeeper_holding: GoalkeeperHoldingEvent
    event_budget_exhausted: jax.Array


def _empty_control_frame_receipt(
    dtype=jnp.float32, player_count: int = 22
) -> ControlFrameReceipt:
    """실행되지 않은 control frame의 canonical event sidecar."""

    return ControlFrameReceipt(
        override_valid=jnp.bool_(True),
        contact_occurred=jnp.bool_(False),
        challenge_attempt_count=jnp.int32(0),
        parameter_consumed=jnp.zeros(player_count, dtype=jnp.uint16),
        applied_contact=empty_applied_contact_event(dtype),
        event_chronology=empty_frame_event_chronology(),
        boundary=empty_boundary_event(dtype),
        restart_released=jnp.bool_(False),
        restart_forced=jnp.bool_(False),
        restart_retouch=empty_restart_retouch_event(dtype),
        woodwork=empty_goal_frame_event(dtype),
        foul=empty_foul_event(dtype),
        offside=empty_offside_event(dtype),
        passive_contact=empty_passive_contact_event(dtype),
        nutmeg=empty_nutmeg_event(dtype),
        goalkeeper_holding=empty_goalkeeper_holding_event(dtype),
        event_budget_exhausted=jnp.bool_(False),
    )


def _parameter_bits(condition: jax.Array, bits: int) -> jax.Array:
    """조건별 fixed-width parameter-consumption bit를 만든다."""

    return jnp.asarray(condition, dtype=jnp.uint16) * jnp.uint16(bits)


class ControlFrameResult(NamedTuple):
    state: object
    receipt: ControlFrameReceipt


class _TrainingCarry(NamedTuple):
    state: object
    contact_attempted: jax.Array
    possession_changed: jax.Array
    body_impact_consumed: jax.Array


class _ControlCarry(NamedTuple):
    state: object
    contact_attempted: jax.Array
    override_valid: jax.Array
    contact_occurred: jax.Array
    challenge_attempt_count: jax.Array
    parameter_consumed: jax.Array
    applied_contact: AppliedContactEvent
    event_chronology: FrameEventChronology
    boundary: BoundaryEvent
    restart_released: jax.Array
    restart_forced: jax.Array
    restart_retouch: RestartRetouchEvent
    woodwork: GoalFrameEvent
    foul: FoulEvent
    offside: OffsideEvent
    passive_contact: PassiveContactEvent
    nutmeg: NutmegEvent
    goalkeeper_holding: GoalkeeperHoldingEvent
    event_budget_exhausted: jax.Array
    possession_changed: jax.Array
    body_impact_consumed: jax.Array


def sample_control_contest_overrides(
    timebase: Timebase = DEFAULT_TIMEBASE,
) -> ContestOverride:
    """한 control frame의 모든 contest를 sampling하는 fixed-shape 입력."""

    shape = (timebase.decimation,)
    return ContestOverride(
        winner=jnp.full(shape, SAMPLE_CONTEST, dtype=jnp.int32),
        outcome=jnp.full(shape, SAMPLE_CONTEST, dtype=jnp.int32),
        discipline=jnp.full(shape, SAMPLE_CONTEST, dtype=jnp.int32),
    )


def sample_control_body_foul_overrides(
    timebase: Timebase = DEFAULT_TIMEBASE,
) -> BodyFoulOverride:
    """한 control frame의 body foul을 sampling하는 fixed-shape 입력."""

    shape = (timebase.decimation,)
    return BodyFoulOverride(
        occurred=jnp.full(shape, SAMPLE_CONTEST, dtype=jnp.int32),
        discipline=jnp.full(shape, SAMPLE_CONTEST, dtype=jnp.int32),
    )


def _decrement_recovery(players):
    def decrement(value):
        return jnp.maximum(value - jnp.int32(1), jnp.int32(0))

    return players._replace(
        challenge_recovery_substeps=decrement(players.challenge_recovery_substeps),
        contact_lock_substeps=decrement(players.contact_lock_substeps),
        aerial_recovery_substeps=decrement(players.aerial_recovery_substeps),
        possession_loss_lock_substeps=decrement(players.possession_loss_lock_substeps),
    )


def _passive_contact_updates_provenance(
    release_occurred: jax.Array,
    passive_occurred: jax.Array,
    passive_actor: jax.Array,
    release_taker: jax.Array,
) -> jax.Array:
    """Actuation 잔여 overlap을 제외한 실제 passive touch mask를 반환한다."""

    same_release_taker_overlap = (
        release_occurred
        & passive_occurred
        & (passive_actor == release_taker)
    )
    return passive_occurred & (~same_release_taker_overlap)


def _validate_frame_inputs(
    state,
    roster,
    action: IntentAction,
    key: jax.Array,
    overrides: ContestOverride | None,
    body_foul_overrides: BodyFoulOverride | None,
    timebase: Timebase,
) -> int:
    validate_prng_key(key, name="key")
    if not isinstance(state, State):
        raise TypeError("state는 State여야 한다")
    if not isinstance(roster, RosterState):
        raise TypeError("roster는 RosterState여야 한다")
    if not isinstance(action, IntentAction):
        raise TypeError("action은 IntentAction이어야 한다")
    player_count = state.players.position.shape[0]
    validate_player_capacity(player_count)
    vector_shape = (player_count,)
    planar_shape = (player_count, 2)
    roster_planar = {"formation_anchor": roster.formation_anchor}
    roster_vector = {
        name: getattr(roster, name)
        for name in roster._fields
        if name != "formation_anchor"
    }
    player_planar = {
        "position": state.players.position,
        "velocity": state.players.velocity,
        "body_forward": state.players.body_forward,
    }
    player_vector = {
        name: getattr(state.players, name)
        for name in state.players._fields
        if name not in player_planar
    }
    action_planar = {
        "move": action.move,
        "force_to_ball": action.force_to_ball,
        "spin": action.spin,
    }
    action_vector = {"intent": action.intent, "launch": action.launch}
    shape_groups = (
        ("roster", roster_vector, vector_shape),
        ("roster", roster_planar, planar_shape),
        ("state.players", player_vector, vector_shape),
        ("state.players", player_planar, planar_shape),
        ("action", action_vector, vector_shape),
        ("action", action_planar, planar_shape),
    )
    for owner, leaves, expected in shape_groups:
        invalid = tuple(
            f"{name}={value.shape}"
            for name, value in leaves.items()
            if value.shape != expected
        )
        if invalid:
            details = ", ".join(invalid)
            raise ValueError(f"{owner} player축 shape는 {expected}여야 한다: {details}")
    if state.offside_flagged.shape != (player_count,):
        raise ValueError("offside_flagged player축은 state player축과 같아야 한다")
    fixed_shape_groups = (
        (
            "state.ball",
            {
                "position": state.ball.position,
                "velocity": state.ball.velocity,
                "spin": state.ball.spin,
            },
            (3,),
        ),
        (
            "state.team",
            {
                "attack_direction": state.attack_direction,
                "score": state.score,
            },
            (2,),
        ),
        (
            "state.scalar",
            {
                "control_tick": state.control_tick,
                "ball.live": state.ball.live,
                "kickoff_team": state.kickoff_team,
                "offside_direct_exempt_team": state.offside_direct_exempt_team,
                "gk_backpass_team": state.gk_backpass_team,
                "dead_ball_control_ticks": state.dead_ball_control_ticks,
                "first_half_wall_end_tick": state.first_half_wall_end_tick,
                "first_half_live_extension_ticks": state.first_half_live_extension_ticks,
                "penalty_completion_active": state.penalty_completion_active,
                "penalty_completion_team": state.penalty_completion_team,
                "restart_layout_ready": state.restart_layout_ready,
                "restart_taker_approach": state.restart_taker_approach,
                **{
                    f"possession.{name}": getattr(state.possession, name)
                    for name in state.possession._fields
                    if name != "last_contact"
                },
                **{
                    f"possession.last_contact.{name}": getattr(
                        state.possession.last_contact, name
                    )
                    for name in state.possession.last_contact._fields
                },
                **{
                    f"restart.{name}": getattr(state.restart, name)
                    for name in state.restart._fields
                },
                **{
                    f"restart_release.{name}": getattr(state.restart_release, name)
                    for name in state.restart_release._fields
                },
            },
            (),
        ),
    )
    for owner, leaves, expected in fixed_shape_groups:
        invalid = tuple(
            f"{name}={value.shape}"
            for name, value in leaves.items()
            if value.shape != expected
        )
        if invalid:
            raise ValueError(
                f"{owner} field shape는 {expected}여야 한다: {', '.join(invalid)}"
            )
    if overrides is not None:
        expected = (timebase.decimation,)
        shapes = (
            overrides.winner.shape,
            overrides.outcome.shape,
            overrides.discipline.shape,
        )
        if any(shape != expected for shape in shapes):
            raise ValueError(
                f"contest override leaf shape는 {expected}여야 한다: {shapes}"
            )
    if body_foul_overrides is not None:
        expected = (timebase.decimation,)
        shapes = (
            body_foul_overrides.occurred.shape,
            body_foul_overrides.discipline.shape,
        )
        if any(shape != expected for shape in shapes):
            raise ValueError(
                f"body foul override leaf shape는 {expected}여야 한다: {shapes}"
            )
    return player_count


def _validate_physics_geometry_compatibility(
    contact: ContactPhysics,
    contest: ContestPhysics,
    goal: GoalGeometry,
    restart: RestartPhysics,
    foul: FoulLaw,
) -> None:
    """한 transition 안에서 중복 표현된 float32 physical geometry를 맞춘다."""

    comparisons = (
        (
            "ContactPhysics.carry_radius_m과 ContestPhysics.carry_radius_m",
            contact.carry_radius_m,
            contest.carry_radius_m,
        ),
        (
            "ContactPhysics.challenge_radius_m과 ContestPhysics.challenge_radius_m",
            contact.challenge_radius_m,
            contest.challenge_radius_m,
        ),
        (
            "GoalGeometry.width_m / 2와 RestartPhysics.goal_half_width_m",
            np.float32(0.5) * np.float32(goal.width_m),
            restart.goal_half_width_m,
        ),
        (
            "RestartPhysics.penalty_area_length_m과 FoulLaw.penalty_area_length_m",
            restart.penalty_area_length_m,
            foul.penalty_area_length_m,
        ),
        (
            "RestartPhysics.penalty_area_half_width_m과 FoulLaw.penalty_area_half_width_m",
            restart.penalty_area_half_width_m,
            foul.penalty_area_half_width_m,
        ),
    )
    for label, left, right in comparisons:
        if np.float32(left) != np.float32(right):
            raise ValueError(f"{label}은 float32에서 같아야 한다")


def _validate_static_restart_pitch_compatibility(
    field_half_extent,
    restart: RestartPhysics,
    body: BodyCollision,
    foul: FoulLaw,
) -> None:
    """Host-static pitch에 restart 합법 배치가 존재하는지 검증한다.

    Low-level transition은 runtime traced extent도 허용한다. JAX leaf가 하나라도 있으면
    host 값을 읽지 않고 기존 shape-only contract에 맡긴다.
    """

    leaves = jax.tree.leaves(field_half_extent)
    if any(isinstance(leaf, (jax.Array, jax.core.Tracer)) for leaf in leaves):
        return
    half_x_value, half_y_value = require_float32_field_half_extent(
        field_half_extent
    )
    half_x = np.float32(half_x_value)
    half_y = np.float32(half_y_value)
    radius = np.float32(0.5) * np.float32(body.effective_diameter_m)
    epsilon = np.float32(1.0e-6)
    inset_x = half_x - radius
    inset_y = half_y - radius

    if inset_x < -epsilon or inset_y < -epsilon:
        raise ValueError(
            "field_half_extent는 BodyCollision player centre inset을 포함해야 한다"
        )
    if half_y + epsilon < np.float32(restart.goal_half_width_m):
        raise ValueError(
            "field_half_extent[1]는 RestartPhysics.goal_half_width_m 이상이어야 한다"
        )
    if inset_x + epsilon < radius:
        raise ValueError(
            "field_half_extent[0]에는 kickoff own-half player centre 구간이 있어야 한다"
        )

    clearance = (
        np.float32(restart.set_piece_opponent_distance_m)
        + radius
        + np.float32(8.0e-6)
    )
    maximum_centre_distance = np.sqrt(
        inset_x * inset_x + inset_y * inset_y, dtype=np.float32
    )
    if maximum_centre_distance + epsilon < clearance:
        raise ValueError(
            "field_half_extent 안에서 RestartPhysics."
            "set_piece_opponent_distance_m clearance를 만들 수 없다"
        )

    mark_progress = half_x - np.float32(foul.penalty_mark_distance_m)
    behind_mark_limit = mark_progress - radius
    area_progress_limit = (
        half_x - np.float32(restart.penalty_area_length_m) - radius
    )
    lateral_limit = np.float32(restart.penalty_area_half_width_m) + radius

    def box_has_penalty_clearance(
        lower_x: np.float32,
        upper_x: np.float32,
        lower_y: np.float32,
        upper_y: np.float32,
    ) -> bool:
        if lower_x > upper_x or lower_y > upper_y:
            return False
        maximum_x = np.maximum(
            np.abs(lower_x - mark_progress),
            np.abs(upper_x - mark_progress),
        )
        maximum_y = np.maximum(np.abs(lower_y), np.abs(upper_y))
        maximum_distance = np.sqrt(
            maximum_x * maximum_x + maximum_y * maximum_y,
            dtype=np.float32,
        )
        return bool(maximum_distance + epsilon >= clearance)

    longitudinal = box_has_penalty_clearance(
        -inset_x,
        np.minimum(inset_x, np.minimum(behind_mark_limit, area_progress_limit)),
        -inset_y,
        inset_y,
    )
    lateral_progress_limit = np.minimum(inset_x, behind_mark_limit)
    upper_lateral = box_has_penalty_clearance(
        -inset_x,
        lateral_progress_limit,
        lateral_limit,
        inset_y,
    )
    lower_lateral = box_has_penalty_clearance(
        -inset_x,
        lateral_progress_limit,
        -inset_y,
        -lateral_limit,
    )
    if not (longitudinal or upper_lateral or lower_lateral):
        raise ValueError(
            "field_half_extent 안에서 penalty_area 밖·mark 뒤·clearance를 "
            "동시에 만족하는 penalty restart 위치를 만들 수 없다"
        )


def _step_control_impl(
    state,
    roster,
    action: IntentAction,
    key: jax.Array,
    contest_overrides: ContestOverride | None,
    body_foul_overrides: BodyFoulOverride | None,
    *,
    collect_events: bool,
    timebase: Timebase = DEFAULT_TIMEBASE,
    field_half_extent: jax.Array = (52.5, 34.0),
    movement_enabled: jax.Array = True,
    goalkeeper_hand_allowed: jax.Array | None = None,
    regulation_elapsed_fraction: jax.Array = 0.0,
    ball_geometry: BallGeometry = BallGeometry(),
    ball_physics: BallPhysics = BallPhysics(),
    goal_geometry: GoalGeometry = GoalGeometry(),
    restart_physics: RestartPhysics = RestartPhysics(),
    contact_physics: ContactPhysics = ContactPhysics(),
    contest_physics: ContestPhysics = ContestPhysics(),
    response_physics: ContactResponsePhysics = ContactResponsePhysics(),
    player_physics: PlayerPhysics = PlayerPhysics(),
    body_collision: BodyCollision = BodyCollision(),
    collision_physics: CollisionPhysics = CollisionPhysics(),
    body_foul_physics: BodyFoulPhysics = BodyFoulPhysics(),
    long_stamina: LongStamina = LongStamina(),
    short_stamina: ShortStamina = ShortStamina(),
    goal_frame_physics: GoalFramePhysics = GoalFramePhysics(),
    foul_law: FoulLaw = DEFAULT_FOUL_LAW,
    offside_geometry: OffsideGeometry = OffsideGeometry(),
    passive_contact_physics: PassiveContactPhysics = PassiveContactPhysics(),
    goalkeeper_holding_physics: GoalkeeperHoldingPhysics = GoalkeeperHoldingPhysics(),
    prepare_restart_layout: bool = True,
) -> ControlFrameResult:
    """Static mode에 따라 lean training 또는 event/replay frame을 전개한다."""

    player_count = _validate_frame_inputs(
        state,
        roster,
        action,
        key,
        contest_overrides,
        body_foul_overrides,
        timebase,
    )
    _validate_physics_geometry_compatibility(
        contact_physics,
        contest_physics,
        goal_geometry,
        restart_physics,
        foul_law,
    )
    movement_enabled = jnp.broadcast_to(
        jnp.asarray(movement_enabled, dtype=jnp.bool_), (player_count,)
    )
    field_half_extent = jnp.asarray(field_half_extent, dtype=jnp.float32)
    derive_goalkeeper_hand_allowed = goalkeeper_hand_allowed is None
    if not derive_goalkeeper_hand_allowed:
        goalkeeper_hand_allowed = jnp.broadcast_to(
            jnp.asarray(goalkeeper_hand_allowed, dtype=jnp.bool_), (player_count,)
        )
    if field_half_extent.shape != (2,):
        raise ValueError("field_half_extent shape는 (2,)여야 한다")
    possession_loss_lock_ticks = duration_to_ticks(
        response_physics.possession_loss_lock_s,
        timebase.dt_phys,
        rounding="nearest",
        minimum=1 if response_physics.possession_loss_lock_s > 0.0 else 0,
        maximum=MAX_INT32_TICKS,
    )

    taker_repair = repair_restart_taker(state, roster)
    state = taker_repair.state

    entry_restart_kind = state.restart.kind
    entry_restart_team = state.restart.team
    entry_restart_taker = state.restart.taker
    entry_restart_ready = (
        restart_active(entry_restart_kind) & state.restart_layout_ready
    )
    if not isinstance(prepare_restart_layout, bool):
        raise TypeError("prepare_restart_layout은 bool이어야 한다")
    if prepare_restart_layout:
        prepared = prepare_restart(
            state,
            roster,
            field_half_extent,
            timebase=timebase,
            ball=ball_geometry,
            body=body_collision,
            contact=contact_physics,
            config=restart_physics,
            defer_non_taker_positioning=True,
        )
    else:
        prepared = RestartPreparation(state, jnp.bool_(False))
    physics_action = decode_physics_action(
        action,
        roster,
        prepared.state.attack_direction,
        prepared.state.players.body_forward,
    )
    taker = prepared.state.restart.taker
    valid_taker_slot = (taker >= 0) & (taker < player_count)
    safe_taker = jnp.clip(taker, 0, player_count - 1)
    taker_valid = (
        restart_active(prepared.state.restart.kind)
        & valid_taker_slot
        & prepared.state.players.active[safe_taker]
        & (roster.team_id[safe_taker] == prepared.state.restart.team)
    )
    approach_actor = taker_valid & (
        jnp.arange(player_count, dtype=jnp.int32) == safe_taker
    )
    approach_target, approach_facing = restart_taker_release_pose(
        prepared.state, roster, ball=ball_geometry, body=body_collision
    )
    approach_delta = approach_target - prepared.state.players.position[safe_taker]
    approach_distance_squared = jnp.sum(approach_delta * approach_delta)
    approach_requested = (
        entry_restart_ready & prepared.state.restart_taker_approach & taker_valid
    )
    approach_timeout_ticks = timebase.control_steps_for(
        restart_physics.approach_timeout_s,
        minimum=1,
        maximum=MAX_INT32_TICKS,
    )
    approach_age = jnp.maximum(
        prepared.state.control_tick - prepared.state.restart.opened_control_tick,
        jnp.int32(0),
    )
    approach_timed_out = approach_requested & (
        approach_age >= jnp.int32(approach_timeout_ticks)
    )
    continuous_approach = continuous_restart_approach_enabled(
        prepared.state.restart.kind, config=restart_physics
    )
    approach_enabled = approach_requested & continuous_approach & (~approach_timed_out)
    approaching = approach_enabled & (
        approach_distance_squared > restart_physics.approach_arrival_tolerance_m**2
    )
    approach_distance = jnp.sqrt(approach_distance_squared)
    approach_direction = approach_delta / jnp.maximum(approach_distance, 1.0e-9)
    # One-frame arrival (distance * control_fps) keeps requesting near-maximum
    # speed until the taker is very close.  The acceleration-limited locomotion
    # then cannot shed momentum before crossing the release pose and oscillates
    # around it.  A braking-derived first-order arrival rate starts reducing the
    # requested speed while the remaining distance can still absorb it.  This
    # changes only the transition-owned restart approach override; ordinary
    # player movement continues to use the submitted policy velocity unchanged.
    approach_max_speed = roster.max_speed[safe_taker]
    approach_arrival_rate = (
        jnp.float32(2.0)
        * jnp.asarray(player_physics.braking_deceleration_mps2, jnp.float32)
        / jnp.maximum(approach_max_speed, jnp.float32(1.0e-6))
    )
    braking_speed = jnp.sqrt(
        jnp.float32(2.0)
        * jnp.asarray(player_physics.braking_deceleration_mps2, jnp.float32)
        * approach_distance
    )
    approach_speed = jnp.minimum(
        approach_max_speed,
        jnp.minimum(
            approach_distance * approach_arrival_rate,
            braking_speed,
        ),
    )
    approach_velocity = approach_direction * approach_speed
    snap_requested = approach_requested & ((~continuous_approach) | (~approaching))
    snap_actor = snap_requested & approach_actor
    desired_velocity = jnp.where(
        (approaching & approach_actor)[:, None],
        approach_velocity[None, :],
        physics_action.desired_velocity,
    )
    approach_state = prepared.state._replace(
        players=prepared.state.players._replace(
            position=jnp.where(
                snap_actor[:, None],
                approach_target,
                prepared.state.players.position,
            ),
            velocity=jnp.where(
                snap_actor[:, None],
                jnp.zeros_like(prepared.state.players.velocity),
                prepared.state.players.velocity,
            ),
            body_forward=jnp.where(
                snap_actor[:, None],
                approach_facing,
                prepared.state.players.body_forward,
            ),
        ),
        restart_taker_approach=jnp.where(
            entry_restart_ready,
            prepared.state.restart_taker_approach & approaching,
            prepared.state.restart_taker_approach,
        ),
    )
    approach_margin = jnp.where(
        approaching,
        0.5 * body_collision.effective_diameter_m + ball_geometry.radius_m + 1.0e-6,
        0.0,
    )
    deadline_within_frame = (
        approach_state.restart.substeps_remaining < timebase.decimation
    )
    non_takers_positioned = jax.lax.cond(
        deadline_within_frame & (~approaching),
        lambda _: ordinary_restart_non_takers_positioned(
            approach_state,
            roster,
            field_half_extent,
            ball=ball_geometry,
            body=body_collision,
            contact=contact_physics,
            config=restart_physics,
        ),
        lambda _: jnp.bool_(True),
        operand=None,
    )
    release_due = deadline_within_frame & (~approaching) & non_takers_positioned
    release_plan = release_restart(
        approach_state,
        roster,
        physics_action,
        entry_restart_ready & release_due,
        deadline_due=release_due,
        ball=ball_geometry,
        contact=contact_physics,
        config=restart_physics,
    )
    release_substep = jnp.where(
        release_plan.occurred,
        jnp.clip(
            approach_state.restart.substeps_remaining,
            0,
            timebase.decimation - 1,
        ),
        0,
    ).astype(jnp.uint32)
    frame_key = frame_random_key(key, state.control_tick)
    goalkeeper_holding_limit = holding_limit_control_ticks(
        timebase=timebase, config=goalkeeper_holding_physics
    )
    held_release = release_held_ball(
        approach_state,
        roster,
        physics_action,
        ball=ball_geometry,
        contact=contact_physics,
    )
    with jax.named_scope("messi/goalkeeper_holding"):
        holding = resolve_goalkeeper_holding_expiry(
            held_release.state,
            roster,
            field_half_extent,
            limit_control_ticks=goalkeeper_holding_limit,
            ball_radius_m=ball_geometry.radius_m,
        )
    held_actor = jnp.arange(player_count, dtype=jnp.int32) == jnp.clip(
        approach_state.possession.player, 0, player_count - 1
    )
    initial_contact_attempted = held_release.released & held_actor
    if collect_events:
        held_velocity = held_release.state.ball.velocity
        held_horizontal_speed = jnp.linalg.norm(held_velocity[:2])
        held_direction = held_velocity[:2] / jnp.maximum(
            held_horizontal_speed, jnp.asarray(1.0e-9, held_velocity.dtype)
        )
        held_contact = held_release.state.possession.last_contact
        held_safe_actor = jnp.clip(held_contact.actor, 0, player_count - 1)
        held_event = _applied_contact_event(
            held_release.released,
            held_contact._replace(restart_kind=jnp.int32(RK_GK_HOLD)),
            roster,
            held_release.state.ball.position,
            jnp.int32(0),
            jnp.asarray(0.0, dtype=state.ball.position.dtype),
            held_direction,
            jnp.arctan2(held_velocity[2], held_horizontal_speed),
            physics_action.spin[held_safe_actor],
            held_velocity,
            held_release.released,
            approach_state.attack_direction[roster.team_id[held_safe_actor]],
        )
        initial = _ControlCarry(
            state=holding.state,
            contact_attempted=initial_contact_attempted,
            # Replay가 body-foul stream을 생략하면 stochastic body foul이 남는다.
            # 이를 strict replay 성공으로 보고하지 않도록 frame validity를 닫는다.
            override_valid=jnp.bool_(body_foul_overrides is not None),
            contact_occurred=jnp.bool_(False),
            challenge_attempt_count=jnp.int32(0),
            parameter_consumed=_parameter_bits(
                held_release.released & held_actor,
                PARAMETER_FORCE_DIRECTION
                | PARAMETER_FORCE_POWER
                | PARAMETER_LAUNCH
                | PARAMETER_SPIN,
            ),
            applied_contact=held_event,
            event_chronology=empty_frame_event_chronology(),
            possession_changed=jnp.bool_(False),
            body_impact_consumed=jnp.bool_(False),
            boundary=empty_boundary_event(state.ball.position.dtype),
            restart_released=jnp.bool_(False),
            restart_forced=jnp.bool_(False),
            restart_retouch=empty_restart_retouch_event(state.ball.position.dtype),
            woodwork=empty_goal_frame_event(state.ball.position.dtype),
            foul=empty_foul_event(state.ball.position.dtype),
            offside=empty_offside_event(state.ball.position.dtype),
            passive_contact=empty_passive_contact_event(state.ball.position.dtype),
            nutmeg=empty_nutmeg_event(state.ball.position.dtype),
            goalkeeper_holding=holding.event,
            event_budget_exhausted=jnp.bool_(False),
        )
    else:
        initial = _TrainingCarry(
            state=holding.state,
            contact_attempted=initial_contact_attempted,
            possession_changed=jnp.bool_(False),
            body_impact_consumed=jnp.bool_(False),
        )
    substep_indices = jnp.arange(timebase.decimation, dtype=jnp.uint32)
    substep_keys = jax.vmap(lambda index: jax.random.fold_in(frame_key, index))(
        substep_indices
    )
    body_foul_draws = jax.vmap(sample_body_foul_draws)(substep_keys)

    def substep(carry, inputs):
        if contest_overrides is None:
            substep_index, contest_key, body_foul_draw = inputs
            body_foul_override = None
        elif body_foul_overrides is None:
            substep_index, contest_key, body_foul_draw, override = inputs
            body_foul_override = None
        else:
            (
                substep_index,
                contest_key,
                body_foul_draw,
                override,
                body_foul_override,
            ) = inputs
        before = carry.state
        same_entry_restart = (
            (before.restart.kind == entry_restart_kind)
            & (before.restart.team == entry_restart_team)
            & (before.restart.taker == entry_restart_taker)
        )
        restart_visible = entry_restart_ready & same_entry_restart
        release_now = (
            restart_visible & release_plan.occurred & (substep_index == release_substep)
        )

        def select(new, old):
            return jnp.where(release_now, new, old)

        pre_physics = before._replace(
            ball=jax.tree.map(select, release_plan.state.ball, before.ball),
            possession=jax.tree.map(
                select, release_plan.state.possession, before.possession
            ),
            restart=jax.tree.map(select, release_plan.state.restart, before.restart),
            restart_release=jax.tree.map(
                select, release_plan.state.restart_release, before.restart_release
            ),
            restart_layout_ready=select(
                release_plan.state.restart_layout_ready,
                before.restart_layout_ready,
            ),
            restart_taker_approach=select(
                release_plan.state.restart_taker_approach,
                before.restart_taker_approach,
            ),
            penalty_completion_active=select(
                release_plan.state.penalty_completion_active,
                before.penalty_completion_active,
            ),
            penalty_completion_team=select(
                release_plan.state.penalty_completion_team,
                before.penalty_completion_team,
            ),
        )
        players = _decrement_recovery(pre_physics.players)
        contact_state = pre_physics._replace(players=players)
        release_actor = jnp.arange(player_count, dtype=jnp.int32) == release_plan.actor
        attempted_before = carry.contact_attempted | (release_now & release_actor)
        substep_goalkeeper_hand_allowed = (
            goalkeeper_hand_allowed_mask(
                contact_state,
                roster,
                field_half_extent,
                penalty_area_length_m=restart_physics.penalty_area_length_m,
                penalty_area_half_width_m=restart_physics.penalty_area_half_width_m,
            )
            if derive_goalkeeper_hand_allowed
            else goalkeeper_hand_allowed
        )
        substep_goalkeeper_hand_restricted = (
            goalkeeper_hand_restricted_mask(contact_state, roster)
            if derive_goalkeeper_hand_allowed
            else jnp.zeros(player_count, dtype=jnp.bool_)
        )
        with jax.named_scope("messi/contact_geometry"):
            geometry = evaluate_contact_geometry(
                contact_state,
                roster,
                physics_action,
                attempted_before,
                goalkeeper_hand_allowed=substep_goalkeeper_hand_allowed,
                goalkeeper_hand_restricted=substep_goalkeeper_hand_restricted,
                ball_geometry=ball_geometry,
                config=contact_physics,
            )
        hand_offence_occurred = jnp.any(geometry.goalkeeper_hand_restricted_attempt)
        hand_offender = jnp.argmax(
            geometry.goalkeeper_hand_restricted_attempt.astype(jnp.int32)
        ).astype(jnp.int32)
        hand_offence_event = detect_indirect_offence(
            contact_state,
            roster,
            hand_offence_occurred,
            hand_offender,
            contact_state.ball.position,
            source_restart_kind=contact_state.restart_release.kind,
            time_fraction=jnp.asarray(0.0, dtype=contact_state.ball.position.dtype),
        )
        geometry = geometry._replace(
            possible=geometry.possible & (~hand_offence_occurred)
        )
        attempted = (
            attempted_before
            | geometry.possible
            | geometry.goalkeeper_hand_restricted_attempt
        )
        if contest_overrides is None:
            contest = resolve_contest_sample(
                contact_state,
                roster,
                geometry,
                contest_key,
                regulation_elapsed_fraction=regulation_elapsed_fraction,
                field_half_extent=field_half_extent,
                config=contest_physics,
            )
        else:
            contest = resolve_contest(
                contact_state,
                roster,
                geometry,
                contest_key,
                override,
                regulation_elapsed_fraction=regulation_elapsed_fraction,
                field_half_extent=field_half_extent,
                config=contest_physics,
            )
        safe_contest_actor = jnp.clip(contest.actor, 0, player_count - 1)
        challenge_selected = (
            contest.selected & geometry.challenge_context[safe_contest_actor]
        )
        with jax.named_scope("messi/contact_response"):
            resolved = resolve_contact_response(
                contact_state,
                roster,
                physics_action,
                geometry,
                contest,
                dt=timebase.dt_phys,
                contact=contact_physics,
                response=response_physics,
                field_half_extent=field_half_extent,
                ball_geometry=ball_geometry,
                ball_physics=ball_physics,
                goal_geometry=goal_geometry,
                goal_frame_physics=goal_frame_physics,
            )
        resolved = resolved._replace(
            state=update_backpass_latch(
                resolved.state,
                roster,
                resolved.contact,
                ball=ball_geometry,
                physics=ball_physics,
                contact_physics=contact_physics,
            )
        )
        retouch_event = detect_restart_retouch(
            contact_state,
            resolved.contact,
            contact_state.ball.position,
            time_fraction=jnp.asarray(0.0, dtype=contact_state.ball.position.dtype),
        )
        retouch_event = jax.tree.map(
            lambda hand, ordinary: jnp.where(hand_offence_occurred, hand, ordinary),
            hand_offence_event,
            retouch_event,
        )
        with jax.named_scope("messi/indirect_restart"):
            retouch = apply_indirect_restart(
                resolved.state,
                roster,
                retouch_event,
                field_half_extent,
                goal=goal_geometry,
                ball=ball_geometry,
            )
        with jax.named_scope("messi/offside"):
            offside = resolve_offside(
                retouch.state,
                roster,
                resolved.contact,
                contest,
                field_half_extent,
                time_fraction=jnp.asarray(0.0, dtype=contact_state.ball.position.dtype),
                ball_radius_m=ball_geometry.radius_m,
                geometry=offside_geometry,
            )
        contest_offender = jnp.clip(contest.foul_actor, 0, player_count - 1)
        contest_victim = jnp.clip(contest.foul_victim, 0, player_count - 1)
        contest_indices_valid = (
            (contest.foul_actor >= 0)
            & (contest.foul_actor < player_count)
            & (contest.foul_victim >= 0)
            & (contest.foul_victim < player_count)
        )
        contest_foul_pending = (
            contest.selected
            & (contest.outcome == OUTCOME_FOUL)
            & contest_indices_valid
            & offside.state.players.active[contest_victim]
            & (roster.team_id[contest_offender] != roster.team_id[contest_victim])
        )
        rule_state = offside.state
        pre_movement_stoppage = contest_foul_pending | offside.event.occurred
        dead_restart = restart_active(rule_state.restart.kind) | contest_foul_pending
        moving_approach_actor = approaching & approach_actor
        positioning_other = (
            dead_restart
            & (rule_state.restart.substeps_remaining > 0)
            & (~approach_actor)
        )
        pinned = jnp.broadcast_to(dead_restart, (player_count,)) & (
            ~(moving_approach_actor | positioning_other)
        )
        with jax.named_scope("messi/player_step"):
            player_step = step_players(
                rule_state.players,
                roster,
                desired_velocity,
                (movement_enabled | moving_approach_actor) & (~pinned),
                rule_state.attack_direction,
                field_half_extent,
                dt=timebase.dt_phys,
                boundary_margin_m=approach_margin,
                pinned=pinned,
                player_physics=player_physics,
                body=body_collision,
                collision_physics=collision_physics,
                long_stamina=long_stamina,
                short_stamina=short_stamina,
            )
        # A collision or residual approach momentum can still carry the taker
        # across the exact pose within one physics substep.  Clamp that crossing
        # to the target immediately; otherwise the next 10 Hz decision observes
        # the taker on the opposite side and requests a visible return leg.
        approach_before = approach_target - rule_state.players.position[safe_taker]
        approach_after = approach_target - player_step.players.position[safe_taker]
        approach_crossed = approaching & (
            jnp.dot(approach_before, approach_after) <= jnp.float32(0.0)
        )
        clamp_approach_actor = approach_actor & approach_crossed
        player_step = player_step._replace(
            players=player_step.players._replace(
                position=jnp.where(
                    clamp_approach_actor[:, None],
                    approach_target,
                    player_step.players.position,
                ),
                velocity=jnp.where(
                    clamp_approach_actor[:, None],
                    jnp.zeros_like(player_step.players.velocity),
                    player_step.players.velocity,
                ),
                body_forward=jnp.where(
                    clamp_approach_actor[:, None],
                    approach_facing,
                    player_step.players.body_forward,
                ),
            )
        )
        # 데드볼 동안 행동은 자유롭게 고르되 두 stamina store는 얼린다. 재개 준비의
        # 걷기가 경기 체력을 깎지 않게 하려는 것이며, 같은 이유로 회복도 멈춘다.
        stamina_frozen = jnp.broadcast_to(
            dead_restart & jnp.bool_(restart_physics.freeze_dead_ball_stamina),
            (player_count,),
        )
        players = player_step.players._replace(
            stamina_long=jnp.where(
                stamina_frozen,
                rule_state.players.stamina_long,
                player_step.players.stamina_long,
            ),
            stamina_short=jnp.where(
                stamina_frozen,
                rule_state.players.stamina_short,
                player_step.players.stamina_short,
            ),
            position=jnp.where(
                pinned[:, None],
                rule_state.players.position,
                player_step.players.position,
            ),
            velocity=jnp.where(
                pinned[:, None],
                rule_state.players.velocity,
                player_step.players.velocity,
            ),
            body_forward=jnp.where(
                pinned[:, None],
                rule_state.players.body_forward,
                player_step.players.body_forward,
            ),
        )
        players, restart_non_takers_positioned = (
            _enforce_restart_countdown_positions_with_status(
                rule_state,
                roster,
                players,
                field_half_extent,
                positioning_dt_s=timebase.dt_phys,
                ball=ball_geometry,
                body=body_collision,
                contact=contact_physics,
                config=restart_physics,
            )
        )
        player_start_position = rule_state.players.position
        player_path_delta = players.position - player_start_position
        enabled_impact = player_step.impact._replace(
            occurred=(
                player_step.impact.occurred
                & (~carry.body_impact_consumed)
                & (~pre_movement_stoppage)
            )
        )
        with jax.named_scope("messi/body_foul"):
            body_foul = resolve_body_impact_foul(
                rule_state._replace(players=players),
                roster,
                enabled_impact,
                physics_action.requested_intent,
                contest_key,
                field_half_extent,
                body_foul_override,
                body_foul_draw,
                regulation_elapsed_fraction=regulation_elapsed_fraction,
                ball_radius_m=ball_geometry.radius_m,
                config=body_foul_physics,
                discipline_config=contest_physics,
                law=foul_law,
                apply_resolution=False,
            )
        use_body_foul = (~contest_foul_pending) & body_foul.contest.selected
        chosen_foul = jax.tree.map(
            lambda body, primary: jnp.where(use_body_foul, body, primary),
            body_foul.contest,
            contest,
        )
        foul = resolve_contest_foul(
            rule_state._replace(players=players),
            roster,
            chosen_foul,
            field_half_extent,
            time_fraction=jnp.where(
                use_body_foul,
                enabled_impact.time_fraction,
                jnp.asarray(0.0, dtype=state.ball.position.dtype),
            ),
            ball_radius_m=ball_geometry.radius_m,
            law=foul_law,
        )
        body_actor = jnp.clip(body_foul.contest.foul_actor, 0, player_count - 1)
        body_yellow = (
            use_body_foul
            & foul.event.occurred
            & (body_foul.contest.discipline == DISCIPLINE_YELLOW)
        )
        body_red = (
            use_body_foul
            & foul.event.occurred
            & (body_foul.contest.discipline == DISCIPLINE_RED)
        )
        body_second_yellow = body_yellow & (players.yellow_cards[body_actor] >= 1)
        disciplined_players = foul.state.players._replace(
            yellow_cards=foul.state.players.yellow_cards.at[body_actor].add(
                body_yellow.astype(jnp.int32)
            ),
            sent_off=foul.state.players.sent_off.at[body_actor].set(
                foul.state.players.sent_off[body_actor] | body_red | body_second_yellow
            ),
        )
        foul = foul._replace(state=foul.state._replace(players=disciplined_players))
        foul_event = foul.event
        rule_stoppage = contest_foul_pending | offside.event.occurred
        raw_offside_event = jax.tree.map(
            lambda value, empty: jnp.where(contest_foul_pending, empty, value),
            offside.event,
            empty_offside_event(state.ball.position.dtype),
        )
        raw_retouch_event = jax.tree.map(
            lambda value, empty: jnp.where(rule_stoppage, empty, value),
            retouch.event,
            empty_restart_retouch_event(state.ball.position.dtype),
        )
        first_body_foul = use_body_foul & foul.event.occurred
        if collect_events:
            foul_offender = jnp.clip(foul_event.offender, 0, player_count - 1)
            foul_victim = jnp.clip(foul_event.victim, 0, player_count - 1)
            approach = players.position[foul_victim] - players.position[foul_offender]
            relative_velocity = (
                players.velocity[foul_offender] - players.velocity[foul_victim]
            )
            closing_speed = jnp.maximum(
                jnp.dot(
                    relative_velocity,
                    approach / jnp.maximum(jnp.linalg.norm(approach), 1.0e-9),
                ),
                0.0,
            )
            offender_speed = jnp.linalg.norm(players.velocity[foul_offender])
            victim_speed = jnp.linalg.norm(players.velocity[foul_victim])
            alignment = jnp.clip(
                jnp.dot(
                    players.velocity[foul_offender],
                    players.velocity[foul_victim],
                )
                / jnp.maximum(offender_speed * victim_speed, 1.0e-9),
                0.0,
                1.0,
            )
            classification = classify_existing_contact_discipline(
                rule_state._replace(players=players),
                roster,
                valid=foul_event.occurred,
                offender=foul_event.offender,
                victim=foul_event.victim,
                contact_position=foul_event.position,
                administrative_discipline=foul_event.discipline,
                attempt_to_play_ball=(
                    physics_action.requested_intent[foul_offender] == INTENT_CHALLENGE
                ),
                closing_speed=jnp.where(
                    first_body_foul, enabled_impact.impact_score, closing_speed
                ),
                approach_direction=approach,
                velocity_alignment=jnp.where(
                    first_body_foul, enabled_impact.velocity_alignment, alignment
                ),
                field_half_extent=field_half_extent,
                law=foul_law,
            )
            foul_event = foul_event._replace(
                severity=classification.severity,
                tactical_effect=classification.tactical_effect,
            )
        # Contest foul은 deliberate-contact 시각(0.0)에 이미 확정되므로 ball
        # timeline보다 앞선다. 위 ``foul`` resolution은 contest가 pending일 때
        # 동일한 입력·시각의 state를 이미 만들었다. 이를 재사용해 restart solver를
        # 두 번 실행하지 않는다. Body foul은 locomotion 뒤의 물리 시각을 가지므로
        # passive rule offence와 boundary를 계산한 뒤 하나만 선택한다.
        moving_rule_state = rule_state._replace(players=players)
        rule_state = jax.tree.map(
            lambda stopped, moving: jnp.where(contest_foul_pending, stopped, moving),
            foul.state,
            moving_rule_state,
        )
        held_state = sync_held_ball(
            rule_state,
            roster,
            ball=ball_geometry,
            contact=contact_physics,
        )
        last_actor = resolved.state.possession.last_contact.actor
        valid_last_actor = (last_actor >= 0) & (last_actor < player_count)
        safe_last_actor = jnp.clip(last_actor, 0, player_count - 1)
        locked_last_actor = valid_last_actor & (
            resolved.state.players.contact_lock_substeps[safe_last_actor] > 0
        )
        passive_excluded_actor = jnp.where(
            locked_last_actor, safe_last_actor, resolved.contact.actor
        ).astype(jnp.int32)
        player_indices = jnp.arange(player_count, dtype=jnp.int32)
        release = held_state.restart_release
        retouch_actor = (
            release.active
            & release.untouched
            # Release와 같은 control frame의 swept passive overlap은 actuation
            # 자체의 잔여 geometry다. 정상적인 두 번째 touch 판정은 다음 frame부터
            # 시작한다. 일반 passive response는 그대로 적용되어 공도 분리된다.
            & (~release_plan.occurred)
            & (player_indices == release.taker)
        )
        passive_offside_actor = held_state.offside_flagged & (
            roster.team_id != held_state.offside_direct_exempt_team
        )
        rule_actor_mask = retouch_actor | passive_offside_actor
        with jax.named_scope("messi/ball_events"):
            ball_events = step_ball_events_chronological(
                held_state.ball,
                held_state.players,
                roster,
                dt=timebase.dt_phys,
                field_half_extent=field_half_extent,
                excluded_actor=passive_excluded_actor,
                # Held-ball PASS/SHOOT actuation은 goalkeeper 몸 안쪽의 anchor에서
                # 시작한다. 같은 physics frame의 swept overlap은 두 번째 touch가
                # 아니라 release 자체의 잔여 geometry이므로 release actor만 passive
                # timeline에서 제외한다. 다른 선수의 contact는 그대로 유효하다.
                excluded_actor_mask=held_release.released & held_actor,
                rule_actor_mask=rule_actor_mask,
                player_start_position=player_start_position,
                player_path_delta=player_path_delta,
                ball_geometry=ball_geometry,
                ball_physics=ball_physics,
                goal_geometry=goal_geometry,
                goal_frame_physics=goal_frame_physics,
                passive_contact_physics=passive_contact_physics,
                collect_events=collect_events,
            )
        passive_contact = ball_events.passive
        rule_passive_contact = ball_events.rule_passive
        woodwork = ball_events.woodwork
        # Release와 같은 control frame에서 taker 몸에 남은 swept overlap은 actuation
        # 자체의 잔여 geometry다. Physics response와 receipt는 보존하되 이를 별도 player
        # touch로 provenance에 쓰면 ``untouched``가 즉시 닫혀 다음 frame의 실제 second
        # kick이 합법화된다. 다른 선수의 same-frame contact는 여전히 release restriction을
        # 정상적으로 끝낸다.
        passive_provenance_occurred = _passive_contact_updates_provenance(
            release_plan.occurred,
            ball_events.passive_occurred,
            ball_events.last_passive_actor,
            held_state.restart_release.taker,
        )
        advanced = apply_passive_contact_provenance(
            held_state._replace(ball=ball_events.ball),
            roster,
            ball_events.last_passive_actor,
            passive_provenance_occurred,
            possession_loss_lock_substeps=possession_loss_lock_ticks,
        )
        advanced = clear_backpass_after_passive_contact(
            advanced,
            roster,
            ball_events.last_passive_actor,
            ball_events.passive_occurred,
        )
        passive_contact_result = held_state.possession.last_contact._replace(
            actor=rule_passive_contact.actor,
            mechanism=jnp.int32(MECHANISM_PASSIVE_BODY),
            intent=jnp.int32(INTENT_MOVE),
            outcome=jnp.where(
                rule_passive_contact.occurred,
                jnp.int32(OUTCOME_DEFLECTION),
                jnp.int32(OUTCOME_NONE),
            ),
            restart_kind=jnp.int32(RK_NONE),
            law11_effect=jnp.int32(LAW11_DEFLECTION_NO_RESET),
            kick_applied=jnp.bool_(False),
            intent_source=jnp.int32(INTENT_SOURCE_NONE),
        )
        passive_retouch_event = detect_restart_retouch(
            held_state,
            passive_contact_result,
            rule_passive_contact.position,
            time_fraction=rule_passive_contact.time_fraction,
        )
        passive_offside = detect_passive_offside(
            advanced,
            roster,
            passive_contact_result,
            rule_passive_contact.position,
            time_fraction=rule_passive_contact.time_fraction,
        )
        passive_retouch_precedes = passive_retouch_event.occurred & (
            (~passive_offside.event.occurred)
            | (
                passive_retouch_event.time_fraction
                <= passive_offside.event.time_fraction
            )
        )
        passive_offside_precedes = passive_offside.event.occurred & (
            (~passive_retouch_event.occurred)
            | (
                passive_offside.event.time_fraction
                < passive_retouch_event.time_fraction
            )
        )
        passive_offside_restart = RestartRetouchEvent(
            occurred=passive_offside_precedes,
            offender=passive_offside.event.actor,
            source_restart_kind=jnp.int32(RK_OFFSIDE),
            team=jnp.where(
                passive_offside_precedes,
                jnp.int32(1) - passive_offside.event.team,
                jnp.int32(NO_TEAM),
            ),
            restart_kind=jnp.where(
                passive_offside_precedes,
                jnp.int32(RK_OFFSIDE),
                jnp.int32(RK_NONE),
            ),
            indirect=passive_offside_precedes,
            time_fraction=passive_offside.event.time_fraction,
            contact_position=passive_offside.event.position,
            restart_position=jnp.zeros_like(passive_offside.event.position),
        )
        passive_indirect_event = jax.tree.map(
            lambda retouch_value, offside_value: jnp.where(
                passive_retouch_precedes, retouch_value, offside_value
            ),
            passive_retouch_event,
            passive_offside_restart,
        )
        with jax.named_scope("messi/passive_indirect_restart"):
            passive_violation = apply_indirect_restart(
                passive_offside.state,
                roster,
                passive_indirect_event,
                field_half_extent,
                goal=goal_geometry,
                ball=ball_geometry,
            )
        raw_retouch_event = jax.tree.map(
            lambda passive_value, previous_value: jnp.where(
                passive_retouch_precedes, passive_value, previous_value
            ),
            passive_violation.event,
            raw_retouch_event,
        )
        raw_offside_event = jax.tree.map(
            lambda passive_value, previous_value: jnp.where(
                passive_offside_precedes, passive_value, previous_value
            ),
            passive_offside.event,
            raw_offside_event,
        )
        selection = select_first_stoppage(
            foul.event.occurred,
            foul.event.time_fraction,
            raw_retouch_event.occurred,
            raw_retouch_event.time_fraction,
            raw_offside_event.occurred,
            raw_offside_event.time_fraction,
            ball_events.boundary.occurred,
            ball_events.boundary.time_fraction,
        )
        use_foul = selection.kind == STOPPAGE_FOUL
        use_passive_retouch = selection.kind == STOPPAGE_RETOUCH
        use_offside = selection.kind == STOPPAGE_OFFSIDE
        use_boundary = selection.kind == STOPPAGE_BOUNDARY
        foul_event = jax.tree.map(
            lambda value, empty: jnp.where(use_foul, value, empty),
            foul_event,
            empty_foul_event(state.ball.position.dtype),
        )
        raw_retouch_event = jax.tree.map(
            lambda value, empty: jnp.where(use_passive_retouch, value, empty),
            raw_retouch_event,
            empty_restart_retouch_event(state.ball.position.dtype),
        )
        raw_offside_event = jax.tree.map(
            lambda value, empty: jnp.where(use_offside, value, empty),
            raw_offside_event,
            empty_offside_event(state.ball.position.dtype),
        )
        boundary_crossing = ball_events.boundary._replace(
            occurred=ball_events.boundary.occurred & use_boundary
        )
        timeline_state = passive_violation.state
        with jax.named_scope("messi/boundary"):
            boundary = resolve_boundary_crossing(
                timeline_state,
                roster,
                boundary_crossing,
                field_half_extent,
                goal=goal_geometry,
                ball=ball_geometry,
            )
        selected_state = jax.tree.map(
            lambda foul_value, ordinary_value: jnp.where(
                use_foul, foul_value, ordinary_value
            ),
            foul.state,
            boundary.state,
        )
        boundary = boundary._replace(state=selected_state)
        boundary_opened = boundary.event.occurred
        if collect_events:
            nutmeg_contact_actor = resolved.contact.actor
            nutmeg_contact_actor_valid = (nutmeg_contact_actor >= 0) & (
                nutmeg_contact_actor < player_count
            )
            nutmeg_contact_occurred = nutmeg_contact_actor_valid & (
                resolved.contact.outcome != OUTCOME_NONE
            )
            last_contact = before.possession.last_contact
            provenance_actor = last_contact.actor
            provenance_valid = (provenance_actor >= 0) & (
                provenance_actor < player_count
            )
            deliberate_provenance = (
                provenance_valid
                & (last_contact.mechanism != MECHANISM_NONE)
                & (last_contact.mechanism != MECHANISM_PASSIVE_BODY)
                & (last_contact.intent_source != INTENT_SOURCE_NONE)
            )
            ground_rebound = (held_state.ball.velocity[2] < 0.0) & (
                boundary.state.ball.velocity[2] >= 0.0
            )
            uninterrupted = (
                (~nutmeg_contact_occurred)
                & (~release_now)
                & (~passive_contact.occurred)
                & (~woodwork.occurred)
                & (~boundary.event.occurred)
                & (~ball_events.event_budget_exhausted)
                & (~ground_rebound)
            )
            raw_nutmeg = detect_nutmeg(
                held_state.ball.position,
                boundary.state.ball.position - held_state.ball.position,
                held_state.ball.live & boundary.state.ball.live,
                players,
                roster,
                attacker=provenance_actor,
                deliberate=deliberate_provenance,
                uninterrupted=uninterrupted,
                ball_radius_m=ball_geometry.radius_m,
                physics=passive_contact_physics,
            )
        penalty_team = after_penalty_team = boundary.state.penalty_completion_team
        penalty_active = boundary.state.penalty_completion_active
        contact_actor = resolved.contact.actor
        valid_contact_actor = (contact_actor >= 0) & (contact_actor < player_count)
        safe_contact_actor = jnp.clip(contact_actor, 0, player_count - 1)
        deliberate_contact = valid_contact_actor & (
            resolved.contact.outcome != OUTCOME_NONE
        )
        defending_goalkeeper_contact = (
            deliberate_contact
            & roster.is_goalkeeper[safe_contact_actor]
            & (roster.team_id[safe_contact_actor] != penalty_team)
            & boundary.state.ball.live
        )
        defending_goalkeeper = roster.is_goalkeeper & (roster.team_id != penalty_team)
        passive_other_player = jnp.any(
            ball_events.passive_actor_mask & (~defending_goalkeeper)
        )
        settling_contact = (
            deliberate_contact & (~defending_goalkeeper_contact)
        ) | passive_other_player
        stopped = jnp.sum(
            boundary.state.ball.velocity * boundary.state.ball.velocity
        ) <= jnp.asarray(1.0e-6, dtype=boundary.state.ball.velocity.dtype)
        penalty_settled = penalty_active & (
            (~boundary.state.ball.live)
            | restart_active(boundary.state.restart.kind)
            | settling_contact
            | stopped
        )
        boundary_state = boundary.state._replace(
            penalty_completion_active=penalty_active & (~penalty_settled),
            penalty_completion_team=jnp.where(
                penalty_active & (~penalty_settled),
                after_penalty_team,
                jnp.int32(NO_TEAM),
            ).astype(jnp.int32),
        )
        after = advance_restart_clock(
            boundary_state,
            restart_visible
            & (~release_now)
            & (~boundary_opened)
            & (~approaching)
            & restart_non_takers_positioned,
        )
        stoppage = restart_active(after.restart.kind)
        after = after._replace(
            offside_flagged=jnp.where(
                stoppage,
                jnp.zeros_like(after.offside_flagged),
                after.offside_flagged,
            ),
            offside_direct_exempt_team=jnp.where(
                stoppage,
                jnp.int32(NO_TEAM),
                after.offside_direct_exempt_team,
            ),
        )
        identity_changed = (
            (before.possession.team != after.possession.team)
            | (before.possession.player != after.possession.player)
            | (before.possession.control_ticks != after.possession.control_ticks)
        )
        if collect_events:
            movement_parameter_rows = (
                rule_state.players.active
                & movement_enabled
                & (~moving_approach_actor)
                & (~pinned)
            )
            release_contact = release_plan.state.possession.last_contact
            release_safe_actor = jnp.clip(release_contact.actor, 0, player_count - 1)
            release_velocity = release_plan.state.ball.velocity
            release_horizontal_speed = jnp.linalg.norm(release_velocity[:2])
            release_direction = release_velocity[:2] / jnp.maximum(
                release_horizontal_speed,
                jnp.asarray(1.0e-9, release_velocity.dtype),
            )
            release_event = _applied_contact_event(
                release_now,
                release_contact,
                roster,
                release_plan.state.ball.position,
                substep_index.astype(jnp.int32),
                jnp.asarray(0.0, dtype=state.ball.position.dtype),
                release_direction,
                jnp.arctan2(release_velocity[2], release_horizontal_speed),
                physics_action.spin[release_safe_actor],
                release_velocity,
                release_now,
                before.attack_direction[roster.team_id[release_safe_actor]],
            )
            resolved_contact = resolved.contact
            resolved_actor = jnp.clip(resolved_contact.actor, 0, player_count - 1)
            resolved_occurred = (
                (resolved_contact.actor >= 0)
                & (resolved_contact.actor < player_count)
                & (resolved_contact.mechanism != MECHANISM_NONE)
            )
            resolved_actor_rows = (
                jnp.arange(player_count, dtype=jnp.int32) == resolved_actor
            ) & resolved_occurred
            contact_parameters_applied = resolved_occurred & (
                resolved_contact.outcome != OUTCOME_CATCH
            )
            automatic_hand_response = (
                resolved_contact.mechanism == MECHANISM_GOALKEEPER_HAND
            ) & (physics_action.force_power[resolved_actor] <= 0.0)
            parameter_rows = resolved_actor_rows & contact_parameters_applied
            directed_rows = parameter_rows & (~automatic_hand_response)
            kick_rows = resolved_actor_rows & resolved_contact.kick_applied
            release_rows = (
                (jnp.arange(player_count, dtype=jnp.int32) == release_plan.actor)
                & release_now
                & (~release_plan.forced)
            )
            release_spin_rows = release_rows & (entry_restart_kind != RK_THROWIN)
            substep_parameters = (
                _parameter_bits(movement_parameter_rows, PARAMETER_MOVE)
                | _parameter_bits(
                    directed_rows | release_rows, PARAMETER_FORCE_DIRECTION
                )
                | _parameter_bits(
                    parameter_rows | release_rows, PARAMETER_FORCE_POWER
                )
                | _parameter_bits(directed_rows | release_rows, PARAMETER_LAUNCH)
                | _parameter_bits(kick_rows | release_spin_rows, PARAMETER_SPIN)
            )
            resolved_possession_changed = (
                contact_state.possession.team != resolved.state.possession.team
            ) | (contact_state.possession.player != resolved.state.possession.player)
            resolved_event = _applied_contact_event(
                resolved_occurred,
                resolved_contact,
                roster,
                contact_state.ball.position,
                substep_index.astype(jnp.int32),
                jnp.asarray(0.0, dtype=state.ball.position.dtype),
                physics_action.force_direction[resolved_actor],
                physics_action.launch[resolved_actor],
                physics_action.spin[resolved_actor],
                resolved.state.ball.velocity,
                resolved_possession_changed,
                contact_state.attack_direction[roster.team_id[resolved_actor]],
            )
            substep_contact = jax.tree.map(
                lambda release_value, resolved_value: jnp.where(
                    release_now, release_value, resolved_value
                ),
                release_event,
                resolved_event,
            )
            first_applied_contact = (
                ~carry.applied_contact.occurred
            ) & substep_contact.occurred
            applied_contact = jax.tree.map(
                lambda new, old: jnp.where(first_applied_contact, new, old),
                substep_contact,
                carry.applied_contact,
            )
            first_boundary = (~carry.boundary.occurred) & boundary.event.occurred
            boundary_event = jax.tree.map(
                lambda new, old: jnp.where(first_boundary, new, old),
                boundary.event,
                carry.boundary,
            )
            first_retouch = (
                ~carry.restart_retouch.occurred
            ) & raw_retouch_event.occurred
            retouch_event = jax.tree.map(
                lambda new, old: jnp.where(first_retouch, new, old),
                raw_retouch_event,
                carry.restart_retouch,
            )
            first_woodwork = (~carry.woodwork.occurred) & woodwork.occurred
            woodwork_event = jax.tree.map(
                lambda new, old: jnp.where(first_woodwork, new, old),
                woodwork,
                carry.woodwork,
            )
            first_foul = (~carry.foul.occurred) & foul_event.occurred
            captured_foul_event = jax.tree.map(
                lambda new, old: jnp.where(first_foul, new, old),
                foul_event,
                carry.foul,
            )
            first_offside = (~carry.offside.occurred) & raw_offside_event.occurred
            offside_event = jax.tree.map(
                lambda new, old: jnp.where(first_offside, new, old),
                raw_offside_event,
                carry.offside,
            )
            first_passive = (~carry.passive_contact.occurred) & passive_contact.occurred
            passive_event = jax.tree.map(
                lambda new, old: jnp.where(first_passive, new, old),
                passive_contact,
                carry.passive_contact,
            )
            event_chronology = FrameEventChronology(
                rule_stoppage_substep=jnp.where(
                    first_foul | first_retouch | first_offside,
                    substep_index.astype(jnp.int32),
                    carry.event_chronology.rule_stoppage_substep,
                ),
                passive_contact_substep=jnp.where(
                    first_passive,
                    substep_index.astype(jnp.int32),
                    carry.event_chronology.passive_contact_substep,
                ),
                woodwork_substep=jnp.where(
                    first_woodwork,
                    substep_index.astype(jnp.int32),
                    carry.event_chronology.woodwork_substep,
                ),
                boundary_substep=jnp.where(
                    first_boundary,
                    substep_index.astype(jnp.int32),
                    carry.event_chronology.boundary_substep,
                ),
            )
            first_nutmeg = (~carry.nutmeg.occurred) & raw_nutmeg.occurred
            nutmeg_event = jax.tree.map(
                lambda new, old: jnp.where(first_nutmeg, new, old),
                raw_nutmeg,
                carry.nutmeg,
            )
            return _ControlCarry(
                state=after,
                contact_attempted=attempted,
                override_valid=(
                    carry.override_valid
                    & contest.override_valid
                    & body_foul.override_valid
                ),
                contact_occurred=(
                    carry.contact_occurred
                    | contest.occurred
                    | passive_contact.occurred
                    | release_now
                ),
                challenge_attempt_count=(
                    carry.challenge_attempt_count
                    + challenge_selected.astype(jnp.int32)
                ),
                parameter_consumed=(
                    carry.parameter_consumed | substep_parameters
                ),
                applied_contact=applied_contact,
                event_chronology=event_chronology,
                possession_changed=carry.possession_changed | identity_changed,
                boundary=boundary_event,
                restart_released=carry.restart_released | release_now,
                restart_forced=(
                    carry.restart_forced | (release_now & release_plan.forced)
                ),
                restart_retouch=retouch_event,
                woodwork=woodwork_event,
                foul=captured_foul_event,
                offside=offside_event,
                passive_contact=passive_event,
                nutmeg=nutmeg_event,
                goalkeeper_holding=carry.goalkeeper_holding,
                event_budget_exhausted=(
                    carry.event_budget_exhausted | ball_events.event_budget_exhausted
                ),
                body_impact_consumed=(carry.body_impact_consumed | body_foul.eligible),
            ), None
        return _TrainingCarry(
            state=after,
            contact_attempted=attempted,
            possession_changed=carry.possession_changed | identity_changed,
            body_impact_consumed=carry.body_impact_consumed | body_foul.eligible,
        ), None

    if contest_overrides is None:
        scan_inputs = (substep_indices, substep_keys, body_foul_draws)
    elif body_foul_overrides is None:
        scan_inputs = (
            substep_indices,
            substep_keys,
            body_foul_draws,
            contest_overrides,
        )
    else:
        scan_inputs = (
            substep_indices,
            substep_keys,
            body_foul_draws,
            contest_overrides,
            body_foul_overrides,
        )
    final, _ = lax.scan(substep, initial, scan_inputs)
    verified_carrier = verified_controlled_carrier(
        final.state,
        roster,
        contact_physics,
        ball_geometry.radius_m,
    )
    possession_claimed = (
        (final.state.possession.team >= 0)
        | (final.state.possession.player >= 0)
        | (final.state.possession.control_ticks > 0)
    )
    live_open_play = final.state.ball.live & (
        final.state.restart.kind == jnp.int32(RK_NONE)
    )
    release_uncontrolled = (
        live_open_play
        & possession_claimed
        & (verified_carrier == jnp.int32(NO_PLAYER))
        & (~fresh_trap_control_grace(final.state, roster))
    )
    old_team = final.state.possession.team
    valid_old_team = (old_team == 0) | (old_team == 1)
    possession = final.state.possession._replace(
        team=jnp.where(release_uncontrolled, NO_TEAM, old_team).astype(jnp.int32),
        player=jnp.where(
            release_uncontrolled, NO_PLAYER, final.state.possession.player
        ).astype(jnp.int32),
        previous_team=jnp.where(
            release_uncontrolled & valid_old_team,
            old_team,
            final.state.possession.previous_team,
        ).astype(jnp.int32),
        control_ticks=jnp.where(
            release_uncontrolled, 0, final.state.possession.control_ticks
        ).astype(jnp.int32),
    )
    has_control = (possession.team >= 0) & (possession.player >= 0)
    possession_changed = final.possession_changed | release_uncontrolled
    continuous_control = has_control & (~possession_changed)
    control_ticks = jnp.where(
        continuous_control,
        state.possession.control_ticks + jnp.int32(1),
        jnp.where(has_control, possession.control_ticks, jnp.int32(0)),
    )
    # 소유 에피소드는 frame 경계의 team·player identity가 유지되는 동안 이어진다.
    # 같은 선수의 반복 control touch는 `control_ticks`만 되돌리고 에피소드는 잇는다.
    next_control_tick = state.control_tick + jnp.int32(1)
    had_control = (state.possession.team >= 0) & (state.possession.player >= 0)
    same_identity = (state.possession.team == possession.team) & (
        state.possession.player == possession.player
    )
    episode_start_tick = jnp.where(
        has_control,
        jnp.where(
            had_control & same_identity,
            state.possession.episode_start_tick,
            next_control_tick,
        ),
        jnp.int32(-1),
    ).astype(jnp.int32)
    next_state = final.state._replace(
        control_tick=next_control_tick,
        possession=possession._replace(
            control_ticks=control_ticks.astype(jnp.int32),
            episode_start_tick=episode_start_tick,
        ),
    )
    if not collect_events:
        return next_state
    return ControlFrameResult(
        state=next_state,
        receipt=ControlFrameReceipt(
            override_valid=final.override_valid,
            contact_occurred=final.contact_occurred,
            challenge_attempt_count=final.challenge_attempt_count,
            parameter_consumed=final.parameter_consumed,
            applied_contact=final.applied_contact,
            event_chronology=final.event_chronology,
            boundary=final.boundary,
            restart_released=final.restart_released,
            restart_forced=final.restart_forced,
            restart_retouch=final.restart_retouch,
            woodwork=final.woodwork,
            foul=final.foul,
            offside=final.offside,
            passive_contact=final.passive_contact,
            nutmeg=final.nutmeg,
            goalkeeper_holding=final.goalkeeper_holding,
            event_budget_exhausted=final.event_budget_exhausted,
        ),
    )


def step_control(state, roster, action: IntentAction, key: jax.Array, **kwargs):
    """Receipt와 override 입력이 없는 일반 학습용 control step."""

    return _step_control_impl(
        state,
        roster,
        action,
        key,
        None,
        None,
        collect_events=False,
        **kwargs,
    )


def step_control_with_events(
    state,
    roster,
    action: IntentAction,
    key: jax.Array,
    **kwargs,
) -> ControlFrameResult:
    """Sampling 결과와 compact event receipt를 반환하는 평가용 step."""

    return _step_control_impl(
        state,
        roster,
        action,
        key,
        None,
        None,
        collect_events=True,
        **kwargs,
    )


def step_control_replay(
    state,
    roster,
    action: IntentAction,
    key: jax.Array,
    contest_overrides: ContestOverride,
    body_foul_overrides: BodyFoulOverride | None = None,
    **kwargs,
) -> ControlFrameResult:
    """Substep override를 검증·적용하고 receipt를 반환하는 replay step.

    ``body_foul_overrides=None``인 기존 호출은 실행 호환성을 위해 허용하지만
    ``receipt.override_valid=False``를 반환하므로 strict replay로 성공할 수 없다.
    """

    return _step_control_impl(
        state,
        roster,
        action,
        key,
        contest_overrides,
        body_foul_overrides,
        collect_events=True,
        **kwargs,
    )


__all__ = [
    "AppliedContactEvent",
    "ControlFrameReceipt",
    "ControlFrameResult",
    "FrameEventChronology",
    "empty_applied_contact_event",
    "empty_frame_event_chronology",
    "sample_control_body_foul_overrides",
    "sample_control_contest_overrides",
    "step_control",
    "step_control_replay",
    "step_control_with_events",
]
