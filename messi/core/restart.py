"""고정 shape restart layout, timing과 deterministic release."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.core.action import INTENT_PASS, INTENT_SHOOT
from messi.core.collision_config import BodyCollision
from messi.core.constants import (
    NO_PLAYER,
    NO_TEAM,
    RESTART_COUNT,
    RK_CORNER,
    RK_FREEKICK,
    RK_GK_HOLD,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_NONE,
    RK_OFFSIDE,
    RK_PENALTY,
    RK_THROWIN,
    TEAM_0,
    TEAM_1,
)
from messi.core.contact import (
    INTENT_SOURCE_ENVIRONMENT_FORCED,
    INTENT_SOURCE_POLICY,
    LAW11_DELIBERATE_PLAY_RESET,
    LAW11_DIRECT_RESTART_EXEMPTION,
    MECHANISM_FOOT,
    MECHANISM_THROW,
    OUTCOME_RELEASE,
    ContactPhysics,
    ContactResult,
    PhysicsAction,
    apply_foot_actuation,
)
from messi.core.formation import (
    ROLE_CENTRE_BACK,
    ROLE_CENTRE_FORWARD,
    ROLE_FULL_BACK,
    ROLE_GOALKEEPER,
    ROLE_WIDE_FORWARD,
    formation_positions,
)
from messi.core.geometry import BallGeometry
from messi.core.numeric import require_float32_representable
from messi.core.state import PlayerState, RestartReleaseProvenance
from messi.core.timebase import DEFAULT_TIMEBASE, MAX_INT32_TICKS, Timebase

_DIV_EPS = 1.0e-9
_GEOMETRY_EPS = 1.0e-6


@dataclass(frozen=True, slots=True)
class RestartPhysics:
    """환경 소유 restart timing과 법적 거리의 static 설정."""

    forced_release_delay_s: float = 3.0
    # Continuous taker locomotion가 다른 선수와의 collision 때문에 release pose에
    # 도달하지 못해도 restart가 무한히 멈추지 않도록 하는 상한이다. 상한 뒤에는
    # release pose로 snap한 다음 기존 positioning countdown을 그대로 수행한다.
    approach_timeout_s: float = 30.0
    throwin_opponent_distance_m: float = 2.0
    set_piece_opponent_distance_m: float = 9.15
    penalty_area_length_m: float = 16.5
    penalty_area_half_width_m: float = 20.16
    goal_half_width_m: float = 3.66
    forced_release_power: float = 0.35
    approach_arrival_tolerance_m: float = 0.08
    continuous_freekick_approach: bool = True
    continuous_offside_approach: bool = True
    continuous_corner_approach: bool = True
    continuous_throwin_approach: bool = True
    # 데드볼 동안 행동은 자유롭게 고르되 장기·단기 stamina를 모두 얼린다. 재개 준비의
    # 걷기가 경기 체력을 깎지 않는다. 회복도 같이 멈추므로 데드볼은 휴식이 아니다.
    freeze_dead_ball_stamina: bool = True
    # 기본 restart는 현재 경기 shape를 보존하고 법적으로 필요한 최소 이동만 적용한다.
    # Formation anchor 재배치는 명시적인 실험 profile에서만 opt-in한다.
    role_aware_positioning: bool = False
    face_ball_after_positioning: bool = True

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("forced_release_delay_s", 0.1, 10.0),
            ("approach_timeout_s", 1.0, 60.0),
            ("throwin_opponent_distance_m", 1.0, 5.0),
            ("set_piece_opponent_distance_m", 5.0, 15.0),
            ("penalty_area_length_m", 10.0, 25.0),
            ("penalty_area_half_width_m", 10.0, 30.0),
            ("goal_half_width_m", 2.0, 5.0),
            ("forced_release_power", 0.05, 1.0),
            ("approach_arrival_tolerance_m", 0.01, 0.5),
        ):
            value = require_float32_representable(name, getattr(self, name))
            if not lower <= value <= upper:
                raise ValueError(f"{name}은 [{lower}, {upper}] 범위여야 한다")
        for name in (
            "continuous_freekick_approach",
            "continuous_offside_approach",
            "continuous_corner_approach",
            "continuous_throwin_approach",
            "freeze_dead_ball_stamina",
            "role_aware_positioning",
            "face_ball_after_positioning",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name}은 bool이어야 한다")


_DEFAULT_RESTART = RestartPhysics()
_DEFAULT_BALL = BallGeometry()
_DEFAULT_BODY = BodyCollision()
_DEFAULT_CONTACT = ContactPhysics()


class RestartPreparation(NamedTuple):
    state: object
    prepared: jax.Array


class RestartRelease(NamedTuple):
    state: object
    occurred: jax.Array
    forced: jax.Array
    actor: jax.Array


class RestartTakerRepair(NamedTuple):
    state: object
    repaired: jax.Array


def restart_active(kind: jax.Array) -> jax.Array:
    kind = jnp.asarray(kind, dtype=jnp.int32)
    return (kind > RK_NONE) & (kind < RESTART_COUNT)


def forced_release_delay_substeps(
    *,
    timebase: Timebase = DEFAULT_TIMEBASE,
    config: RestartPhysics = _DEFAULT_RESTART,
) -> int:
    """3초 positioning deadline을 host에서 한 번 physics substep으로 변환한다."""

    return timebase.physics_steps_for(
        config.forced_release_delay_s,
        rounding="ceil",
        minimum=1,
        maximum=MAX_INT32_TICKS,
    )


def restart_actor_mask(state, roster) -> jax.Array:
    """구조적으로 일관된 restart taker 한 명만 허용한다."""

    count = state.players.position.shape[0]
    taker = state.restart.taker
    valid_taker = (taker >= 0) & (taker < count)
    safe_taker = jnp.clip(taker, 0, count - 1)
    valid_team = (state.restart.team == TEAM_0) | (state.restart.team == TEAM_1)
    coherent = (
        restart_active(state.restart.kind)
        & valid_team
        & valid_taker
        & state.players.active[safe_taker]
        & (roster.team_id[safe_taker] == state.restart.team)
    )
    return coherent & (jnp.arange(count, dtype=jnp.int32) == taker)


def repair_restart_taker(state, roster) -> RestartTakerRepair:
    """무효가 된 pending restart taker를 가장 가까운 active teammate로 복구한다.

    복구 시 layout을 무효화해 새 taker가 policy에 관측되기 전 같은 frame에서 release되지
    않게 한다. 후보가 없으면 state를 변경하지 않고 ``repaired=False``를 반환한다.
    """

    count = state.players.position.shape[0]
    taker = state.restart.taker
    valid_slot = (taker >= 0) & (taker < count)
    safe_taker = jnp.clip(taker, 0, count - 1)
    valid_team = (state.restart.team == TEAM_0) | (state.restart.team == TEAM_1)
    usable = (
        restart_active(state.restart.kind)
        & valid_team
        & valid_slot
        & state.players.active[safe_taker]
        & (roster.team_id[safe_taker] == state.restart.team)
    )
    broken = restart_active(state.restart.kind) & (~usable)
    eligible = (
        valid_team & state.players.active & (roster.team_id == state.restart.team)
    )
    distance_squared = jnp.sum(
        (state.players.position - state.ball.position[:2]) ** 2,
        axis=-1,
    )
    replacement = jnp.argmin(jnp.where(eligible, distance_squared, jnp.inf)).astype(
        jnp.int32
    )
    repaired = broken & jnp.any(eligible)
    next_state = state._replace(
        restart=state.restart._replace(
            taker=jnp.where(repaired, replacement, taker).astype(jnp.int32)
        ),
        restart_layout_ready=jnp.where(
            repaired, jnp.bool_(False), state.restart_layout_ready
        ),
        restart_taker_approach=jnp.where(
            repaired, jnp.bool_(False), state.restart_taker_approach
        ),
    )
    return RestartTakerRepair(next_state, repaired)


def _unit(vector: jax.Array, fallback: jax.Array) -> jax.Array:
    norm = jnp.sqrt(jnp.sum(vector * vector) + _DIV_EPS)
    return jnp.where(norm > _GEOMETRY_EPS, vector / norm, fallback)


def _project_into_box_with_clearance(
    position: jax.Array,
    centre: jax.Array,
    required_distance: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    fallback_direction: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Box 안에서 centre clearance를 만족하는 최소 fixed 후보를 고른다."""

    dtype = position.dtype
    count = position.shape[0]
    centre = jnp.broadcast_to(jnp.asarray(centre, dtype=dtype), (count, 2))
    required = jnp.broadcast_to(jnp.asarray(required_distance, dtype=dtype), (count,))
    projection_distance = jnp.where(
        required > 0.0,
        required + jnp.asarray(8.0 * _GEOMETRY_EPS, dtype=dtype),
        required,
    )
    lower = jnp.broadcast_to(jnp.asarray(lower, dtype=dtype), (count, 2))
    upper = jnp.broadcast_to(jnp.asarray(upper, dtype=dtype), (count, 2))
    clipped = jnp.clip(position, lower, upper)

    delta = clipped - centre
    distance_squared = jnp.sum(delta * delta, axis=-1)
    distance = jnp.sqrt(distance_squared)
    fallback = jnp.asarray(fallback_direction, dtype=dtype)
    fallback_norm = jnp.sqrt(jnp.sum(fallback * fallback, axis=-1, keepdims=True))
    fallback = fallback / jnp.maximum(fallback_norm, _DIV_EPS)
    radial_direction = jnp.where(
        (distance > _GEOMETRY_EPS)[:, None],
        delta / jnp.maximum(distance[:, None], _DIV_EPS),
        fallback,
    )
    radial = centre + projection_distance[:, None] * radial_direction

    corners = jnp.stack(
        (
            jnp.stack((lower[:, 0], lower[:, 1]), axis=-1),
            jnp.stack((lower[:, 0], upper[:, 1]), axis=-1),
            jnp.stack((upper[:, 0], lower[:, 1]), axis=-1),
            jnp.stack((upper[:, 0], upper[:, 1]), axis=-1),
        ),
        axis=1,
    )
    segment = corners - clipped[:, None, :]
    relative = clipped - centre
    quadratic_a = jnp.sum(segment * segment, axis=-1)
    quadratic_b = 2.0 * jnp.sum(relative[:, None, :] * segment, axis=-1)
    quadratic_c = distance_squared - projection_distance * projection_distance
    discriminant = quadratic_b * quadratic_b - (
        4.0 * quadratic_a * quadratic_c[:, None]
    )
    safe_a = jnp.maximum(quadratic_a, _DIV_EPS)
    root = (-quadratic_b + jnp.sqrt(jnp.maximum(discriminant, 0.0))) / (2.0 * safe_a)
    intersection = clipped[:, None, :] + root[:, :, None] * segment
    root_valid = (
        (quadratic_a > _DIV_EPS) & (discriminant >= 0.0) & (root >= 0.0) & (root <= 1.0)
    )

    candidates = jnp.concatenate(
        (clipped[:, None, :], radial[:, None, :], intersection), axis=1
    )
    source_valid = jnp.concatenate(
        (
            jnp.ones((count, 2), dtype=jnp.bool_),
            root_valid,
        ),
        axis=1,
    )
    inside = jnp.all(
        (candidates >= lower[:, None, :] - _GEOMETRY_EPS)
        & (candidates <= upper[:, None, :] + _GEOMETRY_EPS),
        axis=-1,
    )
    candidate_distance = jnp.sqrt(
        jnp.sum((candidates - centre[:, None, :]) ** 2, axis=-1)
    )
    box_valid = jnp.all(lower <= upper, axis=-1)
    valid = (
        source_valid
        & inside
        & box_valid[:, None]
        & jnp.all(jnp.isfinite(candidates), axis=-1)
        & (candidate_distance >= required[:, None] - _GEOMETRY_EPS)
    )
    displacement_squared = jnp.sum((candidates - position[:, None, :]) ** 2, axis=-1)
    masked_cost = jnp.where(valid, displacement_squared, jnp.inf)
    minimum_cost = jnp.min(masked_cost, axis=1)
    tie_tolerance = jnp.float32(8.0 * _GEOMETRY_EPS) * jnp.maximum(
        jnp.float32(1.0), minimum_cost
    )
    minimum_candidate = valid & (
        displacement_squared <= minimum_cost[:, None] + tie_tolerance[:, None]
    )
    choice = jnp.argmax(minimum_candidate.astype(jnp.int32), axis=1)
    selected = candidates[jnp.arange(count, dtype=jnp.int32), choice]
    available = jnp.any(valid, axis=1)
    return jnp.where(available[:, None], selected, clipped), available


def _select_closest_projection(
    position: jax.Array,
    candidates: jax.Array,
    available: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Fixed candidate projection 중 원래 위치와 가장 가까운 것을 고른다."""

    displacement_squared = jnp.sum((candidates - position[None, :, :]) ** 2, axis=-1)
    masked_cost = jnp.where(available, displacement_squared, jnp.inf)
    minimum_cost = jnp.min(masked_cost, axis=0)
    tie_tolerance = jnp.asarray(
        8.0 * _GEOMETRY_EPS, dtype=position.dtype
    ) * jnp.maximum(jnp.asarray(1.0, dtype=position.dtype), minimum_cost)
    minimum_candidate = available & (
        displacement_squared <= minimum_cost[None, :] + tie_tolerance[None, :]
    )
    choice = jnp.argmax(minimum_candidate.astype(jnp.int32), axis=0)
    player = jnp.arange(position.shape[0], dtype=jnp.int32)
    selected = candidates[choice, player]
    any_available = jnp.any(available, axis=0)
    return jnp.where(any_available[:, None], selected, position), any_available


def _progress_limited_box(
    inset: jax.Array,
    direction: jax.Array,
    progress_limit: jax.Array,
    y_lower: jax.Array,
    y_upper: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """``x * direction <= progress_limit``와 field inset의 교집합 box."""

    boundary = direction * progress_limit
    lower_x = jnp.where(
        direction > 0.0,
        -inset[0],
        jnp.maximum(boundary, -inset[0]),
    )
    upper_x = jnp.where(
        direction > 0.0,
        jnp.minimum(boundary, inset[0]),
        inset[0],
    )
    return jnp.stack((lower_x, y_lower)), jnp.stack((upper_x, y_upper))


def _release_forward(state, roster) -> jax.Array:
    team = jnp.clip(state.restart.team, TEAM_0, TEAM_1)
    attack = state.attack_direction[team]
    ball_xy = state.ball.position[:2]
    toward_centre = _unit(
        -ball_xy,
        jnp.asarray((attack, 0.0), dtype=ball_xy.dtype),
    )
    touch_inward = jnp.asarray(
        (0.0, -jnp.where(ball_xy[1] >= 0.0, 1.0, -1.0)),
        dtype=ball_xy.dtype,
    )
    kickoff_back = jnp.asarray((-attack, 0.0), dtype=ball_xy.dtype)
    taker = jnp.clip(state.restart.taker, 0, state.players.position.shape[0] - 1)
    taker_relative_y = state.players.position[taker, 1] - ball_xy[1]
    restart_roster = state.players.active & (roster.team_id == team)
    roster_relative_y = jnp.where(
        restart_roster,
        state.players.position[:, 1] - ball_xy[1],
        0.0,
    )
    roster_chirality = jnp.sum(roster_relative_y**3)
    lateral_signal = jnp.where(
        jnp.abs(taker_relative_y) > _GEOMETRY_EPS,
        taker_relative_y,
        roster_chirality,
    )
    lateral_valid = jnp.abs(lateral_signal) > _GEOMETRY_EPS
    lateral_kickoff = jnp.asarray(
        (0.0, -jnp.where(lateral_signal >= 0.0, 1.0, -1.0)),
        dtype=ball_xy.dtype,
    )
    kickoff_forward = jnp.where(lateral_valid, lateral_kickoff, kickoff_back)
    attack_forward = jnp.asarray((attack, 0.0), dtype=ball_xy.dtype)
    return jnp.where(
        state.restart.kind == RK_KICKOFF,
        kickoff_forward,
        jnp.where(
            state.restart.kind == RK_THROWIN,
            touch_inward,
            jnp.where(state.restart.kind == RK_CORNER, toward_centre, attack_forward),
        ),
    )


# Restart 종류별 role 성향. 행은 throw-in, corner, 자기 진영 free kick, 중앙 free kick,
# 상대 진영 free kick, penalty, goal kick, kickoff 순이고 열은 ROLE_* 순서다.
# FootballWorld가 DFL 7경기에서 뽑은 표를 그대로 쓴다. Free-kick 깊이 행과 penalty는
# 그 표에서도 관측이 아니라 compatibility prior다.
_TAKER_PROPENSITY = (
    (0.003, 0.041, 0.839, 0.062, 0.041, 0.003, 0.010),
    (0.018, 0.071, 0.107, 0.589, 0.161, 0.036, 0.018),
    (0.771, 0.157, 0.040, 0.025, 0.004, 0.002, 0.001),
    (0.099, 0.268, 0.200, 0.278, 0.077, 0.018, 0.060),
    (0.000, 0.014, 0.185, 0.230, 0.269, 0.128, 0.174),
    (0.000, 0.000, 0.023, 0.076, 0.229, 0.505, 0.167),
    (0.889, 0.063, 0.016, 0.008, 0.008, 0.008, 0.008),
    (0.000, 0.010, 0.040, 0.160, 0.130, 0.500, 0.160),
)
_TAKER_AFFINITY = jnp.asarray(
    [
        [12.0 * math.log(max(value, 1.0e-3)) for value in row]
        for row in _TAKER_PROPENSITY
    ],
    dtype=jnp.float32,
)
_TAKER_DISTANCE_WEIGHT = jnp.asarray(
    (1.0, 0.35, 0.25, 0.5, 0.4, 0.05, 0.3, 0.2), dtype=jnp.float32
)
_TAKER_STAMINA_WEIGHT = 10.0
_TAKER_AERIAL_PENALTY = 15.0
# Role 친화도만 보면 공에서 수십 m 떨어진 선수가 뽑힌다. MESSI는 corner·throw-in
# 키커가 release pose까지 걸어서 접근하므로 그 사이 강제 release가 먼저 걸려 재개가
# 망가진다. 반경 안의 후보로 먼저 좁힌 뒤 역할로 순위를 매기고, 반경 안에 아무도
# 없으면 전체로 되돌아간다.
_TAKER_CANDIDATE_RADIUS_M = 15.0


def select_restart_taker(state, roster, kind, team, spot, *, field_half_extent):
    """Restart 종류별 role 친화도·지구력·거리로 taker 한 명을 결정한다.

    거리만 보면 penalty, corner, goal kick에서 그럴듯하지 않은 키커가 나온다. MESSI는
    ``RosterState.role``을 이미 들고 있으므로 FootballWorld와 달리 역할 추정이 필요 없다.
    Corner는 장신 centre forward/back을 타깃으로 남기려고 오히려 감점한다.
    ``argmax``가 낮은 slot을 안정적인 tie break로 쓴다.
    """

    dtype = state.players.position.dtype
    kind = jnp.asarray(kind, dtype=jnp.int32)
    spot_xy = jnp.asarray(spot, dtype=dtype)[:2]
    distance = jnp.linalg.norm(state.players.position - spot_xy[None, :], axis=-1)
    half_length = jnp.asarray(field_half_extent, dtype=dtype)[0]
    safe_team = jnp.clip(team, 0, state.attack_direction.shape[0] - 1)
    progress = jnp.clip(
        (spot_xy[0] * state.attack_direction[safe_team] + half_length)
        / jnp.maximum(2.0 * half_length, 1.0e-6),
        0.0,
        1.0,
    )
    row = jnp.where(
        progress < (1.0 / 3.0), 2, jnp.where(progress < (2.0 / 3.0), 3, 4)
    ).astype(jnp.int32)
    row = jnp.where(kind == RK_THROWIN, 0, row)
    row = jnp.where(kind == RK_CORNER, 1, row)
    row = jnp.where(kind == RK_PENALTY, 5, row)
    row = jnp.where(kind == RK_GOALKICK, 6, row)
    row = jnp.where(kind == RK_KICKOFF, 7, row).astype(jnp.int32)

    safe_role = jnp.clip(roster.role.astype(jnp.int32), 0, 6)
    score = (
        _TAKER_AFFINITY[row][safe_role]
        + _TAKER_STAMINA_WEIGHT
        * jnp.clip(state.players.stamina_long.astype(dtype), 0.0, 1.0)
        - _TAKER_DISTANCE_WEIGHT[row] * distance
    )
    corner_target = (kind == RK_CORNER) & (
        (safe_role == ROLE_CENTRE_FORWARD) | (safe_role == ROLE_CENTRE_BACK)
    )
    score = score - jnp.where(
        corner_target,
        _TAKER_AERIAL_PENALTY
        * jnp.clip(roster.reach_height.astype(dtype) / 3.0, 0.0, 1.0),
        0.0,
    )
    eligible = state.players.active & (roster.team_id == team)
    eligible = eligible & jnp.where(
        kind == RK_GK_HOLD, roster.is_goalkeeper, jnp.bool_(True)
    )
    radius = jnp.asarray(_TAKER_CANDIDATE_RADIUS_M, dtype=dtype)
    near = eligible & (distance <= radius)
    usable = jnp.where(jnp.any(near), near, eligible)
    candidate = jnp.argmax(jnp.where(usable, score, -jnp.inf)).astype(jnp.int32)
    return jnp.where(jnp.any(eligible), candidate, jnp.int32(NO_PLAYER))


def continuous_restart_approach_enabled(
    kind: jax.Array,
    *,
    config: RestartPhysics = _DEFAULT_RESTART,
) -> jax.Array:
    """Ordinary restart의 designated taker continuous approach gate."""

    return (
        ((kind == RK_FREEKICK) & jnp.bool_(config.continuous_freekick_approach))
        | ((kind == RK_OFFSIDE) & jnp.bool_(config.continuous_offside_approach))
        | ((kind == RK_CORNER) & jnp.bool_(config.continuous_corner_approach))
        | ((kind == RK_THROWIN) & jnp.bool_(config.continuous_throwin_approach))
        | (kind == RK_GOALKICK)
        | (kind == RK_PENALTY)
    )


def restart_taker_release_pose(
    state,
    roster,
    *,
    ball: BallGeometry = _DEFAULT_BALL,
    body: BodyCollision = _DEFAULT_BODY,
) -> tuple[jax.Array, jax.Array]:
    """Roster의 같은-team 배치를 사용해 실행 가능한 taker pose를 반환한다."""

    dtype = state.players.position.dtype
    gap = jnp.asarray(
        0.5 * body.effective_diameter_m + ball.radius_m + _GEOMETRY_EPS,
        dtype=dtype,
    )
    facing = _release_forward(state, roster)
    return state.ball.position[:2] - gap * facing, facing


def _project_restart_constraints(
    state,
    roster,
    actor: jax.Array,
    taker_target: jax.Array,
    forward: jax.Array,
    field_half_extent: jax.Array,
    *,
    ball: BallGeometry,
    body: BodyCollision,
    contact: ContactPhysics,
    config: RestartPhysics,
) -> tuple[jax.Array, jax.Array]:
    position = state.players.position
    active = state.players.active
    opponent = active & (roster.team_id != state.restart.team)
    dtype = position.dtype
    radius = jnp.asarray(0.5 * body.effective_diameter_m, dtype=dtype)
    prepared = jnp.where(actor[:, None], taker_target, position)
    bound = jnp.asarray(field_half_extent, dtype=dtype)
    ordinary = active & (~actor)
    prepared = jnp.where(
        ordinary[:, None],
        jnp.clip(prepared, -bound + radius, bound - radius),
        prepared,
    )

    delta = prepared - state.ball.position[:2]
    distance_squared = jnp.sum(delta * delta, axis=-1)
    distance = jnp.sqrt(jnp.maximum(distance_squared, _DIV_EPS))
    fallback = jnp.stack(
        (state.attack_direction[roster.team_id], jnp.zeros_like(distance)), axis=-1
    )
    clearance = jnp.where(
        state.restart.kind == RK_THROWIN,
        config.throwin_opponent_distance_m,
        config.set_piece_opponent_distance_m,
    )
    distance_illegal = opponent & (
        (
            (state.restart.kind == RK_THROWIN)
            | (state.restart.kind == RK_CORNER)
            | (state.restart.kind == RK_FREEKICK)
            | (state.restart.kind == RK_OFFSIDE)
        )
        & (distance < clearance + radius)
    )
    inset = bound - radius
    clearance_target, clearance_available = _project_into_box_with_clearance(
        prepared,
        state.ball.position[:2],
        clearance + radius,
        -inset,
        inset,
        fallback,
    )
    # Law 13: 상대는 원칙적으로 9.15 m를 유지하지만 자기 goal line의 두 post
    # 사이에는 설 수 있다. Goal-area line의 attacking indirect free kick에서 일반
    # radial projection만 쓰면 goalkeeper까지 측면으로 밀려 빈 골문이 된다.
    restart_attack = state.attack_direction[
        jnp.clip(state.restart.team, TEAM_0, TEAM_1)
    ]
    attacking_goal_area_free_kick = (
        ((state.restart.kind == RK_FREEKICK) | (state.restart.kind == RK_OFFSIDE))
        & (
            state.ball.position[0] * restart_attack
            >= field_half_extent[0] - config.penalty_area_length_m
        )
    )
    goal_line_y_limit = jnp.maximum(
        jnp.asarray(config.goal_half_width_m, dtype=dtype) - radius,
        jnp.asarray(0.0, dtype=dtype),
    )
    goal_line_target = jnp.stack(
        (
            jnp.full_like(
                distance,
                restart_attack * (field_half_extent[0] - radius),
            ),
            jnp.clip(prepared[:, 1], -goal_line_y_limit, goal_line_y_limit),
        ),
        axis=-1,
    )
    clearance_cost = jnp.sum((clearance_target - prepared) ** 2, axis=-1)
    goal_line_cost = jnp.sum((goal_line_target - prepared) ** 2, axis=-1)
    use_goal_line_exception = (
        opponent
        & attacking_goal_area_free_kick
        & distance_illegal
        & (goal_line_cost < clearance_cost)
    )
    clearance_target = jnp.where(
        use_goal_line_exception[:, None], goal_line_target, clearance_target
    )
    prepared = jnp.where(
        (distance_illegal & clearance_available)[:, None],
        clearance_target,
        prepared,
    )

    team_direction = state.attack_direction[roster.team_id]
    kickoff_subject = active & (~actor)
    own_half_lower_x = jnp.where(team_direction > 0.0, -inset[0], radius)
    own_half_upper_x = jnp.where(team_direction > 0.0, -radius, inset[0])
    own_half_lower = jnp.stack(
        (own_half_lower_x, jnp.full_like(own_half_lower_x, -inset[1])), axis=-1
    )
    own_half_upper = jnp.stack(
        (own_half_upper_x, jnp.full_like(own_half_upper_x, inset[1])), axis=-1
    )
    # Law 8은 같은 team 비키커에게 centre-mark 최소 거리를 요구하지 않지만,
    # contact model은 player centre가 carry envelope 안에 있으면 다음 visible
    # release를 즉시 새 touch로 처리한다. Own-half projection이 centre-forward를
    # halfway line 바로 뒤에 두더라도 kick이 teammate 몸에서 시작하지 않게 한다.
    kickoff_teammate_distance = jnp.asarray(
        contact.carry_radius_m + ball.radius_m,
        dtype=dtype,
    )
    kickoff_distance = jnp.where(
        opponent,
        config.set_piece_opponent_distance_m + radius,
        kickoff_teammate_distance,
    )
    kickoff_target, kickoff_available = _project_into_box_with_clearance(
        prepared,
        jnp.zeros(2, dtype=dtype),
        kickoff_distance,
        own_half_lower,
        own_half_upper,
        fallback,
    )
    prepared = jnp.where(
        ((state.restart.kind == RK_KICKOFF) & kickoff_subject & kickoff_available)[
            :, None
        ],
        kickoff_target,
        prepared,
    )

    end_sign = jnp.where(state.ball.position[0] >= 0.0, 1.0, -1.0)
    penalty_x = field_half_extent[0] - config.penalty_area_length_m
    area_progress_limit = penalty_x - radius
    lateral_limit = config.penalty_area_half_width_m + radius
    inside_penalty = (
        opponent
        & (prepared[:, 0] * end_sign > area_progress_limit)
        & (jnp.abs(prepared[:, 1]) < lateral_limit)
    )

    goal_x_lower, goal_x_upper = _progress_limited_box(
        inset,
        end_sign,
        area_progress_limit,
        -inset[1],
        inset[1],
    )
    upper_lateral_lower = jnp.stack((-inset[0], lateral_limit))
    upper_lateral_upper = inset
    lower_lateral_lower = -inset
    lower_lateral_upper = jnp.stack((inset[0], -lateral_limit))
    zero_clearance = jnp.zeros_like(distance)
    goal_x_target, goal_x_available = _project_into_box_with_clearance(
        prepared,
        state.ball.position[:2],
        zero_clearance,
        goal_x_lower,
        goal_x_upper,
        fallback,
    )
    goal_upper_target, goal_upper_available = _project_into_box_with_clearance(
        prepared,
        state.ball.position[:2],
        zero_clearance,
        upper_lateral_lower,
        upper_lateral_upper,
        fallback,
    )
    goal_lower_target, goal_lower_available = _project_into_box_with_clearance(
        prepared,
        state.ball.position[:2],
        zero_clearance,
        lower_lateral_lower,
        lower_lateral_upper,
        fallback,
    )
    goal_candidates = jnp.stack(
        (goal_x_target, goal_upper_target, goal_lower_target), axis=0
    )
    goal_available = jnp.stack(
        (goal_x_available, goal_upper_available, goal_lower_available), axis=0
    )
    closest_goal_target, any_goal_available = _select_closest_projection(
        prepared, goal_candidates, goal_available
    )
    # 표준 pitch에서는 기존처럼 goal-line 반대 x축 탈출을 우선한다. 그 box가
    # 존재하지 않는 좁은 pitch에서만 가장 가까운 측면 탈출을 사용한다.
    goal_target = jnp.where(
        goal_x_available[:, None], goal_x_target, closest_goal_target
    )
    prepared = jnp.where(
        ((state.restart.kind == RK_GOALKICK) & inside_penalty & any_goal_available)[
            :, None
        ],
        goal_target,
        prepared,
    )
    penalty = state.restart.kind == RK_PENALTY
    defending_goalkeeper = opponent & roster.is_goalkeeper
    penalty_other = active & (~actor) & (~defending_goalkeeper)
    attack = state.attack_direction[jnp.clip(state.restart.team, TEAM_0, TEAM_1)]
    # IFAB Law 14: kicker/GK 외 선수는 mark 뒤, penalty area 밖,
    # mark에서 9.15 m 밖이어야 한다. 세 legal rectangle과 clearance
    # circle을 동시에 만족하는 fixed 후보 중 최소 이동을 선택한다.
    mark_progress_limit = state.ball.position[0] * attack - radius
    penalty_x_progress_limit = jnp.minimum(mark_progress_limit, area_progress_limit)
    penalty_x_lower, penalty_x_upper = _progress_limited_box(
        inset,
        attack,
        penalty_x_progress_limit,
        -inset[1],
        inset[1],
    )
    penalty_upper_lower, penalty_upper_upper = _progress_limited_box(
        inset,
        attack,
        mark_progress_limit,
        lateral_limit,
        inset[1],
    )
    penalty_lower_lower, penalty_lower_upper = _progress_limited_box(
        inset,
        attack,
        mark_progress_limit,
        -inset[1],
        -lateral_limit,
    )
    penalty_fallback = jnp.stack(
        (
            jnp.full_like(distance, -attack),
            jnp.zeros_like(distance),
        ),
        axis=-1,
    )
    penalty_clearance = config.set_piece_opponent_distance_m + radius
    penalty_x_target, penalty_x_available = _project_into_box_with_clearance(
        prepared,
        state.ball.position[:2],
        penalty_clearance,
        penalty_x_lower,
        penalty_x_upper,
        penalty_fallback,
    )
    penalty_upper_target, penalty_upper_available = _project_into_box_with_clearance(
        prepared,
        state.ball.position[:2],
        penalty_clearance,
        penalty_upper_lower,
        penalty_upper_upper,
        penalty_fallback,
    )
    penalty_lower_target, penalty_lower_available = _project_into_box_with_clearance(
        prepared,
        state.ball.position[:2],
        penalty_clearance,
        penalty_lower_lower,
        penalty_lower_upper,
        penalty_fallback,
    )
    penalty_target, penalty_available = _select_closest_projection(
        prepared,
        jnp.stack(
            (penalty_x_target, penalty_upper_target, penalty_lower_target), axis=0
        ),
        jnp.stack(
            (
                penalty_x_available,
                penalty_upper_available,
                penalty_lower_available,
            ),
            axis=0,
        ),
    )
    prepared = jnp.where(
        (penalty & penalty_other & penalty_available)[:, None],
        penalty_target,
        prepared,
    )
    goalkeeper_target = jnp.asarray(
        (
            attack * (field_half_extent[0] - radius),
            jnp.clip(
                prepared[:, 1][jnp.argmax(defending_goalkeeper)],
                -config.goal_half_width_m + radius,
                config.goal_half_width_m - radius,
            ),
        ),
        dtype=dtype,
    )
    prepared = jnp.where(
        (penalty & defending_goalkeeper)[:, None], goalkeeper_target, prepared
    )
    prepared = jnp.where(
        ordinary[:, None],
        jnp.clip(prepared, -bound + radius, bound - radius),
        prepared,
    )
    facing = jnp.where(actor[:, None], forward, state.players.body_forward)
    return prepared, facing


def enforce_restart_countdown_positions(
    state,
    roster,
    proposed_players: PlayerState,
    field_half_extent: jax.Array,
    *,
    positioning_dt_s: float | None = None,
    ball: BallGeometry = _DEFAULT_BALL,
    body: BodyCollision = _DEFAULT_BODY,
    contact: ContactPhysics = _DEFAULT_CONTACT,
    config: RestartPhysics = _DEFAULT_RESTART,
) -> PlayerState:
    """Countdown locomotion 뒤 비키커의 restart 위치 legality를 유지한다.

    Coherent ordinary restart의 준비 window에서만 동작한다. Designated taker와
    합법적인 비키커 이동은 그대로 둔다. ``positioning_dt_s``가 ``None``이면 기존
    administrative projection을 적용하고, transition이 physics ``dt``를 넘기면
    비키커는 roster speed 상한 안에서 legal target을 향해 이동한다.
    """

    players, _ = _enforce_restart_countdown_positions_with_status(
        state,
        roster,
        proposed_players,
        field_half_extent,
        positioning_dt_s=positioning_dt_s,
        ball=ball,
        body=body,
        contact=contact,
        config=config,
    )
    return players


def _enforce_restart_countdown_positions_with_status(
    state,
    roster,
    proposed_players: PlayerState,
    field_half_extent: jax.Array,
    *,
    positioning_dt_s: float | None,
    ball: BallGeometry = _DEFAULT_BALL,
    body: BodyCollision = _DEFAULT_BODY,
    contact: ContactPhysics = _DEFAULT_CONTACT,
    config: RestartPhysics = _DEFAULT_RESTART,
) -> tuple[PlayerState, jax.Array]:
    """위치 보정과 보정 후 ordinary legality를 같은 projection에서 반환한다."""

    if positioning_dt_s is not None:
        positioning_dt_s = require_float32_representable(
            "positioning_dt_s", positioning_dt_s
        )
        if positioning_dt_s <= 0.0:
            raise ValueError("positioning_dt_s는 0보다 커야 한다")

    actor = restart_actor_mask(state, roster)
    maintain = (
        restart_active(state.restart.kind)
        & (state.restart.kind != RK_GK_HOLD)
        & state.restart_layout_ready
        & jnp.any(actor)
    )
    safe_taker = jnp.clip(
        state.restart.taker, 0, proposed_players.position.shape[0] - 1
    )

    def project(players: PlayerState) -> tuple[PlayerState, jax.Array]:
        def target_for(candidate_players: PlayerState):
            candidate = state._replace(players=candidate_players)
            if config.role_aware_positioning:
                position, _ = _project_restart_layout(
                    candidate,
                    roster,
                    actor,
                    jnp.asarray(field_half_extent, dtype=players.position.dtype),
                    ball=ball,
                    body=body,
                    contact=contact,
                    config=config,
                )
            else:
                position, _ = _project_restart_constraints(
                    candidate,
                    roster,
                    actor,
                    candidate_players.position[safe_taker],
                    candidate_players.body_forward[safe_taker],
                    jnp.asarray(field_half_extent, dtype=players.position.dtype),
                    ball=ball,
                    body=body,
                    contact=contact,
                    config=config,
                )
            return position

        subject = players.active & (~actor)
        if positioning_dt_s is None:
            proposed_target = target_for(players)
            proposed_illegal = subject & jnp.any(
                proposed_target != players.position, axis=-1
            )
            corrected = proposed_illegal
            next_position = proposed_target
            next_velocity = jnp.zeros_like(players.velocity)
            positioned = jnp.bool_(True)
        else:
            # Policy locomotion이 만든 proposed pose를 한 번만 projection한다.
            # 합법인 비키커 이동은 그대로 허용하고, 위반 row만 substep 시작
            # pose에서 가장 가까운 legal target을 향해 speed-bounded 보정한다.
            target = target_for(players)
            proposed_illegal = subject & jnp.any(
                target != players.position, axis=-1
            )
            delta = target - state.players.position
            distance = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + _DIV_EPS)
            maximum_step = roster.max_speed * jnp.asarray(
                positioning_dt_s, dtype=players.position.dtype
            )
            fraction = jnp.minimum(1.0, maximum_step / distance)
            corrected_position = state.players.position + fraction[:, None] * delta
            travel_velocity = (
                corrected_position - state.players.position
            ) / jnp.asarray(positioning_dt_s, dtype=players.velocity.dtype)
            corrected_velocity = jnp.where(
                (
                    (~proposed_illegal)
                    | (distance <= maximum_step + _GEOMETRY_EPS)
                )[:, None],
                jnp.zeros_like(players.velocity),
                travel_velocity,
            )
            corrected = proposed_illegal
            next_position = corrected_position
            next_velocity = corrected_velocity
            positioned = jnp.all(
                (~subject)
                | (~proposed_illegal)
                | (distance <= maximum_step + _GEOMETRY_EPS)
            )
        corrected_players = players._replace(
            position=jnp.where(
                corrected[:, None], next_position, players.position
            ),
            velocity=jnp.where(
                corrected[:, None], next_velocity, players.velocity
            ),
        )
        ordinary = state.restart.kind != RK_KICKOFF
        return corrected_players, (~ordinary) | positioned

    return jax.lax.cond(
        maintain,
        project,
        lambda players: (players, jnp.bool_(True)),
        proposed_players,
    )


def ordinary_restart_non_takers_positioned(
    state,
    roster,
    field_half_extent: jax.Array,
    *,
    ball: BallGeometry = _DEFAULT_BALL,
    body: BodyCollision = _DEFAULT_BODY,
    contact: ContactPhysics = _DEFAULT_CONTACT,
    config: RestartPhysics = _DEFAULT_RESTART,
) -> jax.Array:
    """Ordinary restart의 active 비키커가 현재 legal layout에 있는지 반환한다."""

    actor = restart_actor_mask(state, roster)
    ordinary = (
        restart_active(state.restart.kind)
        & (state.restart.kind != RK_KICKOFF)
        & (state.restart.kind != RK_GK_HOLD)
        & state.restart_layout_ready
        & jnp.any(actor)
    )

    def evaluate(_):
        safe_taker = jnp.clip(
            state.restart.taker, 0, state.players.position.shape[0] - 1
        )
        if config.role_aware_positioning:
            target, _ = _project_restart_layout(
                state,
                roster,
                actor,
                jnp.asarray(field_half_extent, dtype=state.players.position.dtype),
                ball=ball,
                body=body,
                contact=contact,
                config=config,
            )
        else:
            target, _ = _project_restart_constraints(
                state,
                roster,
                actor,
                state.players.position[safe_taker],
                state.players.body_forward[safe_taker],
                jnp.asarray(field_half_extent, dtype=state.players.position.dtype),
                ball=ball,
                body=body,
                contact=contact,
                config=config,
            )
        subject = state.players.active & (~actor)
        return jnp.all(
            (~subject)
            | jnp.all(
                jnp.abs(target - state.players.position) <= _GEOMETRY_EPS,
                axis=-1,
            )
        )

    return jax.lax.cond(ordinary, evaluate, lambda _: jnp.bool_(True), operand=None)


def _restart_layouts_valid(state, layouts, actor, target, field_half_extent, *, body):
    """후보축을 한 번에 펼쳐 공통 mask로 fixed layout들을 audit한다."""
    active = state.players.active
    count = layouts.shape[1]
    diameter = jnp.asarray(body.effective_diameter_m, dtype=layouts.dtype)
    delta = layouts[:, :, None, :] - layouts[:, None, :, :]
    distance_squared = jnp.sum(delta * delta, axis=-1)
    pair = active[:, None] & active[None, :] & (~jnp.eye(count, dtype=jnp.bool_))
    separated = jnp.all(
        (~pair[None, :, :]) | (distance_squared >= diameter * diameter),
        axis=(1, 2),
    )
    radius = 0.5 * diameter
    extent = jnp.asarray(field_half_extent, dtype=layouts.dtype)
    ordinary = active & (~actor)
    inside = jnp.all(
        (~ordinary[None, :])
        | (
            (jnp.abs(layouts[:, :, 0]) <= extent[0] - radius + _GEOMETRY_EPS)
            & (jnp.abs(layouts[:, :, 1]) <= extent[1] - radius + _GEOMETRY_EPS)
        ),
        axis=1,
    )
    actor_error = jnp.max(jnp.abs(layouts - target[None, None, :]), axis=-1)
    actor_valid = jnp.all(
        (~actor[None, :]) | (actor_error <= 8.0 * _GEOMETRY_EPS), axis=1
    )
    return separated & inside & actor_valid


def _project_restart_layout(
    state,
    roster,
    actor,
    field_half_extent,
    *,
    ball,
    body,
    contact,
    config,
):
    """세 fixed candidate 중 legal displacement가 최소인 layout을 선택한다."""
    formation_anchor = formation_positions(
        roster,
        state.attack_direction,
        field_half_extent=field_half_extent,
    )
    kickoff = state.restart.kind == RK_KICKOFF
    kickoff_subject = kickoff & state.players.active
    # Kickoff는 직전 open-play pose와 무관한 opening shape에서 chirality와 legal
    # release pose를 함께 계산한다. 득점 직후 한쪽 box에 몰린 선수 위치가 taker의
    # lateral stance나 다음 kickoff layout을 바꾸지 않게 한다.
    pose_position = jnp.where(
        kickoff_subject[:, None], formation_anchor, state.players.position
    )
    pose_state = state._replace(
        players=state.players._replace(position=pose_position)
    )
    taker_target, release_forward = restart_taker_release_pose(
        pose_state, roster, ball=ball, body=body
    )
    if config.role_aware_positioning:
        direction = state.attack_direction[roster.team_id]
        role = roster.role
        advance_weight = jnp.where(
            role == ROLE_GOALKEEPER,
            0.0,
            jnp.where(
                role == ROLE_CENTRE_BACK,
                0.20,
                jnp.where(
                    role == ROLE_FULL_BACK,
                    0.30,
                    jnp.where(
                        (role == ROLE_CENTRE_FORWARD) | (role == ROLE_WIDE_FORWARD),
                        0.55,
                        0.40,
                    ),
                ),
            ),
        ).astype(state.players.position.dtype)
        ball_progress = state.ball.position[0] * direction
        longitudinal_shift = jnp.clip(
            ball_progress * advance_weight,
            -jnp.asarray(field_half_extent[0], dtype=formation_anchor.dtype)
            * (24.0 / 52.5),
            jnp.asarray(field_half_extent[0], dtype=formation_anchor.dtype)
            * (28.0 / 52.5),
        )
        role_target = formation_anchor.at[:, 0].add(direction * longitudinal_shift)
        tactical_kind = (
            (state.restart.kind == RK_KICKOFF)
            | (state.restart.kind == RK_THROWIN)
            | (state.restart.kind == RK_GOALKICK)
            | (state.restart.kind == RK_CORNER)
            | (state.restart.kind == RK_FREEKICK)
            | (state.restart.kind == RK_OFFSIDE)
        )
        role_target = jnp.where(
            (tactical_kind & state.players.active & (~actor))[:, None],
            role_target,
            state.players.position,
        )
    else:
        role_target = jnp.where(
            (kickoff & state.players.active & (~actor))[:, None],
            formation_anchor,
            state.players.position,
        )
    role_state = state._replace(players=state.players._replace(position=role_target))
    base, facing = _project_restart_constraints(
        role_state,
        roster,
        actor,
        taker_target,
        release_forward,
        field_half_extent,
        ball=ball,
        body=body,
        contact=contact,
        config=config,
    )
    active_movable = state.players.active & (~actor)
    rank = jnp.cumsum(active_movable.astype(jnp.int32)) - jnp.int32(1)
    dtype = base.dtype
    spacing = jnp.asarray(1.25 * body.effective_diameter_m, dtype=dtype)
    relative = base - state.ball.position[:2]
    norm = jnp.sqrt(jnp.sum(relative * relative, axis=-1) + _DIV_EPS)
    radial = relative / norm[:, None]
    tangent = jnp.stack((-radial[:, 1], radial[:, 0]), axis=-1)
    local_sign = jnp.where((rank % 2) == 0, 1.0, -1.0).astype(dtype)
    local_step = ((rank + 2) // 2).astype(dtype)
    local_raw = jnp.where(
        active_movable[:, None],
        base + (spacing * local_sign * local_step)[:, None] * tangent,
        base,
    )
    local_state = state._replace(players=state.players._replace(position=local_raw))
    local, _ = _project_restart_constraints(
        local_state,
        roster,
        actor,
        taker_target,
        release_forward,
        field_half_extent,
        ball=ball,
        body=body,
        contact=contact,
        config=config,
    )
    slot = jnp.maximum(rank, 0).astype(dtype)
    angle = 2.0 * jnp.pi * slot / jnp.maximum(jnp.sum(active_movable), 1)
    ring_radius = config.set_piece_opponent_distance_m + spacing * (
        1.0 + (jnp.maximum(rank, 0) // 12).astype(dtype)
    )
    ring_raw = state.ball.position[:2] + ring_radius[:, None] * jnp.stack(
        (jnp.cos(angle), jnp.sin(angle)), axis=-1
    )
    ring_raw = jnp.where(active_movable[:, None], ring_raw, base)
    ring_state = state._replace(players=state.players._replace(position=ring_raw))
    ring, _ = _project_restart_constraints(
        ring_state,
        roster,
        actor,
        taker_target,
        release_forward,
        field_half_extent,
        ball=ball,
        body=body,
        contact=contact,
        config=config,
    )
    layouts = jnp.stack((base, local, ring), axis=0)
    valid = _restart_layouts_valid(
        state, layouts, actor, taker_target, field_half_extent, body=body
    )
    cost = jnp.sum((layouts - state.players.position[None, :, :]) ** 2, axis=(1, 2))
    choice = jnp.argmin(jnp.where(valid, cost, jnp.inf))
    selected = jnp.where(jnp.any(valid), layouts[choice], base)
    if config.face_ball_after_positioning:
        toward_ball = state.ball.position[:2] - selected
        distance_squared = jnp.sum(toward_ball * toward_ball, axis=-1)
        inverse_distance = jax.lax.rsqrt(jnp.maximum(distance_squared, _DIV_EPS))
        ball_facing = jnp.where(
            (distance_squared > _GEOMETRY_EPS * _GEOMETRY_EPS)[:, None],
            toward_ball * inverse_distance[:, None],
            release_forward,
        )
        facing = jnp.where(
            state.players.active[:, None], ball_facing, state.players.body_forward
        )
    return selected, facing


def prepare_restart(
    state,
    roster,
    field_half_extent: jax.Array,
    *,
    timebase: Timebase = DEFAULT_TIMEBASE,
    ball: BallGeometry = _DEFAULT_BALL,
    body: BodyCollision = _DEFAULT_BODY,
    contact: ContactPhysics = _DEFAULT_CONTACT,
    config: RestartPhysics = _DEFAULT_RESTART,
    defer_non_taker_positioning: bool = False,
) -> RestartPreparation:
    """아직 준비되지 않은 restart를 한 번 fixed-shape legal layout으로 투영한다."""

    if not isinstance(defer_non_taker_positioning, bool):
        raise TypeError("defer_non_taker_positioning은 bool이어야 한다")
    actor = restart_actor_mask(state, roster)
    should_prepare = restart_active(state.restart.kind) & (~state.restart_layout_ready)
    prepared = should_prepare & jnp.any(actor)
    position, facing = _project_restart_layout(
        state,
        roster,
        actor,
        jnp.asarray(field_half_extent, dtype=state.players.position.dtype),
        ball=ball,
        body=body,
        contact=contact,
        config=config,
    )
    delay = jnp.where(
        state.restart.kind == RK_KICKOFF,
        jnp.int32(0),
        jnp.int32(forced_release_delay_substeps(timebase=timebase, config=config)),
    )
    continuous_approach = continuous_restart_approach_enabled(
        state.restart.kind, config=config
    )
    approaching_actor = actor & continuous_approach
    deferred_non_taker = (
        jnp.bool_(defer_non_taker_positioning)
        & (state.restart.kind != RK_KICKOFF)
        & (state.restart.kind != RK_GK_HOLD)
        & state.players.active
        & (~actor)
    )
    candidate = state._replace(
        players=state.players._replace(
            # Production transition은 ordinary 비키커의 legal target만 계산하고
            # 현재 위치에서 speed-bounded positioning을 시작한다. Direct helper의
            # 기본 administrative mode와 kickoff만 target에 즉시 배치한다.
            position=jnp.where(
                (approaching_actor | deferred_non_taker)[:, None],
                state.players.position,
                position,
            ),
            velocity=jnp.zeros_like(state.players.velocity),
            body_forward=jnp.where(
                approaching_actor[:, None], state.players.body_forward, facing
            ),
        ),
        restart=state.restart._replace(substeps_remaining=delay),
        restart_layout_ready=jnp.bool_(True),
        restart_taker_approach=continuous_approach,
    )
    next_state = jax.tree.map(
        lambda new, old: jnp.where(prepared, new, old), candidate, state
    )
    return RestartPreparation(next_state, prepared)


def advance_restart_clock(state, enabled: jax.Array) -> object:
    """준비되어 공개된 ordinary restart의 deadline을 한 substep 전진한다."""

    active = restart_active(state.restart.kind) & state.restart_layout_ready
    remaining = jnp.where(
        active & enabled,
        jnp.maximum(state.restart.substeps_remaining - jnp.int32(1), 0),
        state.restart.substeps_remaining,
    ).astype(jnp.int32)
    return state._replace(restart=state.restart._replace(substeps_remaining=remaining))


def release_restart(
    state,
    roster,
    action: PhysicsAction,
    visible: jax.Array,
    *,
    deadline_due: jax.Array = False,
    ball: BallGeometry = _DEFAULT_BALL,
    contact: ContactPhysics = _DEFAULT_CONTACT,
    config: RestartPhysics = _DEFAULT_RESTART,
) -> RestartRelease:
    """정책 release 또는 deadline fallback을 contest 없이 결정론적으로 적용한다."""

    actor_mask = restart_actor_mask(state, roster)
    actor = jnp.argmax(actor_mask).astype(jnp.int32)
    coherent = jnp.any(actor_mask)
    intent = action.requested_intent[actor]
    pass_or_shoot = (intent == INTENT_PASS) | (intent == INTENT_SHOOT)
    pass_only = (state.restart.kind == RK_THROWIN) | state.restart.indirect
    legal_intent = pass_or_shoot & ((~pass_only) | (intent == INTENT_PASS))
    requested = legal_intent & (action.force_power[actor] > _GEOMETRY_EPS)
    due = (state.restart.substeps_remaining <= 0) | jnp.asarray(
        deadline_due, dtype=jnp.bool_
    )
    occurred = visible & coherent & due
    forced = occurred & (~requested)
    forward = _release_forward(state, roster)
    projected_direction = _unit(action.force_direction[actor], forward)
    into_field = jnp.dot(projected_direction, forward)
    projected_direction = jnp.where(
        (state.restart.kind == RK_THROWIN) & (into_field <= _GEOMETRY_EPS),
        forward,
        projected_direction,
    )
    effective_direction = jnp.where(forced, forward, projected_direction)
    effective_power = jnp.where(
        forced,
        jnp.maximum(action.force_power[actor], config.forced_release_power),
        action.force_power[actor],
    )
    minimum_power = contact.restart_min_ball_speed_mps / contact.kick_speed_max_mps
    effective_power = jnp.maximum(effective_power, minimum_power)
    effective_intent = jnp.where(forced, INTENT_PASS, intent).astype(jnp.int32)
    effective = action._replace(
        contact=action.contact.at[actor].set(True),
        force_direction=action.force_direction.at[actor].set(effective_direction),
        force_power=action.force_power.at[actor].set(effective_power),
        requested_intent=action.requested_intent.at[actor].set(effective_intent),
    )
    live_ball = state.ball._replace(live=jnp.bool_(True))
    actuation = apply_foot_actuation(
        live_ball,
        state.players,
        effective,
        actor,
        ball_geometry=ball,
        config=contact,
    )
    mechanism = jnp.where(
        state.restart.kind == RK_THROWIN, MECHANISM_THROW, MECHANISM_FOOT
    )
    throw_speed = jnp.maximum(
        effective_power * contact.throw_speed_max_mps,
        contact.restart_min_ball_speed_mps,
    )
    launch = effective.launch[actor]
    throw_velocity = throw_speed * jnp.asarray(
        (
            jnp.cos(launch) * effective_direction[0],
            jnp.cos(launch) * effective_direction[1],
            jnp.sin(launch),
        ),
        dtype=state.ball.velocity.dtype,
    )
    throw_position = state.ball.position.at[2].set(
        roster.height[actor] * contact.torso_top_height_factor
        + contact.throw_release_height_addition_m
    )
    released_ball = state.ball._replace(
        position=jnp.where(
            mechanism == MECHANISM_THROW, throw_position, state.ball.position
        ),
        velocity=jnp.where(
            mechanism == MECHANISM_THROW,
            throw_velocity,
            actuation.ball.velocity,
        ),
        spin=jnp.where(
            mechanism == MECHANISM_THROW,
            jnp.zeros_like(state.ball.spin),
            actuation.ball.spin,
        ),
        live=jnp.bool_(True),
    )
    direct_exempt = (
        (state.restart.kind == RK_THROWIN)
        | (state.restart.kind == RK_GOALKICK)
        | (state.restart.kind == RK_CORNER)
    )
    result = ContactResult(
        actor=actor,
        mechanism=mechanism.astype(jnp.int32),
        intent=effective_intent,
        outcome=jnp.int32(OUTCOME_RELEASE),
        restart_kind=state.restart.kind,
        law11_effect=jnp.where(
            direct_exempt,
            LAW11_DIRECT_RESTART_EXEMPTION,
            LAW11_DELIBERATE_PLAY_RESET,
        ).astype(jnp.int32),
        kick_applied=(mechanism == MECHANISM_FOOT),
        intent_source=jnp.where(
            forced, INTENT_SOURCE_ENVIRONMENT_FORCED, INTENT_SOURCE_POLICY
        ).astype(jnp.int32),
    )
    candidate = state._replace(
        ball=released_ball,
        possession=state.possession._replace(
            team=jnp.int32(NO_TEAM),
            player=jnp.int32(NO_PLAYER),
            previous_team=state.possession.team,
            control_ticks=jnp.int32(0),
            last_contact=result,
        ),
        restart=state.restart._replace(
            kind=jnp.int32(RK_NONE),
            team=jnp.int32(NO_TEAM),
            substeps_remaining=jnp.int32(0),
            taker=jnp.int32(NO_PLAYER),
            indirect=jnp.bool_(False),
            opened_control_tick=jnp.int32(-1),
        ),
        restart_release=RestartReleaseProvenance(
            active=jnp.bool_(True),
            untouched=jnp.bool_(True),
            kind=state.restart.kind,
            team=state.restart.team,
            taker=actor,
            indirect=state.restart.indirect,
            law11_direct_exempt=direct_exempt,
            release_mechanism=mechanism.astype(jnp.int32),
        ),
        restart_layout_ready=jnp.bool_(False),
        restart_taker_approach=jnp.bool_(False),
        penalty_completion_active=jnp.where(
            state.restart.kind == RK_PENALTY,
            jnp.bool_(True),
            state.penalty_completion_active,
        ),
        penalty_completion_team=jnp.where(
            state.restart.kind == RK_PENALTY,
            state.restart.team,
            state.penalty_completion_team,
        ).astype(jnp.int32),
    )
    next_state = jax.tree.map(
        lambda new, old: jnp.where(occurred, new, old), candidate, state
    )
    return RestartRelease(
        state=next_state,
        occurred=occurred,
        forced=forced,
        actor=jnp.where(occurred, actor, NO_PLAYER).astype(jnp.int32),
    )


__all__ = [
    "RestartPhysics",
    "RestartPreparation",
    "RestartRelease",
    "RestartTakerRepair",
    "advance_restart_clock",
    "continuous_restart_approach_enabled",
    "enforce_restart_countdown_positions",
    "ordinary_restart_non_takers_positioned",
    "forced_release_delay_substeps",
    "prepare_restart",
    "release_restart",
    "repair_restart_taker",
    "restart_active",
    "restart_actor_mask",
    "restart_taker_release_pose",
]
