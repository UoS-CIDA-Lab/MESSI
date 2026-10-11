"""goal post와 crossbar의 swept capsule collision."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.core.ball import BallPhysics
from messi.core.geometry import BallGeometry
from messi.core.numeric import require_float32_representable

WOODWORK_NONE = 0
WOODWORK_POST = 1
WOODWORK_CROSSBAR = 2


class GoalFrameEvent(NamedTuple):
    occurred: jax.Array
    kind: jax.Array
    time_fraction: jax.Array
    position: jax.Array
    normal: jax.Array


@dataclass(frozen=True, slots=True)
class GoalFramePhysics:
    frame_radius_m: float = 0.06
    # FootballWorld의 goal-frame 계수와 같은 값이다. 기존 0.55/0.72는 tangential
    # retention 단순 반사용 계수였고 근거 기록이 없었다.
    restitution: float = 0.68
    friction: float = 0.35

    def __post_init__(self) -> None:
        radius = require_float32_representable(
            "frame_radius_m", self.frame_radius_m
        )
        if not 0.01 <= radius <= 0.06:
            raise ValueError("frame_radius_m은 [0.01, 0.06]m 범위여야 한다")
        for name in ("restitution", "friction"):
            value = require_float32_representable(name, getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}은 [0, 1] 범위여야 한다")


DEFAULT_GOAL_FRAME_PHYSICS = GoalFramePhysics()
_DEFAULT_BALL_GEOMETRY = BallGeometry()
_DEFAULT_BALL_PHYSICS = BallPhysics()


def empty_goal_frame_event(dtype=jnp.float32) -> GoalFrameEvent:
    return GoalFrameEvent(
        jnp.bool_(False),
        jnp.int32(WOODWORK_NONE),
        jnp.asarray(0.0, dtype),
        jnp.zeros(3, dtype),
        jnp.zeros(3, dtype),
    )


def _swept_entry(origin, delta, centers, radius):
    """[entry, exit] 구간으로 swept 접촉을 푼다.

    최소근만 보고 ``fraction >= 0``을 요구하면 substep 시작 시 이미 capsule과 겹친
    공이 영구히 미검출된다. 겹친 상태는 ``entry < 0 <= exit``이므로 구간 교차로
    판정하고, 움직이지 않는 공은 중심 거리로 직접 본다. 반환하는 마지막 값은
    radial 도함수의 두 배로, 접촉 직후의 egress 상태를 거르는 데 쓴다.
    """

    offset = origin[None, :] - centers
    quadratic = jnp.sum(delta * delta)
    moving = quadratic > 1.0e-12
    linear = 2.0 * jnp.sum(offset * delta[None, :], axis=-1)
    distance_squared = jnp.sum(offset * offset, axis=-1)
    constant = distance_squared - radius * radius
    disc = linear * linear - 4.0 * quadratic * constant
    root = jnp.sqrt(jnp.maximum(disc, 0.0))
    denominator = jnp.where(moving, 2.0 * quadratic, 1.0)
    entry = jnp.where(moving, (-linear - root) / denominator, 0.0)
    exit_ = jnp.where(moving, (-linear + root) / denominator, 1.0)
    valid = jnp.where(
        moving,
        (disc > 0.0) & (exit_ >= 0.0) & (entry <= 1.0),
        distance_squared < radius * radius,
    )
    lower = jnp.maximum(entry, 0.0)
    upper = jnp.minimum(exit_, 1.0)
    hit = valid & (lower <= upper)
    fraction = jnp.clip(lower, 0.0, 1.0)
    at = offset + fraction[:, None] * delta[None, :]
    return hit, fraction, at, linear


def detect_goal_frame_contact(
    position,
    path_delta,
    ball_live,
    *,
    half_length=52.5,
    goal_half_width=3.66,
    goal_height=2.44,
    ball_radius=0.11,
    physics=DEFAULT_GOAL_FRAME_PHYSICS,
) -> GoalFrameEvent:
    radius = ball_radius + physics.frame_radius_m
    # IFAB는 goal 폭을 post 안쪽 면 사이로, 높이를 crossbar 아래 면까지로 정의한다.
    # 따라서 frame의 축은 전달받은 내측 치수보다 frame 반지름만큼 바깥에 있어야
    # 한다. 축을 내측 치수에 두면 개구부가 frame 반지름만큼 잠식돼 boundary의 득점
    # 판정(0.5*width - ball_radius)과 어긋난다.
    post_axis_y = goal_half_width + physics.frame_radius_m
    bar_axis_z = goal_height + physics.frame_radius_m
    post_centers = jnp.asarray(
        (
            (-half_length, -post_axis_y),
            (-half_length, post_axis_y),
            (half_length, -post_axis_y),
            (half_length, post_axis_y),
        ),
        position.dtype,
    )
    post_hit, post_t, post_offset, post_radial = _swept_entry(
        position[:2], path_delta[:2], post_centers, radius
    )
    post_z = position[2] + post_t * path_delta[2]
    post_hit &= (post_z >= 0.0) & (post_z <= bar_axis_z)
    post_normal = jnp.concatenate(
        (post_offset, jnp.zeros((4, 1), position.dtype)), axis=-1
    )
    bar_centers = jnp.asarray(
        ((-half_length, bar_axis_z), (half_length, bar_axis_z)), position.dtype
    )
    bar_hit, bar_t, bar_offset, bar_radial = _swept_entry(
        position[jnp.asarray((0, 2))],
        path_delta[jnp.asarray((0, 2))],
        bar_centers,
        radius,
    )
    bar_y = position[1] + bar_t * path_delta[1]
    bar_hit &= jnp.abs(bar_y) <= post_axis_y
    bar_normal = jnp.stack(
        (bar_offset[:, 0], jnp.zeros(2, position.dtype), bar_offset[:, 1]), axis=-1
    )
    # Post와 crossbar 축이 만나는 네 모서리는 어느 capsule에도 속하지 않아 그대로
    # 두면 공이 관통한다. 같은 반지름의 구 네 개로 접합부를 메운다.
    corner_centers = jnp.asarray(
        (
            (-half_length, -post_axis_y, bar_axis_z),
            (-half_length, post_axis_y, bar_axis_z),
            (half_length, -post_axis_y, bar_axis_z),
            (half_length, post_axis_y, bar_axis_z),
        ),
        position.dtype,
    )
    corner_hit, corner_t, corner_offset, corner_radial = _swept_entry(
        position, path_delta, corner_centers, radius
    )
    hit = jnp.concatenate((post_hit, bar_hit, corner_hit)) & jnp.asarray(ball_live)
    fraction = jnp.concatenate((post_t, bar_t, corner_t))
    normals = jnp.concatenate((post_normal, bar_normal, corner_offset), axis=0)
    kinds = jnp.asarray((1, 1, 1, 1, 2, 2, 1, 1, 1, 1), jnp.int32)
    # 실제 충돌 직후 float32 반올림으로 중심이 표면에 남을 수 있다. 그것은 egress이지
    # 새 충돌이 아니므로 자기 표면에서 밖으로 나가는 time-zero 후보만 버린다.
    radial = jnp.concatenate((post_radial, bar_radial, corner_radial))
    hit &= ~((fraction == 0.0) & (radial > 0.0))
    winner = jnp.argmin(jnp.where(hit, fraction, jnp.inf))
    occurred = jnp.any(hit)
    normal = normals[winner]
    # 재구성된 상태가 축 위에서 정확히 시작하면 normal이 0이 된다. 그때는 진행
    # 반대 방향을 사용한다.
    reverse = -path_delta
    axis_normals = jnp.concatenate(
        (
            jnp.broadcast_to(
                jnp.stack((reverse[0], reverse[1], jnp.zeros((), position.dtype))),
                (4, 3),
            ),
            jnp.broadcast_to(
                jnp.stack((reverse[0], jnp.zeros((), position.dtype), reverse[2])),
                (2, 3),
            ),
            jnp.broadcast_to(reverse, (4, 3)),
        ),
        axis=0,
    )
    degenerate = jnp.linalg.norm(normal) <= 1.0e-6
    normal = jnp.where(degenerate, axis_normals[winner], normal)
    normal /= jnp.maximum(jnp.linalg.norm(normal), 1.0e-6)
    contact = position + fraction[winner] * path_delta
    return GoalFrameEvent(
        occurred,
        jnp.where(occurred, kinds[winner], WOODWORK_NONE),
        jnp.where(occurred, fraction[winner], 0.0),
        jnp.where(occurred, contact, jnp.zeros_like(contact)),
        jnp.where(occurred, normal, jnp.zeros_like(normal)),
    )


def apply_goal_frame_contact(
    ball,
    event,
    *,
    physics=DEFAULT_GOAL_FRAME_PHYSICS,
    ball_geometry=_DEFAULT_BALL_GEOMETRY,
    ball_physics=_DEFAULT_BALL_PHYSICS,
):
    """Frame impulse를 Coulomb 마찰로 풀어 spin과 속도를 함께 갱신한다.

    기존 구현은 ``retention * tangent - restitution * normal`` 단순 반사에 spin을
    같은 비율로 줄이기만 해서, spin 먹은 슛이 post를 맞고 방향이 바뀌는 거동이
    없었다. 지면 bounce(:mod:`messi.core.ball`)가 이미 쓰는 것과 같은
    sticking/sliding impulse로 맞춘다.
    """

    normal = event.normal
    velocity = ball.velocity
    spin = ball.spin
    radius = jnp.asarray(ball_geometry.radius_m, dtype=velocity.dtype)
    inertia = jnp.asarray(ball_physics.inertia_ratio, dtype=velocity.dtype)
    restitution = jnp.asarray(physics.restitution, dtype=velocity.dtype)
    friction = jnp.asarray(physics.friction, dtype=velocity.dtype)

    normal_speed = jnp.dot(velocity, normal)
    # 이미 frame에서 멀어지는 중이면 impulse를 다시 주지 않는다.
    approaching = event.occurred & (normal_speed < 0.0)
    tangential = velocity - normal_speed * normal
    slip = tangential - radius * jnp.cross(spin, normal)
    slip = slip - jnp.dot(slip, normal) * normal
    slip_speed = jnp.linalg.norm(slip)
    slip_direction = slip / jnp.maximum(slip_speed, 1.0e-9)

    normal_impulse = (1.0 + restitution) * jnp.maximum(-normal_speed, 0.0)
    sticking_impulse = (inertia / (1.0 + inertia)) * slip_speed
    tangential_impulse = jnp.minimum(sticking_impulse, friction * normal_impulse)
    tangential_delta = -tangential_impulse * slip_direction
    candidate_velocity = (
        tangential + tangential_delta - restitution * normal_speed * normal
    )
    candidate_spin = spin - jnp.cross(normal, tangential_delta) / jnp.maximum(
        inertia * radius, 1.0e-9
    )

    # Impulse 근사가 에너지를 늘리지 않도록 보정한다.
    rotational = inertia * radius * radius
    incoming = jnp.sum(velocity * velocity) + rotational * jnp.sum(spin * spin)
    outgoing = jnp.sum(candidate_velocity * candidate_velocity) + (
        rotational * jnp.sum(candidate_spin * candidate_spin)
    )
    scale = jnp.sqrt(jnp.minimum(1.0, incoming / (outgoing + 1.0e-9)))
    candidate_velocity = candidate_velocity * scale
    candidate_spin = candidate_spin * scale

    position = event.position + (1.0e-4 * normal)
    return ball._replace(
        position=jnp.where(event.occurred, position, ball.position),
        velocity=jnp.where(approaching, candidate_velocity, velocity),
        spin=jnp.where(approaching, candidate_spin, spin),
    )
