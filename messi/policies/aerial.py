"""정책별 취향과 분리된 고정-shape 공중 PASS 경로 근사.

동일 MESSI 공중 force와 17-step trapezoidal 위치 근사를 사용한다. 실제 environment
transition이나 정확한 interception solver가 아니며 성공 확률로 해석하지 않는다.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.core.ball import BallPhysics, advance_ball_smooth
from messi.core.contact import ContactPhysics
from messi.core.geometry import BallGeometry
from messi.core.passive_contact import PassiveContactPhysics
from messi.core.state import BallState

_STEPS = 17


class AerialRoute(NamedTuple):
    """선수 후보축 [N]의 bounded score, 가능 여부와 예상 접촉 시간(s).

관측된 경로에서 접촉을 찾지 못하면 시간은 +inf다. 공중 경로 실패는 score=0이며
그 원인이 상위 eligibility, 몸 충돌 또는 도착 우위인지 다른 field와 함께 해석한다.
delivery_feasible은 입력·몸 충돌·경계·수신 도달을 확인하되 상대보다 먼저 닿을
요건은 적용하지 않는다. viable은 그 요건까지 포함한 안전 경로다.
"""

    score: jax.Array
    viable: jax.Array
    delivery_feasible: jax.Array
    receiver_time_s: jax.Array
    opponent_time_s: jax.Array
    body_blocked: jax.Array
    teammate_body_blocked: jax.Array
    boundary_before_receiver: jax.Array


def aerial_route(
    *, origin, incoming_velocity, incoming_spin, position, height, reach_height,
    max_speed, opponent, eligible, power, launch_angle, intended_direction, field_half_extent,
    ball_physics: BallPhysics = BallPhysics(),
    contact_physics: ContactPhysics = ContactPhysics(),
    ball_geometry: BallGeometry = BallGeometry(), opponent_contact_radius=None,
    goalkeeper=None, opponent_penalty_x_min=None, penalty_half_width=None,
    nonreceiver_teammate=None, passive_physics: PassiveContactPhysics = PassiveContactPhysics(),
) -> AerialRoute:
    """한 passer 기준 SI/team-frame의 [N] 후보 경로를 보수적으로 비교한다.

origin/velocity/spin [3], position/direction [N,2], 나머지 선수값 [N]. Power와 launch
angle(rad)는 caller가 actuator와 동일하게 계산해 전달한다. Spin action은 0인 PASS를
대상으로 현재 spin을 보존한다. Active opponent와 onside teammate 등 legality는 caller의
opponent/eligible mask가 담당하며 이 함수가 완화하지 않는다. GK legal hand 영역을 이미
    contact radius [N] 또는 후보별 [N,N]를 선택적으로 전달할 수 있다. goalkeeper [N]를
    제공하면 상대 penalty x 최소값·half width도 필요하며 각 경로점에서만 손 반경을 쓴다.
    nonreceiver_teammate [N]는 carrier를 제외한 활성 동료 mask이며 target 자신은 내부에서
    제외한다. 동료의 미래 max_speed 이동은 가정하지 않고 정적 몸 envelope만 검사한다.
    field_half_extent [2]의 실제 field 크기로 공 전체가 경계를 먼저 넘는 경로도 배제한다.

17-step 수는 static이다. 수신자는 reach에서 0.10m를 제외하고 상대에는 0.15m를 더하며
    최초 도착 0.05s 우위를 요구한다. 이는 경기 검증 중인 근사 오차 여유다. Score는 실측
    성공 확률이 아닌 bounded ranking heuristic이다. 무소유 future
intent/recovery와 가속은 예측하지 않으며 상대의 즉시 max_speed disk는 보수적 envelope다.
비유한 공통 관측은 모든 후보를 fail-closed하고 잘못된 후보별 power/angle/direction은 해당
후보만 배제한다. 비행이 3초를 넘는 조건은 아직 오차 검증 범위를 벗어나므로 배제한다.
"""
    origin, incoming_velocity, incoming_spin, position, height, reach_height, max_speed = (
        jnp.asarray(x, jnp.float32)
        for x in (origin, incoming_velocity, incoming_spin, position, height, reach_height, max_speed)
    )
    power, launch_angle, intended_direction = (
        jnp.asarray(x, jnp.float32) for x in (power, launch_angle, intended_direction)
    )
    opponent, eligible = jnp.asarray(opponent, bool), jnp.asarray(eligible, bool)
    field_half_extent = jnp.asarray(field_half_extent, jnp.float32)
    n = position.shape[0]
    finite = (jnp.all(jnp.isfinite(origin)) & jnp.all(jnp.isfinite(incoming_velocity))
              & jnp.all(jnp.isfinite(incoming_spin)) & jnp.all(jnp.isfinite(position))
              & jnp.all(jnp.isfinite(height)) & jnp.all(jnp.isfinite(reach_height))
              & jnp.all(jnp.isfinite(max_speed)) & jnp.all(height >= 0.)
              & jnp.all(reach_height >= height) & jnp.all(max_speed >= 0.)
              & jnp.all(jnp.isfinite(field_half_extent)) & jnp.all(field_half_extent > 0.))
    candidate_valid = (jnp.isfinite(power) & (power >= 0.) & (power <= 1.)
                       & jnp.isfinite(launch_angle) & (launch_angle > 0.)
                       & (launch_angle <= contact_physics.launch_max_radians)
                       & jnp.all(jnp.isfinite(intended_direction), axis=-1))
    origin, incoming_velocity, incoming_spin, position, height, reach_height, max_speed = (
        jnp.nan_to_num(x, nan=0., posinf=0., neginf=0.)
        for x in (origin, incoming_velocity, incoming_spin, position, height, reach_height, max_speed)
    )
    power, launch_angle, intended_direction = (
        jnp.nan_to_num(x, nan=0., posinf=0., neginf=0.) for x in (power, launch_angle, intended_direction)
    )
    direction_norm = jnp.linalg.norm(intended_direction, axis=-1)
    candidate_valid &= direction_norm > 1.e-6
    direction = intended_direction / jnp.maximum(direction_norm[:, None], 1.e-6)
    impulse = power * contact_physics.kick_speed_max_mps
    release = incoming_velocity + jnp.concatenate(
        (impulse[:, None] * jnp.cos(launch_angle)[:, None] * direction,
         (impulse * jnp.sin(launch_angle))[:, None]), axis=-1)
    limit = jnp.maximum(jnp.linalg.norm(incoming_velocity), contact_physics.kick_speed_max_mps)
    release *= jnp.minimum(1., limit / jnp.maximum(jnp.linalg.norm(release, axis=-1), 1.e-6))[:, None]
    radius = jnp.float32(ball_geometry.radius_m)
    gravity = jnp.float32(ball_physics.gravity_mps2)
    bottom = jnp.maximum(origin[2] - radius, 0.)
    vacuum_time = (release[:, 2] + jnp.sqrt(release[:, 2]**2 + 2 * gravity * bottom)) / gravity
    candidate_valid &= (vacuum_time > .05) & (vacuum_time <= 3.) & (release[:, 2] > 0.)
    dt = jnp.clip(vacuum_time, .05, 3.) / _STEPS
    initial = BallState(jnp.broadcast_to(origin, (n, 3)), release,
                        jnp.broadcast_to(incoming_spin, (n, 3)), jnp.ones(n, bool))

    def step(ball, _):
        smooth = jax.vmap(lambda b, delta: advance_ball_smooth(
            b, dt=delta, geometry=ball_geometry, physics=ball_physics))(ball, dt)
        proposed = ball.position + .5 * dt[:, None] * (ball.velocity + smooth.velocity)
        hit = (proposed[:, 2] <= radius) & (smooth.velocity[:, 2] < 0.) & ball.live
        fraction = jnp.clip((ball.position[:, 2] - radius) / jnp.maximum(ball.position[:, 2] - proposed[:, 2], 1.e-6), 0., 1.)
        hit_position = ball.position + fraction[:, None] * (proposed - ball.position)
        new = BallState(jnp.where(hit[:, None], hit_position, proposed),
                        jnp.where(hit[:, None], 0., smooth.velocity), smooth.spin, ball.live & ~hit)
        new = jax.tree.map(lambda x, old: jnp.where(ball.live.reshape((-1,) + (1,) * (x.ndim - 1)), x, old), new, ball)
        return new, (new, fraction)

    # unroll=4는 단일 GPU B32/H60000에서 warm 처리량을 2.7% 높였지만,
    # CPU B32에서는 3.4% 느리고 4-GPU pmap에서는 이득이 없었다.
    scan_unroll = (
        4
        if jax.default_backend() == "gpu" and jax.local_device_count() == 1
        else 1
    )
    _, (path, fractions) = jax.lax.scan(
        step, initial, None, length=_STEPS, unroll=scan_unroll
    )
    xyz = jnp.concatenate((initial.position[:, None], jnp.swapaxes(path.position, 0, 1)), axis=1)
    live_path = jnp.swapaxes(path.live, 0, 1)
    landed = jnp.any(~live_path, axis=-1)
    impact = jnp.where(landed, jnp.argmax(~live_path, axis=-1) + 1, _STEPS)
    impact_fraction = fractions.T[jnp.arange(n), impact - 1]
    duration = dt * (impact - 1 + jnp.where(landed, impact_fraction, 1.))
    z_path = xyz[..., 2] - radius

    def crossings(levels):
        a, b = z_path[:, None, :-1], z_path[:, None, 1:]
        level = levels[None, :, None]
        indices = jnp.arange(_STEPS)[None, None, :]
        crossed = (a >= level) & (b <= level) & (a > b) & (indices < impact[:, None, None])
        fraction = jnp.clip((a - level) / jnp.maximum(a - b, 1.e-6), 0., 1.)
        segment_duration = jnp.where(indices == impact[:, None, None] - 1,
                                     dt[:, None, None] * jnp.where(landed, impact_fraction, 1.)[:, None, None], dt[:, None, None])
        times = indices * dt[:, None, None] + fraction * segment_duration
        first = jnp.min(jnp.where(crossed, times, jnp.inf), axis=-1)
        return jnp.where(jnp.isfinite(first), first, (jnp.argmax(z_path, axis=-1) * dt)[:, None])

    own_reach = jnp.maximum(reach_height - .10, 0.)
    opponent_reach = reach_height + .15
    own_cross = jnp.diag(crossings(own_reach))
    opponent_cross = crossings(opponent_reach)
    samples = jnp.concatenate((duration[:, None] * jnp.linspace(0., 1., 13)[None],
                               own_cross[:, None], jnp.diag(crossings(height))[:, None]), axis=-1)

    def points(t):
        shape = (n, *((1,) * (t.ndim - 1)))
        step_index = t / dt.reshape(shape)
        before = jnp.clip(jnp.floor(step_index).astype(jnp.int32), 0, _STEPS - 1)
        fraction = jnp.clip(step_index - before, 0., 1.)
        final = before >= (impact - 1).reshape(shape)
        final_fraction = (t - (impact - 1).reshape(shape) * dt.reshape(shape)) / jnp.maximum(
            (duration - (impact - 1) * dt).reshape(shape), 1.e-6)
        fraction = jnp.where(final, jnp.clip(final_fraction, 0., 1.), fraction)
        row = jnp.arange(n).reshape(shape)
        point = xyz[row, before] * (1 - fraction[..., None]) + xyz[row, before + 1] * fraction[..., None]
        return point[..., :2], point[..., 2] - radius

    xy, z = points(samples)
    own_radius = jnp.float32(contact_physics.carry_radius_m) + radius
    contact = jnp.asarray(contact_physics.carry_radius_m if opponent_contact_radius is None else opponent_contact_radius, jnp.float32)
    finite &= jnp.all(jnp.isfinite(contact)) & jnp.all(contact >= 0.)
    contact = jnp.broadcast_to(jnp.nan_to_num(contact, nan=0., posinf=0., neginf=0.), (n, n)) + radius
    if goalkeeper is not None:
        if opponent_penalty_x_min is None or penalty_half_width is None:
            raise ValueError("GK 경로에는 상대 penalty x 경계와 half width가 필요하다")
        keeper = jnp.asarray(goalkeeper, bool)
        penalty_x = jnp.asarray(opponent_penalty_x_min, jnp.float32)
        penalty_width = jnp.asarray(penalty_half_width, jnp.float32)
        finite &= jnp.isfinite(penalty_x) & jnp.isfinite(penalty_width) & (penalty_width > 0.)

    def contact_at(point):
        if goalkeeper is None:
            return contact[..., None]
        legal_hand = (point[..., 0] >= penalty_x) & (jnp.abs(point[..., 1]) <= penalty_width)
        keeper_radius = jnp.where(legal_hand, contact_physics.goalkeeper_radius_m,
                                  contact_physics.carry_radius_m) + radius
        return jnp.where(keeper[None, :, None], keeper_radius, contact[..., None])
    own_ok = ((jnp.linalg.norm(xy - position[:, None, :], axis=-1) <= max_speed[:, None] * samples + own_radius)
              & (z <= own_reach[:, None] + 1.e-5) & (z >= -.01))
    own_time = jnp.min(jnp.where(own_ok, samples, jnp.inf), axis=-1)
    own_index = jnp.argmin(jnp.where(own_ok, samples, jnp.inf), axis=-1)
    offered_height = jnp.take_along_axis(z, own_index[:, None], axis=-1)[:, 0]
    gap = jnp.linalg.norm(xy[:, None] - position[None, :, None], axis=-1)
    intercept = (opponent[None, :, None] & (gap <= max_speed[None, :, None] * samples[:, None] + contact_at(xy[:, None]))
                 & (z[:, None] <= opponent_reach[None, :, None]) & (z[:, None] >= -.01))
    opponent_time = jnp.min(jnp.where(intercept, samples[:, None], jnp.inf), axis=(1, 2))

    # 구간별 수비수 projection은 sample 사이의 직접 몸 충돌을 보강한다.
    segment = xyz[:, 1:, :2] - xyz[:, :-1, :2]
    relative = position[None, :, None] - xyz[:, None, :-1, :2]
    u = jnp.clip(jnp.sum(relative * segment[:, None], axis=-1)
                 / jnp.maximum(jnp.sum(segment**2, axis=-1)[:, None], 1.e-6), 0., 1.)
    indices = jnp.arange(_STEPS)[None, None]
    segment_duration = jnp.where(indices == impact[:, None, None] - 1,
                                 (duration - (impact - 1) * dt)[:, None, None], dt[:, None, None])
    projection_time = indices * dt[:, None, None] + u * segment_duration
    projection_time = jnp.minimum(projection_time, duration[:, None, None])
    special = jnp.concatenate((projection_time, opponent_cross[..., None]), axis=-1)
    special_xy, special_z = points(special)
    special_gap = jnp.linalg.norm(special_xy - position[None, :, None], axis=-1)
    valid = (opponent[None, :, None] & (special_gap <= max_speed[None, :, None] * special + contact_at(special_xy))
             & (special_z <= opponent_reach[None, :, None] + 1.e-5) & (special_z >= -.01))
    opponent_time = jnp.minimum(opponent_time, jnp.min(jnp.where(valid, special, jnp.inf), axis=(1, 2)))
    body_radius = max(passive_physics.torso_radius_m, passive_physics.head_radius_m,
                      .5 * passive_physics.shoulder_width_m + passive_physics.leg_radius_m) + radius
    body = jnp.any(opponent[None, :, None] & (special <= .20) & (special_gap <= body_radius)
                   & (special_z <= height[None, :, None] + .10), axis=(1, 2))
    own_safe = jnp.where(jnp.isfinite(own_time), own_time, 100.)
    other_safe = jnp.where(jnp.isfinite(opponent_time), opponent_time, own_safe + 2.)
    lead = other_safe - own_safe
    effort = jnp.clip((offered_height - height) / jnp.maximum(reach_height - height, 1.e-3), 0., 1.)
    teammate_block = jnp.zeros(n, bool)
    if nonreceiver_teammate is not None:
        teammate = jnp.asarray(nonreceiver_teammate, bool)[None, :] & ~jnp.eye(n, dtype=bool)
        teammate_block = jnp.any(teammate[..., None] & (special < own_safe[:, None, None])
                                 & (special_gap <= body_radius)
                                 & (special_z <= height[None, :, None]) & (special_z >= 0.), axis=(1, 2))
    boundary_limit = jnp.nan_to_num(field_half_extent, nan=0., posinf=0., neginf=0.) + radius
    start, end = xyz[:, :-1, :2], xyz[:, 1:, :2]
    motion = end - start
    crossing_fraction = (jnp.sign(end) * boundary_limit - start) / jnp.where(jnp.abs(motion) > 1.e-6, motion, 1.)
    crossed = (jnp.abs(end) > boundary_limit) & (jnp.abs(motion) > 1.e-6)
    segment_indices = jnp.arange(_STEPS)[None, :, None]
    segment_lengths = jnp.where(segment_indices == impact[:, None, None] - 1,
                                (duration - (impact - 1) * dt)[:, None, None], dt[:, None, None])
    crossing_time = segment_indices * dt[:, None, None] + jnp.clip(crossing_fraction, 0., 1.) * segment_lengths
    boundary_time = jnp.min(jnp.where(crossed, crossing_time, jnp.inf), axis=(1, 2))
    boundary_time = jnp.where(jnp.any(jnp.abs(origin[:2]) > boundary_limit), 0., boundary_time)
    boundary_first = jnp.isfinite(boundary_time) & (boundary_time <= own_time)
    delivery_feasible = (eligible & finite & candidate_valid & ~body & ~teammate_block
                         & ~boundary_first & jnp.isfinite(own_time))
    viable = delivery_feasible & (lead >= .05)
    score = jnp.where(viable, jnp.clip(jax.nn.sigmoid(lead / .10) - .03 * effort, 0., 1.), 0.)
    return AerialRoute(score, viable, delivery_feasible, own_time, opponent_time,
                       body, teammate_block, boundary_first)
