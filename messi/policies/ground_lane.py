"""정책의 직선 저고도 패스를 위한 slip-to-roll 도달시간 근사.

환경 solver를 대체하지 않는다. 횡방향 slip, curl과 공중 bounce는 caller가 적용 범위를
제한해야 한다. Receiver별 [N,K] 감속 profile과 query [N,Q]만 보관한다.
"""

import math
import numbers

import jax
import jax.numpy as jnp

from messi.core.ball import BallPhysics


def _integrals(top, bottom, slope, intercept):
    delta = top-bottom
    lower_a = jnp.maximum(slope*bottom+intercept, 1.e-6)
    z = jnp.maximum(slope*delta/lower_a, -.9999999)
    small = jnp.abs(z) < .01
    safe_z = jnp.where(small, 1., z)
    log_term = jnp.where(small, 1-z/2+z**2/3-z**3/4+z**4/5,
                         jnp.log1p(z)/safe_z)
    area_term = jnp.where(small, .5-z/3+z**2/4-z**3/5+z**4/6,
                          (z-jnp.log1p(z))/safe_z**2)
    duration = delta/lower_a*log_term
    distance = bottom*duration+delta**2/lower_a*area_term
    return duration, jnp.maximum(distance, 0.)


def _interp_knots(x, xp, fp):
    """Monotonic knot interpolation without ``jnp.interp``'s search scan."""
    index = jnp.sum(x[..., None] >= xp, axis=-1, dtype=jnp.int32) - 1
    index = jnp.clip(index, 0, xp.shape[0] - 2)
    x0, x1 = xp[index], xp[index + 1]
    y0, y1 = fp[index], fp[index + 1]
    fraction = (x - x0) / (x1 - x0)
    value = y0 + fraction * (y1 - y0)
    return jnp.where(x <= xp[0], fp[0], jnp.where(x >= xp[-1], fp[-1], value))


def _rolling_profile(speed, physics):
    knots = jnp.asarray(physics.rolling_speed_knots_mps, jnp.float32)
    deceleration = jnp.asarray(physics.rolling_deceleration_knots_mps2, jnp.float32)
    low = jnp.minimum(speed[:, None], knots[::-1])
    high = jnp.concatenate((speed[:, None], low[:, :-1]), axis=-1)
    upper_a = _interp_knots(high, knots, deceleration)
    lower_a = _interp_knots(low, knots, deceleration)
    slope = (upper_a-lower_a) / jnp.maximum(high-low, 1.e-6)
    intercept = upper_a-slope*high
    duration, distance = _integrals(high, low, slope, intercept)
    end_distance = jnp.cumsum(distance, axis=-1)
    start_time = jnp.cumsum(duration, axis=-1)-duration
    return high, low, slope, intercept, end_distance, end_distance-distance, start_time


def _rolling_query(distance, speed, physics, integration_dt_s):
    high, low, slope, intercept, ends, starts, start_times = _rolling_profile(speed, physics)
    original_distance = distance
    maximum_a = max(physics.rolling_deceleration_knots_mps2)
    margin = 2*speed*integration_dt_s + .5*maximum_a*integration_dt_s**2
    # 같은 거리 여유를 ETA에도 적용한다. 정지점 gate만 보수적으로 바꾸고
    # 남은 도달시간은 낙관적인 continuous 값으로 두지 않는다.
    distance = distance + margin[:, None]
    # Knot별 loop와 index carry 대신 한 번 비교해 query 축에서 count한다. XLA는
    # 이 bounded [N,Q,K] 비교를 fused reduce로 처리해 production lane workload에서
    # 임시 메모리 증가 없이 warm latency를 낮춘다.
    index = jnp.sum(
        distance[:, :, None] > ends[:, None, :], axis=-1, dtype=jnp.int32
    )
    index = jnp.minimum(index, ends.shape[1]-1)
    def gather(value):
        return jnp.take_along_axis(value, index, axis=1)
    top, bottom, m, b, start, elapsed = map(gather, (high, low, slope, intercept, starts, start_times))
    remaining = jnp.maximum(distance-start, 0.)
    # d(distance)/d(speed) = -speed / deceleration. Newton 단계를 구간 안으로
    # 제한해 고정 16회 bisection보다 적은 적분 평가로 root를 찾는다.
    current = .5*(bottom+top)
    def refine(_, bounds):
        lower, upper, current = bounds
        _, travel = _integrals(top, current, m, b)
        residual = travel-remaining
        too_far = residual > 0.
        lower = jnp.where(too_far, current, lower)
        upper = jnp.where(too_far, upper, current)
        acceleration = jnp.maximum(m*current+b, 1.e-6)
        newton = current + residual*acceleration/jnp.maximum(current, 1.e-6)
        midpoint = .5*(lower+upper)
        newton = jnp.where(jnp.isfinite(newton), newton, midpoint)
        current = jnp.clip(newton, lower, upper)
        return lower, upper, current
    _, _, speed_at_distance = jax.lax.fori_loop(
        0, 8, refine, (bottom, top, current)
    )
    duration, _ = _integrals(top, speed_at_distance, m, b)
    # Continuous 감속과 semi-implicit physics의 정지점 차이를 거리로 제한한다.
    # 이 여유의 검증 범위는 caller가 사용하는 dt 및 test의 physics 조건에 한정된다.
    reachable = distance <= ends[:, -1, None]
    return jnp.where(original_distance == 0., 0., jnp.where(reachable, elapsed+duration, jnp.inf))


def ground_arrival_times(
    distance, release_speed, surface_velocity, *,
    physics: BallPhysics = BallPhysics(), integration_dt_s: float = .0125,
):
    """거리 [N,Q] m, 속도/표면속도 [N] m/s -> 도달시간 [N,Q] s.

    초기 ray 방향 속도만 사용한다. 표면속도는 spin으로 인한 접지점 속도의 ray 성분이다.
    정지거리 여유 안쪽의 도달만 반환하며 미도달·비유한 값·역방향 초기속도는 +inf다.
    거리 0은 유효한 입력에서 0이다. Custom physics의 knot를 그대로 적분하며 이 함수는
    no-player-contact의 연속 지면 근사이지 arbitrary timebase의 정확 solver가 아니다.
    """
    if not isinstance(integration_dt_s, numbers.Real) or isinstance(integration_dt_s, bool):
        raise TypeError('integration_dt_s는 static 실수여야 한다')
    if not math.isfinite(integration_dt_s) or integration_dt_s <= 0.:
        raise ValueError('integration_dt_s는 유한한 양수여야 한다')
    distance, release_speed, surface_velocity = (
        jnp.asarray(value, jnp.float32) for value in (distance, release_speed, surface_velocity)
    )
    valid = (jnp.isfinite(distance) & (distance >= 0.)
             & jnp.isfinite(release_speed[:, None]) & (release_speed[:, None] >= 0.)
             & jnp.isfinite(surface_velocity[:, None]))
    distance, speed, surface = (
        jnp.nan_to_num(value, nan=0., posinf=0., neginf=0.)
        for value in (distance, release_speed, surface_velocity)
    )
    speed = jnp.maximum(speed, 0.)
    delta = -physics.inertia_ratio/(1+physics.inertia_ratio)*(speed+surface)
    slide_a = physics.ground_slide_friction*physics.gravity_mps2
    if slide_a == 0.:
        rolling = jnp.abs(delta) <= 1.e-9
        rolling_time = _rolling_query(distance, speed, physics, integration_dt_s)
        constant_time = jnp.where(speed[:, None] > 0., distance/jnp.maximum(speed[:, None], 1.e-6), jnp.inf)
        result = jnp.where(rolling[:, None], rolling_time, constant_time)
        return jnp.where(valid, jnp.where(distance == 0., 0., result), jnp.inf)
    slip_time = jnp.abs(delta)/slide_a
    rolling_speed = jnp.maximum(speed+delta, 0.)
    slip_distance = .5*(speed+rolling_speed)*slip_time
    deceleration = -jnp.sign(delta)*slide_a
    energy = speed[:, None]**2-2*deceleration[:, None]*distance
    at_distance_speed = jnp.sqrt(jnp.maximum(energy, 0.))
    early_time = 2*distance/jnp.maximum(speed[:, None]+at_distance_speed, 1.e-6)
    early_time = jnp.where(energy >= 0., early_time, jnp.inf)
    rolling_time = _rolling_query(jnp.maximum(distance-slip_distance[:, None], 0.),
                                  rolling_speed, physics, integration_dt_s)
    result = jnp.where(distance <= slip_distance[:, None], early_time, slip_time[:, None]+rolling_time)
    return jnp.where(valid, jnp.where(distance == 0., 0., result), jnp.inf)
