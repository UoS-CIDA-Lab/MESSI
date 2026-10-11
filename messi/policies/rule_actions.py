"""규칙 정책의 기존 슛 판단·방출 예측과 공통 수학 helper leaf."""

import jax
import jax.numpy as jnp
import numpy as np

from messi.core.contact import ContactPhysics
from messi.core.passive_contact import DEFAULT_PASSIVE_CONTACT_PHYSICS
from messi.environment.observation import SharedScene
from messi.policies.rule_config import RulePolicyConfig
from messi.policies.rule_features import _inside_attacking_penalty_area

def _shot_compensation_eligible(position, ball_height, height, config, contact_physics):
    """[N] 거리와 무관하게 실제 foot 접촉 높이에만 보정 후보를 적용한다."""
    return ball_height <= height*jnp.float32(contact_physics.pelvis_height_factor)+jnp.float32(config.ball_radius_m)

def _shot_request_direction(direction, incoming_xy, goal_distance, config, contact_physics, *, compensation_eligible=True):
    """고정 foot impulse로 목표 ray를 만드는 단위 방향. 불가능한 역산은 기존 ray."""
    desired = _unit(direction)
    if not config.shot_residual_velocity_compensation:
        return desired
    power = jnp.where(goal_distance <= jnp.float32(config.close_shot_distance_m),
                      jnp.float32(config.close_shot_power), jnp.float32(1.0))
    launch = _ground_launch_angle(jnp.float32(config.shot_launch), contact_physics)
    impulse = power * jnp.float32(contact_physics.kick_speed_max_mps) * jnp.cos(launch)
    parallel = jnp.sum(incoming_xy * desired, axis=-1)
    perpendicular_squared = jnp.maximum(jnp.sum(incoming_xy**2, axis=-1)-parallel**2, 0.0)
    radicand = impulse**2-perpendicular_squared
    speed = parallel+jnp.sqrt(jnp.maximum(radicand, 0.0))
    request = (speed[..., None]*desired-incoming_xy)/jnp.maximum(impulse[..., None], 1.0e-6)
    feasible = (
        jnp.asarray(compensation_eligible, jnp.bool_)
        & (radicand >= 0.0) & (speed > 1.0e-6) & (impulse > 1.0e-6)
        & jnp.all(jnp.isfinite(request), axis=-1)
        & (jnp.sum(desired**2, axis=-1) > 0.0)
    )
    return jnp.where(feasible[..., None], _unit(request), desired)

def _shot_release_velocity(
    direction: jax.Array,
    incoming_xy: jax.Array,
    incoming_z: jax.Array,
    goal_distance: jax.Array,
    config: "RulePolicyConfig",
    contact_physics: ContactPhysics,
    *,
    compensation_eligible=True,
) -> tuple[jax.Array, jax.Array]:
    """Broadcast 가능한 SI 입력에서 nominal foot 방출 속도(xy, z)를 예측한다.

    목표점 선택과 lane 판정이 같은 impulse·잔여 속도·generated-speed cap을 쓴다.
    실제 접촉 성공이나 이후 비행을 보장하지 않는다.
    """
    power = jnp.where(
        goal_distance <= jnp.float32(config.close_shot_distance_m),
        jnp.float32(config.close_shot_power), jnp.float32(1.0),
    )
    launch = _ground_launch_angle(jnp.float32(config.shot_launch), contact_physics)
    impulse = power * jnp.float32(contact_physics.kick_speed_max_mps)
    force_direction = _shot_request_direction(direction, incoming_xy, goal_distance, config, contact_physics,
                                              compensation_eligible=compensation_eligible)
    force_norm = jnp.sqrt(jnp.sum(force_direction**2, axis=-1))
    candidate_xy = incoming_xy + (impulse * jnp.cos(launch))[..., None] * force_direction
    candidate_z = incoming_z + impulse * force_norm * jnp.sin(launch)
    incoming_speed = jnp.sqrt(jnp.sum(incoming_xy**2, axis=-1) + incoming_z**2)
    candidate_speed = jnp.sqrt(jnp.sum(candidate_xy**2, axis=-1) + candidate_z**2)
    scale = jnp.minimum(
        1.0,
        jnp.maximum(incoming_speed, jnp.float32(contact_physics.kick_speed_max_mps))
        / jnp.maximum(candidate_speed, 1.0e-6),
    )
    return candidate_xy * scale[..., None], candidate_z * scale

def _nominal_shot_target_y(
    shooter_position: jax.Array,
    goalkeeper_position: jax.Array,
    goalkeeper_velocity: jax.Array,
    has_goalkeeper: jax.Array,
    config: "RulePolicyConfig",
    contact_physics: ContactPhysics,
    *,
    ball_position: jax.Array | None = None,
    incoming_velocity: jax.Array | None = None,
    incoming_vertical_velocity: jax.Array | None = None,
    compensation_eligible: jax.Array | None = None,
) -> jax.Array:
    """[N, 2] 공격 +x SI 관측에서 [N] nominal goal-line y(m)를 고른다.

    잔여 속도를 반영한 좌/중앙/우 후보 중 골문으로 향하는 경로에서 등속 GK와의
    최소 거리를 최대화한다. Drag·높이·GK 가속은 모델링하지 않는 목표점 근사다.
    GK가 없거나 최댓값이 같으면 기존 반대편 목표를 우선한다.
    """

    half_width = jnp.float32(
        0.5 * config.goal_width_m * config.shot_target_half_width_fraction
    )
    reference_y = jnp.where(
        has_goalkeeper, goalkeeper_position[:, 1], shooter_position[:, 1]
    )
    fallback = jnp.where(reference_y >= 0.0, -half_width, half_width)
    target_y = jnp.stack((fallback, jnp.zeros_like(fallback), -fallback), axis=-1)
    dx = jnp.float32(config.field_half_length_m) - shooter_position[:, 0]
    ray = jnp.stack(
        (jnp.broadcast_to(dx[:, None], target_y.shape),
         target_y - shooter_position[:, 1, None]),
        axis=-1,
    )
    goal_distance = jnp.sqrt(dx * dx + shooter_position[:, 1] ** 2)
    origin = shooter_position if ball_position is None else ball_position
    eligible = jnp.ones_like(dx, jnp.bool_) if compensation_eligible is None else compensation_eligible
    if config.shot_residual_velocity_compensation:
        request_origin = jnp.where(eligible[:, None], origin, shooter_position)
        ray = jnp.stack(
            (jnp.broadcast_to((jnp.float32(config.field_half_length_m)-request_origin[:, 0])[:, None], target_y.shape),
             target_y-request_origin[:, 1, None]), axis=-1,
        )
    incoming = jnp.zeros_like(shooter_position) if incoming_velocity is None else incoming_velocity
    incoming_z = jnp.zeros_like(dx) if incoming_vertical_velocity is None else jnp.broadcast_to(incoming_vertical_velocity, dx.shape)
    ball_velocity, _ = _shot_release_velocity(
        ray, incoming[:, None, :], incoming_z[:, None], goal_distance[:, None],
        config, contact_physics,
        compensation_eligible=eligible[:, None],
    )
    flight_time = jnp.maximum(
        (jnp.float32(config.field_half_length_m) - origin[:, 0, None])
        / jnp.maximum(ball_velocity[..., 0], 1.0e-6), 0.0,
    )
    crossing_y = origin[:, 1, None] + flight_time * ball_velocity[..., 1]
    on_goal_path = (
        (ball_velocity[..., 0] > 1.0e-6)
        & (flight_time > 0.0)
        & (jnp.abs(crossing_y) <= jnp.float32(0.5 * config.goal_width_m - config.ball_radius_m))
    )
    relative_position = goalkeeper_position[:, None, :] - origin[:, None, :]
    relative_velocity = goalkeeper_velocity[:, None, :] - ball_velocity
    closest_time = jnp.clip(
        -jnp.sum(relative_position * relative_velocity, axis=-1)
        / jnp.maximum(jnp.sum(relative_velocity * relative_velocity, axis=-1), 1.0e-6),
        0.0,
        flight_time,
    )
    separation = relative_position + closest_time[..., None] * relative_velocity
    clearance_squared = jnp.sum(separation * separation, axis=-1)
    candidate_value = jnp.where(on_goal_path, clearance_squared, -jnp.inf)
    selected = jnp.take_along_axis(
        target_y, jnp.argmax(candidate_value, axis=-1)[:, None], axis=-1
    )[:, 0]
    return jnp.where(has_goalkeeper, selected, fallback)

def _shot_lane_open(
    scene: SharedScene,
    goal_vector: jax.Array,
    is_goalkeeper: jax.Array,
    height: jax.Array,
    reach_height: jax.Array,
    ball_height: jax.Array,
    ball_vertical_velocity: jax.Array,
    config: "RulePolicyConfig",
    contact_physics: ContactPhysics,
) -> jax.Array:
    """[N] 슛 lane 가능성: 등속 상대 운동과 접촉 종류별 도달 범위.

    Foot impulse의 속도 cap은 반영하되 drag·bounce·상대 가속/미래 intent는
    풀지 않는다. Passive body는 기본 geometry의 보수적 수평 envelope이며
    세부 capsule이나 custom passive physics와의 동등성을 주장하지 않는다.
    """

    goal_distance = jnp.sqrt(
        (jnp.float32(config.field_half_length_m) - scene.position_team[:, 0]) ** 2
        + scene.position_team[:, 1] ** 2
    )
    velocity, vertical_velocity = _shot_release_velocity(
        goal_vector, scene.ball_velocity_team, ball_vertical_velocity, goal_distance,
        config, contact_physics,
        compensation_eligible=_shot_compensation_eligible(scene.ball_position_team, ball_height, height, config, contact_physics),
    )
    horizontal_speed = jnp.sqrt(jnp.sum(velocity**2, axis=-1))
    goal_time = jnp.maximum(
        (jnp.float32(config.field_half_length_m) - scene.ball_position_team[:, 0])
        / jnp.maximum(velocity[:, 0], 1.0e-6),
        0.0,
    )
    relative_position = (
        scene.relative_position
        + (scene.position_team - scene.ball_position_team)[:, None, :]
    )
    relative_velocity = (
        scene.relative_velocity + (scene.velocity_team - velocity)[:, None, :]
    )
    closest_time = jnp.clip(
        -jnp.sum(relative_position * relative_velocity, axis=-1)
        / jnp.maximum(jnp.sum(relative_velocity**2, axis=-1), 1.0e-6),
        0.0,
        goal_time[:, None],
    )
    closest_delta = relative_position + closest_time[..., None] * relative_velocity
    distance_squared = jnp.sum(closest_delta**2, axis=-1)
    predicted_height = jnp.maximum(
        jnp.float32(config.ball_radius_m),
        ball_height + vertical_velocity[:, None] * closest_time
        - jnp.float32(0.5 * config.gravity_mps2) * closest_time**2,
    )
    ball_radius = jnp.float32(config.ball_radius_m)
    foot_height = predicted_height <= (
        height[None, :] * jnp.float32(contact_physics.pelvis_height_factor) + ball_radius
    )
    foot_speed_allowed = (
        horizontal_speed[:, None]
        + jnp.float32(contact_physics.height_speed_penalty_mps_per_m) * predicted_height
        <= jnp.float32(contact_physics.block_speed_limit_mps)
    )
    # Turf 충돌 이후에는 실제 감속을 풀지 않는 이 근사로 고속 foot 면제를
    # 연장하지 않는다. Bounce 이후 발 차단 가능성은 보수적으로 유지한다.
    gravity = jnp.float32(config.gravity_mps2)
    first_turf_time = (
        vertical_velocity
        + jnp.sqrt(vertical_velocity**2 + 2.0 * gravity * jnp.maximum(ball_height - ball_radius, 0.0))
    ) / gravity
    foot_speed_allowed |= closest_time >= first_turf_time[:, None]
    active_radius = jnp.minimum(
        jnp.float32(config.shot_block_radius_m),
        jnp.float32(contact_physics.carry_radius_m) + ball_radius,
    )
    active_contact = (
        (distance_squared <= active_radius**2)
        & (predicted_height <= reach_height[None, :] + ball_radius)
        & ((~foot_height) | foot_speed_allowed)
    )
    passive = DEFAULT_PASSIVE_CONTACT_PHYSICS
    body_radius = jnp.float32(max(
        passive.torso_radius_m, passive.head_radius_m,
        0.5 * passive.shoulder_width_m + passive.leg_radius_m,
    )) + ball_radius
    body_contact = (
        (distance_squared <= body_radius**2)
        & (predicted_height <= height[None, :] + ball_radius)
    )
    # GK hand는 standing/dive 모두 실제 2 m envelope를 쓴다. Standing 반경은
    # 접촉 후 recovery effort의 기준이지 hand contact의 hard gate가 아니다.
    hand_radius = jnp.float32(contact_physics.goalkeeper_radius_m) + ball_radius
    a = jnp.sum(relative_velocity**2, axis=-1)
    b = jnp.sum(relative_position * relative_velocity, axis=-1)
    c = jnp.sum(relative_position**2, axis=-1) - hand_radius**2
    discriminant = b**2 - a * c
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    moving = a > 1.0e-6
    hand_start = jnp.where(moving, (-b - root) / jnp.maximum(a, 1.0e-6), 0.0)
    hand_end = jnp.where(moving, (-b + root) / jnp.maximum(a, 1.0e-6), goal_time[:, None])
    intersects = jnp.where(moving, discriminant >= 0.0, c <= 0.0)
    start = jnp.maximum(hand_start, 0.0)
    end = jnp.minimum(hand_end, goal_time[:, None])
    goalkeeper_position = scene.ball_position_team[:, None, :] + relative_position
    goalkeeper_velocity = velocity[:, None, :] + relative_velocity

    def inside_axis_interval(position, speed, lower, upper):
        parallel = jnp.abs(speed) <= 1.0e-6
        safe_speed = jnp.where(parallel, 1.0, speed)
        first, second = (lower - position) / safe_speed, (upper - position) / safe_speed
        inside = (position >= lower) & (position <= upper)
        entry = jnp.where(parallel, jnp.where(inside, -jnp.inf, jnp.inf), jnp.minimum(first, second))
        exit_time = jnp.where(parallel, jnp.where(inside, jnp.inf, -jnp.inf), jnp.maximum(first, second))
        return entry, exit_time

    # 법적 손 사용은 미래 GK 중심과 공 중심 모두 opponent penalty area 안일 때다.
    # 상대 슛 이후의 back-pass 제한 해제는 실제 transition이 결정하며 여기서는
    # 해당 미래 intent/recovery나 catch 확률까지 확정하지 않는다.
    for point, motion in ((scene.ball_position_team[:, None, :], velocity[:, None, :]),
                          (goalkeeper_position, goalkeeper_velocity)):
        for axis, lower, upper in (
            (0, config.field_half_length_m - config.penalty_area_length_m, config.field_half_length_m),
            (1, -config.penalty_area_half_width_m, config.penalty_area_half_width_m),
        ):
            entry, exit_time = inside_axis_interval(point[..., axis], motion[..., axis], lower, upper)
            start, end = jnp.maximum(start, entry), jnp.minimum(end, exit_time)
    # 수직 포물선은 concave이므로 교차 구간 양끝 중 낮은 높이가 hand reach에
    # 닿으면 구간 안에 가능한 접촉이 있다. Closest 한 점만 보면 입·출구를 놓친다.
    start_safe = jnp.clip(start, 0.0, goal_time[:, None])
    end_safe = jnp.clip(end, 0.0, goal_time[:, None])
    start_height = ball_height + vertical_velocity[:, None] * start_safe - .5 * gravity * start_safe**2
    end_height = ball_height + vertical_velocity[:, None] * end_safe - .5 * gravity * end_safe**2
    hand_height = jnp.maximum(ball_radius, jnp.minimum(start_height, end_height))
    goalkeeper_hand = intersects & (start <= end) & (hand_height <= reach_height[None, :] + ball_radius)
    goalkeeper_block = goalkeeper_hand | body_contact | active_contact
    blocked = jnp.any(
        scene.opponent
        & jnp.where(is_goalkeeper[None, :], goalkeeper_block, body_contact | active_contact),
        axis=-1,
    )
    return (~blocked) & (velocity[:, 0] > 1.0e-6) & (goal_time > 0.0)

def _shot_aim_noise_scale(
    goal_centre_vector: jax.Array,
    body_forward_team: jax.Array,
    nearest_opponent_distance_squared: jax.Array,
    stamina_short: jax.Array,
    config: "RulePolicyConfig",
) -> jax.Array:
    """거리·압박·몸 방향·피로에 따른 goal-half-width 조준 오차를 계산한다."""

    goal_distance = jnp.sqrt(
        jnp.sum(goal_centre_vector * goal_centre_vector, axis=-1) + jnp.float32(1.0e-6)
    )
    distance = jnp.clip(
        goal_distance / jnp.float32(config.long_shot_distance_m), 0.0, 1.0
    )
    pressure = jnp.float32(1.0) - jnp.clip(
        jnp.sqrt(nearest_opponent_distance_squared + jnp.float32(1.0e-6))
        / jnp.float32(config.pressure_distance_m),
        0.0,
        1.0,
    )
    alignment = jnp.clip(
        jnp.sum(_unit(goal_centre_vector) * body_forward_team, axis=-1), -1.0, 1.0
    )
    misalignment = jnp.float32(0.5) * (jnp.float32(1.0) - alignment)
    fatigue = jnp.float32(1.0) - jnp.clip(stamina_short, 0.0, 1.0)
    clean_opportunity = (
        (jnp.float32(1.0) - distance)
        * (jnp.float32(1.0) - pressure)
        * (jnp.float32(1.0) - misalignment)
        * (jnp.float32(1.0) - fatigue)
    )
    base_noise = jnp.float32(config.shot_aim_noise_fraction) * (
        jnp.float32(1.0)
        - clean_opportunity
        * (jnp.float32(1.0) - jnp.float32(config.shot_aim_clean_base_scale))
    )
    scale = (
        base_noise
        + jnp.float32(config.shot_aim_distance_gain) * distance
        + jnp.float32(config.shot_aim_pressure_gain) * pressure
        + jnp.float32(config.shot_aim_alignment_gain) * misalignment
        + jnp.float32(config.shot_aim_fatigue_gain) * fatigue
    )
    return jnp.minimum(scale, jnp.float32(config.shot_aim_noise_max_fraction))

def _smooth_release_space(
    distance: jax.Array, close_distance: jax.Array, useful_distance: jax.Array
) -> jax.Array:
    """접촉 직전 0에서 유효한 release 간격의 1까지 매끄럽게 증가한다."""

    value = jnp.clip(
        (jnp.asarray(distance, jnp.float32) - jnp.asarray(close_distance, jnp.float32))
        / jnp.maximum(
            jnp.asarray(useful_distance, jnp.float32)
            - jnp.asarray(close_distance, jnp.float32),
            jnp.float32(1.0e-3),
        ),
        0.0,
        1.0,
    )
    return value * value * (jnp.float32(3.0) - jnp.float32(2.0) * value)

def _shot_opportunity_quality(
    goal_distance: jax.Array,
    goal_angle: jax.Array,
    centrality: jax.Array,
    nearest_outfield_distance: jax.Array,
    goalkeeper_distance: jax.Array,
    config: "RulePolicyConfig",
    contact_physics: ContactPhysics,
) -> jax.Array:
    """거리·goal angle·release 간격을 함께 쓰는 bounded shot ordering.

    거리만 짧아질수록 좋아지는 점수와 달리, 박스 진입 전에 quality가 완만하게 열리고
    defender 또는 goalkeeper와 접촉 envelope가 겹치기 시작하면 다시 낮아진다. 이는
    xG가 아니라 SHOOT/PASS/CONTROL macro 선택용 design prior다.
    """

    distance = jnp.asarray(goal_distance, jnp.float32)
    softness = jnp.maximum(
        jnp.float32(0.20) * jnp.asarray(config.shoot_distance_m, jnp.float32),
        jnp.float32(1.0),
    )
    range_quality = jax.nn.sigmoid(
        (jnp.asarray(config.shoot_distance_m, jnp.float32) - distance) / softness
    )
    reference_angle = jnp.float32(2.0) * jnp.arctan2(
        jnp.float32(0.5) * jnp.asarray(config.goal_width_m, jnp.float32),
        jnp.maximum(jnp.asarray(config.close_shot_distance_m, jnp.float32), 1.0),
    )
    angle_quality = jnp.clip(
        jnp.asarray(goal_angle, jnp.float32)
        / jnp.maximum(reference_angle, jnp.float32(1.0e-3)),
        0.0,
        1.0,
    )
    base_quality = jnp.clip(
        jnp.asarray(config.shot_range_weight, jnp.float32) * range_quality
        + jnp.asarray(config.shot_opening_weight, jnp.float32) * angle_quality
        + jnp.asarray(config.shot_centrality_weight, jnp.float32)
        * jnp.asarray(centrality, jnp.float32),
        0.0,
        1.0,
    )

    outfield_space = _smooth_release_space(
        nearest_outfield_distance,
        jnp.asarray(
            contact_physics.challenge_radius_m + config.ball_radius_m, jnp.float32
        ),
        jnp.asarray(config.pressure_distance_m, jnp.float32),
    )
    goalkeeper_space = _smooth_release_space(
        goalkeeper_distance,
        jnp.asarray(
            contact_physics.goalkeeper_radius_m + config.ball_radius_m, jnp.float32
        ),
        jnp.maximum(
            jnp.asarray(config.pressure_distance_m, jnp.float32),
            jnp.asarray(
                contact_physics.goalkeeper_radius_m
                + config.ball_radius_m
                + config.receive_distance_m,
                jnp.float32,
            ),
        ),
    )
    usable_separation = jnp.minimum(outfield_space, goalkeeper_space)
    pressure_floor = jnp.float32(config.shot_pressure_quality_floor)
    separation_factor = pressure_floor + (jnp.float32(1.0) - pressure_floor) * usable_separation
    return jnp.clip(base_quality * separation_factor, 0.0, 1.0)

def _shot_quality_context(
    scene: SharedScene,
    is_goalkeeper: jax.Array,
    config: "RulePolicyConfig",
    contact_physics: ContactPhysics,
    *,
    carrier_release_clear=False,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """한 번의 shared geometry로 선수별 shot quality와 후속 판단 context를 만든다."""

    count = scene.valid.shape[0]
    observer_index = jnp.arange(count, dtype=jnp.int32)
    opponent_goalkeeper = scene.opponent & is_goalkeeper[None, :]
    goal_centre_vector = jnp.stack(
        (
            config.field_half_length_m - scene.position_team[:, 0],
            -scene.position_team[:, 1],
        ),
        axis=-1,
    )
    goal_distance_squared = jnp.sum(goal_centre_vector * goal_centre_vector, axis=-1)
    goal_distance = jnp.sqrt(goal_distance_squared + jnp.float32(1.0e-6))
    centrality = 1.0 - jnp.clip(
        jnp.abs(scene.position_team[:, 1])
        / jnp.maximum(jnp.asarray(config.penalty_area_half_width_m, jnp.float32), 1.0),
        0.0,
        1.0,
    )
    goal_dx = (
        jnp.asarray(config.field_half_length_m, jnp.float32) - scene.position_team[:, 0]
    )
    upper_post_vector = jnp.stack(
        (
            goal_dx,
            jnp.float32(0.5) * jnp.asarray(config.goal_width_m, jnp.float32)
            - scene.position_team[:, 1],
        ),
        axis=-1,
    )
    lower_post_vector = jnp.stack(
        (
            goal_dx,
            -jnp.float32(0.5) * jnp.asarray(config.goal_width_m, jnp.float32)
            - scene.position_team[:, 1],
        ),
        axis=-1,
    )
    goal_post_cross = (
        upper_post_vector[:, 0] * lower_post_vector[:, 1]
        - upper_post_vector[:, 1] * lower_post_vector[:, 0]
    )
    goal_post_dot = jnp.sum(upper_post_vector * lower_post_vector, axis=-1)
    goal_angle = jnp.abs(jnp.arctan2(goal_post_cross, goal_post_dot))
    opponent_outfield = scene.opponent & (~is_goalkeeper[None, :])
    nearest_outfield_distance = jnp.sqrt(
        jnp.min(jnp.where(opponent_outfield, scene.distance_squared, jnp.inf), axis=-1)
    )
    if config.shot_release_contest_relief_enabled:
        # 현재 공의 release 위험은 실제 carrier에만 적용한다. 다른 선수의 값은
        # hypothetical receiver quality이므로 현재 공에서 멀다는 이유로 올리지 않는다.
        nearest_outfield_distance = jnp.where(
            scene.possessor & carrier_release_clear,
            jnp.maximum(nearest_outfield_distance, config.pressure_distance_m),
            nearest_outfield_distance,
        )
    nearest_goalkeeper_distance = jnp.sqrt(
        jnp.min(
            jnp.where(opponent_goalkeeper, scene.distance_squared, jnp.inf), axis=-1
        )
    )
    shot_quality = _shot_opportunity_quality(
        goal_distance,
        goal_angle,
        centrality,
        nearest_outfield_distance,
        nearest_goalkeeper_distance,
        config,
        contact_physics,
    )
    inside_attacking_box = _inside_attacking_penalty_area(scene.position_team, config)
    goalkeeper_index = jnp.argmax(opponent_goalkeeper, axis=-1)
    goalkeeper_y = (
        scene.position_team[:, 1]
        + scene.relative_position[observer_index, goalkeeper_index, 1]
    )
    return (
        goal_centre_vector,
        goal_distance_squared,
        goal_distance,
        centrality,
        opponent_goalkeeper,
        goalkeeper_y,
        shot_quality,
        inside_attacking_box,
    )

def _regular_shot_quality_threshold(
    inside_attacking_box: jax.Array,
    config: "RulePolicyConfig",
) -> jax.Array:
    """공격 penalty area 밖의 일반 슛에만 추가 quality margin을 적용한다.

    Long-shot 전용 분기의 pressure·centrality gate는 별도 contract로 유지한다.
    """

    base = jnp.asarray(config.shot_quality_threshold, jnp.float32)
    if config.outside_box_shot_quality_margin == 0.0:
        return base
    outside = base + jnp.asarray(config.outside_box_shot_quality_margin, jnp.float32)
    return jnp.where(inside_attacking_box, base, outside)

def _shot_macro_ranking_value(
    shot_quality: jax.Array,
    inside_attacking_box: jax.Array,
    fresh_unlinked_possession: jax.Array,
    transition_shot_scale: jax.Array,
    config: "RulePolicyConfig",
    *,
    goal_distance: jax.Array | None = None,
) -> jax.Array:
    """Possession의 SHOT macro 값만 계산한다.

    Outside-box scale은 eligibility, loose-ball first-time release와 실행 제어를
    바꾸지 않는다. Neutral 값 1.0은 기존 산술과 그래프를 그대로 보존한다.
    """

    if config.outside_box_shot_macro_scale == 1.0:
        scaled_quality = shot_quality
    else:
        scaled_quality = shot_quality * jnp.where(
            inside_attacking_box,
            jnp.float32(1.0),
            jnp.asarray(config.outside_box_shot_macro_scale, jnp.float32),
        )
    transition_scale = jnp.where(
        inside_attacking_box,
        jnp.maximum(
            transition_shot_scale,
            jnp.asarray(config.inside_box_transition_shot_floor, jnp.float32),
        ),
        transition_shot_scale,
    )
    value = jnp.clip(
        scaled_quality * jnp.asarray(config.shot_value_gain, jnp.float32),
        0.0,
        1.0,
    ) * jnp.where(fresh_unlinked_possession, transition_scale, 1.0)
    if config.shot_distance_macro_floor == 1.0:
        return value
    if goal_distance is None:
        raise ValueError("거리 감쇠를 사용하면 goal_distance가 필요하다")
    distance = jnp.asarray(goal_distance, jnp.float32)
    fraction = jnp.clip(
        (distance - config.shot_distance_macro_near_m)
        / (config.shot_distance_macro_far_m - config.shot_distance_macro_near_m),
        0.0,
        1.0,
    )
    smooth = fraction * fraction * (3.0 - 2.0 * fraction)
    scale = 1.0 - (1.0 - config.shot_distance_macro_floor) * smooth
    # 잘못된 관측 거리는 새 후보에서 슛 선호를 높이지 않는다.
    scale = jnp.where(jnp.isfinite(distance) & (distance >= 0.0), scale, 0.0)
    return value * scale

def _interior_limit_m(half_extent_m: float, margin_m: float) -> float:
    """Static pitch margin이 clip interval을 뒤집지 않게 내부 한계를 만든다."""

    return float(
        np.maximum(
            np.float32(half_extent_m) - np.float32(margin_m),
            np.float32(0.0),
        )
    )

def _unit(vector: jax.Array) -> jax.Array:
    norm = jnp.sqrt(jnp.sum(vector * vector, axis=-1, keepdims=True) + 1.0e-6)
    return vector / jnp.maximum(norm, 1.0)

def _ground_launch_angle(signed_launch, contact_physics: ContactPhysics):
    """Ground contact에서 signed launch control이 실현하는 actuator 각도(rad)."""

    floor = -jnp.float32(contact_physics.ground_launch_down_max_radians)
    unit = 0.5 * (jnp.asarray(signed_launch, jnp.float32) + 1.0)
    return floor + unit * (jnp.float32(contact_physics.launch_max_radians) - floor)
