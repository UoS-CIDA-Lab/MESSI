"""선수 장기/단기 stamina 전이의 단일 진실원천.

장기 stamina는 한 경기 동안 누적되는 유산소 예산이며 경기 중 회복하지 않는다.
단기 stamina는 고강도 질주의 headroom으로, 고강도 이동에서 소모되고 저강도 구간에서
회복한다. movement, inverse playback, dataset reconstruction은 이 모듈의 순수 함수를
물리 substep ``dt``마다 호출해야 한다.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp

from .spatial import _safe_norm


class StaminaTransition(NamedTuple):
    """한 물리 substep의 두 저장량과 감사 가능한 전이 성분."""

    stamina_long: jnp.ndarray
    stamina_short: jnp.ndarray
    long_consumed: jnp.ndarray
    long_drain_rate: jnp.ndarray
    short_consumed: jnp.ndarray
    short_recovered: jnp.ndarray
    short_drain_rate: jnp.ndarray
    short_recovery_rate: jnp.ndarray
    chargeable: jnp.ndarray
    sprinting: jnp.ndarray
    sprint_extra: jnp.ndarray
    long_speed_load: jnp.ndarray
    long_acceleration_load: jnp.ndarray
    long_workload: jnp.ndarray
    short_load: jnp.ndarray
    short_recovery_factor: jnp.ndarray
    locomotion: jnp.ndarray


def long_stamina_drain_base(
    end_frac: float,
    reference_duration_s: float,
    reference_workload: float,
) -> float:
    """기준 workload 선수가 목표 잔여량에 도달하는 명목 피로율(1/s).

    잔량이 ``long_stamina_tail_knee``보다 클 때는 이 값이 실제 stamina 소모율과
    같다. knee 아래에서는 :func:`long_stamina_after_drain`이 같은 명목 피로를
    가역적인 지수 꼬리로 바꿔 0 클리핑을 피한다.
    """

    if not math.isfinite(float(end_frac)) or not 0.0 <= end_frac <= 1.0:
        raise ValueError(
            "long_stamina_end_frac must be finite and lie in [0, 1], "
            f"got {end_frac!r}"
        )
    if not math.isfinite(float(reference_duration_s)) or reference_duration_s <= 0.0:
        raise ValueError(
            "long_stamina_reference_duration_s must be finite and positive, "
            f"got {reference_duration_s!r}"
        )
    if not math.isfinite(float(reference_workload)) or reference_workload <= 0.0:
        raise ValueError(
            "long_stamina_reference_workload must be finite and positive, "
            f"got {reference_workload!r}"
        )
    return (1.0 - float(end_frac)) / (
        float(reference_duration_s) * float(reference_workload)
    )


def long_stamina_tail_decay(end_frac: float, tail_knee: float) -> float:
    """기준 종료값을 보존하는 장기 stamina 꼬리의 지수 계수.

    명목 선형 피로가 ``1-end_frac``만큼 누적될 때 ``1``에서 시작한 저장량이
    정확히 ``end_frac``에 도달하도록 계수를 정한다. ``tail_knee`` 위에서는
    기존 선형 좌표를 그대로 쓰므로 이 계수는 knee 아래에서만 작동한다.
    """

    if not math.isfinite(float(end_frac)) or not 0.0 < end_frac < 1.0:
        raise ValueError(
            "long_stamina_end_frac must be finite and lie in (0, 1) "
            "when the soft tail is enabled, "
            f"got {end_frac!r}"
        )
    if (
        not math.isfinite(float(tail_knee))
        or not float(end_frac) < float(tail_knee) < 1.0
    ):
        raise ValueError(
            "long_stamina_tail_knee must be finite and lie strictly between "
            f"long_stamina_end_frac and 1, got {tail_knee!r}"
        )
    return math.log(float(tail_knee) / float(end_frac)) / (
        float(tail_knee) - float(end_frac)
    )


def long_stamina_after_drain(
    stamina_long,
    nominal_drain,
    *,
    tail_knee: float,
    tail_decay: float,
):
    """명목 피로 증가를 정보 보존형 장기 stamina 전이로 바꾼다.

    knee까지는 ``stamina -= nominal_drain``인 기존 선형 식이다. 그 아래에서는
    ``stamina *= exp(-tail_decay * nominal_drain)``을 써 어떤 유한 workload도
    정확한 0으로 뭉개지지 않는다. 경계 횡단분을 분리하므로 전이는 연속이고,
    저장된 stamina만으로 다음 상태가 결정되는 Markov 계약도 유지된다.
    """

    nominal_drain = jnp.maximum(jnp.asarray(nominal_drain), 0.0)
    linear_room = jnp.maximum(stamina_long - tail_knee, 0.0)
    remains_linear = nominal_drain <= linear_room
    linear_next = stamina_long - nominal_drain
    tail_start = jnp.minimum(stamina_long, tail_knee)
    tail_drain = jnp.maximum(nominal_drain - linear_room, 0.0)
    tail_next = tail_start * jnp.exp(-tail_decay * tail_drain)
    return jnp.clip(jnp.where(remains_linear, linear_next, tail_next), 0.0, 1.0)


def long_speed_cap(vmax, stamina_long, *, floor: float):
    """장기 피로만 반영한 지속 가능 최고속도."""

    return vmax * (floor + (1.0 - floor) * jnp.clip(stamina_long, 0.0, 1.0))


def short_headroom(stamina_short, *, knee: float):
    """단기 저장량을 부드러운 최고속도 headroom ``[0, 1]``로 변환한다.

    ``knee`` 이상에서는 최고속도를 온전히 허용한다. 그 아래에서는 smoothstep을 써서
    경계에서 속도 상한이 꺾이지 않게 한다.
    """

    x = jnp.clip(stamina_short / knee, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def effective_speed_cap(
    vmax,
    stamina_long,
    stamina_short,
    *,
    long_floor: float,
    short_floor: float,
    short_knee: float,
):
    """장기·단기 stamina를 모두 반영한 순간 최고속도."""

    sustained = long_speed_cap(vmax, stamina_long, floor=long_floor)
    headroom = short_headroom(stamina_short, knee=short_knee)
    return sustained * (short_floor + (1.0 - short_floor) * headroom)


def stamina_step(
    stamina_long,
    stamina_short,
    player_vel,
    previous_player_vel,
    active_player,
    locomotion_mask,
    *,
    dt: float,
    long_drain_base: float,
    long_tail_knee: float,
    long_tail_decay: float,
    sprint_speed: float,
    long_sprint_mult: float,
    long_idle_load: float,
    long_speed_ref: float,
    long_speed_load_weight: float,
    long_accel_ref: float,
    long_accel_load_weight: float,
    vmax,
    endurance_factor=1.0,
    long_vmax_floor: float,
    short_depletion_s: float,
    short_depletion_speed_frac: float,
    short_speed_exponent: float,
    short_accel_ref: float,
    short_accel_load_weight: float,
    short_recovery_tau_s: float,
    short_recovery_speed_frac: float,
    short_recovery_exponent: float,
    short_long_recovery_penalty: float,
) -> StaminaTransition:
    """한 물리 substep의 장기 소모와 단기 소모/회복을 계산한다.

    장기 저장량은 active 선수의 기저 부하와 자기 추진 이동의 속도·가속·질주 부하를
    누적한다. 단기 저장량은 장기 피로만 적용한 속도 상한 대비 고강도 비율 및 양의 속력
    가속에 따라 소모되며, 그 임계값 아래에서 지수 시정수로 회복한다. 세트피스 엔진이
    강제로 옮긴 위치는 locomotion으로 세지 않아 이동 부하를 만들지 않고 휴식으로 처리한다.
    """

    speed = _safe_norm(player_vel, axis=1)
    previous_speed = _safe_norm(previous_player_vel, axis=1)
    acceleration = _safe_norm((player_vel - previous_player_vel) / dt, axis=1)
    positive_speed_accel = jnp.clip((speed - previous_speed) / dt, 0.0)

    chargeable = active_player
    locomotion = chargeable & locomotion_mask

    over = jnp.clip((speed - sprint_speed) / sprint_speed, 0.0, 2.0)
    sprinting_raw = speed > sprint_speed
    sprint_extra_raw = jnp.where(
        sprinting_raw, (long_sprint_mult - 1.0) * over, 0.0
    )
    long_speed_load_raw = long_speed_load_weight * jnp.clip(
        speed / long_speed_ref, 0.0, 4.0
    )
    long_acceleration_load_raw = long_accel_load_weight * jnp.clip(
        acceleration / long_accel_ref, 0.0, 4.0
    ) ** 2
    sprint_extra = jnp.where(locomotion, sprint_extra_raw, 0.0)
    long_speed_load = jnp.where(locomotion, long_speed_load_raw, 0.0)
    long_acceleration_load = jnp.where(
        locomotion, long_acceleration_load_raw, 0.0
    )
    long_workload = jnp.where(
        chargeable,
        long_idle_load + long_speed_load + long_acceleration_load + sprint_extra,
        0.0,
    )
    # ``long_drain_rate``는 선형화된 피로 좌표의 속도다. 실제 저장량 변화는 knee
    # 아래에서 작아지며 ``long_consumed``가 그 물리 substep의 실제 변화량을 낸다.
    # One reciprocal feeds both fatigue horizons.  The factor is an immutable
    # per-player profile within a stint, and spelling the common subexpression
    # once keeps CPU and GPU lowerings from duplicating the vector division.
    endurance_reciprocal = jnp.reciprocal(endurance_factor)
    long_drain_rate = (
        long_drain_base * long_workload * endurance_reciprocal
    )
    stamina_long_next = long_stamina_after_drain(
        stamina_long,
        dt * long_drain_rate,
        tail_knee=long_tail_knee,
        tail_decay=long_tail_decay,
    )

    sustained_cap = long_speed_cap(vmax, stamina_long, floor=long_vmax_floor)
    speed_fraction = speed / jnp.maximum(sustained_cap, 1e-6)
    depletion_width = 1.0 - short_depletion_speed_frac
    speed_intensity = jnp.clip(
        (speed_fraction - short_depletion_speed_frac) / depletion_width,
        0.0,
        1.0,
    ) ** short_speed_exponent
    accel_intensity = short_accel_load_weight * jnp.clip(
        positive_speed_accel / short_accel_ref, 0.0, 1.0
    )
    short_load_raw = jnp.clip(speed_intensity + accel_intensity, 0.0, 2.0)
    short_load = jnp.where(locomotion, short_load_raw, 0.0)
    short_drain_rate = (
        short_load / short_depletion_s * endurance_reciprocal
    )

    recovery_speed_fraction = jnp.where(locomotion, speed_fraction, 0.0)
    short_recovery_factor = jnp.clip(
        1.0 - recovery_speed_fraction / short_recovery_speed_frac,
        0.0,
        1.0,
    ) ** short_recovery_exponent
    recovery_capacity = jnp.clip(
        1.0 - short_long_recovery_penalty * (1.0 - stamina_long), 0.0, 1.0
    )
    short_recovery_rate = jnp.where(
        chargeable,
        (1.0 - stamina_short)
        * short_recovery_factor
        * recovery_capacity
        / short_recovery_tau_s,
        0.0,
    )
    stamina_short_next = jnp.clip(
        stamina_short + dt * (short_recovery_rate - short_drain_rate),
        0.0,
        1.0,
    )

    return StaminaTransition(
        stamina_long=stamina_long_next,
        stamina_short=stamina_short_next,
        long_consumed=stamina_long - stamina_long_next,
        long_drain_rate=long_drain_rate,
        short_consumed=jnp.maximum(stamina_short - stamina_short_next, 0.0),
        short_recovered=jnp.maximum(stamina_short_next - stamina_short, 0.0),
        short_drain_rate=short_drain_rate,
        short_recovery_rate=short_recovery_rate,
        chargeable=chargeable,
        sprinting=locomotion & sprinting_raw,
        sprint_extra=sprint_extra,
        long_speed_load=long_speed_load,
        long_acceleration_load=long_acceleration_load,
        long_workload=long_workload,
        short_load=short_load,
        short_recovery_factor=jnp.where(
            chargeable, short_recovery_factor, 0.0
        ),
        locomotion=locomotion,
    )
