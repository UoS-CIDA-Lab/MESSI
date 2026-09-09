"""Factory-time aerial-kick calibration against the authoritative ball physics.

The immutable configuration is rolled out once, leaving only a small monotone
distance table in the player-policy graph. The table is generated from the
current drag, lift, bounce, spin-generation, and launch-control semantics.

The shipped default table avoids calibration compilation in normal workers.
Custom immutable configurations use the same CPU-only cached generator.  At
runtime an aerial kick costs fixed-shape one-dimensional interpolation only.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from footballworld.config.action import ActionScale
from footballworld.config.ball_physics import BallPhysics
from footballworld.config.geometry import Ball
from footballworld.core.state import BallState
from footballworld.dynamics.ball import step_free_ball


class AerialKickLookup(NamedTuple):
    """Strictly increasing aerial distance-to-launch calibration."""

    distance_m: tuple[float, ...]
    launch_speed_mps: tuple[float, ...]
    travel_time_s: tuple[float, ...]
    heading_offset_radians: tuple[float, ...]
    physical_launch_radians: float


class AerialKickControls(NamedTuple):
    """One selected target's normalized controls and reachability."""

    direction: jax.Array
    power: jax.Array
    launch: jax.Array
    travel_time_s: jax.Array
    reachable: jax.Array


_DEFAULT_BALL = Ball()
_DEFAULT_ACTION_SCALE = ActionScale()
_DEFAULT_BALL_PHYSICS = BallPhysics()


def _physical_launch_angle(
    launch_control: float,
    release_height_m: float,
    ball_radius_m: float,
    launch_max_radians: float,
    ground_launch_down_max_radians: float,
    ground_launch_down_reference_height_m: float,
) -> float:
    height_fraction = np.clip(
        (release_height_m - ball_radius_m)
        / (ground_launch_down_reference_height_m - ball_radius_m),
        0.0,
        1.0,
    )
    launch_floor = -(
        ground_launch_down_max_radians
        + (launch_max_radians - ground_launch_down_max_radians) * height_fraction
    )
    launch_unit = 0.5 * (np.clip(launch_control, -1.0, 1.0) + 1.0)
    return float(launch_floor + launch_unit * (launch_max_radians - launch_floor))


@lru_cache(maxsize=24)
def calibrate_aerial_kick_lookup(
    physics_dt_s: float,
    ball_radius_m: float,
    kick_speed_max_mps: float,
    spin_max_radps: float,
    launch_control: float,
    side_spin_control: float,
    back_spin_control: float,
    release_height_m: float,
    launch_max_radians: float,
    ground_launch_down_max_radians: float,
    ground_launch_down_reference_height_m: float,
    physics: BallPhysics,
    *,
    speed_samples: int = 32,
    maximum_steps: int = 450,
) -> AerialKickLookup:
    """Roll exact free-ball flights and invert their first-landing range."""

    theta = _physical_launch_angle(
        launch_control,
        release_height_m,
        ball_radius_m,
        launch_max_radians,
        ground_launch_down_max_radians,
        ground_launch_down_reference_height_m,
    )
    speeds = jnp.linspace(
        4.0,
        kick_speed_max_mps,
        speed_samples,
        dtype=jnp.float32,
    )
    geometry = Ball(radius=ball_radius_m)

    def one(initial_speed):
        power = initial_speed / jnp.float32(kick_speed_max_mps)
        initial = BallState(
            position=jnp.asarray([0.0, 0.0, release_height_m], dtype=jnp.float32),
            velocity=jnp.asarray(
                [
                    initial_speed * jnp.cos(theta),
                    0.0,
                    initial_speed * jnp.sin(theta),
                ],
                dtype=jnp.float32,
            ),
            # For a +x kick, back-spin is -y and side-spin is +z under the
            # authoritative contact transform. Spin generation scales by power.
            spin=jnp.asarray(
                [
                    0.0,
                    -back_spin_control * spin_max_radps * power,
                    side_spin_control * spin_max_radps * power,
                ],
                dtype=jnp.float32,
            ),
            live=jnp.asarray(True),
        )

        def body(state, _):
            stepped = step_free_ball(
                state,
                dt=physics_dt_s,
                geometry=geometry,
                physics=physics,
            )
            return stepped, (
                stepped.position[0],
                stepped.position[1],
                stepped.position[2],
                stepped.velocity[2],
            )

        _, (x, y, z, vz) = jax.lax.scan(body, initial, None, length=maximum_steps)
        above = z > jnp.float32(ball_radius_m + 0.015)
        seen_above = jnp.cumsum(above.astype(jnp.int32)) > 0
        seen_before = jnp.concatenate(
            (jnp.zeros((1,), dtype=jnp.bool_), seen_above[:-1])
        )
        at_ground = (z <= jnp.float32(ball_radius_m + 0.015)) & seen_before
        bounced = (
            jnp.concatenate(
                (
                    jnp.zeros((1,), dtype=jnp.bool_),
                    (vz[:-1] < 0.0) & (vz[1:] >= 0.0),
                )
            )
            & seen_before
        )
        landing = at_ground | bounced
        index = jnp.where(
            jnp.any(landing),
            jnp.argmax(landing),
            jnp.int32(maximum_steps - 1),
        )
        distance = jnp.sqrt(x[index] * x[index] + y[index] * y[index])
        travel_time = (index.astype(jnp.float32) + 1.0) * physics_dt_s
        heading_offset = jnp.arctan2(y[index], x[index])
        return distance, travel_time, heading_offset

    distance, travel_time, heading_offset = jax.jit(
        lambda values: jax.lax.map(one, values), backend="cpu"
    )(speeds)
    distance, travel_time, speeds = map(
        np.asarray, jax.device_get((distance, travel_time, speeds))
    )
    previous = np.maximum.accumulate(distance)
    keep = np.ones(distance.shape, dtype=bool)
    keep[1:] = distance[1:] > previous[:-1] + 1.0e-4
    if int(np.sum(keep)) < 2:
        raise RuntimeError("aerial-kick physics produced no usable lookup")
    return AerialKickLookup(
        distance_m=tuple(float(value) for value in distance[keep]),
        launch_speed_mps=tuple(float(value) for value in speeds[keep]),
        travel_time_s=tuple(float(value) for value in travel_time[keep]),
        heading_offset_radians=tuple(float(value) for value in heading_offset[keep]),
        physical_launch_radians=theta,
    )


# Filled from ``calibrate_aerial_kick_lookup`` on its recorded default 80 Hz grid.
# The compact 32-sample table is a factory-derived approximation, not a fitted
# empirical coefficient. Against a 56-sample dense reference, interpolation's
# maximum absolute errors are 0.028699 m/s launch speed, 0.005981 s travel time,
# and 7.333e-05 rad heading, with identical represented range. It is declared
# below the generator so regeneration remains a mechanical, auditable change.
_SHIPPED_LOOKUP_PHYSICS_DT_S = 1.0 / 80.0
DEFAULT_CROSS_LOOKUP = AerialKickLookup(
    distance_m=(
        1.163550853729248,
        1.8240946531295776,
        2.6181249618530273,
        3.539421319961548,
        4.581065654754639,
        5.731379985809326,
        7.081395149230957,
        8.569137573242188,
        10.228827476501465,
        12.150725364685059,
        14.242027282714844,
        16.441957473754883,
        18.82435417175293,
        21.156414031982422,
        23.635787963867188,
        26.14906120300293,
        28.793678283691406,
        31.364158630371094,
        34.04851150512695,
        36.73653030395508,
        39.41682815551758,
        42.07448959350586,
        44.808250427246094,
        47.40082550048828,
        50.041221618652344,
        52.529808044433594,
        55.04911422729492,
        57.40670394897461,
        59.691864013671875,
        61.90093231201172,
        64.03096008300781,
        66.07958221435547,
    ),
    launch_speed_mps=(
        4.0,
        4.992258071899414,
        5.984516143798828,
        6.976773738861084,
        7.969031810760498,
        8.96129035949707,
        9.953547477722168,
        10.945805549621582,
        11.938063621520996,
        12.93032169342041,
        13.922579765319824,
        14.914836883544922,
        15.907094955444336,
        16.89935302734375,
        17.891611099243164,
        18.883869171142578,
        19.876127243041992,
        20.868385314941406,
        21.86064338684082,
        22.852901458740234,
        23.84515953063965,
        24.837417602539062,
        25.829675674438477,
        26.82193374633789,
        27.814189910888672,
        28.80644989013672,
        29.7987060546875,
        30.790964126586914,
        31.783222198486328,
        32.775482177734375,
        33.767738342285156,
        34.7599983215332,
    ),
    travel_time_s=(
        0.32500001788139343,
        0.4124999940395355,
        0.5,
        0.5875000357627869,
        0.675000011920929,
        0.762499988079071,
        0.862500011920929,
        0.9625000357627869,
        1.0625,
        1.1750000715255737,
        1.287500023841858,
        1.399999976158142,
        1.524999976158142,
        1.6375000476837158,
        1.7625000476837158,
        1.8875000476837158,
        2.0250000953674316,
        2.1500000953674316,
        2.2875001430511475,
        2.424999952316284,
        2.5625,
        2.700000047683716,
        2.8500001430511475,
        2.987499952316284,
        3.137500047683716,
        3.2750000953674316,
        3.424999952316284,
        3.5625,
        3.700000047683716,
        3.8375000953674316,
        3.9750001430511475,
        4.112500190734863,
    ),
    heading_offset_radians=(
        0.002853210549801588,
        0.004456957336515188,
        0.006398946512490511,
        0.008673046715557575,
        0.011271034367382526,
        0.014173462986946106,
        0.01760626584291458,
        0.021391088142991066,
        0.025568269193172455,
        0.030391378328204155,
        0.03565407916903496,
        0.04129563644528389,
        0.047576796263456345,
        0.05387382209300995,
        0.06076057255268097,
        0.06792502105236053,
        0.0756811872124672,
        0.08338914811611176,
        0.09167590737342834,
        0.10020537674427032,
        0.10895735025405884,
        0.11790609359741211,
        0.12739789485931396,
        0.1366969347000122,
        0.1464901715517044,
        0.15604864060878754,
        0.1660618931055069,
        0.1757965236902237,
        0.18558959662914276,
        0.19542089104652405,
        0.20527130365371704,
        0.2151211053133011,
    ),
    physical_launch_radians=0.42525,
)


def aerial_kick_lookup(
    physics_dt_s: float,
    ball_radius_m: float,
    scale: ActionScale,
    launch_control: float,
    side_spin_control: float,
    back_spin_control: float,
    physics: BallPhysics,
) -> AerialKickLookup:
    """Return the shipped cross table or calibrate a custom configuration."""

    key = (
        physics_dt_s,
        ball_radius_m,
        scale.kick_speed_max_mps,
        scale.spin_max_radps,
        launch_control,
        abs(side_spin_control),
        back_spin_control,
        ball_radius_m,
        scale.launch_max_radians,
        scale.ground_launch_down_max_radians,
        scale.ground_launch_down_reference_height_m,
        physics,
    )
    default_key = (
        _SHIPPED_LOOKUP_PHYSICS_DT_S,
        _DEFAULT_BALL.radius,
        _DEFAULT_ACTION_SCALE.kick_speed_max_mps,
        _DEFAULT_ACTION_SCALE.spin_max_radps,
        0.05,
        0.18,
        0.35,
        _DEFAULT_BALL.radius,
        _DEFAULT_ACTION_SCALE.launch_max_radians,
        _DEFAULT_ACTION_SCALE.ground_launch_down_max_radians,
        _DEFAULT_ACTION_SCALE.ground_launch_down_reference_height_m,
        _DEFAULT_BALL_PHYSICS,
    )
    if key == default_key and DEFAULT_CROSS_LOOKUP is not None:
        return DEFAULT_CROSS_LOOKUP
    return calibrate_aerial_kick_lookup(*key)


def aerial_kick_controls(
    source_xy: jax.Array,
    target_xy: jax.Array,
    incoming_velocity: jax.Array,
    ball_height_m: jax.Array,
    ball_radius_m: float,
    side_spin_control: jax.Array,
    lookup: AerialKickLookup,
    scale: ActionScale,
) -> AerialKickControls:
    """Invert one lookup row and compensate the observed incoming velocity."""

    delta = jnp.asarray(target_xy, dtype=jnp.float32) - jnp.asarray(
        source_xy, dtype=jnp.float32
    )
    distance = jnp.linalg.norm(delta)
    target_direction = delta / jnp.maximum(distance, jnp.float32(1.0e-6))
    lookup_distance = jnp.asarray(lookup.distance_m, dtype=jnp.float32)
    launch_speed = jnp.interp(
        distance,
        lookup_distance,
        jnp.asarray(lookup.launch_speed_mps, dtype=jnp.float32),
    )
    travel_time = jnp.interp(
        distance,
        lookup_distance,
        jnp.asarray(lookup.travel_time_s, dtype=jnp.float32),
    )
    heading_offset = jnp.interp(
        distance,
        lookup_distance,
        jnp.asarray(lookup.heading_offset_radians, dtype=jnp.float32),
    ) * jnp.sign(jnp.asarray(side_spin_control, dtype=jnp.float32))
    offset_cos = jnp.cos(heading_offset)
    offset_sin = jnp.sin(heading_offset)
    # A positive lookup spin curves a +x launch toward +y. Rotate the initial
    # horizontal direction by the opposite simulated landing offset so the
    # curved trajectory, rather than its tangent, reaches the requested target.
    launch_direction = jnp.asarray(
        [
            offset_cos * target_direction[0] + offset_sin * target_direction[1],
            -offset_sin * target_direction[0] + offset_cos * target_direction[1],
        ],
        dtype=jnp.float32,
    )
    theta = jnp.asarray(lookup.physical_launch_radians, dtype=jnp.float32)
    desired_velocity = launch_speed * jnp.asarray(
        [
            jnp.cos(theta) * launch_direction[0],
            jnp.cos(theta) * launch_direction[1],
            jnp.sin(theta),
        ],
        dtype=jnp.float32,
    )
    impulse = desired_velocity - jnp.asarray(incoming_velocity, dtype=jnp.float32)
    impulse_speed = jnp.linalg.norm(impulse)
    horizontal = jnp.linalg.norm(impulse[:2])
    direction = jnp.where(
        horizontal > jnp.float32(1.0e-6),
        impulse[:2] / jnp.maximum(horizontal, jnp.float32(1.0e-6)),
        target_direction,
    )
    impulse_angle = jnp.arctan2(impulse[2], horizontal)
    radius = jnp.float32(ball_radius_m)
    height_span = jnp.maximum(
        jnp.float32(scale.ground_launch_down_reference_height_m) - radius,
        jnp.float32(1.0e-6),
    )
    height_fraction = (
        jnp.clip(
            jnp.asarray(ball_height_m, dtype=jnp.float32) - radius,
            0.0,
            height_span,
        )
        / height_span
    )
    launch_floor = -(
        scale.ground_launch_down_max_radians
        + (scale.launch_max_radians - scale.ground_launch_down_max_radians)
        * height_fraction
    )
    launch_unit = (impulse_angle - launch_floor) / jnp.maximum(
        scale.launch_max_radians - launch_floor,
        jnp.float32(1.0e-6),
    )
    launch = jnp.clip(2.0 * launch_unit - 1.0, -1.0, 1.0)
    reachable = (
        (distance >= lookup_distance[0])
        & (distance <= lookup_distance[-1])
        & (impulse_speed <= scale.kick_speed_max_mps + jnp.float32(1.0e-5))
        & (launch_unit >= 0.0)
        & (launch_unit <= 1.0)
    )
    return AerialKickControls(
        direction=direction.astype(jnp.float32),
        power=jnp.clip(impulse_speed / scale.kick_speed_max_mps, 0.0, 1.0),
        launch=launch.astype(jnp.float32),
        travel_time_s=travel_time.astype(jnp.float32),
        reachable=reachable,
    )


__all__ = [
    "DEFAULT_CROSS_LOOKUP",
    "AerialKickControls",
    "AerialKickLookup",
    "aerial_kick_controls",
    "aerial_kick_lookup",
    "calibrate_aerial_kick_lookup",
]
