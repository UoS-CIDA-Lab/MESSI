"""Factory-time ground-pass calibration against FootballWorld ball physics.

The lookup is built once per immutable environment configuration, never inside
the observation policy graph.  Runtime pass control therefore reduces to
shape-static interpolation while retaining the exact slide, roll, spin, and
settling transitions owned by dynamics.ball.
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


class GroundPassLookup(NamedTuple):
    """Strictly increasing distance-to-launch-speed calibration table."""

    distance_m: tuple[float, ...]
    launch_speed_mps: tuple[float, ...]
    travel_time_s: tuple[float, ...]


# Generated once from ``calibrate_ground_pass_lookup`` at its recorded default 80 Hz
# physics grid. Keeping the float32 results verbatim avoids a fresh 56 x 900
# JAX scan for that exact legacy grid; other grids are calibrated and cached.
_SHIPPED_LOOKUP_PHYSICS_DT_S = 1.0 / 80.0
DEFAULT_GROUND_PASS_LOOKUP = GroundPassLookup(
    distance_m=(
        0.08696351945400238,
        1.0838634967803955,
        2.241891384124756,
        3.47408127784729,
        4.780435562133789,
        6.074097156524658,
        7.528907299041748,
        9.057881355285645,
        10.661017417907715,
        12.251442909240723,
        14.002230644226074,
        15.73869800567627,
        17.445615768432617,
        19.311166763305664,
        21.064754486083984,
        22.856130599975586,
        24.724660873413086,
        26.65365982055664,
        28.522977828979492,
        30.482601165771484,
        32.5889892578125,
        34.6674690246582,
        36.7561149597168,
        38.94794464111328,
        41.197593688964844,
        43.35761260986328,
        45.80525588989258,
        48.22346878051758,
        50.69913101196289,
        53.0699348449707,
        55.569129943847656,
        58.212074279785156,
        60.738006591796875,
        63.49005126953125,
        66.2110824584961,
        68.89016723632812,
        71.80620574951172,
        74.69062805175781,
        77.71749114990234,
        80.51241302490234,
        83.64303588867188,
        86.7378158569336,
        89.75933837890625,
        92.95262145996094,
        96.1969985961914,
        99.49217987060547,
        102.61180114746094,
        106.00508117675781,
        109.44901275634766,
        112.70577239990234,
        116.24748992919922,
        119.8398208618164,
        123.23342895507812,
        126.92404174804688,
        130.5795440673828,
        134.37374877929688,
    ),
    launch_speed_mps=(
        7.0,
        7.504727363586426,
        8.009454727172852,
        8.514181137084961,
        9.018909454345703,
        9.523635864257812,
        10.028363227844238,
        10.533090591430664,
        11.03781795501709,
        11.542545318603516,
        12.047271728515625,
        12.552000045776367,
        13.056726455688477,
        13.561453819274902,
        14.066181182861328,
        14.570908546447754,
        15.07563591003418,
        15.580363273620605,
        16.08509063720703,
        16.58981704711914,
        17.094545364379883,
        17.599271774291992,
        18.104000091552734,
        18.608726501464844,
        19.113452911376953,
        19.618181228637695,
        20.122907638549805,
        20.627635955810547,
        21.132362365722656,
        21.6370906829834,
        22.141817092895508,
        22.64654541015625,
        23.15127182006836,
        23.65599822998047,
        24.16072654724121,
        24.66545295715332,
        25.170181274414062,
        25.674907684326172,
        26.179636001586914,
        26.684362411499023,
        27.189090728759766,
        27.693817138671875,
        28.198543548583984,
        28.703271865844727,
        29.207998275756836,
        29.712726593017578,
        30.217453002929688,
        30.72218132019043,
        31.22690773010254,
        31.73163414001465,
        32.23636245727539,
        32.7410888671875,
        33.24581527709961,
        33.750545501708984,
        34.255271911621094,
        34.7599983215332,
    ),
    travel_time_s=(
        0.012500000186264515,
        0.15000000596046448,
        0.30000001192092896,
        0.45000001788139343,
        0.6000000238418579,
        0.737500011920929,
        0.887499988079071,
        1.037500023841858,
        1.1875,
        1.3250000476837158,
        1.475000023841858,
        1.6125000715255737,
        1.7375000715255737,
        1.875,
        1.9875000715255737,
        2.1000001430511475,
        2.2125000953674316,
        2.325000047683716,
        2.424999952316284,
        2.5250000953674316,
        2.637500047683716,
        2.737499952316284,
        2.8375000953674316,
        2.9375,
        3.0375001430511475,
        3.125,
        3.237499952316284,
        3.3375000953674316,
        3.4375,
        3.5250000953674316,
        3.612499952316284,
        3.7125000953674316,
        3.799999952316284,
        3.9000000953674316,
        3.987499952316284,
        4.075000286102295,
        4.175000190734863,
        4.262500286102295,
        4.362500190734863,
        4.4375,
        4.537499904632568,
        4.625,
        4.712500095367432,
        4.800000190734863,
        4.887500286102295,
        4.974999904632568,
        5.050000190734863,
        5.137500286102295,
        5.224999904632568,
        5.300000190734863,
        5.387500286102295,
        5.474999904632568,
        5.550000190734863,
        5.637500286102295,
        5.712500095367432,
        5.800000190734863,
    ),
)

_DEFAULT_BALL = Ball()
_DEFAULT_ACTION_SCALE = ActionScale()
_DEFAULT_BALL_PHYSICS = BallPhysics()


@lru_cache(maxsize=16)
def calibrate_ground_pass_lookup(
    physics_dt_s: float,
    ball_radius_m: float,
    kick_speed_max_mps: float,
    desired_arrival_speed_mps: float,
    physics: BallPhysics,
    *,
    speed_samples: int = 56,
    maximum_steps: int = 900,
) -> GroundPassLookup:
    """Roll out exact default-spin drives and invert their controlled range.

    Inputs are immutable public environment coefficients, making this cache
    independent of match state and observations.  Record-high filtering keeps
    interpolation monotone even if a discrete transition produces a tiny local
    range reversal.
    """

    speeds = jnp.linspace(
        desired_arrival_speed_mps,
        kick_speed_max_mps,
        speed_samples,
        dtype=jnp.float32,
    )
    geometry = Ball(radius=ball_radius_m)

    def one(initial_speed):
        initial = BallState(
            position=jnp.asarray([0.0, 0.0, ball_radius_m], dtype=jnp.float32),
            velocity=jnp.asarray([initial_speed, 0.0, 0.0], dtype=jnp.float32),
            spin=jnp.zeros((3,), dtype=jnp.float32),
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
                jnp.linalg.norm(stepped.velocity[:2]),
            )

        _, (position_x, horizontal_speed) = jax.lax.scan(
            body, initial, None, length=maximum_steps
        )
        arrived = horizontal_speed <= desired_arrival_speed_mps
        index = jnp.where(
            jnp.any(arrived),
            jnp.argmax(arrived),
            jnp.int32(maximum_steps - 1),
        )
        return (
            position_x[index],
            (index.astype(jnp.float32) + 1.0) * physics_dt_s,
        )

    # Keep one-time calibration off an accelerator reserved for rollout/render.
    distance, travel_time = jax.jit(jax.vmap(one), backend="cpu")(speeds)
    distance, travel_time, speeds = map(
        np.asarray, jax.device_get((distance, travel_time, speeds))
    )
    previous = np.maximum.accumulate(distance)
    keep = np.ones(distance.shape, dtype=bool)
    keep[1:] = distance[1:] > previous[:-1] + 1.0e-4
    if int(np.sum(keep)) < 2:
        raise RuntimeError("ground-pass physics produced no usable lookup")
    return GroundPassLookup(
        distance_m=tuple(float(value) for value in distance[keep]),
        launch_speed_mps=tuple(float(value) for value in speeds[keep]),
        travel_time_s=tuple(float(value) for value in travel_time[keep]),
    )


def ground_pass_lookup(
    physics_dt_s: float,
    ball_radius_m: float,
    kick_speed_max_mps: float,
    desired_arrival_speed_mps: float,
    physics: BallPhysics,
) -> GroundPassLookup:
    """Return the shipped default table or calibrate a custom environment.

    Exact equality is intentional: the precomputed table is valid only for
    the complete default calibration key.  The existing LRU-cached solver
    remains authoritative for every custom timebase, ball, action scale, or
    ball-physics configuration.
    """

    default_key = (
        _SHIPPED_LOOKUP_PHYSICS_DT_S,
        _DEFAULT_BALL.radius,
        _DEFAULT_ACTION_SCALE.kick_speed_max_mps,
        7.0,
        _DEFAULT_BALL_PHYSICS,
    )
    key = (
        physics_dt_s,
        ball_radius_m,
        kick_speed_max_mps,
        desired_arrival_speed_mps,
        physics,
    )
    if key == default_key:
        return DEFAULT_GROUND_PASS_LOOKUP
    return calibrate_ground_pass_lookup(*key)


__all__ = [
    "DEFAULT_GROUND_PASS_LOOKUP",
    "GroundPassLookup",
    "calibrate_ground_pass_lookup",
    "ground_pass_lookup",
]
