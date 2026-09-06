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


# Generated once from ``calibrate_ground_pass_lookup`` at its recorded 90 Hz
# physics grid. Keeping the float32 results verbatim avoids a fresh 56 x 900
# JAX scan for that exact legacy grid; other grids are calibrated and cached.
_SHIPPED_LOOKUP_PHYSICS_DT_S = 1.0 / 90.0
DEFAULT_GROUND_PASS_LOOKUP = GroundPassLookup(
    distance_m=(
        0.07735389471054077,
        1.1228939294815063,
        2.2426071166992188,
        3.436493396759033,
        4.704553127288818,
        6.124203681945801,
        7.540707111358643,
        9.03138256072998,
        10.596234321594238,
        12.312641143798828,
        14.025121688842773,
        15.729137420654297,
        17.419677734375,
        19.26119613647461,
        21.073545455932617,
        22.864294052124023,
        24.73259735107422,
        26.63753318786621,
        28.521862030029297,
        30.46224594116211,
        32.57223129272461,
        34.62665939331055,
        36.735599517822266,
        39.02266311645508,
        41.165367126464844,
        43.438758850097656,
        45.899169921875,
        48.20500946044922,
        50.5640983581543,
        53.05354690551758,
        55.66610336303711,
        58.187007904052734,
        60.76016616821289,
        63.543113708496094,
        66.22370910644531,
        68.95602416992188,
        71.7397689819336,
        74.74600982666016,
        77.63570404052734,
        80.57623291015625,
        83.74634552001953,
        86.70824432373047,
        89.79446411132812,
        93.04142761230469,
        96.22476959228516,
        99.37744903564453,
        102.65412139892578,
        106.10494995117188,
        109.40088653564453,
        112.7433090209961,
        116.42450714111328,
        119.86334228515625,
        123.34857177734375,
        126.88096618652344,
        130.6894073486328,
        134.32058715820312,
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
        0.011111111380159855,
        0.15555556118488312,
        0.30000001192092896,
        0.4444444477558136,
        0.5888888835906982,
        0.7444444894790649,
        0.8888888955116272,
        1.0333333015441895,
        1.1777777671813965,
        1.3333333730697632,
        1.4777778387069702,
        1.6111111640930176,
        1.7333333492279053,
        1.8666666746139526,
        1.9888889789581299,
        2.1000001430511475,
        2.211111068725586,
        2.3222222328186035,
        2.422222375869751,
        2.5222222805023193,
        2.633333444595337,
        2.7333333492279053,
        2.8333334922790527,
        2.944444417953491,
        3.0333333015441895,
        3.133333444595337,
        3.2444446086883545,
        3.3333334922790527,
        3.422222375869751,
        3.5222222805023193,
        3.622222423553467,
        3.711111307144165,
        3.8000001907348633,
        3.9000000953674316,
        3.98888897895813,
        4.077777862548828,
        4.1666669845581055,
        4.266666889190674,
        4.355555534362793,
        4.44444465637207,
        4.544444561004639,
        4.622222423553467,
        4.711111068725586,
        4.800000190734863,
        4.888888835906982,
        4.9666666984558105,
        5.055555820465088,
        5.144444465637207,
        5.222222328186035,
        5.300000190734863,
        5.400000095367432,
        5.47777795791626,
        5.555555820465088,
        5.633333683013916,
        5.722222328186035,
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
