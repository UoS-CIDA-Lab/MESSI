"""A restart spot must land on or inside the field of play.

Every restart kind but the free kick already produced a spot on or inside the
boundary: throw-ins clamp along the touchline, corners and goal kicks and
penalties are fixed marks, and a kickoff is the centre. The free kick took the
offence point verbatim, and an offence can occur with the offender past a
line. A spot one millimetre outside is not a cosmetic error: the restart layout
audit allows a taker to stand off the pitch only through ``external_taker``,
which is itself gated on the ball being inside it, so every candidate layout is
rejected, the projector fails closed, the taker never approaches, the
forced-release countdown never starts, and the match hangs with no recovery.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from footballworld.config.geometry import Ball, Stadium
from footballworld.core.constants import (
    RK_CORNER,
    RK_FREEKICK,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_OFFSIDE,
    RK_PENALTY,
    RK_THROWIN,
)
from footballworld.rules.restart_spot import canonical_restart_spot

_STADIUM = Stadium()
_BALL = Ball()
_ATTACK = jnp.asarray([1.0, -1.0], dtype=jnp.float32)
_KINDS = (
    RK_KICKOFF,
    RK_THROWIN,
    RK_GOALKICK,
    RK_CORNER,
    RK_FREEKICK,
    RK_PENALTY,
    RK_OFFSIDE,
)

# Offence points outside the field, including the corner the collector actually
# deadlocked on, and points past a touchline and past both lines at once.
_OUTSIDE = (
    (52.60, 33.93),
    (-52.60, 33.93),
    (10.0, 34.40),
    (10.0, -34.40),
    (60.0, 40.0),
    (-60.0, -40.0),
    (52.501, 0.0),
)


def _spot(kind, team, point, *, indirect=False):
    return np.asarray(
        canonical_restart_spot(
            jnp.int32(kind),
            jnp.int32(team),
            jnp.asarray([point[0], point[1], 0.0], dtype=jnp.float32),
            _ATTACK,
            indirect=jnp.bool_(indirect),
            stadium=_STADIUM,
            ball=_BALL,
        )
    )


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("point", _OUTSIDE)
@pytest.mark.parametrize("team", (0, 1))
def test_every_restart_spot_is_on_or_inside_the_field(kind, point, team):
    spot = _spot(kind, team, point)
    assert abs(spot[0]) <= _STADIUM.half_length + 1e-6
    assert abs(spot[1]) <= _STADIUM.half_width + 1e-6


@pytest.mark.parametrize("point", _OUTSIDE)
@pytest.mark.parametrize("team", (0, 1))
def test_indirect_free_kick_spot_is_on_or_inside_the_field(point, team):
    spot = _spot(RK_FREEKICK, team, point, indirect=True)
    assert abs(spot[0]) <= _STADIUM.half_length + 1e-6
    assert abs(spot[1]) <= _STADIUM.half_width + 1e-6


@pytest.mark.parametrize("team", (0, 1))
def test_free_kick_spot_inside_the_field_is_untouched(team):
    """The clamp must not move a lawful spot, or it would rewrite Law 13."""

    for point in ((0.0, 0.0), (30.0, -12.5), (-52.4, 33.9), (52.5, 34.0)):
        spot = _spot(RK_FREEKICK, team, point)
        assert spot[0] == pytest.approx(point[0], abs=1e-5)
        assert spot[1] == pytest.approx(point[1], abs=1e-5)
        assert spot[2] == pytest.approx(_BALL.radius, abs=1e-6)


@pytest.mark.parametrize("team", (0, 1))
def test_free_kick_spot_is_brought_to_the_nearest_boundary_point(team):
    """An off-field offence restarts at the nearest point on the boundary."""

    spot = _spot(RK_FREEKICK, team, (52.60, 33.93))
    assert spot[0] == pytest.approx(_STADIUM.half_length, abs=1e-5)
    assert spot[1] == pytest.approx(33.93, abs=1e-5)

    spot = _spot(RK_FREEKICK, team, (10.0, -40.0))
    assert spot[0] == pytest.approx(10.0, abs=1e-5)
    assert spot[1] == pytest.approx(-_STADIUM.half_width, abs=1e-5)
