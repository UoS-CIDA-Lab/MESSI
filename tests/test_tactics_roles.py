"""Formation-relative role classification contracts."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from footballworld.environment.tactics import (
    ROLE_CENTRE_BACK,
    ROLE_CENTRE_FORWARD,
    ROLE_CENTRE_MIDFIELDER,
    ROLE_FULL_BACK,
    ROLE_GOALKEEPER,
    ROLE_WIDE_MIDFIELDER,
    classify_formation_roles,
)


def test_four_line_4231_keeps_attacking_midfield_distinct_from_striker():
    """Only the most advanced line is forward in a four-line formation."""

    anchors = jnp.asarray(
        (
            (-50.0, 0.0),
            (-35.0, -24.0),
            (-35.0, -8.0),
            (-35.0, 8.0),
            (-35.0, 24.0),
            (-23.0, -10.0),
            (-23.0, 10.0),
            (-12.0, -24.0),
            (-12.0, 0.0),
            (-12.0, 24.0),
            (-5.0, 0.0),
        ),
        dtype=jnp.float32,
    )
    roles = classify_formation_roles(
        anchors,
        jnp.zeros((11,), dtype=jnp.int32),
        jnp.asarray((True,) + (False,) * 10, dtype=jnp.bool_),
    )

    np.testing.assert_array_equal(
        np.asarray(roles),
        np.asarray(
            (
                ROLE_GOALKEEPER,
                ROLE_FULL_BACK,
                ROLE_CENTRE_BACK,
                ROLE_CENTRE_BACK,
                ROLE_FULL_BACK,
                ROLE_CENTRE_MIDFIELDER,
                ROLE_CENTRE_MIDFIELDER,
                ROLE_WIDE_MIDFIELDER,
                ROLE_CENTRE_MIDFIELDER,
                ROLE_WIDE_MIDFIELDER,
                ROLE_CENTRE_FORWARD,
            ),
            dtype=np.int32,
        ),
    )
