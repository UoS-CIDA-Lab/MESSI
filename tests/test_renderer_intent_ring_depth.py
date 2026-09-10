"""Focused host-rendering contract for flat, player-occluded intent rings.

FootballWorld projects ground rings from z=0 and paints them before player
bodies so occlusion, requested-intent semantics, and configured reach radii
remain visually consistent.
"""

import numpy as np

from footballworld.rendering.renderer import (
    _INTENT_RING_TURF_HEIGHT_M,
    _INTENT_RING_UNDERLAY_ZORDER,
    _INTENT_RING_ZORDER,
    _PLAYER_MARKER_ZORDER,
    _intent_ring_world_vertices,
    _player_intent_painter_order,
)


def test_intent_rings_are_flat_on_turf_and_painted_before_player_spheres():
    position = np.asarray(((3.0, -4.0), (-8.0, 7.5)), dtype=np.float32)
    radius = np.asarray((1.25, 2.0), dtype=np.float32)
    ring_unit = np.asarray(
        ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)),
        dtype=np.float32,
    )

    world = _intent_ring_world_vertices(position, radius, ring_unit)

    assert world.shape == (2, 4, 3)
    np.testing.assert_array_equal(
        world[..., 2],
        np.full(world.shape[:2], _INTENT_RING_TURF_HEIGHT_M, dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.linalg.norm(world[..., :2] - position[:, None, :], axis=-1),
        np.broadcast_to(radius[:, None], world.shape[:2]),
        rtol=0.0,
        atol=1e-6,
    )

    fan, shadow, underlay, ring, player = (object() for _ in range(5))
    order = _player_intent_painter_order(fan, shadow, underlay, ring, player)
    assert order == (fan, shadow, underlay, ring, player)
    assert order.index(underlay) < order.index(player)
    assert order.index(ring) < order.index(player)
    assert _INTENT_RING_UNDERLAY_ZORDER < _PLAYER_MARKER_ZORDER
    assert _INTENT_RING_ZORDER < _PLAYER_MARKER_ZORDER
