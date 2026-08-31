"""JAX-compatible input commands for one SoccerWorld control frame.

The engine consumes one fixed PyTree instead of growing positional step arguments. Empty commands
use masks and sentinels, not ``None`` or variable-length Python containers, so one schema works
under ``jit`` and ``vmap``. Rule and policy adapters may construct these values, but only the
environment decides whether a proposal is legal and can be applied.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from soccerworld._schema_versions import COMMAND_SCHEMA

NO_PLAYER = -1
NO_FORMATION = -1

__all__ = [
    "COMMAND_SCHEMA",
    "FormationCommand",
    "SetPieceTakerCommand",
    "StepCommand",
    "SubstitutionCommand",
]


def _safe_axis_index(name: str, value, size: int):
    """Validate a host address and fail closed for a traced invalid address."""

    traced = any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(value))
    if not traced:
        try:
            source = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a scalar non-boolean integer") from exc
        if source.shape != ():
            raise ValueError(f"{name} must be scalar, got shape {source.shape}")
        if not np.issubdtype(source.dtype, np.integer) or np.issubdtype(source.dtype, np.bool_):
            raise TypeError(f"{name} must have a non-boolean integer dtype, got {source.dtype}")
        integer = int(source)
        if not 0 <= integer < size:
            raise ValueError(f"{name} must lie in [0, {size}), got {integer}")
        return jnp.int32(integer), jnp.bool_(True)

    array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar, got shape {array.shape}")
    if not jnp.issubdtype(array.dtype, jnp.integer) or jnp.issubdtype(array.dtype, jnp.bool_):
        raise TypeError(f"{name} must have a non-boolean integer dtype, got {array.dtype}")
    valid = (array >= 0) & (array < size)
    safe = jnp.clip(array, 0, size - 1).astype(jnp.int32)
    return safe, valid


def _set_if_valid(array, index, value, valid):
    """Apply one scalar update without letting an invalid traced address alias."""

    current = array[index]
    replacement = jnp.asarray(value, dtype=array.dtype)
    return array.at[index].set(jnp.where(valid, replacement, current))


class SubstitutionCommand(NamedTuple):
    """Fixed-width substitution proposals for both teams.

    Every field has shape ``(2, K)``.  ``requested`` is authoritative; sentinel values in an
    unrequested cell are ignored.  Multiple requested cells form one atomic substitution window.
    """

    requested: Array
    out_slot: Array
    bench_index: Array

    @classmethod
    def empty(cls, max_simultaneous: int) -> SubstitutionCommand:
        shape = (2, max_simultaneous)
        return cls(
            requested=jnp.zeros(shape, dtype=jnp.bool_),
            out_slot=jnp.full(shape, NO_PLAYER, dtype=jnp.int32),
            bench_index=jnp.full(shape, NO_PLAYER, dtype=jnp.int32),
        )

    def with_request(
        self,
        team: int | Array,
        command_index: int | Array,
        *,
        out_slot: int | Array,
        bench_index: int | Array,
    ) -> SubstitutionCommand:
        """Return a command with one substitution proposal enabled.

        This only constructs a proposal. Roster, goalkeeper, window, and limit legality remains
        authoritative inside the environment transition. Invalid eager table addresses raise;
        traced out-of-range addresses are no-ops instead of negative-index aliases.
        """

        team_index, team_valid = _safe_axis_index("team", team, self.requested.shape[0])
        command_slot, command_valid = _safe_axis_index(
            "command_index", command_index, self.requested.shape[1]
        )
        index = (team_index, command_slot)
        valid = team_valid & command_valid
        return SubstitutionCommand(
            requested=_set_if_valid(self.requested, index, True, valid),
            out_slot=_set_if_valid(self.out_slot, index, out_slot, valid),
            bench_index=_set_if_valid(self.bench_index, index, bench_index, valid),
        )


class FormationCommand(NamedTuple):
    """Per-team formation proposals, each with shape ``(2,)``."""

    requested: Array
    layout_index: Array

    @classmethod
    def empty(cls) -> FormationCommand:
        return cls(
            requested=jnp.zeros((2,), dtype=jnp.bool_),
            layout_index=jnp.full((2,), NO_FORMATION, dtype=jnp.int32),
        )

    def with_request(self, team: int | Array, *, layout_index: int | Array) -> FormationCommand:
        """Return a formation proposal; a traced invalid team is a fail-closed no-op."""

        team_index, valid = _safe_axis_index("team", team, self.requested.shape[0])
        return FormationCommand(
            requested=_set_if_valid(self.requested, team_index, True, valid),
            layout_index=_set_if_valid(self.layout_index, team_index, layout_index, valid),
        )


class SetPieceTakerCommand(NamedTuple):
    """Per-team, per-restart-kind taker proposals.

    Fields have shape ``(2, R)`` where ``R`` is the restart vocabulary size fixed by the environment
    schema.  Supplying the complete table lets an external controller nominate a taker even when a
    restart is created inside the upcoming physics scan.
    """

    requested: Array
    player_slot: Array

    @classmethod
    def empty(cls, restart_kind_count: int) -> SetPieceTakerCommand:
        shape = (2, restart_kind_count)
        return cls(
            requested=jnp.zeros(shape, dtype=jnp.bool_),
            player_slot=jnp.full(shape, NO_PLAYER, dtype=jnp.int32),
        )

    def with_request(
        self,
        team: int | Array,
        restart_kind: int | Array,
        *,
        player_slot: int | Array,
    ) -> SetPieceTakerCommand:
        """Return a taker proposal; traced invalid table addresses are no-ops."""

        team_index, team_valid = _safe_axis_index("team", team, self.requested.shape[0])
        kind_index, kind_valid = _safe_axis_index(
            "restart_kind", restart_kind, self.requested.shape[1]
        )
        index = (team_index, kind_index)
        valid = team_valid & kind_valid
        return SetPieceTakerCommand(
            requested=_set_if_valid(self.requested, index, True, valid),
            player_slot=_set_if_valid(self.player_slot, index, player_slot, valid),
        )


class StepCommand(NamedTuple):
    """All externally supplied decisions for one control frame.

    ``player_actions`` has shape ``(N, A)``.  Policy selection and composition happen outside the
    physics engine; the resulting action tensor enters through the same path whether it came from a
    rule policy, a learned policy, replay data, or a human controller.
    """

    player_actions: Array
    substitutions: SubstitutionCommand
    formations: FormationCommand
    set_piece_takers: SetPieceTakerCommand

    def with_player_actions(self, player_actions: Array) -> StepCommand:
        """Return a command with a new action tensor and unchanged manager decisions."""

        return self._replace(player_actions=player_actions)

    def with_substitutions(self, substitutions: SubstitutionCommand) -> StepCommand:
        return self._replace(substitutions=substitutions)

    def with_formations(self, formations: FormationCommand) -> StepCommand:
        return self._replace(formations=formations)

    def with_set_piece_takers(self, set_piece_takers: SetPieceTakerCommand) -> StepCommand:
        return self._replace(set_piece_takers=set_piece_takers)
