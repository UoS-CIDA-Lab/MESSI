"""Mechanism-defined tactical plans shared by rule player and manager policies.

Continuous axes such as line height, tempo, width, aggression, and directness
do not by themselves guarantee a build-up mechanism. The policy keeps those
axes as internal bundle components and names plans after the observable
structure they create. Every number below is an uncalibrated design prior;
none is presented as a tracking-data fit.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.core.randomness import validate_prng_key

_TACTICAL_SELECTION_STREAM = 0x5450434C  # ASCII "TPCL".


class TacticalPlan(str, Enum):
    """Rule-policy plans whose names state their implemented mechanism."""

    SALIDA_LAVOLPIANA = "salida_lavolpiana"
    """A central midfielder drops between centre backs as full backs advance."""

    JUEGO_DE_POSICION = "juego_de_posicion"
    """Wide and half-space occupation preserves three staggered pass lines."""

    GEGENPRESS = "gegenpress"
    """A compact local squeeze follows a recent observed possession loss."""

    CATENACCIO = "catenaccio"
    """A deeper compact block protects the centre before vertical transition."""

    ZONA_MISTA = "zona_mista"
    """A zonal block adds wide-role marking and wide progression."""


@dataclass(frozen=True, slots=True)
class TacticalProfile:
    """Fixed-shape numeric bundle resolved on the host before policy tracing."""

    progressive_pass_gain: float
    wide_pass_gain: float
    attack_depth_scale: float
    attack_width_scale: float
    defend_depth_scale: float
    defend_width_scale: float
    defend_line_shift_m: float
    counterpress_gain: float
    pivot_drop_m: float
    fullback_advance_m: float
    halfspace_gain: float
    overlap_run_m: float
    mixed_wide_mark_gain: float
    settled_pressure_count: float


# These bundles intentionally express relative policy differences only. Their
# values belong to the current action, observation, stamina, and contact
# semantics and must be calibrated as a coupled policy model.
_TACTICAL_PROFILES = MappingProxyType(
    {
        TacticalPlan.SALIDA_LAVOLPIANA: TacticalProfile(
            progressive_pass_gain=0.14,
            wide_pass_gain=0.07,
            attack_depth_scale=1.08,
            attack_width_scale=1.10,
            defend_depth_scale=0.88,
            defend_width_scale=0.90,
            defend_line_shift_m=0.0,
            counterpress_gain=0.85,
            pivot_drop_m=5.0,
            fullback_advance_m=4.5,
            halfspace_gain=0.25,
            overlap_run_m=10.0,
            mixed_wide_mark_gain=0.0,
            settled_pressure_count=1.0,
        ),
        TacticalPlan.JUEGO_DE_POSICION: TacticalProfile(
            progressive_pass_gain=0.10,
            wide_pass_gain=0.08,
            attack_depth_scale=1.08,
            attack_width_scale=1.16,
            defend_depth_scale=0.86,
            defend_width_scale=0.88,
            defend_line_shift_m=0.0,
            counterpress_gain=1.00,
            pivot_drop_m=0.0,
            fullback_advance_m=1.5,
            halfspace_gain=0.72,
            overlap_run_m=7.0,
            mixed_wide_mark_gain=0.0,
            settled_pressure_count=2.0,
        ),
        TacticalPlan.GEGENPRESS: TacticalProfile(
            progressive_pass_gain=0.09,
            wide_pass_gain=0.04,
            attack_depth_scale=1.10,
            attack_width_scale=1.06,
            defend_depth_scale=0.92,
            defend_width_scale=0.90,
            defend_line_shift_m=2.5,
            counterpress_gain=1.45,
            pivot_drop_m=0.0,
            fullback_advance_m=2.0,
            halfspace_gain=0.35,
            overlap_run_m=6.0,
            mixed_wide_mark_gain=0.0,
            settled_pressure_count=3.0,
        ),
        TacticalPlan.CATENACCIO: TacticalProfile(
            progressive_pass_gain=0.16,
            wide_pass_gain=0.03,
            attack_depth_scale=1.03,
            attack_width_scale=1.02,
            defend_depth_scale=0.76,
            defend_width_scale=0.78,
            defend_line_shift_m=-6.0,
            counterpress_gain=0.42,
            pivot_drop_m=0.0,
            fullback_advance_m=0.0,
            halfspace_gain=0.15,
            overlap_run_m=0.0,
            mixed_wide_mark_gain=0.0,
            settled_pressure_count=1.0,
        ),
        TacticalPlan.ZONA_MISTA: TacticalProfile(
            progressive_pass_gain=0.10,
            wide_pass_gain=0.16,
            attack_depth_scale=1.07,
            attack_width_scale=1.13,
            defend_depth_scale=0.84,
            defend_width_scale=0.84,
            defend_line_shift_m=-1.5,
            counterpress_gain=0.86,
            pivot_drop_m=0.0,
            fullback_advance_m=2.5,
            halfspace_gain=0.42,
            overlap_run_m=9.0,
            mixed_wide_mark_gain=0.62,
            # Preserve the hybrid plan's baseline marker behind one presser
            # and one cover; its man-oriented gain acts through that marker.
            settled_pressure_count=1.0,
        ),
    }
)

# [speed, height, reach, control, endurance]. These plan-selection weights are
# structural design priors. They are not measured player-to-tactic effects.
_TACTICAL_ROSTER_WEIGHT = jnp.asarray(
    [
        [0.15, 0.05, 0.05, 0.50, 0.25],  # salida lavolpiana
        [0.18, 0.00, 0.00, 0.55, 0.27],  # juego de posicion
        [0.38, 0.00, 0.00, 0.12, 0.50],  # gegenpress
        [0.10, 0.25, 0.35, 0.05, 0.25],  # catenaccio
        [0.25, 0.10, 0.10, 0.25, 0.30],  # zona mista
    ],
    dtype=jnp.float32,
)


class TacticalPlanSelection(NamedTuple):
    """Two-team softmax receipt after episode abilities have been realized."""

    plan_code: jax.Array
    logits: jax.Array
    probability: jax.Array


def canonical_tactical_plan(value: TacticalPlan | str) -> TacticalPlan:
    """Return one canonical enum value or reject an unknown plan name."""

    if isinstance(value, TacticalPlan):
        return value
    if isinstance(value, str):
        try:
            return TacticalPlan(value)
        except ValueError as error:
            choices = ", ".join(plan.value for plan in TacticalPlan)
            raise ValueError(
                f"unknown tactical_plan {value!r}; choose from {choices}"
            ) from error
    raise TypeError("tactical_plan must be a TacticalPlan or its string value")


TACTICAL_PLAN_COUNT = len(TacticalPlan)


def tactical_plan_code(value: TacticalPlan | str) -> int:
    """Return the stable table row for a canonical tactical plan."""

    return tuple(TacticalPlan).index(canonical_tactical_plan(value))


def tactical_plan_from_code(value: int) -> TacticalPlan:
    """Return the canonical plan for one host integer code."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("tactical plan code must be an integer")
    plans = tuple(TacticalPlan)
    if value < 0 or value >= len(plans):
        raise ValueError(f"tactical plan code must be in [0, {len(plans)})")
    return plans[value]


def select_tactical_plans_from_abilities(
    ability: jax.Array,
    valid: jax.Array,
    is_goalkeeper: jax.Array,
    match_key: jax.Array,
    *,
    prior: jax.Array | None = None,
    temperature: float = 0.35,
) -> TacticalPlanSelection:
    """Sample team plans from roster-fit softmaxes before formation selection."""

    validate_prng_key(match_key, name="match_key")
    ability = jnp.asarray(ability, dtype=jnp.float32)
    valid = jnp.asarray(valid, dtype=jnp.bool_)
    is_goalkeeper = jnp.asarray(is_goalkeeper, dtype=jnp.bool_)
    if ability.ndim != 3 or ability.shape[0] != 2 or ability.shape[2] != 5:
        raise ValueError("ability must have shape [2, candidates, 5]")
    if valid.shape != ability.shape[:2] or is_goalkeeper.shape != valid.shape:
        raise ValueError("valid and is_goalkeeper must match ability[:2]")
    if isinstance(temperature, bool) or not isinstance(temperature, Real):
        raise TypeError("temperature must be a real number")
    if not 0.0 < float(temperature) < float("inf"):
        raise ValueError("temperature must be finite and positive")
    if prior is None:
        plan_prior = jnp.full(
            (2, TACTICAL_PLAN_COUNT),
            jnp.float32(1.0 / TACTICAL_PLAN_COUNT),
        )
    else:
        plan_prior = jnp.asarray(prior, dtype=jnp.float32)
        if plan_prior.ndim == 1:
            plan_prior = jnp.broadcast_to(plan_prior[None, :], (2, TACTICAL_PLAN_COUNT))
        if plan_prior.shape != (2, TACTICAL_PLAN_COUNT):
            raise ValueError("prior must have shape [5] or [2, 5]")
        if bool(jnp.any(~jnp.isfinite(plan_prior))) or bool(jnp.any(plan_prior < 0.0)):
            raise ValueError("prior must be finite and non-negative")
        if bool(jnp.any(jnp.sum(plan_prior, axis=1) <= 0.0)):
            raise ValueError("each team prior must have positive mass")
        plan_prior = plan_prior / jnp.sum(plan_prior, axis=1, keepdims=True)
    outfield = valid & (~is_goalkeeper)
    count = jnp.maximum(jnp.sum(outfield, axis=1), 1).astype(jnp.float32)
    roster_profile = (
        jnp.sum(jnp.where(outfield[..., None], ability, 0.0), axis=1) / count[:, None]
    )
    fit = roster_profile @ _TACTICAL_ROSTER_WEIGHT.T
    logits = jnp.where(
        plan_prior > 0.0,
        (jnp.log(jnp.maximum(plan_prior, jnp.float32(1e-12))) + fit)
        / jnp.float32(temperature),
        -jnp.inf,
    )
    selected = []
    for team in (0, 1):
        key = jax.random.fold_in(match_key, jnp.uint32(_TACTICAL_SELECTION_STREAM))
        key = jax.random.fold_in(key, jnp.uint32(team))
        selected.append(jax.random.categorical(key, logits[team]).astype(jnp.int32))
    return TacticalPlanSelection(
        plan_code=jnp.stack(selected),
        logits=logits,
        probability=jax.nn.softmax(logits, axis=1),
    )


def tactical_profiles() -> tuple[TacticalProfile, ...]:
    """Return profiles in the stable integer-code order."""

    return tuple(_TACTICAL_PROFILES[plan] for plan in TacticalPlan)


def gather_tactical_profile(plan_code):
    """Gather one profile by dynamic int code without changing graph shape."""

    import jax.numpy as jnp

    safe_code = jnp.clip(
        jnp.asarray(plan_code, dtype=jnp.int32), 0, TACTICAL_PLAN_COUNT - 1
    )
    profiles = tactical_profiles()
    values = {}
    for field in fields(TacticalProfile):
        table = jnp.asarray(
            tuple(getattr(profile, field.name) for profile in profiles),
            dtype=jnp.float32,
        )
        values[field.name] = table[safe_code]
    return TacticalProfile(**values)


def tactical_profile(plan: TacticalPlan | str) -> TacticalProfile:
    """Resolve a plan to the immutable fixed-shape numeric bundle."""

    return _TACTICAL_PROFILES[canonical_tactical_plan(plan)]


__all__ = [
    "TACTICAL_PLAN_COUNT",
    "TacticalPlan",
    "TacticalPlanSelection",
    "TacticalProfile",
    "canonical_tactical_plan",
    "gather_tactical_profile",
    "select_tactical_plans_from_abilities",
    "tactical_plan_code",
    "tactical_plan_from_code",
    "tactical_profile",
    "tactical_profiles",
]
