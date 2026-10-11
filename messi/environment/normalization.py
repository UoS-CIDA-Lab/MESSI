"""SI `SharedScene`과 model 입력 사이의 고정·가역 normalization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from messi._json import strict_json_dumps
from messi.core.contact import ContactPhysics
from messi.core.numeric import (
    require_bounded_integer,
    require_float32_field_half_extent,
    require_float32_representable,
)
from messi.core.roster import RosterSampling
from messi.environment.observation import PolicyObservation, SharedScene

_DEFAULT_ROSTER_SAMPLING = RosterSampling()
_DEFAULT_CONTACT_PHYSICS = ContactPhysics()


@jax.tree_util.register_static
@dataclass(frozen=True, slots=True)
class ObservationNormalization:
    """runtime sample이 아니라 immutable environment 상한에서 만든 scale."""

    position_x_m: float
    position_y_m: float
    relative_position_x_m: float
    relative_position_y_m: float
    ball_position_x_m: float
    ball_position_y_m: float
    ball_relative_position_x_m: float
    ball_relative_position_y_m: float
    player_speed_mps: float
    relative_player_speed_mps: float
    ball_speed_mps: float
    relative_ball_speed_mps: float
    ball_spin_radps: float
    distance_squared_m2: float
    score_scale: float
    control_tick_scale: int
    height_m: float
    reach_height_m: float
    ball_control_scale: float
    endurance_factor_scale: float
    ball_height_m: float
    ball_vertical_speed_mps: float
    possession_tick_scale: int

    def __post_init__(self) -> None:
        for name in (
            "position_x_m",
            "position_y_m",
            "relative_position_x_m",
            "relative_position_y_m",
            "ball_position_x_m",
            "ball_position_y_m",
            "ball_relative_position_x_m",
            "ball_relative_position_y_m",
            "player_speed_mps",
            "relative_player_speed_mps",
            "ball_speed_mps",
            "relative_ball_speed_mps",
            "ball_spin_radps",
            "distance_squared_m2",
            "score_scale",
            "height_m",
            "reach_height_m",
            "ball_control_scale",
            "endurance_factor_scale",
            "ball_height_m",
            "ball_vertical_speed_mps",
        ):
            value = require_float32_representable(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name}은 양수여야 한다")
            represented = np.float32(value)
            if represented < np.finfo(np.float32).tiny:
                raise ValueError(
                    f"{name}은 float32 최소 positive normal 이상이어야 한다"
                )
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                reciprocal = np.reciprocal(represented)
            if not np.isfinite(reciprocal):
                raise ValueError(f"{name}의 float32 reciprocal은 유한해야 한다")
        try:
            control_tick_scale = require_bounded_integer(
                "control_tick_scale", self.control_tick_scale, minimum=1
            )
        except TypeError as error:
            raise ValueError("control_tick_scale은 양의 정수여야 한다") from error
        object.__setattr__(
            self,
            "control_tick_scale",
            control_tick_scale,
        )
        try:
            possession_tick_scale = require_bounded_integer(
                "possession_tick_scale", self.possession_tick_scale, minimum=1
            )
        except TypeError as error:
            raise ValueError("possession_tick_scale은 양의 정수여야 한다") from error
        object.__setattr__(self, "possession_tick_scale", possession_tick_scale)


class NormalizedSharedScene(NamedTuple):
    """SharedScene의 구조와 discrete dtype을 보존한 model view."""

    valid: jax.Array
    team_id: jax.Array
    attack_direction: jax.Array
    position_team: jax.Array
    velocity_team: jax.Array
    body_forward_team: jax.Array
    relative_position: jax.Array
    relative_velocity: jax.Array
    distance_squared: jax.Array
    teammate: jax.Array
    opponent: jax.Array
    ball_relative_position: jax.Array
    ball_relative_velocity: jax.Array
    ball_position_team: jax.Array
    ball_velocity_team: jax.Array
    ball_spin_team: jax.Array
    ball_live: jax.Array
    stamina_long: jax.Array
    stamina_short: jax.Array
    possessor: jax.Array
    restart_taker: jax.Array
    score_for: jax.Array
    score_against: jax.Array
    control_tick: jax.Array
    contact_ready: jax.Array
    challenge_ready: jax.Array


class NormalizedPolicyObservation(NamedTuple):
    """PolicyObservation의 구조와 discrete dtype을 보존한 model view."""

    scene: NormalizedSharedScene
    player_id: jax.Array
    is_goalkeeper: jax.Array
    max_speed: jax.Array
    reach_height: jax.Array
    height: jax.Array
    ball_control: jax.Array
    endurance_factor: jax.Array
    role: jax.Array
    formation_anchor: jax.Array
    formation_id: jax.Array
    prospective_offside: jax.Array
    goalkeeper_claim_available: jax.Array
    goalkeeper_holding: jax.Array
    restart_retouch_blocked: jax.Array
    restart_kind: jax.Array
    restart_indirect: jax.Array
    ball_height: jax.Array
    ball_vertical_velocity: jax.Array
    possession_team: jax.Array
    possession_previous_team: jax.Array
    possession_control_ticks: jax.Array
    possession_episode_ticks: jax.Array
    last_contact_actor: jax.Array
    last_contact_intent: jax.Array
    last_contact_kick_applied: jax.Array


OBSERVATION_LAYOUT_SCHEMA = "messi.normalized-shared-scene/3"
# Scene 밖의 leaf(능력치·역할·restart 맥락·last contact)까지 포함한 layout이다.
# Scene만 지문에 넣으면 그 leaf의 순서·dtype이 바뀌어도 checkpoint가 조용히 통과한다.
POLICY_OBSERVATION_LAYOUT_SCHEMA = "messi.policy-observation-layout/1"
OBSERVATION_NORMALIZATION_SCHEMA = "messi.observation-normalization/1"


@dataclass(frozen=True, slots=True)
class ObservationLeafLayout:
    name: str
    shape: tuple[int, ...]
    dtype: str
    block: str
    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class ObservationLayout:
    schema: str
    leaves: tuple[ObservationLeafLayout, ...]
    float_size: int
    integer_size: int
    boolean_size: int
    fingerprint: str


class FlatObservation(NamedTuple):
    floats: jax.Array
    integers: jax.Array
    booleans: jax.Array


def make_observation_normalization(
    *,
    field_half_extent: tuple[float, float] = (52.5, 34.0),
    ball_radius_m: float = 0.11,
    maximum_control_ticks: int = 60_000,
    score_scale: float = 20.0,
    ball_height_m: float = 5.0,
    ball_vertical_speed_mps: float = 20.0,
    possession_tick_scale: int = 6_000,
    roster: RosterSampling = _DEFAULT_ROSTER_SAMPLING,
    contact: ContactPhysics = _DEFAULT_CONTACT_PHYSICS,
) -> ObservationNormalization:
    """환경 정의에서 한 번만 deterministic normalization scale을 만든다."""

    half_x, half_y = require_float32_field_half_extent(field_half_extent)
    radius = require_float32_representable("ball_radius_m", ball_radius_m)
    if half_x <= 0.0 or half_y <= 0.0 or radius <= 0.0:
        raise ValueError("field extent와 ball radius는 양수여야 한다")
    try:
        maximum_control_ticks = require_bounded_integer(
            "maximum_control_ticks", maximum_control_ticks, minimum=1
        )
    except TypeError as error:
        raise ValueError("maximum_control_ticks는 양의 정수여야 한다") from error
    player_speed = roster.max_max_speed_mps
    ball_speed = contact.kick_speed_max_mps
    relative_x = 2.0 * half_x
    relative_y = 2.0 * half_y
    ball_relative_x = relative_x + radius
    ball_relative_y = relative_y + radius
    return ObservationNormalization(
        position_x_m=half_x,
        position_y_m=half_y,
        relative_position_x_m=relative_x,
        relative_position_y_m=relative_y,
        ball_position_x_m=half_x + radius,
        ball_position_y_m=half_y + radius,
        ball_relative_position_x_m=ball_relative_x,
        ball_relative_position_y_m=ball_relative_y,
        player_speed_mps=player_speed,
        relative_player_speed_mps=2.0 * player_speed,
        ball_speed_mps=ball_speed,
        relative_ball_speed_mps=ball_speed + player_speed,
        ball_spin_radps=contact.spin_max_radps,
        distance_squared_m2=relative_x * relative_x + relative_y * relative_y,
        score_scale=score_scale,
        control_tick_scale=maximum_control_ticks,
        height_m=roster.max_height_m,
        reach_height_m=roster.max_height_m + roster.max_reach_margin_m,
        ball_control_scale=roster.max_ball_control,
        endurance_factor_scale=roster.max_endurance_factor,
        ball_height_m=ball_height_m,
        ball_vertical_speed_mps=ball_vertical_speed_mps,
        possession_tick_scale=possession_tick_scale,
    )


DEFAULT_OBSERVATION_NORMALIZATION = make_observation_normalization()


def observation_normalization_fingerprint(
    context: ObservationNormalization = DEFAULT_OBSERVATION_NORMALIZATION,
) -> str:
    """모든 normalization scale을 포함한 deterministic contract 지문을 만든다."""

    payload = {
        "schema": OBSERVATION_NORMALIZATION_SCHEMA,
        "scales": {
            field.name: getattr(context, field.name) for field in fields(context)
        },
    }
    return hashlib.sha256(
        strict_json_dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _xy_scale(x: float, y: float, dtype) -> jax.Array:
    return jnp.asarray((x, y), dtype=dtype)


def normalize_observation(
    scene: SharedScene,
    context: ObservationNormalization = DEFAULT_OBSERVATION_NORMALIZATION,
) -> NormalizedSharedScene:
    """clip 없이 continuous SI leaf만 고정 scale로 나눈다."""

    position_scale = _xy_scale(
        context.position_x_m, context.position_y_m, scene.position_team.dtype
    )
    relative_scale = _xy_scale(
        context.relative_position_x_m,
        context.relative_position_y_m,
        scene.relative_position.dtype,
    )
    ball_position_scale = _xy_scale(
        context.ball_position_x_m,
        context.ball_position_y_m,
        scene.ball_position_team.dtype,
    )
    ball_relative_scale = _xy_scale(
        context.ball_relative_position_x_m,
        context.ball_relative_position_y_m,
        scene.ball_relative_position.dtype,
    )
    return NormalizedSharedScene(
        valid=scene.valid,
        team_id=scene.team_id,
        attack_direction=scene.attack_direction,
        position_team=scene.position_team / position_scale,
        velocity_team=scene.velocity_team / context.player_speed_mps,
        body_forward_team=scene.body_forward_team,
        relative_position=scene.relative_position / relative_scale,
        relative_velocity=scene.relative_velocity / context.relative_player_speed_mps,
        distance_squared=scene.distance_squared / context.distance_squared_m2,
        teammate=scene.teammate,
        opponent=scene.opponent,
        ball_relative_position=scene.ball_relative_position / ball_relative_scale,
        ball_relative_velocity=(
            scene.ball_relative_velocity / context.relative_ball_speed_mps
        ),
        ball_position_team=scene.ball_position_team / ball_position_scale,
        ball_velocity_team=scene.ball_velocity_team / context.ball_speed_mps,
        ball_spin_team=scene.ball_spin_team / context.ball_spin_radps,
        ball_live=scene.ball_live,
        stamina_long=scene.stamina_long,
        stamina_short=scene.stamina_short,
        possessor=scene.possessor,
        restart_taker=scene.restart_taker,
        score_for=scene.score_for.astype(jnp.float32) / context.score_scale,
        score_against=scene.score_against.astype(jnp.float32) / context.score_scale,
        control_tick=(
            scene.control_tick.astype(jnp.float32) / context.control_tick_scale
        ),
        contact_ready=scene.contact_ready,
        challenge_ready=scene.challenge_ready,
    )


def restore_observation(
    scene: NormalizedSharedScene,
    context: ObservationNormalization = DEFAULT_OBSERVATION_NORMALIZATION,
) -> SharedScene:
    """NormalizedSharedScene이 나타내는 SI 값을 복원한다."""

    position_scale = _xy_scale(
        context.position_x_m, context.position_y_m, scene.position_team.dtype
    )
    relative_scale = _xy_scale(
        context.relative_position_x_m,
        context.relative_position_y_m,
        scene.relative_position.dtype,
    )
    ball_position_scale = _xy_scale(
        context.ball_position_x_m,
        context.ball_position_y_m,
        scene.ball_position_team.dtype,
    )
    ball_relative_scale = _xy_scale(
        context.ball_relative_position_x_m,
        context.ball_relative_position_y_m,
        scene.ball_relative_position.dtype,
    )
    return SharedScene(
        valid=scene.valid,
        team_id=scene.team_id,
        attack_direction=scene.attack_direction,
        position_team=scene.position_team * position_scale,
        velocity_team=scene.velocity_team * context.player_speed_mps,
        body_forward_team=scene.body_forward_team,
        relative_position=scene.relative_position * relative_scale,
        relative_velocity=scene.relative_velocity * context.relative_player_speed_mps,
        distance_squared=scene.distance_squared * context.distance_squared_m2,
        teammate=scene.teammate,
        opponent=scene.opponent,
        ball_relative_position=scene.ball_relative_position * ball_relative_scale,
        ball_relative_velocity=(
            scene.ball_relative_velocity * context.relative_ball_speed_mps
        ),
        ball_position_team=scene.ball_position_team * ball_position_scale,
        ball_velocity_team=scene.ball_velocity_team * context.ball_speed_mps,
        ball_spin_team=scene.ball_spin_team * context.ball_spin_radps,
        ball_live=scene.ball_live,
        stamina_long=scene.stamina_long,
        stamina_short=scene.stamina_short,
        possessor=scene.possessor,
        restart_taker=scene.restart_taker,
        score_for=jnp.rint(scene.score_for * context.score_scale).astype(jnp.int32),
        score_against=jnp.rint(scene.score_against * context.score_scale).astype(
            jnp.int32
        ),
        control_tick=jnp.rint(
            scene.control_tick * context.control_tick_scale
        ).astype(jnp.int32),
        contact_ready=scene.contact_ready,
        challenge_ready=scene.challenge_ready,
    )


def normalize_policy_observation(
    observation: PolicyObservation,
    context: ObservationNormalization = DEFAULT_OBSERVATION_NORMALIZATION,
) -> NormalizedPolicyObservation:
    """전체 PolicyObservation의 continuous leaf만 deterministic하게 정규화한다."""

    anchor_scale = _xy_scale(
        context.position_x_m,
        context.position_y_m,
        observation.formation_anchor.dtype,
    )
    return NormalizedPolicyObservation(
        scene=normalize_observation(observation.scene, context),
        player_id=observation.player_id,
        is_goalkeeper=observation.is_goalkeeper,
        max_speed=observation.max_speed / context.player_speed_mps,
        reach_height=observation.reach_height / context.reach_height_m,
        height=observation.height / context.height_m,
        ball_control=observation.ball_control / context.ball_control_scale,
        endurance_factor=(
            observation.endurance_factor / context.endurance_factor_scale
        ),
        role=observation.role,
        formation_anchor=observation.formation_anchor / anchor_scale,
        formation_id=observation.formation_id,
        prospective_offside=observation.prospective_offside,
        goalkeeper_claim_available=observation.goalkeeper_claim_available,
        goalkeeper_holding=observation.goalkeeper_holding,
        restart_retouch_blocked=observation.restart_retouch_blocked,
        restart_kind=observation.restart_kind,
        restart_indirect=observation.restart_indirect,
        ball_height=observation.ball_height / context.ball_height_m,
        ball_vertical_velocity=(
            observation.ball_vertical_velocity / context.ball_vertical_speed_mps
        ),
        possession_team=observation.possession_team,
        possession_previous_team=observation.possession_previous_team,
        possession_control_ticks=(
            observation.possession_control_ticks.astype(jnp.float32)
            / context.possession_tick_scale
        ),
        possession_episode_ticks=(
            observation.possession_episode_ticks.astype(jnp.float32)
            / context.possession_tick_scale
        ),
        last_contact_actor=observation.last_contact_actor,
        last_contact_intent=observation.last_contact_intent,
        last_contact_kick_applied=observation.last_contact_kick_applied,
    )


def _leaf_block(dtype) -> str:
    if jnp.issubdtype(dtype, jnp.floating):
        return "float"
    if jnp.issubdtype(dtype, jnp.integer):
        return "integer"
    if jnp.issubdtype(dtype, jnp.bool_):
        return "boolean"
    raise TypeError(f"지원하지 않는 observation dtype: {dtype}")


def observation_layout(scene: NormalizedSharedScene) -> ObservationLayout:
    """Named leaf 의미·shape·dtype와 typed-block offset을 고정한다."""

    return _layout_from_fields(
        tuple(zip(scene._fields, scene, strict=True)), OBSERVATION_LAYOUT_SCHEMA
    )


def policy_observation_layout(observation) -> ObservationLayout:
    """PolicyObservation 전체(scene + 나머지 leaf)의 layout과 지문을 만든다.

    Scene만 지문에 넣으면 능력치·역할·restart 맥락·last contact leaf의 순서나 dtype이
    바뀌어도 checkpoint가 조용히 통과한다. Scene leaf는 ``scene.`` prefix로 구분한다.
    """

    fields: list[tuple[str, object]] = []
    for name, value in zip(observation._fields, observation, strict=True):
        if name == "scene":
            fields.extend(
                (f"scene.{inner}", leaf)
                for inner, leaf in zip(value._fields, value, strict=True)
            )
        else:
            fields.append((name, value))
    return _layout_from_fields(tuple(fields), POLICY_OBSERVATION_LAYOUT_SCHEMA)


def _layout_from_fields(fields, schema: str) -> ObservationLayout:
    offsets = {"float": 0, "integer": 0, "boolean": 0}
    leaves: list[ObservationLeafLayout] = []
    for name, value in fields:
        array = jnp.asarray(value)
        block = _leaf_block(array.dtype)
        shape = tuple(int(size) for size in array.shape)
        size = int(np.prod(shape, dtype=np.int64)) if shape else 1
        leaves.append(
            ObservationLeafLayout(
                name=name,
                shape=shape,
                dtype=str(array.dtype),
                block=block,
                offset=offsets[block],
                size=size,
            )
        )
        offsets[block] += size
    payload = {
        "schema": schema,
        "leaves": [
            {
                "name": leaf.name,
                "shape": leaf.shape,
                "dtype": leaf.dtype,
                "block": leaf.block,
                "offset": leaf.offset,
                "size": leaf.size,
            }
            for leaf in leaves
        ],
    }
    fingerprint = hashlib.sha256(
        strict_json_dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return ObservationLayout(
        schema=schema,
        leaves=tuple(leaves),
        float_size=offsets["float"],
        integer_size=offsets["integer"],
        boolean_size=offsets["boolean"],
        fingerprint=fingerprint,
    )


def _validate_layout(scene: NormalizedSharedScene, layout: ObservationLayout) -> None:
    actual = observation_layout(scene)
    if actual.schema != layout.schema or actual.fingerprint != layout.fingerprint:
        raise ValueError(
            "observation layout이 schema/shape/dtype contract와 일치하지 않는다"
        )


def flatten_observation(
    scene: NormalizedSharedScene,
    *,
    layout: ObservationLayout | None = None,
) -> tuple[FlatObservation, ObservationLayout]:
    """category dtype을 보존하는 float/int/bool block으로 평탄화한다."""

    resolved = observation_layout(scene) if layout is None else layout
    if layout is not None:
        _validate_layout(scene, layout)
    values = {"float": [], "integer": [], "boolean": []}
    for leaf, value in zip(resolved.leaves, scene, strict=True):
        values[leaf.block].append(jnp.asarray(value).reshape(-1))

    def concatenate(block: str, dtype) -> jax.Array:
        arrays = values[block]
        if not arrays:
            return jnp.zeros((0,), dtype=dtype)
        return jnp.concatenate(arrays).astype(dtype)

    return (
        FlatObservation(
            floats=concatenate("float", jnp.float32),
            integers=concatenate("integer", jnp.int32),
            booleans=concatenate("boolean", jnp.bool_),
        ),
        resolved,
    )


def unflatten_observation(
    flat: FlatObservation,
    layout: ObservationLayout,
) -> NormalizedSharedScene:
    """검증된 typed block을 원래 NormalizedSharedScene으로 복원한다."""

    if layout.schema != OBSERVATION_LAYOUT_SCHEMA:
        raise ValueError(f"지원하지 않는 observation schema: {layout.schema}")
    expected = {
        "float": layout.float_size,
        "integer": layout.integer_size,
        "boolean": layout.boolean_size,
    }
    blocks = {
        "float": jnp.asarray(flat.floats),
        "integer": jnp.asarray(flat.integers),
        "boolean": jnp.asarray(flat.booleans),
    }
    required_dtype = {
        "float": jnp.dtype(jnp.float32),
        "integer": jnp.dtype(jnp.int32),
        "boolean": jnp.dtype(jnp.bool_),
    }
    for block, value in blocks.items():
        if value.ndim != 1 or value.shape[0] != expected[block]:
            raise ValueError(f"{block} block shape가 layout과 일치하지 않는다")
        if value.dtype != required_dtype[block]:
            raise TypeError(f"{block} block dtype이 layout contract와 다르다")
    restored = []
    for leaf in layout.leaves:
        block = blocks[leaf.block]
        value = block[leaf.offset : leaf.offset + leaf.size]
        restored.append(value.astype(jnp.dtype(leaf.dtype)).reshape(leaf.shape))
    return NormalizedSharedScene(*restored)


__all__ = [
    "DEFAULT_OBSERVATION_NORMALIZATION",
    "OBSERVATION_LAYOUT_SCHEMA",
    "OBSERVATION_NORMALIZATION_SCHEMA",
    "POLICY_OBSERVATION_LAYOUT_SCHEMA",
    "FlatObservation",
    "NormalizedPolicyObservation",
    "NormalizedSharedScene",
    "ObservationLayout",
    "ObservationLeafLayout",
    "ObservationNormalization",
    "flatten_observation",
    "make_observation_normalization",
    "normalize_observation",
    "normalize_policy_observation",
    "observation_layout",
    "observation_normalization_fingerprint",
    "policy_observation_layout",
    "restore_observation",
    "unflatten_observation",
]
