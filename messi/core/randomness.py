"""재현 가능한 환경 random stream의 안정적인 주소 contract."""

from __future__ import annotations

import os
from enum import IntEnum
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

RANDOM_EVENT_SCHEMA = "messi.random-event/2"
_MAX_FOLD_IN_ADDRESS = int(np.iinfo(np.uint32).max)


class RandomEvent(IntEnum):
    """추가만 허용하는 stochastic mechanism 주소."""

    ACTIVE_ROSTER_SAMPLING = 0x41435456
    BENCH_ROSTER_SAMPLING = 0x42454E43
    RESTART_TAKER = 0x54414B52
    CONTEST_FOUL = 0x464F554C
    CONTEST_SUCCESS = 0x0057494E
    CONTEST_DEFLECTION = 0x4445464C
    CONTEST_CARD = 0x43415244
    CONTEST_CARD_COLOUR = 0x434F4C52
    CONTEST_WINNER = 0x57494E52
    CONTEST_OUTCOME = 0x4F555443
    BODY_FOUL = 0x424F4459
    BODY_CARD = 0x42434152
    BODY_CARD_COLOUR = 0x42434F4C
    OPENING_CANDIDATE = 0x4F50454E
    OPENING_FORMATION = 0x464F524D
    OPENING_PLAYER_LINE = 0x4C494E45
    OPENING_REGION = 0x52454749
    MANAGER_SUBSTITUTION = 0x53554253
    MANAGER_SUBSTITUTION_TIMING = 0x54494D45
    MANAGER_SUBSTITUTION_OUTGOING = 0x4F555447
    MANAGER_SUBSTITUTION_INCOMING = 0x494E434D
    POLICY_ATTACK_PATTERN = 0x4154504E
    POLICY_CARRY_COMMIT = 0x434D4954
    POLICY_CARRIER_DECISION = 0x43415252
    POLICY_ATTACK_EPISODE = 0x41544550
    POLICY_RUN_BEHIND = 0x52554E52
    POLICY_CHALLENGE = 0x5441434B
    POLICY_DEEP_CLEAR = 0x434C4541
    POLICY_OFFSIDE_TIMING = 0x4F465344
    POLICY_RESTART_DECISION = 0x52535452
    POLICY_REBOUND_SHOT = 0x52425348
    POLICY_REBOUND_CHOICE = 0x52424348
    POLICY_RECEIVER = 0x52454356
    POLICY_MACRO = 0x4D414352
    POLICY_DRIBBLE_DIRECTION = 0x44524942
    POLICY_CLEAR_DIRECTION = 0x434C5244
    POLICY_SHOT_PORTION = 0x53484F54
    POLICY_CURVE = 0x43555256
    POLICY_SHOT_DIRECTION = 0x53484452
    POLICY_SHOT_LAUNCH = 0x53484C4E
    POLICY_GOALKEEPER_DISTRIBUTION = 0x474B4449
    POLICY_GROUND_RESTART = 0x47524E44
    POLICY_AERIAL_RESTART = 0x4145524C
    POLICY_GOALKEEPER_KICK = 0x474B4943
    POLICY_GOAL_TARGET = 0x474F414C


RANDOM_EVENT_NAMES = tuple(event.name.lower() for event in RandomEvent)
RANDOM_EVENT_ADDRESSES = MappingProxyType(
    {name.lower(): int(event) for name, event in RandomEvent.__members__.items()}
)


_AUTOTUNE_FLAG_NAME = "--xla_gpu_autotune_level"
_AUTOTUNE_FLAG = f"{_AUTOTUNE_FLAG_NAME}=0"
_autotuning_checked = False


def _has_deterministic_autotuning(xla_flags: str) -> bool:
    """정확한 autotune 비활성화 token이 하나만 있는지 확인한다."""

    autotuning_tokens = [
        token
        for token in xla_flags.split()
        if token == _AUTOTUNE_FLAG_NAME
        or token.startswith(f"{_AUTOTUNE_FLAG_NAME}=")
    ]
    return autotuning_tokens == [_AUTOTUNE_FLAG]


def require_deterministic_autotuning() -> None:
    """GPU에서 XLA 자동 튜닝이 꺼져 있는지 검증한다.

    XLA는 compile 시점에 GEMM 알고리즘을 실측으로 고르고 그 선택이 실행마다 달라질 수
    있다. 마지막 비트가 흔들리면 긴 rollout은 혼돈계라 골 수까지 갈린다. 같은 코드·
    시드·GPU에서 6,000 step 평가를 두 번 돌려 pass 719/676과 711/685을 얻어 확인했다.
    같은 설정 6회 반복의 통합 편차 폭은 `0.079`로, 후보 간 차이보다 컸다.

    ``--xla_gpu_autotune_level=0``은 B32×H1024 warm median을 `7.22초`에서 `7.17초`로
    바꾸는 수준이라 비용이 사실상 없다. 전체 ``--xla_gpu_deterministic_ops=true``는
    `10.82초`로 58% 느려 쓰지 않는다.
    """

    global _autotuning_checked
    if _autotuning_checked:
        return
    if _has_deterministic_autotuning(os.environ.get("XLA_FLAGS", "")):
        _autotuning_checked = True
        return
    try:
        platforms = {device.platform for device in jax.devices()}
    except RuntimeError:  # backend가 없으면 검증할 대상도 없다.
        _autotuning_checked = True
        return
    if not platforms & {"gpu", "cuda", "rocm"}:
        _autotuning_checked = True
        return
    raise RuntimeError(
        "MESSI는 GPU에서 XLA_FLAGS에 정확히 하나의 "
        "--xla_gpu_autotune_level=0 token을 요구하며 다른 값과의 중복을 "
        "허용하지 않는다. "
        "자동 튜닝이 켜져 있으면 같은 seed의 rollout이 실행마다 달라져 "
        "candidate 비교가 무의미해진다"
    )


def validate_prng_key(
    key: jax.Array,
    *,
    name: str = "key",
    batch_size: int | None = None,
) -> jax.Array:
    """scalar 또는 고정 batch의 Threefry key contract를 검증한다."""

    if bool(jax.config.jax_threefry_partitionable):
        raise RuntimeError(
            "MESSI는 JAX_THREEFRY_PARTITIONABLE=0을 요구한다. 이 설정을 켜면 "
            "seeded 환경 trajectory가 달라진다"
        )
    require_deterministic_autotuning()
    if not isinstance(key, (jax.Array, jax.core.Tracer, np.ndarray)):
        raise TypeError(f"{name}은 JAX PRNG key여야 한다")
    try:
        key_dtype = key.dtype
        key_shape = key.shape
        typed_key = jax.dtypes.issubdtype(key_dtype, jax.dtypes.prng_key)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(f"{name}은 JAX PRNG key여야 한다") from exc
    if typed_key:
        try:
            implementation = jax.random.key_impl(key)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name}은 JAX PRNG key여야 한다") from exc
        expected_shape = () if batch_size is None else (batch_size,)
    else:
        if key_dtype != jnp.uint32:
            raise TypeError(f"{name}은 JAX PRNG key여야 한다")
        implementation = jax.config.jax_default_prng_impl
        expected_shape = (2,) if batch_size is None else (batch_size, 2)
    if str(implementation) != "threefry2x32":
        raise ValueError(
            f"{name}은 재현성을 위해 threefry2x32를 사용해야 한다: {implementation}"
        )
    if key_shape != expected_shape:
        description = (
            "batch가 아닌 하나의" if batch_size is None else f"정확히 {batch_size}개의"
        )
        raise TypeError(f"{name}에는 {description} threefry2x32 PRNG key가 있어야 한다")
    return key


def _validate_host_random_address(value: object, *, name: str) -> object:
    if isinstance(value, (jax.Array, jax.core.Tracer)):
        return value
    try:
        host = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name}은 bool이 아닌 정수 scalar여야 한다") from exc
    if host.shape != ():
        return value
    if not np.issubdtype(host.dtype, np.integer) or np.issubdtype(host.dtype, np.bool_):
        return value
    integer = int(host)
    if not 0 <= integer <= _MAX_FOLD_IN_ADDRESS:
        raise ValueError(
            f"{name}은 [0, {_MAX_FOLD_IN_ADDRESS}] 범위에 있어야 한다: {integer}"
        )
    return np.uint32(integer)


def frame_random_key(match_key: jax.Array, control_tick: jax.Array) -> jax.Array:
    """불변 match key와 절대 control tick으로 frame key를 만든다."""

    validate_prng_key(match_key, name="match_key")
    control_tick = _validate_host_random_address(control_tick, name="control_tick")
    return _frame_random_key_unchecked(match_key, control_tick)


def _frame_random_key_unchecked(
    match_key: jax.Array, control_tick: jax.Array
) -> jax.Array:
    tick = jnp.asarray(control_tick)
    if tick.shape != ():
        raise ValueError("control_tick은 scalar여야 한다")
    if not jnp.issubdtype(tick.dtype, jnp.integer) or tick.dtype == jnp.bool_:
        raise TypeError("control_tick은 bool이 아닌 integer dtype이어야 한다")
    if tick.dtype.itemsize > np.dtype(np.uint32).itemsize:
        raise TypeError("control_tick은 최대 32-bit integer dtype이어야 한다")
    return jax.random.fold_in(match_key, tick.astype(jnp.uint32))


def event_random_key(
    base_key: jax.Array,
    event: RandomEvent,
    *addresses: jax.Array,
) -> jax.Array:
    """이름이 고정된 event와 causal scalar 주소를 key에 순서대로 fold한다."""

    if not isinstance(event, RandomEvent):
        raise TypeError("event는 RandomEvent여야 한다")
    validate_prng_key(base_key, name="base_key")
    addresses = tuple(
        _validate_host_random_address(address, name=f"addresses[{index}]")
        for index, address in enumerate(addresses)
    )
    return _event_random_key_unchecked(base_key, event, *addresses)


def _event_random_key_unchecked(
    base_key: jax.Array,
    event: RandomEvent,
    *addresses: jax.Array,
) -> jax.Array:
    if not isinstance(event, RandomEvent):
        raise TypeError("event는 RandomEvent여야 한다")
    key = jax.random.fold_in(base_key, jnp.uint32(int(event)))
    for address in addresses:
        value = jnp.asarray(address)
        if value.shape != ():
            raise ValueError("random event 주소는 scalar여야 한다")
        if not jnp.issubdtype(value.dtype, jnp.integer) or value.dtype == jnp.bool_:
            raise TypeError("random event 주소는 integer dtype이어야 한다")
        if value.dtype.itemsize > np.dtype(np.uint32).itemsize:
            raise TypeError("random event 주소는 최대 32-bit integer dtype이어야 한다")
        key = jax.random.fold_in(key, value.astype(jnp.uint32))
    return key


__all__ = [
    "RANDOM_EVENT_ADDRESSES",
    "RANDOM_EVENT_NAMES",
    "RANDOM_EVENT_SCHEMA",
    "RandomEvent",
    "event_random_key",
    "frame_random_key",
    "require_deterministic_autotuning",
    "validate_prng_key",
]
