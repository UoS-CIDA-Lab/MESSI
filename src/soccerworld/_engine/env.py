from __future__ import annotations

import copy
import functools
import hashlib
import inspect
import json
import math
import numbers
from collections.abc import Callable, Mapping
from dataclasses import asdict, fields, replace
from functools import wraps

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxmarl.environments import spaces
from jaxmarl.environments.multi_agent_env import MultiAgentEnv

from soccerworld.core.commands import (
    FormationCommand,
    SetPieceTakerCommand,
    StepCommand,
    SubstitutionCommand,
)
from soccerworld.core.randomness import (
    RandomEvent,
    RandomnessControl,
    select_random_key,
    validate_randomness_control,
)
from soccerworld.core.results import CommandReason

from . import formation as formation_module
from . import manager as manager_module
from . import setpiece_taker as setpiece_taker_module
from . import substitution as substitution_module
from .ball import BallPhysics
from .config import (
    MAX_BENCH_SIZE,
    MAX_CONTROL_DECIMATION,
    MAX_ENV_TEAM_PLAYERS,
    MAX_SUBSTITUTION_SCHEDULE_ROWS,
    Agent,
    Ball,
    BenchPlayer,
    Engine,
    Foul,
    Reward,
    Stadium,
    Substitution,
)
from .constants import (
    ACTION_DIM,
    ACTION_KICK_GATE,
    ACTION_MAX,
    ACTION_MIN,
    ACTION_SCHEMA_VERSION,
    AFFORDANCE_SCHEMA_VERSION,
    BALL_ALIVE,
    BALL_DEAD,
    BALL_EVENT_NONE,
    DEFAULT_CONTROL_FPS,
    DEFAULT_FORMATION,
    DEFAULT_MAX_SUBSTITUTIONS,
    DEFAULT_TEAM_SIZE,
    DEPARTED_TAKER,
    DIM_ALL,
    DIM_X,
    DIM_Y,
    DIM_Z,
    DISCIPLINE_OUTCOMES,
    DIV_EPS,
    ENERGY_DYNAMICS_VERSION,
    ENV_DYNAMICS_VERSION,
    FORMATION_DECISION_APPLIED,
    FORMATION_DECISION_OUT_OF_RANGE,
    FORMATION_DECISION_TERMINAL,
    FORMATION_DECISION_UNCHANGED,
    FOUL_NONE,
    FOUL_TACKLE,
    GEOMETRY_EPS,
    IFAB_MAX_SUBSTITUTION_WINDOWS,
    IFAB_MIN_TEAM_PLAYERS,
    INJECTABLE_FOUL_KINDS,
    MOVEMENT_DYNAMICS_VERSION,
    NO_EVENT,
    NO_PLAYER,
    NO_TEAM,
    OBS_SCHEMA_VERSION,
    RESTART_COUNT,
    RK_GK_HOLD,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_NONE,
    RK_THROWIN,
    ROLE_GAIN_EXACT_MAX_SAMPLES,
    ROTATE_180,
    SAMPLED_WINNER,
    STATE_CONTAINER_SCHEMA_VERSION,
    STATE_SCHEMA_VERSION,
    SUB_DECISION_APPLIED,
    SUB_DECISION_BALL_LIVE,
    SUB_DECISION_BENCH_EMPTY,
    SUB_DECISION_BENCH_RANGE,
    SUB_DECISION_GK_ROLE,
    SUB_DECISION_NO_CARD,
    SUB_DECISION_NOT_REQUESTED,
    SUB_DECISION_PLACEMENT,
    SUB_DECISION_SLOT_ALREADY_CHANGED,
    SUB_DECISION_SLOT_INACTIVE,
    SUB_DECISION_SLOT_RANGE,
    SUB_DECISION_TERMINAL,
    SUB_DECISION_WINDOW_BUDGET,
    SUB_DECISION_WRONG_TEAM,
    TEAM_0,
    TEAM_1,
    TEAM_COUNT,
    TOUCH_DRIBBLE,
    TOUCH_EVENT_BODY,
    TOUCH_EVENT_FORCE,
    TOUCH_EVENT_PHASE_COUNT,
    TOUCH_GK_CATCH,
    TOUCH_INTERCEPT,
    TOUCH_NONE,
    TOUCH_PARRY,
    TOUCH_PASS,
    TOUCH_PASS_HEAD,
    TOUCH_SHOOT,
    TOUCH_SHOOT_HEAD,
    TOUCH_TACKLE,
    WOODWORK_NONE,
    YELLOW_CARD_SEND_OFF_COUNT,
)
from .contest import Contest
from .energy import long_stamina_drain_base, long_stamina_tail_decay
from .events import Events
from .fouls import Fouls
from .initialization import build_static_metadata
from .inverse import Inverse
from .movement import Movement
from .observation import Observation
from .offside import Offside
from .render import Render
from .restart import Restart, _validate_float32_runtime, restart_timer_active
from .rewards import Rewards
from .spatial import _safe_norm
from .state import State
from .timebase import DEFAULT_MATCH_DURATION_SECONDS, Timebase
from .validation import (
    _FLOAT32_DYNAMICS_MAGNITUDE_LIMIT,
    _float32_position_resolution,
    _require_float32_scalar,
    validate_configuration,
)
from .validation import (
    _FLOAT32_EXACT_COUNTER_MAX as _FLOAT32_EXACT_COUNTER_MAX,
)

# The rule-policy solver is a derived, policy-only lazy cache and is the sole
# private SoccerEnv attribute that legitimately needs post-construction
# rebinding.  A blanket ``name.startswith('_')`` escape also exposed roster,
# schema and even private mechanics methods to fingerprint-invisible mutation.
_POST_INIT_MUTABLE_PRIVATE_ENV_ATTRIBUTES = frozenset({
    "_rule_policy_solver_cache",
    # 결정자 파라미터를 **한 호출 동안만** 싣는 자리. 정적 계약이 아니라 그 호출의
    # 데이터다 — 어느 결정자를 어떤 규약으로 부르는지는 생성 시 확정되어 얼지만,
    # 가중치는 호출마다 바뀌는 것이 이 기능의 목적 자체다. 키커 지정 호출이 서브스텝
    # 스캔 안쪽 아홉 곳에 흩어져 있어 인자로 꿰려면 규칙 함수 여덟 개의 시그니처를
    # 넓혀야 하고, 그 인자는 거의 항상 ``None``이라 규칙 코드에 잡음만 남는다.
    # ``step_env_array``/``reset_state``가 ``finally``로 되돌리므로 호출 밖에서는 항상
    # ``None``이고, 객체에 트레이서가 남지 않는다.
    "_taker_params",
    # Native ``StepCommand`` taker overrides have the same lifetime as dynamic
    # taker parameters: they are installed only while one trace is being built
    # and restored in ``finally``.  Keeping the fixed-shape PyTree here lets all
    # restart creation sites (including sites inside the physics scan) share the
    # existing taker-selection SSOT without retaining a tracer on the env.
    "_native_taker_command",
    # Privileged event-key overrides have the same trace-local lifetime as
    # native taker commands.  They are never policy actions and are restored
    # after each public reset/transition call.
    "_active_randomness_control",
})


# These Engine values define a host-side real-time clock and integer tick
# quantisation before JAX sees them.  Equal float32 encodings can legitimately
# straddle a floor/ceil boundary, so unlike continuous JAX coefficients they
# retain their validated Python precision in the canonical object/fingerprint.
_ENGINE_HOST_TIME_FIELDS = frozenset({
    "dt_phys",
    "long_stamina_reference_duration_s",
    "challenge_cooldown_extra_s",
    "gk_hold_s",
    "cooldown_s",
    "contact_interval_s",
    "aerial_attempt_lock_s",
    "restart_s",
    "setup_hold_s",
    "throwin_restart_delay_s",
    "goalkick_restart_delay_s",
    "corner_restart_delay_s",
    "freekick_restart_delay_s",
    "offside_restart_delay_s",
    "post_goal_kickoff_delay_s",
    "penalty_restart_delay_s",
    "ctrl_lock_s",
    "penalty_s",
})


def _default_roster(start_id: int) -> list[Agent]:
    """``DEFAULT_FORMATION``으로 팀을 만들며 첫 슬롯만 GK로 둔다."""
    return [
        Agent(id=start_id + i, is_gk=(i == 0), init_pos=pos)
        for i, pos in enumerate(DEFAULT_FORMATION)
    ]


def _is_host_real(value) -> bool:
    """True for scalar real numbers, deliberately excluding bool coercion."""

    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


def _is_host_integral(value) -> bool:
    """True for scalar integral IDs/counters, deliberately excluding bool."""

    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _is_host_real_vector(value, shape) -> bool:
    """Strict host vector contract used before NumPy/JAX narrowing casts."""

    if not isinstance(value, (tuple, list, np.ndarray)):
        return False
    array = np.asarray(value, dtype=object)
    return array.shape == shape and all(_is_host_real(item) for item in array.flat)


def _validate_prng_key(key, *, name="key"):
    """Require the single reproducible PRNG implementation used by dynamics.

    JAX typed keys carry their implementation in the dtype, while legacy
    ``uint32`` keys inherit the process-wide default.  Accepting every
    implementation made ``(seed, state, action, dynamics_fingerprint)``
    insufficient to reproduce a transition: threefry, RBG, and unsafe-RBG
    produce different contest/card draws under the same integer seed.  The
    environment's golden trajectories and reconstruction contract are based on
    threefry2x32, so reject a different static key implementation before any
    random split.  Both legacy ``PRNGKey`` and typed ``key(..., impl='threefry2x32')``
    remain valid and this check is safe while tracing/jitting/vmapping.
    """

    # The environment is a float32 contract.  Construction validates x64, but
    # JAX configuration remains process-global and callers can flip it after an
    # environment (and its fingerprint) already exists.  The next random
    # boundary must fail explicitly before weak Python literals are promoted
    # and ``lax.cond`` branches acquire incompatible float32/float64 leaves.
    # Checking here covers reset and every forward/reconstruction step without
    # adding a data-dependent operation to the compiled transition.
    _validate_float32_runtime()

    # This global changes both direct draws and ``split`` for the exact same
    # threefry raw key.  The pinned dynamics run in the False mode; switching
    # it is a dynamics change, not a silent process-local toggle.
    if bool(jax.config.jax_threefry_partitionable):
        raise RuntimeError(
            "SoccerEnv requires JAX_THREEFRY_PARTITIONABLE=0; enabling it "
            "changes seeded environment trajectories"
        )
    try:
        implementation = jax.random.key_impl(key)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a scalar threefry2x32 JAX PRNG key"
        ) from exc
    if str(implementation) != "threefry2x32":
        raise ValueError(
            f"{name} must use threefry2x32 for reproducible environment "
            f"dynamics, got {implementation}"
        )
    # ``key_impl`` also accepts a *batch* of typed/legacy keys.  Random APIs do
    # not interpret that leading axis as one transition key (callers must vmap),
    # and letting it reach ``split`` produces a late, backend-specific shape
    # error.  A scalar threefry key always has exactly two raw uint32 words.
    if jax.random.key_data(key).shape != (2,):
        raise TypeError(
            f"{name} must be one scalar threefry2x32 key; batch keys must be "
            "mapped with jax.vmap"
        )
    return key


def _canonical_dataclass(instance):
    """Return an immutable config/roster object with annotation-canonical scalars.

    Validation must run before this helper: bool is a Real/Integral subclass and
    coercing first would hide a bad public input.  Once validated, canonical Python
    float/int/bool/tuple values keep JAX promotion and JSON fingerprints independent
    of whether callers used ``15``, ``15.0``, Fraction or NumPy scalar wrappers.
    """

    def canonical_float(value, *, runtime_float32: bool):
        """Return one JSON/JAX-stable Python float.

        IEEE ``-0.0`` and ``+0.0`` have identical environment dynamics, but
        JSON preserves their spelling and therefore used to give them different
        fingerprints.  Normalize the sign only for zero; all nonzero values,
        including subnormals, remain untouched.
        """

        # Continuous environment coefficients execute in float32.  Widening a
        # NumPy float32 wrapper to Python float used to make ``0.1`` and
        # ``np.float32(0.1)`` hash differently despite identical runtime bits.
        # Host clock fields are the deliberate exception documented above.
        result = (
            float(np.float32(value)) if runtime_float32 else float(value)
        )
        return 0.0 if result == 0.0 else result

    updates = {}
    for declared in fields(instance):
        if not declared.init:
            continue
        value = getattr(instance, declared.name)
        expected = declared.type
        runtime_float32 = not (
            isinstance(instance, Engine)
            and declared.name in _ENGINE_HOST_TIME_FIELDS
        )
        if expected is float or declared.name == "norm_spin":
            updates[declared.name] = (
                None if value is None else canonical_float(
                    value, runtime_float32=runtime_float32
                )
            )
        elif expected is int:
            updates[declared.name] = int(value)
        elif expected is bool:
            updates[declared.name] = bool(value)
        elif expected is tuple or str(expected).startswith("tuple["):
            updates[declared.name] = tuple(
                canonical_float(item, runtime_float32=True) for item in value
            )
        else:
            updates[declared.name] = value
    return replace(instance, **updates)


class _ReadOnlyVersions(dict):
    """쓰기를 거부하는 dict — ``json.dumps``와 ``dict(...)``는 그대로 동작한다.

    스키마 버전은 fingerprint의 입력이라 공개 객체를 고칠 수 있으면 manifest·fingerprint와
    조용히 어긋난다. 그렇다고 ``mappingproxy``로 막으면 직렬화가 깨지므로, dict의 인터페이스는
    유지한 채 일반 변경 경로를 닫는다. 이 객체 자체도 내부 원본이 아닌 방어적 복사본이다.
    따라서 ``dict.__setitem__(view, ...)``처럼 서브클래스 훅을 명시적으로 우회해도 그 일회성
    view만 바뀌며 환경의 스키마·metadata·fingerprint에는 닿지 않는다.
    """

    _MESSAGE = ("schema_version is read-only because it feeds the dynamics fingerprint; "
                "copy it with dict(env.schema_version) before modifying")

    def _readonly(self, *args, **kwargs):
        raise TypeError(self._MESSAGE)

    __setitem__ = _readonly
    __delitem__ = _readonly
    clear = _readonly
    pop = _readonly
    popitem = _readonly
    setdefault = _readonly
    update = _readonly
    __ior__ = _readonly

    def __reduce__(self):
        """Rebuild through ``dict.__init__`` instead of blocked item writes."""

        return type(self), (dict(self),)

    def __deepcopy__(self, memo):
        copied = type(self)(
            copy.deepcopy(dict(self), memo)
        )
        memo[id(self)] = copied
        return copied


class _ImmutableMapping(Mapping):
    """Small pickleable immutable mapping used for public space registries.

    ``types.MappingProxyType`` rejects both pickle and deepcopy, which in turn
    made a fully constructed environment unusable with spawn-based workers.
    Store ordered key/value pairs in a tuple: the public mapping remains
    immutable, insertion order is stable, and pickle can reconstruct it without
    a temporary mutation phase.
    """

    __slots__ = ("_items",)

    def __init__(self, values):
        items = tuple(values.items()) if isinstance(values, Mapping) else tuple(values)
        keys = tuple(key for key, _ in items)
        if len(set(keys)) != len(keys):
            raise ValueError("immutable mapping keys must be unique")
        object.__setattr__(self, "_items", items)

    def __getitem__(self, key):
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self):
        return len(self._items)

    def __setattr__(self, name, value):
        if hasattr(self, "_items"):
            raise AttributeError("immutable mapping cannot be modified")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError("immutable mapping cannot be modified")

    def __reduce__(self):
        return type(self), (self._items,)

    def __deepcopy__(self, memo):
        copied = type(self)(copy.deepcopy(self._items, memo))
        memo[id(self)] = copied
        return copied


class _ReadOnlyBox(spaces.Box):
    """A JaxMARL-compatible Box whose declaration cannot be edited in place."""

    def __init__(self, low, high, shape, dtype=jnp.float32):
        object.__setattr__(self, "_declaration_frozen", False)
        super().__init__(low, high, shape, dtype)
        object.__setattr__(self, "_declaration_frozen", True)

    def __setattr__(self, name, value):
        existing_public_name = (
            name in self.__dict__
            or any(name in base.__dict__ for base in type(self).__mro__)
        )
        if getattr(self, "_declaration_frozen", False) and existing_public_name:
            raise AttributeError(
                f"space.{name} is read-only; construct a new environment "
                "to change its declared spaces"
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, "_declaration_frozen", False) and name in self.__dict__:
            raise AttributeError(f"space.{name} is read-only and cannot be deleted")
        object.__delattr__(self, name)


def _decider_takes_params(fn):
    """결정자가 ``(params, view, key)`` 3인자 규약인가.

    학습 결정자는 파라미터를 **동적 인자**로 받아야 체크포인트가 바뀔 때 재컴파일되지
    않는다(:class:`manager.DecisionParams`). 기존 결정자는 ``(view, key)`` 2인자이므로 둘을
    구분해야 하는데, 별도 등록 절차를 두면 남의 결정자를 꽂는 문턱이 올라간다. 그래서
    **필수 위치 인자의 개수**로 가른다 — 문서화된 두 규약의 인자 수가 서로 다르기 때문에
    모호하지 않다.

    판정은 생성 시 한 번만 하므로 정적이다. 시그니처를 읽을 수 없는 호출자(C 확장 등)는
    기존 2인자 규약으로 본다 — 새 규약은 명시적으로 선택하는 쪽이 안전하다.
    """

    target = fn if inspect.isfunction(fn) or inspect.ismethod(fn) else (
        getattr(fn, "__call__", fn))
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return False
    required = 0
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            return False           # *args 결정자는 개수로 가를 수 없다.
        if parameter.kind not in (parameter.POSITIONAL_ONLY,
                                  parameter.POSITIONAL_OR_KEYWORD):
            continue
        if parameter.default is parameter.empty:
            required += 1
    return required >= 3


def _callable_identity(fn):
    """사용자 결정자의 신원 문자열 — 이름만으로는 다른 알고리즘이 같은 지문을 갖는다.

    같은 factory에서 나온 두 클로저는 ``__qualname__``이 같지만 전이가 다르다(실측:
    "레이아웃 16"과 "레이아웃 17"이 정확히 같은 fingerprint였다). 바이트코드와 클로저 셀
    값을 함께 담아 그 둘을 가른다.

    파이썬 함수를 완전히 내용 해시할 수는 없다(자유변수가 가변 객체를 가리킬 수 있다).
    그래서 이것은 **재현 보장이 아니라 구분 장치**다 — manifest만으로 사용자 결정자를
    재현할 수 없다는 사실 자체는 그대로다.
    """

    # ``functools.partial``과 호출가능 객체는 ``__code__``가 없다. 이름만 남기면 서로 다른
    # 알고리즘이 ``functools:?`` 하나로 뭉개진다(실측: partial 두 개가 같은 지문인데 다음
    # 레이아웃은 [16,16]과 [17,17]로 갈렸다). 감싼 대상과 묶인 인자를 따라 들어간다.
    if isinstance(fn, functools.partial):
        inner = _callable_identity(fn.func)
        bound = repr((fn.args, sorted((fn.keywords or {}).items())))
        return (f"partial({inner})#"
                + hashlib.sha256(bound.encode("utf-8")).hexdigest()[:16])
    if not callable(getattr(fn, "__code__", None)) and hasattr(fn, "__call__") \
            and not hasattr(fn, "__code__"):
        # 호출가능 객체 — 클래스 신원과 인스턴스 상태를 담는다.
        cls = type(fn)
        state = getattr(fn, "__dict__", {})
        return (f"{cls.__module__}:{cls.__qualname__}#"
                + hashlib.sha256(repr(sorted(state.items())).encode("utf-8"))
                .hexdigest()[:16])

    parts = [getattr(fn, "__module__", "?"), getattr(fn, "__qualname__", "?")]
    code = getattr(fn, "__code__", None)
    if code is not None:
        parts.append(hashlib.sha256(code.co_code).hexdigest()[:16])
        cells = []
        for cell in (getattr(fn, "__closure__", None) or ()):
            try:
                cells.append(repr(cell.cell_contents))
            except ValueError:
                cells.append("<empty>")
        parts.append(
            hashlib.sha256("|".join(cells).encode("utf-8")).hexdigest()[:16])
    return ":".join(parts)


def _first_free(slots):
    """``NO_PLAYER``인 첫 자리. 자리가 없으면 마지막을 덮는다(가장 오래된 기록이 밀린다)."""

    free = slots < 0
    return jnp.where(jnp.any(free), jnp.argmax(free), slots.shape[0] - 1).astype(
        jnp.int32)


def _require_integer_proposal(value, name):
    """제안 배열이 **정수**인지 변환 전에 확인한다.

    ``jnp.asarray(value, jnp.int32)``를 먼저 하면 실수가 조용히 잘린다. 슬롯/레이아웃
    인덱스에 반올림 규약을 정할 이유가 없으므로 애초에 거부하는 편이 낫다 — 잘린 값은
    합법 인덱스라 승인 층도 잡아내지 못한다.

    **폭도 같은 이유로 좁히기 전에 본다.** int32로 먼저 캐스팅하면 ``2**32 + 1``이 슬롯 1로
    감기고, 감긴 값 역시 합법 인덱스라 승인 층을 그대로 통과한다. 호스트 값일 때만 검사할
    수 있다 — 트레이서는 이미 int32로 좁혀진 뒤라 여기서 볼 것이 남아 있지 않다.
    """

    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.integer):
        raise ValueError(
            f"{name} must be an integer array, got dtype {array.dtype}")
    limits = np.iinfo(np.int32)
    try:
        source = np.asarray(value)
    except (TypeError, ValueError):
        source = None                      # 트레이서 — 호스트에서 읽을 값이 없다.
    if source is not None and source.size and np.issubdtype(
        source.dtype, np.integer
    ):
        if int(source.min()) < limits.min or int(source.max()) > limits.max:
            raise ValueError(
                f"{name} must fit in int32 ([{limits.min}, {limits.max}]), got "
                f"[{int(source.min())}, {int(source.max())}]")
    return array.astype(jnp.int32)


class SoccerEnv(
    Events,
    Observation,
    Movement,
    Restart,
    BallPhysics,
    Contest,
    Fouls,
    Offside,
    Rewards,
    Inverse,
    Render,
    MultiAgentEnv
):
    def __setattr__(self, name, value):
        """Freeze the constructed environment's public static contract.

        Config objects and roster containers are individually immutable, but
        their public attributes used to remain rebindable.  Replacing, for
        example, ``e_cfg`` or ``game_duration`` after construction changed live
        dynamics while cached geometry, metadata, and the fingerprint retained
        their original values.  The explicitly allowlisted private policy
        solver cache remains writable; a caller that needs different mechanics
        must construct a new env.
        """

        frozen = self.__dict__.get("_static_contract_frozen", False)
        # Methods/properties inherited from the mixins and private mechanics
        # are part of the same static contract as stored attributes. Checking only
        # ``self.__dict__`` let a caller shadow e.g. ``reset_state`` with an
        # instance attribute while the dynamics fingerprint stayed unchanged;
        # the next deletion was then rejected, leaving the environment stuck
        # in the corrupted state.  Preserve support for genuinely new public
        # extension attributes, but never allow an existing mechanics/API name
        # to be rebound after construction.
        existing_contract_name = (
            name in self.__dict__
            or any(name in base.__dict__ for base in type(self).__mro__)
        )
        mutable_private = name in _POST_INIT_MUTABLE_PRIVATE_ENV_ATTRIBUTES
        if frozen and existing_contract_name and not mutable_private:
            raise AttributeError(
                f"{name} is read-only after SoccerEnv construction; "
                "construct a new environment to change its static contract"
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        """Prevent delete-then-rebind from bypassing the static freeze."""

        frozen = self.__dict__.get("_static_contract_frozen", False)
        mutable_private = name in _POST_INIT_MUTABLE_PRIVATE_ENV_ATTRIBUTES
        if frozen and name in self.__dict__ and not mutable_private:
            raise AttributeError(
                f"{name} is read-only after SoccerEnv construction and cannot be deleted"
            )
        object.__delattr__(self, name)

    def __init__(
        self,
        n_agents: int = 11,
        n_opponents: int = 11,
        agent_team: list[Agent] | None = None,
        opponent_team: list[Agent] | None = None,
        game_duration: int | None = None,
        control_fps: float = DEFAULT_CONTROL_FPS,
        halftime: bool = True,   # False면 하프타임 전환(공수교대 킥오프) 비활성 — 데모/짧은 롤아웃용
        compress_stamina_to_episode: bool = False,
        *,
        substitutions: list[Substitution] | None = None,
        # 벤치는 팀별 **투입 후보 명단**이다. 관측된 교체를 재현하는 ``substitutions``와
        # 독립적으로 켤 수 있다 — 스케줄은 '언제 누가'라는 확정 사실이고, 벤치는 '누가
        # 들어올 수 있는가'라는 선택지다. 자동 교체 알고리즘은 후자만 쓴다.
        bench: dict[int, list[BenchPlayer]] | None = None,
        bench_size: int | None = None,
        # 팀당 총 교체 상한. 5는 기본값일 뿐이며 벤치 크기와 독립적으로
        # 0..MAX_BENCH_SIZE 범위에서 설정한다.
        max_substitutions: int = DEFAULT_MAX_SUBSTITUTIONS,
        # [IFAB Law 3] 한 정지에서 함께 바꿀 수 있는 인원. 규칙에는 상한이 없으므로 기본값은
        # 교체 인원 전체다 — 다섯 명을 한 번에 넣는 것도 합법이다. ``fori_loop``으로 돌아
        # 이 값이 커져도 그래프 크기는 그대로다.
        max_simultaneous_substitutions: int | None = None,
        substitution_mode: str | Callable = "schedule",
        # 포메이션 지휘 — 교체와 **같은 자리**의 결정이다. 축은 서로 독립이라 어느 조합도
        # 설정만으로 만들어진다(:mod:`formation` 참조).
        formation_mode: str | Callable = "fixed",
        base_layout: str = "kickoff",
        # 감독 — 교체와 포메이션을 **한 뷰에서 함께** 정한다. 주면 위의 두 모드보다 우선한다.
        # 두 모드는 감독을 조립하는 단축표기로 남는다(기존 설정을 깨지 않기 위해서다).
        manager: str | Callable | None = None,
        # 세트피스 키커 — 종전에는 공에 최근접인 선수가 전부 찼다. "nearest"는 그 동작을
        # 그대로 남긴 호환 모드이고, "auto"가 종류별 규칙 선택기다.
        restart_taker_mode: str | Callable = "nearest",
        decision_params=None,
        setpiece_plans: dict | None = None,
        ball_config: Ball | None = None,
        stadium_config: Stadium | None = None,
        engine_config: Engine | None = None,
        foul_config: Foul | None = None,
        reward_config: Reward | None = None,
    ):
        # State, observation and golden-rollout contracts are deliberately
        # float32.  With global JAX x64 enabled, weak Python literals promote
        # selected geometry branches to float64 and lax.cond then either
        # rejects mismatched State leaves or silently changes trajectories.
        # Fail before any cache/fingerprint is built instead of exposing a
        # backend-dependent half-supported mode.
        _validate_float32_runtime()
        if bool(jax.config.jax_threefry_partitionable):
            raise RuntimeError(
                "SoccerEnv's reproducible PRNG contract requires "
                "JAX_THREEFRY_PARTITIONABLE=0; the current dynamics/golden "
                "trajectory uses the non-partitionable threefry split"
            )
        # These are on-pitch slot counts.  Validate the original-width scalar
        # before canonicalising it so NumPy integers cannot narrow through a
        # JAX shape, and reject oversized rosters before touching caller lists
        # or entering the O(N^2) roster/geometry paths.
        for name, value in (
            ("n_agents", n_agents),
            ("n_opponents", n_opponents),
        ):
            if not _is_host_integral(value):
                raise TypeError(
                    f"{name} must be an integer non-boolean scalar, got {value!r}"
                )
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
            if int(value) > MAX_ENV_TEAM_PLAYERS:
                raise ValueError(
                    f"{name} exceeds SoccerEnv's on-pitch team limit: "
                    f"{int(value)} > MAX_ENV_TEAM_PLAYERS={MAX_ENV_TEAM_PLAYERS}"
                )
        n_agents = int(n_agents)
        n_opponents = int(n_opponents)
        if not isinstance(halftime, bool):
            raise ValueError(f"halftime must be a bool, got {halftime!r}")
        if not isinstance(compress_stamina_to_episode, bool):
            raise ValueError(
                "compress_stamina_to_episode must be a bool, "
                f"got {compress_stamina_to_episode!r}"
            )
        if compress_stamina_to_episode:
            raise ValueError(
                "episode-length stamina compression is unsupported; use the "
                "fixed 5,400-second physical reference"
            )
        if agent_team is None and opponent_team is None:
            if n_agents != DEFAULT_TEAM_SIZE or n_opponents != DEFAULT_TEAM_SIZE:
                raise ValueError(
                    "omitted rosters are available only for the default 11v11 setup; "
                    "provide agent_team and opponent_team for custom team sizes"
                )
            agent_team = _default_roster(0)
            opponent_team = _default_roster(DEFAULT_TEAM_SIZE)
        elif agent_team is None or opponent_team is None:
            raise ValueError("agent_team and opponent_team must be provided together")

        # Every config is a frozen dataclass, so ``dataclasses.replace`` accepts
        # *any* dataclass instance.  Without an explicit class check, swapping
        # two keyword arguments survives construction and fails much later with
        # an unrelated AttributeError (for example ``ball_config=Engine()``
        # reports that Engine has no ``radius``).  Keep the public boundary
        # strict and identify the offending argument before deriving metadata.
        for config_name, value, expected_type in (
            ("ball_config", ball_config, Ball),
            ("stadium_config", stadium_config, Stadium),
            ("engine_config", engine_config, Engine),
            ("foul_config", foul_config, Foul),
            ("reward_config", reward_config, Reward),
        ):
            if value is not None and not isinstance(value, expected_type):
                raise TypeError(
                    f"{config_name} must be {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

        super().__init__(num_agents=n_agents + n_opponents)
        self.n_agents: int = n_agents
        self.n_opponents: int = n_opponents
        self.N: int = n_agents + n_opponents
        # Freeze public roster containers.  Mutable lists previously let callers
        # change render metadata after State/static meta/fingerprint were cached.
        self.agent_team: tuple[Agent, ...] = tuple(agent_team)
        self.opponent_team: tuple[Agent, ...] = tuple(opponent_team)

        # config dataclass는 전부 immutable이다. 환경별 인스턴스를 소유해 객체 정체성까지 분리하고,
        # 초기화 뒤 캐시된 파생 기하(hx/goal_w 등)와 config를 외부 mutation으로 갈라놓을 수 없게 한다.
        self.b_cfg = Ball() if ball_config is None else replace(ball_config)
        self.s_cfg = Stadium() if stadium_config is None else replace(stadium_config)
        self.e_cfg = Engine() if engine_config is None else replace(engine_config)
        self.f_cfg = Foul() if foul_config is None else replace(foul_config)
        self.r_cfg = Reward() if reward_config is None else replace(reward_config)
        self._validate_configuration()
        self._validate_rosters()

        # Type checks above deliberately see the original public values.  Only
        # after they pass do we canonicalise all immutable physics inputs.
        self.b_cfg = _canonical_dataclass(self.b_cfg)
        self.s_cfg = _canonical_dataclass(self.s_cfg)
        self.e_cfg = _canonical_dataclass(self.e_cfg)
        self.f_cfg = _canonical_dataclass(self.f_cfg)
        self.r_cfg = _canonical_dataclass(self.r_cfg)
        self.agent_team = tuple(_canonical_dataclass(row) for row in self.agent_team)
        self.opponent_team = tuple(
            _canonical_dataclass(row) for row in self.opponent_team
        )
        # Float32 canonicalisation can collapse two distinct Python values onto
        # one runtime value.  Re-run coupled/strict geometry checks on the
        # values dynamics will actually consume: e.g. 49.9999999 < 50.0 in
        # Python, but both become 50.0f and can no longer satisfy a strict
        # penalty-area-within-pitch relation.  The first pass above is still
        # required so wrong public scalar kinds are diagnosed before coercion.
        self._validate_configuration()
        self._validate_rosters()

        self.timebase = Timebase(
            dt_phys=self.e_cfg.dt_phys,
            control_fps=control_fps,
        )
        # Timebase is also a public general clock utility and deliberately
        # retains its wider int32 ratio contract.  SoccerEnv, however, creates
        # decimation-shaped pins, a static scan and optionally a full State
        # stack for every action, so enforce its practical resource boundary
        # before deriving any duration-sized state.
        if self.timebase.decimation > MAX_CONTROL_DECIMATION:
            raise ValueError(
                "SoccerEnv physics substeps per control frame exceed its "
                "bounded scan contract: "
                f"{self.timebase.decimation} > "
                f"MAX_CONTROL_DECIMATION={MAX_CONTROL_DECIMATION}"
            )
        self.control_fps: float = self.timebase.control_fps
        self.control_dt: float = self.timebase.control_dt

        # 기본 경기 길이의 원천은 초 단위다. 제어 FPS를 바꾸고 game_duration을 생략하면
        # 같은 90분을 나타내는 제어 스텝 수가 시간축에서 자동으로 다시 계산된다.
        if game_duration is None:
            game_duration = self.timebase.control_steps_for(
                DEFAULT_MATCH_DURATION_SECONDS,
                minimum=1,
            )
        if not isinstance(game_duration, int) or isinstance(game_duration, bool) or game_duration <= 0:
            raise ValueError(f"game_duration must be a positive integer, got {game_duration!r}")
        if game_duration > ROLE_GAIN_EXACT_MAX_SAMPLES:
            raise ValueError(
                "game_duration exceeds the lossless role_gain inverse contract: "
                f"{game_duration} > {ROLE_GAIN_EXACT_MAX_SAMPLES} control steps"
            )
        self.game_duration: int = game_duration

        # 필드 반경(경계 클립·분리에서 사용) — Stadium property에서 파생
        self.hx: float = self.s_cfg.half_length
        self.hy: float = self.s_cfg.half_width

        # episode 종료시간과 에너지 물리시간을 분리한다. clip/inverse를 짧게 잘라도 stamina는
        # 항상 Engine의 물리 기준(기본 90분 = 5,400초)으로 적분한다.
        self.episode_duration_s: float = self.timebase.seconds_for_control_steps(self.game_duration)
        self.long_stamina_reference_duration_s: float = (
            self.e_cfg.long_stamina_reference_duration_s
        )
        self.long_stamina_drain_base: float = long_stamina_drain_base(
            self.e_cfg.long_stamina_end_frac,
            self.long_stamina_reference_duration_s,
            self.e_cfg.long_stamina_reference_workload,
        )
        self.long_stamina_tail_decay: float = long_stamina_tail_decay(
            self.e_cfg.long_stamina_end_frac,
            self.e_cfg.long_stamina_tail_knee,
        )
        _require_float32_scalar(
            "long_stamina_drain_base", self.long_stamina_drain_base
        )
        max_stamina_drain = self.long_stamina_drain_base * (
            self.e_cfg.long_stamina_idle_load
            + 4.0 * self.e_cfg.long_stamina_speed_load
            + 16.0 * self.e_cfg.long_stamina_accel_load
            + 2.0 * max(0.0, self.e_cfg.long_stamina_sprint_mult - 1.0)
        )
        _require_float32_scalar("maximum long stamina drain rate", max_stamina_drain)
        _require_float32_scalar(
            "maximum long stamina drain per physics tick",
            max_stamina_drain * self.e_cfg.dt_phys,
        )

        self.r_ball: float = self.b_cfg.radius
        self.r_player: float = self.e_cfg.r_player
        self.players: tuple[Agent, ...] = self.agent_team + self.opponent_team
        self.initial_player_ids = jnp.asarray(
            [int(player.id) for player in self.players],
            dtype=jnp.int32,
        )
        # JaxMARL agent 집합은 에피소드 동안 정적이어야 한다. person id는 교체 시 바뀌므로
        # 외부 dict API는 identity가 아닌 영구 slot key를 사용하고, 사람 identity는 State.player_id로 낸다.
        self._agent_keys: tuple[str, ...] = tuple(
            f"slot_{i}" for i in range(self.N)
        )
        self.minimum_team_players = jnp.asarray(
            [min(IFAB_MIN_TEAM_PLAYERS, self.n_agents),
             min(IFAB_MIN_TEAM_PLAYERS, self.n_opponents)],
            dtype=jnp.int32,
        )
        self.player_indices = jnp.arange(self.N, dtype=jnp.int32)
        self.team_indices = jnp.array([0, self.n_agents], dtype=jnp.int32)
        self.others_idx = jnp.asarray(
            [[j for j in range(self.N) if j != i] for i in range(self.N)],
            dtype=jnp.int32,
        )
        self.substitutions: tuple[Substitution, ...] = self._validate_substitutions(substitutions)
        (self.bench_size, self.max_substitutions, self._bench_plan) = (
            self._validate_bench(bench, bench_size, max_substitutions)
        )
        # 교체가 0명이면 이 하위체계 자체가 꺼진다. 그때 동시 교체 인원을 1 이상으로
        # 요구하면 "0은 합법"이라고 검증해 놓고 생성자가 죽는다.
        if self.max_substitutions == 0:
            self.max_simultaneous_substitutions = 1
        else:
            simultaneous = (self.max_substitutions
                            if max_simultaneous_substitutions is None
                            else max_simultaneous_substitutions)
            if (not isinstance(simultaneous, int)
                    or isinstance(simultaneous, bool)
                    or not 1 <= simultaneous <= self.max_substitutions):
                raise ValueError(
                    "max_simultaneous_substitutions must be an int in "
                    f"[1, {self.max_substitutions}], got {simultaneous!r}")
            self.max_simultaneous_substitutions = simultaneous
        self._substitution_decider = substitution_module.resolve(substitution_mode)
        self.substitution_mode = (
            substitution_mode if isinstance(substitution_mode, str) else "custom"
        )
        self._substitution_plan = self._compile_substitutions(self.substitutions)

        # 페널티 박스 반치수·골대 규격 — 경합/파울/이벤트 판정에서 사용
        self.pen_len: float = self.s_cfg.penalty_area_length
        self.pen_hw: float = self.s_cfg.penalty_area_width / 2.0
        self.goal_w: float = self.s_cfg.goal_width
        self.goal_h: float = self.s_cfg.goal_height

        # 골 프레임 캡슐 축 — ``goal_width``/``goal_height``가 프레임 **안쪽** 치수이므로
        # 축은 반지름만큼 바깥에 있다(포스트는 y로, 크로스바는 z로). 정적 기하라 여기서
        # 한 번 만들고 substep 스캔은 읽기만 한다.
        # ``goal_frame_radius=0``이면 프레임 자체를 끈다 — 반지름 0인 기둥을 계산하면
        # 축이 골문 안쪽 면과 겹쳐 없던 충돌이 생기므로, 파이썬 수준에서 경로를 지운다.
        frame_r: float = self.e_cfg.goal_frame_radius
        self._goal_frame_active: bool = frame_r > 0.0
        post_y = self.goal_w / 2.0 + frame_r
        self._goal_post_axis_xy = jnp.asarray(
            [(sx * self.hx, sy * post_y) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)],
            jnp.float32,
        )
        self._goal_bar_axis_xz = jnp.asarray(
            [(sx * self.hx, self.goal_h + frame_r) for sx in (-1.0, 1.0)],
            jnp.float32,
        )
        # 크로스바 몸통은 포스트 축까지 덮는다. 그래도 두 원기둥의 합집합에는 바깥 위
        # 모서리에 노치가 남으므로(두 축 모두 반지름 안·두 축 범위 모두 밖), 캡슐의 둥근
        # 끝면에 해당하는 모서리 구를 따로 판정한다. 포스트 위 끝과 크로스바 양 끝의
        # 반구가 같은 점에 중심을 두므로 네 점이면 충분하다.
        self._goal_bar_half_span: float = post_y
        self._goal_corner_xyz = jnp.asarray(
            [(sx * self.hx, sy * post_y, self.goal_h + frame_r)
             for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)],
            jnp.float32,
        )

        # 킥오프 기준 포메이션(N,2): team0 원본 + team1 점대칭(초기 배치와 동일).
        # _kickoff_positions가 attack_dir 부호로 점대칭 반전해 골 후·후반 재배치에 재사용.
        self._static_meta = build_static_metadata(
            e_cfg=self.e_cfg,
            n_agents=self.n_agents,
            n_opponents=self.n_opponents,
            agent_infos=self.agent_team,
            opponent_infos=self.opponent_team,
        )
        self.base_formation = self._static_meta[3]
        # 은퇴 명단은 **벤치와 다른 축**이다. 벤치 폭에 묶어 두면 스케줄 전용 env
        # (``bench_size=0``)가 나간 선수를 기록할 자리를 아예 못 갖는다 — 실제로 (2,0)이라
        # 예약 교체 뒤 명단이 비어 있었다.
        #
        # 두 경로는 **배타적이 아니라 가산적**이다. 한 경기에서 스케줄 재현과 벤치 교체를
        # 함께 쓸 수 있으므로 각 경로의 최대치를 더해야 한다 — 큰 쪽만 잡으면(예: 벤치1·
        # 상한1·스케줄1이면 깊이 1) 두 번째로 나간 선수의 신원이 통째로 사라진다.
        scheduled_per_team = [0, 0]
        team_of = np.asarray(self._static_meta[0])
        for row in self.substitutions:
            scheduled_per_team[int(team_of[int(row.slot)])] += 1
        self.retired_depth = (
            max(self.bench_size, self.max_substitutions)
            + max(scheduled_per_team))

        # 레이아웃 표는 킥오프 포메이션에서 파생되므로 base_formation 뒤에 짓는다.
        self._build_formation_plan(formation_mode, base_layout)
        self._build_manager(manager)
        self._restart_taker_decider = setpiece_taker_module.resolve(
            restart_taker_mode)
        # 결정자의 규약은 생성 시 한 번 확정한다(정적). 파라미터는 호출마다 바뀌지만
        # **어느 규약으로 부를지**는 바뀌지 않아야 프로그램이 하나로 유지된다.
        self._manager_takes_params = _decider_takes_params(self._manager)
        self._taker_takes_params = _decider_takes_params(
            self._restart_taker_decider)
        # 키커 파라미터는 호출 범위에 매달아 둔다. 지정 호출이 서브스텝 스캔 안쪽 아홉
        # 곳에 흩어져 있어 인자로 꿰려면 규칙 함수 여덟 개의 시그니처를 전부 넓혀야 하고,
        # 그 인자는 거의 항상 ``None``이라 규칙 코드에 잡음만 남는다. 값은 추적 중에만
        # 실려 있다가 호출이 끝나면 복원되므로 객체에 트레이서가 남지 않는다.
        self._taker_params = None
        # Fixed-shape external taker overrides exist only for one native command
        # call.  They stay separate from learned-decider parameters so enabling
        # an override never changes the configured policy PyTree.
        self._native_taker_command = None
        self._active_randomness_control = None
        # 파라미터 결정자의 **기본값**. 호출마다 넘기는 것이 이 기능의 요점이지만, 모양만
        # 필요한 내부 경로(정책 팩토리가 앵커를 얻으려 부르는 ``reset_state`` 등)는 넘길
        # 자리가 없다. 기본값이 있으면 그런 호출도 같은 결정자로 돌고, 핫패스는 호출 인자로
        # 덮어써 재컴파일을 피한다. 여기 담긴 가중치는 ``dynamics_fingerprint``에 들어가지
        # 않는다 — 정책 신원은 환경 동역학이 아니기 때문이다.
        self._default_decision_params = decision_params
        if decision_params is not None and not isinstance(
            decision_params, manager_module.DecisionParams
        ):
            raise TypeError(
                "decision_params must be a manager.DecisionParams, got "
                f"{type(decision_params).__name__}")
        for axis, takes, field in (
            ("manager", self._manager_takes_params, "manager"),
            ("restart taker", self._taker_takes_params, "taker"),
        ):
            if takes and getattr(decision_params, field, None) is None:
                raise ValueError(
                    f"the {axis} decider uses the (params, view, key) protocol, so "
                    f"SoccerEnv needs decision_params={field}=... as its default; "
                    "per-call parameters then override it without recompiling")
        self.restart_taker_mode = (
            restart_taker_mode if isinstance(restart_taker_mode, str)
            else "custom")
        self._setpiece_plan_rows = setpiece_plans
        self._setpiece_plans = setpiece_taker_module.compile_plans(
            setpiece_plans, (TEAM_0, TEAM_1))
        # 구름 사거리 — 접촉 분류의 **도달성 게이트**가 쓴다. 슛 라벨에서 거리 상한을
        # 없애자 자기 진영에서 중앙으로 길게 찬 공이 상대 골문 폭 안을 지나 슛이 됐는데,
        # 실측으로 발 슛 라벨 31건 중 40 m 초과가 58.1 %였고 **그 18건 전부가 골라인
        # 전에 정지**했다. 물리적으로 도달하지 못하는 슛은 슛이 아니다.
        #
        # 새 상수를 만들지 않고 env 자신의 감속 테이블에서 유도한다. 굴림 감속이
        # ``d(u)``이면 정지까지의 거리는 ``∫ u/d(u) du``이고, 그 적분을 knot마다
        # 미리 쌓아 두면 런타임에는 보간 한 번이면 된다. 공중 구간은 여기 없으므로
        # 이 값은 **보수적 하한**이다 — 띄운 공은 이보다 멀리 간다.
        roll_v = np.asarray(self.e_cfg.roll_v_knots, np.float64)
        roll_d = np.asarray(self.e_cfg.roll_d_knots, np.float64)
        segment = np.diff(roll_v) * 0.5 * (
            roll_v[:-1] / roll_d[:-1] + roll_v[1:] / roll_d[1:])
        self._roll_range_knots = jnp.asarray(
            np.concatenate([[0.0], np.cumsum(segment)]), jnp.float32)
        self.field_half = jnp.array([self.hx, self.hy], dtype=jnp.float32)
        self.field_size = jnp.array([self.s_cfg.length, self.s_cfg.width], dtype=jnp.float32)

        # ── JaxMARL 스페이스 — 트레이너가 env.observation_space(agent)/action_space(agent)
        # (base 메서드, 아래 dict 조회) 또는 obs_dim/state_dim/action_dim 속성으로 조회.
        # 관측은 정규화 목표 ~[-1,1]이되 엄격 클립이 아니므로(env/README.md §5.5) 비유계 Box로 정직하게,
        # 행동은 kick gate만 [0,1], 나머지는 [-1,1]이다(constants.ACTION_DIM).
        # 디코더가 gate > 0.5를 사용하는데 Box를 전 차원 [-1,1]로 선언하면
        # 공간 샘플러와 트레이너가 문서화된 Bernoulli 전구체 분포를 받지 못한다.
        self.obs_dim: int = self.obs_spec()["dim"]
        self.state_dim: int = self.state_spec()["dim"]
        self.action_dim: int = ACTION_DIM
        self.halftime = halftime
        # 공개 dict를 그대로 내주면 호출자가 고친 값과 manifest·fingerprint가 어긋난 채로
        # 남는다(공개 schema는 999인데 fingerprint는 원래 값). 내부는 plain dict로 두고
        # 공개는 pickle-safe read-only 방어 복사본을 돌려주는 property로 분리한다.
        self._schema_version = {
            "environment_dynamics": ENV_DYNAMICS_VERSION,
            "action": ACTION_SCHEMA_VERSION,
            "observation": OBS_SCHEMA_VERSION,
            "state": STATE_SCHEMA_VERSION,
            "state_container": STATE_CONTAINER_SCHEMA_VERSION,
            "affordance": AFFORDANCE_SCHEMA_VERSION,
            "movement_dynamics": MOVEMENT_DYNAMICS_VERSION,
            "energy_dynamics": ENERGY_DYNAMICS_VERSION,
        }
        self._dynamics_metadata = self._build_dynamics_metadata()
        fingerprint_payload = json.dumps(
            self._dynamics_metadata,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.dynamics_fingerprint: str = hashlib.sha256(fingerprint_payload).hexdigest()
        self.observation_spaces = _ImmutableMapping({
            a: _ReadOnlyBox(-jnp.inf, jnp.inf, (self.obs_dim,))
            for a in self._agent_keys
        })
        action_low = jnp.full((self.action_dim,), ACTION_MIN, dtype=jnp.float32)
        action_low = action_low.at[ACTION_KICK_GATE].set(0.0)
        action_high = jnp.full((self.action_dim,), ACTION_MAX, dtype=jnp.float32)
        self.action_spaces = _ImmutableMapping({
            a: _ReadOnlyBox(action_low, action_high, (self.action_dim,))
            for a in self._agent_keys
        })
        # Set only after every public scalar/cache/container has been derived.
        # Private lazy caches (for example the rule-policy solver tables) remain
        # writable through the name-based exception in ``__setattr__``.
        self._static_contract_frozen = True

    @property
    def agents(self) -> list[str]:
        """JaxMARL agent names as a defensive compatibility view.

        Agent keys are permanent roster-slot identities and therefore part of
        the environment's static shape/metadata contract.  Returning the
        internal container used to let ``env.agents.append(...)`` or an in-place
        reorder silently desynchronise dict APIs from the cached spaces and JAX
        arrays.  Keep the runtime single source of truth immutable while
        preserving JaxMARL's customary public list interface.
        """

        return list(self._agent_keys)

    @agents.setter
    def agents(self, value) -> None:
        raise AttributeError(
            "agents is a read-only defensive view of immutable slot keys"
        )

    @property
    def schema_version(self):
        """공개 스키마 버전 — 쓰기를 거부하지만 **dict로 동작하는** 읽기 전용 view.

        이 값은 fingerprint payload에 그대로 들어가므로 나중에 바뀌면 안 된다. 쓰기를 시도하면
        ``TypeError``가 나서 '공개 dict는 바꿨는데 manifest·fingerprint는 그대로'인 모순 상태가
        만들어지지 않는다.

        ``mappingproxy``가 아니라 dict 서브클래스인 이유는 ``json.dumps``가 mappingproxy를
        직렬화하지 못하기 때문이다 — 불변성을 얻겠다고 기존 manifest 코드의 직렬화를 깨면
        안 된다. ``dict(...)``, ``json.dumps(...)``, 인덱싱, 순회는 모두 그대로 동작한다.
        """

        return _ReadOnlyVersions(self._schema_version)

    def _validate_bench(self, bench, bench_size, max_substitutions):
        """벤치 명단을 검증하고 고정 shape 배열로 굳힌다.

        ``bench_size``는 **State shape**라 생성 시점에 정해져야 한다. 명단보다 크면 남는
        자리는 ``NO_PLAYER``로 비워 두고, 작으면 거부한다 — 조용히 잘라내면 사용자가 넣은
        선수가 사라진 것을 모른다.

        ``max_substitutions``는 벤치 크기와 **독립**이다. 9명을 앉히고 5명만 넣을 수 있는
        것이 정규 경기이고, 그 선택이 알고리즘이 푸는 문제다.
        """

        if not isinstance(max_substitutions, int) or isinstance(max_substitutions, bool):
            raise ValueError(
                f"max_substitutions must be an int, got {max_substitutions!r}")
        if not 0 <= max_substitutions <= MAX_BENCH_SIZE:
            raise ValueError(
                f"max_substitutions must lie in [0, {MAX_BENCH_SIZE}], "
                f"got {max_substitutions}")

        rows = {}
        if bench is not None:
            if not isinstance(bench, Mapping):
                raise ValueError(
                    f"bench must be a mapping of team -> [BenchPlayer], "
                    f"got {type(bench).__name__}")
            for team, players in bench.items():
                if team not in (TEAM_0, TEAM_1):
                    raise ValueError(
                        f"bench team must be {TEAM_0} or {TEAM_1}, got {team!r}")
                if not isinstance(players, (list, tuple)):
                    raise ValueError("bench entries must be a list of BenchPlayer")
                for row in players:
                    if not isinstance(row, BenchPlayer):
                        raise ValueError(
                            f"bench entries must be BenchPlayer, got {row!r}")
                rows[int(team)] = list(players)

        needed = max((len(v) for v in rows.values()), default=0)
        if bench_size is None:
            # 명단이 없으면 **0**이다. 교체 상한(5)에 맞춰 자리를 잡아 두면 아무도 못 들어오는
            # 빈 벤치를 State에 싣고, 데드볼마다 K회의 무효 투영 기하를 헛돌린다 —
            # ``bench_size == 0`` 빠른 반환이 영영 안 걸린다.
            bench_size = needed
        if not isinstance(bench_size, int) or isinstance(bench_size, bool):
            raise ValueError(f"bench_size must be an int, got {bench_size!r}")
        if not 0 <= bench_size <= MAX_BENCH_SIZE:
            raise ValueError(
                f"bench_size must lie in [0, {MAX_BENCH_SIZE}], got {bench_size}")
        if needed > bench_size:
            raise ValueError(
                f"bench_size={bench_size} cannot hold {needed} listed players; "
                "raise bench_size or shorten the roster")

        # 필드 검증. 종전에는 ID를 ``int()``로 자르고 나머지를 곧바로 캐스팅해서,
        # player_id=9000.9(런타임 9000 / 메타데이터 9000.9로 갈림), 음수 키, 무한 reach,
        # ball_control=2, 문자열 GK가 전부 통과했다. 잘린 ID는 합법 정수라 나중에 잡히지도
        # 않고, NaN/inf는 strict JSON 직렬화에서 manifest 자체를 깨뜨린다.
        for team, players in rows.items():
            for row in players:
                for name, low, high in (
                    # 속도 0은 생성자를 통과하지만 투영이 ``vmax > 0``을 요구해 **조용히
                    # 투입되지 않는다** — 거부도 동작도 아닌 최악의 형태라 여기서 막는다.
                    ("speed", 0.1, 20.0),
                    ("tall", 0.5, 2.6),
                    ("reach_z_max", 0.5, 4.0),
                    ("ball_control", 0.0, 1.0),
                    ("endurance_factor", 0.01, 100.0),
                ):
                    value = getattr(row, name)
                    if (not isinstance(value, numbers.Real)
                            or isinstance(value, (bool, np.bool_))
                            or not math.isfinite(float(value))
                            or not low <= float(value) <= high):
                        raise ValueError(
                            f"BenchPlayer.{name} must be a finite number in "
                            f"[{low}, {high}], got {value!r}")
                # 결합 조건 — 범위만 봐서는 잡히지 않는다. 손이 머리보다 낮으면 헤딩만
                # 가능한 선수가 되어 접촉 판정이 뒤집힌다.
                if float(row.reach_z_max) < float(row.tall):
                    raise ValueError(
                        f"BenchPlayer.reach_z_max ({row.reach_z_max}) must be at "
                        f"least tall ({row.tall})")
                if not isinstance(row.is_gk, (bool, np.bool_)):
                    raise ValueError(
                        f"BenchPlayer.is_gk must be a bool, got {row.is_gk!r}")
                if (not isinstance(row.player_id, numbers.Integral)
                        or isinstance(row.player_id, (bool, np.bool_))):
                    raise ValueError(
                        "BenchPlayer.player_id must be an int, got "
                        f"{row.player_id!r}")
                anchor = row.role_pos
                if (not isinstance(anchor, (tuple, list)) or len(anchor) != 2
                        or any(not isinstance(v, numbers.Real)
                               or isinstance(v, (bool, np.bool_))
                               or not math.isfinite(float(v)) for v in anchor)):
                    raise ValueError(
                        f"BenchPlayer.role_pos must be two finite numbers, "
                        f"got {anchor!r}")
                if (abs(float(anchor[DIM_X])) > self.hx
                        or abs(float(anchor[DIM_Y])) > self.hy):
                    raise ValueError(
                        f"BenchPlayer.role_pos {anchor!r} lies outside the pitch")

        starters = {int(a.id) for a in self.agent_team} | {
            int(a.id) for a in self.opponent_team
        }
        seen = set()
        for team, players in rows.items():
            for row in players:
                pid = int(row.player_id)
                if pid < 0:
                    raise ValueError(
                        f"bench player_id must be non-negative, got {pid}")
                if pid in starters:
                    raise ValueError(
                        f"bench player_id {pid} is already on the pitch")
                if pid in seen:
                    raise ValueError(f"duplicate bench player_id {pid}")
                seen.add(pid)

        factor = self.e_cfg.reach_height_factor
        def column(getter, dtype, fill):
            table = np.full((TEAM_COUNT, bench_size), fill, dtype)
            for team, players in rows.items():
                for i, row in enumerate(players):
                    table[team, i] = getter(row)
            return jnp.asarray(table)

        plan = {
            "player_id": column(lambda r: int(r.player_id), np.int32, NO_PLAYER),
            "vmax": column(lambda r: float(r.speed), np.float32, 0.0),
            "reach_z": column(lambda r: float(r.reach_z_max), np.float32, 0.0),
            "head_z": column(lambda r: float(r.tall) * factor, np.float32, 0.0),
            "player_ctrl": column(lambda r: float(r.ball_control), np.float32, 0.0),
            "endurance_factor": column(
                lambda r: float(r.endurance_factor), np.float32, 1.0
            ),
            "is_gk": column(lambda r: bool(r.is_gk), np.bool_, False),
        }
        role = np.zeros((TEAM_COUNT, bench_size, 2), np.float32)
        for team, players in rows.items():
            for i, row in enumerate(players):
                role[team, i] = np.asarray(row.role_pos, np.float32)
        plan["role_pos"] = jnp.asarray(role)
        # 명단 원본을 남긴다 — dynamics manifest가 벤치를 기록해야 산출물만 보고
        # 재현할 수 있다. plan은 고정 shape 배열이라 '몇 번째가 비었는지'가 섞인다.
        self._bench_roster = {team: tuple(players) for team, players in rows.items()}
        return bench_size, max_substitutions, plan

    def _validate_substitutions(
        self, substitutions: list[Substitution] | None
    ) -> tuple[Substitution, ...]:
        """구조적으로 성립하지 않는 교체 스케줄을 JIT 전에 거부한다.

        경기 규칙상의 교체 횟수를 실측 코퍼스에 맞춰 좁게 강제하지는 않지만,
        모든 행을 매 step 전이에 포함하므로 정적 안전 상한은 강제한다. 실측 최대인
        팀당 6회(연장·추가 window 포함)보다 넓은 상한이므로 관측된 사실은
        그대로 받아들인다. 그 안에서 slot/tick 범위, 같은 tick의 같은 slot
        중복, 한 번 교체된 사람이 다시 들어오지 않는 identity 규칙을 검사한다.
        """

        if substitutions is None:
            return ()
        if not isinstance(substitutions, (list, tuple)):
            raise ValueError(
                f"substitutions must be a list of Substitution, got {type(substitutions).__name__}")
        # Check length before copying or validating rows.  Apart from bounding
        # the compiled transition, this makes an oversized adversarial list a
        # constant-work constructor failure instead of an O(rows) validation
        # pass followed by an enormous per-step loop.
        if len(substitutions) > MAX_SUBSTITUTION_SCHEDULE_ROWS:
            raise ValueError(
                "substitutions must contain at most "
                f"{MAX_SUBSTITUTION_SCHEDULE_ROWS} rows, got {len(substitutions)}"
            )
        rows = list(substitutions)
        for row in rows:
            if not isinstance(row, Substitution):
                raise ValueError(f"substitutions must contain Substitution, got {row!r}")
            if (
                not _is_host_integral(row.slot)
                or not 0 <= int(row.slot) < self.N
            ):
                raise ValueError(f"substitution slot must lie in [0, {self.N}), got {row.slot!r}")
            if (
                not _is_host_integral(row.tick)
                or not 0 <= int(row.tick) < self.game_duration
            ):
                raise ValueError(
                    f"substitution tick must lie in [0, {self.game_duration}), got {row.tick!r}")
            id_bounds = np.iinfo(np.int32)
            if (
                not _is_host_integral(row.player_id)
                or not 0 <= int(row.player_id) <= id_bounds.max
            ):
                raise ValueError(
                    "substitution player_id must be a non-negative integer fitting the int32 State contract, "
                    f"got {row.player_id!r}"
                )
            vectors = {"entry_pos": row.entry_pos, "role_pos": row.role_pos}
            for name, value in vectors.items():
                if (
                    not _is_host_real_vector(value, (DIM_Z,))
                    or not all(math.isfinite(float(v)) for v in value)
                ):
                    raise ValueError(
                        f"substitution {name} must contain two finite real values, "
                        f"got {value!r}"
                    )
                for component in value:
                    _require_float32_scalar(f"substitution {name}", component)
            if (
                abs(float(row.entry_pos[DIM_X])) > self.hx + self.e_cfg.player_boundary_margin
                or abs(float(row.entry_pos[DIM_Y])) > self.hy + self.e_cfg.player_boundary_margin
            ):
                raise ValueError(
                    "substitution entry_pos lies outside the player boundary: "
                    f"{row.entry_pos!r}"
                )
            # role_pos is stored in the player's attack-folded frame, but folding is
            # only a sign flip, so the same absolute player boundary applies.  An
            # unbounded finite prior is observable while role_pos_count==0 (notably
            # throughout a dead ball) and violates the role-anchor state invariant.
            if (
                abs(float(row.role_pos[DIM_X])) > self.hx + self.e_cfg.player_boundary_margin
                or abs(float(row.role_pos[DIM_Y])) > self.hy + self.e_cfg.player_boundary_margin
            ):
                raise ValueError(
                    "substitution role_pos lies outside the folded player boundary: "
                    f"{row.role_pos!r}"
                )
            positive = {
                "speed": row.speed,
                "tall": row.tall,
                "reach_z_max": row.reach_z_max,
                "endurance_factor": row.endurance_factor,
            }
            if any(
                not _is_host_real(v)
                or not math.isfinite(float(v))
                or float(v) <= 0.0
                for v in positive.values()
            ):
                raise ValueError(f"substitution physical attributes must be finite and positive: {positive}")
            for name, value in positive.items():
                _require_float32_scalar(f"substitution {name}", value)
            if max(
                float(row.speed), float(row.tall), float(row.reach_z_max)
            ) > _FLOAT32_DYNAMICS_MAGNITUDE_LIMIT:
                raise ValueError(
                    "substitution physical profile exceeds the float32 "
                    "quartic/norm-safe dynamics magnitude limit"
                )
            if not 0.01 <= float(row.endurance_factor) <= 100.0:
                raise ValueError(
                    "substitution endurance_factor must lie in [0.01, 100], "
                    f"got {row.endurance_factor!r}"
                )
            if float(row.speed) * self.e_cfg.dt_phys > _FLOAT32_DYNAMICS_MAGNITUDE_LIMIT:
                raise ValueError(
                    "substitution speed * dt_phys exceeds the float32 "
                    "quartic/norm-safe dynamics magnitude limit"
                )
            if (
                min(
                    self.e_cfg.a_max * self.e_cfg.dt_phys,
                    float(row.speed),
                )
                * self.e_cfg.dt_phys
                < _float32_position_resolution(self.s_cfg, self.e_cfg)
            ):
                raise ValueError(
                    "substitution speed cannot produce a representable "
                    "full-action position step at the configured float32 "
                    "spatial resolution"
                )
            entrant_head_z = float(row.tall) * self.e_cfg.reach_height_factor
            if float(row.reach_z_max) < entrant_head_z:
                raise ValueError(
                    "substitution reach_z_max must be at least the derived head height "
                    f"({row.reach_z_max!r} < {entrant_head_z:.6g})"
                )
            if self.e_cfg.body_top_frac * entrant_head_z <= self.e_cfg.leg_top:
                raise ValueError(
                    "substitution body collision band must have positive height: "
                    "body_top_frac * head_z must exceed leg_top"
                )
            if (
                not _is_host_real(row.ball_control)
                or not math.isfinite(float(row.ball_control))
                or not 0.0 <= float(row.ball_control) <= 1.0
            ):
                raise ValueError(
                    f"substitution ball_control must lie in [0, 1], got {row.ball_control!r}")
            _require_float32_scalar("substitution ball_control", row.ball_control)
            if not isinstance(row.is_gk, (bool, np.bool_)):
                raise ValueError(f"substitution is_gk must be bool, got {row.is_gk!r}")
            for stamina_name, stamina_value in (
                ("stamina_long_entry", row.stamina_long_entry),
                ("stamina_short_entry", row.stamina_short_entry),
            ):
                if (
                    not _is_host_real(stamina_value)
                    or not math.isfinite(float(stamina_value))
                    or not 0.0 <= float(stamina_value) <= 1.0
                ):
                    raise ValueError(
                        f"substitution {stamina_name} must lie in [0, 1], "
                        f"got {stamina_value!r}"
                    )
                _require_float32_scalar(
                    f"substitution {stamina_name}", stamina_value
                )
            if (
                not _is_host_integral(row.yellow_cards)
                or not 0 <= int(row.yellow_cards) < YELLOW_CARD_SEND_OFF_COUNT
            ):
                raise ValueError(
                    "an incoming player must have zero or one yellow card and cannot enter already "
                    f"dismissed, got {row.yellow_cards!r}"
                )
        rows.sort(key=lambda row: (int(row.tick), int(row.slot)))
        seen_slots: set[tuple[int, int]] = set()
        for row in rows:
            key = (int(row.tick), int(row.slot))
            if key in seen_slots:
                raise ValueError(f"two substitutions target slot {row.slot} at tick {row.tick}")
            seen_slots.add(key)
        incoming = [int(row.player_id) for row in rows]
        if len(set(incoming)) != len(incoming):
            raise ValueError("a substituted-in player id appears twice; identities are not reusable")
        starters = {int(value) for value in np.asarray(self.initial_player_ids)}
        reused = sorted(set(incoming) & starters)
        if reused:
            raise ValueError(f"substituted-in player ids collide with starters: {reused}")
        initial_gk = [bool(player.is_gk) for player in self.players]
        for row in rows:
            if bool(row.is_gk) != initial_gk[int(row.slot)]:
                raise ValueError(
                    "a substitution must preserve the slot's goalkeeper role; "
                    f"slot {row.slot} starts is_gk={initial_gk[int(row.slot)]}"
                )
        return tuple(_canonical_dataclass(row) for row in rows)

    def _compile_substitutions(self, rows: tuple[Substitution, ...]) -> dict | None:
        """스케줄을 고정 shape의 JAX 배열로 굳힌다(``fori_loop`` 피연산자)."""

        if not rows:
            return None
        factor = self.e_cfg.reach_height_factor
        return {
            "tick": jnp.asarray([int(row.tick) for row in rows], jnp.int32),
            "slot": jnp.asarray([int(row.slot) for row in rows], jnp.int32),
            "player_id": jnp.asarray([int(row.player_id) for row in rows], jnp.int32),
            "entry_pos": jnp.asarray([list(row.entry_pos) for row in rows], jnp.float32),
            "role_pos": jnp.asarray([list(row.role_pos) for row in rows], jnp.float32),
            "vmax": jnp.asarray([float(row.speed) for row in rows], jnp.float32),
            "reach_z": jnp.asarray([float(row.reach_z_max) for row in rows], jnp.float32),
            "head_z": jnp.asarray([float(row.tall) * factor for row in rows], jnp.float32),
            "player_ctrl": jnp.asarray([float(row.ball_control) for row in rows], jnp.float32),
            "endurance_factor": jnp.asarray(
                [float(row.endurance_factor) for row in rows], jnp.float32
            ),
            "is_gk": jnp.asarray([bool(row.is_gk) for row in rows], jnp.bool_),
            "stamina_long_entry": jnp.asarray(
                [float(row.stamina_long_entry) for row in rows], jnp.float32
            ),
            "stamina_short_entry": jnp.asarray(
                [float(row.stamina_short_entry) for row in rows], jnp.float32
            ),
            "yellow_cards": jnp.asarray([int(row.yellow_cards) for row in rows], jnp.int32),
        }

    def _is_terminal(self, state):
        """경기가 이미 끝난 상태인가 — 시간제한 또는 한 팀의 최소 인원 미달."""

        active_per_team = jnp.stack([
            jnp.sum(state.active_player & (state.team_id == TEAM_0)),
            jnp.sum(state.active_player & (state.team_id == TEAM_1)),
        ]).astype(jnp.int32)
        return ((state.t >= self.game_duration)
                | jnp.any(active_per_team < self.minimum_team_players))

    def _release_unattended_possession(self, state):
        """정지한 채 아무도 곁에 없는 공의 소유권을 중립으로 되돌린다.

        소유권은 **접촉으로만** 바뀐다(ball.py의 trap/bounce, contest의 승자 판정). 그래서
        아무도 공을 건드리지 않으면 마지막 터치 팀에 영원히 래치된다. 규칙 정책은 소유가
        중립일 때만 루즈볼 추격을 켜므로, 래치된 채 곁에 아무도 없으면 양 팀이 대형만
        유지하고 공이 멈춰 선 경기가 된다 — 실측에서 최근접 선수가 9.8 m 떨어진 채 90초가
        흘렀고 양 팀 패스가 동시에 0이었다.

        규칙 vs 규칙은 접촉이 끊이지 않아 소유가 계속 뒤집히므로 이 상태에 도달하지 않는다.
        즉 다르게 행동하는 정책(학습 정책 등)만 노출시키는 잠복 결함이며, 정책이 아니라
        환경이 닫아야 하는 종류다.

        데드볼·재개 중에는 적용하지 않는다 — 그때의 소유는 재개 주체를 뜻하는 규칙 상태이지
        '공을 들고 있다'는 물리 사실이 아니다.
        """

        e_cfg = self.e_cfg
        live = (state.ball_state == BALL_ALIVE) & (~restart_timer_active(state.restart_t))
        resting = (_safe_norm(state.ball_vel[jnp.newaxis, :DIM_Z], axis=1)[0]
                   <= e_cfg.possession_release_speed)
        distance = _safe_norm(
            state.player_pos - state.ball_pos[jnp.newaxis, :DIM_Z], axis=1)
        nearest = jnp.min(jnp.where(state.active_player, distance, jnp.inf))
        unattended = (live & resting & (state.poss_team >= 0)
                      & (nearest > e_cfg.possession_release_radius))
        return state._replace(
            poss_team=jnp.where(unattended, jnp.int32(-1),
                                state.poss_team).astype(jnp.int32))

    def _update_possession_context(self, entry_state, state):
        """Advance the observable control-frame possession transition clock.

        A policy action is chosen once per control frame, so changes that happen
        and reverse wholly inside the physics scan are intentionally not exposed
        as a fictitious decision boundary.  ``previous_poss_team`` remembers the
        latest non-neutral controller across a loose interval; this lets both
        teams distinguish "we just lost it" from an unrelated neutral ball.
        """

        changed = state.poss_team != entry_state.poss_team
        previous = jnp.where(
            changed & (entry_state.poss_team >= 0),
            entry_state.poss_team,
            entry_state.previous_poss_team,
        ).astype(jnp.int32)
        age = jnp.where(
            changed,
            jnp.int32(0),
            jnp.minimum(
                entry_state.possession_t + jnp.int32(1),
                jnp.int32(self.game_duration),
            ),
        ).astype(jnp.int32)
        return state._replace(
            possession_t=age,
            previous_poss_team=previous,
        )

    def _track_forced_positions(self, before, after, forced):
        """비정책 단계가 위치를 바꾼 슬롯을 강제 이동 누적기에 더한다.

        위치를 강제로 쓰는 단계는 여럿이고(득점 후 킥오프 포메이션 재배치, 재개 이격 투영,
        키커 스냅, 차징 파울, 교체 투입), 각 단계가 자기 마스크를 따로 **선언**하면 빠뜨린
        단계가 생긴다 — 실제로 득점 프레임의 22명 전원 재배치(최대 55 m)와 교체 투입
        텔레포트가 마스킹되지 않은 채 BC 라벨로 나갔다. 위치 차분에 더해 identity·참여상태·
        득점 경계를 함께 관측한다. 강제 writer가 우연히 같은 좌표를 다시 써도(동일위치 교체,
        이미 킥오프 포메이션에 있던 선수의 득점 리셋, 이미 bench 좌표에 있던 선수의 퇴장)
        counterfactual 정책 이동은 여전히 덮였으므로 마스크되어야 한다.

        정책 이동(``_move_with_energy``)에는 쓰지 않는다 — 그건 액션이 설명하는 변위다.
        """

        moved = jnp.any(after.player_pos != before.player_pos, axis=1)
        identity_changed = after.slot_generation != before.slot_generation
        participation_changed = after.active_player != before.active_player
        # A goal resets every participating player to kickoff formation.  A
        # player already standing on their formation coordinate has no value
        # difference, but their movement command was overwritten just as fully
        # as everybody else's.  Score is team-global, so broadcast this event.
        goal_repositioned = jnp.any(after.score != before.score)
        # 새로 퇴장한 선수는 ``after.active_player=False``지만 같은 frame에 벤치로 수십 m
        # 순간이동한다. after만 보면 그 이동이 BC 자유 이동으로 열리는 정확한 마스크 구멍이
        # 생긴다(실측 33.6 m, move_forced=False). 진입 또는 종료 어느 쪽에서든 참여자였으면
        # 그 frame의 env 위치 쓰기를 라벨에서 제거한다. 계속 비활성인 벤치 슬롯은 제외한다.
        participates = before.active_player | after.active_player
        env_write = (
            moved | identity_changed | participation_changed | goal_repositioned
        )
        return forced | (env_write & participates)

    def _substitution_window_open(self, state):
        """[IFAB Law 3] 지금 교체가 법적으로 가능한가 — 공이 죽어 있어야 한다.

        스케줄 재현과 자동 결정자가 **같은 술어**를 써야 한다. 갈리면 한쪽 경로만 규칙을
        어기고, 그 상태가 리플레이에 남아 학습 데이터를 오염시킨다.

        킥오프 직전(t=0)은 아직 경기가 시작되지 않았으므로 열려 있다 — 관측된 하프타임
        교체를 후반 전용 에피소드로 재생할 때 첫 프레임에 적용돼야 한다.
        """

        return (state.ball_state == BALL_DEAD) | (state.t <= 0)

    @staticmethod
    def _goalkeeper_substitution_role_compatible(
        state, slot, entrant_is_gk, *, team=None
    ):
        """Return whether one replacement preserves the active-GK invariant.

        Ordinary replacements preserve the outgoing slot's goalkeeper role.
        The sole exception repairs a team with no active goalkeeper by replacing
        one active field player with a goalkeeper.  The surrounding approval
        layer remains responsible for the dead-ball, budget, and roster gates.
        """

        slot_team = state.team_id[slot] if team is None else jnp.int32(team)
        outgoing_is_gk = state.gk_indices[slot] == 1
        active_goalkeeper = jnp.any(
            (state.team_id == slot_team)
            & state.active_player
            & (state.gk_indices == 1)
        )
        entrant_is_gk = jnp.asarray(entrant_is_gk, dtype=jnp.bool_)
        return (
            (outgoing_is_gk == entrant_is_gk)
            | ((~active_goalkeeper) & (~outgoing_is_gk) & entrant_is_gk)
        )

    def _is_half_time(self, state):
        """지금이 하프타임 정지인가.

        [IFAB Law 3] 하프타임 교체는 3회 기회에 포함되지 않는다. 승인부
        (:meth:`_apply_decided_substitutions`)와 결정자 뷰가 **같은 식**을 봐야 한다 —
        따로 쓰면 승인은 되는데 제안이 안 되는(또는 그 반대의) 어긋남이 생긴다.
        """

        return jnp.bool_(self.halftime) & (
            state.t == jnp.int32(self.game_duration // 2))

    def _substitution_view(self, state):
        """결정자에게 줄 입력 — 전부 관측 가능한 값이다."""

        return substitution_module.SubstitutionView(
            t=state.t,
            game_duration=jnp.int32(self.game_duration),
            ball_dead=self._substitution_window_open(state),
            team_id=state.team_id,
            active_player=state.active_player,
            # ``gk_indices``는 0/1 int다. 결정자 규약은 bool이라고 적어 두었고 남의
            # 알고리즘은 ``~view.is_gk``를 쓴다 — int에 ``~``를 걸면 0이 -1(참)이 되어
            # 조용히 뒤집힌다. 경계에서 한 번 bool로 못박는다.
            is_gk=state.gk_indices == 1,
            stamina_long=state.stamina_long,
            stamina_short=state.stamina_short,
            yellow_cards=state.yellow_cards,
            score=state.score,
            bench_player_id=state.bench_player_id,
            bench_is_gk=state.bench_is_gk,
            subs_remaining=state.subs_remaining,
            sub_windows_used=state.sub_windows_used,
            sub_window_open_t=state.sub_window_open_t,
            half_time=self._is_half_time(state),
            max_simultaneous=self.max_simultaneous_substitutions,
        )

    def _build_formation_plan(self, formation_mode, base_layout):
        """레이아웃 표를 생성 시점에 한 번 편다.

        레이아웃이 작은 categorical이라 모든 모양의 앵커를 미리 계산할 수 있다. 덕분에
        런타임에는 인덱싱과 보간만 남고, **규칙 정책도 같은 표를 들고 같은 값을 재구성**한다 —
        관측에 앵커를 실어 보내고 정책이 그것을 읽는 구조가 성립하는 이유다.
        """

        self._formation_decider = formation_module.resolve(formation_mode)
        self.formation_mode = (
            formation_mode if isinstance(formation_mode, str) else "custom"
        )
        names = formation_module.LAYOUT_NAMES
        if base_layout not in names:
            raise ValueError(
                f"base_layout must be one of {list(names)}, got {base_layout!r}")
        self.base_layout = names.index(base_layout)

        teams = np.asarray(self._static_meta[0])
        att_dir = np.asarray(self._static_meta[1], np.float32)
        gk = np.asarray(self._static_meta[2]) > 0
        # ``_kickoff_positions``와 같은 규약으로 접는다 — 공격 프레임(+x=전방).
        flip = 1.0 if float(att_dir[0]) > 0 else -1.0
        home_att = (np.asarray(self.base_formation, np.float32) * flip
                    * att_dir[:, None])
        self._formation_ranks = formation_module.slot_ranks(home_att, gk, teams)
        self._formation_home_att = jnp.asarray(home_att, jnp.float32)
        self._formation_gk = jnp.asarray(gk)
        self._formation_teams = jnp.asarray(teams, jnp.int32)
        # 라인 기하는 레이아웃 × **인원수**로 편다. 퇴장으로 인원이 줄면 남은 선수들이
        # 그 인원용 형태로 다시 선다 — 종전에는 킥오프 인원 표에 묶여 대형에 구멍이 남았다.
        max_outfield = int(max(1, self.N))
        self._formation_plan = formation_module.layout_line_plan(
            self.hx, self.hy, max_outfield,
            formation_module.kickoff_line_shape(home_att, gk, teams))

    def _build_manager(self, manager):
        """감독을 조립한다.

        ``manager``를 주면 그것이 교체와 포메이션을 **함께** 정한다. 주지 않으면 기존
        ``substitution_mode``/``formation_mode`` 두 결정자를 하나로 묶은 어댑터를 만든다 —
        기존 설정이 그대로 돌아야 하기 때문이다.
        """

        if manager is not None:
            self._manager = manager_module.resolve(manager)
            self.manager_mode = manager if isinstance(manager, str) else "custom"
            return

        # 클로저가 아니라 클래스로 묶는다 — 로컬 함수를 env 속성으로 두면 pickle이 깨진다.
        self._manager = manager_module.ComposedManager(
            self._substitution_decider, self._formation_decider,
            self._legacy_substitution_view, self._legacy_formation_view)
        self.manager_mode = "composed"

    @staticmethod
    def _legacy_substitution_view(view):
        """감독 뷰에서 옛 :class:`SubstitutionView`를 만든다 — 기존 결정자 호환."""

        return substitution_module.SubstitutionView(
            t=view.t, game_duration=view.game_duration,
            ball_dead=view.ball_dead, team_id=view.team_id,
            active_player=view.active_player, is_gk=view.is_gk,
            stamina_long=view.stamina_long, stamina_short=view.stamina_short,
            yellow_cards=view.yellow_cards, score=view.score,
            bench_player_id=view.bench_player_id,
            bench_is_gk=view.bench_is_gk,
            subs_remaining=view.subs_remaining,
            sub_windows_used=view.sub_windows_used,
            sub_window_open_t=jnp.where(view.sub_window_open, view.t, -1),
            half_time=view.half_time,
            max_simultaneous=view.max_simultaneous)

    @staticmethod
    def _legacy_formation_view(view):
        """감독 뷰에서 옛 :class:`FormationView`를 만든다 — 기존 지휘관 호환."""

        return formation_module.FormationView(
            t=view.t, game_duration=view.game_duration,
            ball_dead=view.ball_dead, score=view.score, team_id=view.team_id,
            active_player=view.active_player, is_gk=view.is_gk,
            formation_home=view.formation_home,
            stamina_long=view.stamina_long,
            endurance_factor=view.endurance_factor,
            territory=view.ball_progress,
            layout_index=view.layout_index,
            layout_since_t=view.layout_since_t,
            control_fps=view.control_fps)

    @staticmethod
    def _canonical_manager_view_counters(view):
        """공개·수집용 감독 뷰의 스칼라 자원을 고정 dtype 배열로 만든다."""

        return view._replace(
            max_substitutions=jnp.int32(view.max_substitutions),
            max_simultaneous=jnp.int32(view.max_simultaneous),
        )

    def manager_view(self, state):
        """공개 감독 입력 — 교체와 포메이션이 **같은 값**을 본다.

        공개 PyTree와 capture는 eager/JIT/scan에서 같은 스키마를 가져야 하므로 정수
        스칼라도 명시적인 ``int32`` 배열로 낸다. 내장·사용자 감독을 환경 안에서 호출할
        때는 고정 폭 출력을 만들 수 있도록 :meth:`_manager_decision_view`의 호스트 정적
        정수를 사용한다.
        """

        return self._canonical_manager_view_counters(
            self._manager_decision_view(state)
        )

    def _manager_decision_view(self, state):
        """환경 내부 감독 호출용 입력 — 출력 폭 자원은 호스트 정적 정수다.

        두 뷰를 따로 만들면 스태미나·스코어·팀 구성을 두 번 계산하게 되고, 두 결정이 서로를
        모르므로 연동된 판단(수비수를 빼면서 동시에 전진 배치)이 표현되지 않는다.

        전부 관측에서 얻을 수 있는 값만 담는다 — 학습된 감독이 배포 시점에 없는 입력에
        의존하면 안 된다.
        """

        folded_x = state.ball_pos[DIM_X] * state.attack_dir
        home = self.formation_home(state)

        def per_team(values, mask_extra=None):
            out = []
            for team in (TEAM_0, TEAM_1):
                mine = (state.team_id == team) & state.active_player
                if mask_extra is not None:
                    mine = mine & mask_extra
                count = jnp.maximum(jnp.sum(mine), 1).astype(jnp.float32)
                out.append(jnp.sum(jnp.where(mine, values, 0.0)) / count)
            return jnp.stack(out)

        folded_pos = state.player_pos[:, DIM_X] * state.attack_dir
        centroid = per_team(folded_pos) / jnp.maximum(self.hx, DIV_EPS)
        spread = per_team(jnp.abs(state.player_pos[:, DIM_Y])) / jnp.maximum(
            self.hy, DIV_EPS)
        return manager_module.ManagerView(
            t=state.t,
            game_duration=jnp.int32(self.game_duration),
            control_fps=jnp.float32(self.control_fps),
            ball_dead=self._substitution_window_open(state),
            score=state.score,
            team_id=state.team_id,
            active_player=state.active_player,
            is_gk=state.gk_indices == 1,
            stamina_long=state.stamina_long,
            endurance_factor=state.endurance_factor,
            stamina_short=state.stamina_short,
            yellow_cards=state.yellow_cards,
            formation_home=home,
            vmax=state.vmax,
            player_ctrl=state.player_ctrl,
            bench_player_id=state.bench_player_id,
            bench_is_gk=state.bench_is_gk,
            bench_role_pos=state.bench_role_pos,
            bench_vmax=state.bench_vmax,
            bench_ctrl=state.bench_player_ctrl,
            bench_endurance_factor=state.bench_endurance_factor,
            subs_remaining=state.subs_remaining,
            max_substitutions=self.max_substitutions,
            sub_windows_used=state.sub_windows_used,
            sub_window_open=state.sub_window_open_t >= 0,
            half_time=self._is_half_time(state),
            max_simultaneous=self.max_simultaneous_substitutions,
            layout_index=state.layout_index,
            layout_since_t=state.layout_since_t,
            ball_progress=per_team(jnp.broadcast_to(folded_x, (self.N,)))
            / jnp.maximum(self.hx, DIV_EPS),
            team_centroid=centroid,
            team_width=spread,
        )

    def formation_layout_anchors(self, layout_index, active=None):
        """레이아웃별 앵커 스냅샷 ``(N, 2)`` — 호스트에서 표를 만들 때 쓴다.

        ``active``를 주지 않으면 전원 출전으로 본다(정책 빌드·렌더 그룹핑처럼 킥오프 기준
        기하가 필요한 곳). 런타임 앵커는 :meth:`formation_home`이고 그쪽은 실제 활성
        마스크를 쓴다.
        """

        if active is None:
            active = jnp.ones(self.N, bool)
        return formation_module.active_anchors(
            self._formation_plan,
            jnp.broadcast_to(jnp.asarray(layout_index, jnp.int32), (TEAM_COUNT,)),
            active, self._formation_gk, self._formation_teams,
            self._formation_ranks, self._formation_home_att)

    def formation_home(self, state):
        """지금의 규범 앵커 (N,2) — 공격 접힘 프레임.

        ``role_pos``와 헷갈리면 안 된다. ``role_pos``는 **서술적**이다(현재 전술 epoch에서
        이 선수가 실제로 어디 있었나의 누적평균). 이쪽은 **규범적**이다(어디에 서야 하나).
        서술값을 전술 홈으로 쓰면 정책이 자기 과거 평균을 쫓게 되므로 둘을 나눈다.
        """

        return formation_module.active_anchors(
            self._formation_plan, state.layout_index,
            state.active_player, self._formation_gk,
            state.team_id, self._formation_ranks, self._formation_home_att)

    def formation_view(self, state):
        """포메이션 지휘관이 보는 입력 — 학습 헤드도 같은 것을 본다.

        :meth:`substitution_view`와 같은 정보 경계다 — 전부 관측에서 얻을 수 있는 값뿐이다.
        """

        folded_ball_x = state.ball_pos[DIM_X] * state.attack_dir
        # 팀별 **자기 인원**으로 나눈다. 전체 평균에 2를 곱하면 양 팀 인원이 같다고 가정하는
        # 셈이라 비대칭 로스터에서 편향된다(실측 4v3, 공 x=10: 기대 ±0.190476 대신
        # [0.217687, -0.163265]).
        def side(team):
            mine = state.team_id == team
            count = jnp.maximum(jnp.sum(mine), 1).astype(jnp.float32)
            return jnp.sum(jnp.where(mine, folded_ball_x, 0.0)) / count

        territory = jnp.stack([side(TEAM_0), side(TEAM_1)]) / jnp.maximum(
            self.hx, DIV_EPS)
        return formation_module.FormationView(
            t=state.t,
            game_duration=jnp.int32(self.game_duration),
            ball_dead=state.ball_state == BALL_DEAD,
            score=state.score,
            team_id=state.team_id,
            active_player=state.active_player,
            is_gk=state.gk_indices == 1,
            formation_home=self.formation_home(state),
            stamina_long=state.stamina_long,
            endurance_factor=state.endurance_factor,
            territory=territory,
            layout_index=state.layout_index,
            layout_since_t=state.layout_since_t,
            control_fps=jnp.float32(self.control_fps),
        )

    def _apply_formation_command(self, state, key, proposal=None):
        """지휘관의 제안을 승인 검사 뒤 적용한다 — State만 필요한 호출자용."""

        return self._apply_formation_command_with_trace(
            state, key, proposal=proposal)[0]

    def _formation_trace_zero(self, state, code=FORMATION_DECISION_UNCHANGED):
        """포메이션 결정이 돌지 않은 프레임의 고정 shape 트레이스."""

        return {
            "formation_proposed_layout": state.layout_index,
            "formation_layout": state.layout_index,
            "formation_applied": jnp.zeros(TEAM_COUNT, bool),
            "formation_decision_code": jnp.full(TEAM_COUNT, code, jnp.int32),
        }

    def _apply_formation_command_with_trace(self, state, key, proposal=None):
        """지휘관의 제안을 **승인 검사 뒤** 적용하고 승인 결과를 함께 낸다 — 교체와 같은 규약이다.

        승인된 목표 레이아웃은 라이브볼·데드볼 모두 이 명령 경계에서 즉시 바뀐다. 이 함수는
        선수 좌표를 쓰지 않는다. 실제 형태 변화는 새 규범 앵커를 읽은 행동 정책과 물리가
        이후 프레임에 걸쳐 만든다.

        승인된 팀은 ``role_pos`` 누적기도 새 전술 epoch로 연다. prior는 새
        ``formation_home``이 아니라 **명령 경계의 실제 위치**다. 아직 도달하지 않은 목표를
        과거 실제 위치처럼 기록하면 서술값과 규범값의 경계가 무너지기 때문이다.

        이 재기준화는 identity 변경이 아니다. 교체의 단일 진실원천은
        ``(player_id, slot_generation)``이고 포메이션 경계의 단일 진실원천은 아래 ``legal`` /
        공개 ``FormationResult.applied``다. 둘은 같은 프레임에 동시에 참일 수 있으므로
        ``role_pos_count`` 감소만으로 교체를 추론해서는 안 된다.
        """

        want = self._validate_formation_proposal(proposal)
        want = jnp.asarray(want, jnp.int32)
        layouts = len(formation_module.LAYOUTS)
        legal = (
            (want >= 0) & (want < layouts)
            & (want != state.layout_index)
            & (~self._is_terminal(state))
        )
        safe = jnp.clip(want, 0, layouts - 1)
        # 전술 epoch는 승인된 팀의 현재 참여자에게만 열린다. 공격 접힘 프레임에 저장하는
        # 기존 role_pos 규약을 지키되, 새 목표 앵커를 복사하지 않고 현재 실제 위치를 prior로
        # 둔다. 이 함수 뒤의 _accumulate_role_anchor가 라이브볼이면 곧바로 첫 표본(count=1)을
        # 넣고, 데드볼이면 count=0 prior를 재개 시점까지 보존한다.
        formation_epoch = legal[state.team_id] & state.active_player
        folded_actual = state.player_pos * state.attack_dir[:, None]
        after = state._replace(
            layout_index=jnp.where(legal, safe, state.layout_index),
            layout_since_t=jnp.where(legal, state.t, state.layout_since_t),
            role_pos=jnp.where(
                formation_epoch[:, None], folded_actual, state.role_pos
            ).astype(state.role_pos.dtype),
            role_pos_count=jnp.where(
                formation_epoch,
                jnp.zeros((), dtype=state.role_pos_count.dtype),
                state.role_pos_count,
            ),
        )
        # 사유는 위 ``legal`` 논리곱의 항 순서를 그대로 따른다. '지금과 같은 레이아웃'은
        # 거절이 아니라 **바꿀 것이 없음**이라 따로 구분한다 — 그것까지 거절로 세면
        # 아무것도 요청하지 않은 프레임이 전부 실패로 보고된다.
        reason = jnp.where(
            self._is_terminal(state), jnp.int32(FORMATION_DECISION_TERMINAL),
            jnp.where(
                (want < 0) | (want >= layouts),
                jnp.int32(FORMATION_DECISION_OUT_OF_RANGE),
            jnp.where(
                want == state.layout_index,
                jnp.int32(FORMATION_DECISION_UNCHANGED),
                jnp.int32(FORMATION_DECISION_APPLIED))))
        return after, {
            "formation_proposed_layout": want,
            "formation_layout": after.layout_index,
            "formation_applied": legal,
            "formation_decision_code": reason,
        }

    def _validate_formation_proposal(self, proposal):
        """주입된 포메이션 제안을 형태만 확인한다.

        값의 합법성(범위 안인가, 종료 전인가)은 승인 층의 일이다 — 여기서 중복하면 두 곳이
        갈릴 수 있다. 막는 것은 아예 해석할 수 없는 입력뿐이다.
        """

        want = _require_integer_proposal(proposal, "formation proposal")
        if want.shape != (TEAM_COUNT,):
            raise ValueError(
                f"formation proposal must have shape ({TEAM_COUNT},), "
                f"got {want.shape}")
        return want

    def substitution_view(self, state):
        """교체 결정자가 보는 입력 — 학습 헤드도 같은 것을 본다.

        env 안에 사는 결정자(``substitution_mode="auto"``, 사용자 함수)는 이 값을 그대로
        받는다. 밖에서 학습되는 교체 정책은 파라미터가 매 갱신마다 바뀌어 env 생성 시
        클로저로 묶을 수 없으므로 ``step_env_array(..., substitution=...)``로 결정을
        주입하는데, **무엇을 보고 결정했는가**는 같아야 한다. 그래서 같은 뷰를 공개한다.

        여기 담기는 값은 전부 관측 가능한 것뿐이다(:mod:`substitution` 참조). 특권 정보를
        넣으면 학습된 교체 정책이 배포 시점에 없는 입력에 의존하게 된다.
        """

        return self._substitution_view(state)

    def _apply_decided_substitutions(self, state, key, proposal=None,
                                     boundary_generation=None):
        """결정자의 제안을 승인 검사 뒤 적용한다 — State만 필요한 호출자용."""

        return self._apply_decided_substitutions_with_trace(
            state, key, proposal=proposal,
            boundary_generation=boundary_generation)[0]

    def _substitution_trace_zero(self, code=SUB_DECISION_NOT_REQUESTED):
        """아무 교체 결정도 없었을 때의 고정 shape 트레이스.

        ``lax.cond``의 두 가지가 같은 pytree를 내야 하고, 종료 프레임처럼 결정 단계 자체가
        돌지 않는 경로도 이 값을 낸다 — shape가 갈리면 소비자가 프레임마다 다른 배열을 받는다.
        """

        wide = (TEAM_COUNT, self.max_simultaneous_substitutions)
        return {
            "substitution_proposed_out_slot": jnp.full(wide, NO_PLAYER, jnp.int32),
            "substitution_proposed_bench_index": jnp.full(wide, NO_PLAYER, jnp.int32),
            "substitution_applied": jnp.zeros(wide, bool),
            "substitution_decision_code": jnp.full(wide, code, jnp.int32),
        }

    def _apply_decided_substitutions_with_trace(self, state, key, proposal=None,
                                                boundary_generation=None):
        """결정자의 제안을 **승인 검사 뒤** 적용하고 승인 결과를 함께 낸다.

        결정자는 제안하고 환경이 승인한다. 타인이 만든 알고리즘이 규칙을 어겨도(인플레이
        중 교체, 잔여 0에서 교체, GK 자리에 필드 선수, 이미 나간 벤치 자리) 여기서 거부되므로
        환경 상태가 깨지지 않는다. 그래서 결정자 쪽 검사를 신뢰하지 않고 전부 다시 본다.

        한 데드볼에 두 팀이 각각 한 명씩 바꿀 수 있다. 같은 팀이 한 번에 여러 명을 바꾸는
        것은 이 구조에서 연속 프레임으로 표현된다 — 같은 데드볼이면 window가 늘지 않는다.

        반환하는 트레이스는 **제안과 승인 결과를 같은 호출에서** 담는다. 거절은 규칙이라
        없앨 수 없지만(공이 살아났으면 못 바꾼다) 조용한 거절은 없앨 수 있다 — 학습된 교체
        정책은 자기 제안이 왜 사라졌는지 알아야 고칠 수 있고, 모방학습 QC도 규칙 정책의
        제안값과 환경이 승인한 값을 구분해 저장해야 한다.
        """

        # 제안 검증은 **config보다 앞**이다. 벤치가 없다고 검증을 건너뛰면 같은 잘못된
        # 제안이 config에 따라 통과했다 거부됐다 한다 — 실측으로 ``bench_size=0``에서는
        # ``out_slot=10**18``·실수 제안·틀린 shape가 셋 다 조용히 통과했고, 벤치를
        # 붙이는 순간 셋 다 거부됐다. 벤치 없는 env로 개발한 학습 교체 헤드는 그때까지
        # 아무 신호도 받지 못한다. 승인 규칙은 config에 따라 달라져도 되지만 **입력이
        # 해석 가능한가**는 달라지면 안 된다.
        out_slot, bench_idx = self._validate_substitution_proposal(proposal)
        out_slot = self._widen_substitution(jnp.asarray(out_slot, jnp.int32))
        bench_idx = self._widen_substitution(jnp.asarray(bench_idx, jnp.int32))
        if self.bench_size == 0 or self.max_substitutions == 0:
            # 벤치 없음과 명시적인 0명 교체 예산은 서로 다른 계약이다. 둘을 같은 코드로
            # 뭉치면 학습된 감독이 명단을 보완해야 하는지, 예산상 교체가 금지된 경기인지
            # 구분할 수 없다. 둘 다 no-op이어도 제안과 정확한 거절 사유는 그대로 돌려준다.
            unavailable_code = (
                SUB_DECISION_NO_CARD
                if self.max_substitutions == 0
                else SUB_DECISION_BENCH_EMPTY
            )
            return state, {
                "substitution_proposed_out_slot": out_slot,
                "substitution_proposed_bench_index": bench_idx,
                "substitution_applied": jnp.zeros_like(out_slot, bool),
                "substitution_decision_code": jnp.where(
                    out_slot < 0, jnp.int32(SUB_DECISION_NOT_REQUESTED),
                    jnp.int32(unavailable_code)),
            }
        # [IFAB Law 3] 교체 기회는 **정지 구간** 단위다. 팀이 정지를 만들어 내는 것이 아니라,
        # 알아서 생기는 데드볼 중 교체 절차를 끼워 넣을 수 있는 개수가 3개다(교체 절차 자체가
        # 정지를 20~30 s 늘리기 때문에 둔 제한이다).
        #
        # 그래서 창은 공이 다시 살아날 때 닫는다. 종전에는 창을 '같은 control tick'으로
        # 봤는데, 이 구현은 한 tick에 팀당 한 명만 바꾸므로 3명이면 필연적으로 3 tick이
        # 걸리고 매 tick이 새 기회로 잡혔다 — 실측으로 한 번의 정지에서 3명을 바꾼 팀이
        # 기회 3개를 다 써 남은 인원을 인플레이 중 영영 못 넣었다(잔여4/윈도1 → 2/윈도3).
        state = state._replace(
            sub_window_open_t=jnp.where(
                state.ball_state == BALL_ALIVE,
                jnp.int32(-1),
                state.sub_window_open_t,
            )
        )
        # 제안은 감독(또는 호출자)이 이미 만들어 넘긴다. 출처와 무관하게 **같은 검증기**를
        # 지난다 — 출처에 따라 승인 강도가 달라지면 "감독은 제안하고 환경이 승인한다"는
        # 원칙 자체가 무너진다.
        # 학습 경로도 같은 자리로 들어온다. 파라미터가 매 갱신마다 바뀌는 학습 정책을 env
        # 생성 시 클로저로 묶으면 재추적이 일어나므로 결정을 밖에서 만들어 주입한다.
        # 승인 층은 그대로다 — 제안의 **출처**만 바뀐다.
        # (해석 검증과 폭 정규화는 위에서 이미 끝났다.)

        window_open = self._substitution_window_open(state)
        terminal = self._is_terminal(state)
        # 같은 정지에서 같은 슬롯을 두 번 바꾸면 방금 들어온 선수가 곧바로 나가고 교체
        # 카드만 한 장 더 사라진다. 결정자의 실수가 규칙 자원을 축내지 않도록 막는다.
        #
        # 기준은 이 **경계가 시작될 때**의 generation이다. 자기 진입 시점으로 잡으면 같은
        # tick에 먼저 돈 스케줄 경로의 교체를 못 보고 그 슬롯을 다시 바꾼다(실측: 한 경계에서
        # 2 -> 9002 -> 992003, generation +2, 잔여 5 -> 3).
        generation_before = (state.slot_generation if boundary_generation is None
                             else boundary_generation)

        def apply_team(team, rank, carry):
            current, code = carry
            slot = out_slot[team, rank]
            idx = bench_idx[team, rank]
            safe_slot = jnp.clip(slot, 0, self.N - 1)
            safe_idx = jnp.clip(idx, 0, self.bench_size - 1)
            incoming = current.bench_player_id[team, safe_idx]
            entrant_is_gk = current.bench_is_gk[team, safe_idx]
            # 평시에는 역할을 보존한다. 다만 퇴장 등으로 활동 가능한 GK가 전혀 없으면
            # active field player 한 명을 빼고 벤치 GK를 넣는 복구만 허용한다. 이미 GK가
            # 있으면 field→GK는 계속 거절되고, 기존 GK→GK 교체는 equality 경로를 탄다.
            role_compatible = self._goalkeeper_substitution_role_compatible(
                current, safe_slot, entrant_is_gk, team=team
            )
            # 같은 정지 안이면 이미 연 기회를 다시 세지 않는다. 창은 공이 살아날 때
            # 닫히므로(위) '열려 있다'는 것만 보면 된다 — tick 수와 무관하다.
            same_window = current.sub_window_open_t[team] >= 0
            # [IFAB Law 3] 하프타임 교체는 3회에 포함되지 않는다. 하프타임은 어차피 멈춰
            # 있는 시간이라 교체 절차가 경기 흐름을 **추가로** 끊지 않기 때문이다.
            # 기회를 세지 않을 뿐 아니라 창을 열지도 않는다 — 열어 두면 뒤이은 킥오프
            # 데드볼의 교체까지 같은 창으로 묶여 공짜가 된다.
            free_window = self._is_half_time(current)
            legal = (
                window_open
                & (~terminal)
                & (slot >= 0) & (slot < self.N)
                & (idx >= 0) & (idx < self.bench_size)
                & (incoming >= 0)
                & (current.subs_remaining[team] > 0)
                & (current.team_id[safe_slot] == team)
                & current.active_player[safe_slot]
                & (current.slot_generation[safe_slot]
                   == generation_before[safe_slot])
                & role_compatible
                & (
                    same_window
                    | free_window
                    | (current.sub_windows_used[team]
                       < IFAB_MAX_SUBSTITUTION_WINDOWS)
                )
            )
            entry_xy = self._substitution_entry_position(current, safe_slot)
            projected = self._project_substitution(
                current,
                safe_slot,
                incoming,
                player_pos=entry_xy,
                role_pos=current.bench_role_pos[team, safe_idx],
                vmax=current.bench_vmax[team, safe_idx],
                reach_z=current.bench_reach_z[team, safe_idx],
                head_z=current.bench_head_z[team, safe_idx],
                player_ctrl=current.bench_player_ctrl[team, safe_idx],
                is_gk=entrant_is_gk,
                endurance_factor=current.bench_endurance_factor[team, safe_idx],
            )
            outgoing = current.player_id[safe_slot]
            # ``legal``은 **사전** 검사다. 통과해도 투영이 자체 사유로 no-op이 될 수 있다
            # (예: 들어올 선수가 이미 경기장에 있는 중복 identity). 그때 자원만 확정하면
            # 카드와 벤치 자리가 사라지고 **나가지도 않은 선수가 은퇴 명단에 오른다**
            # (실측: 슬롯 그대로인데 벤치 990001→-1, 잔여 −1, 기회 +1, 은퇴기록에 2번).
            # 그래서 확정 조건은 "투영이 실제로 identity를 바꿨는가"다.
            applied = legal & (
                projected.slot_generation[safe_slot]
                != current.slot_generation[safe_slot]
            )
            committed = projected._replace(
                # 벤치 자리를 비우고, 나간 사람을 기록한다. 신원을 남기지 않으면 렌더가
                # 퇴장 선수와 교체 아웃 선수를 구분해 그릴 수 없다.
                bench_player_id=projected.bench_player_id.at[team, safe_idx].set(
                    jnp.int32(NO_PLAYER)
                ),
                # 나간 사람은 **은퇴 순서**로 쌓는다. 벤치 인덱스에 넣으면 벤치를 쓰지 않는
                # 스케줄 경로가 기록할 자리를 못 찾는다 — 실제로 예약 교체 뒤 명단이 전부
                # 비어 있었고, 렌더의 OUT 구역에서 선수가 사라졌다.
                retired_player_id=projected.retired_player_id.at[
                    team, _first_free(projected.retired_player_id[team])
                ].set(outgoing),
                subs_remaining=projected.subs_remaining.at[team].add(-1),
                sub_windows_used=projected.sub_windows_used.at[team].add(
                    jnp.where(same_window | free_window, 0, 1)
                ),
                sub_window_open_t=projected.sub_window_open_t.at[team].set(
                    jnp.where(free_window, current.sub_window_open_t[team],
                              current.t)
                ),
            )
            # 승인 결과를 **거절 사유까지** 남긴다. 순서는 위 ``legal`` 논리곱의 항과
            # 같고, 먼저 걸리는 사유를 보고한다 — 여러 조건이 동시에 틀린 제안에도
            # 결정론적으로 하나의 코드가 붙는다.
            reason = jnp.where(
                slot < 0, jnp.int32(SUB_DECISION_NOT_REQUESTED),
                jnp.where(
                    slot >= self.N, jnp.int32(SUB_DECISION_SLOT_RANGE),
                jnp.where(
                    (idx < 0) | (idx >= self.bench_size),
                    jnp.int32(SUB_DECISION_BENCH_RANGE),
                jnp.where(
                    incoming < 0, jnp.int32(SUB_DECISION_BENCH_EMPTY),
                jnp.where(
                    current.team_id[safe_slot] != team,
                    jnp.int32(SUB_DECISION_WRONG_TEAM),
                jnp.where(
                    ~current.active_player[safe_slot],
                    jnp.int32(SUB_DECISION_SLOT_INACTIVE),
                jnp.where(
                    current.slot_generation[safe_slot]
                    != generation_before[safe_slot],
                    jnp.int32(SUB_DECISION_SLOT_ALREADY_CHANGED),
                jnp.where(
                    ~role_compatible,
                    jnp.int32(SUB_DECISION_GK_ROLE),
                jnp.where(
                    current.subs_remaining[team] <= 0,
                    jnp.int32(SUB_DECISION_NO_CARD),
                jnp.where(
                    ~(same_window | free_window
                      | (current.sub_windows_used[team]
                         < IFAB_MAX_SUBSTITUTION_WINDOWS)),
                    jnp.int32(SUB_DECISION_WINDOW_BUDGET),
                jnp.where(
                    applied, jnp.int32(SUB_DECISION_APPLIED),
                    jnp.int32(SUB_DECISION_PLACEMENT),
                )))))))))))
            code = code.at[team, rank].set(reason)
            return jax.tree_util.tree_map(
                lambda taken, kept: jnp.where(applied, taken, kept),
                committed,
                current,
            ), code

        # [IFAB Law 3] 한 정지에서 여러 명을 함께 바꿀 수 있다. 그 의도는 **원자적**이라
        # 연속 tick으로 흩어 놓으면 도중에 공이 살아났을 때 남은 교체가 사라진다.
        #
        # 반복은 ``lax.fori_loop``으로 돈다 — 파이썬으로 펼치면 K에 비례해 HLO가 커진다
        # (스케줄 경로가 행 수에 대해 같은 이유로 fori_loop을 쓴다). 그래서 K를 IFAB 상한인
        # 교체 인원 전체로 둬도 그래프 크기가 그대로다.
        def one_rank(rank, carry):
            return apply_team(TEAM_1, rank, apply_team(TEAM_0, rank, carry))

        wide = (TEAM_COUNT, self.max_simultaneous_substitutions)
        # 창이 닫혀 있으면 통째로 건너뛴다. 실측으로 인플레이가 99.96 %라, 이 분기가
        # 없으면 아무 일도 일어나지 않는 스텝마다 K회의 투영 기하를 헛돌린다.
        # 그 경우의 사유는 창이 닫힌 이유 그대로다 — 요청이 없던 자리는 그대로 둔다.
        closed_reason = jnp.where(
            out_slot < 0, jnp.int32(SUB_DECISION_NOT_REQUESTED),
            jnp.where(terminal, jnp.int32(SUB_DECISION_TERMINAL),
                      jnp.int32(SUB_DECISION_BALL_LIVE)))
        state, code = lax.cond(
            window_open & (~terminal),
            lambda carry: lax.fori_loop(
                0, self.max_simultaneous_substitutions, one_rank, carry),
            lambda carry: (carry[0], closed_reason),
            (state, jnp.full(wide, SUB_DECISION_NOT_REQUESTED, jnp.int32)),
        )
        return state, {
            "substitution_proposed_out_slot": out_slot,
            "substitution_proposed_bench_index": bench_idx,
            "substitution_applied": code == SUB_DECISION_APPLIED,
            "substitution_decision_code": code,
        }

    def _validate_substitution_proposal(self, proposal):
        """주입된 교체 제안을 형태만 확인해 두 배열로 편다.

        값의 합법성(데드볼·잔여·GK)은 확인하지 않는다 — 그건 승인 층의 일이고, 여기서
        중복하면 두 곳이 갈릴 수 있다. 여기서 막는 것은 shape/dtype처럼 **아예 해석할 수
        없는** 입력뿐이다.
        """

        if isinstance(proposal, Mapping):
            missing = {"out_slot", "bench_index"} - set(proposal)
            if missing:
                raise ValueError(
                    f"substitution proposal is missing {sorted(missing)}")
            out_slot, bench_idx = proposal["out_slot"], proposal["bench_index"]
        else:
            try:
                out_slot, bench_idx = proposal
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "substitution must be (out_slot, bench_index) or a mapping "
                    f"with those keys, got {proposal!r}"
                ) from exc
        # dtype은 **변환 전에** 본다. 먼저 int32로 캐스팅하면 실수 제안이 조용히 잘린다
        # (실측: [1.9, -1.2] -> [1, -1]). 슬롯 인덱스에 반올림 규약을 정할 이유가 없으므로
        # 애초에 정수가 아닌 입력은 거부한다.
        out_slot = _require_integer_proposal(out_slot, "substitution out_slot")
        bench_idx = _require_integer_proposal(bench_idx, "substitution bench_index")
        wide = (TEAM_COUNT, self.max_simultaneous_substitutions)
        for name, value in (("out_slot", out_slot), ("bench_index", bench_idx)):
            # 한 명만 바꾸는 흔한 경우를 위해 ``(2,)``도 받는다 — 나머지 자리는
            # 비운다(NO_PLAYER). 계약을 넓히면서 기존 호출을 깨지 않기 위한 것이다.
            if value.shape not in (wide, (TEAM_COUNT,)):
                raise ValueError(
                    f"substitution {name} must have shape {wide} or "
                    f"({TEAM_COUNT},), got {value.shape}")
        return (self._widen_substitution(out_slot),
                self._widen_substitution(bench_idx))

    def _widen_substitution(self, value):
        """``(2,)`` 제안을 ``(2, K)``로 편다 — 첫 자리만 쓰고 나머지는 비운다."""

        if value.ndim == 1:
            pad = jnp.full(
                (TEAM_COUNT, self.max_simultaneous_substitutions - 1),
                NO_PLAYER, jnp.int32)
            return jnp.concatenate([value[:, None], pad], axis=1)
        return value

    def _substitution_entry_position(self, state, slot):
        """투입 선수의 첫 좌표 — 나간 선수 자리에서 출발한다.

        스케줄 재현은 관측된 좌표를 쓰지만 자동 교체에는 그런 관측이 없다. 나간 선수의
        위치가 전술적으로 가장 자연스러운 출발점이고, 겹침은 ``_project_substitution``의
        배치 해결기가 처리한다.
        """

        return state.player_pos[slot]

    def _apply_scheduled_substitutions(self, state):
        """이 tick에 예정된 교체를 원자적으로 적용한다.

        스케줄 행은 ``lax.fori_loop``에서 순서대로 처리해 길이가 JIT 그래프를
        정적으로 폭발시키지 않게 한다. 교체가 없는 환경은 이 경로 자체가 없어
        기존 롤아웃과 완전히 동일하다.
        """

        return self._apply_scheduled_substitutions_with_mask(state)[0]

    def _apply_scheduled_substitutions_with_mask(self, state):
        """Apply due substitutions and expose their final restart projection.

        The public/internal callers that only need a State keep using
        ``_apply_scheduled_substitutions``.  ``step_env_array`` additionally
        needs the projector's exact moved-slot mask: a substitution can place
        an entrant inside an active restart exclusion zone at the control-frame
        boundary, and that projection does not run again on the next frame.
        Returning the mask from the already-required final projection avoids a
        second geometry solve and does not confuse ordinary entry collision
        placement with referee restart intervention.
        """

        plan = self._substitution_plan
        if plan is None:
            return state, jnp.zeros(self.N, dtype=bool)
        # All rows sharing a tick describe one exogenous roster boundary.  Its
        # membership must be decided from the entry State, not from the
        # partially updated State produced by the static row loop.  Otherwise
        # an early entrant may commit even when a later entrant cannot be
        # placed, leaving a half-applied identity transition (and, in crowded
        # geometry, two active players at the same coordinate).
        state_before = state
        generation_before = state_before.slot_generation
        row_indices = jnp.arange(len(self.substitutions), dtype=jnp.int32)
        # 행은 경기당 **한 번만** 발화한다. '경기장에 없는가'만 보면 그 사람이 나중에
        # 다시 빠지는 순간 같은 행이 되살아나 되돌아온다(실측: 4 -> 9201 -> 9202 뒤 다음
        # 데드볼에 9202 -> 9201). 신원은 재사용되지 않는다고 스케줄 검증이 이미 보장하므로,
        # '경기장에도 없고 은퇴 명단에도 없다' = '아직 등장한 적 없다'가 정확한 술어다.
        appeared = jnp.any(
            state_before.player_id[:, None] == plan["player_id"][None, :],
            axis=0,
        ) | jnp.any(
            state_before.retired_player_id.reshape(-1)[:, None]
            == plan["player_id"][None, :],
            axis=0,
        )
        incoming_available_before = ~appeared
        # [IFAB Law 3] 교체는 **경기 중단 시에만** 가능하다. 스케줄된 tick에 env가 인플레이면
        # 그 교체는 사라지는 것이 아니라 **다음 데드볼까지 미뤄진다** — 그래서 조건이
        # ``== t``가 아니라 ``<= t``다. 실측 재현의 목적은 '누가 언제쯤 들어왔는가'이지
        # 프레임 단위 정확도가 아니고, 규칙을 어긴 상태는 BC 학습 데이터로도 오염이다.
        # 지연은 보통 수 초이고, 실제 지연량은 텔레메트리의 substitution 이벤트에 남는다.
        dead_ball = self._substitution_window_open(state_before)
        target_rows = (
            (plan["tick"] <= state_before.t)
            & dead_ball
            & state_before.on_pitch[plan["slot"]]
            & (~state_before.sent_off[plan["slot"]])
            & incoming_available_before
            & (~self._is_terminal(state_before))
        )
        # 한 슬롯은 한 경계에서 **한 번만** 바뀐다. 지연 규약(``tick <= t``) 때문에 서로
        # 다른 tick의 같은 슬롯 교체가 한 데드볼에 겹칠 수 있는데, 그대로 두면 정적 행
        # 루프가 둘 다 적용해 identity가 A -> B -> C로 두 번 넘어간다. 회계와 은퇴 기록은
        # **슬롯 단위**(``entered = generation != before``)라 그때 카드는 한 장만 빠지고
        # 중간 사람 B는 경기장에도 은퇴 명단에도 남지 않는다(실측: 2 -> 9201 -> 9202,
        # generation +2, 잔여 -1, 명단에 9201 없음).
        #
        # 회계를 전이 단위로 바꾸는 대신 **직렬화**한다. B가 실제로 뛴 것이 관측된 사실이고,
        # 다음 데드볼로 미루는 것은 이 경로가 이미 쓰는 규약이라 새 의미를 만들지 않는다.
        # 행은 ``(tick, slot)``으로 정렬돼 있으므로 앞선 인덱스가 곧 이른 tick이다.
        earlier_same_slot_due = jnp.any(
            (plan["slot"][:, None] == plan["slot"][None, :])
            & (row_indices[None, :] < row_indices[:, None])
            & target_rows[None, :],
            axis=1,
        )
        target_rows = target_rows & (~earlier_same_slot_due)

        def apply_row(index, carry):
            current, batch_succeeded = carry
            slot = plan["slot"][index]
            # A dismissal permanently removes that person/slot from this match.
            # Terminal, absent/off-pitch and already-present incoming identities
            # are non-target rows at this boundary rather than transaction
            # failures; they cannot prevent another independently eligible row
            # at the same tick from committing.
            due = target_rows[index]

            # Rows at the same tick are one simultaneous identity boundary,
            # even though this static loop applies them in slot order.  A
            # not-yet-processed outgoing player vacates their position at that
            # boundary and must not push an earlier entrant away from a spot
            # that will be free in the final State.  Already processed entrants
            # remain blockers, providing a deterministic reservation order when
            # two entrants request the same point.
            future_due_rows = (
                (row_indices > index)
                & target_rows
            )
            future_due_outgoing = jnp.any(
                (self.player_indices[:, None] == plan["slot"][None, :])
                & future_due_rows[None, :],
                axis=1,
            )

            def apply(inner_state):
                return self._project_substitution(
                    inner_state,
                    plan["slot"][index],
                    plan["player_id"][index],
                    player_pos=plan["entry_pos"][index],
                    role_pos=plan["role_pos"][index],
                    vmax=plan["vmax"][index],
                    reach_z=plan["reach_z"][index],
                    head_z=plan["head_z"][index],
                    player_ctrl=plan["player_ctrl"][index],
                    is_gk=plan["is_gk"][index],
                    endurance_factor=plan["endurance_factor"][index],
                    stamina_long_entry=plan["stamina_long_entry"][index],
                    stamina_short_entry=plan["stamina_short_entry"][index],
                    yellow_cards=plan["yellow_cards"][index],
                    _placement_exempt=future_due_outgoing,
                    _scheduled_identity=True,
                    _defer_restart_projection=True,
                )

            generation_before_row = current.slot_generation[slot]
            current = lax.cond(
                due, apply, lambda inner_state: inner_state, current
            )
            row_succeeded = (
                current.slot_generation[slot] != generation_before_row
            )
            batch_succeeded = batch_succeeded & ((~due) | row_succeeded)
            return current, batch_succeeded

        # A single compiled loop keeps HLO graph size independent of schedule
        # length while retaining the old row-order reservation semantics.
        state, batch_succeeded = lax.fori_loop(
            0,
            len(self.substitutions),
            apply_row,
            (state, jnp.bool_(True)),
        )

        # A same-tick substitution batch is all-or-nothing.  In particular,
        # future-vacating exemptions are optimistic reservations: if any target
        # row cannot use a legal entry point, every identity/state mutation made
        # by its siblings must disappear as well.  The scalar tree select keeps
        # the rollback JIT/vmap compatible and restores *all* person-owned
        # latches, not merely ids and positions.
        state = jax.tree_util.tree_map(
            lambda updated, original: jnp.where(
                batch_succeeded, updated, original
            ),
            state,
            state_before,
        )
        # Rows sharing a tick are one atomic identity boundary.  Projecting a
        # restart after each sequential row would let an early row move a
        # player that a later row is about to replace, and would run the
        # relatively expensive geometry solver repeatedly.  Defer every row,
        # then enforce the final roster exactly once — and only on a frame
        # where an identity transition actually succeeded.  Invalid/no-due
        # rows therefore retain the old zero-cost path.
        # 스케줄 재현도 교체 카드를 쓴다. 세지 않으면 스케줄과 자동/학습 결정자를 함께
        # 쓸 때 한 팀이 상한을 넘겨 바꿀 수 있고, 결정자에게 주는 뷰가 남은 카드를 거짓으로
        # 말한다. 관측된 사실이므로 **막지는 않고 기록만** 한다 — 0에서 눌러 멈춘다.
        entered = state.slot_generation != generation_before
        # 스케줄 경로도 나간 사람을 남긴다. 종전에는 아예 기록하지 않아 렌더의 OUT 구역과
        # State의 은퇴 명단에서 그 선수가 사라졌다(교체 이벤트만 generation 차분으로 남았다).
        #
        # 슬롯마다 ``lax.cond``를 펼치면 N에 비례해 HLO가 커진다. 팀별로 한 번에 편다.
        retired = state.retired_player_id
        depth = retired.shape[1]
        position = jnp.arange(depth, dtype=jnp.int32)
        for team in (TEAM_0, TEAM_1):
            mine = entered & (state.team_id == team)
            used = jnp.sum(retired[team] >= 0).astype(jnp.int32)
            # 나간 사람을 앞으로 모은다(슬롯 순서 유지 — 안정 정렬).
            order = jnp.argsort(jnp.where(mine, 0, 1), stable=True)
            ordered_ids = state_before.player_id[order]
            offset = position - used
            fresh = (position >= used) & (offset < jnp.sum(mine))
            row = jnp.where(
                fresh,
                ordered_ids[jnp.clip(offset, 0, ordered_ids.shape[0] - 1)],
                retired[team],
            )
            retired = retired.at[team].set(row)
        state = state._replace(retired_player_id=retired)

        per_team = jnp.stack([
            jnp.sum(entered & (state.team_id == TEAM_0)),
            jnp.sum(entered & (state.team_id == TEAM_1)),
        ]).astype(jnp.int32)
        opened = per_team > 0
        # 두 경로가 **같은 술어**를 써야 한다. 결정자는 정지 단위(`>= 0`)로 세는데 여기만
        # tick 단위(`== t`)로 세면, 같은 정지의 다음 tick 스케줄 교체가 기회를 또 소모하고
        # 하프타임 스케줄 교체도 기회를 쓴다 — 회계가 경로마다 갈린다.
        same_window = state.sub_window_open_t >= 0
        free_window = self._is_half_time(state)
        exempt = same_window | free_window
        state = state._replace(
            subs_remaining=jnp.maximum(state.subs_remaining - per_team, 0),
            sub_windows_used=state.sub_windows_used + jnp.where(
                opened & (~exempt), 1, 0).astype(jnp.int32),
            sub_window_open_t=jnp.where(
                opened & (~free_window), state.t, state.sub_window_open_t),
        )

        generation_changed = batch_succeeded & jnp.any(
            state.slot_generation != generation_before
        )
        return lax.cond(
            generation_changed,
            self._project_restart_positions,
            lambda current: (
                current, jnp.zeros(self.N, dtype=bool)
            ),
            state,
        )

    def _accumulate_role_anchor(self, state):
        """``role_pos``를 현재 전술 epoch의 **인과적 라이브볼 누적평균**으로 갱신한다.

        epoch는 경기/하프 시작, identity 교체, 승인된 포메이션 변경에서 열린다. 그 경계부터
        지금까지 관측된 라이브볼 위치의 평균이며 미래 프레임을 읽지 않는다. 명시적인 포메이션
        경계가 없는 실경기 코퍼스는 period/identity 경계를 쓰고, 포메이션 변경을 식별한 코퍼스는
        같은 지점에서 epoch를 잘라야 한다. 교체 투입 선수도 선발과 동일하게 자기 anchor를 처음부터
        쌓는다.

        데드볼을 제외하는 이유는 세트피스 정렬·킥오프 배치가 전술 역할이 아니라 상황 배치이기
        때문이다. 표본이 아직 없으면 진입 prior(킥오프 포메이션 또는 교체 지정값)를 그대로 둔다.

        누적은 **공격 접힘 프레임**에 한다. 하프타임에는 진영이 바뀌므로 누적기를 리셋한다.
        포메이션 변경은 명령 경계의 실제 위치를 prior로 삼고, 접힘 프레임에서는 prior가 그대로
        유지된다.

        갱신은 누적합이 아니라 **증분(Welford) 형식**으로 한다::

            n' = n + sample
            role' = role + (x - role) / n'          (sample일 때만)

        누적합을 따로 들고 있으면 ``(role_pos, role_count)``가 같아도 분자가 미세하게 다른 두
        상태가 존재하게 되어 다음 role이 갈린다 — 즉 누적합 자체가 숨은 전이 상태가 된다.
        증분 형식은 ``(role_pos, role_count)``만으로 다음 값이 결정되므로 그 구멍을 없애고,
        상태 필드도 하나 줄인다. 첫 표본에서 ``n'=1``이라 이득이 1이 되어 진입 prior가 정확히
        관측값으로 대체된다.
        """

        live = state.ball_state == BALL_ALIVE
        sample = live & state.active_player
        folded = state.player_pos * state.attack_dir[:, None]
        role_count = jnp.where(sample, state.role_pos_count + 1.0, state.role_pos_count)
        gain = jnp.where(sample, 1.0 / jnp.maximum(role_count, 1.0), 0.0)
        role_pos = state.role_pos + (folded - state.role_pos) * gain[:, None]
        return state._replace(
            role_pos=role_pos.astype(state.role_pos.dtype),
            role_pos_count=role_count.astype(state.role_pos_count.dtype),
        )

    @staticmethod
    def _json_default(value):
        """NumPy scalar/vector를 reproducibility manifest의 JSON 값으로 바꾼다."""

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            # np.longdouble.item() may return another np.longdouble and recurse
            # forever in json.default; an explicit Python float closes the type.
            return float(value)
        raise TypeError(f"not JSON serializable: {type(value).__name__}")

    def _build_dynamics_metadata(self) -> dict:
        stadium_dynamics = asdict(self.s_cfg)
        # Goal-area width is part of Law 13 mechanics: an attacking indirect
        # free kick awarded for an offence inside the defenders' goal area is
        # moved to the nearest point on its front line.  It therefore belongs
        # in the dynamics fingerprint together with goal-area length; treating
        # it as a render-only marking would merge environments whose restart
        # spots actually differ.
        config_metadata = {
            "ball": asdict(self.b_cfg),
            "stadium": stadium_dynamics,
            "engine": asdict(self.e_cfg),
            "foul": asdict(self.f_cfg),
            "reward": asdict(self.r_cfg),
        }
        raw = {
            # 내부 plain dict를 쓴다 — 공개 property는 매 호출 방어 복사본을 만든다.
            "schema_version": self._schema_version,
            "timebase": {
                "dt_phys": self.timebase.dt_phys,
                "control_fps": self.timebase.control_fps,
                "decimation": self.timebase.decimation,
            },
            "prng": {
                "implementation": "threefry2x32",
                "threefry_partitionable": False,
                "key_contract": "SCALAR_THREEFRY_V1",
            },
            "duration": {
                "game_duration_steps": self.game_duration,
                "episode_duration_s": self.episode_duration_s,
                "halftime": self.halftime,
                "long_stamina_reference_duration_s": self.long_stamina_reference_duration_s,
                "long_stamina_drain_base": self.long_stamina_drain_base,
            },
            "config": config_metadata,
            "teams": {
                "n_agents": self.n_agents,
                "n_opponents": self.n_opponents,
                "minimum_active_players": [
                    int(self.minimum_team_players[TEAM_0]),
                    int(self.minimum_team_players[TEAM_1]),
                ],
                "agent_keys": "STABLE_SLOT_V1",
            },
            "roster": [asdict(player) for player in self.players],
        }
        # 교체는 dynamics의 일부다. 같은 config라도 스케줄·벤치·결정자가 다르면 궤적이
        # 달라지므로 fingerprint가 갈라져야 하고, 산출물 manifest만 보고 재현할 수 있어야
        # 한다. 셋 다 비어 있으면 동역학이 교체 도입 이전과 완전히 같으므로 키 자체를 넣지
        # 않는다 — 궤적이 그대로인데 fingerprint만 갈라지면 기존 산출물이 거짓으로 stale이 된다.
        roster = getattr(self, "_bench_roster", {})
        seated = sum(len(players) for players in roster.values())
        autonomous = self.substitution_mode != "schedule"
        if self.substitutions or seated or autonomous:
            block = {
                "decider": self.substitution_mode,
                "bench_size": self.bench_size,
                # 사용자 함수는 내용을 해시할 수 없다. 이름이라도 남겨야 두 알고리즘이
                # 같은 지문으로 합쳐지지 않고, 지문만으로는 재현이 안 된다는 사실이
                # manifest에 드러난다.
                **({"decider_ref": _callable_identity(self._substitution_decider)}
                   if self.substitution_mode == "custom" else {}),
                "max_per_team": self.max_substitutions,
                # 한 정지에 몇 명까지 함께 바꾸는가는 궤적을 바꾼다 — 1과 5는 다른 경기다.
                "max_simultaneous": self.max_simultaneous_substitutions,
                "bench": {
                    str(int(team)): [asdict(row) for row in players]
                    for team, players in sorted(roster.items())
                },
            }
            if self.substitutions:
                block["schedule"] = {
                    "mode": "EXOGENOUS_SCHEDULE",
                    "count": len(self.substitutions),
                    "per_team_count": [
                        sum(1 for row in self.substitutions
                            if int(np.asarray(self._static_meta[0])[int(row.slot)]) == team)
                        for team in (TEAM_0, TEAM_1)
                    ],
                    "entries": [asdict(row) for row in self.substitutions],
                }
            raw["substitutions"] = block
        # 세트피스 키커 규칙도 dynamics의 일부다 — 누가 차느냐가 궤적을 바꾼다. 기본값
        # ``nearest``와 지정 없음이면 이 기능 도입 이전과 완전히 같으므로 키를 넣지 않는다.
        # 감독 설정도 dynamics의 일부다 — 교체·포메이션 결정이 궤적을 바꾼다. 감독을 주지
        # 않으면 두 모드 어댑터가 돌고 그 둘은 이미 위에 기록돼 있으므로 키를 넣지 않는다.
        if self.manager_mode != "composed":
            raw["manager"] = {
                "mode": self.manager_mode,
                **({"rules_version": manager_module.MANAGER_RULES_VERSION}
                   if self.manager_mode == "auto" else {}),
                **({"manager_ref": _callable_identity(self._manager)}
                   if self.manager_mode == "custom" else {}),
            }
        if self.restart_taker_mode != "nearest" or self._setpiece_plan_rows:
            raw["restart_taker"] = {
                "mode": self.restart_taker_mode,
                "rules_version": setpiece_taker_module.TAKER_RULES_VERSION,
                "plans": {
                    str(int(team)): asdict(plan)
                    for team, plan in sorted(
                        (self._setpiece_plan_rows or {}).items())
                },
                **({"decider_ref": _callable_identity(
                    self._restart_taker_decider)}
                   if self.restart_taker_mode == "custom" else {}),
            }
        # 포메이션도 dynamics의 일부다 — 지휘관이 다르면 궤적이 다르다. 기본값(고정 +
        # 킥오프 모양)에서는 궤적이 이 기능 도입 이전과 완전히 같으므로 키를 넣지 않는다.
        if self.formation_mode != "fixed" or self.base_layout != 0:
            raw["formation"] = {
                "commander": self.formation_mode,
                "base_layout": formation_module.LAYOUT_NAMES[self.base_layout],
                "layouts": len(formation_module.LAYOUTS),
                **({"commander_ref": _callable_identity(self._formation_decider)}
                   if self.formation_mode == "custom" else {}),
            }
        payload = json.dumps(
            raw,
            sort_keys=True,
            separators=(",", ":"),
            default=self._json_default,
        )
        return json.loads(payload)

    def dynamics_metadata(self) -> dict:
        """dataset/replay manifest에 저장할 JSON-safe dynamics 계약의 복사본."""

        return json.loads(json.dumps(self._dynamics_metadata))

    def _validate_configuration(self) -> None:
        """JIT 전에 잘못된 물리/규칙 조합을 즉시 거부한다."""
        validate_configuration(
            self.b_cfg,
            self.s_cfg,
            self.e_cfg,
            self.f_cfg,
            self.r_cfg,
        )

    def _validate_rosters(self) -> None:
        if len(self.agent_team) != self.n_agents:
            raise ValueError(f"agent_team has {len(self.agent_team)} players, expected {self.n_agents}")
        if len(self.opponent_team) != self.n_opponents:
            raise ValueError(f"opponent_team has {len(self.opponent_team)} players, expected {self.n_opponents}")
        players = self.agent_team + self.opponent_team
        if any(not isinstance(player, Agent) for player in players):
            raise ValueError("agent_team and opponent_team must contain only Agent objects")
        ids = [p.id for p in players]
        if any(not _is_host_integral(pid) for pid in ids):
            raise ValueError("all Agent.id values must be integers")
        id_bounds = np.iinfo(np.int32)
        if any(not 0 <= int(pid) <= id_bounds.max for pid in ids):
            raise ValueError(
                "Agent.id values must be non-negative and fit the JAX int32 "
                "State identity contract"
            )
        if len(set(map(int, ids))) != len(ids):
            raise ValueError("Agent.id values must be unique across both teams")
        for p in players:
            if not isinstance(p.is_gk, (bool, np.bool_)):
                raise ValueError(f"player {p.id} is_gk must be bool, got {p.is_gk!r}")
            if not _is_host_real_vector(p.init_pos, (DIM_Z,)):
                raise ValueError(
                    f"player {p.id} init_pos must contain exactly two real values"
                )
            values = (
                p.speed,
                p.tall,
                p.reach_z_max,
                p.ball_control,
                p.endurance_factor,
                *p.init_pos,
            )
            if not all(_is_host_real(v) and math.isfinite(float(v)) for v in values):
                raise ValueError(f"player {p.id} has a non-real or non-finite attribute")
            for field_name, value in zip(
                (
                    "speed",
                    "tall",
                    "reach_z_max",
                    "ball_control",
                    "endurance_factor",
                    "init_pos.x",
                    "init_pos.y",
                ),
                values,
            ):
                _require_float32_scalar(f"player {p.id} {field_name}", value)
            if p.speed <= 0 or p.tall <= 0 or p.reach_z_max <= 0:
                raise ValueError(f"player {p.id} speed, tall, and reach_z_max must be positive")
            if max(float(p.speed), float(p.tall), float(p.reach_z_max)) > _FLOAT32_DYNAMICS_MAGNITUDE_LIMIT:
                raise ValueError(
                    f"player {p.id} physical profile exceeds the float32 "
                    "quartic/norm-safe dynamics magnitude limit"
                )
            if float(p.speed) * self.e_cfg.dt_phys > _FLOAT32_DYNAMICS_MAGNITUDE_LIMIT:
                raise ValueError(
                    f"player {p.id} speed * dt_phys exceeds the float32 "
                    "quartic/norm-safe dynamics magnitude limit"
                )
            if (
                min(
                    self.e_cfg.a_max * self.e_cfg.dt_phys,
                    float(p.speed),
                )
                * self.e_cfg.dt_phys
                < _float32_position_resolution(self.s_cfg, self.e_cfg)
            ):
                raise ValueError(
                    f"player {p.id} cannot make a representable full-action "
                    "position step at the configured float32 spatial resolution"
                )
            head_z = float(p.tall) * self.e_cfg.reach_height_factor
            if float(p.reach_z_max) < head_z:
                raise ValueError(
                    f"player {p.id} reach_z_max must be at least head_z "
                    f"({p.reach_z_max!r} < {head_z:.6g})"
                )
            if self.e_cfg.body_top_frac * head_z <= self.e_cfg.leg_top:
                raise ValueError(
                    f"player {p.id} body collision band must have positive "
                    "height: body_top_frac * head_z must exceed leg_top"
                )
            if not 0.0 <= p.ball_control <= 1.0:
                raise ValueError(f"player {p.id} ball_control must lie in [0, 1]")
            if not 0.01 <= p.endurance_factor <= 100.0:
                raise ValueError(
                    f"player {p.id} endurance_factor must lie in [0.01, 100]"
                )
            if abs(p.init_pos[DIM_X]) > self.s_cfg.half_length or abs(p.init_pos[DIM_Y]) > self.s_cfg.half_width:
                raise ValueError(f"player {p.id} init_pos lies outside the pitch: {p.init_pos!r}")
        # ``Agent.init_pos`` is expressed in each team's attacking frame.
        # Initialization rotates the opponent formation by 180 degrees before the first
        # State is returned, so overlap validation must use those same *world* coordinates.
        # Otherwise two individually valid formations can become coincident only
        # after the rotation and reset exposes a physically impossible first
        # observation (the normal movement separator does not run until step 1).
        team0_world = np.asarray(
            [player.init_pos for player in self.agent_team], dtype=np.float64)
        team1_world = np.asarray(
            [player.init_pos for player in self.opponent_team], dtype=np.float64
        ) * np.asarray([ROTATE_180, ROTATE_180], dtype=np.float64)
        world_positions = np.concatenate([team0_world, team1_world], axis=0)
        # ``_validate_rosters`` runs before the public ``self.r_player`` cache is
        # assigned, so use the immutable Engine SSOT directly here.
        min_distance = 2.0 * self.e_cfg.r_player
        for left in range(self.N):
            for right in range(left + 1, self.N):
                distance = float(np.linalg.norm(
                    world_positions[left] - world_positions[right]))
                if distance < min_distance:
                    raise ValueError(
                        "initial world positions overlap: "
                        f"slot {left} (player {int(players[left].id)}) and "
                        f"slot {right} (player {int(players[right].id)}) are "
                        f"{distance:.6f} m apart, below 2*r_player={min_distance:.6f} m"
                    )
        for name, team in (("agent_team", self.agent_team), ("opponent_team", self.opponent_team)):
            goalkeeper_count = sum(bool(player.is_gk) for player in team)
            if goalkeeper_count != 1:
                raise ValueError(f"{name} must contain exactly one goalkeeper, got {goalkeeper_count}")

    def reset(self, key, decision_params=None, randomness=None):
        """JaxMARL 규약 dict 어댑터, 계산은 reset_array(단일 진실원천)"""
        obs, state = self.reset_array(
            key, decision_params=decision_params, randomness=randomness
        )
        return {a: obs[i] for i, a in enumerate(self._agent_keys)}, state

    def reset_state(self, key, decision_params=None, randomness=None):
        """관측 조립 없이 초기 State만 만든다. 물리/렌더/팩토리용 경량 reset 경로.

        리셋도 킥오프 키커를 **지정한다**. 그래서 학습 키커를 쓰는 환경은 reset에도 같은
        파라미터를 넘겨야 한다 — 안 넘기면 첫 킥오프만 다른 정책이 고르게 되고, 그 어긋남은
        롤아웃 어디에도 표시가 나지 않는다. ``_split_decision_params``가 그 짝을 강제한다.
        """
        key = _validate_prng_key(key)
        if randomness is not None:
            randomness = validate_randomness_control(
                randomness, decimation=self.timebase.decimation
            )
        _, taker_params = self._split_decision_params(decision_params)
        previous = self._taker_params
        previous_randomness = self._active_randomness_control
        self._taker_params = taker_params
        self._active_randomness_control = randomness
        try:
            return self._reset_state(key, randomness=randomness)
        finally:
            self._active_randomness_control = previous_randomness
            self._taker_params = previous

    def _reset_state(self, key, randomness=None):
        """리셋 본체 — 키커 파라미터는 호출 범위에 이미 실려 있다."""

        (
            teams,
            att_dir,
            gk,
            init_position,
            vmax,
            reach_z,
            head_z,
            player_ctrl,
            endurance_factor,
        ) = self._static_meta

        key, k_kick = jax.random.split(key)
        k_kick = select_random_key(
            randomness, RandomEvent.RESET_KICKOFF_TEAM, 0, k_kick
        )
        kick_team = jax.random.bernoulli(k_kick).astype(jnp.int32)
        # facing은 속도 파생값 — 정지 상태이므로 자기 공격 방향(team0=0, team1=π)이 된다.
        init_vel = jnp.zeros((self.N, DIM_Z), dtype=jnp.float32)
        init_facing = self.facing_from_velocity(init_vel, att_dir)

        state = State(
            t = jnp.int32(0),
            ball_pos = jnp.array([0.0, 0.0, self.r_ball], dtype=jnp.float32),
            ball_vel = jnp.zeros(DIM_ALL, dtype=jnp.float32),
            ball_spin = jnp.zeros(DIM_ALL, dtype=jnp.float32),
            player_ctrl = player_ctrl,
            ball_state = jnp.int32(BALL_DEAD),
            player_pos = init_position,
            player_vel = init_vel,
            player_facing = init_facing,
            attack_dir = att_dir,
            team_id = teams,
            gk_indices = gk,
            vmax = vmax,
            reach_z = reach_z,
            head_z = head_z,
            cooldown=jnp.zeros(self.N, dtype=jnp.float32),
            contact_lock_t=jnp.zeros(self.N, dtype=jnp.int32),
            stamina_long=jnp.ones(self.N, dtype=jnp.float32),
            stamina_short=jnp.ones(self.N, dtype=jnp.float32),
            endurance_factor=endurance_factor,
            aerial_recovery_t=jnp.zeros(self.N, dtype=jnp.int32),
            role_pos=init_position * att_dir[:, None],   # 킥오프 포메이션(공격 접힘) = 표본 0일 때의 prior
            role_pos_count=jnp.zeros(self.N, jnp.float32),

            ctrl_lock_t=jnp.zeros(self.N, dtype=jnp.int32),
            kickoff_team=kick_team,
            poss_team = kick_team,
            possession_t=jnp.int32(0),
            previous_poss_team=kick_team,
            last_touch_team = kick_team,
            gk_handling_restricted_team=jnp.int32(NO_TEAM),
            restart_team = kick_team,
            # Pre-match organisation happens before the simulated match clock.
            # A one-tick timer makes the opening kickoff immediately eligible;
            # goal-created kickoffs use the full 80 s restart window instead.
            restart_t=jnp.int32(1),
            restart_kind=jnp.int32(RK_KICKOFF),
            offside_flag=jnp.zeros(self.N, dtype=jnp.bool_),
            pass_team=jnp.int32(NO_TEAM),
            pass_t=jnp.int32(0),
            foul_kind=jnp.int32(FOUL_NONE),
            foul_actor=jnp.int32(NO_PLAYER),
            foul_victim=jnp.int32(NO_PLAYER),
            pending_taker=jnp.int32(NO_PLAYER),
            setpiece_taker=jnp.int32(NO_PLAYER),
            throw_taker=jnp.int32(NO_PLAYER),
            touch=jnp.zeros(self.N, dtype=jnp.int32),
            touch_event_actor=jnp.full(
                (self.timebase.decimation, TOUCH_EVENT_PHASE_COUNT),
                NO_PLAYER,
                dtype=jnp.int32,
            ),
            touch_event_code=jnp.full(
                (self.timebase.decimation, TOUCH_EVENT_PHASE_COUNT),
                TOUCH_NONE,
                dtype=jnp.int32,
            ),
            touch_event_player_id=jnp.full(
                (self.timebase.decimation, TOUCH_EVENT_PHASE_COUNT),
                NO_PLAYER,
                dtype=jnp.int32,
            ),
            touch_event_control_t=jnp.full(
                (self.timebase.decimation, TOUCH_EVENT_PHASE_COUNT),
                -1,
                dtype=jnp.int32,
            ),
            touch_event_toi=jnp.zeros(
                (self.timebase.decimation, TOUCH_EVENT_PHASE_COUNT),
                dtype=jnp.float32,
            ),
            touch_event_ball_pos=jnp.zeros(
                (self.timebase.decimation, TOUCH_EVENT_PHASE_COUNT, DIM_ALL),
                dtype=jnp.float32,
            ),
            touch_event_ball_vel_before=jnp.zeros(
                (self.timebase.decimation, TOUCH_EVENT_PHASE_COUNT, DIM_ALL),
                dtype=jnp.float32,
            ),
            touch_event_ball_vel_after=jnp.zeros(
                (self.timebase.decimation, TOUCH_EVENT_PHASE_COUNT, DIM_ALL),
                dtype=jnp.float32,
            ),
            touch_event_impulse=jnp.zeros(
                (self.timebase.decimation, TOUCH_EVENT_PHASE_COUNT),
                dtype=jnp.float32,
            ),
            ball_event_kind=jnp.full(
                (self.timebase.decimation,), BALL_EVENT_NONE, dtype=jnp.int32,
            ),
            ball_event_team=jnp.full(
                (self.timebase.decimation,), NO_TEAM, dtype=jnp.int32,
            ),
            ball_event_pos=jnp.zeros(
                (self.timebase.decimation, DIM_ALL), dtype=jnp.float32,
            ),
            ball_event_vel=jnp.zeros(
                (self.timebase.decimation, DIM_ALL), dtype=jnp.float32,
            ),
            ball_event_control_t=jnp.full(
                (self.timebase.decimation,), -1, dtype=jnp.int32,
            ),
            woodwork_kind=jnp.full(
                (self.timebase.decimation,), WOODWORK_NONE, dtype=jnp.int32,
            ),
            woodwork_pos=jnp.zeros(
                (self.timebase.decimation, DIM_ALL), dtype=jnp.float32,
            ),
            woodwork_vel_in=jnp.zeros(
                (self.timebase.decimation, DIM_ALL), dtype=jnp.float32,
            ),
            woodwork_control_t=jnp.full(
                (self.timebase.decimation,), -1, dtype=jnp.int32,
            ),
            score=jnp.zeros(TEAM_COUNT, jnp.int32),
            yellow_cards=jnp.zeros(self.N, jnp.int32),
            player_id=self.initial_player_ids,
            slot_generation=jnp.zeros(self.N, jnp.int32),
            on_pitch=jnp.ones(self.N, jnp.bool_),
            sent_off=jnp.zeros(self.N, jnp.bool_),
            restart_indirect=jnp.bool_(False),
            last_touch_code=jnp.int32(TOUCH_NONE),
            last_touch_actor=jnp.int32(NO_PLAYER),
            bench_player_id=self._bench_plan["player_id"],
            bench_vmax=self._bench_plan["vmax"],
            bench_reach_z=self._bench_plan["reach_z"],
            bench_head_z=self._bench_plan["head_z"],
            bench_player_ctrl=self._bench_plan["player_ctrl"],
            bench_endurance_factor=self._bench_plan["endurance_factor"],
            bench_is_gk=self._bench_plan["is_gk"],
            bench_role_pos=self._bench_plan["role_pos"],
            retired_player_id=jnp.full(
                (TEAM_COUNT, self.retired_depth), NO_PLAYER, jnp.int32
            ),
            subs_remaining=jnp.full(
                (TEAM_COUNT,), self.max_substitutions, jnp.int32
            ),
            sub_windows_used=jnp.zeros((TEAM_COUNT,), jnp.int32),
            sub_window_open_t=jnp.full((TEAM_COUNT,), -1, jnp.int32),
            # 킥오프 모양에서 출발한다.
            layout_index=jnp.full((TEAM_COUNT,), self.base_layout, jnp.int32),
            layout_since_t=jnp.zeros((TEAM_COUNT,), jnp.int32),
            # reset 키를 한 칸으로 접는다 — scan 깊숙이의 결정자가 쓸 유일한 에피소드 신원.
            episode_seed=jax.random.bits(
                select_random_key(
                    randomness,
                    RandomEvent.RESET_EPISODE_SEED,
                    0,
                    jax.random.fold_in(key, 0xE9),
                ),
                (),
                jnp.uint32,
            ).astype(jnp.int32),
        )
        state = state._replace(pending_taker=self._kickoff_taker(state, kick_team, init_position))
        # 킥오프 키커를 frame-0 전에 스폿으로 스냅한다. 첫 step 안에서만 이동시키면
        # 진입 action_agency가 forced kick을 못 보고 인과 킥 라벨을 닫는다.
        state = self._snap_kickoff_taker(state)
        # tick 0에 예정된 교체(예: 관측된 하프타임 교체를 후반 전용 에피소드로 재생할 때)도
        # step과 같은 "t=T 상태부터 새 사람" 계약을 따르게 한다.
        state = self._apply_scheduled_substitutions(state)
        return state

    def reset_array(self, key, decision_params=None, randomness=None):
        """배열형 JaxMARL reset: (obs[N,D], State). State-only 코드는 reset_state를 쓴다."""
        state = self.reset_state(
            key, decision_params=decision_params, randomness=randomness
        )
        return self.get_obs_array(state), state

    def project_substitution(
        self,
        state,
        slot,
        incoming_player_id,
        *,
        player_pos,
        role_pos,
        vmax,
        reach_z,
        head_z,
        player_ctrl,
        is_gk,
        endurance_factor=1.0,
        stamina_long_entry=1.0,
        stamina_short_entry=1.0,
        yellow_cards=0,
        player_vel=None,
    ):
        """Project one externally observed substitution atomically.

        This is the public playback adapter.  It always protects identities
        reserved by the environment schedule, treats every active player as a
        placement blocker, and completes active-restart projection before the
        returned State becomes observable.  Those are authority boundaries,
        not caller-selectable tuning flags: exposing the scheduler's internal
        bypasses here previously let an external caller consume a reserved id,
        return an entrant one metre from a free-kick spot, or place two active
        players at exactly the same coordinate.
        """

        _validate_float32_runtime()
        return self._project_substitution(
            state,
            slot,
            incoming_player_id,
            player_pos=player_pos,
            role_pos=role_pos,
            vmax=vmax,
            reach_z=reach_z,
            head_z=head_z,
            player_ctrl=player_ctrl,
            endurance_factor=endurance_factor,
            is_gk=is_gk,
            stamina_long_entry=stamina_long_entry,
            stamina_short_entry=stamina_short_entry,
            yellow_cards=yellow_cards,
            player_vel=player_vel,
            _placement_exempt=None,
            _scheduled_identity=False,
            _defer_restart_projection=False,
        )

    def _project_substitution(
        self,
        state,
        slot,
        incoming_player_id,
        *,
        player_pos,
        role_pos,
        vmax,
        reach_z,
        head_z,
        player_ctrl,
        is_gk,
        endurance_factor=1.0,
        stamina_long_entry=1.0,
        stamina_short_entry=1.0,
        yellow_cards=0,
        player_vel=None,
        _placement_exempt=None,
        _scheduled_identity=False,
        _defer_restart_projection=False,
    ):
        """Internal implementation shared with the static schedule executor.

        ``_placement_exempt``는 같은 tick의 나중 스케줄 row가 비울 slot을
        순차 배치 blocker에서만 제외하는 내부 마스크다.
        ``_scheduled_identity``도 내부 스케줄 실행기만 쓰는 정적 표식이다. public direct
        호출은 미래 행을 포함한 정적 스케줄의 모든 person id를 예약 자원으로 취급한다.
        ``_defer_restart_projection`` 역시 같은 tick의 모든 스케줄 row를 처리한 뒤 재개
        제약을 한 번만 적용하기 위한 내부 정적 표식이다. public direct 호출은 원자적으로
        합법인 State를 반환하도록 활성 재개 투영까지 이 함수 안에서 끝낸다.

        Identity uniqueness has one deliberate adapter-side boundary: ``State``
        stores current identities and slot generations, not an unbounded list of
        identities previously supplied by *ad-hoc direct* calls.  Therefore this
        method can enforce current/starter/static-schedule collisions exactly,
        while a playback adapter making several direct calls must never reuse one
        of its own departed ids.  Use the constructor ``substitutions`` schedule
        when the environment itself must validate the complete match-lifetime
        identity set; exact arbitrary direct-history validation would require a
        new State ledger/schema rather than pretending ``slot_generation`` retains
        information it does not contain.
        """

        # A direct eager call is a host API even when its scalars arrive as NumPy/JAX
        # 0-D arrays.  Previously only Python ``int`` pairs entered validation, so
        # ``slot=3.9, incoming_player_id=999.7`` silently truncated to slot 3 / id 999
        # and performed a real identity transition.  Validate scalar integer *dtype*
        # before any int32 cast.  Traced values cannot raise dynamically; their static
        # dtype/rank and dynamic range are folded into ``eligible`` below.
        host_values = (
            state.player_id, slot, incoming_player_id,
            player_pos, role_pos, vmax, reach_z, head_z, player_ctrl,
            endurance_factor,
            is_gk, stamina_long_entry, stamina_short_entry,
            yellow_cards, player_vel,
        )
        host_call = not any(
            isinstance(value, jax.core.Tracer)
            for value in jax.tree_util.tree_leaves(host_values)
        )
        if type(_scheduled_identity) is not bool:
            raise TypeError(
                "_scheduled_identity must be a Python bool internal marker"
            )
        if type(_defer_restart_projection) is not bool:
            raise TypeError(
                "_defer_restart_projection must be a Python bool internal marker"
            )
        id_bounds = np.iinfo(np.int32)
        if host_call:
            slot_host = np.asarray(slot)
            if (
                slot_host.shape != ()
                or not np.issubdtype(slot_host.dtype, np.integer)
                or np.issubdtype(slot_host.dtype, np.bool_)
                or not 0 <= int(slot_host) < self.N
            ):
                raise ValueError(
                    f"slot must be an integer scalar in [0, {self.N}), got {slot!r}"
                )
            incoming_host = np.asarray(incoming_player_id)
            if (
                incoming_host.shape != ()
                or not np.issubdtype(incoming_host.dtype, np.integer)
                or np.issubdtype(incoming_host.dtype, np.bool_)
                or not 0 <= int(incoming_host) <= id_bounds.max
            ):
                raise ValueError(
                    "incoming_player_id must be a non-negative integer scalar fitting the JAX "
                    f"int32 State identity contract, got {incoming_player_id!r}"
                )
        if host_call:
            slot_i = int(slot)
            incoming_i = int(incoming_player_id)
            if bool(np.asarray(self._is_terminal(state))):
                raise ValueError("a substitution cannot be projected after the match is terminal")
            if incoming_i in set(map(int, np.asarray(state.player_id))):
                raise ValueError(f"incoming_player_id {incoming_i} is already present in the match")
            if incoming_i in set(map(int, np.asarray(self.initial_player_ids))):
                raise ValueError(
                    f"incoming_player_id {incoming_i} is a starter identity and "
                    "cannot re-enter this match"
                )
            if (
                not _scheduled_identity
                and self._substitution_plan is not None
                and incoming_i in set(map(
                    int, np.asarray(self._substitution_plan["player_id"])
                ))
            ):
                raise ValueError(
                    f"incoming_player_id {incoming_i} is reserved by the "
                    "environment's substitution schedule"
                )
            vectors = {
                "player_pos": player_pos,
                "role_pos": role_pos,
                "player_vel": (0.0, 0.0) if player_vel is None else player_vel,
            }
            for name, value in vectors.items():
                if (
                    not _is_host_real_vector(value, (DIM_Z,))
                    or not all(math.isfinite(float(item)) for item in value)
                ):
                    raise ValueError(
                        f"{name} must contain exactly two finite real values, got {value!r}"
                    )
                for component in value:
                    _require_float32_scalar(name, component)
            if (
                abs(float(player_pos[DIM_X])) > self.hx + self.e_cfg.player_boundary_margin
                or abs(float(player_pos[DIM_Y])) > self.hy + self.e_cfg.player_boundary_margin
            ):
                raise ValueError(f"player_pos lies outside the player boundary: {player_pos!r}")
            if (
                abs(float(role_pos[DIM_X])) > self.hx + self.e_cfg.player_boundary_margin
                or abs(float(role_pos[DIM_Y])) > self.hy + self.e_cfg.player_boundary_margin
            ):
                raise ValueError(
                    "role_pos lies outside the folded player boundary: "
                    f"{role_pos!r}"
                )
            positive = {
                "vmax": vmax,
                "reach_z": reach_z,
                "head_z": head_z,
                "endurance_factor": endurance_factor,
            }
            if any(
                not _is_host_real(v)
                or not math.isfinite(float(v))
                or float(v) <= 0.0
                for v in positive.values()
            ):
                raise ValueError(f"physical attributes must be finite and positive: {positive}")
            for name, value in positive.items():
                _require_float32_scalar(name, value)
            if max(float(vmax), float(reach_z), float(head_z)) > _FLOAT32_DYNAMICS_MAGNITUDE_LIMIT:
                raise ValueError(
                    "incoming physical profile exceeds the float32 "
                    "quartic/norm-safe dynamics magnitude limit"
                )
            if float(vmax) * self.e_cfg.dt_phys > _FLOAT32_DYNAMICS_MAGNITUDE_LIMIT:
                raise ValueError(
                    "vmax * dt_phys exceeds the float32 quartic/norm-safe "
                    "dynamics magnitude limit"
                )
            if not 0.01 <= float(endurance_factor) <= 100.0:
                raise ValueError(
                    "endurance_factor must lie in [0.01, 100], "
                    f"got {endurance_factor!r}"
                )
            if (
                min(
                    self.e_cfg.a_max * self.e_cfg.dt_phys,
                    float(vmax),
                )
                * self.e_cfg.dt_phys
                < _float32_position_resolution(self.s_cfg, self.e_cfg)
            ):
                raise ValueError(
                    "vmax cannot produce a representable full-action position "
                    "step at the configured float32 spatial resolution"
                )
            if float(reach_z) < float(head_z):
                raise ValueError(
                    "reach_z must be at least head_z for an incoming player"
                )
            if self.e_cfg.body_top_frac * float(head_z) <= self.e_cfg.leg_top:
                raise ValueError(
                    "incoming player body collision band must have positive "
                    "height: body_top_frac * head_z must exceed leg_top"
                )
            if (
                not _is_host_real(player_ctrl)
                or not math.isfinite(float(player_ctrl))
                or not 0.0 <= float(player_ctrl) <= 1.0
            ):
                raise ValueError(f"player_ctrl must lie in [0, 1], got {player_ctrl!r}")
            _require_float32_scalar("player_ctrl", player_ctrl)
            if not isinstance(is_gk, (bool, np.bool_)):
                raise ValueError(f"is_gk must be boolean, got {is_gk!r}")
            if not bool(np.asarray(
                self._goalkeeper_substitution_role_compatible(
                    state, slot_i, is_gk
                )
            )):
                raise ValueError("a substitution must preserve the slot's goalkeeper role")
            for stamina_name, stamina_value in (
                ("stamina_long_entry", stamina_long_entry),
                ("stamina_short_entry", stamina_short_entry),
            ):
                if (
                    not _is_host_real(stamina_value)
                    or not math.isfinite(float(stamina_value))
                    or not 0.0 <= float(stamina_value) <= 1.0
                ):
                    raise ValueError(
                        f"{stamina_name} must lie in [0, 1], got {stamina_value!r}"
                    )
                _require_float32_scalar(stamina_name, stamina_value)
            # 진입 속도는 그 선수의 실효 최고속을 넘을 수 없다. 다른 모든 입력은 경계가 있는데
            # 속도만 유한성만 봤다 — 초과 상태를 주입하면 다음 서브스텝의 vmax 캡이 마찰 타원을
            # 수십 배 넘는 Δv를 만들고, 그 프레임은 이동 액션으로 역산되지 않는다.
            entry_speed_cap = float(self.effective_vmax(
                float(vmax), float(stamina_long_entry), float(stamina_short_entry)
            ))
            entry_speed = float(np.linalg.norm(np.asarray(vectors["player_vel"], float)))
            if entry_speed > entry_speed_cap + GEOMETRY_EPS:
                raise ValueError(
                    "player_vel must not exceed the incoming player's effective vmax "
                    f"({entry_speed:.4f} > {entry_speed_cap:.4f})"
                )
            if (
                not _is_host_integral(yellow_cards)
                or not 0 <= int(yellow_cards) < YELLOW_CARD_SEND_OFF_COUNT
            ):
                raise ValueError("incoming yellow_cards must be zero or one")

        slot_input = jnp.asarray(slot)
        slot_type_valid = (
            slot_input.shape == ()
            and jnp.issubdtype(slot_input.dtype, jnp.integer)
            and not jnp.issubdtype(slot_input.dtype, jnp.bool_)
        )
        if slot_type_valid:
            # Range-check before int32 conversion so uint32/int64 cannot wrap into a
            # different valid slot.
            slot_valid = (slot_input >= 0) & (slot_input < self.N)
            raw_slot = slot_input.astype(jnp.int32)
        else:
            slot_valid = jnp.bool_(False)
            raw_slot = jnp.int32(0)
        slot = jnp.clip(raw_slot, 0, self.N - 1)

        incoming_input = jnp.asarray(incoming_player_id)
        incoming_type_valid = (
            incoming_input.shape == ()
            and jnp.issubdtype(incoming_input.dtype, jnp.integer)
            and not jnp.issubdtype(incoming_input.dtype, jnp.bool_)
        )
        if incoming_type_valid:
            if jnp.issubdtype(incoming_input.dtype, jnp.unsignedinteger):
                incoming_range_valid = incoming_input <= id_bounds.max
            else:
                incoming_range_valid = (
                    (incoming_input >= 0)
                    & (incoming_input <= id_bounds.max)
                )
            incoming_player_id = incoming_input.astype(state.player_id.dtype)
        else:
            incoming_range_valid = jnp.bool_(False)
            incoming_player_id = jnp.zeros((), dtype=state.player_id.dtype)
        def numeric_value(value, shape, dtype):
            """Sanitize traced continuous inputs without lossy bool/complex casts."""

            array = jnp.asarray(value)
            type_valid = (
                array.shape == shape
                and (
                    jnp.issubdtype(array.dtype, jnp.floating)
                    or jnp.issubdtype(array.dtype, jnp.integer)
                )
                and not jnp.issubdtype(array.dtype, jnp.bool_)
            )
            if type_valid:
                return array.astype(dtype), jnp.bool_(True)
            return jnp.zeros(shape, dtype=dtype), jnp.bool_(False)

        player_pos, player_pos_type_valid = numeric_value(
            player_pos, (DIM_Z,), state.player_pos.dtype)
        role_pos, role_pos_type_valid = numeric_value(
            role_pos, (DIM_Z,), state.role_pos.dtype)
        incoming_vel, incoming_vel_type_valid = numeric_value(
            (0.0, 0.0) if player_vel is None else player_vel,
            (DIM_Z,), state.player_vel.dtype)
        vmax, vmax_type_valid = numeric_value(vmax, (), state.vmax.dtype)
        reach_z, reach_z_type_valid = numeric_value(reach_z, (), state.reach_z.dtype)
        head_z, head_z_type_valid = numeric_value(head_z, (), state.head_z.dtype)
        player_ctrl, player_ctrl_type_valid = numeric_value(
            player_ctrl, (), state.player_ctrl.dtype)
        endurance_factor, endurance_factor_type_valid = numeric_value(
            endurance_factor, (), state.endurance_factor.dtype
        )
        stamina_long_entry, stamina_long_type_valid = numeric_value(
            stamina_long_entry, (), state.stamina_long.dtype)
        stamina_short_entry, stamina_short_type_valid = numeric_value(
            stamina_short_entry, (), state.stamina_short.dtype)

        is_gk_input = jnp.asarray(is_gk)
        is_gk_type_valid = (
            is_gk_input.shape == ()
            and jnp.issubdtype(is_gk_input.dtype, jnp.bool_)
        )
        is_gk_value_valid = (
            ((is_gk_input == 0) | (is_gk_input == 1))
            if is_gk_type_valid
            else jnp.bool_(False)
        )
        is_gk = (
            is_gk_input.astype(state.gk_indices.dtype)
            if is_gk_type_valid
            else jnp.zeros((), dtype=state.gk_indices.dtype)
        )

        yellow_input = jnp.asarray(yellow_cards)
        yellow_type_valid = (
            yellow_input.shape == ()
            and jnp.issubdtype(yellow_input.dtype, jnp.integer)
            and not jnp.issubdtype(yellow_input.dtype, jnp.bool_)
        )
        yellow_value_valid = (
            ((yellow_input >= 0) & (yellow_input < YELLOW_CARD_SEND_OFF_COUNT))
            if yellow_type_valid
            else jnp.bool_(False)
        )
        yellow_cards = (
            yellow_input.astype(state.yellow_cards.dtype)
            if yellow_type_valid
            else jnp.zeros((), dtype=state.yellow_cards.dtype)
        )

        player_vel_all = state.player_vel.at[slot].set(incoming_vel)
        player_facing_all = self.facing_from_velocity(player_vel_all, state.attack_dir)

        def clear_slot_reference(value):
            return jnp.where(value == slot, jnp.int32(NO_PLAYER), value).astype(jnp.int32)

        duplicate_identity = jnp.any(state.player_id == incoming_player_id)
        starter_identity = jnp.any(self.initial_player_ids == incoming_player_id)
        reserved_identity = (
            jnp.bool_(False)
            if self._substitution_plan is None
            else jnp.any(
                self._substitution_plan["player_id"] == incoming_player_id
            )
        )
        role_compatible = self._goalkeeper_substitution_role_compatible(
            state, slot, is_gk == 1
        )
        values_valid = (
            player_pos_type_valid & role_pos_type_valid & incoming_vel_type_valid
            & vmax_type_valid & reach_z_type_valid & head_z_type_valid
            & player_ctrl_type_valid
            & endurance_factor_type_valid
            & stamina_long_type_valid & stamina_short_type_valid
            & jnp.bool_(is_gk_type_valid) & is_gk_value_valid
            & jnp.bool_(yellow_type_valid) & yellow_value_valid
            & jnp.all(jnp.isfinite(player_pos))
            & jnp.all(jnp.isfinite(role_pos))
            & jnp.all(jnp.isfinite(incoming_vel))
            & jnp.isfinite(vmax) & (vmax > 0)
            & (vmax <= jnp.asarray(
                _FLOAT32_DYNAMICS_MAGNITUDE_LIMIT, dtype=state.vmax.dtype
            ))
            & (vmax * self.e_cfg.dt_phys <= jnp.asarray(
                _FLOAT32_DYNAMICS_MAGNITUDE_LIMIT, dtype=state.vmax.dtype
            ))
            & (
                jnp.minimum(
                    jnp.asarray(
                        self.e_cfg.a_max * self.e_cfg.dt_phys,
                        dtype=state.vmax.dtype,
                    ),
                    vmax,
                )
                * self.e_cfg.dt_phys
                >= jnp.asarray(
                    _float32_position_resolution(self.s_cfg, self.e_cfg),
                    dtype=state.vmax.dtype,
                )
            )
            # |v| <= effective_vmax — 상태 불변식을 주입 경로에서도 닫는다.
            & (jnp.linalg.norm(incoming_vel)
               <= self.effective_vmax(
                   vmax, stamina_long_entry, stamina_short_entry
               ) + GEOMETRY_EPS)
            & jnp.isfinite(reach_z) & (reach_z > 0)
            & jnp.isfinite(head_z) & (head_z > 0)
            & (reach_z <= jnp.asarray(
                _FLOAT32_DYNAMICS_MAGNITUDE_LIMIT, dtype=state.reach_z.dtype
            ))
            & (head_z <= jnp.asarray(
                _FLOAT32_DYNAMICS_MAGNITUDE_LIMIT, dtype=state.head_z.dtype
            ))
            & (reach_z >= head_z)
            & (self.e_cfg.body_top_frac * head_z > self.e_cfg.leg_top)
            & jnp.isfinite(player_ctrl) & (player_ctrl >= 0) & (player_ctrl <= 1)
            & jnp.isfinite(endurance_factor) & (endurance_factor >= 0.01)
            & (endurance_factor <= 100.0)
            & jnp.isfinite(stamina_long_entry)
            & (stamina_long_entry >= 0) & (stamina_long_entry <= 1)
            & jnp.isfinite(stamina_short_entry)
            & (stamina_short_entry >= 0) & (stamina_short_entry <= 1)
            & role_compatible
            & (jnp.abs(player_pos[DIM_X]) <= self.hx + self.e_cfg.player_boundary_margin)
            & (jnp.abs(player_pos[DIM_Y]) <= self.hy + self.e_cfg.player_boundary_margin)
            & (jnp.abs(role_pos[DIM_X]) <= self.hx + self.e_cfg.player_boundary_margin)
            & (jnp.abs(role_pos[DIM_Y]) <= self.hy + self.e_cfg.player_boundary_margin)
        )
        eligible = (slot_valid & incoming_range_valid & (~self._is_terminal(state))
                    & state.on_pitch[slot] & (~state.sent_off[slot])
                    & (~duplicate_identity) & (~starter_identity) & values_valid)
        eligible = eligible & (
            jnp.bool_(_scheduled_identity) | (~reserved_identity)
        )
        had_pending_restart = (
            restart_timer_active(state.restart_t) & (state.pending_taker == slot)
        )
        # Foul identities are person-owned, while the storage references a
        # reusable roster slot.  Once either referenced person leaves, keeping
        # the kind would attribute that foul to the entrant (or expose a
        # non-NONE kind with NO_PLAYER identities).  End the telemetry latch
        # atomically with the identity transition; the referee restart itself
        # remains represented independently by restart_kind/team/timer.
        replaced_foul_reference = (
            (state.foul_actor == slot) | (state.foul_victim == slot)
        )
        projected = state._replace(
            player_id=state.player_id.at[slot].set(incoming_player_id),
            slot_generation=state.slot_generation.at[slot].add(jnp.int32(1)),
            on_pitch=state.on_pitch.at[slot].set(True),
            sent_off=state.sent_off.at[slot].set(False),
            player_pos=state.player_pos.at[slot].set(player_pos),
            player_vel=player_vel_all,
            player_facing=player_facing_all,
            # 투입 선수는 자기 anchor를 처음부터 쌓는다. 지정 role_pos는 표본이 생기기 전의
            # prior일 뿐이고, 나간 선수의 누적은 승계하지 않는다.
            role_pos=state.role_pos.at[slot].set(role_pos),
            role_pos_count=state.role_pos_count.at[slot].set(
                jnp.zeros((), dtype=state.role_pos_count.dtype)),
            vmax=state.vmax.at[slot].set(vmax),
            reach_z=state.reach_z.at[slot].set(reach_z),
            head_z=state.head_z.at[slot].set(head_z),
            player_ctrl=state.player_ctrl.at[slot].set(
                player_ctrl
            ),
            endurance_factor=state.endurance_factor.at[slot].set(
                endurance_factor
            ),
            aerial_recovery_t=state.aerial_recovery_t.at[slot].set(jnp.int32(0)),
            gk_indices=state.gk_indices.at[slot].set(
                is_gk
            ),
            stamina_long=state.stamina_long.at[slot].set(
                jnp.clip(stamina_long_entry, 0.0, 1.0)
            ),
            stamina_short=state.stamina_short.at[slot].set(
                jnp.clip(stamina_short_entry, 0.0, 1.0)
            ),
            cooldown=state.cooldown.at[slot].set(0.0),
            contact_lock_t=state.contact_lock_t.at[slot].set(jnp.int32(0)),
            ctrl_lock_t=state.ctrl_lock_t.at[slot].set(jnp.int32(0)),
            yellow_cards=state.yellow_cards.at[slot].set(
                yellow_cards
            ),
            offside_flag=state.offside_flag.at[slot].set(False),
            # 마지막 터치 신원은 슬롯이 아니라 **사람**에 붙는다. 슬롯 번호만 남겨 두면
            # 투입 선수가 "내가 마지막으로 찼다"는 관측을 물려받고, 그 위에 세운 수신
            # 게이트가 새 선수를 패스 수신 후보에서 빼 버린다.
            last_touch_actor=jnp.where(
                state.last_touch_actor == slot,
                jnp.int32(NO_PLAYER),
                state.last_touch_actor,
            ),
            touch=state.touch.at[slot].set(jnp.int32(TOUCH_NONE)),
            pending_taker=clear_slot_reference(state.pending_taker),
            # The outgoing person can no longer commit a double touch, but
            # identity replacement is not a touch by another player.  Keep
            # the ball's direct-goal provenance until the next real contact;
            # a dedicated negative sentinel avoids aliasing the entrant who
            # now occupies this slot.
            setpiece_taker=jnp.where(
                state.setpiece_taker == slot,
                jnp.int32(DEPARTED_TAKER),
                state.setpiece_taker,
            ).astype(jnp.int32),
            throw_taker=jnp.where(
                state.throw_taker == slot,
                jnp.int32(DEPARTED_TAKER),
                state.throw_taker,
            ).astype(jnp.int32),
            foul_kind=jnp.where(
                replaced_foul_reference,
                jnp.int32(FOUL_NONE),
                state.foul_kind,
            ).astype(jnp.int32),
            foul_actor=clear_slot_reference(state.foul_actor),
            foul_victim=clear_slot_reference(state.foul_victim),
        )
        # The replaced identity's offside flag was cleared above.  If it was the
        # phase's last flagged attacker, carrying pass_t/team into the public
        # post-substitution State creates an active phase with no subject until
        # the next physics step.  Normalize inside the atomic identity transition.
        projected = self._normalize_pass_latch(projected)
        projected = self._normalize_foul_latch(projected)
        # 투입 좌표는 호출자가 주는 값이라 이미 온피치인 선수와 겹칠 수 있다. 이 전이는
        # 스캔·하프타임 뒤에 적용되고 그 뒤에는 어떤 분리도 돌지 않으므로, 겹친 채로 반환하면
        # 물리적으로 불가능한 상태가 그대로 관측·데이터셋에 실린다(실측: 두 활성 슬롯이 정확히
        # 같은 좌표, 다음 프레임의 _separate가 풀 때까지 한 프레임 유지). 순간이동해 들어온
        # 쪽이 겹침을 흡수해야 하므로 나머지를 고정하고 투입 선수만 민다 — 재개 투영이 침범자에게
        # 겹침을 지우는 것과 같은 소유 규칙이다. 겹치지 않는 투입은 이 호출이 항등이라 실데이터
        # playback의 좌표는 그대로 보존된다.
        # 지정 좌표가 이미 점유돼 있으면 가장 가까운 빈 자리로 옮긴다. 이 전이는 스캔·하프타임
        # 뒤에 적용되고 그 뒤에는 어떤 정합도 돌지 않으므로, 겹친 채로 반환하면 물리적으로
        # 불가능한 상태가 그대로 관측·데이터셋에 실린다(실측: 간격 0.0000 m).
        entering = self.player_indices == slot
        if _placement_exempt is None:
            placement_exempt = jnp.zeros(self.N, dtype=bool)
        else:
            placement_exempt = jnp.asarray(_placement_exempt)
            if placement_exempt.shape != (self.N,):
                raise ValueError(
                    "_placement_exempt must have shape "
                    f"({self.N},), got {placement_exempt.shape}"
                )
            if placement_exempt.dtype != jnp.dtype(jnp.bool_):
                raise TypeError("_placement_exempt must have boolean dtype")
        entry_point = self.nearest_free_position(
            player_pos,
            projected.player_pos,
            projected.active_player & (~entering) & (~placement_exempt),
            orientation=projected.attack_dir[slot],
        )
        entry_gap = jnp.linalg.norm(
            entry_point[None, :] - projected.player_pos, axis=1
        )
        entry_blockers = (
            projected.active_player & (~entering) & (~placement_exempt)
        )
        entry_point_valid = (
            jnp.all(jnp.where(
                entry_blockers,
                entry_gap >= 2.0 * self.r_player + GEOMETRY_EPS,
                True,
            ))
            & (jnp.abs(entry_point[DIM_X])
               <= self.hx + self.e_cfg.player_boundary_margin)
            & (jnp.abs(entry_point[DIM_Y])
               <= self.hy + self.e_cfg.player_boundary_margin)
        )
        # In a deliberately tiny/crowded custom geometry the finite local and
        # global candidate set can be exhausted.  The outgoing player already
        # occupied one point that becomes free at this identity boundary; keep
        # that exact point as the final minimal-intervention fallback when it
        # is still inside the boundary and clear of every *remaining* active
        # blocker.  If even that point is illegal, preserve fail-closed rather
        # than returning an overlapping State.
        outgoing_point = state.player_pos[slot]
        outgoing_gap = jnp.linalg.norm(
            outgoing_point[None, :] - projected.player_pos, axis=1
        )
        outgoing_point_valid = (
            jnp.all(jnp.where(
                entry_blockers,
                outgoing_gap >= 2.0 * self.r_player + GEOMETRY_EPS,
                True,
            ))
            & jnp.all(jnp.isfinite(outgoing_point))
            & (jnp.abs(outgoing_point[DIM_X])
               <= self.hx + self.e_cfg.player_boundary_margin)
            & (jnp.abs(outgoing_point[DIM_Y])
               <= self.hy + self.e_cfg.player_boundary_margin)
        )
        use_outgoing_point = (~entry_point_valid) & outgoing_point_valid
        entry_point = jnp.where(
            use_outgoing_point, outgoing_point, entry_point
        )
        entry_point_valid = entry_point_valid | outgoing_point_valid
        eligible = eligible & entry_point_valid
        projected = projected._replace(
            player_pos=projected.player_pos.at[slot].set(
                entry_point.astype(projected.player_pos.dtype))
        )
        # Replacing the designated player during a dead ball must not leave an
        # active restart without a taker.  Choose from the *final* post-placement
        # roster: ``nearest_free_position`` can move an entrant away from the
        # requested spot, and selecting before that move can leave a farther
        # entrant designated while a teammate is standing on the ball.
        # Goal kicks prefer the goalkeeper but legally fall back to a teammate.
        # A GK hold is different: possession in the hands belongs to an actual
        # goalkeeper, so falling back to an outfielder creates an illegal hold.
        replacement_any = self._designate_taker_when(
            had_pending_restart,
            projected,
            projected.ball_pos[:DIM_Z],
            projected.restart_team,
            projected.restart_kind == RK_GOALKICK,
            # 진행 중인 재개라 state의 종류가 곧 현재 종류다. 이 인자를 빠뜨리면 교체 후
            # 키커 재지정이 TypeError로 죽는다 — 필수 인자로 바꾼 효과가 여기서 났다.
            projected.restart_kind,
            projected.restart_indirect,
        )
        hold_candidates = (
            projected.active_player
            & (projected.team_id == projected.restart_team)
            & (projected.gk_indices == 1)
        )
        hold_distance = jnp.linalg.norm(
            projected.player_pos - projected.ball_pos[:DIM_Z][None, :], axis=1
        )
        replacement_gk = jnp.where(
            jnp.any(hold_candidates),
            jnp.argmin(jnp.where(hold_candidates, hold_distance, jnp.inf)),
            jnp.int32(NO_PLAYER),
        ).astype(jnp.int32)
        replacement_taker = jnp.where(
            projected.restart_kind == RK_GK_HOLD,
            replacement_gk,
            replacement_any,
        )
        projected = projected._replace(
            pending_taker=jnp.where(
                had_pending_restart, replacement_taker, projected.pending_taker
            ).astype(jnp.int32)
        )
        # Public playback callers use this function directly as well as through the
        # schedule.  Keep the permanent-dismissal rule inside this atomic transition so
        # no call path can resurrect a sent-off slot.
        result = jax.tree_util.tree_map(
            lambda new, old: jnp.where(eligible, new, old), projected, state
        )
        if not _defer_restart_projection:
            # A direct public transition is itself an observable State
            # boundary.  An entrant placed inside an active restart exclusion
            # zone must not remain illegal until the following control frame.
            # Gate on successful eligibility so invalid/terminal calls remain
            # bitwise fail-closed, including under jit where they cannot raise.
            result = lax.cond(
                eligible,
                lambda current: self._project_restart_positions(current)[0],
                lambda current: current,
                result,
            )
        return result

    def action_agency(self, state):
        """행동 강제성 마스크(per-agent) — 이 state에서 각 선수의 액션이 자유 결정으로 결과에
        반영되는지, 아니면 env가 강제/무시하는지. **주어진 state(액션이 조건으로 삼은 스텝 진입
        시점)에서 결정론적으로 계산** — 제출한 액션값과 무관(국면 소유권·강제 규칙만으로 판정).
        BC/RL에서 강제 프레임을 정책 결정처럼 학습하지 않도록 마스킹하는 신호.

        반환 dict(모두 bool[N]):
          move_forced: 이동이 env에 의해 대체됨 — 세트피스 키커 강제 워킹, 현재 재개 거리 위반의
                       최소 합법 위치 투영 또는 비활성 선수. ※vmax·스태미나·마찰 타원
                       물리 클립은 방향 의도가 반영되므로 강제에 포함하지 않는다.
          kick_gated:  해당 control frame의 F2B 파라미터 채널이 **국면/상태상**
                       완전히 무효 — 비활성, 데드볼, 재개 중 비지정 키커. 도달·
                       타이머·재터치는 frame 안에 이동/소유전이/타 선수 접촉으로 바뀌어
                       entry값으로 hard-gate하면 실제 킥의 false-negative가 생긴다. 그들은
                       ``f2b_avail``(현재 정확 ready-now)과 ``kick_applied``(사후 인과)로 분리한다.
          kick_forced: 키커 도착 후 gameplay hold 소진으로 킥이 의지와 무관하게
                       강제 발사(`_kick_gate`의 forced). 진입 타이머가 해당 종류의 ready 경계를
                       이번 control step 안에 통과하는지 판정한다.
                       '언제 찰지'는 강제지만 실제 킥 파라미터는 에이전트 액션이므로 ``bc_action_mask``는
                       킥 차원을 열어 두고, 이동 강제 여부만 별도로 마스킹한다.
          halftime_reset: 이 control frame 끝의 하프타임 전환이 post-state를 덮어써 제출 액션과
                          결과의 인과를 끊음. 사전 행동 권한은 유지하되 BC 라벨은 전 차원 차단한다.
        move_forced/kick_gated/kick_forced는 상호배타적이지 않다(예: 재개 준비 중 지정 키커는
        강제 접근 때문에 move_forced=T이고 아직 발사 전이면 kick_gated=T다). kick_forced일 땐
        kick_gated=False로 정리(강제 발사=파라미터 효력 있음)."""
        N = self.N
        ar = self.player_indices
        _, _, _, sp_active = self._setpiece_kick_lock(state)
        alive = state.ball_state == BALL_ALIVE
        restart_active = restart_timer_active(state.restart_t)
        active = state.active_player
        is_designated = (ar == state.pending_taker) & (state.pending_taker >= 0)
        # A policy command persists for the whole control frame.  Agency is a
        # potential/causality mask, not an entry-substep contact prediction:
        # action-dependent reach and a retouch latch that another player may
        # clear inside the frame cannot safely hard-gate parameters here.
        # Exact geometry/rule checks stay in ``_kick_gate`` and realised
        # causality is reported by ``kick_applied``.
        kick_can_fire = (~restart_active) & alive & active
        # [B] 킥 타이밍 강제: 지정 키커의 킥은 setup 완료 시 결정론적 발사(_kick_gate와 정렬). BC가 '언제
        # 찰지'를 정책 결정처럼 학습하지 않게 강제 프레임을 forced로 잡되, **킥 파라미터는 bc_action_mask에서
        # 학습 대상으로 열려 있다**(kick_ok=~kick_gated, 실현 여부는 kick_applied와 AND).
        #  ★진입 setup_done만 보면 발화 스텝을 놓친다: restart_t는 서브스텝당 1 감소하고,
        #  종류별 임계(window - data-derived total delay)를 통과하는 도달 완료 서브스텝에 발사된다.
        #  진입 restart_t가 임계보다 decimation 이내면 이 스텝 안에 발사되므로 kick_forced로
        #  잡아 마스크를 연다(놓치면 실제 세트피스 킥 라벨의 대부분이 kick_gated=True로 드롭됨).
        #  경기 시작·후반 시작 킥오프는 준비 시간을 경기 시계 밖에서 보낸 것으로 timer=1이다.
        fires_this_step = self._setpiece_release_within_control_frame(state)
        kick_forced = restart_active & is_designated & active & fires_this_step
        kick_gated = (~kick_can_fire) & (~kick_forced)
        # 현재 거리 위반자는 step 시작과 동시에 최소 합법 위치로 투영된다. 이 이동은
        # 정책 액션의 결과가 아니므로 해당 control frame의 이동 라벨/권한을 마스킹한다.
        # 진입 시점에 확정적으로 규칙 투영될 침범자. 충돌 해결 때문에 함께 밀리는 원래
        # 비침범자는 실제 solver를 실행해야만 알 수 있으므로, step 말미에 누적된
        # ``restart_position_forced``를 move_forced에 OR해 BC mask를 사후 정합한다. 여기서
        # 전체 projector를 다시 실행하면 action_agency를 포함하는 모든 JIT 그래프가 쌍별
        # solver를 중복 인라인해 컴파일 시간이 수배로 늘어난다.
        projection_forced = self._restart_encroacher_mask(state)
        move_forced = (is_designated & sp_active) | (~active) | projection_forced
        # 하프타임 전이 프레임 — 이 스텝이 끝난 뒤 `_halftime_switch`가 위치·공격방향·속도·공·재개를
        # 통째로 덮으므로 **제출 액션이 post-state를 전혀 설명하지 못한다**. `t`는 스텝당 정확히 1
        # 증가하고 전환 판정은 `state.t == game_duration // 2`(증가 후)이므로, 진입 시점에
        # `(t + 1) == game_duration // 2`로 결정적으로 알 수 있다.
        # move_forced/kick_gated에 섞지 않고 별도 신호로 두는 이유: 이 둘은 `get_avail_actions_array`가
        # 그대로 투영하는 **사전 행동 권한**인데, 하프타임 프레임에도 에이전트는 액션을 제출해야 하고
        # 그 액션은 전환 전 서브스텝 물리에 실제로 작용한다. 막을 것은 권한이 아니라 **라벨**이다.
        halftime_reset = jnp.broadcast_to(
            jnp.bool_(self.halftime)
            & ((state.t + 1) == jnp.int32(self.game_duration // 2)),
            (N,),
        )
        # 종료 상태에는 더 이상 정책 행동이 존재하지 않는다. 최종 State만 얼리고 agency를
        # 열어 두면 terminal duplicate frame이 BC 표본으로 남는다. 모든 이동을 forced,
        # 모든 킥을 gated로 만들어 terminal action mask를 구조적으로 전부 닫는다.
        terminal = self._is_terminal(state)
        move_forced = move_forced | terminal
        kick_gated = kick_gated | terminal
        kick_forced = kick_forced & (~terminal)
        halftime_reset = halftime_reset & (~terminal)
        return {"move_forced": move_forced, "kick_gated": kick_gated,
                "kick_forced": kick_forced, "halftime_reset": halftime_reset}

    def _parameter_consumed_mask(self, entry_state, kick_applied):
        """접촉 파라미터를 실제로 소비했는가 (N,5) — action dim 3:8 대응.

        ``bc_action_mask``의 킥 dim은 **진입 가능성**이고 ``kick_applied``는 그 프레임에
        제출 파라미터가 공 속도를 정했는가다. 이 sidecar는 그 위에서 한 겹 더 좁힌다 —
        같은 인과킥이라도 실행 branch에 따라 읽지 않는 차원이 있기 때문이다. 스로인
        릴리스가 그 경우로, ``throw_vel``은 방향과 발사각만 쓰고 스핀을 버린다.
        스핀 dim을 열어 두면 BC가 버려진 값을 라벨로 학습한다.
        """

        consumed = jnp.broadcast_to(
            kick_applied[:, None], (self.N, ACTION_DIM - 3))
        throw_take = (
            (entry_state.restart_kind == RK_THROWIN)
            & (self.player_indices == entry_state.pending_taker)
        )
        spin_open = ~throw_take[:, None]
        keep = jnp.concatenate(
            [jnp.ones((self.N, 3), bool), jnp.broadcast_to(spin_open, (self.N, 2))],
            axis=1,
        )
        return consumed & keep

    def bc_action_mask(self, info, kick_applied=None):
        """BC/RL 손실용 per-dim 액션 후보 마스크 (N,ACTION_DIM=8) bool.

        True는 downstream task가 학습 대상으로 **쓸 수 있음**, False는 반드시 차단이라는 뜻이다.
        gate/contact task는 아래의 인과 규약으로 이 public 가능성 마스크를 한 번 더 좁힌다.
        `action_agency`의 세 행동 권한 신호와 `halftime_reset`을 8-D 액션 레이아웃
        (_decode / L∞ stretch)에 매핑한다.

        레이아웃: [0]=want_f2b · [1:3]=move(L∞ stretch) · [3:5]=f2b(L∞ stretch) · [5]=launch · [6:8]=spin.
        정책:
          - 이동 dim[1:3]     ← ~move_forced   : 키커 강제워킹·퇴장이면 이동 차단.
          - 킥 dim[0,3:8]     ← ~kick_gated    : 무의미(gated) 프레임만 후보에서 차단. 강제킥이면
                               public 행은 열리지만, 환경이 정한 timing은 정책 결정이 아니므로 downstream
                               gate mask는 ``[:, 0] & ~kick_forced``로 다시 좁힌다. direction·power·
                               launch·spin은 실제 공에 적용된 ``kick_applied`` row에서 학습한다.
                               kick_forced는 지정 키커에게만 참이므로 재개 국면의 후보 언마스킹은 그
                               키커에게만 작용한다. 라이브 오프볼 파라미터는 frame 중 도달 가능성 때문에
                               사전에 열릴 수 있지만 역시 ``kick_applied``와 AND해야 한다.
        키커 강제워킹·재개 위치 투영 프레임은 이동(move_forced)은 통째 차단되지만, 킥 행은 살아난다.
        재개 중 비지정 선수의 `kick_gated`는 킥 dim만 막고 이동은 살린다(자유 결정이라
        학습 대상 — 통째 마스킹하면 필드 전원 이동 감독의 대부분이 소실).

        [BC 킥 라벨 계약] 이 마스크의 킥 dim은 **진입(step-entry) 가능성 게이트**일 뿐 최종
        task mask가 아니다.
        킥의 인과성은 스텝 내부의 확률적 경합에서 갈리므로(승자만 파라미터 적용·굴절/GK캐치는 무효),
        진입 마스크만으론 경합 패자·굴절·GK 자동클레임 킥을 과포함한다.

        ``gate_mask = cols[:, 0] & ~info["kick_forced"]``
        ``contact_parameter_mask = cols[:, 3:8] & info["kick_applied"][:, None] & parameter_consumed_mask``

        ``parameter_consumed_mask``는 스로인 spin처럼 해당 실행 branch가 무시한 차원을
        downstream materializer가 닫는 per-dimension sidecar다. 강제킥은 정상적으로
        ``kick_forced=True``이면서 ``kick_applied=True``다. 따라서 timing gate
        loss에서는 빠지고 실제 소비된 contact parameter loss에는 들어간다. 이동은 진입 마스크 하나로
        충분하지만, contact parameter는 사후 신호와 반드시 AND해야 인과 라벨이 된다.
        (ctrl_lock 등 사후 무효화도 kick_applied로 흡수.)"""
        def require_bool_vector(name, value):
            """Validate one mask before bitwise ops can coerce/broadcast it.

            Shape and dtype are trace-time static, so the same diagnostics are
            available in eager and JIT calls.  In particular, accepting an
            integer mask here makes ``~0``/``~1`` become ``-1``/``-2`` and a
            scalar or length-one optional mask silently broadcasts across the
            whole roster — both produce plausible-shaped but false BC labels.
            """

            try:
                array = jnp.asarray(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} must be a bool array") from exc
            if array.shape != (self.N,):
                raise ValueError(
                    f"{name} must have shape ({self.N},), got {array.shape}"
                )
            if array.dtype != jnp.dtype(jnp.bool_):
                raise TypeError(
                    f"{name} must have bool dtype, got {array.dtype}"
                )
            return array

        move_forced = require_bool_vector(
            "move_forced", info["move_forced"]
        )
        kick_gated = require_bool_vector(
            "kick_gated", info["kick_gated"]
        )
        move_ok = ~move_forced                                  # (N,)
        kick_ok = ~kick_gated                                   # 킥 후보 행: gate/contact task가 인과 mask로 더 좁힘
        # ★진입 게이트는 **스텝 진입 시점 거리**로 reach를 판정하는데, 실제 전이는 _move로 선수를
        # 옮긴 뒤 _kick_gate를 다시 계산한다. 그래서 진입엔 reach 밖이었지만 같은 0.04 s 안에
        # 이동·공 접근으로 도달해 **실제로 찬** 킥이 존재한다(실측: 진입 1.815 m > 임계 1.71 m인데
        # touch=PASS·kick_applied=True). 진입 게이트만 쓰면 그 라벨이 통째로 버려지므로, 실현
        # 인과킥은 게이트를 연다: kick_ok = (~kick_gated) | kick_applied.
        # (물리는 건드리지 않는다 — 마스크만 실현 결과와 정렬한다.)
        if kick_applied is not None:
            kick_applied = require_bool_vector(
                "kick_applied", kick_applied
            )
            kick_ok = kick_ok | kick_applied
        # ★하프타임 전이 프레임은 **전 차원 하드 마스크**이며 위 kick_applied보다 우선한다.
        # 스텝 안에서 킥이 실제로 적용됐더라도 `_halftime_switch`가 공·위치·재개를 전부 덮어써
        # 그 결과가 사라지므로, 이 프레임을 자유 행동 라벨로 세면 안 된다. 진입 시점 마스크만으로는
        # 잡히지 않는다 — 하프타임은 스텝 **끝**에 오기 때문(action_agency의 halftime_reset 참조).
        halftime = info.get("halftime_reset")
        if halftime is not None:
            halftime = require_bool_vector("halftime_reset", halftime)
            live = ~halftime
            move_ok = move_ok & live
            kick_ok = kick_ok & live
        cols = jnp.stack([
            kick_ok,                          # 0  want_f2b
            move_ok, move_ok,                 # 1,2  move (L∞ stretch 2D)
            kick_ok, kick_ok,                 # 3,4  f2b  (L∞ stretch 2D)
            kick_ok,                          # 5    f2b_launch
            kick_ok, kick_ok,                 # 6,7  spin_side, spin_back
        ], axis=1)
        return cols.astype(jnp.bool_)                         # (N, ACTION_DIM=8) bool

    def _kick_applied(self, state, touch_before, force_touch_mask=None):
        """실현 인과킥 마스크(bool[N]) — 이 선수의 제출 킥 파라미터(f2b_dir·pow·launch·spin)가 이번 스텝
        공 속도를 **실제로 결정**했는가. contest에서 params가 공에 적용되는 분기는 free_play(자발/강제
        세트피스 킥) 와 tackle_ok 둘뿐이고, 그 결과가 per-player touch **코드**로 남는다(contest._apply_force2ball).
        즉 ``kick_applied``는 자발 킥 전용 신호가 아니다. 정상 강제 재개 킥도 timing만 환경이 정하고
        제출 parameter가 공에 적용되므로 이 마스크에 포함된다.
        인과 집합 = {PASS, PASS_HEAD, SHOOT, SHOOT_HEAD, DRIBBLE, TACKLE, INTERCEPT}.
        제외: DEFLECT(출구각 랜덤 재추첨 — params 무시)·GK_CATCH/PARRY(무액션 env 클레임)·NONE.
        ※태클/인터셉트도 포함(params 적용됨) — 오펜시브 킥만 원하면 다운스트림에서
        {PASS*,SHOOT*,DRIBBLE}로 좁히면 된다(contest의 is_played 집합).

        ★파울 태클 제외: 태클 파울은 공을 정지(new_vel=0)시키면서도 touch를 TOUCH_TACKLE로 라벨하므로
        (contest: `code = where(foul, TOUCH_TACKLE)` + `new_vel = where(foul, 0)`), 제출 params가 공 속도를
        결정하지 않았는데도 위 집합에 걸려 **거짓 인과킥**이 된다. foul_kind==FOUL_TACKLE인 foul_actor를 뺀다.

        [BC 킥 라벨 계약] 순수 킥 라벨 = bc_action_mask 킥 dim(진입 게이트) ∧ 이 신호(실현 진실).
        진입 마스크만으론 확률적 경합 패자·굴절·GK 자동클레임을 과포함하므로 반드시 AND 한다.

        ★touch_before: **contest 접촉만 세기 위한 서브스텝 스냅샷**(`_apply_force2ball` 직전 값).
        위 인과 집합은 "이 코드들은 contest가 params를 공에 적용한 결과"라는 전제 위에 있는데,
        `ball._ball_body`의 **몸통 트랩도 DRIBBLE/INTERCEPT를 기록**한다. 트랩은 출구속도가
        `trap_velocity_keep · v`인 순수 수동 물리라 제출 params와 무관하므로, 컨트롤 스텝 끝의
        누적 `state.touch`만 보면 트랩이 인과킥으로 오탐된다. 스냅샷을 주면 그 서브스텝에
        **새로 생긴 contest 접촉**만 집계한다(`step_env_array`가 `_ball_body` 이전에 호출·누적).
        호출자는 반드시 스냅샷을 넘겨 몸통 트랩을 인과킥으로 오인하지 않아야 한다."""
        touch = state.touch
        applied = ((touch == TOUCH_PASS) | (touch == TOUCH_PASS_HEAD)
                   | (touch == TOUCH_SHOOT) | (touch == TOUCH_SHOOT_HEAD)
                   | (touch == TOUCH_DRIBBLE) | (touch == TOUCH_TACKLE)
                   | (touch == TOUCH_INTERCEPT))
        applied = applied & (
            (touch != touch_before)
            if force_touch_mask is None
            else jnp.asarray(force_touch_mask, dtype=bool)
        )
        foul_tackle_actor = (
            (state.foul_kind == FOUL_TACKLE)
            & (self.player_indices == state.foul_actor)
        )
        return applied & (~foul_tackle_actor)

    def _record_woodwork_event(self, state, substep_index, woodwork):
        """Store one goal-frame (post/crossbar) rebound in the current frame.

        The frame is not a player contact, so it has no home in the
        ``touch_event_*`` stream, and it cannot share the ``ball_event_*`` slot:
        a ball that comes off the crossbar and crosses the goal line does both
        inside one physics substep.  Its own per-substep buffer is what keeps
        the interesting case — the one the shot-outcome classifier needs — from
        being overwritten by the line ruling that follows it.
        """

        kind, pos, vel_in = woodwork
        struck = kind != jnp.int32(WOODWORK_NONE)
        return state._replace(
            woodwork_kind=state.woodwork_kind.at[substep_index].set(
                kind.astype(jnp.int32)
            ),
            woodwork_pos=state.woodwork_pos.at[substep_index].set(
                jnp.asarray(pos, dtype=state.woodwork_pos.dtype)
            ),
            woodwork_vel_in=state.woodwork_vel_in.at[substep_index].set(
                jnp.asarray(vel_in, dtype=state.woodwork_vel_in.dtype)
            ),
            woodwork_control_t=state.woodwork_control_t.at[substep_index].set(
                jnp.where(struck, state.t, jnp.int32(-1))
            ),
        )

    def _record_touch_event(
        self,
        state,
        substep_index,
        phase,
        event_mask,
        *,
        ball_pos,
        ball_vel_before,
        ball_vel_after,
        toi,
        administrative_stop=False,
    ):
        """Store one ordered physical contact in the current control frame.

        The force-to-ball and passive-body modules each elect at most one actor
        per physics substep.  Their explicit event masks are the causal source;
        ``State.touch`` cannot be used here because it is only a per-player
        accumulator and repeated equal-code contacts leave it unchanged.

        ``phase`` is a static ``TOUCH_EVENT_*`` index.  The fixed
        ``(decimation, 2)`` layout avoids a dynamic-length JAX buffer while its
        row-major order is exactly the runtime order: force, then body, for each
        successive physics substep.  Contact kinematics are sampled at the
        module boundary, before later contacts, ball integration, restarts, or
        frame-boundary projections can overwrite them.  ``administrative_stop``
        excludes a foul whistle's forced zero velocity from physical Δv while
        retaining the actual before/after vectors for auditability.
        """

        mask = jnp.asarray(event_mask, dtype=jnp.bool_)
        actor = jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int32)
        occurred = jnp.any(mask)
        safe_actor = jnp.clip(actor, 0, self.N - 1)
        recorded_actor = jnp.where(
            occurred, safe_actor, jnp.int32(NO_PLAYER)
        ).astype(jnp.int32)
        recorded_code = jnp.where(
            occurred, state.touch[safe_actor], jnp.int32(TOUCH_NONE)
        ).astype(jnp.int32)
        recorded_player_id = jnp.where(
            occurred, state.player_id[safe_actor], jnp.int32(NO_PLAYER)
        ).astype(jnp.int32)
        pos = jnp.asarray(ball_pos, dtype=state.ball_pos.dtype)
        vel_before = jnp.asarray(
            ball_vel_before, dtype=state.ball_vel.dtype
        )
        vel_after = jnp.asarray(
            ball_vel_after, dtype=state.ball_vel.dtype
        )
        contact_toi = jnp.clip(
            jnp.asarray(toi, dtype=state.ball_pos.dtype), 0.0, 1.0
        )
        physical_impulse = jnp.where(
            jnp.asarray(administrative_stop, dtype=jnp.bool_),
            jnp.asarray(0.0, dtype=state.ball_vel.dtype),
            jnp.linalg.norm(vel_after - vel_before),
        )
        zero_pos = jnp.zeros_like(state.ball_pos)
        zero_vel = jnp.zeros_like(state.ball_vel)
        return state._replace(
            touch_event_actor=state.touch_event_actor.at[
                substep_index, phase
            ].set(recorded_actor),
            touch_event_code=state.touch_event_code.at[
                substep_index, phase
            ].set(recorded_code),
            touch_event_player_id=state.touch_event_player_id.at[
                substep_index, phase
            ].set(recorded_player_id),
            touch_event_control_t=state.touch_event_control_t.at[
                substep_index, phase
            ].set(jnp.where(occurred, state.t, jnp.int32(-1))),
            touch_event_toi=state.touch_event_toi.at[
                substep_index, phase
            ].set(jnp.where(occurred, contact_toi, 0.0)),
            touch_event_ball_pos=state.touch_event_ball_pos.at[
                substep_index, phase
            ].set(jnp.where(occurred, pos, zero_pos)),
            touch_event_ball_vel_before=state.touch_event_ball_vel_before.at[
                substep_index, phase
            ].set(jnp.where(occurred, vel_before, zero_vel)),
            touch_event_ball_vel_after=state.touch_event_ball_vel_after.at[
                substep_index, phase
            ].set(jnp.where(occurred, vel_after, zero_vel)),
            touch_event_impulse=state.touch_event_impulse.at[
                substep_index, phase
            ].set(jnp.where(occurred, physical_impulse, 0.0)),
        )

    def step_env(self, key, state, actions):
        """JaxMARL 규약 dict 어댑터 — 순수 전이(auto-reset 없음). 계산은 step_env_array에 위임.
        키는 __init__에서 만든 영구 slot key(``slot_0`` …)를 재사용한다. person id는 교체 시
        바뀌므로 dict key로 쓰지 않고 ``state.player_id``/``info["player_id"]``로 제공한다.
        ※핫루프(데이터 생성·학습)에선 이 dict 경로 대신 배열 경로 step_env_array + jit/vmap을 쓸 것
        (dict 조립은 파이썬 오버헤드라 jit 밖에서 반복 호출하면 병목)."""
        act_arr = jnp.stack([actions[a] for a in self._agent_keys], axis=0)
        obs_arr, state, reward_arr, done_all, info = self.step_env_array(key, state, act_arr)
        obs = {a: obs_arr[i] for i, a in enumerate(self._agent_keys)}
        reward = {a: reward_arr[i] for i, a in enumerate(self._agent_keys)}
        done = {a: done_all for a in self._agent_keys}
        done["__all__"] = done_all
        return obs, state, reward, done, info

    def _split_decision_params(self, decision_params):
        """:class:`manager.DecisionParams`를 축별로 갈라 검증한다.

        결정자와 파라미터는 **짝이 맞아야** 한다. 파라미터를 받는 결정자에게 아무것도 주지
        않으면 호출이 그대로 실패하거나 조용히 fallback으로 떨어지고 — 그 순간 학습 정책은
        실행되지 않는데 롤아웃은 정상으로 보인다 — 2인자 결정자에게 파라미터를 주면 그
        가중치는 아무 데도 쓰이지 않는다. 둘 다 조용히 넘어가면 안 되는 실수라 여기서 막는다.
        """

        if decision_params is None:
            decision_params = self._default_decision_params
        if decision_params is None:
            manager_params = taker_params = None
        elif isinstance(decision_params, manager_module.DecisionParams):
            manager_params = decision_params.manager
            taker_params = decision_params.taker
        else:
            raise TypeError(
                "decision_params must be a manager.DecisionParams, got "
                f"{type(decision_params).__name__}")
        for axis, params, takes in (
            ("manager", manager_params, self._manager_takes_params),
            ("restart taker", taker_params, self._taker_takes_params),
        ):
            if takes and params is None:
                raise ValueError(
                    f"the {axis} decider uses the (params, view, key) protocol "
                    "but neither this call nor the constructor supplied its "
                    "parameters")
            if params is not None and not takes:
                raise ValueError(
                    f"decision_params carries {axis} parameters, but that decider "
                    "uses the (view, key) protocol and would ignore them")
        return manager_params, taker_params

    def _validate_step_command(self, command):
        """Validate only static PyTree structure, shape, and dtype.

        Data-dependent legality belongs to the JAX transition. This method
        never converts a value to NumPy or branches on a command leaf, so it is
        safe when ``command`` contains JIT/vmap tracers.
        """

        if not isinstance(command, StepCommand):
            raise TypeError(
                f"command must be StepCommand, got {type(command).__name__}"
            )

        def require(name, value, shape, dtype):
            try:
                array = jnp.asarray(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} must be an array") from exc
            if array.shape != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got {array.shape}"
                )
            expected = jnp.dtype(dtype)
            if array.dtype != expected:
                raise TypeError(
                    f"{name} must have dtype {expected}, got {array.dtype}"
                )
            return array

        actions = jnp.asarray(command.player_actions)
        if actions.shape != (self.N, ACTION_DIM):
            raise ValueError(
                "command.player_actions must have shape "
                f"({self.N}, {ACTION_DIM}), got {actions.shape}"
            )
        if not jnp.issubdtype(actions.dtype, jnp.floating):
            raise TypeError(
                "command.player_actions must have floating dtype, "
                f"got {actions.dtype}"
            )

        substitution_shape = (
            TEAM_COUNT, self.max_simultaneous_substitutions
        )
        substitution_requested = require(
            "command.substitutions.requested",
            command.substitutions.requested,
            substitution_shape,
            jnp.bool_,
        )
        substitution_out = require(
            "command.substitutions.out_slot",
            command.substitutions.out_slot,
            substitution_shape,
            jnp.int32,
        )
        substitution_bench = require(
            "command.substitutions.bench_index",
            command.substitutions.bench_index,
            substitution_shape,
            jnp.int32,
        )
        formation_requested = require(
            "command.formations.requested",
            command.formations.requested,
            (TEAM_COUNT,),
            jnp.bool_,
        )
        formation_layout = require(
            "command.formations.layout_index",
            command.formations.layout_index,
            (TEAM_COUNT,),
            jnp.int32,
        )
        taker_requested = require(
            "command.set_piece_takers.requested",
            command.set_piece_takers.requested,
            (TEAM_COUNT, RESTART_COUNT),
            jnp.bool_,
        )
        taker_slot = require(
            "command.set_piece_takers.player_slot",
            command.set_piece_takers.player_slot,
            (TEAM_COUNT, RESTART_COUNT),
            jnp.int32,
        )
        return (
            actions,
            substitution_requested,
            substitution_out,
            substitution_bench,
            formation_requested,
            formation_layout,
            taker_requested,
            taker_slot,
        )

    def empty_command(self):
        """Return the canonical no-override command for this environment.

        Command widths are derived from the constructed environment, keeping
        roster, simultaneous-substitution, action, and restart vocab shapes in
        one place instead of requiring callers to duplicate engine constants.
        """

        return StepCommand(
            player_actions=jnp.zeros((self.N, ACTION_DIM), jnp.float32),
            substitutions=SubstitutionCommand.empty(
                self.max_simultaneous_substitutions
            ),
            formations=FormationCommand.empty(),
            set_piece_takers=SetPieceTakerCommand.empty(RESTART_COUNT),
        )

    def empty_randomness_control(self) -> RandomnessControl:
        """Return the fixed-shape no-override control for this environment.

        Normal training should omit the privileged control entirely.  Reconstruction and
        conditional-native data generation use this builder so event and decimation widths never
        need to be duplicated outside the environment.
        """

        return RandomnessControl.empty(self.timebase.decimation)

    def _apply_native_entry_taker_override(self, state, entry_done):
        """Apply a matching command to an already-observed restart.

        New restarts consult the native table at their existing designation
        sites. An entry restart already has a valid ``pending_taker``, so the
        broken-reference repair path intentionally leaves it alone. Route an
        explicit re-designation through the same selector and legality gate,
        then preserve the current taker when the proposal is rejected.

        The caller invokes this only while ``_native_taker_command`` is
        installed. ``requested`` and ``accepted`` are scalar event facts used
        later to retain an exact trace even if this restart is consumed during
        the physics scan.
        """

        command = self._native_taker_command
        team = jnp.asarray(state.restart_team, jnp.int32)
        kind = jnp.asarray(state.restart_kind, jnp.int32)
        addressable = (
            (team >= TEAM_0)
            & (team <= TEAM_1)
            & (kind >= RK_KICKOFF)
            & (kind < RESTART_COUNT)
        )
        safe_team = jnp.clip(team, TEAM_0, TEAM_1)
        safe_kind = jnp.clip(kind, RK_NONE, RESTART_COUNT - 1)
        requested = (
            (~jnp.asarray(entry_done, jnp.bool_))
            & restart_timer_active(state.restart_t)
            & addressable
            & command.requested[safe_team, safe_kind]
        )
        proposed = command.player_slot[safe_team, safe_kind].astype(jnp.int32)
        selected = self._designate_taker_when(
            requested,
            state,
            state.ball_pos[:DIM_Z],
            team,
            (kind == RK_GOALKICK) | (kind == RK_GK_HOLD),
            kind,
            state.restart_indirect,
        ).astype(jnp.int32)
        # An illegal proposal resolves to the selector's legal fallback, but
        # that fallback must not silently replace an already-observed valid
        # internal designation. Reuse the selector's exact proposal gate to
        # distinguish approval from fallback, including the no-candidate case.
        accepted = requested & self._taker_proposal_legal(
            state, proposed, team, kind
        )
        state = state._replace(
            pending_taker=jnp.where(
                accepted, selected, state.pending_taker
            ).astype(jnp.int32)
        )
        return state, requested, accepted

    def _native_taker_trace(
        self,
        entry_state,
        state,
        command,
        restart_opened,
        entry_taker_repaired,
        entry_taker_override_requested,
        entry_taker_override_accepted,
        entry_done,
    ):
        """Project frame-end taker adjudication to fixed ``(2, R)`` arrays.

        Decision codes are the numeric values of public ``CommandReason``.
        Only the team/kind whose designation was created or repaired in this
        frame can be accepted. Other submitted table cells remain future
        proposals and report ``NO_MATCHING_RESTART``.
        """

        requested = command.requested
        proposed = command.player_slot
        safe = jnp.clip(proposed, 0, self.N - 1)
        in_range = (proposed >= 0) & (proposed < self.N)
        active = in_range & state.active_player[safe]

        teams = jnp.arange(TEAM_COUNT, dtype=jnp.int32)[:, None]
        kinds = jnp.arange(RESTART_COUNT, dtype=jnp.int32)[None, :]
        correct_team = active & (state.team_id[safe] == teams)
        role_eligible = correct_team & (
            (kinds != RK_GK_HOLD) | (state.gk_indices[safe] == 1)
        )

        entry_active = restart_timer_active(entry_state.restart_t)
        current_active = restart_timer_active(state.restart_t)
        designation_changed = (
            jnp.asarray(restart_opened, jnp.bool_)
            | jnp.asarray(entry_taker_repaired, jnp.bool_)
            | (~entry_active)
            | (state.restart_kind != entry_state.restart_kind)
            | (state.restart_team != entry_state.restart_team)
            | (state.pending_taker != entry_state.pending_taker)
        )
        designation = current_active & designation_changed & (~entry_done)
        matching = (
            designation
            & (teams == state.restart_team)
            & (kinds == state.restart_kind)
        )
        external_accepted = requested & matching & role_eligible
        external_applied = external_accepted & (
            proposed == state.pending_taker
        )

        # The post-state may already be live after an entry restart was
        # released, and a same-slot re-designation does not change
        # ``pending_taker``. Preserve the entry event independently, using
        # entry-state legality rather than any later substitution or restart.
        entry_matching = (
            jnp.asarray(entry_taker_override_requested, jnp.bool_)
            & (teams == entry_state.restart_team)
            & (kinds == entry_state.restart_kind)
        )
        entry_external_accepted = entry_matching & jnp.asarray(
            entry_taker_override_accepted, jnp.bool_
        )
        external_accepted = external_accepted | entry_external_accepted
        external_applied = external_applied | entry_external_accepted
        matching = matching | entry_matching
        selected = state.pending_taker
        safe_selected = jnp.clip(selected, 0, self.N - 1)
        selected_valid = (
            (selected >= 0)
            & (selected < self.N)
            & state.active_player[safe_selected]
            & (state.team_id[safe_selected] == state.restart_team)
            & (
                (state.restart_kind != RK_GK_HOLD)
                | (state.gk_indices[safe_selected] == 1)
            )
        )
        internal_applied = matching & (~requested) & selected_valid
        accepted = external_accepted | internal_applied
        applied = external_applied | internal_applied
        reported_proposed = jnp.where(
            internal_applied, selected, proposed
        ).astype(jnp.int32)

        not_requested = jnp.int32(int(CommandReason.NOT_REQUESTED))
        reason = jnp.full(
            (TEAM_COUNT, RESTART_COUNT), not_requested, dtype=jnp.int32
        )
        reason = jnp.where(
            requested,
            jnp.int32(int(CommandReason.NO_MATCHING_RESTART)),
            reason,
        )
        reason = jnp.where(
            requested & (~in_range),
            jnp.int32(int(CommandReason.INVALID_SLOT)),
            reason,
        )
        reason = jnp.where(
            requested & in_range & (~active),
            jnp.int32(int(CommandReason.INACTIVE_PLAYER)),
            reason,
        )
        reason = jnp.where(
            requested & active & (~correct_team),
            jnp.int32(int(CommandReason.WRONG_TEAM)),
            reason,
        )
        reason = jnp.where(
            requested & correct_team & (~role_eligible),
            jnp.int32(int(CommandReason.INELIGIBLE_TAKER)),
            reason,
        )
        reason = jnp.where(
            accepted,
            jnp.where(
                external_applied,
                jnp.int32(int(CommandReason.APPLIED)),
                jnp.int32(int(CommandReason.ACCEPTED_PENDING)),
            ),
            reason,
        )
        reason = jnp.where(
            matching & (~requested),
            jnp.int32(int(CommandReason.INTERNAL_FALLBACK)),
            reason,
        )

        # Final-state validation above is correct for designations opened
        # inside the scan. Override the one entry cell with the legality that
        # was actually adjudicated before physics, so a consumed restart (or a
        # later roster mutation) cannot rewrite its result.
        entry_active_player = in_range & entry_state.active_player[safe]
        entry_correct_team = entry_active_player & (
            entry_state.team_id[safe] == teams
        )
        entry_role_eligible = entry_correct_team & (
            (kinds != RK_GK_HOLD) | (entry_state.gk_indices[safe] == 1)
        )
        entry_reason = jnp.full(
            (TEAM_COUNT, RESTART_COUNT),
            jnp.int32(int(CommandReason.NO_MATCHING_RESTART)),
            dtype=jnp.int32,
        )
        entry_reason = jnp.where(
            ~in_range,
            jnp.int32(int(CommandReason.INVALID_SLOT)),
            entry_reason,
        )
        entry_reason = jnp.where(
            in_range & (~entry_active_player),
            jnp.int32(int(CommandReason.INACTIVE_PLAYER)),
            entry_reason,
        )
        entry_reason = jnp.where(
            entry_active_player & (~entry_correct_team),
            jnp.int32(int(CommandReason.WRONG_TEAM)),
            entry_reason,
        )
        entry_reason = jnp.where(
            entry_correct_team & (~entry_role_eligible),
            jnp.int32(int(CommandReason.INELIGIBLE_TAKER)),
            entry_reason,
        )
        entry_reason = jnp.where(
            entry_external_accepted,
            jnp.int32(int(CommandReason.APPLIED)),
            entry_reason,
        )
        reason = jnp.where(entry_matching, entry_reason, reason)
        return {
            "set_piece_taker_requested": requested,
            "set_piece_taker_accepted": accepted,
            "set_piece_taker_applied": applied,
            "set_piece_taker_proposed_player_slot": reported_proposed,
            "set_piece_taker_decision_code": reason,
        }

    def step_command(
        self,
        key,
        state,
        command,
        *,
        external_substitutions: bool = True,
        external_formations: bool = True,
        external_takers: bool = True,
        collect_substeps=False,
        include_bc_info=True,
        compute_observation=True,
        include_decision_trace=True,
        randomness=None,
    ):
        """Apply one fixed-shape :class:`StepCommand` inside JIT/vmap.

        A true ``requested`` cell overrides only that manager decision. False
        cells retain the configured internal manager/taker, so an empty command
        is transition-identical to ``step_env_array``. The three
        ``external_*`` flags are static capability gates; disabling one masks
        every request on that axis without inspecting command values on host.
        """

        for name, value in (
            ("external_substitutions", external_substitutions),
            ("external_formations", external_formations),
            ("external_takers", external_takers),
        ):
            if type(value) is not bool:
                raise TypeError(
                    f"{name} must be a Python bool, got {type(value).__name__}"
                )

        if randomness is not None:
            randomness = validate_randomness_control(
                randomness, decimation=self.timebase.decimation
            )

        (
            actions,
            substitution_requested,
            substitution_out,
            substitution_bench,
            formation_requested,
            formation_layout,
            taker_requested,
            taker_slot,
        ) = self._validate_step_command(command)
        if not external_substitutions:
            substitution_requested = jnp.zeros_like(substitution_requested)
        if not external_formations:
            formation_requested = jnp.zeros_like(formation_requested)
        if not external_takers:
            taker_requested = jnp.zeros_like(taker_requested)
        taker_command = SetPieceTakerCommand(taker_requested, taker_slot)

        manager_params, taker_params = self._split_decision_params(None)
        previous_params = self._taker_params
        previous_command = self._native_taker_command
        previous_randomness = self._active_randomness_control
        self._taker_params = taker_params
        self._native_taker_command = taker_command
        self._active_randomness_control = randomness
        try:
            return self._step_env_array(
                key,
                state,
                actions,
                substitution=(substitution_out, substitution_bench),
                formation=formation_layout,
                collect_substeps=collect_substeps,
                include_bc_info=include_bc_info,
                compute_observation=compute_observation,
                include_decision_trace=include_decision_trace,
                manager_params=manager_params,
                native_substitution_requested=substitution_requested,
                native_formation_requested=formation_requested,
                randomness=randomness,
            )
        finally:
            self._active_randomness_control = previous_randomness
            self._native_taker_command = previous_command
            self._taker_params = previous_params

    def step_env_array(self, key, state, act_arr, forced_winner=None, forced_freeplay=None,
                       substitution=None,
                       formation=None,
                       forced_gk_touch=None, suppress_charge=None, suppress_body=None,
                       suppress_restart=None, inject_charge=None, collect_substeps=False,
                       include_bc_info=True, compute_observation=True,
                       include_decision_trace=True, decision_params=None,
                       randomness=None):
        """결정자 파라미터를 이 호출 범위에 실은 뒤 본체로 넘긴다.

        키커 파라미터를 인자로 꿰지 않는 이유는 :class:`manager.DecisionParams`에 적어 두었다 —
        지정 호출이 서브스텝 스캔 안쪽에 흩어져 있어서다. 값은 추적 중에만 실려 있고
        ``finally``가 되돌리므로, 예외로 빠져나가도 객체에 트레이서가 남지 않는다.
        """

        manager_params, taker_params = self._split_decision_params(decision_params)
        previous = self._taker_params
        previous_randomness = self._active_randomness_control
        self._taker_params = taker_params
        if randomness is not None:
            randomness = validate_randomness_control(
                randomness, decimation=self.timebase.decimation
            )
        self._active_randomness_control = randomness
        try:
            return self._step_env_array(
                key, state, act_arr,
                forced_winner=forced_winner, forced_freeplay=forced_freeplay,
                substitution=substitution, formation=formation,
                forced_gk_touch=forced_gk_touch,
                suppress_charge=suppress_charge, suppress_body=suppress_body,
                suppress_restart=suppress_restart, inject_charge=inject_charge,
                collect_substeps=collect_substeps,
                include_bc_info=include_bc_info,
                compute_observation=compute_observation,
                include_decision_trace=include_decision_trace,
                manager_params=manager_params,
                randomness=randomness,
            )
        finally:
            self._active_randomness_control = previous_randomness
            self._taker_params = previous

    def _step_env_array(self, key, state, act_arr, forced_winner=None, forced_freeplay=None,
                       substitution=None,
                       formation=None,
                       forced_gk_touch=None, suppress_charge=None, suppress_body=None,
                       suppress_restart=None, inject_charge=None, collect_substeps=False,
                       include_bc_info=True, compute_observation=True,
                       include_decision_trace=True, manager_params=None,
                       native_substitution_requested=None,
                       native_formation_requested=None,
                       randomness=None):
        """step 계산 본체(단일 진실원천) — (N,·) 배열 경로. 반환 (obs (N,·), State, reward (N,),
        done_all 스칼라, info). 종료는 시간제한 또는 한 팀의 최소 인원 미달이며 done은 경기 공용
        스칼라 하나다.

        서브스텝 파이프라인: 키커이동 → 이동 → 차징파울 → (킥 게이트 재계산) → 경합 승자 →
        force2ball → 몸통충돌 → 스로인 재터치 → 오프사이드 → 공 자유물리 → 이벤트.

        forced_winner: reconstruct용 경합 승자 주입. (decimation,) int 배열 — 서브스텝별로
        ≥0=그 선수 승자 / -1=무승자 / -2=정상 샘플. None이면 전 서브스텝 샘플(포워드 불변).
        pin도 합법 후보 게이트(도달·쿨다운·게이트)를 통과해야 성립 — 게이트 밖 pin은 불발(-1).
        forced_freeplay: reconstruct용 분기 pin. (decimation,) bool 배열 — True인 서브스텝은
        force2ball의 opp_poss를 젖혀 관측 터치를 free_play(결정론 킥)로 강제(상태 무변경 pin).
        None이면 전부 False(포워드 불변).
        forced_gk_touch: reconstruct용 GK catch/parry 결과 pin. (decimation,) signed-int 배열 —
        TOUCH_NONE=정상 샘플 / TOUCH_GK_CATCH=캐치 / TOUCH_PARRY=parry. 도달·박스·백패스 같은
        확정 게이트는 그대로이며, 잘못된 traced pin은 TOUCH_NONE으로 닫고 info에 표시한다.
        suppress_charge / suppress_body: reconstruct용 추첨 봉쇄 pin(스칼라 bool, 스텝 전체 적용).
        True면 각각 차징 파울 추첨·공-몸통 hit 추첨을 '불발'로 고정 — 관측에 없는 확률 이벤트가
        복원 창을 탈선시키지 않게. 추첨은 소비되므로 RNG 열 불변. None이면 False(포워드 불변).
        suppress_restart: reconstruct용 경계 이벤트 봉쇄 pin(스칼라 bool, 스텝 전체 적용).
        True면 시뮬 공의 짧은 드리프트가 골·아웃 재개를 만들어 복원 창을 끊는 것만 막는다.
        선수·공 물리와 기존 재개 카운트다운은 계속 진행하며 None은 False와 같다.
        inject_charge: suppress_charge의 짝인 파울 주입 pin(dict 또는 None) — 관측에 있는데
        추상 모델의 게이트가 만들 수 없는 파울(핸드볼·오프볼·공격 측)을 확정 발생시킨다.
        ``{"actor", "victim", "pos"}`` 필수, ``"kind"``와 ``"discipline"`` 선택.
        discipline은 -1=상황별 추첨, 0=카드 없음, 1=옐로, 2=직접 레드다.
        actor<0이면 주입하지 않는다.
        계약과 근거는 ``Fouls._charge_foul``의 독스트링에 있다. None이면 포워드 불변.
        include_bc_info: 정적 bool. False면 ``action_agency``·BC 마스크·kick_applied 산출과
        info 삽입을 생략한다. RL처럼 이 라벨을 쓰지 않는 핫루프용.
        compute_observation: 정적 bool. False면 관측 대신 shape=(N,0) 빈 배열을 반환한다.
        렌더/물리 검증처럼 state만 필요한 롤아웃에서 O(N²) 관측 조립을 생략하는 경로다.
        """
        key = _validate_prng_key(key)
        if randomness is not None:
            randomness = validate_randomness_control(
                randomness, decimation=self.timebase.decimation
            )
        # Static graph-shape flags must be real Python bools.  Accepting 0/1,
        # numpy scalars or strings silently creates a different JIT program (or
        # lets truthiness choose a branch the caller did not request).
        for name, value in (
            ("collect_substeps", collect_substeps),
            ("include_bc_info", include_bc_info),
            ("compute_observation", compute_observation),
            ("include_decision_trace", include_decision_trace),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be a Python bool, got {type(value).__name__}")

        def strict_bool_scalar(name, value):
            """Validate a dynamic scalar pin without Python truth coercion."""

            if value is None:
                return jnp.bool_(False)
            try:
                array = jnp.asarray(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} must be a scalar bool") from exc
            if array.shape != ():
                raise ValueError(f"{name} must be a scalar bool, got shape {array.shape}")
            if array.dtype != jnp.dtype(jnp.bool_):
                raise TypeError(f"{name} must have bool dtype, got {array.dtype}")
            return array

        suppress_charge = strict_bool_scalar("suppress_charge", suppress_charge)
        suppress_body = strict_bool_scalar("suppress_body", suppress_body)
        suppress_restart = strict_bool_scalar("suppress_restart", suppress_restart)

        if inject_charge is not None:
            if not isinstance(inject_charge, Mapping):
                raise TypeError(
                    "inject_charge must be a mapping with actor/victim/pos, got "
                    f"{type(inject_charge).__name__}"
                )
            missing = {"actor", "victim", "pos"} - set(inject_charge)
            if missing:
                raise ValueError(
                    f"inject_charge missing required keys: {sorted(missing)}"
                )
            unknown = set(inject_charge) - {
                "actor", "victim", "pos", "kind", "discipline"
            }
            if unknown:
                raise ValueError(
                    f"inject_charge has unknown keys: {sorted(unknown)}"
                )
            pos_arr = jnp.asarray(inject_charge["pos"])
            if pos_arr.shape != (DIM_Z,):
                raise ValueError(
                    f"inject_charge['pos'] must have shape ({DIM_Z},), got {pos_arr.shape}"
                )
            # 값도 형태와 같은 자리에서 본다. 주입은 **관측된 사실**을 재현하는 pin이라
            # 잘못된 값이 조용히 통과하면 그대로 State와 BC 라벨이 된다 — 실측으로
            # victim=999·kind=999가 저장됐고 pos=NaN은 중앙 프리킥으로 바뀌었다.
            # 호스트 값일 때만 여기서 막을 수 있고, 트레이서는 ``_charge_foul``의
            # 런타임 게이트가 같은 조건으로 fail-closed한다(두 층이 같은 규칙이다).
            for field, upper in (("actor", self.N), ("victim", self.N)):
                try:
                    host = np.asarray(inject_charge[field])
                except (TypeError, ValueError):
                    continue                     # 트레이서 — 런타임 게이트가 받는다.
                if not np.issubdtype(host.dtype, np.integer):
                    raise ValueError(
                        f"inject_charge['{field}'] must be an integer slot, got "
                        f"dtype {host.dtype}")
                if host.shape != ():
                    raise ValueError(
                        f"inject_charge['{field}'] must be a scalar, got shape "
                        f"{host.shape}")
                # ``NO_PLAYER``는 두 자리 모두에서 뜻이 있는 값이다 — actor면 '주입하지
                # 않음', victim이면 '피해자 없음'(핸드볼·오프볼처럼 상대가 없는 반칙).
                # 그 하나만 허용하고 나머지 음수는 거부한다.
                if not NO_PLAYER <= int(host) < upper:
                    raise ValueError(
                        f"inject_charge['{field}'] must lie in "
                        f"[{NO_PLAYER}, {upper}), got {int(host)}")
            if "kind" in inject_charge:
                try:
                    host_kind = np.asarray(inject_charge["kind"])
                except (TypeError, ValueError):
                    host_kind = None
                if host_kind is not None:
                    if not np.issubdtype(host_kind.dtype, np.integer):
                        raise ValueError(
                            "inject_charge['kind'] must be an integer foul code, "
                            f"got dtype {host_kind.dtype}")
                    if int(host_kind) not in INJECTABLE_FOUL_KINDS:
                        raise ValueError(
                            "inject_charge['kind'] must be one of "
                            f"{sorted(INJECTABLE_FOUL_KINDS)} "
                            f"(FOUL_TACKLE/FOUL_CHARGE), got {int(host_kind)}")
            if "discipline" in inject_charge:
                try:
                    host_discipline = np.asarray(inject_charge["discipline"])
                except (TypeError, ValueError):
                    host_discipline = None
                if host_discipline is not None:
                    if (
                        host_discipline.shape != ()
                        or not np.issubdtype(host_discipline.dtype, np.signedinteger)
                    ):
                        raise ValueError(
                            "inject_charge['discipline'] must be a signed-integer "
                            "scalar"
                        )
                    if int(host_discipline) not in DISCIPLINE_OUTCOMES:
                        raise ValueError(
                            "inject_charge['discipline'] must be one of "
                            f"{sorted(DISCIPLINE_OUTCOMES)}, got "
                            f"{int(host_discipline)}"
                        )
            try:
                host_pos = np.asarray(inject_charge["pos"], np.float64)
            except (TypeError, ValueError):
                host_pos = None
            if host_pos is not None and not np.isfinite(host_pos).all():
                raise ValueError(
                    f"inject_charge['pos'] must be finite, got {host_pos.tolist()}")

        try:
            act_arr = jnp.asarray(act_arr)
        except (TypeError, ValueError) as exc:
            raise TypeError("act_arr must be a real floating array") from exc
        expected_shape = (self.N, ACTION_DIM)
        if act_arr.shape != expected_shape:
            raise ValueError(f"act_arr must have shape {expected_shape}, got {act_arr.shape}")
        if not jnp.issubdtype(act_arr.dtype, jnp.floating):
            raise TypeError(
                "act_arr must have floating dtype; bool/integer/complex actions "
                f"are not valid Box actions, got {act_arr.dtype}"
            )
        action_invalid = jnp.any(~jnp.isfinite(act_arr))
        # A closed-over concrete JAX action may itself not be a Tracer while
        # this reduction *is* traced by an enclosing ``jax.jit``.  Branch on
        # the value that would cross into Python, otherwise a perfectly valid
        # jitted step raises TracerArrayConversionError during tracing.
        if (not isinstance(action_invalid, jax.core.Tracer)
                and bool(np.asarray(action_invalid))):
            raise ValueError("act_arr must contain only finite values")
        # Traced inputs cannot raise data-dependent Python exceptions.  Fail
        # closed by replacing invalid elements with no-op commands and expose
        # the fault in info rather than letting NaNs poison the whole State.
        act_arr = jnp.where(jnp.isfinite(act_arr), act_arr, 0.0)
        # 종료된 상태에서 다시 step하면 전이는 항등이어야 한다. auto-reset이 없는 순수 전이라
        # 종료 준수는 호출자 책임이지만, 벡터 환경이나 수집기가 done 마스크를 잘못 적용하면
        # terminal 이후 프레임(t 증가, stamina 감소, role_pos_count 누적)이 데이터에 섞인다.
        # 여기서 얼려 두면 그 오염이 구조적으로 불가능해진다.
        entry_state = state
        entry_done = self._is_terminal(state)
        # post-state만 보면 이번 frame에 소비된 재개가 restart_t=0이 되어, 데드볼 킥의
        # 행정적 공 이동에 dense 전진/소유 보상이 샌다. 진입 인플레이 여부를 인과적으로 보존한다.
        reward_entry_inplay = (
            (state.ball_state == BALL_ALIVE)
            & (~restart_timer_active(state.restart_t))
        )
        # A decision made while a restart is active belongs to that restart
        # phase for the whole control frame.  Once the designated taker releases
        # in an early physics tick, do not let another player's already-held
        # command become a second voluntary kick before the policy has observed
        # the new live-ball state.  This mirrors ``action_agency``/public
        # availability, which intentionally opens only the designated taker on
        # an entry-restart frame.
        entry_restart_active = restart_timer_active(state.restart_t)
        (
            want_f2b, mv_dir, mv_pow,
            f2b_dir, f2b_pow, f2b_launch,
            spin_side, spin_back
        ) = self._decode(act_arr, state.attack_dir)

        e_cfg = self.e_cfg
        decimation = self.timebase.decimation
        forced_winner_invalid = jnp.bool_(False)
        forced_gk_touch_invalid = jnp.bool_(False)
        if forced_winner is None:
            forced_winner = jnp.full(
                (decimation,), SAMPLED_WINNER, dtype=jnp.int32
            )   # 전 서브스텝 샘플
        else:
            try:
                winner_array = jnp.asarray(forced_winner)
            except (TypeError, ValueError) as exc:
                raise TypeError("forced_winner must be a signed-integer array") from exc
            if winner_array.shape != (decimation,):
                raise ValueError(
                    "forced_winner must have shape "
                    f"({decimation},), got {winner_array.shape}"
                )
            if not jnp.issubdtype(winner_array.dtype, jnp.signedinteger):
                raise TypeError(
                    "forced_winner must have signed-integer dtype (bool/float/unsigned "
                    f"are not pins), got {winner_array.dtype}"
                )
            # Check host-visible values before narrowing to the runtime int32
            # representation, so an oversized int64 cannot wrap into a legal slot.
            if not isinstance(winner_array, jax.core.Tracer):
                values = np.asarray(forced_winner)
                legal = ((values == SAMPLED_WINNER) | (values == NO_PLAYER)
                         | ((values >= 0) & (values < self.N)))
                if not bool(legal.all()):
                    raise ValueError(
                        "forced_winner entries must be a slot in "
                        f"[0, {self.N}), NO_PLAYER({NO_PLAYER}) or "
                        f"SAMPLED_WINNER({SAMPLED_WINNER}); got "
                        f"{values[~legal].tolist()}"
                    )
            legal_jax = (
                (winner_array == SAMPLED_WINNER)
                | (winner_array == NO_PLAYER)
                | ((winner_array >= 0) & (winner_array < self.N))
            )
            forced_winner_invalid = jnp.any(~legal_jax)
            # Sanitize in the original integer width before narrowing.  In
            # x64/traced callers, ``2**32 + slot`` must not wrap into a legal
            # int32 pin.
            forced_winner = jnp.where(
                legal_jax, winner_array, jnp.asarray(NO_PLAYER, winner_array.dtype)
            ).astype(jnp.int32)
            # 값이 보이는 eager 호출은 일반 ValueError로 거절한다. traced/JIT 입력은 Python
            # 예외를 순수 XLA graph에서 낼 수 없으므로 안전하게 무효 pin으로 처리하되,
            # info["forced_winner_invalid"]을 반드시 세운다. host callback은 vmap이 cond 양쪽을
            # 평가할 때 정상 batch까지 직렬화하므로 reconstruction 핫패스에는 두지 않는다.
        if forced_freeplay is None:
            forced_freeplay = jnp.zeros((decimation,), dtype=bool)         # 전부 off(포워드 불변)
        else:
            try:
                forced_freeplay = jnp.asarray(forced_freeplay)
            except (TypeError, ValueError) as exc:
                raise TypeError("forced_freeplay must be a bool array") from exc
            if forced_freeplay.shape != (decimation,):
                raise ValueError(
                    "forced_freeplay must have shape "
                    f"({decimation},), got {forced_freeplay.shape}"
                )
            if forced_freeplay.dtype != jnp.dtype(jnp.bool_):
                raise TypeError(
                    f"forced_freeplay must have bool dtype, got {forced_freeplay.dtype}"
                )
        gk_touch_pin_requested = forced_gk_touch is not None
        if forced_gk_touch is None:
            forced_gk_touch = jnp.full(
                (decimation,), TOUCH_NONE, dtype=jnp.int32
            )
        else:
            try:
                gk_touch_array = jnp.asarray(forced_gk_touch)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "forced_gk_touch must be a signed-integer array"
                ) from exc
            if gk_touch_array.shape != (decimation,):
                raise ValueError(
                    "forced_gk_touch must have shape "
                    f"({decimation},), got {gk_touch_array.shape}"
                )
            if not jnp.issubdtype(gk_touch_array.dtype, jnp.signedinteger):
                raise TypeError(
                    "forced_gk_touch must have signed-integer dtype "
                    f"(bool/float/unsigned are not pins), got {gk_touch_array.dtype}"
                )
            if not isinstance(gk_touch_array, jax.core.Tracer):
                values = np.asarray(forced_gk_touch)
                legal = (
                    (values == TOUCH_NONE)
                    | (values == TOUCH_GK_CATCH)
                    | (values == TOUCH_PARRY)
                )
                if not bool(legal.all()):
                    raise ValueError(
                        "forced_gk_touch entries must be TOUCH_NONE, "
                        f"TOUCH_GK_CATCH or TOUCH_PARRY; got {values[~legal].tolist()}"
                    )
            legal_jax = (
                (gk_touch_array == TOUCH_NONE)
                | (gk_touch_array == TOUCH_GK_CATCH)
                | (gk_touch_array == TOUCH_PARRY)
            )
            forced_gk_touch_invalid = jnp.any(~legal_jax)
            forced_gk_touch = jnp.where(
                legal_jax,
                gk_touch_array,
                jnp.asarray(TOUCH_NONE, gk_touch_array.dtype),
            ).astype(jnp.int32)
        entry_taker_override_requested = jnp.bool_(False)
        entry_taker_override_accepted = jnp.bool_(False)
        if self._native_taker_command is not None:
            (
                state,
                entry_taker_override_requested,
                entry_taker_override_accepted,
            ) = self._apply_native_entry_taker_override(state, entry_done)
        decision_state = state
        if include_bc_info:
            # A native taker selection is part of this decision. Agency must
            # therefore describe the selected player, not the stale internal
            # designation in the pre-command observation.
            agency = self.action_agency(decision_state)  # 명령 적용 진입 상태와 액션 조건 정렬
            # bc_mask는 스캔이 끝나 kick_applied가 확정된 뒤에 만든다(bc_action_mask 참조).
        # Repair an externally injected inactive/wrong-team pending slot before
        # referee projection.  The replacement was not present in the policy's
        # entry observation, so the scalar also erects a full decision-frame
        # boundary below: no forced walk, timer spend, or release until the
        # next observation has exposed the legitimate taker.
        state, entry_taker_repaired = self._repair_broken_pending_taker(state)
        # 직접 거리 위반자는 frame 동안 정책 이동을 동결한다. 충돌 해결 때문에 같이 밀린
        # 합법 선수는 위치 개입을 BC에서 마스킹하되 이후 정책 이동은 그대로 반영한다.
        restart_position_frozen = self._restart_encroacher_mask(state)
        state, restart_position_forced = self._project_restart_positions(state)
        # ``touch`` holds the per-player final code.  The parallel
        # fixed-layout buffers retain every force/body contact in physical
        # substep order so downstream event extraction never has to infer order
        # from player slot indices.
        state = state._replace(
            touch=jnp.zeros(self.N, dtype=jnp.int32),
            touch_event_actor=jnp.full_like(
                state.touch_event_actor, jnp.int32(NO_PLAYER)
            ),
            touch_event_code=jnp.full_like(
                state.touch_event_code, jnp.int32(TOUCH_NONE)
            ),
            touch_event_player_id=jnp.full_like(
                state.touch_event_player_id, jnp.int32(NO_PLAYER)
            ),
            touch_event_control_t=jnp.full_like(
                state.touch_event_control_t, jnp.int32(-1)
            ),
            touch_event_toi=jnp.zeros_like(state.touch_event_toi),
            touch_event_ball_pos=jnp.zeros_like(state.touch_event_ball_pos),
            touch_event_ball_vel_before=jnp.zeros_like(
                state.touch_event_ball_vel_before
            ),
            touch_event_ball_vel_after=jnp.zeros_like(
                state.touch_event_ball_vel_after
            ),
            touch_event_impulse=jnp.zeros_like(state.touch_event_impulse),
            ball_event_kind=jnp.full_like(
                state.ball_event_kind, jnp.int32(BALL_EVENT_NONE)
            ),
            ball_event_team=jnp.full_like(
                state.ball_event_team, jnp.int32(NO_TEAM)
            ),
            ball_event_pos=jnp.zeros_like(state.ball_event_pos),
            ball_event_vel=jnp.zeros_like(state.ball_event_vel),
            ball_event_control_t=jnp.full_like(
                state.ball_event_control_t, jnp.int32(-1)
            ),
            woodwork_kind=jnp.full_like(
                state.woodwork_kind, jnp.int32(WOODWORK_NONE)
            ),
            woodwork_pos=jnp.zeros_like(state.woodwork_pos),
            woodwork_vel_in=jnp.zeros_like(state.woodwork_vel_in),
            woodwork_control_t=jnp.full_like(
                state.woodwork_control_t, jnp.int32(-1)
            ),
        )
        ball_x0 = state.ball_pos[DIM_X]          # dense 전진/소유획득 차분 기준(스텝 전)
        poss0 = state.poss_team
        def _kick_gate(st, setup_done, f2b_attempted, restart_opened):
            """현재 상태에서 f2b 킥 허용 마스크(N,) 계산 — 오픈플레이=alive, 재개=재개팀 & setup 완료.
            페널티도 물리 플레이라 포함(setup 후 키커가 실제 킥). in_reach가 사실상 키커로 한정한다.

            세트피스 시간 만료 강제: 세트업이 끝난 뒤에도 키커가 끝내 안 차서 카운트다운이 이번
            서브스텝에 소진(restart_t==1→0)되면, 슛 게이트를 강제로 열어 킥을 발생시킨다 —
            시간초과가 흐지부지 루즈볼로 새는 대신 반드시 인플레이 킥으로 전환된다. 강제 대상은
            지정 키커(pending_taker) 하나뿐이고 in_reach·cooldown을 우회한다(키커는 스폿 뒤에 정렬
            완료). 킥 파라미터(방향·파워·발사각·스핀)는 그대로 에이전트 액션이라 결과는 여전히 역산 가능."""
            reach_context = self._reach_context(st)
            in_reach = reach_context.in_reach
            alive = st.ball_state == BALL_ALIVE
            restart_active = restart_timer_active(st.restart_t)
            # 재개 킥은 지정 키커만 — 팀 단위로 열면 스폿 근처 동료가 세트피스를 가로채거나
            # (페널티는 침범자 본인이 차는 것도 가능) GK 홀드 중인 공을 동료가 차버릴 수 있다.
            restart_kick_ok = (
                (self.player_indices == st.pending_taker)
                & setup_done
                # ★킥은 **의사결정 스텝**의 산물이다. 이 control frame 안에서 막 생긴 재개는
                # 정책이 아직 관측하지 못했으므로 이번 frame에는 찰 수 없다. 이 게이트가 없으면
                # 득점 다음 서브스텝에 킥오프가 즉시 소비돼(decimation=6이면 5/6 확률로 같은
                # frame), 킥 파라미터로 **득점 이전에 제출된 액션**이 쓰인다 — 정책은 킥오프를
                # 한 번도 보지 못한 채 그 킥의 주체가 된다. 한 frame 미루면 다음 스텝 진입
                # 상태가 재개를 담고 arrived=True라 action_agency가 kick_forced로 킥 dim을
                # 열어 준다(오프닝 킥오프가 이미 그렇게 동작한다).
                & (~restart_opened)
            )
            allowed = jnp.where(restart_active, restart_kick_ok, alive)
            # [재터치 원천 차단] 직전 재개 수행자(throw/setpiece_taker, 타 선수 터치 시 해제)는
            # 자발 킥 게이트를 봉쇄 — 사후 파울(FOUL_THROW) 판정 대신 규칙을 구조로 강제.
            # 정확한 서브스텝 합법성은 여기서 막고, 사전 agency는 frame 중
            # 해제 가능성을 보존하려고 더 넓게 연다. BC는 kick_applied와 AND해 정합한다.
            # 몸통 접촉 재터치 등 비킥 경로는 기존 파울 기계가 백스톱으로 잔존.
            retouch_ok = self._retouch_allows_f2b(st)
            # One active force-to-ball contact per player per control step.
            # ``touch`` is cleared immediately before the substep scan, then
            # records the first real contest/body contact.  This replaces the
            # old blanket cooldown debounce, which also starved possessors in
            # later control frames.  Forced restarts intentionally bypass both
            # this touch gate and challenge cooldown below.
            first_contact = st.touch == TOUCH_NONE
            attempt_ready = ~f2b_attempted
            entry_phase_allows = (~entry_restart_active) | restart_active
            voluntary = (want_f2b & in_reach & self._cooldown_allows_f2b(st)
                         & self._contact_lock_allows_f2b(st)
                         & self._aerial_recovery_allows_f2b(st)
                         & self._ctrl_lock_allows_f2b(st)
                         & allowed & retouch_ok & first_contact & attempt_ready
                         & entry_phase_allows)
            # [B] 킥 타이밍 강제: 자율 킥 창을 없애고 setup 완료(=키커 도착·정렬) 즉시 결정론적 발사.
            # 타이밍은 env가 정하고 킥 파라미터(방향·파워·발사각·스핀)는 그대로 정책 액션 — 은닉 '언제 찰지'
            # 결정이 사라져 BC가 킥을 인과 라벨로 학습 가능. 킥오프 즉시발동은 setup_done을
            # 즉시 참으로 만드는 restart.py 로직으로 자연 흡수(별도 분기 불필요). 카운트다운 소진(restart_t<=1)
            # 타임아웃 강제도 setup_done 즉시 발사에 포섭된다.
            forced = (restart_active & setup_done & attempt_ready
                      & self._contact_lock_allows_f2b(st)
                      & self._aerial_recovery_allows_f2b(st)
                      # Arrival is a setup tolerance, not ball-contact reach.
                      # A large kicker_arrive_r can become true while the ball
                      # is still physically unreachable; forced release must
                      # obey the same reach predicate as voluntary contact.
                      & in_reach
                      & (self.player_indices == st.pending_taker) & (st.pending_taker >= 0)
                      & st.active_player      # 퇴장자면 강제 안 함(방어 가드) — 실해소는 events의 taker_dead 재지정
                      # 강제 발사도 의사결정 스텝 경계를 지킨다. forced는 allowed를 거치지 않으므로
                      # 여기에 같은 게이트를 따로 걸어야 한다 — 안 걸면 이 frame에 막 생긴 재개가
                      # 다음 서브스텝에 그대로 소비된다.
                      & (~restart_opened))
            do_kick = voluntary | forced
            # Committing to a reachable ball above standing height represents
            # a jump/aerial challenge even when the actor loses the contest or
            # misses the touch.  Recovery therefore follows the eligible
            # attempt, not the sampled winner.  ``head_z`` is the calibrated
            # body/head reference (tall * reach_height_factor), so invert that
            # static factor to recover the player's actual configured height.
            standing_height = st.head_z / jnp.float32(
                e_cfg.reach_height_factor
            )
            aerial_attempt = do_kick & (
                st.ball_pos[DIM_Z] > standing_height
            )
            return do_kick, reach_context, aerial_attempt

        def live_substep(carry, xs):
            (
                forced_winner_sub,
                forced_freeplay_sub,
                forced_gk_touch_sub,
                substep_index,
            ) = xs
            (
                st,
                scored,
                kick_acc,
                k,
                stamina_long_consumed,
                stamina_short_consumed,
                stamina_short_recovered,
                stamina_chargeable_seconds,
                stamina_sprint_seconds,
                stamina_sprint_extra_integral,
                stamina_locomotion_seconds,
                stamina_long_speed_load_integral,
                stamina_long_acceleration_load_integral,
                stamina_long_workload_integral,
                stamina_short_load_integral,
                stamina_short_recovery_factor_integral,
                stamina_short_drain_integral,
                stamina_short_recovery_integral,
                f2b_attempted,
                position_forced,
                position_frozen,
                position_env_written,
                restart_opened,
            ) = carry
            k, k_win, k_app, k_body, k_chg = jax.random.split(k, 5)
            k_win = select_random_key(
                randomness,
                RandomEvent.CONTEST_WINNER,
                substep_index,
                k_win,
            )
            poss_before = st.poss_team
            # 이 서브스텝 진입 시점의 재개 상태 — 아래에서 '이 frame에 새로 생긴 재개'를 가른다.
            restart_t_at_substep = st.restart_t
            restart_kind_at_substep = st.restart_kind
            restart_team_at_substep = st.restart_team

            # 직전 substep에 새 재개가 생겼을 수 있으므로 이동 전에 먼저 투영한다.
            frozen_pre = self._restart_encroacher_mask(st)
            st, forced_pre = self._project_restart_positions(st)
            position_forced = position_forced | forced_pre
            position_env_written = position_env_written | forced_pre
            position_frozen = position_frozen | frozen_pre
            # 키커 강제이동 + 세트피스 봉인 판정(키커는 이동 봉인)
            kicker_before = st
            st = lax.cond(
                entry_taker_repaired,
                lambda current: current,
                self._apply_kicker_move,
                st,
            )
            position_env_written = self._track_forced_positions(kicker_before, st, position_env_written)
            _, _, _, sp_active = self._setpiece_kick_lock(st)
            # The replacement was not the designated player in the entry
            # observation. Keep it an ordinary policy mover throughout this
            # boundary frame.
            sp_active = sp_active & (~entry_taker_repaired)
            is_kicker = (self.player_indices == st.pending_taker) & sp_active
            move_mask = (~is_kicker) & (~position_frozen)
            # ``_apply_kicker_move`` already traversed this tick and stores
            # the truthful displacement velocity for observation/facing.  Do
            # not feed that velocity into the ordinary position integrator or
            # the taker walks the same distance twice.  Restore it after the
            # ordinary movers and collision separator have run (the taker is
            # pinned there).
            forced_kicker_velocity = st.player_vel
            move_input_velocity = jnp.where(
                is_kicker[:, None], jnp.zeros_like(st.player_vel), st.player_vel
            )
            move_input = st._replace(
                player_vel=move_input_velocity,
                player_facing=self.facing_from_velocity(
                    move_input_velocity, st.attack_dir
                ),
            )
            movement = self._move_with_energy(
                move_input,
                move_mask,
                mv_dir,
                mv_pow,
                separation_pinned=is_kicker,
                position_update_mask=move_mask,
                locomotion_mask=move_mask,
                return_domain_projection=include_bc_info,
            )
            if include_bc_info:
                st, energy, domain_projection = movement
                position_env_written = (
                    position_env_written | domain_projection
                )
            else:
                st, energy = movement
            restored_velocity = jnp.where(
                is_kicker[:, None], forced_kicker_velocity, st.player_vel
            )
            st = st._replace(
                player_vel=restored_velocity,
                player_facing=self.facing_from_velocity(
                    restored_velocity, st.attack_dir
                ),
            )
            stamina_long_consumed = (
                stamina_long_consumed + energy.long_consumed
            )
            stamina_short_consumed = (
                stamina_short_consumed + energy.short_consumed
            )
            stamina_short_recovered = (
                stamina_short_recovered + energy.short_recovered
            )
            stamina_chargeable_seconds = stamina_chargeable_seconds + (
                energy.chargeable.astype(st.stamina_long.dtype) * e_cfg.dt_phys
            )
            stamina_sprint_seconds = stamina_sprint_seconds + (
                energy.sprinting.astype(st.stamina_long.dtype) * e_cfg.dt_phys
            )
            stamina_sprint_extra_integral = stamina_sprint_extra_integral + (
                energy.sprint_extra * e_cfg.dt_phys
            )
            stamina_locomotion_seconds = stamina_locomotion_seconds + (
                energy.locomotion.astype(st.stamina_long.dtype) * e_cfg.dt_phys
            )
            stamina_long_speed_load_integral = (
                stamina_long_speed_load_integral
                + energy.long_speed_load * e_cfg.dt_phys
            )
            stamina_long_acceleration_load_integral = (
                stamina_long_acceleration_load_integral
                + energy.long_acceleration_load * e_cfg.dt_phys
            )
            stamina_long_workload_integral = (
                stamina_long_workload_integral
                + energy.long_workload * e_cfg.dt_phys
            )
            stamina_short_load_integral = (
                stamina_short_load_integral + energy.short_load * e_cfg.dt_phys
            )
            stamina_short_recovery_factor_integral = (
                stamina_short_recovery_factor_integral
                + energy.short_recovery_factor * e_cfg.dt_phys
            )
            stamina_short_drain_integral = (
                stamina_short_drain_integral
                + energy.short_drain_rate * e_cfg.dt_phys
            )
            stamina_short_recovery_integral = (
                stamina_short_recovery_integral
                + energy.short_recovery_rate * e_cfg.dt_phys
            )

            # 자유 이동이 제한구역 안으로 들어온 경우 접촉 판정 전에 다시 최소 투영한다.
            # 새 개입자는 남은 substep에서 동결되고 control-frame 실제 마스크에 누적된다.
            frozen_post = self._restart_encroacher_mask(st)
            st, forced_post = self._project_restart_positions(st)
            position_forced = position_forced | forced_post
            position_env_written = position_env_written | forced_post
            position_frozen = position_frozen | frozen_post

            # 차징 파울이 공을 데드로 만들 수 있으므로, 경합·킥 적용 전 킥 게이트를 '차징 후' 상태로 재계산
            charge_before = st
            st = self._charge_foul(
                st,
                k_chg,
                suppress=suppress_charge,
                inject=inject_charge,
                randomness=randomness,
                substep_index=substep_index,
            )
            position_env_written = self._track_forced_positions(charge_before, st, position_env_written)
            # 차징은 이 substep **중간**에 새 FK/페널티를 만들 수 있다. 기존 래치는 substep
            # 끝에서만 갱신돼, 지정 키커가 스폿 반경 안이면 정책이 재개를 관측하기도 전에
            # 파울 이전 액션으로 즉시 강제킥했다. 차징 직후 경계를 먼저 세워 이번 control
            # frame의 모든 남은 substep에서 재개 소비를 막는다.
            charge_opened = self.restart_reopened(
                restart_t_at_substep, restart_kind_at_substep, restart_team_at_substep, st)
            restart_opened_now = restart_opened | charge_opened
            # 새 재개의 제한구역은 킥을 미루더라도 즉시 합법화한다.
            frozen_charge = self._restart_encroacher_mask(st)
            st, forced_charge = self._project_restart_positions(st)
            position_forced = position_forced | forced_charge
            position_env_written = position_env_written | forced_charge
            position_frozen = position_frozen | frozen_charge
            _, setup_done2, _, _ = self._setpiece_kick_lock(st)
            do_kick, reach_context, aerial_attempt = _kick_gate(
                st, setup_done2, f2b_attempted, restart_opened_now
            )
            # A control command gets at most one physical f2b attempt per
            # player.  Mark every eligible actor, not only the contest winner:
            # losing a contest (TOUCH_NONE) must not
            # resubmit the same action on every physics substep.  The latch is
            # scan-local and therefore resets next control frame.
            f2b_attempted = f2b_attempted | do_kick

            # GK 리액티브 클레임(박스 안 GK 본능)을 경합 후보에만 더한다 — 이동 facing용 do_kick과 분리.
            contest_cand = do_kick | self._gk_reactive_claim(st, reach_context)
            # 재탈취 지연(ctrl_lock) 중인 상대는 후보에서 제외 — 안 그러면 방금 뺏긴 선수가 경합 argmax를
            # 이겨 승자 슬롯을 먹고 no-op(locked)하며 새 점유팀 캐리어를 lock창(~0.16s) 동안 굶기고,
            # 헛쿨다운까지 받는다. 도달·쿨다운처럼 '확정 규칙 게이트'라 forced_winner pin도 통과해야 성립.
            locked_opp = (st.poss_team >= 0) & (st.team_id != st.poss_team) & (st.ctrl_lock_t > 0)
            contest_cand = contest_cand & (~locked_opp)
            winner, any_cand = self._contest_winner(
                st,
                contest_cand,
                reach_context.distance_xy,
                k_win,
                forced_winner_sub,
            )
            # 재터치 제한은 컨트롤 스텝 입구가 아니라 매 물리 서브스텝의 접촉 직전 상태를
            # 기준으로 판정해야 한다. touch_before로 이번 서브스텝에 새로 생긴 접촉만 구분한다.
            throw_taker_before = st.throw_taker
            setpiece_taker_before = st.setpiece_taker
            touch_before = st.touch
            contest_before = st
            st, force_touch_mask = self._apply_force2ball(
                st,
                winner,
                any_cand,
                f2b_dir,
                f2b_pow,
                f2b_launch,
                k_app,
                spin_side,
                spin_back,
                want_kick=do_kick,
                forced_freeplay=forced_freeplay_sub,
                contest_candidate=contest_cand,
                forced_gk_touch=(
                    forced_gk_touch_sub if gk_touch_pin_requested else None
                ),
                randomness=randomness,
                substep_index=substep_index,
                return_event=True,
            )
            # A tackle foul can draw a red card inside ``_apply_force2ball``;
            # ``_draw_cards`` then projects the offender to the bench.  Track
            # that non-policy position write just like the charge-foul path.
            # Without this wrapper a 30--50 m dismissal teleport remains an
            # apparently learnable movement label.
            position_env_written = self._track_forced_positions(
                contest_before, st, position_env_written
            )
            # Preserve the fixed physical ordering for rule adjudication:
            # active force-to-ball contact happens before the optional passive
            # body collision in this substep.
            touch_after_force = st.touch
            throw_taker_after_force = st.throw_taker
            setpiece_taker_after_force = st.setpiece_taker
            # Ordinary kicks contact the ball at its pre-contest position.
            # A throw-in is released from the taker's hands and the contest
            # primitive materialises that exact release point in ``st``.
            force_contact_pos = jnp.where(
                (contest_before.restart_kind == RK_THROWIN)
                & restart_timer_active(contest_before.restart_t),
                st.ball_pos,
                contest_before.ball_pos,
            )
            st = self._record_touch_event(
                st,
                substep_index,
                TOUCH_EVENT_FORCE,
                force_touch_mask,
                ball_pos=force_contact_pos,
                ball_vel_before=contest_before.ball_vel,
                ball_vel_after=st.ball_vel,
                toi=jnp.float32(0.0),
                administrative_stop=(
                    (contest_before.foul_kind == FOUL_NONE)
                    & (st.foul_kind != FOUL_NONE)
                ),
            )
            # ★인과킥 집계는 반드시 _ball_body **이전**에 — 몸통 트랩이 남기는 DRIBBLE/INTERCEPT는
            # 수동 물리(trap_velocity_keep·v)라 제출 킥 params와 무관한데, 컨트롤 스텝 끝의 누적
            # touch만 보면 인과킥으로 오탐된다(_kick_applied 참조). 여기서 서브스텝별 contest
            # 접촉만 뽑아 OR 누적한다.
            if include_bc_info:
                kick_acc = kick_acc | self._kick_applied(
                    st, touch_before, force_touch_mask=force_touch_mask
                )
            # touch_before를 넘겨 몸통 충돌 배제를 '이번 서브스텝의 새 접촉'으로 한정한다 —
            # 누적 touch를 그대로 보면 배제 창 길이가 decimation(즉 제어 frame 길이)에 종속된다.
            body_ball_pos = st.ball_pos
            body_ball_vel = st.ball_vel
            st, body_touch_mask, body_toi = self._ball_body(
                st, k_body, suppress=suppress_body, touch_before=touch_before,
                force_touch_mask=force_touch_mask,
                randomness=randomness,
                substep_index=substep_index,
                return_event=True,
            )
            body_contact_pos = (
                body_ball_pos
                + body_ball_vel * e_cfg.dt_phys * body_toi
            )
            st = self._record_touch_event(
                st,
                substep_index,
                TOUCH_EVENT_BODY,
                body_touch_mask,
                ball_pos=body_contact_pos,
                ball_vel_before=body_ball_vel,
                ball_vel_after=st.ball_vel,
                toi=body_toi,
            )
            st = self._throwin_restriction(
                st, throw_taker_before, setpiece_taker_before, touch_before,
                touch_after_force=touch_after_force,
                throw_taker_after_force=throw_taker_after_force,
                setpiece_taker_after_force=setpiece_taker_after_force,
                force_touch_mask=force_touch_mask,
                body_touch_mask=body_touch_mask,
                ball_pos_before_body=body_ball_pos,
            )
            # 오프사이드 콜도 재터치 판정과 같은 규약 — 이번 서브스텝에 새로 생긴 접촉만 트리거로
            # 삼는다. touch_before 없이 누적 touch를 보면 플래그가 서기 전(앞 서브스텝)의 터치로
            # 오프사이드가 즉시 오검된다(같은 컨트롤 스텝 내 다중 터치 국면).
            st = self._offside_check(
                st, touch_before, touch_after_force,
                body_ball_pos=body_ball_pos,
                force_touch_mask=force_touch_mask,
                body_touch_mask=body_touch_mask,
            )
            st, ball_pos_before, woodwork = self._ball_step_after_body(
                st,
                body_ball_pos,
                body_ball_vel,
                body_touch_mask,
                body_toi,
                return_event_start=True,
                return_woodwork=True,
            )
            # 프레임 반사는 선수 접촉이 아니라 순수 물리다 — touch/last_touch/오프사이드
            # 국면은 그대로 두고 telemetry만 남긴다. 그래야 포스트 리바운드가 코너/골킥
            # 라우팅에서 여전히 '마지막으로 찬 선수' 기준으로 갈린다(IFAB 규약).
            st = self._record_woodwork_event(st, substep_index, woodwork)

            # 접촉이 끝난 뒤, 이벤트 판정 전에 방치된 공의 소유권을 푼다. 여기서 풀어야
            # 다음 서브스텝의 관측과 정책이 곧바로 루즈볼 국면을 본다.
            st = self._release_unattended_possession(st)

            # 이벤트 판정 직전 상태로 재개 도착/활성 재계산(≤1서브스텝 카운트다운 편향 제거)
            _, _, arrived_ev, sp_active_ev = self._setpiece_kick_lock(st)
            # Preserve the timer for the entire identity-repair frame.  At
            # restart_t==1 a single hidden decrement would otherwise revive a
            # loose ball and erase the repaired restart before it is observed.
            arrived_ev = arrived_ev & (~entry_taker_repaired)
            sp_active_ev = sp_active_ev | entry_taker_repaired
            events_before = st
            st, s, event_restart_forced = self._events_with_restart_mask(
                st, arrived_ev, sp_active_ev, suppress_restart=suppress_restart,
                ball_pos_before=ball_pos_before,
                # The outer transition repaired this identity before entering
                # the substep scan.  Events can no longer infer that boundary
                # from the now-valid State, so carry the fact explicitly and
                # preserve the complete restart timer for this decision frame.
                preserve_restart_timer=entry_taker_repaired,
                substep_index=substep_index,
            )
            # Goal-created kickoffs are made legal atomically inside the
            # kickoff snap.  Preserve that exact half/circle projection mask;
            # the following substep sees an already-legal State and cannot
            # reconstruct the intervention retrospectively.
            position_forced = position_forced | event_restart_forced
            # The public foul fields form one latch.  Goal/restart/GK writers
            # live in different modules and may end the foul by clearing only
            # its kind; normalize after all rule writers in every substep so
            # FOUL_NONE can never coexist with stale actor/victim identities.
            st = self._normalize_foul_latch(st)
            # Offside/double-touch and other ordered rule writers may open a
            # restart after the force/body contact modules have updated the
            # causal GK-handling provenance.  Normalize once after all writers
            # so no dead-ball frame exposes a stale restriction.
            st = self._normalize_gk_handling_latch(st)
            # 득점은 22명 전원을 킥오프 포메이션으로 되돌린다(실측 최대 55 m). 정책 액션이
            # 설명할 수 없는 변위이므로 라벨에서 지운다.
            position_env_written = self._track_forced_positions(events_before, st, position_env_written)
            scored = jnp.where(scored >= 0, scored, s)

            # 재탈취 지연(per-player): 소유를 잃은 팀 선수에게 lock 세팅, 그 외 감쇠. 쿨다운 감쇠.
            changed = (st.poss_team != poss_before) & (poss_before >= 0)
            lock_set = changed & (st.team_id == poss_before)
            # The possession change happened inside this physical tick, so
            # that tick consumes the first unit of the newly installed lock.
            # Contact lock and challenge cooldown already follow this rule;
            # storing the full configured value here made ctrl_lock last one
            # tick longer than ``ctrl_lock_s`` and delayed eligibility by one.
            ctrl_lock_t = jnp.where(
                lock_set,
                jnp.maximum(jnp.int32(0), jnp.int32(e_cfg.ctrl_lock_substeps) - 1),
                jnp.maximum(0, st.ctrl_lock_t - 1),
            )
            cooldown = jnp.maximum(0.0, st.cooldown - 1.0)
            contact_lock_t = jnp.where(
                # Aerial attempts use their own active-only recovery timer.
                # Applying the generic contact debounce as well would suppress
                # the passive torso deflections that must remain live during
                # aerial recovery.
                do_kick & (~aerial_attempt),
                jnp.int32(e_cfg.contact_lock_substeps),
                st.contact_lock_t,
            )
            contact_lock_t = jnp.maximum(0, contact_lock_t - 1)
            aerial_recovery_t = jnp.where(
                aerial_attempt,
                jnp.int32(e_cfg.aerial_attempt_lock_substeps),
                st.aerial_recovery_t,
            )
            # The physical tick containing the attempt consumes the first
            # unit, matching contact_lock_t and ctrl_lock_t duration semantics.
            aerial_recovery_t = jnp.maximum(0, aerial_recovery_t - 1)
            st = st._replace(
                ctrl_lock_t=ctrl_lock_t,
                cooldown=cooldown,
                contact_lock_t=contact_lock_t,
                aerial_recovery_t=aerial_recovery_t,
            )
            # 이 서브스텝에서 재개가 새로 열렸는가 — 라이브에서 데드로 넘어갔거나(0 -> >0),
            # 데드 상태에서 종류가 바뀌었으면(소비 후 새 재개) 새 재개다. 한 번 서면 이 frame이
            # 끝날 때까지 유지되고, 다음 frame의 캐리는 다시 False로 시작한다.
            restart_opened = restart_opened_now | self.restart_reopened(
                restart_t_at_substep, restart_kind_at_substep, restart_team_at_substep, st)
            # collect_substeps(정적 플래그)면 서브스텝 종료 상태를 ys로 스택 — 물리 tick 밀도 렌더용.
            # False(학습 경로)면 None → jit 특수화로 오버헤드 0.
            return (
                st,
                scored,
                kick_acc,
                k,
                stamina_long_consumed,
                stamina_short_consumed,
                stamina_short_recovered,
                stamina_chargeable_seconds,
                stamina_sprint_seconds,
                stamina_sprint_extra_integral,
                stamina_locomotion_seconds,
                stamina_long_speed_load_integral,
                stamina_long_acceleration_load_integral,
                stamina_long_workload_integral,
                stamina_short_load_integral,
                stamina_short_recovery_factor_integral,
                stamina_short_drain_integral,
                stamina_short_recovery_integral,
                f2b_attempted,
                position_forced,
                position_frozen,
                position_env_written,
                restart_opened,
            ), (st if collect_substeps else None)

        def substep(carry, xs):
            """Freeze the remaining physical ticks once this frame abandons.

            A red card can reduce a team from seven to six in the first physics
            tick.  The transition that creates that terminal state is real and is
            retained, but no later tick may move the ball/players, drain stamina,
            or decrement timers after the match has ended.  Keeping the complete
            carry unchanged also freezes telemetry and RNG consumption.  With
            ``collect_substeps`` every remaining sample is therefore an exact copy
            of the first terminal state.
            """

            def frozen(args):
                frozen_carry, _ = args
                return frozen_carry, (
                    frozen_carry[0] if collect_substeps else None
                )

            return lax.cond(
                self._is_terminal(carry[0]),
                frozen,
                lambda args: live_substep(*args),
                (carry, xs),
            )

        energy_zero = jnp.zeros(self.N, dtype=state.stamina_long.dtype)
        (
            state,
            scored,
            kick_applied,
            _,
            stamina_long_consumed,
            stamina_short_consumed,
            stamina_short_recovered,
            stamina_chargeable_seconds,
            stamina_sprint_seconds,
            stamina_sprint_extra_integral,
            stamina_locomotion_seconds,
            stamina_long_speed_load_integral,
            stamina_long_acceleration_load_integral,
            stamina_long_workload_integral,
            stamina_short_load_integral,
            stamina_short_recovery_factor_integral,
            stamina_short_drain_integral,
            stamina_short_recovery_integral,
            _,
            restart_position_forced,
            _,
            position_env_written,
            restart_opened,
        ), sub_states = lax.scan(
            substep,
            (
                state,
                jnp.int32(NO_EVENT),
                jnp.zeros(self.N, dtype=bool),
                key,
                energy_zero,
                energy_zero,
                energy_zero,
                energy_zero,
                energy_zero,
                energy_zero,
                energy_zero,
                energy_zero,
                energy_zero,
                energy_zero,
                energy_zero,
                energy_zero,
                energy_zero,
                energy_zero,
                jnp.zeros(self.N, dtype=bool),
                restart_position_forced,
                restart_position_frozen,
                restart_position_forced,
                # A normal entry restart was observed by the policy.  A
                # repaired designated identity was not, so it gets the same
                # release barrier as a restart opened inside this frame.
                entry_taker_repaired,
            ),
            (
                forced_winner,
                forced_freeplay,
                forced_gk_touch,
                jnp.arange(decimation, dtype=jnp.int32),
            ),
            length=decimation,
        )
        state = state._replace(t=state.t + 1)
        # A dismissal inside the scan can make this very frame terminal.
        # Boundary-only mechanics must not execute after abandonment/time
        # expiry: otherwise a 7->6 red at halftime becomes a second-half
        # kickoff, or a due substitution mutates the terminal roster.
        transition_terminal = self._is_terminal(state)
        if self.halftime:                      # 정적 플래그 — 데모/클립 렌더에선 비활성 가능
            halftime_before = state
            do_halftime = ((state.t == (self.game_duration // 2))
                           & (~transition_terminal))
            state, halftime_restart_forced = self._halftime_switch_with_mask(
                state, do_halftime
            )
            restart_position_forced = (
                restart_position_forced | halftime_restart_forced
            )
            position_env_written = self._track_forced_positions(
                halftime_before, state, position_env_written)
        # 교체는 tick 경계에서 적용한다. 하프타임 전환 뒤에 두어야 투입 위치가 전환된
        # 좌표계에서 해석되고, 다음 관측이 이미 새 identity를 반영한다.
        substitution_before = state
        # 스케줄이 **먼저**다. 관측된 확정 사실이라 우선권이 있고, 이 순서라야 결정자가
        # 같은 슬롯을 덮어써 카드를 두 장 쓰는 일이 생기지 않는다. 결정자에게는 이 경계가
        # 시작될 때의 generation을 넘겨, 스케줄이 방금 바꾼 슬롯을 건드리지 않게 한다.
        boundary_generation = state.slot_generation
        # 감독을 **한 번** 부른다. 교체와 포메이션이 같은 뷰를 보고 함께 결정된다.
        # 주입이 있으면 그쪽이 이긴다 — 제안은 행동과 같은 매 스텝 데이터다.
        #
        # 둘 다 주입되면 감독의 출력은 **어디에도 쓰이지 않는다**. 재생·강제 실행에서
        # 매 tick 감독 추론을 돌리는 비용은 그대로 남으므로 아예 부르지 않는다. 주입
        # 여부는 파이썬 인자의 ``None`` 여부라 정적이고, 감독 키는 순차 split이 아니라
        # 주소 기반 ``fold_in``이라 호출을 건너뛰어도 다른 난수열이 밀리지 않는다.
        # Keep the submitted arrays separate from the merged proposal. The
        # legacy subsystem uses a negative slot as its no-request sentinel,
        # while native ``requested=True`` makes that same value an invalid
        # proposal which must remain visible in the result trace.
        native_substitution_proposal = (
            substitution if native_substitution_requested is not None else None
        )
        native_manager_merge = (
            native_substitution_requested is not None
            or native_formation_requested is not None
        )
        manager_may_be_called = (
            native_manager_merge
            or substitution is None
            or formation is None
        )
        if native_manager_merge:
            substitution_complete = (
                jnp.all(native_substitution_requested)
                if native_substitution_requested is not None
                else jnp.bool_(False)
            )
            formation_complete = (
                jnp.all(native_formation_requested)
                if native_formation_requested is not None
                else jnp.bool_(False)
            )
            manager_needed = (~substitution_complete) | (~formation_complete)
            manager_called = manager_needed
        else:
            manager_needed = None
            manager_called = jnp.bool_(manager_may_be_called)
        # 수집기는 결정자가 **실제로 본** 뷰를 라벨과 같은 호출에서 요구한다. 감독을
        # 부르지 않는 프레임에도 같은 shape의 자리를 남겨야 info의 pytree 구조가 흔들리지
        # 않으므로, 그때는 같은 경계 State로 만든 뷰를 sentinel로 쓴다(수집기는 called로
        # 거른다).
        # 둘 다 주입됐고 트레이스도 끄면 뷰조차 만들 필요가 없다 — 두 조건 모두
        # 파이썬 정적값이라 이 분기가 프로그램을 갈라 놓지 않는다.
        manager_decision_view = (
            self._manager_decision_view(state)
            if (
                include_decision_trace
                or (manager_may_be_called and not native_manager_merge)
            )
            else None
        )
        manager_trace_view = (
            self._canonical_manager_view_counters(manager_decision_view)
            if include_decision_trace
            else None
        )
        if manager_may_be_called:
            manager_key = select_random_key(
                randomness,
                RandomEvent.MANAGER_DECISION,
                0,
                jax.random.fold_in(key, 0x5B),
            )

            def configured_manager(_):
                view = (
                    manager_decision_view
                    if manager_decision_view is not None
                    else self._manager_decision_view(state)
                )
                return (
                    self._manager(manager_params, view, manager_key)
                    if self._manager_takes_params
                    else self._manager(view, manager_key)
                )

            if native_manager_merge:
                external_decision = manager_module.ManagerDecision(
                    out_slot=substitution[0],
                    bench_index=substitution[1],
                    layout=formation,
                )
                decision = lax.cond(
                    manager_needed,
                    configured_manager,
                    lambda _: external_decision,
                    None,
                )
            else:
                decision = configured_manager(None)
            if native_substitution_requested is not None:
                # Native commands override individual fixed-width cells. False
                # cells preserve the configured manager decision, which makes
                # an all-empty command exactly the frozen path.
                substitution = (
                    jnp.where(
                        native_substitution_requested,
                        substitution[0],
                        decision.out_slot,
                    ),
                    jnp.where(
                        native_substitution_requested,
                        substitution[1],
                        decision.bench_index,
                    ),
                )
            elif substitution is None:
                substitution = (decision.out_slot, decision.bench_index)
            if native_formation_requested is not None:
                formation = jnp.where(
                    native_formation_requested, formation, decision.layout
                )
            elif formation is None:
                formation = decision.layout

        state, substitution_restart_forced = lax.cond(
            transition_terminal,
            lambda current: (
                current, jnp.zeros(self.N, dtype=bool)
            ),
            self._apply_scheduled_substitutions_with_mask,
            state,
        )
        state, substitution_trace = lax.cond(
            transition_terminal,
            lambda current: (
                current,
                self._substitution_trace_zero(SUB_DECISION_TERMINAL),
            ),
            lambda current: self._apply_decided_substitutions_with_trace(
                current, jax.random.fold_in(key, 0x5B), proposal=substitution,
                boundary_generation=boundary_generation,
            ),
            state,
        )
        if native_substitution_requested is not None:
            submitted_out, submitted_bench = native_substitution_proposal
            substitution_trace = dict(substitution_trace)
            substitution_trace["substitution_proposed_out_slot"] = jnp.where(
                native_substitution_requested,
                submitted_out,
                substitution_trace["substitution_proposed_out_slot"],
            )
            substitution_trace["substitution_proposed_bench_index"] = jnp.where(
                native_substitution_requested,
                submitted_bench,
                substitution_trace["substitution_proposed_bench_index"],
            )
            invalid_slot = native_substitution_requested & (
                (submitted_out < 0) | (submitted_out >= self.N)
            )
            invalid_bench = (
                native_substitution_requested
                & (~invalid_slot)
                & (
                    (submitted_bench < 0)
                    | (submitted_bench >= self.bench_size)
                )
            )
            substitution_trace["substitution_applied"] = (
                substitution_trace["substitution_applied"]
                & (~invalid_slot)
                & (~invalid_bench)
            )
            substitution_trace["substitution_decision_code"] = jnp.where(
                invalid_slot,
                jnp.int32(SUB_DECISION_SLOT_RANGE),
                jnp.where(
                    invalid_bench,
                    jnp.int32(SUB_DECISION_BENCH_RANGE),
                    substitution_trace["substitution_decision_code"],
                ),
            )
        # 포메이션 지휘 — 교체와 같은 자리의 결정이다. 교체 **뒤**에 두는 이유는 새로 들어온
        # 선수도 이 tick부터 같은 목표 모양을 보고 서야 하기 때문이다.
        state, formation_trace = lax.cond(
            transition_terminal,
            lambda current: (
                current,
                self._formation_trace_zero(current, FORMATION_DECISION_TERMINAL),
            ),
            lambda current: self._apply_formation_command_with_trace(
                current,
                jax.random.fold_in(key, 0x5F),
                proposal=formation,
            ),
            state,
        )
        # Boundary substitutions defer restart enforcement until the complete
        # final roster exists.  Preserve the exact projector mask here; after
        # this projection the next frame is already legal, so discarding it
        # would permanently under-report restart intervention even though
        # ``move_forced`` correctly observes the same non-policy displacement.
        restart_position_forced = (
            restart_position_forced | substitution_restart_forced
        )
        # 교체 투입은 슬롯이 전후 모두 활성이라 ``~active`` 항이 걸리지 않는다. 지정 좌표로의
        # 순간이동(실측 13 m)이 정책 이동으로 학습되지 않도록 여기서 잡는다.
        position_env_written = self._track_forced_positions(
            substitution_before, state, position_env_written)
        # role anchor는 교체·하프타임·포메이션 epoch 경계가 모두 끝난 뒤 이 tick의 표본을
        # 더한다. 그래야 투입 선수와 새 포메이션의 첫 표본이 자기 구간에 들어가고,
        # role_pos가 t까지의 정보만 담는다. count 감소는 identity 증거가 아니다 — 교체는
        # 오직 slot_generation/player_id 경계로 판정한다.
        state = self._accumulate_role_anchor(state)
        state = self._update_possession_context(entry_state, state)

        active_per_team = jnp.stack([
            jnp.sum(state.active_player & (state.team_id == TEAM_0)),
            jnp.sum(state.active_player & (state.team_id == TEAM_1)),
        ]).astype(jnp.int32)
        time_limit = state.t >= self.game_duration
        abandoned = jnp.any(active_per_team < self.minimum_team_players)
        done_all = time_limit | abandoned
        reward = self._reward_array(
            state, scored, ball_x0, poss0, reward_entry_inplay, done_all
        )
        # terminal freeze — 진입이 이미 종료였다면 이 스텝은 일어나지 않은 것으로 만든다.
        state = jax.tree_util.tree_map(
            lambda frozen, stepped: jnp.where(entry_done, frozen, stepped),
            entry_state, state)
        reward = jnp.where(entry_done, jnp.zeros_like(reward), reward)
        scored = jnp.where(entry_done, jnp.int32(NO_EVENT), scored)
        kick_applied = kick_applied & (~entry_done)
        # 상태를 되돌린 뒤 info의 roster/종료 근거도 **되돌린 상태에서 다시 계산**한다.
        # 그렇지 않으면 terminal 내부 가상 전이에서 한 명이 퇴장했을 때 State는 11:11인데
        # info["active_per_team"]만 11:10인 자기모순이 생긴다.
        active_per_team = jnp.stack([
            jnp.sum(state.active_player & (state.team_id == TEAM_0)),
            jnp.sum(state.active_player & (state.team_id == TEAM_1)),
        ]).astype(jnp.int32)
        time_limit = state.t >= self.game_duration
        abandoned = jnp.any(active_per_team < self.minimum_team_players)
        done_all = time_limit | abandoned | entry_done
        # terminal duplicate frame에는 물리시간이 흐르지 않는다. State뿐 아니라 파생
        # telemetry/강제위치 라벨도 0으로 닫아야 소비자가 terminal_frozen을 한 번 놓쳐도
        # 가짜 스태미나·이동 표본이 생기지 않는다.
        stamina_long_consumed = jnp.where(
            entry_done, jnp.zeros_like(stamina_long_consumed),
            stamina_long_consumed
        )
        stamina_short_consumed = jnp.where(
            entry_done, jnp.zeros_like(stamina_short_consumed),
            stamina_short_consumed
        )
        stamina_short_recovered = jnp.where(
            entry_done, jnp.zeros_like(stamina_short_recovered),
            stamina_short_recovered
        )
        stamina_chargeable_seconds = jnp.where(
            entry_done, jnp.zeros_like(stamina_chargeable_seconds),
            stamina_chargeable_seconds)
        stamina_sprint_seconds = jnp.where(
            entry_done, jnp.zeros_like(stamina_sprint_seconds), stamina_sprint_seconds)
        stamina_sprint_extra_integral = jnp.where(
            entry_done, jnp.zeros_like(stamina_sprint_extra_integral),
            stamina_sprint_extra_integral)
        stamina_locomotion_seconds = jnp.where(
            entry_done, jnp.zeros_like(stamina_locomotion_seconds),
            stamina_locomotion_seconds)
        stamina_long_speed_load_integral = jnp.where(
            entry_done, jnp.zeros_like(stamina_long_speed_load_integral),
            stamina_long_speed_load_integral)
        stamina_long_acceleration_load_integral = jnp.where(
            entry_done, jnp.zeros_like(stamina_long_acceleration_load_integral),
            stamina_long_acceleration_load_integral)
        stamina_long_workload_integral = jnp.where(
            entry_done, jnp.zeros_like(stamina_long_workload_integral),
            stamina_long_workload_integral)
        stamina_short_load_integral = jnp.where(
            entry_done, jnp.zeros_like(stamina_short_load_integral),
            stamina_short_load_integral)
        stamina_short_recovery_factor_integral = jnp.where(
            entry_done, jnp.zeros_like(stamina_short_recovery_factor_integral),
            stamina_short_recovery_factor_integral)
        stamina_short_drain_integral = jnp.where(
            entry_done, jnp.zeros_like(stamina_short_drain_integral),
            stamina_short_drain_integral)
        stamina_short_recovery_integral = jnp.where(
            entry_done, jnp.zeros_like(stamina_short_recovery_integral),
            stamina_short_recovery_integral)
        restart_position_forced = restart_position_forced & (~entry_done)
        position_env_written = position_env_written & (~entry_done)
        # ``State.touch`` belongs to the last real control frame and must stay
        # unchanged for terminal-state identity.  ``info['touch']`` is event
        # telemetry for *this call*, however, so an absorbing duplicate step
        # must not replay that last contact indefinitely.
        touch_info = jnp.where(
            entry_done, jnp.zeros_like(state.touch), state.touch
        )
        # info에 BC 라벨링·reconstruct 검증용 터치/파울 신호 노출(스텝 내 누적된 per-player touch 등)
        info = {"scored": scored, "poss_team": state.poss_team, "score": state.score,
                "truncated": time_limit, "terminated": abandoned,
                # 이 스텝이 종료 이후라 항등으로 처리됐는가 — 수집기가 걸러 낼 수 있게 노출한다.
                "terminal_frozen": entry_done,
                "action_invalid": action_invalid,
                "forced_winner_invalid": forced_winner_invalid,
                "forced_gk_touch_invalid": forced_gk_touch_invalid,
                "abandoned": abandoned, "active_per_team": active_per_team,
                "active_player": state.active_player,
                "player_id": state.player_id, "slot_generation": state.slot_generation,
                "on_pitch": state.on_pitch, "sent_off": state.sent_off,
                "touch": touch_info, "last_touch_team": state.last_touch_team,
                "foul_kind": state.foul_kind, "foul_actor": state.foul_actor,
                "foul_victim": state.foul_victim,
                "stamina_long_consumed": stamina_long_consumed,
                "stamina_long_drain_rate": stamina_long_consumed / self.control_dt,
                "stamina_short_consumed": stamina_short_consumed,
                "stamina_short_recovered": stamina_short_recovered,
                "stamina_short_delta": (
                    stamina_short_recovered - stamina_short_consumed
                ),
                "stamina_short_drain_rate": (
                    stamina_short_drain_integral / self.control_dt
                ),
                "stamina_short_recovery_rate": (
                    stamina_short_recovery_integral / self.control_dt
                ),
                "stamina_chargeable": stamina_chargeable_seconds > 0.0,
                "stamina_chargeable_seconds": stamina_chargeable_seconds,
                "stamina_sprint_seconds": stamina_sprint_seconds,
                "stamina_long_sprint_extra": (
                    stamina_sprint_extra_integral / self.control_dt
                ),
                "stamina_locomotion_seconds": stamina_locomotion_seconds,
                "stamina_long_speed_load": (
                    stamina_long_speed_load_integral / self.control_dt
                ),
                "stamina_long_acceleration_load": (
                    stamina_long_acceleration_load_integral / self.control_dt
                ),
                "stamina_long_workload": (
                    stamina_long_workload_integral / self.control_dt
                ),
                "stamina_short_load": (
                    stamina_short_load_integral / self.control_dt
                ),
                "stamina_short_recovery_factor": (
                    stamina_short_recovery_factor_integral / self.control_dt
                ),
                "stamina_dynamics_version": jnp.int32(ENERGY_DYNAMICS_VERSION)}
        if randomness is not None:
            # The mask/key words are privileged replay provenance, not an
            # observation or action.  They appear only in the statically
            # controlled transition graph and therefore add zero payload to
            # ordinary training calls that omit ``randomness``.
            info["randomness_override_mask"] = randomness.key_override_mask
            info["randomness_key_data"] = randomness.key_data
        if include_decision_trace:
            # 결정 트레이스 — **제안과 승인 결과를 같은 호출에서** 낸다.
            #
            # 거절 자체는 규칙이라 없앨 수 없다(경계에서 공이 살아 있으면 못 바꾼다).
            # 없앨 수 있는 것은 **조용한** 거절이다. 종전에는 info에 교체·포메이션 결과가
            # 하나도 없어서, 주입된 학습 정책은 자기 제안이 적용됐는지조차 알 수 없었다.
            # 모방학습도 이 구분이 필요하다 — 학습 타깃은 규칙 정책의 **제안값**이고
            # 승인 결과는 legality/QC용이라, 둘을 한 프레임에서 같이 저장해야 어긋나지 않는다.
            #
            # 코드는 문자열이 아니라 정수다(JIT 안에서 만들어진다). terminal freeze 프레임은
            # 아무 결정도 일어나지 않은 것으로 닫는다 — State를 되돌리면서 트레이스만
            # 살려 두면 일어나지 않은 교체가 보고된다.
            frozen_substitution = self._substitution_trace_zero(
                SUB_DECISION_TERMINAL)
            frozen_formation = self._formation_trace_zero(
                state, FORMATION_DECISION_TERMINAL)
            for source, frozen in ((substitution_trace, frozen_substitution),
                                   (formation_trace, frozen_formation)):
                for field, value in source.items():
                    info[field] = jnp.where(entry_done, frozen[field], value)
            info["manager_called"] = (
                jnp.asarray(manager_called, jnp.bool_) & (~entry_done)
            )
            # 모방학습 capture가 읽는 중첩 형태. 평평한 키들과 같은 값이며 축별로 한
            # 덩어리를 가져갈 수 있도록 묶는다. ``state_hash``는 JAX 안에서 만들지 않고,
            # 저장기가 이 ``raw_view``를 host에서 해시한다.
            info["manager_decision_trace"] = {
                "called": info["manager_called"],
                "raw_view": manager_trace_view,
                "teacher_out_slot": info["substitution_proposed_out_slot"],
                "teacher_bench_index": info["substitution_proposed_bench_index"],
                "teacher_layout": info["formation_proposed_layout"],
                "approved_pair_mask": info["substitution_applied"],
                "approved_layout": info["formation_layout"],
                "rejection_reason": info["substitution_decision_code"].astype(
                    jnp.uint8),
                "layout_decision_code": info["formation_decision_code"].astype(
                    jnp.uint8),
            }
            # 키커 축은 아직 제안과 승인을 **분리해서** 낼 수 없다. 지정 호출이 서브스텝
            # 스캔 안의 여섯 개 규칙 함수에 흩어져 있어서, 제안값을 밖으로 빼려면 스캔
            # 캐리를 그 함수들 전부에 걸쳐 넓혀야 한다. 지금 낼 수 있는 것은 **이 프레임에
            # 지정이 열렸는가**와 그 결과이며, 그 조건은 ``_designate_taker_when``의 게이트가
            # 보는 사건과 같으므로 호출 여부의 정확한 상한이다.
            info["taker_pending"] = jnp.where(
                entry_done, entry_state.pending_taker, state.pending_taker)
            info["taker_restart_opened"] = restart_opened & (~entry_done)
            info["taker_changed"] = (
                (state.pending_taker != entry_state.pending_taker)
                & (~entry_done))
            if self._native_taker_command is not None:
                info.update(self._native_taker_trace(
                    entry_state,
                    state,
                    self._native_taker_command,
                    restart_opened,
                    entry_taker_repaired,
                    entry_taker_override_requested,
                    entry_taker_override_accepted,
                    entry_done,
                ))
        if include_bc_info:
            # 진입 시 이미 위반한 선수뿐 아니라 제출 이동으로 substep 중 경계를 침범해
            # 실제 투영된 선수도 사후 마스크한다.
            agency = dict(agency)
            # Entry state alone predicts a halftime boundary, but a scan-time
            # abandonment cancels the actual reset.  Reconcile the label with
            # the transition that really ran, avoiding a false sequence cut.
            agency["halftime_reset"] = (
                agency["halftime_reset"] & (~transition_terminal)
            )
            # 재개 투영뿐 아니라 **비정책 단계가 위치를 쓴 모든 슬롯**을 마스크한다 —
            # 득점 후 포메이션 재배치, 키커 스냅, 차징, 하프타임 전환, 교체 투입 포함.
            agency["move_forced"] = agency["move_forced"] | position_env_written
            # per-dim BC 손실 마스크 (N,8). 킥 dim은 진입 게이트 ∨ 실현 인과킥이다 —
            # info["kick_gated"]는 문서화된 **진입 시점** 신호 그대로 내보내고(RL이 사전 판단에 씀),
            # 마스크만 사후 실현을 반영한다.
            bc_mask = self.bc_action_mask(agency, kick_applied=kick_applied)
            info.update({
                # 실현 인과킥(선수별 bool) — 제출 킥 params가 이번 스텝 공 속도를 실제로 결정했는가.
                # [BC 킥 라벨 계약] 순수 킥 라벨 = bc_action_mask 킥 dim ∧ kick_applied.
                # 서브스텝 스캔이 contest 접촉만 골라 누적한 값(몸통 트랩 오탐 제외 — _kick_applied 참조).
                "kick_applied": kick_applied,
                # 행동 강제성 마스크 — 기본은 스텝 진입 상태 기준이고, move_forced만은 같은
                # 스텝 안에서 실제 발생한 재개 위치 투영까지 사후 OR한다.
                # 원신호 3종 + 손실에 바로 곱하는 per-dim 마스크(N,8). RL은 원신호로 자체 정책 구성 가능.
                "move_forced": agency["move_forced"], "kick_gated": agency["kick_gated"],
                "kick_forced": agency["kick_forced"],
                "restart_position_forced": restart_position_forced,
                # 하프타임 전이 프레임(bool[N], 프레임 단위 균일) — 시퀀스 hard boundary이자
                # 전 차원 라벨 마스크의 근거. 데이터셋의 `halftime_reset` 필드가 이 값을 그대로 쓴다.
                "halftime_reset": agency["halftime_reset"],
                "bc_action_mask": bc_mask,
                # 접촉 파라미터 sidecar (N,5) — action dim 3:8에 대응한다.
                # ``bc_action_mask``의 독스트링이 이미 계약으로 적어 둔 값인데
                # 생산자가 없었다(소비자만 있었다). True는 그 실행 branch가 제출
                # 파라미터를 **실제로 읽었다**는 뜻이다.
                #
                #   kick_applied=False        전부 False — 공에 적용된 것이 없다
                #   스로인 테이크             spin 두 dim은 False — throw_vel이
                #                             방향·발사각만 읽고 스핀을 버린다
                #   그 외 인과킥              전부 True
                "parameter_consumed_mask": self._parameter_consumed_mask(
                    decision_state, kick_applied),
            })
        if collect_substeps:
            # (decimation,)-선두 스택 State — 스텝 내 물리 서브스텝별 상태(위치·공 등). 렌더 밀도 up용.
            # terminal freeze에서는 내부 계산 흔적도 노출하지 않는다. 모든 물리 tick을 진입
            # terminal State의 복제본으로 바꿔 light render/진단이 가짜 움직임을 보지 않게 한다.
            sub_states = jax.tree_util.tree_map(
                lambda stepped, frozen: jnp.where(
                    entry_done,
                    jnp.broadcast_to(frozen, stepped.shape),
                    stepped,
                ),
                sub_states,
                entry_state,
            )
            info["substeps"] = sub_states
        obs = (
            self.get_obs_array(state)
            if compute_observation
            else jnp.empty((self.N, 0), dtype=state.ball_pos.dtype)
        )
        return obs, state, reward, done_all, info


_FLOAT32_RUNTIME_GUARDED_METHODS = (
    # Environment transitions and action/label projections.
    "reset", "reset_array", "reset_state", "step_env", "step_env_array",
    "empty_command", "step_command",
    "project_substitution", "prepare_observed_restart",
    "synchronize_observed_restart", "action_agency", "bc_action_mask",
    # Observation/state/affordance serialization.
    "get_obs", "get_obs_array", "get_state", "get_avail_actions",
    "get_avail_actions_array", "affordance_array", "affordance_view",
    "substitution_view", "formation_view", "formation_home",
    "squad_resources", "manager_view", "formation_layout_anchors",
    # 키커 뷰도 다른 세 결정 축의 뷰와 같은 계약이다 — float32 배열을 내주는 공개
    # 수치 메서드이므로 같은 런타임 가드를 받아야 한다.
    "setpiece_taker_view",
    # Public physics, geometry and inverse/playback helpers.  Although several
    # are normally reached through step_env_array, they are callable adapters
    # in their own right and must not emit silent float64/mixed-width results
    # after a process-global JAX x64 toggle.
    "ball_step_only", "canonical_restart_spot", "effective_vmax",
    "facing_from_velocity", "goal_frame_impact",
    "infer_contact", "infer_deflect",
    "infer_kick_action", "infer_move_action", "infer_throwin_action",
    "launch_lo", "nearest_free_position", "predict_parry",
    "reconcile_positions", "repair_causal_overlaps", "restart_reopened",
    "stamina_transition",
)


def _float32_runtime_guard(method):
    """Wrap one public numeric method without changing its signature/docs."""

    @wraps(method)
    def guarded(*args, **kwargs):
        _validate_float32_runtime()
        return method(*args, **kwargs)

    guarded._soccer_float32_runtime_guarded = True
    return guarded


# Mixins own most public numeric methods, so guards installed only in their
# source files are easy to miss and create import cycles.  Materialise an
# explicit allowlist on SoccerEnv once, after class creation.  Host-only schema,
# metadata, space and renderer APIs are intentionally excluded.
for _guarded_method_name in _FLOAT32_RUNTIME_GUARDED_METHODS:
    setattr(
        SoccerEnv,
        _guarded_method_name,
        _float32_runtime_guard(getattr(SoccerEnv, _guarded_method_name)),
    )


if __name__ == "__main__":
    env = SoccerEnv(
        control_fps=DEFAULT_CONTROL_FPS,
    )
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key = key)

    for i in range(10):
        key, k_step = jax.random.split(key)
        one_act = jnp.array(
            [-1.0, 1.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1.0],
            dtype=jnp.float32,
        )
        actions = {agent: one_act for agent in env.agents}
        obs, state, reward, done, info = env.step(key = k_step, state = state, actions = actions)
        print(f"Step {i+1}: t={int(state.t)} ball_state={int(state.ball_state)} "
              f"poss={int(state.poss_team)} score={state.score.tolist()} done={bool(done['__all__'])}")
        print("-" * 30)
