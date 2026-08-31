import math
import numbers

import jax
import jax.numpy as jnp
import numpy as np

from .constants import (
    BALL_ALIVE,
    BALL_DEAD,
    COINCIDENT_DISTANCE_EPS,
    DEPARTED_TAKER,
    DIM_ALL,
    DIM_X,
    DIM_Y,
    DIM_Z,
    DIV_EPS,
    FOUL_NONE,
    FOUL_SETPIECE,
    FOUL_THROW,
    GEOMETRY_EPS,
    NO_PLAYER,
    NO_TEAM,
    RESTART_COUNT,
    RK_CORNER,
    RK_FREEKICK,
    RK_GK_HOLD,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_NONE,
    RK_OFFSIDE,
    RK_PENALTY,
    RK_THROWIN,
    TEAM_0,
    TEAM_1,
    TEAM_COUNT,
    TOUCH_NONE,
)


def _validate_float32_runtime():
    """Reject a process-global x64 toggle outside the float32 contract."""

    if bool(jax.config.jax_enable_x64):
        raise RuntimeError(
            "SoccerEnv is a float32 dynamics contract and does not support "
            "JAX_ENABLE_X64=1; disable jax_enable_x64 before constructing or "
            "using the environment"
        )


# Public mechanics helpers accept convenient Python/NumPy real values at an
# eager host boundary, but all actual dynamics execute as float32 JAX arrays.
# Inspecting only ``jnp.asarray(value)`` loses the source width when x64 is
# disabled (for example ``np.uint64(2**32 + 1) -> uint32(1)``).  These small
# adapters therefore validate concrete host values *before* JAX sees them.
# Traced/device inputs cannot be inspected without synchronising or breaking
# jit/vmap, so their static shape/dtype is strict and data-dependent invalid
# values are replaced with a bounded fallback.
_FLOAT32_PUBLIC_MAGNITUDE_LIMIT = float(np.finfo(np.float32).max) ** 0.25 / 4.0


def _contains_jax_value(value) -> bool:
    """Whether ``value`` contains a tracer/device array rather than host data."""

    return any(
        isinstance(leaf, (jax.Array, jax.core.Tracer))
        for leaf in jax.tree_util.tree_leaves(value)
    )


def _coerce_public_float32(
    name,
    value,
    shape=None,
    *,
    fallback=0.0,
    minimum=None,
    maximum=None,
    allowed_values=None,
):
    """Canonicalise one public real scalar/array without hidden narrowing.

    Both host and JAX inputs must have a floating dtype; integer convenience is
    deliberately limited to higher-level adapters such as observed-position
    ingestion.  Keeping mechanics floating-only makes the exact same NumPy
    input obey the same eager/jit dtype contract and prevents a wide integer
    from narrowing before a compiled boundary.  The returned array is always
    float32.  The second return value is a scalar validity bit for traced
    callers; host-invalid values raise before conversion.
    """

    expected_shape = None if shape is None else tuple(shape)
    is_jax_value = _contains_jax_value(value)
    if not is_jax_value:
        try:
            host = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a real numeric value") from exc
        if expected_shape is not None and host.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {host.shape}"
            )
        if not np.issubdtype(host.dtype, np.floating):
            raise TypeError(
                f"{name} must have floating dtype, got {host.dtype}"
            )
        if not bool(np.isfinite(host).all()):
            raise ValueError(f"{name} must contain only finite values")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            host32 = host.astype(np.float32)
        if not bool(np.isfinite(host32).all()):
            raise ValueError(f"{name} is not representable as finite float32")
        # Domain checks apply to the value dynamics actually consumes.  A
        # float64 just above 1.0, or a tiny negative that becomes -0.0, arrives
        # at a jitted function as exactly 1.0f/0.0f and must receive the same
        # decision on the eager path.
        safe_limit32 = np.float32(_FLOAT32_PUBLIC_MAGNITUDE_LIMIT)
        if bool((np.abs(host32) > safe_limit32).any()):
            raise ValueError(
                f"{name} exceeds the safe float32 dynamics magnitude "
                f"{_FLOAT32_PUBLIC_MAGNITUDE_LIMIT:g}"
            )
        if minimum is not None and bool((host32 < np.float32(minimum)).any()):
            raise ValueError(f"{name} must be >= {minimum}")
        if maximum is not None and bool((host32 > np.float32(maximum)).any()):
            raise ValueError(f"{name} must be <= {maximum}")
        if allowed_values is not None and not bool(
            np.isin(
                host32,
                np.asarray(tuple(allowed_values), dtype=np.float32),
            ).all()
        ):
            raise ValueError(
                f"{name} must contain only {tuple(allowed_values)}, got {host!r}"
            )
        # Finite values below float32 range canonically become zero.  Rejecting
        # them only on the eager path would disagree with jit, whose NumPy
        # float64 arguments are already canonical float32 tracers at entry.
        return jnp.asarray(host32, dtype=jnp.float32), jnp.bool_(True)

    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real floating array") from exc
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {array.shape}"
        )
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(
            f"{name} must have floating dtype in JAX-compiled calls, "
            f"got {array.dtype}"
        )
    array = array.astype(jnp.float32)
    element_valid = (
        jnp.isfinite(array)
        & (jnp.abs(array) <= jnp.float32(_FLOAT32_PUBLIC_MAGNITUDE_LIMIT))
    )
    if minimum is not None:
        element_valid = element_valid & (array >= jnp.float32(minimum))
    if maximum is not None:
        element_valid = element_valid & (array <= jnp.float32(maximum))
    if allowed_values is not None:
        allowed = jnp.asarray(tuple(allowed_values), dtype=jnp.float32)
        element_valid = element_valid & jnp.any(
            array[..., None] == allowed, axis=-1
        )
    safe = jnp.where(element_valid, array, jnp.float32(fallback))
    return safe, jnp.all(element_valid)


def _coerce_public_bool(name, value, shape):
    """Require one bool array with an exact public shape."""

    expected_shape = tuple(shape)
    if not _contains_jax_value(value):
        try:
            host = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a boolean array") from exc
        if host.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {host.shape}"
            )
        if not np.issubdtype(host.dtype, np.bool_):
            raise TypeError(f"{name} must have bool dtype, got {host.dtype}")
        return jnp.asarray(host, dtype=jnp.bool_)

    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a boolean array") from exc
    if array.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {array.shape}"
        )
    if array.dtype != jnp.dtype(jnp.bool_):
        raise TypeError(f"{name} must have bool dtype, got {array.dtype}")
    return array


def _coerce_public_signed_int(
    name, value, shape=None, *, minimum, maximum, fallback
):
    """Validate public signed integers before any possible int32 narrowing."""

    expected_shape = None if shape is None else tuple(shape)
    if not _contains_jax_value(value):
        try:
            host = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a signed integer array") from exc
        if expected_shape is not None and host.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {host.shape}"
            )
        if not np.issubdtype(host.dtype, np.signedinteger):
            raise TypeError(
                f"{name} must have signed-integer dtype, got {host.dtype}"
            )
        legal = (host >= minimum) & (host <= maximum)
        if not bool(legal.all()):
            raise ValueError(
                f"{name} entries must lie in [{minimum}, {maximum}], got "
                f"{host[~legal].tolist() if host.shape else int(host)}"
            )
        return (
            jnp.asarray(host, dtype=jnp.int32),
            jnp.ones(host.shape, dtype=jnp.bool_),
        )

    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a signed integer array") from exc
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {array.shape}"
        )
    if not jnp.issubdtype(array.dtype, jnp.signedinteger):
        raise TypeError(
            f"{name} must have signed-integer dtype, got {array.dtype}"
        )
    valid = (array >= minimum) & (array <= maximum)
    safe = jnp.where(valid, array, jnp.asarray(fallback, array.dtype))
    return safe.astype(jnp.int32), valid


def _coerce_public_signed_int_scalar(
    name, value, *, minimum, maximum, fallback
):
    """Scalar specialization used by bounded public loop counts."""

    return _coerce_public_signed_int(
        name, value, (),
        minimum=minimum, maximum=maximum, fallback=fallback,
    )


def restart_timer_active(restart_t):
    """재개 인스턴스가 진행 중인지 판정하는 단일 진실원천.

    물리·관측·행동 agency·에너지·검증이 이 술어를 공유해야 타이머 표현을
    바꿀 때 한 경로만 옛 의미에 남는 조용한 불일치를 막을 수 있다.
    """

    return restart_t > 0


class Restart:
    def _restart_window_ready_threshold(self, restart_kind):
        """Return ``(window, ready_threshold)`` for one restart kind.

        ``restart_t`` starts at ``window``.  A valid taker may release once the
        remaining timer reaches ``ready_threshold`` *and* the ball is physically
        in reach.  Except for kickoffs, the timer is frozen during approach and
        counts only after physical arrival, giving the explicit
        ``approach time + configured dead-ball hold`` contract.

        Opening and half-time kickoffs are constructed with ``restart_t=1``
        and are therefore immediately ready.  A kickoff opened by a goal uses
        the full ordinary window but has a zero delay, so the goal frame still
        crosses the decision boundary and the following step is ready.
        """

        e_cfg = self.e_cfg
        kind = jnp.asarray(restart_kind, dtype=jnp.int32)
        if kind.shape != ():
            raise ValueError(
                f"restart_kind must be scalar, got shape {kind.shape}"
            )
        ordinary_delay = jnp.where(
            kind == RK_KICKOFF,
            e_cfg.post_goal_kickoff_delay_substeps,
            jnp.where(
                kind == RK_THROWIN,
                e_cfg.throwin_restart_delay_substeps,
                jnp.where(
                    kind == RK_GOALKICK,
                    e_cfg.goalkick_restart_delay_substeps,
                    jnp.where(
                        kind == RK_CORNER,
                        e_cfg.corner_restart_delay_substeps,
                        jnp.where(
                            kind == RK_FREEKICK,
                            e_cfg.freekick_restart_delay_substeps,
                            jnp.where(
                                kind == RK_OFFSIDE,
                                e_cfg.offside_restart_delay_substeps,
                                e_cfg.setup_hold_substeps,
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        window = jnp.where(
            kind == RK_PENALTY,
            e_cfg.penalty_substeps,
            jnp.where(
                kind == RK_GK_HOLD,
                e_cfg.gk_hold_substeps,
                e_cfg.restart_substeps,
            ),
        ).astype(jnp.int32)
        delay = jnp.where(
            kind == RK_PENALTY,
            e_cfg.penalty_restart_delay_substeps,
            jnp.where(
                kind == RK_GK_HOLD,
                e_cfg.setup_hold_substeps,
                ordinary_delay,
            ),
        ).astype(jnp.int32)
        return window, jnp.maximum(jnp.int32(0), window - delay)

    def _sanitize_observed_restart(
        self, state, restart_kind, restart_team, pending_taker, restart_t
    ):
        """외부 재개 입력을 host에서는 거부하고 traced 경로에서는 합법 상태로 투영한다."""

        names = ("restart_kind", "restart_team", "pending_taker", "restart_t")
        values = (restart_kind, restart_team, pending_taker, restart_t)
        arrays = []
        concrete = {}
        for name, value in zip(names, values):
            value_traced = any(
                isinstance(leaf, jax.core.Tracer)
                for leaf in jax.tree_util.tree_leaves(value)
            )
            if not value_traced:
                # Inspect the original host width before JAX canonicalisation.
                # With x64 disabled, jnp.asarray(np.int64(2**32 + code))
                # narrows to int32 and can wrap into a legal enum/slot.  A
                # host adapter must reject that source value, not validate its
                # truncated image.
                try:
                    host_array = np.asarray(value)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"{name} must be a scalar non-boolean integer"
                    ) from exc
                if host_array.shape != ():
                    raise ValueError(
                        f"{name} must be a scalar, got shape {host_array.shape}"
                    )
                if (
                    not np.issubdtype(host_array.dtype, np.integer)
                    or np.issubdtype(host_array.dtype, np.bool_)
                ):
                    raise TypeError(
                        f"{name} must have a non-boolean integer dtype, "
                        f"got {host_array.dtype}"
                    )
                concrete[name] = int(host_array)
            array = jnp.asarray(value)
            if array.shape != ():
                raise ValueError(f"{name} must be a scalar, got shape {array.shape}")
            if (not jnp.issubdtype(array.dtype, jnp.integer)
                    or jnp.issubdtype(array.dtype, jnp.bool_)):
                raise TypeError(
                    f"{name} must have a non-boolean integer dtype, got {array.dtype}"
                )
            arrays.append(array)

        # Validate every eager-visible field independently.  The old all-or-
        # nothing Python-int check let one float disable validation for the
        # other three fields, after which int32 conversion silently truncated
        # it.  Traced integer values retain the documented sanitize/fallback
        # behaviour because their values are unavailable at trace time.
        if "restart_kind" in concrete:
            kind_i = concrete["restart_kind"]
            if not RK_KICKOFF <= kind_i < RESTART_COUNT:
                raise ValueError(f"restart_kind must be an active restart code, got {kind_i}")
        if "restart_team" in concrete:
            team_i = concrete["restart_team"]
            if team_i not in (TEAM_0, TEAM_1):
                raise ValueError(f"restart_team must be 0 or 1, got {team_i}")
        if "pending_taker" in concrete:
            taker_i = concrete["pending_taker"]
            if not 0 <= taker_i < self.N:
                raise ValueError(f"pending_taker must lie in [0, {self.N}), got {taker_i}")
        if "restart_t" in concrete:
            timer_i = concrete["restart_t"]
            if timer_i <= 0:
                raise ValueError(f"restart_t must be positive, got {timer_i}")
        if "restart_kind" in concrete and "restart_t" in concrete:
            kind_i = concrete["restart_kind"]
            timer_i = concrete["restart_t"]
            window_i = (
                self.e_cfg.penalty_substeps if kind_i == RK_PENALTY
                else self.e_cfg.gk_hold_substeps if kind_i == RK_GK_HOLD
                else self.e_cfg.restart_substeps
            )
            if timer_i > window_i:
                raise ValueError(f"restart_t exceeds the configured window {window_i}: {timer_i}")

        state_visible = not (
            isinstance(state.active_player, jax.core.Tracer)
            or isinstance(state.team_id, jax.core.Tracer)
        )
        if (state_visible
                and "restart_team" in concrete
                and "pending_taker" in concrete):
            team_i = concrete["restart_team"]
            taker_i = concrete["pending_taker"]
            active = bool(np.asarray(state.active_player)[taker_i])
            team_ok = int(np.asarray(state.team_id)[taker_i]) == team_i
            if not active or not team_ok:
                raise ValueError("pending_taker must be active and belong to restart_team")
            if (concrete.get("restart_kind") == RK_GK_HOLD
                    and not bool(np.asarray(state.gk_indices)[taker_i])):
                raise ValueError("RK_GK_HOLD pending_taker must be a goalkeeper")

        kind_input, team_input, taker_input, timer_input = arrays
        # Validate/sanitize in the source width before int32 narrowing.  A
        # traced int64 value such as 2**32+RK_FREEKICK must not wrap into a
        # legal enum; the same applies to team, taker and timer.
        kind_valid = (
            (kind_input >= RK_KICKOFF) & (kind_input < RESTART_COUNT)
        )
        kind = jnp.where(
            kind_valid, kind_input,
            jnp.asarray(RK_FREEKICK, dtype=kind_input.dtype),
        ).astype(jnp.int32)
        team_valid = (team_input == TEAM_0) | (team_input == TEAM_1)
        team = jnp.where(
            team_valid, team_input,
            jnp.asarray(TEAM_0, dtype=team_input.dtype),
        ).astype(jnp.int32)
        taker_range_valid = (taker_input >= 0) & (taker_input < self.N)
        safe_taker = jnp.where(
            taker_range_valid, taker_input,
            jnp.zeros((), dtype=taker_input.dtype),
        ).astype(jnp.int32)
        taker_valid = (
            taker_range_valid
            & state.active_player[safe_taker]
            & (state.team_id[safe_taker] == team)
            & ((kind != RK_GK_HOLD) | (state.gk_indices[safe_taker] == 1))
        )
        # ``restart_indirect``는 이 함수 뒤쪽에서 정규화된다. 여기서는 현재 state의 값을
        # 쓴다 — 관측된 재개를 정리하는 경로라 종류는 이미 확정돼 있고, 간접 여부는 키커
        # 선택에서 직접/간접 프리킥만 가른다.
        fallback_any = self._designate_taker_when(
            ~taker_valid,
            state, state.ball_pos[:DIM_Z], team, kind == RK_GOALKICK,
            kind, state.restart_indirect
        )
        # A goal kick may legally fall back to any team-mate when its goalkeeper
        # is unavailable.  A goalkeeper hold cannot: it is the result of a GK
        # hand claim inside that player's own penalty area.  Keep the traced
        # adapter fail-closed when no active goalkeeper exists instead of
        # silently turning an outfielder into a goalkeeper.
        hold_gk = (
            state.active_player
            & (state.team_id == team)
            & (state.gk_indices == 1)
        )
        hold_gk_distance = jnp.linalg.norm(
            state.player_pos - state.ball_pos[:DIM_Z][None, :], axis=1
        )
        hold_gk_taker = jnp.where(
            jnp.any(hold_gk),
            jnp.argmin(jnp.where(hold_gk, hold_gk_distance, jnp.inf)),
            jnp.int32(NO_PLAYER),
        ).astype(jnp.int32)
        fallback = jnp.where(
            kind == RK_GK_HOLD, hold_gk_taker, fallback_any
        ).astype(jnp.int32)
        taker = jnp.where(
            taker_valid, safe_taker, fallback
        ).astype(jnp.int32)
        window = jnp.where(
            kind == RK_PENALTY,
            self.e_cfg.penalty_substeps,
            jnp.where(kind == RK_GK_HOLD, self.e_cfg.gk_hold_substeps,
                      self.e_cfg.restart_substeps),
        )
        timer = jnp.where(
            timer_input < 1,
            jnp.ones((), dtype=timer_input.dtype),
            jnp.where(
                timer_input > window,
                window.astype(timer_input.dtype),
                timer_input,
            ),
        ).astype(jnp.int32)
        return kind.astype(jnp.int32), team.astype(jnp.int32), taker, timer

    def _sanitize_observed_restart_indirect(
        self, restart_kind, restart_indirect
    ):
        """Validate/sanitize the observed direct-vs-indirect provenance.

        Offside is intrinsically indirect.  A generic observed free kick may
        be either direct or indirect, while no other restart kind can carry
        this flag.  Host-visible misuse raises; traced data-dependent misuse
        is projected to the only legal value for its kind.
        """

        if restart_indirect is None:
            return (restart_kind == RK_OFFSIDE).astype(jnp.bool_)

        value_traced = any(
            isinstance(leaf, jax.core.Tracer)
            for leaf in jax.tree_util.tree_leaves(restart_indirect)
        )
        if not value_traced:
            try:
                host = np.asarray(restart_indirect)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "restart_indirect must be a scalar bool"
                ) from exc
            if host.shape != ():
                raise ValueError(
                    "restart_indirect must be a scalar, "
                    f"got shape {host.shape}"
                )
            if not np.issubdtype(host.dtype, np.bool_):
                raise TypeError(
                    "restart_indirect must have bool dtype, "
                    f"got {host.dtype}"
                )
            if not isinstance(restart_kind, jax.core.Tracer):
                kind = int(np.asarray(restart_kind))
                indirect = bool(host)
                if kind == RK_OFFSIDE and not indirect:
                    raise ValueError("RK_OFFSIDE must be indirect")
                if kind not in (RK_FREEKICK, RK_OFFSIDE) and indirect:
                    raise ValueError(
                        "restart_indirect=True is valid only for "
                        "RK_FREEKICK or RK_OFFSIDE"
                    )

        value = jnp.asarray(restart_indirect)
        if value.shape != ():
            raise ValueError(
                "restart_indirect must be a scalar, "
                f"got shape {value.shape}"
            )
        if value.dtype != jnp.dtype(jnp.bool_):
            raise TypeError(
                "restart_indirect must have bool dtype, "
                f"got {value.dtype}"
            )
        return jnp.where(
            restart_kind == RK_OFFSIDE,
            jnp.bool_(True),
            jnp.where(
                restart_kind == RK_FREEKICK,
                value,
                jnp.bool_(False),
            ),
        ).astype(jnp.bool_)

    @staticmethod
    def _canonical_observed_ball_pos(observed_ball_pos):
        """Canonicalise one public observed position without integer wrapping.

        JAX's x64-disabled default narrows a host ``int64`` array to ``int32``
        before ordinary arithmetic.  For external coordinates that made, for
        example, ``2**32 + 1`` silently become one metre.  Host-visible values
        are therefore checked at their source width and converted *directly*
        to float32.  Traced values cannot be inspected for finiteness, but rank
        and dtype remain static and are enforced during tracing; the caller's
        geometry then handles non-finite values with its documented fail-closed
        fallback.
        """

        leaves = jax.tree_util.tree_leaves(observed_ball_pos)
        traced = any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)
        message = (
            "observed_ball_pos must contain three finite real non-boolean "
            "values representable as float32"
        )

        if not traced:
            try:
                raw = np.asarray(observed_ball_pos, dtype=object)
            except (TypeError, ValueError) as exc:
                raise ValueError(message) from exc
            if raw.shape != (DIM_ALL,):
                raise ValueError(message)
            try:
                valid = all(
                    isinstance(value, numbers.Real)
                    and not isinstance(value, (bool, np.bool_))
                    and math.isfinite(float(value))
                    for value in raw.flat
                )
            except (OverflowError, TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError(message)
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    canonical = np.asarray(
                        [float(value) for value in raw.flat], dtype=np.float32
                    ).reshape((DIM_ALL,))
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(message) from exc
            if not bool(np.isfinite(canonical).all()):
                raise ValueError(message)
            return jnp.asarray(canonical, dtype=jnp.float32)

        # A mixed pytree such as ``(tracer, True, 0.0)`` promotes to float32
        # if inspected only after stacking.  Check every leaf's static dtype
        # first so a boolean/complex constant cannot hide beside a tracer.
        for leaf in leaves:
            try:
                leaf_dtype = jnp.asarray(leaf).dtype
            except (TypeError, ValueError) as exc:
                raise ValueError(message) from exc
            if (
                jnp.issubdtype(leaf_dtype, jnp.bool_)
                or jnp.issubdtype(leaf_dtype, jnp.complexfloating)
                or not (
                    jnp.issubdtype(leaf_dtype, jnp.integer)
                    or jnp.issubdtype(leaf_dtype, jnp.floating)
                )
            ):
                raise ValueError(message)
        try:
            observed = jnp.asarray(observed_ball_pos, dtype=jnp.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(message) from exc
        if observed.shape != (DIM_ALL,):
            raise ValueError(message)
        return observed

    def _sanitize_canonical_restart_inputs(
        self, restart_kind, restart_team, restart_indirect
    ):
        """Validate public canonical-spot scalars and safely project tracers.

        Eager callers are checked before any int32 narrowing.  Under JIT/vmap,
        dtype and scalar rank remain statically enforceable while invalid
        dynamic enum values are mapped onto the helper's safe fallback
        geometry: unknown kinds use an ordinary free-kick spot and teams are
        clipped to the two legal team frames.
        """

        arrays = []
        concrete = {}
        for name, value in (
            ("restart_kind", restart_kind),
            ("restart_team", restart_team),
        ):
            value_traced = any(
                isinstance(leaf, jax.core.Tracer)
                for leaf in jax.tree_util.tree_leaves(value)
            )
            if not value_traced:
                try:
                    host = np.asarray(value)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"{name} must be a scalar non-boolean integer"
                    ) from exc
                if host.shape != ():
                    raise ValueError(
                        f"{name} must be a scalar, got shape {host.shape}"
                    )
                if (
                    not np.issubdtype(host.dtype, np.integer)
                    or np.issubdtype(host.dtype, np.bool_)
                ):
                    raise TypeError(
                        f"{name} must have a non-boolean integer dtype, "
                        f"got {host.dtype}"
                    )
                concrete[name] = int(host)
                # Reject in the original host width *before* constructing an
                # int32 JAX scalar.  Otherwise uint64(max) can raise an
                # unrelated OverflowError (or 2**32+code can wrap) before the
                # public range diagnostic below has a chance to run.
                if name == "restart_kind" and not (
                    RK_KICKOFF <= concrete[name] < RESTART_COUNT
                ):
                    raise ValueError(
                        "restart_kind must be an active restart code, "
                        f"got {concrete[name]}"
                    )
                if name == "restart_team" and concrete[name] not in (
                    TEAM_0, TEAM_1
                ):
                    raise ValueError(
                        f"restart_team must be 0 or 1, got {concrete[name]}"
                    )
                array = jnp.asarray(concrete[name], dtype=jnp.int32)
            else:
                array = jnp.asarray(value)
                if array.shape != ():
                    raise ValueError(
                        f"{name} must be a scalar, got shape {array.shape}"
                    )
                if (
                    not jnp.issubdtype(array.dtype, jnp.integer)
                    or jnp.issubdtype(array.dtype, jnp.bool_)
                ):
                    raise TypeError(
                        f"{name} must have a non-boolean integer dtype, "
                        f"got {array.dtype}"
                    )
            arrays.append(array)

        kind_input, team_input = arrays
        kind_valid = (
            (kind_input >= RK_KICKOFF) & (kind_input < RESTART_COUNT)
        )
        kind = jnp.where(
            kind_valid,
            kind_input,
            jnp.asarray(RK_FREEKICK, dtype=kind_input.dtype),
        ).astype(jnp.int32)
        team = jnp.clip(team_input, TEAM_0, TEAM_1).astype(jnp.int32)

        # This geometry helper accepts ``False`` as its default even for
        # RK_OFFSIDE.  Keep that public signature: enforce scalar
        # bool type/rank, then derive the only legal value from the sanitized
        # kind.  The higher-level prepare/sync adapters retain their stricter
        # host-visible semantic-conflict rejection.
        indirect_traced = any(
            isinstance(leaf, jax.core.Tracer)
            for leaf in jax.tree_util.tree_leaves(restart_indirect)
        )
        if not indirect_traced:
            try:
                indirect_host = np.asarray(restart_indirect)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "restart_indirect must be a scalar bool"
                ) from exc
            if indirect_host.shape != ():
                raise ValueError(
                    "restart_indirect must be a scalar, "
                    f"got shape {indirect_host.shape}"
                )
            if not np.issubdtype(indirect_host.dtype, np.bool_):
                raise TypeError(
                    "restart_indirect must have bool dtype, "
                    f"got {indirect_host.dtype}"
                )
            indirect = jnp.asarray(bool(indirect_host), dtype=jnp.bool_)
        else:
            indirect = jnp.asarray(restart_indirect)
            if indirect.shape != ():
                raise ValueError(
                    "restart_indirect must be a scalar, "
                    f"got shape {indirect.shape}"
                )
            if indirect.dtype != jnp.dtype(jnp.bool_):
                raise TypeError(
                    "restart_indirect must have bool dtype, "
                    f"got {indirect.dtype}"
                )
        indirect = jnp.where(
            kind == RK_OFFSIDE,
            jnp.bool_(True),
            jnp.where(kind == RK_FREEKICK, indirect, jnp.bool_(False)),
        ).astype(jnp.bool_)
        return kind, team, indirect

    def _observed_gk_hold_spot_valid(
        self, state, restart_kind, restart_team, ball_pos
    ):
        """Whether an observed GK hold lies inside that team's own box."""

        safe_team = jnp.clip(restart_team, TEAM_0, TEAM_1)
        team_slot = jnp.where(safe_team == TEAM_0, 0, self.n_agents)
        attack = state.attack_dir[team_slot]
        in_own_box = self._in_own_box(ball_pos, attack, clamp_x=True)
        return (restart_kind != RK_GK_HOLD) | in_own_box

    @staticmethod
    def _reject_visible_invalid_gk_hold(valid):
        """Raise for eager playback; traced callers are handled as a no-op."""

        if not isinstance(valid, jax.core.Tracer) and not bool(np.asarray(valid)):
            raise ValueError(
                "RK_GK_HOLD ball position must lie inside the restart team's own "
                "penalty area"
            )

    def _observed_restart_terminal_guard(self, state):
        """Reject host mutation of terminal games; make traced adapters no-op."""

        terminal = self._is_terminal(state)
        if not isinstance(terminal, jax.core.Tracer) and bool(np.asarray(terminal)):
            raise ValueError("observed restart adapters cannot mutate a terminal state")
        return terminal

    def canonical_restart_spot(
        self,
        state,
        restart_kind,
        restart_team,
        observed_ball_pos=None,
        restart_indirect=False,
    ):
        """Return SoccerEnv's legal spot for an externally observed restart.

        Reconstruction supplies only the observed kind/team boundary.  Spot
        geometry remains owned by the environment so DFL adapters cannot grow
        a second, subtly different implementation of touchline/goal-area
        insets.
        """

        e_cfg = self.e_cfg
        restart_kind, restart_team, restart_indirect = (
            self._sanitize_canonical_restart_inputs(
                restart_kind, restart_team, restart_indirect
            )
        )
        observed = (
            state.ball_pos
            if observed_ball_pos is None
            else self._canonical_observed_ball_pos(observed_ball_pos)
        )
        if observed.shape != (DIM_ALL,):
            raise ValueError(
                f"observed_ball_pos must have shape ({DIM_ALL},), got {observed.shape}"
            )
        observed = jnp.where(jnp.all(jnp.isfinite(observed)), observed, state.ball_pos)
        inset = e_cfg.restart_field_inset
        safe_team = jnp.clip(restart_team, TEAM_0, TEAM_1)
        team_slot = jnp.where(safe_team == TEAM_0, 0, self.n_agents)
        attack = state.attack_dir[team_slot]
        # An observed point exactly on the pitch centreline contains no side
        # information.  Resolve that genuine tie in the restart team's frame;
        # a fixed ``+DIV_EPS`` fallback would map both a state and its
        # 180-degree/team-swapped counterpart to +y.  Nonzero observations
        # retain their actual side exactly.
        observed_side = jnp.where(
            observed[DIM_Y] != 0.0,
            jnp.sign(observed[DIM_Y]),
            attack,
        )
        center = jnp.asarray([0.0, 0.0, self.r_ball])
        throwin = jnp.asarray([
            jnp.clip(observed[DIM_X], -self.hx + inset, self.hx - inset),
            observed_side * (self.hy - e_cfg.throwin_line_inset),
            self.r_ball,
        ])
        corner = jnp.asarray([
            attack * (self.hx - inset),
            observed_side * (self.hy - inset),
            self.r_ball,
        ])
        goalkick = jnp.asarray([
            -attack * (self.hx - e_cfg.goalkick_depth),
            observed_side * e_cfg.goalkick_lateral_offset,
            self.r_ball,
        ])
        penalty = jnp.asarray([
            attack * (self.hx - e_cfg.penalty_spot),
            0.0,
            self.r_ball,
        ])
        # A catch hold is not a marked restart spot.  Forward dynamics place
        # the stationary ball at the goalkeeper's catch x/y, so observed
        # playback must preserve that location (after the own-box guard) rather
        # than applying the generic free-kick boundary inset.
        gk_hold = jnp.asarray([
            observed[DIM_X], observed[DIM_Y], self.r_ball,
        ])
        free = jnp.asarray([
            jnp.clip(
                observed[DIM_X],
                -self.hx + e_cfg.free_kick_boundary_inset,
                self.hx - e_cfg.free_kick_boundary_inset,
            ),
            jnp.clip(
                observed[DIM_Y],
                -self.hy + e_cfg.free_kick_boundary_inset,
                self.hy - e_cfg.free_kick_boundary_inset,
            ),
            self.r_ball,
        ])
        # Only an attacking IDFK against the defending team receives Law
        # 13's goal-area-line projection.  RK_OFFSIDE is a defending-team
        # free kick and therefore keeps the ordinary offence spot.
        offender_attack = -attack
        attacking_idfk = self._attacking_indirect_fk_spot(
            observed[:DIM_Z], offender_attack
        )
        free = jnp.where(
            (restart_kind == RK_FREEKICK)
            & jnp.asarray(restart_indirect, dtype=bool),
            attacking_idfk,
            free,
        )
        return jnp.where(
            restart_kind == RK_KICKOFF,
            center,
            jnp.where(
                restart_kind == RK_THROWIN,
                throwin,
                jnp.where(
                    restart_kind == RK_CORNER,
                    corner,
                    jnp.where(
                        restart_kind == RK_GOALKICK,
                        goalkick,
                        jnp.where(
                            restart_kind == RK_PENALTY,
                            penalty,
                            jnp.where(restart_kind == RK_GK_HOLD, gk_hold, free),
                        ),
                    ),
                ),
            ),
        )

    def prepare_observed_restart(
        self,
        state,
        restart_kind,
        restart_team,
        pending_taker,
        restart_t,
        observed_ball_pos=None,
        restart_indirect=None,
    ):
        """Initialize one observed referee boundary without advancing physics.

        This pure adapter is used by offline reconstruction exactly once at a
        DFL referee-boundary onset. Player positions are not teleported: the
        normal restart path walks the designated taker while submitted policy
        actions control every other active player. Goalkeeper possession is
        the live-ball exception; all set-piece restarts remain dead until
        release. Random or simulated outs remain independently suppressible.
        """

        terminal = self._observed_restart_terminal_guard(state)
        if observed_ball_pos is not None:
            observed_ball_pos = self._canonical_observed_ball_pos(
                observed_ball_pos
            )
        restart_kind, restart_team, pending_taker, restart_t = self._sanitize_observed_restart(
            state, restart_kind, restart_team, pending_taker, restart_t
        )
        restart_indirect = self._sanitize_observed_restart_indirect(
            restart_kind, restart_indirect
        )
        hold_ball_pos = (
            state.ball_pos
            if observed_ball_pos is None
            else observed_ball_pos
        )
        hold_spot_valid = self._observed_gk_hold_spot_valid(
            state, restart_kind, restart_team, hold_ball_pos
        )
        self._reject_visible_invalid_gk_hold(hold_spot_valid)
        hold_taker_valid = (restart_kind != RK_GK_HOLD) | (pending_taker >= 0)
        spot = self.canonical_restart_spot(
            state,
            restart_kind,
            restart_team,
            observed_ball_pos,
            restart_indirect,
        )
        # 재개팀에 활성 선수가 전무하면 sanitizer의 폴백도 ``NO_PLAYER``(-1)를 낸다. 그 값을
        # 그대로 scatter 인덱스로 쓰면 JAX가 마지막 슬롯으로 해석해 **상대팀 선수의**
        # contact lock을 지운다 — 이 어댑터가 지키기로 한 '범위 밖 슬롯을 건드리지 않는다'는
        # 계약을 정면으로 어긴다. 인덱스를 안전화하고 쓰기 자체를 taker 존재 여부로 닫는다.
        safe_taker = jnp.clip(pending_taker, 0, self.N - 1)
        taker_contact_lock = jnp.where(
            pending_taker >= 0, jnp.int32(0), state.contact_lock_t[safe_taker])
        prepared = state._replace(
            ball_pos=spot,
            ball_vel=jnp.zeros(DIM_ALL),
            ball_spin=jnp.zeros(DIM_ALL),
            ball_state=jnp.where(
                restart_kind == RK_GK_HOLD,
                jnp.int32(BALL_ALIVE),
                jnp.int32(BALL_DEAD),
            ),
            poss_team=restart_team,
            last_touch_team=restart_team,
            # Administrative restart ownership is not a player touch.  Keep
            # the team/code pair honest and prevent an old PASS/DRIBBLE from
            # becoming a false goalkeeper back-pass after a timeout fallback.
            last_touch_code=jnp.int32(TOUCH_NONE),
            last_touch_actor=jnp.int32(NO_PLAYER),
            gk_handling_restricted_team=jnp.int32(NO_TEAM),
            restart_team=restart_team,
            restart_t=restart_t,
            restart_kind=restart_kind,
            pending_taker=pending_taker,
            contact_lock_t=state.contact_lock_t.at[safe_taker].set(taker_contact_lock),
            setpiece_taker=jnp.int32(NO_PLAYER),
            throw_taker=jnp.int32(NO_PLAYER),
            offside_flag=jnp.zeros_like(state.offside_flag),
            pass_team=jnp.int32(NO_TEAM),
            pass_t=jnp.int32(0),
            foul_kind=jnp.int32(FOUL_NONE),
            foul_actor=jnp.int32(NO_PLAYER),
            foul_victim=jnp.int32(NO_PLAYER),
            restart_indirect=restart_indirect,
        )
        prepared = self._normalize_pass_latch(prepared, clear=True)
        blocked = terminal | (~hold_spot_valid) | (~hold_taker_valid)
        return jax.tree_util.tree_map(
            lambda new, old: jnp.where(blocked, old, new), prepared, state
        )

    def synchronize_observed_restart(
        self,
        state,
        restart_kind,
        restart_team,
        pending_taker,
        restart_t,
        restart_indirect=None,
    ):
        """Pin only the observed referee phase/timer until physical release."""

        terminal = self._observed_restart_terminal_guard(state)
        restart_kind, restart_team, pending_taker, restart_t = self._sanitize_observed_restart(
            state, restart_kind, restart_team, pending_taker, restart_t
        )
        requested_indirect = self._sanitize_observed_restart_indirect(
            restart_kind, restart_indirect
        )
        # ``synchronize`` pins an already observed phase; ``None`` therefore
        # means "no new provenance observation", not "coerce this FK to
        # direct".  The latter silently changed an IDFK prepared at onset into
        # a direct FK on the very next playback frame.  A restart instance can
        # only count down (or hold), so the shared reopened predicate also
        # distinguishes a new same-kind/team phase whose timer reset upward.
        incoming_phase = state._replace(
            restart_kind=restart_kind,
            restart_team=restart_team,
            restart_t=restart_t,
        )
        same_phase = ~self.restart_reopened(
            state.restart_t,
            state.restart_kind,
            state.restart_team,
            incoming_phase,
        )
        preserve_fk_provenance = same_phase & (restart_kind == RK_FREEKICK)
        if restart_indirect is not None:
            provenance_conflict = (
                preserve_fk_provenance
                & (requested_indirect != state.restart_indirect)
            )
            if (
                not isinstance(provenance_conflict, jax.core.Tracer)
                and bool(np.asarray(provenance_conflict))
            ):
                raise ValueError(
                    "restart_indirect cannot change within one observed "
                    "free-kick phase"
                )
        restart_indirect = jnp.where(
            preserve_fk_provenance,
            state.restart_indirect,
            requested_indirect,
        ).astype(jnp.bool_)
        hold_spot_valid = self._observed_gk_hold_spot_valid(
            state, restart_kind, restart_team, state.ball_pos
        )
        self._reject_visible_invalid_gk_hold(hold_spot_valid)
        hold_taker_valid = (restart_kind != RK_GK_HOLD) | (pending_taker >= 0)
        synchronized = state._replace(
            ball_state=jnp.where(
                restart_kind == RK_GK_HOLD,
                jnp.int32(BALL_ALIVE),
                jnp.int32(BALL_DEAD),
            ),
            gk_handling_restricted_team=jnp.int32(NO_TEAM),
            restart_team=restart_team,
            restart_t=restart_t,
            restart_kind=restart_kind,
            pending_taker=pending_taker,
            restart_indirect=restart_indirect,
        )
        synchronized = self._normalize_pass_latch(synchronized, clear=True)
        blocked = terminal | (~hold_spot_valid) | (~hold_taker_valid)
        return jax.tree_util.tree_map(
            lambda new, old: jnp.where(blocked, old, new), synchronized, state
        )

    def _restart_clearance_spot(self, state):
        """Return the Laws-of-the-Game reference point for radial clearance.

        The simulated throw-in ball is stored slightly inside the pitch so a
        legal inward release is not immediately classified as another out.
        Law 15, however, measures the opponents' 2 m distance from the point
        *on the touchline* where the throw is taken.  Keeping those two points
        distinct prevents the numerical inset from weakening the rule for a
        player waiting outside the line.

        A corner uses the corner flag as the centre of an equivalent on-pitch
        radial boundary; its radius adds the corner-arc radius separately in
        ``_encroach_geometry``.  Other radial restarts use the stored ball
        position.  Zero-coordinate fallbacks are defensive for externally
        reconstructed states and preserve 180-degree/team-swap covariance.
        """

        ball = state.ball_pos[:DIM_Z]
        team_slot = jnp.where(state.restart_team == TEAM_0, 0, self.n_agents)
        fallback_side = state.attack_dir[team_slot]
        touchline_side = jnp.where(
            ball[DIM_Y] != 0.0,
            jnp.sign(ball[DIM_Y]),
            fallback_side,
        )
        throw_point = jnp.asarray([ball[DIM_X], touchline_side * self.hy])
        corner_x_side = jnp.where(
            ball[DIM_X] != 0.0,
            jnp.sign(ball[DIM_X]),
            fallback_side,
        )
        corner_point = jnp.asarray([
            corner_x_side * self.hx,
            touchline_side * self.hy,
        ])
        return jnp.where(
            state.restart_kind == RK_THROWIN,
            throw_point,
            jnp.where(state.restart_kind == RK_CORNER, corner_point, ball),
        )

    def _encroach_geometry(self, state):
        """세트피스 제한구역 판정 기하(단일 진실원천) — 위치 투영과 obs 인코딩이 공유.
        render_rich의 제한구역 오버레이도 이 기하 규약을 따른다(numpy 측 복제 — light 렌더엔 오버레이 없음).
        재개 활성 게이팅은 호출자 책임(여기는 순수 기하만).
        반환:
          clear_r     : 규정 이격 반경(m) — 킥오프=센터서클 / 스로인=throwin_clear /
                        코너=corner_arc_radius+clear_dist / 그 외 clear_dist.
                        코너의 합성 반경은 피치 안쪽에서 '코너 아크로부터 9.15m'와 동치다.
          encroachers : bool[N] 침범자.
                        · 골킥: 상대팀 & 온피치 & 재개팀 자기 박스 안 — 반경 조항 없음
                          (IFAB Law 16은 '박스 밖'만 요구. 반경 9.15 추가 요구는 과엄격이라
                          박스 밖 8m 합법 압박을 오판하지 않음)
                        · 킥오프: 키커 외 전원은 자기 진영, 상대는 추가로 센터서클 밖
                        · 페널티: 키커 외 필드 선수는 공격 끝 박스+아크 밖,
                          수비GK는 자기 골라인 밴드 안(이탈 시 침범; ★양 팀 걸침)
                        · GK_HOLD: 침범자 없음(상대는 '도전 불가'일 뿐 이격 의무 없음)
                        · 자기 페널티구역 안 FK/오프사이드 IDFK: 상대팀은 반경 밖 **그리고**
                          페널티구역 밖(IFAB Law 13)
                        · 그 외: 상대팀 & 온피치 & 반경 안 & **자기 골라인 밴드 제외**(IFAB Law 13)
          margin      : float[N] 규정 준수까지의 서명 여유거리(m) — 음수=침범 깊이, 양수=여유.
                        골킥=자기 박스 이탈 sd / 자기 박스 FK=박스·반경 여유의 최솟값 /
                        페널티 필드선수=공격 박스+아크 이탈 sd /
                        페널티 수비GK=골라인 밴드 잔여 margin /
                        골라인 면제자와 GK_HOLD는 Engine의 양의 margin floor로 클램프.
        ★주의: 페널티 박스/아크 의무자는 '키커·수비GK 제외 전원'이고, 수비GK는
          대신 골라인 의무를 가진다. obs 인코딩에서
          기존 골킥용 is_def(수비팀만) 마스크를 페널티에 그대로 쓰면 공격수 쇄도를 놓친다."""
        e_cfg = self.e_cfg
        s_cfg = self.s_cfg
        pen_hw = s_cfg.penalty_area_width / 2.0
        pen_len = s_cfg.penalty_area_length
        hx = self.hx
        clear_r = jnp.where(
            state.restart_kind == RK_KICKOFF,
            s_cfg.center_circle_radius,
            jnp.where(
                state.restart_kind == RK_THROWIN,
                e_cfg.throwin_clear,
                jnp.where(
                    state.restart_kind == RK_CORNER,
                    e_cfg.clear_dist + s_cfg.corner_arc_radius,
                    e_cfg.clear_dist,
                ),
            ),
        )
        clearance_spot = self._restart_clearance_spot(state)
        d_spot = jnp.linalg.norm(
            state.player_pos - clearance_spot[None, :], axis=1
        )
        px, py = state.player_pos[:, 0], state.player_pos[:, 1]
        # adir_rt = 재개팀의 공격 방향(골킥이면 수비팀, 페널티면 공격팀 — restart_kind에 따라 팀 역할이 다름).
        adir_rt = jnp.where(state.restart_team == 0, state.attack_dir[0], state.attack_dir[self.n_agents])

        # 박스 서명거리 헬퍼: x∈[x_lo,x_hi] & |py|≤pen_hw 직사각형. 내부 = -(최근접 경계 깊이), 외부 = 경계 유클리드.
        def _box_signed(goal_x):
            box_back = goal_x - jnp.sign(goal_x) * pen_len   # 골라인에서 필드 안쪽으로 pen_len
            x_lo = jnp.minimum(goal_x, box_back); x_hi = jnp.maximum(goal_x, box_back)
            in_box = (x_lo <= px) & (px <= x_hi) & (jnp.abs(py) <= pen_hw)
            depth_in = jnp.minimum(jnp.minimum(px - x_lo, x_hi - px), pen_hw - jnp.abs(py))
            dx_out = jnp.maximum(jnp.maximum(x_lo - px, px - x_hi), 0.0)
            dy_out = jnp.maximum(jnp.abs(py) - pen_hw, 0.0)
            sd = jnp.where(in_box, -depth_in, jnp.sqrt(dx_out ** 2 + dy_out ** 2))
            return in_box, sd

        # 골킥: 재개팀 '자기' 박스(수비 끝, -adir_rt*hx). 페널티: '공격 끝' 박스(상대 골대, +adir_rt*hx).
        gk_in_box, gk_sd = _box_signed(-adir_rt * hx)
        pen_in_box, pen_sd = _box_signed(adir_rt * hx)
        # IFAB Law 13 adds a second constraint when the defending team takes a
        # free kick inside its own penalty area: every opponent must remain
        # outside that area until the ball is in play.  The ordinary 9.15 m
        # circle alone is insufficient near the goal-area end, where a player
        # can be more than 9.15 m from the ball while still deep inside the
        # penalty area.  Offside restarts are indirect free kicks and share
        # the same procedure.
        restart_own_goal_x = -adir_rt * hx
        restart_own_front_x = restart_own_goal_x + adir_rt * pen_len
        restart_own_lo = jnp.minimum(restart_own_goal_x, restart_own_front_x)
        restart_own_hi = jnp.maximum(restart_own_goal_x, restart_own_front_x)
        spot_in_restart_own_box = (
            (clearance_spot[DIM_X] >= restart_own_lo)
            & (clearance_spot[DIM_X] <= restart_own_hi)
            & (jnp.abs(clearance_spot[DIM_Y]) <= pen_hw)
        )
        is_box_freekick = (
            ((state.restart_kind == RK_FREEKICK)
             | (state.restart_kind == RK_OFFSIDE))
            & spot_in_restart_own_box
        )

        # 페널티 면제자: 키커(pending_taker) + 수비팀 GK(골라인 잔류 허용). 그 외 전원이 박스 밖이어야 함.
        ar = self.player_indices
        taker_in_range = (
            (state.pending_taker >= 0) & (state.pending_taker < self.N)
        )
        safe_taker = jnp.clip(state.pending_taker, 0, self.N - 1)
        valid_kicker = (
            taker_in_range
            & state.active_player[safe_taker]
            & (state.team_id[safe_taker] == state.restart_team)
        )
        is_kicker = (ar == state.pending_taker) & valid_kicker
        defending_team = 1 - state.restart_team
        is_def_gk = (state.gk_indices == 1) & (state.team_id == defending_team)
        # IFAB Law 8: at kick-off every non-kicker remains in their own half;
        # opponents additionally remain outside the centre circle.  The
        # halfway line itself is legal, hence the strict negative-margin test.
        own_half_margin = -px * state.attack_dir
        ko_opponent = state.team_id != state.restart_team
        ko_subject = state.active_player & (~is_kicker)
        ko_encroach = ko_subject & (
            (own_half_margin < 0.0)
            | (ko_opponent & (d_spot < s_cfg.center_circle_radius))
        )
        ko_margin = jnp.where(
            ko_opponent,
            jnp.minimum(own_half_margin, d_spot - s_cfg.center_circle_radius),
            own_half_margin,
        )
        # IFAB Law 14: 박스 밖 **그리고** 스폿 9.15m(페널티 아크) 밖이어야 규정 준수 —
        # 박스 밖이라도 아크 안(박스 모서리 바깥 초승달 지대) 대기는 침범.
        pen_arc = d_spot < s_cfg.penalty_arc_radius
        # Law 14 also requires every non-kicker other than the defending
        # goalkeeper to remain *behind* the penalty mark.  Box+arc alone is
        # insufficient: a player can stand laterally outside the penalty-area
        # width, more than 9.15 m from the spot, yet several metres closer to
        # goal than the ball.  Positive signed margin means the player is on
        # the legal (opposite-goal) side of the mark.
        behind_mark_margin = (
            (state.ball_pos[DIM_X] - px) * adir_rt
        )
        ahead_of_mark = behind_mark_margin <= 0.0
        # Law 13의 일반 수비수는 실제 골라인과 교차해야 하지만, Law 14의 페널티
        # GK는 골라인 뒤도 합법이다. 같은 절댓값 밴드를 두 조항에 쓰면 뒤쪽 GK를
        # 오판하고, 같은 방향성 반평면을 쓰면 골라인 뒤 일반 수비수까지 면제한다.
        own_gx = -state.attack_dir * hx
        goal_line_depth = (px - own_gx) * state.attack_dir
        between_posts = (
            jnp.abs(py) <= s_cfg.goal_width / 2.0 + e_cfg.goal_post_tolerance
        )
        on_goal_line = (
            (jnp.abs(goal_line_depth) <= e_cfg.goal_line_tolerance)
            & between_posts
        )
        penalty_gk_legal = (
            (goal_line_depth <= e_cfg.goal_line_tolerance)
            & between_posts
        )
        # 페널티 침범: 비-GK는 박스·아크 밖 의무, 수비 GK는 '골라인 이탈' 시 침범(Law 14의 GK
        # 골라인 의무 — 전면 면제하면 GK가 스폿 옆에 주차해도 합법이 된다). 키커는 면제.
        pen_encroach = ((((pen_in_box | pen_arc | ahead_of_mark) & (~is_def_gk))
                         | (is_def_gk & (~penalty_gk_legal)))) \
                       & state.active_player & (~is_kicker)

        margin_r = d_spot - clear_r
        is_pen = state.restart_kind == RK_PENALTY
        is_gkk = state.restart_kind == RK_GOALKICK
        is_ko = state.restart_kind == RK_KICKOFF
        is_def = state.team_id != state.restart_team
        # 수비 GK의 페널티 의무는 박스/아크 이탈이 아니라 골라인 밴드 안에 남는 것이다.
        # 반환 margin도 같은 의미여야 obs가 실제 침범 판정을 설명한다.
        goal_line_margin = jnp.minimum(
            e_cfg.goal_line_tolerance - goal_line_depth,
            s_cfg.goal_width / 2.0 + e_cfg.goal_post_tolerance - jnp.abs(py),
        )
        outfield_pen_margin = jnp.minimum(
            jnp.minimum(pen_sd, d_spot - s_cfg.penalty_arc_radius),
            behind_mark_margin,
        )
        pen_margin = jnp.where(is_def_gk, goal_line_margin, outfield_pen_margin)
        ordinary_margin = jnp.where(
            is_box_freekick & is_def,
            jnp.minimum(margin_r, gk_sd),
            margin_r,
        )
        margin = jnp.where(
            is_pen,
            pen_margin,
            jnp.where(is_ko, ko_margin,
                      jnp.where(is_gkk, gk_sd, ordinary_margin)),
        )   # 골킥=박스 서명거리만(Law 16)
        margin = jnp.where((~is_pen) & is_def & on_goal_line,
                           jnp.maximum(margin, e_cfg.legal_margin_floor), margin)
        # GK 홀드는 이격 의무가 없다(상대는 '도전 불가'일 뿐) — 기본 clear_dist가 흘러들면
        # obs가 존재하지 않는 9.15m 후퇴 의무를 신호(음수 margin·any_enc)하게 된다.
        is_hold = state.restart_kind == RK_GK_HOLD
        margin = jnp.where(
            is_hold, jnp.maximum(margin, e_cfg.unrestricted_margin), margin
        )
        encroachers = jnp.where(
            is_pen,
            pen_encroach,
            jnp.where(
                is_ko,
                ko_encroach,
                is_def & state.active_player & (~on_goal_line) & (~is_hold)
                & jnp.where(
                    is_gkk,
                    gk_in_box,
                    (d_spot < clear_r) | (is_box_freekick & gk_in_box),
                ),
            ),
        )
        return clear_r, encroachers, margin

    def _nearest_radial_legal_points(
        self, player_pos, center, radius, fallback_direction
    ):
        """Return nearest points outside a circle, constrained to the 5 m player box.

        The unconstrained answer is the radial projection.  Near touchlines and
        corners that point can lie outside the legal player margin, so circle/box
        intersections are included as exact constrained candidates.  This keeps the
        referee displacement minimal instead of clipping a radial point back inside
        the forbidden circle.
        """

        bx = self.hx + self.e_cfg.player_boundary_margin
        by = self.hy + self.e_cfg.player_boundary_margin
        rel = player_pos - center[None, :]
        dist = jnp.linalg.norm(rel, axis=1)
        # A deterministic direction is required for player==spot.  Its order
        # belongs to the player's within-team identity and its world bearing
        # to that team's attacking frame; global slot golden angles do not
        # rotate when two even-sized roster halves are exchanged.
        fallback = jnp.asarray(fallback_direction, dtype=player_pos.dtype)
        if fallback.shape != player_pos.shape:
            raise ValueError(
                "fallback_direction must match player_pos shape, got "
                f"{fallback.shape} and {player_pos.shape}"
            )
        direction = jnp.where(
            (dist > GEOMETRY_EPS)[:, None], rel / (dist[:, None] + DIV_EPS), fallback
        )
        radial = center[None, :] + direction * radius
        radial_valid = (
            (jnp.abs(radial[:, DIM_X]) <= bx + GEOMETRY_EPS)
            & (jnp.abs(radial[:, DIM_Y]) <= by + GEOMETRY_EPS)
        )

        # ``stack(..., axis=1).reshape(4, 2)``는 행을 **모서리별로** 묶는다
        # (edge0+root, edge0-root, edge1+root, edge1-root). 따라서 유효성도 모서리마다
        # 두 번 반복(``repeat``)해야 행과 정렬된다. ``tile``을 쓰면 [v0, v1, v0, v1]이 되어
        # 1·2번 행이 서로의 유효성을 갖는다 — 원이 닿지도 않는 모서리의 퇴화점(root=0)이
        # 유효로 둔갑하고 진짜 최근접점은 버려져, 코너·스로인 근처 침범자가 반대편
        # 터치라인으로 수십 미터 순간이동한다.
        x_edges = jnp.asarray([-bx, bx])
        x_rad = radius ** 2 - (x_edges - center[DIM_X]) ** 2
        x_root = jnp.sqrt(jnp.maximum(x_rad, 0.0))
        x_points = jnp.stack([
            jnp.stack([x_edges, center[DIM_Y] + x_root], axis=1),
            jnp.stack([x_edges, center[DIM_Y] - x_root], axis=1),
        ], axis=1).reshape(4, 2)
        x_valid = jnp.repeat(x_rad >= 0.0, 2) & (jnp.abs(x_points[:, DIM_Y]) <= by + GEOMETRY_EPS)

        y_edges = jnp.asarray([-by, by])
        y_rad = radius ** 2 - (y_edges - center[DIM_Y]) ** 2
        y_root = jnp.sqrt(jnp.maximum(y_rad, 0.0))
        y_points = jnp.stack([
            jnp.stack([center[DIM_X] + y_root, y_edges], axis=1),
            jnp.stack([center[DIM_X] - y_root, y_edges], axis=1),
        ], axis=1).reshape(4, 2)
        y_valid = jnp.repeat(y_rad >= 0.0, 2) & (jnp.abs(y_points[:, DIM_X]) <= bx + GEOMETRY_EPS)

        edge_points = jnp.concatenate([x_points, y_points], axis=0)
        edge_valid = jnp.concatenate([x_valid, y_valid], axis=0)
        candidates = jnp.concatenate(
            [radial[:, None, :], jnp.broadcast_to(edge_points[None, :, :], (self.N, 8, 2))],
            axis=1,
        )
        valid = jnp.concatenate(
            [radial_valid[:, None], jnp.broadcast_to(edge_valid[None, :], (self.N, 8))],
            axis=1,
        )
        sq = jnp.sum((candidates - player_pos[:, None, :]) ** 2, axis=2)
        choice = jnp.argmin(jnp.where(valid, sq, jnp.inf), axis=1)
        return candidates[self.player_indices, choice]


    def _spread_shared_radial_targets(self, target, encroachers, center):
        """같은 좌표로 몰린 침범자들을 원호를 따라 최소거리씩 흩뜨린다.

        ``_nearest_radial_legal_points``의 모서리 후보 8개는 **전 선수 공용**이다. 반경
        투영점이 5 m 박스를 벗어나는 침범자(코너 깃발 바깥쪽 등)는 전부 같은 모서리점을
        고르므로 문자 그대로 같은 좌표에 겹쳐 놓인다(실측: 두 슬롯이 정확히
        (-44.680, -39.000)). 그 점은 박스 경계 위라 접선 한쪽이 막혀 있어 사후 분리로도
        풀리지 않는다 — 애초에 서로 다른 지점을 주는 수밖에 없다.

        슬롯 순서대로 순위를 매겨 접선 방향으로 ``rank × min_d``만큼 민다. 접선 이동은
        현(chord)이라 반경이 늘어나므로 금지 원에 다시 들어가지 않고, 필드 중앙을 향하는
        부호를 골라 허용 경계 밖으로도 나가지 않는다.
        """

        min_d = 2.0 * self.r_player
        gap = target[:, None, :] - target[None, :, :]
        shared = (
            (jnp.sum(gap * gap, axis=2) < min_d ** 2)
            & encroachers[:, None] & encroachers[None, :]
        )
        order = self.player_indices
        rank = jnp.sum(shared & (order[None, :] < order[:, None]), axis=1).astype(jnp.float32)

        radial = target - center[None, :]
        radial = radial / (jnp.linalg.norm(radial, axis=1, keepdims=True) + DIV_EPS)
        tangent = jnp.stack([-radial[:, DIM_Y], radial[:, DIM_X]], axis=1)
        sign = jnp.where(
            jnp.sum(tangent * (-target), axis=1, keepdims=True) < 0.0, -1.0, 1.0)
        tdir = tangent * sign

        def room(axis, bound):
            step = tdir[:, axis]
            safe = jnp.where(jnp.abs(step) > DIV_EPS, step, 1.0)
            edge = jnp.where(step >= 0.0, bound, -bound)
            return jnp.where(
                jnp.abs(step) > DIV_EPS, (edge - target[:, axis]) / safe, jnp.inf)

        allowance = jnp.maximum(0.0, jnp.minimum(
            room(DIM_X, self.hx + self.e_cfg.player_boundary_margin),
            room(DIM_Y, self.hy + self.e_cfg.player_boundary_margin),
        ))
        offset = jnp.minimum(rank * min_d, allowance)
        return target + tdir * offset[:, None]

    def _spread_halfplane_targets(
        self, target, original, encroachers, active, away_direction
    ):
        """Place x-half-plane targets without stacking them on their boundary.

        Penalty behind-mark and kick-off own-half projection both map lateral
        violators to one x coordinate while preserving y.  A crowded group at
        the 5 m outer boundary can therefore become a one-dimensional stack
        that local separation cannot fully unwind in its configured rounds.
        Place encroachers in slot order at the nearest collision-free point in
        their legal x direction. ``away_direction`` may be one scalar
        (penalty) or one sign per slot (kick-off teams face opposite ways).
        """

        min_d = 2.0 * self.r_player
        floor = self.e_cfg.legal_margin_floor
        bound_x = self.hx + self.e_cfg.player_boundary_margin
        bound_y = self.hy + self.e_cfg.player_boundary_margin
        order = self.player_indices
        away = jnp.broadcast_to(
            jnp.asarray(away_direction, dtype=target.dtype), (self.N,)
        )
        # Non-encroachers are blockers at their real positions, not at the
        # unused rule target computed for their branch.
        placed = jnp.where(encroachers[:, None], target, original)

        def place_one(index, points):
            base = target[index]
            away_i = away[index]
            delta = base[None, :] - points
            along = delta[:, DIM_X] * away_i
            lateral = delta[:, DIM_Y]
            span = jnp.sqrt(jnp.maximum(min_d ** 2 - lateral ** 2, 0.0))
            # Move just beyond each prior blocker's forbidden interval.  The
            # zero candidate preserves the exact nearest target when already
            # clear.
            exits = jnp.maximum(0.0, span - along + floor)
            shifts = jnp.concatenate([jnp.zeros(1, dtype=target.dtype), exits])
            candidates = base[None, :] + jnp.stack(
                [away_i * shifts, jnp.zeros_like(shifts)], axis=1
            )
            prior = (order < index) & active
            distances = jnp.linalg.norm(
                candidates[:, None, :] - points[None, :, :], axis=2
            )
            clear = jnp.all(
                (~prior)[None, :] | (distances >= min_d), axis=1
            )
            inside = (
                (jnp.abs(candidates[:, DIM_X]) <= bound_x)
                & (jnp.abs(candidates[:, DIM_Y]) <= bound_y)
            )
            cost = jnp.where(clear & inside, shifts, jnp.inf)
            best = jnp.argmin(cost)
            chosen = jnp.where(jnp.isfinite(cost[best]), candidates[best], base)
            chosen = jnp.where(encroachers[index], chosen, points[index])
            return points.at[index].set(chosen)

        return jax.lax.fori_loop(0, self.N, place_one, placed)

    def _restart_projection_active(self, state):
        """재개 거리 투영이 활성인지를 판정하는 단일 진실원천."""

        return (
            restart_timer_active(state.restart_t)
            & (state.restart_kind >= RK_KICKOFF)
            & (state.restart_kind < RESTART_COUNT)
            & (state.restart_kind != RK_GK_HOLD)
        )

    def restart_reopened(self, before_t, before_kind, before_team, after):
        """``after``의 재개가 스냅샷 시점의 그것과 **다른 인스턴스**인가.

        단일 전이의 scalar 스냅샷과 render/replay time-axis의 동일-shape batch를
        모두 지원하며, 여섯 정수 field는 정확히 같은 shape여야 한다.

        킥은 의사결정 스텝의 산물이므로, control frame 안에서 막 생긴 재개는 정책이 아직
        관측하지 못했고 이번 frame에 소비되면 안 된다. 그 판정의 단일 진실원천이다.

        종류 비교만으로는 부족하다 — 스로인을 던지자마자 공이 다시 나가면 같은 종류·같은
        타이머 창의 재개로 갈아탄다(실측: ``rt 184 -> 450``, THROWIN -> THROWIN, 키커 20 -> 5).
        그 경우 ``before_t <= 0``도 ``kind != before_kind``도 거짓이라 새 재개가 아닌 것으로
        판정되어 게이트를 통째로 우회했다.

        재개 타이머는 한 인스턴스 안에서 **절대 늘지 않는다** — ``events``의 카운트다운은
        ``max(0, t - 1)``이거나 키커 미도착 시 정지이고, 전체 창으로 되감기는 것은 새 재개
        이벤트뿐이다. 따라서 되감김이 곧 새 인스턴스이며, 재개팀 교체도 마찬가지다.
        """

        max_restart_t = max(
            self.e_cfg.restart_substeps,
            self.e_cfg.penalty_substeps,
            self.e_cfg.gk_hold_substeps,
        )
        before_t, valid_before_t = _coerce_public_signed_int(
            "before_t", before_t,
            minimum=0, maximum=max_restart_t, fallback=0,
        )
        before_kind, valid_before_kind = _coerce_public_signed_int(
            "before_kind", before_kind,
            minimum=RK_NONE, maximum=RESTART_COUNT - 1, fallback=RK_NONE,
        )
        before_team, valid_before_team = _coerce_public_signed_int(
            "before_team", before_team,
            minimum=NO_TEAM, maximum=TEAM_COUNT - 1, fallback=NO_TEAM,
        )
        after_t, valid_after_t = _coerce_public_signed_int(
            "after.restart_t", after.restart_t,
            minimum=0, maximum=max_restart_t, fallback=0,
        )
        after_kind, valid_after_kind = _coerce_public_signed_int(
            "after.restart_kind", after.restart_kind,
            minimum=RK_NONE, maximum=RESTART_COUNT - 1, fallback=RK_NONE,
        )
        after_team, valid_after_team = _coerce_public_signed_int(
            "after.restart_team", after.restart_team,
            minimum=NO_TEAM, maximum=TEAM_COUNT - 1, fallback=NO_TEAM,
        )
        snapshot_shape = before_t.shape
        named_shapes = (
            ("before_kind", before_kind.shape),
            ("before_team", before_team.shape),
            ("after.restart_t", after_t.shape),
            ("after.restart_kind", after_kind.shape),
            ("after.restart_team", after_team.shape),
        )
        for name, actual_shape in named_shapes:
            if actual_shape != snapshot_shape:
                raise ValueError(
                    "restart snapshots must have one exact common shape; "
                    f"before_t has {snapshot_shape}, {name} has {actual_shape}"
                )
        valid_snapshot = (
            valid_before_t & valid_before_kind & valid_before_team
            & valid_after_t & valid_after_kind & valid_after_team
        )
        reopened = restart_timer_active(after_t) & (
            (~restart_timer_active(before_t))
            | (after_kind != before_kind)
            | (after_team != before_team)
            | (after_t > before_t)
        )
        # An invalid traced snapshot cannot prove phase continuity.  Treat it
        # as newly opened so the current policy frame is prevented from
        # consuming a restart it may not have observed.
        return (~valid_snapshot) | reopened

    def _restart_encroacher_mask(self, state):
        """현재 state에서 규칙 이격 때문에 직접 투영·동결될 슬롯만 반환한다.

        충돌 해결 과정에서 함께 밀리는 합법 선수는 포함하지 않는다. 그 선수의 정책 이동은
        투영 뒤에도 적용되며, 강제 위치 보정이 섞인 BC 라벨만 사후 mask한다.
        """

        return jax.lax.cond(
            self._restart_projection_active(state),
            lambda current: self._encroach_geometry(current)[1],
            lambda current: jnp.zeros(self.N, dtype=bool),
            state,
        )

    def _project_restart_positions(self, state):
        """Minimally enforce restart exclusion zones and return ``(state, mask)``.

        There are no ceremonial free kicks or encroachment retakes in this
        environment.  During every active restart, an illegal non-taker is moved to
        the nearest legal point.  Inside either penalty area, displacement is
        restricted to the x direction away from that goal; the defending penalty
        goalkeeper is the sole exception and is projected to the required goal-line
        band.  Elsewhere the nearest feasible point outside the exclusion circle is
        used.  Only velocity back into the forbidden region is removed.
        """

        def project(current):
            e_cfg, s_cfg = self.e_cfg, self.s_cfg

            def resolve(cur):
                clear_r, encroachers, _ = self._encroach_geometry(cur)
                clearance_spot = self._restart_clearance_spot(cur)
                pos = cur.player_pos
                px, py = pos[:, DIM_X], pos[:, DIM_Y]
                floor = e_cfg.legal_margin_floor
                target_radius = clear_r + floor
                radial_target = self._nearest_radial_legal_points(
                    pos,
                    clearance_spot,
                    target_radius,
                    self._covariant_slot_directions(current),
                )
                radial_target = self._spread_shared_radial_targets(
                    radial_target, encroachers, clearance_spot
                )

                # Kick-off is an intersection of two constraints, not merely
                # the old opponent-only centre-circle rule.  A team-mate who
                # crossed halfway needs only an x clamp.  An opponent who is
                # both across halfway and inside the circle is projected to
                # the nearest circle/half-plane intersection; radial-only or
                # half-only violations retain their respective minimal moves.
                own_half_margin = -px * current.attack_dir
                half_illegal = own_half_margin < 0.0
                ko_opponent = current.team_id != current.restart_team
                half_target_x = -current.attack_dir * floor
                half_target = jnp.stack([half_target_x, py], axis=1)
                ko_rel_y = py - current.ball_pos[DIM_Y]
                half_circle_dx = half_target_x - current.ball_pos[DIM_X]
                half_circle_span = target_radius ** 2 - half_circle_dx ** 2
                half_circle_y = jnp.sqrt(jnp.maximum(half_circle_span, 0.0))
                y_sign = jnp.where(
                    jnp.abs(ko_rel_y) > GEOMETRY_EPS,
                    jnp.sign(ko_rel_y),
                    self._covariant_slot_side(current),
                )
                half_circle_target = jnp.stack([
                    half_target_x,
                    current.ball_pos[DIM_Y] + y_sign * half_circle_y,
                ], axis=1)
                needs_intersection = (
                    ko_opponent & half_illegal
                    & (jnp.abs(ko_rel_y) < target_radius)
                    & (half_circle_span >= 0.0)
                )
                opponent_ko_target = jnp.where(
                    needs_intersection[:, None],
                    half_circle_target,
                    jnp.where(half_illegal[:, None], half_target, radial_target),
                )
                kickoff_target = jnp.where(
                    ko_opponent[:, None], opponent_ko_target, half_target
                )
                kickoff_target = self._spread_halfplane_targets(
                    kickoff_target,
                    pos,
                    encroachers,
                    current.active_player,
                    -current.attack_dir,
                )

                # Both penalty areas share the same intervention rule: move only toward
                # the opposite goal (away from the nearby goal), never sideways or deeper.
                in_right_box = (
                    (px >= self.hx - self.pen_len) & (px <= self.hx)
                    & (jnp.abs(py) <= self.pen_hw)
                )
                in_left_box = (
                    (px <= -self.hx + self.pen_len) & (px >= -self.hx)
                    & (jnp.abs(py) <= self.pen_hw)
                )
                in_either_box = in_left_box | in_right_box
                nearby_goal_sign = jnp.where(in_right_box, 1.0, -1.0)
                away_dir = -nearby_goal_sign
                rel_x = px - clearance_spot[DIM_X]
                rel_y = py - clearance_spot[DIM_Y]
                ray_root = jnp.sqrt(jnp.maximum(target_radius ** 2 - rel_y ** 2, 0.0))
                ray_step = jnp.maximum(0.0, -away_dir * rel_x + ray_root)
                box_radial_target = jnp.stack([px + away_dir * ray_step, py], axis=1)
                generic_target = jnp.where(in_either_box[:, None], box_radial_target, radial_target)

                team_slot = jnp.where(current.restart_team == TEAM_0, 0, self.n_agents)
                restart_attack = current.attack_dir[team_slot]

                # Goal kick: opponents leave the restart team's own box through its front
                # edge, along the direction toward the opposite goal.
                goal_x_gk = -restart_attack * self.hx
                gk_away = restart_attack
                gk_front = goal_x_gk + gk_away * self.pen_len
                goalkick_target = jnp.stack([
                    jnp.broadcast_to(gk_front + gk_away * floor, px.shape), py
                ], axis=1)

                # Law 13: a free kick (including an offside IDFK) taken by the
                # defending team inside its own penalty area requires both the
                # ordinary 9.15 m distance and departure from the area.  For a
                # player already inside that box both exits lie in the same
                # upfield direction, so the farther x target is the exact
                # minimum under this engine's penalty-area x-only intervention
                # rule.  Players outside the box retain the ordinary radial
                # projection.
                gk_lo = jnp.minimum(goal_x_gk, gk_front)
                gk_hi = jnp.maximum(goal_x_gk, gk_front)
                in_restart_own_box = (
                    (px >= gk_lo) & (px <= gk_hi)
                    & (jnp.abs(py) <= self.pen_hw)
                )
                spot_in_restart_own_box = (
                    (clearance_spot[DIM_X] >= gk_lo)
                    & (clearance_spot[DIM_X] <= gk_hi)
                    & (jnp.abs(clearance_spot[DIM_Y]) <= self.pen_hw)
                )
                is_box_freekick = (
                    ((current.restart_kind == RK_FREEKICK)
                     | (current.restart_kind == RK_OFFSIDE))
                    & spot_in_restart_own_box
                )
                box_exit_x = gk_front + gk_away * floor
                combined_box_x = jnp.where(
                    gk_away > 0.0,
                    jnp.maximum(generic_target[:, DIM_X], box_exit_x),
                    jnp.minimum(generic_target[:, DIM_X], box_exit_x),
                )
                box_freekick_target = jnp.where(
                    in_restart_own_box[:, None],
                    jnp.stack([combined_box_x, generic_target[:, DIM_Y]], axis=1),
                    generic_target,
                )

                # Penalty: outfield players must clear both box and arc.  Along the only
                # permitted direction, use whichever exit requires the larger movement.
                penalty_goal_x = restart_attack * self.hx
                penalty_away = -restart_attack
                penalty_front = penalty_goal_x + penalty_away * self.pen_len
                # Live-play movement permits players up to 5 m behind a goal.
                # At a newly awarded penalty those players are encroachers via
                # the behind-mark predicate.  Limiting this exit to the closed
                # goal-line..front rectangle moved them only to the mark side
                # and thereby *into* the penalty area.  The whole goal-side
                # half-ray within the area width needs the same x-only exit.
                pen_box_inside = (
                    (restart_attack * px >= restart_attack * penalty_front)
                    & (jnp.abs(py) <= self.pen_hw)
                )
                box_step = jnp.where(
                    pen_box_inside,
                    jnp.maximum(0.0, penalty_away * (penalty_front + penalty_away * floor - px)),
                    0.0,
                )
                # Arc geometry is ball-relative just like _encroach_geometry.  Observed
                # replay states may place the spot a few centimetres away from the
                # canonical coordinate, so projection must not introduce a second center.
                spot_x = current.ball_pos[DIM_X]
                spot_y = current.ball_pos[DIM_Y]
                arc_radius = s_cfg.penalty_arc_radius + floor
                arc_dx = px - spot_x
                arc_dy = py - spot_y
                arc_span = arc_radius ** 2 - arc_dy ** 2
                arc_root = jnp.sqrt(jnp.maximum(arc_span, 0.0))
                # 아크 탈출은 **현재 아크 안에 있는지**가 아니라 **이동 후 아크에 들어가는지**로
                # 판정해야 한다. 박스 탈출과 아크 탈출을 각자의 시작점에서 독립으로 계산하고
                # 최댓값을 쓰면, 아크 밖·박스 안이던 선수가 박스를 빠져나오면서 아크 안으로
                # 들어간다(실측: d_spot 9.72 → 8.39, margin -0.757). 진행 방향이 하나뿐이므로
                # 두 금지구간의 x-구간 탈출점 중 더 먼 쪽을 쓰면 합집합을 정확히 벗어난다.
                # ``max(0, ...)``가 이미 '이미 지나쳤으면 0'을 처리하므로 in_arc 게이트는 필요
                # 없고, 그 y에서 아크가 아예 없을 때(arc_span<=0)만 막으면 된다.
                arc_step = jnp.where(
                    arc_span > 0.0, jnp.maximum(0.0, -penalty_away * arc_dx + arc_root), 0.0
                )
                # The legal half-plane lies behind the mark, in
                # ``penalty_away``.  Keep the same minimal x-only intervention
                # used for penalty-box violations and add the numerical floor
                # on the legal side so the next predicate evaluation is stable.
                mark_target_x = spot_x + penalty_away * floor
                mark_step = jnp.maximum(
                    0.0, penalty_away * (mark_target_x - px)
                )
                penalty_step = jnp.maximum(jnp.maximum(box_step, arc_step), mark_step)
                penalty_outfield_target = jnp.stack(
                    [px + penalty_away * penalty_step, py], axis=1
                )

                # The defending goalkeeper has a different Law-14 constraint: restore
                # the nearest point in the goal-line band rather than moving upfield.
                defending_team = TEAM_1 - current.restart_team
                is_def_gk = (
                    (current.gk_indices == 1) & (current.team_id == defending_team)
                )
                own_goal_x = -current.attack_dir * self.hx
                line_inner = jnp.maximum(e_cfg.goal_line_tolerance - floor, 0.0)
                mouth_inner = jnp.maximum(
                    s_cfg.goal_width / 2.0 + e_cfg.goal_post_tolerance - floor, 0.0
                )
                gk_depth = (px - own_goal_x) * current.attack_dir
                gk_target_depth = jnp.minimum(gk_depth, line_inner)
                gk_line_target = jnp.stack([
                    own_goal_x + current.attack_dir * gk_target_depth,
                    jnp.clip(
                        py,
                        -mouth_inner,
                        mouth_inner,
                    ),
                ], axis=1)
                spread_penalty_outfield = self._spread_halfplane_targets(
                    penalty_outfield_target,
                    pos,
                    encroachers & (~is_def_gk),
                    current.active_player,
                    penalty_away,
                )
                penalty_target = jnp.where(
                    is_def_gk[:, None], gk_line_target, spread_penalty_outfield
                )

                target = jnp.where(
                    (current.restart_kind == RK_GOALKICK), goalkick_target,
                    jnp.where(
                        current.restart_kind == RK_PENALTY,
                        penalty_target,
                        jnp.where(
                            current.restart_kind == RK_KICKOFF,
                            kickoff_target,
                            jnp.where(
                                is_box_freekick,
                                box_freekick_target,
                                generic_target,
                            ),
                        ),
                    ),
                )
                # 제약면의 바깥 법선 — 재투영이 되돌리는 방향. 목표 지점 기준으로 잡아야
                # 한다. 코너처럼 원이 피치 밖으로 잘리면 투영 변위 방향은 법선이 아니다.
                zero = jnp.zeros_like(px)
                to_target = target - clearance_spot[None, :]
                radial_normal = to_target / (
                    jnp.linalg.norm(to_target, axis=1, keepdims=True) + DIV_EPS)
                box_normal = jnp.stack([away_dir, zero], axis=1)
                generic_normal = jnp.where(in_either_box[:, None], box_normal, radial_normal)
                gk_normal = jnp.stack([jnp.broadcast_to(gk_away, px.shape), zero], axis=1)
                box_freekick_normal = jnp.where(
                    in_restart_own_box[:, None], gk_normal, generic_normal
                )
                # 수비 골키퍼의 목표는 골라인 밴드로의 클립이라 단일 법선이 없다 —
                # 0 법선은 아래에서 '접선 제한 없음'으로 해석된다.
                pen_normal = jnp.where(
                    is_def_gk[:, None], 0.0,
                    jnp.stack([jnp.broadcast_to(penalty_away, px.shape), zero], axis=1))
                ko_half_normal = jnp.stack([-current.attack_dir, zero], axis=1)
                ko_normal = jnp.where(
                    half_illegal[:, None], ko_half_normal, radial_normal
                )
                normal = jnp.where(
                    current.restart_kind == RK_GOALKICK, gk_normal,
                    jnp.where(
                        current.restart_kind == RK_PENALTY,
                        pen_normal,
                        jnp.where(
                            current.restart_kind == RK_KICKOFF,
                            ko_normal,
                            jnp.where(
                                is_box_freekick,
                                box_freekick_normal,
                                generic_normal,
                            ),
                        ),
                    ),
                )
                return jnp.where(encroachers[:, None], target, pos), encroachers, normal

            pos = current.player_pos
            first, encroachers, normal = resolve(current)
            # 투영은 침범자마다 독립으로 계산되고, 서브스텝의 유일한 _separate **뒤에**
            # 호출된다. 그래서 두 침범자가 사실상 같은 지점으로 밀리면 그 겹침이 관측
            # 상태에 그대로 남는다(수정 전 실측: 두 슬롯이 8.4 m / 3.4 m 순간이동해 간격
            # 0.193 m < 0.46 m). 우선 침범자가 겹침을 흡수하되, 합법 영역의 바깥쪽을 이미
            # 비침범자가 막고 있으면 그 선수도 최소한으로 밀어야 한다. 그렇지 않으면 양립
            # 가능한 두 불변식(재개 이격·선수 비침투) 중 하나를 영구히 포기하게 된다.
            # 실제로 움직인 모든 슬롯을 반환 마스크에 넣으므로 BC는 이 규칙 개입을 정책
            # 이동으로 학습하지 않는다.
            taker = jnp.clip(current.pending_taker, 0, self.N - 1)
            valid_taker = (
                (current.pending_taker >= 0)
                & (current.pending_taker < self.N)
                & current.active_player[taker]
                & (current.team_id[taker] == current.restart_team)
            )
            taker_pinned = (
                (self.player_indices == taker)
                & valid_taker
            )

            def reproject(points):
                return resolve(current._replace(player_pos=points))

            new_pos = self.reconcile_positions(
                first, current.active_player,
                pinned=taker_pinned, constrained=encroachers, normal=normal,
                resolve=reproject,
                tie_direction=self._covariant_slot_directions(current),
            )

            # The bounded relaxed solver handles ordinary clusters, but a
            # legal near-contact chain can propagate farther than its fixed
            # sweep budget and leave a deep overlap whose location depends on
            # arbitrary slot order.  Detect only pairs causally connected to
            # this projection; unrelated live-play approximation remains out
            # of referee scope.  The global-placement fallback is therefore a
            # rare branch and has no cost on already reconciled restarts.
            moved_once = jnp.any(new_pos != pos, axis=1)
            causal = encroachers | moved_once
            delta = new_pos[:, None, :] - new_pos[None, :, :]
            residual = (
                (jnp.sum(delta * delta, axis=2)
                 < (2.0 * self.r_player - COINCIDENT_DISTANCE_EPS) ** 2)
                & current.active_player[:, None]
                & current.active_player[None, :]
                & (~jnp.eye(self.N, dtype=bool))
                & (causal[:, None] | causal[None, :])
            )
            new_pos = jax.lax.cond(
                jnp.any(residual),
                lambda points: self.repair_causal_overlaps(
                    points,
                    current.active_player,
                    causal,
                    taker_pinned,
                    encroachers,
                    reproject,
                    orientation=current.attack_dir,
                ),
                lambda points: points,
                new_pos,
            )

            displacement = new_pos - pos
            disp_norm = jnp.linalg.norm(displacement, axis=1)
            outward = displacement / (disp_norm[:, None] + DIV_EPS)
            inward_speed = jnp.sum(current.player_vel * outward, axis=1)
            forced = current.active_player & (disp_norm > GEOMETRY_EPS)
            new_vel = current.player_vel - jnp.where(
                (forced & (inward_speed < 0.0))[:, None],
                inward_speed[:, None] * outward,
                0.0,
            )
            new_facing = self.facing_from_velocity(new_vel, current.attack_dir)
            return current._replace(
                player_pos=new_pos, player_vel=new_vel, player_facing=new_facing
            ), forced

        return jax.lax.cond(
            self._restart_projection_active(state),
            project,
            lambda current: (current, jnp.zeros(self.N, dtype=bool)),
            state,
        )

    def _setpiece_kick_lock(self, state):
        """키커 강제이동/정렬 중 킥 불가 여부 판단.
        키커는 setup 완료 전까지 강제이동, 도착 후엔 제자리 고정. setup 완료 전까지 킥 불가.
        setup 완료: 도착 && 판정 후 종류별 준비시간 카운트다운 소진.
        Args:
            state: State
        Return: (전부 스칼라 bool — pending_taker 단일 인덱스 기준)
            kicker_locked: 키커 강제이동/정렬 중 킥 불가 여부
            setup_done: setup 완료 여부 (도착 && 카운트다운 소진)
            arrived: 키커 도착 여부
            active: 키커 강제이동/정렬 활성화 여부
        """
        e_cfg = self.e_cfg
        restart_active = restart_timer_active(state.restart_t)
        has_taker = (state.pending_taker >= 0) & (state.pending_taker < self.N)
        active = restart_active & has_taker            # 페널티도 포함(물리 플레이: 키커 스폿 접근·킥)
        taker = jnp.clip(state.pending_taker, 0, self.N - 1)
        valid_taker = (
            state.active_player[taker]
            & (state.team_id[taker] == state.restart_team)
        )
        # Invalid/inactive pending slots belong to the broken-taker repair
        # path; they must not be force-moved or counted as having arrived.
        active = active & valid_taker
        tgt = self._kicker_target(state)
        d_tgt = jnp.linalg.norm(state.player_pos[taker] - tgt)
        taker_in_reach = self._in_reach(state)[0][taker]
        # ``kicker_arrive_r`` is only an approach envelope.  The restart clock
        # and setup cannot advance until the taker can physically contact the
        # ball; otherwise an observed timer at 1 can expire into a live loose
        # ball without any kick.
        arrived = active & (d_tgt <= e_cfg.kicker_arrive_r) & taker_in_reach
        # 일반 재개와 페널티는 키커 도착 뒤 3초를, GK 홀드는
        # 별도 배급 prior를 쓴다. 접근 중에는 아래 events 경로가 시계를 얼린다.
        _window, ready_threshold = self._restart_window_ready_threshold(
            state.restart_kind
        )
        time_ready = state.restart_t <= ready_threshold
        setup_done = arrived & time_ready
        kicker_locked = active & (~setup_done)        # 도착 전 or 정렬 중 = 킥 불가
        return kicker_locked, setup_done, arrived, active

    def _advance_kicker_position(
        self, current, target, restart_kind, maximum_speed=None
    ):
        """Advance one forced-kicker physics tick using the runtime SSOT.

        Both :meth:`movement.Movement._apply_kicker_move` and the control-frame
        agency predictor use this primitive.  Keeping the floating-point
        expression shared matters at the arrival-radius boundary: an analytic
        ``ceil(distance / speed)`` predictor can disagree with the actual
        normalized step by one substep because the runtime protects division
        with ``DIV_EPS``.
        """

        to_target = target - current
        distance = jnp.linalg.norm(to_target) + DIV_EPS
        walk_speed = jnp.asarray(self.e_cfg.kicker_speed, dtype=current.dtype)
        if maximum_speed is not None:
            maximum_speed = jnp.asarray(maximum_speed, dtype=current.dtype)
            if maximum_speed.shape != ():
                raise ValueError(
                    f"maximum_speed must be scalar, got {maximum_speed.shape}"
                )
            walk_speed = jnp.minimum(walk_speed, jnp.maximum(maximum_speed, 0.0))
        walk = jnp.minimum(distance, walk_speed * self.e_cfg.dt_phys)
        # Kickoff is the sole instantaneous placement.  Every other restart
        # approaches at the configured physical speed.
        step_length = jnp.where(restart_kind == RK_KICKOFF, distance, walk)
        return current + to_target / distance * step_length

    def _setpiece_release_within_control_frame(self, state):
        """Whether the designated taker will be forced to release this frame.

        A control action is held across ``decimation`` physics ticks.  At each
        tick the runtime order is: forced kicker movement, setup test, kick
        opportunity, then restart/contact-timer decrement.  Looking only at
        the entry ``arrived`` or timer value therefore closes the action mask
        on frames that actually consume the restart.  This tiny scalar scan
        mirrors that order while sharing the exact movement primitive above.

        The result is action-independent.  It intentionally models only an
        already-open restart; restarts created inside this control frame are
        protected by ``restart_opened`` and cannot be consumed until the next
        decision frame.
        """

        taker = jnp.clip(state.pending_taker, 0, self.N - 1)
        has_taker = (state.pending_taker >= 0) & (state.pending_taker < self.N)
        # Keep the predictor identical to ``_setpiece_kick_lock`` and the
        # runtime forced-release gate.  An active slot from the wrong team is
        # a broken external/playback state which ``events`` repairs; treating
        # it as eligible here opens the public kick channel for a kick that
        # cannot happen in this control frame.
        valid_taker = (
            has_taker
            & state.active_player[taker]
            & (state.team_id[taker] == state.restart_team)
        )
        target = self._kicker_target(state)
        _window, ready_threshold = self._restart_window_ready_threshold(
            state.restart_kind
        )

        def tick(_, carry):
            position, restart_t, contact_lock, aerial_recovery, fired = carry
            active = restart_timer_active(restart_t) & has_taker & (~fired)
            maximum_speed = self.effective_vmax(
                state.vmax[taker],
                state.stamina_long[taker],
                state.stamina_short[taker],
            )
            advanced = self._advance_kicker_position(
                position, target, state.restart_kind, maximum_speed
            )
            position = jnp.where(active, advanced, position)
            probe_pos = state.player_pos.at[taker].set(position)
            in_reach = self._in_reach(
                state._replace(player_pos=probe_pos)
            )[0][taker]
            arrived = (
                active
                & (jnp.linalg.norm(position - target)
                   <= self.e_cfg.kicker_arrive_r)
                & in_reach
            )
            setup_done = arrived & (restart_t <= ready_threshold)
            fires_now = (
                active & valid_taker & setup_done
                & in_reach & (contact_lock <= 0) & (aerial_recovery <= 0)
            )
            fired = fired | fires_now

            # Except for kickoff (whose taker is placed instantly), the hold
            # clock starts only after physical arrival.  Both active-contact
            # locks decay at the end of the physics tick.
            # Mirror ``events._events_with_restart_mask``: the final timer
            # tick cannot create a live, untaken restart while the valid taker
            # is still outside physical reach or has a transient active lock.
            # Every approach tick preserves the timer.
            preserve_expiry = (
                active
                & (restart_t <= 1)
                & ((~arrived) | (contact_lock > 0) | (aerial_recovery > 0))
            )
            restart_t = jnp.where(
                active & arrived & (~fires_now) & (~preserve_expiry),
                jnp.maximum(jnp.int32(0), restart_t - 1),
                restart_t,
            ).astype(jnp.int32)
            contact_lock = jnp.maximum(jnp.int32(0), contact_lock - 1)
            aerial_recovery = jnp.maximum(
                jnp.int32(0), aerial_recovery - 1
            )
            return position, restart_t, contact_lock, aerial_recovery, fired

        initial = (
            state.player_pos[taker],
            state.restart_t.astype(jnp.int32),
            state.contact_lock_t[taker].astype(jnp.int32),
            state.aerial_recovery_t[taker].astype(jnp.int32),
            jnp.bool_(False),
        )
        eligible = restart_timer_active(state.restart_t) & valid_taker
        return jax.lax.cond(
            eligible,
            lambda carry: jax.lax.fori_loop(
                0, self.timebase.decimation, tick, carry
            )[4],
            lambda carry: jnp.bool_(False),
            initial,
        )

    def _attacking_indirect_fk_spot(self, offense_xy, offender_attack_dir):
        """Canonical spot for an IDFK awarded against the defending team.

        Normally the kick is taken where the offence occurred.  Law 13 moves
        an attacking indirect free kick whose offence lies inside the
        defenders' goal area to the nearest point on the goal-area line
        parallel to the goal line.  Malformed/out-of-play public coordinates
        first use the ordinary field-inset clamp; the resulting legal field
        coordinate then decides whether the goal-area exception applies.  In
        particular, clamping a point from behind the goal line cannot leave an
        attacking IDFK illegally inside the goal area.
        """

        offense_xy = jnp.asarray(offense_xy)
        inset = self.e_cfg.free_kick_boundary_inset
        x = offense_xy[DIM_X]
        y = offense_xy[DIM_Y]
        ordinary_x = jnp.clip(x, -self.hx + inset, self.hx - inset)
        ordinary_y = jnp.clip(y, -self.hy + inset, self.hy - inset)

        own_goal_x = -offender_attack_dir * self.hx
        goal_area_front = (
            own_goal_x
            + offender_attack_dir * self.s_cfg.goal_area_length
        )
        # Public/playback coordinates can be in the five-metre outer player
        # margin.  Classifying the *raw* point first and only then clamping it
        # can manufacture an illegal final IDFK inside the goal area: e.g. a
        # point one metre behind the goal line clamps just inside that line.
        # Apply the ordinary field normalization first, then enforce Law 13 on
        # the actual restart coordinate that would otherwise be returned.
        in_goal_area = (
            (jnp.abs(own_goal_x - ordinary_x)
             <= self.s_cfg.goal_area_length)
            & (jnp.abs(ordinary_y) <= self.s_cfg.goal_area_width / 2.0)
        )
        spot_x = jnp.where(in_goal_area, goal_area_front, ordinary_x)
        spot_y = ordinary_y
        return jnp.asarray([spot_x, spot_y, self.r_ball])

    def _throwin_restriction(
        self, state, throw_taker_before, setpiece_taker_before, touch_before,
        touch_after_force=None, throw_taker_after_force=None,
        setpiece_taker_after_force=None, force_touch_mask=None,
        body_touch_mask=None, ball_pos_before_body=None,
    ):
        """세트피스 키커 재터치 금지(Law 15 스로인 + 코너/골킥/FK/킥오프 일반).

        키커는 공이 다른 선수에 닿기 전 재터치하면 위반 → 상대팀 프리킥. 다른 선수가 먼저 닿으면
        제한 해제(합법). ``*_before``와 ``touch_before``는 바로 이 물리 서브스텝의 접촉 직전
        값이다. 컨트롤 스텝 입구 값을 재사용하면 같은 스텝 후반에 제3자가 먼저 터치한 사실을
        놓쳐 다음 프레임에 원 키커를 오심 처리한다. ``touch_after_force``로
        force2ball→ball_body 순서를 보존해, 키커가 먼저 다시 닿은 반칙을 나중 타인
        몸접촉이 소급해서 없애지 못하게 한다. touch 확정 뒤 호출한다.
        """
        e_cfg = self.e_cfg
        N = self.N

        if touch_after_force is None:
            touch_after_force = state.touch
        if throw_taker_after_force is None:
            throw_taker_after_force = state.throw_taker
        if setpiece_taker_after_force is None:
            setpiece_taker_after_force = state.setpiece_taker
        if ball_pos_before_body is None:
            ball_pos_before_body = state.ball_pos
        else:
            ball_pos_before_body = jnp.asarray(ball_pos_before_body)
            if ball_pos_before_body.shape != (DIM_ALL,):
                raise ValueError(
                    "ball_pos_before_body must have shape "
                    f"({DIM_ALL},), got {ball_pos_before_body.shape}"
                )
        force_touch = (
            ((touch_after_force > TOUCH_NONE)
             & (touch_after_force != touch_before))
            if force_touch_mask is None
            else jnp.asarray(force_touch_mask, dtype=bool)
        )
        body_touch = (
            ((state.touch > TOUCH_NONE)
             & (state.touch != touch_after_force))
            if body_touch_mask is None
            else jnp.asarray(body_touch_mask, dtype=bool)
        )

        def _detect(taker_before, taker_after_force):
            # Adjudicate in physical order.  A taker's force/GK contact is an
            # offence immediately and cannot be retroactively cancelled by a
            # later body touch from somebody else.  Conversely, somebody
            # else's force contact releases the latch before a later taker
            # body contact.  A latch created by the force phase is the original
            # restart touch, so only the subsequent body phase can violate it.
            existed = (taker_before >= 0) & (taker_before < N)
            departed = taker_before == DEPARTED_TAKER
            provenance_existed = existed | departed
            after_valid = (
                (taker_after_force >= 0) & (taker_after_force < N)
            )
            after_departed = taker_after_force == DEPARTED_TAKER
            provenance_after = after_valid | after_departed
            taker_force = jnp.clip(taker_before, 0, N - 1)
            force_others = self.player_indices != taker_force
            foul_force = existed & force_touch[taker_force]
            release_force = (
                (~foul_force)
                & ((existed & jnp.any(force_touch & force_others))
                   | (departed & jnp.any(force_touch)))
            )

            created = (~provenance_existed) & provenance_after
            carried = (
                provenance_existed & provenance_after
                & (~foul_force) & (~release_force)
            )
            body_active = created | carried
            # The after-force latch owns body-phase identity.  In particular,
            # DEPARTED_TAKER has provenance but no player who can commit a
            # double touch; the first body contact by anybody releases it.
            taker_body_raw = taker_after_force
            taker_body = jnp.clip(taker_body_raw, 0, N - 1)
            body_others = self.player_indices != taker_body
            foul_body = body_active & after_valid & body_touch[taker_body]
            release_body = (
                body_active & (~foul_body)
                & ((after_valid & jnp.any(body_touch & body_others))
                   | (after_departed & jnp.any(body_touch)))
            )
            foul = foul_force | foul_body
            released = release_force | release_body
            taker = jnp.where(foul_force, taker_force, taker_body)
            return foul, released, taker

        foul_throw, other_throw, taker_throw = _detect(
            throw_taker_before, throw_taker_after_force
        )
        foul_sp, other_sp, taker_sp = _detect(
            setpiece_taker_before, setpiece_taker_after_force
        )
        foul = foul_throw | foul_sp
        taker = jnp.where(foul_throw, taker_throw, taker_sp)

        offender_team = state.team_id[taker].astype(jnp.int32)
        defender = (TEAM_1 - offender_team).astype(jnp.int32)
        # Rule adjudication is ordered force contact -> body response.  A
        # later BODY_TRAP can relocate ``state.ball_pos`` to its drop point,
        # but it cannot move the already committed force-retouch offence.
        # The runtime therefore supplies the post-force/pre-body ball sample,
        # which is also the body-contact phase's causal location.  Direct
        # callers that omit it fall back to the final state.
        px = ball_pos_before_body[DIM_X]
        py = ball_pos_before_body[DIM_Y]
        fk_spot = self._attacking_indirect_fk_spot(
            jnp.asarray([px, py]), state.attack_dir[taker]
        )
        taker_fk = self._designate_taker_when(
            foul,
            state, fk_spot[:DIM_Z], defender, jnp.bool_(False),
            jnp.int32(RK_FREEKICK), jnp.bool_(True))

        restart_kind = jnp.where(foul, RK_FREEKICK, state.restart_kind).astype(jnp.int32)
        restart_team = jnp.where(foul, defender, state.restart_team).astype(jnp.int32)
        restart_t = jnp.where(foul, jnp.int32(e_cfg.restart_substeps), state.restart_t).astype(jnp.int32)
        ball_state = jnp.where(foul, BALL_DEAD, state.ball_state).astype(jnp.int32)
        ball_pos = jnp.where(foul, fk_spot, state.ball_pos)
        ball_vel = jnp.where(foul, jnp.zeros(DIM_ALL), state.ball_vel)
        ball_spin = jnp.where(foul, jnp.zeros(DIM_ALL), state.ball_spin)
        poss = jnp.where(foul, defender, state.poss_team).astype(jnp.int32)
        pending_taker = jnp.where(foul, taker_fk, state.pending_taker).astype(jnp.int32)
        # 재터치 출처를 보존한다 — 제재는 같아도 스로인과 세트피스는 다른 위반이다.
        # ``foul_throw``가 우선인 것은 위의 ``taker`` 선택과 같은 순서다.
        retouch_kind = jnp.where(
            foul_throw, jnp.int32(FOUL_THROW), jnp.int32(FOUL_SETPIECE)
        )
        foul_kind = jnp.where(foul, retouch_kind, state.foul_kind)
        # actor/victim 갱신 — 누락 시 렌더·통계가 stale 값 또는 -1(P[-1]=마지막 선수)을 지목한다.
        foul_actor = jnp.where(foul, taker.astype(jnp.int32), state.foul_actor)
        foul_victim = jnp.where(foul, jnp.int32(-1), state.foul_victim)   # 재터치는 피해자 없음
        # 각 제한 해제: 자기 케이스가 반칙이거나 다른 선수가 먼저 닿으면 해제
        valid_throw_taker = (
            ((state.throw_taker >= 0) & (state.throw_taker < N))
            | (state.throw_taker == DEPARTED_TAKER)
        )
        valid_setpiece_taker = (
            ((state.setpiece_taker >= 0) & (state.setpiece_taker < N))
            | (state.setpiece_taker == DEPARTED_TAKER)
        )
        throw_taker_in = jnp.where(
            valid_throw_taker, state.throw_taker, jnp.int32(NO_PLAYER)
        )
        setpiece_taker_in = jnp.where(
            valid_setpiece_taker, state.setpiece_taker, jnp.int32(NO_PLAYER)
        )
        throw_taker = jnp.where(
            foul | foul_throw | other_throw, jnp.int32(NO_PLAYER), throw_taker_in
        ).astype(jnp.int32)
        setpiece_taker = jnp.where(
            foul | foul_sp | other_sp, jnp.int32(NO_PLAYER), setpiece_taker_in
        ).astype(jnp.int32)
        # 간접FK 플래그: 재터치 반칙은 IFAB Law 13/15상 **간접 FK**(직접골 무효)로 선다 — 특히
        # 자기 박스 안 재터치(GK 배급 재캐치 등)가 직접 FK면 상대에게 공짜 골 각이 된다.
        # 두 번째 터치(other_*)는 기존 IDFK 보호 종료(False).
        malformed_setpiece_taker = (
            (state.setpiece_taker != NO_PLAYER) & (~valid_setpiece_taker)
        )
        restart_indirect = jnp.where(
            foul,
            jnp.bool_(True),
            jnp.where(
                other_sp | other_throw | malformed_setpiece_taker,
                jnp.bool_(False),
                state.restart_indirect,
            ),
        )
        # 새 재개는 오프사이드 창을 리셋(다른 재개 전이와 동일 규약) — 스테일 pass_signal obs 방지.
        off_clear = jnp.where(foul, jnp.zeros(self.N, bool), state.offside_flag)
        pass_t_clear = jnp.where(foul, jnp.int32(0), state.pass_t).astype(jnp.int32)
        pass_team_clear = jnp.where(foul, jnp.int32(-1), state.pass_team).astype(jnp.int32)
        result = state._replace(
            restart_kind=restart_kind, restart_team=restart_team, restart_t=restart_t,
            ball_state=ball_state, ball_pos=ball_pos, ball_vel=ball_vel, ball_spin=ball_spin,
            poss_team=poss, pending_taker=pending_taker, foul_kind=foul_kind,
            foul_actor=foul_actor, foul_victim=foul_victim,
            throw_taker=throw_taker, setpiece_taker=setpiece_taker,
            restart_indirect=restart_indirect,
            offside_flag=off_clear, pass_t=pass_t_clear, pass_team=pass_team_clear,
            gk_handling_restricted_team=jnp.where(
                foul, jnp.int32(NO_TEAM), state.gk_handling_restricted_team
            ).astype(jnp.int32),
        )
        return self._normalize_pass_latch(result, clear=foul)
