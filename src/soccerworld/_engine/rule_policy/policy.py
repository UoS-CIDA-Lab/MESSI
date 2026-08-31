"""
policy.py — 관측(obs) 기반 룰 정책. env가 제대로 구현됐는지 눈으로(렌더) 검증하는 용도.

원본 SOCCER policy는 월드 절대상태(st)를 직접 읽었지만, 이 클론 정책은 **obs 벡터와
env가 제공하는 affordance view**를 입력받아 action(8-D)을 낸다. `from_state`는 같은
상태에서 두 입력을 만드는 편의 adapter이며, 정책은 월드 절대상태를 직접 읽지 않는다.

핵심 기하: 클론 obs는 **공격 프레임 폴딩**(에이전트별 자기중심, ×attack_dir)이고 action의
방향도 공격 프레임(env가 ×attack_dir로 월드 복원)이다. 따라서 정책은 월드 좌표를 몰라도 된다 —
모든 판단이 "내 공격 프레임"에서 이뤄지며 상대 골대는 항상 [+hx, 0], 우리 골대는 [-hx, 0]이다.
각 에이전트가 자기 obs로 전원·공의 공격 프레임 기하를 복원하고, env affordance의
규칙 파생량을 합쳐 분산적으로 자기 행동을 정한다.

팀 성향(team_styles, (2,5) ∈[0,1]): 초기화 세팅으로 팀 색깔을 바꾼다.
  0 line_height 로우블록↔하이라인 · 1 tempo 점유↔직접 · 2 width 중앙↔측면
  3 aggression 지역↔맨압박 · 4 directness 숏빌드업↔롱볼
obs는 팀 정체를 감추므로(폴딩) 팀별 성향은 팩토리에서 team_id로 주입한다(정적 메타).

env 규칙 준수(기하는 obs, 규칙 파생량은 affordance로 판정):
  · 카드(self yellow): 경고 1장 선수는 태클 라인에 안 들어가고 컨테인/조키 — 2차 경고=퇴장 자충수 방지.
  · 간접 FK(is_fk_indirect, IFAB Law 13): 키커는 골 직격 금지, 반드시 동료로 연결(백패스·재터치 IDFK 포함).
  · GK 캐치/홀드(RK_GK_HOLD): 홀드 만료 전 GK가 롱 클리어(롱볼/무옵션) 또는 열린 동료 배급.

API:
  make_rule_based_policy(env, match_key=None, team_styles=None, policy_config=None) -> policy_fn
  policy_fn(obs, key, aff) -> action   # obs (N, obs_dim), aff = env.affordance_view(state)
  policy_fn.policy_with_trace(obs, key, aff) -> (action, trace)
  policy_fn.from_state(state, key)     # obs·affordance를 함께 만들어 주는 편의 어댑터
  프리셋: STYLE_PRESETS["gegenpress"|"park_the_bus"|"tiki_taka"|"long_ball"|"balanced"]

Causal-Compact 관측(v14)부터 파생 affordance는 관측 벡터에 없다. in_reach/f2b_avail/off_line/
kicker_ready 같은 값은 ``env.affordance_view(state)``가 단일 구현으로 계산해 넘겨준다. 정책이
관측에서 이들을 다시 유도하면 env와 정책에 같은 규칙의 사본이 둘 생기고, 그 둘은 반드시
갈린다(그 중복을 없애려고 만든 것이 이 view다).
"""
from __future__ import annotations

import hashlib
import json
import numbers
import types
from collections.abc import Mapping
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np

from soccerworld.policies.contracts import RULE_POLICY_EXECUTION_PROFILES

from .. import setpiece_taker as player_roles
from ..config import RulePolicy
from ..constants import (
    ACTION_MAX,
    ACTION_MIN,
    AFFORDANCE_FEATURES,
    BALL_ALIVE,
    DIM_Z,
    DIV_EPS,
    ENDURANCE_DECISION_DELTA_CAP,
    ENDURANCE_DECISION_GAIN,
    ENDURANCE_FACTOR_REFERENCE,
    GEOMETRY_EPS,
    POSSESSION_CONTEXT_SECONDS,
    PROB_EPS,
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
    ROLE_DEFENDER,
    ROLE_FORWARD,
    ROLE_GAIN_EXACT_MAX_SAMPLES,
    ROLE_GK,
    ROLE_MIDFIELDER,
    SLOT_ACTIVE,
    STYLE_AGGRESSION,
    STYLE_DIM,
    STYLE_DIRECTNESS,
    STYLE_LINE,
    STYLE_PRESETS,
    STYLE_TEMPO,
    STYLE_WIDTH,
    TAKER_BIT_PENDING,
    TAKER_BIT_SETPIECE,
    TAKER_BIT_THROW,
    TEAM_COUNT,
    TOUCH_DRIBBLE,
    TOUCH_PASS,
    TOUCH_PASS_HEAD,
    TOUCH_SHOOT,
    TOUCH_SHOOT_HEAD,
)
from ..energy import effective_speed_cap
from ..formation import LAYOUT_INDEX_CAPACITY
from ..formation import LAYOUTS as F_LAYOUTS
from ..spatial import _safe_norm, _unit, reach_blockable, stretch_encode
from ..timebase import DEFAULT_TIMEBASE
from . import deadball as DB
from . import positioning as POS
from . import tactics as T

RULE_POLICY_VERSION = 61
"""규칙 정책의 전술 의사결정 버전. 환경 dynamics/schema 버전과 독립이다.

teacher 행동 분포가 바뀌면 올린다 — 배치·패스 선택·압박 기준이 달라지면 같은 관측에서
다른 행동이 나오므로, 이 값이 다른 정책으로 만든 BC 데이터셋은 서로 섞을 수 없다.
생성기는 이 버전과 config fingerprint를 shard manifest에 함께 박고, 로더가 불일치하면
데이터를 열기 전에 실패한다."""


def _decision_speed_for_endurance(speed, endurance_factor):
    """Bound the policy-only capacity estimate around the neutral factor 1.0."""

    centered = jnp.asarray(endurance_factor, jnp.float32) - jnp.float32(
        ENDURANCE_FACTOR_REFERENCE
    )
    delta = (
        jnp.clip(
            centered,
            -jnp.float32(ENDURANCE_DECISION_DELTA_CAP),
            jnp.float32(ENDURANCE_DECISION_DELTA_CAP),
        )
        * jnp.float32(ENDURANCE_DECISION_GAIN)
    )
    return jnp.asarray(speed, jnp.float32) * (jnp.float32(1.0) + delta)


def _resolve_execution_profile(execution_profile: str) -> str:
    """Resolve the factory-time lowering profile without a per-tick branch."""

    if not isinstance(execution_profile, str):
        raise TypeError(
            "execution_profile must be one of auto, cpu_causal, or gpu_dense"
        )
    if execution_profile not in RULE_POLICY_EXECUTION_PROFILES:
        raise ValueError(
            "execution_profile must be one of auto, cpu_causal, or gpu_dense"
        )
    if execution_profile != "auto":
        return execution_profile
    backend = jax.default_backend()
    if backend == "cpu":
        return "cpu_causal"
    if backend in {"gpu", "cuda"}:
        return "gpu_dense"
    raise ValueError(
        "execution_profile='auto' supports CPU and GPU backends; "
        f"got {backend!r}"
    )

RECEPTION_CONTROL_HYSTERESIS_MPS = 0.10
"""Speed margin keeping a soft reception inside carry and DRIBBLE bands."""


def _bound_policy_action(action):
    """Enforce the exact public Box bound after float32 action assembly."""

    return jnp.clip(action, ACTION_MIN, ACTION_MAX)


def _dribble_recontact_ready(
    ball_offset,
    relative_ball_velocity,
    ball_speed,
    self_touched_last,
    last_touch_code,
    unattended_ball_speed,
):
    """Re-arm a self-follow-up touch only when the carrier catches the ball.

    A carried ball is kicked ahead of its carrier.  The time derivative of its
    squared carrier distance has the sign of ``relative_velocity · offset``:
    positive immediately after a good touch and negative only when the carrier
    catches it again.  This observable dot product is rotation invariant and
    needs neither a hidden timer nor a norm.  A stalled ball remains recoverable
    even when its offset is exactly zero.
    """

    # The vectors are exactly 2-D.  Writing the dot explicitly avoids creating
    # a tiny reduction barrier in every scalar and batched policy graph.
    distance_derivative_sign = (
        relative_ball_velocity[..., 0] * ball_offset[..., 0]
        + relative_ball_velocity[..., 1] * ball_offset[..., 1]
    )
    self_dribble_followup = self_touched_last & (
        last_touch_code == TOUCH_DRIBBLE
    )
    carrier_caught_ball = distance_derivative_sign < 0.0
    stalled_ball = ball_speed <= unattended_ball_speed
    return (~self_dribble_followup) | carrier_caught_ball | stalled_ball


_OBSERVED_FORMATION_TABLE_TOL_M = 1e-3
"""Maximum reconstruction error for the stable-layout role fast path."""


_POLICY_AFFORDANCE_KEYS = tuple(name for name, width in AFFORDANCE_FEATURES)
_POLICY_FLOAT32_DTYPE = np.dtype(np.float32)


def _validate_policy_key(key, *, name):
    """Use the environment's single seeded-reproducibility key contract."""

    # Lazy import keeps policy helpers usable during module discovery without
    # introducing an env->policy cycle.  The implementation itself remains a
    # single source in env.py, shared with reset/step.
    from ..env import _validate_prng_key

    return _validate_prng_key(key, name=name)


def _validate_exact_mapping_keys(mapping, expected_keys, *, name):
    """Validate a static public mapping schema before JAX can ignore extra keys."""

    if not isinstance(mapping, Mapping):
        raise TypeError(f"{name} must be a mapping")
    actual = set(mapping.keys())
    expected = set(expected_keys)
    if actual != expected:
        missing = tuple(key for key in expected_keys if key not in actual)
        extra = tuple(sorted((key for key in actual if key not in expected), key=repr))
        raise ValueError(
            f"{name} keys must exactly match {tuple(expected_keys)!r}; "
            f"missing={missing!r}, extra={extra!r}"
        )


def _as_policy_float32_array(value, *, name, shape):
    """Preserve the documented float32 array contract without silent narrowing."""

    source_dtype = getattr(value, "dtype", None)
    if source_dtype is not None:
        try:
            source_dtype = np.dtype(source_dtype)
        except TypeError as exc:
            raise TypeError(f"{name} must have float32 dtype") from exc
        # Check before jnp.asarray: with global x64 disabled, JAX otherwise
        # silently narrows a NumPy float64 array to float32.
        if source_dtype != _POLICY_FLOAT32_DTYPE:
            raise TypeError(
                f"{name} must have float32 dtype, got {source_dtype}"
            )
    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a float32 array") from exc
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != jnp.dtype(jnp.float32):
        raise TypeError(f"{name} must have float32 dtype, got {array.dtype}")
    return array


def _validate_policy_inputs(obs, aff, *, n_players, obs_dim):
    """Validate the public observation and 14-column affordance mapping."""

    obs = _as_policy_float32_array(
        obs, name="policy observation", shape=(n_players, obs_dim)
    )
    _validate_exact_mapping_keys(
        aff, _POLICY_AFFORDANCE_KEYS, name="policy affordance"
    )
    checked_aff = {
        name: _as_policy_float32_array(
            aff[name], name=f"policy affordance[{name!r}]", shape=(n_players,)
        )
        for name in _POLICY_AFFORDANCE_KEYS
    }
    finite_flags = jnp.stack(
        [jnp.all(jnp.isfinite(obs))]
        + [jnp.all(jnp.isfinite(checked_aff[name]))
           for name in _POLICY_AFFORDANCE_KEYS]
    )
    all_finite = jnp.all(finite_flags)
    if isinstance(all_finite, jax.core.Tracer):
        # Runtime values cannot raise a Python exception from a pure JAX graph.
        # Sanitize before any tactical arithmetic, then let policy_fn close the
        # entire action row-set to exact zero when this predicate is false.
        obs = jnp.where(jnp.isfinite(obs), obs, 0.0)
        checked_aff = {
            name: jnp.where(jnp.isfinite(value), value, 0.0)
            for name, value in checked_aff.items()
        }
        return obs, checked_aff, all_finite
    if not bool(np.asarray(all_finite)):
        names = ("observation",) + _POLICY_AFFORDANCE_KEYS
        flags = np.asarray(finite_flags, dtype=bool)
        bad = tuple(name for name, finite in zip(names, flags) if not finite)
        raise ValueError(
            f"policy inputs must be finite; non-finite fields: {bad!r}"
        )
    return obs, checked_aff, None


def prefix_stable_keys(base_key, steps):
    """전체 rollout 길이와 무관하게 같은 seed의 prefix가 같은 frame key를 갖게 한다."""

    base_key = _validate_policy_key(base_key, name="base_key")
    if not isinstance(steps, numbers.Integral) or isinstance(
        steps, (bool, np.bool_)
    ):
        raise TypeError("steps must be a scalar integer")
    steps = int(steps)
    # ``jnp.arange`` materialises every frame index before ``vmap``.  Merely
    # fitting uint32 is therefore not a useful public bound: uint32 max would
    # request more than 34 GiB for the indices and generated raw keys alone.
    # The environment already has one stricter maximum episode/control-step
    # contract for exact role-state reconstruction; use that same SSOT here so
    # a policy rollout cannot request a prefix longer than any legal episode.
    if not 0 <= steps <= ROLE_GAIN_EXACT_MAX_SAMPLES:
        raise ValueError(
            "steps must lie in [0, ROLE_GAIN_EXACT_MAX_SAMPLES] "
            f"([0, {ROLE_GAIN_EXACT_MAX_SAMPLES}])"
        )
    return jax.vmap(lambda frame: jax.random.fold_in(base_key, frame))(
        jnp.arange(steps, dtype=jnp.uint32)
    )

def _taker_bits(taker_mask):
    """``taker_mask`` 열 → (pending, setpiece_retouch, throw_retouch) bool.

    ``Observation.taker_bits``와 같은 디코딩이다. env 인스턴스 없이도 관측만으로 풀 수 있게
    정책 쪽에도 순수 함수를 둔다(값 규약은 constants.TAKER_BIT_*가 단일 진실원천).
    """

    code = jnp.round(taker_mask).astype(jnp.int32)
    return ((code & TAKER_BIT_PENDING) > 0,
            (code & TAKER_BIT_SETPIECE) > 0,
            (code & TAKER_BIT_THROW) > 0)


def policy_config_fingerprint(policy_config: RulePolicy) -> str:
    """Canonical SHA-256 receipt for policy semantics and complete config."""

    if not isinstance(policy_config, RulePolicy):
        raise TypeError(
            "policy_config must be RulePolicy, got "
            f"{type(policy_config).__name__}"
        )
    payload = json.dumps(
        {
            "rule_policy_version": RULE_POLICY_VERSION,
            "config": asdict(policy_config),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sample_style(key, policy_config):
    """균등난수 평균 성향 샘플 (5,)∈[0,1]; 기본 2개는 삼각분포다."""
    return jnp.mean(
        jax.random.uniform(key, (policy_config.style_sample_count, STYLE_DIM)),
        axis=0,
    )


def _resolve_styles(match_key, team_styles, policy_config):
    """team_styles((2,5) 배열/이름/None)를 (2,5) jnp 배열로 정규화.
    None이면 match_key로 팀별 독립 샘플, 이름이면 프리셋."""
    def one(spec, key):
        if spec is None:
            return _sample_style(key, policy_config)
        if isinstance(spec, str):
            if spec not in STYLE_PRESETS:
                raise ValueError(
                    f"unknown team style {spec!r}; choose from {tuple(STYLE_PRESETS)}"
                )
            return jnp.asarray(STYLE_PRESETS[spec], jnp.float32)
        array = np.asarray(spec)
        if array.shape != (STYLE_DIM,):
            raise ValueError(
                f"each team style must have shape {(STYLE_DIM,)}, got {array.shape}"
            )
        if (
            not np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.bool_)
            or np.issubdtype(array.dtype, np.complexfloating)
        ):
            raise TypeError(
                "team style vectors must contain real non-boolean numbers"
            )
        array = array.astype(np.float64, copy=False)
        if not np.isfinite(array).all() or not np.all(
            (array >= 0.0) & (array <= 1.0)
        ):
            raise ValueError("team style values must be finite and lie in [0, 1]")
        return jnp.asarray(array, jnp.float32)

    keys = jax.random.split(
        match_key if match_key is not None else jax.random.PRNGKey(0),
        TEAM_COUNT,
    )
    if team_styles is None or isinstance(team_styles, str):
        s = team_styles
        styles = jnp.stack([one(s, key) for key in keys])
    else:
        if len(team_styles) != TEAM_COUNT:
            raise ValueError(f"team_styles must contain {TEAM_COUNT} teams")
        styles = jnp.stack([one(team_styles[i], keys[i]) for i in range(TEAM_COUNT)])
    if styles.shape != (TEAM_COUNT, STYLE_DIM):
        raise ValueError(
            f"team_styles must resolve to shape {(TEAM_COUNT, STYLE_DIM)}, got {styles.shape}"
        )
    if not bool(jnp.all(jnp.isfinite(styles))) or not bool(
        jnp.all((styles >= 0.0) & (styles <= 1.0))
    ):
        raise ValueError("team style values must be finite and lie in [0, 1]")
    return styles


def _coarse_roles_from_detailed(detailed):
    """Collapse shared CB/FB/CM/WM/CF/WF roles to policy line bands."""

    coarse = jnp.where(
        detailed == player_roles.ROLE_GK,
        ROLE_GK,
        jnp.where(
            (detailed == player_roles.ROLE_CENTRE_BACK)
            | (detailed == player_roles.ROLE_FULL_BACK),
            ROLE_DEFENDER,
            jnp.where(
                (detailed == player_roles.ROLE_CENTRE_MID)
                | (detailed == player_roles.ROLE_WIDE_MID),
                ROLE_MIDFIELDER,
                ROLE_FORWARD,
            ),
        ),
    )
    return coarse.astype(jnp.int32)


def _formation_roles_from_home(home_att, gk, team_id, active):
    """Return coarse policy lines and detailed roles from one relative classifier.

    The set-piece and manager paths already define role identity from distinct
    formation depths and within-line width ranks. Reusing that classifier keeps a
    block-height shift, pitch direction, or formation width from changing what the
    policy means by defender, midfielder, forward, central, and wide.
    """

    detailed = player_roles.classify_roles(home_att, gk, team_id, active)
    return _coarse_roles_from_detailed(detailed), detailed


def _first_landing_index(zs, ground, vzs=None):
    """각 궤적의 최초 공중 이탈 뒤 첫 지면 복귀 index.

    전체 궤적의 ``argmax(z)`` 이후를 찾으면 첫 바운스 뒤 골 구조물 충돌처럼 더 늦은 최고점이
    생겼을 때 그 후속 접촉을 '착지'로 오인한다. 공중 상태를 한 번이라도 지난 시점의 직후만
    허용해 최초 비행을 인과적으로 닫는다.
    """

    above = zs > ground
    seen = jnp.cumsum(above.astype(jnp.int32), axis=1) > 0
    seen_before = jnp.concatenate(
        [jnp.zeros((zs.shape[0], 1), dtype=bool), seen[:, :-1]], axis=1
    )
    landing = (zs <= ground) & seen_before
    if vzs is not None:
        # 접촉 적분은 지면 충돌 뒤 같은 substep에 반발 속도를 적용해 반환 위치가 ground 문턱보다
        # 몇 cm 높을 수 있다. 하강→상승 부호 전환도 첫 접지로 인정하지 않으면 첫 바운스를 놓친다.
        bounced = jnp.concatenate(
            [
                jnp.zeros((vzs.shape[0], 1), dtype=bool),
                (vzs[:, :-1] < 0.0) & (vzs[:, 1:] > 0.0),
            ],
            axis=1,
        )
        landing = landing | (bounced & seen_before)
    fallback = zs.shape[1] - 1
    return jnp.where(
        jnp.any(landing, axis=1), jnp.argmax(landing, axis=1), fallback
    )


def _air_travel_distance(horizontal_speed, time_s, drag):
    """이차 항력 ``dv/dt=-c|v|v`` 아래의 수평 이동거리.

    ``drag``는 팩토리가 스냅샷한 정적 config다. 0일 때 나눗셈을 그래프에
    남기지 않고 정확한 등속 극한 ``v*t``를 쓴다. 수신 주자 선택과 최종
    공중볼 추격이 반드시 같은 수식을 공유하게 하는 SSOT다.
    """

    speed = jnp.asarray(horizontal_speed)
    time = jnp.asarray(time_s)
    if float(drag) == 0.0:
        return speed * time
    c_drag = jnp.asarray(drag, dtype=speed.dtype)
    return jnp.log1p(c_drag * speed * time) / c_drag


def _strict_record_table(distance, speed, *extras, eps=1e-4):
    """거리→속도 역함수의 엄격 단조 record-high 표를 만든다.

    이산 충돌·접지 적분은 속도를 높여도 착지거리가 미세하게 줄 수 있다. cummax로 평탄화한 뒤
    중복 x를 보간하면 지배당한 더 높은 속도를 선택하므로, 이전 최대거리를 실제로 갱신한 표본만
    남긴다. ``extras``(예: 체공시간)는 같은 원본 index를 따라간다.
    """

    d = np.asarray(distance, dtype=np.float32)
    v = np.asarray(speed, dtype=np.float32)
    previous = np.maximum.accumulate(d)
    keep = np.ones(d.shape, dtype=bool)
    keep[1:] = d[1:] > previous[:-1] + eps
    if int(keep.sum()) < 2:
        raise RuntimeError("physical calibration did not produce a usable distance table")
    result = [jnp.asarray(d[keep]), jnp.asarray(v[keep])]
    result.extend(jnp.asarray(np.asarray(extra)[keep]) for extra in extras)
    return tuple(result)


def _calibrate_kick_solver(env, arrive_speed=8.0, n_speed=56, nstep=450):
    """팩토리 1회: env의 **실제 공 물리**(ball_step_only)로 (발사속도→도달거리)를 롤아웃 캘리브하고
    (도달거리→필요 발사속도) 역테이블을 만든다. 런타임에 jnp.interp로 목표거리에 맞는 정밀 파워를 낸다.
    드래그·굴림·바운스가 전부 반영된 실물리 기반이라 R≈v²sin2θ/g 같은 근사보다 정확하다.

    두 궤적 모드:
      loft  — 높은 아크(수비수 넘김). '착지 거리'(apex 후 첫 지면 복귀 x) 기준. 크로스·롱패스·골킥·GK 롱배급.
      drive — 낮은 지상 드라이브. '도착 트래핑 속도까지 감속한 거리' 기준. 짧은 발밑 패스(리시버가 트랩 가능).
    반환: {loft_R, loft_v, loft_T, drive_R, drive_v(모두 (n_speed,), interp용 단조증가),
    loft_launch01, drive_launch01}. ``loft_T``는 각 착지거리와 짝지은 첫 착지 시간이다.
    """
    e = env.e_cfg
    rb = env.r_ball
    theta_loft, theta_cross, theta_drive = 0.55, 0.38, 0.0
    # rad — 롱패스 높은 아크 / 빠른 백스핀 크로스 / 순수 지면
    # 낮게 띄운 0.06 rad 드라이브는 바운스 횟수가 바뀌는 속도에서 도착거리 곡선이
    # 불연속·비단조가 된다. 특히 DFL7 bounce_h_keep 적용 뒤 18 m 역보간이 실제로는
    # 13.7 m에 멈췄다. 0 rad 굴림은 전 속도 범위에서 단조라 거리 역함수가 유효하다.
    arrive_speed = float(arrive_speed)          # drive 패스 도착 목표 속력(트래핑 가능)
    ground = rb + 0.03
    speeds = jnp.linspace(4.0, e.f2b_speed_max, n_speed)

    def rollout(v0, theta, initial_spin):
        pos = jnp.array([0.0, 0.0, rb])
        vel = jnp.array([v0 * jnp.cos(theta), 0.0, v0 * jnp.sin(theta)])
        spin = initial_spin
        def body(carry, _):
            p, v, s = carry
            p2, v2, s2 = env.ball_step_only(p, v, s)
            return (p2, v2, s2), (
                p2[0], p2[2], v2[2], jnp.linalg.norm(v2[:2])
            )
        _, out = jax.lax.scan(body, (pos, vel, spin), None, length=nstep)
        return out                              # (xs, zs, spd) 각 (nstep,)

    zero_spin = jnp.zeros(3)
    cross_spin = jnp.array([0.0, -0.38 * e.spin_max, 0.0])
    xs_l, zs_l, vzs_l, _ = jax.vmap(
        lambda v: rollout(v, theta_loft, zero_spin)
    )(speeds)
    xs_d, _, _, spd_d = jax.vmap(
        lambda v: rollout(v, theta_drive, zero_spin)
    )(speeds)
    xs_c, zs_c, vzs_c, _ = jax.vmap(
        lambda v: rollout(v, theta_cross, cross_spin)
    )(speeds)
    # 착지거리 = 최초 공중 이탈 뒤 첫 지면 복귀. 전역 apex는 후속 바운스·골 구조물 충돌을
    # 첫 비행으로 오인하므로 사용하지 않는다.
    land_idx = _first_landing_index(zs_l, ground, vzs_l)
    loft_R = jnp.take_along_axis(xs_l, land_idx[:, None], axis=1)[:, 0]
    loft_T = (land_idx.astype(jnp.float32) + 1.0) * e.dt_phys
    land_c_idx = _first_landing_index(zs_c, ground, vzs_c)
    cross_R = jnp.take_along_axis(xs_c, land_c_idx[:, None], axis=1)[:, 0]
    cross_T = (land_c_idx.astype(jnp.float32) + 1.0) * e.dt_phys
    # drive 도착거리 = xy속력이 arrive_speed 이하로 처음 감속한 지점의 x
    dm = spd_d <= arrive_speed
    d_idx = jnp.where(jnp.any(dm, axis=1), jnp.argmax(dm, axis=1), nstep - 1)
    drive_R = jnp.take_along_axis(xs_d, d_idx[:, None], axis=1)[:, 0]
    # 지배당한 표본을 버려 interp의 x를 엄격 단조로 만든다. cross 체공시간도 같은 record index를
    # 따라야 거리표와 서로 다른 속도의 시간이 짝지어지지 않는다.
    loft_R, loft_v, loft_T = _strict_record_table(
        loft_R, speeds, loft_T
    )
    drive_R, drive_v = _strict_record_table(drive_R, speeds)
    cross_R, cross_v, cross_T = _strict_record_table(
        cross_R, speeds, cross_T
    )
    lf = -e.launch_down_ground                  # 지면공 발사각 하한(launch_lo(r_ball))
    to01 = lambda th: float((th - lf) / (e.launch_max - lf))   # 목표각→action launch01 역매핑
    return dict(loft_R=loft_R, loft_v=loft_v, loft_T=loft_T,
                drive_R=drive_R, drive_v=drive_v,
                cross_R=cross_R, cross_v=cross_v, cross_T=cross_T,
                loft_launch01=to01(theta_loft), cross_launch01=to01(theta_cross),
                drive_launch01=to01(theta_drive))


def _calibrate_throw_solver(env, launch_h, n_speed=48, nstep=350):
    """스로인(손 던지기) 전용 캘리브 — 발사가 발킥이 아니라 **손 높이(launch_h)**에서 throw_speed_max
    스케일로 나간다(env _apply_force2ball의 throw 경로). 발킥 솔버로는 틀리므로 별도 캘리브.
    반환: {throw_R, throw_v(=속도, interp용 단조증가), throw_launch01}."""
    e = env.e_cfg
    rb = env.r_ball
    theta = 0.5
    ground = rb + 0.03
    speeds = jnp.linspace(3.0, e.throw_speed_max, n_speed)

    def rollout(v0):
        pos = jnp.array([0.0, 0.0, launch_h])
        vel = jnp.array([v0 * jnp.cos(theta), 0.0, v0 * jnp.sin(theta)])
        spin = jnp.zeros(3)
        def body(carry, _):
            p, v, s = carry
            p2, v2, s2 = env.ball_step_only(p, v, s)
            return (p2, v2, s2), (p2[0], p2[2], v2[2])
        _, (xs, zs, vzs) = jax.lax.scan(
            body, (pos, vel, spin), None, length=nstep
        )
        return xs, zs, vzs

    xs, zs, vzs = jax.vmap(rollout)(speeds)
    land_idx = _first_landing_index(zs, ground, vzs)
    throw_R = jnp.take_along_axis(xs, land_idx[:, None], axis=1)[:, 0]
    throw_R, throw_v = _strict_record_table(throw_R, speeds)
    lf = -e.launch_down_ground                    # 스로인 launch_ang도 스폿(지면) 기준 remap을 탄다
    throw_launch01 = float((theta - lf) / (e.launch_max - lf))
    return dict(throw_R=throw_R, throw_v=throw_v, throw_launch01=throw_launch01)


def _calibrate_shot_solver(env, n_theta=56, nstep=220):
    """슛 전용 캘리브 — 목적이 '착지'가 아니라 '골 구석에 빠르게'. 고파워(≈0.9) 슛이 목표거리 d에서
    낮은 코너 높이(z_target)를 통과하도록 **발사각(launch01)을 거리별로 역산**한다. 파워는 높게 고정.
    반환: {shot_d, shot_launch01(거리별, interp용), shot_pow(스칼라)}."""
    e = env.e_cfg
    rb = env.r_ball
    v_shot = 0.97 * e.f2b_speed_max              # 거의 최대 파워 — GK 반응시간 최소화(피니싱)
    z_target = 0.9                                # 낮은 코너 목표 높이(m)
    # 오픈플레이 감아차기의 평균 탑스핀(spin_back≈-0.31)을 포함한다. 제로스핀 각도로 찬 뒤
    # 탑스핀만 더하면 공이 목표 전에 떨어지므로 launch와 spin은 반드시 같은 물리로 역산해야 한다.
    shot_spin = jnp.array([0.0, 0.31 * e.spin_max, 0.0])
    thetas = jnp.linspace(-0.08, 0.5, n_theta)

    def rollout(theta):
        pos = jnp.array([0.0, 0.0, rb])
        vel = jnp.array([v_shot * jnp.cos(theta), 0.0, v_shot * jnp.sin(theta)])
        spin = shot_spin
        def body(carry, _):
            p, v, s = carry
            p2, v2, s2 = env.ball_step_only(p, v, s)
            return (p2, v2, s2), (p2[0], p2[2])
        _, (xs, zs) = jax.lax.scan(body, (pos, vel, spin), None, length=nstep)
        return xs, zs

    XS, ZS = jax.vmap(rollout)(thetas)            # (n_theta, nstep)
    d_grid = jnp.linspace(4.0, e.f2b_shoot_range, 40)
    # 각 (d, theta): 궤적에서 x=d일 때의 z (x는 전진 중 단조증가)
    Z_at = jax.vmap(lambda d: jax.vmap(lambda i: jnp.interp(d, XS[i], ZS[i]))(jnp.arange(n_theta)))(d_grid)
    valid = (Z_at > 0.1) & (Z_at < env.goal_h - 0.2)                           # 지면~크로스바 하단 사이
    cost = jnp.where(valid, jnp.abs(Z_at - z_target), 1e6)
    best = jnp.argmin(cost, axis=1)               # (n_d,) 목표높이에 가장 근접한 발사각
    lf = -e.launch_down_ground
    shot_launch01 = jnp.clip((thetas[best] - lf) / (e.launch_max - lf), 0.0, 1.0)

    # 같은 거리·launch에서 side-spin action=+1이 만드는 실제 횡변위를 계산한다. 정책은 이 표에
    # 실제 curve_spin을 곱해 직선 조준점을 반대쪽으로 이동시킨다. 고정 몇 m 보정은 거리²에 가까운
    # 마그누스 변위를 설명하지 못해 짧은 슛은 중앙, 긴 슛은 포스트 밖으로 보내므로 쓰지 않는다.
    def rollout_curve(theta):
        pos = jnp.array([0.0, 0.0, rb])
        vel = jnp.array([v_shot * jnp.cos(theta), 0.0, v_shot * jnp.sin(theta)])
        spin = shot_spin + jnp.array([0.0, 0.0, e.spin_max])
        def body(carry, _):
            p, v, s = carry
            p2, v2, s2 = env.ball_step_only(p, v, s)
            return (p2, v2, s2), (p2[0], p2[1])
        _, (xs, ys) = jax.lax.scan(body, (pos, vel, spin), None, length=nstep)
        return xs, ys

    curve_x, curve_y = jax.vmap(rollout_curve)(thetas[best])
    curve_y_unit = jax.vmap(
        lambda d, xs, ys: jnp.interp(d, xs, ys)
    )(d_grid, curve_x, curve_y)
    return dict(
        shot_d=d_grid,
        shot_launch01=shot_launch01,
        shot_curve_y_unit=curve_y_unit,
        shot_pow=float(v_shot / e.f2b_speed_max),
    )


def make_rule_based_policy(
    env,
    match_key=None,
    team_styles=None,
    policy_config=None,
    *,
    execution_profile="auto",
):
    """팩토리: env에서 정적 기하(팀·역할·홈·정규화 상수)를 스냅샷하고 팀 성향을 고정해
    obs+env affordance→action 클로저를 반환. `from_state`는 두 입력을 같은 State에서
    구성하는 편의 adapter이며 입력 State를 변경하지 않는다."""
    policy_config = RulePolicy() if policy_config is None else policy_config
    if not isinstance(policy_config, RulePolicy):
        raise TypeError(
            f"policy_config must be RulePolicy or None, got {type(policy_config).__name__}"
        )
    resolved_execution_profile = _resolve_execution_profile(execution_profile)
    if match_key is not None:
        match_key = _validate_policy_key(match_key, name="match_key")
    e, s = env.e_cfg, env.s_cfg
    if policy_config.curl_distance_start >= e.f2b_shoot_range:
        raise ValueError(
            "policy_config.curl_distance_start must be smaller than "
            "engine.f2b_shoot_range"
        )
    if policy_config.first_time_shot_distance > e.f2b_shoot_range:
        raise ValueError(
            "policy_config.first_time_shot_distance must not exceed "
            "engine.f2b_shoot_range (the shot-classification range)"
        )
    if policy_config.drive_arrive_speed_mps >= e.f2b_speed_max:
        raise ValueError(
            "policy_config.drive_arrive_speed_mps must be smaller than "
            "engine.f2b_speed_max"
        )
    if policy_config.quick_relay_min_incoming_speed_mps >= e.f2b_speed_max:
        raise ValueError(
            "policy_config.quick_relay_min_incoming_speed_mps must be smaller "
            "than engine.f2b_speed_max"
        )
    if policy_config.quick_relay_max_incoming_speed_mps >= e.f2b_speed_max:
        raise ValueError(
            "policy_config.quick_relay_max_incoming_speed_mps must be smaller "
            "than engine.f2b_speed_max"
        )
    if policy_config.carried_release_speed_mps >= e.f2b_speed_max:
        raise ValueError(
            "policy_config.carried_release_speed_mps must be smaller than "
            "engine.f2b_speed_max"
        )
    if policy_config.gk_foot_pass_max_distance > 50.0:
        raise ValueError(
            "policy_config.gk_foot_pass_max_distance must not exceed the "
            "policy's 50 m pass-candidate horizon"
        )
    if policy_config.ground_control_touch_speed_mps >= e.dribble_speed_max:
        raise ValueError(
            "policy_config.ground_control_touch_speed_mps must be smaller than "
            "engine.dribble_speed_max so a control touch remains a dribble"
        )
    reception_control_cap = min(
        policy_config.carried_release_speed_mps,
        e.dribble_speed_max,
    ) - RECEPTION_CONTROL_HYSTERESIS_MPS
    if policy_config.ground_control_touch_speed_mps >= reception_control_cap:
        raise ValueError(
            "policy_config.ground_control_touch_speed_mps must remain below "
            "the carried-release/DRIBBLE reception cap by at least "
            f"{RECEPTION_CONTROL_HYSTERESIS_MPS:.2f} m/s"
        )
    if policy_config.challenge_control_touch_speed_mps >= e.dribble_speed_max:
        raise ValueError(
            "policy_config.challenge_control_touch_speed_mps must be smaller "
            "than engine.dribble_speed_max so a challenge control remains a dribble"
        )
    if policy_config.challenge_outlet_max_incoming_speed_mps >= e.f2b_speed_max:
        raise ValueError(
            "policy_config.challenge_outlet_max_incoming_speed_mps must be smaller "
            "than engine.f2b_speed_max"
        )
    if policy_config.challenge_outlet_max_distance > 50.0:
        raise ValueError(
            "policy_config.challenge_outlet_max_distance must not exceed the "
            "policy's 50 m pass-candidate horizon"
        )
    if policy_config.aerial_pass_max_distance > 50.0:
        raise ValueError(
            "policy_config.aerial_pass_max_distance must not exceed the "
            "policy's 50 m pass-candidate horizon"
        )
    header_speed_cap = e.header_cap * e.f2b_speed_max
    if max(
        policy_config.aerial_pass_max_speed_mps,
        policy_config.aerial_shot_speed_mps,
        policy_config.aerial_clear_speed_mps,
    ) > header_speed_cap:
        raise ValueError(
            "policy aerial output speeds must not exceed the engine header "
            f"speed cap ({header_speed_cap:.6g} m/s)"
        )
    if policy_config.aerial_defender_cover_distance >= min(env.hx, env.hy):
        raise ValueError(
            "policy_config.aerial_defender_cover_distance must be smaller "
            "than both pitch half-dimensions"
        )
    if max(
        policy_config.box_mark_runner_margin,
        policy_config.box_mark_ball_margin,
        policy_config.box_mark_goal_side_distance,
    ) >= min(env.hx, env.hy):
        raise ValueError(
            "policy box-mark distances must be smaller than both pitch "
            "half-dimensions"
        )
    if policy_config.retouch_pitch_inset >= min(env.hx, env.hy):
        raise ValueError(
            "policy_config.retouch_pitch_inset must be smaller than both "
            "pitch half-dimensions"
        )
    resolved_styles = _resolve_styles(match_key, team_styles, policy_config)
    st0 = env.reset_state(jax.random.PRNGKey(0))
    adir0 = st0.attack_dir
    home_att = env._kickoff_positions(st0) * adir0[:, None]      # 공격 프레임 홈(+x=전방)
    gk = st0.gk_indices.astype(jnp.float32)
    # 데이터 유래 오프더볼 위치장: 각 선수를 자기 팀 포메이션 안에서 실측 슬롯에 배정하고
    # 그 슬롯의 (국면 × ball_x × [x0,y0,dx_dby,dy_dby]) 표를 스냅샷한다. 배정은 정적 기하라
    # 호스트에서 한 번만 계산한다.
    # 실측 위치장 표는 **포메이션마다 다르다** — 슬롯 배정이 홈 좌표에서 나오기 때문이다.
    # 경기 중에 모양이 바뀌므로 레이아웃마다 표를 미리 만들어 두고 런타임에는 고르기만 한다.
    # 레이아웃이 작은 categorical인 덕분에 가능하고, 이것이 자유 앵커 대신 표를 택한 이유다.
    gk_mask = np.asarray(gk) > 0.5
    teams0 = np.asarray(st0.team_id)
    # 실측 위치장 슬롯 배정은 **킥오프 기준 기하**로 한 번 정한다. 경기 중 인원이 줄면
    # 앵커는 따라 줄지만(``formation_home``), 표 배정까지 매 프레임 다시 푸는 것은 비용이
    # 크고 이득이 불분명하다 — 그 실험은 별도 항목으로 둔다.
    layout_table = np.stack([
        np.asarray(env.formation_layout_anchors(index))
        for index in range(len(F_LAYOUTS))
    ])
    detailed_role_table = jax.vmap(
        lambda anchors: player_roles.classify_roles(
            anchors, gk, st0.team_id, st0.active_player
        )
    )(jnp.asarray(layout_table))
    detailed_roles = detailed_role_table[0]
    roles = _coarse_roles_from_detailed(detailed_roles)
    per_layout = [
        POS.player_tables(layout_table[index], gk_mask, teams0)
        for index in range(layout_table.shape[0])
    ]
    anchor_tables = np.stack([rows[0] for rows in per_layout])   # (L, N, P, K, 4)
    anchor_slots = np.stack([rows[1] for rows in per_layout])    # (L, N)
    # 킥오프 레이아웃(0번)이 종전 값이다 — 명령이 없으면 정확히 예전과 같아야 한다.
    solver_cache = getattr(env, "_rule_policy_solver_cache", None)
    solver_cache_key = (
        RULE_POLICY_VERSION,
        policy_config.drive_arrive_speed_mps,
        env.dynamics_fingerprint,
    )
    cache_current = (
        isinstance(solver_cache, tuple)
        and len(solver_cache) == 4
        and solver_cache[0] == solver_cache_key
    )
    if not cache_current:
        ksolve = _calibrate_kick_solver(
            env, policy_config.drive_arrive_speed_mps
        )                                           # 발킥 솔버(거리→속도 역테이블)
        launch_h = float(st0.head_z[1]) + e.throw_height       # 스로인 손 높이(대표값)
        tsolve = _calibrate_throw_solver(env, launch_h)        # 스로인 전용 솔버
        ssolve = _calibrate_shot_solver(env)                   # 슛 발사각 솔버
        solver_cache = (solver_cache_key, ksolve, tsolve, ssolve)
        env._rule_policy_solver_cache = solver_cache
    else:
        _, ksolve, tsolve, ssolve = solver_cache

    spec = env.obs_spec()
    an = spec["anchor"]["features"]
    pf = spec["players"]["features"]
    bf = spec["ball"]["features"]
    ctx = spec["context"]["features"]

    ctx_ns = types.SimpleNamespace(
        N=env.N, n=env.n_agents, hx=env.hx, hy=env.hy, length=s.length, width=s.width,
        goal_w=env.goal_w, r_ball=env.r_ball, r_player=env.r_player,
        f2b_max=e.f2b_speed_max, shoot_range=e.f2b_shoot_range,
        drib_max=e.dribble_speed_max, launch_max=e.launch_max,
        launch_down_ground=e.launch_down_ground,
        launch_down_ref=e.launch_down_ref,
        n_pvel=e.norm_player_vel, n_bvel=e.norm_ball_vel, n_bz=e.norm_ball_z,
        n_body_z=e.norm_body_z,
        clear_dist=e.clear_dist, game_dur=env.game_duration, gravity=e.g,
        ball_drag=e.c_drag,
        control_dt=env.control_dt,
        f2b_action_horizon=max(
            0.0, env.control_dt - env.timebase.dt_phys
        ),
        reach_xy_carry=e.reach_xy_carry,
        reach_xy_challenge=e.reach_xy_challenge,
        reach_block_limit=e.reach_block_limit,
        reach_height_penalty=e.reach_height_penalty,
        long_stamina_vmax_floor=e.long_stamina_vmax_floor,
        short_stamina_vmax_floor=e.short_stamina_vmax_floor,
        short_stamina_headroom_knee=e.short_stamina_headroom_knee,
        offside_margin=e.offside_margin,
        roll_v_knots=jnp.asarray(e.roll_v_knots), roll_d_knots=jnp.asarray(e.roll_d_knots),
        team_id=st0.team_id.astype(jnp.int32), gk=gk,
        roles=roles, detailed_roles=detailed_roles, home_att=home_att,
        detailed_role_tab=detailed_role_table,
        role_home_tab=jnp.asarray(layout_table),
        anchor_tab=jnp.asarray(anchor_tables), anchor_slot=jnp.asarray(anchor_slots),
        anchor_knots=jnp.asarray(POS.BALL_X_KNOTS),
        anchor_mean_x=jnp.asarray(POS.ANCHOR_MEAN_X),
        anchor_spread_x=jnp.asarray([
            policy_config.anchor_spread_x_attack,
            policy_config.anchor_spread_x_defend,
            0.5 * (policy_config.anchor_spread_x_attack
                   + policy_config.anchor_spread_x_defend),
        ], dtype=jnp.float32),
        anchor_spread_y=jnp.asarray([
            policy_config.anchor_spread_y_attack,
            policy_config.anchor_spread_y_defend,
            0.5 * (policy_config.anchor_spread_y_attack
                   + policy_config.anchor_spread_y_defend),
        ], dtype=jnp.float32),
        styles=resolved_styles,
        policy=policy_config,
        execution_profile=resolved_execution_profile,
        # Causal-Compact: 전 슬롯이 같은 22차원 토큰이고 관측자 자신도 그 안에 있다.
        players_start=spec["players"]["start"], players_size=spec["players"]["size"],
        others_idx=env.others_idx,
        i_self_pos=an["abs_pos"], i_self_vel=an["abs_vel"],
        p_relpos=pf["rel_pos"], p_relvel=pf["rel_vel"], p_gk=pf["is_gk"],
        p_taker=pf["taker_mask"], p_team=pf["team_relation"], p_status=pf["status_code"],
        p_offside=pf["offside_latch"], p_yellow=pf["yellow"], p_cooldown=pf["cooldown"],
        p_vmax=pf["vmax"], p_ctrl=pf["ball_ctrl"],
        p_formation_home=pf["formation_home"],
        p_layout_index=pf["layout_index"],
        layout_count=int(layout_table.shape[0]),
        # 관측이 인덱스를 나누는 분모와 **같은 값**이어야 한다. 표 길이로 나누면
        # 레이아웃이 하나 늘 때마다 모든 인덱스가 어긋난다 — 실측: 63으로 인코딩된
        # 3-4-3(36번)을 36으로 복원해 21번 레이아웃으로 읽었다.
        layout_scale=float(LAYOUT_INDEX_CAPACITY),
        p_stamina_short=pf["stamina_short"],
        p_stamina_long=pf["stamina_long"],
        p_endurance=pf["endurance_factor"],
        p_reach_z=pf["reach_z"],
        p_head_z=pf["head_z"],
        i_ball_pos=bf["rel_pos"], i_ball_z=bf["abs_z"], i_ball_alive=bf["ball_alive"],
        i_poss=bf["possession_relation"],
        i_ball_vel=bf["rel_vel"], i_ball_vel_z=bf["abs_vel_z"],
        i_rt=ctx["restart_steps"][0], i_rk=ctx["restart_kind_code"][0],
        i_fk_indirect=ctx["restart_indirect"][0],
        i_last_touch=ctx["last_touch_relation"][0],
        i_last_touch_code=ctx["last_touch_code"][0],
        i_possession_steps=ctx["possession_steps"][0],
        i_previous_possession=ctx["previous_possession_relation"][0],
        possession_context_s=POSSESSION_CONTEXT_SECONDS,
        i_gk_handling_restricted=ctx["gk_handling_restricted_relation"][0],
        pen_len=env.pen_len, pen_hw=env.pen_hw, gk_catch_cap=e.gk_catch_speed_cap,
        center_circle_radius=s.center_circle_radius,
        penalty_arc_radius=s.penalty_arc_radius,
        corner_arc_radius=s.corner_arc_radius,
        throwin_clear=e.throwin_clear,
        player_boundary_margin=e.player_boundary_margin,
        goal_line_tolerance=e.goal_line_tolerance,
        goal_post_tolerance=e.goal_post_tolerance,
        legal_margin_floor=e.legal_margin_floor,
        loft_R=ksolve["loft_R"], loft_v=ksolve["loft_v"], loft_T=ksolve["loft_T"],
        drive_R=ksolve["drive_R"], drive_v=ksolve["drive_v"],
        cross_R=ksolve["cross_R"], cross_v=ksolve["cross_v"], cross_T=ksolve["cross_T"],
        loft_launch01=ksolve["loft_launch01"], cross_launch01=ksolve["cross_launch01"],
        drive_launch01=ksolve["drive_launch01"],
        throw_R=tsolve["throw_R"], throw_v=tsolve["throw_v"], throw_launch01=tsolve["throw_launch01"],
        throw_speed_max=e.throw_speed_max,
        shot_d=ssolve["shot_d"], shot_launch01=ssolve["shot_launch01"], shot_pow=ssolve["shot_pow"],
        shot_curve_y_unit=ssolve["shot_curve_y_unit"],
    )

    def policy_fn(obs, key, aff):
        """(obs, key, affordance view) → action (N, ACTION_DIM=8).

        ``aff``는 ``env.affordance_view(state)``의 반환 dict다. 관측에 없는 파생량을 정책이
        스스로 유도하지 않고 env의 단일 구현에서 받는다.
        """
        key = _validate_policy_key(key, name="policy key")
        obs, aff, inputs_finite = _validate_policy_inputs(
            obs, aff, n_players=env.N, obs_dim=env.obs_dim
        )
        action = _rule_based_actions(obs, key, ctx_ns, aff)
        if inputs_finite is None:
            return action
        return jnp.where(inputs_finite, action, jnp.zeros_like(action))

    def policy_with_trace(obs, key, aff):
        """Return the action and its fixed-shape Phase S decision trace.

        The tactical core is evaluated exactly once.  The four ``teacher_*``
        leaves can be copied directly into the equally named
        ``ACTION_SCENE_SPECS`` fields.  ``coord_*`` leaves are the policy-known
        part of ``EpisodeCaptureBuilder.append_scene(..., coordination=...)``;
        the rollout adapter supplies entry-State slot generations and the
        one-step ``teacher_kick_applied`` outcome.
        """
        key = _validate_policy_key(key, name="policy key")
        obs, aff, inputs_finite = _validate_policy_inputs(
            obs, aff, n_players=env.N, obs_dim=env.obs_dim
        )
        action, trace = _rule_based_actions(
            obs, key, ctx_ns, aff, with_trace=True
        )
        if inputs_finite is None:
            return action, trace
        action = jnp.where(inputs_finite, action, jnp.zeros_like(action))
        absent_slots = frozenset(
            {"coord_passer_slot", "coord_planned_receiver_slot"}
        )
        trace = {
            name: jnp.where(
                inputs_finite,
                value,
                (
                    jnp.full_like(value, -1)
                    if name in absent_slots
                    else jnp.zeros_like(value)
                ),
            )
            for name, value in trace.items()
        }
        return action, trace

    def from_state(state, key):
        """state → action. 관측과 affordance를 같은 상태에서 함께 만든다."""
        return policy_fn(env.get_obs_array(state), key, env.affordance_view(state))

    def act_dict(obs_dict, key, aff):
        """JaxMARL dict 어댑터 — get_obs dict → action dict. env.step(dict API)와 함께 쓸 때."""
        _validate_exact_mapping_keys(
            obs_dict, env._agent_keys, name="policy observation dict"
        )
        rows = [
            _as_policy_float32_array(
                obs_dict[agent],
                name=f"policy observation dict[{agent!r}]",
                shape=(env.obs_dim,),
            )
            for agent in env._agent_keys
        ]
        stacked = jnp.stack(rows)
        act = policy_fn(stacked, key, aff)
        return {a: act[i] for i, a in enumerate(env._agent_keys)}

    policy_fn.act_dict = act_dict
    policy_fn.from_state = from_state
    policy_fn.policy_with_trace = policy_with_trace
    policy_fn.team_styles = ctx_ns.styles
    policy_fn.roles = ctx_ns.roles
    policy_fn.policy_config = policy_config
    policy_fn.rule_policy_version = RULE_POLICY_VERSION
    policy_fn.execution_profile = resolved_execution_profile
    # env의 접촉 게이트와 정책의 예측 게이트가 같은 선을 쓰는지 밖에서 확인할 수 있게
    # 스냅샷된 상수를 노출한다. 둘이 갈리면 정책이 잡을 수 없는 공에 주자를 붙인다.
    policy_fn.reach_gate = (ctx_ns.reach_block_limit, ctx_ns.reach_height_penalty)
    policy_fn.positional_field_version = POS.POSITIONAL_FIELD_VERSION
    policy_fn.deadball_positioning_version = DB.DEADBALL_POSITIONING_VERSION
    policy_fn.deadball_positioning_source_digest = DB.METADATA["source"]["source_digest"]
    return policy_fn


def _nearest_radial_wait_points(
    player_pos, center, radius, fallback_direction, c, boundary_margin=0.0
):
    """선택한 직사각 경계 안에서 제한 원 밖의 가장 가까운 점을 반환한다.

    정상 데드볼 대기 목표는 ``boundary_margin=0``으로 실제 피치와 원의 교집합을
    사용한다. 현재 침범자의 즉시 탈출만 env와 같은 5 m 선수 물리 경계에서 계산한다.
    두 공간을 합치면 라인 밖 침범자가 피치 안으로 원을 가로질러 심판 투영과 싸운다.
    """

    boundary_margin = jnp.asarray(boundary_margin, dtype=player_pos.dtype)
    bx = c.hx + boundary_margin
    by = c.hy + boundary_margin
    rel = player_pos - center
    dist = jnp.linalg.norm(rel, axis=1)
    direction = jnp.where(
        (dist > GEOMETRY_EPS)[:, None],
        rel / (dist[:, None] + DIV_EPS),
        _unit(fallback_direction),
    )
    radial = center + direction * radius[:, None]
    radial_valid = (
        (jnp.abs(radial[:, 0]) <= bx + GEOMETRY_EPS)
        & (jnp.abs(radial[:, 1]) <= by + GEOMETRY_EPS)
    )

    x_edges = jnp.asarray([-bx, bx], dtype=player_pos.dtype)
    x_delta = x_edges[None, :] - center[:, 0:1]
    x_sq = radius[:, None] ** 2 - x_delta ** 2
    x_root = jnp.sqrt(jnp.maximum(x_sq, 0.0))
    x_coord = jnp.broadcast_to(x_edges[None, :], x_root.shape)
    x_plus = jnp.stack([x_coord, center[:, 1:2] + x_root], axis=2)
    x_minus = jnp.stack([x_coord, center[:, 1:2] - x_root], axis=2)
    x_points = jnp.concatenate([x_plus, x_minus], axis=1)
    x_valid_base = (x_sq >= 0.0)
    x_valid = jnp.concatenate([
        x_valid_base & (jnp.abs(x_plus[:, :, 1]) <= by + GEOMETRY_EPS),
        x_valid_base & (jnp.abs(x_minus[:, :, 1]) <= by + GEOMETRY_EPS),
    ], axis=1)

    y_edges = jnp.asarray([-by, by], dtype=player_pos.dtype)
    y_delta = y_edges[None, :] - center[:, 1:2]
    y_sq = radius[:, None] ** 2 - y_delta ** 2
    y_root = jnp.sqrt(jnp.maximum(y_sq, 0.0))
    y_coord = jnp.broadcast_to(y_edges[None, :], y_root.shape)
    y_plus = jnp.stack([center[:, 0:1] + y_root, y_coord], axis=2)
    y_minus = jnp.stack([center[:, 0:1] - y_root, y_coord], axis=2)
    y_points = jnp.concatenate([y_plus, y_minus], axis=1)
    y_valid_base = (y_sq >= 0.0)
    y_valid = jnp.concatenate([
        y_valid_base & (jnp.abs(y_plus[:, :, 0]) <= bx + GEOMETRY_EPS),
        y_valid_base & (jnp.abs(y_minus[:, :, 0]) <= bx + GEOMETRY_EPS),
    ], axis=1)

    candidates = jnp.concatenate(
        [radial[:, None, :], x_points, y_points], axis=1
    )
    valid = jnp.concatenate(
        [radial_valid[:, None], x_valid, y_valid], axis=1
    )
    sq_distance = jnp.sum((candidates - player_pos[:, None, :]) ** 2, axis=2)
    choice = jnp.argmin(jnp.where(valid, sq_distance, jnp.inf), axis=1)
    return candidates[jnp.arange(player_pos.shape[0]), choice]


def _clip_policy_target_to_player_domain(target, c, boundary_margin=None):
    """Keep every voluntary movement target inside the current player domain.

    Most formation targets live comfortably inside the pitch, but support offsets,
    aerial landing extrapolation and pressure leads are deliberately relative to a
    moving player/ball.  Near a touchline those relative targets can therefore lie
    beyond the hard player boundary.  Letting the environment clip the resulting
    motion makes the policy push against an immovable wall on every frame.
    """

    margin = jnp.asarray(
        c.player_boundary_margin if boundary_margin is None else boundary_margin,
        dtype=target.dtype,
    )
    limit = jnp.stack(
        [c.hx + margin, c.hy + margin], axis=-1
    )
    return jnp.clip(target, -limit, limit)


def _remove_outward_boundary_motion(
    player_pos, move_dir, c, boundary_margin=None
):
    """Remove only the outward component at the hard player boundary.

    This is applied after anti-clump blending: a legal target alone is insufficient,
    because teammate repulsion can rotate its final command back through the wall.
    Tangential and inward motion remain untouched.
    """

    margin = jnp.asarray(
        c.player_boundary_margin if boundary_margin is None else boundary_margin,
        dtype=player_pos.dtype,
    )
    bx = c.hx + margin
    by = c.hy + margin
    block_x = (
        ((player_pos[:, 0] >= bx - GEOMETRY_EPS) & (move_dir[:, 0] > 0.0))
        | ((player_pos[:, 0] <= -bx + GEOMETRY_EPS) & (move_dir[:, 0] < 0.0))
    )
    block_y = (
        ((player_pos[:, 1] >= by - GEOMETRY_EPS) & (move_dir[:, 1] > 0.0))
        | ((player_pos[:, 1] <= -by + GEOMETRY_EPS) & (move_dir[:, 1] < 0.0))
    )
    constrained = jnp.stack(
        [
            jnp.where(block_x, 0.0, move_dir[:, 0]),
            jnp.where(block_y, 0.0, move_dir[:, 1]),
        ],
        axis=1,
    )
    return _unit(constrained)


def _slot_pick(entries, slot):
    """슬롯 번호로 배치 후보 중 하나를 고른다.

    ``entries``는 (N, 2) 배열 10개(포메이션 슬롯 수)다.  GK나 미배정(-1)은 0번으로
    접히지만 호출부가 GK를 따로 덮으므로 결과에 영향이 없다.
    """
    stacked = jnp.stack(entries, axis=1)                       # (N, S, 2)
    idx = jnp.clip(slot, 0, len(entries) - 1)
    return jnp.take_along_axis(stacked, idx[:, None, None], axis=1)[:, 0, :]


def _wall_points(ball_field, own_goal, goal_w, radius, index, count):
    """프리킥 벽의 index번째 자리.

    벽은 '골 중심 선'이 아니라 **근포스트 선**에서 시작해 중앙으로 이어 붙인다.  실제
    수비 벽이 가리는 것은 키커가 감아 넣는 근포스트 쪽이고, 골 중심 기준으로 세우면
    벽이 통째로 반대편으로 밀려 근포스트가 열린다.
    """
    near_sign = jnp.sign(ball_field[:, 1] + DIV_EPS)
    near_post = jnp.stack([
        jnp.full(ball_field.shape[0], own_goal[0]),
        near_sign * (goal_w * 0.5),
    ], axis=1)
    to_near = _unit(near_post - ball_field)
    to_goal = _unit(own_goal[None, :] - ball_field)
    anchor = ball_field + to_near * radius
    perp = jnp.stack([-to_near[:, 1], to_near[:, 0]], axis=1)
    # 중앙(골 중심 선) 쪽 부호 — 벽이 근포스트에서 골 중심으로 자란다.
    center_ref = ball_field + to_goal * radius
    step_sign = jnp.sign(
        jnp.sum(perp * (center_ref - anchor), axis=1) + DIV_EPS
    )
    offset = (index.astype(jnp.float32) + 0.5) * 0.78
    point = anchor + perp * (step_sign * offset)[:, None]
    in_wall = index < count
    return point, in_wall


def _setpiece_structure(
    ball_field,
    anchor_target,
    slot,
    is_gk,
    is_sp_ours,
    rk,
    c,
    *,
    return_measured_valid=False,
):
    """세트피스 비키커의 데이터 유래 배치 목표를 만든다(합법성 투영 전).

    구판은 재개 중 비키커를 오픈플레이 대형 좌표에 그대로 세워 두었다.  그래서 코너에
    박스 점유가 없고, 프리킥에 벽이 서지 않고, 스로인에 지원 삼각형이 없었다 — 실제
    경기를 보는 느낌이 가장 크게 깨지는 지점이다.  여기서는 슬롯(정적 포메이션 서열)에
    세트피스 임무를 배정했다. v14부터 그 구조는 표본이 없는 종류·구간의 fallback이고,
    코너·프리킥·스로인·골킥은 K리그 이벤트 시점 선수 위치의 조건부 중심을 사용한다.
    슬롯은 관측자마다 같은 정적 배열이므로 22개 행이 각자 계산해도 배치가 일관된다.

    좌표는 관측자 자기 공격 프레임이다: 상대 골 ``+hx``, 우리 골 ``-hx``.
    """
    N = ball_field.shape[0]
    hx, hy = c.hx, c.hy
    ones = jnp.ones(N)
    own_goal_v = jnp.stack([-ones * hx, jnp.zeros(N)], axis=1)
    own_goal = jnp.array([-hx, 0.0])
    bx, by = ball_field[:, 0], ball_field[:, 1]
    side = jnp.sign(by + DIV_EPS)                     # 공이 있는 터치라인/포스트 쪽
    ours = is_sp_ours > 0.5
    theirs = is_sp_ours < -0.5

    def pt(x, y):
        return jnp.stack([x, y], axis=1)

    # ── 코너: 우리 공격 ──────────────────────────────────────────────────
    # 2명은 뒤에 남기고(역습 대비), 6명이 박스와 박스 앞을 채운다. 실제 코너의 기본형이다.
    corner_ours = [
        pt(-4.0 * ones, 14.0 * ones),                       # 0 FB  하프웨이 커버
        pt(-4.0 * ones, -14.0 * ones),                      # 1 FB  하프웨이 커버
        pt(ones * (hx - 5.5), side * 3.0),                  # 2 CB  니어포스트
        pt(ones * (hx - 5.5), -side * 4.5),                 # 3 CB  파포스트
        pt(bx - 7.5, by - side * 4.0),                      # 4 WM  숏코너 지원
        pt(ones * (hx - 24.0), -side * 7.0),                # 5 WM  세컨볼
        pt(ones * (hx - 19.0), side * 2.0),                 # 6 CM  박스 앞
        pt(ones * (hx - 10.5), -side * 1.0),                # 7 W   페널티 스팟
        pt(ones * (hx - 14.0), -side * 7.0),                # 8 W   파포스트 뒤쪽
        pt(ones * (hx - 7.5), side * 7.5),                  # 9 ST  니어포스트 앞 침투
    ]
    corner_ours_gk = pt(-ones * (hx - 2.0), jnp.zeros(N))

    # ── 코너: 상대 공격(우리 수비) ────────────────────────────────────────
    # 포스트 2명 + 6야드 2명 + 스팟 2명 + 박스 앞 1명 + 숏코너 견제 1명 + 아웃렛 2명.
    corner_theirs = [
        pt(-ones * (hx - 0.8), side * 3.4),                 # 0 FB  니어포스트
        pt(-ones * (hx - 0.8), -side * 3.4),                # 1 FB  파포스트
        pt(-ones * (hx - 5.5), side * 1.0),                 # 2 CB  6야드 니어
        pt(-ones * (hx - 5.5), -side * 5.0),                # 3 CB  6야드 파
        pt(-ones * (hx - 11.0), side * 2.5),                # 4 WM  스팟 니어
        pt(-ones * (hx - 11.0), -side * 5.5),               # 5 WM  스팟 파
        pt(-ones * (hx - 18.5), -side * 1.0),               # 6 CM  박스 앞(세컨볼)
        ball_field + _unit(own_goal_v - ball_field) * (c.clear_dist + 1.4),
        pt(-ones * (hx - 32.0), side * 11.0),               # 8 W   아웃렛
        pt(4.0 * ones, -side * 8.0),                        # 9 ST  하프웨이 아웃렛
    ]
    corner_theirs_gk = pt(-ones * (hx - 1.2), side * 1.2)

    # ── 프리킥/오프사이드 재개: 상대 공격(우리 수비) — 벽 + 라인 ─────────────
    d_ball_own = jnp.linalg.norm(ball_field - own_goal[None, :], axis=1)
    central = jnp.abs(by) < 22.0
    # 공이 골에 너무 가까우면 규정 이격(9.15 m)을 지킨 벽이 골라인 뒤에 놓인다 — 그런 프리킥은
    # 벽이 아니라 골라인 수비다. 벽이 물리적으로 설 수 있는 거리에서만 세운다.
    wall_possible = d_ball_own > c.clear_dist + 4.0
    wall_count = jnp.where(
        (d_ball_own < 24.0) & central, 4,
        jnp.where(d_ball_own < 30.0, 3, jnp.where(d_ball_own < 38.0, 2, 0)),
    )
    wall_count = jnp.where(wall_possible, wall_count, 0)
    wall_radius = c.clear_dist + c.policy.restart_wait_margin
    # 벽에 서는 슬롯: 이미 골사이드에 있는 센터백 1명 + 미드 3명. 미드만으로 세우면 전원이
    # 공 반대편에서 출발해 벽이 서기 전에 킥이 나간다(측정: 벽 인원 0~1명).
    wall_slot_index = jnp.array([9, 9, 0, 9, 1, 2, 3, 9, 9, 9])
    my_wall_index = jnp.take(wall_slot_index, jnp.clip(slot, 0, 9))
    wall_pos, in_wall = _wall_points(
        ball_field, own_goal, c.goal_w, wall_radius, my_wall_index, wall_count
    )
    # 벽 밖 수비수는 공보다 살짝 앞에 라인을 만들고(오프사이드 트랩) 대형 y를 지킨다.
    # 단 이 라인은 **우리 골을 위협하는 프리킥**에서만 의미가 있다. 상대가 자기 진영 깊은
    # 곳에서 재개할 때 같은 규칙을 쓰면 우리 팀 전체가 상대 박스 앞까지 끌려간다.
    fk_line_x = jnp.clip(bx + 2.0, -hx + 7.0, 0.0)
    fk_line = jnp.stack([fk_line_x, anchor_target[:, 1] * 0.85], axis=1)
    fk_threatening = d_ball_own < 40.0
    fk_theirs = jnp.where(
        in_wall[:, None],
        wall_pos,
        jnp.where(fk_threatening[:, None], fk_line, anchor_target),
    )
    # GK는 벽이 가리지 않는 파포스트 쪽으로 치우쳐 선다.
    fk_theirs_gk = pt(-ones * (hx - 1.6), -side * 1.6)

    # ── 프리킥: 우리 공격 ────────────────────────────────────────────────
    attack_third = bx > hx * 0.28
    fk_box = [
        pt(-2.0 * ones, 13.0 * ones),                       # 0 FB
        pt(-2.0 * ones, -13.0 * ones),                      # 1 FB
        pt(ones * (hx - 7.0), side * 3.5),                  # 2 CB 니어
        pt(ones * (hx - 7.0), -side * 4.5),                 # 3 CB 파
        pt(bx - 1.5, by + side * 6.0),                      # 4 WM 공 옆(레이오프)
        pt(ones * (hx - 21.0), -side * 8.0),                # 5 WM 세컨볼
        pt(bx - 4.0, by - side * 5.0),                      # 6 CM 백패스 옵션
        pt(ones * (hx - 11.5), side * 1.5),                 # 7 W  스팟
        pt(ones * (hx - 12.5), -side * 6.5),                # 8 W  파포스트
        pt(ones * (hx - 8.0), side * 6.0),                  # 9 ST 니어 침투
    ]
    fk_build = [
        anchor_target, anchor_target,
        anchor_target, anchor_target,
        pt(bx + 7.0, by + side * 9.0),                      # 4 WM 전진 옵션
        anchor_target,
        pt(bx - 6.0, by - side * 7.0),                      # 6 CM 백 옵션
        anchor_target, anchor_target,
        pt(jnp.minimum(bx + 24.0, hx - 12.0), by * 0.4),    # 9 ST 롱 타깃
    ]
    fk_ours = jnp.where(
        attack_third[:, None], _slot_pick(fk_box, slot), _slot_pick(fk_build, slot)
    )

    # ── 스로인 ─────────────────────────────────────────────────────────
    throw_line = hy - 2.0
    throw_ours = [
        anchor_target, anchor_target,
        anchor_target,
        pt(bx - 12.0, side * (throw_line - 4.0)),           # 3 CB  안전 백패스
        pt(bx + 7.0, side * throw_line),                    # 4 WM  라인 따라 앞
        anchor_target,
        pt(bx + 1.5, side * (throw_line - 10.0)),           # 6 CM  안쪽 지원
        pt(bx - 8.0, side * (throw_line - 2.5)),            # 7 W   뒤 지원
        anchor_target,
        pt(jnp.minimum(bx + 17.0, hx - 10.0), side * (throw_line - 6.0)),   # 9 ST 롱 타깃
    ]
    throw_ours = _slot_pick(throw_ours, slot)
    # 수비는 지원 옵션의 골사이드를 잡고 한 명이 던지는 선수를 견제한다.
    throw_theirs = [
        anchor_target, anchor_target,
        anchor_target,
        pt(bx - 13.5, side * (throw_line - 5.0)),
        pt(bx + 5.5, side * (throw_line - 1.5)),
        anchor_target,
        pt(bx + 0.0, side * (throw_line - 10.0)),
        ball_field + _unit(own_goal_v - ball_field) * (c.throwin_clear + c.policy.restart_wait_margin),
        anchor_target,
        pt(jnp.minimum(bx + 15.0, hx - 12.0), side * (throw_line - 7.0)),
    ]
    throw_theirs = _slot_pick(throw_theirs, slot)

    # ── 골킥 ───────────────────────────────────────────────────────────
    gkk_ours = [
        pt(-ones * (hx - 33.0), ones * (hy - 4.0)),         # 0 FB 높고 넓게
        pt(-ones * (hx - 33.0), -ones * (hy - 4.0)),        # 1 FB
        pt(-ones * (hx - 16.0), 13.0 * ones),               # 2 CB 박스 모서리로 벌림
        pt(-ones * (hx - 16.0), -13.0 * ones),              # 3 CB
        pt(-ones * (hx - 40.0), 9.0 * ones),                # 4 WM
        pt(-ones * (hx - 40.0), -9.0 * ones),               # 5 WM
        pt(-ones * (hx - 25.0), jnp.zeros(N)),              # 6 CM 내려받기
        pt(2.0 * ones, 14.0 * ones),                        # 7 W  롱볼 세컨볼
        pt(2.0 * ones, -14.0 * ones),                       # 8 W
        pt(5.0 * ones, jnp.zeros(N)),                       # 9 ST 롱볼 타깃
    ]
    press_x = hx - c.pen_len - 3.0
    gkk_theirs = [
        anchor_target, anchor_target, anchor_target, anchor_target,
        pt(ones * (press_x - 8.0), 12.0 * ones),
        pt(ones * (press_x - 8.0), -12.0 * ones),
        pt(ones * (press_x - 12.0), jnp.zeros(N)),
        pt(ones * press_x, 9.0 * ones),
        pt(ones * press_x, -9.0 * ones),
        pt(ones * press_x, jnp.zeros(N)),
    ]

    # ── 킥오프 ─────────────────────────────────────────────────────────
    # 킥오프는 양 팀 모두 자기 진영이라는 강한 제약이 있다. 위치장(ball_x=0)의 공격 행은
    # 절반 이상이 상대 진영이라 그대로 쓰면 합법성 투영이 전원을 하프웨이 라인에 일렬로
    # 붙여 버린다. 실제 킥오프 대형은 바로 기본 포메이션이므로 그것을 쓴다.
    kickoff_base = c.home_att
    kickoff_ours = [
        kickoff_base, kickoff_base, kickoff_base, kickoff_base,
        kickoff_base, kickoff_base,
        pt(-2.5 * ones, 2.0 * ones),                        # 6 CM 킥오프 파트너
        kickoff_base, kickoff_base,
        pt(-1.2 * ones, -1.2 * ones),                       # 9 ST 공 옆
    ]

    # ── 페널티: 리바운드 주자 ──────────────────────────────────────────
    pen_line_x = hx - c.pen_len - 1.5
    pen_ours = [
        anchor_target, anchor_target, anchor_target, anchor_target,
        pt(ones * pen_line_x, 8.0 * ones),
        pt(ones * pen_line_x, -8.0 * ones),
        pt(ones * (pen_line_x - 4.0), jnp.zeros(N)),
        pt(ones * pen_line_x, 3.0 * ones),
        pt(ones * pen_line_x, -3.0 * ones),
        pt(ones * (pen_line_x - 1.0), 0.8 * ones),
    ]
    pen_theirs = [
        pt(-ones * (hx - c.pen_len - 1.5), 6.0 * ones),
        pt(-ones * (hx - c.pen_len - 1.5), -6.0 * ones),
        pt(-ones * (hx - c.pen_len - 1.5), 2.0 * ones),
        pt(-ones * (hx - c.pen_len - 1.5), -2.0 * ones),
        anchor_target, anchor_target, anchor_target,
        anchor_target, anchor_target, anchor_target,
    ]

    corner = jnp.where(ours[:, None], _slot_pick(corner_ours, slot),
                       _slot_pick(corner_theirs, slot))
    corner_gk = jnp.where(ours[:, None], corner_ours_gk, corner_theirs_gk)
    freekick = jnp.where(ours[:, None], fk_ours, fk_theirs)
    freekick_gk = jnp.where(ours[:, None], anchor_target, fk_theirs_gk)
    throw = jnp.where(ours[:, None], throw_ours, throw_theirs)
    goalkick = jnp.where(ours[:, None], _slot_pick(gkk_ours, slot),
                         _slot_pick(gkk_theirs, slot))
    kickoff = jnp.where(
        ours[:, None], _slot_pick(kickoff_ours, slot), kickoff_base
    )
    penalty = jnp.where(ours[:, None], _slot_pick(pen_ours, slot),
                        _slot_pick(pen_theirs, slot))

    out = anchor_target
    out = jnp.where((rk[:, RK_CORNER] > 0.5)[:, None], corner, out)
    out = jnp.where(
        ((rk[:, RK_FREEKICK] > 0.5) | (rk[:, RK_OFFSIDE] > 0.5))[:, None],
        freekick, out,
    )
    out = jnp.where((rk[:, RK_THROWIN] > 0.5)[:, None], throw, out)
    out = jnp.where((rk[:, RK_GOALKICK] > 0.5)[:, None], goalkick, out)
    out = jnp.where((rk[:, RK_KICKOFF] > 0.5)[:, None], kickoff, out)
    out = jnp.where((rk[:, RK_PENALTY] > 0.5)[:, None], penalty, out)

    # GK는 세트피스에서도 자기 골문을 지킨다. 우리 재개면 배급 준비 위치(박스 앞),
    # 상대 재개면 실점 각도를 줄이는 라인 위치다.
    # 위협 거리 안의 상대 재개에서만 골라인 각도수비를 서고, 그 밖에는 실측 GK 위치장을
    # 그대로 쓴다. 미드필드 스로인에 GK가 골라인에 붙어 있으면 스위퍼 높이가 사라진다.
    gk_default = jnp.where(
        (theirs & (d_ball_own < 32.0))[:, None],
        pt(-ones * (hx - 1.4), jnp.clip(by * 0.10, -c.goal_w * 0.4, c.goal_w * 0.4)),
        anchor_target,
    )
    gk_out = gk_default
    gk_out = jnp.where((rk[:, RK_CORNER] > 0.5)[:, None], corner_gk, gk_out)
    gk_out = jnp.where(
        ((rk[:, RK_FREEKICK] > 0.5) | (rk[:, RK_OFFSIDE] > 0.5))[:, None],
        freekick_gk, gk_out,
    )
    gk_out = jnp.where((rk[:, RK_GOALKICK] > 0.5)[:, None], anchor_target, gk_out)
    gk_out = jnp.where((rk[:, RK_GK_HOLD] > 0.5)[:, None], anchor_target, gk_out)
    out = jnp.where(is_gk[:, None], gk_out, out)
    # Common restarts use the event-aligned K League positioning field.  The
    # handcrafted structure above remains an explicit fallback for kickoff,
    # penalty and statistically sparse cells.  Legality is deliberately not
    # baked into the fit: `_restart_policy_plan` applies the runtime SSOT's
    # distance/box/half constraints after this target is selected.
    measured, measured_valid = DB.target(
        ball_field, slot, is_gk, is_sp_ours, rk
    )
    out = jnp.where(measured_valid[:, None], measured, out)
    # The event timestamp gives an excellent team shape but does not reliably
    # distinguish which nearby defenders constitute the formal 9.15 m wall.
    # Preserve the measured targets for everyone else and impose only this
    # restart-law/tactical assignment on the selected wall roles.
    measured_wall = (
        theirs
        & (~is_gk)
        & in_wall
        & ((rk[:, RK_FREEKICK] > 0.5) | (rk[:, RK_OFFSIDE] > 0.5))
    )
    out = jnp.where(measured_wall[:, None], wall_pos, out)
    if return_measured_valid:
        return out, measured_valid
    return out


def _orbit_detour(point, goal, center, radius):
    """``point``→``goal`` 직선이 원 ``(center, radius)``를 지나면 원을 따라 돌아가게 한다.

    재개 제한구역은 심판이 매 프레임 강제 투영하는 실제 장벽이다. 가로지르는 직선 목표를
    주면 선수는 경계에서 계속 밀려나 영영 도착하지 못한다(측정: 프리킥 벽 인원 10초 뒤 0명).
    접점을 목표로 주는 방식은 **이미 원 위에 선 선수에게 자기 자신을 가리켜** 그대로 굳어
    버리므로 쓸 수 없다. 대신 목표의 방위각 쪽으로 한 걸음(최대 ~34°) 돌린 원 위의 점을
    준다 — 실제 선수도 공을 빙 돌아 벽 자리로 간다.

    원 **안**에 있는 선수는 건드리지 않는다. 그쪽은 심판의 최소 투영과 같은 기하로 먼저
    빠져나오는 것이 계약이고, :mod:`tests.test_rule_policy_restart`가 그것을 고정한다.
    """
    rel_p = point - center
    d = _safe_norm(rel_p)
    radius = jnp.broadcast_to(jnp.asarray(radius, d.dtype), d.shape)
    seg = goal - point
    seg_len2 = jnp.sum(seg * seg, axis=1)
    t = jnp.clip(
        jnp.sum((center - point) * seg, axis=1) / jnp.maximum(seg_len2, DIV_EPS),
        0.0,
        1.0,
    )
    closest = point + seg * t[:, None]
    crosses = (_safe_norm(center - closest) < radius) & (d >= radius - GEOMETRY_EPS)

    ang_p = jnp.arctan2(rel_p[:, 1], rel_p[:, 0])
    rel_g = goal - center
    ang_g = jnp.arctan2(rel_g[:, 1], rel_g[:, 0])
    diff = ang_g - ang_p
    sin_d, cos_d = jnp.sin(diff), jnp.cos(diff)
    delta = jnp.arctan2(sin_d, cos_d)
    # 정확히 반대편(≈180°)이면 delta 부호가 수치 잡음으로 매 프레임 뒤집혀 제자리 진동한다.
    # 그 퇴화점만 한쪽으로 고정한다.
    antipodal = (jnp.abs(sin_d) < 1e-3) & (cos_d < 0.0)
    delta = jnp.where(antipodal, jnp.abs(delta), delta)
    step = jnp.clip(delta, -0.60, 0.60)
    ring = jnp.maximum(d, radius + 0.30)
    ang = ang_p + step
    orbit = center + jnp.stack(
        [jnp.cos(ang), jnp.sin(ang)], axis=1
    ) * ring[:, None]
    return jnp.where(crosses[:, None], orbit, goal)


def _restart_policy_plan(my_field, ball_field, base_target, is_gk, is_kicker,
                         is_sp_ours, rk, c):
    """관측만으로 합법적인 재개 대기 목표를 만든다.

    두 층으로 나뉜다.

    1. **배치**  ``base_target``(세트피스 구조가 원하는 자리)을 제한구역 밖으로 투영한다.
       구판은 이격 대상 팀을 "그 자리에 서 있기"로 묶어 두어 코너의 포스트 배치도,
       프리킥의 벽도, 스로인의 마킹도 만들어지지 않았다.
    2. **탈출**  지금 서 있는 자리가 이미 불법이면 배치보다 **먼저** 규칙을 푼다. 이때의
       기하는 심판(env)의 최소 투영과 같다 — 페널티박스 안은 가까운 골 반대쪽 x축으로만,
       그 밖의 반경 제한은 현재 위치에서 가장 가까운 원 경계점으로.  두 층을 섞지 않아야
       "정책이 스스로 합법을 유지한다"는 계약이 유지되고 env 투영이 반복되지 않는다.

    페널티박스 안 위반은 y를 바꾸지 않고 가까운 골의 반대 방향(x)으로만 해소한다.
    """

    policy = c.policy
    restart_active = rk[:, RK_NONE] < 0.5
    is_kickoff = rk[:, RK_KICKOFF] > 0.5
    is_corner = rk[:, RK_CORNER] > 0.5
    is_goalkick = rk[:, RK_GOALKICK] > 0.5
    is_throw = rk[:, RK_THROWIN] > 0.5
    is_freekick = rk[:, RK_FREEKICK] > 0.5
    is_offside = rk[:, RK_OFFSIDE] > 0.5
    is_penalty = rk[:, RK_PENALTY] > 0.5
    is_hold = rk[:, RK_GK_HOLD] > 0.5
    opponent_restart = restart_active & (is_sp_ours < -0.5)
    non_taker = restart_active & (~is_kicker)
    safety = policy.restart_wait_margin

    target = jnp.where(non_taker[:, None], base_target, my_field)
    # 데이터 위치장의 희소-cell 외삽은 몇 cm 정도 라인 밖을 가리킬 수
    # 있다. 합법성 투영 *뒤*에 clip하면 원/박스 제약을 다시 깨뜨리므로,
    # 먼저 실제 피치에 놓고 아래의 모든 재개 제약을 그 좌표에 적용한다.
    field_limit = jnp.asarray([c.hx, c.hy], dtype=target.dtype)
    target = jnp.where(
        non_taker[:, None], jnp.clip(target, -field_limit, field_limit), target
    )
    wait_mask = jnp.zeros(my_field.shape[0], dtype=bool)

    # 원형 이격: 킥오프/스로인/코너/FK/오프사이드의 수비팀만 대상이다.
    radial_kind = is_kickoff | is_throw | is_corner | is_freekick | is_offside
    clear_r = jnp.where(
        is_kickoff,
        c.center_circle_radius,
        jnp.where(
            is_throw,
            c.throwin_clear,
            jnp.where(is_corner, c.clear_dist + c.corner_arc_radius, c.clear_dist),
        ),
    )
    target_r = clear_r + safety
    # The observed throw-in ball is deliberately inset from the touchline for
    # stable release physics.  Law 15's opponent clearance and the measured
    # support distances are referenced to the point on the touchline, so the
    # policy must not learn/command the 15 cm numerical inset as a rule spot.
    throw_side = jnp.where(
        ball_field[:, 1] != 0.0,
        jnp.sign(ball_field[:, 1]),
        jnp.where(c.home_att[:, 1] != 0.0, jnp.sign(c.home_att[:, 1]), 1.0),
    )
    throw_point = jnp.stack(
        [ball_field[:, 0], throw_side * c.hy], axis=1
    )
    corner_point = jnp.stack([
        jnp.where(ball_field[:, 0] != 0.0,
                  jnp.sign(ball_field[:, 0]), 1.0) * c.hx,
        throw_side * c.hy,
    ], axis=1)
    radial_center = jnp.where(
        is_throw[:, None],
        throw_point,
        jnp.where(is_corner[:, None], corner_point, ball_field),
    )

    home_rel = c.home_att - radial_center
    home_dist = jnp.linalg.norm(home_rel, axis=1)
    # 선수와 공이 정확히 겹친 퇴화점에서도 결정적인 바깥 방향을 갖는다.
    deterministic_fallback = jnp.stack([
        -jnp.where(radial_center[:, 0] >= 0.0, 1.0, -1.0),
        jnp.where(c.home_att[:, 1] >= radial_center[:, 1], 1.0, -1.0),
    ], axis=1)
    fallback = jnp.where(
        (home_dist > GEOMETRY_EPS)[:, None],
        home_rel / (home_dist[:, None] + DIV_EPS),
        _unit(deterministic_fallback),
    )

    def radial_plan(point, boundary_margin=0.0, *, referee_box_escape=False):
        """``point``를 제한 원 밖의 가장 가까운 합법 대기점으로 옮긴다.

        ``referee_box_escape``는 **현재 침범자**가 페널티지역 안에 있을 때만 쓴다.
        심판은 그 선수를 가까운 골의 반대 x방향으로 밀지만, 최종 대형의 합법적인 목표가
        페널티지역 안에 있다는 이유만으로 같은 처리를 하면 공격 프리킥의 벽이 공 반대편으로
        뒤집힌다. 계획 목표는 아래의 보통 최근접 원 투영을 그대로 사용한다.
        """
        rel = point - radial_center
        dist = jnp.linalg.norm(rel, axis=1)
        radial = _nearest_radial_wait_points(
            point,
            radial_center,
            target_r,
            fallback,
            c,
            boundary_margin=boundary_margin,
        )
        px_, py_ = point[:, 0], point[:, 1]
        in_right_box = (
            (px_ >= c.hx - c.pen_len) & (px_ <= c.hx) & (jnp.abs(py_) <= c.pen_hw)
        )
        in_left_box = (
            (px_ <= -c.hx + c.pen_len) & (px_ >= -c.hx) & (jnp.abs(py_) <= c.pen_hw)
        )
        away = jnp.where(in_right_box, -1.0, 1.0)
        rel_x = px_ - radial_center[:, 0]
        rel_y = py_ - radial_center[:, 1]
        x_root = jnp.sqrt(jnp.maximum(target_r ** 2 - rel_y ** 2, 0.0))
        x_step = jnp.maximum(0.0, -away * rel_x + x_root)
        box_radial = jnp.stack([px_ + away * x_step, py_], axis=1)
        radial = jnp.where(
            (referee_box_escape & (in_left_box | in_right_box))[:, None],
            box_radial,
            radial,
        )
        # 경계 '위'는 합법이다. 부동소수 오차로 위반 판정이 되면 원 위에 선 선수가
        # 매 프레임 최근접 탈출로 다시 고정돼 벽·마킹 자리로 이동하지 못한다.
        return radial, dist < target_r - GEOMETRY_EPS

    def on_own_goal_line(point):
        return (
            (jnp.abs(point[:, 0] + c.hx) <= c.goal_line_tolerance)
            & (jnp.abs(point[:, 1]) <= c.goal_w * 0.5 + c.goal_post_tolerance)
        )

    px, py = my_field[:, 0], my_field[:, 1]
    current_goal_line = on_own_goal_line(my_field)
    # A labelled restart actor is deliberately excluded from the measured
    # restart-team field (dead-ball table v2).  Keep a data-derived p05 floor
    # as a safety net for sparse cells/handcrafted fallback: a non-taker should
    # never become a second thrower standing on the ball.
    support_rel = target - radial_center
    support_dist = jnp.linalg.norm(support_rel, axis=1)
    support_floor = policy.throwin_support_min_distance
    support_target = _nearest_radial_wait_points(
        target,
        radial_center,
        jnp.full_like(support_dist, support_floor),
        fallback,
        c,
    )
    own_throw_support = is_throw & non_taker & (is_sp_ours > 0.5)
    target = jnp.where(
        (own_throw_support & (support_dist < support_floor))[:, None],
        support_target,
        target,
    )

    radial_family = opponent_restart & radial_kind
    planned_goal_line = on_own_goal_line(target)
    plan_radial_subject = radial_family & (~planned_goal_line)
    exit_radial_subject = radial_family & (~current_goal_line)
    plan_radial, plan_inside = radial_plan(target)
    exit_radial, now_inside = radial_plan(
        my_field,
        boundary_margin=c.player_boundary_margin,
        referee_box_escape=True,
    )
    # 배치 목표가 제한구역 안이면 밖으로 옮긴다.
    target = jnp.where(
        (plan_radial_subject & plan_inside)[:, None], plan_radial, target
    )
    # 지금 자리가 불법이면 심판의 최소 투영과 같은 기하로 먼저 나간다.
    target = jnp.where(
        (exit_radial_subject & now_inside)[:, None], exit_radial, target
    )

    # Law 13 permits an opponent to remain between the posts on their own
    # goal line even inside 9.15 m. That exception belongs to the *current
    # point*, not to an unrelated planned point. To leave it without entering
    # the forbidden interior, first travel along the line to a circle/goal-line
    # intersection. If the whole goal mouth lies inside the circle, there is
    # no continuous legal pre-kick exit and the realistic action is to stay.
    goal_line_dx = -c.hx - radial_center[:, 0]
    goal_line_span = target_r ** 2 - goal_line_dx ** 2
    goal_line_root = jnp.sqrt(jnp.maximum(goal_line_span, 0.0))
    goal_line_y = jnp.stack(
        [radial_center[:, 1] + goal_line_root,
         radial_center[:, 1] - goal_line_root],
        axis=1,
    )
    goal_mouth_half = c.goal_w * 0.5 + c.goal_post_tolerance
    goal_line_candidate_valid = (
        (goal_line_span >= 0.0)[:, None]
        & (jnp.abs(goal_line_y) <= goal_mouth_half + GEOMETRY_EPS)
    )
    goal_line_candidate_distance = jnp.abs(goal_line_y - py[:, None])
    goal_line_choice = jnp.argmin(
        jnp.where(goal_line_candidate_valid, goal_line_candidate_distance, jnp.inf),
        axis=1,
    )
    selected_goal_line_y = jnp.take_along_axis(
        goal_line_y, goal_line_choice[:, None], axis=1
    )[:, 0]
    has_goal_line_exit = jnp.any(goal_line_candidate_valid, axis=1)
    goal_line_exit = jnp.stack(
        [jnp.full_like(selected_goal_line_y, -c.hx), selected_goal_line_y],
        axis=1,
    )
    goal_line_exit = jnp.where(
        has_goal_line_exit[:, None], goal_line_exit, my_field
    )
    leaves_goal_line_exception = (
        radial_family & current_goal_line & now_inside & (~planned_goal_line)
    )
    target = jnp.where(
        leaves_goal_line_exception[:, None], goal_line_exit, target
    )
    # 제한구역 **바깥**(또는 경계 위)에 있는데 목표까지 직선이 그 구역을 가로지르면
    # 원을 따라 돌아간다.
    target = jnp.where(
        radial_family[:, None],
        _orbit_detour(my_field, target, radial_center, target_r),
        target,
    )
    wait_mask = wait_mask | radial_family

    # Law 13 adds a box constraint to the ordinary 9.15 m radius when the
    # opponents take a free kick (including an offside IDFK) inside *their*
    # penalty area.  In each observer's attacking frame that area is the +x
    # box.  The referee enforces the same condition in ``restart.py``; owning
    # it here too prevents a legal radial target that is still in the box from
    # being commanded back through the referee projection on every frame.
    restart_own_box_spot = (
        (ball_field[:, 0] >= c.hx - c.pen_len)
        & (ball_field[:, 0] <= c.hx)
        & (jnp.abs(ball_field[:, 1]) <= c.pen_hw)
    )
    box_freekick_subject = (
        opponent_restart
        & (is_freekick | is_offside)
        & restart_own_box_spot
    )
    box_front_x = c.hx - c.pen_len - safety

    def leave_restart_box(point):
        px_, py_ = point[:, 0], point[:, 1]
        inside = (
            (px_ >= c.hx - c.pen_len)
            & (px_ <= c.hx)
            & (jnp.abs(py_) <= c.pen_hw)
        )
        return jnp.stack([jnp.minimum(px_, box_front_x), py_], axis=1), inside

    box_plan_target, box_plan_inside = leave_restart_box(target)
    # ``exit_radial`` is the exact x-only 9.15 m exit for a point in either
    # penalty area.  Start from it only when the current point also violates
    # the radius, then take the farther of the radial and box-front exits.
    current_radial = jnp.where(now_inside[:, None], exit_radial, my_field)
    box_exit_target, box_exit_inside = leave_restart_box(current_radial)
    target = jnp.where(
        (box_freekick_subject & box_plan_inside)[:, None],
        box_plan_target,
        target,
    )
    target = jnp.where(
        (box_freekick_subject & box_exit_inside)[:, None],
        box_exit_target,
        target,
    )
    wait_mask = wait_mask | box_freekick_subject

    # Kick-off is the intersection of a half-plane and (for opponents) the
    # centre-circle exclusion.  The environment projects both constraints;
    # keeping only the old radial rule made an opponent who was outside the
    # circle but across halfway command an exact standstill while the referee
    # moved them.  When both constraints are violated, aim directly at their
    # intersection so the voluntary policy cannot satisfy one by re-entering
    # the other.
    kickoff_subject = is_kickoff & non_taker

    def kickoff_plan(point, inside):
        px_, py_ = point[:, 0], point[:, 1]
        half_illegal = px_ > 0.0
        half_target_x = jnp.full_like(px_, -safety)
        half_target = jnp.stack([half_target_x, py_], axis=1)
        half_circle_dx = half_target_x - ball_field[:, 0]
        half_circle_span = target_r ** 2 - half_circle_dx ** 2
        half_circle_y = jnp.sqrt(jnp.maximum(half_circle_span, 0.0))
        rel_y = py_ - ball_field[:, 1]
        half_y_sign = jnp.where(
            jnp.abs(rel_y) > GEOMETRY_EPS,
            jnp.sign(rel_y),
            jnp.where(c.home_att[:, 1] >= ball_field[:, 1], 1.0, -1.0),
        )
        half_circle_target = jnp.stack(
            [half_target_x, ball_field[:, 1] + half_y_sign * half_circle_y],
            axis=1,
        )
        both = opponent_restart & is_kickoff & half_illegal & inside & (
            half_circle_span >= 0.0
        )
        return half_target, half_illegal, half_circle_target, both

    p_half, p_illegal, p_both_target, p_both = kickoff_plan(target, plan_inside)
    target = jnp.where((kickoff_subject & p_illegal)[:, None], p_half, target)
    target = jnp.where(p_both[:, None], p_both_target, target)
    e_half, e_illegal, e_both_target, e_both = kickoff_plan(my_field, now_inside)
    target = jnp.where((kickoff_subject & e_illegal)[:, None], e_half, target)
    target = jnp.where(e_both[:, None], e_both_target, target)
    wait_mask = wait_mask | kickoff_subject

    # 골킥 상대는 재개팀 자기 박스의 전면으로 빠져나온 뒤 그 자리를 유지한다. 각
    # 에이전트의 공격 프레임에서 상대 골은 항상 +hx이다.
    goalkick_subject = opponent_restart & is_goalkick

    def goalkick_plan(point):
        px_, py_ = point[:, 0], point[:, 1]
        inside = (
            (px_ >= c.hx - c.pen_len) & (px_ <= c.hx) & (jnp.abs(py_) <= c.pen_hw)
        )
        out = jnp.stack([jnp.minimum(px_, c.hx - c.pen_len - safety), py_], axis=1)
        return out, inside

    gk_plan_target, gk_plan_inside = goalkick_plan(target)
    gk_exit_target, gk_exit_inside = goalkick_plan(my_field)
    target = jnp.where(
        (goalkick_subject & gk_plan_inside)[:, None], gk_plan_target, target
    )
    target = jnp.where(
        (goalkick_subject & gk_exit_inside)[:, None], gk_exit_target, target
    )
    wait_mask = wait_mask | goalkick_subject

    # A player can be lawfully *outside* the penalty area while standing
    # behind the goal line or beside the box.  A straight command from there
    # to another legal target can nevertheless cross the rectangular area;
    # the referee would then project the player from just inside the goal line
    # all the way to the box front.  Stage a deterministic route around the
    # rectangle: (1) leave the goal-width lane laterally while still behind
    # goal, (2) travel along the box side to its front, (3) align with a
    # side-target from the front.  The following frame advances to the next
    # stage, so no hidden policy state is required.
    box_path_subject = box_freekick_subject | goalkick_subject
    my_x, my_y = my_field[:, 0], my_field[:, 1]
    side_y = c.pen_hw + safety
    side_sign = jnp.where(
        jnp.abs(my_y) > GEOMETRY_EPS,
        jnp.sign(my_y),
        jnp.where(
            jnp.abs(target[:, 1]) > GEOMETRY_EPS,
            jnp.sign(target[:, 1]),
            jnp.where(c.home_att[:, 1] >= 0.0, 1.0, -1.0),
        ),
    )
    current_box_inside = (
        (my_x >= c.hx - c.pen_len)
        & (my_x <= c.hx)
        & (jnp.abs(my_y) <= c.pen_hw)
    )
    behind_goal_lane = (
        (my_x > c.hx)
        & (jnp.abs(my_y) < side_y - GEOMETRY_EPS)
    )
    lateral_waypoint = jnp.stack([my_x, side_sign * side_y], axis=1)
    beside_or_behind = (
        (my_x > box_front_x + GEOMETRY_EPS)
        & (jnp.abs(my_y) >= c.pen_hw)
    )
    front_waypoint = jnp.stack([
        jnp.full_like(my_x, box_front_x), my_y
    ], axis=1)
    target_beside_box = (
        (target[:, 0] > box_front_x + GEOMETRY_EPS)
        & (jnp.abs(target[:, 1]) >= c.pen_hw)
    )
    target_side_sign = jnp.where(
        jnp.abs(target[:, 1]) > GEOMETRY_EPS,
        jnp.sign(target[:, 1]),
        side_sign,
    )
    side_alignment_waypoint = jnp.stack([
        jnp.full_like(my_x, box_front_x), target_side_sign * side_y
    ], axis=1)
    # The strip between the actual box front and our safety-offset waypoint is
    # already legally in front of the area.  Treat it as the front region;
    # otherwise a diagonal to a side target can shave through the real corner.
    at_or_in_front = my_x <= c.hx - c.pen_len + GEOMETRY_EPS

    path_target = target
    path_target = jnp.where(
        (box_path_subject & at_or_in_front & target_beside_box)[:, None],
        side_alignment_waypoint,
        path_target,
    )
    path_target = jnp.where(
        (box_path_subject & beside_or_behind & (~current_box_inside))[:, None],
        front_waypoint,
        path_target,
    )
    path_target = jnp.where(
        (box_path_subject & behind_goal_lane & (~current_box_inside))[:, None],
        lateral_waypoint,
        path_target,
    )
    target = path_target

    # 페널티는 키커 이외 전원이 박스와 아크 밖이고 마크 뒤다. 고정된 y에서 세 금지영역을
    # 모두 벗어나는 최소 x 이동을 취한다. 단, 수비 GK는 자기 골라인 밴드가 합법 대기점이다.
    spot_sign = jnp.where(ball_field[:, 0] >= 0.0, 1.0, -1.0)
    pen_away = -spot_sign
    front_x = spot_sign * (c.hx - c.pen_len)
    arc_r = c.penalty_arc_radius + safety

    def penalty_plan(point):
        px_, py_ = point[:, 0], point[:, 1]
        # A player can legally drift behind the goal line during live play.
        # If a penalty is then awarded, moving that player merely behind the
        # mark would put them *back inside* the penalty area. Treat the whole
        # goal-side half-ray (not only front-line..goal-line) as requiring the
        # same away-from-goal exit.
        in_pen_box = (
            (spot_sign * px_ >= spot_sign * front_x)
            & (jnp.abs(py_) <= c.pen_hw)
        )
        pen_boundary_x = front_x + pen_away * safety
        box_step = jnp.where(
            in_pen_box, jnp.maximum(0.0, pen_away * (pen_boundary_x - px_)), 0.0
        )
        pen_rel_x = px_ - ball_field[:, 0]
        pen_rel_y = py_ - ball_field[:, 1]
        arc_span = arc_r ** 2 - pen_rel_y ** 2
        arc_root = jnp.sqrt(jnp.maximum(arc_span, 0.0))
        # Compute the exit point on the whole permitted x ray, rather than gating
        # on the starting point being in the arc.  A player can start outside the
        # arc near the goal, leave the box along x, and cross into the arc en route.
        arc_step = jnp.where(
            arc_span > 0.0,
            jnp.maximum(0.0, -pen_away * pen_rel_x + arc_root),
            0.0,
        )
        # Law 14 also requires every non-kicker except the defending goalkeeper to
        # remain behind the penalty mark.  Lateral distance alone is insufficient:
        # a player outside both box and arc can still stand closer to goal than the
        # ball.  Use the same goal-opposite x-only intervention as the referee.
        mark_target_x = ball_field[:, 0] + pen_away * safety
        mark_step = jnp.maximum(0.0, pen_away * (mark_target_x - px_))
        step = jnp.maximum(jnp.maximum(box_step, arc_step), mark_step)
        return jnp.stack([px_ + pen_away * step, py_], axis=1), step > 0.0

    penalty_subject = is_penalty & non_taker
    pen_plan_target, _ = penalty_plan(target)
    pen_exit_target, pen_exit_need = penalty_plan(my_field)
    defending_gk = is_penalty & opponent_restart & is_gk
    gk_line_target = jnp.stack([
        jnp.full_like(px, -c.hx + 0.5 * c.goal_line_tolerance),
        jnp.clip(py, -c.goal_w * 0.5, c.goal_w * 0.5),
    ], axis=1)
    target = jnp.where(penalty_subject[:, None], pen_plan_target, target)
    target = jnp.where(
        (penalty_subject & pen_exit_need)[:, None], pen_exit_target, target
    )
    target = jnp.where(defending_gk[:, None], gk_line_target, target)
    wait_mask = wait_mask | penalty_subject

    # GK 홀드에는 이격 의무가 없고 키커는 언제나 공으로 간다.
    # 재개 중 비키커의 목표는 '대형 표류'가 아니라 의도된 배치다. anti-clump 반발이 그 위를
    # 덮으면 코너의 니어포스트·스팟 배치(서로 7m 남짓)가 통째로 흩어진다. 그래서 GK 홀드를
    # 뺀 모든 재개에서 비키커도 committed로 표시한다.
    wait_mask = (wait_mask | non_taker) & (~is_hold)
    # Event-aligned linear fits are deliberately allowed to extrapolate inside
    # each pitch third.  At the extreme edge of a sparse cell that can put a
    # non-taker a few centimetres behind a goal/touch line (most visibly the
    # goalkeeper at a deep throw-in).  The five-metre player physics margin is
    # for natural live-play momentum, not a dead-ball formation target.  Keep
    # every non-taker's voluntary waiting target on the field; the designated
    # thrower is handled separately by the referee/kicker path below.
    bounded_wait_target = jnp.clip(target, -field_limit, field_limit)
    # A current encroacher may legally be in the five-metre live-play margin.
    # Preserve the env-identical nearest exit for this first phase even when it
    # lies outside the touchline; on the next frame ``now_inside`` is false and
    # the ordinary in-pitch target/orbit takes over.
    preserve_immediate_radial_exit = exit_radial_subject & now_inside
    target = jnp.where(
        (non_taker & (~preserve_immediate_radial_exit))[:, None],
        bounded_wait_target,
        target,
    )
    # Orbit detours are computed on a circle and can briefly point outside the
    # rectangular pitch near a corner. If the final safety clip changed such a
    # target, restore the radius on the pitch/clearance-circle intersection.
    final_radial, final_inside = radial_plan(target)
    final_radial_subject = radial_family & (~on_own_goal_line(target))
    target = jnp.where(
        (final_radial_subject & final_inside)[:, None], final_radial, target
    )
    # An orbit can end inside the restart team's penalty area even though its
    # original measured target was outside. Reapply the Law-13 half-plane and
    # then the circle: moving from the box front away from a spot inside that
    # box gives the second radial projection a goal-opposite x component, so
    # the two passes converge on their intersection rather than alternating.
    final_box_target, final_box_inside = leave_restart_box(target)
    target = jnp.where(
        (box_freekick_subject & final_box_inside)[:, None],
        final_box_target,
        target,
    )
    final_radial, final_inside = radial_plan(target)
    final_radial_subject = radial_family & (~on_own_goal_line(target))
    target = jnp.where(
        (final_radial_subject & final_inside)[:, None], final_radial, target
    )
    target = jnp.where(is_kicker[:, None], ball_field, target)
    distance = jnp.linalg.norm(target - my_field, axis=1)
    power = jnp.clip(
        distance / policy.restart_target_slowdown_radius, 0.0, 1.0
    )
    return target, power, wait_mask


def _arrival_power(distance, policy):
    """목표까지 남은 거리로 이동 파워를 정한다(도착 감속).

    실측 속도 분포는 걷기·조깅이 절반을 넘는다.  고정 파워는 그 분포를 만들 수 없고
    (전원 상시 질주), 스태미너도 비현실적으로 빨리 마른다.  ``arrival_radius`` 밖은
    최대 순항 파워, 안쪽은 선형 감속이며 완전 정지 대신 최소 파워를 남긴다.
    """
    ramp = jnp.clip(distance / policy.arrival_radius, 0.0, 1.0)
    # 반경 밖으로 크게 벗어났으면(전환 직후 복귀·침투 주행) 순항을 넘어 전력까지 올린다.
    # 실측 스프린트는 대부분 이 '자리 회복/침투' 국면에서 나온다.
    surge = jnp.clip(
        (distance - policy.arrival_radius) / (3.5 * policy.arrival_radius), 0.0, 1.0
    )
    return (
        policy.offball_walk_floor
        + (policy.offball_cruise - policy.offball_walk_floor) * ramp
        + jnp.maximum(policy.offball_surge_cap - policy.offball_cruise, 0.0) * surge
    )


def _professional_ability(value, floor, ceiling):
    """Map a raw ability to a bounded lower-pro→elite-pro tactical scale.

    The environment may intentionally contain values outside the calibrated
    professional band.  Rule-policy decisions saturate at both ends so a low
    value never turns a valid professional into an incapable amateur, while the
    engine's physical contest model remains free to use the original value.
    """

    return jnp.clip((value - floor) / (ceiling - floor + DIV_EPS), 0.0, 1.0)


def _execution_noise_multiplier(ability, gain):
    """Return a symmetric, bounded technique multiplier around ability 0.5."""

    return 1.0 + gain * (0.5 - jnp.clip(ability, 0.0, 1.0))


def _dribble_escape_scores(
    space_m,
    signed_progress,
    directions,
    player_velocity,
    ability,
    policy,
):
    """Score local dribble exits with space, progress and body momentum.

    Lower-pro players favour the safer open lane; elite-pro players can accept
    a moderately tighter lane for progression.  Both ends retain every term,
    which prevents ability from becoming a binary can/cannot-dribble switch.
    """

    directions = jnp.asarray(directions)
    if directions.ndim == 2:
        directions = directions[None, :, :]
    velocity_speed = _safe_norm(player_velocity, axis=1)
    velocity_dir = player_velocity / (velocity_speed[:, None] + DIV_EPS)
    alignment = jnp.sum(directions * velocity_dir[:, None, :], axis=2)
    momentum = jnp.where(
        (velocity_speed > 0.8)[:, None],
        0.5 * (jnp.clip(alignment, -1.0, 1.0) + 1.0),
        0.5,
    )
    space = jnp.clip(space_m / 10.0, 0.0, 1.0)
    progress = 0.5 * (jnp.clip(signed_progress, -1.0, 1.0) + 1.0)
    ability = jnp.clip(ability, 0.0, 1.0)[:, None]
    space_weight = (
        policy.dribble_escape_space_weight
        + policy.dribble_escape_skill_tradeoff * (1.0 - ability)
    )
    progress_weight = (
        policy.dribble_escape_progress_weight
        + policy.dribble_escape_skill_tradeoff * ability
    )
    return (
        space_weight * space
        + progress_weight * progress
        + policy.dribble_escape_momentum_weight * momentum
    )


def _contextual_quick_relay_probability(
    base_probability,
    ability,
    alignment,
    player_speed,
    pressure,
    return_like,
    policy,
):
    """Contextual first-time-pass probability without hidden intention state."""

    ability_factor = (
        1.0
        - 0.5 * policy.quick_relay_skill_gain
        + policy.quick_relay_skill_gain * jnp.clip(ability, 0.0, 1.0)
    )
    alignment_level = jnp.clip(
        (alignment - policy.quick_relay_alignment_floor)
        / (1.0 - policy.quick_relay_alignment_floor + DIV_EPS),
        0.0,
        1.0,
    )
    body_factor = jnp.where(
        player_speed < 0.8, 0.90, 0.55 + 0.45 * alignment_level
    )
    pressure_level = jnp.clip(
        pressure / policy.pass_pressure_scale, 0.0, 1.0
    )
    pressure_factor = 0.85 + 0.15 * pressure_level
    return_factor = jnp.where(
        return_like, policy.quick_relay_return_scale, 1.0
    )
    contextual = jnp.clip(
        base_probability
        * ability_factor
        * body_factor
        * pressure_factor
        * return_factor,
        0.0,
        1.0,
    )
    # A public override of one has always meant a deterministic contract in the
    # focused tests and calibration CLI; preserve that exact endpoint.
    return jnp.where(base_probability >= 1.0, 1.0, contextual)


def _box_mark_plan(
    my_field,
    others_field,
    opponent_future,
    opponent_threat,
    opponent_candidate,
    self_available,
    other_available,
    own_goal,
    max_runners,
    goal_side_distance,
    player_slot,
    other_slot,
):
    """Greedily pair the top box threats with distinct free defenders.

    The old nearest-to-each-runner claim could still leave two runners to the
    same defender: that defender kept only the higher threat and the other was
    silently unmarked.  Assigning in threat order and removing each chosen
    defender gives one marker per runner while preserving the press/cover
    exclusions supplied by the caller.
    """

    candidate = jnp.asarray(opponent_candidate, dtype=bool)
    threat = jnp.where(candidate, opponent_threat, -jnp.inf)
    defender_position = jnp.concatenate(
        [my_field[:, None, :], others_field], axis=1
    )
    defender_available = jnp.concatenate(
        [self_available[:, None], other_available], axis=1
    )
    batch_size, defender_count = defender_available.shape
    opponent_count = candidate.shape[1]
    assigned_threat = -jnp.ones(
        (batch_size, defender_count), dtype=jnp.int32
    )
    used_defender = jnp.zeros_like(defender_available)
    used_threat = jnp.zeros((batch_size, opponent_count), dtype=bool)

    def assign_one(_, carry):
        assignments, defenders_used, threats_used = carry
        available_threat = candidate & (~threats_used)
        threat_index = _stable_argmin_index(
            -threat, available_threat, other_slot
        )
        has_threat = jnp.any(available_threat, axis=1)
        threat_position = jnp.take_along_axis(
            opponent_future, threat_index[:, None, None], axis=1
        )[:, 0, :]
        distance = _safe_norm(
            defender_position - threat_position[:, None, :], axis=2
        )
        free_defender = defender_available & (~defenders_used)
        defender_index = _stable_argmin_index(
            distance, free_defender, player_slot
        )
        can_assign = has_threat & jnp.any(free_defender, axis=1)
        chosen_defender = (
            jax.nn.one_hot(defender_index, defender_count, dtype=jnp.int32)
            .astype(bool)
            & can_assign[:, None]
        )
        chosen_threat = (
            jax.nn.one_hot(threat_index, opponent_count, dtype=jnp.int32)
            .astype(bool)
            & has_threat[:, None]
        )
        assignments = jnp.where(
            chosen_defender, threat_index[:, None], assignments
        )
        return (
            assignments,
            defenders_used | chosen_defender,
            threats_used | chosen_threat,
        )

    assigned_threat, _, _ = jax.lax.fori_loop(
        0,
        max_runners,
        assign_one,
        (assigned_threat, used_defender, used_threat),
    )
    selected = jnp.maximum(assigned_threat[:, 0], 0)
    has_assignment = assigned_threat[:, 0] >= 0
    selected_position = jnp.take_along_axis(
        opponent_future, selected[:, None, None], axis=1
    )[:, 0, :]
    own_goal = jnp.broadcast_to(own_goal, selected_position.shape)
    target = (
        selected_position
        + _unit(own_goal - selected_position) * goal_side_distance
    )
    return has_assignment, target, selected


def _press_engagement_line(line_height, aggression, policy):
    """Style-conditioned first-pressure line in the observer attack frame."""

    return (
        policy.press_engagement_x_base
        + (aggression - 0.5) * policy.press_engagement_aggression_gain
        + (line_height - 0.5) * policy.press_engagement_line_gain
    )


def _pass_cadence_terms(
    carrier_x, mate_adv, press_self, fwd_blocked, hx, policy
):
    """Return contextual back-pass cost, release cost and open-dribble gain.

    All inputs are observable current-frame quantities.  Keeping this pure avoids a
    hidden possession timer (which would break stateless batch independence) while
    making the cadence contract directly testable.
    """

    pressure_relief_level = jnp.clip(
        press_self / policy.pass_pressure_scale, 0.0, 1.0
    )
    backward_amount = jnp.clip(
        -mate_adv / policy.backpass_distance_scale, 0.0, 1.0
    )
    own_third_depth = jnp.clip(
        (-hx / 3.0 - carrier_x) / (2.0 * hx / 3.0), 0.0, 1.0
    )
    backpass_context_cost = (
        policy.unpressured_backpass_cost
        * backward_amount
        * (1.0 - pressure_relief_level[:, None])
        * (1.0 - policy.deep_backpass_relief * own_third_depth[:, None])
    )
    dynamic_release_cost = policy.pass_release_cost * (
        1.0 - policy.pass_pressure_relief * pressure_relief_level
    )
    open_dribble_bonus = (
        policy.open_dribble_gain
        * (1.0 - pressure_relief_level)
        * (~fwd_blocked).astype(jnp.float32)
    )
    return backpass_context_cost, dynamic_release_cost, open_dribble_bonus


def _pass_release_probability(
    press_self, target_x, through_strength, hx, policy, closing_self=None
):
    """Seeded per-frame release hazard for a currently preferred ordinary pass.

    ``closing_self``는 수비가 캐리어로 좁혀 오는 속도다. 압박 수준만 보면 수비가
    도달한 뒤에야 위험이 올라 '붙은 다음에 찬다'가 된다. 접근률을 함께 보면 아직
    멀어도 빠르게 다가오는 상황에서 미리 놓는다.
    """

    pressure_level = jnp.clip(
        press_self / policy.pass_pressure_scale, 0.0, 1.0
    )
    closing_level = (
        jnp.zeros_like(pressure_level) if closing_self is None
        else jnp.clip(closing_self / policy.pass_release_closing_scale, 0.0, 1.0)
    )
    final_third_depth = jnp.clip(
        (target_x - hx / 3.0) / (2.0 * hx / 3.0), 0.0, 1.0
    )
    return jnp.clip(
        policy.pass_release_base_probability
        + policy.pass_release_pressure_gain * pressure_level
        + policy.pass_release_closing_gain * closing_level
        + policy.pass_release_final_third_gain * final_third_depth
        + policy.pass_release_through_gain * jnp.clip(
            through_strength / policy.pass_release_through_scale, 0.0, 1.0
        ),
        0.0,
        1.0,
    )


def _ground_pass_completion_floor(
    pass_is_forward, creative_lane, policy
):
    """Return the receiver-wise ground-pass safety floor.

    Ordinary forward balls pay the stricter forward floor.  A genuinely
    progressive receiver run is the sole exception and pays the through-ball
    floor irrespective of the coarse ±60° direction bucket.
    """

    ordinary = jnp.where(
        pass_is_forward,
        policy.pass_forward_completion_floor,
        policy.pass_completion_floor,
    )
    return jnp.where(
        creative_lane,
        policy.through_completion_floor,
        ordinary,
    )


def _pass_direction_masks(pass_forward_cos):
    """Return the policy's single forward/backward/lateral partition.

    The DFL reporting bucket treats angles in broad bands, but tactical safety
    must use signed pitch progression: any negative attack-frame x component is
    a backward pass. Exactly lateral and shallow-forward switches remain in the
    lateral bucket until the ordinary +60 degree forward boundary.
    """

    forward = pass_forward_cos > 0.5
    backward = pass_forward_cos < 0.0
    sideways = (~forward) & (~backward)
    return forward, backward, sideways


def _safe_live_pass_direction(
    pass_forward_cos, target_distance, transition_live, policy
):
    """Keep live backward outlets short without narrowing restart fallback."""

    _, backward, _ = _pass_direction_masks(pass_forward_cos)
    return (
        (~transition_live)[:, None]
        | (~backward)
        | (target_distance <= policy.challenge_outlet_max_distance)
    )


def _prefer_loft_pass(
    ground_target_distance,
    ground_comp,
    ground_completion_floor,
    pass_forward_cos,
    directness,
    policy,
):
    """Choose height without turning every long progression into an aerial ball.

    Distance only opens the loft option.  A possession-oriented side keeps a
    sufficiently safe route on the grass; an unsafe long route may go over the
    blocking line, while a deliberately direct style may loft it regardless.
    A backward receiver is never upgraded into a loft: ordinary recycling is a
    short, low outlet, while the separate emergency-clearance branch owns hard
    aerial relief in the defensive danger zone.
    """

    loft_threshold = (
        policy.loft_distance_base
        - policy.loft_distance_direct_relief * directness[:, None]
    )
    long_enough = ground_target_distance > loft_threshold
    ground_blocked = ground_comp < ground_completion_floor
    deliberately_direct = directness[:, None] >= policy.loft_force_directness
    # Direction buckets are useful for profile statistics, but a shallow
    # negative x component is still a backward ball on the pitch. Preserve an
    # exactly lateral switch while keeping every actual backward loft closed.
    _, backward, _ = _pass_direction_masks(pass_forward_cos)
    not_backward = ~backward
    return long_enough & not_backward & (ground_blocked | deliberately_direct)


def _pass_direction_balance_value(
    pass_forward_cos,
    through_strength,
    target_distance,
    directness,
    carrier_x,
    press_self,
    own_fast_pass,
    hx,
    policy,
):
    """Contextual direction prior for realistic progression and recycling.

    A short wall pass must not pay the same forward tax as a long direct ball,
    nor lose automatically to the sideways bonus in a possession-oriented side.
    Conversely, the full backward bonus belongs to pressure relief and deep
    build-up, not an unpressured midfield return immediately after reception.
    """

    pass_is_forward, pass_is_backward, pass_is_sideways = (
        _pass_direction_masks(pass_forward_cos)
    )
    ordinary_forward = 1.0 - jnp.clip(
        through_strength / policy.pass_release_through_scale, 0.0, 1.0
    )
    forward_distance_scale = jnp.clip(
        (target_distance - policy.pass_min_distance)
        / (
            policy.continuation_max_distance
            - policy.pass_min_distance
            + DIV_EPS
        ),
        0.0,
        1.0,
    )
    possession_short_need = 1.0 - jnp.clip(
        (
            directness - policy.support_progression_directness_start
        )
        / (
            policy.support_progression_directness_full
            - policy.support_progression_directness_start
        ),
        0.0,
        1.0,
    )
    short_forward_value = (
        policy.pass_short_forward_bonus
        * possession_short_need[:, None]
        * (1.0 - forward_distance_scale)
        - policy.pass_forward_direction_cost * forward_distance_scale
    )

    pressure_need = jnp.clip(
        press_self / policy.pass_pressure_scale, 0.0, 1.0
    )
    # 단순히 자기 진영이라는 이유만으로 후방 보너스를 절반씩 복원하면 센터백 라인에서
    # 전진 출구가 계속 짧은 후방 수신자에게 진다. 자기 1/3보다 깊은 곳에서만 빌드업
    # 안전판을 열고, 그 밖에서는 실제 압박이 있어야 전체 보너스를 쓴다.
    buildout_depth = jnp.clip(
        (-hx / 3.0 - carrier_x) / (2.0 * hx / 3.0), 0.0, 1.0
    )
    backward_need = jnp.maximum(pressure_need, buildout_depth)
    recycle_floor = jnp.where(
        own_fast_pass,
        0.0,
        policy.pass_backward_recycle_floor,
    )
    backward_scale = backward_need + (1.0 - backward_need) * recycle_floor

    return (
        short_forward_value
        * ordinary_forward
        * pass_is_forward.astype(jnp.float32)
        + policy.pass_sideways_direction_bonus
        * pass_is_sideways.astype(jnp.float32)
        + policy.pass_backward_direction_bonus
        * backward_scale[:, None]
        * pass_is_backward.astype(jnp.float32)
    )


def _combination_support_slot(srank, support_scale, progression_blend=1.0):
    """Return one carrier-relative support lane for each teammate rank.

    The first two distances retain the measured K-League 1st/2nd-neighbour
    medians (10.2/14.2 m), but their forward cosine now exceeds 0.5.  The old
    (2,11)/(3,-14) geometry was classified sideways by construction, so even a
    perfectly executed triangle could not advance through short ground passes.
    The third player remains the 17.7 m safety outlet behind the ball.
    """

    old_slot = jnp.where(
        (srank < 0.5)[:, None],
        jnp.asarray([[2.0, 11.0]]),
        jnp.where(
            (srank < 1.5)[:, None],
            jnp.asarray([[3.0, -14.0]]),
            jnp.where(
                (srank < 2.5)[:, None],
                jnp.asarray([[-17.0, 0.0]]),
                jnp.where(
                    (srank < 3.5)[:, None],
                    jnp.asarray([[10.0, 6.0]]),
                    jnp.asarray([[10.0, -6.0]]),
                ),
            ),
        ),
    )
    progressive_slot = jnp.where(
        (srank < 0.5)[:, None],
        jnp.asarray([[5.5, 8.6]]),
        jnp.where(
            (srank < 1.5)[:, None],
            jnp.asarray([[7.5, -12.0]]),
            jnp.where(
                (srank < 2.5)[:, None],
                jnp.asarray([[-17.7, 0.0]]),
                jnp.where(
                    (srank < 3.5)[:, None],
                    jnp.asarray([[10.0, 6.0]]),
                    jnp.asarray([[10.0, -6.0]]),
                ),
            ),
        ),
    )
    blend = jnp.broadcast_to(
        jnp.asarray(progression_blend, dtype=progressive_slot.dtype),
        srank.shape,
    )[:, None]
    slot = old_slot + blend * (progressive_slot - old_slot)
    return slot * support_scale


def _style_support_progression_blend(directness, policy):
    """Map team directness continuously onto lateral→forward support geometry."""

    style_fraction = jnp.clip(
        (
            directness - policy.support_progression_directness_start
        )
        / (
            policy.support_progression_directness_full
            - policy.support_progression_directness_start
        ),
        0.0,
        1.0,
    )
    return policy.support_progression_blend * style_fraction


def _tackle_attempt_probability(
    danger,
    instability,
    approach_quality,
    contact_proximity,
    counterpress,
    aggression,
    control_dt,
    policy,
):
    """통제된 캐리어 상대 태클의 control-frame 확률.

    입력은 모두 ``[0,1]``의 관측 기반 연속 점수다. 구성값은 초당 hazard라서
    ``1-exp(-lambda*dt)``로 변환한다. 같은 물리시간을 두 개의 절반 frame으로
    나눠도 누적 시도 확률이 같아 control FPS에 종속되지 않는다.
    """

    hazard = (
        policy.tackle_hazard_base_per_s
        + policy.tackle_hazard_danger_gain_per_s * jnp.clip(danger, 0.0, 1.0)
        + policy.tackle_hazard_instability_gain_per_s
        * jnp.clip(instability, 0.0, 1.0)
        + policy.tackle_hazard_approach_gain_per_s
        * jnp.clip(approach_quality, 0.0, 1.0)
        + policy.tackle_hazard_contact_gain_per_s
        * jnp.clip(contact_proximity, 0.0, 1.0)
        + policy.tackle_hazard_counterpress_gain_per_s
        * jnp.clip(counterpress, 0.0, 1.0)
        + policy.tackle_hazard_aggression_gain_per_s
        * jnp.clip(aggression, 0.0, 1.0)
    )
    hazard = jnp.clip(hazard, 0.0, policy.tackle_hazard_cap_per_s)
    return -jnp.expm1(-hazard * control_dt), hazard


def _interception_attempt_probability(
    ball_speed_level,
    approach_quality,
    control_dt,
    policy,
):
    """소유 표기만 남은 이동 공에 대한 인터셉트 시도 확률."""

    hazard = (
        policy.interception_hazard_base_per_s
        + policy.interception_hazard_speed_gain_per_s
        * jnp.clip(ball_speed_level, 0.0, 1.0)
        + policy.interception_hazard_approach_gain_per_s
        * jnp.clip(approach_quality, 0.0, 1.0)
    )
    hazard = jnp.clip(
        hazard, 0.0, policy.interception_hazard_cap_per_s
    )
    return -jnp.expm1(-hazard * control_dt)


def _reception_control_plan(
    player_velocity,
    preferred_direction,
    speed_floor,
    lead_speed,
    speed_cap,
):
    """Return a momentum-compatible soft first-touch direction and speed.

    ``player_velocity`` and ``preferred_direction`` share the attack frame.
    A fixed world-speed trap is valid for a stationary receiver but makes a
    running receiver overtake the ball.  Preserve the receiver's current
    velocity vector and add only a small tactical lead; the caller supplies a
    cap that remains inside both the carried-control and DRIBBLE contracts.
    """

    preferred_direction = _unit(preferred_direction)
    raw_velocity = player_velocity + lead_speed * preferred_direction
    raw_speed = _safe_norm(raw_velocity, axis=1)
    direction = jnp.where(
        (raw_speed > DIV_EPS)[:, None],
        raw_velocity / (raw_speed[:, None] + DIV_EPS),
        preferred_direction,
    )
    speed = jnp.clip(raw_speed, speed_floor, speed_cap)
    return direction, speed


def _emergency_clearance_context(
    ball_x,
    pressure,
    has_safe_outlet,
    hx,
    policy,
):
    """Return the shared, deliberately narrow emergency-clearance context."""

    return (
        (~has_safe_outlet)
        & (ball_x <= -hx * policy.challenge_clearance_depth_fraction)
        & (pressure >= policy.challenge_clearance_pressure_min)
    )


def _challenge_contact_modes(
    challenge_intent,
    airborne,
    has_safe_outlet,
    preserve_control,
    ball_x,
    pressure,
    hx,
    policy,
):
    """Split a ground challenge into control, outlet, or emergency clearance.

    A challenge is a request to contest the ball, not an instruction to launch it.
    Safe low outlets take precedence.  A long clearance is legal only deep in the
    defender's own end, under pressure, and when no measured outlet exists.
    Airborne contacts remain owned by the aerial receive/clearance contract.
    """

    ground_challenge = challenge_intent & (~airborne)
    outlet = ground_challenge & has_safe_outlet
    clearance = (
        ground_challenge
        & (~preserve_control)
        & _emergency_clearance_context(
            ball_x, pressure, has_safe_outlet, hx, policy
        )
    )
    control = ground_challenge & (~outlet) & (~clearance)
    return control, outlet, clearance


def _trajectory_receiver_team_mask(
    player_team_base,
    own_live_pass,
    ahead_of_ball,
):
    """Return eligible team runners for a moving own-team pass.

    While at least one eligible outfielder remains ahead of the moving ball,
    only those ahead runners may receive it; this prevents the original kicker
    from chasing and repeatedly touching their own service. If the ball has
    passed every eligible runner, the restriction falls back to the whole
    pre-filtered team. Otherwise a backward clearance from a corner can leave
    every team-mate watching while the opponent alone pursues it.

    ``player_team_base`` already excludes goalkeepers and restart retouch
    latches, so falling back cannot reactivate the original kicker.
    """

    ahead_team_ok = player_team_base & ahead_of_ball
    has_ahead_receiver = jnp.any(ahead_team_ok, axis=1)
    return player_team_base & (
        (~own_live_pass[:, None])
        | ahead_of_ball
        | (~has_ahead_receiver[:, None])
    )


def _stable_nearest_player(
    self_distance,
    other_distance,
    self_eligible,
    other_eligible,
    other_slot,
):
    """Select one nearest eligible slot per independently observed group.

    Distance comparisons retain the policy's ``GEOMETRY_EPS`` tolerance, then
    exact/epsilon ties resolve by stable global slot order.  Every observation
    row must describe the same group with itself removed from ``other_slot``;
    this makes all rows agree on one carrier without a cross-team reduction.
    The existing nearest-other distance is returned with the selection so the
    caller does not repeat that reduction for its later rank calculations.
    """

    player_count = self_distance.shape[0]
    self_slot = jnp.arange(player_count, dtype=jnp.int32)
    eligible_other_distance = jnp.where(
        other_eligible, other_distance, jnp.inf
    )
    nearest_other_distance = jnp.min(eligible_other_distance, axis=1)
    nearest_distance = jnp.minimum(
        jnp.where(self_eligible, self_distance, jnp.inf),
        nearest_other_distance,
    )
    self_tied = self_eligible & (
        self_distance <= nearest_distance + GEOMETRY_EPS
    )
    other_tied_slot = jnp.min(
        jnp.where(
            other_eligible
            & (other_distance <= nearest_distance[:, None] + GEOMETRY_EPS),
            other_slot,
            jnp.int32(player_count),
        ),
        axis=1,
    )
    winner_slot = jnp.minimum(
        jnp.where(self_tied, self_slot, jnp.int32(player_count)),
        other_tied_slot,
    )
    return self_tied & (self_slot == winner_slot), nearest_other_distance


def _stable_lower_rank(
    self_value,
    other_value,
    self_eligible,
    other_eligible,
    other_slot,
):
    """Rank one player by value, resolving tolerance ties by global slot."""

    player_count = self_value.shape[0]
    self_slot = jnp.arange(player_count, dtype=jnp.int32)
    strictly_lower = other_value < self_value[:, None] - GEOMETRY_EPS
    tied_before = (
        jnp.abs(other_value - self_value[:, None]) <= GEOMETRY_EPS
    ) & (other_slot < self_slot[:, None])
    rank = jnp.sum(
        (other_eligible & (strictly_lower | tied_before)).astype(jnp.int32),
        axis=1,
    )
    return jnp.where(self_eligible, rank, jnp.int32(player_count))


def _stable_argmin_index(value, eligible, slot):
    """Return each row's minimum local index with a global-slot tie-break."""

    width = value.shape[1]
    best_value = jnp.min(jnp.where(eligible, value, jnp.inf), axis=1)
    tied = eligible & (value <= best_value[:, None] + GEOMETRY_EPS)
    slot_sentinel = jnp.iinfo(jnp.int32).max
    winner_slot = jnp.min(
        jnp.where(tied, slot, slot_sentinel), axis=1
    )
    winner = tied & (slot == winner_slot[:, None])
    return jnp.argmin(
        jnp.where(
            winner,
            jnp.arange(width, dtype=jnp.int32)[None, :],
            jnp.int32(width),
        ),
        axis=1,
    )


def _stable_lower_ranks(value, eligible, slot):
    """Rank every row entry by value and stable global slot."""

    width = value.shape[1]
    strictly_lower = value[:, :, None] < (
        value[:, None, :] - GEOMETRY_EPS
    )
    tied_before = (
        jnp.abs(value[:, :, None] - value[:, None, :]) <= GEOMETRY_EPS
    ) & (slot[:, :, None] < slot[:, None, :])
    rank = jnp.sum(
        eligible[:, :, None] & (strictly_lower | tied_before),
        axis=1,
        dtype=jnp.int32,
    )
    return jnp.where(eligible, rank, jnp.int32(width))


def _stable_first_team_candidate(candidate, team_id):
    """Select the lowest candidate slot independently for each team."""

    player_count = candidate.shape[0]
    slot = jnp.arange(player_count, dtype=jnp.int32)
    side = jnp.arange(TEAM_COUNT, dtype=jnp.int32)
    winner_by_team = jnp.min(
        jnp.where(
            candidate[:, None] & (team_id[:, None] == side[None, :]),
            slot[:, None],
            jnp.int32(player_count),
        ),
        axis=0,
    )
    winner = winner_by_team[team_id]
    return candidate & (slot == winner), winner < player_count


def _assign_box_runner_roles(
    box_approach_attack,
    final_third_attack,
    deep_final_attack,
    detailed_roles,
    am_carrier,
    carrier_role,
    carrier_y,
    home_y,
    team_id,
):
    """Assign one player to each compatible box lane."""

    central_forward = detailed_roles == player_roles.ROLE_CENTRE_FORWARD
    wide_forward = detailed_roles == player_roles.ROLE_WIDE_FORWARD
    central_midfielder = detailed_roles == player_roles.ROLE_CENTRE_MID
    central_candidate = box_approach_attack & central_forward & (~am_carrier)
    central, has_central = _stable_first_team_candidate(
        central_candidate, team_id
    )
    backup_candidate = (
        box_approach_attack
        & central_midfielder
        & (carrier_role == ROLE_FORWARD)
        & (jnp.abs(carrier_y) < 8.0)
        & (~am_carrier)
        & (~has_central)
    )
    backup, _ = _stable_first_team_candidate(backup_candidate, team_id)
    carrier_wide = jnp.abs(carrier_y) > 11.0
    far_side = jnp.sign(home_y + DIV_EPS) == -jnp.sign(
        carrier_y + DIV_EPS
    )
    far_candidate = (
        final_third_attack
        & (carrier_wide | (jnp.abs(carrier_y) < 8.0))
        & wide_forward
        & far_side
        & (~am_carrier)
    )
    far, _ = _stable_first_team_candidate(far_candidate, team_id)
    cutback_candidate = (
        deep_final_attack
        & central_midfielder
        & (~am_carrier)
        & (~backup)
    )
    cutback, _ = _stable_first_team_candidate(cutback_candidate, team_id)
    return central, backup, far, cutback


def _aerial_runner_roles(
    aerial_live,
    trajectory_runner,
    own_live_pass,
    opponent_live_pass,
    own_best_cost,
    opponent_best_cost,
    duel_eta_window_s,
):
    """Split an aerial trajectory winner into direct-duel and cover roles.

    The service team always sends its winner to the contact point. The defending
    winner does so only when its best trajectory cost is within the configured
    ETA window; otherwise it prepares the goal-side second ball. Aerial balls
    without deliberate-play provenance retain the symmetric two-team contest.
    """

    defender_can_contest = (
        jnp.isfinite(own_best_cost)
        & jnp.isfinite(opponent_best_cost)
        & (own_best_cost <= opponent_best_cost + duel_eta_window_s)
    )
    without_played_provenance = (~own_live_pass) & (~opponent_live_pass)
    direct = (
        aerial_live
        & trajectory_runner
        & (
            own_live_pass
            | without_played_provenance
            | (opponent_live_pass & defender_can_contest)
        )
    )
    cover = (
        aerial_live
        & trajectory_runner
        & opponent_live_pass
        & (~defender_can_contest)
    )
    return direct, cover


def _anchor_positions(c, ball_field, phase_weight):
    """데이터 유래 위치장에서 이 프레임의 오프더볼 기준점을 읽는다.

    ``positioning.ANCHOR_*`` 는 (국면, ball_x 격자, [x0, y0, dx_dby, dy_dby]) 표다.
    ball_x는 격자 사이를 선형보간하고 격자 밖은 상수 유지(끝 구간은 실측 표본이 적어
    외삽하면 라인이 골라인을 넘는다).  ball_y는 적합된 기울기로 선형 반영한다.

    ``phase_weight``는 (N, 3) 가중치다 — [공격, 수비, 중립], 합 1.  중립을 따로 둔 이유는
    실측이 그것을 **별도 대형**이라고 말하기 때문이다: 소유 미상은 인플레이의 16%이고, 팀 폭이
    25.7 m로 공격(35.2)·수비(28.9) **어느 쪽보다도 좁다**.  두 국면의 중간으로 근사하면 폭이
    6 m 넓게 나오고, 그냥 '수비'로 접으면 양 팀이 동시에 물러나 화면이 통째로 소극적이 된다.

    반환은 관측자 자기 공격 프레임 좌표 (N, 2)다.
    """
    knots = c.anchor_knots
    n_knots = knots.shape[0]
    w = jnp.asarray(phase_weight, jnp.float32)                     # (N, P)
    tab = jnp.einsum("np,npkc->nkc", w, c.anchor_tab)              # (N, K, 4)
    # 표는 조건부 **평균**이라 순간 산포보다 항상 좁다(젠슨 효과). 실측/표 비율로 팀 중심
    # 기준 깊이와 좌우 폭을 되돌린다. 국면 가중치가 섞이면 배율도 같이 섞인다.
    spread_x = jnp.einsum("np,p->n", w, c.anchor_spread_x)[:, None]
    spread_y = jnp.einsum("np,p->n", w, c.anchor_spread_y)[:, None]
    mean_x = jnp.einsum("np,pk->nk", w, c.anchor_mean_x)            # (N, K)
    tab = tab.at[:, :, 0].set(mean_x + (tab[:, :, 0] - mean_x) * spread_x)
    tab = tab.at[:, :, 1].set(tab[:, :, 1] * spread_y)
    tab = tab.at[:, :, 3].set(tab[:, :, 3] * spread_y)
    bx = jnp.clip(ball_field[:, 0], knots[0], knots[-1])
    idx = jnp.clip(jnp.searchsorted(knots, bx) - 1, 0, n_knots - 2)
    x0 = knots[idx]
    x1 = knots[idx + 1]
    frac = jnp.clip((bx - x0) / jnp.maximum(x1 - x0, DIV_EPS), 0.0, 1.0)
    lo = jnp.take_along_axis(tab, idx[:, None, None], axis=1)[:, 0, :]
    hi = jnp.take_along_axis(tab, (idx + 1)[:, None, None], axis=1)[:, 0, :]
    row = lo + (hi - lo) * frac[:, None]
    by = ball_field[:, 1]
    return jnp.stack([row[:, 0] + row[:, 2] * by, row[:, 1] + row[:, 3] * by], axis=1)


def _receive_prediction_times(policy):
    """End at the horizon; shrink the first sample only for sub-0.20s windows."""

    start = (
        policy.receive_prediction_horizon_s / policy.receive_prediction_samples
        if policy.receive_prediction_horizon_s < 0.20
        else 0.20
    )
    return jnp.linspace(
        start,
        policy.receive_prediction_horizon_s,
        policy.receive_prediction_samples,
    )


def _with_observed_formation(c, self_tok, half, self_active):
    """관측에서 읽은 현재 포메이션으로 문맥을 다시 묶는다.

    바뀌는 것은 다섯이다 — 규범 앵커, coarse·상세 역할, 실측 위치장 표, 그리고
    그 표의 슬롯 배정. 승인된 목표 layout은 명령 경계에서 즉시 관측되며, 실제 선수
    이동은 이후 행동과 물리가 만든다.
    """

    home_att = self_tok[:, c.p_formation_home[0]:c.p_formation_home[1]] * half
    scale = jnp.float32(c.layout_scale)
    target = jnp.round(self_tok[:, c.p_layout_index[0]] * scale).astype(jnp.int32)
    target = jnp.clip(target, 0, c.layout_count - 1)

    slots = jnp.arange(home_att.shape[0], dtype=jnp.int32)
    tab_target = c.anchor_tab[target, slots]
    tabulated_roles = c.detailed_role_tab[target, slots]
    tabulated_home = c.role_home_tab[target, slots]
    table_matches_observation = (
        jnp.all(self_active)
        & jnp.all(
            jnp.abs(home_att - tabulated_home) <= _OBSERVED_FORMATION_TABLE_TOL_M
        )
    )
    detailed_roles = jax.lax.cond(
        table_matches_observation,
        lambda: tabulated_roles,
        lambda: player_roles.classify_roles(
            home_att, c.gk, c.team_id, self_active
        ),
    )
    roles = _coarse_roles_from_detailed(detailed_roles)
    return _replace_ctx(
        c,
        home_att=home_att,
        roles=roles,
        detailed_roles=detailed_roles,
        anchor_tab=tab_target,
        anchor_slot=c.anchor_slot[target, slots],
    )


def _replace_ctx(c, **overrides):
    """문맥의 얕은 사본에 프레임별 값을 덮어쓴다.

    ``types.SimpleNamespace``라 사본이 싸고, 원본은 생성 시점 상수 그대로 남는다 —
    같은 정책 객체를 여러 프레임에 재사용해도 서로 오염되지 않는다.
    """

    fields = dict(vars(c))
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _rule_based_actions(obs, key, c, aff, *, with_trace=False):
    """관측과 affordance view로 국면(세트피스/공격/수비/루즈볼)을 분해해 각 행동을 정한다.
    모든 좌표는 자기 공격 프레임(m): 상대 골 [+hx,0], 우리 골 [-hx,0], +x=전방."""
    N = c.N
    policy = c.policy
    # Receiver choice, ordinary release cadence, first-time relay, execution
    # noise.  Separate streams keep a relay calibration change from silently
    # changing pass targets or shot error for the same match key.
    k_pass, k_release, k_relay, k_noise = jax.random.split(key, 4)
    # 기존 패스·수신자·슛 난수 stream은 보존하고, challenge 의도만 독립 fold-in한다.
    # split 개수를 늘리면 같은 seed의 나머지 전술까지 전부 바뀌어 paired 진단이 불가능하다.
    k_challenge = jax.random.fold_in(key, jnp.uint32(0x5441434B))

    # ── obs 복원(공격 프레임, 미터) ─────────────────────────────────────────
    # Causal-Compact 규약: 앵커와 토큰 상대량이 같은 정규화 상수를 쓰므로 절대 좌표는
    # 정규화 공간에서 그냥 더한 뒤 한 번만 미터로 되돌리면 된다.
    def col(i0, i1):
        return obs[:, i0:i1]

    half = jnp.array([c.hx, c.hy])
    my_field = col(*c.i_self_pos) * half                               # 내 위치
    my_vel = col(*c.i_self_vel) * c.n_pvel                             # 내 절대속도(공격 프레임, m/s)
    my_speed = jnp.linalg.norm(my_vel, axis=1)

    # 전 슬롯 토큰 (N, N, players_size) — 관측자 자신도 포함(rel=0)이라 self 값은 대각에서 읽는다.
    tokens = obs[:, c.players_start:c.players_start + N * c.players_size]
    tokens = tokens.reshape(N, N, c.players_size)
    diag = jnp.arange(N)
    self_tok = tokens[diag, diag]                                      # (N, players_size)

    # ── 포메이션을 관측에서 읽는다 ─────────────────────────────────────────
    # 종전에는 ``home_att``가 생성 시점 상수라 킥오프 이후 모양이 영원히 고정이었다.
    # 지휘관이 상태를 바꿔도 정책이 볼 방법이 없었다는 뜻이다. 자기 토큰의 규범 앵커를
    # 읽으면 관측 경계를 지키면서 매 프레임 현재 모양을 따른다.
    #
    # 자기 토큰이라 프레임이 정확히 맞는다: 토큰은 관측자 부호로 접히고 앵커는 대상 부호로
    # 저장되므로, i == j에서 두 부호가 상쇄돼 그 선수 자신의 공격 프레임 좌표가 남는다.
    _, sp_bit, throw_bit = _taker_bits(self_tok[:, c.p_taker[0]])
    retouch = sp_bit | throw_bit
    is_gk = self_tok[:, c.p_gk[0]] > 0.5
    booked = self_tok[:, c.p_yellow[0]] > 0.5   # 경고 1장 — 2차 경고=퇴장이라 태클 자제(조키 전환)
    self_active = (
        jnp.round(self_tok[:, c.p_status[0]]).astype(jnp.int32) == SLOT_ACTIVE
    )
    c = _with_observed_formation(c, self_tok, half, self_active)

    # 파생 affordance는 env의 단일 구현에서 받는다(정책이 규칙 사본을 갖지 않는다).
    in_reach = aff["in_reach"] > 0.5
    f2b_avail = aff["f2b_avail"] > 0.5
    kick_potential = aff["kick_gated"] < 0.5

    ball_rel_n = col(*c.i_ball_pos)
    ball_field = (col(*c.i_self_pos) + ball_rel_n) * half
    ball_z = obs[:, c.i_ball_z[0]] * c.n_bz
    # 공 절대속도 = 관측자 속도 + 상대속도. 두 항의 정규화 상수가 달라 각각 되돌린다.
    ball_vel_xy = my_vel + col(*c.i_ball_vel) * c.n_bvel
    ball_vel = jnp.concatenate([ball_vel_xy, obs[:, c.i_ball_vel_z[0]:c.i_ball_vel_z[1]] * c.n_bvel],
                               axis=1)
    ball_speed = jnp.linalg.norm(ball_vel[:, :2], axis=1)
    ball_alive = obs[:, c.i_ball_alive[0]] > 0.5
    ball_xy_speed = ball_speed
    poss = obs[:, c.i_poss[0]]                                        # +1 우리 / -1 상대 / 0 없음
    possession_age_s = (
        jnp.clip(obs[:, c.i_possession_steps], 0.0, 1.0)
        * c.possession_context_s
    )
    previous_possession = obs[:, c.i_previous_possession]
    # 국면 확정은 공까지의 거리를 계산한 뒤로 미룬다(아래 `unattended` 참조). 소유 표기만으로
    # 국면을 정하면 방치된 공 앞에서 양 팀이 모두 대형만 유지한다.

    # 자기 토큰을 뺀 (N, N-1, size) 뷰 — 아래 전술 계산은 전부 '타 선수' 집합 기준이다.
    others = tokens[jnp.arange(N)[:, None], c.others_idx]
    o_rel = others[:, :, c.p_relpos[0]:c.p_relpos[1]] * half
    others_field = my_field[:, None, :] + o_rel                      # (N,N-1,2) 상대·동료 위치
    o_gk = others[:, :, c.p_gk[0]] > 0.5
    o_team = others[:, :, c.p_team[0]]
    mate = o_team > 0.5
    opp = o_team < -0.5
    # 비활성 슬롯은 토큰이 0이지만 status_code만은 살아 있다. 팀 부호가 아니라 이 코드로
    # 판정해야 '교체로 빠진 슬롯'과 '퇴장한 슬롯'을 구분할 수 있다.
    active_o = jnp.round(others[:, :, c.p_status[0]]).astype(jnp.int32) == SLOT_ACTIVE
    # [지표] 타 선수 절대속도(공격프레임, m/s) — 압박·패스차단·마킹 선점의 핵심 신호.
    # 토큰은 관측자 기준 상대속도라 자기 속도를 더해 절대로 되돌린다.
    o_vel = my_vel[:, None, :] + others[:, :, c.p_relvel[0]:c.p_relvel[1]] * c.n_pvel
    o_vel = jnp.where(active_o[:, :, None], o_vel, 0.0)              # (N,O,2)
    o_offside = others[:, :, c.p_offside[0]] > 0.5                   # 오프사이드 위치 동료(패스 무효 게이트)
    # 관측에 이미 공개된 이질 능력치를 정책 선택에도 쓴다. 원값의 0/1 끝단을 그대로
    # 배율로 쓰면 저능력 선수가 아마추어처럼 붕괴하므로 하·상위 프로 band에서 포화한다.
    self_control = jnp.clip(self_tok[:, c.p_ctrl[0]], 0.0, 1.0)
    o_control = jnp.clip(others[:, :, c.p_ctrl[0]], 0.0, 1.0)
    self_ability = _professional_ability(
        self_control,
        policy.professional_control_floor,
        policy.professional_control_ceiling,
    )
    o_ability = _professional_ability(
        o_control,
        policy.professional_control_floor,
        policy.professional_control_ceiling,
    )
    self_reach_z = self_tok[:, c.p_reach_z[0]] * c.n_body_z
    self_head_z = self_tok[:, c.p_head_z[0]] * c.n_body_z
    o_reach_z = others[:, :, c.p_reach_z[0]] * c.n_body_z
    o_reach_ability = _professional_ability(
        o_reach_z,
        policy.professional_reach_floor_m,
        policy.professional_reach_ceiling_m,
    )

    is_sp_ours = aff["is_sp_ours"]                                   # +1 우리 재개 / -1 상대 / 0
    self_is_taker = aff["is_taker"] > 0.5
    self_kicker_ready = aff["kicker_ready"] > 0.5
    # categorical 재개 코드를 one-hot으로 되돌린다 — 아래 국면 분해가 종류별 비트를 쓰고,
    # 이 왕복이 정확히 무손실이라는 것이 코드 인코딩의 근거다.
    rk_code = jnp.round(obs[:, c.i_rk]).astype(jnp.int32)
    rk = jax.nn.one_hot(rk_code, RESTART_COUNT)
    restart_active = rk_code != RK_NONE
    last_touch_code = jnp.round(obs[:, c.i_last_touch_code]).astype(jnp.int32)
    i_touched_last = obs[:, c.i_last_touch] > 1.5
    last_touch_was_played = (
        (last_touch_code == TOUCH_PASS)
        | (last_touch_code == TOUCH_PASS_HEAD)
        | (last_touch_code == TOUCH_SHOOT)
        | (last_touch_code == TOUCH_SHOOT_HEAD)
    )
    last_touch_was_pass = (
        (last_touch_code == TOUCH_PASS)
        | (last_touch_code == TOUCH_PASS_HEAD)
    )
    # 마지막 터치 팀/코드는 최초 수신 전까지 유지되는 causal provenance다. 속도만 보면
    # 감속 순간 이 정보를 버리게 되고, 원 패서가 다시 루즈볼 ETA 후보로 들어가 자기
    # 패스를 먼저 재접촉한다. 거의 정지한 방치 공만 일반 루즈볼로 되돌린다.
    own_live_pass = (
        ball_alive
        & (ball_speed > policy.unattended_ball_speed)
        & (obs[:, c.i_last_touch] > 0.5)
        & last_touch_was_played
        & (~restart_active)
    )
    opponent_live_pass = (
        ball_alive
        & (ball_speed > policy.unattended_ball_speed)
        & (obs[:, c.i_last_touch] < -0.5)
        & last_touch_was_played
        & (~restart_active)
    )
    own_fast_pass = (
        own_live_pass
        & (ball_speed > policy.quick_relay_min_incoming_speed_mps)
    )

    sty = c.styles[c.team_id]
    line_h, tempo, width_s, aggr, direct = (
        sty[:, STYLE_LINE],
        sty[:, STYLE_TEMPO],
        sty[:, STYLE_WIDTH],
        sty[:, STYLE_AGGRESSION],
        sty[:, STYLE_DIRECTNESS],
    )

    # ── 캐리어/최근접 판정(자기 obs 내 거리 비교) ───────────────────────────
    ball_dist = jnp.linalg.norm(ball_field - my_field, axis=1)
    # 세트피스 taker(is_taker=재터치 금지/킥 예정)와 재터치 제한 중인 자신은 '공 회수자'에서 제외 —
    # 방금 찬 키커가 제 공 위에서 얼어붙지 않고, 다른 동료가 collector로 나서게 한다.
    o_pending, o_sp, o_throw = _taker_bits(others[:, :, c.p_taker[0]])
    o_is_taker = o_pending | o_sp | o_throw
    d_ball_o = jnp.linalg.norm(ball_field[:, None, :] - others_field, axis=2)
    mate_eligible = mate & (~o_is_taker)
    mate_ball = jnp.where(mate_eligible, d_ball_o, jnp.inf)
    am_nearest_mate, nearest_mate_ball = _stable_nearest_player(
        ball_dist,
        d_ball_o,
        self_active & (~retouch),
        mate_eligible,
        c.others_idx,
    )
    am_nearest_field, _ = _stable_nearest_player(
        ball_dist,
        d_ball_o,
        self_active & (~is_gk) & (~retouch),
        mate_eligible & active_o & (~o_gk),
        c.others_idx,
    )

    # ── 국면 확정: 방치된 공은 소유 표기를 무시하고 루즈볼로 다룬다 ──────────
    # 소유권은 접촉으로만 바뀌므로, 아무도 건드리지 않으면 마지막 터치 팀에 래치된 채
    # 남는다. 그때 소유 표기만 믿으면 "우리가 갖고 있으니 대형 유지", 상대는 "쟤들이
    # 갖고 있으니 대형 유지"가 되어 **아무도 공을 주우러 가지 않는다**. 실측에서 최근접
    # 선수가 9.8 m 떨어진 채 경기가 90초간 멈췄다. env도 같은 조건으로 소유권을 중립으로
    # 되돌리지만(``possession_release_*``), 정책이 스스로도 판단하게 두어 이중으로 막는다.
    nearest_any = jnp.minimum(
        ball_dist, jnp.min(jnp.where(active_o, d_ball_o, jnp.inf), axis=1))
    unattended = (
        (ball_xy_speed <= policy.unattended_ball_speed)
        & (nearest_any > policy.unattended_ball_radius)
        & ball_alive
        & (~restart_active)
    )
    attacking = (poss > 0.5) & (~unattended)
    defending = (poss < -0.5) & (~unattended)
    loose_phase = (~attacking) & (~defending)
    transition_live = ball_alive & (~restart_active)
    just_lost = (
        defending
        & transition_live
        & (previous_possession > 0.5)
        & (possession_age_s < policy.counterpress_window_s)
    )
    own_recent_loose = (
        loose_phase
        & transition_live
        & (previous_possession > 0.5)
        & (possession_age_s < policy.counterpress_window_s)
    )
    secure_phase = (
        attacking
        & transition_live
        & (possession_age_s < policy.secure_possession_window_s)
    )
    team_rank = _stable_lower_rank(
        ball_dist,
        d_ball_o,
        self_active & (~retouch),
        mate_eligible,
        c.others_idx,
    )
    carrier_j = jnp.argmin(mate_ball, axis=1)
    carrier_field = jnp.where(am_nearest_mate[:, None], my_field, jnp.take_along_axis(
        others_field, carrier_j[:, None, None], axis=1)[:, 0, :])
    carrier_other_slot = jnp.take_along_axis(
        c.others_idx, carrier_j[:, None], axis=1
    )[:, 0]
    carrier_slot = jnp.where(
        am_nearest_mate, jnp.arange(N, dtype=jnp.int32), carrier_other_slot
    )
    carrier_role = c.roles[carrier_slot]

    opp_goal = jnp.array([c.hx, 0.0])
    own_goal = jnp.array([-c.hx, 0.0])
    ball_adv = ball_field[:, 0]                                      # 공 전진도(+ = 상대 진영)

    # ── 국면 B: 공격(우리 점유) — 지표 기반 기대가치 결정 ─────────────────────
    # 캐리어는 슛/패스/드리블/전개를 **공통 통화(≈득점 기여 기대값)**로 비교해 argmax. 모든
    # 에이전트가 '내가 캐리어라면'을 계산하고 am_nearest_mate 게이팅으로 실제 캐리어만 실행(분산·일관).
    to_goal_vec = opp_goal[None, :] - my_field
    d_goal = jnp.linalg.norm(to_goal_vec, axis=1) + DIV_EPS
    # 전술 가치는 선수 위치에서 평가하지만 킥 물리는 실제 공 좌표에서 시작한다. carry reach 안에서도
    # 둘은 1m 이상 어긋날 수 있으므로 방향·파워 솔버의 원점을 선수 중심으로 두면 그 오차가 패스
    # 목표와 포스트에 그대로 남는다.
    ball_to_goal = opp_goal[None, :] - ball_field
    ball_d_goal = jnp.linalg.norm(ball_to_goal, axis=1) + DIV_EPS

    # 지표(tactics): 자기·동료 슛 xg, 압박(상대속도 선점), 위협통화 threat, 패스 성공확률(인터셉트 예측)
    xg_self = T.shot_xg(my_field, others_field, opp, o_gk, c.hx, c.goal_w)        # (N,)
    press_self = T.pressure(my_field, others_field, o_vel, opp)                          # (N,)
    # 압박의 **변화율** — 아직 멀어도 빠르게 좁혀 오면 릴리스를 앞당긴다.
    closing_self = T.closing(my_field, my_vel, others_field, o_vel, opp)                 # (N,)
    opp_fwd = jnp.where(
        opp, others_field[:, :, 0] - my_field[:, 0:1], -1e9
    )
    opp_side = jnp.where(
        opp, jnp.abs(others_field[:, :, 1] - my_field[:, 1:2]), 1e9
    )
    fwd_blocked = jnp.any(
        (opp_fwd > 0.5)
        & (opp_fwd < policy.dribble_cone_length)
        & (opp_side < 3.0),
        axis=1,
    )
    # ★슛 가치 통화 통일: '이 위치에서의 슛 가치' = xg·RulePolicy.shoot_gain.
    # 슛/패스/드리블이 모두 같은 단위로
    #  '지금 슛 vs 더 좋은 슛을 만든 뒤 슛'을 비교한다 → range에 들었다고 때리지 않고, 패스·드리블로
    #  xg가 더 오르면 그쪽을 택하다가 지금 슛이 최선일 때만 슛(argmax가 crossover를 자동 결정).
    sv_self = xg_self * policy.shoot_gain                                                       # (N,)
    # 슛 조준: 중앙에서는 GK 반대쪽, 하프스페이스에서는 먼 포스트의 안쪽을 향해 차고
    # 사이드스핀이 마지막 구간을 더 휘게 한다. 목표점과 스핀 부호를 같은 횡방향으로 묶어
    # "직선은 한쪽, 커브는 반대쪽"인 비물리 조합을 만들지 않는다.
    opp_gk_y = jnp.sum(jnp.where(opp & o_gk, others_field[:, :, 1], 0.0), axis=1)
    far_side = jnp.where(
        jnp.abs(my_field[:, 1]) > 2.0,
        -jnp.sign(my_field[:, 1]),
        -jnp.sign(opp_gk_y + DIV_EPS),
    )
    desired_goal_y = far_side * (c.goal_w * 0.5 * policy.curl_goal_fraction)
    curve_progress = jnp.clip(
        (ball_d_goal - policy.curl_distance_start)
        / (c.shoot_range - policy.curl_distance_start + DIV_EPS),
        0.0,
        1.0,
    )
    curve_spin = jnp.clip(
        policy.curl_spin_min
        + (policy.curl_spin_max - policy.curl_spin_min)
        * (0.55 * curve_progress + 0.45 * jnp.clip(jnp.abs(my_field[:, 1]) / c.hy, 0.0, 1.0)),
        policy.curl_spin_min,
        policy.curl_spin_max,
    )
    # 같은 공 물리에서 캘리브한 거리별 횡변위만큼 직선 조준을 안쪽으로 옮긴다. 여기서 단순히
    # ``aim_y -= curve``로 끝내면 대각선 슛은 틀린다. 로컬 lateral 변위 w는 골라인 x에서
    # world-y 변위 ``w / dir_x``가 되고, 커브 때문에 골라인까지의 종방향 비행거리도 늘어난다.
    # 첫 추정으로 그 두 기하 효과를 계산한 뒤 거리표를 한 번 더 조회한다(고정 반복의 1회 전개).
    goal_dx = jnp.maximum(c.hx - ball_field[:, 0], DIV_EPS)
    curve_displacement0 = jnp.abs(
        jnp.interp(ball_d_goal, c.shot_d, c.shot_curve_y_unit)
    ) * curve_spin
    aim_y0 = desired_goal_y - far_side * curve_displacement0
    straight_d0 = jnp.sqrt(goal_dx * goal_dx + (aim_y0 - ball_field[:, 1]) ** 2)
    dir_x0 = goal_dx / jnp.maximum(straight_d0, DIV_EPS)
    dir_y0 = (aim_y0 - ball_field[:, 1]) / jnp.maximum(straight_d0, DIV_EPS)
    signed_curve0 = far_side * curve_displacement0
    curve_path_d0 = straight_d0 + dir_y0 * signed_curve0 / jnp.maximum(dir_x0, 0.1)
    curve_displacement = jnp.abs(
        jnp.interp(curve_path_d0, c.shot_d, c.shot_curve_y_unit)
    ) * curve_spin
    aim_y = desired_goal_y - far_side * curve_displacement / jnp.maximum(dir_x0, 0.1)
    aim = jnp.stack([
        jnp.full(N, c.hx),
        aim_y,
    ], axis=1)
    dir_shot = _unit(aim - ball_field)

    # 패스 후보 평가: 현재 발밑이 아니라 공이 도착할 때의 수신 위치를 겨냥한다. 전방으로 달리는
    # 온사이드 수신자와 최종 수비선 사이 공간은 킬패스 목표로 쓰되, 현재 오프사이드 선수는 제외한다.
    nearest_opp_carrier = jnp.min(
        jnp.where(opp, d_ball_o, jnp.inf), axis=1
    )
    pass_contact_clear = (
        nearest_opp_carrier >= policy.pass_contact_clearance
    )
    mate_ball_d = jnp.linalg.norm(
        others_field - ball_field[:, None, :], axis=2
    )
    mate_adv = others_field[:, :, 0] - my_field[:, 0:1]
    backpass_context_cost, dynamic_release_cost, open_dribble_bonus = (
        _pass_cadence_terms(
            my_field[:, 0], mate_adv, press_self, fwd_blocked, c.hx, policy
        )
    )
    off_line_x = aff["off_line"] * c.hx
    # Law 11의 온사이드 경계는 두 번째 최종 수비수와 공 중 더 전방인 쪽이다.
    # 패스 후보(``onside_now``)는 이미 이 규칙을 썼지만 오프더볼 목표는 v38까지
    # 수비선만 cap으로 사용했다. 캐리어가 수비선 뒤로 돌파하면 합법적으로 공 뒤를
    # 따라갈 동료까지 남겨 두는 불일치였으므로 모든 공격 목표가 같은 경계를 쓴다.
    onside_line_x = jnp.maximum(off_line_x, ball_adv)
    onside_now = (
        (others_field[:, :, 0] <= off_line_x[:, None] + c.offside_margin)
        | (others_field[:, :, 0] <= ball_adv[:, None] + c.offside_margin)
        | (others_field[:, :, 0] <= 0.0)
    )
    pass_candidate = (
        mate
        & (mate_ball_d > policy.pass_min_distance)
        & (mate_ball_d < 50.0)
        & (~o_offside)
        & onside_now
    )
    pass_target_all, pass_lead_all, through_strength = T.moving_pass_target(
        ball_field,
        others_field,
        o_vel,
        off_line_x,
        c.hx,
        c.hy,
        lead_time_cap=policy.pass_lead_time_cap,
        velocity_weight=policy.pass_lead_velocity_weight,
        lead_distance_cap=policy.pass_lead_distance_cap,
        through_gap_weight=policy.through_gap_weight,
        through_shoulder_cue_weight=policy.through_shoulder_cue_weight,
        through_shoulder_gap_m=policy.through_shoulder_gap_m,
        through_run_ahead=policy.through_run_ahead,
        through_min_progress=policy.through_min_progress,
    )
    # 지상 패스의 0.9 s lead를 로프트에 그대로 쓰면 2~3 s 체공 중
    # 달리는 수신자 뒤로 떨어진다. 실물리 loft 거리↔시간 테이블로 추가
    # lead를 한 번 재수렴하되, 전체 선행거리는 기존 cap의 2배로 제한한다.
    drive_lead_t = jnp.clip(
        mate_ball_d / 18.0, 0.0, policy.pass_lead_time_cap
    ) * policy.pass_lead_velocity_weight
    loft_distance1 = jnp.linalg.norm(
        pass_target_all - ball_field[:, None, :], axis=2
    )
    loft_flight_t = jnp.interp(loft_distance1, c.loft_R, c.loft_T)
    loft_flight_extra = jnp.maximum(
        0.0,
        policy.pass_lead_velocity_weight * loft_flight_t - drive_lead_t,
    )
    loft_target_all = pass_target_all + o_vel * loft_flight_extra[:, :, None]
    loft_offset = loft_target_all - others_field
    loft_offset_norm = _safe_norm(loft_offset, axis=-1)
    loft_offset = loft_offset * jnp.minimum(
        1.0,
        (2.0 * policy.pass_lead_distance_cap)
        / (loft_offset_norm + DIV_EPS),
    )[:, :, None]
    loft_target_all = others_field + loft_offset
    loft_target_all = jnp.stack(
        [
            jnp.clip(loft_target_all[:, :, 0], -c.hx + 2.0, c.hx - 1.5),
            jnp.clip(loft_target_all[:, :, 1], -c.hy + 1.5, c.hy - 1.5),
        ],
        axis=-1,
    )
    loft_target_distance = jnp.linalg.norm(
        loft_target_all - ball_field[:, None, :], axis=2
    )
    loft_flight_t = jnp.interp(
        loft_target_distance, c.loft_R, c.loft_T
    )
    xg_target = T.shot_xg(pass_target_all, others_field, opp, o_gk, c.hx, c.goal_w)
    threat_target = jnp.maximum(
        xg_target * policy.shoot_gain,
        T.pitch_value(pass_target_all, c.hx, c.hy),
    )
    receiver_vmax = _decision_speed_for_endurance(
        effective_speed_cap(
            others[:, :, c.p_vmax[0]] * c.n_pvel,
            others[:, :, c.p_stamina_long[0]],
            others[:, :, c.p_stamina_short[0]],
            long_floor=c.long_stamina_vmax_floor,
            short_floor=c.short_stamina_vmax_floor,
            short_knee=c.short_stamina_headroom_knee,
        ),
        others[:, :, c.p_endurance[0]],
    )
    receive_space_m = T.openness(pass_target_all, others_field, o_vel, opp)
    receive_space_10 = jnp.clip(receive_space_m / 10.0, 0.0, 1.0)
    receive_space_now_m = T.openness(
        others_field, others_field, o_vel, opp
    )
    # DFL 성공 수신자의 직전 1초 이동 중앙값(2.42m)을 중심으로, 정지 수신자에는
    # 보너스를 주지 않되 전력질주자가 끝없이 유리해지지도 않게 한다. ``pass_lead``는
    # 실제 이동거리와 동일한 값은 아니지만 현재 속도×도착 선점시간이라 인과적으로 얻을
    # 수 있는 가장 가까운 proxy다. 2.4m 이후에는 보너스를 완만히 되돌리고, 새로 만든
    # openness는 별도 항으로 계속 온전히 평가한다.
    receiver_movement_score = (
        jnp.clip(pass_lead_all / 2.4, 0.0, 1.0)
        * (
            1.0
            - 0.5 * jnp.clip((pass_lead_all - 2.4) / 2.4, 0.0, 1.0)
        )
    )
    receiver_motion_value = (
        policy.pass_receiver_movement_gain * receiver_movement_score
        + policy.pass_space_creation_gain
        * jnp.clip(
            (receive_space_m - receive_space_now_m) / 5.0,
            0.0,
            1.0,
        )
    )
    ground_target_distance = loft_distance1
    ground_launch_speed = jnp.interp(
        ground_target_distance, c.drive_R, c.drive_v
    )
    ground_average_speed = 0.5 * (
        ground_launch_speed + policy.drive_arrive_speed_mps
    )
    lane_comp = T.lane_completion(
        ball_field,
        pass_target_all,
        others_field,
        o_vel,
        opp,
        v_ball=ground_average_speed[:, :, None],
        r_int=policy.pass_lane_intercept_radius,
        react=policy.pass_lane_reaction_s,
        opponent_vmax=receiver_vmax,
        defender_accel=policy.pass_lane_defender_acceleration_mps2,
    )
    # drive 역솔버는 목표 도착속도를 고정하므로 등가 평균속도로 비행시간을 복원할 수 있다.
    # 단순 수신점 openness는 가까운 수비수가 '누가 먼저 닿나'를 말하지 못했다. 실제 수신자와
    # 상대의 stamina-aware ETA 우위를 계산해 레인 차단확률과 곱하면, 2차 커버를 늘린 뒤에도
    # 정책이 접전 발밑 패스를 안전하다고 오인하지 않는다.
    ground_flight_t = (
        2.0 * ground_target_distance
        / (
            ground_launch_speed
            + policy.drive_arrive_speed_mps
            + DIV_EPS
        )
    )
    ground_receive_advantage = T.aerial_reception(
        pass_target_all,
        others_field,
        receiver_vmax,
        ground_flight_t,
        others_field,
        o_vel,
        receiver_vmax,
        opp,
        arrival_slack=0.18,
        arrival_soft=0.35,
        contest_soft=0.48,
    )
    ground_comp = lane_comp * (0.25 + 0.75 * ground_receive_advantage)
    # 공중 접촉에서의 동료 연결은 별도 행동 빈도를 만들지 않고, 이미 지불한 패스
    # 레인·수신 ETA 계산을 재사용한다. 짧은 헤더는 일반 패스의 6m 하한보다 가까울 수
    # 있으므로 팀/온사이드/활성 계약만 다시 쓰되, 느린 공중 연결의 낙관을 막기 위해
    # challenge 출구와 같은 높은 완성률 하한을 요구한다.
    aerial_pass_floor = (
        policy.aerial_pass_completion_floor
        + policy.aerial_control_floor_relief * (0.5 - self_ability)
    )
    aerial_pass_candidate = (
        mate
        & active_o
        & (~o_offside)
        & onside_now
        & (ground_target_distance >= policy.aerial_pass_min_distance)
        & (ground_target_distance <= policy.aerial_pass_max_distance)
        & (ground_comp >= aerial_pass_floor[:, None])
    )
    aerial_pass_value = (
        ground_comp
        + 0.08 * receive_space_10
        + 0.06 * T.pitch_value(pass_target_all, c.hx, c.hy)
        + policy.receiver_control_value_gain * (o_ability - 0.5)
        - 0.08
        * jnp.clip(
            ground_target_distance / policy.aerial_pass_max_distance,
            0.0,
            1.0,
        )
    )
    aerial_pass_j = jnp.argmax(
        jnp.where(aerial_pass_candidate, aerial_pass_value, -jnp.inf), axis=1
    )
    has_aerial_pass = jnp.any(aerial_pass_candidate, axis=1)
    aerial_pass_target = jnp.take_along_axis(
        pass_target_all, aerial_pass_j[:, None, None], axis=1
    )[:, 0, :]
    aerial_pass_distance = jnp.linalg.norm(
        aerial_pass_target - ball_field, axis=1
    )
    aerial_pass_speed = jnp.interp(
        aerial_pass_distance,
        jnp.asarray(
            [policy.aerial_pass_min_distance, policy.aerial_pass_max_distance]
        ),
        jnp.asarray(
            [policy.aerial_pass_min_speed_mps, policy.aerial_pass_max_speed_mps]
        ),
    )
    pass_vector = pass_target_all - ball_field[:, None, :]
    # ``ground_target_distance`` is the norm of this exact vector. Reusing it
    # avoids a second receiver-wise reduction/square-root on every policy tick.
    pass_forward_cos = pass_vector[:, :, 0] / (
        ground_target_distance + DIV_EPS
    )
    pass_is_forward, _, _ = _pass_direction_masks(pass_forward_cos)
    creative_lane = (
        (mate_adv > policy.through_min_progress)
        & ((through_strength > 0.02) | (pass_lead_all > 1.25))
    )
    # 실제 전방 러너를 겨냥한 creative lane만 낮은 through 하한을 쓴다. 종전 코드는
    # ``pass_is_forward``일 때 오히려 일반 전방 하한을 다시 골라, 이 예외가 전방에서
    # 죽고 비전방 후보에만 열리는 반대 조건이었다.
    ground_completion_floor = _ground_pass_completion_floor(
        pass_is_forward,
        creative_lane,
        policy,
    )
    # 짧은 전진 레인의 지상 완성률이 낮다는 이유만으로 로프트로 우회하지 않는다.
    # 그 낮은 값은 패스 길목이 막혔다는 뜻일 수도 있지만, 수신점 자체가 경합 중이라는
    # 뜻일 수도 있다. 후자는 공을 띄워도 해결되지 않는데 종전 분기는 둘을 구분하지 않아
    # 180초 기준 전방 로프트 15회 중 4회만 연결됐다. 짧은 전진은 아래의 수신자 이동·공간
    # 생성과 through 하한으로 만들고, 로프트는 스타일별 장거리 기준을 넘을 때만 쓴다.
    # 그 거리 판정은 **지상 수신 목표**로 해야 한다. 긴 체공시간만큼 러너를 추가
    # 외삽한 loft_target_distance로 판정하면, 가까운 전방 러너가 먼저 먼 목표로
    # 부풀고 그 부푼 값 때문에 로프트가 되는 자기강화 분기가 생긴다.
    # v48부터 거리 기준은 로프트의 *허용 조건*일 뿐 자동 선택 조건이 아니다. 안전한
    # 장거리 drive는 balanced/tiki-taka가 낮게 연결하고, 지상 레인이 막혔거나
    # long-ball처럼 directness가 명시적으로 높은 경우에만 공중 경로를 쓴다.
    candidate_loft = _prefer_loft_pass(
        ground_target_distance,
        ground_comp,
        ground_completion_floor,
        pass_forward_cos,
        direct,
        policy,
    )
    loft_control = T.aerial_reception(
        loft_target_all,
        others_field,
        receiver_vmax,
        loft_flight_t,
        others_field,
        o_vel,
        receiver_vmax,
        opp,
    )
    # 로프트는 지상 레인 완성률이 아니라 체공시간 내 도착·상대보다
    # 선점할 가능성이 품질 통화다. 일반 롱패스에도 크로스와 같은 물리 게이트를 쓴다.
    comp = jnp.where(candidate_loft, loft_control, ground_comp)
    ground_pass_ok = ground_comp >= ground_completion_floor
    # Live-ball recycling shares the configured low-outlet envelope used by
    # a challenge escape. Beyond it, a backward receiver cannot become the
    # default merely because every progressive lane carries some risk. Restarts
    # keep their dedicated fallback and are therefore not narrowed here.
    safe_live_direction = _safe_live_pass_direction(
        pass_forward_cos,
        ground_target_distance,
        transition_live,
        policy,
    )
    pass_ok = (
        pass_candidate
        & safe_live_direction
        & jnp.where(
            candidate_loft,
            loft_control >= policy.loft_control_floor,
            ground_pass_ok,
        )
    )
    receive_space = receive_space_m / 15.0
    # 연결 보너스는 원 패서까지 포함하되 각 후보 자신의 현재 위치는 제외한다.
    # v36의 단순 거리/피치가치는 출구 레인이 막혔어도 삼각형으로 오인했다.
    # v37은 첫 패스 비행 동안 제3선수를 짧게 선점하고, 수신점→지원점
    # drive의 실제 공속·수비 가속 ETA·순전진을 함께 본다.
    support_pos = jnp.concatenate([my_field[:, None, :], others_field], axis=1)
    support_vel = jnp.concatenate([my_vel[:, None, :], o_vel], axis=1)
    support_base = jnp.concatenate(
        [jnp.ones((N, 1), dtype=bool), mate], axis=1
    )
    receiver_is_self = jnp.concatenate(
        [jnp.zeros((N - 1, 1), dtype=bool), jnp.eye(N - 1, dtype=bool)], axis=1
    )
    support_mask = support_base[:, None, :] & (~receiver_is_self[None, :, :])
    static_combination = T.combination_value(
        pass_target_all, support_pos, support_mask, c.hx, c.hy
    )
    continuation_t = (
        jnp.minimum(
            ground_flight_t,
            policy.continuation_support_lead_cap_s,
        )
        * policy.continuation_support_lead_weight
    )
    # Physical two-hop scoring can affect the submitted action only on the
    # live-ball carrier, a reachable goalkeeper foot distribution, or the
    # designated restart taker. Select the causal row before entering the
    # receiver×support×opponent producer; ``argmax`` is harmless when the mask
    # is empty because the surrounding scalar condition skips the gather.
    decision_candidate = (
        (restart_active & self_is_taker)
        | ((~restart_active) & attacking & am_nearest_mate)
    )
    decision_row = jnp.argmax(decision_candidate).reshape(1).astype(jnp.int32)
    if c.execution_profile == "gpu_dense":
        decision_rows = []
        for team in range(TEAM_COUNT):
            team_rows = c.team_id == team
            decision_rows.extend(
                (
                    jnp.argmax(team_rows & am_nearest_mate),
                    jnp.argmax(team_rows & is_gk),
                    jnp.argmax(team_rows & self_is_taker),
                )
            )
        decision_rows = jnp.stack(decision_rows).astype(jnp.int32)
    else:
        gk_contact_candidate = (
            (~restart_active)
            & is_gk
            & in_reach
            & f2b_avail
            & (~decision_candidate)
        )
        gk_contact_rows = []
        gk_contact_valid = []
        for team in range(TEAM_COUNT):
            team_rows = c.team_id == team
            valid = jnp.any(team_rows & gk_contact_candidate)
            # The team representative is a distinct no-op scatter destination
            # when that team's goalkeeper cannot contact the ball.
            row = jnp.where(
                valid,
                jnp.argmax(team_rows & gk_contact_candidate),
                jnp.argmax(team_rows),
            )
            gk_contact_rows.append(row)
            gk_contact_valid.append(valid)
        gk_contact_rows = jnp.stack(gk_contact_rows).astype(jnp.int32)
        gk_contact_valid = jnp.stack(gk_contact_valid)

    def physical_continuation_for(rows):
        row_count = rows.shape[0]
        row_pass_target = pass_target_all[rows]
        row_support_pos = support_pos[rows]
        row_support_vel = support_vel[rows]
        row_continuation_t = continuation_t[rows]
        row_support_future = (
            row_support_pos[:, None, :, :]
            + row_support_vel[:, None, :, :]
            * row_continuation_t[:, :, None, None]
        )
        row_support_future = jnp.stack(
            [
                jnp.clip(
                    row_support_future[:, :, :, 0], -c.hx + 2.0, c.hx - 1.5
                ),
                jnp.clip(
                    row_support_future[:, :, :, 1], -c.hy + 1.5, c.hy - 1.5
                ),
            ],
            axis=-1,
        )
        row_shot_value = T.shot_xg(
            row_support_future.reshape(row_count, -1, 2),
            others_field[rows],
            opp[rows],
            o_gk[rows],
            c.hx,
            c.goal_w,
        ).reshape(row_support_future.shape[:3]) * policy.shoot_gain
        row_distance = jnp.linalg.norm(
            row_support_future - row_pass_target[:, :, None, :], axis=3
        )
        row_launch_speed = jnp.interp(row_distance, c.drive_R, c.drive_v)
        row_ball_speed = 0.5 * (
            row_launch_speed + policy.drive_arrive_speed_mps
        )
        return T.two_hop_continuation(
            row_pass_target,
            row_support_future,
            support_mask[rows],
            others_field[rows],
            o_vel[rows],
            receiver_vmax[rows],
            opp[rows],
            row_ball_speed,
            c.hx,
            c.hy,
            min_distance=policy.continuation_min_distance,
            max_distance=policy.continuation_max_distance,
            backward_tolerance=policy.continuation_backward_tolerance,
            intercept_radius=policy.pass_lane_intercept_radius,
            reaction_s=policy.pass_lane_reaction_s,
            defender_accel=policy.pass_lane_defender_acceleration_mps2,
            shot_value=row_shot_value,
            shot_gain=policy.pass_shot_chain_gain,
        )

    if c.execution_profile == "gpu_dense":
        decision_continuation = physical_continuation_for(decision_rows)
        physical_continuation = static_combination.at[decision_rows].set(
            decision_continuation
        )
    else:
        physical_continuation = jax.lax.cond(
            jnp.any(decision_candidate),
            lambda current: current.at[decision_row].set(
                physical_continuation_for(decision_row)
            ),
            lambda current: current,
            static_combination,
        )

        def add_goalkeeper_continuation(current):
            continuation = physical_continuation_for(gk_contact_rows)
            continuation = jnp.where(
                gk_contact_valid[:, None],
                continuation,
                current[gk_contact_rows],
            )
            return current.at[gk_contact_rows].set(continuation)

        physical_continuation = jax.lax.cond(
            jnp.any(gk_contact_valid),
            add_goalkeeper_continuation,
            lambda current: current,
            physical_continuation,
        )
    # 공중 수신 후의 즉시 2차 drive는 신뢰할 수 없으므로 기존 정적
    # 지원 신호만 절반 남긴다. 땅볼 후보만 물리 2-hop으로 선택한다.
    combination = jnp.where(
        candidate_loft,
        0.5 * static_combination,
        physical_continuation,
    )
    # 측면전환 소보너스(동률깨기·스타일) — 결정은 위협통화가 지배.
    wide_progress = jnp.clip(
        (pass_target_all[:, :, 0] - 0.05 * c.hx) / (0.45 * c.hx), 0.0, 1.0
    )
    switch_bonus = (
        jnp.clip(
            (jnp.abs(others_field[:, :, 1]) - jnp.abs(my_field[:, 1:2])) / c.hy,
            0.0,
            1.0,
        )
        * policy.wide_progression_gain
        * (0.5 + width_s[:, None])
        * wide_progress
    )
    # 전진 보상: 안전한 옆·뒤 패스(comp↑·threat 비슷)보다 전방 찔러주기를 우선. directness에 비례.
    prog = jnp.clip(mate_adv, 0.0, 25.0) / 25.0                                          # (N,M) 전진량 정규화
    prog_w = (
        policy.pass_progress_base
        + policy.pass_progress_direct_gain * direct[:, None]
    )
    risk_adjusted = 0.24 + 0.76 * comp
    combination_weight = policy.combination_gain * (
        1.0 + policy.possession_combination_gain * (1.0 - direct[:, None])
    )
    direct_distance_bonus = (
        policy.direct_distance_gain
        * direct[:, None]
        * jnp.clip((mate_ball_d - 14.0) / 28.0, 0.0, 1.0)
        * jnp.clip(mate_adv / 20.0, 0.0, 1.0)
    )
    possession_distance_cost = (
        policy.possession_distance_penalty
        * (1.0 - direct[:, None])
        * jnp.clip((mate_ball_d - 15.0) / 25.0, 0.0, 1.0)
    )
    V_pass_mate = jnp.where(
        pass_ok,
        comp
        * (
            threat_target
            + combination_weight * combination
            + 0.10 * receive_space
            + receiver_motion_value
        )
        + risk_adjusted
        * (
            prog_w * prog
            + policy.through_bonus * through_strength
            + direct_distance_bonus
        )
        + policy.receiver_control_value_gain * (o_ability - 0.5)
        + switch_bonus,
        -1.0,
    )
    # DFL PlayAngle과 같은 ±60도 구간으로 수신자 선택 prior를 준다. v36 첫 표본은
    # 안전 하한을 올린 뒤 전/횡/후가 51/35/14%로 전방만 남았다. 실제 라인 뒤 공간을
    # 가진 through 후보는 전방 비용을 연속적으로 해제해, 방향 균형이 킬패스를 지우지 않는다.
    direction_balance_value = _pass_direction_balance_value(
        pass_forward_cos,
        through_strength,
        ground_target_distance,
        direct,
        my_field[:, 0],
        press_self,
        own_fast_pass,
        c.hx,
        policy,
    )
    V_pass_mate = jnp.where(
        pass_ok, V_pass_mate + direction_balance_value, -1.0
    )
    # A goalkeeper is not merely an emergency endpoint.  DFL build-up cycles
    # show deliberate outfielder→GK→outfielder circulation, especially when a
    # team is pressed in its own half.  Reward only a legal, ground-pass
    # candidate; the ordinary lane/completion gate still owns safety.
    buildout_pressure = jnp.clip(
        press_self / policy.pass_pressure_scale, 0.0, 1.0
    )
    buildout_depth = jnp.clip(-my_field[:, 0] / c.hx, 0.0, 1.0)
    gk_buildout_candidate = (
        pass_ok
        & o_gk
        & (~candidate_loft)
        & (my_field[:, 0:1] < 0.0)
    )
    gk_buildout_bonus = (
        policy.gk_buildup_gain
        * (0.55 + 0.45 * buildout_pressure[:, None])
        * (0.55 + 0.45 * buildout_depth[:, None])
    )
    V_pass_mate = jnp.where(
        pass_ok,
        V_pass_mate + jnp.where(gk_buildout_candidate, gk_buildout_bonus, 0.0),
        -1.0,
    )
    V_pass_mate = jnp.where(
        pass_ok,
        V_pass_mate - possession_distance_cost,
        -1.0,
    )
    # 후방 패스는 그 자체로 잘못이 아니다. 다만 압박도 없고 자기 골문과도 멀 때 안전한
    # 수신확률·연계 보너스만으로 A→B→A를 반복해서는 안 된다. 후퇴거리만큼 비용을 키우되
    # 강압박에서는 완전히 해제하고, 자기 진영 깊이에서는 빌드업 안전판을 대부분 보존한다.
    V_pass_mate = jnp.where(
        pass_ok, V_pass_mate - backpass_context_cost, -1.0
    )
    # A newly won ball is not a completed counterpress until the team can keep
    # it.  During the short observable secure window, prefer a nearby flat exit
    # with a DFL-compatible safety margin.  If none exists, the carrier keeps
    # the ball and the escape-dribble branch below turns away from pressure.
    secure_outlet = (
        pass_ok
        & (~candidate_loft)
        & (mate_ball_d <= 20.0)
        & (ground_comp >= policy.secure_pass_completion_floor)
        & (mate_adv <= 10.0)
    )
    # 탈취 접촉의 즉시 출구는 다음 프레임의 possession label을 기다릴 수 없다. 현재
    # 수신점/레인 모델에서 짧고 낮으며 충분히 안전한 후보를 별도로 고른다. 캐리어 바로 옆
    # 상대 때문에 ``pass_contact_clear``는 필연적으로 거짓이므로, 그 릴리스 게이트를 여기
    # 재사용하지 않고 실제 lane/ETA 완성률이 안전성을 소유한다.
    challenge_outlet_candidate = (
        pass_ok
        & (~candidate_loft)
        & (mate_ball_d <= policy.challenge_outlet_max_distance)
        & (ground_comp >= policy.challenge_outlet_completion_floor)
        & (mate_adv <= 10.0)
    )
    challenge_outlet_value = (
        ground_comp
        + 0.08 * receive_space_10
        - 0.12 * jnp.clip(
            mate_ball_d / policy.challenge_outlet_max_distance, 0.0, 1.0
        )
    )
    challenge_outlet_j = jnp.argmax(
        jnp.where(challenge_outlet_candidate, challenge_outlet_value, -jnp.inf),
        axis=1,
    )
    has_challenge_outlet = jnp.any(challenge_outlet_candidate, axis=1)
    challenge_outlet_target = jnp.take_along_axis(
        pass_target_all, challenge_outlet_j[:, None, None], axis=1
    )[:, 0, :]
    secure_outlet_bonus = (
        0.30 * ground_comp
        + 0.08 * receive_space_10
    )
    V_pass_mate = jnp.where(
        secure_phase[:, None],
        jnp.where(secure_outlet, V_pass_mate + secure_outlet_bonus, -1.0),
        V_pass_mate,
    )
    effective_pass_ok = pass_ok & ((~secure_phase)[:, None] | secure_outlet)
    pass_u = jnp.clip(
        jax.random.uniform(k_pass, (N, N - 1)), PROB_EPS, 1.0 - PROB_EPS
    )
    gumbel = -jnp.log(-jnp.log(pass_u)) * 0.03
    # 수신자 선택의 거리 편향은 '패스를 할지'의 기대가치와 분리한다. 같은 값을 V_pass에서
    # 빼면 짧은 패스로 바뀌는 대신 전부 드리블로 폴백해 빈도 캘리브레이션을 훼손한다.
    # 50 m 후보 상한까지 선형 감점하되 direct 팀은 절반까지 완화해 스타일 차이를 남긴다.
    target_distance_fraction = jnp.clip(
        (mate_ball_d - policy.pass_min_distance)
        / (50.0 - policy.pass_min_distance),
        0.0,
        1.0,
    )
    target_distance_cost = (
        policy.pass_target_distance_penalty
        * (1.0 - 0.5 * direct[:, None])
        * target_distance_fraction
    )
    best_j = jnp.argmax(V_pass_mate - target_distance_cost + gumbel, axis=1)
    V_pass = jnp.take_along_axis(V_pass_mate, best_j[:, None], axis=1)[:, 0]             # (N,)
    has_pass = jnp.any(effective_pass_ok, axis=1)
    has_restart_pass = jnp.any(pass_candidate, axis=1)
    fallback_value = jnp.where(
        pass_candidate,
        lane_comp + 0.20 * threat_target + 0.08 * receive_space,
        -jnp.inf,
    )
    fallback_j = jnp.argmax(fallback_value, axis=1)
    # 오픈플레이의 보수적 완성률 하한이 세트피스 배급까지 막아서는 안 된다. viable 후보가
    # 없으면 가장 안전한 기하 후보를 dir/distance의 fallback으로 보존하고, V_pass는 -1로 남긴다.
    best_j = jnp.where(has_pass, best_j, fallback_j)
    best_ground_target = jnp.take_along_axis(
        pass_target_all, best_j[:, None, None], axis=1
    )[:, 0, :]
    best_loft_target = jnp.take_along_axis(
        loft_target_all, best_j[:, None, None], axis=1
    )[:, 0, :]
    best_lofted = jnp.take_along_axis(
        candidate_loft, best_j[:, None], axis=1
    )[:, 0]
    best_ground_comp = jnp.take_along_axis(
        ground_comp, best_j[:, None], axis=1
    )[:, 0]
    best_secure_outlet = jnp.take_along_axis(
        secure_outlet, best_j[:, None], axis=1
    )[:, 0]
    best_forward_candidate = jnp.take_along_axis(
        pass_is_forward, best_j[:, None], axis=1
    )[:, 0]
    best_pass_target = jnp.where(
        best_lofted[:, None], best_loft_target, best_ground_target
    )
    # An indirect free kick may never fall back to a direct goal attempt.  If
    # the quality/distance/offside filters leave no ordinary pass candidate,
    # ``argmax(all -inf)`` is merely index 0 and can name an opponent or an
    # inactive slot.  Preserve a guaranteed active-team fallback separately
    # from the open-play value decision.
    has_any_mate = jnp.any(mate, axis=1)
    nearest_mate_j = jnp.argmin(
        jnp.where(mate, mate_ball_d, jnp.inf), axis=1
    )
    nearest_mate_target = jnp.take_along_axis(
        others_field, nearest_mate_j[:, None, None], axis=1
    )[:, 0, :]
    no_mate_target = ball_field + jnp.asarray([1.0, 0.0])
    nearest_mate_target = jnp.where(
        has_any_mate[:, None], nearest_mate_target, no_mate_target
    )
    restart_pass_target = jnp.where(
        has_restart_pass[:, None], best_ground_target, nearest_mate_target
    )
    restart_pass_distance = (
        jnp.linalg.norm(restart_pass_target - ball_field, axis=1) + DIV_EPS
    )
    restart_pass_dir = _unit(restart_pass_target - ball_field)
    best_through = jnp.take_along_axis(through_strength, best_j[:, None], axis=1)[:, 0]
    # GK 롱배급은 고정된 '측면 지역'보다 실제 동료의 예상 도착점을
    # 우선한다. 도달 가능성·상대 선점·최소 전진/거리를 모두 통과한 후보만
    # 롱 타겟이며, 후보가 없을 때만 기존 인-피치 지역 클리어로 폴백한다.
    gk_long_candidate = (
        pass_candidate
        & (~o_gk)
        & (loft_target_distance >= policy.gk_long_min_distance)
        & (mate_adv >= policy.gk_long_min_progress)
        & (loft_control >= policy.loft_control_floor)
    )
    gk_long_value = (
        loft_control
        + 0.25 * jnp.clip(mate_adv / 25.0, 0.0, 1.0)
        + 0.10 * receive_space
    )
    gk_long_j = jnp.argmax(
        jnp.where(gk_long_candidate, gk_long_value, -jnp.inf), axis=1
    )
    has_gk_long_receiver = jnp.any(gk_long_candidate, axis=1)
    gk_long_receiver_target = jnp.take_along_axis(
        loft_target_all, gk_long_j[:, None, None], axis=1
    )[:, 0, :]
    # Keep a goalkeeper-specific short outlet independent of the carrier's
    # globally best option.  The latter may deliberately be a 30m loft; using
    # it as the only foot-play candidate made a nearby free centre-back
    # invisible and converted every restricted back-pass into a clearance.
    gk_short_candidate = (
        pass_candidate
        & (~o_gk)
        & (mate_ball_d <= policy.gk_foot_pass_max_distance)
        & ground_pass_ok
    )
    gk_short_value = (
        ground_comp
        + 0.12 * receive_space
        + 0.08 * combination
        + policy.receiver_control_value_gain * (o_ability - 0.5)
        - 0.10 * jnp.clip(mate_ball_d / policy.gk_foot_pass_max_distance, 0.0, 1.0)
    )
    gk_short_j = jnp.argmax(
        jnp.where(gk_short_candidate, gk_short_value, -jnp.inf), axis=1
    )
    has_gk_short_receiver = jnp.any(gk_short_candidate, axis=1)
    gk_short_target = jnp.take_along_axis(
        pass_target_all, gk_short_j[:, None, None], axis=1
    )[:, 0, :]
    d_gk_short = jnp.linalg.norm(gk_short_target - ball_field, axis=1) + DIV_EPS
    dir_gk_short = _unit(gk_short_target - ball_field)
    pass_release_probability = _pass_release_probability(
        press_self, best_pass_target[:, 0], best_through, c.hx, policy,
        closing_self=closing_self,
    )
    pass_release_probability = jnp.where(
        secure_phase & best_secure_outlet,
        jnp.maximum(
            pass_release_probability,
            policy.secure_release_probability,
        ),
        pass_release_probability,
    )
    pass_release_draw = jax.random.uniform(k_release, (N,))
    pass_release_now = pass_release_draw < pass_release_probability
    # A close opponent must not create a policy deadlock.  Ordinary passes and
    # crosses still require contact clearance, but a short ground outlet whose
    # lane/receiver ETA clears the stricter challenge floor may be played away
    # from the pressing body.  The physical simultaneous-contact resolver still
    # decides whether a defender who actually challenges blocks that release.
    pressured_outlet_ready = (
        attacking
        & am_nearest_mate
        & (~pass_contact_clear)
        & has_challenge_outlet
        & (
            pass_release_draw
            < jnp.maximum(
                pass_release_probability,
                policy.contact_release_probability,
            )
        )
    )
    relay_release_draw = jax.random.uniform(k_relay, (N,))
    best_pass_delta = best_pass_target - ball_field
    best_pass_direction = _unit(best_pass_delta)
    incoming_direction = _unit(ball_vel[:, :2])
    my_velocity_direction = _unit(my_vel)
    relay_alignment = jnp.sum(
        best_pass_direction * my_velocity_direction, axis=1
    )
    relay_return_like = (
        jnp.sum(best_pass_direction * incoming_direction, axis=1)
        < policy.quick_relay_return_cos
    )
    relay_body_ready = (
        (my_speed < 0.8)
        | (relay_alignment >= policy.quick_relay_alignment_floor)
    )
    relay_speed_ceiling = (
        policy.quick_relay_min_incoming_speed_mps
        + (
            policy.quick_relay_max_incoming_speed_mps
            - policy.quick_relay_min_incoming_speed_mps
        )
        * (
            policy.quick_relay_low_skill_speed_fraction
            + (1.0 - policy.quick_relay_low_skill_speed_fraction)
            * self_ability
        )
    )
    relay_completion_floor = (
        policy.quick_relay_completion_floor
        + policy.quick_relay_floor_skill_relief * (0.5 - self_ability)
    )
    contextual_relay_probability = _contextual_quick_relay_probability(
        policy.quick_relay_probability,
        self_ability,
        relay_alignment,
        my_speed,
        press_self,
        relay_return_like,
        policy,
    )
    quick_relay_ready = (
        own_fast_pass
        & last_touch_was_pass
        & (ball_speed <= relay_speed_ceiling)
        & relay_body_ready
        & am_nearest_mate
        & in_reach
        & f2b_avail
        & has_pass
        & (~best_lofted)
        & (
            best_ground_comp
            >= jnp.where(
                best_forward_candidate,
                jnp.maximum(
                    relay_completion_floor,
                    policy.pass_forward_completion_floor,
                ),
                relay_completion_floor,
            )
        )
        & (relay_release_draw < contextual_relay_probability)
    )
    d_pass = jnp.linalg.norm(best_pass_delta, axis=1) + DIV_EPS
    dir_pass = best_pass_direction

    # 크로스는 일반 패스를 고른 뒤 목표를 고정 좌표로 바꾸지 않는다. 박스에 실제로 침투한 수신자별
    # 도착점·공간·xG를 평가해 별도 선택지로 비교한다. 덕분에 빈 박스 크로스와 수신자 불일치가 사라진다.
    cross_flight_t = jnp.interp(mate_ball_d, c.cross_R, c.cross_T)
    cross_flight_extra = jnp.maximum(
        0.0, 0.76 * cross_flight_t - drive_lead_t
    )
    cross_target_all = pass_target_all + o_vel * (
        cross_flight_extra
    )[:, :, None]
    cross_target_all = jnp.stack(
        [
            jnp.clip(cross_target_all[:, :, 0], -c.hx + 2.0, c.hx - 1.5),
            jnp.clip(cross_target_all[:, :, 1], -c.pen_hw - 5.0, c.pen_hw + 5.0),
        ],
        axis=-1,
    )
    # 첫 lead로 목표거리가 바뀌었으므로 실제 백스핀 거리표에서 체공시간을 다시 읽고 한 번
    # 재수렴시킨다. 현재 mate 거리의 시간만 쓰면 빠른 대각선 러너를 짧게 예측한다.
    cross_distance1 = jnp.linalg.norm(
        cross_target_all - ball_field[:, None, :], axis=2
    )
    cross_flight_t = jnp.interp(cross_distance1, c.cross_R, c.cross_T)
    cross_flight_extra = jnp.maximum(
        0.0, 0.76 * cross_flight_t - drive_lead_t
    )
    cross_target_all = pass_target_all + o_vel * cross_flight_extra[:, :, None]
    cross_target_all = jnp.stack(
        [
            jnp.clip(cross_target_all[:, :, 0], -c.hx + 2.0, c.hx - 1.5),
            jnp.clip(cross_target_all[:, :, 1], -c.pen_hw - 5.0, c.pen_hw + 5.0),
        ],
        axis=-1,
    )
    # 박스 안 선수만 기다리면 수비 블록도 함께 내려와 크로스가 거의 생기지 않는다. 박스 전방
    # 14m부터 달려드는 선수도 비행시간 예측으로 받되, 캐리어의 별도 cross_start가 너무 이른 공급을 막는다.
    cross_entry_x = c.hx - c.pen_len - 14.0
    # '내 앞으로 올린다' 조건은 오픈플레이 전용이다. 코너 키커는 골라인 위(x≈hx)에 서므로
    # 박스 안 어떤 목표도 자기보다 앞설 수 없고, 그대로 두면 코너가 크로스 후보를 하나도
    # 갖지 못해 발밑 로프트로 되돌아간다.
    cross_ahead_ok = (cross_target_all[:, :, 0] > my_field[:, 0:1] + 2.0) | (
        restart_active[:, None]
    )
    cross_candidate = (
        pass_candidate
        & (cross_target_all[:, :, 0] > cross_entry_x)
        & cross_ahead_ok
        & (jnp.abs(cross_target_all[:, :, 1]) < c.pen_hw + 5.0)
    )
    # Flight geometry remains full-row because its selected arrival is exposed
    # in the Phase-S trace. The expensive receiver×opponent quality metrics and
    # argmax can affect only the causal carrier/restart-taker row.
    def cross_value_for(rows):
        row_target = cross_target_all[rows]
        row_others = others_field[rows]
        row_opp = opp[rows]
        row_o_gk = o_gk[rows]
        row_o_vel = o_vel[rows]
        row_vmax = receiver_vmax[rows]
        row_candidate = cross_candidate[rows]
        row_xg = T.shot_xg(
            row_target,
            row_others,
            row_opp,
            row_o_gk,
            c.hx,
            c.goal_w,
        )
        row_space = T.openness(
            row_target, row_others, row_o_vel, row_opp
        ) / 15.0
        row_control = T.aerial_reception(
            row_target,
            row_others,
            row_vmax,
            cross_flight_t[rows],
            row_others,
            row_o_vel,
            row_vmax,
            row_opp,
        )
        receiver_profile = (
            0.65 * o_ability[rows] + 0.35 * o_reach_ability[rows]
        )
        # Only services whose runner reaches the target before the defense
        # survive this quality gate.
        row_candidate = row_candidate & (
            row_control >= policy.cross_control_floor
        )
        far_post = (
            jnp.sign(row_target[:, :, 1] + DIV_EPS)
            != jnp.sign(my_field[rows, 1:2] + DIV_EPS)
        ).astype(jnp.float32)
        value = jnp.where(
            row_candidate,
            policy.cross_gain
            * (
                0.70 * row_xg * policy.shoot_gain
                + 0.18 * row_space
                + 0.08 * far_post
                + 0.12 * through_strength[rows]
                + policy.cross_receiver_ability_gain
                * (receiver_profile - 0.5)
            )
            * (0.52 + 0.48 * row_control),
            -1.0,
        )
        best_j = jnp.argmax(value, axis=1)
        return (
            best_j,
            jnp.take_along_axis(value, best_j[:, None], axis=1)[:, 0],
            jnp.take_along_axis(
                row_target, best_j[:, None, None], axis=1
            )[:, 0, :],
            jnp.any(row_candidate, axis=1),
        )

    if c.execution_profile == "gpu_dense":
        best_cross_j, V_cross, best_cross_target, has_cross = cross_value_for(
            jnp.arange(N, dtype=jnp.int32)
        )
    else:
        def active_cross_value(_):
            best_j, best_value, best_target, present = cross_value_for(
                decision_row
            )
            return (
                jnp.zeros((N,), dtype=jnp.int32).at[decision_row].set(best_j),
                jnp.full((N,), -1.0).at[decision_row].set(best_value),
                jnp.zeros_like(my_field).at[decision_row].set(best_target),
                jnp.zeros((N,), dtype=bool).at[decision_row].set(present),
            )

        def empty_cross_value(_):
            return (
                jnp.zeros((N,), dtype=jnp.int32),
                jnp.full((N,), -1.0),
                jnp.zeros_like(my_field),
                jnp.zeros((N,), dtype=bool),
            )

        best_cross_j, V_cross, best_cross_target, has_cross = jax.lax.cond(
            jnp.any(decision_candidate),
            active_cross_value,
            empty_cross_value,
            jnp.int32(0),
        )

    # 드리블 가치: 전방 5m 위협 × 유지확률(압박·제어력 반영). 열린 평상시는 골 방향을
    # 유지하되, 탈취 직후·전방 차단·강압박에서는 8개 4m 후보의 공간·전진·운동량을
    # 함께 평가한다. 그래서 모든 캐리어가 같은 +x 직선으로 수비 몸에 들어가지 않는다.
    fwd_goal = _unit(opp_goal[None, :] - my_field)
    keep_prob = jnp.clip(
        (1.0 - 0.45 * press_self)
        * (
            1.0
            + policy.dribble_control_retention_gain * (self_ability - 0.5)
        ),
        0.25,
        1.0,
    )
    # 드리블 목표 = 골 방향 5m 전진(단, 골 2m 앞까지만 — 오버런 방지). 그 지점의 슛 가치(xg·GAIN)를
    # 전개 가치(pitch)와 max로 결합 → **드리블로 더 좋은 슛 위치를 만드는 것**이 슛 통화로 평가된다.
    step_len = jnp.clip(d_goal - 2.0, 0.0, 5.0)
    forward_drib_step = my_field + fwd_goal * step_len[:, None]
    escape_dirs = _unit(jnp.asarray([
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
        [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0],
    ], dtype=jnp.float32))
    escape_candidates = my_field[:, None, :] + 4.0 * escape_dirs[None, :, :]
    escape_candidates = jnp.stack([
        jnp.clip(escape_candidates[:, :, 0], -c.hx + 2.0, c.hx - 2.0),
        jnp.clip(escape_candidates[:, :, 1], -c.hy + 2.0, c.hy - 2.0),
    ], axis=-1)
    escape_open = T.openness(
        escape_candidates, others_field, o_vel, opp
    )
    escape_progress = jnp.clip(
        (escape_candidates[:, :, 0] - my_field[:, 0:1]) / 4.0,
        -1.0,
        1.0,
    )
    escape_score = _dribble_escape_scores(
        escape_open,
        escape_progress,
        escape_dirs,
        my_vel,
        self_ability,
        policy,
    )
    escape_j = jnp.argmax(escape_score, axis=1)
    escape_step = jnp.take_along_axis(
        escape_candidates, escape_j[:, None, None], axis=1
    )[:, 0, :]
    best_escape_open = jnp.take_along_axis(
        escape_open, escape_j[:, None], axis=1
    )[:, 0]
    evade_dribble = (
        secure_phase
        | fwd_blocked
        | (press_self >= policy.dribble_evade_pressure)
    )
    drib_step = jnp.where(
        evade_dribble[:, None], escape_step, forward_drib_step
    )
    xg_step = T.shot_xg(drib_step, others_field, opp, o_gk, c.hx, c.goal_w)       # 드리블 후 슛 가치
    # 드리블 후 슛 가치 xg_step은 **GAIN 없이**(≤1) — 드리블은 '접근수단'이라 pitch와 같은 스케일로만
    # 반영(GAIN 스케일 시 무한 접근-드리블로 과드리블). 근접해 xg_self가 크면 V_shoot=xg·GAIN이 이겨 슛.
    # 0.85 배 — 드리블은 보조수단(맨 제치기·근접). 전진패스를 1차 진행수단으로.
    V_drib = jnp.maximum(T.pitch_value(drib_step, c.hx, c.hy), xg_step) * keep_prob
    blocked_escape_quality = 0.70 + 0.30 * jnp.clip(
        best_escape_open / 8.0, 0.0, 1.0
    )
    V_drib = jnp.where(
        fwd_blocked & (~secure_phase),
        V_drib * blocked_escape_quality,
        V_drib,
    )

    # 슛 가치: 사거리 안에서만(밖이면 0 → 반드시 접근).
    # 통화는 sv_self=xg·RulePolicy.shoot_gain.
    in_range = d_goal < c.shoot_range
    V_shoot = jnp.where(in_range, sv_self, 0.0)

    # 전개(클리어) 가치: challenge와 같은 좁은 긴급 문맥에서만 선택지를 연다.
    # 구 ``else=0.02``는 주석과 달리 패스가 없고 드리블 가치가 낮은 모든 위치에서
    # 클리어가 argmax를 이길 수 있었다. 문맥 밖은 -1로 닫아 그 묵시적 롱볼을 없앤다.
    emergency_clear_context = _emergency_clearance_context(
        ball_adv, press_self, has_pass, c.hx, policy
    )
    V_clear = jnp.where(emergency_clear_context, 0.14, -1.0)

    # 스타일 변조: tempo↑ 패스선호. 낮은 directness도 짧은 점유 연계를 자주 해야 하므로
    # pass 빈도를 tempo 하나로만 결정하지 않는다. directness↑는 먼 전방 후보·슛을 선호한다.
    V_pass = V_pass * (
        0.80 + 0.25 * tempo + 0.22 * (1.0 - direct)
    ) - dynamic_release_cost
    V_drib = V_drib * (
        0.90 + 0.25 * (1.0 - tempo) + open_dribble_bonus
    )
    V_shoot = V_shoot * (0.90 + 0.25 * direct)
    wide = jnp.abs(my_field[:, 1]) > c.hy * policy.cross_wide_fraction
    cross_zone = (
        wide
        & (ball_adv > c.hx * policy.cross_start_fraction)
        & (d_goal > 12.0)
        & has_cross
    )
    V_cross = jnp.where(
        cross_zone,
        V_cross * (
            0.82 + 0.22 * width_s + 0.18 * tempo + 0.24 * direct
        ),
        -1.0,
    )

    # ★추상 행동 선택 — 공통 통화 argmax(0 슛 / 1 패스 / 2 드리블 / 3 전개 / 4 크로스)
    opt = jnp.argmax(
        jnp.stack([V_shoot, V_pass, V_drib, V_clear, V_cross], axis=1), axis=1
    )
    # 슛 = argmax가 '지금 슛'을 택했을 때(=패스·드리블로 더 좋은 슛을 만드는 것보다 지금 슛이 나을 때).
    # 하드 거리강제(d_goal<14) 제거 — range에 들었다고 무조건 때리지 않는다. 단 확실한 대찬스(xg>0.55)는
    # 개선 여지 무관하게 즉시 슛(탭인 흘리지 않음).
    first_time_finish = (
        own_fast_pass
        & am_nearest_mate
        & attacking
        & (d_goal < policy.first_time_shot_distance)
        & (xg_self >= policy.first_time_shot_xg)
    )
    # 킬패스를 받은 직후 범위 밖에서 한 번 전진 터치했는데 다음 프레임 곧바로 횡패스로
    # 되돌리는 단절을 막는다. 마지막 자기 팀 터치가 DRIBBLE이고 슛 범위가 가까우며 현재
    # 위협이 충분할 때만 범위 안까지 운반한다. 터치 주체가 바뀌거나 수비 접촉이 생기면
    # last_touch_code가 바뀌므로 별도 숨은 메모리 없이 즉시 해제된다.
    own_shot_setup_touch = (
        (obs[:, c.i_last_touch] > 0.5)
        & (last_touch_code == TOUCH_DRIBBLE)
    )
    shot_approach = (
        own_shot_setup_touch
        & am_nearest_mate
        & attacking
        & (~in_range)
        & (d_goal < c.shoot_range + policy.shot_approach_margin)
        & (xg_self >= policy.shot_approach_xg)
    )
    # xG 하한은 '평범한 슛'에만 건다. 원터치 마무리는 별도 계약(first_time_shot_xg)을
    # 가지므로 일반 하한이 그것을 조용히 덮으면 안 된다 — 덮으면 인입 패스를 받은 선수가
    # 16 m 밖에서 절대 때리지 않는 비축구가 된다.
    shot_floor = jnp.where(
        first_time_finish, policy.first_time_shot_xg, policy.shot_min_xg
    )
    shoot = (
        (opt == 0) | (xg_self > policy.shot_commit_xg) | first_time_finish
    ) & in_range & (xg_self > shot_floor)
    # 폭을 잡은 캐리어에게 실재 박스 러너가 있으면, 크로스 가치가 일반 패스의 72%만 되어도
    # 서비스의 구조적 이점(수비 라인을 한 번에 넘김)을 인정한다. 빈 박스는 has_cross에서 차단된다.
    cross_preference_ratio = policy.cross_preference_ratio * (
        1.0 + policy.cross_direct_preference_gain * (0.5 - direct)
    )
    cross_preferred = (
        cross_zone
        & (V_cross >= cross_preference_ratio * V_pass)
        & (V_cross >= policy.cross_dribble_preference_ratio * V_drib)
    )
    crossing = (
        ((opt == 4) | cross_preferred)
        & (~shoot)
        & (~shot_approach)
        & cross_zone
        & pass_contact_clear
    )
    quick_relay = (
        quick_relay_ready
        & (~shoot)
        & (~crossing)
        & (~shot_approach)
        & pass_contact_clear
    )
    chose_pass = (opt == 1) & (~shoot) & (~crossing)
    pressured_outlet = (
        pressured_outlet_ready
        & (~shoot)
        & (~crossing)
        & (~shot_approach)
    )
    carrier_pass_all = (
        quick_relay
        | (
            chose_pass
            & has_pass
            & (~shot_approach)
            & pass_release_now
            & pass_contact_clear
        )
        | pressured_outlet
    )
    cross_side = jnp.sign(my_field[:, 1] + DIV_EPS)
    dir_cross = _unit(best_cross_target - ball_field)
    carrier_pass = carrier_pass_all
    carrier_clear = (
        (opt == 3)
        & (~shoot)
        & (~shot_approach)
        & (~carrier_pass_all)
    )                                                               # V_clear가 이미 무옵션·강압박·자기진영 게이트
    # 나머지 전부 드리블(항상 폴백) — '드리블을 골랐다'와 '패스를 골랐지만 무옵션'을 함께 흡수한다.
    carrier_drib = (~shoot) & (~carrier_pass_all) & (~carrier_clear) & (~crossing)

    forward_ball = _unit(opp_goal[None, :] - ball_field)
    dribble_dir = _unit(drib_step - ball_field)
    carrier_pass_target = jnp.where(
        pressured_outlet[:, None], challenge_outlet_target, best_pass_target
    )
    carrier_pass_distance = (
        jnp.linalg.norm(carrier_pass_target - ball_field, axis=1) + DIV_EPS
    )
    carrier_pass_dir = _unit(carrier_pass_target - ball_field)
    c_kick_dir = jnp.where(shoot[:, None], dir_shot,
                 jnp.where(crossing[:, None], dir_cross,
                 jnp.where(carrier_pass[:, None], carrier_pass_dir,
                 jnp.where(carrier_drib[:, None], dribble_dir, forward_ball))))
    # 정밀 킥 솔버(팩토리 캘리브): 목표거리 R → 필요 파워(=발사속도/f2b_max). loft=아크(착지)·drive=지상(트래핑 도착).
    def loft_pow_of(R):  return jnp.clip(jnp.interp(R, c.loft_R, c.loft_v) / c.f2b_max, 0.15, 1.0)
    def drive_pow_of(R): return jnp.clip(jnp.interp(R, c.drive_R, c.drive_v) / c.f2b_max, 0.12, 1.0)
    def cross_pow_of(R): return jnp.clip(jnp.interp(R, c.cross_R, c.cross_v) / c.f2b_max, 0.15, 1.0)
    def throw_pow_of(R): return jnp.clip(jnp.interp(R, c.throw_R, c.throw_v) / c.throw_speed_max, 0.12, 1.0)  # 손 던지기
    # 횡커브 때문에 실제 종방향 경로는 골 중심 직선거리보다 길다. 위에서 계산한 물리 경로로
    # 발사각도 조회해야 먼 포스트 보정 뒤 크로스바 높이가 달라지지 않는다.
    final_straight_d = jnp.sqrt(goal_dx * goal_dx + (aim_y - ball_field[:, 1]) ** 2)
    final_dir_x = goal_dx / jnp.maximum(final_straight_d, DIV_EPS)
    final_dir_y = (aim_y - ball_field[:, 1]) / jnp.maximum(final_straight_d, DIV_EPS)
    signed_curve = far_side * curve_displacement
    shot_path_d = final_straight_d + final_dir_y * signed_curve / jnp.maximum(final_dir_x, 0.1)
    shot_launch = jnp.interp(
        shot_path_d, c.shot_d, c.shot_launch01
    )  # 커브 포함 물리 경로별 저각 강타(지면 접촉 기준 action fraction)

    # 수신자별 지상/로프트 선택은 위의 체공시간·ETA 게이트에서 이미
    # 확정됐다. 킥 직전에 거리만으로 다시 판정하면 검증한 수신 품질과 킥 모드가 갈린다.
    lofted = carrier_pass & best_lofted & (~pressured_outlet)
    pass_pow = drive_pow_of(carrier_pass_distance)                  # 지상 드라이브 — 리시버에 트래핑 속도로 도착
    loft_pow = loft_pow_of(carrier_pass_distance)                   # 아크 — 수비수 넘겨 목표 지점 착지
    carry_touch_speed_cap = (
        jnp.minimum(policy.carried_release_speed_mps, c.drib_max)
        - RECEPTION_CONTROL_HYSTERESIS_MPS
    )
    drib_pow = jnp.clip(
        (my_speed + 1.5) / c.f2b_max,
        0.03,
        carry_touch_speed_cap / c.f2b_max,
    )                                                               # 내 속도보다 살짝 앞서되 통제 운반 밴드 안
    d_cross = jnp.linalg.norm(best_cross_target - ball_field, axis=1)
    cross_pow = cross_pow_of(d_cross)                              # 백스핀 포함 물리로 목표점 정밀 착지
    clear_pow = loft_pow_of(jnp.clip(ball_d_goal, 20.0, 42.0))    # 캐리어 클리어 — 전방 롱 아크(솔버)
    c_kick_pow = jnp.where(shoot, (c.shot_pow if policy.use_shot_solver else 0.95), jnp.where(crossing, cross_pow,
                 jnp.where(carrier_pass, jnp.where(lofted, loft_pow, pass_pow),
                 jnp.where(carrier_drib, drib_pow, clear_pow))))
    # 발사각: 슛은 거리별 저각(shot_launch), 크로스·클리어는 loft, 드라이브 패스는 drive.
    c_launch = jnp.where(shoot, (shot_launch if policy.use_shot_solver else 0.05), jnp.where(crossing, c.cross_launch01,
               jnp.where(carrier_pass, jnp.where(lofted, c.loft_launch01, c.drive_launch01),
               jnp.where(carrier_drib, 0.0, c.loft_launch01))))
    c_spin_s = jnp.where(
        shoot,
        far_side * curve_spin,
        jnp.where(crossing, -cross_side * (0.55 + 0.25 * width_s), 0.0),
    )
    # 슛은 약한 탑스핀으로 떨어뜨리고 크로스는 백스핀으로 체공·감속시킨다.
    c_spin_b = jnp.where(
        shoot,
        -(0.22 + 0.18 * curve_progress),
        jnp.where(crossing, 0.38, 0.0),
    )

    # 통제된(느린) 공에서만 의도적 킥. 빠른 공은 아래 트래핑이 먼저 잡는다(2-터치).
    settled = ball_speed < policy.settled_ball_speed_mps
    carried_release = (
        own_shot_setup_touch
        & (ball_speed < policy.carried_release_speed_mps)
    )
    release_controlled = settled | carried_release
    dribble_recontact_ready = _dribble_recontact_ready(
        ball_field - my_field,
        ball_vel_xy - my_vel,
        ball_speed,
        i_touched_last,
        last_touch_code,
        policy.unattended_ball_speed,
    )
    drib_touch = (
        carrier_drib & (ball_speed < 2.5) & dribble_recontact_ready
    )
    shoot_ok = shoot & (
        release_controlled | (d_goal < 16.0) | first_time_finish
    )  # 근거리와 명확한 인입 패스는 트래핑 대기 없이 원터치 마무리
    carrier_kick = (
        shoot_ok
        | quick_relay
        | (
            (crossing | carrier_pass | carrier_clear) & release_controlled
        )
        | (drib_touch & settled)
    )

    # 오프더볼 배치: 홈 형태의 좌우 간격·서열을 보존한 채(과대 확장 금지) 공 전진도만큼 전방 이동하고
    # 블록 전체를 공 쪽으로 살짝 슬라이드(볼사이드 컴팩트). y를 ×1.15~1.65로 부풀리고 |y|>18
    # 전원(6/10)을 터치라인(±29m)에 붙박으면 '양끝 바벨·중앙 공동'이 된다(측정: 측면 51%·중앙 18%).
    # 공격(우리 점유) 시 전방 침투 가속 — 포워드는 파이널서드로, 미드는 그 뒤로 올라가 '전진 패스 타깃'을
    # 만든다(전방 주자가 없으면 전진 패스를 넣을 곳이 없어 볼이 미드필드에서 맴돈다 = 전진성 패스의 전제).
    # 기준점은 실측 위치장(``positioning.py``)에서 읽는다. 구판은 홈 포메이션에
    # ``ball_adv*0.30 + line_h*10 + 역할상수``를 더한 손튜닝 수식이었는데, K리그 45경기
    # 적합은 블록의 ball_x 추종 계수가 국면·슬롯마다 0.55~0.68이고 ball_y 슬라이드도
    # 0.26~0.52로 갈린다고 말한다. 하나의 상수로는 재현할 수 없어 표를 그대로 쓴다.
    # ``attacking``이 국면 축을 고르므로 이 한 배열이 공격·수비 기준점을 모두 준다.
    phase_w = jnp.stack(
        [attacking, defending, loose_phase], axis=1
    ).astype(jnp.float32)
    anchor = _anchor_positions(c, ball_field, phase_w)
    line_shift = (line_h - 0.5) * policy.style_line_span
    width_scale = policy.width_base + policy.width_gain * (width_s - 0.5) * 2.0
    off_x = anchor[:, 0] + line_shift
    # 온사이드 유지: 공격 시 전방 주자는 최종수비 라인(off_line)까지만 — 라인을 넘으면 오프사이드로
    # 전진 패스가 무효가 된다. 라인 어깨에서 대기하다 스루패스에 뛰어들게(전진성 패스 성립).
    forward_shoulders_line = (
        attacking
        & (c.roles == ROLE_FORWARD)
        & (onside_line_x > ball_adv + policy.forward_run_min_gap)
    )
    off_x = jnp.where(
        forward_shoulders_line,
        jnp.maximum(off_x, onside_line_x - policy.forward_line_margin),
        off_x,
    )
    off_x = jnp.where(attacking, jnp.minimum(off_x, onside_line_x), off_x)
    off_y = anchor[:, 1] * width_scale
    # 폭 유지 자체는 표가 담당한다(측면 슬롯의 y0와 ball_y 기울기가 '볼 근처 윙어는 라인을
    # 밟고 반대편 윙어는 안으로 접힌다'를 이미 재현한다). 이 플래그는 측면 슬롯을 캐리어
    # 서포트로 차출하지 않기 위한 역할 표시로만 남는다.
    is_wide = (
        (c.detailed_roles == player_roles.ROLE_FULL_BACK)
        | (c.detailed_roles == player_roles.ROLE_WIDE_MID)
        | (c.detailed_roles == player_roles.ROLE_WIDE_FORWARD)
    )
    hold_wide = is_wide & (ball_adv > c.hx * policy.wide_progress_gate)
    off_target = jnp.stack([jnp.clip(off_x, -c.hx + 4.0, c.hx - 6.0),
                            jnp.clip(off_y, -c.hy + 3.0, c.hy - 3.0)], axis=1)

    # 파이널서드의 조건부 역할 점유. 실측의 박스 안 공격수 중앙값은 1명이지 전원 쇄도가
    # 아니다. 중앙 9번은 공이 파이널서드에 들어온 뒤 출발하면 도착이 늦으므로 경계 8.75m
    # 전부터 라인 어깨/니어 중앙을 선점한다. 실제 파이널서드에서는 약한 쪽 와이드 포워드
    # 한 명이 파포스트 레인을, 깊은 전개에서 중앙 미드필더가 컷백 존을 맡는다. 세 역할 모두
    # 관측 가능한 현재 위치와 정적 포메이션 슬롯만으로 정해져 tracking에서도 복원 가능하다.
    box_approach_attack = attacking & (carrier_field[:, 0] > c.hx / 6.0)
    final_third_attack = attacking & (carrier_field[:, 0] > c.hx / 3.0)
    deep_final_attack = attacking & (
        carrier_field[:, 0] > c.hx - c.pen_len - 2.0
    )
    (
        central_box_runner,
        central_box_backup,
        far_box_runner,
        cutback_runner,
    ) = _assign_box_runner_roles(
        box_approach_attack,
        final_third_attack,
        deep_final_attack,
        c.detailed_roles,
        am_nearest_mate,
        carrier_role,
        carrier_field[:, 1],
        c.home_att[:, 1],
        c.team_id,
    )
    central_box_target = jnp.stack(
        [
            jnp.full_like(carrier_field[:, 0], c.hx - 9.0),
            jnp.clip(0.22 * carrier_field[:, 1], -5.0, 5.0),
        ],
        axis=1,
    )
    far_box_target = jnp.stack(
        [
            jnp.full_like(carrier_field[:, 0], c.hx - 7.0),
            -jnp.sign(carrier_field[:, 1] + DIV_EPS)
            * jnp.minimum(0.42 * c.goal_w, c.pen_hw - 2.0),
        ],
        axis=1,
    )
    cutback_target = jnp.stack(
        [
            jnp.full_like(carrier_field[:, 0], c.hx - c.pen_len - 2.5),
            jnp.clip(-0.25 * carrier_field[:, 1], -9.0, 9.0),
        ],
        axis=1,
    )
    box_target = jnp.where(
        (central_box_runner | central_box_backup)[:, None],
        central_box_target,
        jnp.where(far_box_runner[:, None], far_box_target, cutback_target),
    )
    box_target = jnp.stack(
        [
            jnp.minimum(box_target[:, 0], onside_line_x - 0.25),
            jnp.clip(box_target[:, 1], -c.pen_hw + 1.0, c.pen_hw - 1.0),
        ],
        axis=1,
    )
    box_runner = (
        central_box_runner
        | central_box_backup
        | far_box_runner
        | cutback_runner
    )

    # 오버래핑 풀백: 내 쪽 측면에서 동료가 공을 잡고 나보다 앞서 있으면 풀백이 그 바깥을
    # 돌아 나간다. 위치장은 조건부 평균이라 이 '한 명이 갑자기 라인을 타고 넘어가는' 사건을
    # 만들지 못한다 — 축구를 보고 있다는 느낌의 상당 부분이 이런 국소 패턴에서 온다.
    my_side = jnp.sign(c.home_att[:, 1] + DIV_EPS)
    is_fullback = (c.anchor_slot == 0) | (c.anchor_slot == 1)
    carrier_side_ok = (
        (jnp.sign(carrier_field[:, 1] + DIV_EPS) == my_side)
        & (jnp.abs(carrier_field[:, 1]) > 9.0)
    )
    overlap = (
        attacking
        & is_fullback
        & (~am_nearest_mate)
        & (~box_runner)
        & carrier_side_ok
        & (carrier_field[:, 0] > my_field[:, 0] - 3.0)
        & (carrier_field[:, 0] > policy.overlap_start_x)
    )
    overlap_target = jnp.stack([
        jnp.minimum(carrier_field[:, 0] + policy.overlap_run_ahead, c.hx - 5.0),
        my_side * (c.hy - 3.0),
    ], axis=1)
    overlap_target = overlap_target.at[:, 0].set(
        jnp.minimum(overlap_target[:, 0], onside_line_x)
    )
    off_target = jnp.where(overlap[:, None], overlap_target, off_target)
    # 서포트: 대형 유지를 위해 '캐리어 최근접 동료 상위 3명만' 지정 슬롯(좌/우/후방)으로 —
    # 거리 환대(7~16m) 방식은 조건 맞는 전원이 ±11m 두 슬롯에 수렴해 겹침·대형 붕괴를 만들었다.
    # 각자 자기 obs에서 같은 팀 거리를 복원하되, 같은 거리에서는 전역 선수 슬롯으로
    # 순서를 고정한다. 캐리어는 후보에서 명시적으로 제외한다.
    d_carrier = jnp.linalg.norm(my_field - carrier_field, axis=1)
    d_carrier_o = jnp.linalg.norm(others_field - carrier_field[:, None, :], axis=2)
    support_other_eligible = (
        mate
        & active_o
        & (c.others_idx != carrier_slot[:, None])
    )
    srank = _stable_lower_rank(
        d_carrier,
        d_carrier_o,
        self_active & (~am_nearest_mate),
        support_other_eligible,
        c.others_idx,
    )
    penetration_runner = (
        attacking
        & (c.roles == ROLE_FORWARD)
        & (my_field[:, 0] > carrier_field[:, 0] + 3.0)
    ) | box_runner
    is_support = ((~am_nearest_mate) & (~hold_wide) & (~is_gk) & (~penetration_runner)
                  & (~overlap)
                  & (d_carrier < 18.0)
                  & (srank < policy.support_count))
    # DFL의 캐리어 최근접 1/2/3번째 동료 거리 중앙값(10.2/14.2/17.7m)에 맞춘
    # 비대칭 삼각형. 전방 두 점을 같은 반경에 두지 않아 수비 한 줄 뒤에 겹치지 않는다.
    support_progression_blend = _style_support_progression_blend(
        direct,
        policy,
    )
    slot = _combination_support_slot(
        srank,
        policy.support_scale,
        support_progression_blend,
    )
    support_pos = carrier_field + slot
    # [지표] 고정 삼각형 주변의 9개 점 가운데 열린 패스 레인·슛 연계·동료 간격·현재
    # 이동 관성을 함께 최대화한다. 실측 수신자의 패스 전 1초 이동은 전진 일변도가 아니라
    # 횡/대각선 비중이 크므로 좌우 후보를 대칭으로 두고, 현재 속도를 끊지 않는 후보만 살짝
    # 우대한다. 이 목표 자체가 곧 pass_target의 속도 선점으로 들어가 '움직여 만든 공간'에
    # 실제로 공을 넣게 된다.
    cand_off = jnp.array([
        [0., 0.], [3., 0.], [-3., 0.], [0., 3.], [0., -3.],
        [3., 3.], [3., -3.], [-3., 3.], [-3., -3.],
    ])
    cand = support_pos[:, None, :] + cand_off[None, :, :]                       # (N,K,2)
    cand = jnp.stack(
        [
            jnp.minimum(
                jnp.clip(cand[:, :, 0], -c.hx + 2.0, c.hx - 2.0),
                onside_line_x[:, None] - 0.25,
            ),
            jnp.clip(cand[:, :, 1], -c.hy + 2.0, c.hy - 2.0),
        ],
        axis=-1,
    )
    op_c = T.openness(cand, others_field, o_vel, opp) / 12.0                    # (N,K) 공간
    support_pass_distance = jnp.linalg.norm(
        cand - carrier_field[:, None, :], axis=2
    )
    support_launch_speed = jnp.interp(
        support_pass_distance, c.drive_R, c.drive_v
    )
    support_ball_speed = 0.5 * (
        support_launch_speed + policy.drive_arrive_speed_mps
    )
    lane_c = T.lane_completion(
        carrier_field,
        cand,
        others_field,
        o_vel,
        opp,
        v_ball=support_ball_speed[:, :, None],
        r_int=policy.pass_lane_intercept_radius,
        react=policy.pass_lane_reaction_s,
        opponent_vmax=receiver_vmax,
        defender_accel=policy.pass_lane_defender_acceleration_mps2,
    )                                                               # (N,K) 캐리어→후보 레인
    pv_c = T.pitch_value(cand, c.hx, c.hy)                                      # (N,K) 전진위협
    xg_c = T.shot_xg(cand, others_field, opp, o_gk, c.hx, c.goal_w)
    support_heading = jnp.sum(
        _unit(cand - my_field[:, None, :])
        * my_velocity_direction[:, None, :],
        axis=2,
    ) * jnp.clip(my_speed[:, None] / 3.0, 0.0, 1.0)
    cand_mate_distance = jnp.linalg.norm(
        cand[:, :, None, :] - others_field[:, None, :, :], axis=3
    )
    nearest_mate_c = jnp.min(
        jnp.where(mate[:, None, :], cand_mate_distance, jnp.inf), axis=2
    )
    spacing_c = jnp.clip(nearest_mate_c / 8.0, 0.0, 1.0)
    # 위협 우선 가중은 유지하되 openness만 좇아 파이널서드에서 후퇴하지 않게 xG와
    # pitch value를 함께 쓴다. spacing은 동일 좌표로 여러 지원자가 수렴하는 것을 막는다.
    support_score = (
        0.60 * op_c
        + 0.90 * lane_c
        + 0.95 * pv_c
        + 0.70 * xg_c
        + policy.support_motion_alignment_gain * support_heading
        + 0.22 * spacing_c
    )
    best_c = jnp.argmax(support_score, axis=1)
    support_ref = jnp.take_along_axis(cand, best_c[:, None, None], axis=1)[:, 0, :]
    support_pos = 0.45 * support_pos + 0.55 * support_ref
    support_pos = support_pos.at[:, 0].set(
        jnp.where(
            attacking,
            jnp.minimum(support_pos[:, 0], onside_line_x - 0.25),
            support_pos[:, 0],
        )
    )
    off_target = jnp.where(is_support[:, None], support_pos, off_target)
    off_target = jnp.where(box_runner[:, None], box_target, off_target)

    # 패스 비행 중 수신 주자: "현재 공에 가장 가까운 선수"가 아니라, 실제 장거리 백스핀
    # 크로스 체공시간까지의 공 궤적에
    # 가장 먼저 도달 가능한 같은 팀 필드 선수를 고른다. 이 오버라이드가 없으면 킥 직후에도 원 패서가
    # 잠시 최근접이라 자기 패스를 쫓고, 진짜 수신자는 대형 목표에 머물러 킬패스가 죽는다.
    receive_times = _receive_prediction_times(policy)
    roll_decel = jnp.interp(ball_xy_speed, c.roll_v_knots, c.roll_d_knots)
    stop_distance = ball_xy_speed ** 2 / (2.0 * roll_decel + DIV_EPS)
    roll_distance = jnp.clip(
        ball_xy_speed[:, None] * receive_times[None, :]
        - 0.5 * roll_decel[:, None] * receive_times[None, :] ** 2,
        0.0,
        stop_distance[:, None],
    )
    roll_future = (
        ball_field[:, None, :]
        + incoming_direction[:, None, :] * roll_distance[:, :, None]
    )
    # quadratic drag dv/dt=-c|v|v의 수평 근사 해. 최종 낙하 목표와 같은
    # helper를 쓴다. 선형 외삽은 25 m/s·2.5 s 롱볼을 약 20 m 지나친다.
    air_distance = _air_travel_distance(
        ball_xy_speed[:, None], receive_times[None, :], c.ball_drag
    )
    air_future = (
        ball_field[:, None, :]
        + incoming_direction[:, None, :] * air_distance[:, :, None]
    )
    airborne = ball_z > c.r_ball + 0.08
    future_ball = jnp.where(
        airborne[:, None, None], air_future, roll_future
    )
    future_ball = jnp.stack(
        [
            jnp.clip(future_ball[:, :, 0], -c.hx + 0.5, c.hx - 0.5),
            jnp.clip(future_ball[:, :, 1], -c.hy + 0.5, c.hy - 0.5),
        ],
        axis=-1,
    )
    player_field = jnp.concatenate([my_field[:, None, :], others_field], axis=1)
    player_is_gk = jnp.concatenate([is_gk[:, None], o_gk], axis=1)
    player_team_base = jnp.concatenate(
        [self_active[:, None], mate & active_o], axis=1
    ) & (~player_is_gk)
    player_retouch = jnp.concatenate(
        [retouch[:, None], o_is_taker], axis=1
    )
    player_team_base = player_team_base & (~player_retouch)
    opponent_team_base = jnp.concatenate(
        [jnp.zeros_like(self_active[:, None]), opp & active_o], axis=1
    ) & (~player_is_gk) & (~player_retouch)
    ball_travel_dir = incoming_direction
    player_along_ball = jnp.sum(
        (player_field - ball_field[:, None, :]) * ball_travel_dir[:, None, :], axis=2
    )
    ahead_of_ball = (
        player_along_ball > policy.pass_receive_ahead_margin
    )
    # 진행방향 앞선 선수 제한은 킥 직후 원 패서의 15Hz 재접촉을 막는다. 그러나 코너
    # 클리어처럼 공이 모든 동료를 지나 자기 진영으로 흐르면 앞선 후보가 0명이 되어 팀 전체가
    # 추격을 포기했다. 앞선 후보가 하나라도 있을 때만 제한하고, 모두 지나친 뒤에는 retouch
    # 키커를 제외한 팀 전체 ETA 경쟁을 다시 연다.
    player_team_ok = _trajectory_receiver_team_mask(
        player_team_base,
        own_live_pass,
        ahead_of_ball,
    )
    # 같은 관측 안에서 상대 서비스 팀의 수신 경쟁도 동일한 계약으로 복원한다. 수비자가
    # 자기 팀 ETA만 보면 공격 수신자보다 1초 늦어도 같은 궤적점에 달려들므로, 양 팀 역할을
    # 나누려면 서비스 팀의 최선 비용과 예상 접촉점이 함께 필요하다.
    opponent_team_ok = _trajectory_receiver_team_mask(
        opponent_team_base,
        opponent_live_pass,
        ahead_of_ball,
    )
    self_vmax = _decision_speed_for_endurance(
        effective_speed_cap(
            self_tok[:, c.p_vmax[0]] * c.n_pvel,
            self_tok[:, c.p_stamina_long[0]],
            self_tok[:, c.p_stamina_short[0]],
            long_floor=c.long_stamina_vmax_floor,
            short_floor=c.short_stamina_vmax_floor,
            short_knee=c.short_stamina_headroom_knee,
        ),
        self_tok[:, c.p_endurance[0]],
    )
    o_vmax = _decision_speed_for_endurance(
        effective_speed_cap(
            others[:, :, c.p_vmax[0]] * c.n_pvel,
            others[:, :, c.p_stamina_long[0]],
            others[:, :, c.p_stamina_short[0]],
            long_floor=c.long_stamina_vmax_floor,
            short_floor=c.short_stamina_vmax_floor,
            short_knee=c.short_stamina_headroom_knee,
        ),
        others[:, :, c.p_endurance[0]],
    )
    player_vmax = jnp.concatenate([self_vmax[:, None], o_vmax], axis=1)
    # 미래 수신점의 수직 자격은 env._in_reach와 같이 선수별
    # reach_z + r_ball이 소유한다. 속도·높이 결합선(reach_blockable)은
    # '빠른 공을 의도적으로 막을 수 있나'이지 선수의 물리적 최대
    # 높이가 아니다. 두 게이트를 함께 걸지 않으면 낙하지점에 있던
    # 선수가 4~5 m 높이의 조기 교차점을 잡으러 거꾸로 달린다.
    player_reach_z = jnp.concatenate(
        [self_reach_z[:, None], o_reach_z], axis=1
    )
    intercept_d = jnp.linalg.norm(
        player_field[:, :, None, :] - future_ball[:, None, :, :], axis=-1
    )
    arrival_t = intercept_d / (player_vmax[:, :, None] + DIV_EPS)
    # env의 reach 속도 게이트(v34)와 **같은 선**을 예측에도 건다. 빠르거나 높은 공은 몸에
    # 직접 부딪히지 않는 한 env가 경합 자체를 허용하지 않으므로, 도착시간만 보고 주자를
    # 붙이면 잡을 수 없는 공에 사람을 보내는 셈이 된다. 위치 예측과 같은 감속 모형을 써야
    # 시점별 판정이 서로 어긋나지 않는다 — 구름은 선형 감속, 체공은 이차 항력의 해다.
    roll_speed = jnp.maximum(
        ball_xy_speed[:, None] - roll_decel[:, None] * receive_times[None, :], 0.0
    )
    air_speed = ball_xy_speed[:, None] / (
        1.0 + c.ball_drag * ball_xy_speed[:, None] * receive_times[None, :]
    )
    future_speed = jnp.where(airborne[:, None], air_speed, roll_speed)
    future_z = jnp.where(
        airborne[:, None],
        jnp.maximum(
            ball_z[:, None]
            + ball_vel[:, DIM_Z][:, None] * receive_times[None, :]
            - 0.5 * c.gravity * receive_times[None, :] ** 2,
            c.r_ball,
        ),
        c.r_ball,
    )
    catchable = reach_blockable(
        future_speed, future_z, c.reach_block_limit, c.reach_height_penalty
    )
    height_reachable = (
        future_z[:, None, :]
        <= player_reach_z[:, :, None] + c.r_ball
    )
    feasible = (
        (arrival_t <= receive_times[None, None, :] + 0.10)
        & catchable[:, None, :]
        & height_reachable
    )
    base_intercept_cost = jnp.where(
        feasible,
        receive_times[None, None, :] + 0.015 * intercept_d,
        8.0 + arrival_t - receive_times[None, None, :],
    )
    intercept_cost = jnp.where(
        player_team_ok[:, :, None], base_intercept_cost, jnp.inf
    )
    opponent_intercept_cost = jnp.where(
        opponent_team_ok[:, :, None], base_intercept_cost, jnp.inf
    )
    best_time_idx = jnp.argmin(intercept_cost, axis=2)
    best_player_cost = jnp.min(intercept_cost, axis=2)
    player_slot = jnp.concatenate(
        [
            jnp.arange(N, dtype=jnp.int32)[:, None],
            c.others_idx,
        ],
        axis=1,
    )
    receive_j = _stable_argmin_index(
        best_player_cost, player_team_ok, player_slot
    )
    receive_time_j = jnp.take_along_axis(
        best_time_idx, receive_j[:, None], axis=1
    )[:, 0]
    receive_target = jnp.take_along_axis(
        future_ball, receive_time_j[:, None, None], axis=1
    )[:, 0, :]
    opponent_best_time_idx = jnp.argmin(opponent_intercept_cost, axis=2)
    opponent_best_player_cost = jnp.min(opponent_intercept_cost, axis=2)
    opponent_receive_j = _stable_argmin_index(
        opponent_best_player_cost, opponent_team_ok, player_slot
    )
    opponent_receive_time_j = jnp.take_along_axis(
        opponent_best_time_idx, opponent_receive_j[:, None], axis=1
    )[:, 0]
    opponent_receive_target = jnp.take_along_axis(
        future_ball, opponent_receive_time_j[:, None, None], axis=1
    )[:, 0, :]
    # 각 관측에서 self는 0번 열이다. 값이 같은 수신 후보는 전역 선수 슬롯으로 순서를
    # 고정해야 모든 관측자가 한 명의 1순위와 한 명의 2순위에 합의한다.
    self_trajectory_cost = best_player_cost[:, 0]
    trajectory_rank = _stable_lower_rank(
        self_trajectory_cost,
        best_player_cost[:, 1:],
        player_team_ok[:, 0] & self_active & (~is_gk),
        player_team_ok[:, 1:],
        c.others_idx,
    )
    trajectory_runner = trajectory_rank == 0
    trajectory_cover_runner = (
        trajectory_rank == 1
    )
    receive_runner = own_live_pass & trajectory_runner
    # ``receive_runner``는 '누가 공을 향해 **달릴** 것인가'의 배정이지 '발 앞의 공을 찰 수
    # 있는가'가 아니다. 그 배정은 공 진행축 투영(``ahead_of_ball``)에서 나오는데, 수신자가
    # 실제로 공에 도달하면 투영이 0으로 수렴해 **도착과 동시에 자격을 잃는다** — 방향·파워를
    # 다 계산해 놓고 want_kick=False가 나온다.
    #
    # 기하 추정은 반대 방향으로도 틀렸다. 막으려던 원 패서의 15 Hz 재접촉을 실제로는 막지
    # 못했다(스냅샷 실측: 1프레임 간격 재접촉 38건/120초, 동일 선수 최장 10회 연속).
    # 이제 관측이 '내가 마지막으로 찼는가'를 직접 준다(``last_touch_relation == +2``).
    # 추정 대신 그 사실로 **원 패서만** 배제한다.
    receive_or_arrived = (receive_runner | in_reach) & (~i_touched_last)
    own_best_trajectory_cost = jnp.min(best_player_cost, axis=1)
    opponent_best_trajectory_cost = jnp.min(
        opponent_best_player_cost, axis=1
    )
    aerial_live = airborne & ball_alive & (~restart_active)
    # 서비스 팀의 1순위는 실제 수신점으로 간다. 상대 1순위는 ETA가 duel 창 안일 때만
    # 같은 지점에서 contest action을 제출한다. 명백히 늦으면 접촉을 시도하지 않고 아래의
    # 골사이드 세컨드볼 역할로 전환한다. 실제 50:50 공은 기존 env contest가 계속 판정한다.
    aerial_runner, aerial_defender_cover = _aerial_runner_roles(
        aerial_live,
        trajectory_runner,
        own_live_pass,
        opponent_live_pass,
        own_best_trajectory_cost,
        opponent_best_trajectory_cost,
        policy.aerial_duel_eta_window_s,
    )
    drib_target = ball_field                                        # 캐리어는 공에 직접 호밍(오버런 방지)
    # 패스/크로스/슛/클리어를 릴리스하는 같은 control frame에 캐리어가 계속 공으로
    # 달려들면 contact lock이 풀린 직후 자기 패스를 먼저 재접촉한다. 킥을 고른 순간부터
    # 오프볼 목표로 전환하고, 실제 드리블 폴백만 공 호밍을 유지한다.
    carrier_releasing_ball = carrier_pass | crossing | shoot | carrier_clear
    chase_as_carrier = (
        am_nearest_mate
        & (~own_live_pass)
        & (~carrier_releasing_ball)
    )
    attack_target = jnp.where(chase_as_carrier[:, None], drib_target, off_target)
    attack_target = jnp.where(receive_runner[:, None], receive_target, attack_target)
    carrier_pow = jnp.where(
        ball_dist < 2.5,
        policy.carrier_control_power,
        policy.carrier_chase_power,
    )                                                               # 근접 감속(오버런 억제)
    # 오프더볼은 '목표까지 남은 거리'로 파워를 정한다(도착 감속). 고정 파워 0.7은 대형이
    # 이미 잡힌 프레임에도 전원을 달리게 해 실측 속도 분포(중앙값 1.97 m/s)를 깨뜨렸다.
    d_off = jnp.linalg.norm(off_target - my_field, axis=1)
    cruise_pow = _arrival_power(d_off, policy)
    attack_pow = jnp.where(chase_as_carrier, carrier_pow,
                 jnp.where(
                     is_support,
                     jnp.maximum(cruise_pow, policy.support_run_power),
                     cruise_pow,
                 ))
    attack_pow = jnp.where(
        box_runner, jnp.maximum(cruise_pow, policy.box_run_power), attack_pow
    )
    attack_pow = jnp.where(receive_runner, policy.receive_run_power, attack_pow)

    # ── 국면 C: 수비(상대 점유) — 능동 압박 + 맨마킹 + 컴팩트 블록(골라인 붕괴 방지) ──
    # 최근접 수비수가 볼로 강압박(공격성/위험지역이면 2차 압박 가세), 비압박 수비수는 침투한
    # 위협 상대를 골사이드 마킹, 나머지는 공 높이의 컴팩트 라인 유지. '전원 골대 앞 뭉침' 제거.
    ball_deep = ball_adv < -c.hx / 3.0
    # 압박 완화: 상시 3인 스웜은 미드필드에서 공을 즉시 뺏어 공격이 파이널서드까지 전개되지 못하게
    # 한다(측정: 파이널서드 공격 터치 ~0). 1차만 상시 압박하고, 2차는 공격적 팀·위험지역, 3차는 드물게 —
    # 캐리어에 빌드업·전진패스 시간을 줘 공격이 전개되게(→슛·골 기회 생성).
    # 압박은 '조금만' 완화 — 높은 압박이 턴오버로 공격 기회를 만들어(게겐프레싱), 너무 빼면 양 팀이
    # 로우블록으로 앉아 침투가 죽는다(측정: 완전 완화 시 슛 6→0). 2차 압박 문턱만 살짝 올려(0.5→0.65)
    # 소극적 팀은 미드필드 더블프레스를 줄이되, 공격적 팀·위험지역은 유지.
    # 실제 직접 압박 위치의 스타일별 분위수로 개시선을 만든다. balanced는 p75(+12.2m),
    # gegenpress는 p95 부근, park-the-bus는 중앙값 부근이다. 선 밖의 느린 통제 빌드업에는
    # 최근접 선수도 달려들지 않고 위치장 블록을 유지해 안전한 소유 구간을 허용한다.
    fast_ball = ball_speed > policy.press_loose_ball_speed
    press_line_x = _press_engagement_line(line_h, aggr, policy)
    # 개시선은 새 압박을 시작할 위치이지 이미 붙은 압박을 순간 해제할 경계가 아니다.
    # 팀 최근접이 DFL 중앙값(4.8m) 안이면 개시선 밖에서도 1차+커버 구조를 유지한다.
    nearest_team_ball = jnp.minimum(
        ball_dist, nearest_mate_ball
    )
    active_retain_distance = jnp.where(
        just_lost,
        policy.counterpress_retain_distance,
        policy.press_retain_distance,
    )
    press_engaged = (
        (ball_adv <= press_line_x)
        | fast_ball
        | (nearest_team_ball <= active_retain_distance)
    )
    press1 = defending & press_engaged & (team_rank < 0.5)
    press2 = (defending & press_engaged & (team_rank >= 0.5) & (team_rank < 1.5)
              & ((aggr > 0.5) | ball_deep) & (~just_lost))
    press3 = (defending & press_engaged & (team_rank >= 1.5) & (team_rank < 2.5)
              & (aggr > 0.65) & ball_deep & (~just_lost))
    press = press1 | press2 | press3
    # balanced/지역 압박에서도 2순위가 사라지지 않게 하되, 직접 발을 넣는 press2와는
    # 분리한다. 커버는 패스 경로와 튀어나온 공을 준비할 뿐 아래 do_kick에 들어가지 않는다.
    press_cover = (
        defending
        & press_engaged
        & (team_rank >= 0.5)
        & (team_rank < 1.5)
        & (~press2)
        & (~is_gk)
    )
    counterpress_rest = (
        just_lost
        & press_engaged
        & (team_rank >= 1.5)
        & (team_rank < 2.5)
        & (~is_gk)
    )
    # 2·3차는 공 진행방향을 살짝 리드해 패스레인 차단·커버한다.
    # ── 실제 challenge 의도 ────────────────────────────────────────────────
    # ``poss``는 패스가 이동하는 동안에도 마지막 소유팀에 래치된다. 따라서 상대팀 공 최근접
    # 선수가 carry reach 안에서 실제로 공을 제어하는지 먼저 복원하고, 캐리어 태클과 이동 공
    # 인터셉트를 서로 다른 hazard로 다룬다. ``ball_deep | fast_ball`` 같은 bool 하나로 묶으면
    # 중원에서 2.2m 조키만 하다가 문턱을 넘는 순간 0.9m·매 frame 발 넣기로 바뀌어 gate가 burst한다.
    opponent_ball_distance = jnp.where(
        opp & active_o, d_ball_o, jnp.inf
    )
    opponent_carrier_j = jnp.argmin(opponent_ball_distance, axis=1)
    has_opponent_carrier = jnp.any(opp & active_o, axis=1)
    opponent_carrier_distance = jnp.take_along_axis(
        opponent_ball_distance, opponent_carrier_j[:, None], axis=1
    )[:, 0]
    opponent_carrier_pos = jnp.take_along_axis(
        others_field, opponent_carrier_j[:, None, None], axis=1
    )[:, 0, :]
    opponent_carrier_vel = jnp.take_along_axis(
        o_vel, opponent_carrier_j[:, None, None], axis=1
    )[:, 0, :]
    opponent_carrier_reach_z = jnp.take_along_axis(
        o_reach_z, opponent_carrier_j[:, None], axis=1
    )[:, 0]
    carrier_control_radius = c.reach_xy_carry + c.r_ball
    opponent_carrier_controls = (
        defending
        & has_opponent_carrier
        & (opponent_carrier_distance <= carrier_control_radius)
        & (ball_z <= opponent_carrier_reach_z + c.r_ball)
    )

    control_edge = jnp.clip(
        (opponent_carrier_distance - 0.45)
        / jnp.maximum(carrier_control_radius - 0.45, DIV_EPS),
        0.0,
        1.0,
    )
    ball_carrier_relative_speed = jnp.linalg.norm(
        ball_vel[:, :2] - opponent_carrier_vel, axis=1
    )
    carrier_instability = jnp.maximum(
        control_edge,
        jnp.clip(
            ball_carrier_relative_speed
            / policy.tackle_instability_speed_scale_mps,
            0.0,
            1.0,
        ),
    )
    to_carrier = _unit(opponent_carrier_pos - my_field)
    closing_speed = jnp.sum(
        (my_vel - opponent_carrier_vel) * to_carrier, axis=1
    )
    closing_quality = jnp.clip(
        closing_speed / policy.tackle_closing_speed_scale_mps, 0.0, 1.0
    )
    carrier_to_defender = _unit(my_field - opponent_carrier_pos)
    carrier_to_own_goal = _unit(own_goal[None, :] - opponent_carrier_pos)
    goal_side_quality = jnp.clip(
        jnp.sum(carrier_to_defender * carrier_to_own_goal, axis=1),
        0.0,
        1.0,
    )
    approach_quality = 0.65 * closing_quality + 0.35 * goal_side_quality
    danger = jnp.clip(-ball_adv / c.hx, 0.0, 1.0)
    challenge_contact_radius = c.reach_xy_challenge + c.r_ball
    contact_proximity = jnp.clip(
        (challenge_contact_radius - ball_dist)
        / jnp.maximum(
            challenge_contact_radius - policy.press_tackle_distance,
            DIV_EPS,
        ),
        0.0,
        1.0,
    )
    tackle_probability, tackle_hazard = _tackle_attempt_probability(
        danger,
        carrier_instability,
        approach_quality,
        contact_proximity,
        just_lost.astype(jnp.float32),
        aggr,
        c.control_dt,
        policy,
    )
    interception_probability = _interception_attempt_probability(
        ball_speed / policy.press_loose_ball_speed,
        approach_quality,
        c.control_dt,
        policy,
    )
    challenge_draw = jax.random.uniform(k_challenge, (N,))
    carrier_tackle_intent = (
        opponent_carrier_controls & (challenge_draw < tackle_probability)
    )
    travelling_intercept_intent = (
        defending
        & (~opponent_carrier_controls)
        & (challenge_draw < interception_probability)
    )
    commit_tackle = (
        (~booked)
        & (~is_gk)
        & (carrier_tackle_intent | travelling_intercept_intent)
    )

    # 접근 거리도 같은 연속 위험도로 보간한다. 캐리어가 안정적이면 2.2m 조키에 가깝고,
    # 위험·긴 터치·좋은 접근각이 겹칠수록 0.9m 접촉권으로 들어간다. 이동 공은 다음
    # frame의 확률 draw를 놓치지 않도록 접촉 거리까지 추적하지만 실제 발은 위 hazard가 정한다.
    tackle_entry_strength = jnp.clip(
        tackle_hazard / policy.tackle_entry_hazard_scale_per_s,
        0.0,
        1.0,
    )
    tackle_entry_strength = jnp.where(
        opponent_carrier_controls, tackle_entry_strength, 1.0
    )
    tackle_entry_strength = jnp.where(
        booked | is_gk, 0.0, tackle_entry_strength
    )
    press_gap = (
        policy.press_contain_distance
        - tackle_entry_strength
        * (policy.press_contain_distance - policy.press_tackle_distance)
    )
    own_goal_direction = _unit(own_goal[None, :] - ball_field)
    press_target = jnp.where(press1[:, None],
                             ball_field + own_goal_direction * press_gap[:, None],
                             ball_field + ball_vel[:, :2] * 0.3 + own_goal_direction * 3.0)
    normal_press_cover_target = (
        ball_field
        + ball_vel[:, :2] * 0.18
        + own_goal_direction
        * policy.press_cover_distance
    )
    # 전환 시 두 번째 선수는 캐리어에게 함께 달려들지 않고 가장 가까운 상대 출구의
    # 패스선을 먼저 닫는다. 세 번째 선수는 공의 골사이드에 남아 첫 압박이 벗겨졌을 때의
    # 전진 패스를 회수한다. DFL의 손실 직후 속도는 전원 상승하지 않았으므로 역할만 바꾸고
    # 전체 주행 파워를 올리지 않는다.
    outlet_candidate = opp & active_o & (~o_gk) & (d_ball_o > 2.0)
    outlet_j = jnp.argmin(
        jnp.where(outlet_candidate, d_ball_o, jnp.inf), axis=1
    )
    outlet_pos = jnp.take_along_axis(
        others_field, outlet_j[:, None, None], axis=1
    )[:, 0, :]
    outlet_distance = jnp.linalg.norm(outlet_pos - ball_field, axis=1)
    lane_step = jnp.minimum(
        policy.counterpress_cover_distance,
        jnp.maximum(2.5, 0.42 * outlet_distance),
    )
    counterpress_cover_target = (
        ball_field
        + _unit(outlet_pos - ball_field) * lane_step[:, None]
        + own_goal_direction * 1.5
    )
    press_cover_target = jnp.where(
        just_lost[:, None], counterpress_cover_target, normal_press_cover_target
    )
    counterpress_rest_target = (
        ball_field
        + own_goal_direction
        * policy.counterpress_rest_distance
    )
    counterpress_rest_target = jnp.stack(
        [
            jnp.clip(counterpress_rest_target[:, 0], -c.hx + 2.0, c.hx - 3.0),
            jnp.clip(counterpress_rest_target[:, 1], -c.hy + 2.0, c.hy - 2.0),
        ],
        axis=1,
    )

    # [지표·선점] 위협 상대 = 우리 골쪽 침투 깊이 + 골로 달리는 속도 가점. 상대를 0.25s
    # 선점해 '지금 위치'가 아니라 '곧 있을 위치'를 마킹한다. 오픈필드에서는 최고 위협
    # 한 명만 전담하지만, 박스 부근에서는 압박·커버 역할을 뺀 수비수에게 최대 세 러너를
    # 서로 다르게 배정한다. 이 구분이 없으면 중앙·니어·파포스트가 모두 같은 마커를 향한다.
    opp_fut = others_field + o_vel * 0.25                                       # 선점 위치
    toward_own = jnp.clip(-o_vel[:, :, 0], 0.0, 8.0)                            # 우리 골(-x) 방향 이동속도
    opp_threat = jnp.where(
        opp & active_o & (~o_gk),
        -opp_fut[:, :, 0]
        + 0.4 * toward_own
        + 0.08
        * jnp.clip(
            c.pen_hw + policy.box_mark_runner_margin
            - jnp.abs(opp_fut[:, :, 1]),
            0.0,
            c.pen_hw + policy.box_mark_runner_margin,
        ),
        -jnp.inf,
    )
    has_threat = jnp.any(opp & (~o_gk), axis=1)
    mark_j = jnp.argmax(opp_threat, axis=1)
    mark_pos = jnp.take_along_axis(opp_fut, mark_j[:, None, None], axis=1)[:, 0, :]   # 선점 위치 마킹
    mark_target = mark_pos + _unit(own_goal[None, :] - mark_pos) * 2.0
    mark_goalside = (-mark_pos[:, 0]) > (-ball_adv - 3.0)
    # 마킹 클레임: 그 위협에 '내가 최근접 동료(GK 제외)'일 때만 마킹 — 전원이 동일 argmax
    # 타깃(최심 침투자)으로 수렴해 수비 라인이 통째로 무너지던 것을 1명 전담으로 제한.
    # 나머지 수비수는 컴팩트 라인(def_target)을 유지해 서로 간의 진영이 보존된다.
    d_mark_me = jnp.linalg.norm(mark_pos - my_field, axis=1)
    d_mark_mates = jnp.linalg.norm(others_field - mark_pos[:, None, :], axis=2)
    claim_mark, _ = _stable_nearest_player(
        d_mark_me,
        d_mark_mates,
        self_active & (~is_gk),
        mate & active_o & (~o_gk),
        c.others_idx,
    )
    ordinary_do_mark = (
        defending
        & (~press)
        & (~press_cover)
        & (~is_gk)
        & has_threat
        & mark_goalside
        & claim_mark
    )

    box_line_x = -c.hx + c.pen_len
    box_defense_live = (
        defending
        & (ball_adv <= box_line_x + policy.box_mark_ball_margin)
    )
    box_runner_candidate = (
        opp
        & active_o
        & (~o_gk)
        & (
            opp_fut[:, :, 0]
            <= box_line_x + policy.box_mark_runner_margin
        )
        & (
            jnp.abs(opp_fut[:, :, 1])
            <= c.pen_hw + policy.box_mark_runner_margin
        )
    )
    # 공에 직접 접근하는 1차·2차(공격적 깊은 수비는 3차까지), 그리고 전환 때의
    # rest defender를 러너 마킹 풀에서 제외한다. 팀 공거리 rank를 전 선수에 대해
    # 같은 방식으로 복원하므로 각 관측자가 서로 모순되지 않는 배정을 계산한다.
    team_player = jnp.concatenate(
        [
            (self_active & (~is_gk))[:, None],
            mate & active_o & (~o_gk),
        ],
        axis=1,
    )
    team_ball_distance = jnp.concatenate(
        [ball_dist[:, None], d_ball_o], axis=1
    )
    team_ball_distance = jnp.where(
        team_player, team_ball_distance, jnp.inf
    )
    team_ball_rank = _stable_lower_ranks(
        team_ball_distance,
        team_player,
        player_slot,
    )
    reserved_ball_roles = jnp.where(
        press_engaged,
        jnp.where(
            just_lost,
            3,
            jnp.where((aggr > 0.65) & ball_deep, 3, 2),
        ),
        1,
    )
    self_box_available = (
        self_active
        & (~is_gk)
        & (team_ball_rank[:, 0] >= reserved_ball_roles)
    )
    other_box_available = (
        mate
        & active_o
        & (~o_gk)
        & (team_ball_rank[:, 1:] >= reserved_ball_roles[:, None])
    )
    has_box_assignment, box_mark_target, _ = _box_mark_plan(
        my_field,
        others_field,
        opp_fut,
        opp_threat,
        box_runner_candidate,
        self_box_available,
        other_box_available,
        own_goal,
        policy.box_mark_max_runners,
        policy.box_mark_goal_side_distance,
        player_slot,
        c.others_idx,
    )
    box_do_mark = (
        box_defense_live
        & (~press)
        & (~press_cover)
        & has_box_assignment
    )
    do_mark = box_do_mark | (ordinary_do_mark & (~box_defense_live))
    mark_target = jnp.where(
        box_do_mark[:, None], box_mark_target, mark_target
    )

    # 수비 블록도 같은 실측 위치장을 쓴다 — ``attacking=False``인 슬롯은 anchor가 이미
    # defend 국면 행을 읽었다. 구판의 (홈 x·0.35 + line_x·0.78) / (홈 y·0.66 + ball_y·0.28)
    # 손튜닝 결합은 라인 높이도 폭도 실측 계수와 어긋났다(특히 공이 상대 진영일 때 x가 14m로
    # 잘려 블록이 따라 올라가지 못했다).
    def_target = jnp.stack([
        jnp.clip(anchor[:, 0] + line_shift, -c.hx + 2.0, c.hx - 4.0),
        jnp.clip(anchor[:, 1] * width_scale, -c.hy + 2.0, c.hy - 2.0),
    ], axis=1)
    def_target = jnp.where(do_mark[:, None], mark_target, def_target)
    def_target = jnp.where(
        counterpress_rest[:, None], counterpress_rest_target, def_target
    )
    def_target = jnp.where(
        press_cover[:, None], press_cover_target, def_target
    )
    def_target = jnp.where(press[:, None], press_target, def_target)
    d_def = jnp.linalg.norm(def_target - my_field, axis=1)
    # 압박·마킹만 전력이고, 블록을 유지하는 나머지는 같은 도착 감속을 쓴다.
    def_pow = jnp.where(
        press,
        policy.press_run_power,
        jnp.where(
            press_cover,
            policy.press_cover_run_power,
            jnp.where(
                counterpress_rest,
                policy.press_cover_run_power,
                jnp.where(
                    do_mark,
                    policy.mark_run_power,
                    _arrival_power(d_def, policy),
                ),
            ),
        ),
    )

    # ── 국면 루즈볼(중립 poss=-1): 팀별 궤적 ETA 1순위 + 세컨드볼 커버 ─────────
    # [2026-08-13] 소유 전이가 K리그 정합(비통제 접촉→중립)으로 바뀌며 중립이 실제
    # 국면이 됐다. 현재 공 거리만으로 1명을 고르면 빠르게 굴러가거나 튀는 공의 뒤쪽 선수가
    # 선택된다. 위의 stamina-aware 궤적 ETA를 그대로 써 양 팀 1순위는 예상 접촉점으로 보내고,
    # 2순위는 그 지점의 골사이드에서 세컨드볼을 준비한다. 2순위는 접촉 action을 내지 않는다.
    loose_primary = (
        loose_phase
        & ball_alive
        & (~restart_active)
        & trajectory_runner
    )
    loose_cover = (
        loose_phase
        & ball_alive
        & (~restart_active)
        & trajectory_cover_runner
    )
    loose_rest = (
        own_recent_loose
        & (trajectory_rank >= 1.5)
        & (trajectory_rank < 2.5)
        & player_team_ok[:, 0]
        & (~is_gk)
    )
    receive_own_goal_direction = _unit(own_goal[None, :] - receive_target)
    loose_cover_target = (
        receive_target
        + receive_own_goal_direction
        * policy.loose_cover_distance
    )
    loose_rest_target = (
        receive_target
        + receive_own_goal_direction
        * policy.counterpress_rest_distance
    )
    loose_target = jnp.where(
        loose_rest[:, None], loose_rest_target, off_target
    )
    loose_target = jnp.where(
        loose_cover[:, None], loose_cover_target, loose_target
    )
    loose_target = jnp.where(
        loose_primary[:, None], receive_target, loose_target
    )
    loose_pow = jnp.where(
        loose_primary,
        policy.loose_run_power,
        jnp.where(
            loose_cover,
            policy.loose_cover_run_power,
            jnp.where(
                loose_rest,
                policy.loose_cover_run_power,
                _arrival_power(d_off, policy),
            ),
        ),
    )

    # ── 국면 통합(오픈플레이) ───────────────────────────────────────────────
    target = jnp.where(attacking[:, None], attack_target,
             jnp.where(defending[:, None], def_target, loose_target))
    move_pow = jnp.where(attacking, attack_pow, jnp.where(defending, def_pow, loose_pow))

    # ── 공중볼 역할: 서비스 팀 수신자와 ETA가 맞는 수비 경합자만 실제 궤적점을
    # 추적한다. 늦은 수비 1순위는 같은 좌표로 복제 이동하지 않고 상대 수신점의
    # 골사이드에서 세컨드볼을 준비하며 접촉 action도 제출하지 않는다.
    aerial_defender_cover_target = (
        opponent_receive_target
        + _unit(own_goal[None, :] - opponent_receive_target)
        * policy.aerial_defender_cover_distance
    )
    aerial_go = aerial_runner | aerial_defender_cover
    aerial_target = jnp.where(
        aerial_defender_cover[:, None],
        aerial_defender_cover_target,
        receive_target,
    )
    target = jnp.where(aerial_go[:, None], aerial_target, target)
    aerial_move_power = jnp.where(
        aerial_defender_cover,
        policy.aerial_cover_run_power,
        policy.aerial_run_power,
    )
    move_pow = jnp.where(aerial_go, aerial_move_power, move_pow)

    kick_dir = jnp.where(am_nearest_mate[:, None] & attacking[:, None], c_kick_dir,
                         forward_ball)
    # 일반 비캐리어 접촉의 기본 fallback. 지상 challenge는 아래에서 통제/출구/긴급
    # 걷어내기로 전부 덮어쓰며, 공중 경합은 뒤의 aerial contract가 최종 의도를 정한다.
    kick_pow = jnp.where(
        am_nearest_mate & attacking,
        c_kick_pow,
        loft_pow_of(jnp.clip(ball_d_goal, 18.0, 40.0)),
    )
    # 루즈볼 통제 터치는 클리어가 아니라 다음 행동을 위한 소프트 터치다. 구 상수 0.35는
    # 기본 f2b_speed_max에서 11.9 m/s여서 주석과 반대로 강한 패스였고, 아래 지상 트랩의
    # 3 m/s와도 물리 의미가 갈렸다. 같은 m/s config를 공유해 공속 캡과 독립시킨다.
    ground_control_power = (
        policy.ground_control_touch_speed_mps / c.f2b_max
    )
    kick_pow = jnp.where(
        loose_primary, ground_control_power, kick_pow
    )
    launch01 = jnp.where(am_nearest_mate & attacking, c_launch, c.loft_launch01)
    spin_s = jnp.where(am_nearest_mate & attacking, c_spin_s, 0.0)
    spin_b = jnp.where(am_nearest_mate & attacking, c_spin_b, 0.0)

    challenge_kick = press1 & commit_tackle & in_reach & (~booked)
    challenge_safe_outlet = (
        has_challenge_outlet
        & (ball_speed <= policy.challenge_outlet_max_incoming_speed_mps)
    )
    (
        challenge_ground_control,
        challenge_ground_outlet,
        challenge_emergency_clearance,
    ) = _challenge_contact_modes(
        challenge_kick,
        airborne,
        challenge_safe_outlet,
        has_challenge_outlet & (~challenge_safe_outlet),
        ball_adv,
        press_self,
        c.hx,
        policy,
    )
    ground_challenge_action = (
        challenge_ground_control
        | challenge_ground_outlet
        | challenge_emergency_clearance
    )
    challenge_outlet_distance = (
        jnp.linalg.norm(challenge_outlet_target - ball_field, axis=1) + DIV_EPS
    )
    challenge_control_dir = _unit(escape_step - ball_field)
    challenge_clear_dir = forward_ball
    challenge_dir = jnp.where(
        challenge_ground_outlet[:, None],
        _unit(challenge_outlet_target - ball_field),
        jnp.where(
            challenge_emergency_clearance[:, None],
            challenge_clear_dir,
            challenge_control_dir,
        ),
    )
    challenge_control_power = (
        policy.challenge_control_touch_speed_mps / c.f2b_max
    )
    challenge_power = jnp.where(
        challenge_ground_outlet,
        drive_pow_of(challenge_outlet_distance),
        jnp.where(
            challenge_emergency_clearance,
            clear_pow,
            challenge_control_power,
        ),
    )
    challenge_launch = jnp.where(
        challenge_ground_outlet,
        c.drive_launch01,
        jnp.where(
            challenge_emergency_clearance, c.loft_launch01, 0.0
        ),
    )
    kick_dir = jnp.where(ground_challenge_action[:, None], challenge_dir, kick_dir)
    kick_pow = jnp.where(ground_challenge_action, challenge_power, kick_pow)
    launch01 = jnp.where(ground_challenge_action, challenge_launch, launch01)
    spin_s = jnp.where(ground_challenge_action, 0.0, spin_s)
    spin_b = jnp.where(ground_challenge_action, 0.0, spin_b)
    # 부킹된 선수는 압박 태클(공 다툼 킥)을 안 한다 — 파울→2차 경고→퇴장의 자충수 방지.
    # 루즈볼 도달 시 통제 터치(소프트 전진 터치) — 이겨야 중립이 소유로 전환된다.
    do_kick = ((am_nearest_mate & attacking & carrier_kick & in_reach
                & ((~own_live_pass) | receive_or_arrived))
               | challenge_kick
               | (loose_primary & in_reach & (~booked)
                  & ((~own_live_pass) | receive_or_arrived))) & f2b_avail

    # ── 국면 A: 세트피스 오버라이드 ─────────────────────────────────────────
    # 지정 키커는 obs의 self_is_taker로 직접 식별한다. 거리 순위 휴리스틱은 공 근처 동료를
    # 키커로 오인할 수 있으므로 쓰지 않는다. 비키커는 `_restart_policy_plan`이 환경의
    # 최소투영 규약과 같은 기하로 합법 대기점을 정하고, 다음 frame부터 스스로 그 위치를 지킨다.
    is_corner = rk[:, RK_CORNER] > 0.5
    is_gkk = rk[:, RK_GOALKICK] > 0.5
    is_throw = rk[:, RK_THROWIN] > 0.5
    is_pen = rk[:, RK_PENALTY] > 0.5
    is_kicker = restart_active & (is_sp_ours > 0.5) & self_is_taker
    # 일반 세트피스 키커는 공 반대쪽으로 물러난다. 스로어에는 이 방향을 쓰지 않는다. 공이
    # 안쪽·앞쪽으로 날아가면 (player-ball)의 y 목표는 clip되고 x 성분만 남아, 스로어가
    # 터치라인을 따라 공과 반대 방향으로 달리는 인공적인 궤적이 된다.
    retreat_limit = jnp.array([
        c.hx - policy.retouch_pitch_inset,
        c.hy - policy.retouch_pitch_inset,
    ])
    # 코너 키커에게 공 반대 방향을 강제하면 그 방향은 필연적으로 골라인/터치라인 바깥이다.
    # retouch latch가 타인 접촉까지 오래 남는 동안 키커가 벽에 계속 몸을 밀고 마커까지
    # 가두었다. 경계 1m 안에 복귀할 때만 자기 골 쪽 대각선으로 이동시키고, 그 뒤에는
    # 일반 오프더볼 대형에 즉시 합류시킨다. 재터치 킥 금지는 아래 ``retouch`` 마스크가
    # 독립적으로 계속 소유한다.
    setpiece_needs_recovery = sp_bit & jnp.any(
        jnp.abs(my_field) > retreat_limit[None, :], axis=1
    )
    setpiece_recovery_target = my_field + policy.retouch_retreat_distance * _unit(
        own_goal[None, :] - my_field
    )
    setpiece_recovery_target = jnp.clip(
        setpiece_recovery_target, -retreat_limit, retreat_limit
    )
    setpiece_recovery_dir = _unit(setpiece_recovery_target - my_field)

    # 스로어는 현재 x를 고정해 터치라인에 수직으로 들어온 뒤 정상 대형에 합류한다. 타인
    # 접촉 전 재터치 금지(`retouch`)는 계속 유지하되, 피치 안 1 m에 도달한 뒤에는 공을
    # 따라 움직이는 별도 후퇴 목표를 더 이상 덮어쓰지 않는다.
    throw_recovery_y = c.hy - policy.retouch_pitch_inset
    throw_side = jnp.where(my_field[:, 1] >= 0.0, 1.0, -1.0)
    throw_recovery_target = jnp.stack(
        [my_field[:, 0], throw_side * throw_recovery_y], axis=1
    )
    throw_needs_recovery = throw_bit & (
        jnp.abs(my_field[:, 1]) > throw_recovery_y
    )
    throw_recovery_dir = _unit(throw_recovery_target - my_field)
    retouch_move = setpiece_needs_recovery | throw_needs_recovery
    retouch_move_dir = jnp.where(
        throw_needs_recovery[:, None], throw_recovery_dir, setpiece_recovery_dir
    )

    long_opt = direct > policy.gk_long_directness_threshold
    # 골킥 정밀화: 롱옵션이면 ETA 게이트를 통과한 실제 동료의 예상
    # 수신점에 loft한다. 그런 수신자가 없을 때만 기존 다운필드 측면
    # 인-피치 지점을 안전 클리어 폴백으로 쓴다.
    gk_far_tgt = jnp.stack([jnp.clip(my_field[:, 0] + 42.0, -c.hx + 5.0, c.hx - 8.0),
                            jnp.sign(my_field[:, 1] + DIV_EPS) * (c.hy - 14.0)], axis=1)
    gkk_long_target = jnp.where(
        has_gk_long_receiver[:, None], gk_long_receiver_target, gk_far_tgt
    )
    d_gk_far = jnp.linalg.norm(gkk_long_target - ball_field, axis=1)
    gkk_long = is_gkk & (long_opt | (~has_restart_pass))        # 롱 골킥(무옵션 포함)
    gkk_short_loft = (
        is_gkk & (~gkk_long) & (d_pass > policy.gk_long_min_distance)
    )                                                            # 숏 골킥이지만 먼 동료 → 아크
    gkk_dir = jnp.where(
        gkk_long[:, None],
        _unit(gkk_long_target - ball_field),
        jnp.where(gkk_short_loft[:, None], dir_pass, restart_pass_dir),
    )
    gkk_pow = jnp.where(gkk_long, loft_pow_of(d_gk_far),
              jnp.where(gkk_short_loft, loft_pow_of(d_pass),
                        drive_pow_of(restart_pass_distance)))
    gkk_launch = jnp.where(gkk_long | gkk_short_loft, c.loft_launch01, c.drive_launch01)
    # 코너·공격진영 FK는 '열린 동료 발밑'이 아니라 **박스 서비스**다. 위에서 이미 계산한
    # 박스 러너별 체공시간·xG·선점 우위(cross 후보)를 그대로 쓰고, 파워·발사각·스핀도 크로스
    # 물리 역테이블로 페어링한다. loft 테이블에 발밑 목표를 얹으면 실제 코너처럼
    # 니어포스트·스팟으로 떨어지지 않는다.
    is_fk = rk[:, RK_FREEKICK] > 0.5
    # 간접 FK는 직접 골 시도가 금지되고 아래에서 동료 연결로 덮이므로 서비스 대상이 아니다.
    indirect_now = obs[:, c.i_fk_indirect] > 0.5
    box_service = (
        (is_corner | (is_fk & (ball_adv > c.hx * 0.30)))
        & has_cross
        & (~indirect_now)
    )
    # 방향: 페널티=슛, 골킥=gkk, 박스 서비스=크로스 목표, 그 외 짧은 재개=열린 동료(dir_pass).
    sp_kick_dir = jnp.where(is_pen[:, None], dir_shot,
                  jnp.where(is_gkk[:, None], gkk_dir,
                  jnp.where(box_service[:, None], dir_cross,
                  jnp.where(has_restart_pass[:, None], restart_pass_dir,
                            forward_ball))))
    # 파워·발사각을 종류별 솔버에 페어링: 페널티=슛, 코너=loft(아크), 스로=손던지기, 킥오프/FK=drive(지상).
    d_sp = jnp.clip(restart_pass_distance, 4.0, 40.0)             # 재개 동료까지 지상 목표 거리(클램프)
    sp_kick_pow = jnp.where(is_pen, (c.shot_pow if policy.use_shot_solver else 0.9),
                  jnp.where(is_gkk, gkk_pow,
                  jnp.where(box_service, cross_pow,
                  jnp.where(is_corner, loft_pow_of(d_sp),
                  jnp.where(is_throw, throw_pow_of(d_sp), drive_pow_of(d_sp))))))
    sp_launch = jnp.where(is_pen, (shot_launch if policy.use_shot_solver else 0.04),
                jnp.where(is_gkk, gkk_launch,
                jnp.where(box_service, c.cross_launch01,
                jnp.where(is_corner, c.loft_launch01,
                jnp.where(is_throw, c.throw_launch01, c.drive_launch01)))))

    # 간접 FK(IFAB Law 13, is_fk_indirect): 직접골 무효 → 키커는 골 조준 금지, 반드시 동료로 연결.
    # 일반 후보가 없을 때도 restart_pass_dir는 최근접 활성 동료를 가리켜 골대 직격과
    # ``argmax(all -inf)``의 상대/비활성 슬롯 선택을 함께 막는다.
    is_fk_indirect = indirect_now
    sp_kick_dir = jnp.where(
        is_fk_indirect[:, None], restart_pass_dir, sp_kick_dir
    )
    sp_kick_pow = jnp.where(
        is_fk_indirect, drive_pow_of(restart_pass_distance), sp_kick_pow
    )                                                                   # drive(솔버)
    sp_launch = jnp.where(is_fk_indirect, c.drive_launch01, sp_launch)   # drive와 페어링

    # GK 캐치/홀드 배급(RK_GK_HOLD): 롱볼은 도달 가능한 동료에게, 수신자가
    # 없을 때만 전방 측면 안전 구역으로 보낸다. 비롱옵션은 기존 열린 동료 짧은 배급을 유지한다.
    is_hold = rk[:, RK_GK_HOLD] > 0.5
    gk_dist_long = jnp.stack([jnp.clip(my_field[:, 0] + 45.0, -c.hx + 5.0, c.hx - 10.0),
                              jnp.sign(my_field[:, 1] + DIV_EPS) * (c.hy - 12.0)], axis=1)
    gk_dist_target = jnp.where(
        has_gk_long_receiver[:, None], gk_long_receiver_target, gk_dist_long
    )
    gk_dist_long_opt = long_opt | (~has_restart_pass)
    d_gk_dist = jnp.linalg.norm(gk_dist_target - ball_field, axis=1)
    sp_kick_dir = jnp.where(is_hold[:, None],
                            jnp.where(gk_dist_long_opt[:, None], _unit(gk_dist_target - ball_field), restart_pass_dir),
                            sp_kick_dir)
    sp_kick_pow = jnp.where(is_hold,
                            jnp.where(gk_dist_long_opt, loft_pow_of(d_gk_dist),
                                      drive_pow_of(restart_pass_distance)),  # 솔버 정밀
                            sp_kick_pow)
    sp_launch = jnp.where(is_hold, jnp.where(gk_dist_long_opt, c.loft_launch01, c.drive_launch01), sp_launch)

    # 세트피스 배치의 기본값은 '재개 소유' 기준 위치장이다. 데드볼 동안 poss_team이
    # 흔들려도 공격/수비 구조가 뒤집히지 않는다.
    sp_phase = jnp.stack(
        [is_sp_ours > 0.5, is_sp_ours < -0.5, jnp.abs(is_sp_ours) <= 0.5], axis=1
    ).astype(jnp.float32)
    sp_anchor = _anchor_positions(c, ball_field, sp_phase)
    sp_anchor = jnp.stack([
        jnp.clip(sp_anchor[:, 0] + line_shift, -c.hx + 2.0, c.hx - 3.0),
        jnp.clip(sp_anchor[:, 1] * width_scale, -c.hy + 2.0, c.hy - 2.0),
    ], axis=1)
    sp_structure, measured_structure = _setpiece_structure(
        ball_field,
        sp_anchor,
        c.anchor_slot,
        is_gk,
        is_sp_ours,
        rk,
        c,
        return_measured_valid=True,
    )
    restart_target, restart_power, restart_wait = _restart_policy_plan(
        my_field, ball_field, sp_structure, is_gk, is_kicker, is_sp_ours, rk, c
    )
    measured_corner = (
        self_active
        & (~is_kicker)
        & measured_structure
        & (rk[:, RK_CORNER] > 0.5)
    )
    def deconflict_corner(value):
        planned_target, planned_power = value
        deconflicted = POS.deconflict_measured_corner_targets(
            planned_target,
            measured_corner,
            c.team_id,
            r_player=c.r_player,
            hx=c.hx,
            hy=c.hy,
        )
        changed = measured_corner & jnp.any(
            deconflicted != planned_target,
            axis=1,
        )
        distance = jnp.linalg.norm(deconflicted - my_field, axis=1)
        deconflicted_power = jnp.where(
            changed,
            jnp.clip(
                distance / policy.restart_target_slowdown_radius,
                0.0,
                1.0,
            ),
            planned_power,
        )
        return deconflicted, deconflicted_power

    restart_target, restart_power = jax.lax.cond(
        jnp.any(measured_corner),
        deconflict_corner,
        lambda value: value,
        (restart_target, restart_power),
    )
    target = jnp.where(restart_active[:, None], restart_target, target)
    move_pow = jnp.where(restart_active, restart_power, move_pow)
    # 지정 키커의 위치는 env의 강제 재개 이동이 소유한다. 여기서 이동 파워를 낮춰 배치
    # 시간을 벌 수는 없다 — runtime이 같은 프레임에 그 액션 이동을 마스킹하고
    # ``Engine.kicker_speed``로 덮으므로 관측 가능한 효과가 없다. 배치 시간은
    # whistle/out 기준의 종류별 데이터 지연이 직접 보장한다.
    kick_dir = jnp.where(is_kicker[:, None], sp_kick_dir, kick_dir)
    kick_pow = jnp.where(is_kicker, sp_kick_pow, kick_pow)
    launch01 = jnp.where(is_kicker, sp_launch, launch01)
    # 세트피스 킥은 스핀 커맨드를 기본적으로 쓰지 않는다 — 사이드스핀만 0으로 두고 백스핀을
    # 남기면 슛/크로스 분기의 c_spin_b가 키커에게 그대로 새어 든다(비대칭 누락). 다만 박스
    # 서비스는 실제 코너·FK 크로스와 같은 백스핀/인스윙을 유지해야 체공과 착지가 크로스
    # 역테이블과 맞는다.
    sp_spin_s = jnp.where(box_service, -cross_side * (0.55 + 0.25 * width_s), 0.0)
    sp_spin_b = jnp.where(box_service, 0.38, 0.0)
    spin_s = jnp.where(is_kicker, sp_spin_s, spin_s)
    spin_b = jnp.where(is_kicker, sp_spin_b, spin_b)
    do_kick = jnp.where(
        restart_active, is_kicker & self_kicker_ready, do_kick
    )                                                                  # 세트피스 중엔 준비된 지정 키커만 킥

    # ── GK: 스위퍼-키퍼 — 각도수비 + 위협 예측 시 조기 진출해 인터셉트·소유(캐치), 필요시만 클리어 ──
    own_gx = own_goal[0]
    bvx = ball_vel[:, 0]
    ball_vxy = ball_vel[:, :2]
    d_ball_goal = jnp.linalg.norm(ball_field - own_goal[None, :], axis=1) + DIV_EPS
    heading_goal = (bvx * jnp.sign(own_gx) > 0.5) & (d_ball_goal < 32.0)
    # 골라인 교차 예측(슛 커버 각도수비의 기본 위치)
    t_cross = jnp.clip((own_gx - ball_field[:, 0]) / jnp.where(jnp.abs(bvx) < 0.3, jnp.sign(own_gx) * 0.3, bvx), 0.0, 3.0)
    cross_y = jnp.clip(ball_field[:, 1] + ball_vel[:, 1] * t_cross, -c.goal_w * 0.5, c.goal_w * 0.5)
    gk_out = jnp.clip(0.6 + d_ball_goal * 0.045, 0.6, 4.0)
    angle_pos = own_goal[None, :] + _unit(ball_field - own_goal[None, :]) * gk_out[:, None]
    gk_y = jnp.where(heading_goal, cross_y, jnp.clip(angle_pos[:, 1], -c.goal_w * 0.5, c.goal_w * 0.5))
    gk_pos = jnp.stack([jnp.clip(own_gx + gk_out, own_gx + 0.3, own_gx + 4.5), gk_y], axis=1)
    # 실측 GK는 위협이 멀면 골라인에 붙어 있지 않다. K리그 적합에서 공이 상대 골라인 근처일 때
    # GK의 평균 x는 골라인에서 25m 앞(박스 밖)이다 — 스위퍼 높이가 수비라인 높이를 지탱한다.
    # 구판은 항상 4.5m 안이라 뒷공간이 통째로 비고, 화면에서도 GK만 혼자 뒤에 남았다.
    # 위협이 가까우면(또는 공이 골로 향하면) 순수 각도수비로 되돌린다.
    gk_anchor_pos = jnp.stack([
        jnp.clip(anchor[:, 0], own_gx + 0.3, own_gx + 25.0),
        jnp.clip(anchor[:, 1], -c.pen_hw + 1.0, c.pen_hw - 1.0),
    ], axis=1)
    sweeper_w = jnp.where(
        heading_goal, 0.0, jnp.clip((d_ball_goal - 20.0) / 16.0, 0.0, 1.0)
    )
    gk_pos = gk_pos * (1.0 - sweeper_w[:, None]) + gk_anchor_pos * sweeper_w[:, None]

    # 공 미래 경로 예측(굴림+드래그 평균 감속 근사) → GK가 시간 내 닿는 최이른 인터셉트점.
    # 같은 프레임의 수신 경쟁에서 이미 계산한 자기 실효속도를 재사용한다.
    my_vmax = self_vmax
    ts = jnp.linspace(0.08, 2.2, 12)                                # 미래 시각 샘플(s)
    bsp = jnp.linalg.norm(ball_vxy, axis=1) + DIV_EPS
    bhat = ball_vxy / bsp[:, None]
    s_stop = 0.5 * bsp ** 2 / policy.gk_prediction_decel             # 정지까지 이동거리
    s_t = jnp.minimum(jnp.maximum(
        0.0,
        bsp[:, None] * ts[None, :]
        - 0.5 * policy.gk_prediction_decel * ts[None, :] ** 2,
    ),
                      s_stop[:, None])
    ball_fut = ball_field[:, None, :] + bhat[:, None, :] * s_t[:, :, None]       # (N,12,2)
    d_gk_fut = jnp.linalg.norm(ball_fut - my_field[:, None, :], axis=2)          # (N,12)
    reach_by = d_gk_fut <= my_vmax[:, None] * ts[None, :] + 0.5     # GK가 그 시각까지 닿나(+마진)
    any_reach = jnp.any(reach_by, axis=1)
    first_k = jnp.argmax(reach_by, axis=1)
    intercept = jnp.take_along_axis(ball_fut, first_k[:, None, None], axis=1)[:, 0, :]
    # 조기 진출 판정: 상대보다 먼저 붙고(gk_wins) + 잡을 수 있는 속도(claimable). '소유=캐치'는
    # env상 자기 박스 안에서만 가능하므로 진출 목표를 박스 안으로 클리핑(claim_pt) — 밖으로 나가면
    # 클리어가 돼 소유 실패 + 골문 노출. 빠른 슛엔 스위핑하지 않고 라인 각도수비(gk_pos·cross_y) 유지.
    d_opp_int = jnp.min(jnp.where(opp, jnp.linalg.norm(others_field - intercept[:, None, :], axis=2), jnp.inf), axis=1)
    # GK는 '명확히 이길 때만' 스위핑(마진 -1.5m) — 경합 스루볼을 다 수거하면 전진 패스 공격이 전부
    # 죽는다(스위퍼 vs 전진패스 자기충돌). 확실히 먼저 닿는 공만 나가고, 50/50 볼은 공격수에게 넘겨 슛 기회.
    gk_wins_int = jnp.linalg.norm(intercept - my_field, axis=1) <= d_opp_int - 1.5
    ball_low = ball_z < 1.7
    claimable = ball_speed < c.gk_catch_cap * 0.9                   # 잡을 수 있는 속도(슛엔 라인 유지)
    claim_pt = jnp.stack([jnp.clip(intercept[:, 0], own_gx + 0.3, own_gx + c.pen_len - 1.0),
                          jnp.clip(intercept[:, 1], -c.pen_hw + 1.0, c.pen_hw - 1.0)], axis=1)  # 박스 안으로 제한
    # ★위협은 '공이 자기 골로 향하는가'(소유 무관)로 판정 — 우리 팀 백패스·굴절이 자기 골로 굴러가는
    # 자살골 상황을 소유(attacking) 게이트로 놓치던 버그 수정. 상대 슛뿐 아니라 아군 공도 위협이면 진출.
    ball_toward_own = ball_vel[:, 0] * jnp.sign(own_gx) > 0.3       # 자기 골 방향 이동
    threat = ball_toward_own | (~attacking)                        # 골로 향함 or 상대/루즈볼
    sweep = (is_gk & (~restart_active) & any_reach & ball_low & claimable & gk_wins_int
             & (ball_field[:, 0] < -c.hx * 0.30) & threat & (d_ball_goal < 24.0))
    # 근접 스매더: 아주 가까우면(<7m) 소유 무관 돌진, 골로 향하는 근거리는 잡을 수 있을 때만 진출.
    gk_rush = is_gk & (~restart_active) & ((d_ball_goal < 7.0)
                                           | (heading_goal & (d_ball_goal < 12.0) & claimable))
    gk_come = sweep | gk_rush                                       # 진출(스위퍼 or 최종 돌진)
    # 재개 중 GK도 위의 합법 대기 목표를 따른다. 특히 페널티 수비 GK를 오픈플레이
    # 각도수비 위치로 다시 덮으면 매 frame 골라인 투영이 반복된다.
    gk_active = is_gk & (~is_kicker) & (~restart_active)
    gk_target = jnp.where(gk_rush[:, None], intercept, jnp.where(sweep[:, None], claim_pt, gk_pos))
    target = jnp.where(gk_active[:, None], gk_target, target)
    # GK도 각도수비 위치를 이미 잡았으면 걷는다. 진출(스위핑·돌진)만 전력이다.
    gk_hold_pow = _arrival_power(
        jnp.linalg.norm(gk_pos - my_field, axis=1), policy
    )
    move_pow = jnp.where(gk_active, jnp.where(gk_come, 1.0, gk_hold_pow), move_pow)

    # 손처리 vs 클리어: 자기 박스 안에서 합법적으로 손을 쓸 수 있으면 속도와 무관하게
    # hand claim(do_kick=False)을 시도한다. catch/parry 결과를 정책 쪽 하드캡으로 먼저
    # 잘라 버리면 환경의 실측 확률곡선이 고속 save 구간을 전혀 보지 못한다. 박스 밖이거나
    # 고의적 아군 발 백패스일 때만 솔버로 인-피치 정밀 클리어한다.
    # Hand-use legality is determined at the ball contact point, not at the
    # goalkeeper's feet (movement._ball_in_each_own_box is the runtime SSOT).
    # Include the real goal-line bound as players may legally stand up to 5 m
    # behind it while a ball there is outside the penalty area.
    gk_ball_in_box = (
        (jnp.abs(own_gx - ball_field[:, 0]) <= c.pen_len)
        & (jnp.abs(ball_field[:, 1]) <= c.pen_hw)
        & (jnp.abs(ball_field[:, 0]) <= c.hx)
    )
    # f2b_avail is the SSOT for the challenger-aware cooldown: a possession-team
    # action bypasses the re-challenge lock, while loose/opponent claims obey it.
    gk_reach_ball = is_gk & in_reach & f2b_avail & (~restart_active)
    # ★우리 팀 GK에게 의도적으로 보낸 공은 env가 '손 캐치'를 금지한다 → 캐치 시도(do_kick=False)하면
    # GK가 아무것도 못 하고 공이 골로 굴러 들어간다(자살골 원인). 그런 공은 '발 클리어'(do_kick=True)로 처리.
    # Runtime and policy share the causal relation latch.  It is armed from the
    # submitted target direction, so semantic labels such as INTERCEPT/TACKLE
    # neither bypass the rule nor make unrelated team-mate shots illegal.
    gk_handling_relation = obs[:, c.i_gk_handling_restricted]
    gk_handling_restricted = gk_handling_relation > 0.5
    gk_own_distribution_live = gk_handling_relation > 1.5
    gk_hand_claim = (
        gk_reach_ball
        & gk_ball_in_box
        & (~gk_handling_restricted)
    )
    # +2는 이 GK가 손에서 방금 배급한 뒤 아직 타인이 접촉하지 않은 공이다. 발 재터치는
    # 합법일 수 있지만, 빠르게 골 밖으로 진행 중인 자기 배급을 매 0.067초마다 다시 차는
    # 정책 행동은 아니다. 정지했거나 자기 골로 되돌아오는 긴급 공은 계속 발로 처리하고,
    # +1인 동료 백패스도 종전처럼 발로 처리한다.
    gk_release_foot_needed = (
        (~gk_own_distribution_live)
        | ball_toward_own
        | (ball_speed <= policy.unattended_ball_speed)
    )
    gk_foot_play = (
        gk_reach_ball
        & (~gk_hand_claim)
        & gk_release_foot_needed
    )
    # A deliberate back-pass forbids hands, not build-up.  If a safe nearby
    # ground receiver exists and pressure is manageable, play through that
    # receiver; only the remaining foot contacts are emergency lofted clears.
    gk_foot_distribution = (
        gk_foot_play
        & has_gk_short_receiver
        & (press_self <= policy.gk_foot_pass_pressure_max)
    )
    clear_tgt = gk_dist_long
    d_clear = jnp.linalg.norm(clear_tgt - ball_field, axis=1)
    gk_foot_dir = jnp.where(
        gk_foot_distribution[:, None], dir_gk_short, _unit(clear_tgt - ball_field)
    )
    gk_foot_pow = jnp.where(
        gk_foot_distribution, drive_pow_of(d_gk_short), loft_pow_of(d_clear)
    )
    gk_foot_launch = jnp.where(
        gk_foot_distribution, c.drive_launch01, c.loft_launch01
    )
    kick_dir = jnp.where(gk_foot_play[:, None], gk_foot_dir, kick_dir)
    kick_pow = jnp.where(gk_foot_play, gk_foot_pow, kick_pow)
    launch01 = jnp.where(gk_foot_play, gk_foot_launch, launch01)
    spin_s = jnp.where(gk_foot_play, 0.0, spin_s)
    spin_b = jnp.where(gk_foot_play, 0.0, spin_b)
    # Legal hand claims remain do_kick=False so the environment catches; all
    # other reachable GK contacts are explicit foot plays.
    do_kick = jnp.where(is_gk & (~restart_active), gk_foot_play, do_kick)

    # ── 재터치 가드: 세트피스 키커 후퇴 / 스로어 수직 복귀 + 공통 킥 금지 ──
    movement_boundary_margin = jnp.where(
        ball_alive & (~restart_active),
        jnp.float32(c.r_player),
        jnp.float32(c.player_boundary_margin),
    )
    target = _clip_policy_target_to_player_domain(
        target, c, movement_boundary_margin
    )
    move_dir = _unit(target - my_field)
    move_dir = jnp.where(retouch_move[:, None], retouch_move_dir, move_dir)
    # 실제 후퇴/재진입 중에는 커밋된 파워를 쓴다. 피치 안으로 돌아온 스로어는 일반 오프볼
    # 이동 강도를 회복하지만, 아래 retouch 킥 마스크는 타인 접촉 전까지 그대로 남는다.
    move_pow = jnp.where(
        retouch_move,
        jnp.maximum(move_pow, policy.retouch_retreat_power),
        move_pow,
    )
    do_kick = do_kick & (~retouch)

    # ── 캐리어 드리블: 킥 안 하는 캐리어는 공 지나 3m로 몰고 감(오버런 자동 회수) ──
    is_dribbling = (am_nearest_mate & attacking & carrier_drib & (~do_kick)
                    & (~restart_active) & (~is_gk) & (~own_live_pass))
    move_dir = jnp.where(is_dribbling[:, None], _unit(drib_target - my_field), move_dir)

    # ── 인입 볼 트래핑/헤더: 빠른 지상볼은 소프트 트래핑, 공중볼은 그 자리서 헤더/발리(env가 접촉높이로 판정) ──
    # 글루가 없어 굴러오는 패스/루즈볼은 능동 트래핑, 떨어지는 공은 최근접이 그대로 때려 경합.
    # v24: 무압박 공중볼은 머리 아래로 하강할 때까지 기다려 소프트 터치,
    # 경합 압박 중의 공중볼만 즉시 헤더/발리를 쓴다.
    # 지상볼은 기존 전역 최근접 트랩을 유지한다. 공중볼은 서비스 팀 runner와 ETA 창
    # 안의 수비 runner만 동시에 제출해 env contest가 실제 거리·높이·ball control로
    # 승자를 가리고, 늦은 수비 cover는 접촉 후보에서 제외한다.
    receive_claimant = jnp.where(
        airborne,
        aerial_runner,
        jnp.where(loose_phase, loose_primary, am_nearest_field),
    )
    aerial_space = press_self <= policy.aerial_control_pressure_max

    # An action is held for the entire control frame, while ``f2b_avail`` is
    # exact only at its entry substep.  A fast descending ball can cross the
    # player's reach boundary and leave it again before the next policy call.
    # Arm the selected runner when the ball can enter reach during this frame;
    # movement._kick_gate still owns the exact per-substep geometry, locks and
    # cooldowns.  For an unpressured soft control, arming is safe only after the
    # ball is already below the requested control ceiling: otherwise the env
    # would legally apply the held action as soon as it crosses head reach,
    # earlier than the policy intended.
    # The contest gate runs before ball integration in each physics substep.
    # Therefore the last actionable instant is (decimation - 1) * dt_phys,
    # not the post-frame state at control_dt.
    arm_t = jnp.asarray(c.f2b_action_horizon, dtype=ball_z.dtype)
    arm_air_distance = _air_travel_distance(ball_xy_speed, arm_t, c.ball_drag)
    arm_ball_xy = ball_field + ball_travel_dir * arm_air_distance[:, None]
    arm_ball_z = jnp.maximum(
        ball_z + ball_vel[:, DIM_Z] * arm_t - 0.5 * c.gravity * arm_t ** 2,
        c.r_ball,
    )
    arm_ball_vz = ball_vel[:, DIM_Z] - c.gravity * arm_t
    arm_ball_speed = ball_xy_speed / (
        1.0 + c.ball_drag * ball_xy_speed * arm_t
    )
    arm_radius = jnp.where(
        poss > 0.5, c.reach_xy_carry, c.reach_xy_challenge
    ) + c.r_ball
    arm_xy_reachable = (
        jnp.linalg.norm(arm_ball_xy - my_field, axis=1)
        <= arm_radius + self_vmax * arm_t
    )
    arm_height_reachable = arm_ball_z <= self_reach_z + c.r_ball
    arm_blockable = reach_blockable(
        arm_ball_speed,
        arm_ball_z,
        c.reach_block_limit,
        c.reach_height_penalty,
    )
    soft_control_can_arm = (
        (ball_z <= policy.aerial_control_max_height)
        & (arm_ball_vz < 0.0)
    )
    aerial_contact_imminent = (
        aerial_runner
        & (~f2b_avail)
        & kick_potential
        & arm_xy_reachable
        & arm_height_reachable
        & arm_blockable
        & ((~aerial_space) | soft_control_can_arm)
    )
    aerial_shot_intent = am_nearest_mate & attacking & shoot_ok
    expected_contact_z = jnp.where(f2b_avail, ball_z, arm_ball_z)
    expected_header_contact = expected_contact_z > self_head_z
    receive_touch_candidate = (
        ((receive_claimant & in_reach & f2b_avail) | aerial_contact_imminent)
        & (~restart_active)
        & (~retouch)
        & ((~own_live_pass) | receive_or_arrived)
        # A recent low DRIBBLE is already the carrier's controlled ball.  The
        # tactical carrier branch may pass it or wait for the next dribble
        # touch; the generic incoming-ball trap must not alternately overwrite
        # that decision with a fixed 3 m/s touch.
        & (~(
            am_nearest_mate
            & attacking
            & carried_release
            & (ball_z <= policy.pressured_aerial_control_max_height)
        ))
        & ((~settled) | airborne)
        & (~quick_relay)
        # 낮은 발리·바운드 슛은 기존 슛 솔버가 소유한다. 실제 head_z보다 높은 예상
        # 접촉만 아래의 거리별 헤더 속도 계약으로 넘겨 일반 슛 파워를 조용히 낮추지 않는다.
        & (~(aerial_shot_intent & (~expected_header_contact)))
        & (~ground_challenge_action)
    )
    aerial_control_ready = (
        airborne
        & aerial_space
        & (ball_vel[:, DIM_Z] < 0.0)
        & (ball_z <= policy.aerial_control_max_height)
    )
    pressured_low_control = (
        airborne
        & (~aerial_space)
        & (
            ball_z
            <= policy.pressured_aerial_control_max_height
        )
    )
    aerial_soft_touch_ready = (
        aerial_control_ready
        | (aerial_contact_imminent & aerial_space)
        | pressured_low_control
    )
    aerial_control_wait = (
        airborne
        & aerial_space
        & (~aerial_soft_touch_ready)
        & (~aerial_shot_intent)
    )
    trap_now = receive_touch_candidate & (~aerial_control_wait)
    aerial_clear_now = trap_now & airborne & (~aerial_soft_touch_ready)
    # 높은 공을 무조건 전방 0.5 power로 돌려보내지 않는다. 명시적 원터치 슛은 골을
    # 겨냥하고, 그 외에는 높은 품질의 짧은 동료 연결을 우선하며, 후보가 없을 때만
    # 상대 진영으로 걷어낸다. 출구속도는 행동별 m/s config라 엔진 max 변경에도 의미가 같다.
    aerial_shot_now = aerial_clear_now & aerial_shot_intent
    aerial_pass_now = (
        aerial_clear_now & (~aerial_shot_now) & has_aerial_pass
    )
    aerial_pass_dir = _unit(aerial_pass_target - ball_field)
    aerial_action_dir = jnp.where(
        aerial_shot_now[:, None],
        dir_shot,
        jnp.where(
            aerial_pass_now[:, None], aerial_pass_dir, challenge_clear_dir
        ),
    )
    aerial_action_speed = jnp.where(
        aerial_shot_now,
        policy.aerial_shot_speed_mps,
        jnp.where(
            aerial_pass_now,
            aerial_pass_speed,
            policy.aerial_clear_speed_mps,
        ),
    )
    aerial_action_power = aerial_action_speed / c.f2b_max
    aerial_action_launch = jnp.where(
        aerial_shot_now,
        shot_launch if policy.use_shot_solver else 0.05,
        jnp.where(
            aerial_pass_now,
            policy.aerial_pass_launch01,
            policy.aerial_clear_launch01,
        ),
    )
    receive_preferred_dir = jnp.where(
        aerial_space[:, None], dribble_dir, challenge_control_dir
    )
    receive_control_dir, receive_control_speed = _reception_control_plan(
        my_vel,
        receive_preferred_dir,
        policy.ground_control_touch_speed_mps,
        policy.receive_control_lead_speed_mps,
        carry_touch_speed_cap,
    )
    soft_pass_control = (
        trap_now
        & own_live_pass
        & receive_or_arrived
        & (~aerial_clear_now)
    )
    trap_dir = jnp.where(
        soft_pass_control[:, None],
        receive_control_dir,
        jnp.where(
            aerial_clear_now[:, None],
            aerial_action_dir,
            jnp.where(
                pressured_low_control[:, None],
                challenge_control_dir,
                challenge_clear_dir,
            ),
        ),
    )
    kick_dir = jnp.where(trap_now[:, None], trap_dir, kick_dir)
    kick_pow = jnp.where(
        trap_now,
        jnp.where(
            aerial_clear_now,
            aerial_action_power,
            jnp.where(
                soft_pass_control,
                receive_control_speed / c.f2b_max,
                ground_control_power,
            ),
        ),
        kick_pow,
    )  # 공중볼은 세게(헤더/발리), 지상볼은 config 기반 통제 터치
    launch01 = jnp.where(
        trap_now, jnp.where(aerial_clear_now, aerial_action_launch, 0.0), launch01
    )
    spin_s = jnp.where(trap_now, 0.0, spin_s)
    spin_b = jnp.where(trap_now, 0.0, spin_b)
    do_kick = do_kick | trap_now

    # A receive action is held across all physics substeps in this control
    # frame.  Stop commanding the old predictive sprint as soon as the touch is
    # armed: keep the receiver's current speed along the same vector as the
    # controlled ball.  The normal carrier homing branch owns the next frame.
    receive_hold_power = jnp.clip(
        my_speed / (self_vmax + DIV_EPS), 0.0, 1.0
    )
    move_dir = jnp.where(
        soft_pass_control[:, None], receive_control_dir, move_dir
    )
    move_pow = jnp.where(
        soft_pass_control, receive_hold_power, move_pow
    )

    # 모든 policy launch 값은 지면 캘리브가 의도한 **물리각**의 fraction이다. Env action은
    # 접촉 높이에 따라 달라지는 [launch_lo(z), launch_max] fraction이므로 마지막에 단 한 번
    # 역매핑한다. 이를 슛에만 적용하면 낮게 튀는 공의 패스·크로스·GK 클리어가 같은 이유로
    # 과도하게 아래로 향한다. 지면공에서는 항등이며 스로인도 take 직전 공이 스폿 지면에 있다.
    ground_floor = -c.launch_down_ground
    intended_launch_angle = ground_floor + launch01 * (
        c.launch_max - ground_floor
    )
    contact_height_frac = jnp.clip(
        (ball_z - c.r_ball) / (c.launch_down_ref - c.r_ball + DIV_EPS),
        0.0,
        1.0,
    )
    contact_floor = -(
        c.launch_down_ground
        + (c.launch_max - c.launch_down_ground) * contact_height_frac
    )
    launch01 = jnp.clip(
        (intended_launch_angle - contact_floor)
        / (c.launch_max - contact_floor + DIV_EPS),
        0.0,
        1.0,
    )

    # ── anti-clump: 공 미커밋 오프더볼은 동료 반발을 이동방향에 강하게 블렌딩(넓게 벌리기) ──
    # 세트피스 홀드/후퇴 중인 선수도 committed — 반발 블렌딩이 홀드 타겟을 표류시키면
    # 제한구역 경계 재진입 진동과 반복 env 투영이 발생한다.
    committed = (am_nearest_field | receive_runner | aerial_go | press
                 | press_cover | loose_primary | loose_cover | is_kicker
                 | is_gk | do_kick | retouch_move | is_dribbling | trap_now
                 | restart_wait)
    rel_tm = my_field[:, None, :] - others_field
    dist_tm = jnp.linalg.norm(rel_tm, axis=2) + DIV_EPS
    near_tm = mate & (dist_tm < policy.anti_clump_radius)
    push = jnp.sum(jnp.where(near_tm[:, :, None], rel_tm / dist_tm[:, :, None]
                             * (1.0 - dist_tm / policy.anti_clump_radius)[:, :, None], 0.0), axis=1)
    move_dir = jnp.where(
        (~committed)[:, None],
        _unit(move_dir + policy.anti_clump_gain * push),
        move_dir,
    )
    move_dir = _remove_outward_boundary_motion(
        my_field, move_dir, c, movement_boundary_margin
    )

    # ── 실행 노이즈(킥 각오차) + 액션 조립(공격 프레임) ───────────────────────
    # nominal solver는 먼 포스트 안쪽을 정확히 겨냥하지만 실제 피니싱은 완벽하지 않다. 좋은 xG는
    # 오차가 작고, 먼/차폐된 낮은 xG와 압박 속 슛은 커진다. 구 0.012rad 상수는 30m에서도 횡오차
    # 표준편차가 0.36m뿐이라 규칙 정책이 사실상 매번 GK 리치 밖 구석을 맞혔다.
    # 기저값은 K리그 슛 결말(골대밖 43%)에서 역산했다 — config 주석 참고.
    shot_noise = (
        policy.shot_noise_base_rad
        + policy.shot_noise_xg_rad * (1.0 - xg_self)
        + policy.shot_noise_pressure_rad * jnp.clip(press_self, 0.0, 1.0)
    )
    # ``shoot``는 모든 슬롯이 "내가 캐리어라면"을 병렬 계산한 후보다. 실제 실행 종류와
    # 분리하지 않으면 트랩·GK 배급·일반 재개가 그 슬롯의 가상 shoot=True 때문에 슛 분산을
    # 받는다. 실제 오픈플레이 캐리어 슛과 페널티만 명시적으로 고른다.
    actual_open_shot = (
        do_kick
        & shoot
        & am_nearest_mate
        & attacking
        & (~restart_active)
        & ((~trap_now) | aerial_shot_now)
        & (~is_gk)
    )
    actual_penalty = do_kick & restart_active & is_pen & is_kicker
    noise_scale = jnp.where(
        actual_open_shot | actual_penalty,
        shot_noise,
        0.04 + 0.0028 * ball_dist,
    )
    # 같은 선택을 해도 기술 좋은 프로는 킥 방향을 더 좁게 반복하고, 하위 프로는 조금
    # 더 흔들린다. 능력 band 양 끝의 차이는 기본값으로 ±16%뿐이라 누구도 패스를 못 할
    # 수준으로 붕괴하지 않지만, 여러 킥이 누적되는 경기 영상과 성공률에는 분명히 남는다.
    noise_scale = noise_scale * _execution_noise_multiplier(
        self_ability, policy.execution_control_noise_gain
    )
    ang = jax.random.normal(k_noise, (N,)) * noise_scale
    cs, sn = jnp.cos(ang), jnp.sin(ang)
    kick_dir = jnp.stack([cs * kick_dir[:, 0] - sn * kick_dir[:, 1],
                          sn * kick_dir[:, 0] + cs * kick_dir[:, 1]], axis=1)

    # 슛의 실행 오차는 **등방**이다 — 슈터는 좌우만 틀리는 것이 아니라 높이도 틀린다.
    # 수평 오차만 두면 발사각이 역산값 그대로라 공이 크로스바를 넘는 일이 아예 없다:
    # 실측으로 900초·3시드에서 골킥과 코너가 각각 0건이었고(K리그 14.3·8.6건) 슛은
    # 골 아니면 GK 처리로만 끝났다. 수평과 같은 표준편차를 발사각에도 준다.
    # 세트피스·패스의 발사각은 착지거리 역테이블에 묶여 있어 건드리지 않는다 — 여기서
    # 흔들면 스로인 릴리스나 코너 체공이 캘리브 값에서 벗어난다.
    # 별도 stream을 fold_in으로 파생해 기존 패스·수신자·수평오차 난수열을 밀지 않는다.
    k_launch = jax.random.fold_in(key, jnp.uint32(0x4C4E4348))
    launch_span = jnp.float32(c.launch_max + c.launch_down_ground)
    launch01 = jnp.where(
        actual_open_shot | actual_penalty,
        jnp.clip(
            launch01
            + jax.random.normal(k_launch, (N,))
            * shot_noise
            * _execution_noise_multiplier(
                self_ability, policy.execution_control_noise_gain
            )
            / launch_span,
            0.0,
            1.0,
        ),
        launch01,
    )

    # 이동·킥 = L∞ radial stretch 2D(방향+크기를 한 벡터로). env _decode의 stretch_decode와 역쌍.
    mv_v = stretch_encode(move_dir, jnp.clip(move_pow, 0.0, 1.0))       # (N,2)
    kick_v = stretch_encode(kick_dir, jnp.clip(kick_pow, 0.0, 1.0))     # (N,2)
    action = jnp.stack([
        jnp.where(do_kick, 1.0, 0.0),        # 킥 게이트 dim[0]∈[0,1](0.5 초과=킥): 킥 1.0 / 미킥 0.0
        mv_v[:, 0], mv_v[:, 1],
        kick_v[:, 0], kick_v[:, 1],
        2.0 * jnp.clip(launch01, 0.0, 1.0) - 1.0,
        jnp.clip(spin_s, -1.0, 1.0), jnp.clip(spin_b, -1.0, 1.0),
    ], axis=1)
    # Inactive slots have no policy identity, and a frame where *every* row has
    # both movement and kick agency closed is a terminal/no-agency frame.  Emit
    # exact zero there.  Do not erase an individual restart encroacher's policy
    # target merely because the referee also projects it this frame: that target
    # is the documented rule-policy response and remains useful diagnostically.
    no_agency_frame = jnp.all(
        (aff["move_forced"] > 0.5) & (aff["kick_gated"] > 0.5)
    ) & (~jnp.any(aff["kick_forced"] > 0.5))
    row_active = self_active & (~no_agency_frame)
    action = jnp.where(row_active[:, None], action, 0.0)
    # L-infinity stretch arithmetic may overshoot a Box face by one float32
    # ULP on an accelerator.  The public action-space bound is exact.
    action = _bound_policy_action(action)
    if not with_trace:
        return action

    # ------------------------------------------------------------------
    # Same-call imitation decision trace
    # ------------------------------------------------------------------
    # These stable mode codes are part of the public capture/dataset contract;
    # keeping them local avoids coupling the environment runtime to a trainer.
    mode_none = jnp.uint8(0)
    mode_chase = jnp.uint8(1)
    mode_pass = jnp.uint8(2)
    mode_cross = jnp.uint8(3)
    mode_shot = jnp.uint8(4)
    mode_clear = jnp.uint8(5)
    mode_restart = jnp.uint8(6)

    slot = jnp.arange(N, dtype=jnp.int32)

    def take_scalar(values, local_index):
        return jnp.take_along_axis(
            values, local_index[:, None], axis=1
        )[:, 0]

    def other_slot(local_index):
        return take_scalar(c.others_idx, local_index).astype(jnp.int32)

    # Open-play pass/cross: expose the receiver from the final branch, not the
    # unused per-player argmaxes.  A pressured outlet has its own receiver.
    open_receiver_j = jnp.where(
        crossing,
        best_cross_j,
        jnp.where(pressured_outlet, challenge_outlet_j, best_j),
    )
    open_receiver_valid = carrier_pass | crossing
    open_target = jnp.where(
        crossing[:, None], best_cross_target, carrier_pass_target
    )
    best_ground_arrival = take_scalar(ground_flight_t, best_j)
    best_loft_arrival = take_scalar(loft_flight_t, best_j)
    outlet_ground_arrival = take_scalar(
        ground_flight_t, challenge_outlet_j
    )
    cross_arrival = take_scalar(cross_flight_t, best_cross_j)
    open_arrival = jnp.where(
        crossing,
        cross_arrival,
        jnp.where(
            pressured_outlet,
            outlet_ground_arrival,
            jnp.where(best_lofted, best_loft_arrival, best_ground_arrival),
        ),
    )
    outlet_through = take_scalar(through_strength, challenge_outlet_j)
    cross_through = take_scalar(through_strength, best_cross_j)
    open_through = jnp.where(
        crossing,
        cross_through,
        jnp.where(pressured_outlet, outlet_through, best_through),
    )
    open_loft = crossing | (carrier_pass & best_lofted & (~pressured_outlet))
    open_cross = crossing

    # A legal open-play GK foot distribution is a real planned pass; an
    # emergency foot play without the short receiver is a clearance.
    gk_receiver_j = gk_short_j
    gk_receiver_valid = gk_foot_distribution
    gk_target_trace = gk_short_target
    gk_arrival = take_scalar(ground_flight_t, gk_short_j)
    gk_through = take_scalar(through_strength, gk_short_j)

    # Restart plans are evaluated every waiting draw as well as on the release
    # draw.  Apply overrides in the same order as the action path above:
    # ordinary/box service -> goal kick -> penalty -> indirect FK -> GK hold.
    restart_default_j = jnp.where(
        has_restart_pass, best_j, nearest_mate_j
    )
    restart_receiver_j = jnp.where(
        box_service, best_cross_j, restart_default_j
    )
    restart_receiver_valid = box_service | has_restart_pass
    restart_target_trace = jnp.where(
        box_service[:, None], best_cross_target, restart_pass_target
    )
    restart_loft = box_service | is_corner | is_throw
    restart_cross = box_service
    restart_pass_through = jnp.where(has_restart_pass, best_through, 0.0)
    restart_through = jnp.where(
        box_service,
        cross_through,
        restart_pass_through,
    )

    gkk_receiver_valid = jnp.where(
        gkk_long, has_gk_long_receiver, has_restart_pass
    )
    gkk_receiver_j = jnp.where(gkk_long, gk_long_j, best_j)
    gkk_target_trace = jnp.where(
        gkk_long[:, None],
        gkk_long_target,
        jnp.where(
            gkk_short_loft[:, None], best_pass_target, restart_pass_target
        ),
    )
    gkk_through = jnp.where(
        gkk_long,
        take_scalar(through_strength, gk_long_j),
        best_through,
    )
    restart_receiver_j = jnp.where(
        is_gkk, gkk_receiver_j, restart_receiver_j
    )
    restart_receiver_valid = jnp.where(
        is_gkk, gkk_receiver_valid, restart_receiver_valid
    )
    restart_target_trace = jnp.where(
        is_gkk[:, None], gkk_target_trace, restart_target_trace
    )
    restart_loft = jnp.where(
        is_gkk, gkk_long | gkk_short_loft, restart_loft
    )
    restart_cross = jnp.where(is_gkk, False, restart_cross)
    restart_through = jnp.where(
        is_gkk, gkk_through, restart_through
    )

    restart_receiver_valid = jnp.where(
        is_pen, False, restart_receiver_valid
    )
    restart_cross = jnp.where(is_pen, False, restart_cross)
    restart_loft = jnp.where(is_pen, False, restart_loft)
    restart_through = jnp.where(is_pen, 0.0, restart_through)

    indirect_receiver_valid = has_restart_pass | has_any_mate
    restart_receiver_j = jnp.where(
        is_fk_indirect, restart_default_j, restart_receiver_j
    )
    restart_receiver_valid = jnp.where(
        is_fk_indirect, indirect_receiver_valid, restart_receiver_valid
    )
    restart_target_trace = jnp.where(
        is_fk_indirect[:, None], restart_pass_target, restart_target_trace
    )
    restart_loft = jnp.where(is_fk_indirect, False, restart_loft)
    restart_cross = jnp.where(is_fk_indirect, False, restart_cross)
    restart_through = jnp.where(
        is_fk_indirect,
        restart_pass_through,
        restart_through,
    )

    hold_receiver_valid = jnp.where(
        gk_dist_long_opt, has_gk_long_receiver, has_restart_pass
    )
    hold_receiver_j = jnp.where(gk_dist_long_opt, gk_long_j, best_j)
    hold_target_trace = jnp.where(
        gk_dist_long_opt[:, None], gk_dist_target, restart_pass_target
    )
    hold_through = jnp.where(
        gk_dist_long_opt,
        take_scalar(through_strength, gk_long_j),
        best_through,
    )
    restart_receiver_j = jnp.where(
        is_hold, hold_receiver_j, restart_receiver_j
    )
    restart_receiver_valid = jnp.where(
        is_hold, hold_receiver_valid, restart_receiver_valid
    )
    restart_target_trace = jnp.where(
        is_hold[:, None], hold_target_trace, restart_target_trace
    )
    restart_loft = jnp.where(is_hold, gk_dist_long_opt, restart_loft)
    restart_cross = jnp.where(is_hold, False, restart_cross)
    restart_through = jnp.where(is_hold, hold_through, restart_through)

    restart_distance = jnp.linalg.norm(
        restart_target_trace - ball_field, axis=1
    ) + DIV_EPS
    restart_drive_speed = jnp.interp(
        restart_distance, c.drive_R, c.drive_v
    )
    restart_drive_arrival = (
        2.0
        * restart_distance
        / (restart_drive_speed + policy.drive_arrive_speed_mps + DIV_EPS)
    )
    restart_loft_arrival = jnp.interp(
        restart_distance, c.loft_R, c.loft_T
    )
    restart_cross_arrival = jnp.interp(
        restart_distance, c.cross_R, c.cross_T
    )
    restart_arrival = jnp.where(
        restart_cross,
        restart_cross_arrival,
        jnp.where(restart_loft, restart_loft_arrival, restart_drive_arrival),
    )

    open_mode = jnp.where(
        shoot,
        mode_shot,
        jnp.where(
            crossing,
            mode_cross,
            jnp.where(
                carrier_pass,
                mode_pass,
                jnp.where(carrier_clear, mode_clear, mode_chase),
            ),
        ),
    ).astype(jnp.uint8)
    gk_mode = jnp.where(
        gk_foot_distribution,
        mode_pass,
        jnp.where(gk_foot_play, mode_clear, mode_chase),
    ).astype(jnp.uint8)
    restart_mode = jnp.where(
        restart_receiver_valid,
        jnp.where(restart_cross, mode_cross, mode_pass),
        jnp.where(
            is_pen,
            mode_shot,
            jnp.where(
                (is_gkk & gkk_long) | (is_hold & gk_dist_long_opt),
                mode_clear,
                mode_restart,
            ),
        ),
    ).astype(jnp.uint8)

    # ``poss`` is observer-relative, so only the possession team can nominate
    # an open-play carrier.  A live-pass receiver becomes a carrier only for an
    # actual quick relay; a GK back-pass foot play is the analogous exception.
    open_causal = (
        attacking
        & am_nearest_mate
        & (~restart_active)
        & (~is_gk)
        & ((~own_live_pass) | quick_relay)
    )
    gk_causal = (
        attacking
        & am_nearest_mate
        & (~restart_active)
        & is_gk
        & ((~own_live_pass) | gk_foot_play)
    )
    restart_causal = is_kicker
    causal_candidate = (
        open_causal | gk_causal | restart_causal
    ) & row_active
    # Exact distance ties can nominate two rows under GEOMETRY_EPS.  Phase S's
    # schema permits one causal carrier, so resolve only the trace tie by stable
    # global slot order; this does not alter either row's action.
    has_causal = jnp.any(causal_candidate)
    causal_slot = jnp.argmin(
        jnp.where(causal_candidate, slot, jnp.int32(N))
    )
    causal = causal_candidate & (slot == causal_slot) & has_causal

    per_mode = jnp.where(
        restart_causal,
        restart_mode,
        jnp.where(gk_causal, gk_mode, open_mode),
    ).astype(jnp.uint8)
    per_releasing = jnp.where(
        restart_causal,
        do_kick,
        jnp.where(gk_causal, gk_foot_play, carrier_releasing_ball),
    )
    teacher_carrier_mode = jnp.where(
        causal, per_mode, mode_none
    ).astype(jnp.uint8)
    teacher_carrier_releasing = causal & per_releasing

    per_receiver_j = jnp.where(
        restart_causal,
        restart_receiver_j,
        jnp.where(gk_causal, gk_receiver_j, open_receiver_j),
    )
    per_receiver_valid = jnp.where(
        restart_causal,
        restart_receiver_valid,
        jnp.where(gk_causal, gk_receiver_valid, open_receiver_valid),
    )
    per_target = jnp.where(
        restart_causal[:, None],
        restart_target_trace,
        jnp.where(gk_causal[:, None], gk_target_trace, open_target),
    )
    per_arrival = jnp.where(
        restart_causal,
        restart_arrival,
        jnp.where(gk_causal, gk_arrival, open_arrival),
    )
    per_through = jnp.clip(
        jnp.where(
            restart_causal,
            restart_through,
            jnp.where(gk_causal, gk_through, open_through),
        ),
        0.0,
        1.0,
    )
    per_loft = jnp.where(
        restart_causal,
        restart_loft,
        jnp.where(gk_causal, False, open_loft),
    )
    per_cross = jnp.where(
        restart_causal,
        restart_cross,
        jnp.where(gk_causal, False, open_cross),
    )

    chosen_mode = per_mode[causal_slot]
    coordination_present = (
        has_causal
        & per_receiver_valid[causal_slot]
        & ((chosen_mode == mode_pass) | (chosen_mode == mode_cross))
    )
    receiver_slot = other_slot(per_receiver_j)[causal_slot]
    teacher_intended_receiver = (
        (slot == receiver_slot) & coordination_present & row_active
    )

    teacher_receive_runner = receive_runner & row_active

    selected_loft = per_loft[causal_slot]
    selected_cross = per_cross[causal_slot]
    selected_through = per_through[causal_slot]
    pass_family = (
        jnp.where(selected_loft, jnp.uint8(2), jnp.uint8(1))
        | jnp.where(selected_through > 0.02, jnp.uint8(4), jnp.uint8(0))
        | jnp.where(selected_cross, jnp.uint8(8), jnp.uint8(0))
    )
    selected_arrival = per_arrival[causal_slot].astype(jnp.float32)
    coord_trace_valid = coordination_present & (selected_arrival > 0.0)
    coord_release_submitted = (
        coordination_present & do_kick[causal_slot]
    )

    trace = {
        # Direct ACTION_SCENE_SPECS leaves.
        "teacher_carrier_mode": teacher_carrier_mode,
        "teacher_carrier_releasing": teacher_carrier_releasing,
        "teacher_intended_receiver": teacher_intended_receiver,
        "teacher_receive_runner": teacher_receive_runner,
        # Fixed-shape policy-known part of append_scene(coordination=...).
        # The adapter emits a sparse row only when coordination_present=True.
        "coordination_present": coordination_present,
        "coord_passer_slot": jnp.where(
            coordination_present, causal_slot, jnp.int32(-1)
        ).astype(jnp.int16),
        "coord_planned_receiver_slot": jnp.where(
            coordination_present, receiver_slot, jnp.int32(-1)
        ).astype(jnp.int16),
        "coord_pass_family_mask": jnp.where(
            coordination_present, pass_family, jnp.uint8(0)
        ).astype(jnp.uint8),
        "coord_target_xy": jnp.where(
            coordination_present,
            per_target[causal_slot],
            jnp.zeros((2,), dtype=jnp.float32),
        ).astype(jnp.float32),
        "coord_expected_arrival_s": jnp.where(
            coordination_present, selected_arrival, jnp.float32(0.0)
        ).astype(jnp.float32),
        "coord_through_strength": jnp.where(
            coordination_present, selected_through, jnp.float32(0.0)
        ).astype(jnp.float32),
        "coord_release_submitted": coord_release_submitted,
        "coord_trace_valid": coord_trace_valid,
        # No invalid sparse row is emitted for an absent plan.  A present plan
        # is fully computed, so schema-v1's zero = known convention applies.
        "coord_unknown_reason": jnp.uint8(0),
    }
    return action, trace


if __name__ == "__main__":
    import argparse
    import os

    from ..env import SoccerEnv

    parser = argparse.ArgumentParser(description="관측 전용 룰 정책 light 렌더")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--style-a", default="balanced", choices=tuple(STYLE_PRESETS))
    parser.add_argument("--style-b", default="balanced", choices=tuple(STYLE_PRESETS))
    parser.add_argument(
        "--out",
        default=os.path.join("replays", "rule_policy_restart_60s_light.mp4"),
    )
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0.0:
        parser.error("--seconds must be a positive finite number")

    rollout_steps = DEFAULT_TIMEBASE.control_steps_for(args.seconds, minimum=1)
    # 짧은 검증 클립은 시간축 중간에 공수 방향과 킥오프를 삽입하지 않는다.
    env = SoccerEnv(game_duration=rollout_steps, halftime=False)

    policy = make_rule_based_policy(
        env,
        match_key=jax.random.PRNGKey(0),
        team_styles=(args.style_a, args.style_b),
    )
    print("team_styles=\n", np.asarray(policy.team_styles))

    reset_key, rollout_key = jax.random.split(jax.random.PRNGKey(args.seed))
    obs, state = env.reset_array(reset_key)

    def step(carry, k):
        obs, st = carry
        k_pol, k_env = jax.random.split(k)
        act = policy(obs, k_pol, env.affordance_view(st))
        obs2, st2, rew, done, info = env.step_env_array(k_env, st, act)
        metrics = (
            info["restart_position_forced"],
            info["halftime_reset"],
            done,
        )
        return (obs2, st2), (st2, metrics)

    rollout = jax.jit(
        lambda initial_obs, initial_state, keys: jax.lax.scan(
            step, (initial_obs, initial_state), keys
        )
    )
    (obs, state), (states, metrics) = rollout(
        obs,
        state,
        prefix_stable_keys(rollout_key, rollout_steps),
    )
    forced, halftime_reset, done = metrics
    alive_frac = float(jnp.mean(states.ball_state == BALL_ALIVE))
    forced_player_frames = int(jnp.sum(forced))
    forced_frames = int(jnp.sum(jnp.any(forced, axis=1)))
    halftime_frames = int(jnp.sum(jnp.any(halftime_reset, axis=1)))
    if halftime_frames != 0:
        raise RuntimeError("halftime=False rollout emitted a halftime reset")
    if not bool(done[-1]):
        raise RuntimeError("rollout did not terminate at the requested duration")

    out_path = env.render_mp4(
        states,
        out_path=args.out,
        fps=env.control_fps,
        mode="light",
        title=(
            f"Rule policy · {args.style_a} vs {args.style_b} · "
            "halftime off"
        ),
    )
    print(f"{rollout_steps} steps: final score={state.score.tolist()} "
          f"ball-alive frac={alive_frac:.2f} NaN={bool(jnp.isnan(state.ball_pos).any())}")
    print(
        f"restart projection: {forced_frames} frames / "
        f"{forced_player_frames} player-frames; halftime resets={halftime_frames}"
    )
    print(f"light render: {out_path}")
