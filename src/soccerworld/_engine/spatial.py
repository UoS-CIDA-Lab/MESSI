"""순수 벡터 헬퍼 — 정규화·단위벡터. grad-안전(NaN 차단)."""
import jax.numpy as jnp

from .constants import DIV_EPS, SAFE_NORM_EPS

def _u01(x: jnp.ndarray) -> jnp.ndarray:
    """값의 범위 정규화 [-1,1] -> [0,1]
    Args:
        x: jnp.ndarray, shape (...,)
    Returns:
        jnp.ndarray, shape (...,)
    """
    return jnp.clip((x + 1.0) * 0.5, 0.0, 1.0)

def _safe_norm(v: jnp.ndarray, axis=-1, keepdims=False, eps=SAFE_NORM_EPS) -> jnp.ndarray:
    """ 벡터의 크기 계산하는 L2 norm. 0 벡터의 경우 grad가 NaN이 되는 문제를 방지하기 위해 eps를 더함.
    Args:
        v: jnp.ndarray, shape (..., D)
        axis: int, axis along which to compute the norm
        keepdims: bool, whether to keep the reduced dimension
        eps: float, small value to determine the minimum norm value
    Returns:
        jnp.ndarray, shape (...,) if keepdims=False else (..., 1)
    """
    # Keep the ordinary expression on the calibrated/default path so this
    # numerical hardening does not perturb existing trajectories bitwise.
    # A public float32 component can nevertheless be perfectly finite while
    # its square is not (e.g. 1e30).  In that case the old expression returned
    # ``inf`` and callers such as ``_unit`` silently collapsed a valid
    # direction to zero.  Re-evaluate only that overflow case after scaling by
    # max(abs(v)); this is the usual overflow-safe BLAS ``nrm2`` construction.
    direct = jnp.sqrt(jnp.sum(v * v, axis=axis, keepdims=keepdims) + eps)
    scale = jnp.max(jnp.abs(v), axis=axis, keepdims=True)
    safe_scale = jnp.where((scale > 0.0) & jnp.isfinite(scale), scale, 1.0)
    scaled_v = v / safe_scale
    scaled = safe_scale * jnp.sqrt(
        jnp.sum(scaled_v * scaled_v, axis=axis, keepdims=True)
    )
    # A vector of finite components can have a mathematical L2 norm above the
    # largest representable scalar.  Saturation keeps downstream ratios and
    # clipping finite without inventing a direction.  Non-finite input is not
    # hidden: ``direct`` remains the result in that case.
    scaled = jnp.minimum(scaled, jnp.finfo(v.dtype).max)
    if not keepdims:
        scaled = jnp.squeeze(scaled, axis=axis)
    finite_input = jnp.all(jnp.isfinite(v), axis=axis, keepdims=keepdims)
    return jnp.where(jnp.isfinite(direct) | (~finite_input), direct, scaled)

def _unit(v: jnp.ndarray) -> jnp.ndarray:
    """ 벡터를 단위벡터로 변환. 0 벡터의 경우 eps를 더해 NaN 방지.
    Args:
        v: jnp.ndarray, shape (..., D)
    Returns:
        jnp.ndarray, shape (..., D)
    """
    n = _safe_norm(v, axis=-1, keepdims=True)
    return v / n

def _linf(v: jnp.ndarray, axis=-1, keepdims=False) -> jnp.ndarray:
    """L∞ 노름(성분 절댓값 최댓값) — L∞ radial stretch 매핑에 사용."""
    return jnp.max(jnp.abs(v), axis=axis, keepdims=keepdims)

def stretch_decode(v: jnp.ndarray):
    """**L∞ radial stretch** — box `[-1,1]²` raw 액션 → (단위방향, 크기비율[0,1]).
    box를 반지름 1 원판으로 방향보존·전단사 사영하는 위상동형: `stretch(v)=v·‖v‖∞/‖v‖₂`.
    분해하면 방향 `unit(v)`, 크기 `‖v‖∞`. `accel = 크기·a_max·방향`으로 쓰면 **등방 원판(모든 방향 최대
    a_max)·전단사(잉여 없음)·방향보존**이 동시에 성립(잉여 극좌표·비등방 Cartesian의 문제를 함께 해소).
    Return: (unit_dir (…,D), frac (…,))"""
    l2 = _safe_norm(v, axis=-1, keepdims=True)
    return v / l2, _linf(v, axis=-1)

def stretch_encode(unit_dir: jnp.ndarray, frac: jnp.ndarray) -> jnp.ndarray:
    """`stretch_decode`의 역 — (단위방향 (…,D), 크기비율 frac (…,)) → raw box 액션 v.
    `unit(v)=unit_dir`, `‖v‖∞=frac`을 만족(범위 내면 box 안, 방향 정확). 역산·정책 조립에서 사용."""
    linf = _linf(unit_dir, axis=-1)                            # (…,)
    return unit_dir * (frac / (linf + DIV_EPS))[..., None]


def reach_blockable(ball_speed_xy, ball_z, block_limit, height_penalty):
    """reach 안의 공을 **의도적으로 제어할 수 있는가**의 이진 판정 — env·정책 공용 SSOT.

    실측(DFL 127경기, reach 반경 통과 기회 9,569건)에서 차단률은 높이와 진입속도 어느
    한쪽이 아니라 둘의 **선형 결합**으로 갈렸다. 지면을 구르는 공도 26 m/s를 넘으면
    막히지 않고, 가슴 높이면 그보다 훨씬 낮은 속도에서 이미 막히지 않는다. 높이만 보는
    규칙은 0.869, 이 결합은 0.914를 맞춘다.

    env의 접촉 판정과 정책의 인터셉트 예측이 **반드시 같은 선**을 써야 한다. 다르면
    정책이 env가 허용하지 않는 공에 주자를 붙이거나, 잡을 수 있는 공을 포기한다.

    몸통 충돌은 이 게이트를 거치지 않는다 — "발 뻗어 잡기"만 막고 "몸으로 막기"는 남는다.
    """

    return ball_speed_xy + height_penalty * ball_z <= block_limit
