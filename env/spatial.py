"""순수 벡터 헬퍼 — 정규화·단위벡터. grad-안전(NaN 차단)."""
import jax.numpy as jnp

from constants import DIV_EPS, SAFE_NORM_EPS

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
    return jnp.sqrt(jnp.sum(v * v, axis=axis, keepdims=keepdims) + eps)

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
