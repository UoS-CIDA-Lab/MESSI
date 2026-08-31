"""XLA 영속 컴파일 캐시 — ``import jax`` **전에** 켜야 효과가 있다.

이 환경의 전이 그래프는 HLO 133,781줄 · 약 84,000 op이라 XLA 최적화 패스가 오래 걸린다
(실측: GPU 단일 step 컴파일 60.1 s, 정책 27.2 s). 같은 그래프를 프로세스마다 다시
컴파일할 이유가 없다 — 캐시 키는 소스 파일이 아니라 **HLO 내용 주소**라, 주석만 바꾼
편집은 캐시를 무효화하지 않고 dynamics가 실제로 바뀌면 키가 달라져 stale executable이
재사용되지도 않는다.

지금까지 이 설정은 ``test_exe.py`` 안에만 있어 테스트 러너만 혜택을 봤다. 렌더·측정·
데이터 생성 같은 다른 진입점은 매번 전체 컴파일을 물었다. 그래서 공용 모듈로 뺀다.

    from soccerworld._engine.jax_cache import enable
    enable()            # ← ``import jax`` 보다 먼저
    import jax

``JAX_COMPILATION_CACHE_MAX_SIZE``의 기본값 -1은 **무제한**이라 장기 개발 환경에서 조용히
수 GB로 자란다. 여기서는 2 GiB 상한을 항상 함께 건다. 호출자가 명시한 환경변수는 존중한다.
"""

from __future__ import annotations

import os
import tempfile

DEFAULT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "soccerworld-jax-cache-v1")
DEFAULT_MAX_BYTES = 2 * 1024 ** 3
"""하드 상한(2 GiB). 무제한 기본값을 그대로 두면 캐시가 계속 자란다."""
MIN_COMPILE_TIME_SECS = "1"
"""이보다 짧은 컴파일은 캐시하지 않는다 — 저장 비용이 이득을 넘는다."""


def enable(cache_dir: str | None = None, max_bytes: int | None = None) -> str:
    """영속 캐시를 켜고 사용 중인 디렉터리를 돌려준다.

    ``import jax`` 이후에 불러도 예외는 나지 않지만 그 프로세스에는 적용되지 않는다.
    JAX가 이 환경변수를 import 시점에 읽기 때문이다.
    """

    path = cache_dir or os.environ.get("JAX_COMPILATION_CACHE_DIR") or DEFAULT_CACHE_DIR
    os.makedirs(path, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = path
    os.environ.setdefault(
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", MIN_COMPILE_TIME_SECS
    )
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_MAX_SIZE",
        str(DEFAULT_MAX_BYTES if max_bytes is None else int(max_bytes)),
    )
    return path


def is_enabled() -> bool:
    return bool(os.environ.get("JAX_COMPILATION_CACHE_DIR"))
