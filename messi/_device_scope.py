"""공유 서버에서 JAX가 암묵적으로 모든 GPU를 보지 않게 하는 process scope."""

from __future__ import annotations

import os
import re
from collections.abc import MutableMapping

_NUMBERED_GPU_LIST = re.compile(r"[0-9]+(?:,[0-9]+)*")


def configure_device_scope(
    environment: MutableMapping[str, str] | None = None,
) -> str:
    """번호가 지정되지 않은 process를 CPU로 제한한다.

    GPU 실행은 ``CUDA_VISIBLE_DEVICES``에 쉼표로 구분한 숫자 ID를 명시한 경우에만
    허용한다. 빈 값과 ``-1``은 명시적인 GPU 비활성화로 취급한다.
    """

    environ = os.environ if environment is None else environment
    platforms = environ.get("JAX_PLATFORMS", "").strip()
    if platforms == "cpu":
        return "cpu"

    visible_devices = environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None or not visible_devices.strip():
        environ["JAX_PLATFORMS"] = "cpu"
        return "cpu"
    visible_devices = visible_devices.strip()
    if visible_devices == "-1":
        environ["JAX_PLATFORMS"] = "cpu"
        return "cpu"
    if _NUMBERED_GPU_LIST.fullmatch(visible_devices) is None:
        raise RuntimeError(
            "GPU 실행에는 CUDA_VISIBLE_DEVICES=0 또는 0,1처럼 숫자 ID를 "
            "명시해야 한다"
        )
    return "gpu"


__all__ = ["configure_device_scope"]

configure_device_scope()
