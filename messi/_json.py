"""공개 artifact에 사용하는 strict JSON encode/decode helper."""

from __future__ import annotations

import json
import math
from typing import Any


def _reject_constant(value: str):
    raise ValueError(f"JSON non-finite constant는 허용되지 않는다: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"JSON number가 finite float 범위를 벗어난다: {value}")
    return parsed


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object key가 중복됐다: {key!r}")
        result[key] = value
    return result


def strict_json_dumps(value: Any, **kwargs) -> str:
    """Non-finite number를 거부하면서 caller의 formatting을 보존한다."""

    return json.dumps(value, allow_nan=False, **kwargs)


def strict_json_loads(value: str | bytes | bytearray) -> Any:
    """Non-finite number와 중복 object key를 거부해 JSON을 읽는다."""

    return json.loads(
        value,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
        object_pairs_hook=_unique_object,
    )
