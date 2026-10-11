"""MESSI 환경과 내장 rule-based policy의 공개 API."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

import messi.control_rollout as _control_rollout
import messi.core as _core
import messi.environment as _environment
import messi.policies as _policies
from messi._device_scope import configure_device_scope
from messi.control_rollout import *  # noqa: F403
from messi.core import *  # noqa: F403
from messi.environment import *  # noqa: F403
from messi.policies import *  # noqa: F403


def _resolve_version() -> str:
    try:
        return _distribution_version(__name__)
    except _PackageNotFoundError:
        try:
            import tomllib
            from pathlib import Path

            pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
            with pyproject_path.open("rb") as pyproject_file:
                return str(tomllib.load(pyproject_file)["project"]["version"])
        except (IndexError, KeyError, OSError, TypeError, ValueError):
            return "0+unknown"


__version__ = _resolve_version()

__all__ = sorted(
    {
        "__version__",
        "configure_device_scope",
        *_control_rollout.__all__,
        *_core.__all__,
        *_environment.__all__,
        *_policies.__all__,
    }
)
