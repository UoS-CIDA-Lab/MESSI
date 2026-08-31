"""Public, numerically lightweight access to SoccerWorld's clock contract.

Importing this module does not initialize JAX, the environment, or rendering.  Data
pipelines can therefore validate their sampling rate against the same ``Timebase``
used by ``SoccerEnv`` without acquiring accelerator resources.
"""

from soccerworld._engine.timebase import (
    DEFAULT_MATCH_DURATION_SECONDS,
    DEFAULT_TIMEBASE,
    Timebase,
    duration_to_ticks,
)

__all__ = [
    "DEFAULT_MATCH_DURATION_SECONDS",
    "DEFAULT_TIMEBASE",
    "Timebase",
    "duration_to_ticks",
]
