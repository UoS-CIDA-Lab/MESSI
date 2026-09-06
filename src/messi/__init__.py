"""MESSI — A Multi-Agent Environment for Soccer Simulation and Intelligence.

Alias package: ``import messi`` re-exports the ``soccerworld`` public API so that both
names work. New code may use either; the engine itself lives in ``soccerworld``.
"""
from soccerworld import *  # noqa: F401,F403
from soccerworld import __all__, __version__  # noqa: F401
