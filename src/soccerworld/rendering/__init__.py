"""Optional rendering and replay surface.

Importing this module is safe in a core-only installation. Pillow, imageio,
ffmpeg, and matplotlib are loaded only after a render entry point is called.
"""

from soccerworld._engine.render import (
    CANONICAL_EVENTS_SCHEMA,
    DEFAULT_RENDER_FPS,
    RESTART_LABEL,
    Render,
    RichRenderer,
)

__all__ = (
    "CANONICAL_EVENTS_SCHEMA",
    "DEFAULT_RENDER_FPS",
    "RESTART_LABEL",
    "Render",
    "RichRenderer",
)
