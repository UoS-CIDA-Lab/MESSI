"""Optional replay rendering API.

Importing this package does not import Matplotlib, imageio, or ffmpeg.  Those
libraries are loaded only when rendering begins.
"""

from footballworld.rendering.capture import (
    EventMatchRenderResult,
    ManagedEventMatchRenderResult,
    render_event_match,
    render_managed_event_match,
)
from footballworld.rendering.defaults import DEFAULT_RENDER_FPS, RenderStyle
from footballworld.rendering.events import EventStream, open_events
from footballworld.rendering.publication import PublishedReplay, open_published_replay
from footballworld.rendering.renderer import RenderResult, ReplayRenderer, render_mp4
from footballworld.rendering.replay import open_tracking, open_tracking_jsonl
from footballworld.rendering.sparse_events import (
    restore_sparse_action_trace,
    restore_sparse_frame_events,
)
from footballworld.rendering.tracking import export_tracking_jsonl
from footballworld.rendering.window import ReplayWindow

__all__ = [
    "DEFAULT_RENDER_FPS",
    "EventMatchRenderResult",
    "EventStream",
    "ManagedEventMatchRenderResult",
    "PublishedReplay",
    "RenderResult",
    "RenderStyle",
    "ReplayRenderer",
    "ReplayWindow",
    "export_tracking_jsonl",
    "open_events",
    "open_published_replay",
    "open_tracking",
    "open_tracking_jsonl",
    "render_event_match",
    "render_managed_event_match",
    "render_mp4",
    "restore_sparse_action_trace",
    "restore_sparse_frame_events",
]
