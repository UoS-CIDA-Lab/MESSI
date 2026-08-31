"""Training-data contracts and host-side sinks."""

from soccerworld.data.capture import CaptureSpec
from soccerworld.data.events import (
    BallBoundaryEventCode,
    DisciplinaryOutcome,
    EventBatch,
    EventDomain,
    FoulEventCode,
    TouchEventCode,
    WoodworkEventCode,
)
from soccerworld.data.records import (
    ActionProvenance,
    DatasetMetadata,
    FrameIdentity,
    ManagerProvenance,
    TransitionRecord,
)
from soccerworld.data.writers import NpzShardSink, RecordSink

__all__ = [
    "ActionProvenance",
    "BallBoundaryEventCode",
    "CaptureSpec",
    "DatasetMetadata",
    "DisciplinaryOutcome",
    "EventBatch",
    "EventDomain",
    "FoulEventCode",
    "FrameIdentity",
    "ManagerProvenance",
    "NpzShardSink",
    "RecordSink",
    "TouchEventCode",
    "TransitionRecord",
    "WoodworkEventCode",
]
