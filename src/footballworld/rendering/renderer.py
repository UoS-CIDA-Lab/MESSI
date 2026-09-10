"""Fast production MP4 replay renderer."""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import subprocess
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from footballworld.config.contact_timing import ContactTiming
from footballworld.config.geometry import Ball, Stadium
from footballworld.config.perception import Perception
from footballworld.config.reach import Reach
from footballworld.core.constants import (
    BALL_EVENT_GOAL,
    DISCIPLINE_NONE,
    DISCIPLINE_RED,
    DISCIPLINE_YELLOW,
    RK_CORNER,
    RK_FREEKICK,
    RK_GK_HOLD,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_NONE,
    RK_OFFSIDE,
    RK_PENALTY,
    RK_THROWIN,
)
from footballworld.core.contact import (
    INTENT_CHALLENGE,
    INTENT_CLEAR,
    INTENT_CONTROL,
    INTENT_MOVE,
    INTENT_PASS,
    INTENT_SHOT,
    MECHANISM_HEAD,
)
from footballworld.core.timebase import DEFAULT_TIMEBASE
from footballworld.rendering.defaults import DEFAULT_RENDER_FPS, RenderStyle
from footballworld.rendering.integrity import (
    artifact_receipt,
    write_completion_manifest,
)
from footballworld.rendering.receipt import render_settings_receipt
from footballworld.rendering.replay import (
    _transition_identity_arrays,
    write_replay_sidecars,
)
from footballworld.rendering.transfer import HostFrame, prepare_host_frames
from footballworld.rules.foul import (
    OFFENCE_DANGEROUS_PLAY,
    OFFENCE_DELIBERATE_GK_TRICK,
    OFFENCE_DIRECT_CONTACT,
    OFFENCE_GK_ILLEGAL_HANDLING,
    OFFENCE_HANDBALL,
    OFFENCE_HOLD,
    OFFENCE_IMPEDE_CONTACT,
    OFFENCE_IMPEDE_NO_CONTACT,
    OFFENCE_OTHER_INDIRECT,
    OFFENCE_PREVENT_GK_RELEASE,
    OFFENCE_RESTART_SECOND_TOUCH,
    OFFENCE_THROW_OBJECT,
    OFFENCE_VIOLENT_CONDUCT,
    SEVERITY_CARELESS,
    SEVERITY_EXCESSIVE_FORCE,
    SEVERITY_NONE,
    SEVERITY_RECKLESS,
    TACTICAL_DOGSO,
    TACTICAL_NONE,
    TACTICAL_SPA,
)

TEAM_COLORS = ("#ee4b5f", "#3178e8")
GK_COLORS = ("#ffad33", "#35d0ba")
TEAM_LABELS = ("HOME", "AWAY")
TEAM_COLOR_ARRAY = np.asarray(TEAM_COLORS, dtype=object)
GK_COLOR_ARRAY = np.asarray(GK_COLORS, dtype=object)
SCOREBOARD_SCORE_Y = 0.965
SCOREBOARD_CLOCK_Y = 0.915

_MARKER_POLYGON_SIDES = 32
_FOV_FAN_INNER_M = 2.00
_FOV_FAN_OUTER_M = 3.00
_FOV_FAN_ALPHA = 0.23
_FOV_FAN_SAMPLES = 9
_MINIMAP_RECT = (0.765, 0.035, 0.215, 0.205)
_PLAYER_MARKER_HEIGHT_M = 0.72
_PLAYER_HUD_HEIGHT_M = 1.85
_AERIAL_LIFT_M = 0.78
_STAMINA_BAR_HALF_WIDTH_PX = 7.0
_STAMINA_BAR_OFFSET_PX = 6.0
_STAMINA_BAR_HALF_GAP_PX = 2.25
_STAMINA_BAR_RAIL_WIDTH_PT = 4.0
_STAMINA_BAR_FILL_WIDTH_PT = 2.8
_STAMINA_BAR_RAIL_COLOR = "#071019"
_STAMINA_LONG_COLOR = "#31e36c"
_STAMINA_SHORT_COLOR = "#39d9ff"
_MIN_PARALLEL_RENDER_FRAMES = 64
_ASYNC_FRAME_BUFFER_COUNT = 2
_INTENT_RING_TURF_HEIGHT_M = np.float32(0.0)
_INTENT_RING_UNDERLAY_ZORDER = 6.0
_INTENT_RING_ZORDER = 6.2
_PLAYER_MARKER_ZORDER = 7.0


@dataclass(frozen=True, slots=True)
class _InteractionOverlayGeometry:
    ordinary_ring_radius_m: float
    challenge_ring_radius_m: float
    goalkeeper_control_ring_radius_m: float
    fov_inner_radius_m: float
    fov_outer_radius_m: float


def _interaction_overlay_geometry(
    reach: Reach,
    ball_radius_m: float,
) -> _InteractionOverlayGeometry:
    """Derive host-only overlays from authoritative horizontal reach envelopes."""

    ball_radius = float(ball_radius_m)
    radii = np.asarray(
        (
            float(reach.carry_radius_m) + ball_radius,
            float(reach.challenge_radius_m) + ball_radius,
            float(reach.goalkeeper_radius_m) + ball_radius,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(radii).all() or np.any(radii <= 0.0):
        raise ValueError("interaction overlay radii must be finite and positive")
    ordinary, challenge, goalkeeper = (float(value) for value in radii)
    if challenge < ordinary:
        raise ValueError(
            "challenge ring radius must not be smaller than ordinary reach"
        )
    return _InteractionOverlayGeometry(
        ordinary_ring_radius_m=ordinary,
        challenge_ring_radius_m=challenge,
        goalkeeper_control_ring_radius_m=goalkeeper,
        fov_inner_radius_m=_FOV_FAN_INNER_M,
        fov_outer_radius_m=_FOV_FAN_OUTER_M,
    )


def _intent_ring_radii(
    requested_intent: np.ndarray,
    is_goalkeeper: np.ndarray,
    geometry: _InteractionOverlayGeometry,
) -> np.ndarray:
    """Map each requested action to its maximum horizontal ball-interaction reach."""

    intent = np.asarray(requested_intent)
    goalkeeper = np.asarray(is_goalkeeper, dtype=bool)
    radius = np.where(
        intent == INTENT_CHALLENGE,
        geometry.challenge_ring_radius_m,
        geometry.ordinary_ring_radius_m,
    )
    return np.where(
        goalkeeper & (intent == INTENT_CONTROL),
        geometry.goalkeeper_control_ring_radius_m,
        radius,
    ).astype(np.float32)


def _intent_ring_world_vertices(
    player_position: np.ndarray,
    ring_radius: np.ndarray,
    ring_unit: np.ndarray,
) -> np.ndarray:
    """Return flat turf-plane circles for every player's intent envelope."""

    ring_xy = (
        player_position[:, None, :] + ring_radius[:, None, None] * ring_unit[None, :, :]
    )
    return np.concatenate(
        (
            ring_xy,
            np.full(
                (*ring_xy.shape[:2], 1),
                _INTENT_RING_TURF_HEIGHT_M,
                dtype=np.float32,
            ),
        ),
        axis=2,
    )


def _player_intent_painter_order(
    field_of_view_fan: Any,
    player_shadow: Any,
    intent_ring_underlay: Any,
    intent_rings: Any,
    players: Any,
) -> tuple[Any, ...]:
    """Return back-to-front field cues so player spheres occlude intent rings."""

    return (
        field_of_view_fan,
        player_shadow,
        intent_ring_underlay,
        intent_rings,
        players,
    )


_RENDER_SPAWN_ENV_LOCK = threading.Lock()


def _team_label(team: int) -> str:
    """Return a presentation label without exposing the internal team index."""

    return TEAM_LABELS[team] if team in (0, 1) else "UNKNOWN"


INTENT_COLORS = {
    INTENT_CONTROL: "#00d5c8",
    INTENT_PASS: "#ffd166",
    INTENT_SHOT: "#ff7f0e",
    INTENT_CLEAR: "#b388ff",
    INTENT_CHALLENGE: "#f03b73",
}

_OFFENCE_NAMES = {
    OFFENCE_DIRECT_CONTACT: "DIRECT CONTACT",
    OFFENCE_HANDBALL: "HANDBALL",
    OFFENCE_HOLD: "HOLDING",
    OFFENCE_IMPEDE_CONTACT: "IMPEDING WITH CONTACT",
    OFFENCE_THROW_OBJECT: "THROWING AN OBJECT",
    OFFENCE_DANGEROUS_PLAY: "DANGEROUS PLAY",
    OFFENCE_IMPEDE_NO_CONTACT: "IMPEDING WITHOUT CONTACT",
    OFFENCE_PREVENT_GK_RELEASE: "PREVENTING GK RELEASE",
    OFFENCE_GK_ILLEGAL_HANDLING: "ILLEGAL GK HANDLING",
    OFFENCE_RESTART_SECOND_TOUCH: "RESTART SECOND TOUCH",
    OFFENCE_DELIBERATE_GK_TRICK: "DELIBERATE GK TRICK",
    OFFENCE_OTHER_INDIRECT: "OTHER INDIRECT OFFENCE",
    OFFENCE_VIOLENT_CONDUCT: "VIOLENT CONDUCT",
}
_SEVERITY_NAMES = {
    SEVERITY_NONE: "SEVERITY: NONE",
    SEVERITY_CARELESS: "CARELESS",
    SEVERITY_RECKLESS: "RECKLESS",
    SEVERITY_EXCESSIVE_FORCE: "EXCESSIVE FORCE",
}
_TACTICAL_NAMES = {
    TACTICAL_NONE: "TACTICAL: NONE",
    TACTICAL_SPA: "SPA",
    TACTICAL_DOGSO: "DOGSO",
}
_DISCIPLINE_NAMES = {
    DISCIPLINE_NONE: "NO CARD",
    DISCIPLINE_YELLOW: "YELLOW CARD",
    DISCIPLINE_RED: "RED CARD",
}
_RESTART_NAMES = {
    RK_NONE: "NO RESTART",
    RK_KICKOFF: "KICK-OFF",
    RK_THROWIN: "THROW-IN",
    RK_GOALKICK: "GOAL KICK",
    RK_CORNER: "CORNER",
    RK_FREEKICK: "FREE KICK",
    RK_PENALTY: "PENALTY",
    RK_OFFSIDE: "OFFSIDE FREE KICK",
    RK_GK_HOLD: "GK HOLD",
}

warnings.filterwarnings(
    "ignore", message=r".*os\.fork\(\) was called.*", category=RuntimeWarning
)


@contextmanager
def _renderer_spawn_environment():
    """Keep host-only render workers from claiming an accelerator.

    ``spawn`` imports the caller's main module before unpickling the render
    target. FootballWorld applications normally import JAX there, even though
    the renderer itself is NumPy/Agg-only. Make that import select the CPU in
    children; an already initialized parent backend is unaffected.
    """

    with _RENDER_SPAWN_ENV_LOCK:
        variable = "JAX_PLATFORMS"
        missing = object()
        previous: str | object = os.environ.get(variable, missing)
        os.environ[variable] = "cpu"
        try:
            yield
        finally:
            if previous is missing:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous


class _PerspectiveCamera:
    """Fixed broadcast-style pinhole camera over world-space metres."""

    def __init__(self, style: RenderStyle) -> None:
        azimuth = np.deg2rad(style.camera_azimuth_degrees)
        elevation = np.deg2rad(style.camera_elevation_degrees)
        direction = np.asarray(
            (
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ),
            dtype=np.float64,
        )
        self.position = style.camera_distance_m * direction
        forward = -direction
        right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        self.rotation = np.stack((right, up, forward))
        self.focal_length = float(style.camera_focal_length)
        self.reference_depth = float(style.camera_distance_m)

    def _camera_space(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=np.float64) - self.position) @ self.rotation.T

    def project(self, points: np.ndarray) -> np.ndarray:
        camera = self._camera_space(points)
        depth = np.clip(camera[..., 2], 1e-3, None)
        return np.stack(
            (
                self.focal_length * camera[..., 0] / depth,
                self.focal_length * camera[..., 1] / depth,
            ),
            axis=-1,
        )

    def relative_scale(self, points: np.ndarray) -> np.ndarray:
        depth = np.clip(self._camera_space(points)[..., 2], 1e-3, None)
        return np.clip(self.reference_depth / depth, 0.62, 1.55)

    def project_with_relative_scale(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project points and reuse their camera-space depth for marker scale."""

        camera = self._camera_space(points)
        depth = np.clip(camera[..., 2], 1e-3, None)
        projected = np.stack(
            (
                self.focal_length * camera[..., 0] / depth,
                self.focal_length * camera[..., 1] / depth,
            ),
            axis=-1,
        )
        scale = np.clip(self.reference_depth / depth, 0.62, 1.55)
        return projected, scale


@dataclass(frozen=True, slots=True)
class RenderResult:
    video: Path
    event: Path
    tracking: Path
    metadata: Path
    frames: int
    seconds: float
    throughput_fps: float
    workers: int
    video_generated: bool = True


@dataclass(frozen=True, slots=True)
class _Adjudication:
    """One exact high-value decision held briefly by the host renderer."""

    origin_control_tick: int
    title: str
    detail: str
    accent: str


@dataclass(frozen=True, slots=True)
class _VisualFrame:
    """Renderer-only host payload, excluding replay and observation trees."""

    control_tick: int
    video_time_s: float | None
    first_half_wall_end_tick: int
    ball_position: np.ndarray
    ball_live: bool
    player_position: np.ndarray
    player_body_forward: np.ndarray
    player_gaze_yaw: np.ndarray
    aerial_progress: np.ndarray
    high_head_contact: np.ndarray
    team_id: np.ndarray
    is_goalkeeper: np.ndarray
    active: np.ndarray
    sent_off: np.ndarray
    stamina_long: np.ndarray
    stamina_short: np.ndarray
    yellow_cards: np.ndarray
    possession_player: int
    score: np.ndarray
    requested_intent: np.ndarray
    action_executed: bool
    adjudication: _Adjudication | None


def _named(code: int, names: dict[int, str], family: str) -> str:
    """Render a code without inventing semantics for future values."""

    return names.get(int(code), f"{family} {int(code)}")


def _transition_actor_label(frame: HostFrame, slot: int) -> str:
    """Name an event actor using the identity that executed the transition."""

    player_ids, _, _, _ = _transition_identity_arrays(frame)
    if slot < 0 or slot >= player_ids.size:
        return "NO PLAYER"
    return f"PLAYER {int(player_ids[slot])} (#{slot + 1})"


def _frame_adjudication(frame: HostFrame) -> _Adjudication | None:
    """Select one exact high-value decision from this transition's events.

    This is not a historical event feed: ordinary contacts and requested
    intents remain in event.json. Titles and details use only leaves emitted
    by step_with_events. Unknown future enum values stay visible as codes.
    """

    tick = int(frame.control_tick)
    candidates: list[tuple[int, int, _Adjudication]] = []
    substitution: _Adjudication | None = None
    substitution_events = frame.substitution_events
    if substitution_events is not None:
        occurred = np.asarray(substitution_events.occurred, dtype=bool)
        team = np.asarray(substitution_events.team, dtype=np.int32)
        outgoing = np.asarray(substitution_events.outgoing_player_id, dtype=np.int64)
        incoming = np.asarray(substitution_events.incoming_player_id, dtype=np.int64)
        if not (
            team.shape == occurred.shape
            and outgoing.shape == occurred.shape
            and incoming.shape == occurred.shape
        ):
            raise ValueError("substitution event fields must share one shape")
        rows = [tuple(index) for index in np.argwhere(occurred)]
        if occurred.ndim == 0 and bool(occurred):
            rows = [()]
        if rows:
            details = [
                (
                    f"{_team_label(int(team[index]))}: "
                    f"P{int(outgoing[index])} OUT → P{int(incoming[index])} IN"
                )
                for index in rows
            ]
            represented_teams = {int(team[index]) for index in rows}
            accent = (
                TEAM_COLORS[next(iter(represented_teams))]
                if len(represented_teams) == 1
                and next(iter(represented_teams)) in (0, 1)
                else "#f4f4f4"
            )
            # Make bench transitions visible from the exact host-only manager
            # sidecar without restoring bench arrays to JAX State.
            substitution = _Adjudication(
                origin_control_tick=tick,
                title="SUBSTITUTION",
                detail=" · ".join(details),
                accent=accent,
            )
    events = frame.frame_events
    if events is None:
        return substitution

    boundary = events.boundary
    occurred = np.asarray(boundary.occurred, dtype=bool).reshape(-1)
    kind = np.asarray(boundary.kind, dtype=np.int32).reshape(-1)
    scoring_team = np.asarray(boundary.scoring_team, dtype=np.int32).reshape(-1)
    for substep in np.flatnonzero(occurred & (kind == BALL_EVENT_GOAL)):
        team = int(scoring_team[substep])
        candidates.append(
            (
                6,
                int(substep),
                _Adjudication(
                    origin_control_tick=tick,
                    title=f"GOAL — {_team_label(team)}",
                    detail=f"SCORE {int(frame.score[0])} : {int(frame.score[1])}",
                    accent=TEAM_COLORS[team] if team in (0, 1) else "#ffffff",
                ),
            )
        )

    foul = events.foul
    foul_occurred = np.asarray(foul.occurred, dtype=bool).reshape(-1)
    foul_fields = {
        name: np.asarray(getattr(foul, name)).reshape(-1)
        for name in (
            "contest_source",
            "offender",
            "victim",
            "offender_team",
            "offence_type",
            "severity",
            "tactical_effect",
            "restart_kind",
            "discipline",
            "advantage_applied",
        )
    }
    for substep in np.flatnonzero(foul_occurred):
        offender_team = int(foul_fields["offender_team"][substep])
        offender = int(foul_fields["offender"][substep])
        victim = int(foul_fields["victim"][substep])
        discipline = int(foul_fields["discipline"][substep])
        source = (
            "CONTEST" if bool(foul_fields["contest_source"][substep]) else "NON-CONTEST"
        )
        consequence = (
            "ADVANTAGE"
            if bool(foul_fields["advantage_applied"][substep])
            else _named(
                int(foul_fields["restart_kind"][substep]),
                _RESTART_NAMES,
                "RESTART",
            )
        )
        candidates.append(
            (
                5 if discipline == DISCIPLINE_RED else 4,
                int(substep),
                _Adjudication(
                    origin_control_tick=tick,
                    title=(
                        f"FOUL — {_team_label(offender_team)} · "
                        f"{_transition_actor_label(frame, offender)} ON "
                        f"{_transition_actor_label(frame, victim)}"
                    ),
                    detail=" · ".join(
                        (
                            _named(
                                int(foul_fields["offence_type"][substep]),
                                _OFFENCE_NAMES,
                                "OFFENCE",
                            ),
                            source,
                            _named(
                                int(foul_fields["severity"][substep]),
                                _SEVERITY_NAMES,
                                "SEVERITY",
                            ),
                            _named(
                                int(foul_fields["tactical_effect"][substep]),
                                _TACTICAL_NAMES,
                                "TACTICAL",
                            ),
                            _named(discipline, _DISCIPLINE_NAMES, "DISCIPLINE"),
                            consequence,
                        )
                    ),
                    accent=("#ff3b3b" if discipline == DISCIPLINE_RED else "#ffd166"),
                ),
            )
        )

    offside = events.offside
    offside_occurred = np.asarray(offside.occurred, dtype=bool).reshape(-1)
    offside_actor = np.asarray(offside.actor, dtype=np.int32).reshape(-1)
    offside_team = np.asarray(offside.team, dtype=np.int32).reshape(-1)
    for substep in np.flatnonzero(offside_occurred):
        team = int(offside_team[substep])
        actor = int(offside_actor[substep])
        candidates.append(
            (
                3,
                int(substep),
                _Adjudication(
                    origin_control_tick=tick,
                    title=(
                        f"OFFSIDE — {_team_label(team)} · "
                        f"{_transition_actor_label(frame, actor)}"
                    ),
                    detail="FLAGGED PLAYER BECAME INVOLVED",
                    accent="#f4f4f4",
                ),
            )
        )

    if not candidates:
        return substitution
    selected = max(candidates, key=lambda item: (item[0], item[1]))[2]
    if substitution is not None:
        selected = replace(
            selected,
            detail=f"{selected.detail} · {substitution.title}: {substitution.detail}",
        )
    return selected


def _carry_adjudication(
    frame: _VisualFrame,
    previous: _Adjudication | None,
    *,
    control_fps: float,
    duration_seconds: float,
) -> tuple[_VisualFrame, _Adjudication | None]:
    """Hold a decision by control tick so render chunks cannot cut it short."""

    current = frame.adjudication
    if current is not None:
        return frame, current
    if previous is None:
        return frame, None
    age_ticks = frame.control_tick - previous.origin_control_tick
    if age_ticks < 0:
        raise ValueError("visual control_tick must be monotonic")
    if age_ticks < duration_seconds * control_fps:
        return replace(frame, adjudication=previous), previous
    return frame, None


def _propagate_adjudications(
    frames: list[_VisualFrame],
    *,
    control_fps: float,
    duration_seconds: float,
) -> list[_VisualFrame]:
    """Precompute one bounded host-only caption reference per visual frame."""

    previous: _Adjudication | None = None
    propagated: list[_VisualFrame] = []
    for frame in frames:
        frame, previous = _carry_adjudication(
            frame,
            previous,
            control_fps=control_fps,
            duration_seconds=duration_seconds,
        )
        propagated.append(frame)
    return propagated


def _high_head_contact_mask(frame: HostFrame, *, ball_radius_m: float) -> np.ndarray:
    """Locate realized above-stature headers from exact contact sidecars."""
    mask = np.zeros(frame.player_position.shape[0], dtype=bool)
    events = frame.frame_events
    if events is None:
        return mask

    deliberate = events.deliberate_contact
    contacts = events.contacts
    deliberate_actor = np.asarray(deliberate.actor, dtype=np.int32).reshape(-1)
    deliberate_mechanism = np.asarray(deliberate.mechanism, dtype=np.int32).reshape(-1)
    occurred = np.asarray(contacts.occurred, dtype=bool)
    actor = np.asarray(contacts.actor, dtype=np.int32)
    mechanism = np.asarray(contacts.mechanism, dtype=np.int32)
    position = np.asarray(contacts.position, dtype=np.float32)
    if occurred.ndim == 1:
        occurred = occurred[None, :]
        actor = actor[None, :]
        mechanism = mechanism[None, :]
        position = position[None, :, :]
    if occurred.shape[0] != deliberate_actor.size:
        raise ValueError("contact sidecar substep axes do not match")

    player_count = mask.size
    player_height = np.asarray(frame.player_height, dtype=np.float32)
    identity = frame.pre_management_identity
    if isinstance(identity, dict) and "player_height" in identity:
        pre_management_height = np.asarray(identity["player_height"], dtype=np.float32)
        if pre_management_height.shape != player_height.shape:
            raise ValueError("pre-management player_height shape does not match frame")
        player_height = pre_management_height
    safe_actor = np.clip(actor, 0, max(player_count - 1, 0))
    valid_actor = (actor >= 0) & (actor < player_count)
    deliberate_head = (
        (deliberate_actor >= 0)
        & (deliberate_actor < player_count)
        & (deliberate_mechanism == MECHANISM_HEAD)
    )
    realized_head = (
        occurred
        & valid_actor
        & (mechanism == MECHANISM_HEAD)
        & deliberate_head[:, None]
        & (actor == deliberate_actor[:, None])
    )
    # The contact engine measures aerial effort from the ball's near surface,
    # not its centre.  Use the same geometry so the gold jump cue cannot claim
    # a recovery-producing header when the engine installed no aerial lock.
    above_stature = (
        position[..., 2] - np.float32(ball_radius_m) > player_height[safe_actor]
    )
    qualifying = realized_head & above_stature
    np.logical_or.at(mask, actor[qualifying], True)
    return mask


def _visual_frame(
    frame: HostFrame,
    *,
    max_outfield_aerial_recovery_substeps: int,
    max_goalkeeper_aerial_recovery_substeps: int,
    ball_radius_m: float,
) -> _VisualFrame:
    recovery_scale = np.where(
        frame.is_goalkeeper,
        max_goalkeeper_aerial_recovery_substeps,
        max_outfield_aerial_recovery_substeps,
    )
    aerial_progress = np.clip(
        np.asarray(frame.aerial_recovery_substeps, dtype=np.float32) / recovery_scale,
        0.0,
        1.0,
    )
    if frame.action_trace is None:
        requested_intent = np.full(
            frame.player_position.shape[0], INTENT_MOVE, dtype=np.int32
        )
        action_executed = False
    else:
        requested_intent = np.asarray(
            frame.action_trace.requested_intent, dtype=np.int32
        )
        if requested_intent.shape != (frame.player_position.shape[0],):
            raise ValueError("requested intent must have one value per player")
        action_executed = bool(np.asarray(frame.action_trace.executed))
    return _VisualFrame(
        control_tick=frame.control_tick,
        video_time_s=frame.video_time_s,
        first_half_wall_end_tick=frame.first_half_wall_end_tick,
        ball_position=frame.ball_position,
        ball_live=frame.ball_live,
        player_position=frame.player_position,
        player_body_forward=frame.player_body_forward,
        player_gaze_yaw=frame.player_gaze_yaw,
        aerial_progress=aerial_progress,
        high_head_contact=_high_head_contact_mask(frame, ball_radius_m=ball_radius_m),
        team_id=frame.team_id,
        is_goalkeeper=frame.is_goalkeeper,
        active=frame.active,
        sent_off=frame.sent_off,
        stamina_long=frame.stamina_long,
        stamina_short=frame.stamina_short,
        yellow_cards=frame.yellow_cards,
        possession_player=frame.possession_player,
        score=frame.score,
        requested_intent=requested_intent,
        action_executed=action_executed,
        adjudication=_frame_adjudication(frame),
    )


def _dependencies():
    try:
        import imageio.v2 as imageio
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection, PolyCollection
        from matplotlib.patches import Arc, Circle, Rectangle
        from matplotlib.path import Path as MplPath
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Replay rendering requires matplotlib, imageio, Pillow, and "
            "imageio-ffmpeg. Install FootballWorld's render dependencies."
        ) from exc
    return (
        imageio,
        plt,
        LineCollection,
        PolyCollection,
        MplPath,
        Arc,
        Circle,
        Rectangle,
    )


def _flatten_circle_markers(mpl_path: Any, *collections: Any) -> None:
    """Replace Bézier circle markers with visually equivalent 32-gons once.

    Agg repeatedly solves curved marker extrema while drawing scatter
    collections. A 32-sided marker differs from the original radius by less
    than a pixel fraction at replay sizes and makes those extents a cheap
    polygon calculation. Equal area keeps player, ball, shadow, intent, and
    aerial-effect sizes visually stable.
    """

    sides = _MARKER_POLYGON_SIDES
    angle = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    equal_area_scale = np.sqrt(2.0 * np.pi / (sides * np.sin(2.0 * np.pi / sides)))
    unit = np.column_stack((np.cos(angle), np.sin(angle))) * equal_area_scale
    for collection in collections:
        paths = collection.get_paths()
        if len(paths) != 1 or paths[0].codes is None:
            continue
        if mpl_path.CURVE4 not in paths[0].codes:
            continue
        extent = paths[0].get_extents()
        center_x = extent.x0 + 0.5 * extent.width
        center_y = extent.y0 + 0.5 * extent.height
        vertices = np.column_stack(
            (
                center_x + 0.5 * extent.width * unit[:, 0],
                center_y + 0.5 * extent.height * unit[:, 1],
            )
        )
        collection.set_paths([mpl_path(vertices, closed=True)])


def _update_line_vertices(collection: Any, vertices: np.ndarray) -> None:
    """Update fixed-shape LineCollection paths without rebuilding Path objects."""

    values = np.asarray(vertices, dtype=np.float64)
    shape = values.shape
    if getattr(collection, "_footballworld_vertex_shape", None) != shape:
        collection.set_segments(values)
        collection._footballworld_vertex_shape = shape
        return
    paths = collection.get_paths()
    if len(paths) != values.shape[0] or any(
        path.vertices.shape != values[index].shape for index, path in enumerate(paths)
    ):
        collection.set_segments(values)
        return
    for path, row in zip(paths, values, strict=True):
        path.vertices[...] = row
    collection.stale = True


def _update_polygon_vertices(collection: Any, vertices: np.ndarray) -> None:
    """Update closed, fixed-shape PolyCollection paths in place."""

    values = np.asarray(vertices, dtype=np.float64)
    shape = values.shape
    if getattr(collection, "_footballworld_vertex_shape", None) != shape:
        collection.set_verts(values)
        collection._footballworld_vertex_shape = shape
        return
    paths = collection.get_paths()
    expected_points = values.shape[1] + 1
    if len(paths) != values.shape[0] or any(
        path.vertices.shape != (expected_points, 2) for path in paths
    ):
        collection.set_verts(values)
        return
    for path, row in zip(paths, values, strict=True):
        path.vertices[:-1] = row
        path.vertices[-1] = row[0]
    collection.stale = True


def _install_renderer_font_path_cache(renderer: Any) -> bool:
    """Cache stable Agg font-path resolution for one render segment.

    Every text draw asks Matplotlib to resolve an immutable ``FontProperties``
    object to the same font-file tuple. Keep that tuple on this RendererAgg
    instance while preserving Agg's normal font loading, clearing, sizing,
    hinting, and rasterization. The identity cache owns a strong reference to
    each property, so Python cannot reuse an identity for a different object.

    Matplotlib's backend hooks are private and may change after the supported
    minimum version. Fail closed to the original implementation when the
    expected hooks are unavailable.
    """

    try:
        from matplotlib.backends import backend_agg

        manager = backend_agg._fontManager
        find_fonts = manager._find_fonts_by_props
        get_font = backend_agg.get_font
        original_prepare = renderer._prepare_font
    except (AttributeError, ImportError):
        return False

    resolved: dict[int, tuple[Any, tuple[str, ...]]] = {}
    shared_paths: dict[tuple[Any, ...], tuple[str, ...]] = {}

    def prepare_font(font_properties: Any) -> Any:
        identity = id(font_properties)
        cached = resolved.get(identity)
        if cached is None or cached[0] is not font_properties:
            try:
                property_key = (
                    font_properties.get_file(),
                    tuple(font_properties.get_family()),
                    font_properties.get_style(),
                    font_properties.get_variant(),
                    font_properties.get_weight(),
                    font_properties.get_stretch(),
                    font_properties.get_size_in_points(),
                )
                paths = shared_paths.get(property_key)
                if paths is None:
                    paths = tuple(find_fonts(font_properties))
                    shared_paths[property_key] = paths
            except (AttributeError, TypeError, ValueError):
                return original_prepare(font_properties)
            cached = (font_properties, paths)
            resolved[identity] = cached
        font = get_font(cached[1])
        font.clear()
        font.set_size(font_properties.get_size_in_points(), renderer.dpi)
        return font

    renderer._prepare_font = prepare_font
    return True


class _AsyncWriter:
    """Overlap Agg rendering with ffmpeg pipe writes."""

    def __init__(
        self,
        path: Path,
        fps: float,
        style: RenderStyle,
        *,
        faststart: bool = True,
    ):
        imageio, *_ = _dependencies()
        threads = max(1, min(style.encoder_threads, os.cpu_count() or 1))
        output_params = [
            "-preset",
            style.encoder_preset,
            "-crf",
            str(style.crf),
            "-threads",
            str(threads),
        ]
        if faststart:
            output_params.extend(("-movflags", "+faststart"))
        self._writer = imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            quality=None,
            pixelformat="yuv420p",
            macro_block_size=None,
            output_params=output_params,
        )
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=_ASYNC_FRAME_BUFFER_COUNT)
        self._free: queue.LifoQueue[np.ndarray] | None = None
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def _consume(self) -> None:
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    return
                try:
                    if self._error is None:
                        self._writer.append_data(frame)
                except Exception as exc:  # noqa: BLE001 - propagate encoder failures
                    self._error = exc
                finally:
                    if self._free is not None:
                        self._free.put(frame)
        except Exception as exc:  # noqa: BLE001 - propagate encoder failures
            self._error = exc

    def append(self, frame: np.ndarray) -> None:
        if self._error is not None:
            raise self._error
        source = np.asarray(frame)
        if self._thread is None:
            # Let imageio spawn ffmpeg before a Python worker thread exists.
            # This call is synchronous, so the first canvas view does not need
            # an owned copy before its caller starts the next draw.
            self._writer.append_data(source)
            self._free = queue.LifoQueue(maxsize=_ASYNC_FRAME_BUFFER_COUNT)
            for _ in range(_ASYNC_FRAME_BUFFER_COUNT):
                self._free.put(np.empty_like(source))
            self._thread = threading.Thread(target=self._consume, daemon=True)
            self._thread.start()
            return
        assert self._free is not None
        owned = self._free.get()
        if self._error is not None:
            self._free.put(owned)
            raise self._error
        np.copyto(owned, source)
        self._queue.put(owned)

    def close(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
        try:
            if self._error is not None:
                raise self._error
        finally:
            self._writer.close()


class ReplayRenderer:
    """Fixed-camera replay renderer with persistent, update-in-place artists."""

    def __init__(
        self,
        *,
        stadium: Stadium | None = None,
        control_fps: float = DEFAULT_TIMEBASE.control_fps,
        halftime_seconds: float = 45.0 * 60.0,
        fulltime_seconds: float = 90.0 * 60.0,
        halftime_enabled: bool = True,
        horizontal_fov_degrees: float = Perception().horizontal_fov_degrees,
        style: RenderStyle | None = None,
        reach: Reach | None = None,
        ball_radius_m: float = Ball().radius,
    ) -> None:
        self.stadium = Stadium() if stadium is None else stadium
        if reach is not None and type(reach) is not Reach:
            raise TypeError("reach must be Reach or None")
        self.reach = Reach() if reach is None else reach
        self.ball_radius_m = float(ball_radius_m)
        self.overlay = _interaction_overlay_geometry(self.reach, self.ball_radius_m)
        self.control_fps = float(control_fps)
        self.fulltime_seconds = float(fulltime_seconds)
        if type(halftime_enabled) is not bool:
            raise TypeError("halftime_enabled must be bool")
        if style is not None and type(style) is not RenderStyle:
            raise TypeError("style must be RenderStyle or None")
        self.halftime_enabled = halftime_enabled
        self.halftime_seconds = (
            float(halftime_seconds) if halftime_enabled else self.fulltime_seconds
        )
        self.horizontal_fov_degrees = float(horizontal_fov_degrees)
        self.style = RenderStyle() if style is None else style
        self.camera = _PerspectiveCamera(self.style)
        if not np.isfinite(self.control_fps) or self.control_fps <= 0.0:
            raise ValueError("control_fps must be finite and positive")
        if not np.isfinite(self.fulltime_seconds) or self.fulltime_seconds <= 0.0:
            raise ValueError("fulltime_seconds must be finite and positive")
        if self.halftime_enabled and (
            not np.isfinite(self.halftime_seconds)
            or self.halftime_seconds <= 0.0
            or self.fulltime_seconds <= self.halftime_seconds
        ):
            raise ValueError(
                "halftime_seconds must be finite, positive, and less than "
                "fulltime_seconds when halftime is enabled"
            )
        if (
            not np.isfinite(self.horizontal_fov_degrees)
            or not 0.0 < self.horizontal_fov_degrees <= 360.0
        ):
            raise ValueError("horizontal_fov_degrees must be in (0, 360]")

    def _clock_label(self, frame: _VisualFrame) -> str:
        second_half = self.halftime_enabled and frame.first_half_wall_end_tick >= 0
        first_half_nominal = round(self.halftime_seconds * self.control_fps)
        first_half_added = (
            max(frame.first_half_wall_end_tick - first_half_nominal, 0)
            if second_half
            else 0
        )
        absolute_ticks = (
            frame.control_tick
            if frame.video_time_s is None
            else frame.video_time_s * self.control_fps
        )
        display_ticks = absolute_ticks - first_half_added
        nominal_seconds = (
            self.fulltime_seconds
            if second_half or not self.halftime_enabled
            else self.halftime_seconds
        )
        display_seconds = max(0, int(np.floor(display_ticks / self.control_fps)))
        nominal = round(nominal_seconds)
        nominal_ticks = round(nominal_seconds * self.control_fps)
        if display_ticks <= nominal_ticks:
            return f"{display_seconds // 60:02d}:{display_seconds % 60:02d}"
        added = display_seconds - nominal
        return (
            f"{nominal // 60:02d}:{nominal % 60:02d} "
            f"+{added // 60:02d}:{added % 60:02d}"
        )

    @staticmethod
    def _ground_polyline(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return np.column_stack((points, np.zeros(points.shape[0])))

    def _draw_static(
        self,
        ax: Any,
        *,
        line_collection: Any,
        poly_collection: Any,
    ) -> None:
        """Draw the pitch, stands, and regulation goal frames in 3D projection."""

        s = self.stadium
        hx, hy = s.half_length, s.half_width
        camera = self.camera
        ax.set_facecolor("#0c1622")

        stripe_width = s.length / 12.0
        pitch_quads = [
            self._ground_polyline(
                np.asarray(
                    (
                        (-hx + index * stripe_width, -hy),
                        (-hx + (index + 1) * stripe_width, -hy),
                        (-hx + (index + 1) * stripe_width, hy),
                        (-hx + index * stripe_width, hy),
                    )
                )
            )
            for index in range(12)
        ]

        apron = 2.0
        stand_depth = 8.0
        stand_height = 5.0
        pitch_corners = np.asarray(
            ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)), dtype=np.float64
        )
        inner_corners = np.asarray(
            (
                (-hx - apron, -hy - apron),
                (hx + apron, -hy - apron),
                (hx + apron, hy + apron),
                (-hx - apron, hy + apron),
            ),
            dtype=np.float64,
        )
        outer_corners = np.asarray(
            (
                (-hx - apron - stand_depth, -hy - apron - stand_depth),
                (hx + apron + stand_depth, -hy - apron - stand_depth),
                (hx + apron + stand_depth, hy + apron + stand_depth),
                (-hx - apron - stand_depth, hy + apron + stand_depth),
            ),
            dtype=np.float64,
        )
        apron_quads: list[np.ndarray] = []
        stand_quads: list[np.ndarray] = []
        for index in range(4):
            nxt = (index + 1) % 4
            apron_quads.append(
                self._ground_polyline(
                    np.asarray(
                        (
                            pitch_corners[index],
                            pitch_corners[nxt],
                            inner_corners[nxt],
                            inner_corners[index],
                        )
                    )
                )
            )
            stand_quads.append(
                np.asarray(
                    (
                        (*inner_corners[index], 0.0),
                        (*inner_corners[nxt], 0.0),
                        (*outer_corners[nxt], stand_height),
                        (*outer_corners[index], stand_height),
                    ),
                    dtype=np.float64,
                )
            )
        ax.add_collection(
            poly_collection(
                [camera.project(quad) for quad in stand_quads],
                facecolors=("#3a4150", "#424958", "#343b49", "#3d4554"),
                edgecolors="#596373",
                linewidths=0.7,
                zorder=-5,
            )
        )
        ax.add_collection(
            poly_collection(
                [camera.project(quad) for quad in apron_quads],
                facecolors="#1c2a17",
                edgecolors="none",
                zorder=-4,
            )
        )
        ax.add_collection(
            poly_collection(
                [camera.project(quad) for quad in pitch_quads],
                facecolors=[("#2f8f3e", "#37a449")[index & 1] for index in range(12)],
                edgecolors="none",
                zorder=-3,
            )
        )

        ground_lines: list[np.ndarray] = []

        def rectangle(x0: float, x1: float, y0: float, y1: float) -> None:
            ground_lines.append(
                self._ground_polyline(
                    np.asarray(((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)))
                )
            )

        rectangle(-hx, hx, -hy, hy)
        ground_lines.append(self._ground_polyline(np.asarray(((0.0, -hy), (0.0, hy)))))
        angle = np.linspace(0.0, 2.0 * np.pi, 72)
        ground_lines.append(
            self._ground_polyline(
                np.column_stack(
                    (
                        s.center_circle_radius * np.cos(angle),
                        s.center_circle_radius * np.sin(angle),
                    )
                )
            )
        )
        for sign in (-1.0, 1.0):
            goal_x = sign * hx
            penalty_front = goal_x - sign * s.penalty_area_length
            goal_area_front = goal_x - sign * s.goal_area_length
            rectangle(
                goal_x,
                penalty_front,
                -s.penalty_area_width / 2.0,
                s.penalty_area_width / 2.0,
            )
            rectangle(
                goal_x,
                goal_area_front,
                -s.goal_area_width / 2.0,
                s.goal_area_width / 2.0,
            )
            spot_x = sign * (hx - 11.0)
            spot_angle = np.linspace(0.0, 2.0 * np.pi, 18)
            ground_lines.append(
                self._ground_polyline(
                    np.column_stack(
                        (
                            spot_x + 0.16 * np.cos(spot_angle),
                            0.16 * np.sin(spot_angle),
                        )
                    )
                )
            )
            arc_angle = np.deg2rad(
                np.linspace(-53.0, 53.0, 40)
                if sign < 0.0
                else np.linspace(127.0, 233.0, 40)
            )
            ground_lines.append(
                self._ground_polyline(
                    np.column_stack(
                        (
                            spot_x + s.penalty_arc_radius * np.cos(arc_angle),
                            s.penalty_arc_radius * np.sin(arc_angle),
                        )
                    )
                )
            )

        corner_specs = (
            (-hx, -hy, 0.0, 90.0),
            (-hx, hy, -90.0, 0.0),
            (hx, -hy, 90.0, 180.0),
            (hx, hy, 180.0, 270.0),
        )
        for cx, cy, start, end in corner_specs:
            corner_angle = np.deg2rad(np.linspace(start, end, 16))
            ground_lines.append(
                self._ground_polyline(
                    np.column_stack(
                        (
                            cx + s.corner_arc_radius * np.cos(corner_angle),
                            cy + s.corner_arc_radius * np.sin(corner_angle),
                        )
                    )
                )
            )
        ax.add_collection(
            line_collection(
                [camera.project(line) for line in ground_lines],
                colors="#f4f4f4",
                linewidths=1.25,
                zorder=1,
            )
        )

        goal_segments: list[np.ndarray] = []
        net_segments: list[np.ndarray] = []
        half_goal = s.goal_width / 2.0
        net_depth = 2.4
        top_back = 0.55 * net_depth

        def net_profile(
            goal_x: float, sign: float, y: float, fraction: float
        ) -> np.ndarray:
            if fraction <= 0.5:
                alpha = fraction / 0.5
                return np.asarray((goal_x + sign * top_back * alpha, y, s.goal_height))
            alpha = (fraction - 0.5) / 0.5
            return np.asarray(
                (
                    goal_x + sign * (top_back + (net_depth - top_back) * alpha),
                    y,
                    s.goal_height * (1.0 - alpha),
                )
            )

        for sign in (-1.0, 1.0):
            goal_x = sign * hx
            goal_segments.extend(
                (
                    np.asarray(
                        ((goal_x, -half_goal, 0.0), (goal_x, -half_goal, s.goal_height))
                    ),
                    np.asarray(
                        ((goal_x, half_goal, 0.0), (goal_x, half_goal, s.goal_height))
                    ),
                    np.asarray(
                        (
                            (goal_x, -half_goal, s.goal_height),
                            (goal_x, half_goal, s.goal_height),
                        )
                    ),
                )
            )
            for y in np.linspace(-half_goal, half_goal, 7):
                net_segments.extend(
                    (
                        np.asarray(
                            (
                                net_profile(goal_x, sign, y, 0.0),
                                net_profile(goal_x, sign, y, 0.5),
                            )
                        ),
                        np.asarray(
                            (
                                net_profile(goal_x, sign, y, 0.5),
                                net_profile(goal_x, sign, y, 1.0),
                            )
                        ),
                        np.asarray(
                            (
                                (goal_x, y, 0.0),
                                net_profile(goal_x, sign, y, 1.0),
                            )
                        ),
                    )
                )
            for fraction in np.linspace(0.0, 1.0, 6):
                net_segments.append(
                    np.asarray(
                        (
                            net_profile(goal_x, sign, -half_goal, fraction),
                            net_profile(goal_x, sign, half_goal, fraction),
                        )
                    )
                )
            net_segments.append(
                np.asarray(
                    (
                        (goal_x + sign * net_depth, -half_goal, 0.0),
                        (goal_x + sign * net_depth, half_goal, 0.0),
                    )
                )
            )
        ax.add_collection(
            line_collection(
                [camera.project(segment) for segment in net_segments],
                colors="#cfd8e2",
                linewidths=0.6,
                alpha=0.42,
                zorder=1.5,
            )
        )
        ax.add_collection(
            line_collection(
                [camera.project(segment) for segment in goal_segments],
                colors="#ffffff",
                linewidths=2.5,
                zorder=2,
            )
        )

        # Frame tightly around the regulation surface and goals. The retained
        # stands deliberately continue beyond the crop, like a broadcast shot,
        # instead of consuming screen area as a fully visible diorama border.
        framing_margin = 3.0
        bounds = camera.project(
            np.asarray(
                [
                    (x, y, z)
                    for x in (-hx - framing_margin, hx + framing_margin)
                    for y in (-hy - framing_margin, hy + framing_margin)
                    for z in (0.0, s.goal_height)
                ]
            )
        )
        span = np.ptp(bounds, axis=0)
        pad = 0.018 * max(span[0], span[1])
        ax.set_xlim(float(bounds[:, 0].min() - pad), float(bounds[:, 0].max() + pad))
        ax.set_ylim(float(bounds[:, 1].min() - pad), float(bounds[:, 1].max() + pad))
        ax.set_aspect("equal")
        ax.axis("off")

    def _draw_minimap_static(self, ax: Any, patches: tuple[Any, ...]) -> None:
        _Arc, Circle, Rectangle = patches
        s = self.stadium
        hx, hy = s.half_length, s.half_width
        ax.set_facecolor((0.03, 0.08, 0.05, 0.92))
        line = {
            "fill": False,
            "edgecolor": "#dce9df",
            "linewidth": 0.65,
            "zorder": 1,
        }
        ax.add_patch(Rectangle((-hx, -hy), s.length, s.width, **line))
        ax.plot((0.0, 0.0), (-hy, hy), color="#dce9df", linewidth=0.55, zorder=1)
        ax.add_patch(Circle((0.0, 0.0), s.center_circle_radius, **line))
        for sign in (-1.0, 1.0):
            goal_x = sign * hx
            front = goal_x - sign * s.penalty_area_length
            ax.add_patch(
                Rectangle(
                    (min(goal_x, front), -s.penalty_area_width / 2.0),
                    s.penalty_area_length,
                    s.penalty_area_width,
                    **line,
                )
            )
        ax.set_xlim(-hx - 1.0, hx + 1.0)
        ax.set_ylim(-hy - 1.0, hy + 1.0)
        ax.set_aspect("equal")
        ax.set_xticks(())
        ax.set_yticks(())
        for spine in ax.spines.values():
            spine.set_color("#81928a")
            spine.set_linewidth(0.8)

    def render_frames(
        self,
        frames: list[_VisualFrame],
        path: str | Path,
        *,
        fps: float,
        faststart: bool = True,
    ) -> Path:
        """Render an independent contiguous frame range to one MP4."""
        if not frames:
            raise ValueError("frames must not be empty")
        if (
            isinstance(fps, (bool, np.bool_))
            or not isinstance(fps, (int, float, np.integer, np.floating))
            or not np.isfinite(float(fps))
            or float(fps) <= 0.0
        ):
            raise ValueError("fps must be a finite positive real scalar")
        frames = _propagate_adjudications(
            frames,
            control_fps=self.control_fps,
            duration_seconds=self.style.adjudication_seconds,
        )
        (
            imageio,
            plt,
            LineCollection,
            PolyCollection,
            MplPath,
            Arc,
            Circle,
            Rectangle,
        ) = _dependencies()
        del imageio
        style = self.style
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=style.figsize, dpi=style.dpi, facecolor="#0c1622")
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        self._draw_static(
            ax,
            line_collection=LineCollection,
            poly_collection=PolyCollection,
        )
        axm = fig.add_axes(_MINIMAP_RECT)
        axm.set_zorder(35)
        self._draw_minimap_static(axm, (Arc, Circle, Rectangle))
        first = frames[0]
        count = first.player_position.shape[0]
        team = first.team_id
        gk = first.is_goalkeeper
        colors = np.where(gk, GK_COLOR_ARRAY[team], TEAM_COLOR_ARRAY[team])
        colors = np.asarray(colors, dtype=object)
        colors[first.sent_off] = "#7c2834"

        # Static broadcast header. A separate single transient adjudication
        # banner may appear beneath it; there is no historical event feed.
        ax.add_patch(
            Rectangle(
                (0.33, 0.905),
                0.34,
                0.084,
                transform=ax.transAxes,
                color="#071019",
                alpha=0.94,
                zorder=30,
            )
        )
        team0 = ax.text(
            0.395,
            SCOREBOARD_SCORE_Y,
            TEAM_LABELS[0],
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=TEAM_COLORS[0],
            fontsize=12,
            weight="bold",
            zorder=31,
        )
        team1 = ax.text(
            0.605,
            SCOREBOARD_SCORE_Y,
            TEAM_LABELS[1],
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=TEAM_COLORS[1],
            fontsize=12,
            weight="bold",
            zorder=31,
        )
        del team0, team1
        score_text = ax.text(
            0.5,
            SCOREBOARD_SCORE_Y,
            "0 : 0",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            weight="bold",
            zorder=32,
        )
        clock_text = ax.text(
            0.5,
            SCOREBOARD_CLOCK_Y,
            "00:00",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#b8c7d6",
            fontsize=10.5,
            family="monospace",
            zorder=32,
        )

        legend = (
            ("CONTROL", INTENT_COLORS[INTENT_CONTROL]),
            ("PASS", INTENT_COLORS[INTENT_PASS]),
            ("SHOT", INTENT_COLORS[INTENT_SHOT]),
            ("CLEAR", INTENT_COLORS[INTENT_CLEAR]),
            ("CHALLENGE", INTENT_COLORS[INTENT_CHALLENGE]),
        )
        ax.add_patch(
            Rectangle(
                (0.014, 0.025),
                0.425,
                0.056,
                transform=fig.transFigure,
                color="#071019",
                alpha=0.90,
                clip_on=False,
                zorder=30,
            )
        )
        for index, (label, color) in enumerate(legend):
            x = 0.034 + index * 0.082
            ax.scatter(
                [x],
                [0.053],
                s=52.0,
                transform=fig.transFigure,
                facecolors="none",
                edgecolors=color,
                linewidths=1.9,
                clip_on=False,
                zorder=31,
            )
            ax.text(
                x + 0.013,
                0.053,
                label,
                transform=fig.transFigure,
                ha="left",
                va="center",
                color="#f4f7fa",
                fontsize=7.0,
                weight="bold",
                zorder=31,
            )

        event_panel = Rectangle(
            (0.285, 0.810),
            0.430,
            0.073,
            transform=ax.transAxes,
            color="#071019",
            alpha=0.94,
            zorder=33,
            visible=False,
        )
        event_accent = Rectangle(
            (0.285, 0.810),
            0.006,
            0.073,
            transform=ax.transAxes,
            color="#ffffff",
            zorder=34,
            visible=False,
        )
        ax.add_patch(event_panel)
        ax.add_patch(event_accent)
        event_title = ax.text(
            0.302,
            0.857,
            "",
            transform=ax.transAxes,
            ha="left",
            va="center",
            color="#ffffff",
            fontsize=12.0,
            weight="bold",
            zorder=35,
            visible=False,
        )
        event_detail = ax.text(
            0.302,
            0.828,
            "",
            transform=ax.transAxes,
            ha="left",
            va="center",
            color="#d9e3ec",
            fontsize=7.5,
            weight="bold",
            zorder=35,
            visible=False,
        )

        zeros = np.zeros((count, 2), np.float32)
        player_shadow = ax.scatter(
            zeros[:, 0],
            zeros[:, 1],
            s=style.player_size * 1.08,
            c="#07120b",
            alpha=0.28,
            linewidths=0,
            zorder=4,
        )
        players = ax.scatter(
            zeros[:, 0],
            zeros[:, 1],
            s=style.player_size,
            c=colors,
            edgecolors="#101010",
            linewidths=1.0,
            zorder=_PLAYER_MARKER_ZORDER,
        )
        # Intent rings live in turf coordinates rather than screen-marker
        # coordinates. The fixed camera therefore foreshortens them with the
        # pitch instead of showing front-facing circles around every player.
        empty_rings = [np.zeros((2, 2)) for _ in range(count)]
        intent_ring_underlay = LineCollection(
            empty_rings,
            colors="#071019",
            linewidths=6.4,
            alpha=0.0,
            capstyle="round",
            zorder=_INTENT_RING_UNDERLAY_ZORDER,
        )
        intent_rings = LineCollection(
            empty_rings,
            colors="#ffffff",
            linewidths=3.8,
            alpha=0.0,
            capstyle="round",
            zorder=_INTENT_RING_ZORDER,
        )
        ax.add_collection(intent_ring_underlay)
        ax.add_collection(intent_rings)
        aerial_effect = ax.scatter(
            zeros[:, 0],
            zeros[:, 1],
            s=np.zeros(count),
            facecolors="none",
            edgecolors="#a993ff",
            linewidths=1.5,
            zorder=10,
        )
        # One translucent collection shows the realized state gaze without
        # bright boundary rays competing with the player and intent cues.
        field_of_view_fan = PolyCollection(
            [np.zeros((2 * _FOV_FAN_SAMPLES, 2))] * count,
            facecolors=np.zeros((count, 4), dtype=np.float32),
            edgecolors="none",
            zorder=5.5,
        )
        ax.add_collection(field_of_view_fan)
        number_labels = tuple("GK" if gk[i] else str(i + 1) for i in range(count))
        numbers = [
            ax.text(
                0,
                0,
                number_labels[i],
                color="white",
                fontsize=8.0,
                weight="bold",
                ha="center",
                va="center",
                zorder=9,
            )
            for i in range(count)
        ]
        stamina_long_rail = LineCollection(
            [np.zeros((2, 2)) for _ in range(count)],
            colors=_STAMINA_BAR_RAIL_COLOR,
            linewidths=_STAMINA_BAR_RAIL_WIDTH_PT,
            alpha=0.0,
            capstyle="round",
            zorder=9,
        )
        stamina_short_rail = LineCollection(
            [np.zeros((2, 2)) for _ in range(count)],
            colors=_STAMINA_BAR_RAIL_COLOR,
            linewidths=_STAMINA_BAR_RAIL_WIDTH_PT,
            alpha=0.0,
            capstyle="round",
            zorder=9,
        )
        stamina_long = LineCollection(
            [np.zeros((2, 2)) for _ in range(count)],
            colors=_STAMINA_LONG_COLOR,
            linewidths=_STAMINA_BAR_FILL_WIDTH_PT,
            capstyle="round",
            zorder=10,
        )
        stamina_short = LineCollection(
            [np.zeros((2, 2)) for _ in range(count)],
            colors=_STAMINA_SHORT_COLOR,
            linewidths=_STAMINA_BAR_FILL_WIDTH_PT,
            capstyle="round",
            zorder=10,
        )
        ax.add_collection(stamina_long_rail)
        ax.add_collection(stamina_short_rail)
        ax.add_collection(stamina_long)
        ax.add_collection(stamina_short)
        cards = ax.scatter(
            zeros[:, 0],
            zeros[:, 1],
            marker="s",
            s=np.zeros(count),
            c="#ffd21c",
            edgecolors="#332900",
            linewidths=0.6,
            zorder=12,
        )
        (possession,) = ax.plot(
            [], [], color=TEAM_COLORS[0], linewidth=2.2, alpha=0.0, zorder=5
        )
        ball_shadow = ax.scatter([0], [0], s=55, c="#07120b", alpha=0.35, zorder=5)
        (ball_drop,) = ax.plot(
            [], [], color="#d9e1e8", linewidth=0.8, linestyle=(0, (2, 2)), zorder=6
        )
        ball = ax.scatter(
            [0], [0], s=58, c="#fafafa", edgecolors="#202020", linewidths=0.8, zorder=11
        )
        minimap_players = axm.scatter(
            zeros[:, 0],
            zeros[:, 1],
            s=18,
            c=colors,
            edgecolors="#101010",
            linewidths=0.35,
            zorder=4,
        )
        minimap_ball = axm.scatter(
            [0], [0], s=22, c="#ffffff", edgecolors="#202020", linewidths=0.4, zorder=5
        )

        _flatten_circle_markers(
            MplPath,
            player_shadow,
            players,
            aerial_effect,
            ball_shadow,
            ball,
            minimap_players,
            minimap_ball,
        )

        dynamic_main = [
            *_player_intent_painter_order(
                field_of_view_fan,
                player_shadow,
                intent_ring_underlay,
                intent_rings,
                players,
            ),
            aerial_effect,
            stamina_long_rail,
            stamina_short_rail,
            stamina_long,
            stamina_short,
            cards,
            possession,
            ball_shadow,
            ball_drop,
            ball,
            score_text,
            clock_text,
            *numbers,
            # draw_artist follows this explicit order rather than z-order.
            # Keep the one transient decision above every field-space cue.
            event_panel,
            event_accent,
            event_title,
            event_detail,
        ]
        dynamic_minimap = [minimap_players, minimap_ball]
        for artist in dynamic_main + dynamic_minimap:
            artist.set_animated(True)
        _install_renderer_font_path_cache(fig.canvas.get_renderer())
        fig.canvas.draw()
        text_layout_cache: dict[str, Any] = {}

        def set_number_label(text: Any, label: str) -> None:
            """Keep Agg's hinted glyph rasterization but cache static layout."""

            text.set_text(label)
            layout = text_layout_cache.get(label)
            if layout is None:
                layout = type(text)._get_layout(text, fig.canvas.get_renderer())
                text_layout_cache[label] = layout
            text._get_layout = lambda _renderer, cached=layout: cached

        for text, label in zip(numbers, number_labels, strict=True):
            set_number_label(text, label)
        background = fig.canvas.copy_from_bbox(fig.bbox)
        writer = _AsyncWriter(path, fps, style, faststart=faststart)
        angles = np.linspace(0.0, 2.0 * np.pi, 40)
        ring_unit = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        half_fov = np.deg2rad(0.5 * self.horizontal_fov_degrees)
        fan_angles = np.linspace(-half_fov, half_fov, _FOV_FAN_SAMPLES)
        fan_cos = np.cos(fan_angles)
        fan_sin = np.sin(fan_angles)
        cached_team = np.asarray(first.team_id).copy()
        cached_goalkeeper = np.asarray(first.is_goalkeeper).copy()
        cached_sent_off = np.asarray(first.sent_off).copy()
        cached_active: np.ndarray | None = None
        cached_high_head_contact: np.ndarray | None = None
        cached_ball_live: bool | None = None
        intent_visible = False
        try:
            for frame in frames:
                active = frame.active
                active_changed = cached_active is None or not np.array_equal(
                    active, cached_active
                )
                pos = frame.player_position
                aerial = np.where(active, frame.aerial_progress, 0.0)
                # The post-contact recovery lock is a causal countdown. Map it
                # through one sine arch so the single player marker rises and
                # descends without inventing a body pose or frame history.
                jump_phase = np.sin(np.pi * (1.0 - np.clip(aerial, 0.0, 1.0)))
                aerial_lift = _AERIAL_LIFT_M * jump_phase
                display_world = np.column_stack(
                    (pos, _PLAYER_MARKER_HEIGHT_M + aerial_lift)
                )
                hud_world = np.column_stack((pos, _PLAYER_HUD_HEIGHT_M + aerial_lift))
                ground_world = np.column_stack((pos, np.zeros(count, dtype=np.float32)))
                projected, projected_scale = self.camera.project_with_relative_scale(
                    np.concatenate((display_world, hud_world, ground_world), axis=0)
                )
                display_pos, hud_pos, ground_pos = np.split(projected, 3)
                depth_scale = projected_scale[:count]
                size = np.where(active, style.player_size * np.square(depth_scale), 0.0)
                player_shadow.set_offsets(ground_pos)
                # A jump stays anchored to the authoritative ground position.
                # The shrinking shadow and ground pulse make the recovery arc
                # visible without moving the physical player coordinate.
                player_shadow.set_sizes(size * (0.92 - 0.24 * jump_phase))
                players.set_offsets(display_pos)
                players.set_sizes(size)
                goalkeeper_changed = not np.array_equal(
                    frame.is_goalkeeper, cached_goalkeeper
                )
                player_style_changed = not (
                    np.array_equal(frame.team_id, cached_team)
                    and not goalkeeper_changed
                    and np.array_equal(frame.sent_off, cached_sent_off)
                )
                if player_style_changed:
                    cached_team[...] = frame.team_id
                    cached_goalkeeper[...] = frame.is_goalkeeper
                    cached_sent_off[...] = frame.sent_off
                    player_colors = np.where(
                        frame.is_goalkeeper,
                        GK_COLOR_ARRAY[frame.team_id],
                        TEAM_COLOR_ARRAY[frame.team_id],
                    )
                    player_colors = np.asarray(player_colors, dtype=object)
                    player_colors[frame.sent_off] = "#7c2834"
                    players.set_facecolors(player_colors)
                    minimap_players.set_facecolors(player_colors)
                direction = frame.player_body_forward
                # Contact intent occupies the small turf-space gap between the
                # player and the FOV fan. Projecting the world-space circle
                # makes it lie on the pitch under the fixed camera.
                shown_intent = (
                    active
                    & frame.action_executed
                    & (frame.requested_intent != INTENT_MOVE)
                )
                if np.any(shown_intent):
                    ring_radius = _intent_ring_radii(
                        frame.requested_intent,
                        frame.is_goalkeeper,
                        self.overlay,
                    )
                    ring_world = _intent_ring_world_vertices(
                        pos,
                        ring_radius,
                        ring_unit,
                    )
                    ring_segments = self.camera.project(
                        ring_world.reshape(-1, 3)
                    ).reshape(count, angles.size, 2)
                    _update_line_vertices(intent_ring_underlay, ring_segments)
                    _update_line_vertices(intent_rings, ring_segments)
                    intent_rings.set_color(
                        [
                            INTENT_COLORS.get(int(intent), "#ffffff")
                            for intent in frame.requested_intent
                        ]
                    )
                    intent_ring_underlay.set_alpha(np.where(shown_intent, 0.72, 0.0))
                    intent_rings.set_alpha(shown_intent.astype(np.float32))
                    intent_visible = True
                elif intent_visible:
                    intent_ring_underlay.set_alpha(np.zeros(count, dtype=np.float32))
                    intent_rings.set_alpha(np.zeros(count, dtype=np.float32))
                    intent_visible = False
                aerial_effect.set_offsets(ground_pos)
                aerial_effect.set_sizes(
                    np.where(
                        active & ((aerial > 0.0) | frame.high_head_contact),
                        size
                        * np.where(
                            frame.high_head_contact,
                            2.60,
                            1.65 + 0.45 * jump_phase,
                        ),
                        0.0,
                    )
                )
                if cached_high_head_contact is None or not np.array_equal(
                    frame.high_head_contact, cached_high_head_contact
                ):
                    cached_high_head_contact = np.asarray(
                        frame.high_head_contact
                    ).copy()
                    effect_colors = np.tile(
                        np.array([0.663, 0.576, 1.0, 0.68]), (count, 1)
                    )
                    effect_colors[frame.high_head_contact] = (
                        1.0,
                        0.949,
                        0.478,
                        0.95,
                    )
                    aerial_effect.set_edgecolors(effect_colors)
                    aerial_effect.set_linewidths(
                        np.where(frame.high_head_contact, 3.0, 2.0)
                    )
                gaze_cos = np.cos(frame.player_gaze_yaw)
                gaze_sin = np.sin(frame.player_gaze_yaw)
                view_direction = np.column_stack(
                    (
                        direction[:, 0] * gaze_cos - direction[:, 1] * gaze_sin,
                        direction[:, 0] * gaze_sin + direction[:, 1] * gaze_cos,
                    )
                )
                fan_direction = np.stack(
                    (
                        view_direction[:, 0, None] * fan_cos
                        - view_direction[:, 1, None] * fan_sin,
                        view_direction[:, 0, None] * fan_sin
                        + view_direction[:, 1, None] * fan_cos,
                    ),
                    axis=-1,
                )
                outer_fan_xy = (
                    pos[:, None, :] + self.overlay.fov_outer_radius_m * fan_direction
                )
                inner_fan_xy = (
                    pos[:, None, :]
                    + self.overlay.fov_inner_radius_m * fan_direction[:, ::-1, :]
                )
                fan_xy = np.concatenate((outer_fan_xy, inner_fan_xy), axis=1)
                fan_world = np.concatenate(
                    (
                        fan_xy,
                        np.full(
                            (count, 2 * _FOV_FAN_SAMPLES, 1),
                            0.025,
                            dtype=np.float32,
                        ),
                    ),
                    axis=2,
                )
                _update_polygon_vertices(
                    field_of_view_fan,
                    self.camera.project(fan_world.reshape(-1, 3)).reshape(
                        count, 2 * _FOV_FAN_SAMPLES, 2
                    ),
                )
                if active_changed:
                    fan_colors = np.ones((count, 4), dtype=np.float32)
                    fan_colors[:, 3] = np.where(active, _FOV_FAN_ALPHA, 0.0)
                    field_of_view_fan.set_facecolors(fan_colors)
                bar_half_width = _STAMINA_BAR_HALF_WIDTH_PX * depth_scale
                base_y = hud_pos[:, 1] + _STAMINA_BAR_OFFSET_PX * depth_scale
                left = hud_pos[:, 0] - bar_half_width
                right = hud_pos[:, 0] + bar_half_width
                long_y = base_y - _STAMINA_BAR_HALF_GAP_PX * depth_scale
                short_y = base_y + _STAMINA_BAR_HALF_GAP_PX * depth_scale
                _update_line_vertices(
                    stamina_long_rail,
                    np.stack((np.c_[left, long_y], np.c_[right, long_y]), axis=1),
                )
                _update_line_vertices(
                    stamina_short_rail,
                    np.stack((np.c_[left, short_y], np.c_[right, short_y]), axis=1),
                )
                _update_line_vertices(
                    stamina_long,
                    np.stack(
                        (
                            np.c_[left, long_y],
                            np.c_[
                                left
                                + 2.0
                                * bar_half_width
                                * np.clip(frame.stamina_long, 0, 1),
                                long_y,
                            ],
                        ),
                        axis=1,
                    ),
                )
                _update_line_vertices(
                    stamina_short,
                    np.stack(
                        (
                            np.c_[left, short_y],
                            np.c_[
                                left
                                + 2.0
                                * bar_half_width
                                * np.clip(frame.stamina_short, 0, 1),
                                short_y,
                            ],
                        ),
                        axis=1,
                    ),
                )
                if active_changed:
                    for collection in (stamina_long_rail, stamina_short_rail):
                        collection.set_alpha(np.where(active, 0.88, 0.0))
                    for collection in (stamina_long, stamina_short):
                        collection.set_alpha(np.where(active, 1.0, 0.0))
                    for index, text in enumerate(numbers):
                        text.set_visible(bool(active[index]))
                    cached_active = np.asarray(active).copy()
                if goalkeeper_changed:
                    number_labels = tuple(
                        "GK" if frame.is_goalkeeper[i] else str(i + 1)
                        for i in range(count)
                    )
                    for text, label in zip(numbers, number_labels, strict=True):
                        set_number_label(text, label)
                for index, text in enumerate(numbers):
                    text.set_position(display_pos[index])
                cards.set_offsets(
                    hud_pos + np.column_stack((4.5 * depth_scale, 4.5 * depth_scale))
                )
                cards.set_sizes(
                    np.where(
                        active & (frame.yellow_cards > 0),
                        34.0 * np.square(depth_scale),
                        0.0,
                    )
                )
                owner = frame.possession_player
                if 0 <= owner < count and active[owner]:
                    radius = 1.15
                    ring = self.camera.project(
                        np.column_stack(
                            (
                                pos[owner, 0] + radius * ring_unit[:, 0],
                                pos[owner, 1] + radius * ring_unit[:, 1],
                                np.zeros_like(angles),
                            )
                        )
                    )
                    possession.set_data(ring[:, 0], ring[:, 1])
                    possession.set_color(TEAM_COLORS[int(frame.team_id[owner])])
                    possession.set_alpha(0.95)
                else:
                    possession.set_alpha(0.0)
                ball_xy = frame.ball_position[:2]
                ball_height = max(0.0, float(frame.ball_position[2]))
                ball_ground_world = np.asarray(((ball_xy[0], ball_xy[1], 0.0),))
                ball_world = np.asarray(((ball_xy[0], ball_xy[1], ball_height),))
                ball_projection, ball_scales = self.camera.project_with_relative_scale(
                    np.concatenate((ball_ground_world, ball_world), axis=0)
                )
                ball_ground = ball_projection[:1]
                ball_projected = ball_projection[1:]
                ball_scale = float(ball_scales[1])
                ball_shadow.set_offsets(ball_ground)
                ball.set_offsets(ball_projected)
                ball.set_sizes([58.0 * ball_scale * ball_scale])
                if frame.ball_live != cached_ball_live:
                    ball.set_facecolors(["#fafafa" if frame.ball_live else "#8b949e"])
                    cached_ball_live = frame.ball_live
                ball_drop.set_data(
                    [ball_ground[0, 0], ball_projected[0, 0]],
                    [ball_ground[0, 1], ball_projected[0, 1]],
                )
                minimap_players.set_offsets(pos)
                minimap_players.set_sizes(np.where(active, 18.0, 0.0))
                minimap_ball.set_offsets(ball_xy[None])
                score_text.set_text(f"{frame.score[0]} : {frame.score[1]}")
                clock_text.set_text(self._clock_label(frame))
                caption = frame.adjudication
                banner_visible = caption is not None
                for artist in (
                    event_panel,
                    event_accent,
                    event_title,
                    event_detail,
                ):
                    artist.set_visible(banner_visible)
                if caption is not None:
                    event_accent.set_facecolor(caption.accent)
                    event_title.set_text(caption.title)
                    event_detail.set_text(caption.detail)
                fig.canvas.restore_region(background)
                for artist in dynamic_main:
                    ax.draw_artist(artist)
                for artist in dynamic_minimap:
                    axm.draw_artist(artist)
                fig.canvas.blit(fig.bbox)
                # The writer synchronously consumes the first canvas view and
                # copies subsequent views into a fixed two-buffer handoff
                # pool. This preserves async ownership without one fresh
                # 1920x1080 RGBA allocation per frame.
                writer.append(fig.canvas.buffer_rgba())
        finally:
            writer.close()
            plt.close(fig)
        return path


def _render_segment(
    frames: list[_VisualFrame],
    path: str,
    stadium: Stadium,
    reach: Reach,
    ball_radius_m: float,
    control_fps: float,
    halftime_seconds: float,
    fulltime_seconds: float,
    halftime_enabled: bool,
    horizontal_fov_degrees: float,
    style: RenderStyle,
    fps: float,
) -> tuple[str, int]:
    ReplayRenderer(
        stadium=stadium,
        reach=reach,
        ball_radius_m=ball_radius_m,
        control_fps=control_fps,
        halftime_seconds=halftime_seconds,
        fulltime_seconds=fulltime_seconds,
        halftime_enabled=halftime_enabled,
        horizontal_fov_degrees=horizontal_fov_degrees,
        style=style,
    ).render_frames(frames, path, fps=fps, faststart=False)
    rendered = Path(path)
    if not rendered.is_file() or rendered.stat().st_size <= 0:
        raise RuntimeError(f"encoder produced no segment: {rendered}")
    return path, len(frames)


def _concat_segments(segments: list[Path], output: Path) -> None:
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as exc:
        raise ImportError("parallel rendering requires imageio-ffmpeg") from exc
    manifest = segments[0].parent / "concat.txt"
    manifest.write_text(
        "".join(f"file '{path.name}'\n" for path in segments), encoding="utf-8"
    )
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(f"segment concatenation produced no video: {output}")


def _resample_indices(
    frames: list[HostFrame], source_fps: float, target_fps: float
) -> np.ndarray:
    """Causal hold resampling that preserves control-tick match duration."""
    if len(frames) == 1:
        return np.zeros(1, dtype=np.int64)
    ticks = np.asarray([frame.control_tick for frame in frames], dtype=np.float64)
    if np.any(np.diff(ticks) <= 0.0):
        raise ValueError("control_tick must be strictly increasing for rendering")
    times = (ticks - ticks[0]) / source_fps
    # Every post-control source row represents one complete control cell. Use
    # the covered interval, including the final cell, rather than only the
    # first-to-last endpoint span. Otherwise 10 -> 20 Hz produces 2N-1 frames
    # and silently shortens every video by 50 ms.
    covered_seconds = times[-1] + 1.0 / source_fps
    target_count = max(1, int(np.floor(covered_seconds * target_fps + 0.5)))
    target_times = np.arange(target_count, dtype=np.float64) / target_fps
    return np.clip(
        np.searchsorted(times, target_times, side="right") - 1, 0, len(frames) - 1
    )


def render_mp4(
    states: Any,
    output_dir: str | Path,
    *,
    video_name: str = "match.mp4",
    env: Any = None,
    ball_radius_m: float | None = None,
    frame_events: Any = None,
    observations: Any = None,
    slot_generations: Any = None,
    substitution_events: Any = None,
    acting_goalkeeper_events: Any = None,
    metadata: Any = None,
    match_index: int = 0,
    fps: float = DEFAULT_RENDER_FPS,
    every: int = 1,
    workers: int = 1,
    chunk_frames: int | None = None,
    style: RenderStyle | None = None,
    exact_actions: bool = False,
) -> RenderResult:
    """Render one selected match and write replay sidecars.

    Batched inputs always render only ``match_index`` (zero by default).  The
    host boundary and all Matplotlib/encoding work are outside JAX transitions.
    ``workers > 1`` renders independent contiguous segments and concatenates
    identical H.264 streams without re-encoding.
    ``slot_generations`` carries exact reusable-slot identity; omitted values
    are written as the untracked ``-1`` sentinel and are never inferred from
    player ids. Per-frame ``substitution_events`` and
    ``acting_goalkeeper_events`` may contain ``None`` where no manager
    command ran. Committed substitutions and acting-goalkeeper assignments are
    written only to ``event.json`` and never drawn on the video. These
    arguments are post-rollout sidecars and add no JAX transition compilation
    or per-step runtime cost.

    When ``env`` is omitted, ``ball_radius_m`` is required because the replay
    state contains the ball centre but not its physical radius.  When ``env``
    is supplied, its configured radius is authoritative.

    ``fps`` is the causal video-resampling rate before decimation. ``every``
    retains every Nth resampled frame. The encoder rate is derived from the
    retained and sampled frame counts so a partial final group does not change
    total playback duration. Tracking and event
    sidecars remain on the original source-frame grid and use ``control_tick``
    as their authoritative clock.
    """
    if (
        isinstance(fps, (bool, np.bool_))
        or not isinstance(fps, (int, float, np.integer, np.floating))
        or not np.isfinite(float(fps))
        or float(fps) <= 0.0
    ):
        raise ValueError("fps must be a finite positive real scalar")
    if not isinstance(every, int) or isinstance(every, bool) or every < 1:
        raise ValueError("every must be a positive integer")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if chunk_frames is not None and (
        not isinstance(chunk_frames, int)
        or isinstance(chunk_frames, bool)
        or chunk_frames < 1
    ):
        raise ValueError("chunk_frames must be None or a positive integer")
    if type(exact_actions) is not bool:
        raise TypeError("exact_actions must be bool")
    if style is not None and type(style) is not RenderStyle:
        raise TypeError("style must be RenderStyle or None")
    style = RenderStyle() if style is None else style
    if env is not None:
        configured_ball_radius_m = float(env.ball.radius)
        if ball_radius_m is not None:
            tolerance = np.finfo(np.float32).eps * max(
                1.0, abs(configured_ball_radius_m)
            )
            if not np.isclose(
                float(ball_radius_m),
                configured_ball_radius_m,
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError("ball_radius_m must match env.ball.radius")
        ball_radius_m = configured_ball_radius_m
    elif ball_radius_m is None:
        raise ValueError("ball_radius_m is required when env is omitted")
    ball_radius_m = float(ball_radius_m)
    reach = env.reach if env is not None else Reach()
    overlay = _interaction_overlay_geometry(reach, ball_radius_m)
    if not np.isfinite(ball_radius_m) or ball_radius_m <= 0.0:
        raise ValueError("ball_radius_m must be finite and positive")
    output_dir = Path(output_dir)
    if Path(video_name).name != video_name or not video_name.lower().endswith(".mp4"):
        raise ValueError("video_name must be a plain .mp4 filename")
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = output_dir / "completion.json"
    # This convenience path is intentionally non-atomic, but a failed rerender
    # must never leave a stale success receipt from an earlier invocation.
    completion_path.unlink(missing_ok=True)
    video = output_dir / video_name
    control_fps = float(env.timebase.control_fps) if env is not None else float(fps)
    stadium = env.stadium if env is not None else Stadium()
    if env is not None:
        fulltime_tick, halftime_tick = env.match.clock_ticks(env.timebase)
        halftime_seconds = halftime_tick / control_fps
        fulltime_seconds = fulltime_tick / control_fps
    else:
        halftime_seconds = 45.0 * 60.0
        fulltime_seconds = 90.0 * 60.0
    halftime_enabled = env.match.halftime_enabled if env is not None else True
    perception = env.perception if env is not None else Perception()
    environment_view_limited = bool(perception.limit_by_view_angle)
    environment_horizontal_fov_degrees = float(perception.horizontal_fov_degrees)
    horizontal_fov_degrees = (
        environment_horizontal_fov_degrees
        if environment_view_limited
        else float(style.gaze_cue_degrees)
    )
    contact_timing = env.contact_timing if env is not None else ContactTiming()
    timebase = env.timebase if env is not None else DEFAULT_TIMEBASE
    max_outfield_aerial_recovery_substeps = max(
        1,
        round(contact_timing.aerial_attempt_recovery_s / timebase.dt_phys),
    )
    max_goalkeeper_aerial_recovery_substeps = max(
        1,
        round(contact_timing.goalkeeper_dive_recovery_s / timebase.dt_phys),
    )
    started = time.perf_counter()
    frames = prepare_host_frames(
        states,
        frame_events=frame_events,
        observations=observations,
        slot_generations=slot_generations,
        substitution_events=substitution_events,
        acting_goalkeeper_events=acting_goalkeeper_events,
        match_index=match_index,
    )
    source_frame_count = len(frames)
    if source_frame_count == 0:
        raise ValueError("states must contain at least one source frame")
    start_control_tick = int(frames[0].control_tick)
    final_control_tick = int(frames[-1].control_tick)
    sample_indices = _resample_indices(frames, control_fps, float(fps))
    indices = sample_indices[::every]
    video_sample_frame_count = len(sample_indices)
    video_frame_count = len(indices)
    render_fps = video_frame_count * float(fps) / video_sample_frame_count
    if workers == 1 or video_frame_count < _MIN_PARALLEL_RENDER_FRAMES:
        planned_chunk_frames = video_frame_count
        segment_count = 1
        used_workers = 1
        encoder_threads_per_segment = style.encoder_threads
    else:
        planned_chunk_frames = (
            max(1, (video_frame_count + workers - 1) // workers)
            if chunk_frames is None
            else chunk_frames
        )
        segment_count = (
            video_frame_count + planned_chunk_frames - 1
        ) // planned_chunk_frames
        used_workers = min(workers, segment_count)
        encoder_threads_per_segment = max(1, style.encoder_threads // used_workers)
    render_metadata = render_settings_receipt(
        style=style,
        video_fps=render_fps,
        workers_requested=workers,
        workers_effective=used_workers,
        encoder_threads_per_segment=encoder_threads_per_segment,
        render_chunk_frame_cap=planned_chunk_frames,
        segment_count=segment_count,
        process_start_method=("spawn" if used_workers > 1 else None),
        environment_view_limited=environment_view_limited,
        environment_horizontal_fov_degrees=(environment_horizontal_fov_degrees),
        gaze_yaw_limit_degrees=float(perception.gaze_yaw_limit_degrees),
        gaze_slew_rate_degrees_s=float(perception.gaze_slew_rate_degrees_s),
        rendered_fov_degrees=horizontal_fov_degrees,
        fov_fan_inner_m=overlay.fov_inner_radius_m,
        fov_fan_outer_m=overlay.fov_outer_radius_m,
        fov_fan_alpha=_FOV_FAN_ALPHA,
        fov_fan_samples=_FOV_FAN_SAMPLES,
        intent_ring_ordinary_radius_m=overlay.ordinary_ring_radius_m,
        intent_ring_challenge_radius_m=overlay.challenge_ring_radius_m,
        intent_ring_goalkeeper_control_radius_m=(
            overlay.goalkeeper_control_ring_radius_m
        ),
        event_chunk_steps=None,
        async_rgba_buffers_per_worker=_ASYNC_FRAME_BUFFER_COUNT,
    )
    event, tracking, meta, _ = write_replay_sidecars(
        frames,
        video,
        control_fps=control_fps,
        video_sample_fps=float(fps),
        sample_every=every,
        video_sample_frame_count=video_sample_frame_count,
        video_frame_count=video_frame_count,
        video_fps=render_fps,
        match_index=match_index,
        stadium=stadium,
        halftime_seconds=halftime_seconds,
        fulltime_seconds=fulltime_seconds,
        halftime_enabled=halftime_enabled,
        metadata=metadata,
        render_metadata=render_metadata,
        collect_exact_events=False,
        include_all_action_controls=exact_actions,
    )
    source_render_frames = [
        _visual_frame(
            frame,
            max_outfield_aerial_recovery_substeps=(
                max_outfield_aerial_recovery_substeps
            ),
            max_goalkeeper_aerial_recovery_substeps=(
                max_goalkeeper_aerial_recovery_substeps
            ),
            ball_radius_m=ball_radius_m,
        )
        for frame in frames
    ]
    source_render_frames = _propagate_adjudications(
        source_render_frames,
        control_fps=control_fps,
        duration_seconds=style.adjudication_seconds,
    )
    render_frames = [source_render_frames[int(index)] for index in indices]
    del source_render_frames
    del frames
    if used_workers == 1:
        ReplayRenderer(
            stadium=stadium,
            reach=reach,
            ball_radius_m=ball_radius_m,
            control_fps=control_fps,
            halftime_seconds=halftime_seconds,
            fulltime_seconds=fulltime_seconds,
            halftime_enabled=halftime_enabled,
            horizontal_fov_degrees=horizontal_fov_degrees,
            style=style,
        ).render_frames(render_frames, video, fps=render_fps)
    else:
        chunks = [
            render_frames[i : i + planned_chunk_frames]
            for i in range(0, len(render_frames), planned_chunk_frames)
        ]
        worker_style = replace(style, encoder_threads=encoder_threads_per_segment)
        with TemporaryDirectory(
            prefix=".footballworld-render-", dir=output_dir
        ) as temp:
            temp_path = Path(temp)
            segments = [temp_path / f"segment_{i:05d}.mp4" for i in range(len(chunks))]
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=used_workers, mp_context=context
            ) as pool:
                with _renderer_spawn_environment():
                    futures = [
                        pool.submit(
                            _render_segment,
                            chunk,
                            str(segment),
                            stadium,
                            reach,
                            ball_radius_m,
                            control_fps,
                            halftime_seconds,
                            fulltime_seconds,
                            halftime_enabled,
                            horizontal_fov_degrees,
                            worker_style,
                            render_fps,
                        )
                        for chunk, segment in zip(chunks, segments, strict=True)
                    ]
                for future in futures:
                    future.result()
            _concat_segments(segments, video)
    completion = {
        "done": False,
        "complete": False,
        "termination_reason": "render_only",
        "terminal_basis": "render_only",
        "full_duration_complete": False,
        "start_control_tick": start_control_tick,
        "final_control_tick": final_control_tick,
    }
    verified_counts = {
        "source": source_frame_count,
        "tracking": source_frame_count,
        "event": source_frame_count,
        "video": video_frame_count,
        "sidecar_verification": (
            "successful_write_from_caller_materialized_source_grid"
        ),
        "video_verification": "successful_encoder_close_and_segment_count",
    }
    with meta.open(encoding="utf-8") as stream:
        metadata_record = json.load(stream)
    metadata_record["completion"] = completion
    metadata_record["verified_counts"] = verified_counts
    with meta.open("w", encoding="utf-8") as stream:
        json.dump(
            metadata_record,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        stream.write("\n")
    artifacts = {
        "video": artifact_receipt(video),
        "event": artifact_receipt(event),
        "tracking": artifact_receipt(tracking),
        "metadata": artifact_receipt(meta),
    }
    write_completion_manifest(
        output_dir,
        completion=completion,
        outputs=[
            {
                "video": video.relative_to(output_dir).as_posix(),
                "event": event.relative_to(output_dir).as_posix(),
                "tracking": tracking.relative_to(output_dir).as_posix(),
                "metadata": meta.relative_to(output_dir).as_posix(),
                "source_frame_count": source_frame_count,
                "tracking_frame_count": source_frame_count,
                "event_frame_count": source_frame_count,
                "video_frame_count": video_frame_count,
                "tracking_storage": metadata_record["tracking_storage"],
                "artifacts": artifacts,
                "video_decode_verification": {
                    "enabled": False,
                    "method": "not_requested",
                },
            }
        ],
    )
    elapsed = time.perf_counter() - started
    return RenderResult(
        video=video,
        event=event,
        tracking=tracking,
        metadata=meta,
        frames=len(render_frames),
        seconds=elapsed,
        throughput_fps=len(render_frames) / max(elapsed, 1e-9),
        workers=used_workers,
    )


__all__ = ["RenderResult", "ReplayRenderer", "render_mp4"]
