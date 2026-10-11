"""JAX import 없이 실행하는 NumPy·Agg MP4 화면 생성과 encoder worker."""

from __future__ import annotations

import json
import math
import subprocess
from bisect import bisect_right
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_TEAM_COLORS = np.asarray(("#ee4b5f", "#3178e8"), dtype=object)
_GK_COLORS = np.asarray(("#ffad33", "#35d0ba"), dtype=object)
_INTENT_COLORS = np.asarray(
    ("#000000", "#00d5c8", "#ffd166", "#ff7f0e", "#f03b73"), dtype=object
)


def _poisson_wdl_probabilities(score, rates) -> np.ndarray:
    """현재 점수와 양팀의 남은 Poisson 득점률에서 HOME/DRAW/AWAY 확률을 구한다."""

    score = np.asarray(score, dtype=np.int64)
    rates = np.asarray(rates, dtype=np.float64)
    if score.shape != (2,) or rates.shape != (2,):
        raise ValueError("Poisson 승률 입력은 score와 rates 각각 (2,)여야 한다")
    if np.any(score < 0) or not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("Poisson 승률 입력 점수·득점률은 유효한 비음수여야 한다")
    maximum_goals = 20
    values = np.arange(maximum_goals + 1, dtype=np.float64)
    distributions = []
    for rate in rates:
        probabilities = np.empty(maximum_goals + 1, dtype=np.float64)
        probabilities[0] = math.exp(-float(rate))
        for goals in range(1, maximum_goals + 1):
            probabilities[goals] = probabilities[goals - 1] * rate / goals
        probabilities /= probabilities.sum()
        distributions.append(probabilities)
    joint = distributions[0][:, None] * distributions[1][None, :]
    final_margin = int(score[0]) - int(score[1]) + values[:, None] - values[None, :]
    probabilities = np.asarray(
        (
            joint[final_margin > 0].sum(),
            joint[final_margin == 0].sum(),
            joint[final_margin < 0].sum(),
        ),
        dtype=np.float64,
    )
    return probabilities / probabilities.sum()


def _remaining_goal_rates(rate_curve, minute: float) -> np.ndarray:
    """5분 간격 fitted curve를 현재 경기 분으로 선형 보간한다."""

    curve = np.asarray(rate_curve, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] != 3 or curve.shape[0] < 2:
        raise ValueError("Poisson rate curve는 (minute, HOME, AWAY) [K,3]이어야 한다")
    if not np.all(np.isfinite(curve)) or np.any(curve[:, 1:] < 0.0):
        raise ValueError("Poisson rate curve는 finite nonnegative 값이어야 한다")
    return np.asarray(
        [np.interp(minute, curve[:, 0], curve[:, team]) for team in (1, 2)],
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class RenderStyle:
    """MP4 presentation과 encoder의 static 설정."""

    width_px: int = 1920
    height_px: int = 1080
    dpi: int = 150
    player_size: float = 210.0
    crf: int = 20
    encoder_preset: str = "veryfast"
    encoder_threads: int = 4
    camera_azimuth_degrees: float = -90.0
    camera_elevation_degrees: float = 33.0
    camera_distance_m: float = 150.0
    camera_focal_length: float = 820.0
    goal_hold_seconds: float = 2.0
    ball_trail_seconds: float = 0.8
    player_trail_seconds: float = 0.5

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("width_px", self.width_px, 320),
            ("height_px", self.height_px, 240),
            ("dpi", self.dpi, 50),
            ("encoder_threads", self.encoder_threads, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name}은 {minimum} 이상의 정수여야 한다")
        if self.width_px % 2 or self.height_px % 2:
            raise ValueError("H.264 yuv420p 출력 크기는 짝수여야 한다")
        if not all(
            math.isfinite(float(value)) and value > 0
            for value in (
                self.player_size,
                self.camera_distance_m,
                self.camera_focal_length,
                self.goal_hold_seconds,
            )
        ):
            raise ValueError("renderer 실수 설정은 finite positive여야 한다")
        if not all(
            math.isfinite(float(value)) and value >= 0.0
            for value in (self.ball_trail_seconds, self.player_trail_seconds)
        ):
            raise ValueError("trail duration은 finite nonnegative여야 한다")
        if not 0 < self.camera_elevation_degrees < 90:
            raise ValueError("camera elevation은 (0, 90) 범위여야 한다")
        if not -90 <= self.camera_azimuth_degrees <= 90:
            raise ValueError("camera azimuth는 [-90, 90] 범위여야 한다")
        if not 0 <= self.crf <= 51:
            raise ValueError("crf는 [0, 51] 범위여야 한다")

    @property
    def figsize(self) -> tuple[float, float]:
        return self.width_px / self.dpi, self.height_px / self.dpi


class _PerspectiveCamera:
    def __init__(self, style: RenderStyle) -> None:
        azimuth = np.deg2rad(style.camera_azimuth_degrees)
        elevation = np.deg2rad(style.camera_elevation_degrees)
        direction = np.asarray(
            (
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            )
        )
        self.position = style.camera_distance_m * direction
        forward = -direction
        right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
        right /= np.linalg.norm(right)
        self.rotation = np.stack((right, np.cross(right, forward), forward))
        self.focal_length = float(style.camera_focal_length)
        self.reference_depth = float(style.camera_distance_m)

    def project_with_scale(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        camera = (np.asarray(points) - self.position) @ self.rotation.T
        depth = np.clip(camera[..., 2], 1.0e-3, None)
        projected = np.stack(
            (
                self.focal_length * camera[..., 0] / depth,
                self.focal_length * camera[..., 1] / depth,
            ),
            axis=-1,
        )
        return projected, np.clip(self.reference_depth / depth, 0.62, 1.55)

    def project(self, points: np.ndarray) -> np.ndarray:
        return self.project_with_scale(points)[0]


def _ground(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return np.column_stack((points, np.zeros(points.shape[0])))


def _validated_field_half_extent(field_half_extent) -> tuple[float, float]:
    extent = np.asarray(field_half_extent, dtype=np.float64)
    if extent.shape != (2,) or not np.all(np.isfinite(extent)):
        raise ValueError("field_half_extent는 유한한 (2,) 값이어야 한다")
    if np.any(extent <= 0.0):
        raise ValueError("field_half_extent는 양수여야 한다")
    return float(extent[0]), float(extent[1])


def _pitch_geometry(
    field_half_extent=(52.5, 34.0),
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Pitch polygon, marking과 regulation goal/net segment를 만든다."""

    hx, hy = _validated_field_half_extent(field_half_extent)
    stripe_width = 2.0 * hx / 12.0
    stripes = [
        _ground(
            np.asarray(
                (
                    (-hx + i * stripe_width, -hy),
                    (-hx + (i + 1) * stripe_width, -hy),
                    (-hx + (i + 1) * stripe_width, hy),
                    (-hx + i * stripe_width, hy),
                )
            )
        )
        for i in range(12)
    ]
    lines: list[np.ndarray] = []

    def rectangle(x0: float, x1: float, y0: float, y1: float) -> None:
        lines.append(
            _ground(np.asarray(((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))))
        )

    rectangle(-hx, hx, -hy, hy)
    lines.append(_ground(np.asarray(((0.0, -hy), (0.0, hy)))))
    theta = np.linspace(0.0, 2.0 * np.pi, 72)
    lines.append(_ground(np.column_stack((9.15 * np.cos(theta), 9.15 * np.sin(theta)))))
    center_spot_theta = np.linspace(0.0, 2.0 * np.pi, 20)
    lines.append(
        _ground(
            np.column_stack(
                (0.18 * np.cos(center_spot_theta), 0.18 * np.sin(center_spot_theta))
            )
        )
    )
    goals: list[np.ndarray] = []
    penalty_depth = min(16.5, hx)
    penalty_half_width = min(20.16, hy)
    goal_area_depth = min(5.5, hx)
    goal_area_half_width = min(9.16, hy)
    goal_half_width = min(3.66, hy)
    for sign in (-1.0, 1.0):
        gx = sign * hx
        rectangle(
            gx,
            gx - sign * penalty_depth,
            -penalty_half_width,
            penalty_half_width,
        )
        rectangle(
            gx,
            gx - sign * goal_area_depth,
            -goal_area_half_width,
            goal_area_half_width,
        )
        spot = sign * max(hx - 11.0, 0.0)
        spot_theta = np.linspace(0.0, 2.0 * np.pi, 20)
        lines.append(
            _ground(
                np.column_stack(
                    (spot + 0.18 * np.cos(spot_theta), 0.18 * np.sin(spot_theta))
                )
            )
        )
        arc_theta = np.linspace(0.0, 2.0 * np.pi, 100)
        outside_box = sign * np.cos(arc_theta) <= -(penalty_depth - 11.0) / 9.15
        penalty_arc = np.column_stack(
            (spot + 9.15 * np.cos(arc_theta), 9.15 * np.sin(arc_theta))
        )[outside_box]
        if penalty_arc.shape[0] > 1:
            lines.append(_ground(penalty_arc))
        back = gx + sign * 2.44
        for y in (-goal_half_width, goal_half_width):
            goals.extend(
                (
                    np.asarray(((gx, y, 0.0), (gx, y, 2.44))),
                    np.asarray(((gx, y, 2.44), (back, y, 2.44))),
                    np.asarray(((back, y, 2.44), (back, y, 0.0))),
                )
            )
        goals.extend(
            (
                np.asarray(((gx, -goal_half_width, 2.44), (gx, goal_half_width, 2.44))),
                np.asarray(
                    ((back, -goal_half_width, 2.44), (back, goal_half_width, 2.44))
                ),
            )
        )
        for fraction in np.linspace(0.0, 1.0, 7):
            y = -goal_half_width + 2.0 * goal_half_width * fraction
            goals.append(np.asarray(((gx, y, 0.0), (back, y, 2.44))))
        for z in np.linspace(0.0, 2.44, 5):
            goals.append(
                np.asarray(((gx, -goal_half_width, z), (back, -goal_half_width, z)))
            )
            goals.append(
                np.asarray(((gx, goal_half_width, z), (back, goal_half_width, z)))
            )
    corner_radius = min(1.0, hx, hy)
    corner_theta = np.linspace(0.0, 0.5 * np.pi, 18)
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            lines.append(
                _ground(
                    np.column_stack(
                        (
                            x_sign * (hx - corner_radius * np.cos(corner_theta)),
                            y_sign * (hy - corner_radius * np.sin(corner_theta)),
                        )
                    )
                )
            )
    return stripes, lines, goals


_RING_THETA = np.linspace(0.0, 2.0 * np.pi, 48)
_RING_UNIT = np.column_stack((np.cos(_RING_THETA), np.sin(_RING_THETA)))
_EMPTY_RING = np.zeros((2, 2))


def _intent_ring_segments(
    camera: _PerspectiveCamera,
    position: np.ndarray,
    intent: np.ndarray,
    goalkeeper: np.ndarray,
    active: np.ndarray,
    *,
    contact_physics,
    ball_geometry,
) -> tuple[list[np.ndarray], np.ndarray]:
    ball_radius = ball_geometry.radius_m
    radius = np.full(intent.shape, contact_physics.carry_radius_m + ball_radius)
    radius = np.where(
        intent == 4,
        contact_physics.challenge_radius_m + ball_radius,
        radius,
    )
    radius = np.where(
        goalkeeper & ((intent == 1) | (intent == 3)),
        contact_physics.goalkeeper_radius_m + ball_radius,
        radius,
    )
    visible = active & (intent != 0)
    segments = [_EMPTY_RING] * position.shape[0]
    indices = np.flatnonzero(visible)
    if indices.size:
        world = np.zeros((indices.size, 48, 3))
        world[:, :, :2] = (
            position[indices, None, :] + radius[indices, None, None] * _RING_UNIT
        )
        projected = camera.project(world.reshape(-1, 3)).reshape(-1, 48, 2)
        for index, ring in zip(indices, projected):
            segments[index] = ring
    return segments, visible


def _install_text_bitmap_cache(renderer) -> bool:
    """문자별 Agg bitmap을 그대로 재생하는 artist별 한 항목 cache."""

    if not hasattr(renderer, "_draw_text_glyphs_and_boxes"):
        return False
    original = renderer.draw_text
    cache = {}

    class Recorder:
        def __init__(self, target):
            self.target = target
            self.images = []

        def draw_text_image(self, image, x, y, angle, gc):
            self.images.append((np.asarray(image).copy(), x, y, angle))
            self.target.draw_text_image(image, x, y, angle, gc)

        def __getattr__(self, name):
            return getattr(self.target, name)

    def draw_text(gc, x, y, text, prop, angle, ismath=False, mtext=None):
        if ismath or mtext is None:
            return original(gc, x, y, text, prop, angle, ismath, mtext)
        key = (
            text,
            hash(prop),
            x,
            y,
            angle,
            gc.get_antialiased(),
            mtext.get_fontfeatures(),
            mtext.get_language(),
        )
        previous = cache.get(id(mtext))
        if previous is not None and previous[0] == key:
            for image, ix, iy, rotation in previous[1]:
                renderer._renderer.draw_text_image(image, ix, iy, rotation, gc)
            return
        target = renderer._renderer
        recorder = Recorder(target)
        renderer._renderer = recorder
        try:
            original(gc, x, y, text, prop, angle, ismath, mtext)
        finally:
            renderer._renderer = target
        cache[id(mtext)] = (key, recorder.images)

    renderer.draw_text = draw_text
    return True


def _number_sprites(style: RenderStyle, player_id: np.ndarray):
    """선수 번호를 한 번만 rasterize해 RGBA sprite로 만든다.

    matplotlib text는 frame마다 font shaping을 다시 해서 22개 라벨이 draw 시간의
    59%를 차지했다(장당 20.8 ms 중 12.3 ms). Sprite를 미리 만들고 frame에서는
    numpy alpha compositing만 하면 그 비용이 사라진다.
    """

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib import patheffects

    scale = style.height_px / 1080.0
    size = max(8.0, 10.5 * scale)
    labels = ["GK"] + [str(int(identity)) for identity in np.unique(player_id)]
    sprites: dict[str, np.ndarray] = {}
    figure = plt.figure(figsize=(0.6, 0.3), dpi=style.dpi)
    try:
        for label in labels:
            figure.clf()
            figure.patch.set_alpha(0.0)
            figure.text(
                0.5,
                0.5,
                label,
                ha="center",
                va="center",
                color="#ffffff",
                fontsize=size,
                weight="bold",
                path_effects=(
                    patheffects.withStroke(
                        linewidth=max(1.0, 1.25 * scale), foreground="#14202b"
                    ),
                ),
            )
            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba()).copy()
            alpha = rgba[:, :, 3]
            rows = np.flatnonzero(alpha.any(axis=1))
            columns = np.flatnonzero(alpha.any(axis=0))
            if rows.size and columns.size:
                rgba = rgba[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]
            sprites[label] = rgba
    finally:
        plt.close(figure)
    return sprites


def _composite_sprite(
    frame: np.ndarray, sprite: np.ndarray, x: float, y: float
) -> None:
    """Sprite를 frame RGBA에 중심 정렬로 alpha compositing한다."""

    height, width = sprite.shape[:2]
    top = round(float(y)) - height // 2
    left = round(float(x)) - width // 2
    frame_height, frame_width = frame.shape[:2]
    source_top = max(0, -top)
    source_left = max(0, -left)
    top = max(0, top)
    left = max(0, left)
    rows = min(height - source_top, frame_height - top)
    columns = min(width - source_left, frame_width - left)
    if rows <= 0 or columns <= 0:
        return
    patch = sprite[source_top : source_top + rows, source_left : source_left + columns]
    target = frame[top : top + rows, left : left + columns]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    target[:, :, :3] = (
        patch[:, :, :3].astype(np.float32) * alpha
        + target[:, :, :3].astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)


class _ReplayRenderer:
    def __init__(
        self,
        style: RenderStyle,
        player_count: int,
        contact_physics,
        ball_geometry,
        field_half_extent=(52.5, 34.0),
        player_id: np.ndarray | None = None,
        analysis: dict | None = None,
        mode: str = "analysis",
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.collections import LineCollection, PolyCollection
        from matplotlib.colors import to_rgba
        from matplotlib.font_manager import FontProperties, findfont
        from matplotlib.patches import Circle, Rectangle

        self.style = style
        if not isinstance(mode, str) or mode not in ("analysis", "rollout"):
            raise ValueError("render mode는 'analysis' 또는 'rollout'이어야 한다")
        self.mode = mode
        self.camera = _PerspectiveCamera(style)
        self.contact_physics = contact_physics
        self.ball_geometry = ball_geometry
        self._to_rgba = to_rgba
        self.field_half_extent = _validated_field_half_extent(field_half_extent)
        self.fig = plt.figure(figsize=style.figsize, dpi=style.dpi, facecolor="#0c1622")
        try:
            self.ax = self.fig.add_axes((0.0, 0.235, 1.0, 0.765))
            self.ax.set_facecolor("#0c1622")
            self.ax.axis("off")
            stripes, markings, goals = _pitch_geometry(self.field_half_extent)
            self.ax.add_collection(
                PolyCollection(
                    [self.camera.project(item) for item in stripes],
                    facecolors=[("#2f8f3e", "#37a449")[i & 1] for i in range(12)],
                    edgecolors="none",
                    zorder=-3,
                )
            )
            self.ax.add_collection(
                LineCollection(
                    [self.camera.project(item) for item in markings],
                    colors="#eef7ec",
                    linewidths=1.2,
                    zorder=1,
                )
            )
            self.ax.add_collection(
                LineCollection(
                    [self.camera.project(item) for item in goals],
                    colors="#e6edf3",
                    linewidths=1.15,
                    alpha=0.90,
                    zorder=3,
                )
            )
            # FootballWorld와 같은 regulation-surface framing이다. `equal`이 없으면
            # Figure 종횡비가 perspective y축을 늘려 실제보다 고각으로 보인다.
            margin = 3.0
            half_length, half_width = self.field_half_extent
            bounds = self.camera.project(
                np.asarray(
                    [
                        (x, y, z)
                        for x in (-half_length - margin, half_length + margin)
                        for y in (-half_width - margin, half_width + margin)
                        for z in (0.0, 2.44)
                    ]
                )
            )
            span = np.ptp(bounds, axis=0)
            pad = 0.018 * max(span[0], span[1])
            self.ax.set_xlim(bounds[:, 0].min() - pad, bounds[:, 0].max() + pad)
            self.ax.set_ylim(bounds[:, 1].min() - pad, bounds[:, 1].max() + pad)
            self.ax.set_aspect("equal")
            zeros = np.zeros((player_count, 2), dtype=np.float32)
            self.shadows = self.ax.scatter(
                zeros[:, 0],
                zeros[:, 1],
                s=style.player_size * 0.86,
                c="#07120b",
                alpha=0.30,
                linewidths=0,
                zorder=4,
            )
            self.player_body_shadow = LineCollection(
                [], colors="#06110a", linewidths=6.0, alpha=0.72, zorder=6.0
            )
            self.player_body = LineCollection([], linewidths=2.8, zorder=6.2)
            self.player_heads = self.ax.scatter(
                zeros[:, 0], zeros[:, 1], s=42.0, c="#d9c4a5",
                edgecolors="#18202a", linewidths=0.7, zorder=7.2,
            )
            self.ax.add_collection(self.player_body_shadow)
            self.ax.add_collection(self.player_body)
            self.ring_underlay = LineCollection(
                [np.zeros((2, 2))] * player_count,
                colors="#071019",
                linewidths=6.4,
                zorder=5,
            )
            self.rings = LineCollection(
                [np.zeros((2, 2))] * player_count,
                colors="#ffffff",
                linewidths=3.8,
                zorder=6,
            )
            self.ax.add_collection(self.ring_underlay)
            self.ax.add_collection(self.rings)
            self.players = self.ax.scatter(
                zeros[:, 0],
                zeros[:, 1],
                s=style.player_size,
                c="#ffffff",
                edgecolors="#101010",
                linewidths=1.0,
                zorder=7,
            )
            if player_id is None:
                player_id = np.arange(1, player_count + 1, dtype=np.int32)
            self.player_id = np.asarray(player_id, dtype=np.int32)
            if self.player_id.shape != (player_count,):
                raise ValueError(
                    "rendering player_id shape은 (player,)여야 한다: "
                    f"{self.player_id.shape}"
                )
            self.sprites = _number_sprites(style, self.player_id)
            self.labels = tuple(
                self.ax.text(
                    0.0,
                    0.0,
                    "",
                    color="white",
                    fontsize=8.0,
                    weight="bold",
                    ha="center",
                    va="center",
                    zorder=9,
                )
                for _ in range(player_count)
            )
            self.ball_shadow = self.ax.scatter(
                (0.0,),
                (0.0,),
                s=58.0,
                c="#07120b",
                alpha=0.32,
                linewidths=0,
                zorder=8,
            )
            self.ball = self.ax.scatter(
                (0.0,),
                (0.0,),
                s=104.0,
                c="white",
                edgecolors="#111827",
                linewidths=1.1,
                zorder=12,
            )
            self.ball_shading = self.ax.scatter(
                (0.0,), (0.0,), s=46.0, c="#aeb8c2", alpha=0.78,
                edgecolors="none", zorder=13,
            )
            self.ball_highlight = self.ax.scatter(
                (0.0,), (0.0,), s=13.0, c="#ffffff", alpha=0.98,
                edgecolors="none", zorder=14,
            )
            self.player_trails = LineCollection([], linewidths=2.0, zorder=6.7)
            self.ball_trail = LineCollection([], linewidths=2.8, zorder=7.5)
            self.ax.add_collection(self.player_trails)
            self.ax.add_collection(self.ball_trail)
            self.minimap_ax = self.fig.add_axes((0.83, 0.775, 0.15, 0.18))
            self.minimap_ax.set_facecolor("#164b2a")
            hx, hy = self.field_half_extent
            self.minimap_ax.set_xlim(-hx, hx)
            self.minimap_ax.set_ylim(-hy, hy)
            self.minimap_ax.set_aspect("equal")
            self.minimap_ax.set_xticks([])
            self.minimap_ax.set_yticks([])
            for spine in self.minimap_ax.spines.values():
                spine.set_color("#d5e4d8")
                spine.set_linewidth(1.2)
            self.minimap_ax.add_patch(Rectangle((-hx, -hy), 2 * hx, 2 * hy,
                fill=False, edgecolor="#d5e4d8", linewidth=0.8))
            self.minimap_ax.plot((0, 0), (-hy, hy), color="#d5e4d8", linewidth=0.6)
            self.minimap_ax.add_patch(Circle((0, 0), 9.15, fill=False,
                edgecolor="#d5e4d8", linewidth=0.6))
            for sign in (-1.0, 1.0):
                self.minimap_ax.add_patch(Rectangle(
                    (sign * hx if sign < 0 else hx - min(16.5, hx), -min(20.16, hy)),
                    min(16.5, hx), 2 * min(20.16, hy),
                    fill=False, edgecolor="#d5e4d8", linewidth=0.55,
                ))
                self.minimap_ax.add_patch(Rectangle(
                    (sign * hx if sign < 0 else hx - min(5.5, hx), -min(9.16, hy)),
                    min(5.5, hx), 2 * min(9.16, hy),
                    fill=False, edgecolor="#d5e4d8", linewidth=0.45,
                ))
            self.minimap_players = self.minimap_ax.scatter(
                np.zeros(player_count), np.zeros(player_count), s=22,
                c="#ffffff", edgecolors="#101010", linewidths=0.5, zorder=3)
            self.minimap_ball = self.minimap_ax.scatter((0,), (0,), s=26,
                c="#ffffff", edgecolors="#101010", linewidths=0.4, zorder=4)
            self.minimap_players.set_animated(True)
            self.minimap_ball.set_animated(True)
            self.analysis = analysis or {}
            self.team_names = tuple(self.analysis.get("team_names", ("HOME", "AWAY")))
            self.metric_history = tuple(self.analysis.get("metric_history", ()))
            self.metric_times = tuple(
                float(point["time_min"]) for point in self.metric_history
            )
            self._metric_visible_index = -1
            self.metric_dynamic_artists = []
            self.metric_block_artists = {}
            self.stamina_block_artists = []
            self.win_probability_rate_model = self.analysis.get(
                "win_probability_rate_model"
            )
            self._win_probability_key = None
            self._win_probability_last_score = None
            self._win_probability_last_phase = None
            ui_scale = max(0.55, min(1.0, style.height_px / 1080.0))
            try:
                analysis_font = FontProperties(
                    fname=findfont("Noto Sans CJK KR", fallback_to_default=False)
                )
            except ValueError:
                analysis_font = None
            dashboard = self.fig.add_axes((0.025, 0.02, 0.95, 0.26))
            self.dashboard_ax = dashboard
            dashboard_text = lambda *args, **kwargs: dashboard.text(
                *args, fontproperties=analysis_font, **kwargs
            )
            dashboard.set_facecolor("#111a25")
            dashboard.set_xticks([])
            dashboard.set_yticks([])
            for spine in dashboard.spines.values():
                spine.set_color("#324252")
            dashboard.set_facecolor("#111a25")
            columns = (0.02, 0.155, 0.290, 0.425, 0.605, 0.800)
            for divider in (0.140, 0.275, 0.410, 0.590, 0.785):
                dashboard.axvline(divider, color="#324252", linewidth=0.7)
            leaders = (
                self.metric_history[0].get("leaders", {})
                if self.metric_history
                else self.analysis.get("leaders", {})
            )
            for column, title, metric in (
                (0, "DISTANCE (km)", "distance_km"),
                (1, "TOP SPEED (km/h)", "top_speed_kmh"),
            ):
                x = columns[column]
                dashboard_text(x, 0.86, title, color="#d9e5ef", fontsize=9.4 * ui_scale,
                    weight="bold", transform=dashboard.transAxes)
                entries = leaders.get(metric, ())
                for team_index in range(2):
                    group_top = 0.76 - 0.39 * team_index
                    dashboard_text(x, group_top, self.team_names[team_index],
                        color=_TEAM_COLORS[team_index], fontsize=9.2 * ui_scale,
                        weight="bold", transform=dashboard.transAxes)
                    team_entries = [row for row in entries if int(row["team"]) == team_index][:5]
                    block = dashboard_text(
                        x + 0.008, group_top - 0.035,
                        "", color=_TEAM_COLORS[team_index], fontsize=7.0 * ui_scale,
                        family="monospace", va="top", linespacing=1.15,
                        transform=dashboard.transAxes,
                    )
                    if not self.metric_history and team_entries:
                        lines = []
                        for rank, row in enumerate(team_entries, start=1):
                            value = (
                                f"{float(row[metric]):.3f}"
                                if metric == "distance_km"
                                else f"{float(row[metric]):.1f}"
                            )
                            lines.append(f"{rank}. #{int(row['player_id']):>2}  {value:>5}")
                        block.set_text("\n".join(lines))
                    elif not self.metric_history:
                        block.set_text("No data")
                    if self.metric_history:
                        self.metric_dynamic_artists.append(block)
                    self.metric_block_artists.setdefault(metric, []).append(block)
                if metric == "top_speed_kmh":
                    dashboard_text(x, 0.04, "Based on 10 Hz positions", color="#9aabba",
                        fontsize=5.5 * ui_scale, transform=dashboard.transAxes)
            x = columns[3]
            dashboard_text(x, 0.86, "BALL POSSESSION (min)", color="#d9e5ef",
                fontsize=8.5 * ui_scale, weight="bold", transform=dashboard.transAxes)
            for team_index in range(2):
                group_top = 0.76 - 0.39 * team_index
                dashboard_text(x, group_top, self.team_names[team_index],
                    color=_TEAM_COLORS[team_index], fontsize=9.0 * ui_scale,
                    weight="bold", transform=dashboard.transAxes)
                team_entries = [row for row in leaders.get("possession_min", ())
                                if int(row["team"]) == team_index][:5]
                block = dashboard_text(
                    x + 0.008, group_top - 0.035, "",
                    color=_TEAM_COLORS[team_index], fontsize=6.8 * ui_scale,
                    family="monospace", va="top", linespacing=1.15,
                    transform=dashboard.transAxes,
                )
                if not self.metric_history and team_entries:
                    block.set_text("\n".join(
                        f"{rank}. #{int(row['player_id']):>2}  {float(row['possession_min']):>4.1f}"
                        for rank, row in enumerate(team_entries, start=1)
                    ))
                elif not self.metric_history:
                    block.set_text("No data")
                if self.metric_history:
                    self.metric_dynamic_artists.append(block)
                self.metric_block_artists.setdefault("possession_min", []).append(block)
            stamina = self.analysis.get("final_stamina", ())
            stamina_players = self.analysis.get("final_stamina_players", ())
            if self.metric_history:
                stamina = self.metric_history[0].get("stamina", ())
            x = columns[2]
            dashboard_text(x, 0.86, "STAMINA · LOWEST LONG", color="#d9e5ef",
                fontsize=9.4 * ui_scale, weight="bold", transform=dashboard.transAxes)
            if len(stamina) == 2:
                for team_index, team_name in enumerate(self.team_names):
                    row = stamina[team_index]
                    group_top = 0.75 - 0.36 * team_index
                    dashboard_text(
                        x, group_top,
                        team_name,
                        color=_TEAM_COLORS[team_index], fontsize=9.0 * ui_scale,
                        weight="bold",
                        transform=dashboard.transAxes,
                    )
                    if "players" in row:
                        team_players = list(row["players"])
                    else:
                        team_players = sorted(
                            (item for item in stamina_players if int(item["team"]) == team_index),
                            key=lambda item: float(item["long"]),
                        )[:3]
                    block = dashboard_text(
                        x + 0.008, group_top - 0.035, "",
                        color=_TEAM_COLORS[team_index], fontsize=6.3 * ui_scale,
                        family="monospace", va="top", linespacing=1.15,
                        transform=dashboard.transAxes,
                    )
                    if not self.metric_history:
                        lines = [
                            f"AVG L{100*row['long_mean']:.0f} S{100*row['short_mean']:.0f}"
                        ]
                        lines.extend(
                            f"{rank}. #{int(player['player_id']):>2} "
                            f"L{100*float(player['long']):.0f} S{100*float(player['short']):.0f}"
                            for rank, player in enumerate(team_players, start=1)
                        )
                        block.set_text("\n".join(lines))
                    if self.metric_history:
                        self.metric_dynamic_artists.append(block)
                    self.stamina_block_artists.append(block)
            else:
                dashboard_text(x, 0.52, "No data", color="#9aabba",
                    fontsize=7.0 * ui_scale, transform=dashboard.transAxes)
            x = columns[4]
            dashboard_text(x, 0.88, "CUMULATIVE POSSESSION (%)", color="#d9e5ef", fontsize=8.2 * ui_scale,
                weight="bold", transform=dashboard.transAxes)
            possession_history = self.analysis.get("possession_history", ())
            self.possession_history = tuple(
                (float(point["time_min"]), float(point["team_share"][0]),
                 float(point["team_share"][1]))
                for point in possession_history
            )
            self.possession_times = tuple(point[0] for point in self.possession_history)
            self._possession_visible_count = -1
            self.possession_ax = self.fig.add_axes((0.615, 0.14, 0.16, 0.09))
            self.possession_ax.set_facecolor("#111a25")
            self.possession_ax.set_xlim(0.0, 110.0)
            self.possession_ax.set_ylim(0.0, 1.0)
            self.possession_ax.set_xticks((0, 45, 90), labels=("0", "45", "90"))
            self.possession_ax.set_yticks((0.0, 0.5, 1.0), labels=("0", "50", "100"))
            self.possession_ax.tick_params(
                colors="#9aabba", labelsize=5.8 * ui_scale, length=1.4, pad=1.2
            )
            self.possession_ax.grid(axis="y", color="#324252", linewidth=0.45)
            for spine in self.possession_ax.spines.values():
                spine.set_color("#324252")
            self.possession_lines = tuple(
                self.possession_ax.plot(
                    (), (), color=_TEAM_COLORS[team_index], linewidth=2.0,
                    label=self.team_names[team_index], animated=True,
                )[0]
                for team_index in range(2)
            )
            possession_legend = self.possession_ax.legend(
                loc="upper center", ncol=2, frameon=False,
                fontsize=5.5 * ui_scale, handlelength=1.0,
                labelcolor="#d9e5ef", borderaxespad=0.1,
            )
            if possession_legend is not None:
                possession_legend.set_zorder(5)
            substitutions = self.analysis.get("substitutions")
            if substitutions is None:
                dashboard_text(x, 0.28, "No substitution data",
                    color="#9aabba", fontsize=7.2 * ui_scale,
                    transform=dashboard.transAxes)
            else:
                counts = tuple(sum(int(row["team"]) == t for row in substitutions)
                               for t in range(2))
                dashboard_text(x, 0.30,
                    f"SUB  {self.team_names[0]} {counts[0]} · {self.team_names[1]} {counts[1]}",
                    color="#d9e5ef", fontsize=7.0 * ui_scale,
                    transform=dashboard.transAxes)
                latest_by_team = {
                    team_index: next(
                        (row for row in reversed(substitutions)
                         if int(row["team"]) == team_index),
                        None,
                    )
                    for team_index in range(2)
                }
                for team_index, row in latest_by_team.items():
                    if row is None:
                        text = f"{self.team_names[team_index]}  No substitutions"
                    else:
                        text = (f"{self.team_names[team_index]}  OUT #{row['out_player_id']} → "
                                f"IN #{row['in_player_id']}  {int(row['minute'])}′")
                    dashboard_text(x, 0.19 - 0.10 * team_index, text,
                        color=_TEAM_COLORS[team_index], fontsize=6.6 * ui_scale,
                        transform=dashboard.transAxes)
            timeline_x = columns[5]
            dashboard_text(timeline_x, 0.89, "WIN PROBABILITY · Poisson", color="#d9e5ef",
                fontsize=8.2 * ui_scale, weight="bold", transform=dashboard.transAxes)
            dashboard_text(timeline_x, 0.31, "MATCH MOMENTUM · LAST 5 MIN", color="#9aabba",
                fontsize=6.4 * ui_scale, transform=dashboard.transAxes)
            momentum = self.analysis.get("momentum", ())
            self.momentum_ax = None
            self.momentum_collection = None
            self.momentum_times = ()
            self._momentum_colors = None
            self._momentum_visible_count = -1
            if momentum:
                times = np.asarray([float(point["time_min"]) for point in momentum])
                balance = np.asarray([float(point["balance"]) for point in momentum])
                momentum_ax = self.fig.add_axes((0.805, 0.035, 0.165, 0.045))
                self.momentum_ax = momentum_ax
                self.momentum_times = tuple(float(value) for value in times)
                momentum_ax.set_facecolor("#111a25")
                width = max(float(np.median(np.diff(times))) if len(times) > 1 else 0.5, 0.2)
                colors = np.asarray([
                    self._to_rgba(_TEAM_COLORS[0] if value >= 0 else _TEAM_COLORS[1])
                    for value in balance
                ])
                colors[:, 3] = 0.88
                self._momentum_colors = colors
                vertices = [
                    ((time - width / 2, 0.0), (time - width / 2, value),
                     (time + width / 2, value), (time + width / 2, 0.0))
                    for time, value in zip(times, balance)
                ]
                self.momentum_collection = PolyCollection(
                    vertices,
                    facecolors=np.column_stack((colors[:, :3], np.zeros(len(colors)))),
                    edgecolors="none",
                    animated=True,
                )
                momentum_ax.add_collection(self.momentum_collection)
                momentum_ax.axhline(0.0, color="#9aabba", linewidth=0.55)
                momentum_ax.set_ylim(-1.0, 1.0)
                momentum_ax.set_xlim(0.0, max(float(times[-1]), 1.0))
                momentum_ax.set_yticks(
                    (-1.0, 0.0, 1.0),
                    labels=(self.team_names[1][:6], "0", self.team_names[0][:6]),
                )
                momentum_ax.tick_params(colors="#9aabba", labelsize=4.8 * ui_scale, length=1)
                for spine in momentum_ax.spines.values():
                    spine.set_color("#324252")
            else:
                dashboard_text(timeline_x, 0.20, "No data", color="#9aabba", fontsize=6 * ui_scale,
                    transform=dashboard.transAxes)
            self.win_probability_ax = self.fig.add_axes((0.805, 0.15, 0.165, 0.08))
            self.win_probability_ax.set_facecolor("#111a25")
            self.win_probability_ax.set_xlim(0.0, 110.0)
            self.win_probability_ax.set_ylim(0.0, 1.0)
            self.win_probability_ax.set_xticks((0, 45, 90), labels=("0", "45", "90"))
            self.win_probability_ax.set_yticks((0.0, 0.5, 1.0), labels=("0", "50", "100"))
            self.win_probability_ax.tick_params(
                colors="#9aabba", labelsize=6.8 * ui_scale, length=1.8, pad=1.8
            )
            self.win_probability_ax.grid(axis="y", color="#324252", linewidth=0.45)
            for spine in self.win_probability_ax.spines.values():
                spine.set_color("#324252")
            self.win_probability_points = [
                tuple(float(value) for value in point)
                for point in self.analysis.get("win_probability_history", ())
            ]
            self.win_probability_lines = tuple(
                self.win_probability_ax.plot(
                    (), (), color=color, linewidth=2.0, label=label, animated=True
                )[0]
                for label, color in (
                    (self.team_names[0], _TEAM_COLORS[0]),
                    ("DRAW", "#d9e5ef"),
                    (self.team_names[1], _TEAM_COLORS[1]),
                )
            )
            if self.win_probability_rate_model is None:
                dashboard_text(timeline_x, 0.64, "Analysis report required", color="#9aabba",
                    fontsize=5.8 * ui_scale, transform=dashboard.transAxes)
            else:
                legend = self.win_probability_ax.legend(
                    loc="upper center", ncol=3, frameon=False,
                    fontsize=6.0 * ui_scale, handlelength=1.3, columnspacing=1.0,
                    labelcolor="#d9e5ef", borderaxespad=0.1,
                )
                if legend is not None:
                    legend.set_zorder(5)
            for line_index, line in enumerate(self.win_probability_lines):
                if self.win_probability_points:
                    line.set_data(
                        [point[0] for point in self.win_probability_points],
                        [point[line_index + 1] for point in self.win_probability_points],
                    )
            self.ax.add_patch(
                Rectangle(
                    (0.33, 0.905),
                    0.34,
                    0.084,
                    transform=self.ax.transAxes,
                    color="#071019",
                    alpha=0.94,
                    zorder=30,
                )
            )
            self.score = self.ax.text(
                0.5,
                0.965,
                "0 : 0",
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                color="white",
                fontsize=14,
                weight="bold",
                zorder=31,
            )
            self.clock = self.ax.text(
                0.5,
                0.918,
                "00:00",
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                color="#b8c7d6",
                fontsize=11,
                family="monospace",
                zorder=31,
            )
            for x, label, color in (
                (0.395, self.team_names[0], _TEAM_COLORS[0]),
                (0.605, self.team_names[1], _TEAM_COLORS[1]),
            ):
                self.ax.text(
                    x,
                    0.965,
                    label,
                    transform=self.ax.transAxes,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=12,
                    weight="bold",
                    zorder=31,
                )
            self.goal = self.ax.text(
                0.5,
                0.55,
                "GOAL",
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                color="#fff5c2",
                fontsize=60.0 * style.height_px / 1080.0,
                weight="bold",
                visible=False,
                zorder=40,
            )
            # Intent ring 색을 경기장 하단에 설명한다. Static이므로 배경에 포함된다.
            legend = (
                ("CONTROL", _INTENT_COLORS[1]),
                ("PASS", _INTENT_COLORS[2]),
                ("SHOOT", _INTENT_COLORS[3]),
                ("CHALLENGE", _INTENT_COLORS[4]),
            )
            self.ax.add_patch(
                Rectangle(
                    (0.014, 0.020),
                    0.360,
                    0.052,
                    transform=self.fig.transFigure,
                    color="#071019",
                    alpha=0.90,
                    clip_on=False,
                    zorder=30,
                )
            )
            for index, (label, color) in enumerate(legend):
                x = 0.034 + index * 0.088
                self.ax.scatter(
                    (x,),
                    (0.305,),
                    s=52.0,
                    transform=self.fig.transFigure,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.9,
                    clip_on=False,
                    zorder=31,
                )
                self.ax.text(
                    x + 0.013,
                    0.305,
                    label,
                    transform=self.fig.transFigure,
                    ha="left",
                    va="center",
                    color="#f4f7fa",
                    fontsize=7.0 * style.height_px / 1080.0,
                    weight="bold",
                    zorder=31,
                )

            # Static pitch를 한 번만 rasterize하고 매 frame은 동적 artist만 다시
            # 그린다. 기존 구현은 frame마다 canvas.draw()로 전체를 다시 그려
            # 1080p 108,000 frame에서 인코딩이 rollout보다 10배 오래 걸렸다.
            # 선수 번호는 matplotlib text가 아니라 sprite compositing으로 그린다.
            for label in self.labels:
                label.set_visible(False)
            self._dynamic = (
                self.shadows,
                self.ring_underlay,
                self.rings,
                self.player_trails,
                self.ball_trail,
                self.player_body_shadow,
                self.player_body,
                self.player_heads,
                self.players,
                self.ball_shadow,
                self.ball,
                self.ball_shading,
                self.ball_highlight,
                self.score,
                self.clock,
                self.goal,
            )
            for artist in self._dynamic:
                artist.set_animated(True)
            if self.metric_history:
                self._update_metric_snapshot(self.metric_history[0])
                self._metric_visible_index = 0
            self.fig.canvas.draw()
            if self.mode == "rollout":
                self.ax.set_position((0.0, 0.0, 1.0, 1.0))
                for analysis_ax in (
                    self.dashboard_ax,
                    self.possession_ax,
                    self.win_probability_ax,
                    self.momentum_ax,
                ):
                    if analysis_ax is not None:
                        analysis_ax.set_visible(False)
                self.fig.canvas.draw()
            self._background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
            _install_text_bitmap_cache(self.fig.canvas.get_renderer())
            self._roster_key = None
            self._intent_key = None
            self._label_keys = ()
        except BaseException:
            with suppress(BaseException):
                plt.close(self.fig)
            raise

    def draw(
        self,
        *,
        position: np.ndarray,
        active: np.ndarray,
        ball: np.ndarray,
        score: np.ndarray,
        intent: np.ndarray,
        team: np.ndarray,
        goalkeeper: np.ndarray,
        seconds: float,
        second_half: bool = False,
        elapsed_minutes: float | None = None,
        goal_visible: bool,
        player_trail: np.ndarray | None = None,
        player_active_trail: np.ndarray | None = None,
        ball_trail: np.ndarray | None = None,
    ) -> memoryview:
        if elapsed_minutes is None:
            elapsed_minutes = float(seconds) / 60.0
        count = position.shape[0]
        world = np.column_stack((position, np.full(count, 0.72)))
        projected, scale = self.camera.project_with_scale(world)
        ground = self.camera.project(np.column_stack((position, np.zeros(count))))
        projected[~active] = np.nan
        ground[~active] = np.nan
        self.shadows.set_offsets(ground)
        self.players.set_offsets(projected)
        self.shadows.set_sizes(self.style.player_size * 0.28 * scale * scale * active)
        self.players.set_sizes(self.style.player_size * 0.12 * scale * scale * active)
        offsets = np.asarray((
            (-0.18, 0.0, 0.03), (-0.12, 0.0, 0.40),
            (0.18, 0.0, 0.03), (0.12, 0.0, 0.40), (0.0, 0.0, 0.72),
            (-0.42, 0.0, 0.86), (-0.20, 0.0, 1.20), (0.0, 0.0, 1.32),
            (0.20, 0.0, 1.20), (0.42, 0.0, 0.86), (0.0, 0.0, 1.58),
        ), dtype=np.float64)
        body_world = np.zeros((count, offsets.shape[0], 3), dtype=np.float64)
        body_world[:, :, :2] = position[:, None, :] + offsets[None, :, :2]
        body_world[:, :, 2] = offsets[None, :, 2]
        body_projected, body_scale = self.camera.project_with_scale(
            body_world.reshape(-1, 3)
        )
        body_projected = body_projected.reshape(count, offsets.shape[0], 2)
        body_segments = []
        body_colors = []
        body_widths = []
        body_shadows = []
        for player in np.flatnonzero(active):
            paths = (
                body_projected[player, (0, 1, 4, 3, 2)],
                body_projected[player, (4, 7)],
                body_projected[player, (5, 6, 8, 9)],
            )
            for path_index, segment in enumerate(paths):
                body_segments.append(segment)
                body_shadows.append(segment)
                color = list(self._to_rgba(
                    "#27313b" if path_index == 0 else
                    (_GK_COLORS[int(team[player])] if goalkeeper[player]
                     else _TEAM_COLORS[int(team[player])])
                ))
                color[3] = 0.96
                body_colors.append(color)
                body_widths.append((2.8, 4.4, 3.1)[path_index])
        self.player_body_shadow.set_segments(body_shadows)
        self.player_body.set_segments(body_segments)
        self.player_body.set_color(body_colors)
        self.player_body.set_linewidth(body_widths)
        head_positions = body_projected[:, 10, :].copy()
        head_positions[~active] = np.nan
        self.player_heads.set_offsets(head_positions)
        head_scale = body_scale.reshape(count, offsets.shape[0])[:, 10]
        self.player_heads.set_sizes(42.0 * head_scale * head_scale * active)
        self.player_trails.set_segments([])
        self.ball_trail.set_segments([])
        if player_trail is not None and len(player_trail) > 1:
            history = np.asarray(player_trail, dtype=np.float64)
            active_history = np.asarray(player_active_trail, dtype=bool)
            world_history = np.concatenate(
                (history, np.zeros((*history.shape[:-1], 1))), axis=-1
            )
            projected_history = self.camera.project(world_history.reshape(-1, 3)).reshape(
                history.shape[0], history.shape[1], 2
            )
            segments = np.stack(
                (projected_history[:-1], projected_history[1:]), axis=2
            )
            distances = np.linalg.norm(np.diff(history, axis=0), axis=-1)
            visible = (
                active_history[:-1]
                & active_history[1:]
                & (distances <= 8.0)
                & np.isfinite(segments).all(axis=(2, 3))
            )
            denominator = max(history.shape[0] - 2, 1)
            alpha = 0.04 + 0.42 * np.arange(
                1, history.shape[0], dtype=np.float64
            ) / denominator
            team_colors = np.asarray(
                [self._to_rgba(_TEAM_COLORS[int(team_index)]) for team_index in team],
                dtype=np.float64,
            )
            colors = np.broadcast_to(
                team_colors[None, :, :], (*visible.shape, 4)
            ).copy()
            colors[:, :, 3] = alpha[:, None]
            self.player_trails.set_segments(segments[visible])
            if np.any(visible):
                self.player_trails.set_color(colors[visible])
        if ball_trail is not None and len(ball_trail) > 1:
            history = np.asarray(ball_trail, dtype=np.float64)
            projected_history = self.camera.project(history)
            segments = np.stack((projected_history[:-1], projected_history[1:]), axis=1)
            distances = np.linalg.norm(np.diff(history[:, :2], axis=0), axis=-1)
            visible = (distances <= 18.0) & np.isfinite(segments).all(axis=(1, 2))
            denominator = max(history.shape[0] - 2, 1)
            colors = np.tile(self._to_rgba("#ffffff"), (len(visible), 1))
            colors[:, 3] = np.clip(
                0.06 + 0.68 * np.arange(
                    1, history.shape[0], dtype=np.float64
                ) / denominator,
                0.0,
                1.0,
            )
            self.ball_trail.set_segments(segments[visible])
            if np.any(visible):
                self.ball_trail.set_color(colors[visible])
        minimap_position = np.asarray(position).copy()
        minimap_position[~active] = np.nan
        self.minimap_players.set_offsets(minimap_position)
        self.minimap_ball.set_offsets(np.asarray(ball)[None, :2])
        roster_key = (team.tobytes(), goalkeeper.tobytes())
        if roster_key != self._roster_key:
            self.players.set_facecolors(
                np.where(goalkeeper, _GK_COLORS[team], _TEAM_COLORS[team])
            )
            self.minimap_players.set_facecolors(
                np.where(goalkeeper, _GK_COLORS[team], _TEAM_COLORS[team])
            )
            self._label_keys = tuple(
                "GK" if goalkeeper[index] else str(int(self.player_id[index]))
                for index in range(count)
            )
            self._roster_key = roster_key
        segments, shown = _intent_ring_segments(
            self.camera,
            position,
            intent,
            goalkeeper,
            active,
            contact_physics=self.contact_physics,
            ball_geometry=self.ball_geometry,
        )
        self.ring_underlay.set_segments(segments)
        self.rings.set_segments(segments)
        intent_key = (intent.tobytes(), shown.tobytes())
        if intent_key != self._intent_key:
            self.rings.set_color(_INTENT_COLORS[np.clip(intent, 0, 4)])
            self.rings.set_alpha(shown.astype(np.float32))
            self.ring_underlay.set_alpha(shown.astype(np.float32) * 0.72)
            self._intent_key = intent_key
        ball_ground = self.camera.project(np.asarray(((ball[0], ball[1], 0.0),)))
        ball_projected, ball_scale = self.camera.project_with_scale(np.asarray((ball,)))
        self.ball_shadow.set_offsets(ball_ground)
        self.ball.set_offsets(ball_projected)
        self.ball_size = 104.0 * ball_scale[0] ** 2
        self.ball.set_sizes(np.asarray((self.ball_size,)))
        ball_display = self.ax.transData.transform(ball_projected[0])
        ball_radius_px = np.sqrt(self.ball_size) * self.style.dpi / 144.0
        inverse_transform = self.ax.transData.inverted()
        shade_display = ball_display + np.asarray((0.16, -0.18)) * ball_radius_px
        highlight_display = ball_display + np.asarray((-0.30, 0.28)) * ball_radius_px
        self.ball_shading.set_offsets(inverse_transform.transform(shade_display)[None, :])
        self.ball_shading.set_sizes(np.asarray((46.0 * ball_scale[0] ** 2,)))
        self.ball_highlight.set_offsets(inverse_transform.transform(highlight_display)[None, :])
        self.ball_highlight.set_sizes(np.asarray((13.0 * ball_scale[0] ** 2,)))
        self.score.set_text(f"{int(score[0])} : {int(score[1])}")
        self.clock.set_text(f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}")
        if self.win_probability_rate_model is not None:
            probability_second = int(max(seconds, 0.0) // 5.0)
            probability_key = (
                int(score[0]), int(score[1]), bool(second_half), probability_second
            )
            if probability_key != self._win_probability_key:
                phase = "second_half" if second_half else "first_half"
                curve = self.win_probability_rate_model[phase]
                sample_seconds = probability_second * 5.0
                rates = _remaining_goal_rates(curve, sample_seconds / 60.0)
                home, draw, away = _poisson_wdl_probabilities(score, rates)
                self._win_probability_key = probability_key
                point = (float(elapsed_minutes), float(home), float(draw), float(away))
                previous_score = self._win_probability_last_score
                previous_phase = self._win_probability_last_phase
                previous_minute = (
                    self.win_probability_points[-1][0]
                    if self.win_probability_points else None
                )
                should_append = (
                    previous_minute is None
                    or elapsed_minutes - previous_minute >= 0.5
                    or previous_score != (int(score[0]), int(score[1]))
                    or previous_phase != bool(second_half)
                )
                if should_append:
                    if previous_minute is not None and elapsed_minutes <= previous_minute:
                        self.win_probability_points[-1] = point
                    else:
                        self.win_probability_points.append(point)
                    if len(self.win_probability_points) > 512:
                        del self.win_probability_points[:-512]
                    for line_index, line in enumerate(self.win_probability_lines):
                        line.set_data(
                            [item[0] for item in self.win_probability_points],
                            [item[line_index + 1] for item in self.win_probability_points],
                        )
                self._win_probability_last_score = (int(score[0]), int(score[1]))
                self._win_probability_last_phase = bool(second_half)
        possession_count = bisect_right(
            self.possession_times, float(elapsed_minutes) + 1.0e-9
        )
        if possession_count != self._possession_visible_count:
            visible = self.possession_history[:possession_count]
            for team_index, line in enumerate(self.possession_lines):
                line.set_data(
                    [point[0] for point in visible],
                    [point[team_index + 1] for point in visible],
                )
            self._possession_visible_count = possession_count
        if self.metric_history:
            metric_count = bisect_right(
                self.metric_times, float(elapsed_minutes) + 1.0e-9
            )
            metric_index = max(metric_count - 1, 0)
            if metric_index != self._metric_visible_index:
                self.fig.canvas.restore_region(self._background)
                self._update_metric_snapshot(self.metric_history[metric_index])
                self._metric_visible_index = metric_index
                for artist in self.metric_dynamic_artists:
                    self.dashboard_ax.draw_artist(artist)
                self.fig.canvas.blit(self.fig.bbox)
                self._background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        if self.momentum_collection is not None:
            momentum_count = bisect_right(
                self.momentum_times, float(elapsed_minutes) + 1.0e-9
            )
            if momentum_count != self._momentum_visible_count:
                colors = self._momentum_colors.copy()
                colors[momentum_count:, 3] = 0.0
                self.momentum_collection.set_facecolors(colors)
                self._momentum_visible_count = momentum_count
        self.goal.set_visible(goal_visible)
        self.fig.canvas.restore_region(self._background)
        for artist in self._dynamic:
            self.ax.draw_artist(artist)
        for line in self.win_probability_lines:
            self.win_probability_ax.draw_artist(line)
        for line in self.possession_lines:
            self.possession_ax.draw_artist(line)
        if self.momentum_collection is not None:
            self.momentum_ax.draw_artist(self.momentum_collection)
        self.minimap_ax.draw_artist(self.minimap_players)
        self.minimap_ax.draw_artist(self.minimap_ball)
        self.fig.canvas.blit(self.fig.bbox)
        frame = np.asarray(self.fig.canvas.buffer_rgba())
        height = frame.shape[0]
        # 흰 번호 sprite는 선수 몸의 투영 bbox 바깥 네 방향 중 빈 곳에 둔다.
        # 변환 결과는 sprite compositor용 왼쪽 위 원점 pixel 좌표다.
        frame_height, frame_width = frame.shape[:2]
        body_display = self.ax.transData.transform(
            body_projected.reshape(-1, 2)
        ).reshape(count, offsets.shape[0], 2)
        body_display[:, :, 1] = frame_height - body_display[:, :, 1]
        body_boxes = np.full((count, 4), np.nan, dtype=np.float64)
        for player in np.flatnonzero(active):
            points = body_display[player]
            body_boxes[player] = (
                np.min(points[:, 0]) - 2.0,
                np.min(points[:, 1]) - 2.0,
                np.max(points[:, 0]) + 2.0,
                np.max(points[:, 1]) + 2.0,
            )
        ball_display = self.ax.transData.transform(ball_projected[0])
        ball_display[1] = frame_height - ball_display[1]
        ball_radius = max(4.0, np.sqrt(self.ball_size) * self.style.dpi / 144.0 + 2.0)
        ball_box = np.asarray((
            ball_display[0] - ball_radius,
            ball_display[1] - ball_radius,
            ball_display[0] + ball_radius,
            ball_display[1] + ball_radius,
        ))
        label_boxes: list[np.ndarray] = []
        for index in range(count):
            if not active[index]:
                continue
            sprite = self.sprites.get(self._label_keys[index])
            if sprite is None:
                continue
            sprite_height, sprite_width = sprite.shape[:2]
            left, top, right, bottom = body_boxes[index]
            body_center_x = 0.5 * (left + right)
            body_center_y = 0.5 * (top + bottom)
            gap = max(3.0, 0.035 * sprite_height)
            near = (
                (body_center_x, top - gap - sprite_height / 2),
                (right + gap + sprite_width / 2, body_center_y),
                (left - gap - sprite_width / 2, body_center_y),
                (body_center_x, bottom + gap + sprite_height / 2),
            )
            far = (
                (body_center_x, top - 2.0 * (sprite_height + gap) - sprite_height / 2),
                (right + 2.0 * (sprite_width + gap) + sprite_width / 2, body_center_y),
                (left - 2.0 * (sprite_width + gap) - sprite_width / 2, body_center_y),
                (body_center_x, bottom + 2.0 * (sprite_height + gap) + sprite_height / 2),
            )
            offsets = near + far
            candidates = []
            other_bodies = body_boxes[active].copy()
            other_indices = np.flatnonzero(active)
            other_bodies = other_bodies[other_indices != index]
            for candidate_x, candidate_y in offsets:
                candidate_x = float(np.clip(
                    candidate_x, sprite_width / 2, frame_width - sprite_width / 2
                ))
                candidate_y = float(np.clip(
                    candidate_y, sprite_height / 2, frame_height - sprite_height / 2
                ))
                box = (
                    candidate_x - sprite_width / 2,
                    candidate_y - sprite_height / 2,
                    candidate_x + sprite_width / 2,
                    candidate_y + sprite_height / 2,
                )
                obstacle_overlap = (
                    (other_bodies[:, 0] < box[2])
                    & (other_bodies[:, 2] > box[0])
                    & (other_bodies[:, 1] < box[3])
                    & (other_bodies[:, 3] > box[1])
                )
                obstacle_overlap |= (
                    (ball_box[0] < box[2])
                    & (ball_box[2] > box[0])
                    & (ball_box[1] < box[3])
                    & (ball_box[3] > box[1])
                )
                label_overlap = sum(
                    box[0] < previous[2] and box[2] > previous[0]
                    and box[1] < previous[3] and box[3] > previous[1]
                    for previous in label_boxes
                )
                overlaps = int(np.count_nonzero(obstacle_overlap)) + label_overlap
                candidates.append((overlaps, candidate_x, candidate_y, box))
                if overlaps == 0:
                    break
            _, center_x, center_y, chosen_box = min(candidates, key=lambda item: item[0])
            label_boxes.append(chosen_box)
            _composite_sprite(
                frame, sprite, center_x, center_y
            )
        return frame

    def close(self) -> None:
        from matplotlib import pyplot as plt

        plt.close(self.fig)

    def _update_metric_snapshot(self, snapshot: dict) -> None:
        for metric, team_artists in self.metric_block_artists.items():
            entries = snapshot.get("leaders", {}).get(metric, ())
            for team_index, artist in enumerate(team_artists):
                selected = [row for row in entries if int(row["team"]) == team_index]
                lines = []
                for rank, row in enumerate(selected, start=1):
                    if metric == "distance_km":
                        value = f"{float(row[metric]):.3f}"
                    elif metric == "top_speed_kmh":
                        value = f"{float(row[metric]):.1f}"
                    else:
                        value = f"{float(row[metric]):.1f}"
                    lines.append(f"{rank}. #{int(row['player_id']):>2}  {value:>5}")
                artist.set_text("\n".join(lines) if lines else "No data")

        for team_index, summary in enumerate(snapshot.get("stamina", ())):
            players = summary.get("players", ())
            if team_index >= len(self.stamina_block_artists):
                continue
            lines = [
                f"AVG L{100*float(summary['long_mean']):.0f} "
                f"S{100*float(summary['short_mean']):.0f}"
            ]
            lines.extend(
                f"{rank}. #{int(player['player_id']):>2} "
                f"L{100*float(player['long']):.0f} S{100*float(player['short']):.0f}"
                for rank, player in enumerate(players, start=1)
            )
            self.stamina_block_artists[team_index].set_text("\n".join(lines))


def _start_ffmpeg(
    executable: str,
    destination: Path,
    *,
    fps: int,
    style: RenderStyle,
) -> subprocess.Popen[bytes]:
    command = (
        executable,
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgba",
        "-video_size",
        f"{style.width_px}x{style.height_px}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        style.encoder_preset,
        "-crf",
        str(style.crf),
        "-threads",
        str(style.encoder_threads),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    )
    try:
        return subprocess.Popen(command, stdin=subprocess.PIPE)
    except OSError as error:
        raise RuntimeError(
            f"FFmpeg MP4 encoder를 시작할 수 없다: {executable}"
        ) from error


def _abort_ffmpeg(process: subprocess.Popen[bytes]) -> None:
    """실패한 encoder의 pipe와 child를 best-effort로 정리하고 reap한다."""

    stream = process.stdin
    if stream is not None and not stream.closed:
        with suppress(BaseException):
            stream.close()
    running = True
    with suppress(BaseException):
        running = process.poll() is None
    if running:
        with suppress(BaseException):
            process.terminate()
    waited = False
    with suppress(BaseException):
        process.wait(timeout=5.0)
        waited = True
    if not waited:
        with suppress(BaseException):
            process.kill()
        with suppress(BaseException):
            process.wait()


def _output_frame_sample(
    output_index: int, *, control_fps: float, fps: int, frame_count: int
) -> tuple[int, int, float]:
    """Video 시각을 감싸는 두 control frame과 선형 보간 비율을 반환한다."""

    source_time = output_index * control_fps / fps
    lower = min(math.floor(source_time), frame_count - 1)
    upper = min(lower + 1, frame_count - 1)
    alpha = 0.0 if upper == lower else float(source_time - lower)
    return lower, upper, alpha


def _interpolated_visual_state(host, lower: int, upper: int, alpha: float):
    """연속 visual leaf만 보간하고 사건 경계의 순간이동은 보간하지 않는다."""

    lower_position = np.asarray(host.states.players.position[lower])
    lower_ball = np.asarray(host.states.ball.position[lower])
    continuous = (
        upper != lower
        and np.array_equal(
            host.states.players.active[lower], host.states.players.active[upper]
        )
        and np.array_equal(host.states.score[lower], host.states.score[upper])
        and int(host.states.first_half_wall_end_tick[lower])
        == int(host.states.first_half_wall_end_tick[upper])
    )
    blend = alpha if continuous else 0.0
    if blend == 0.0:
        return lower_position, lower_ball
    upper_position = np.asarray(host.states.players.position[upper])
    upper_ball = np.asarray(host.states.ball.position[upper])
    return (
        lower_position + blend * (upper_position - lower_position),
        lower_ball + blend * (upper_ball - lower_ball),
    )


def _display_seconds(control_tick, first_half_end, *, halftime_ticks, control_fps):
    added = max(int(first_half_end) - halftime_ticks, 0) if first_half_end >= 0 else 0
    return float(int(control_tick) - added) / control_fps


def _build_win_probability_history(
    control_tick,
    first_half_wall_end_tick,
    score,
    *,
    control_fps: float,
    halftime_ticks: int,
    rate_model,
):
    """병렬 구간 renderer에 전달할 sparse causal win-probability history."""

    ticks = np.asarray(control_tick, dtype=np.int64)
    half_end = np.asarray(first_half_wall_end_tick, dtype=np.int64)
    scores = np.asarray(score, dtype=np.int64)
    if ticks.ndim != 1 or half_end.shape != ticks.shape or scores.shape != (ticks.size, 2):
        raise ValueError("win probability history의 frame shape가 서로 일치해야 한다")
    if not ticks.size:
        return ()
    sample_stride = max(1, round(30.0 * control_fps))
    selected = set(range(0, ticks.size, sample_stride))
    selected.add(0)
    selected.update(
        (np.flatnonzero(np.any(scores[1:] != scores[:-1], axis=1)) + 1).tolist()
    )
    second_half = (half_end >= 0) & (ticks >= half_end)
    selected.update((np.flatnonzero(second_half[1:] != second_half[:-1]) + 1).tolist())
    history = []
    for index in sorted(selected):
        seconds = _display_seconds(
            ticks[index],
            half_end[index],
            halftime_ticks=halftime_ticks,
            control_fps=control_fps,
        )
        curve = rate_model["second_half" if second_half[index] else "first_half"]
        rates = _remaining_goal_rates(curve, max(float(seconds), 0.0) / 60.0)
        probabilities = _poisson_wdl_probabilities(scores[index], rates)
        history.append(
            (
                int(index),
                float(ticks[index]) / control_fps / 60.0,
                float(probabilities[0]),
                float(probabilities[1]),
                float(probabilities[2]),
            )
        )
    return tuple(history)


def _goal_visible_mask(goal_frames, output_count, *, control_fps, fps, hold_frames):
    """Output frame별 GOAL overlay 가시성을 미리 계산한다.

    원래 loop는 ``previous_source_index``와 ``goal_until``을 carry해서 세그먼트를
    독립적으로 시작할 수 없었다. 미리 mask로 만들면 output index만으로 결정된다.
    """

    visible = np.zeros(output_count, dtype=bool)
    # Source frame을 처음 소비하는 output 시각부터 overlay 구간을 채운다.
    for source in np.flatnonzero(goal_frames):
        begin = math.ceil(int(source) * fps / control_fps)
        while begin < output_count and int(begin * control_fps / fps) < source:
            begin += 1
        while begin > 0 and int((begin - 1) * control_fps / fps) >= source:
            begin -= 1
        visible[begin : min(output_count, begin + hold_frames)] = True
    return visible


def _render_segment(
    job: dict, *, renderer_factory=_ReplayRenderer, encoder_factory=_start_ffmpeg
) -> str:
    """한 output frame 구간만 독립 MP4로 encode한다.

    ``spawn`` 으로 시작하므로 필요한 배열만 담은 자기완결 payload를 받는다. 구간이
    쓰는 source frame만 잘라 보내 worker마다 전체 trajectory를 옮기지 않는다.
    """

    style = job["style"]
    renderer_kwargs = {"player_id": job["player_id"]}
    if job.get("analysis") is not None:
        renderer_kwargs["analysis"] = job["analysis"]
    renderer_kwargs["mode"] = job.get("mode", "analysis")
    renderer = renderer_factory(
        style,
        job["team"].shape[0],
        job["contact_physics"],
        job["ball_geometry"],
        job["field_half_extent"],
        **renderer_kwargs,
    )
    process = None
    try:
        process = encoder_factory(
            job["ffmpeg"], Path(job["path"]), fps=job["fps"], style=style
        )
        stream = process.stdin
        if stream is None:
            raise RuntimeError("FFmpeg MP4 encoder의 input pipe가 없다")
        control_fps = job["control_fps"]
        fps = job["fps"]
        offset = job["source_offset"]
        local_host = SimpleNamespace(
            states=SimpleNamespace(
                players=SimpleNamespace(position=job["position"], active=job["active"]),
                ball=SimpleNamespace(position=job["ball"]),
                score=job["score"],
                first_half_wall_end_tick=job["first_half_end"],
            )
        )
        for local, output_index in enumerate(range(job["start"], job["stop"])):
            global_lower, global_upper, alpha = _output_frame_sample(
                output_index,
                control_fps=control_fps,
                fps=fps,
                frame_count=job["frame_count"],
            )
            lower = global_lower - offset
            upper = global_upper - offset
            position, ball = _interpolated_visual_state(local_host, lower, upper, alpha)
            player_trail_start = max(
                offset,
                global_lower - math.ceil(style.player_trail_seconds * control_fps),
            ) - offset
            ball_trail_start = max(
                offset,
                global_lower - math.ceil(style.ball_trail_seconds * control_fps),
            ) - offset
            seconds = _display_seconds(
                job["control_tick"][lower],
                job["first_half_end"][lower],
                halftime_ticks=job["halftime_ticks"],
                control_fps=control_fps,
            )
            first_half_end = int(job["first_half_end"][lower])
            second_half = (
                first_half_end >= 0
                and int(job["control_tick"][lower]) >= first_half_end
            )
            player_trail = np.asarray(
                job["position"][player_trail_start : lower + 1]
            ).copy()
            if player_trail.shape[0]:
                player_trail[-1] = position
            ball_trail = np.asarray(job["ball"][ball_trail_start : lower + 1]).copy()
            if ball_trail.shape[0]:
                ball_trail[-1] = ball
            frame = renderer.draw(
                position=position,
                active=job["active"][lower],
                ball=ball,
                score=job["score"][lower],
                intent=job["intent"][lower],
                team=job["team"],
                goalkeeper=job["goalkeeper"],
                seconds=seconds,
                second_half=second_half,
                elapsed_minutes=(
                    float(job["control_tick"][lower]) / control_fps / 60.0
                ),
                goal_visible=bool(job["goal_visible"][local]),
                player_trail=player_trail,
                player_active_trail=job["active"][player_trail_start : lower + 1],
                ball_trail=ball_trail,
            )
            try:
                stream.write(frame)
            except OSError as error:
                raise RuntimeError("FFmpeg MP4 frame 전달에 실패했다") from error
        stream.close()
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"FFmpeg MP4 encoding에 실패했다 (exit code {code})")
        output = Path(job["path"])
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("FFmpeg MP4 encoder가 유효한 output을 만들지 않았다")
    except BaseException:
        if process is not None:
            _abort_ffmpeg(process)
        with suppress(BaseException):
            renderer.close()
        raise
    else:
        renderer.close()
    return job["path"]


def _worker_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("payload")
    parser.add_argument("metadata")
    args = parser.parse_args()
    job = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    job["style"] = RenderStyle(**job["style"])
    job["contact_physics"] = SimpleNamespace(**job["contact_physics"])
    job["ball_geometry"] = SimpleNamespace(**job["ball_geometry"])
    with np.load(args.payload, allow_pickle=False) as payload:
        job.update({key: payload[key] for key in payload.files})
    _render_segment(job)


if __name__ == "__main__":
    _worker_main()
