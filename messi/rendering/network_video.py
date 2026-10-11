"""Optional native-video preview for the passing-network report panel."""

from __future__ import annotations

from math import ceil, sqrt
from pathlib import Path
from subprocess import DEVNULL, PIPE, Popen

from messi.rendering.network_identity import (
    network_passes_with_generations,
    network_shots_with_generations,
)


def render_passing_network_video(report, path: str | Path, *, team: int) -> Path:
    """Encode a seekable preview where one video second is one match minute."""
    if team not in (0, 1):
        raise ValueError("team must be 0 or 1")
    from PIL import Image, ImageDraw, ImageFont
    import imageio_ffmpeg

    data = report.as_dict()
    data["pass_map"] = network_passes_with_generations(data)
    data["shot_map"] = network_shots_with_generations(data)
    dt = float(data["control_dt_s"])
    if dt <= 0:
        raise ValueError("control_dt_s must be positive")
    frames = int(data["frames"])
    if frames <= 0:
        raise ValueError("frames must be positive")
    destination = Path(path)
    if destination.suffix.lower() != ".mp4":
        raise ValueError("network video output must end in .mp4")
    destination.parent.mkdir(parents=True, exist_ok=True)

    fps = 4
    width, height = 640, 420
    x_edges = data["spatial"]["x_edges_m"]
    y_edges = data["spatial"]["y_edges_m"]
    xmin, xmax = float(x_edges[0]), float(x_edges[-1])
    ymin, ymax = float(y_edges[0]), float(y_edges[-1])
    pitch_left, pitch_top = 38, 46
    pitch_width, pitch_height = 564, 354

    def xy(point: tuple[float, float] | list[float]) -> tuple[float, float]:
        return (
            pitch_left + (float(point[0]) - xmin) * pitch_width / (xmax - xmin),
            pitch_top + (float(point[1]) - ymin) * pitch_height / (ymax - ymin),
        )

    passes = sorted(
        (
            row for row in data["pass_map"]
            if row.get("team") == team
            and row.get("completed")
            and row.get("terminal_actor", -1) >= 0
            and row.get("actor") != row.get("terminal_actor")
            and "_network_source_generation" in row
            and "_network_target_generation" in row
            and row.get("attack_normalized_source_position")
            and row.get("attack_normalized_terminal_position")
        ),
        key=lambda row: row["terminal_frame"],
    )
    shots = sorted(
        (
            row for row in data["shot_map"]
            if row.get("team") == team
            and "_network_shooter_generation" in row
        ),
        key=lambda row: row["terminal_frame"],
    )
    players = {
        (int(row["slot"]), int(row["generation"])): row
        for row in data["player_generations"]
    }
    nodes: dict[tuple[int, int], list[float]] = {}
    edges: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    next_pass = 0
    next_shot = 0
    shot_links = []
    team_color = "#38bdf8" if team == 0 else "#fb7185"
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(font_path, 17)
        small_font = ImageFont.truetype(font_path, 13)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "slow", "-crf", "27", "-g", str(fps * 4),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ]
    process = Popen(command, stdin=PIPE, stdout=DEVNULL, stderr=PIPE)
    total_minutes = frames * dt / 60
    output_frames = ceil(total_minutes * fps) + 1
    try:
        for video_frame in range(output_frames):
            minute = min(video_frame / fps, total_minutes)
            cut = round(minute * 60 / dt)
            while next_pass < len(passes) and passes[next_pass]["terminal_frame"] <= cut:
                row = passes[next_pass]
                source = int(row["actor"]), int(row["_network_source_generation"])
                target = int(row["terminal_actor"]), int(row["_network_target_generation"])
                for player, point in (
                    (source, row["attack_normalized_source_position"]),
                    (target, row["attack_normalized_terminal_position"]),
                ):
                    node = nodes.setdefault(player, [0.0, 0.0, 0.0])
                    node[0] += point[0]
                    node[1] += point[1]
                    node[2] += 1
                key = source, target
                edges[key] = edges.get(key, 0) + 1
                next_pass += 1
            while next_shot < len(shots) and shots[next_shot]["terminal_frame"] <= cut:
                row = shots[next_shot]
                shooter = int(row["actor"]), int(row["_network_shooter_generation"])
                start = row["attack_normalized_source_position"]
                node = nodes.setdefault(shooter, [0.0, 0.0, 0.0])
                node[0] += start[0]
                node[1] += start[1]
                node[2] += 1
                shot_links.append((shooter, row["attack_normalized_terminal_position"]))
                next_shot += 1

            image = Image.new("RGB", (width, height), "#0b1628")
            draw = ImageDraw.Draw(image)
            draw.text((38, 13), f"Team {team}  ·  {minute:.1f} / {total_minutes:.1f} min  ·  {sum(edges.values())} passes  ·  {len(shot_links)} shots", fill="#e5edf5", font=font)
            draw.rectangle((pitch_left, pitch_top, pitch_left + pitch_width, pitch_top + pitch_height), fill="#173f31", outline="#f4f7fa", width=2)
            center_x, center_y = xy((0, 0))
            draw.line((center_x, pitch_top, center_x, pitch_top + pitch_height), fill="#f4f7fa", width=2)
            radius = 9.15 * pitch_height / (ymax - ymin)
            draw.ellipse((center_x-radius, center_y-radius, center_x+radius, center_y+radius), outline="#f4f7fa", width=2)
            for side in (-1, 1):
                box_x = xmax - 16.5 if side == 1 else xmin
                a = xy((box_x, -20.16))
                b = xy((box_x + 16.5, 20.16))
                draw.rectangle((*a, *b), outline="#f4f7fa", width=2)
            for (source, target), count in edges.items():
                a, b = nodes[source], nodes[target]
                start = xy((a[0] / a[2], a[1] / a[2]))
                end = xy((b[0] / b[2], b[1] / b[2]))
                color = "#facc15" if players[source]["is_goalkeeper"] or players[target]["is_goalkeeper"] else team_color
                draw.line((*start, *end), fill=color, width=min(7, 2 + round(sqrt(count))))
            for shooter, terminal in shot_links:
                node = nodes[shooter]
                start = xy((node[0] / node[2], node[1] / node[2]))
                end = xy(terminal)
                draw.line((*start, *end), fill="#fb923c", width=3)
                draw.ellipse((end[0]-3, end[1]-3, end[0]+3, end[1]+3), fill="#fb923c")
            for player, node in nodes.items():
                x, y = xy((node[0] / node[2], node[1] / node[2]))
                goalkeeper = bool(players[player]["is_goalkeeper"])
                radius = 9 if goalkeeper else 7
                color = "#facc15" if goalkeeper else team_color
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline="white", width=1)
                label = f"GK #{players[player]['player_id']}" if goalkeeper else f"#{players[player]['player_id']}"
                draw.text((x+radius+2, y-7), label, fill="white", font=small_font)
            assert process.stdin is not None
            process.stdin.write(image.tobytes())
        process.stdin.close()
        assert process.stderr is not None
        error = process.stderr.read().decode("utf-8", "replace")
        if process.wait() != 0:
            raise RuntimeError(f"network video encoding failed: {error.strip()}")
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        destination.unlink(missing_ok=True)
        raise
    return destination
