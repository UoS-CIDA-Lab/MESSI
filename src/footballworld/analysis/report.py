"""Write portable JSON and HTML match reports."""

from __future__ import annotations

import html
import json
import math
import tempfile
from itertools import pairwise
from pathlib import Path
from typing import Any

from filelock import FileLock

from footballworld.analysis.dataset import MatchDataset
from footballworld.analysis.metrics import (
    SHOT_ROUTE_MAX_PRIOR_NODES,
    build_match_report,
)
from footballworld.analysis.reference import (
    attach_policy_reference,
    load_policy_reference,
)


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _team_rows(report: dict[str, Any]) -> str:
    labels = (
        (
            "Full-match controlled possession",
            "controlled_possession_share",
            lambda x: "Unavailable" if x is None else f"{100 * x:.1f}%",
        ),
        (
            "Share of all live time",
            "live_time_possession_share",
            lambda x: "Unavailable" if x is None else f"{100 * x:.1f}%",
        ),
        ("Live possession", "possession_s", lambda x: f"{x / 60:.1f} min"),
        ("Goals", "goals", str),
        ("Realized shots", "realized_shots", str),
        ("Shots on target (including goals)", "shots_on_target", str),
        ("Goals from realized shots", "shot_goals", str),
        ("Saved on-target shots", "saved_on_target_shots", str),
        ("Off-target shots", "off_target_shots", str),
        ("Unresolved shots", "unresolved_shots", str),
        ("Realized open-play passes", "open_play_pass_attempts", str),
        ("Completed open-play passes", "open_play_completed_passes", str),
        (
            "Open-play pass receipt rate",
            "open_play_pass_completion",
            lambda x: "Unavailable" if x is None else f"{100 * x:.1f}%",
        ),
        (
            "Rule-policy cross signatures",
            "rule_policy_cross_control_signatures",
            lambda x: "Unavailable" if x is None else str(x),
        ),
        (
            "Completed rule-policy cross signatures",
            "completed_rule_policy_cross_control_signatures",
            lambda x: "Unavailable" if x is None else str(x),
        ),
        (
            "Defensive-line-breaking passes (through-pass proxy)",
            "defensive_line_breaking_pass_proxies",
            str,
        ),
        (
            "Completed defensive-line-breaking passes",
            "completed_defensive_line_breaking_pass_proxies",
            str,
        ),
        ("Penalty-area entries", "penalty_area_entries", str),
        ("Corners", "corners", str),
        ("Fouls committed", "fouls_committed", str),
        ("Offsides", "offsides", str),
    )
    teams = report["teams"]
    return "".join(
        f"<tr><th>{html.escape(label)}</th><td>{formatter(teams[0][key])}</td><td>{formatter(teams[1][key])}</td></tr>"
        for label, key, formatter in labels
    )


def _player_rows(report: dict[str, Any]) -> str:
    rows = []
    for player in report["players"]:
        rows.append(
            "<tr>"
            f"<td><span class='team-dot t{player['team']}'></span>Team {player['team']}</td>"
            f"<td>{player['player_id']}</td>"
            f"<td>{player['slot_generation']}</td>"
            f"<td>{player['active_s'] / 60:.1f}</td>"
            f"<td>{player['distance_m'] / 1000:.2f}</td>"
            f"<td>{player['max_speed_mps']:.2f}</td>"
            "</tr>"
        )
    return "".join(rows)


def _timeline_rows(report: dict[str, Any]) -> str:
    if not report["timeline"]:
        return "<p class='empty'>No goals, fouls, or management changes in this capture.</p>"
    rows = []
    for item in report["timeline"]:
        team = item.get("team")
        badge = "" if team not in (0, 1) else f"<span class='team-dot t{team}'></span>"
        rows.append(
            "<div class='timeline-row static'>"
            f"<time>{html.escape(item['clock_label'])}</time>"
            f"<span>{badge}{html.escape(item['label'])}</span>"
            "</div>"
        )
    return "".join(rows)


def _event_rows(report: dict[str, Any]) -> str:
    counts = report["events"]["exact_event_counts"]
    if not counts:
        return "<p class='empty'>No exact events were recorded.</p>"
    return "".join(
        f"<div class='event-stat'><strong>{count:,}</strong><span>{html.escape(kind.replace('_', ' '))}</span></div>"
        for kind, count in counts.items()
    )


def _series_chart(
    report: dict[str, Any],
    *,
    title: str,
    metric: str,
    unit: str,
    percent: bool = False,
) -> str:
    windows = report["visualizations"]["windows"]
    team_values: list[list[float | None]] = [[], []]
    for window in windows:
        values = window[metric]
        for team in (0, 1):
            value = values[team]
            team_values[team].append(None if value is None else float(value))
    finite_values = [
        value for values in team_values for value in values if value is not None
    ]
    maximum = 1.0 if percent else max(finite_values, default=1.0)
    if not percent:
        maximum = max(1.0, maximum * 1.12)
    width, height = 720.0, 250.0
    left, right, top, bottom = 48.0, 18.0, 18.0, 38.0
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_position(index: int) -> float:
        if len(windows) <= 1:
            return left + plot_width / 2.0
        return left + plot_width * index / (len(windows) - 1)

    def y_position(value: float) -> float:
        return top + plot_height * (1.0 - value / maximum)

    grid = []
    for tick in range(5):
        ratio = tick / 4.0
        y = top + plot_height * (1.0 - ratio)
        label_value = maximum * ratio
        label = f"{100 * label_value:.0f}%" if percent else f"{label_value:.0f}"
        grid.append(
            f"<line x1='{left}' y1='{y:.2f}' x2='{width - right}' y2='{y:.2f}' class='grid-line'/>"
            f"<text x='{left - 8}' y='{y + 4:.2f}' text-anchor='end' class='axis-label'>{label}</text>"
        )
    lines = []
    colors = ("#d1495b", "#167d9a")
    for team in (0, 1):
        points = [
            f"{x_position(index):.2f},{y_position(value):.2f}"
            for index, value in enumerate(team_values[team])
            if value is not None
        ]
        if points:
            lines.append(
                f"<polyline points='{' '.join(points)}' fill='none' stroke='{colors[team]}' "
                "stroke-width='3' stroke-linejoin='round' stroke-linecap='round'/>"
            )
        for index, value in enumerate(team_values[team]):
            if value is None:
                continue
            display = f"{100 * value:.1f}%" if percent else f"{value:.1f} {unit}"
            lines.append(
                f"<circle cx='{x_position(index):.2f}' cy='{y_position(value):.2f}' r='3.5' "
                f"fill='{colors[team]}'><title>Team {team}, {display}</title></circle>"
            )
    duration = report["summary"]["captured_duration_s"]
    return (
        f"<figure class='viz'><figcaption><strong>{html.escape(title)}</strong>"
        f"<span>{len(windows)} cumulative checkpoints  |  {report['visualizations']['window_s']:.0f}s sampling</span></figcaption>"
        f"<svg viewBox='0 0 {width:.0f} {height:.0f}' role='img' aria-label='{html.escape(title)}'>"
        + "".join(grid)
        + "".join(lines)
        + f"<text x='{left}' y='{height - 9}' class='axis-label'>0:00</text>"
        + f"<text x='{width - right}' y='{height - 9}' text-anchor='end' class='axis-label'>{duration / 60:.1f} min</text>"
        + "<g class='legend'><circle cx='520' cy='17' r='5' fill='#d1495b'/><text x='531' y='21'>Team 0</text>"
        + "<circle cx='610' cy='17' r='5' fill='#167d9a'/><text x='621' y='21'>Team 1</text></g>"
        + "</svg></figure>"
    )


def _speed_histogram(report: dict[str, Any]) -> str:
    histogram = report["visualizations"]["speed_histogram"]
    bins = histogram["bin_starts_mps"]
    shares = histogram["sample_shares"]
    maximum = max((value for team in shares for value in team), default=0.0)
    maximum = max(0.01, maximum * 1.12)
    width, height = 720.0, 290.0
    left, right, top, bottom = 48.0, 18.0, 18.0, 48.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    group_width = plot_width / max(1, len(bins))
    bar_width = group_width * 0.34
    pieces = []
    for tick in range(5):
        ratio = tick / 4.0
        y = top + plot_height * (1.0 - ratio)
        pieces.append(
            f"<line x1='{left}' y1='{y:.2f}' x2='{width - right}' y2='{y:.2f}' class='grid-line'/>"
            f"<text x='{left - 8}' y='{y + 4:.2f}' text-anchor='end' class='axis-label'>{100 * maximum * ratio:.0f}%</text>"
        )
    colors = ("#d1495b", "#167d9a")
    for index, start in enumerate(bins):
        for team in (0, 1):
            value = float(shares[team][index])
            bar_height = plot_height * value / maximum
            x = left + index * group_width + group_width * 0.12 + team * bar_width
            y = top + plot_height - bar_height
            upper = "+" if start == histogram["overflow_from_mps"] else f"–{start + 1}"
            pieces.append(
                f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_width:.2f}' height='{bar_height:.2f}' "
                f"fill='{colors[team]}'><title>Team {team}, {start}{upper} m/s: {100 * value:.1f}%</title></rect>"
            )
        if index % 2 == 0 or index == len(bins) - 1:
            label = (
                f"{start}+" if start == histogram["overflow_from_mps"] else str(start)
            )
            pieces.append(
                f"<text x='{left + (index + 0.5) * group_width:.2f}' y='{height - 24}' text-anchor='middle' class='axis-label'>{label}</text>"
            )
    return (
        "<figure class='viz'><figcaption><strong>Player speed distribution</strong>"
        "<span>Share of active on-pitch tracking samples</span></figcaption>"
        f"<svg viewBox='0 0 {width:.0f} {height:.0f}' role='img' aria-label='Player speed histogram'>"
        + "".join(pieces)
        + "<text x='360' y='282' text-anchor='middle' class='axis-label'>Speed bin start (m/s)</text>"
        + "<g class='legend'><circle cx='520' cy='17' r='5' fill='#d1495b'/><text x='531' y='21'>Team 0</text>"
        + "<circle cx='610' cy='17' r='5' fill='#167d9a'/><text x='621' y='21'>Team 1</text></g>"
        + "</svg></figure>"
    )


def _territory_chart(report: dict[str, Any]) -> str:
    shares = report["visualizations"]["ball_territory"]["live_frame_shares"]
    maximum = max(shares, default=0.0)
    pieces = []
    pitch_x, pitch_y, pitch_width, pitch_height = 30.0, 24.0, 660.0, 242.0
    zone_width = pitch_width / max(1, len(shares))
    for index, share in enumerate(shares):
        intensity = 0.12 if not maximum else 0.16 + 0.68 * share / maximum
        x = pitch_x + index * zone_width
        pieces.append(
            f"<rect x='{x:.2f}' y='{pitch_y}' width='{zone_width:.2f}' height='{pitch_height}' "
            f"fill='#f2b134' fill-opacity='{intensity:.3f}'><title>Zone {index + 1}: {100 * share:.1f}% of live frames</title></rect>"
            f"<text x='{x + zone_width / 2:.2f}' y='150' text-anchor='middle' class='territory-label'>{100 * share:.1f}%</text>"
        )
    return (
        "<figure class='viz territory'><figcaption><strong>Ball territory</strong>"
        "<span>Live-ball frame share across six equal pitch zones</span></figcaption>"
        "<svg viewBox='0 0 720 300' role='img' aria-label='Longitudinal ball territory'>"
        "<rect x='30' y='24' width='660' height='242' rx='3' fill='#236746'/>"
        + "".join(pieces)
        + "<g class='pitch-lines'><rect x='30' y='24' width='660' height='242'/><line x1='360' y1='24' x2='360' y2='266'/>"
        + "<circle cx='360' cy='145' r='38'/><rect x='30' y='73' width='105' height='144'/><rect x='585' y='73' width='105' height='144'/></g>"
        + "<text x='30' y='289' class='axis-label'>negative x</text><text x='690' y='289' text-anchor='end' class='axis-label'>positive x</text>"
        + "</svg></figure>"
    )


def _ball_density_chart(report: dict[str, Any]) -> str:
    """Render cumulative live-ball density controlled by a match-time slider."""

    density = report.get("visualizations", {}).get("ball_density_2d", {})
    windows = density.get("windows", [])
    x_edges = density.get("x_edges_m", [])
    y_edges = density.get("y_edges_m", [])
    rows = density.get("rows", [])
    if (
        not isinstance(windows, list)
        or not windows
        or not isinstance(rows, list)
        or not isinstance(x_edges, list)
        or len(x_edges) < 2
        or not isinstance(y_edges, list)
        or len(y_edges) < 2
    ):
        return (
            "<figure class='viz spatial-wide'><figcaption><strong>Ball-position density over time</strong>"
            "<span>No continuous in-pitch live-ball intervals are available.</span></figcaption>"
            "<p class='empty'>Time-resolved ball density is unavailable for this capture.</p></figure>"
        )
    payload = json.dumps(
        {
            "x_edges_m": x_edges,
            "y_edges_m": y_edges,
            "windows": windows,
            "rows": rows,
        },
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    x_bins = len(x_edges) - 1
    y_bins = len(y_edges) - 1
    pitch_x, pitch_y, pitch_width = 48.0, 72.0, 600.0
    pitch_length = float(x_edges[-1]) - float(x_edges[0])
    pitch_span_y = float(y_edges[-1]) - float(y_edges[0])
    pitch_height = pitch_width * pitch_span_y / max(pitch_length, 1.0)
    box_width = pitch_width * 16.5 / 105.0
    box_height = pitch_height * 40.32 / 68.0
    box_y = pitch_y + (pitch_height - box_height) / 2.0
    circle_radius = pitch_width * 9.15 / 105.0
    footer_y = pitch_y + pitch_height + 24.0
    total = float(density.get("observed_live_seconds", 0.0))
    excluded = float(density.get("excluded_out_of_pitch_seconds", 0.0))
    return f"""<figure class="viz spatial-wide ball-density-player">
<figcaption><strong>Ball-position density over time</strong><span>{x_bins} x {y_bins} time-weighted cells | {total:.1f}s in pitch | {excluded:.1f}s excluded</span></figcaption>
<div class="network-controls"><button type="button" class="network-play density-play">Play</button><input class="network-time density-time" type="range" min="0" max="{len(windows) - 1}" value="0" step="1" aria-label="Ball density cumulative match time"></div>
<div class="network-hover density-status" aria-live="polite">Loading cumulative live-ball density…</div>
<svg viewBox="0 0 720 {footer_y + 14:.0f}" role="img" aria-label="Interactive cumulative two-dimensional live-ball position density">
<rect x="{pitch_x}" y="{pitch_y}" width="{pitch_width}" height="{pitch_height:.2f}" fill="#173f31"/>
<g class="density-cells"></g>
<g class="pitch-lines"><rect x="{pitch_x}" y="{pitch_y}" width="{pitch_width}" height="{pitch_height:.2f}"/><line x1="{pitch_x + pitch_width / 2:.2f}" y1="{pitch_y}" x2="{pitch_x + pitch_width / 2:.2f}" y2="{pitch_y + pitch_height:.2f}"/><circle cx="{pitch_x + pitch_width / 2:.2f}" cy="{pitch_y + pitch_height / 2:.2f}" r="{circle_radius:.2f}"/><rect x="{pitch_x}" y="{box_y:.2f}" width="{box_width:.2f}" height="{box_height:.2f}"/><rect x="{pitch_x + pitch_width - box_width:.2f}" y="{box_y:.2f}" width="{box_width:.2f}" height="{box_height:.2f}"/></g>
<text x="{pitch_x}" y="{footer_y:.2f}" class="axis-label">{float(x_edges[0]):.1f} m</text><text x="{pitch_x + pitch_width / 2:.2f}" y="{footer_y:.2f}" text-anchor="middle" class="axis-label">absolute pitch x</text><text x="{pitch_x + pitch_width:.2f}" y="{footer_y:.2f}" text-anchor="end" class="axis-label">{float(x_edges[-1]):.1f} m</text>
<g class="legend"><rect x="500" y="15" width="13" height="8" fill="#f6b73c"/><text x="520" y="23">cell time (sqrt color)</text><rect x="500" y="31" width="13" height="8" fill="#36a6d9"/><text x="520" y="39">marginal time</text></g>
</svg><noscript><p class="empty">Enable JavaScript to move the cumulative time bar.</p></noscript></figure>
<script type="application/json" id="ball-density-data">{payload}</script>
<script>
(()=>{{const d=JSON.parse(document.getElementById("ball-density-data").textContent),F=document.querySelector(".ball-density-player"),S=F.querySelector(".density-time"),G=F.querySelector(".density-cells"),T=F.querySelector(".density-status"),B=F.querySelector(".density-play"),nx=d.x_edges_m.length-1,ny=d.y_edges_m.length-1,byWindow=new Map,prefix=[],running=new Float64Array(nx*ny);let timer=null;
d.rows.forEach(r=>{{const a=byWindow.get(+r[0])||[];a.push(r);byWindow.set(+r[0],a)}});
d.windows.forEach(w=>{{(byWindow.get(+w.index)||[]).forEach(r=>{{const xi=+r[1],yi=+r[2],v=+r[3];if(xi>=0&&xi<nx&&yi>=0&&yi<ny&&Number.isFinite(v)&&v>0)running[yi*nx+xi]+=v}});prefix.push(running.slice())}});
const clock=s=>{{s=Math.max(0,Math.round(+s));return String(Math.floor(s/60)).padStart(2,"0")+":"+String(s%60).padStart(2,"0")}};
const draw=()=>{{const cut=+S.value,w=d.windows[cut],cells=prefix[cut],xt=new Float64Array(nx),yt=new Float64Array(ny);let total=0,max=0;cells.forEach((v,k)=>{{total+=v;max=Math.max(max,v);xt[k%nx]+=v;yt[Math.floor(k/nx)]+=v}});const mx=Math.max(0,...xt),my=Math.max(0,...yt),cw=600/nx,ch={pitch_height:.8f}/ny;let out="";for(let yi=0;yi<ny;yi++)for(let xi=0;xi<nx;xi++){{const v=cells[yi*nx+xi];if(!(v>0))continue;const x=48+xi*cw,y=72+(ny-yi-1)*ch,a=.12+.78*Math.sqrt(v/Math.max(max,1e-12)),share=total?100*v/total:0;out+='<rect x="'+x.toFixed(2)+'" y="'+y.toFixed(2)+'" width="'+(cw+.12).toFixed(2)+'" height="'+(ch+.12).toFixed(2)+'" fill="#f6b73c" fill-opacity="'+a.toFixed(3)+'"><title>'+v.toFixed(2)+' s · '+share.toFixed(2)+'%</title></rect>'}}for(let xi=0;xi<nx;xi++){{const h=mx?42*xt[xi]/mx:0;out+='<rect x="'+(48+xi*cw).toFixed(2)+'" y="'+(64-h).toFixed(2)+'" width="'+Math.max(.5,cw-.5).toFixed(2)+'" height="'+h.toFixed(2)+'" fill="#36a6d9"><title>x marginal '+xt[xi].toFixed(2)+' s</title></rect>'}}for(let yi=0;yi<ny;yi++){{const bw=my?42*yt[yi]/my:0,y=72+(ny-yi-1)*ch;out+='<rect x="656" y="'+y.toFixed(2)+'" width="'+bw.toFixed(2)+'" height="'+Math.max(.5,ch-.5).toFixed(2)+'" fill="#36a6d9"><title>y marginal '+yt[yi].toFixed(2)+' s</title></rect>'}}G.innerHTML=out;let excluded=0;for(let i=0;i<=cut;i++)excluded+=+(d.windows[i].excluded_out_of_pitch_seconds||0);T.textContent="Kickoff–"+clock(w.end_clock_s)+" | "+total.toFixed(1)+" s in pitch · "+excluded.toFixed(1)+" s excluded"}};
S.oninput=draw;B.onclick=()=>{{if(timer){{clearInterval(timer);timer=null;B.textContent="Play";return}}if(+S.value>=+S.max)S.value=0;B.textContent="Pause";timer=setInterval(()=>{{S.value=Math.min(+S.max,+S.value+1);draw();if(+S.value>=+S.max){{clearInterval(timer);timer=null;B.textContent="Play"}}}},220)}};draw()}})();
</script>"""


_SPATIAL_RED = (255, 54, 95)
_SPATIAL_BLUE = (39, 135, 255)
_SPATIAL_BALANCE_STEPS = 16


def _spatial_balance_bucket(team_0_share: float, team_1_share: float) -> int:
    """Quantize a cell's normalized team-density balance for stable SVG IDs."""

    total = team_0_share + team_1_share
    if total <= 0.0:
        return 0
    return min(
        _SPATIAL_BALANCE_STEPS,
        max(
            0,
            math.floor(_SPATIAL_BALANCE_STEPS * team_1_share / total + 0.5),
        ),
    )


def _spatial_balance_color(bucket: int) -> str:
    """Return a bright red-purple-blue interpolation for one balance bucket."""

    blue_weight = bucket / _SPATIAL_BALANCE_STEPS
    rgb = tuple(
        round(red + blue_weight * (blue - red))
        for red, blue in zip(_SPATIAL_RED, _SPATIAL_BLUE, strict=True)
    )
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def _spatial_balance_gradients() -> str:
    """Return bounded gradients shared by all single-pass occupancy cells."""

    gradients = []
    for bucket in range(_SPATIAL_BALANCE_STEPS + 1):
        color = _spatial_balance_color(bucket)
        gradients.append(
            f'<radialGradient id="spatial-balance-{bucket}">'
            f'<stop offset="0%" stop-color="{color}" stop-opacity=".98"/>'
            f'<stop offset="42%" stop-color="{color}" stop-opacity=".78"/>'
            f'<stop offset="72%" stop-color="{color}" stop-opacity=".28"/>'
            f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/>'
            "</radialGradient>"
        )
    return "".join(gradients)


def _space_occupancy_snapshot(occupancy: dict[str, Any], cut: int) -> tuple[str, str]:
    """Return a visible SVG snapshot and status without requiring JavaScript."""

    windows = occupancy["windows"]
    x_edges = occupancy["x_edges_m"]
    y_edges = occupancy["y_edges_m"]
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    selected_windows = {
        int(window["index"])
        for window in windows[: cut + 1]
        if isinstance(window, dict) and "index" in window
    }
    cells = [0.0] * (nx * ny * 2)
    for row in occupancy["rows"]:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        try:
            window_index = int(row[0])
            team = int(row[1])
            xi = int(row[2])
            yi = int(row[3])
            value = float(row[4])
        except (TypeError, ValueError, OverflowError):
            continue
        if (
            window_index not in selected_windows
            or team not in (0, 1)
            or not 0 <= xi < nx
            or not 0 <= yi < ny
            or not math.isfinite(value)
            or value <= 0.0
        ):
            continue
        cells[2 * (yi * nx + xi) + team] += value

    totals = [sum(cells[2 * cell + team] for cell in range(nx * ny)) for team in (0, 1)]
    shares = [
        [
            cells[2 * cell + team] / totals[team] if totals[team] else 0.0
            for cell in range(nx * ny)
        ]
        for team in (0, 1)
    ]
    combined = [shares[0][cell] + shares[1][cell] for cell in range(nx * ny)]
    combined_peak = max(combined, default=0.0)
    attack = [0.0, 0.0]
    fields: list[str] = []
    radius = 1.72 * max(660.0 / nx, 427.4 / ny)
    for yi in range(ny):
        for xi in range(nx):
            cell = yi * nx + xi
            p0 = shares[0][cell]
            p1 = shares[1][cell]
            center = (float(x_edges[xi]) + float(x_edges[xi + 1])) / 2.0
            if center > 0.0:
                attack[0] += cells[2 * cell]
            if center < 0.0:
                attack[1] += cells[2 * cell + 1]
            if not (p0 > 0.0 or p1 > 0.0):
                continue
            cx = 30.0 + (xi + 0.5) / nx * 660.0
            cy = 38.0 + (ny - yi - 0.5) / ny * 427.4
            label = html.escape(
                f"Team occupancy | T0 {100.0 * p0:.2f}% | T1 {100.0 * p1:.2f}%"
            )
            balance_bucket = _spatial_balance_bucket(p0, p1)
            opacity = 0.12 + 0.88 * math.sqrt(
                combined[cell] / max(combined_peak, 1e-12)
            )
            fields.append(
                f'<circle class="spatial-density-cell" cx="{cx:.2f}" '
                f'cy="{cy:.2f}" r="{radius:.2f}" '
                f'fill="url(#spatial-balance-{balance_bucket})" '
                f'opacity="{opacity:.3f}">'
                f"<title>{label}</title></circle>"
            )

    end_seconds = max(0, round(float(windows[cut].get("end_clock_s", 0.0))))
    clock = f"{end_seconds // 60:02d}:{end_seconds % 60:02d}"
    status = (
        f"Kickoff–{clock} | cumulative attacking-half occupancy: "
        f"T0 {100.0 * attack[0] / totals[0] if totals[0] else 0.0:.1f}% · "
        f"T1 {100.0 * attack[1] / totals[1] if totals[1] else 0.0:.1f}%"
    )
    markup = '<g class="spatial-density-fields">' + "".join(fields) + "</g>"
    return markup, status


def _space_occupancy_chart(report: dict[str, Any]) -> str:
    """Render a smooth cumulative team occupancy comparison over time."""

    occupancy = report.get("visualizations", {}).get("team_space_occupancy", {})
    windows = occupancy.get("windows", [])
    x_edges = occupancy.get("x_edges_m", [])
    y_edges = occupancy.get("y_edges_m", [])
    rows = occupancy.get("rows", [])
    if (
        not isinstance(windows, list)
        or not windows
        or not isinstance(rows, list)
        or not isinstance(x_edges, list)
        or len(x_edges) < 2
        or not isinstance(y_edges, list)
        or len(y_edges) < 2
    ):
        return (
            "<figure class='viz'><figcaption><strong>Team spatial occupancy over time</strong>"
            "<span>No continuous live-ball player intervals are available.</span></figcaption>"
            "<p class='empty'>Spatial occupancy is unavailable for this capture.</p></figure>"
        )
    payload = json.dumps(
        occupancy, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).replace("</", "<\\/")
    maximum = len(windows) - 1
    initial_cells, initial_status = _space_occupancy_snapshot(occupancy, maximum)
    return f"""<figure class="viz spatial-player">
<figcaption><strong>Team spatial occupancy over time</strong><span class="spatial-status">{initial_status}</span></figcaption>
<div class="network-controls"><button type="button" class="network-play spatial-play">Play</button><input class="network-time spatial-time" type="range" min="0" max="{maximum}" value="{maximum}" step="1" aria-label="Team occupancy cumulative match time"></div>
<div class="network-hover">Cumulative live-ball player density from kickoff through the selected time. Cell hue compares each team's normalized density (red through balanced purple to blue); opacity shows their combined density. This is not modeled territory control.</div>
<svg viewBox="0 0 720 500" role="img" aria-label="Interactive smoothed cumulative team spatial occupancy from kickoff">
<defs>{_spatial_balance_gradients()}<clipPath id="spatial-pitch-clip"><rect x="30" y="38" width="660" height="427.4" rx="3"/></clipPath></defs>
<rect x="30" y="38" width="660" height="427.4" rx="3" fill="#0b1628"/>
<g class="spatial-cells" clip-path="url(#spatial-pitch-clip)">{initial_cells}</g>
<g class="pitch-lines"><rect x="30" y="38" width="660" height="427.4"/><line x1="360" y1="38" x2="360" y2="465.4"/><circle cx="360" cy="251.7" r="57.5"/><rect x="30" y="125.3" width="103.7" height="253.6"/><rect x="586.3" y="125.3" width="103.7" height="253.6"/></g>
<text x="30" y="487" class="axis-label">Team 1 attacks &lt;-</text><text x="690" y="487" text-anchor="end" class="axis-label">-&gt; Team 0 attacks</text>
<g class="legend"><circle cx="210" cy="18" r="6" fill="#ff365f"/><text x="224" y="22">Team 0 edge</text><circle cx="344" cy="18" r="6" fill="{_spatial_balance_color(8)}"/><text x="358" y="22">balanced</text><circle cx="454" cy="18" r="6" fill="#2787ff"/><text x="468" y="22">Team 1 edge</text></g>
</svg><noscript><p class="empty">Enable JavaScript to move the cumulative time bar.</p></noscript></figure>
<script type="application/json" id="space-occupancy-data">{payload}</script>
<script>
(()=>{{const d=JSON.parse(document.getElementById("space-occupancy-data").textContent),F=document.querySelector(".spatial-player"),S=F.querySelector(".spatial-time"),G=F.querySelector(".spatial-cells"),T=F.querySelector(".spatial-status"),B=F.querySelector(".spatial-play"),nx=d.x_edges_m.length-1,ny=d.y_edges_m.length-1,byWindow=new Map,prefix=[],running=new Float64Array(nx*ny*2);let timer=null;
d.rows.forEach(r=>{{const a=byWindow.get(+r[0])||[];a.push(r);byWindow.set(+r[0],a)}});
d.windows.forEach(w=>{{(byWindow.get(+w.index)||[]).forEach(r=>{{const tm=+r[1],xi=+r[2],yi=+r[3],v=+r[4];if((tm===0||tm===1)&&xi>=0&&xi<nx&&yi>=0&&yi<ny&&Number.isFinite(v)&&v>0)running[2*(yi*nx+xi)+tm]+=v}});prefix.push(running.slice())}});
const clock=s=>{{s=Math.max(0,Math.round(+s));return String(Math.floor(s/60)).padStart(2,"0")+":"+String(s%60).padStart(2,"0")}};
const draw=()=>{{const cut=+S.value,w=d.windows[cut],cells=prefix[cut],tot=[0,0],attack=[0,0],combined=new Float64Array(nx*ny);for(let yi=0;yi<ny;yi++)for(let xi=0;xi<nx;xi++){{const k=2*(yi*nx+xi),center=(+d.x_edges_m[xi]+ +d.x_edges_m[xi+1])/2;tot[0]+=cells[k];tot[1]+=cells[k+1];if(center>0)attack[0]+=cells[k];if(center<0)attack[1]+=cells[k+1]}}let combinedPeak=0;for(let cell=0;cell<nx*ny;cell++){{const k=2*cell,p0=tot[0]?cells[k]/tot[0]:0,p1=tot[1]?cells[k+1]/tot[1]:0;combined[cell]=p0+p1;combinedPeak=Math.max(combinedPeak,combined[cell])}}let fields="",radius=1.72*Math.max(660/nx,427.4/ny);for(let yi=0;yi<ny;yi++)for(let xi=0;xi<nx;xi++){{const cell=yi*nx+xi,k=2*cell,p0=tot[0]?cells[k]/tot[0]:0,p1=tot[1]?cells[k+1]/tot[1]:0;if(!(p0>0||p1>0))continue;const cx=30+(xi+.5)/nx*660,cy=38+(ny-yi-.5)/ny*427.4,label="Team occupancy | T0 "+(100*p0).toFixed(2)+"% | T1 "+(100*p1).toFixed(2)+"%",balance=Math.max(0,Math.min(16,Math.round(16*p1/(p0+p1)))),a=.12+.88*Math.sqrt(combined[cell]/Math.max(combinedPeak,1e-12));fields+='<circle class="spatial-density-cell" cx="'+cx.toFixed(2)+'" cy="'+cy.toFixed(2)+'" r="'+radius.toFixed(2)+'" fill="url(#spatial-balance-'+balance+')" opacity="'+a.toFixed(3)+'"><title>'+label+'</title></circle>'}}G.innerHTML='<g class="spatial-density-fields">'+fields+'</g>';T.textContent="Kickoff–"+clock(w.end_clock_s)+" | cumulative attacking-half occupancy: T0 "+(tot[0]?100*attack[0]/tot[0]:0).toFixed(1)+"% · T1 "+(tot[1]?100*attack[1]/tot[1]:0).toFixed(1)+"%"}};
S.oninput=draw;B.onclick=()=>{{if(timer){{clearInterval(timer);timer=null;B.textContent="Play";return}}if(+S.value>=+S.max)S.value=0;B.textContent="Pause";timer=setInterval(()=>{{S.value=Math.min(+S.max,+S.value+1);draw();if(+S.value>=+S.max){{clearInterval(timer);timer=null;B.textContent="Play"}}}},220)}};draw()}})();
</script>"""


def _shot_map_chart(report: dict[str, Any], team: int) -> str:
    """Render shot outcomes with an on-pitch prior-action progression overlay."""

    visualizations = report.get("visualizations", {})
    shot_map = visualizations.get("shot_map", {})
    rows = [
        row
        for row in shot_map.get("rows", [])
        if isinstance(row, dict) and int(row.get("team", -1)) == team
    ]
    density = visualizations.get("ball_density_2d", {})
    x_edges = density.get("x_edges_m", [])
    y_edges = density.get("y_edges_m", [])
    if (
        not isinstance(x_edges, list)
        or len(x_edges) < 2
        or not isinstance(y_edges, list)
        or len(y_edges) < 2
    ):
        return (
            f"<figure class='viz shot-map'><figcaption><strong>Team {team} realized shot map</strong>"
            "<span>Pitch geometry is unavailable.</span></figcaption>"
            "<p class='empty'>Shot locations cannot be rendered without verified pitch edges.</p></figure>"
        )
    pitch_x, pitch_y, pitch_width = 30.0, 66.0, 660.0
    pitch_length = float(x_edges[-1]) - float(x_edges[0])
    pitch_span_y = float(y_edges[-1]) - float(y_edges[0])
    pitch_height = pitch_width * pitch_span_y / max(pitch_length, 1.0)
    scale_x = pitch_width / max(pitch_length, 1.0)
    scale_y = pitch_height / max(pitch_span_y, 1.0)
    team_color = "#ff365f" if team == 0 else "#2787ff"
    event_badges = {
        "control_touch": ("T", "#00d5c8"),
        "pass": ("P", "#42b6ff"),
        "shot": ("S", "#ff365f"),
        "clear": ("C", "#c3cad8"),
        "challenge": ("X", "#ffb229"),
        "sequence_start": ("R", "#b37cff"),
    }

    def point(raw: list[float]) -> tuple[float, float]:
        return (
            pitch_x + (float(raw[0]) - float(x_edges[0])) * scale_x,
            pitch_y + pitch_height - (float(raw[1]) - float(y_edges[0])) * scale_y,
        )

    def faded_route_segments(
        route_points: list[tuple[float, float]],
        *,
        segment_class: str,
        shadow_class: str,
    ) -> str:
        segments = list(pairwise(route_points))
        pieces: list[str] = []
        for index, (start, end) in enumerate(segments):
            age = 1.0 if len(segments) == 1 else index / (len(segments) - 1)
            opacity = 0.2 + 0.8 * age
            shadow_opacity = 0.12 + 0.58 * age
            arrow = (
                f" marker-end='url(#{route_arrow})'"
                if index == len(segments) - 1
                else ""
            )
            coordinates = (
                f"x1='{start[0]:.2f}' y1='{start[1]:.2f}' "
                f"x2='{end[0]:.2f}' y2='{end[1]:.2f}'"
            )
            pieces.append(
                f"<line class='{shadow_class}' {coordinates} "
                f"opacity='{shadow_opacity:.3f}'/>"
                f"<line class='{segment_class}' {coordinates} "
                f"data-route-age='{age:.3f}' opacity='{opacity:.3f}'{arrow}/>"
            )
        return "".join(pieces)

    def positioned(event: Any) -> bool:
        if not isinstance(event, dict):
            return False
        raw = event.get("position_m")
        if not isinstance(raw, list) or len(raw) != 2:
            return False
        try:
            values = [float(raw[0]), float(raw[1])]
        except (TypeError, ValueError):
            return False
        return (
            all(math.isfinite(value) for value in values)
            and float(x_edges[0]) <= values[0] <= float(x_edges[-1])
            and float(y_edges[0]) <= values[1] <= float(y_edges[-1])
        )

    counts = {
        category: sum(row.get("category") == category for row in rows)
        for category in ("goal", "on_target", "off_target", "unresolved")
    }
    route_arrow = f"shot-route-arrow-{team}"
    pieces = [
        f"<defs><marker id='{route_arrow}' viewBox='0 0 10 10' refX='8' refY='5' markerUnits='userSpaceOnUse' markerWidth='8' markerHeight='8' orient='auto'><path d='M0 0L10 5L0 10Z' fill='#ffffff'/></marker></defs>",
        f"<rect x='{pitch_x}' y='{pitch_y}' width='{pitch_width}' height='{pitch_height:.2f}' fill='#236746'/>",
        (
            "<g class='legend shot-event-legend'>"
            "<circle cx='68' cy='42' r='7' fill='#b37cff'/><text class='shot-route-glyph' x='68' y='45'>R</text><text x='79' y='46'>attack start</text>"
            "<circle cx='176' cy='42' r='7' fill='#00d5c8'/><text class='shot-route-glyph' x='176' y='45'>T</text><text x='187' y='46'>control</text>"
            "<circle cx='263' cy='42' r='7' fill='#42b6ff'/><text class='shot-route-glyph' x='263' y='45'>P</text><text x='274' y='46'>pass</text>"
            "<circle cx='334' cy='42' r='7' fill='#ff365f'/><text class='shot-route-glyph' x='334' y='45'>S</text><text x='345' y='46'>shot</text>"
            "<circle cx='404' cy='42' r='7' fill='#c3cad8'/><text class='shot-route-glyph' x='404' y='45'>C</text><text x='415' y='46'>clear</text>"
            "<circle cx='482' cy='42' r='7' fill='#ffb229'/><text class='shot-route-glyph' x='482' y='45'>X</text><text x='493' y='46'>challenge</text>"
            "</g>"
        ),
    ]
    box_width = pitch_width * 16.5 / 105.0
    box_height = pitch_height * 40.32 / 68.0
    box_y = pitch_y + (pitch_height - box_height) / 2.0
    circle_radius = pitch_width * 9.15 / 105.0
    pieces.append(
        f"<g class='pitch-lines'><rect x='{pitch_x}' y='{pitch_y}' width='{pitch_width}' height='{pitch_height:.2f}'/>"
        f"<line x1='{pitch_x + pitch_width / 2:.2f}' y1='{pitch_y}' x2='{pitch_x + pitch_width / 2:.2f}' y2='{pitch_y + pitch_height:.2f}'/>"
        f"<circle cx='{pitch_x + pitch_width / 2:.2f}' cy='{pitch_y + pitch_height / 2:.2f}' r='{circle_radius:.2f}'/>"
        f"<rect x='{pitch_x}' y='{box_y:.2f}' width='{box_width:.2f}' height='{box_height:.2f}'/>"
        f"<rect x='{pitch_x + pitch_width - box_width:.2f}' y='{box_y:.2f}' width='{box_width:.2f}' height='{box_height:.2f}'/></g>"
    )
    symbol_pieces: list[str] = []
    overlay_pieces: list[str] = []
    for row in rows:
        raw_shot = row.get("position_m")
        if not isinstance(raw_shot, list) or len(raw_shot) != 2:
            continue
        try:
            x, y = point(raw_shot)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        category = str(row.get("category", "unresolved"))
        context = row.get("pre_shot_context", {})
        context_label = str(context.get("label", "Context unavailable"))
        title = html.escape(
            f"{row.get('clock_label', 'time unavailable')} | #{row.get('player_id')} | "
            f"{category.replace('_', ' ')} | {str(row.get('resolution', 'unknown')).replace('_', ' ')} | "
            f"{context_label} | ({float(raw_shot[0]):.1f}, {float(raw_shot[1]):.1f}) m"
        )
        if category == "goal":
            symbol = (
                f"<circle cx='{x:.2f}' cy='{y:.2f}' r='8.0' fill='#f6b73c' stroke='white' stroke-width='1.8'/>"
                f"<text x='{x:.2f}' y='{y + 3.2:.2f}' text-anchor='middle' fill='#18201c' font-size='9' font-weight='800'>G</text>"
            )
        elif category == "on_target":
            symbol = f"<circle cx='{x:.2f}' cy='{y:.2f}' r='6.2' fill='{team_color}' stroke='white' stroke-width='1.6'/>"
        elif category == "off_target":
            symbol = f"<circle cx='{x:.2f}' cy='{y:.2f}' r='6.2' fill='none' stroke='{team_color}' stroke-width='2.2'/>"
        else:
            symbol = f"<path d='M {x:.2f} {y - 6:.2f} L {x + 6:.2f} {y:.2f} L {x:.2f} {y + 6:.2f} L {x - 6:.2f} {y:.2f} Z' fill='#a7afbd' stroke='white' stroke-width='1.2'/>"

        history = [
            event
            for event in list(row.get("preceding_events", []))[
                -SHOT_ROUTE_MAX_PRIOR_NODES:
            ]
            if positioned(event)
        ]
        history_pieces = [
            "<g class='shot-history'>",
            f"<text class='shot-route-context' x='{pitch_x + 10:.1f}' y='{pitch_y + 18:.1f}'>{html.escape(context_label)} · "
            + (
                "actual 15 s tracking position history → shot"
                if row.get("tracking_path_basis") == "bounded_tracking_lookback"
                else "actual live-ball tracking path + exact events → shot"
            )
            + "</text>",
        ]
        route_nodes = [(event, *point(event["position_m"])) for event in history]
        tracking_points: list[tuple[float, float]] = []
        for item in row.get("tracking_path", []):
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            try:
                raw_tracking_position = [float(item[1]), float(item[2])]
            except (TypeError, ValueError, OverflowError):
                continue
            if (
                all(math.isfinite(value) for value in raw_tracking_position)
                and float(x_edges[0]) <= raw_tracking_position[0] <= float(x_edges[-1])
                and float(y_edges[0]) <= raw_tracking_position[1] <= float(y_edges[-1])
            ):
                tracking_points.append(point(raw_tracking_position))
        if tracking_points:
            tracking_points.append((x, y))
            track_class = "shot-route-track"
            if row.get("tracking_path_basis") == "bounded_tracking_lookback":
                track_class += " shot-route-lookback"
            route_title = (
                "Actual tracking position history; may include dead-ball placement."
                if row.get("tracking_path_basis") == "bounded_tracking_lookback"
                else "Actual continuous live-ball tracking path."
            )
            history_pieces.append(
                f"<g stroke='{team_color}'><title>{route_title}</title>"
                + faded_route_segments(
                    tracking_points,
                    segment_class=track_class,
                    shadow_class="shot-route-track-shadow",
                )
                + "</g>"
            )
        if history:
            if not tracking_points:
                route_points = [
                    (node_x, node_y) for _, node_x, node_y in route_nodes
                ] + [(x, y)]
                history_pieces.append(
                    faded_route_segments(
                        route_points,
                        segment_class="shot-route-segment",
                        shadow_class="shot-route-shadow",
                    )
                )
            for event_index, (event, node_x, node_y) in enumerate(route_nodes):
                event_type = str(event.get("type", "event"))
                badge, color = event_badges.get(event_type, ("E", "#c3cad8"))
                player = (
                    ""
                    if event.get("player_id") is None
                    else f" #{event.get('player_id')}"
                )
                label = str(event.get("label", event_type)).replace("_", " ")
                anchor = "middle"
                label_x = node_x
                label_offset = 15.0 if event_index % 2 else -12.0
                label_y = max(
                    pitch_y + 34.0,
                    min(pitch_y + pitch_height - 8.0, node_y + label_offset),
                )
                node_title = html.escape(
                    f"{event.get('clock_label', '--:--')} | {label}{player} | "
                    f"({float(event['position_m'][0]):.1f}, {float(event['position_m'][1]):.1f}) m"
                )
                history_pieces.extend(
                    (
                        f"<circle class='shot-route-node shot-event-{html.escape(event_type)}' cx='{node_x:.2f}' cy='{node_y:.2f}' r='7' fill='{color}'><title>{node_title}</title></circle>",
                        f"<text class='shot-route-glyph' x='{node_x:.2f}' y='{node_y + 3.0:.2f}'>{html.escape(badge)}</text>",
                        f"<text class='shot-route-label' x='{label_x:.2f}' y='{label_y:.2f}' text-anchor='{anchor}'>{html.escape(str(event.get('clock_label', '--:--')))}</text>",
                    )
                )
        elif not tracking_points:
            history_pieces.extend(
                (
                    f"<rect class='shot-route-note-bg' x='{pitch_x + 170:.1f}' y='{pitch_y + pitch_height / 2 - 19:.1f}' width='320' height='38' rx='6'/>",
                    f"<text class='shot-route-note' x='{pitch_x + pitch_width / 2:.1f}' y='{pitch_y + pitch_height / 2 + 4:.1f}'>No pre-shot tracking samples or positioned events are available.</text>",
                )
            )
        history_pieces.append("</g>")
        aria = html.escape(f"{title}. Focus to show prior on-pitch event progression")
        symbol_pieces.append(
            f"<g class='shot-symbol'><title>{title}</title>{symbol}</g>"
        )
        overlay_pieces.append(
            f"<g class='shot-marker' tabindex='0' role='img' onpointerup='this.focus()' aria-label='{aria}'>"
            f"<title>{title}</title><circle class='shot-hit-target' cx='{x:.2f}' cy='{y:.2f}' r='12' fill='transparent' pointer-events='all'/>"
            f"{''.join(history_pieces)}</g>"
        )
    pieces.append("<g class='shot-symbol-layer'>" + "".join(symbol_pieces) + "</g>")
    pieces.append("<g class='shot-overlay-layer'>" + "".join(overlay_pieces) + "</g>")
    footer_y = pitch_y + pitch_height + 26.0
    return (
        f"<figure class='viz shot-map'><figcaption><strong>Team {team} realized shot map</strong>"
        f"<span>{len(rows)} shots | {counts['goal']} goals | {counts['on_target']} saved on target | {counts['off_target']} off target"
        + (f" | {counts['unresolved']} unresolved" if counts["unresolved"] else "")
        + "</span></figcaption>"
        f"<div class='network-hover'>Tap, hover, or keyboard-focus a shot to draw its actual tracking ball path and any exact positioned events above every shot marker.</div>"
        f"<svg viewBox='0 0 720 {footer_y + 18:.0f}' role='img' aria-label='Team {team} attack-normalized realized shot outcomes'>"
        + "".join(pieces)
        + f"<text x='{pitch_x}' y='{footer_y:.2f}' class='axis-label'>own goal</text>"
        + f"<text x='{pitch_x + pitch_width:.2f}' y='{footer_y:.2f}' text-anchor='end' class='axis-label'>attacking direction -&gt;</text>"
        + f"<g class='legend'><circle cx='270' cy='18' r='6' fill='#f6b73c' stroke='white'/><text x='282' y='22'>goal</text><circle cx='340' cy='18' r='5' fill='{team_color}' stroke='white'/><text x='351' y='22'>saved on target</text><circle cx='475' cy='18' r='5' fill='none' stroke='{team_color}' stroke-width='2'/><text x='486' y='22'>off target</text><path d='M 570 12 L 576 18 L 570 24 L 564 18 Z' fill='#a7afbd'/><text x='582' y='22'>unresolved</text></g>"
        + "</svg></figure>"
    )


def _shot_context_chart(report: dict[str, Any]) -> str:
    """Compare the pre-shot context composition of all shots and goals."""

    context = report.get("visualizations", {}).get("shot_context", {})
    rows = list(context.get("rows", []))
    if not rows:
        return (
            "<figure class='viz'><figcaption><strong>Pre-shot context mix</strong>"
            "<span>No realized-shot context is available.</span></figcaption>"
            "<p class='empty'>Context classification is unavailable.</p></figure>"
        )
    total_shots = sum(int(row["shots"]) for row in rows)
    total_goals = sum(int(row["goals"]) for row in rows)
    width = 720.0
    left = 166.0
    plot_width = 430.0
    top = 48.0
    row_height = 43.0
    height = top + row_height * len(rows) + 38.0
    pieces = []
    for tick in (0, 25, 50, 75, 100):
        x = left + plot_width * tick / 100.0
        pieces.append(
            f"<line class='grid-line' x1='{x:.2f}' y1='35' x2='{x:.2f}' y2='{height - 28:.2f}'/>"
            f"<text class='axis-label' x='{x:.2f}' y='{height - 10:.2f}' text-anchor='middle'>{tick}%</text>"
        )
    for index, row in enumerate(rows):
        y = top + index * row_height
        shot_share = 100.0 * float(row["shot_share"])
        goal_share = 100.0 * float(row["goal_share"])
        conversion = row.get("goal_conversion")
        conversion_text = (
            "Unavailable" if conversion is None else f"{100.0 * float(conversion):.1f}%"
        )
        label = html.escape(str(row["label"]))
        pieces.extend(
            (
                f"<text x='{left - 10:.2f}' y='{y + 15:.2f}' text-anchor='end' class='context-label'>{label}</text>",
                f"<rect x='{left:.2f}' y='{y:.2f}' width='{plot_width * shot_share / 100.0:.2f}' height='10' fill='#236746'><title>{int(row['shots'])} shots · {shot_share:.1f}% of shots · {conversion_text} conversion</title></rect>",
                f"<rect x='{left:.2f}' y='{y + 15:.2f}' width='{plot_width * goal_share / 100.0:.2f}' height='10' fill='#c08a20'><title>{int(row['goals'])} goals · {goal_share:.1f}% of goals</title></rect>",
                f"<text x='{left + plot_width + 9:.2f}' y='{y + 9:.2f}' class='context-value'>{int(row['shots'])} · {shot_share:.1f}%</text>",
                f"<text x='{left + plot_width + 9:.2f}' y='{y + 24:.2f}' class='context-value'>{int(row['goals'])} · {goal_share:.1f}%</text>",
            )
        )
    return (
        "<figure class='viz shot-context-chart'><figcaption><strong>Pre-shot context mix</strong>"
        f"<span>{total_shots} realized shots | {total_goals} goals | deterministic heuristic, not a measured football constant</span></figcaption>"
        f"<svg viewBox='0 0 {width:.0f} {height:.0f}' role='img' aria-label='Shares of shots and goals by pre-shot attacking context'>"
        + "".join(pieces)
        + "<g class='legend'><rect x='450' y='10' width='13' height='8' fill='#236746'/><text x='470' y='18'>share of shots</text><rect x='555' y='10' width='13' height='8' fill='#c08a20'/><text x='575' y='18'>share of goals</text></g></svg>"
        + "<div class='network-hover'>Priority: penalty kick → free kick → restart attack → counterattack → quick after regain → sustained buildup → other open play → unclassified. Hover bars for counts and conversion.</div></figure>"
    )


def _pass_map_chart(report: dict[str, Any], team: int) -> str:
    density = report["visualizations"]["ball_density_2d"]
    pass_map = report["visualizations"]["pass_map"]
    rows = [row for row in pass_map["rows"] if int(row["team"]) == team]
    x_edges = density["x_edges_m"]
    y_edges = density["y_edges_m"]
    pitch_x, pitch_y, pitch_width = 30.0, 38.0, 660.0
    pitch_length = float(x_edges[-1]) - float(x_edges[0])
    pitch_span_y = float(y_edges[-1]) - float(y_edges[0])
    pitch_height = pitch_width * pitch_span_y / max(pitch_length, 1.0)
    scale_x = pitch_width / max(pitch_length, 1.0)
    scale_y = pitch_height / max(pitch_span_y, 1.0)

    def point(raw: list[float]) -> tuple[float, float]:
        return (
            pitch_x + (float(raw[0]) - float(x_edges[0])) * scale_x,
            pitch_y + pitch_height - (float(raw[1]) - float(y_edges[0])) * scale_y,
        )

    same_team = sum(row["outcome"] == "same_team_next_contact" for row in rows)
    intended_matches = sum(row["intended_receiver_match"] is True for row in rows)
    same_marker = f"pass-arrow-same-{team}"
    opponent_marker = f"pass-arrow-opponent-{team}"
    shot_marker = f"shot-direction-{team}"
    pieces = [
        (
            "<defs>"
            f"<marker id='{same_marker}' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='5' markerHeight='5' orient='auto-start-reverse'><path d='M 0 0 L 10 5 L 0 10 z' fill='#f4f7f4'/></marker>"
            f"<marker id='{opponent_marker}' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='5' markerHeight='5' orient='auto-start-reverse'><path d='M 0 0 L 10 5 L 0 10 z' fill='#f0a35b'/></marker>"
            f"<marker id='{shot_marker}' viewBox='0 0 10 10' refX='9' refY='5' markerUnits='userSpaceOnUse' markerWidth='7' markerHeight='7' orient='auto'><path d='M 0 0 L 10 5 L 0 10 z' fill='#ff7f0e'/></marker>"
            "</defs>"
        ),
        f"<rect x='{pitch_x}' y='{pitch_y}' width='{pitch_width}' height='{pitch_height:.2f}' fill='#236746'/>",
    ]
    for row in rows:
        start_x, start_y = point(row["start_m"])
        target = row["intended_target_m"]
        if target is not None:
            target_x, target_y = point(target)
            pieces.append(
                f"<line x1='{start_x:.2f}' y1='{start_y:.2f}' x2='{target_x:.2f}' y2='{target_y:.2f}' "
                "stroke='#8cb4c0' stroke-width='1.2' stroke-dasharray='4 4' opacity='.42'/>"
                f"<circle cx='{target_x:.2f}' cy='{target_y:.2f}' r='3.2' fill='none' stroke='#8cb4c0' opacity='.72'/>"
            )
        end = row["actual_end_m"]
        if end is not None:
            end_x, end_y = point(end)
            same_team_outcome = row["outcome"] == "same_team_next_contact"
            color = "#f4f7f4" if same_team_outcome else "#f0a35b"
            marker = same_marker if same_team_outcome else opponent_marker
            pieces.append(
                f"<g><title>#{row['passer_player_id']} intended #{row['intended_receiver_player_id']}; "
                f"next #{row['actual_next_player_id']}; {row['outcome'].replace('_', ' ')}; "
                f"{_fmt(row['distance_m'])} m; {row['direction_family']}</title>"
                f"<line x1='{start_x:.2f}' y1='{start_y:.2f}' x2='{end_x:.2f}' y2='{end_y:.2f}' "
                f"stroke='{color}' stroke-width='1.6' opacity='.74' marker-end='url(#{marker})'/>"
                f"<circle cx='{start_x:.2f}' cy='{start_y:.2f}' r='2.3' fill='{color}'/></g>"
            )
        else:
            pieces.append(
                f"<g stroke='#f2d04f' stroke-width='1.8'><title>#{row['passer_player_id']}: "
                f"{row['outcome'].replace('_', ' ')}</title>"
                f"<line x1='{start_x - 3:.2f}' y1='{start_y - 3:.2f}' x2='{start_x + 3:.2f}' y2='{start_y + 3:.2f}'/>"
                f"<line x1='{start_x - 3:.2f}' y1='{start_y + 3:.2f}' x2='{start_x + 3:.2f}' y2='{start_y - 3:.2f}'/></g>"
            )
    intent_rows = [
        row
        for row in report["visualizations"].get("intent_actions", {}).get("rows", [])
        if int(row["team"]) == team and int(row["intent"]) in (3, 4, 5)
    ]
    intent_style = {
        3: ("shot", "#ff7f0e", "circle"),
        4: ("clear", "#b388ff", "square"),
        5: ("challenge", "#f03b73", "diamond"),
    }
    for action in intent_rows:
        action_x, action_y = point(action["position_m"])
        name, color, shape = intent_style[int(action["intent"])]
        title = f"#{action['player_id']} {name} intent"
        if shape == "circle":
            direction = action.get("direction_unit")
            valid_direction = (
                isinstance(direction, (list, tuple))
                and len(direction) == 2
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in direction
                )
            )
            if valid_direction:
                direction_x, direction_y = float(direction[0]), float(direction[1])
                magnitude = math.hypot(direction_x, direction_y)
                valid_direction = magnitude > 1.0e-9
            if valid_direction:
                direction_x /= magnitude
                direction_y /= magnitude
                arrow_length_m = 9.0
                end_m = [
                    min(
                        max(
                            float(action["position_m"][0])
                            + arrow_length_m * direction_x,
                            float(x_edges[0]),
                        ),
                        float(x_edges[-1]),
                    ),
                    min(
                        max(
                            float(action["position_m"][1])
                            + arrow_length_m * direction_y,
                            float(y_edges[0]),
                        ),
                        float(y_edges[-1]),
                    ),
                ]
                end_x, end_y = point(end_m)
                title += f"; direction ({direction_x:.3f}, {direction_y:.3f})"
                symbol = (
                    f"<line x1='{action_x:.2f}' y1='{action_y:.2f}' x2='{end_x:.2f}' y2='{end_y:.2f}' "
                    f"stroke='{color}' stroke-width='2.2' marker-end='url(#{shot_marker})'/>"
                    f"<circle cx='{action_x:.2f}' cy='{action_y:.2f}' r='3.2' fill='{color}'/>"
                )
            else:
                symbol = f"<circle cx='{action_x:.2f}' cy='{action_y:.2f}' r='4.2' fill='{color}'/>"
        elif shape == "square":
            symbol = f"<rect x='{action_x - 3.8:.2f}' y='{action_y - 3.8:.2f}' width='7.6' height='7.6' fill='{color}'/>"
        else:
            symbol = f"<path d='M {action_x:.2f} {action_y - 4.8:.2f} L {action_x + 4.8:.2f} {action_y:.2f} L {action_x:.2f} {action_y + 4.8:.2f} L {action_x - 4.8:.2f} {action_y:.2f} Z' fill='{color}'/>"
        pieces.append(f"<g opacity='.68'><title>{title}</title>{symbol}</g>")
    box_width = pitch_width * 16.5 / 105.0
    box_height = pitch_height * 40.32 / 68.0
    box_y = pitch_y + (pitch_height - box_height) / 2.0
    circle_radius = pitch_width * 9.15 / 105.0
    pieces.append(
        f"<g class='pitch-lines'><rect x='{pitch_x}' y='{pitch_y}' width='{pitch_width}' height='{pitch_height:.2f}'/>"
        f"<line x1='{pitch_x + pitch_width / 2:.2f}' y1='{pitch_y}' x2='{pitch_x + pitch_width / 2:.2f}' y2='{pitch_y + pitch_height:.2f}'/>"
        f"<circle cx='{pitch_x + pitch_width / 2:.2f}' cy='{pitch_y + pitch_height / 2:.2f}' r='{circle_radius:.2f}'/>"
        f"<rect x='{pitch_x}' y='{box_y:.2f}' width='{box_width:.2f}' height='{box_height:.2f}'/>"
        f"<rect x='{pitch_x + pitch_width - box_width:.2f}' y='{box_y:.2f}' width='{box_width:.2f}' height='{box_height:.2f}'/></g>"
    )
    footer_y = pitch_y + pitch_height + 25.0
    return (
        f"<figure class='viz'><figcaption><strong>Team {team} pass map</strong>"
        f"<span>{len(rows)} realized | {same_team} same-team next contacts | {intended_matches} intended matches</span></figcaption>"
        f"<svg viewBox='0 0 720 {footer_y + 18:.0f}' role='img' aria-label='Team {team} attack-normalized realized pass map'>"
        + "".join(pieces)
        + f"<text x='{pitch_x}' y='{footer_y:.2f}' class='axis-label'>own goal</text>"
        + f"<text x='{pitch_x + pitch_width:.2f}' y='{footer_y:.2f}' text-anchor='end' class='axis-label'>attacking direction -&gt;</text>"
        + "<g class='legend'><line x1='210' y1='18' x2='230' y2='18' stroke='#f4f7f4' stroke-width='2'/><text x='237' y='22'>same team</text>"
        + "<line x1='324' y1='18' x2='344' y2='18' stroke='#f0a35b' stroke-width='2'/><text x='351' y='22'>opponent</text>"
        + f"<line x1='390' y1='18' x2='410' y2='18' stroke='#8cb4c0' stroke-dasharray='4 4'/><text x='417' y='22'>target</text><line x1='482' y1='18' x2='500' y2='18' stroke='#ff7f0e' stroke-width='2' marker-end='url(#{shot_marker})'/><text x='506' y='22'>shot</text><rect x='548' y='14' width='8' height='8' fill='#b388ff'/><text x='561' y='22'>clear</text><path d='M 614 13 L 619 18 L 614 23 L 609 18 Z' fill='#f03b73'/><text x='624' y='22'>challenge</text></g>"
        + "</svg></figure>"
    )


def _passing_network_charts(report: dict[str, Any]) -> str:
    """Render one cumulative, identity-aware pass network controlled by match time."""

    visualizations = report.get("visualizations", {})
    rows = visualizations.get("pass_map", {}).get("rows", [])
    shots = [
        row
        for row in visualizations.get("intent_actions", {}).get("rows", [])
        if int(row.get("intent", -1)) == 3
    ]
    position_data = visualizations.get("player_position_occupancy", {})
    occupancy_rows = position_data.get("rows", [])
    activity_rows = list(position_data.get("activity", []))
    x_edges = position_data.get("x_edges_m", [-52.5, 52.5])
    y_edges = position_data.get("y_edges_m", [-34.0, 34.0])
    timeline = report.get("timeline", [])

    if not activity_rows:
        fallback: dict[tuple[int, int, int], dict[str, Any]] = {}

        def observe_fallback(
            team: int, identity: list[int], tick: int, position: list[float]
        ) -> None:
            key = (team, int(identity[0]), int(identity[1]))
            current = fallback.setdefault(
                key,
                {
                    "team": team,
                    "player_id": key[1],
                    "slot_generation": key[2],
                    "first_tick": tick,
                    "last_tick": tick,
                    "initial_position_m": position,
                },
            )
            current["first_tick"] = min(int(current["first_tick"]), tick)
            current["last_tick"] = max(int(current["last_tick"]), tick)

        for row in rows:
            team = int(row["team"])
            tick = int(row["control_tick"])
            observe_fallback(
                team,
                [
                    int(row["passer_player_id"]),
                    int(row.get("passer_slot_generation", 0)),
                ],
                tick,
                row["start_m"],
            )
            if (
                row.get("actual_next_player_id") is not None
                and row.get("actual_end_m") is not None
            ):
                observe_fallback(
                    team,
                    [
                        int(row["actual_next_player_id"]),
                        int(row.get("actual_next_slot_generation") or 0),
                    ],
                    tick,
                    row["actual_end_m"],
                )
        for row in shots:
            observe_fallback(
                int(row["team"]),
                [int(row["player_id"]), int(row.get("slot_generation", 0))],
                int(row["control_tick"]),
                row["position_m"],
            )
        activity_rows = list(fallback.values())

    final_tick = max(
        [
            *(int(row["control_tick"]) for row in rows),
            *(int(row["control_tick"]) for row in shots),
            *(int(row[0]) for row in occupancy_rows),
            *(int(row["last_tick"]) for row in activity_rows),
            *(
                int(item["control_tick"])
                for item in timeline
                if item.get("control_tick") is not None
            ),
            1,
        ]
    )
    clock_samples = [
        int(item["control_tick"]) / float(item["clock_s"])
        for item in timeline
        if int(item.get("control_tick", 0)) > 0
        and isinstance(item.get("clock_s"), (int, float))
        and float(item["clock_s"]) > 0.0
    ]
    control_fps = sum(clock_samples) / len(clock_samples) if clock_samples else 10.0

    payloads = []
    figures = []
    for team in (0, 1):
        completed = [
            {
                "tick": int(row["control_tick"]),
                "source": [
                    int(row["passer_player_id"]),
                    int(row.get("passer_slot_generation", 0)),
                ],
                "target": [
                    int(row["actual_next_player_id"]),
                    int(row.get("actual_next_slot_generation") or 0),
                ],
            }
            for row in rows
            if int(row["team"]) == team
            and row.get("outcome") == "same_team_next_contact"
            and row.get("actual_next_player_id") is not None
        ]
        team_shots = [
            {
                "tick": int(row["control_tick"]),
                "player": [
                    int(row["player_id"]),
                    int(row.get("slot_generation", 0)),
                ],
            }
            for row in shots
            if int(row["team"]) == team
        ]
        milestones = [
            {
                "tick": int(item["control_tick"]),
                "type": str(item["type"]),
                "label": str(item.get("label", item["type"])),
            }
            for item in timeline
            if item.get("type") in {"substitution", "formation"}
            and item.get("control_tick") is not None
            and item.get("team") in (None, team)
        ]
        players = [
            [
                int(row["player_id"]),
                int(row.get("slot_generation", 0)),
                int(row["first_tick"]),
                int(row["last_tick"]),
                float(row["initial_position_m"][0]),
                float(row["initial_position_m"][1]),
            ]
            for row in activity_rows
            if int(row["team"]) == team
        ]
        occupancy = [
            [
                int(row[0]),
                int(row[2]),
                int(row[3]),
                int(row[4]),
                int(row[5]),
                float(row[6]),
            ]
            for row in occupancy_rows
            if int(row[1]) == team
        ]
        payloads.append(
            {
                "team": team,
                "final": final_tick,
                "fps": control_fps,
                "passes": completed,
                "shots": team_shots,
                "milestones": milestones,
                "players": players,
                "occupancy": occupancy,
                "x_edges": x_edges,
                "y_edges": y_edges,
            }
        )
        figures.append(
            f"""<figure class="viz network-player">
<figcaption><strong>Team {team} cumulative attack network</strong><span class="network-status"></span></figcaption>
<div class="network-controls"><button type="button" class="network-play">Play</button><input class="network-time" type="range" min="0" max="{final_tick}" value="{final_tick}" step="1"></div>
<div class="network-hover" aria-live="polite">Hover a player node for its movement heatmap or a connection for directional pass counts.</div>
<svg viewBox="0 0 720 500" role="img" aria-label="Team {team} interactive cumulative passing network">
<rect x="30" y="38" width="660" height="427.4" fill="#163f31"/>
<g class="pitch-lines"><rect x="30" y="38" width="660" height="427.4"/><line x1="360" y1="38" x2="360" y2="465.4"/><circle cx="360" cy="251.7" r="57.5"/><rect x="30" y="125.3" width="103.7" height="253.6"/><rect x="586.3" y="125.3" width="103.7" height="253.6"/><line x1="690" y1="228" x2="690" y2="275" stroke-width="5"/></g>
<g class="network-heatmap" pointer-events="none"></g><g class="network-dynamic"></g>
<text x="30" y="487" class="axis-label">own goal</text><text x="690" y="487" text-anchor="end" class="axis-label">attacking direction - shots end at goal line</text>
</svg></figure>"""
        )
    packed = json.dumps(
        payloads, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).replace("</", "<\\/")
    script = (
        '<script type="application/json" id="network-data">'
        + packed
        + """</script>
<script>
(()=>{const D=JSON.parse(document.getElementById("network-data").textContent);
const K=x=>x[0]+":"+x[1],P=p=>[30+(+p[0]+52.5)/105*660,38+427.4-(+p[1]+34)/68*427.4];
document.querySelectorAll(".network-player").forEach((F,i)=>{const d=D[i],S=F.querySelector(".network-time"),G=F.querySelector(".network-dynamic"),H=F.querySelector(".network-heatmap"),T=F.querySelector(".network-status"),V=F.querySelector(".network-hover"),B=F.querySelector(".network-play");let timer=null,hoverKey=null,centers={},activeSeconds={};
const cellCenter=r=>[(d.x_edges[r[3]]+d.x_edges[r[3]+1])/2,(d.y_edges[r[4]]+d.y_edges[r[4]+1])/2];
const idleHint="Hover a player node for its movement heatmap or a connection for directional pass counts.";
const hideHeat=()=>{H.innerHTML="";V.textContent=idleHint};
const showHeat=key=>{const cut=+S.value,cells={},rows=d.occupancy.filter(r=>r[0]<=cut&&(r[1]+":"+r[2])===key);rows.forEach(r=>{const c=r[3]+":"+r[4];cells[c]=(cells[c]||0)+r[5]});const max=Math.max(0,...Object.values(cells));if(!max){hideHeat();return}let out="";Object.entries(cells).forEach(([cell,value])=>{const [xi,yi]=cell.split(":").map(Number),a=P([d.x_edges[xi],d.y_edges[yi]]),b=P([d.x_edges[xi+1],d.y_edges[yi+1]]),opacity=.12+.68*Math.sqrt(value/max);out+='<rect x="'+Math.min(a[0],b[0])+'" y="'+Math.min(a[1],b[1])+'" width="'+Math.abs(b[0]-a[0])+'" height="'+Math.abs(b[1]-a[1])+'" fill="#ffd166" opacity="'+opacity.toFixed(3)+'"/>'});H.innerHTML=out;const id=key.split(":"),p=centers[key],mins=(activeSeconds[key]||0)/60;V.textContent="Player #"+id[0]+" | generation "+id[1]+" | "+mins.toFixed(1)+" active min | cumulative mean "+(p?"("+((p[0]-30)/660*105-52.5).toFixed(1)+", "+(34-(p[1]-38)/427.4*68).toFixed(1)+") m":"unavailable")};
const draw=()=>{const cut=+S.value,last=d.milestones.filter(x=>x.tick<=cut).sort((a,b)=>b.tick-a.tick)[0],people=d.players.filter(p=>p[2]<=cut&&cut<=p[3]),active=new Set(people.map(K)),occ=d.occupancy.filter(r=>r[0]<=cut&&active.has(r[1]+":"+r[2])),pos={},pairs={},shotCounts={};activeSeconds={};
people.forEach(p=>pos[K(p)]={id:[p[0],p[1]],x:p[4],y:p[5],n:0});
occ.forEach(r=>{const k=r[1]+":"+r[2],c=cellCenter(r),v=pos[k]||(pos[k]={id:[r[1],r[2]],x:0,y:0,n:0});if(v.n===0){v.x=0;v.y=0}v.x+=c[0]*r[5];v.y+=c[1]*r[5];v.n+=r[5];activeSeconds[k]=(activeSeconds[k]||0)+r[5]});
centers={};Object.entries(pos).forEach(([k,v])=>centers[k]=P(v.n?[v.x/v.n,v.y/v.n]:[v.x,v.y]));
d.passes.filter(x=>x.tick<=cut).forEach(x=>{const a=K(x.source),b=K(x.target);if(active.has(a)&&active.has(b)&&a!==b){const first=a<b?a:b,second=a<b?b:a,key=first+"|"+second,e=pairs[key]||(pairs[key]={a:first,b:second,ab:0,ba:0});if(a===first)e.ab++;else e.ba++}});
d.shots.filter(x=>x.tick<=cut).forEach(x=>{const k=K(x.player);if(active.has(k))shotCounts[k]=(shotCounts[k]||0)+1});
const involvement={},radii={};Object.values(pairs).forEach(e=>{const n=e.ab+e.ba;involvement[e.a]=(involvement[e.a]||0)+n;involvement[e.b]=(involvement[e.b]||0)+n});Object.keys(centers).forEach(k=>radii[k]=10+Math.min(8,Math.sqrt(involvement[k]||0)));
const totals=Object.values(pairs).map(e=>e.ab+e.ba),maxPair=Math.max(1,...totals),headSize=n=>n?Math.min(10,4+1.8*Math.sqrt(n)):0;
const arrowHead=(tipX,tipY,ux,uy,size)=>{if(!size)return "";const baseX=tipX-ux*size,baseY=tipY-uy*size,px=-uy*size*.58,py=ux*size*.58;return '<polygon points="'+tipX+','+tipY+' '+(baseX+px)+','+(baseY+py)+' '+(baseX-px)+','+(baseY-py)+'" fill="#e8eef2"/>'};
let out='<defs><marker id="shot-arrow-'+d.team+'" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#ff9f43"/></marker></defs>';
Object.values(pairs).forEach(e=>{const x=centers[e.a],y=centers[e.b],total=e.ab+e.ba;if(!x||!y)return;const dx=y[0]-x[0],dy=y[1]-x[1],len=Math.hypot(dx,dy);if(len<1)return;const ux=dx/len,uy=dy/len,sx=x[0]+ux*radii[e.a],sy=x[1]+uy*radii[e.a],ex=y[0]-ux*radii[e.b],ey=y[1]-uy*radii[e.b],label="#"+e.a.split(":")[0]+" -> #"+e.b.split(":")[0]+": "+e.ab+" completed | #"+e.b.split(":")[0]+" -> #"+e.a.split(":")[0]+": "+e.ba+" completed | "+total+" total",opacity=.28+.62*Math.sqrt(total/maxPair),width=.9+Math.min(4.1,.6*Math.sqrt(total));out+='<g class="network-edge" tabindex="0" data-label="'+label+'" aria-label="'+label+'"><line x1="'+sx+'" y1="'+sy+'" x2="'+ex+'" y2="'+ey+'" stroke="#e8eef2" stroke-width="'+width+'" opacity="'+opacity+'"><title>'+label+'</title></line>'+arrowHead(ex,ey,ux,uy,headSize(e.ab))+arrowHead(sx,sy,-ux,-uy,headSize(e.ba))+'</g>'});
Object.entries(shotCounts).forEach(([k,n])=>{const a=centers[k];if(a)out+='<line class="network-shot" x1="'+a[0]+'" y1="'+a[1]+'" x2="687" y2="251.7" stroke="#ff9f43" stroke-width="'+(2+Math.min(7,2.2*Math.sqrt(n)))+'" opacity=".82" marker-end="url(#shot-arrow-'+d.team+')"><title>'+n+' shots by #'+k.split(":")[0]+'</title></line>'});
Object.entries(centers).forEach(([k,p])=>{const v=pos[k],r=radii[k],color=d.team?"#167d9a":"#d1495b";out+='<g class="network-node" data-player="'+k+'" tabindex="0"><circle cx="'+p[0]+'" cy="'+p[1]+'" r="'+r+'" fill="'+color+'" stroke="white" stroke-width="1.5"><title>Player #'+v.id[0]+', generation '+v.id[1]+'</title></circle><text x="'+p[0]+'" y="'+(p[1]+4)+'" text-anchor="middle" fill="white" font-size="11" font-weight="700">'+v.id[0]+'</text></g>'});
G.innerHTML=out;G.querySelectorAll(".network-edge").forEach(edge=>{const enter=()=>{H.innerHTML="";V.textContent=edge.dataset.label},leave=()=>{if(!hoverKey)hideHeat()};edge.addEventListener("pointerenter",enter);edge.addEventListener("pointerleave",leave);edge.addEventListener("focus",enter);edge.addEventListener("blur",leave)});G.querySelectorAll(".network-node").forEach(node=>{const enter=()=>{hoverKey=node.dataset.player;showHeat(hoverKey)},leave=()=>{hoverKey=null;hideHeat()};node.addEventListener("pointerenter",enter);node.addEventListener("pointerleave",leave);node.addEventListener("focus",enter);node.addEventListener("blur",leave)});
if(hoverKey&&active.has(hoverKey))showHeat(hoverKey);else hideHeat();
T.textContent=(cut/d.fps/60).toFixed(1)+" min | "+(last?last.label:"opening shape")+" | "+Object.values(pairs).reduce((n,e)=>n+e.ab+e.ba,0)+" passes | "+Object.values(shotCounts).reduce((a,b)=>a+b,0)+" shots"};
S.oninput=draw;B.onclick=()=>{if(timer){clearInterval(timer);timer=null;B.textContent="Play";return}if(+S.value>=d.final)S.value=0;B.textContent="Pause";timer=setInterval(()=>{S.value=Math.min(d.final,+S.value+Math.max(1,Math.round(d.fps*15)));draw();if(+S.value>=d.final){clearInterval(timer);timer=null;B.textContent="Play"}},80)};draw()})})();
</script>"""
    )
    return "<div class='viz-grid'>" + "".join(figures) + "</div>" + script


def _comparison_rows(report: dict[str, Any]) -> str:
    metrics = (
        (
            "Full-match controlled possession",
            "Controlled live-ball time; loose-ball time is excluded.",
            "controlled_possession_share",
            100.0,
            "%",
        ),
        ("Goals", "Confirmed goals in the final score.", "goals", 1.0, ""),
        (
            "Realized shots",
            "Applied SHOT contacts with exact source positions.",
            "realized_shots",
            1.0,
            "",
        ),
        (
            "Shots on target",
            "Goals plus shots stopped by a physics-labelled deliberate save.",
            "shots_on_target",
            1.0,
            "",
        ),
        (
            "Completed open-play passes",
            "Pass contacts followed by a same-team next contact.",
            "open_play_completed_passes",
            1.0,
            "",
        ),
        (
            "Open-play pass receipt rate",
            "Same-team next contacts divided by realized open-play passes.",
            "open_play_pass_completion",
            100.0,
            "%",
        ),
        (
            "Rule-policy cross signatures",
            "Exact shipped-policy cross spin signature on realized open-play PASS contacts; not a generic cross label.",
            "rule_policy_cross_control_signatures",
            1.0,
            "",
        ),
        (
            "Defensive-line-breaking passes",
            "Source-to-next-contact second-last-opponent line breaks; a through-pass geometry proxy, not an event label.",
            "defensive_line_breaking_pass_proxies",
            1.0,
            "",
        ),
        (
            "Penalty-area entries",
            "Possession-retaining entries across the penalty-area boundary.",
            "penalty_area_entries",
            1.0,
            "",
        ),
        ("Corners", "Confirmed corner restarts awarded.", "corners", 1.0, ""),
        (
            "Fouls committed",
            "Confirmed fouls attributed to the offending team.",
            "fouls_committed",
            1.0,
            "",
        ),
        ("Offsides", "Confirmed offside offences.", "offsides", 1.0, ""),
    )
    rows = []
    teams = report["teams"]
    for label, definition, key, multiplier, unit in metrics:
        raw_values = [teams[team].get(key) for team in (0, 1)]
        values = [
            0.0 if value is None else float(value) * multiplier for value in raw_values
        ]
        scale = 100.0 if unit == "%" else max(1.0, *values)
        formatted = [
            "Unavailable"
            if raw_values[index] is None
            else (f"{value:.1f}%" if unit == "%" else f"{value:.0f}")
            for index, value in enumerate(values)
        ]
        rows.append(
            "<div class='compare-row'>"
            f"<strong><span>{html.escape(label)}</span><small>{html.escape(definition)}</small></strong>"
            f"<span class='value t0-text'>{formatted[0]}</span>"
            f"<div class='compare-track left'><i style='width:{100 * values[0] / scale:.2f}%'></i></div>"
            f"<div class='compare-track right'><i style='width:{100 * values[1] / scale:.2f}%'></i></div>"
            f"<span class='value t1-text'>{formatted[1]}</span></div>"
        )
    return "".join(rows)


def _substitution_badges(player: dict[str, Any]) -> str:
    substitution = player.get("substitution") or {}
    badges = []
    for key, symbol, label in (
        ("entered", "&uarr;", "Entered"),
        ("exited", "&darr;", "Left"),
    ):
        timing = substitution.get(key)
        if not isinstance(timing, dict):
            continue
        clock_label = str(timing.get("clock_label", "time unavailable"))
        escaped = html.escape(clock_label)
        badges.append(
            f"<span class='sub-badge {key}' title='{label} at {escaped}' aria-label='{label} at {escaped}'>{symbol} {escaped}</span>"
        )
    return "".join(badges)


def _workload_charts(report: dict[str, Any]) -> str:
    maximum = max(
        1.0,
        max((float(player["distance_m"]) for player in report["players"]), default=1.0),
    )
    columns = []
    for team in (0, 1):
        rows = []
        players = sorted(
            (player for player in report["players"] if player["team"] == team),
            key=lambda player: float(player["distance_m"]),
            reverse=True,
        )
        for player in players:
            distance = float(player["distance_m"])
            rows.append(
                "<div class='workload-row'>"
                f"<span class='workload-player'><b>#{player['player_id']}</b>{_substitution_badges(player)}</span>"
                f"<div><i class='t{team}-fill' style='width:{100 * distance / maximum:.2f}%'></i></div>"
                f"<strong>{distance / 1000:.2f} km</strong></div>"
            )
        columns.append(
            f"<div class='workload-team'><h3><span class='team-dot t{team}'></span>Team {team}</h3>{''.join(rows)}</div>"
        )
    return "".join(columns)


def _event_bars(report: dict[str, Any]) -> str:
    counts = report["events"]["exact_event_counts"]
    if not counts:
        return "<p class='empty'>No exact events were recorded.</p>"
    maximum = max(counts.values())
    return "".join(
        "<div class='event-bar'>"
        f"<span>{html.escape(kind.replace('_', ' ').title())}</span>"
        f"<div><i style='width:{100 * count / maximum:.2f}%'></i></div><strong>{count:,}</strong></div>"
        for kind, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def _reference_value(value: float | None, unit: str) -> str:
    if value is None:
        return "Unavailable"
    if unit == "ratio":
        return f"{100.0 * value:.1f}%"
    return f"{value:.3f}"


def _reference_chart(report: dict[str, Any]) -> str:
    reference = report.get("external_reference")
    if not isinstance(reference, dict):
        return ""
    badge_labels = {
        "closest_comparable": "Closest comparable",
        "proxy": "Proxy",
        "diagnostic_neighbor": "Diagnostic neighbor",
    }
    cards = []
    for item in reference["comparisons"]:
        current = item["current"]
        target = float(item["target"])
        unit = str(item["unit"])
        values = [
            value
            for value in (current, target)
            if value is not None and float(value) >= 0.0
        ]
        scale = 1.0 if unit == "ratio" else max(1.0e-9, max(values, default=1.0) * 1.14)
        current_width = (
            0.0 if current is None else 100.0 * min(float(current), scale) / scale
        )
        target_left = 100.0 * min(target, scale) / scale
        relative = item["relative_delta"]
        gap = (
            "Insufficient source facts"
            if relative is None
            else f"{relative:+.1%} vs DFL"
        )
        caution = item.get("caution") or item.get("definition") or ""
        cards.append(
            "<article class='benchmark-card'>"
            f"<div class='benchmark-head'><strong>{html.escape(item['label'])}</strong>"
            f"<span class='metric-badge {html.escape(item['comparability'])}'>"
            f"{html.escape(badge_labels[item['comparability']])}</span></div>"
            "<div class='benchmark-values'>"
            f"<span><b>{_reference_value(current, unit)}</b>Current match</span>"
            f"<span><b>{_reference_value(target, unit)}</b>DFL guide</span></div>"
            "<div class='benchmark-track'>"
            f"<span class='current-bar' style='width:{current_width:.2f}%'></span>"
            f"<i class='target-mark' style='left:{target_left:.2f}%'></i></div>"
            f"<div class='benchmark-foot'><span>{html.escape(gap)}</span>"
            f"<span>{html.escape(caution)}</span></div></article>"
        )
    source = reference["source"]
    provenance = (
        f"{source.get('provider', 'External')} aggregate, "
        f"{source.get('matches', '?')} matches at "
        f"{source.get('tracking_hz', '?')} Hz"
    )
    cautions = "".join(
        f"<li>{html.escape(item)}</li>" for item in reference.get("cautions", [])
    )
    return (
        "<section class='reference-section'><div class='section-head'>"
        "<div><h2>DFL policy alignment</h2>"
        f"<p>{html.escape(reference['title'])}</p></div>"
        "<div class='benchmark-legend'><span class='current-key'>Current match</span>"
        "<span class='target-key'>DFL guide</span></div></div>"
        "<div class='benchmark-grid'>"
        + "".join(cards)
        + "</div><div class='reference-note'>"
        f"<strong>{html.escape(provenance)}</strong>"
        f"<span>{html.escape(reference['interpretation'])}</span>"
        f"<ul>{cautions}</ul></div></section>"
    )


def _pass_completion_timeline(report: dict[str, Any]) -> str:
    """Render the audit's fixed 15-minute pass-receipt windows."""

    team_rows = [team.get("pass_completion_by_15m", []) for team in report["teams"]]
    if len(team_rows) != 2 or not any(team_rows):
        return "<p class='empty'>Time-binned pass completion is unavailable.</p>"
    rows = []
    for index in range(max(len(team_rows[0]), len(team_rows[1]))):
        cells = []
        label = f"{15 * index}–{15 * (index + 1)}"
        for team in (0, 1):
            row = team_rows[team][index] if index < len(team_rows[team]) else {}
            attempts = int(row.get("attempts", 0))
            completed = int(row.get("completed", 0))
            completion = row.get("completion")
            cells.append(
                "n/a"
                if completion is None
                else f"{100.0 * float(completion):.1f}% ({completed}/{attempts})"
            )
        rows.append(f"<tr><td>{label}</td><td>{cells[0]}</td><td>{cells[1]}</td></tr>")
    return (
        "<table><thead><tr><th>Minute</th><th><span class='team-dot t0'></span>Team 0</th>"
        "<th><span class='team-dot t1'></span>Team 1</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_html(
    dataset: MatchDataset,
    report: dict[str, Any],
    *,
    output_dir: Path,
) -> str:
    """Render a dependency-free metrics report without replay media."""

    json_href = "report.json"
    score = report["summary"]["score"]
    status = (
        "Authoritative full match"
        if report["quality"]["authoritative"]
        else "Diagnostic capture"
    )
    warnings = "".join(
        f"<li>{html.escape(item)}</li>" for item in report["quality"]["warnings"]
    )
    warning_band = (
        ""
        if not warnings
        else f"<aside class='warning'><strong>Data quality</strong><ul>{warnings}</ul></aside>"
    )
    shot_count = sum(int(team.get("realized_shots", 0)) for team in report["teams"])
    on_target_count = sum(
        int(team.get("shots_on_target", 0)) for team in report["teams"]
    )
    reference_section = _reference_chart(report)
    revision = report["source"].get("git_revision")
    provenance_parts = []
    if isinstance(revision, str) and revision:
        provenance_parts.append(f"source {revision[:8]}")
    provenance_parts.append(
        f"{report['summary']['captured_duration_s'] / 60:.1f} captured minutes"
    )
    provenance_label = " | ".join(provenance_parts)
    facts = f"""<div class="facts"><div class="fact"><strong>{report["summary"]["live_ball_s"] / 60:.1f}</strong><span>live-ball minutes</span></div><div class="fact"><strong>{shot_count:,}</strong><span>realized shots</span></div><div class="fact"><strong>{on_target_count:,}</strong><span>shots on target</span></div></div>"""
    overview = f"""<div class="metrics-overview">{facts}</div>"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FootballWorld Match Report</title>
<style>
:root {{--ink:#18201c;--muted:#647069;--paper:#f3f5f1;--line:#d7ddd8;--green:#236746;--gold:#c08a20;--red:#d1495b;--blue:#167d9a;--white:#fff;}}
* {{box-sizing:border-box}} body {{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 system-ui,sans-serif;letter-spacing:0}}
header {{background:var(--white);border-bottom:1px solid var(--line)}} .wrap {{width:min(1560px,calc(100% - 28px));margin:auto}}
.mast {{display:flex;align-items:end;justify-content:space-between;gap:20px;padding:19px 0 14px}} h1 {{font-size:27px;margin:0}} h2 {{font-size:18px;margin:0}} h3 {{font-size:11px;text-transform:uppercase;color:var(--muted);margin:0 0 7px}} p {{margin:0}}
.status {{color:var(--muted);text-align:right}} .score {{font-size:29px;font-weight:780;color:var(--green)}} .section-head {{display:flex;align-items:baseline;justify-content:space-between;gap:14px;margin-bottom:8px}} .section-head p {{color:var(--muted);font-size:11px}}
main {{padding:14px 0 36px}} .warning {{border-left:4px solid var(--gold);background:#fff8e8;padding:9px 13px;margin-bottom:12px}} .warning ul {{margin:4px 0 0;padding-left:18px}}
.facts {{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--line);background:var(--white)}} .fact {{padding:9px 12px;border-right:1px solid var(--line)}} .fact:last-child {{border:0}} .fact strong {{display:block;font-size:18px}} .fact span {{color:var(--muted);font-size:11px}}
.panel,.viz {{background:var(--white);border:1px solid var(--line);border-radius:4px}} .panel {{padding:16px}} .timeline {{max-height:520px;overflow:auto;border-top:1px solid var(--line)}} .timeline-row {{appearance:none;width:100%;border:0;border-bottom:1px solid var(--line);background:var(--white);display:grid;grid-template-columns:58px 1fr auto;gap:10px;text-align:left;padding:11px 4px;cursor:pointer;color:inherit}} .timeline-row:hover {{background:#edf5ef}} .timeline-row time {{font-variant-numeric:tabular-nums;color:var(--green);font-weight:700}} .seek {{font-size:12px;color:var(--muted)}}
.metrics-overview .facts {{border-top:1px solid var(--line)}} .timeline-row.static {{grid-template-columns:58px 1fr;cursor:default}} .timeline-row.static:hover {{background:var(--white)}}
section {{margin-top:22px}} .viz-grid {{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}} .viz-grid.single {{grid-template-columns:1fr}} .viz-grid.three {{grid-template-columns:repeat(3,minmax(0,1fr))}} .dashboard-pair {{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px}} .viz {{margin:0;padding:9px;min-width:0}} .viz figcaption {{display:flex;justify-content:space-between;gap:8px;align-items:baseline;padding:0 1px 5px}} .viz figcaption strong {{font-size:13px}} .viz figcaption span {{font-size:10px;color:var(--muted);text-align:right}} .viz svg {{display:block;width:100%;height:auto}} .grid-line {{stroke:#dfe4df;stroke-width:1}} .axis-label,.legend text {{font:11px system-ui,sans-serif;fill:#6b746e}} .pitch-lines {{fill:none;stroke:#fff;stroke-width:2;opacity:.72}} .territory-label {{font:bold 14px system-ui,sans-serif;fill:#fff;paint-order:stroke;stroke:#32463a;stroke-width:2px}}
.team-comparison {{background:var(--white);border:1px solid var(--line);padding:6px 12px}} .compare-row {{display:grid;grid-template-columns:220px 54px minmax(70px,1fr) minmax(70px,1fr) 54px;gap:7px;align-items:center;min-height:36px;border-bottom:1px solid #edf0ed}} .compare-row:last-child {{border:0}} .compare-row strong {{font-size:12px}} .compare-row strong span,.compare-row strong small {{display:block}} .compare-row strong small {{margin-top:2px;color:var(--muted);font-size:10px;font-weight:400;line-height:1.25}} .value {{font-variant-numeric:tabular-nums;font-weight:750}} .t0-text {{text-align:right;color:var(--red)}} .t1-text {{color:var(--blue)}} .compare-track {{height:9px;background:#e8ece8;overflow:hidden}} .compare-track i {{display:block;height:100%}} .compare-track.left i {{background:var(--red);margin-left:auto}} .compare-track.right i {{background:var(--blue)}}
.benchmark-grid {{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}} .benchmark-card {{background:var(--white);border:1px solid var(--line);border-radius:4px;padding:11px;min-width:0}} .benchmark-head {{display:flex;justify-content:space-between;align-items:center;gap:12px}} .benchmark-head strong {{font-size:15px}} .metric-badge {{font-size:10px;text-transform:uppercase;padding:3px 6px;border:1px solid var(--line);color:var(--muted);white-space:nowrap}} .metric-badge.closest_comparable {{border-color:#7cad91;color:var(--green)}} .metric-badge.proxy {{border-color:#d7b66f;color:#805c16}} .metric-badge.diagnostic_neighbor {{border-style:dashed}}
.benchmark-values {{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin:15px 0 11px}} .benchmark-values span {{font-size:10px;color:var(--muted);text-transform:uppercase}} .benchmark-values b {{display:block;color:var(--ink);font-size:17px;text-transform:none;font-variant-numeric:tabular-nums}} .benchmark-track {{position:relative;height:12px;background:#e7ebe8;margin:18px 7px 14px 0}} .current-bar {{display:block;height:100%;background:var(--green);min-width:2px}} .target-mark {{position:absolute;top:-6px;width:3px;height:24px;background:var(--gold)}} .baseline-mark {{position:absolute;top:2px;width:8px;height:8px;margin-left:-4px;border-radius:50%;background:var(--ink);box-shadow:0 0 0 2px var(--white)}} .benchmark-foot {{display:grid;grid-template-columns:120px 1fr;gap:12px;color:var(--muted);font-size:11px}} .benchmark-foot span:first-child {{font-weight:700;color:var(--ink)}} .benchmark-legend {{display:flex;gap:16px;color:var(--muted);font-size:11px}} .benchmark-legend span::before {{content:"";display:inline-block;width:14px;height:7px;margin-right:6px;vertical-align:0}} .current-key::before {{background:var(--green)}} .target-key::before {{background:var(--gold)}} .baseline-key::before {{background:var(--ink);border-radius:50%}} .reference-note {{display:grid;grid-template-columns:240px 1fr;gap:10px 22px;margin-top:12px;padding:12px 14px;border-left:3px solid var(--gold);background:#fffaf0;color:var(--muted);font-size:11px}} .reference-note strong {{color:var(--ink)}} .reference-note ul {{grid-column:1/-1;margin:0;padding-left:18px}}
.workload-grid {{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;background:var(--white);border:1px solid var(--line);padding:10px 12px}} .workload-team {{min-width:0}} .workload-row {{display:grid;grid-template-columns:132px minmax(70px,1fr) 62px;gap:7px;align-items:center;min-height:27px;font-variant-numeric:tabular-nums}} .workload-row>div {{height:8px;background:#e1e6e2}} .workload-row i {{display:block;height:100%}} .t0-fill {{background:var(--red)}} .t1-fill {{background:var(--blue)}} .workload-row strong {{font-size:11px;text-align:right}} .workload-player {{display:flex;align-items:center;gap:5px;min-width:0}} .sub-badge {{padding:1px 4px;border-radius:3px;font-size:9px;font-weight:750;white-space:nowrap}} .sub-badge.entered {{background:#dff3e6;color:#17653b}} .sub-badge.exited {{background:#fae1e4;color:#9d2935}}
.event-grid {{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 14px;background:var(--white);border:1px solid var(--line);padding:7px 10px}} .event-bar {{display:grid;grid-template-columns:130px minmax(60px,1fr) 42px;gap:7px;align-items:center;min-height:26px}} .event-bar>div {{height:7px;background:#e1e6e2}} .event-bar i {{display:block;height:100%;min-width:2px;background:var(--gold)}} .event-bar strong {{text-align:right}} .empty {{color:var(--muted);padding:14px 0}} .network-controls {{display:grid;grid-template-columns:auto 1fr;gap:10px;align-items:center;padding:2px 2px 10px}} .network-play {{border:1px solid var(--line);background:var(--white);color:var(--green);font-weight:700;border-radius:3px;padding:5px 12px;cursor:pointer}} .network-time {{width:100%;accent-color:var(--green)}} .network-hover {{min-height:25px;padding:1px 3px 8px;color:var(--muted);font-size:11px}} .network-node {{cursor:pointer;outline:none}} .network-node:focus circle,.network-node:hover circle {{stroke:#ffd166;stroke-width:3px}}
.shot-marker {{cursor:help;outline:none}} .shot-marker:focus>circle,.shot-marker:hover>circle,.shot-marker:focus>path,.shot-marker:hover>path {{stroke:#ffd166;stroke-width:3px}} .shot-history {{visibility:hidden;opacity:0;pointer-events:none;transition:opacity .12s ease}} .shot-marker:focus .shot-history,.shot-marker:hover .shot-history {{visibility:visible;opacity:1}}
.shot-route-shadow {{stroke:#07111d;stroke-width:7;stroke-linecap:round;opacity:.8}} .shot-route-segment {{stroke:#fff;stroke-width:2.5;stroke-linecap:round;opacity:.96}} .shot-route-track-shadow {{fill:none;stroke:#07111d;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;opacity:.86}} .shot-route-track {{fill:none;stroke-width:3.4;stroke-linecap:round;stroke-linejoin:round;opacity:1}} .shot-route-lookback {{stroke-dasharray:7 5}} .shot-route-node {{stroke:#fff;stroke-width:1.6}} .shot-route-glyph {{font:800 8px system-ui,sans-serif;fill:#07111d;text-anchor:middle;pointer-events:none}} .shot-route-label,.shot-route-context {{font:700 10px system-ui,sans-serif;fill:#fff;paint-order:stroke;stroke:#07111d;stroke-width:3px;stroke-linejoin:round}} .shot-route-context {{font-size:11px;fill:#ffe082}} .shot-route-note-bg {{fill:#07111d;stroke:#fff;stroke-width:1.2;opacity:.92}} .shot-route-note {{font:700 11px system-ui,sans-serif;fill:#fff;text-anchor:middle}} .context-label {{font:11px system-ui,sans-serif;fill:#26332c}} .context-value {{font:10px ui-monospace,monospace;fill:#58655e}}
details {{margin-top:28px}} summary {{cursor:pointer;font-weight:700;padding:12px 0}} table {{width:100%;border-collapse:collapse;background:var(--white);border:1px solid var(--line)}} th,td {{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}} th:first-child,td:first-child,td:nth-child(2) {{text-align:left}} thead th {{background:#e9eeea;color:#415047;font-size:11px;text-transform:uppercase}} .team-dot {{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}} .t0 {{background:var(--red)}} .t1 {{background:var(--blue)}}
.foot {{margin-top:28px;color:var(--muted);font-size:12px;display:flex;justify-content:space-between;gap:20px}} a {{color:var(--green)}}
@media(max-width:1100px) {{.viz-grid.three,.benchmark-grid {{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
@media(max-width:900px) {{.viz-grid,.viz-grid.three,.dashboard-pair {{grid-template-columns:1fr}} .mast {{align-items:start}} .timeline {{max-height:320px}}}}
@media(max-width:650px) {{.wrap {{width:min(100% - 20px,1240px)}} .mast,.section-head {{display:block}} .status {{text-align:left;margin-top:10px}} .section-head p {{margin-top:3px}} .facts {{grid-template-columns:repeat(2,1fr)}} .workload-grid {{grid-template-columns:1fr}} .fact:nth-child(2) {{border-right:0}} .compare-row {{grid-template-columns:112px 48px 1fr 1fr 48px;gap:5px}} .timeline-row {{grid-template-columns:48px 1fr}} .seek {{display:none}} .viz {{padding:9px}} .viz figcaption {{display:block}} .viz figcaption span {{display:block;margin-top:2px;text-align:left}} th,td {{padding:7px 6px;font-size:11px}}}}
@media(max-width:900px) {{.benchmark-grid {{grid-template-columns:1fr}}}}
@media(max-width:650px) {{.benchmark-head {{align-items:start}} .benchmark-values b {{font-size:15px}} .benchmark-foot,.reference-note {{grid-template-columns:1fr}} .benchmark-legend {{margin-top:8px;flex-wrap:wrap}}}}
</style>
</head>
<body>
<header><div class="wrap mast"><div><h3>FootballWorld analysis</h3><h1>Match report</h1></div><div class="status"><div class="score">Team 0&nbsp; {score[0]} : {score[1]} &nbsp;Team 1</div><div>{html.escape(status)} | {html.escape(provenance_label)}</div></div></div></header>
<main class="wrap">
{warning_band}
{overview}
<section><div class="section-head"><h2>Realized shot outcomes</h2><p>Tap, hover, or focus a shot for its on-pitch attack-start/action progression; context is an explicit report heuristic</p></div><div class="viz-grid three">{_shot_map_chart(report, 0)}{_shot_map_chart(report, 1)}{_shot_context_chart(report)}</div></section>
<section><div class="section-head"><h2>Interactive cumulative passing networks</h2><p>Shared trunks show total completed passes; independently capped arrowheads show each direction; shots grow faster; hover nodes or connections for detail</p></div>{_passing_network_charts(report)}</section>
<section><div class="section-head"><h2>Realized pass maps</h2><p>Actual next-contact paths, policy-intended receiver locations, and submitted shot directions; both teams attack right</p></div><div class="viz-grid">{_pass_map_chart(report, 0)}{_pass_map_chart(report, 1)}</div></section>
<section><div class="section-head"><h2>Movement and exact events</h2><p>Active on-pitch tracking samples and confirmed event counts</p></div><div class="dashboard-pair"><div>{_speed_histogram(report)}</div><div class="event-grid">{_event_bars(report)}</div></div></section>
<section><div class="section-head"><h2>Team comparison</h2><p><span class="team-dot t0"></span>Team 0 &nbsp;&nbsp; <span class="team-dot t1"></span>Team 1</p></div><div class="team-comparison">{_comparison_rows(report)}</div></section>
<section><div class="section-head"><h2>Pass completion over time</h2><p>15-minute realized open-play pass windows; completion means same-team next distinct contact</p></div>{_pass_completion_timeline(report)}</section>
{reference_section}
<section><div class="section-head"><h2>Player workload</h2><p>Distance accumulated while identity-continuous, active, and on pitch</p></div><div class="workload-grid">{_workload_charts(report)}</div></section>
<details><summary>Detailed metrics and match timeline</summary>
<h3>Match timeline</h3><div class="panel timeline">{_timeline_rows(report)}</div>
<h3>Team metrics</h3><table><thead><tr><th>Metric</th><th><span class="team-dot t0"></span>Team 0</th><th><span class="team-dot t1"></span>Team 1</th></tr></thead><tbody>{_team_rows(report)}</tbody></table>
<h3>Player metrics</h3><table><thead><tr><th>Team</th><th>Player ID</th><th>Generation</th><th>Minutes</th><th>Distance km</th><th>Max m/s</th></tr></thead><tbody>{_player_rows(report)}</tbody></table>
</details>
<section class="control-space-final"><div class="section-head"><h2>Match control and space</h2><p>Cumulative possession, time-resolved ball density, and smoothed player occupancy from kickoff</p></div><div class="viz-grid three">
{_series_chart(report, title="Cumulative controlled possession", metric="controlled_possession_share", unit="%", percent=True)}
{_ball_density_chart(report)}
{_space_occupancy_chart(report)}
</div></section>
<div class="foot"><span>Derived metrics schema: {html.escape(report["metrics_schema"])}. Raw tracking and event files remain the source of truth.</span><a href="{json_href}" download="footballworld-report.json">Open machine-readable report</a></div>
</main>
</body></html>"""


def write_match_report(
    dataset: MatchDataset,
    output_dir: str | Path,
    *,
    policy_reference: str | Path | None = None,
) -> tuple[Path, Path]:
    """Build metrics once, then atomically replace the JSON and HTML products."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    report = build_match_report(dataset)
    if policy_reference is not None:
        attach_policy_reference(report, load_policy_reference(policy_reference))
    json_path = target / "report.json"
    html_path = target / "report.html"
    lock = FileLock(str(target / ".report.lock"))
    with lock:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target,
            prefix=".report-json-",
            delete=False,
        ) as handle:
            json_tmp = Path(handle.name)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target,
            prefix=".report-html-",
            delete=False,
        ) as handle:
            html_tmp = Path(handle.name)
        try:
            json_tmp.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            html_tmp.write_text(
                render_html(dataset, report, output_dir=target), encoding="utf-8"
            )
            json_tmp.replace(json_path)
            html_tmp.replace(html_path)
        finally:
            json_tmp.unlink(missing_ok=True)
            html_tmp.unlink(missing_ok=True)
    return html_path, json_path


__all__ = ["render_html", "write_match_report"]
