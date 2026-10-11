# Match artifacts and rendering

`messi.rendering` presents a recorded match. The report data and its presentation are separate so the same `MatchReport` can be saved as JSON or Markdown, shown in standalone HTML, or supplied to the MP4 renderer without rerunning the match.

Install the optional MP4 dependencies with `python -m pip install -e '.[render]'`. JSON, Markdown, and HTML output do not need these extras. On a shared server, select a specific GPU with `CUDA_VISIBLE_DEVICES=0` or use `JAX_PLATFORMS=cpu` before starting Python.

## Record and export

```python
import jax
import messi
from messi.analysis import make_rule_policy_recording_chunk
from messi.rendering import record_and_export_match

reset_key, match_key = jax.random.split(jax.random.key(0))
episode = messi.reset_environment(reset_key)
kernel = make_rule_policy_recording_chunk(512)
recording, paths = record_and_export_match(
    kernel, episode.state, episode.roster, match_key, episode.setup,
    maximum_steps=1_000,
    root_directory="output",
    title="Example match",
)
print(recording.report.final_score, paths.report_html)
```

This writes `match-report.html` under `output/Example-match/` and puts `match-report.json`, `match-report.md`, and `artifacts.json` in its `artifact/` subdirectory. An optional match MP4 is also saved in `artifact/`. The output directory is ignored by Git. The standalone HTML shows both starting formations and realized player attributes from `roster_summary`, along with pre-rendered match charts. The formation diagram uses reference-pitch anchors; its player attributes are model inputs rather than observed player ratings. The bar-shaped passing-network timeline works in viewers that do not run JavaScript, sampling the network every minute; JavaScript enables frame-level scrubbing and the player heatmap layer selector. `maximum_steps` caps the capture; increase it for a full match and check the recording's termination reason before treating the result as complete.

For a draggable seek bar in a viewer without JavaScript, install `.[render]`, call `render_passing_network_video(report, path, team=0)` and `render_passing_network_video(report, path, team=1)`, then pass both paths as `network_videos=(team0_path, team1_path)` to `render_match_report_html`. Save the source MP4s in the report's `artifact/` directory. The videos are embedded in the standalone HTML. When videos are included, the HTML omits the redundant static network snapshots; reports without videos retain the minute-by-minute static timeline. One video second represents one match minute, with four video frames per second. This optional report preview is separate from the full-match MP4 renderer and increases the HTML file size.

Passing-network nodes identify a player by roster slot and generation. After a substitution, the incoming player's node starts with no pass or shot history; the outgoing player's node keeps its earlier history. Node positions average pass contacts and shot releases for that player generation. Completed passes enter the network at the terminal contact frame. Valid shot attempts with an observed terminal position appear as orange links from the shooter node to that position at the terminal frame; clearance releases and shots without an observed terminal position do not create a link. The interactive chart, static timeline, and optional preview videos use the same identity rule.

The realized pass map places an arrow halfway from each recorded release point to its terminal point. Solid lines mark completed passes and dashed lines mark uncompleted passes. Gold marks an elevated-height proxy when the recorded vertical release speed is at least 4 m/s or either recorded endpoint is at least 1 m high; otherwise the pass uses its team color. Missing vertical observations are gray. These categories do not reconstruct a flight path or measure its maximum height.

Set `include_video=True` to also write `match.mp4`. Video capture retains all recorded states, actions, and event receipts on the host; report-only capture reduces each chunk and discards its dense trajectory. This makes report-only capture preferable for long matches when a replay is unnecessary. MP4 output uses the report's control interval and pitch dimensions.

`export_match_artifacts(recording, root_directory, title, ...)` saves an already recorded match. Pass a corresponding trajectory and roster to include a video. If the match used a custom half-time setting, also pass `halftime_seconds`. `render_match_report_html(report, path)` and `render_match_video(trajectory, roster, path, ...)` remain available separately.

The video renderer includes a Poisson win-probability display calibrated on earlier training matches. It has not been recalibrated for this release and should be treated as a visual estimate.
