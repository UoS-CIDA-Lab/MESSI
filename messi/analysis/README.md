# Match recording and analysis

`messi.analysis` turns a match-aware rollout into an event stream and a structured `MatchReport`. Recording is an explicit evaluation path; ordinary `advance_control` and `evaluate_control` still return only the final state.

## Bounded recording

`make_match_recording_chunk(max_steps, policy, ...)` creates a fixed-size JAX capture kernel. The policy reads the current state and roster and returns an action. `record_match` compiles and runs the kernel in chunks, transfers each bounded chunk to the host, accumulates statistics, and returns a `MatchRecording`. Its `maximum_steps` argument is a caller-specified cap; reaching it does not mean that the match ended naturally.

`make_rule_policy_recording_chunk` is the built-in policy adapter. It obtains a `PolicyObservation`, uses `rule_policy_with_diagnostics`, and records decision telemetry. Pass a tactical plan through `record_match(..., policy_context=plan)`; the default uses `DEFAULT_RULE_TACTICAL_PLAN`.

The report includes scores, possession, actions, event counts, passing and shooting outcomes, player movement, team shape, spatial occupancy, causal goal attribution, and `roster_summary`. The schema is `messi.match-report/28`. `roster_summary` captures each team's starting formation and the realized speed, height, reach, ball-control, and endurance attributes for each player generation. Starting formation describes the reset lineup, not the team's shape throughout the match. Some statistics require events and trajectory samples, so this path costs more than final-state training. `record_match_batch` provides the same aggregation for independent matches, using `vmap` on one local device and `pmap` across multiple local devices.

## Outputs

`write_match_report(report, directory, title=...)` writes `match-report.json` and `match-report.md`. The JSON contains the full structured report; the English Markdown file highlights key fields and links conceptually to the same report object. For an HTML view and optional video, use [the rendering guide](../rendering/README.md).

The DFL values in `reference_context` are descriptive reference data. Their source measure differs from MESSI's next-distinct-contact measure, so the comparison does not establish statistical equivalence.

Recording does not automatically resolve manager requests such as substitutions. If your match requires a management decision, use the management API at that boundary.
