# Changelog

All notable changes to MESSI are recorded here.

## 0.1.0 - 2026-09-09

Current research-preview baseline. This version is not a formal stable release.

### Added

- Fixed-shape JAX football state, action, observation, physics, and rules
  transitions for configurable team sizes up to full 11-vs-11 matches.
- Six categorical player intents paired with bounded continuous movement,
  contact, launch, spin, and gaze controls.
- Full and optional view-limited observations with explicit visibility and
  known-value masks.
- Separate compact and exact-event step paths, plus fixed-length rollout and
  managed-match factories.
- Seeded player, manager, opening-lineup, formation, and set-piece rule
  policies behind framework-neutral interfaces.
- Three-dimensional ball flight, bounce, rolling, spin, stamina, oriented body
  contact, challenges, goalkeeper handling, restarts, offside, fouls, cards,
  substitutions, halftime, and causally accumulated added time.
- Host-side 1080p replay rendering with lossless NPZ tracking, exact JSON event
  sidecars, integrity receipts, and bounded-memory chunk processing.
- Calibration, coefficient provenance, replay auditing, and CPU/GPU profiling
  tools kept outside the lean environment transition.
