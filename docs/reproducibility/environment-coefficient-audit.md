# FootballWorld environment coefficient review

Date: 2026-09-03

## Verdict and evidence policy

Scope: the public runtime defaults and hidden numerical constants reviewed for
this development snapshot. The rows form a reviewed evidence ledger, not an
exhaustive coverage certificate. Newly surfaced fields remain `DEFER` until
they receive an estimand- and equation-matched review.

**CHANGE: 0.** No current value should be changed in isolation.
`KEEP` means that a law/physical anchor or an intentional numerical/interface
contract is adequate for this release. `DEFER` means retain the current value
but do not call it validated and do not tune it outside its coupled model.

Evidence classes are `LAW`, `PHYSICAL`, `PAPER_FIT`,
`DATA_VALIDATED`, `DESIGN_PRIOR`, and `NUMERICAL`. “Law profile” means a
lawful selected default, not the only value permitted by the Laws.
“Paper-anchored design” means a paper motivates a range or shape but did not
fit the exact runtime scalar.

No local receipt presently identifies a fitted coefficient for the current
FootballWorld ball/contact equation. `BallPhysics.bounce_h_keep=0.732` is a
compatibility prior inherited from an older FootballWorld reconstruction whose
named held-out artifact is not present locally and whose ball equation predates
the current swept-event, spin/slip, and rolling implementation. The four
post-foul discipline fields inherit a conditional-card aggregate: three remain
narrow conditional priors, while the elapsed-time coefficient is rejected as a
runtime mapping because the provider wall clock and FootballWorld live clock
are different estimands. None identifies foul occurrence.

Scope exclusion: `runtime.enable_compilation_cache(max_bytes=2 * 1024**3,
min_compile_seconds=1.0)` is operational compilation-cache housekeeping. It
does not alter environment state-transition, action, observation, physics, or
rule semantics, so its two thresholds are not environment coefficients and
are excluded from the decision inventory. The same exclusion applies to
`MAX_ROLLING_TABLE_KNOTS=256`, nonzero float32-underflow rejection, the
coupled float32 arithmetic envelope, and the stadium-edge ULP checks in
`environment/validation.py`: these are host resource/numerical-validity
guards that reject unrepresentable public configurations without selecting a
physical model value.

`SubstitutionRequest` fields are per-event dynamic inputs, not configuration
defaults. Its incoming profile values replace one roster slot and are checked
against the same profile domain; `incoming_yellow_cards` carries current
disciplinary/law state. Neither the request profile nor yellow-card state is an
environment coefficient, so they are excluded from both the public-default and
decision inventories.

The substitution kernel reuses `YELLOW_CARD_SEND_OFF_COUNT` for the
incoming-card validity boundary, keeping one law-state SSOT without adding or
changing a coefficient decision.

The public control transition canonicalizes `IntentAction` through its
categorical and finite float32 box contracts before physics, and contact
resolution retains current possession only for the already verified physical
carrier or a coherent goalkeeper hold. These are input-boundary and
authoritative-state consistency fixes: they add no action dimension, threshold,
config default, or numeric model coefficient, so they do not add decision rows.

## Public config inventory

Every defaulted public config scalar/boolean is listed. Tuple leaves are
expanded: both initial-position coordinates and all 20 roll-table values count
separately. Required player IDs/profiles have no defaults.

### ActionScale (13)

| Field | Default | Class | Decision | Assessment |
|---|---:|---|---|---|
| `kick_speed_max_mps` | 34.76 | DESIGN_PRIOR | DEFER | Legacy compatibility cap; fit jointly with launch, spin, and reach. |
| `throw_speed_max_mps` | 21.5 | DESIGN_PRIOR | DEFER | Legacy release anchor; couple to release height and throw reconstruction. |
| `control_request_speed_max_mps` | 9.0 | DESIGN_PRIOR | KEEP | Action-taxonomy boundary, not incoming-ball controllability. |
| `launch_max_radians` | 1.0 | DESIGN_PRIOR | DEFER | Data-informed envelope without a complete current receipt. |
| `ground_launch_down_max_radians` | 0.21 | DESIGN_PRIOR | DEFER | Height-coupled launch envelope. |
| `ground_launch_down_reference_height_m` | 1.0 | DESIGN_PRIOR | DEFER | Interpolation design point. |
| `pelvis_height_factor` | 0.55 | DESIGN_PRIOR | KEEP | Effective mechanism boundary, not literal anatomy. |
| `header_speed_retention` | 0.90 | DESIGN_PRIOR | DEFER | Redirect/damping prior; header adds no energy. |
| `chest_speed_retention` | 0.10 | DESIGN_PRIOR | DEFER | Trap-response prior. |
| `challenge_speed_retention` | 1.0 | DESIGN_PRIOR | KEEP | Explicit energy-neutral redirect upper bound. |
| `spin_max_radps` | 150.0 | DESIGN_PRIOR | KEEP | Evidence-bounded rounded envelope; published contexts reach about 146–147 rad/s, but 150 is not a paper fit. |
| `restart_min_ball_speed_mps` | 0.5 | DESIGN_PRIOR | KEEP | Quantifies “clearly moves”; not a typical restart speed. |
| `throw_release_height_addition_m` | 0.47 | DESIGN_PRIOR | DEFER | Stature, source-coordinate bias, and release localization are confounded. |

### BallPhysics (44 numeric leaves)

The airborne mechanism cites *Trajectory analysis of a soccer ball*, *Soccer
ball lift coefficients via trajectory analysis*, and *Investigations into
soccer aerodynamics via trajectory analysis and dust experiments*. The 2010
study supplies the direct non-spinning and spin-dependent drag fits and the
measured lift regime; the 2009 and 2012 studies supply trajectory and flow
context. None of these papers supplies evidence for turf friction, rolling, or
bounce coefficients.

| Field | Default | Class | Decision | Assessment |
|---|---:|---|---|---|
| `g` | 9.81 | PHYSICAL | KEEP | Near-surface gravity. |
| `ground_settle_vz` | 0.5 | NUMERICAL | KEEP | Bounce/settle hybrid-event boundary. |
| `ball_mass_kg` | 0.424 | PHYSICAL | KEEP | Lawful regulation-ball profile. |
| `air_density_kgpm3` | 1.2 | PHYSICAL | KEEP | Standard-atmosphere profile; venue variability should be explicit. |
| `drag_coefficient_high_re` | 0.155 | PAPER_FIT | KEEP | Goff/Carré non-spinning drag fit. |
| `drag_crisis_drop` | 0.346 | PAPER_FIT | KEEP | Same fit. |
| `drag_crisis_speed_mps` | 12.19 | PAPER_FIT | KEEP | Same transition fit. |
| `drag_crisis_width_mps` | 1.309 | PAPER_FIT | KEEP | Same transition width. |
| `spin_drag_scale` | 0.4127 | PAPER_FIT | KEEP | Spin-dependent drag fit. |
| `spin_drag_exponent` | 0.3056 | PAPER_FIT | KEEP | Spin-parameter exponent. |
| `spin_drag_min_parameter` | 0.05 | DESIGN_PRIOR | DEFER | Branch floor; validate continuity and trajectory tails. |
| `lift_coefficient_limit` | 0.42 | DESIGN_PRIOR | KEEP | Paper-anchored bounded surrogate used as `0.42*tanh(Sp/0.42)`, not an exact fit of that formula. |
| `air_spin_decay` | 0.075 | DESIGN_PRIOR | DEFER | Not independently identified by the cited flight papers. |
| `c_ground_curl` | 0.00506 | DESIGN_PRIOR | DEFER | Fit only with rolling and spin-decay group. |
| `ground_slide_friction` | 0.35 | PHYSICAL | DEFER | Effective venue-dependent turf friction. |
| `ground_spin_decay` | 0.31 | DESIGN_PRIOR | DEFER | Fit with roll/curl group. |
| `e_rest` | 0.773 | PHYSICAL | DEFER | External estimate is impact-timing sensitive; validate a bounded speed response and timebase. |
| `ball_inertia_ratio` | 0.667 | PHYSICAL | DEFER | Ideal thin-shell approximation; the layered real ball was not identified. |
| `bounce_tangential_e` | 0.0 | DESIGN_PRIOR | DEFER | Sticking-impulse convention; requires spin-observed impacts. |
| `bounce_spin_vmin` | 0.7 | NUMERICAL | KEEP | Excludes low-amplitude ground vibration from impulse coupling. |
| `bounce_h_keep` | 0.732 | DESIGN_PRIOR | DEFER | FootballWorld compatibility prior; refit the complete bounce group against the current equations. |
| `goal_frame_radius` | 0.06 | DESIGN_PRIOR | KEEP | Law-constrained circular profile selecting the 0.12 m maximum frame width/depth. |
| `goal_frame_e_rest` | 0.68 | PHYSICAL | DEFER | Explicit unfitted hollow-frame prior. |
| `goal_frame_mu` | 0.35 | PHYSICAL | DEFER | Explicit unfitted frame/ball friction prior. |
| `roll_v_knots[0:10]` | 0, 2, 4, 6, 9, 12, 16, 20, 26, 40 | DESIGN_PRIOR | DEFER | Ten speed knots. |
| `roll_d_knots[0:10]` | 0.70, 0.95, 1.09, 1.70, 6.04, 7.85, 9.72, 13.29, 17.39, 26.96 | DESIGN_PRIOR | DEFER | Ten deceleration knots; slow roll exists, but no checked-in field receipt validates the table. |

The cited flight model computes `Sp=r*|omega_perp|/|v|`, drag, bounded lift,
and per-substep rotation. Scalars from a different aerodynamic formulation are
not drop-in replacements. The cited flight papers do not validate turf,
bounce, or rolling values.

### BodyContact (8)

| Field | Default | Class | Decision | Assessment |
|---|---:|---|---|---|
| `shoulder_width_m` | 0.50 | DESIGN_PRIOR | KEEP | Effective torso-capsule width used by collision, separation, offside, and restarts. |
| `torso_depth_m` | 0.20 | DESIGN_PRIOR | KEEP | With width: 0.30 m shoulder core plus 0.10 m end-cap radius. |
| `leg_apex_height_factor` | 0.50 | DESIGN_PRIOR | DEFER | Fixed isosceles-stance apex and torso boundary as a fraction of stature; fit with pose/contact data. |
| `leg_radius_m` | 0.05 | DESIGN_PRIOR | DEFER | Effective radius of each diagonal leg capsule; no gait phase is represented. |
| `foot_forward_extent_m` | 0.30 | DESIGN_PRIOR | DEFER | Symmetric playable-foot gait envelope used only for Law 11 support, not collision or control reach. |
| `torso_top_height_factor` | 0.85 | DESIGN_PRIOR | KEEP | Semantic chest/head boundary. |
| `head_radius_m` | 0.10 | DESIGN_PRIOR | DEFER | Effective collision sphere. |
| `restitution` | 0.50 | PHYSICAL | DEFER | Fit jointly with geometry, trapping, and pre/post velocity. |

### ContactTiming (6)

| Field | Default | Class | Decision | Assessment |
|---|---:|---|---|---|
| `challenge_recovery_s` | 0.16 | DESIGN_PRIOR | DEFER | Challenge-recovery proxy; generic touch gaps do not identify it. |
| `max_lunge_extra_recovery_s` | 0.84 | DESIGN_PRIOR | KEEP | Explicit anti-repeat-tackle prior: up to 1.0 s with base. |
| `active_contact_interval_s` | 1/15 | NUMERICAL | KEEP | Physical-time debounce and intended contact-rate ceiling. |
| `aerial_attempt_recovery_s` | 0.5 | DESIGN_PRIOR | DEFER | Maximum full-jump recovery; reachable attempts interpolate from standing stature to maximum reach, but selected attempts are not provider-labelled. |
| `goalkeeper_dive_recovery_s` | 1.0 | DESIGN_PRIOR | DEFER | Maximum full-extension dive recovery; current tracking does not identify attempt posture or recovery time. |
| `possession_loss_lock_s` | 0.16 | DESIGN_PRIOR | DEFER | Causal old-team challenge gate, not generic retouch time. |

### Contest (15)

| Field | Default | Class | Decision | Assessment |
|---|---:|---|---|---|
| `distance_weight` | 1.0 | DESIGN_PRIOR | DEFER | Coupled score coordinate. |
| `reach_time_weight` | 1.0 | DESIGN_PRIOR | DEFER | Collinear with distance under uniform-speed rosters. |
| `height_fit_weight` | 0.55 | DESIGN_PRIOR | DEFER | Non-discriminating under uniform reach. |
| `possession_weight` | 0.0 | DESIGN_PRIOR | KEEP | Deliberately neutral, avoiding unaudited bias. |
| `ball_control_weight` | 0.3 | DESIGN_PRIOR | DEFER | Identifiable only with heterogeneous ability. |
| `temperature` | 0.66 | DESIGN_PRIOR | DEFER | Fit with all score weights. |
| `tackle_success_probability` | 0.28 | DESIGN_PRIOR | DEFER | External duel units differ from simulator-eligible attempts. |
| `tackle_foul_probability` | 0.04 | DESIGN_PRIOR | DEFER | Non-zero compatibility prior without a runtime-matched opportunity denominator. |
| `tackle_deflection_probability` | 0.0 | DESIGN_PRIOR | KEEP | Physical swept collisions remain authoritative. |
| `card_probability_midpoint` | 0.1297 | DESIGN_PRIOR | DEFER | Inherited conditional prior applied only after a verified carrier challenge; not a foul-occurrence rate. |
| `card_attack_progress_logit_weight` | -1.2860 | DESIGN_PRIOR | DEFER | Inherited conditional prior whose source opportunity population differs from the runtime population. |
| `card_elapsed_fraction_logit_weight` | 0.8028 | DESIGN_PRIOR | DEFER | Source and runtime clock definitions differ; retain only as a transfer prior. |
| `direct_red_given_card_probability` | 0.03125 | DESIGN_PRIOR | DEFER | Sparse conditional aggregate prior without a contextual red model. |
| `goalkeeper_catch_speed_midpoint_mps` | 21.3 | DESIGN_PRIOR | DEFER | Selection and ball-response evidence are mixed. |
| `goalkeeper_catch_speed_scale_mps` | 9.9 | DESIGN_PRIOR | DEFER | Fit jointly with midpoint from 3-D opportunities including misses. |

### Geometry (12)

| Field | Default | Class | Decision | Assessment |
|---|---:|---|---|---|
| `Ball.radius` | 0.11 | PHYSICAL | KEEP | Effective lawful regulation-ball radius. |
| `Stadium.width` | 68.0 | LAW PROFILE | KEEP | Common lawful pitch profile, not the only permitted width. |
| `Stadium.length` | 105.0 | LAW PROFILE | KEEP | Common lawful pitch profile, not the only permitted length. |
| `goal_width` | 7.32 | LAW | KEEP | Inside-post width. |
| `goal_height` | 2.44 | LAW | KEEP | Ground-to-crossbar clearance. |
| `penalty_area_length` | 16.5 | LAW | KEEP | Canonical field marking. |
| `penalty_area_width` | 40.32 | LAW | KEEP | Derived from the two 16.5 m offsets plus goal width. |
| `goal_area_length` | 5.5 | LAW | KEEP | Canonical field marking. |
| `goal_area_width` | 18.32 | LAW | KEEP | Derived from the two 5.5 m offsets plus goal width. |
| `center_circle_radius` | 9.15 | LAW | KEEP | Canonical field marking. |
| `penalty_arc_radius` | 9.15 | LAW | KEEP | Lawful marking; currently has no runtime caller. |
| `corner_arc_radius` | 1.0 | LAW | KEEP | Canonical corner-area radius. |

### GoalkeeperHolding (1)

| Field | Default | Class | Decision | Assessment |
|---|---:|---|---|---|
| `hand_control_limit_s` | 8.0 | LAW | KEEP | Exact current Law 12.3 ceiling; opponent corner on expiry. |

### RestartTiming (1)

| Field | Default | Class | Decision | Assessment |
|---|---:|---|---|---|
| `forced_release_delay_s` | 3.0 | DESIGN_PRIOR | KEEP | Intentional benchmark dead-ball and goalkeeper-distribution delay; the environment, not the policy, forces release. |

### Perception (2)

| Field | Default | Class | Decision | Assessment |
|---|---:|---|---|---|
| `limit_by_view_angle` | false | DESIGN_PRIOR | KEEP | Full-state default; observation only. |
| `horizontal_fov_degrees` | 160.0 | DESIGN_PRIOR | KEEP | Optional learning-task view, not a biological estimate. |

### ManagementRules (2)

| Field | Default | Class | Decision | Assessment |
|---|---:|---|---|---|
| `max_substitutions_per_team` | 5 | LAW PROFILE | KEEP | Common competition profile; callers must override it when competition rules differ. |
| `max_windows_per_team` | 3 | LAW PROFILE | KEEP | Common regulation-time opportunity profile; repeated substitutions at one control tick share a window. |

### PlayerPhysics (4)

| Field | Default | Class | Decision | Assessment |
|---|---:|---|---|---|
| `forward_acceleration_mps2` | 7.95 | DESIGN_PRIOR | DEFER | Validate jointly with the lateral and braking axes before changing. |
| `lateral_acceleration_mps2` | 8.46 | DESIGN_PRIOR | DEFER | External windowed acceleration is not the instantaneous solver cap. |
| `braking_deceleration_mps2` | 10.51 | DESIGN_PRIOR | DEFER | Validate all three movement axes together. |
| `separation_iterations` | 3 | NUMERICAL | KEEP | Bounded fixed-iteration operational choice; not a measured football coefficient. |

### Reach (6)

| Field | Default | Class | Decision | Assessment |
|---|---:|---|---|---|
| `carry_radius_m` | 1.1 | DESIGN_PRIOR | DEFER | Effective control envelope, not anatomy. |
| `challenge_radius_m` | 1.4 | DESIGN_PRIOR | DEFER | Coupled to recovery and foul exposure. |
| `goalkeeper_standing_radius_m` | 1.1 | DESIGN_PRIOR | DEFER | Standing hand-contact baseline used only to infer minimum dive effort; initialized from the existing close-control envelope, not measured anatomy. |
| `goalkeeper_radius_m` | 2.0 | DESIGN_PRIOR | DEFER | Effective dive/hand cover; also used by the backpass target proxy. |
| `block_speed_limit_mps` | 31.0 | DESIGN_PRIOR | DEFER | Event-preserving eligibility boundary, not a unique human limit. |
| `height_speed_penalty_mps_per_m` | 3.5 | DESIGN_PRIOR | DEFER | Coupled speed/height reach slope. |

FootballWorld retains FootballWorld's useful separation between an active-only
aerial recovery lock and passive swept body contact: a recovering player still
has a solid body. It also retains the player-specific maximum vertical reach
and the goalkeeper's larger lawful hand envelope. The fixed 0.5 second lock for
every head-height opportunity is rejected because a standing header and a
maximum jump did not imply the same recovery. Likewise, the 2.0 metre keeper
envelope alone could not distinguish a standing claim from a full dive.
FootballWorld now derives vertical effort from ball-bottom height between
stature and maximum reach, and goalkeeper horizontal effort between standing
and maximum radii, at the exact swept-entry state. The larger effort controls
the one existing aerial-recovery timer. These are causal geometry semantics;
the three default endpoints remain `DEFER` pending 3-D opportunities that
include attempted misses and post-attempt recovery.

### Roster defaults (8)

| Field | Default | Class | Decision | Assessment |
|---|---:|---|---|---|
| `PlayerProfile.max_speed_mps` | 7.96 | DESIGN_PRIOR | DEFER | Population/action-coordinate fallback, not individual capacity. |
| `height_m` | 1.8 | DESIGN_PRIOR | KEEP | Population fallback; no authoritative local stature join. |
| `max_reach_height_m` | 2.7 | DESIGN_PRIOR | DEFER | Effective posture/jump/limb reach. |
| `ball_control` | 0.5 | DESIGN_PRIOR | KEEP | Latent-coordinate anchor, not 50% success. |
| `endurance_factor` | 1.0 | DESIGN_PRIOR | KEEP | Neutral normalized anchor, confounded with drain/recovery scale. |
| `is_goalkeeper` | false | DESIGN_PRIOR | KEEP | Role default; public reset requires exactly one GK/team. |
| `Player.initial_position[0]` | 0.0 | DESIGN_PRIOR | KEEP | Constructor fallback; multi-player all-zero setup is invalid. |
| `Player.initial_position[1]` | 0.0 | DESIGN_PRIOR | KEEP | Same. |

### LongStamina (12)

| Field | Default | Class | Decision | Assessment |
|---|---:|---|---|---|
| `sprint_speed` | 5.5 | DESIGN_PRIOR | DEFER | Workload-coordinate threshold. |
| `sprint_mult` | 3.0 | DESIGN_PRIOR | DEFER | Extra-load slope. |
| `idle_load` | 0.12 | DESIGN_PRIOR | DEFER | Basal on-pitch load. |
| `speed_ref` | 2.4 | DESIGN_PRIOR | DEFER | Speed-load normalizer. |
| `speed_load` | 0.90 | DESIGN_PRIOR | DEFER | Do not lower merely to repair in-play fraction. |
| `accel_ref` | 4.0 | DESIGN_PRIOR | DEFER | Acceleration normalizer. |
| `accel_load` | 0.10 | DESIGN_PRIOR | DEFER | Squared acceleration weight. |
| `vmax_floor` | 0.99 | DESIGN_PRIOR | KEEP | Intentional small sustained-speed collapse; stratified data do not support a larger decline. |
| `end_frac` | 0.05 | DESIGN_PRIOR | KEEP | Latent endpoint, not measured “5% stamina.” |
| `tail_knee` | 0.20 | DESIGN_PRIOR | KEEP | Information-preserving latent tail. |
| `reference_duration_s` | 5400.0 | DESIGN_PRIOR | KEEP | Law-anchored 90-minute workload horizon; its use as normalization is a model choice. |
| `reference_workload` | 0.82 | DESIGN_PRIOR | DEFER | Mixed-population anchor; field-player 0.8386 alone does not justify replacement. |

### ShortStamina (10)

| Field | Default | Class | Decision | Assessment |
|---|---:|---|---|---|
| `vmax_floor` | 0.70 | DESIGN_PRIOR | DEFER | Fit within the entire repeated-sprint response. |
| `headroom_knee` | 0.25 | DESIGN_PRIOR | DEFER | Coupled observable speed-cap knee. |
| `depletion_s` | 10.0 | DESIGN_PRIOR | DEFER | Coupled to intensity saturation and endurance. |
| `depletion_speed_frac` | 0.70 | DESIGN_PRIOR | DEFER | Only about 1.95% of audited frames entered this region. |
| `speed_exponent` | 2.0 | DESIGN_PRIOR | DEFER | Shape is not independently identified. |
| `accel_ref` | 6.0 | DESIGN_PRIOR | DEFER | Coupled positive-acceleration normalizer. |
| `accel_load` | 0.20 | DESIGN_PRIOR | DEFER | Coupled contribution weight. |
| `recovery_tau_s` | 60.0 | DESIGN_PRIOR | DEFER | Hidden-store time constant, not observed time until speed cap clears. |
| `recovery_speed_frac` | 0.70 | DESIGN_PRIOR | DEFER | Recovery opportunity, dead ball, and tactics are confounded. |
| `recovery_exponent` | 1.0 | DESIGN_PRIOR | DEFER | Shape is not independently identified. |

The short-stamina fields cannot be identified separately from observed
repeat-sprint gaps. The halftime recovery uses the exact resting ODE solution,
not repeated physics stepping.

## Match, timebase, and law configuration

| Surface | Value | Class | Decision | Assessment |
|---|---:|---|---|---|
| `MatchConfig.halftime_seconds` | 2700 s | LAW PROFILE | KEEP | Standard first-half boundary; callers may include observed first-half added time. |
| `MatchConfig.fulltime_seconds` | 5400 s | LAW PROFILE | KEEP | Independent full-time boundary permits asymmetric added-time replay. |
| `halftime_enabled` | true | LAW PROFILE | KEEP | Standard match mode. |
| `halftime_interval_s` | 900 s | LAW-CONSTRAINED PROFILE | KEEP | Selects Law 7 maximum; not universally mandated. |
| `minimum_team_players` | (7, 7) | LAW | KEEP | Default Law 3 threshold; explicit diagnostic range is 0..11. |
| `DEFAULT_MATCH_DURATION_SECONDS` | 5400 s | LAW PROFILE | KEEP | Stamina reference SSOT; no longer owns episode termination. |
| `DEFAULT_TIMEBASE.dt_phys` | 1/80 s | NUMERICAL DESIGN | KEEP | 80 Hz collision clock selected as a speed/accuracy trade-off; it is not a measured football constant. |
| `DEFAULT_TIMEBASE.control_fps` | 10 Hz | NUMERICAL DESIGN | KEEP | Policy clock; decimation 8 and control interval 0.1 s. It is not a measured football constant. |
| conversion `minimum` default | 0 ticks | NUMERICAL | KEEP | Zero allowed unless caller requests a positive floor. |
| integer-ratio tolerance | `max(1e-9,4*ulp)` | NUMERICAL | KEEP | Host-only representability check. |
| counter ceiling | 2^31-1 | NUMERICAL | KEEP | int32 recurrent counters. |
| `RestartSpotLaw.penalty_mark_distance_m` | 11.0 m | LAW | KEEP | Canonical penalty mark. |
| `RestartPositionLaw.ordinary_clearance_m` | 9.15 m | LAW | KEEP | Free-kick/penalty distance anchor. |
| `RestartPositionLaw.throwin_clearance_m` | 2.0 m | LAW | KEEP | Throw-in opponent distance. |
| `YELLOW_CARD_SEND_OFF_COUNT` | 2 | LAW | KEEP | Second caution sends off. |
| `IFAB_MIN_TEAM_PLAYERS` | 7 | LAW | KEEP | Match-continuation threshold. |
| `IFAB_MAX_TEAM_PLAYERS` | 11 | LAW | KEEP | Team maximum; one must be the goalkeeper. |

The 80/10 default was selected on 2026-09-07 as a numerical speed/accuracy
operating point and was not fitted from DFL, K-League, or literature data.
Every generated artifact must embed its effective `physics_fps` and
`control_fps`; historical receipts are not part of this public distribution.

Law sources checked against the 2026/27 official pages:

- Law 1 field and goal dimensions:
  <https://www.theifab.com/laws/latest/the-field-of-play/>.
- Law 3 team limits:
  <https://www.theifab.com/laws/latest/the-players/>.
- Law 7 match/interval timing:
  <https://www.theifab.com/laws/latest/the-duration-of-the-match/>.
- Law 12 eight-second goalkeeper limit:
  <https://www.theifab.com/laws/latest/fouls-and-misconduct/>.
- Laws 13–15 restart distances:
  <https://www.theifab.com/laws/latest/free-kicks/>,
  <https://www.theifab.com/laws/latest/the-penalty-kick/>,
  <https://www.theifab.com/laws/latest/the-throw-in/>.

## Hidden numeric model and interface constants

Zero/one algebra identities, indices, enum values, random-key salts, and
sentinel IDs are implementation vocabulary. The following non-public numbers
materially alter environment semantics.

### Action, numerical guards, and defaults

| Surface | Value | Class | Decision | Assessment |
|---|---:|---|---|---|
| Explicit action shape | categorical + 8 continuous | NUMERICAL | KEEP | Six-way intent stored separately from move(2) + force-to-ball(2) + launch + spin(2) + torso-relative gaze target. |
| Continuous bounds | [-1,1] | DESIGN_PRIOR | KEEP | Static normalized interface. |
| `FootballWorld.boundary_margin_m` default | 0.0 m | DESIGN_PRIOR | KEEP | The public facade passes the zero default through validation into episode/control stepping, so there is no hidden inset. |
| `GEOMETRY_EPS` | 1e-6 | NUMERICAL | KEEP | Geometry branches and tolerances. |
| `DIV_EPS` | 1e-9 | NUMERICAL | KEEP | Safe denominators. |
| `SQUARED_EPS` | 1e-12 | NUMERICAL | KEEP | Squared-motion threshold. |
| `SAFE_NORM_EPS` | 1e-18 | NUMERICAL | KEEP | Finite zero norm. |
| `STATIONARY_SPEED_EPS` | 1e-3 m/s | NUMERICAL | KEEP | Facing and zero-direction fallback. |
| `COINCIDENT_DISTANCE_EPS` | 1e-4 m | NUMERICAL | KEEP | Overlap tie-break. |
| `GOLDEN_ANGLE` | pi(3-sqrt(5)) | NUMERICAL | KEEP | Deterministic separation direction. |
| Observation coincident-view literal | 1e-6 m | NUMERICAL | KEEP | Equals `GEOMETRY_EPS` but is duplicated; deduplicate for provenance later. |

### Hidden stamina saturation

| Expression | Cap | Class | Decision | Assessment |
|---|---:|---|---|---|
| sprint excess | 2.0 | DESIGN_PRIOR | DEFER | Caps incremental long sprint load. |
| long speed intensity | 4.0 | DESIGN_PRIOR | DEFER | Extreme-speed saturation. |
| long acceleration intensity before square | 4.0 | DESIGN_PRIOR | DEFER | Allows 16x normalized contribution before configured weight. |
| short positive-acceleration intensity | 1.0 | DESIGN_PRIOR | DEFER | Not separately identified. |
| combined short load | 2.0 | DESIGN_PRIOR | DEFER | Coupled to `depletion_s` and `accel_load`. |
| sustained-limit denominator floor | 1e-6 m/s | NUMERICAL | KEEP | Division guard, not physiology. |

Any stamina calibration must include these hidden caps rather than treating
the 22 public stamina fields as the complete model.

### Fixed-shape solvers and timer conversion

| Surface | Value | Class | Decision | Assessment |
|---|---:|---|---|---|
| Swept ball-event budget | 3/substep | NUMERICAL | KEEP | Three passive-event slots; deliberate contact is a separate fourth occurrence. |
| Restart layout count | 5 | NUMERICAL | KEEP | Local, ring, penalty, corner, wall; function docstring corrected. |
| Local/canonical spacing | 0.60 m | NUMERICAL | DEFER | `0.50 + 0.5*0.20`; search spacing. |
| General/corner base radius | 10.65 m | NUMERICAL | DEFER | `9.15 + 1.00 + 0.50`; search radius. |
| General/corner ring step | 0.70 m | NUMERICAL | DEFER | `0.50 + 0.20`; search step. |
| General half-turn slots | 16 | NUMERICAL | KEEP | Static compiled angular resolution. |
| Corner quadrant slots | 8 | NUMERICAL | KEEP | Static compiled angular resolution. |
| Both slot offsets | 0.5 slot | NUMERICAL | KEEP | Avoids exact sector boundaries. |
| Reprojection/wall tolerance | 8e-6 m | NUMERICAL | KEEP | `8*GEOMETRY_EPS`; audit tolerance, not law slack. |
| Contact/aerial/loss timers | nearest substep, min 1 | NUMERICAL | KEEP | `max(1,round(seconds/dt))`. |
| Challenge recovery | nearest substep, min 1 | NUMERICAL | KEEP | `rint` after distance interpolation; not floor. |
| GK hand-control ceiling | floor complete substeps | LAW/NUMERICAL | KEEP | The following interval first produces “over eight seconds.” |

`Stadium.penalty_arc_radius` is currently unused. The penalty projector
already enforces 9.15 m from the mark; adding a second path merely to consume
the field would increase inconsistency risk.

## Reused-coefficient algorithms

### Goalkeeper backpass target proxy

No independent coefficient was added. The finite outgoing-ray proxy reuses
`goalkeeper_radius_m + Ball.radius = 2.11 m`, `g`, `e_rest`,
`bounce_h_keep`, `ground_settle_vz`, and all roll knots. Bounce count is
analytically derived from impact speed, restitution, and settle speed; there
is no configured horizon, bounce cap, intent angle, or new tolerance radius.
Rolling distance integrates `v / d(v)` across every configured piecewise-linear
deceleration interval. Reusing only the release-speed deceleration for the
whole path was unsound: fast passes enter lower-speed intervals with less
resistance, so that shortcut materially underestimated their finite range and
could fail to arm the restriction for an intended moving goalkeeper receiver.

This remains an intent proxy, not trajectory truth. Do not tune authoritative
flight/bounce/roll/GK reach to repair its labels. Audit false positives and
negatives by loft, lateral miss, intermediate team-mate, and interception.

### Observable offside challenge

The integrated challenge branch adds no numeric threshold. It consumes the
already selected, physically verified `ContestResult.challenger`; snapshots
reuse torso/head/ball geometry. Its time fraction now comes from the common
swept active/passive event scheduler rather than assuming endpoint eligibility
at time zero. The touch reducer continues to exclude non-contact involvement;
the separate resolver handles the observable selected-challenge branch.

### Opening tactical and formation selection

| Surface | Value | Class | Decision | Assessment |
|---|---:|---|---|---|
| roster-to-plan ability weights | 5 x 5 rows summing to 1 | DESIGN_PRIOR | DEFER | Relative speed/body/control/endurance compatibility for the five rule plans; not a measured causal tactic-selection model. |
| automatic plan temperature | 0.35 | DESIGN_PRIOR | DEFER | Softmax diversity control applied once after episode abilities are realized. |
| formation-fit weight | 2.0 | DESIGN_PRIOR | DEFER | Joint gain for best-XI and plan/shape compatibility in the formation softmax logit. |
| plan-to-formation weights | 5 x 5 signed table | DESIGN_PRIOR | DEFER | Ranks attack depth, width, defender share, central midfield and wide-role structure; it never makes a legal formation impossible. |

The automatic order is ability realization, roster-conditioned tactical-plan
softmax, feasible-XI evaluation for every formation, then formation sampling.
Explicit fixture abilities, plans, formations, and XIs bypass only their own
automatic decision. Gumbel-max is used as the exact categorical sampler for
the declared softmax logits, with match-key and content-addressed streams.

### Restart taker selection

| Surface | Value | Class | Decision | Assessment |
|---|---:|---|---|---|
| External role propensity matrix | 8 x 7 probabilities | DESIGN_PRIOR | DEFER | Supplied compatibility table; the kickoff row is authored rather than measured. |
| log-affinity scale | 12.0 | DESIGN_PRIOR | DEFER | Converts propensity ratios to the same score scale as distance and ability. |
| propensity floor | 1e-3 | NUMERICAL | KEEP | Keeps unobserved role/restart pairs finite without making them preferred. |
| restart distance weights | (1.0, 0.35, 0.25, 0.5, 0.4, 0.05, 0.3, 0.2) | DESIGN_PRIOR | DEFER | Kind-specific transfer prior, not an independently fitted metric coefficient. |
| long-stamina weight | 10.0 | DESIGN_PRIOR | DEFER | Readiness prior, not a measured causal fatigue effect on nomination. |
| corner aerial-target cost | 15.0 | DESIGN_PRIOR | DEFER | Preserves likely targets in the box; requires lineup-role ablation. |
| taker Gumbel temperature | 12.0 | DESIGN_PRIOR | DEFER | Inherited stochastic ranking scale; fit against repeated taker choices after conditioning on available lineups. |
| current-depth role thirds | 1/3 and 2/3 | DESIGN_PRIOR | DEFER | Lightweight proxy because immutable formation anchors are absent from rollout state. |
| wide-role ranking | outer 2 when line has >=3; 1e-3 m tie tolerance | DESIGN_PRIOR | DEFER | Formation-robust structural rule inherited in simplified form; validate on formation changes. |

The rule manager uses registered formation roles and identity-keyed Gumbel
noise; the environment fallback remains deterministic and position-derived.
The PRNG stream tag and fold order are reproducibility protocol, not football
coefficients. They isolate the draw from contest, roster, batching, and rollout
chunk order.

### Rule-manager substitution priors

| Surface | Value | Class | Decision | Assessment |
|---|---:|---|---|---|
| substitution timing anchors | 57.7, 68.3, 80.1 min | DESIGN_PRIOR | DEFER | Runtime scheduling anchors, not fitted live-clock hazards. |
| halftime substitution mode | 15.9% | DESIGN_PRIOR | DEFER | Compatibility prior whose source and runtime decision units differ. |
| per-window size probabilities | 1: 0.650, 2: 0.310, 3: 0.040 | DESIGN_PRIOR | DEFER | Runtime compatibility prior capped by command and quota eligibility. |
| role replacement propensities | GK 0.000, CB 0.049, FB 0.100, CM 0.165, WM 0.220, CF 0.221, WF 0.259 | DESIGN_PRIOR | DEFER | Ranking prior, not a lineup-conditioned outgoing-player model. |
| fatigue / booking / role weights | 1.0 / 0.25 / 0.15 | DESIGN_PRIOR | DEFER | Readable coupled ranking; stamina calibration was explicitly deferred and these must not be presented as fitted effects. |
| substitution candidate temperature | 0.30 | DESIGN_PRIOR | DEFER | Candidate-normalized identity-keyed diversity prior for routine outgoing/incoming selection; no lineup-conditioned choice fit supports its value. |

These substitution settings are runtime compatibility priors. Their source decision unit and clock do not match FootballWorld live-clock team-window decisions, so they must not be described as fitted hazards.

## Coupling and residual DEFER priorities

- Action launch, speed, spin, reach, and control boundaries form one contact
  model; do not calibrate a marginal action scalar.
- Aerodynamic fit values belong to their stated `Sp` formulation. Ground
  curl/spin/roll and bounce must be fit together with spin-observed trajectories.
- GK reach, catch midpoint/scale, and target-proxy error form one audit.
- Body geometry, body restitution, and trapping outcome form one audit.
- The three movement axes form one feasible ellipse and need one response fit.
- Contest weights and temperature require heterogeneous ability/opportunity
  data; provider duel rates are not simulator attempt rates.
- All public short-stamina fields plus hidden saturation caps require one
  repeated-sprint trajectory fit.
- Restart packing spacing/radius/step require adversarial formation fixtures
  plus compile/runtime profiling.

Immediate provenance work: serialize the complete config and an
evidence-manifest digest with every experiment, and require field-addressed,
match-held-out receipts for coefficient changes.


## Public provenance boundary

This ledger publishes runtime defaults and their evidence class so that users
can distinguish laws, physical profiles, paper fits, compatibility priors, and
unresolved design choices. Provider files, provider-specific conversion or
fitting code, derived aggregates, split manifests, and private validation
receipts are intentionally not distributed.

Only the six aerodynamic equation parameters identified as `PAPER_FIT` are
presented as fitted constants. External match-derived values remain transfer
priors unless a runtime-matched public evidence contract is supplied.
