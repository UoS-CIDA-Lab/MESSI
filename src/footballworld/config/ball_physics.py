"""Configuration for free-ball and goal-frame dynamics.

The airborne model follows the force definitions and measured soccer-ball
regimes reported by Goff and Carré:

- "Trajectory analysis of a soccer ball", American Journal of Physics 77
  (2009), 1020-1027, doi:10.1119/1.3197187.
- "Soccer ball lift coefficients via trajectory analysis", European Journal
  of Physics 31 (2010), 775-784, doi:10.1088/0143-0807/31/4/007.
- "Investigations into soccer aerodynamics via trajectory analysis and dust
  experiments", Procedia Engineering 34 (2012), 158-163,
  doi:10.1016/j.proeng.2012.04.028.

The 2010 paper supplies the non-spinning drag fit in its equation (3), the
spin-dependent drag fit in equation (4), and the observed high-spin leveling
of the lift coefficient. It also states that laminar/turbulent transitions are
not instantaneous. The runtime therefore bridges the two measured fits with
the already fitted drag-crisis width and a compact smoothstep over the lowest
measured spin interval; no additional aerodynamic coefficient is invented.
The 2009 and 2012 papers provide the trajectory and flow-visualization context
used to choose the bounded lift surrogate. In all three,
``Sp = radius * angular_speed / air_speed``. Ground-contact coefficients remain
separate because the cited experiments concern airborne trajectories, not
ball-turf friction.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BallPhysics:
    """Coefficients consumed by the free-ball integrator."""

    g: float = 9.81
    ground_settle_vz: float = 0.5

    ball_mass_kg: float = 0.424
    air_density_kgpm3: float = 1.2
    drag_coefficient_high_re: float = 0.155
    drag_crisis_drop: float = 0.346
    drag_crisis_speed_mps: float = 12.19
    drag_crisis_width_mps: float = 1.309
    spin_drag_scale: float = 0.4127
    spin_drag_exponent: float = 0.3056
    spin_drag_min_parameter: float = 0.05
    lift_coefficient_limit: float = 0.42
    air_spin_decay: float = 0.075

    c_ground_curl: float = 0.00506
    ground_slide_friction: float = 0.35
    ground_spin_decay: float = 0.31

    e_rest: float = 0.773
    ball_inertia_ratio: float = 0.667
    bounce_tangential_e: float = 0.0
    bounce_spin_vmin: float = 0.7
    bounce_h_keep: float = 0.732

    goal_frame_radius: float = 0.06
    goal_frame_e_rest: float = 0.68
    goal_frame_mu: float = 0.35

    roll_v_knots: tuple[float, ...] = (
        0.0,
        2.0,
        4.0,
        6.0,
        9.0,
        12.0,
        16.0,
        20.0,
        26.0,
        40.0,
    )
    roll_d_knots: tuple[float, ...] = (
        0.70,
        0.95,
        1.09,
        1.70,
        6.04,
        7.85,
        9.72,
        13.29,
        17.39,
        26.96,
    )
