from __future__ import annotations

from dataclasses import replace
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from jaxmarl.environments import spaces
from jaxmarl.environments.multi_agent_env import MultiAgentEnv

from constants import *
from state import State
from setup import _meta

from events import Events
from restart import Restart
from movement import Movement
from observation import Observation
from ball import BallPhysics
from contest import Contest
from fouls import Fouls
from offside import Offside
from rewards import Rewards
from inverse import Inverse
from render import Render
from config import Agent, Ball, DeadBall, Engine, Foul, Reward, Stadium


def _default_roster(start_id: int) -> list[Agent]:
    """``DEFAULT_FORMATION``으로 팀을 만들며 첫 슬롯만 GK로 둔다."""
    return [
        Agent(id=start_id + i, is_gk=(i == 0), init_pos=pos)
        for i, pos in enumerate(DEFAULT_FORMATION)
    ]


class SoccerEnv(
    Events,
    Observation,
    Movement,
    Restart,
    BallPhysics,
    Contest,
    Fouls,
    Offside,
    Rewards,
    Inverse,
    Render,
    MultiAgentEnv
):
    def __init__(
        self,
        n_agents: int = 11,
        n_opponents: int = 11,
        agent_team: list[Agent] | None = None,
        opponent_team: list[Agent] | None = None,
        game_duration: int = DEFAULT_GAME_DURATION, # 90 minutes in 25 fps
        control_fps: float = DEFAULT_CONTROL_FPS,
        *,
        ball_config: Ball | None = None,
        stadium_config: Stadium | None = None,
        deadball_config: DeadBall | None = None,
        engine_config: Engine | None = None,
        foul_config: Foul | None = None,
        reward_config: Reward | None = None,
    ):
        if not isinstance(n_agents, int) or isinstance(n_agents, bool) or n_agents <= 0:
            raise ValueError(f"n_agents must be a positive integer, got {n_agents!r}")
        if not isinstance(n_opponents, int) or isinstance(n_opponents, bool) or n_opponents <= 0:
            raise ValueError(f"n_opponents must be a positive integer, got {n_opponents!r}")
        if agent_team is None and opponent_team is None:
            if n_agents != DEFAULT_TEAM_SIZE or n_opponents != DEFAULT_TEAM_SIZE:
                raise ValueError(
                    "omitted rosters are available only for the default 11v11 setup; "
                    "provide agent_team and opponent_team for custom team sizes"
                )
            agent_team = _default_roster(0)
            opponent_team = _default_roster(DEFAULT_TEAM_SIZE)
        elif agent_team is None or opponent_team is None:
            raise ValueError("agent_team and opponent_team must be provided together")

        super().__init__(num_agents=n_agents + n_opponents)
        self.n_agents: int = n_agents
        self.n_opponents: int = n_opponents
        self.N: int = n_agents + n_opponents
        if not isinstance(game_duration, int) or isinstance(game_duration, bool) or game_duration <= 0:
            raise ValueError(f"game_duration must be a positive integer, got {game_duration!r}")
        self.game_duration: int = game_duration

        self.agent_team: list[Agent] = list(agent_team)
        self.opponent_team: list[Agent] = list(opponent_team)

        # config dataclass는 전부 immutable이다. 환경별 인스턴스를 소유해 객체 정체성까지 분리하고,
        # 초기화 뒤 캐시된 파생 기하(hx/goal_w 등)와 config를 외부 mutation으로 갈라놓을 수 없게 한다.
        self.b_cfg = Ball() if ball_config is None else replace(ball_config)
        self.s_cfg = Stadium() if stadium_config is None else replace(stadium_config)
        self.d_cfg = DeadBall() if deadball_config is None else replace(deadball_config)
        self.e_cfg = Engine() if engine_config is None else replace(engine_config)
        self.f_cfg = Foul() if foul_config is None else replace(foul_config)
        self.r_cfg = Reward() if reward_config is None else replace(reward_config)
        self._validate_configuration()
        self._validate_rosters()

        if not math.isfinite(control_fps) or control_fps <= 0:
            raise ValueError(f"control_fps must be finite and positive, got {control_fps!r}")
        substeps_exact = 1.0 / (float(control_fps) * self.e_cfg.dt_phys)
        decimation = int(round(substeps_exact))
        if decimation < 1 or not math.isclose(
            substeps_exact, decimation, rel_tol=DIV_EPS, abs_tol=DIV_EPS
        ):
            physics_hz = 1.0 / self.e_cfg.dt_phys
            supported = (
                f"{physics_hz:g} / positive_integer Hz "
                f"(e.g. {physics_hz:g}, {physics_hz / 2:g}, {physics_hz / 4:g})"
            )
            raise ValueError(
                f"control_fps={control_fps!r} is not exactly representable with "
                f"dt_phys={self.e_cfg.dt_phys}; supported rates are {supported}"
            )
        self.e_cfg = replace(self.e_cfg, decimation=decimation)
        self.control_fps: float = 1.0 / (decimation * self.e_cfg.dt_phys)

        self.control_dt = self.e_cfg.decimation * self.e_cfg.dt_phys

        # 필드 반경(경계 클립·분리에서 사용) — Stadium property에서 파생
        self.hx: float = self.s_cfg.half_length
        self.hy: float = self.s_cfg.half_width

        # 기저 스태미나 소모율(1/s): 경기 종료 시 기저활동 선수가 stamina_end_frac에 수렴하도록 역산.
        # game_duration(경기 길이)이 3000이든 135000이든 "끝나면 stamina_end_frac"이 유지된다.
        match_seconds = max(GEOMETRY_EPS, self.game_duration * self.control_dt)
        self.stamina_drain_base: float = (1.0 - self.e_cfg.stamina_end_frac) / match_seconds

        self.r_ball: float = self.b_cfg.radius
        self.r_player: float = self.e_cfg.r_player
        self.players: list = self.agent_team + self.opponent_team
        self.agents: list = [f"{p.id}" for p in self.players]   # JaxMARL 규약 agent 이름(dict 키와 일치)
        self.player_indices = jnp.arange(self.N, dtype=jnp.int32)
        self.team_indices = jnp.array([0, self.n_agents], dtype=jnp.int32)
        self.others_idx = jnp.asarray(
            [[j for j in range(self.N) if j != i] for i in range(self.N)],
            dtype=jnp.int32,
        )

        # 페널티 박스 반치수·골대 규격 — 경합/파울/이벤트 판정에서 사용
        self.pen_len: float = self.s_cfg.penalty_area_length
        self.pen_hw: float = self.s_cfg.penalty_area_width / 2.0
        self.goal_w: float = self.s_cfg.goal_width
        self.goal_h: float = self.s_cfg.goal_height

        # 킥오프 기준 포메이션(N,2): team0 원본 + team1 점대칭(setup._meta 초기배치와 동일).
        # _kickoff_positions가 attack_dir 부호로 점대칭 반전해 골 후·후반 재배치에 재사용.
        self._static_meta = _meta(
            e_cfg=self.e_cfg,
            n_agents=self.n_agents,
            n_opponents=self.n_opponents,
            agent_infos=self.agent_team,
            opponent_infos=self.opponent_team,
        )
        self.base_formation = self._static_meta[3]
        self.field_half = jnp.array([self.hx, self.hy], dtype=jnp.float32)
        self.field_size = jnp.array([self.s_cfg.length, self.s_cfg.width], dtype=jnp.float32)

        # ── JaxMARL 스페이스 — 트레이너가 env.observation_space(agent)/action_space(agent)
        # (base 메서드, 아래 dict 조회) 또는 obs_dim/state_dim/action_dim 속성으로 조회.
        # 관측은 정규화 목표 ~[-1,1]이되 엄격 클립이 아니므로(env/README.md §5.5) 비유계 Box로 정직하게,
        # 행동은 전 차원 [-1,1](env가 클립·스케일, constants.ACTION_DIM)이라 유계 Box.
        self.obs_dim: int = self.obs_spec()["dim"]
        self.state_dim: int = self.state_spec()["dim"]
        self.action_dim: int = ACTION_DIM
        self.schema_version = {
            "action": ACTION_SCHEMA_VERSION,
            "observation": OBS_SCHEMA_VERSION,
            "state": STATE_SCHEMA_VERSION,
        }
        self.observation_spaces = {a: spaces.Box(-jnp.inf, jnp.inf, (self.obs_dim,))
                                   for a in self.agents}
        self.action_spaces = {a: spaces.Box(ACTION_MIN, ACTION_MAX, (self.action_dim,))
                              for a in self.agents}

    def _validate_configuration(self) -> None:
        """JIT 전에 잘못된 물리/규칙 조합을 즉시 거부한다."""
        b, s, d, e, f = self.b_cfg, self.s_cfg, self.d_cfg, self.e_cfg, self.f_cfg
        positive = {
            "ball.radius": b.radius,
            "ball.mass": b.mass,
            "ball.area": b.area,
            "stadium.width": s.width,
            "stadium.length": s.length,
            "stadium.goal_width": s.goal_width,
            "stadium.goal_height": s.goal_height,
            "stadium.penalty_area_length": s.penalty_area_length,
            "stadium.penalty_area_width": s.penalty_area_width,
            "stadium.goal_area_length": s.goal_area_length,
            "stadium.goal_area_width": s.goal_area_width,
            "stadium.center_circle_radius": s.center_circle_radius,
            "stadium.penalty_arc_radius": s.penalty_arc_radius,
            "engine.dt_phys": e.dt_phys,
            "engine.r_player": e.r_player,
            "engine.bench_first_x_offset": e.bench_first_x_offset,
            "engine.bench_spacing": e.bench_spacing,
            "engine.bench_boundary_inset": e.bench_boundary_inset,
            "engine.bench_touchline_inset": e.bench_touchline_inset,
            "engine.a_max": e.a_max,
            "engine.accel_norm_max": e.accel_norm_max,
            "engine.brake_decel_max": e.brake_decel_max,
            "engine.turn_rate": e.turn_rate,
            "engine.reach_xy": e.reach_xy,
            "engine.gk_reach_xy": e.gk_reach_xy,
            "engine.kicker_arrive_r": e.kicker_arrive_r,
            "engine.throwin_clear": e.throwin_clear,
            "engine.clear_dist": e.clear_dist,
            "engine.kicker_speed": e.kicker_speed,
            "engine.contest_temp": e.contest_temp,
            "engine.f2b_speed_max": e.f2b_speed_max,
            "engine.spin_max": e.spin_max,
            "engine.launch_max": e.launch_max,
            "engine.g": e.g,
            "engine.z_ground": e.z_ground,
            "engine.ground_settle_vz": e.ground_settle_vz,
            "engine.ball_inertia_ratio": e.ball_inertia_ratio,
            "engine.body_r": e.body_r,
            "engine.trap_speed_ref": e.trap_speed_ref,
            "engine.sprint_speed": e.sprint_speed,
            "engine.penalty_spot": e.penalty_spot,
            "engine.f2b_shoot_range": e.f2b_shoot_range,
            "engine.throw_speed_max": e.throw_speed_max,
            "engine.legal_margin_floor": e.legal_margin_floor,
            "engine.unrestricted_margin": e.unrestricted_margin,
            "engine.norm_player_vel": e.norm_player_vel,
            "engine.norm_ball_vel": e.norm_ball_vel,
            "engine.norm_ball_z": e.norm_ball_z,
            "engine.norm_spin": e.norm_spin,
            "engine.norm_body_z": e.norm_body_z,
            "engine.norm_score": e.norm_score,
        }
        bad = [name for name, value in positive.items()
               if not math.isfinite(float(value)) or float(value) <= 0.0]
        if bad:
            raise ValueError(f"configuration values must be finite and positive: {', '.join(bad)}")
        if s.goal_width > s.width or s.penalty_area_width > s.width or s.goal_area_width > s.width:
            raise ValueError("goal/penalty/goal-area width cannot exceed stadium width")
        if max(s.penalty_area_length, s.goal_area_length) >= s.half_length:
            raise ValueError("penalty/goal-area length must be smaller than half the pitch")
        if e.launch_down_ref <= b.radius:
            raise ValueError("engine.launch_down_ref must be greater than ball.radius")
        if e.z_ground < b.radius:
            raise ValueError("engine.z_ground must be at least ball.radius")
        if e.penalty_spot >= s.half_length:
            raise ValueError("engine.penalty_spot must be smaller than half the pitch")
        if not 0.0 <= e.launch_down_ground <= e.launch_max:
            raise ValueError("launch_down_ground must lie in [0, launch_max]")
        windows = (e.restart_substeps, e.penalty_substeps, e.gk_hold_substeps)
        if any((not isinstance(v, int) or isinstance(v, bool) or v <= 0) for v in windows):
            raise ValueError("restart, penalty, and GK-hold windows must be positive integers")
        if not isinstance(e.setup_hold_substeps, int) or not 0 <= e.setup_hold_substeps < min(windows):
            raise ValueError("setup_hold_substeps must be an integer smaller than every restart window")
        if not isinstance(e.sep_iters, int) or isinstance(e.sep_iters, bool) or e.sep_iters < 1:
            raise ValueError("engine.sep_iters must be a positive integer")
        counters = {
            "cooldown_substeps": e.cooldown_substeps,
            "ctrl_lock_substeps": e.ctrl_lock_substeps,
            "pass_protect": e.pass_protect,
        }
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in counters.values()):
            raise ValueError(f"engine counters must be positive integers: {counters}")
        if not math.isclose(
            e.norm_spin, e.spin_max, rel_tol=GEOMETRY_EPS, abs_tol=GEOMETRY_EPS
        ):
            raise ValueError("engine.norm_spin and engine.spin_max must stay synchronized")
        probs = {
            "tackle_prob": e.tackle_prob,
            "deflect_prob": e.deflect_prob,
            "body_hit_prob": e.body_hit_prob,
            "trap_base": e.trap_base,
            "spin_head_cap": e.spin_head_cap,
            "spin_chest_cap": e.spin_chest_cap,
            "tackle_out_cap": e.tackle_out_cap,
            "body_spin_keep": e.body_spin_keep,
            "trap_velocity_keep": e.trap_velocity_keep,
            "vmax_floor": e.vmax_floor,
            "stamina_end_frac": e.stamina_end_frac,
            "bounce_tangential_e": e.bounce_tangential_e,
            "bounce_h_keep": e.bounce_h_keep,
            "e_rest": e.e_rest,
            "card_per_foul": f.card_per_foul,
            "red_given_card": f.red_given_card,
        }
        if any(not math.isfinite(float(v)) or not 0.0 <= float(v) <= 1.0 for v in probs.values()):
            raise ValueError(f"probabilities must lie in [0, 1]: {probs}")
        if not (0.0 <= f.tackle_p_min <= f.tackle_p_max <= 1.0):
            raise ValueError("tackle foul probability bounds must satisfy 0 <= min <= max <= 1")
        if not (0.0 <= f.charge_p_min <= f.charge_p_max <= 1.0):
            raise ValueError("charge foul probability bounds must satisfy 0 <= min <= max <= 1")
        if f.charge_contact_padding < 0.0 or not math.isfinite(f.charge_contact_padding):
            raise ValueError("charge_contact_padding must be finite and non-negative")
        if len(e.roll_v_knots) != len(e.roll_d_knots) or len(e.roll_v_knots) < 2:
            raise ValueError("roll_v_knots and roll_d_knots must have the same length >= 2")
        if any(not a < b for a, b in zip(e.roll_v_knots, e.roll_v_knots[1:])):
            raise ValueError("roll_v_knots must be strictly increasing")
        if any((not math.isfinite(float(v)) or v < 0.0) for v in e.roll_d_knots):
            raise ValueError("roll_d_knots must be finite and non-negative")
        if not 0.0 < e.deflect_angle_max <= math.pi:
            raise ValueError("deflect_angle_max must lie in (0, pi]")
        if e.restart_field_inset < 0.0 or e.restart_field_inset >= min(s.half_length, s.half_width):
            raise ValueError("restart_field_inset is outside the pitch")
        if e.throwin_line_inset < 0.0 or e.throwin_line_inset >= s.half_width:
            raise ValueError("throwin_line_inset is outside the pitch")
        if e.free_kick_boundary_inset < 0.0 or e.free_kick_boundary_inset >= min(s.half_length, s.half_width):
            raise ValueError("free_kick_boundary_inset is outside the pitch")
        if d.field_inset >= min(s.half_length, s.half_width):
            raise ValueError("deadball.field_inset is outside the pitch")
        if max(
            d.attack_own_inset,
            d.attack_opp_inset,
            d.defend_field_inset,
            d.throw_attack_own_inset,
            d.throw_attack_opp_inset,
            d.throw_defend_field_inset,
            d.free_kick_own_inset,
            d.free_kick_opp_inset,
            d.gk_line_offset,
        ) >= s.half_length:
            raise ValueError("deadball longitudinal inset is outside the pitch")
        if max(
            d.throw_attack_y_inset,
            d.throw_defend_field_inset,
            d.field_inset,
        ) >= s.half_width:
            raise ValueError("deadball lateral inset is outside the pitch")
        if e.bench_boundary_inset >= s.half_length or e.bench_touchline_inset >= s.half_width:
            raise ValueError("engine bench inset is outside the pitch")
        if e.stamina_sprint_mult < 1.0:
            raise ValueError("stamina_sprint_mult must be at least 1")
        if e.spin_decay * e.dt_phys > 1.0:
            raise ValueError("spin_decay * dt_phys must not exceed 1")
        if not 0.0 <= self.r_cfg.shaping_gamma <= 1.0:
            raise ValueError("reward.shaping_gamma must lie in [0, 1]")
        if self.r_cfg.mode not in {"sparse", "dense"}:
            raise ValueError(f"reward.mode must be 'sparse' or 'dense', got {self.r_cfg.mode!r}")

    def _validate_rosters(self) -> None:
        if len(self.agent_team) != self.n_agents:
            raise ValueError(f"agent_team has {len(self.agent_team)} players, expected {self.n_agents}")
        if len(self.opponent_team) != self.n_opponents:
            raise ValueError(f"opponent_team has {len(self.opponent_team)} players, expected {self.n_opponents}")
        players = self.agent_team + self.opponent_team
        ids = [p.id for p in players]
        if any(not isinstance(pid, (int, np.integer)) or isinstance(pid, bool) for pid in ids):
            raise ValueError("all Agent.id values must be integers")
        if len(set(map(int, ids))) != len(ids):
            raise ValueError("Agent.id values must be unique across both teams")
        for p in players:
            values = (p.speed, p.tall, p.reach_z_max, p.ball_control, *p.init_pos)
            if not all(math.isfinite(float(v)) for v in values):
                raise ValueError(f"player {p.id} has a non-finite attribute")
            if p.speed <= 0 or p.tall <= 0 or p.reach_z_max <= 0:
                raise ValueError(f"player {p.id} speed, tall, and reach_z_max must be positive")
            if not 0.0 <= p.ball_control <= 1.0:
                raise ValueError(f"player {p.id} ball_control must lie in [0, 1]")
            if abs(p.init_pos[DIM_X]) > self.s_cfg.half_length or abs(p.init_pos[DIM_Y]) > self.s_cfg.half_width:
                raise ValueError(f"player {p.id} init_pos lies outside the pitch: {p.init_pos!r}")

    def reset(self, key):
        """JaxMARL 규약 dict 어댑터, 계산은 reset_array(단일 진실원천)"""
        obs, state = self.reset_array(key)
        return {a: obs[i] for i, a in enumerate(self.agents)}, state
    
    def reset_state(self, key):
        """관측 조립 없이 초기 State만 만든다. 물리/렌더/팩토리용 경량 reset 경로."""
        (
            teams,
            att_dir,
            gk,
            init_position,
            vmax,
            reach_z,
            head_z,
            player_ctrl
        ) = self._static_meta
        
        key, k_kick = jax.random.split(key)
        kick_team = jax.random.bernoulli(k_kick).astype(jnp.int32)
        # facing은 속도 파생값 — 정지 상태이므로 자기 공격 방향(team0=0, team1=π)이 된다.
        init_vel = jnp.zeros((self.N, DIM_Z))
        init_facing = self.facing_from_velocity(init_vel, att_dir)

        state = State(
            t = jnp.int32(0),
            ball_pos = jnp.array([0.0, 0.0, self.r_ball]),
            ball_vel = jnp.zeros(DIM_ALL),
            ball_spin = jnp.zeros(DIM_ALL),
            player_ctrl = player_ctrl,
            ball_state = jnp.int32(BALL_DEAD),
            player_pos = init_position,
            player_vel = init_vel,
            player_facing = init_facing,
            attack_dir = att_dir,
            team_id = teams,
            gk_indices = gk,
            vmax = vmax,
            reach_z = reach_z,
            head_z = head_z,
            cooldown=jnp.zeros(self.N),
            stamina=jnp.ones(self.N),
            ctrl_lock_t=jnp.zeros(self.N, dtype=jnp.int32),
            kickoff_team=kick_team,
            poss_team = kick_team,
            last_touch_team = kick_team,
            restart_team = kick_team,
            restart_t=jnp.int32(self.e_cfg.restart_substeps),
            restart_kind=jnp.int32(RK_KICKOFF),
            offside_flag=jnp.zeros(self.N, dtype=jnp.bool_),
            pass_team=jnp.int32(NO_TEAM),
            pass_t=jnp.int32(NO_EVENT),
            foul_kind=jnp.int32(FOUL_NONE),
            foul_actor=jnp.int32(NO_PLAYER),
            foul_victim=jnp.int32(NO_PLAYER),
            pending_taker=jnp.int32(NO_PLAYER),
            setpiece_taker=jnp.int32(NO_PLAYER),
            throw_taker=jnp.int32(NO_PLAYER),
            touch=jnp.zeros(self.N, dtype=jnp.int32),
            score=jnp.zeros(TEAM_COUNT, jnp.int32),
            yellow_cards=jnp.zeros(self.N, jnp.int32),
            active_player=jnp.ones(self.N, jnp.bool_),
            penalty_flight_team=jnp.int32(NO_TEAM),
            penalty_encroach_mask=jnp.zeros(self.N, jnp.bool_),
            restart_indirect=jnp.bool_(False),
            last_touch_code=jnp.int32(TOUCH_NONE),
        )
        state = state._replace(pending_taker=self._kickoff_taker(state, kick_team, init_position))
        # 즉시 킥오프(kickoff_instant)면 키커를 스폿에 미리 스냅해 frame-0 obs가 '순간이동 후' 상태가
        # 되게 한다 — 관측 시퀀스의 1프레임 위치 점프 제거. step 로직은 불변(step1의 _apply_kicker_move가
        # 이미 스폿인 키커에 멱등)이라 킥 타이밍·frame-1 이후 궤적이 그대로다. 비-instant(걸어오는)
        # 킥오프는 frame-0을 포메이션으로 두어(스냅 안 함) 걸어오는 과정이 보이게 유지한다.
        if bool(self.e_cfg.kickoff_instant):
            state = self._apply_kicker_move(state)
        return state

    def reset_array(self, key):
        """배열형 JaxMARL reset: (obs[N,D], State). State-only 코드는 reset_state를 쓴다."""
        state = self.reset_state(key)
        return self.get_obs_array(state), state
    
    def action_agency(self, state):
        """행동 강제성 마스크(per-agent) — 이 state에서 각 선수의 액션이 자유 결정으로 결과에
        반영되는지, 아니면 env가 강제/무시하는지. **주어진 state(액션이 조건으로 삼은 스텝 진입
        시점)에서 결정론적으로 계산** — 제출한 액션값과 무관(도달·게이트·재개 규칙만으로 판정).
        BC/RL에서 강제 프레임을 정책 결정처럼 학습하지 않도록 마스킹하는 신호.

        반환 dict(모두 bool[N]):
          move_forced: 이동이 env에 의해 대체됨 — 세트피스 키커가 스폿까지 강제 워킹(`_apply_kicker_move`,
                       `_move`가 키커 제외) 또는 퇴장자. ※vmax·스태미나·plant&cut 물리 클립은 '강제' 아님
                       (방향 의도는 반영되므로 제외).
          kick_gated:  자발적 킥 서브액션(f2b dir/pow/launch/spin)이 이번 스텝 효력 불가 — 미도달 /
                       쿨다운 / 재개 중 비지정키커·세트업 미완 / 데드볼 / 퇴장. 킥 dim은 무의미하니 마스킹 권장.
                       (경합 패배·ctrl_lock 무효화는 확률/사후라 여기 미포함 — 실현 결과는 info["touch"] 참조.)
          kick_forced: 세트피스 카운트다운 소진으로 킥이 의지와 무관하게 강제 발사(`_kick_gate`의 forced).
                       진입 restart_t ≤ decimation을 판정(서브스텝마다 1 감소하므로 이 스텝 안에 발사됨).
                       킥 파라미터 자체는 에이전트 액션이지만 '언제 찰지' 결정이 강제된 프레임이라, BC는
                       그래디언트 위생상 이 행 전체를 마스킹한다(bc_action_mask — 아래). 라벨 데이터로
                       파라미터를 쓰고 싶으면 이 신호로 별도 취급할 것.
        move_forced/kick_gated/kick_forced는 상호배타적이지 않다(예: 정상 오프더볼 비캐리어는
        move_forced=F·kick_gated=T). kick_forced일 땐 kick_gated=False로 정리(강제 발사=효력 있음)."""
        N = self.N
        ar = self.player_indices
        in_reach, _ = self._in_reach(state)
        _, setup_done, arrived, sp_active = self._setpiece_kick_lock(state)
        alive = state.ball_state == BALL_ALIVE
        restart_active = state.restart_t > 0
        active = state.active_player
        is_designated = (ar == state.pending_taker) & (state.pending_taker >= 0)
        allowed = jnp.where(restart_active, is_designated & setup_done, alive)
        kick_can_fire = in_reach & (state.cooldown <= 0) & allowed & active
        # [B] 킥 타이밍 강제: 지정 키커의 킥은 setup 완료 시 결정론적 발사(_kick_gate와 정렬). BC가 '언제
        # 찰지'를 정책 결정처럼 학습하지 않게 강제 프레임을 forced로 잡되, **킥 파라미터는 bc_action_mask에서
        # 학습 대상으로 열려 있다**(kick_ok=~kick_gated, 실현 여부는 kick_applied와 AND).
        #  ★진입 setup_done만 보면 발화 스텝을 놓친다: restart_t는 서브스텝당 1 감소하는데 킥은 setup_done이
        #  '넘어가는' 서브스텝(restart_t가 임계=window-setup_hold를 통과)에 발사된다. 진입 restart_t가 임계보다
        #  decimation 이내면 이 스텝 안에 발사되므로 kick_forced로 잡아 마스크를 연다(놓치면 실제 세트피스 킥
        #  라벨의 대부분이 kick_gated=True로 드롭됨). 킥오프 즉시발동(instant_ko)은 setup_done 즉시 성립이라 포함.
        e_cfg = self.e_cfg
        sp_window = jnp.where(state.restart_kind == RK_PENALTY, e_cfg.penalty_substeps,
                    jnp.where(state.restart_kind == RK_GK_HOLD, e_cfg.gk_hold_substeps, e_cfg.restart_substeps))
        instant_ko = jnp.bool_(e_cfg.kickoff_instant) & (state.restart_kind == RK_KICKOFF)
        fires_this_step = arrived & (instant_ko | (state.restart_t <= (sp_window - e_cfg.setup_hold_substeps + e_cfg.decimation)))
        kick_forced = restart_active & is_designated & active & fires_this_step
        kick_gated = (~kick_can_fire) & (~kick_forced)
        # 데드볼 내장 엔진(deadball_engine) ON이면 재개 중 전원의 이동을 엔진이 강제 → 전원 move_forced.
        # 이로써 골 후·하프타임 킥오프 재배치(데드볼)의 비-키커 텔레포트도 자동 마스킹된다(2번 흡수).
        dead_engine = jnp.bool_(self.e_cfg.deadball_engine) & restart_active
        move_forced = (is_designated & sp_active) | (~active) | dead_engine
        # 하프타임 전이 프레임 — 이 스텝이 끝난 뒤 `_halftime_switch`가 위치·공격방향·속도·공·재개를
        # 통째로 덮으므로 **제출 액션이 post-state를 전혀 설명하지 못한다**. `t`는 스텝당 정확히 1
        # 증가하고 전환 판정은 `state.t == game_duration // 2`(증가 후)이므로, 진입 시점에
        # `(t + 1) == game_duration // 2`로 결정적으로 알 수 있다.
        # move_forced/kick_gated에 섞지 않고 별도 신호로 두는 이유: 이 둘은 `get_avail_actions_array`가
        # 그대로 투영하는 **사전 행동 권한**인데, 하프타임 프레임에도 에이전트는 액션을 제출해야 하고
        # 그 액션은 전환 전 서브스텝 물리에 실제로 작용한다. 막을 것은 권한이 아니라 **라벨**이다.
        halftime_reset = jnp.broadcast_to(
            (state.t + 1) == jnp.int32(self.game_duration // 2), (N,)
        )
        return {"move_forced": move_forced, "kick_gated": kick_gated,
                "kick_forced": kick_forced, "halftime_reset": halftime_reset}

    def bc_action_mask(self, info, kick_applied=None):
        """BC/RL 손실용 per-dim 액션 마스크 (N,ACTION_DIM=8) bool — True=학습(그래디언트 허용), False=차단.
        `action_agency`의 세 신호를 8-D 액션 레이아웃(_decode / L∞ stretch)에 매핑한다.

        레이아웃: [0]=want_f2b · [1:3]=move(L∞ stretch) · [3:5]=f2b(L∞ stretch) · [5]=launch · [6:8]=spin.
        정책:
          - 이동 dim[1:3]     ← ~move_forced   : 키커 강제워킹·퇴장이면 이동 차단.
          - 킥 dim[0,3:8]     ← ~kick_gated    : 무의미(gated) 프레임만 차단. **강제킥(kick_forced)이어도
                               킥 행(결정 want + 파라미터 dir·pow·launch·spin)은 학습 대상으로 열어둔다** —
                               강제킥은 실제로 공에 적용되는 인과 이벤트이고, 키커는 그 세트피스를 실제로
                               차는 주체라 '여기서 찬다(want)'도 진짜 라벨이다(역산은 릴리즈를 자발 킥으로 주입).
                               kick_forced는 지정 키커에게만 참이므로 이 언마스킹은 키커 킥 행에만 작용하고,
                               오프볼은 여전히 kick_gated=True로 킥 dim이 막힌다.
        키커 강제워킹 프레임은 이동(move_forced)은 통째 차단되지만, 킥 행은 살아난다.
        off-ball `kick_gated`는 킥 dim만 막고 이동은 살린다(자유 결정이라 학습 대상 — 통째 마스킹하면
        필드 전원 이동 감독의 99.8%가 소실).

        [BC 킥 라벨 계약] 이 마스크의 킥 dim은 **진입(step-entry) 결정론 게이트 = 필요조건**일 뿐이다.
        킥의 인과성은 스텝 내부의 확률적 경합에서 갈리므로(승자만 파라미터 적용·굴절/GK캐치는 무효),
        진입 마스크만으론 경합 패자·굴절·GK 자동클레임 킥을 과포함한다. **순수 킥 라벨 = 킥 dim(여기)
        ∧ `info["kick_applied"]`(실현 인과킥, `_kick_applied`)**. 이동은 진입 마스크 하나로 충분하지만,
        킥은 이 사후 신호와 반드시 AND 해야 인과 라벨이 된다. (ctrl_lock 등 사후 무효화도 kick_applied로 흡수.)"""
        move_ok = ~info["move_forced"]                       # (N,)
        kick_ok = ~info["kick_gated"]                        # 킥 결정·파라미터: 강제킥이어도 인과 → 학습(단 info["kick_applied"]와 AND)
        # ★진입 게이트는 **스텝 진입 시점 거리**로 reach를 판정하는데, 실제 전이는 _move로 선수를
        # 옮긴 뒤 _kick_gate를 다시 계산한다. 그래서 진입엔 reach 밖이었지만 같은 0.04 s 안에
        # 이동·공 접근으로 도달해 **실제로 찬** 킥이 존재한다(실측: 진입 1.815 m > 임계 1.71 m인데
        # touch=PASS·kick_applied=True). 진입 게이트만 쓰면 그 라벨이 통째로 버려지므로, 실현
        # 인과킥은 게이트를 연다: kick_ok = (~kick_gated) | kick_applied.
        # (물리는 건드리지 않는다 — 마스크만 실현 결과와 정렬한다.)
        if kick_applied is not None:
            kick_ok = kick_ok | kick_applied
        # ★하프타임 전이 프레임은 **전 차원 하드 마스크**이며 위 kick_applied보다 우선한다.
        # 스텝 안에서 킥이 실제로 적용됐더라도 `_halftime_switch`가 공·위치·재개를 전부 덮어써
        # 그 결과가 사라지므로, 이 프레임을 자유 행동 라벨로 세면 안 된다. 진입 시점 마스크만으로는
        # 잡히지 않는다 — 하프타임은 스텝 **끝**에 오기 때문(action_agency의 halftime_reset 참조).
        halftime = info.get("halftime_reset")
        if halftime is not None:
            live = ~halftime
            move_ok = move_ok & live
            kick_ok = kick_ok & live
        cols = jnp.stack([
            kick_ok,                          # 0  want_f2b
            move_ok, move_ok,                 # 1,2  move (L∞ stretch 2D)
            kick_ok, kick_ok,                 # 3,4  f2b  (L∞ stretch 2D)
            kick_ok,                          # 5    f2b_launch
            kick_ok, kick_ok,                 # 6,7  spin_side, spin_back
        ], axis=1)
        return cols                                          # (N, ACTION_DIM=8) bool

    def _kick_applied(self, state, touch_before=None):
        """실현 인과킥 마스크(bool[N]) — 이 선수의 제출 킥 파라미터(f2b_dir·pow·launch·spin)가 이번 스텝
        공 속도를 **실제로 결정**했는가. contest에서 params가 공에 적용되는 분기는 free_play(자발/강제
        세트피스 킥) 와 tackle_ok 둘뿐이고, 그 결과가 per-player touch **코드**로 남는다(contest._apply_force2ball).
        인과 집합 = {PASS, PASS_HEAD, SHOOT, SHOOT_HEAD, DRIBBLE, TACKLE, INTERCEPT}.
        제외: DEFLECT(출구각 랜덤 재추첨 — params 무시)·GK_CATCH/PARRY(무액션 env 클레임)·NONE.
        ※태클/인터셉트도 포함(params 적용됨) — 오펜시브 킥만 원하면 다운스트림에서
        {PASS*,SHOOT*,DRIBBLE}로 좁히면 된다(contest의 is_played 집합).

        ★파울 태클 제외: 태클 파울은 공을 정지(new_vel=0)시키면서도 touch를 TOUCH_TACKLE로 라벨하므로
        (contest: `code = where(foul, TOUCH_TACKLE)` + `new_vel = where(foul, 0)`), 제출 params가 공 속도를
        결정하지 않았는데도 위 집합에 걸려 **거짓 인과킥**이 된다. foul_kind==FOUL_TACKLE인 foul_actor를 뺀다.

        [BC 킥 라벨 계약] 순수 킥 라벨 = bc_action_mask 킥 dim(진입 게이트) ∧ 이 신호(실현 진실).
        진입 마스크만으론 확률적 경합 패자·굴절·GK 자동클레임을 과포함하므로 반드시 AND 한다.

        ★touch_before: **contest 접촉만 세기 위한 서브스텝 스냅샷**(`_apply_force2ball` 직전 값).
        위 인과 집합은 "이 코드들은 contest가 params를 공에 적용한 결과"라는 전제 위에 있는데,
        `ball._ball_body`의 **몸통 트랩도 DRIBBLE/INTERCEPT를 기록**한다. 트랩은 출구속도가
        `trap_velocity_keep · v`인 순수 수동 물리라 제출 params와 무관하므로, 컨트롤 스텝 끝의
        누적 `state.touch`만 보면 트랩이 인과킥으로 오탐된다. 스냅샷을 주면 그 서브스텝에
        **새로 생긴 contest 접촉**만 집계한다(`step_env_array`가 `_ball_body` 이전에 호출·누적).
        None이면 종전(누적 touch 전수) 동작 — 하위호환용이며 위 오탐이 남는다."""
        touch = state.touch
        applied = ((touch == TOUCH_PASS) | (touch == TOUCH_PASS_HEAD)
                   | (touch == TOUCH_SHOOT) | (touch == TOUCH_SHOOT_HEAD)
                   | (touch == TOUCH_DRIBBLE) | (touch == TOUCH_TACKLE)
                   | (touch == TOUCH_INTERCEPT))
        if touch_before is not None:
            applied = applied & (touch != touch_before)
        foul_tackle_actor = (
            (state.foul_kind == FOUL_TACKLE)
            & (self.player_indices == state.foul_actor)
        )
        return applied & (~foul_tackle_actor)

    def step_env(self, key, state, actions):
        """JaxMARL 규약 dict 어댑터 — 순수 전이(auto-reset 없음). 계산은 step_env_array에 위임.
        키는 __init__에서 만든 self.agents(= f"{p.id}") 재사용 — 매 호출 f-string 재생성 제거.
        ※핫루프(데이터 생성·학습)에선 이 dict 경로 대신 배열 경로 step_env_array + jit/vmap을 쓸 것
        (dict 조립은 파이썬 오버헤드라 jit 밖에서 반복 호출하면 병목)."""
        act_arr = jnp.stack([actions[a] for a in self.agents], axis=0)
        obs_arr, state, reward_arr, done_all, info = self.step_env_array(key, state, act_arr)
        obs = {a: obs_arr[i] for i, a in enumerate(self.agents)}
        reward = {a: reward_arr[i] for i, a in enumerate(self.agents)}
        done = {a: done_all for a in self.agents}
        done["__all__"] = done_all
        return obs, state, reward, done, info

    def step_env_array(self, key, state, act_arr, forced_winner=None, forced_freeplay=None,
                       suppress_charge=None, suppress_body=None, suppress_retake=None,
                       suppress_restart=None, collect_substeps=False,
                       include_bc_info=True, compute_observation=True):
        """step 계산 본체(단일 진실원천) — (N,·) 배열 경로. 반환 (obs (N,·), State, reward (N,),
        done_all 스칼라, info). 종료는 시간제한(t≥game_duration)뿐이라 done은 스칼라 하나로 충분.

        서브스텝 파이프라인: 키커이동 → 이동 → 차징파울 → (킥 게이트 재계산) → 경합 승자 →
        force2ball → 몸통충돌 → 스로인 재터치 → 오프사이드 → 공 자유물리 → 이벤트.

        forced_winner: reconstruct용 경합 승자 주입. (decimation,) int 배열 — 서브스텝별로
        ≥0=그 선수 승자 / -1=무승자 / -2=정상 샘플. None이면 전 서브스텝 샘플(포워드 불변).
        pin도 합법 후보 게이트(도달·쿨다운·게이트)를 통과해야 성립 — 게이트 밖 pin은 불발(-1).
        forced_freeplay: reconstruct용 분기 pin. (decimation,) bool 배열 — True인 서브스텝은
        force2ball의 opp_poss를 젖혀 관측 터치를 free_play(결정론 킥)로 강제(상태 무변경 pin).
        None이면 전부 False(포워드 불변).
        suppress_charge / suppress_body: reconstruct용 추첨 봉쇄 pin(스칼라 bool, 스텝 전체 적용).
        True면 각각 차징 파울 추첨·공-몸통 hit 추첨을 '불발'로 고정 — 관측에 없는 확률 이벤트가
        복원 창을 탈선시키지 않게. 추첨은 소비되므로 RNG 열 불변. None이면 False(포워드 불변).
        suppress_retake: 세트피스 에피소드 복원용 pin(스칼라 bool) — 재개 소비 시 침범 retake와
        페널티 결과 판정의 침범 재실행(events)을 봉쇄(근거=관측: 실경기에서 심판이 진행시킴/
        결과를 인정함). None이면 False(포워드 불변).
        include_bc_info: 정적 bool. False면 ``action_agency``·BC 마스크·kick_applied 산출과
        info 삽입을 생략한다. RL처럼 이 라벨을 쓰지 않는 핫루프용.
        compute_observation: 정적 bool. False면 관측 대신 shape=(N,0) 빈 배열을 반환한다.
        렌더/물리 검증처럼 state만 필요한 롤아웃에서 O(N²) 관측 조립을 생략하는 경로다.
        """
        act_arr = jnp.asarray(act_arr)
        expected_shape = (self.N, ACTION_DIM)
        if act_arr.shape != expected_shape:
            raise ValueError(f"act_arr must have shape {expected_shape}, got {act_arr.shape}")
        (
            want_f2b, mv_dir, mv_pow,
            f2b_dir, f2b_pow, f2b_launch,
            spin_side, spin_back
        ) = self._decode(act_arr, state.attack_dir)

        e_cfg = self.e_cfg
        if forced_winner is None:
            forced_winner = jnp.full(
                (e_cfg.decimation,), SAMPLED_WINNER, dtype=jnp.int32
            )   # 전 서브스텝 샘플
        else:
            forced_winner = jnp.asarray(forced_winner, dtype=jnp.int32)
            if forced_winner.shape != (e_cfg.decimation,):
                raise ValueError(
                    "forced_winner must have shape "
                    f"({e_cfg.decimation},), got {forced_winner.shape}"
                )
        if forced_freeplay is None:
            forced_freeplay = jnp.zeros((e_cfg.decimation,), dtype=bool)         # 전부 off(포워드 불변)
        else:
            forced_freeplay = jnp.asarray(forced_freeplay, dtype=bool)
            if forced_freeplay.shape != (e_cfg.decimation,):
                raise ValueError(
                    "forced_freeplay must have shape "
                    f"({e_cfg.decimation},), got {forced_freeplay.shape}"
                )
        if include_bc_info:
            agency = self.action_agency(state)      # 스텝 진입 상태서 행동 강제성 마스크(액션 조건과 정렬)
            # bc_mask는 스캔이 끝나 kick_applied가 확정된 뒤에 만든다(bc_action_mask 참조).
        state = state._replace(touch=jnp.zeros(self.N, dtype=jnp.int32))
        ball_x0 = state.ball_pos[DIM_X]          # dense 전진/소유획득 차분 기준(스텝 전)
        poss0 = state.poss_team

        def _kick_gate(st, setup_done):
            """현재 상태에서 f2b 킥 허용 마스크(N,) 계산 — 오픈플레이=alive, 재개=재개팀 & setup 완료.
            페널티도 물리 플레이라 포함(setup 후 키커가 실제 킥). in_reach가 사실상 키커로 한정한다.

            세트피스 시간 만료 강제: 세트업이 끝난 뒤에도 키커가 끝내 안 차서 카운트다운이 이번
            서브스텝에 소진(restart_t==1→0)되면, 슛 게이트를 강제로 열어 킥을 발생시킨다 —
            시간초과가 흐지부지 루즈볼로 새는 대신 반드시 인플레이 킥으로 전환된다. 강제 대상은
            지정 키커(pending_taker) 하나뿐이고 in_reach·cooldown을 우회한다(키커는 스폿 뒤에 정렬
            완료). 킥 파라미터(방향·파워·발사각·스핀)는 그대로 에이전트 액션이라 결과는 여전히 역산 가능."""
            in_reach, dist_xy = self._in_reach(st)
            alive = st.ball_state == BALL_ALIVE
            restart_active = st.restart_t > 0
            # 재개 킥은 지정 키커만 — 팀 단위로 열면 스폿 근처 동료가 세트피스를 가로채거나
            # (페널티는 침범자 본인이 차는 것도 가능) GK 홀드 중인 공을 동료가 차버릴 수 있다.
            restart_kick_ok = (self.player_indices == st.pending_taker) & setup_done
            allowed = jnp.where(restart_active, restart_kick_ok, alive)
            voluntary = want_f2b & in_reach & (st.cooldown <= 0) & allowed
            # [B] 킥 타이밍 강제: 자율 킥 창을 없애고 setup 완료(=키커 도착·정렬) 즉시 결정론적 발사.
            # 타이밍은 env가 정하고 킥 파라미터(방향·파워·발사각·스핀)는 그대로 정책 액션 — 은닉 '언제 찰지'
            # 결정이 사라져 BC가 킥을 인과 라벨로 학습 가능. 킥오프 즉시발동(kickoff_instant)은 setup_done을
            # 즉시 참으로 만드는 restart.py 로직으로 자연 흡수(별도 분기 불필요). 카운트다운 소진(restart_t<=1)
            # 타임아웃 강제도 setup_done 즉시 발사에 포섭된다.
            forced = (restart_active & setup_done
                      & (self.player_indices == st.pending_taker) & (st.pending_taker >= 0)
                      & st.active_player)      # 퇴장자면 강제 안 함(방어 가드) — 실해소는 events의 taker_dead 재지정
            do_kick = voluntary | forced
            return do_kick, dist_xy

        def substep(carry, xs):
            forced_winner_sub, forced_freeplay_sub = xs
            st, scored, kick_acc, k = carry
            k, k_win, k_app, k_evt, k_body, k_chg = jax.random.split(k, 6)
            poss_before = st.poss_team

            # 키커 강제이동 + 세트피스 봉인 판정(키커는 이동 봉인)
            st = self._apply_kicker_move(st)
            _, _, _, sp_active = self._setpiece_kick_lock(st)
            is_kicker = (self.player_indices == st.pending_taker) & sp_active
            # 데드볼 내장 엔진: 토글 ON & 데드볼이면 비-키커 전원 이동을 엔진 타깃으로 대체(정책 이동 무시).
            # 키커는 _apply_kicker_move가 이미 몰고 아래 ~is_kicker로 _move서 제외되므로 그대로 둔다.
            # 킥 파라미터·f2b_dir·do_kick(release)은 정책 소유 — 이동만 엔진.
            # (킥 게이트는 _move 뒤 차징 파울까지 반영해 한 번만 계산한다 — facing이 킥 방향을
            #  더는 참조하지 않으므로 이동 전 선계산이 필요 없다.)
            if self.e_cfg.deadball_engine:
                eng_dir, eng_pow = self._deadball_move(st)
                forced_move = (
                    (st.restart_t > 0) & (~is_kicker) & st.active_player
                )
                mv_dir_s = jnp.where(forced_move[:, None], eng_dir, mv_dir)
                mv_pow_s = jnp.where(forced_move, eng_pow, mv_pow)
            else:
                mv_dir_s, mv_pow_s = mv_dir, mv_pow
            st = self._move(st, ~is_kicker, mv_dir_s, mv_pow_s)

            # 차징 파울이 공을 데드로 만들 수 있으므로, 경합·킥 적용 전 킥 게이트를 '차징 후' 상태로 재계산
            st = self._charge_foul(st, k_chg, suppress=suppress_charge)
            _, setup_done2, _, _ = self._setpiece_kick_lock(st)
            do_kick, dist_xy = _kick_gate(st, setup_done2)

            # GK 리액티브 클레임(박스 안 GK 본능)을 경합 후보에만 더한다 — 이동 facing용 do_kick과 분리.
            contest_cand = do_kick | self._gk_reactive_claim(st)
            # 재탈취 지연(ctrl_lock) 중인 상대는 후보에서 제외 — 안 그러면 방금 뺏긴 선수가 경합 argmax를
            # 이겨 승자 슬롯을 먹고 no-op(locked)하며 새 점유팀 캐리어를 lock창(~0.16s) 동안 굶기고,
            # 헛쿨다운까지 받는다. 도달·쿨다운처럼 '확정 규칙 게이트'라 forced_winner pin도 통과해야 성립.
            locked_opp = (st.poss_team >= 0) & (st.team_id != st.poss_team) & (st.ctrl_lock_t > 0)
            contest_cand = contest_cand & (~locked_opp)
            winner, any_cand = self._contest_winner(st, contest_cand, dist_xy, k_win, forced_winner_sub)
            # 재터치 제한은 컨트롤 스텝 입구가 아니라 매 물리 서브스텝의 접촉 직전 상태를
            # 기준으로 판정해야 한다. touch_before로 이번 서브스텝에 새로 생긴 접촉만 구분한다.
            throw_taker_before = st.throw_taker
            setpiece_taker_before = st.setpiece_taker
            touch_before = st.touch
            st = self._apply_force2ball(
                st,
                winner,
                any_cand,
                f2b_dir,
                f2b_pow,
                f2b_launch,
                k_app,
                spin_side,
                spin_back,
                want_kick=do_kick,
                forced_freeplay=forced_freeplay_sub,
                suppress_retake=suppress_retake,
            )
            # ★인과킥 집계는 반드시 _ball_body **이전**에 — 몸통 트랩이 남기는 DRIBBLE/INTERCEPT는
            # 수동 물리(trap_velocity_keep·v)라 제출 킥 params와 무관한데, 컨트롤 스텝 끝의 누적
            # touch만 보면 인과킥으로 오탐된다(_kick_applied 참조). 여기서 서브스텝별 contest
            # 접촉만 뽑아 OR 누적한다.
            if include_bc_info:
                kick_acc = kick_acc | self._kick_applied(st, touch_before)
            # touch_before를 넘겨 몸통 충돌 배제를 '이번 서브스텝의 새 접촉'으로 한정한다 —
            # 누적 touch를 그대로 보면 배제 창 길이가 decimation(=control_fps)에 종속된다.
            st = self._ball_body(
                st, k_body, suppress=suppress_body, touch_before=touch_before
            )
            st = self._throwin_restriction(
                st, throw_taker_before, setpiece_taker_before, touch_before
            )
            # 오프사이드 콜도 재터치 판정과 같은 규약 — 이번 서브스텝에 새로 생긴 접촉만 트리거로
            # 삼는다. touch_before 없이 누적 touch를 보면 플래그가 서기 전(앞 서브스텝)의 터치로
            # 오프사이드가 즉시 오검된다(같은 컨트롤 스텝 내 다중 터치 국면).
            st = self._offside_check(st, touch_before)
            st = self._ball_step(st)

            # 이벤트 판정 직전 상태로 재개 도착/활성 재계산(≤1서브스텝 카운트다운 편향 제거)
            _, _, arrived_ev, sp_active_ev = self._setpiece_kick_lock(st)
            st, s = self._events(st, k_evt, arrived_ev, sp_active_ev,
                                 suppress_retake=suppress_retake, suppress_restart=suppress_restart)
            scored = jnp.where(scored >= 0, scored, s)

            # 재탈취 지연(per-player): 소유를 잃은 팀 선수에게 lock 세팅, 그 외 감쇠. 쿨다운 감쇠.
            changed = (st.poss_team != poss_before) & (poss_before >= 0)
            lock_set = changed & (st.team_id == poss_before)
            ctrl_lock_t = jnp.where(lock_set, jnp.int32(e_cfg.ctrl_lock_substeps),
                                    jnp.maximum(0, st.ctrl_lock_t - 1))
            cooldown = jnp.maximum(0.0, st.cooldown - 1.0)
            st = st._replace(ctrl_lock_t=ctrl_lock_t, cooldown=cooldown)
            # collect_substeps(정적 플래그)면 서브스텝 종료 상태를 ys로 스택 — 물리 100Hz 부드러운 렌더용.
            # False(학습 경로)면 None → jit 특수화로 오버헤드 0.
            return (st, scored, kick_acc, k), (st if collect_substeps else None)

        (state, scored, kick_applied, _), sub_states = lax.scan(
            substep,
            (state, jnp.int32(NO_EVENT), jnp.zeros(self.N, dtype=bool), key),
            (forced_winner, forced_freeplay), length=e_cfg.decimation)
        state = state._replace(t=state.t + 1)
        state = self._halftime_switch(state, state.t == (self.game_duration // 2))

        done_all = state.t >= self.game_duration
        reward = self._reward_array(state, scored, ball_x0, poss0)
        # info에 BC 라벨링·reconstruct 검증용 터치/파울 신호 노출(스텝 내 누적된 per-player touch 등)
        info = {"scored": scored, "poss_team": state.poss_team, "score": state.score,
                "truncated": done_all, "active_player": state.active_player,
                "touch": state.touch, "last_touch_team": state.last_touch_team,
                "foul_kind": state.foul_kind, "foul_actor": state.foul_actor,
                "foul_victim": state.foul_victim}
        if include_bc_info:
            # per-dim BC 손실 마스크 (N,8). 킥 dim은 진입 게이트 ∨ 실현 인과킥이다 —
            # info["kick_gated"]는 문서화된 **진입 시점** 신호 그대로 내보내고(RL이 사전 판단에 씀),
            # 마스크만 사후 실현을 반영한다.
            bc_mask = self.bc_action_mask(agency, kick_applied=kick_applied)
            info.update({
                # 실현 인과킥(선수별 bool) — 제출 킥 params가 이번 스텝 공 속도를 실제로 결정했는가.
                # [BC 킥 라벨 계약] 순수 킥 라벨 = bc_action_mask 킥 dim ∧ kick_applied.
                # 서브스텝 스캔이 contest 접촉만 골라 누적한 값(몸통 트랩 오탐 제외 — _kick_applied 참조).
                "kick_applied": kick_applied,
                # 행동 강제성 마스크(스텝 진입 상태 기준) — BC/RL 마스킹용. action_agency()/bc_action_mask() 참조.
                # 원신호 3종 + 손실에 바로 곱하는 per-dim 마스크(N,8). RL은 원신호로 자체 정책 구성 가능.
                "move_forced": agency["move_forced"], "kick_gated": agency["kick_gated"],
                "kick_forced": agency["kick_forced"],
                # 하프타임 전이 프레임(bool[N], 프레임 단위 균일) — 시퀀스 hard boundary이자
                # 전 차원 라벨 마스크의 근거. 데이터셋의 `halftime_reset` 필드가 이 값을 그대로 쓴다.
                "halftime_reset": agency["halftime_reset"],
                "bc_action_mask": bc_mask,
            })
        if collect_substeps:
            # (decimation,)-선두 스택 State — 스텝 내 물리 서브스텝별 상태(위치·공 등). 렌더 밀도 up용.
            info["substeps"] = sub_states
        obs = (
            self.get_obs_array(state)
            if compute_observation
            else jnp.empty((self.N, 0), dtype=state.ball_pos.dtype)
        )
        return obs, state, reward, done_all, info


if __name__ == "__main__":
    env = SoccerEnv(
        game_duration=3000,  # 데모 전용 축약값. 정식 기본은 DEFAULT_GAME_DURATION을 쓴다.
        control_fps=25,
    )
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key = key)
    
    for i in range(10):
        key, k_step = jax.random.split(key)
        one_act = jnp.array(
            [-1.0, 1.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1.0],
            dtype=jnp.float32,
        )
        actions = {f"{player.id}": one_act for player in env.players}
        obs, state, reward, done, info = env.step(key = k_step, state = state, actions = actions)
        print(f"Step {i+1}: t={int(state.t)} ball_state={int(state.ball_state)} "
              f"poss={int(state.poss_team)} score={state.score.tolist()} done={bool(done['__all__'])}")
        print("-" * 30)
