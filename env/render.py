"""경량 궤적 렌더러 — State 시퀀스 → mp4. 속도 지향(경량 경로는 순수 numpy 래스터, matplotlib 무의존).
`render_mp4`는 `mode="rich"` 디스패치(render_rich, 지연 import)와 트래킹/이벤트 덤프(_write_*)의
공용 진입점이기도 하다 — 덤프 산출물은 모드와 무관하게 동일 경로에서 나온다.

원본 SOCCER render.py(3D matplotlib, 정밀 브로드캐스트 뷰)와 목적이 다르다: 여기서는 **궤적 재현
확인**에 필요한 최소 정보만 탑다운 2D로 그려 렌더 속도를 극대화한다(원본 대비 수십 배).
프레임 합성이 전부 벡터화된 numpy 연산(사전계산 피치 + 감쇠 트레일 레이어 + 원판 스탬핑)이고,
인코딩은 원본과 동일하게 imageio+libx264(번들 ffmpeg).

표현(필수 state만, 전부 State 필드에서 직접 — 원본의 Frame dataclass 어댑터 불필요):
 · 선수: 팀색 원판(적/청, GK는 밝은 톤), 퇴장(active_player=False)은 피치에서 제거하고 근경
   터치라인 아래 벤치에 흐린 팀색+레드카드로 표시. 머리 위 스태미나 바(녹→적), 옐로카드는
   노란 사각 마커.
 · 공: 지면 그림자(회색) + 높이만큼 화면 위로 띄운 흰 원판(반지름도 z에 비례) — 3D 정보를 2D에 투영.
 · 궤적: 감쇠 트레일 레이어(공 백색, 선수 팀색 저휘도) — 프레임당 전체 배열 곱 1회로 페이드.
 · 이벤트: 터치코드별 색 링 플래시(패스 백/슛 주황/태클 적/인터셉트 황/GK 하늘/굴절 회), 파울은
   actor 적색 링+victim 연결선, 득점은 GOAL 오버레이, 카드/퇴장 온셋 링. HUD에 시계·스코어·소유·
   재개종류·최근 이벤트 티커.

원본 대비 구현 차이(클론 규약 준수): `on_pitch`→`active_player`, 터치코드에 DEFLECT(9)/PARRY(10)
추가, 득점은 Frame.scored가 아니라 score 차분으로 검출, 파울은 state.foul_kind/actor/victim 직접
사용. 월드 좌표로만 그리므로 obs 폴딩 규약(z회전 vs x미러)과 무관.

사용:
    frames = [state0, state1, ...]          # step_env가 반환한 State들(원 State 그대로)
    env.render_mp4(frames)                  # → ./replays/<unix초>/match.mp4 (실행 경로 기준)
    env.render_mp4(frames, "path/to/out.mp4", fps=25)   # 경로 직접 지정도 가능
"""
from __future__ import annotations

import os
import json
import time

import numpy as np
import jax
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from constants import *  # noqa: F401,F403

# imageio-ffmpeg가 ffmpeg를 fork+exec로 띄울 때, JAX가 스레드를 띄운 상태라 CPython이
# "os.fork() was called ... multithreaded" RuntimeWarning을 낸다. 이 fork는 즉시 exec하는
# async-signal-safe 경로(그 사이 파이썬 락 안 잡음)라 실제 데드락 위험이 없어 렌더는 매번 완료된다 —
# 과보수적인 이 특정 경고만 좁게 억제한다(다른 fork 경고엔 영향 없음).
import warnings
warnings.filterwarnings("ignore", message=r".*os\.fork\(\) was called.*",
                        category=RuntimeWarning)

from types import SimpleNamespace   # rich 프레임 뷰(아래 RichRenderer)에서 사용

# ── rich 모드 전용 matplotlib 의존은 지연 로드 ────────────────────────────────
# light 모드(순수 numpy)는 matplotlib 없이 돌아야 하므로 아래 심볼들을 모듈 전역에 '빈 자리'로
# 두고, rich 렌더가 실제로 호출될 때 _ensure_mpl()이 채운다(RichRenderer 메서드들은 이 전역을
# 참조 — render_rich.py가 모듈 상단 import로 쓰던 것과 동일한 이름 해석). 단일 파일 통합 후에도
# 경량 경로는 무의존 유지.
plt = None
LineCollection = PolyCollection = Rectangle = MplPolygon = None


def _ensure_mpl():
    """rich 렌더 진입 시 1회 호출 — matplotlib을 로드해 모듈 전역(plt/LineCollection/…)을 채운다."""
    global plt, LineCollection, PolyCollection, Rectangle, MplPolygon
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        from matplotlib.collections import LineCollection as _LC, PolyCollection as _PC
        from matplotlib.patches import Rectangle as _R, Polygon as _MP
        plt, LineCollection, PolyCollection, Rectangle, MplPolygon = _plt, _LC, _PC, _R, _MP

TOUCH_LABEL = {TOUCH_PASS: "PASS", TOUCH_SHOOT: "SHOOT", TOUCH_PASS_HEAD: "H-PASS",
               TOUCH_SHOOT_HEAD: "H-SHOOT", TOUCH_DRIBBLE: "DRIBBLE", TOUCH_TACKLE: "TACKLE",
               TOUCH_GK_CATCH: "GK-CATCH", TOUCH_INTERCEPT: "INTERCEPT",
               TOUCH_DEFLECT: "DEFLECT", TOUCH_PARRY: "PARRY"}
TOUCH_RING = {TOUCH_PASS: (240, 240, 240), TOUCH_SHOOT: (255, 150, 40),
              TOUCH_PASS_HEAD: (240, 240, 240), TOUCH_SHOOT_HEAD: (255, 150, 40),
              TOUCH_DRIBBLE: (120, 210, 140), TOUCH_TACKLE: (255, 70, 70),
              TOUCH_GK_CATCH: (90, 200, 255), TOUCH_INTERCEPT: (255, 210, 70),
              TOUCH_DEFLECT: (170, 170, 170), TOUCH_PARRY: (90, 200, 255)}

# ── 터치 링 = '선수가 공에 가한 힘'의 시각화 ─────────────────────────────────
# 공 질량은 상수이므로 임펄스 ∝ 프레임 간 공 속도변화 ‖Δv‖(3D, m/s). 이 값을 링 반지름에
# **선형** 매핑한다(반지름 ∝ 힘의 크기). 터치 코드는 색(무엇을 했나)만 정하고 크기에는
# 관여하지 않는다 — 스친 굴절은 작은 링, 강슛은 큰 링. 그래서 드리블 터치도 배제하지 않는다
# (힘이 작으니 작은 링으로 저절로 눌린다). 링 폭·잔상시간도 같은 비율로 커진다.
#
# 주의 ①: Δv는 컨트롤 프레임 차분이라 그 프레임의 decimation 서브스텝 동안 작용한 중력·항력·
#   지면 바운스가 함께 섞인다(25fps 기준 중력분 ≈0.39 m/s). RING_IMPULSE_MIN이 그 규모의
#   잡음을 걸러낸다. substep_trajectory로 렌더하면 차분 간격이 dt_phys라 오염이 더 작다.
# 주의 ②: 한 프레임에 여러 선수가 공을 건드리면 Δv를 나눌 방법이 없어 같은 크기를 공유한다.
# 주의 ③: 파울 휘슬은 공 속도를 행정적으로 0으로 만든다(물리적 힘이 아님) → 온셋 프레임 제외.
RING_IMPULSE_MIN = 1.0                  # m/s — 이 미만 Δv는 링 없음(스친 접촉·프레임 내 잡음)
RING_R_MIN, RING_R_MAX = 3.0, 15.0      # px(근경 기준). Δv MIN..f2b_speed_max를 선형으로 채운다
RING_CARD_R = 7.0                       # 옐로카드 링 — 힘과 무관한 고정 반지름
RING_SPOT_R = 9.0                        # 퇴장 스팟 · 파울 링(고정)
RESTART_LABEL = {RK_KICKOFF: "KICKOFF", RK_THROWIN: "THROW-IN", RK_GOALKICK: "GOALKICK",
                 RK_CORNER: "CORNER", RK_FREEKICK: "FREEKICK", RK_PENALTY: "PENALTY",
                 RK_OFFSIDE: "OFFSIDE-FK", RK_GK_HOLD: "GK-HOLD"}
FOUL_LABEL = {FOUL_TACKLE: "FOUL(TACKLE)", FOUL_CHARGE: "FOUL(CHARGE)", FOUL_THROW: "FOUL(THROW)"}

COL_PITCH = (30, 74, 40); COL_LINE = (150, 185, 158); COL_HUD = (16, 20, 17)
COL_TEAM = ((208, 68, 58), (66, 118, 208))          # T0 적 / T1 청
COL_GK = ((255, 140, 120), (140, 190, 255))         # GK는 밝은 톤
COL_BALL = (248, 248, 248); COL_SHADOW = (16, 36, 22)
COL_TEXT = (225, 232, 226)
COL_ENGINE = (60, 230, 210)                         # 데드볼 내장 엔진 활성 표시(시안)


def _rgb_hex(color):
    """light RGB 팔레트에서 rich용 HEX를 파생한다(색상 정의의 단일 진실원천)."""
    return "#" + "".join(f"{channel:02x}" for channel in color)


def _disc(radius):
    """반지름 radius 원판의 (dy, dx) 오프셋 — 스탬핑용 사전계산."""
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    m = yy * yy + xx * xx <= radius * radius
    return yy[m], xx[m]


def _ring(radius, width=1.4):
    r = int(np.ceil(radius + width))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    d2 = yy * yy + xx * xx
    m = (d2 <= (radius + width) ** 2) & (d2 >= (radius - width) ** 2)
    return yy[m], xx[m]


_RING_CACHE = {}


def _ring_cached(radius, width=1.4):
    """반지름/폭을 0.5px 격자로 양자화해 링 스탬프를 캐시. 임펄스 링은 이벤트마다 반지름이
    달라 사전계산 테이블을 못 쓰므로, 재사용되는 이산 크기만 만들어 두고 돌려쓴다."""
    key = (round(float(radius) * 2.0) / 2.0, round(float(width) * 2.0) / 2.0)
    hit = _RING_CACHE.get(key)
    if hit is None:
        hit = _RING_CACHE[key] = _ring(max(key[0], 1.0), key[1])
    return hit


class Render:
    # ── 지오메트리(레이아웃 상수) ─────────────────────────────────────────
    # 키스톤(사다리꼴) 경사 투영: 위에서 살짝 기울여 보는 TV 카메라 뷰. 좌표 변환이 선형이라
    # 탑다운 대비 비용 증가 없음(픽셀당 곱셈 몇 회). 원경(화면 위)일수록 축소(_FAR_SCALE),
    # 세로는 _Y_FAR.._Y_NEAR로 압축. 원판 크기도 깊이 버킷으로 원근 단서 제공.
    _PPM = 6                      # px per meter (근경 기준)
    _HUD_H = 40                   # 상단 HUD 밴드 높이(px)
    _W, _H = 672, 448             # 총 캔버스(둘 다 /16 — 인코더 친화)
    _FAR_SCALE = 0.64             # 원경(윗쪽 터치라인) 가로 축소율 — 낮을수록 카메라가 누움
    _Y_FAR, _Y_NEAR = 96, 368     # 원경/근경 터치라인의 화면 y — 간격 좁을수록 누움

    def render_mp4(self, states, out_path=None, fps=25, mode="light", trail_decay=0.88,
                   event_ttl=8, title=None, verbose=True, batch_index=0,
                   dump_state=False, dump_events=False, state_stride=1,
                   frame_annotations=None, **rich_kwargs):
        """State 시퀀스를 mp4로 렌더. states: State 리스트 또는 T-선두축으로 스택된 State pytree.
        out_path=None이면 실행 경로 기준 ./replays/<unix초>/match.mp4에 저장(호출마다 새 폴더,
        같은 초 충돌 시 +1s). title은 HUD 라벨(팀/스타일 표기용).

        mode: "light"(기본) = 순수 numpy 탑다운 경량 렌더(수백 fps). "rich" = 원본 SOCCER 방송형
        렌더(render_rich.RichRenderer — 3D 카메라·관절 스켈레톤·관중석·미니맵 등, matplotlib 필요·저속).
        rich 전용 옵션(dynamic_cam·cam_zoom·feed_n·feed_secs·every·dpi 등)은 **rich_kwargs로 전달**된다.

        dump_state=True면 mp4 옆에 `state.jsonl`(1행=meta, 이후 **행당 1프레임** state — 스트리밍
        기록·부분 읽기/tail 가능), dump_events=True면 `events.txt`(경합승자·패스·슛·골·파울·카드·재개
        등 이벤트 발생 순간 로그)를 함께 쓴다. **이 트래킹/이벤트 덤프는 렌더 모드와 무관하게 동일
        코드경로(_stack_full→_write_*)로 생성**되므로 light·rich가 완전히 같은 산출물을 낸다.
        state_stride>1이면 프레임 행을 그만큼 솎는다(이벤트 로그는 항상 전 프레임 스캔이라 무손실).

        batch_index: **배치 차원 호환** — 학습용 vmap 롤아웃처럼 states에 배치 축이 있으면(규약: 시간-
        선두·배치-2번축 `(T, B, ...)`, 즉 lax.scan(시간)+vmap(env)의 자연 출력) 그 중 한 경기만 골라
        `(T, ...)`로 낮춰 렌더한다(기본은 첫 경기). 배치가 없으면 무시. light·rich·덤프 전부에 적용.
        반환: dump_* 미요청 시 mp4 경로(str, 하위호환), 요청 시 {"mp4","state","events"} dict."""
        if out_path is None:
            run_id = int(time.time())
            while os.path.exists(os.path.join("replays", str(run_id))):   # 'replay'→'replays': calib/replay.py 임포트 섀도잉 방지(B23)
                run_id += 1
            out_path = os.path.join("replays", str(run_id), "match.mp4")
        if os.path.dirname(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # 배치 차원 호환: (T, B, ...) 롤아웃이면 한 경기(batch_index)만 골라 (T, ...)로 — 이후 경로
        # (light·rich·덤프)는 전부 배치 없는 단일 궤적만 본다(단일 진입 정규화).
        states = self._normalize_states(states, batch_index, verbose=verbose)
        if frame_annotations is not None:
            frame_annotations = list(frame_annotations)
            frame_count = int(np.shape(states.ball_pos)[0])
            if len(frame_annotations) != frame_count:
                raise ValueError(
                    "frame_annotations length "
                    f"{len(frame_annotations)} != state frames {frame_count}"
                )

        if mode == "rich":
            # RichRenderer는 이 파일 하단에 통합됨(구 render_rich.py). matplotlib은 render()에서 지연 로드.
            # rich_kwargs를 생성자(카메라·해상도)와 render()(카메라 팬·피드) 인자로 분배.
            ctor_keys = ("azim", "elev", "dist", "focal", "figsize", "dpi", "trail_len", "player_scale")
            ctor_kw = {k: rich_kwargs[k] for k in ctor_keys if k in rich_kwargs}
            render_kw = {k: v for k, v in rich_kwargs.items() if k not in ctor_keys}
            RichRenderer(self, **ctor_kw).render(states, out_path, fps=fps, title=title,
                                                 verbose=verbose,
                                                 frame_annotations=frame_annotations,
                                                 **render_kw)
        else:
            self._render_light(states, out_path, fps=fps, trail_decay=trail_decay,
                               event_ttl=event_ttl, title=title, verbose=verbose,
                               frame_annotations=frame_annotations)

        # ── 트래킹/이벤트 덤프(모드 공유) ── mp4를 무엇으로 그렸든 동일 산출물.
        if not (dump_state or dump_events):
            return out_path
        outputs = {"mp4": out_path}
        d = os.path.dirname(out_path) or "."
        F = self._stack_full(states)                   # 전 필드 numpy 스택(1회, 두 덤프 공유)
        if dump_events:
            ev_path = os.path.join(d, "events.txt")
            n_ev = self._write_event_log(F, ev_path)
            outputs["events"] = ev_path
            if verbose:
                print(f"[render] 이벤트 {n_ev}건 → {ev_path}")
        if dump_state:
            js_path = os.path.join(d, "state.jsonl")
            nf = self._write_state_jsonl(F, js_path, state_stride)
            outputs["state"] = js_path
            if verbose:
                print(f"[render] state {nf}프레임(stride {state_stride}) → {js_path}")
        return outputs

    def substep_trajectory(self, substeps_stack, render_fps=None):
        """`collect_substeps=True` 롤아웃이 모은 서브스텝 스택 → render_fps 밀도의 (F, State) 궤적.
        결정 fps와 무관하게 ``1 / Engine.dt_phys``인 물리 fps까지 렌더하는 경로.

        입력: substeps_stack — step_env_array info["substeps"](decimation-선두 State)를 롤아웃 전 구간에
              스택한 (T, decimation, ...)-선두 State pytree.
        render_fps: 목표 렌더 프레임레이트. None이면 전 서브스텝(물리 fps 그대로 = 최대 부드러움).
              물리 fps에서 stride=round(physics_fps/render_fps)로 서브샘플(≤ 물리 fps, ≥ 1).
        반환: (F, ...)-선두 State pytree — 그대로 `render_mp4(traj, fps=render_fps)`에 전달.

        예)
          def step(c, _):
              o, st, k = c; k, ka, ks = jax.random.split(k, 3)
              o2, st2, _, _, info = env.step_env_array(ks, st, policy(o, ka), collect_substeps=True)
              return (o2, st2, k), info["substeps"]           # (decimation, State)
          (_, _, _), subs = lax.scan(step, init, None, length=T)   # subs: (T, decimation, State)
          traj = env.substep_trajectory(subs, render_fps=100)
          env.render_mp4(traj, fps=100, mode="rich")
        """
        physics_fps = int(round(1.0 / self.e_cfg.dt_phys))
        dec = int(self.e_cfg.decimation)
        leaves = jax.tree_util.tree_leaves(substeps_stack)
        T = int(leaves[0].shape[0])
        flat = jax.tree_util.tree_map(
            lambda x: x.reshape((T * dec,) + x.shape[2:]), substeps_stack)   # (T*dec, ...) 물리순 평탄화
        if render_fps is None:
            return flat
        stride = max(1, int(round(physics_fps / float(render_fps))))
        return jax.tree_util.tree_map(lambda x: x[::stride], flat)

    def _render_light(self, states, out_path, fps=25, trail_decay=0.88,
                      event_ttl=8, title=None, verbose=True,
                      frame_annotations=None):
        """경량 numpy 탑다운 렌더(원 render_mp4 본체) — mp4만 기록.
        프레임 합성은 전부 numpy(플롯 라이브러리 무사용)라 수백 fps로 인코딩 제한까지 닿는다.
        경로 해석·모드 분기·트래킹 덤프는 render_mp4가 담당(여긴 순수 프레임 래스터+인코딩)."""
        S = self._stack_states(states)
        T = len(S["ball_pos"])
        H, W, ppm = self._H, self._W, self._PPM
        hy = self.s_cfg.width / 2.0
        far, yfar, ynear = self._FAR_SCALE, float(self._Y_FAR), float(self._Y_NEAR)
        cxs = W // 2

        def to_px(wx, wy):
            """월드 → (화면y, 화면x, 깊이 0원경~1근경). 키스톤 경사 투영(선형 — 비용 동일)."""
            d = (hy - wy) / (2.0 * hy)
            return (int(yfar + d * (ynear - yfar)),
                    int(cxs + wx * ppm * (far + (1.0 - far) * d)), d)

        base = self._pitch_base(to_px)
        trail = np.zeros((H, W, 3), np.float32)
        d_player = [_disc(3.0 + 0.45 * i) for i in range(4)]   # 깊이 버킷별 원판(원근 단서)
        d_trail, d_ball_sh = _disc(1.2), _disc(2.0)
        BV = S["ball_vel"]                             # 임펄스 링용 — 프레임 차분 Δv의 원천
        imp_span = max(float(self.e_cfg.f2b_speed_max) - RING_IMPULSE_MIN, 1e-3)
        team = np.asarray(S["team_id"][0]).astype(int)
        gk = np.asarray(S["gk_indices"][0]).astype(int)

        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=None,
                                    pixelformat="yuv420p", macro_block_size=None,
                                    output_params=["-preset", "veryfast", "-crf", "26"])
        effects = []                                   # (ttl, kind, data) 활성 이벤트 플래시
        ticker = ""; goal_ttl = 0
        annotation_text = ""; annotation_ttl = 0
        prev_score = S["score"][0].copy(); prev_foul = 0
        prev_active = S["active_player"][0].copy(); prev_yellow = S["yellow_cards"][0].copy()
        bench = [i for i in range(self.N) if not prev_active[i]]   # 퇴장 선수 대기석(퇴장 순서 유지)
        d_bench = _disc(3.0)
        t_start = time.time()

        for f in range(T):
            P = S["player_pos"][f]; B = S["ball_pos"][f]
            stam = S["stamina"][f]; yellow = S["yellow_cards"][f]
            active = S["active_player"][f]; touch = S["touch"][f]
            clock = int(S["t"][f]) / float(self.control_fps)
            mmss = f"{int(clock // 60):02d}:{int(clock % 60):02d}"
            if frame_annotations is not None and frame_annotations[f]:
                annotation_text = str(frame_annotations[f])
                annotation_ttl = max(1, event_ttl * 2)

            # 트레일: 감쇠 후 현재 위치 스탬프(선수 팀색 저휘도, 공 백색)
            # 선수는 저휘도 스탬프(짧은 꼬리), 공은 풀휘도(궤적 확인용으로 상대적으로 길게) —
            # 같은 감쇠 버퍼 하나로 체감 길이만 차등(버퍼 추가 없음 = 비용 동일).
            trail *= trail_decay
            for i in range(self.N):
                if not active[i]:
                    continue
                cy, cx, _ = to_px(P[i, 0], P[i, 1])
                self._stamp(trail, cy, cx, d_trail, np.array(COL_TEAM[team[i]], np.float32) * 0.40)
            bcy, bcx, bd = to_px(B[0], B[1])
            self._stamp(trail, bcy, bcx, d_trail, np.array(COL_BALL, np.float32))

            frame = np.clip(base.astype(np.int16) + trail.astype(np.int16), 0, 255).astype(np.uint8)

            # 이 프레임에 공이 받은 힘: 질량 상수이므로 임펄스 ∝ ‖Δv‖. 파울 휘슬 프레임은
            # 규칙이 공 속도를 0으로 만드는 **행정 정지**(물리적 힘 아님)라 임펄스에서 제외한다.
            fk = int(S["foul_kind"][f])
            whistle = fk > 0 and prev_foul == 0
            dv = 0.0 if (f == 0 or whistle) else float(np.linalg.norm(BV[f] - BV[f - 1]))
            imp = float(np.clip((dv - RING_IMPULSE_MIN) / imp_span, 0.0, 1.0))   # 0..1 정규화 힘

            # 이벤트 온셋 수집 → 플래시/티커
            for i in range(self.N):
                code = int(touch[i])
                if code > 0 and active[i] and dv >= RING_IMPULSE_MIN:
                    # 반지름은 힘에 선형 비례. 깊이 스케일은 선수 원판·공과 같은 투영 계수를
                    # 곱해 원경에서 함께 작아지게 한다(원근 일관성).
                    _, _, dfrac = to_px(P[i, 0], P[i, 1])
                    r_px = ((RING_R_MIN + (RING_R_MAX - RING_R_MIN) * imp)
                            * (far + (1.0 - far) * dfrac))
                    effects.append([max(1, int(round(event_ttl * (0.7 + 0.8 * imp)))), "ring",
                                    (i, TOUCH_RING.get(code, COL_TEXT), r_px)])
                    ticker = f"{mmss} {TOUCH_LABEL.get(code, '?')} P{i}(T{team[i]}) {dv:.1f}m/s"
                if yellow[i] > prev_yellow[i]:
                    effects.append([event_ttl * 2, "ring", (i, (250, 210, 40), RING_CARD_R)])
                    ticker = f"{mmss} YELLOW P{i}(T{team[i]})"
                if prev_active[i] and not active[i]:
                    effects.append([event_ttl * 3, "spot", (to_px(P[i, 0], P[i, 1])[:2], (230, 40, 40))])
                    ticker = f"{mmss} SENT OFF P{i}(T{team[i]})"
                    bench.append(i)
            if fk > 0 and prev_foul == 0:
                a, v = int(S["foul_actor"][f]), int(S["foul_victim"][f])
                effects.append([event_ttl * 2, "foul", (a, v)])
                # 재터치 반칙 등 피해자 없는 파울은 v=-1 — 음수 인덱싱(P[-1]=마지막 선수) 방지
                ticker = (f"{mmss} {FOUL_LABEL.get(fk, 'FOUL')} P{a}" if v < 0
                          else f"{mmss} {FOUL_LABEL.get(fk, 'FOUL')} P{a}>P{v}")
            score = S["score"][f]
            if int(score[0]) != int(prev_score[0]) or int(score[1]) != int(prev_score[1]):
                goal_ttl = fps; ticker = f"{mmss} GOAL  {int(score[0])}-{int(score[1])}"
            prev_foul = fk; prev_score = score.copy()
            prev_active = active.copy(); prev_yellow = yellow.copy()

            # 이벤트 플래시 렌더
            alive_fx = []
            for fx in effects:
                fx[0] -= 1
                ttl, kind, data = fx
                if kind == "ring":
                    i, col, r_px = data
                    cy, cx, _ = to_px(P[i, 0], P[i, 1])
                    # 폭도 반지름에 따라 굵어진다(큰 힘일수록 진하게). ±1px 깜빡임은 유지.
                    self._stamp(frame, cy, cx,
                                _ring_cached(r_px + (1.0 if ttl % 2 else 0.0), 1.2 + 0.06 * r_px),
                                col)
                elif kind == "spot":
                    (cy, cx), col = data
                    self._stamp(frame, cy, cx, _ring_cached(RING_SPOT_R), col)
                elif kind == "foul":
                    a, v = data
                    if a >= 0:
                        ay, ax_, _ = to_px(P[a, 0], P[a, 1])
                        if v >= 0:                       # 피해자 없는 파울(재터치 등)은 링만
                            vy, vx, _ = to_px(P[v, 0], P[v, 1])
                            self._line(frame, ay, ax_, vy, vx, (255, 80, 80))
                        self._stamp(frame, ay, ax_, _ring_cached(RING_SPOT_R), (255, 80, 80))
                if fx[0] > 0:
                    alive_fx.append(fx)
            effects = alive_fx

            # 선수: 원판 + 스태미나 바 + 옐로 마커. 페인터 순서(원경→근경)로 근경이 위에 겹침.
            for i in np.argsort(-P[:, 1]):
                if not active[i]:
                    continue
                cy, cx, dfrac = to_px(P[i, 0], P[i, 1])
                col = (COL_GK if gk[i] == 1 else COL_TEAM)[team[i]]
                self._stamp(frame, cy, cx, d_player[min(3, int(dfrac * 4))], col)
                sw = int(round(8 * float(np.clip(stam[i], 0, 1))))
                sy, sx = cy - 7, cx - 4
                if 0 <= sy < self._H - 2 and 0 <= sx and sx + 8 < self._W:
                    frame[sy:sy + 2, sx:sx + 8] = (40, 44, 40)
                    if sw:
                        s = float(stam[i])
                        frame[sy:sy + 2, sx:sx + sw] = (int(220 * (1 - s) + 40), int(180 * s + 50), 55)
                if yellow[i] >= 1:
                    frame[max(cy - 9, 0):max(cy - 9, 0) + 3, min(cx + 5, self._W - 3):min(cx + 5, self._W - 3) + 3] = (250, 210, 40)

            # 퇴장 선수 대기석: 근경 터치라인 아래 줄지어 표시(흐린 팀색 + 레드카드 + 번호)
            for k, i in enumerate(bench):
                by, bx = self._Y_NEAR + 34, 30 + k * 20
                self._stamp(frame, by, bx, d_bench, tuple(int(c * 0.5) for c in COL_TEAM[team[i]]))
                frame[by - 9:by - 5, bx + 3:bx + 6] = (225, 45, 45)
                self._blit_text(frame, by + 5, bx - 6, f"P{i}", 1, (140, 148, 142))

            # 공: 그림자(지면) + 높이 투영 원판(z만큼 위로, 반지름도 z 비례. 깊이 스케일 일관).
            # 공중볼(z>0.5m)은 그림자-공 수직 연결선으로 높이를 즉시 읽히게.
            bz = float(B[2])
            lift = int(bz * ppm * (far + (1.0 - far) * bd))
            self._stamp(frame, bcy, bcx, d_ball_sh, COL_SHADOW)
            if bz > 0.5:
                self._line(frame, bcy, bcx, bcy - lift, bcx, (190, 195, 190))
            self._stamp(frame, bcy - lift, bcx, _disc(2.0 + min(bz, 6.0) * 0.5), COL_BALL)

            # HUD: 시계 · 스코어 · 소유 · 재개 · 티커 · (득점 오버레이)
            frame[:self._HUD_H] = COL_HUD
            poss = int(S["poss_team"][f])
            self._blit_text(frame, 4, 8, f"{mmss}  {int(score[0])} - {int(score[1])}", 2)
            if poss >= 0:
                frame[10:22, 118:130] = COL_TEAM[poss]
            rk = int(S["restart_kind"][f])
            if rk != RK_NONE and int(S["restart_t"][f]) > 0:
                self._blit_text(frame, 4, 150, RESTART_LABEL.get(rk, ""), 2, (255, 220, 120))
            # 데드볼 내장 엔진 활성(토글 ON & 데드볼) — 육안 식별용 프레임 테두리 + 배지.
            if self.e_cfg.deadball_engine and int(S["restart_t"][f]) > 0:
                b = 3
                frame[self._HUD_H:self._HUD_H + b, :] = COL_ENGINE
                frame[self._H - b:self._H, :] = COL_ENGINE
                frame[self._HUD_H:self._H, :b] = COL_ENGINE
                frame[self._HUD_H:self._H, self._W - b:self._W] = COL_ENGINE
                self._blit_text(frame, 22, 150, "ENGINE", 2, COL_ENGINE)
            if title:
                tw = self._text_mask(title, 2).shape[1]
                self._blit_text(frame, 4, max(self._W - tw - 8, 320), title, 2, (190, 200, 192))
            if ticker:
                self._blit_text(frame, 26, 8, ticker, 1)
            if goal_ttl > 0:
                goal_ttl -= 1
                self._blit_text(frame, self._H // 2 - 16, self._W // 2 - 70, "G O A L", 4, (255, 230, 90))
            if annotation_ttl > 0:
                annotation_ttl -= 1
                border = 4
                frame[self._HUD_H:self._HUD_H + border, :] = (245, 75, 75)
                frame[self._H - border:self._H, :] = (245, 75, 75)
                frame[self._HUD_H:self._H, :border] = (245, 75, 75)
                frame[self._HUD_H:self._H, self._W - border:self._W] = (245, 75, 75)
                self._blit_text(
                    frame,
                    self._HUD_H + 8,
                    10,
                    annotation_text,
                    2,
                    (255, 110, 110),
                )

            writer.append_data(frame)
        writer.close()
        if verbose:
            dt = time.time() - t_start
            print(f"[render] {T}프레임 → {out_path}  ({dt:.1f}s, {T / max(dt, 1e-9):.0f} fps)")

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────
    def _normalize_states(self, states, batch_index=0, verbose=True):
        """states → 스택된 numpy State pytree `(T, ...)`. **배치 차원이 있으면 한 경기만 선택**한다.
        규약: 시간-선두·배치-2번축 `(T, B, ...)`(lax.scan(시간)+vmap(env)의 자연 출력). 배치 판정은
        player_pos 랭크로 — 무배치 `(T,N,2)`=rank3, 배치 `(T,B,N,2)`=rank4 → 배치면 전 필드 `[:, bi]`.
        ⚠ `(B, T, ...)` 레이아웃은 이 규약과 달라 오작동하므로, 그 경우 호출 전에 `swapaxes(0,1)`로
        시간-선두로 바꾸거나 직접 한 경기를 슬라이스할 것.

        State 리스트/이미-스택 pytree 양쪽을 받는다(기존 _stack_* 규약과 동일 판정). 이후 렌더·덤프
        경로는 전부 이 정규화 결과(배치 없는 (T,...))만 소비한다."""
        if isinstance(states, (list, tuple)) and not hasattr(states, "_fields"):
            stacked = jax.tree_util.tree_map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *states)
        else:
            stacked = jax.tree_util.tree_map(np.asarray, states)
        pp = np.asarray(stacked.player_pos)
        if pp.ndim >= 4:                                    # (T, B, N, 2) — 배치 축(1) 존재
            B = pp.shape[1]
            bi = int(np.clip(batch_index, 0, B - 1))
            stacked = jax.tree_util.tree_map(lambda x: x[:, bi] if np.ndim(x) >= 2 else x, stacked)
            if verbose:
                print(f"[render] 배치 감지 (B={B}) → 경기 {bi}만 렌더")
        return stacked

    def _stack_states(self, states):
        """State 리스트/스택 pytree → 필요한 필드만 numpy dict (T, ...)로."""
        need = ("player_pos", "ball_pos", "ball_vel", "stamina", "yellow_cards", "active_player",
                "touch", "foul_kind", "foul_actor", "foul_victim", "restart_kind", "restart_t",
                "score", "poss_team", "t", "team_id", "gk_indices")
        # State 자체가 NamedTuple(=tuple)이므로 '스택된 단일 State'와 'State 리스트'를
        # _fields 유무로 구분한다(그냥 isinstance(tuple)로 가르면 전자가 리스트 분기로 오폭).
        if isinstance(states, (list, tuple)) and not hasattr(states, "_fields"):
            stacked = jax.tree_util.tree_map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *states)
        else:
            stacked = jax.tree_util.tree_map(np.asarray, states)
        return {k: np.asarray(getattr(stacked, k)) for k in need}

    def _stack_full(self, states):
        """State 리스트/스택 pytree → 모든 State 필드를 numpy (T,...) dict로. 덤프(json/이벤트)용 —
        mp4 핫패스의 부분집합 스택(_stack_states)과 달리 전 필드를 보존한다."""
        if isinstance(states, (list, tuple)) and not hasattr(states, "_fields"):
            stacked = jax.tree_util.tree_map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *states)
        else:
            stacked = jax.tree_util.tree_map(np.asarray, states)
        return {k: np.asarray(getattr(stacked, k)) for k in stacked._fields}

    def _write_event_log(self, F, path):
        """이벤트 발생 순간을 텍스트 로그로 — 전 프레임 스캔(무손실). 각 줄:
        `f=NNNNNN MM:SS  EVENT  detail`. 이벤트: 터치(경합 승자 — PASS/SHOOT/헤더/DRIBBLE/TACKLE/
        INTERCEPT/DEFLECT/GK-CATCH/PARRY), 골, 파울(actor>victim), 옐로/퇴장, 재개 온셋(킥오프/스로인/
        코너/골킥/프리킥·간접/페널티/GK-홀드). 반환 기록한 이벤트 수."""
        T = len(F["t"]); N = self.N
        team = F["team_id"][0].astype(int); fps = float(self.control_fps)
        lines, n = [], 0

        def clock(f):
            sec = int(F["t"][f]) / fps
            return f"{int(sec // 60):02d}:{int(sec % 60):02d}"

        def emit(f, ev, detail=""):
            nonlocal n
            lines.append(f"f={f:06d} {clock(f)}  {ev:<12s} {detail}")
            n += 1

        for f in range(T):
            touch = F["touch"][f]; active = F["active_player"][f]
            for i in range(N):
                c = int(touch[i])
                if c > 0 and active[i]:
                    lt = int(F["last_touch_team"][f])
                    extra = "→GK-HOLD" if c == TOUCH_GK_CATCH and int(F["restart_kind"][f]) == RK_GK_HOLD else ""
                    emit(f, TOUCH_LABEL.get(c, f"TOUCH{c}"), f"P{i}(T{team[i]}) last=T{lt} {extra}".rstrip())
            if f > 0:
                sc, psc = F["score"][f], F["score"][f - 1]
                if int(sc[0]) != int(psc[0]) or int(sc[1]) != int(psc[1]):
                    scorer = 0 if int(sc[0]) > int(psc[0]) else 1
                    emit(f, "GOAL", f"T{scorer} scores → {int(sc[0])}-{int(sc[1])}")
                fk, pfk = int(F["foul_kind"][f]), int(F["foul_kind"][f - 1])
                if fk > 0 and pfk == 0:
                    a, v = int(F["foul_actor"][f]), int(F["foul_victim"][f])
                    who = f"P{a}(T{team[a]})" if a >= 0 else "P?"
                    on = f" on P{v}(T{team[v]})" if v >= 0 else ""
                    emit(f, "FOUL", f"{FOUL_LABEL.get(fk, 'FOUL')} {who}{on}")
                yc, pyc = F["yellow_cards"][f], F["yellow_cards"][f - 1]
                for i in range(N):
                    if int(yc[i]) > int(pyc[i]):
                        emit(f, "YELLOW", f"P{i}(T{team[i]})")
                    if bool(F["active_player"][f - 1][i]) and not bool(active[i]):
                        emit(f, "SENT-OFF", f"P{i}(T{team[i]})")
                rk, prk = int(F["restart_kind"][f]), int(F["restart_kind"][f - 1])
                if rk != prk and rk != RK_NONE:
                    idfk = " (indirect)" if bool(F["restart_indirect"][f]) else ""
                    tm = int(F["restart_team"][f])
                    emit(f, "RESTART", f"{RESTART_LABEL.get(rk, f'RK{rk}')} T{tm}{idfk}")

        with open(path, "w") as fh:
            fh.write(f"# UOS-FootballMARL-Env match event log — {T} frames @ {fps:.0f}fps, {n} events\n")
            fh.write(f"# teams: T0={self.n_agents} players, T1={self.n_opponents} players\n")
            fh.write("\n".join(lines) + "\n")
        return n

    def _write_state_jsonl(self, F, path, stride=1):
        """프레임별 전체 state를 JSONL로 — 1행 = {"meta": ...}(정적 per-player·스키마), 이후
        **행당 1프레임**(동적 필드: 공·선수·소유·재개·스코어·터치·파울). 스트리밍 기록이라 풀경기
        (135k행)도 메모리 상주 없이 쓰고, 소비 측은 부분 읽기/tail/grep이 가능하다.
        float는 3자리 반올림. 반환 기록한 프레임 수."""
        T = len(F["t"]); N = self.N
        r3 = lambda a: np.round(np.asarray(a, np.float64), 3).tolist()
        meta = {
            "n_agents": self.n_agents, "n_opponents": self.n_opponents, "N": N,
            "control_fps": self.control_fps, "num_frames_total": T, "stride": stride,
            "pitch": {"length": self.s_cfg.length, "width": self.s_cfg.width,
                      "goal_width": self.goal_w, "goal_height": self.goal_h},
            "agents": [str(p.id) for p in self.players],
            "team_id": F["team_id"][0].astype(int).tolist(),
            "gk": F["gk_indices"][0].astype(int).tolist(),
            "vmax": r3(F["vmax"][0]), "reach_z": r3(F["reach_z"][0]), "head_z": r3(F["head_z"][0]),
            "restart_kinds": {v: k for k, v in RESTART_LABEL.items()},
            "touch_codes": {TOUCH_LABEL[k]: k for k in TOUCH_LABEL},
            "frame_schema": "ball{pos,vel,spin,state} poss last_touch score "
                            "restart{kind,t,team,taker,indirect} players{pos,vel,facing,stamina,"
                            "active,yellow,cooldown,offside} touch foul{kind,actor,victim}",
        }
        n = 0
        with open(path, "w") as fh:
            fh.write(json.dumps({"meta": meta}, separators=(",", ":")) + "\n")
            for f in range(0, T, stride):
                row = {
                    "f": int(f), "t": int(F["t"][f]), "clock": round(int(F["t"][f]) / self.control_fps, 3),
                    "ball": {"pos": r3(F["ball_pos"][f]), "vel": r3(F["ball_vel"][f]),
                             "spin": r3(F["ball_spin"][f]), "state": int(F["ball_state"][f])},
                    "poss": int(F["poss_team"][f]), "last_touch": int(F["last_touch_team"][f]),
                    "score": F["score"][f].astype(int).tolist(),
                    "restart": {"kind": int(F["restart_kind"][f]), "t": int(F["restart_t"][f]),
                                "team": int(F["restart_team"][f]), "taker": int(F["pending_taker"][f]),
                                "indirect": bool(F["restart_indirect"][f])},
                    "players": {"pos": r3(F["player_pos"][f]), "vel": r3(F["player_vel"][f]),
                                "facing": r3(F["player_facing"][f]), "stamina": r3(F["stamina"][f]),
                                "active": F["active_player"][f].astype(int).tolist(),
                                "yellow": F["yellow_cards"][f].astype(int).tolist(),
                                "cooldown": r3(F["cooldown"][f]),
                                "offside": F["offside_flag"][f].astype(int).tolist()},
                    "touch": F["touch"][f].astype(int).tolist(),
                    "foul": {"kind": int(F["foul_kind"][f]), "actor": int(F["foul_actor"][f]),
                             "victim": int(F["foul_victim"][f])},
                }
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
                n += 1
        return n

    def _stamp(self, img, cy, cx, disc, color):
        dy, dx = disc
        ys = dy + cy; xs = dx + cx
        m = (ys >= 0) & (ys < img.shape[0]) & (xs >= 0) & (xs < img.shape[1])
        img[ys[m], xs[m]] = color

    def _line(self, img, y1, x1, y2, x2, color):
        n = max(abs(y2 - y1), abs(x2 - x1), 1)
        ys = np.linspace(y1, y2, n).astype(int); xs = np.linspace(x1, x2, n).astype(int)
        m = (ys >= 0) & (ys < img.shape[0]) & (xs >= 0) & (xs < img.shape[1])
        img[ys[m], xs[m]] = color

    def _pitch_base(self, to_px):
        """정적 피치 배경 1회 래스터 — 키스톤 투영 좌표로 잔디 사다리꼴·라인·박스·센터서클·골문.
        전부 세그먼트(짧은 직선)로 그려 투영이 바뀌어도 이 함수는 그대로 동작한다."""
        img = np.zeros((self._H, self._W, 3), np.uint8); img[:] = (13, 24, 16)   # 장외 어두운 톤
        img[:self._HUD_H] = COL_HUD
        hx, hy = self.s_cfg.length / 2.0, self.s_cfg.width / 2.0
        cxs = self._W // 2
        yfar, xfar, _ = to_px(hx, hy)
        ynear, xnear, _ = to_px(hx, -hy)
        for y in range(yfar, ynear + 1):                       # 잔디: 행별 사다리꼴 폭 채우기
            u = (y - yfar) / max(ynear - yfar, 1)
            halfw = int((xfar - cxs) + u * ((xnear - cxs) - (xfar - cxs)))
            img[y, cxs - halfw:cxs + halfw + 1] = COL_PITCH

        def seg(w1, w2, color=COL_LINE):
            y1, x1, _ = to_px(*w1); y2, x2, _ = to_px(*w2)
            self._line(img, y1, x1, y2, x2, color)

        seg((-hx, hy), (hx, hy)); seg((hx, hy), (hx, -hy))     # 외곽 사다리꼴
        seg((hx, -hy), (-hx, -hy)); seg((-hx, -hy), (-hx, hy))
        seg((0, hy), (0, -hy))                                  # 하프웨이
        th = np.linspace(0, 2 * np.pi, 200)
        center_r = self.s_cfg.center_circle_radius
        pts = [to_px(center_r * np.cos(a), center_r * np.sin(a)) for a in th]
        for (y1, x1, _), (y2, x2, _) in zip(pts[:-1], pts[1:]):
            self._line(img, y1, x1, y2, x2, COL_LINE)
        pl = self.s_cfg.penalty_area_length; pw2 = self.s_cfg.penalty_area_width / 2.0
        gw2 = self.s_cfg.goal_width / 2.0; gh = self.s_cfg.goal_height
        far = self._FAR_SCALE
        for side in (-1, 1):
            xg, xb = side * hx, side * (hx - pl)
            seg((xg, pw2), (xb, pw2)); seg((xb, pw2), (xb, -pw2)); seg((xb, -pw2), (xg, -pw2))
            # 골대 3D 프레임: 골라인(지면) + 양 포스트(높이 goal_height 투영) + 크로스바.
            # 정적이라 사전계산에 포함 — 프레임당 비용 0.
            gy1, gx1, d1 = to_px(xg, gw2); gy2, gx2, d2 = to_px(xg, -gw2)
            self._line(img, gy1, gx1, gy2, gx2, (235, 235, 235))
            h1 = int(gh * self._PPM * (far + (1.0 - far) * d1))
            h2 = int(gh * self._PPM * (far + (1.0 - far) * d2))
            for off in (0, side):                               # 포스트/크로스바 2px 두께
                self._line(img, gy1, gx1 + off, gy1 - h1, gx1 + off, (250, 250, 250))
                self._line(img, gy2, gx2 + off, gy2 - h2, gx2 + off, (250, 250, 250))
                self._line(img, gy1 - h1 - (0 if off == 0 else 1), gx1,
                           gy2 - h2 - (0 if off == 0 else 1), gx2, (250, 250, 250))
        return img

    _FONT = None
    _TEXT_CACHE = {}

    def _text_mask(self, text, scale=1):
        """PIL 기본 비트맵 폰트로 텍스트 마스크를 1회 렌더 후 캐시(프레임당 텍스트 비용 ≈ 복사)."""
        key = (text, scale)
        mask = Render._TEXT_CACHE.get(key)
        if mask is None:
            if Render._FONT is None:
                Render._FONT = ImageFont.load_default()
            l, t, r, b = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=Render._FONT)
            im = Image.new("L", (max(r - l, 1) + 2, max(b - t, 1) + 2), 0)
            ImageDraw.Draw(im).text((1 - l, 1 - t), text, fill=255, font=Render._FONT)
            if scale > 1:
                im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
            mask = np.asarray(im) > 96
            if len(Render._TEXT_CACHE) > 512:                # 티커 다양성으로 무한증식 방지
                Render._TEXT_CACHE.clear()
            Render._TEXT_CACHE[key] = mask
        return mask

    def _blit_text(self, img, y, x, text, scale=1, color=COL_TEXT):
        mask = self._text_mask(text, scale)
        h, w = mask.shape
        y2, x2 = min(y + h, img.shape[0]), min(x + w, img.shape[1])
        if y2 <= y or x2 <= x:
            return
        region = img[y:y2, x:x2]
        region[mask[:y2 - y, :x2 - x]] = color


# ============================================================================
# rich 모드 (구 render_rich.py 통합) — 방송형 matplotlib 3D 렌더.
# matplotlib 심볼은 위 _ensure_mpl()이 지연 로드하며, RichRenderer.render() 진입 시 호출.
# 팀색 TEAM/GKC는 light 팔레트(COL_TEAM/COL_GK)와 동일 RGB(모드 간 색 일관).
# ============================================================================

TEAM = tuple(_rgb_hex(color) for color in COL_TEAM)
GKC = tuple(_rgb_hex(color) for color in COL_GK)
PASS_COL = (0.10, 0.88, 0.84); SHOOT_COL = (1.00, 0.69, 0.13)
TACKLE_COL = (1.00, 0.95, 0.30); INTERCEPT_COL = (1.00, 0.62, 0.30)
CATCH_COL = (0.30, 1.00, 0.85); MISS_COL = (0.62, 0.66, 0.72)

# 임펄스 링(rich) — light와 **같은 힘 정의·같은 색**을 쓰고 크기 단위만 다르다. rich는 3D 투영이라
# 반지름을 픽셀이 아닌 **미터**로 주면 원근이 투영에서 저절로 처리된다(지면 링 = carrier_ring 계열).
TOUCH_RING_RGB = {code: tuple(c / 255.0 for c in col) for code, col in TOUCH_RING.items()}
RING_R_MIN_M, RING_R_MAX_M = 0.7, 3.5     # m — Δv MIN..f2b_speed_max를 선형으로 채운다(비교: 소유링 1.5m)

# 프레임 뷰가 참조하는 State 필드(전부 numpy로 스택). 원본 Frame 속성명으로 매핑된다.
_NEED = ("player_pos", "player_vel", "player_facing", "touch", "ball_pos", "ball_vel",
         "ball_spin", "ball_state", "poss_team", "last_touch_team", "restart_kind",
         "restart_t", "restart_team", "pending_taker", "attack_dir", "offside_flag",
         "pass_t", "stamina", "yellow_cards", "active_player", "foul_kind",
         "foul_actor", "foul_victim", "score", "t", "team_id", "gk_indices")


def _stam_color(s):
    s = float(np.clip(s, 0, 1))
    return (np.clip(2 * (1 - s), 0, 1), np.clip(1.4 * s, 0, 1), 0.18)


class _Cam:
    def __init__(self, target, azim, elev, dist, focal):
        self.t = np.asarray(target, float); self.focal = focal
        a, e = np.radians(azim), np.radians(elev)
        d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        self.pos = self.t + dist * d
        f = self.t - self.pos; f /= np.linalg.norm(f)
        r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
        self.R = np.stack([r, u, f])

    def proj(self, pts):
        cam = (np.asarray(pts, float) - self.pos) @ self.R.T
        Z = np.clip(cam[..., 2], 1e-3, None)
        return np.stack([self.focal * cam[..., 0] / Z, self.focal * cam[..., 1] / Z], -1)

    def unproj_ground(self, uv):
        """uv 점들을 지면(z=0)으로 역투영 — proj의 역(z=0 평면 한정). 카메라 시야영역 계산용."""
        uv = np.asarray(uv, float)
        dir_cam = np.concatenate([uv / self.focal, np.ones(uv.shape[:-1] + (1,))], -1)
        dir_w = dir_cam @ self.R
        s = -self.pos[2] / np.where(np.abs(dir_w[..., 2]) < 1e-9, 1e-9, dir_w[..., 2])
        return self.pos[:2] + dir_w[..., :2] * s[..., None]


class RichRenderer:
    """State 시퀀스 → 방송형 mp4. `RichRenderer(env).render(states, out_path, fps=...)`."""
    GRASS = ("#2f8f3e", "#37a449"); LINE = "#f4f4f4"; BG = "#0c1622"
    APRON = "#1c2a17"; STAND = "#3a4150"

    def __init__(self, env, *, azim=-90.0, elev=42.0, dist=150.0, focal=820.0,
                 figsize=(12.8, 7.2), dpi=150, trail_len=18, player_scale=1.7):
        # figsize×dpi = 1920×1080(Full HD, 짝수 해상도). 느리면 dpi를 낮춰 호출.
        self.env = env
        self.E = env.e_cfg
        st = env.reset(jax.random.PRNGKey(0))[1]          # 정적 속성(team_id·gk)만 취함
        self.team = np.asarray(st.team_id).astype(int); self.gk = np.asarray(st.gk_indices).astype(int)
        self.N = env.N
        self.L = env.s_cfg.length; self.W = env.s_cfg.width
        self.bench_x0 = (-24.0, 4.0)
        self.gw = env.s_cfg.goal_width; self.ch = env.s_cfg.goal_height
        self.control_fps = float(env.control_fps)
        self.cam = _Cam([0, 0, 0], azim, elev, dist, focal)
        self.figsize, self.dpi = figsize, dpi
        self.trail_len, self.player_scale = trail_len, player_scale
        self.pcol = [GKC[self.team[i]] if self.gk[i] else TEAM[self.team[i]] for i in range(self.N)]
        self.pnum = [""] * self.N
        for tm in (0, 1):
            c = 1
            for i in np.where(self.team == tm)[0]:
                self.pnum[i] = "GK" if self.gk[i] else str(c); c += 0 if self.gk[i] else 1
        self._precompute_static()

    # ── State 시퀀스 → 원본 Frame 호환 뷰 ────────────────────────────────
    def _frames_from_states(self, states):
        """State 리스트/스택 pytree → 원본 Frame과 동일 속성의 프레임 리스트.
        `scored`는 State에 없으므로 score 차분의 **온셋 프레임에만** 득점팀을, 그 외엔 -1을 둔다
        (원본 골 검출 `scored>=0 & 직전<0`이 그대로 성립).
        State→numpy 스택은 light/덤프와 **공유 헬퍼** `Render._stack_full`을 재사용(중복 제거)."""
        F = self.env._stack_full(states)     # {필드: (T,...) numpy} — 전 State 필드(≥_NEED) 공유 스택
        T = len(F["t"])
        score = F["score"].astype(int)
        scored = np.full(T, -1, int)
        for t in range(1, T):
            if score[t][0] > score[t - 1][0]:
                scored[t] = 0
            elif score[t][1] > score[t - 1][1]:
                scored[t] = 1
        frames = []
        for t in range(T):
            frames.append(SimpleNamespace(
                P=F["player_pos"][t], facing=F["player_facing"][t],
                speed=np.linalg.norm(F["player_vel"][t], axis=1), touch=F["touch"][t],
                ball=F["ball_pos"][t], poss_team=int(F["poss_team"][t]), scored=int(scored[t]),
                restart_kind=int(F["restart_kind"][t]), last_team=int(F["last_touch_team"][t]),
                stamina=F["stamina"][t], foul_kind=int(F["foul_kind"][t]),
                ball_spin=F["ball_spin"][t], ball_vel=F["ball_vel"][t],
                foul_actor=int(F["foul_actor"][t]), foul_victim=int(F["foul_victim"][t]),
                attack_dir=F["attack_dir"][t], offside_flag=F["offside_flag"][t].astype(bool),
                sp_taker=int(F["pending_taker"][t]), ball_state=int(F["ball_state"][t]),
                restart_t=int(F["restart_t"][t]), restart_team=int(F["restart_team"][t]),
                pass_t=int(F["pass_t"][t]), yellow_cards=F["yellow_cards"][t],
                on_pitch=F["active_player"][t].astype(bool), clock=float(F["t"][t]) / self.control_fps))
        return frames

    def _precompute_static(self):
        L, W = self.L, self.W; hx, hy = L / 2, W / 2
        lines = []

        def rect(x0, x1, y0, y1):
            lines.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], float))
        rect(-hx, hx, -hy, hy); lines.append(np.array([[0, -hy], [0, hy]], float))
        center_r = self.env.s_cfg.center_circle_radius
        th = np.linspace(0, 2 * np.pi, 60)
        lines.append(np.stack([center_r * np.cos(th), center_r * np.sin(th)], 1))
        pen_len = self.env.s_cfg.penalty_area_length
        pen_hw = self.env.s_cfg.penalty_area_width / 2.0
        goal_len = self.env.s_cfg.goal_area_length
        goal_hw = self.env.s_cfg.goal_area_width / 2.0
        for s in (-1, 1):
            gx = s * hx
            rect(gx, gx - s * pen_len, -pen_hw, pen_hw)
            rect(gx, gx - s * goal_len, -goal_hw, goal_hw)
        self.lines = lines
        self.stripes = []; xe = np.linspace(-hx, hx, 13)
        for i in range(12):
            self.stripes.append((np.array([[xe[i], -hy], [xe[i + 1], -hy], [xe[i + 1], hy], [xe[i], hy]], float), self.GRASS[i % 2]))
        # 골대: 흰 프레임(포스트 2 + 크로스바) + 뒤로 뻗는 네트(측면 프로파일·백패널·바닥, 그물 메시).
        self.goal_segs = []
        self.net_segs = []
        g = self.gw / 2; c = self.ch; D = 2.4; tb = 0.55 * D    # D=네트 깊이, tb=상단이 뒤로 가는 거리

        def _profile(gx, s, y, u):
            if u <= 0.5:
                a = u / 0.5; return [gx + s * tb * a, y, c]
            a = (u - 0.5) / 0.5; return [gx + s * tb + s * (D - tb) * a, y, c * (1 - a)]
        for s in (-1, 1):
            gx = s * hx
            self.goal_segs += [np.array([[gx, -g, 0], [gx, -g, c]]), np.array([[gx, g, 0], [gx, g, c]]),
                               np.array([[gx, -g, c], [gx, g, c]])]
            for y in np.linspace(-g, g, 7):
                self.net_segs += [np.array([_profile(gx, s, y, 0.0), _profile(gx, s, y, 0.5)]),
                                  np.array([_profile(gx, s, y, 0.5), _profile(gx, s, y, 1.0)]),
                                  np.array([[gx, y, 0], _profile(gx, s, y, 1.0)])]
            for u in np.linspace(0.0, 1.0, 6):
                self.net_segs.append(np.array([_profile(gx, s, -g, u), _profile(gx, s, g, u)]))
            self.net_segs.append(np.array([[gx + s * D, -g, 0], [gx + s * D, g, 0]]))
        ap, sd, sh = 2.5, 6.0, 4.0
        axi, ayi = hx + ap, hy + ap; axo, ayo = axi + sd, ayi + sd
        pc = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        ic = [(-axi, -ayi), (axi, -ayi), (axi, ayi), (-axi, ayi)]
        oc = [(-axo, -ayo), (axo, -ayo), (axo, ayo), (-axo, ayo)]
        self.stand_quads = []
        for k in range(4):
            a, b = pc[k], pc[(k + 1) % 4]; d, e = ic[(k + 1) % 4], ic[k]
            self.stand_quads.append((np.array([[a[0], a[1], 0], [b[0], b[1], 0], [d[0], d[1], 0], [e[0], e[1], 0]]), self.APRON))
        for k in range(4):
            a, b = ic[k], ic[(k + 1) % 4]; d, e = oc[(k + 1) % 4], oc[k]
            self.stand_quads.append((np.array([[a[0], a[1], 0], [b[0], b[1], 0], [d[0], d[1], sh], [e[0], e[1], sh]]), self.STAND))
        cor = [[x, y, z] for x in (-axo, axo) for y in (-ayo, ayo) for z in (0, sh)]
        uv = self.cam.proj(np.array(cor)); pad = 0.02 * (uv[:, 0].max() - uv[:, 0].min())
        self.xlim = (uv[:, 0].min() - pad, uv[:, 0].max() + pad)
        self.ylim = (uv[:, 1].min() - pad, uv[:, 1].max() + pad)

    def _skeletons(self, P, FAC, ph, amp, kick, head):
        N = self.N; X, Y = P[:, 0], P[:, 1]
        c, s = np.cos(FAC), np.sin(FAC); sc = self.player_scale

        def W(xf, yl, z):
            xf = np.broadcast_to(np.asarray(xf, float), (N,)) * sc
            yl = np.broadcast_to(np.asarray(yl, float), (N,)) * sc
            z = np.broadcast_to(np.asarray(z, float), (N,)) * sc
            return np.stack([X + xf * c - yl * s, Y + xf * s + yl * c, z], 1)
        H = head
        amp = amp * (1 - 0.6 * H); lift = 0.05 + 0.10 * amp
        swL = amp * np.sin(ph); swR = amp * np.sin(ph + np.pi)
        fzL = np.maximum(0, lift * np.sin(ph)); fzR = np.maximum(0, lift * np.sin(ph + np.pi))
        kf = kick * (1 - H)
        swR = (1 - kf) * swR + kf * 0.34; fzR = (1 - kf) * fzR + kf * 0.22
        # 2등신 비율: 짧은 몸통·다리 + 큰 머리(heads 스캐터). lead=머리 전진, hipz/shz/hdz=엉덩이/어깨/머리 높이.
        lead = 0.35 * H; hipz, shz, hdz = 0.55, 0.8, 1.0
        hipL, hipR = W(0, 0.10, hipz), W(0, -0.10, hipz)
        shL, shR = W(lead, 0.15, shz - 0.05 * H), W(lead, -0.15, shz - 0.05 * H)
        neck = W(lead, 0, shz + 0.05 - 0.05 * H); headp = W(lead + 0.06 + 0.30 * H, 0, hdz - 0.10 * H)

        def leg(yl, sw, fz):
            return W(sw * 0.55 + 0.04, yl, hipz * 0.5 + fz * 0.4), W(sw, yl, fz)

        def arm(yl, sw, rz):
            return W(sw * 0.6 + 0.10 * rz, yl, shz - 0.22 + 0.30 * rz), W(sw + 0.04 * rz, yl, shz - 0.42 + 0.55 * rz)
        knL, ftL = leg(0.10, swL, fzL); knR, ftR = leg(-0.10, swR, fzR)
        elL, haL = arm(0.15, swR * 0.6, H); elR, haR = arm(-0.15, swL * 0.6, H)
        pairs = [(hipL, shL), (hipR, shR), (shL, shR), (neck, headp),
                 (hipL, knL), (knL, ftL), (hipR, knR), (knR, ftR),
                 (shL, elL), (elL, haL), (shR, elR), (elR, haR)]
        segs = np.stack([np.stack([p0, p1], 1) for p0, p1 in pairs], 1)
        return segs, headp, ftR

    def _precompute(self, frames):
        T, N = len(frames), self.N
        ph = np.zeros(N); kf = np.zeros(N); kh = np.zeros(N); kt = np.zeros(N, int)
        PH = np.zeros((T, N)); AMP = np.zeros((T, N)); KF = np.zeros((T, N)); KH = np.zeros((T, N)); KT = np.zeros((T, N), int)
        score = np.zeros((T, TEAM_COUNT), int)
        poss = np.zeros((T, TEAM_COUNT))
        shots = np.zeros((T, TEAM_COUNT), int)
        passes = np.zeros((T, TEAM_COUNT), int)
        owner = np.full(T, NO_PLAYER, int)
        pc = np.zeros(TEAM_COUNT)
        sc = np.zeros(TEAM_COUNT, int)
        sh = np.zeros(TEAM_COUNT, int)
        pa = np.zeros(TEAM_COUNT, int)
        foot = {TOUCH_PASS, TOUCH_SHOOT, TOUCH_TACKLE, TOUCH_DRIBBLE, TOUCH_INTERCEPT}
        head = {TOUCH_PASS_HEAD, TOUCH_SHOOT_HEAD}
        shotc = {TOUCH_SHOOT, TOUCH_SHOOT_HEAD}; passc = {TOUCH_PASS, TOUCH_PASS_HEAD}
        DECAY = 5.0
        for t, fr in enumerate(frames):
            spd = np.clip(fr.speed / 8.0, 0, 1); ph = ph + 0.5 + 1.8 * spd
            tc = fr.touch
            isf = np.isin(tc, list(foot)); ish = np.isin(tc, list(head))
            kf = np.where(isf, DECAY, np.maximum(0.0, kf - 1)); kt = np.where(isf | ish, tc, kt)
            kh = np.where(ish, DECAY, np.maximum(0.0, kh - 1))
            PH[t] = ph; AMP[t] = 0.06 + 0.18 * spd; KF[t] = np.clip(kf / DECAY, 0, 1)
            KH[t] = np.clip(kh / DECAY, 0, 1); KT[t] = kt
            if fr.poss_team >= 0: pc[fr.poss_team] += 1
            for tm in (0, 1):
                m = (self.team == tm)
                sh[tm] += int(np.sum(m & np.isin(tc, list(shotc)))); pa[tm] += int(np.sum(m & np.isin(tc, list(passc))))
            if t > 0 and fr.scored >= 0 and frames[t - 1].scored < 0: sc[fr.scored] += 1
            tot = max(1.0, pc.sum())
            score[t] = sc; shots[t] = sh; passes[t] = pa; poss[t] = pc / tot
            d = np.linalg.norm(fr.P - fr.ball[:2][None, :], axis=1)
            ni = int(np.argmin(d)); owner[t] = ni if d[ni] < 2.0 else -1
        # 킥 비행 구간: 발사터치(PASS/SHOOT/헤더)~다음터치/정지까지. 프레임별 비행 시작 프레임(KSEG).
        launch_codes = [TOUCH_PASS, TOUCH_SHOOT, TOUCH_PASS_HEAD, TOUCH_SHOOT_HEAD]
        KSEG = np.full(T, -1, int)
        seg = -1
        for t, fr in enumerate(frames):
            tc = fr.touch
            launched = bool(np.any(np.isin(tc, launch_codes)))
            touched = bool(np.any(tc > 0))
            if launched:
                seg = t
            elif seg >= 0 and t > seg and touched:
                seg = -1
            if seg >= 0:
                spd = float(np.linalg.norm(fr.ball_vel)) if fr.ball_vel is not None else 1.0
                if (t - seg > 2 and spd < 0.6) or fr.restart_kind != frames[seg].restart_kind:
                    seg = -1
            KSEG[t] = seg
        return dict(PH=PH, AMP=AMP, KF=KF, KH=KH, KT=KT, score=score, poss=poss,
                    shots=shots, passes=passes, owner=owner, KSEG=KSEG, events=self._events(frames))

    def _first_touch_after(self, frames, t0, look=60):
        for t2 in range(t0 + 1, min(len(frames), t0 + look)):
            chg = (frames[t2].touch != frames[t2 - 1].touch) & (frames[t2].touch > 0)
            cand = np.where(chg)[0]
            if len(cand):
                bxy = frames[t2].ball[:2]
                return int(cand[np.argmin(np.linalg.norm(frames[t2].P[cand] - bxy[None, :], axis=1))])
        return -1

    def _events(self, frames):
        T = len(frames); out = [[] for _ in range(T)]
        # 재개 코드는 constants.RK_*가 단일 진실원천 — 리터럴 정수를 쓰면 코드 재배치 시 조용히 어긋난다.
        SP = {RK_THROWIN: ("THROW-IN", (0.80, 0.86, 0.55)), RK_GOALKICK: ("GOAL KICK", (0.55, 0.80, 0.95)),
              RK_CORNER: ("CORNER", (0.55, 0.90, 0.75)), RK_FREEKICK: ("FREE KICK", (0.95, 0.80, 0.45)),
              RK_KICKOFF: ("KICK-OFF", (0.92, 0.92, 0.96)), RK_PENALTY: ("PENALTY!", (1.0, 0.45, 0.45)),
              RK_OFFSIDE: ("OFFSIDE", (1.0, 0.78, 0.30)), RK_GK_HOLD: ("GK HOLD", (0.55, 0.90, 0.98))}
        DRIB_COL = (0.55, 0.80, 0.98)
        mrg = max(8, int(round(self.control_fps * 0.8)))
        last_drib = {}
        last_loose = -10 ** 9
        for t in range(1, T):
            fr, pv = frames[t], frames[t - 1]; tc = fr.touch
            if fr.restart_kind > 0 and fr.restart_kind != pv.restart_kind and fr.restart_kind in SP:
                nm, col = SP[fr.restart_kind]; tm = fr.poss_team
                if fr.restart_kind == RK_OFFSIDE and tm >= 0:
                    out[t].append((f"* OFFSIDE  T{1 - tm} (against)", col)); continue
                if fr.restart_kind in (RK_FREEKICK, RK_PENALTY):
                    a, v = fr.foul_actor, fr.foul_victim
                    ftype = {FOUL_TACKLE: "tackle", FOUL_CHARGE: "charge"}.get(fr.foul_kind, "foul")
                    if 0 <= a < self.N and 0 <= v < self.N:
                        anm = "GK" if self.gk[a] else f"#{self.pnum[a]}"
                        vnm = "GK" if self.gk[v] else f"#{self.pnum[v]}"
                        label = (f"* {nm}  T{int(self.team[a])} {anm} fouled "
                                 f"T{int(self.team[v])} {vnm} [{ftype}]")
                    else:
                        label = "* " + nm + (f"  T{tm}" if tm >= 0 else "") + f" [{ftype}]"
                    out[t].append((label, col)); continue
                out[t].append(("* " + nm + (f"  T{tm}" if tm >= 0 else ""), col))
            if fr.scored >= 0 and pv.scored < 0:
                out[t].append((f"* GOAL!   T{fr.scored}", TEAM[fr.scored])); continue
            actors = np.where((tc != pv.touch) & (tc > 0))[0]
            if len(actors):
                bxy = fr.ball[:2]; actors = actors[np.argsort(np.linalg.norm(fr.P[actors] - bxy[None, :], axis=1))]
            hard_win = False
            for i in actors:
                code = int(tc[i]); tm = int(self.team[i]); num = self.pnum[i]
                if code == TOUCH_GK_CATCH:
                    out[t].append((f"* SAVE!   T{tm} GK", CATCH_COL)); hard_win = True
                elif code == TOUCH_PARRY:
                    out[t].append((f"* SAVE! (parry)  T{tm} GK", CATCH_COL)); hard_win = True
                elif code == TOUCH_TACKLE:
                    out[t].append((f"* TACKLE  T{tm} #{num}", TACKLE_COL)); hard_win = True
                elif code == TOUCH_INTERCEPT:
                    out[t].append((f"* INTERCEPT  T{tm} #{num}", INTERCEPT_COL)); hard_win = True
                elif code in (TOUCH_SHOOT, TOUCH_SHOOT_HEAD):
                    out[t].append((f"* {'HEADER SHOT' if code == TOUCH_SHOOT_HEAD else 'SHOT'}  T{tm} #{num}", SHOOT_COL))
                elif code == TOUCH_DRIBBLE:
                    if t - last_drib.get(i, -10 ** 9) >= mrg:
                        out[t].append((f"* DRIBBLE  T{tm} #{num}", DRIB_COL))
                    last_drib[i] = t
                elif code in (TOUCH_PASS, TOUCH_PASS_HEAD):
                    lbl = "HEADER" if code == TOUCH_PASS_HEAD else "PASS"
                    r = self._first_touch_after(frames, t)
                    if r == i:
                        if t - last_drib.get(i, -10 ** 9) >= mrg:
                            out[t].append((f"* DRIBBLE  T{tm} #{num}", DRIB_COL))
                        last_drib[i] = t
                    elif r >= 0 and self.team[r] == tm:
                        out[t].append((f"* {lbl}  T{tm} #{num} -> #{self.pnum[r]}", PASS_COL))
                    else:
                        if t - last_loose >= mrg:
                            won = "" if r < 0 else f" -> won T{int(self.team[r])}"
                            out[t].append((f"* LOOSE BALL  T{tm} #{num}{won}", MISS_COL))
                        last_loose = t
            if (fr.poss_team >= 0 and pv.poss_team >= 0 and fr.poss_team != pv.poss_team
                    and not hard_win):
                out[t].append((f"* TURNOVER -> T{fr.poss_team}", (0.9, 0.9, 0.95)))
        return out

    def _ground_ring(self, cx, cy, r, n=28):
        th = np.linspace(0, 2 * np.pi, n)
        pts = np.stack([cx + r * np.cos(th), cy + r * np.sin(th), np.zeros(n)], 1)
        return self.cam.proj(pts)

    def _camera_path(self, frames, zoom=0.60, smooth=0.14, speed_out=0.20):
        """공을 따라가는 방송형 팬/줌 경로(고정 3D투영 uv평면 위 뷰창). EMA 평활 + 경기장 클램프."""
        fx0, fx1 = self.xlim; fy0, fy1 = self.ylim
        fw, fh = fx1 - fx0, fy1 - fy0
        buv = self.cam.proj(np.array([[fr.ball[0], fr.ball[1], 0.0] for fr in frames]))
        sm = np.empty_like(buv); sm[0] = buv[0]
        for t in range(1, len(buv)):
            sm[t] = (1.0 - smooth) * sm[t - 1] + smooth * buv[t]
        wins = []
        for t, fr in enumerate(frames):
            spd = float(np.linalg.norm(fr.ball_vel)) if fr.ball_vel is not None else 0.0
            z = min(1.0, zoom + speed_out * min(spd / 25.0, 1.0))
            ww, wh = fw * z, fh * z
            cx, cy = sm[t]
            x0 = float(np.clip(cx - ww / 2, fx0, fx1 - ww))
            y0 = float(np.clip(cy - wh / 2, fy0, fy1 - wh))
            wins.append((x0, x0 + ww, y0, y0 + wh))
        return wins

    def render(self, states, out_path="match.mp4", fps=25, title=None, verbose=True,
               every=1, feed_n=10, feed_secs=12.0, dynamic_cam=False, cam_zoom=0.66,
               frame_annotations=None):
        _ensure_mpl()   # matplotlib 지연 로드(모듈 전역 plt/LineCollection/… 채움) — rich 진입점
        frames = self._frames_from_states(states)[::every]
        if frame_annotations is not None:
            frame_annotations = list(frame_annotations)[::every]
        S = self._precompute(frames)
        T = len(frames); N = self.N
        cam_win = self._camera_path(frames, zoom=cam_zoom) if dynamic_cam else None
        t_start = time.time()
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi); fig.patch.set_facecolor(self.BG)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0]); ax.set_facecolor(self.BG)
        ax.set_xlim(*self.xlim); ax.set_ylim(*self.ylim); ax.set_aspect("equal"); ax.axis("off")
        axm = fig.add_axes([0.752, 0.014, 0.238, 0.225]); axm.set_zorder(30)

        stand_uv = [self.cam.proj(v) for v, _ in self.stand_quads]
        ax.add_collection(PolyCollection(stand_uv, facecolors=[c for _, c in self.stand_quads], edgecolors="none", zorder=-3))
        stripe_uv = [self.cam.proj(np.c_[q, np.zeros(4)]) for q, _ in self.stripes]
        ax.add_collection(PolyCollection(stripe_uv, facecolors=[c for _, c in self.stripes], edgecolors="none", zorder=0))
        byi = -(self.W / 2 + 1.6); byo = -(self.W / 2 + 5.4)
        dug = []
        for x0, x1, col in [(self.bench_x0[0] - 2, self.bench_x0[0] + 22, "#141b28"),
                            (self.bench_x0[1] - 2, self.bench_x0[1] + 22, "#141b28")]:
            dug.append(np.array([[x0, byi, 0], [x1, byi, 0], [x1, byo, 0], [x0, byo, 0]], float))
        dug_uv = [self.cam.proj(q) for q in dug]
        ax.add_collection(PolyCollection(dug_uv, facecolors=["#141b28", "#141b28"],
                                         edgecolors=[TEAM[0], TEAM[1]], linewidths=1.2, zorder=-1))
        line_uv = [self.cam.proj(np.c_[ln, np.zeros(len(ln))]) for ln in self.lines]
        ax.add_collection(LineCollection(line_uv, colors=self.LINE, linewidths=1.3, zorder=1))
        net_uv = [self.cam.proj(s) for s in self.net_segs]
        ax.add_collection(LineCollection(net_uv, colors="#cfd8e2", linewidths=0.6, alpha=0.38, zorder=1.5))
        goal_uv = [self.cam.proj(s) for s in self.goal_segs]
        ax.add_collection(LineCollection(goal_uv, colors="#ffffff", linewidths=2.8, zorder=2))
        ax.add_patch(Rectangle((0, 0.90), 1, 0.10, transform=ax.transAxes, facecolor="#0a121c", alpha=0.84, edgecolor="none", zorder=20))
        ax.text(0.445, 0.965, "T0", transform=ax.transAxes, ha="right", va="center", color=TEAM[0], fontsize=13, fontweight="bold", zorder=21)
        ax.text(0.555, 0.965, "T1", transform=ax.transAxes, ha="left", va="center", color=TEAM[1], fontsize=13, fontweight="bold", zorder=21)
        ax.add_patch(Rectangle((0.006, 0.165), 0.165, 0.715, transform=ax.transAxes,
                               facecolor="#070e18", alpha=0.55, edgecolor="#24405e", linewidth=1.0, zorder=18))
        ax.text(0.020, 0.860, "MATCH EVENTS", transform=ax.transAxes, ha="left", va="center",
                color="#8fb6e0", fontsize=10.5, fontweight="bold", family="monospace", zorder=19)
        if title:
            ax.text(0.020, 0.924, str(title), transform=ax.transAxes, ha="left", va="center",
                    color="#9fb4c8", fontsize=9.5, fontweight="bold", family="monospace", zorder=22)
        self._draw_minimap_static(axm)

        ps = self.player_scale
        shadows = ax.scatter(np.zeros(N), np.zeros(N), s=34, c="#06140b", alpha=0.22, zorder=5)
        bodies = LineCollection([np.zeros((2, 2))] * (N * 12),
                                colors=[self.pcol[i] for i in range(N) for _ in range(12)], linewidths=2.4, zorder=6)
        ax.add_collection(bodies)
        heads = ax.scatter(np.zeros(N), np.zeros(N), s=100, c=self.pcol, edgecolors="#0a0a0a", linewidths=0.8, zorder=7)
        nums = [ax.text(0, 0, self.pnum[i], color="white", fontsize=7.0, fontweight="bold",
                        ha="center", va="center", zorder=11) for i in range(N)]
        stam_bg = LineCollection([np.zeros((2, 2))] * N, colors=[(0, 0, 0, 0.5)] * N, linewidths=3.0, zorder=9)
        stam_fg = LineCollection([np.zeros((2, 2))] * N, colors=[(0, 1, 0)] * N, linewidths=3.0, zorder=10)
        ax.add_collection(stam_bg); ax.add_collection(stam_fg)
        kick_traj = LineCollection([], colors=[(0.20, 0.95, 0.95, 0.9)], linewidths=2.2, zorder=4)
        ax.add_collection(kick_traj)
        kick_base, = ax.plot([0, 0], [0, 0], color="#ffd400", lw=1.3, ls=(0, (4, 3)), zorder=3, alpha=0.0)
        ball_sh = ax.scatter([0], [0], s=30, c="#06140b", alpha=0.35, zorder=3)
        ball_drop, = ax.plot([0, 0], [0, 0], color="#d8d8d8", lw=0.8, ls=(0, (2, 2)), zorder=4)
        ball = ax.scatter([0], [0], s=44, c="#fbfbfb", edgecolors="#222", linewidths=0.7, zorder=8)
        spin_tick, = ax.plot([0, 0], [0, 0], color="#ffd400", lw=1.8, zorder=9, alpha=0.0)
        ball_trail = LineCollection([], zorder=3); ax.add_collection(ball_trail)
        carrier_glow, = ax.plot([], [], lw=6.5, zorder=2, alpha=0.0, solid_capstyle="round")
        carrier_ring, = ax.plot([], [], lw=2.0, zorder=3, alpha=0.0, solid_capstyle="round", color="#ffffff")
        imp_rings = LineCollection([], zorder=2.4)      # 임펄스 링(터치당 1개, ttl 동안 페이드)
        ax.add_collection(imp_rings)
        sp_label = ax.text(0, 0, "", color="#ffe14d", fontsize=8.5, fontweight="bold",
                           ha="center", va="bottom", zorder=24,
                           bbox=dict(boxstyle="square,pad=0.2", fc="#1a1200", ec="#ffd400", lw=0.8, alpha=0.85))
        sp_label.set_visible(False)
        zone_fill = ax.add_patch(MplPolygon([[0, 0]], closed=True, facecolor="#ffffff", edgecolor="none", alpha=0.0, zorder=1.9))
        zone_edge, = ax.plot([], [], lw=1.6, ls="--", alpha=0.0, zorder=2.3)
        zone_box_fill = ax.add_patch(MplPolygon([[0, 0]], closed=True, facecolor="#ffffff", edgecolor="none", alpha=0.0, zorder=1.9))
        zone_box_edge, = ax.plot([], [], lw=1.6, ls="--", alpha=0.0, zorder=2.3)
        enc_marks = ax.scatter(np.zeros(N), np.zeros(N), s=np.zeros(N), marker="o",
                               facecolors="none", edgecolors="#ff3b3b", linewidths=1.8, zorder=12)
        offside_glow, = ax.plot([], [], lw=5.5, zorder=2.6, alpha=0.0, solid_capstyle="round")
        offside_line, = ax.plot([], [], lw=2.2, zorder=2.8, alpha=0.0, solid_capstyle="round")
        off_marks = ax.scatter(np.zeros(N), np.zeros(N), s=np.zeros(N), marker="v",
                               facecolors="#ff4d4d", edgecolors="#ffdada", linewidths=0.5, zorder=13)
        card_marks = ax.scatter(np.zeros(N), np.zeros(N), s=np.zeros(N), marker="s",
                                facecolors=[(0, 0, 0, 0)] * N, edgecolors="#141414", linewidths=0.6, zorder=14)
        score_txt = ax.text(0.50, 0.965, "0 : 0", transform=ax.transAxes, ha="center", va="center", color="#fff", fontsize=15, fontweight="bold", zorder=21)
        clock_txt = ax.text(0.50, 0.924, "00:00", transform=ax.transAxes, ha="center", va="center", color="#9fb4c8", fontsize=10, fontweight="bold", family="monospace", zorder=22)
        poss_bg = ax.add_patch(Rectangle((0.035, 0.957), 0.20, 0.014, transform=ax.transAxes, facecolor="#223", edgecolor="none", zorder=20))
        poss_bar0 = ax.add_patch(Rectangle((0.035, 0.957), 0.10, 0.014, transform=ax.transAxes, facecolor=TEAM[0], edgecolor="none", zorder=21))
        poss_bar1 = ax.add_patch(Rectangle((0.135, 0.957), 0.10, 0.014, transform=ax.transAxes, facecolor=TEAM[1], edgecolor="none", zorder=21))
        poss_txt = ax.text(0.035, 0.936, "", transform=ax.transAxes, ha="left", va="center", color="#dfe6ee", fontsize=8.5, zorder=21)
        goal_flash = ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#ffffff", alpha=0.0, edgecolor="none", zorder=19))
        goal_txt = ax.text(0.50, 0.55, "", transform=ax.transAxes, ha="center", va="center", color="#ffd400", fontsize=30, fontweight="bold", zorder=23)
        # 데드볼 내장 엔진 활성 표시(토글 ON & 데드볼) — 육안 식별용 시안 테두리 + 배지.
        eng_on = bool(self.env.e_cfg.deadball_engine)
        eng_border = ax.add_patch(Rectangle((0.004, 0.004), 0.992, 0.992, transform=ax.transAxes,
                                            facecolor="none", edgecolor="#3ce6d2", linewidth=2.6, alpha=0.0, zorder=25))
        eng_txt = ax.text(0.50, 0.892, "", transform=ax.transAxes, ha="center", va="center",
                          color="#3ce6d2", fontsize=9.5, fontweight="bold", family="monospace", zorder=24)
        annotation_border = ax.add_patch(
            Rectangle(
                (0.004, 0.004),
                0.992,
                0.992,
                transform=ax.transAxes,
                facecolor="none",
                edgecolor="#ff5353",
                linewidth=3.2,
                alpha=0.0,
                zorder=27,
            )
        )
        annotation_txt = ax.text(
            0.50,
            0.855,
            "",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#ff7777",
            fontsize=12,
            fontweight="bold",
            family="monospace",
            zorder=28,
        )
        feed_txt = [ax.text(0.020, 0.822 - 0.0345 * k, "", transform=ax.transAxes, ha="left", va="top", fontsize=8.3, fontweight="bold", family="monospace", zorder=23) for k in range(feed_n)]
        mm_view, = axm.plot([], [], color="#ffd400", lw=1.3, alpha=0.9, zorder=3)
        mm_dots = axm.scatter(np.zeros(N), np.zeros(N), s=34, c=self.pcol,
                              edgecolors=["white" if self.gk[i] else "#0a0a0a" for i in range(N)], linewidths=0.6, zorder=4)
        mm_ball = axm.scatter([0], [0], s=40, c="#ffffff", edgecolors="#222", linewidths=0.6, zorder=5)

        dyn_ax = [shadows, kick_traj, kick_base, ball_trail, offside_glow, offside_line, carrier_glow, carrier_ring,
                  imp_rings,
                  zone_fill, zone_edge, zone_box_fill, zone_box_edge, enc_marks,
                  bodies, heads, *nums, stam_bg, stam_fg, off_marks, card_marks, ball_sh, ball_drop, ball, spin_tick,
                  goal_flash, sp_label, eng_border, eng_txt, annotation_border,
                  annotation_txt, score_txt, clock_txt, poss_bg, poss_bar0, poss_bar1, poss_txt,
                  goal_txt, *feed_txt]
        dyn_axm = [mm_view, mm_dots, mm_ball]
        bg = None
        if not dynamic_cam:
            for art in dyn_ax + dyn_axm: art.set_animated(True)
            fig.canvas.draw(); bg = fig.canvas.copy_from_bbox(fig.bbox)

        is_gif = str(out_path).lower().endswith(".gif")
        writer = (imageio.get_writer(out_path, mode="I", duration=1.0 / fps, loop=0) if is_gif
                  else imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8,
                                          pixelformat="yuv420p", macro_block_size=None))
        feed = []; feed_life = max(40, int(fps * feed_secs))
        feed_state = [None] * feed_n
        trail_buf = []
        # 임펄스 링 상태: [ttl0, ttl, 선수, rgb, 반지름m, 선폭]. light의 event_ttl=8@25fps와 같은 체감 길이.
        imp_fx = []
        imp_ttl = max(2, int(round(fps * 0.32)))
        imp_span = max(float(self.E.f2b_speed_max) - RING_IMPULSE_MIN, 1e-3)
        goal_hold_team = -1; goal_hold_until = -1
        annotation_hold = ""; annotation_until = -1

        def _set_text(art, s):
            if art.get_text() != s: art.set_text(s)
        HW = 7.0
        for t, fr in enumerate(frames):
            if frame_annotations is not None and frame_annotations[t]:
                annotation_hold = str(frame_annotations[t])
                annotation_until = t + max(1, int(round(fps * 0.6)))
            if t < annotation_until:
                annotation_border.set_alpha(0.95)
                _set_text(annotation_txt, annotation_hold)
                annotation_txt.set_visible(True)
            else:
                annotation_border.set_alpha(0.0)
                annotation_txt.set_visible(False)
            for e in S["events"][t]: feed.append((e[0], e[1], t))
            kt = S["KT"][t]
            op = fr.on_pitch if fr.on_pitch is not None else np.ones(N, bool)
            if bool(op.all()):
                Pd = fr.P; facing_d = fr.facing
            else:
                Pd = fr.P.copy(); facing_d = fr.facing.copy()
                for tm in (0, 1):
                    slot = 0
                    for i in range(N):
                        if (not op[i]) and self.team[i] == tm:
                            Pd[i] = [self.bench_x0[tm] + slot * 2.4, -(self.W / 2 + 3.5)]
                            facing_d[i] = np.pi / 2
                            slot += 1
            segs, hd, _ = self._skeletons(Pd, facing_d, S["PH"][t], S["AMP"][t], S["KF"][t], S["KH"][t])
            seg_uv = self.cam.proj(segs.reshape(-1, 3)).reshape(-1, 2, 2); bodies.set_segments(list(seg_uv))
            kcol = []
            for i in range(N):
                act = (S["KF"][t][i] > 0.25) or (S["KH"][t][i] > 0.25)
                col = ("#5a626e" if not op[i] else
                       (SHOOT_COL if kt[i] in (TOUCH_SHOOT, TOUCH_SHOOT_HEAD) else PASS_COL) if act else self.pcol[i])
                kcol.extend([col] * 12)
            bodies.set_color(kcol)
            head_uv = self.cam.proj(hd); heads.set_offsets(head_uv)
            shadows.set_offsets(self.cam.proj(np.c_[Pd, np.zeros(N)]))
            num_uv = self.cam.proj(np.c_[Pd, np.full(N, 0.95 * ps)])
            for i in range(N): nums[i].set_position((num_uv[i, 0], num_uv[i, 1]))
            by_off = head_uv[:, 1] + 5.0
            x0 = head_uv[:, 0] - HW; x1 = head_uv[:, 0] + HW
            xw = x0 + 2 * HW * np.clip(fr.stamina, 0, 1)
            bg_segs = np.stack([np.stack([x0, by_off], 1), np.stack([x1, by_off], 1)], 1)
            fg_segs = np.stack([np.stack([x0, by_off], 1), np.stack([xw, by_off], 1)], 1)
            stam_bg.set_segments(bg_segs); stam_fg.set_segments(fg_segs)
            stam_fg.set_color([_stam_color(fr.stamina[i]) for i in range(N)])
            bx, by, bz = fr.ball
            ball.set_offsets(self.cam.proj(np.array([[bx, by, bz]])))
            ball_sh.set_offsets(self.cam.proj(np.array([[bx, by, 0]])))
            dl = self.cam.proj(np.array([[bx, by, bz], [bx, by, 0]])); ball_drop.set_data(dl[:, 0], dl[:, 1])
            spv = fr.ball_spin if fr.ball_spin is not None else np.zeros(3)
            smag = float(np.linalg.norm(spv))
            if smag > 0.5:
                phase = t * 0.9 + smag * 0.3
                rad = 0.45 + 0.5 * min(smag / self.E.spin_max, 1.0)
                tip = self.cam.proj(np.array([[bx + rad * np.cos(phase), by + rad * np.sin(phase), bz]]))[0]
                tip2 = self.cam.proj(np.array([[bx - rad * np.cos(phase), by - rad * np.sin(phase), bz]]))[0]
                spin_tick.set_data([tip2[0], tip[0]], [tip2[1], tip[1]])
                spin_tick.set_color("#39b0ff" if spv[2] >= 0 else "#ff6b6b")
                spin_tick.set_alpha(min(0.9, 0.3 + smag / self.E.spin_max))
            else:
                spin_tick.set_alpha(0.0)
            ks = int(S["KSEG"][t]); cap = 90
            if ks >= 0 and t - ks >= 2:
                i0 = max(ks, t - cap)
                path = np.array([frames[i].ball for i in range(i0, t + 1)])
                traj_uv = self.cam.proj(path); kick_traj.set_segments([traj_uv]); kick_traj.set_alpha(1.0)
                is_shot_seg = bool(np.any(np.isin(frames[ks].touch, [TOUCH_SHOOT, TOUCH_SHOOT_HEAD])))
                kick_traj.set_color((1.0, 0.62, 0.12, 0.95) if is_shot_seg else (0.20, 0.95, 0.95, 0.9))
                start = frames[ks].ball
                v0 = frames[ks].ball_vel if frames[ks].ball_vel is not None else np.zeros(3)
                vdir = v0[:2] / (np.linalg.norm(v0[:2]) + 1e-9)
                downrange = max(0.0, float(np.dot(path[-1, :2] - start[:2], vdir)))
                bl = np.array([[start[0], start[1], 0.0],
                               [start[0] + vdir[0] * downrange, start[1] + vdir[1] * downrange, 0.0]])
                bl_uv = self.cam.proj(bl); kick_base.set_data(bl_uv[:, 0], bl_uv[:, 1]); kick_base.set_alpha(0.85)
            else:
                kick_traj.set_alpha(0.0); kick_base.set_alpha(0.0)
            alive = (int(fr.ball_state) == BALL_ALIVE)
            trail_buf.append(fr.ball.copy())
            if len(trail_buf) > self.trail_len: trail_buf.pop(0)
            if len(trail_buf) >= 2:
                tuv = self.cam.proj(np.array(trail_buf))
                tsegs = np.stack([tuv[:-1], tuv[1:]], 1); m = len(tsegs)
                tcol = np.zeros((m, 4)); tcol[:, :3] = (0.95, 0.97, 1.0)
                tcol[:, 3] = np.linspace(0.04, 0.55, m)
                ball_trail.set_segments(list(tsegs)); ball_trail.set_color(tcol)
                ball_trail.set_linewidths(np.linspace(0.5, 2.2, m))
            else:
                ball_trail.set_segments([])
            ball.set_color("#fbfbfb" if alive else "#8a94a0")
            own = int(S["owner"][t])
            if own >= 0 and alive:
                ring = self._ground_ring(fr.P[own, 0], fr.P[own, 1], 1.5)
                carrier_glow.set_data(ring[:, 0], ring[:, 1]); carrier_glow.set_color(self.pcol[own]); carrier_glow.set_alpha(0.55)
                carrier_ring.set_data(ring[:, 0], ring[:, 1]); carrier_ring.set_alpha(0.95)
            else:
                carrier_glow.set_alpha(0.0); carrier_ring.set_alpha(0.0)
            # ── 임펄스 링 ── 이 프레임에 공이 받은 힘(질량 상수 → ∝‖Δv‖)에 반지름이 비례.
            # 규약은 _render_light와 동일: 색=터치 코드(무엇을 했나), 크기·선폭·잔상=힘의 크기.
            # 파울 휘슬 프레임은 규칙이 공을 세우는 행정 정지(물리적 힘 아님)라 제외한다.
            # 주의: every>1로 솎아 렌더하면 Δv가 every 프레임 구간 차분이라 값이 뭉개진다.
            dv = 0.0
            if t > 0 and not (int(fr.foul_kind) > 0 and int(frames[t - 1].foul_kind) == 0):
                pv, cv = frames[t - 1].ball_vel, fr.ball_vel
                if pv is not None and cv is not None:
                    dv = float(np.linalg.norm(np.asarray(cv) - np.asarray(pv)))
            if dv >= RING_IMPULSE_MIN:
                imp = float(np.clip((dv - RING_IMPULSE_MIN) / imp_span, 0.0, 1.0))
                r_m = RING_R_MIN_M + (RING_R_MAX_M - RING_R_MIN_M) * imp
                ttl0 = max(1, int(round(imp_ttl * (0.7 + 0.8 * imp))))
                for i in np.where((fr.touch > 0) & op)[0]:
                    imp_fx.append([ttl0, ttl0, int(i),
                                   TOUCH_RING_RGB.get(int(fr.touch[i]), (0.88, 0.91, 0.89)),
                                   r_m, 1.6 + 2.2 * imp])
            imp_segs, imp_cols, imp_lws, imp_keep = [], [], [], []
            for fx in imp_fx:
                fx[1] -= 1
                if fx[1] < 0:
                    continue
                ring = self._ground_ring(fr.P[fx[2], 0], fr.P[fx[2], 1], fx[4], n=36)
                imp_segs.append(ring)
                imp_cols.append((*fx[3], 0.95 * (fx[1] + 1) / fx[0]))   # ttl 동안 선형 페이드아웃
                imp_lws.append(fx[5])
                imp_keep.append(fx)
            imp_fx = imp_keep
            imp_rings.set_segments(imp_segs)
            if imp_segs:
                imp_rings.set_color(imp_cols); imp_rings.set_linewidths(imp_lws)
            spk = int(fr.sp_taker) if fr.sp_taker is not None else -1
            if 0 <= spk < N and int(fr.restart_t) > 0:
                # 카운트다운 창은 config의 재개 종류별 값을 사용한다(obs rt_norm과 동일 규약).
                E = self.E; rt = int(fr.restart_t); rk9 = int(fr.restart_kind)
                window = (E.penalty_substeps if rk9 == RK_PENALTY
                          else E.gk_hold_substeps if rk9 == RK_GK_HOLD else E.restart_substeps)
                pre = "PK" if rk9 == RK_PENALTY else ("HOLD" if rk9 == RK_GK_HOLD else "TAKER")
                if rt >= window:
                    lbl = f"{pre}  SET UP"
                else:
                    secs = max(0.0, (rt - (window - E.setup_hold_substeps))) * E.dt_phys
                    lbl = f"{pre}  {secs:0.1f}s" if secs > 0.05 else f"{pre}  READY"
                hd_uv = self.cam.proj(np.array([[fr.P[spk, 0], fr.P[spk, 1], 2.2 * ps]]))[0]
                sp_label.set_position((hd_uv[0], hd_uv[1])); _set_text(sp_label, lbl); sp_label.set_visible(True)
            else:
                sp_label.set_visible(False)
            # 데드볼 내장 엔진 활성(토글 ON & 데드볼) 시 테두리·배지 점등.
            if eng_on and int(fr.restart_t) > 0:
                eng_border.set_alpha(0.9); _set_text(eng_txt, "DEAD-BALL ENGINE"); eng_txt.set_visible(True)
            else:
                eng_border.set_alpha(0.0); eng_txt.set_visible(False)
            rk_now = int(fr.restart_kind)
            rteam = int(fr.restart_team)
            if rteam < 0:
                rteam = int(fr.poss_team)
            # 반경형 이격이 있는 재개만 존 오버레이 — 페널티(박스+아크)·GK홀드(이격 의무 없음)는 제외.
            if (int(fr.restart_t) > 0
                    and rk_now in (RK_KICKOFF, RK_THROWIN, RK_GOALKICK,
                                   RK_CORNER, RK_FREEKICK, RK_OFFSIDE)
                    and 0 <= rteam < len(TEAM)):
                E = self.E
                zr = (self.env.s_cfg.center_circle_radius if rk_now == RK_KICKOFF
                      else (E.throwin_clear if rk_now == RK_THROWIN else E.clear_dist))
                zcol = TEAM[rteam]
                ring = self._ground_ring(fr.ball[0], fr.ball[1], zr, n=48)
                zone_fill.set_xy(ring); zone_fill.set_facecolor(zcol); zone_fill.set_alpha(0.10)
                zone_edge.set_data(ring[:, 0], ring[:, 1]); zone_edge.set_color(zcol); zone_edge.set_alpha(0.8)
                d_spot = np.linalg.norm(fr.P - fr.ball[None, :2], axis=1)
                inz = (self.team != rteam) & op & (d_spot < zr)
                # env 골라인 예외(restart._encroach_geometry, IFAB Law 13) 복제 — 자기 골라인
                # 밴드의 수비수는 합법이므로 위반 할로에서 제외(안 하면 박스 앞 IDFK에서 골라인
                # GK가 위반자로 표시돼 'retake가 왜 없지'로 오도).
                if fr.attack_dir is not None:
                    own_gx_all = -np.asarray(fr.attack_dir) * self.L / 2.0
                    on_line = (
                        (np.abs(fr.P[:, 0] - own_gx_all) <= self.E.goal_line_tolerance)
                        & (
                            np.abs(fr.P[:, 1])
                            <= self.env.goal_w / 2.0 + self.E.goal_post_tolerance
                        )
                    )
                    inz = inz & (~on_line)
                if rk_now == RK_GOALKICK and fr.attack_dir is not None:
                    adir_rt = float(fr.attack_dir[np.where(self.team == rteam)[0][0]])
                    gx = -adir_rt * self.L / 2.0
                    bf = gx + adir_rt * self.env.pen_len; hw = self.env.pen_hw
                    corners = np.array([[gx, -hw, 0.0], [gx, hw, 0.0], [bf, hw, 0.0], [bf, -hw, 0.0]])
                    buv = self.cam.proj(corners)
                    zone_box_fill.set_xy(buv); zone_box_fill.set_facecolor(zcol); zone_box_fill.set_alpha(0.10)
                    bcl = np.vstack([buv, buv[:1]])
                    zone_box_edge.set_data(bcl[:, 0], bcl[:, 1]); zone_box_edge.set_color(zcol); zone_box_edge.set_alpha(0.8)
                    xlo, xhi = min(gx, bf), max(gx, bf)
                    in_box = (fr.P[:, 0] >= xlo) & (fr.P[:, 0] <= xhi) & (np.abs(fr.P[:, 1]) <= hw)
                    inz = inz | ((self.team != rteam) & op & in_box)
                else:
                    zone_box_fill.set_alpha(0.0); zone_box_edge.set_alpha(0.0)
                enc_uv = self.cam.proj(np.c_[fr.P, np.zeros(N)])
                enc_marks.set_offsets(enc_uv); enc_marks.set_sizes(np.where(inz, 150.0, 0.0))
            else:
                zone_fill.set_alpha(0.0); zone_edge.set_alpha(0.0)
                zone_box_fill.set_alpha(0.0); zone_box_edge.set_alpha(0.0)
                enc_marks.set_sizes(np.zeros(N))
            adir = fr.attack_dir; off = fr.offside_flag
            att = fr.poss_team if fr.poss_team >= 0 else fr.last_team
            if adir is not None and att is not None and att >= 0 and alive and int(fr.restart_t) == 0:
                adt = float(adir[np.where(self.team == att)[0][0]])
                x_def = np.where((self.team != att) & op, fr.P[:, 0] * adt, -np.inf)
                lx_att = max(float(np.sort(x_def)[self.N - 2]), 0.0)
                lx = lx_att * adt
                ol = self.cam.proj(np.array([[lx, -self.W / 2, 0.0], [lx, self.W / 2, 0.0]]))
                offside_glow.set_data(ol[:, 0], ol[:, 1]); offside_glow.set_color(TEAM[att]); offside_glow.set_alpha(0.22)
                offside_line.set_data(ol[:, 0], ol[:, 1]); offside_line.set_color(TEAM[att]); offside_line.set_alpha(0.95)
            else:
                offside_glow.set_alpha(0.0); offside_line.set_alpha(0.0)
            fresh = int(getattr(fr, "pass_t", 0)) > (self.E.pass_protect - 50)
            if off is not None and fresh and bool(np.any(off)):
                mk_uv = self.cam.proj(np.c_[fr.P, np.full(N, 2.0 * ps)])
                off_marks.set_offsets(mk_uv); off_marks.set_sizes(np.where(off, 55.0, 0.0))
            else:
                off_marks.set_sizes(np.zeros(N))
            yc = fr.yellow_cards if fr.yellow_cards is not None else np.zeros(N)
            card_uv = head_uv.copy(); card_uv[:, 0] += 10.0; card_uv[:, 1] += 2.0
            card_marks.set_offsets(card_uv)
            ccol = np.zeros((N, 4)); csz = np.zeros(N)
            for i in range(N):
                if not op[i]:
                    ccol[i] = (0.90, 0.15, 0.15, 1.0); csz[i] = 62.0
                elif yc[i] >= 1:
                    ccol[i] = (0.98, 0.82, 0.10, 1.0); csz[i] = 48.0
            card_marks.set_facecolors(ccol); card_marks.set_sizes(csz)
            if not bool(op.all()):
                heads.set_color([self.pcol[i] if op[i] else "#6a7078" for i in range(N)])
            else:
                heads.set_color(self.pcol)
            sc = S["score"][t]; pp = S["poss"][t]
            _set_text(score_txt, f"{sc[0]} : {sc[1]}")
            msec = int(round(fr.clock))
            _set_text(clock_txt, f"{msec // 60:02d}:{msec % 60:02d}")
            poss_bar0.set_width(0.20 * pp[0]); poss_bar1.set_x(0.035 + 0.20 * pp[0]); poss_bar1.set_width(0.20 * pp[1])
            _set_text(poss_txt, f"POSS {pp[0] * 100:.0f}% / {pp[1] * 100:.0f}%")
            if fr.scored >= 0:
                goal_hold_team = int(fr.scored); goal_hold_until = t + int(fps * 2.2)
            if t < goal_hold_until and goal_hold_team >= 0:
                _set_text(goal_txt, "G O A L !")
                goal_flash.set_facecolor(TEAM[goal_hold_team]); goal_flash.set_alpha(float(0.14 + 0.12 * abs(np.sin(t * 0.6))))
            else:
                _set_text(goal_txt, ""); goal_flash.set_alpha(0.0)
            while feed and t - feed[0][2] >= feed_life:
                feed.pop(0)
            shown = feed[-feed_n:][::-1]
            for k in range(feed_n):
                if k < len(shown):
                    txt, col, bt = shown[k]; age = (t - bt) / feed_life
                    a = 1.0 if age < 0.75 else float(np.clip(1.0 - (age - 0.75) / 0.25, 0.3, 1.0))
                    fresh_ev = (t - bt) < int(fps * 1.5)
                    state = (txt, col, fresh_ev, a)
                    if feed_state[k] != state:
                        feed_state[k] = state
                        feed_txt[k].set_text(("> " if fresh_ev else "  ") + txt)
                        feed_txt[k].set_color((1, 1, 1) if fresh_ev else col)
                        feed_txt[k].set_alpha(a); feed_txt[k].set_fontsize(9.2 if fresh_ev else 8.3)
                elif feed_state[k] is not None:
                    feed_state[k] = None; feed_txt[k].set_text("")
            mm_dots.set_offsets(Pd); mm_ball.set_offsets(fr.ball[:2][None, :])

            if dynamic_cam:
                x0, x1, y0, y1 = cam_win[t]
                ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
                cor = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
                gv = self.cam.unproj_ground(cor)
                gv[:, 0] = np.clip(gv[:, 0], -self.L / 2 - 3.0, self.L / 2 + 3.0)
                gv[:, 1] = np.clip(gv[:, 1], -self.W / 2 - 3.0, self.W / 2 + 3.0)
                mm_view.set_data(gv[:, 0], gv[:, 1])
                fig.canvas.draw()
            else:
                fig.canvas.restore_region(bg)
                for art in dyn_ax: ax.draw_artist(art)
                for art in dyn_axm: axm.draw_artist(art)
                fig.canvas.blit(fig.bbox)
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            if verbose and t % 50 == 0: print(f"  [rich] rendering {t + 1}/{T}")
        writer.close(); plt.close(fig)
        if verbose:
            dt = time.time() - t_start
            print(f"[render:rich] {T}프레임 → {out_path}  ({dt:.1f}s, {T / max(dt, 1e-9):.1f} fps)")
        return out_path

    def _draw_minimap_static(self, axm):
        hx, hy = self.L / 2, self.W / 2; axm.set_facecolor("#0e3a1c")
        lc = dict(color="#cfe8d6", lw=0.8, zorder=2)
        axm.add_patch(Rectangle((-hx, -hy), 2 * hx, 2 * hy, fill=False, **lc))
        axm.plot([0, 0], [-hy, hy], **lc); th = np.linspace(0, 2 * np.pi, 40)
        center_r = self.env.s_cfg.center_circle_radius
        axm.plot(center_r * np.cos(th), center_r * np.sin(th), **lc); g = self.gw / 2
        pen_len = self.env.s_cfg.penalty_area_length
        pen_hw = self.env.s_cfg.penalty_area_width / 2.0
        for sx in (-hx, hx):
            axm.plot([sx, sx], [-g, g], color="#fff", lw=1.8, zorder=3)
            x0 = -hx if sx < 0 else hx - pen_len
            axm.add_patch(Rectangle((x0, -pen_hw), pen_len, 2.0 * pen_hw, fill=False, **lc))
        axm.set_xlim(-hx - 2, hx + 2); axm.set_ylim(-hy - 2, hy + 2); axm.set_aspect("equal"); axm.axis("off")
