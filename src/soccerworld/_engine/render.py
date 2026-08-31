"""경량 궤적 렌더러 — State 시퀀스 → mp4. 속도 지향(경량 경로는 순수 numpy 래스터, matplotlib 무의존).
`render_mp4`는 `mode="rich"` 디스패치(render_rich, 지연 import)와 트래킹/이벤트 덤프(_write_*)의
공용 진입점이기도 하다 — 덤프 산출물은 모드와 무관하게 동일 경로에서 나온다.

원본 SOCCER render.py(3D matplotlib, 정밀 브로드캐스트 뷰)와 목적이 다르다: 여기서는 **궤적 재현
확인**에 필요한 최소 정보만 탑다운 2D로 그려 렌더 속도를 극대화한다(원본 대비 수십 배).
프레임 합성이 전부 벡터화된 numpy 연산(사전계산 피치 + 감쇠 트레일 레이어 + 원판 스탬핑)이고,
인코딩은 원본과 동일하게 imageio+libx264(번들 ffmpeg).

표현(필수 state만, 전부 State 필드에서 직접 — 원본의 Frame dataclass 어댑터 불필요):
 · 선수: 팀색 원판(적/청, GK는 밝은 톤), 비활성(active_player=False)은 피치에서 제거하고 근경
   터치라인 아래 벤치에 흐린 팀색+레드카드로 표시. 머리 위에는 단기(위)·장기(아래)
   stamina 바를 표시하고, 옐로카드는 노란 사각 마커로 표시.
 · 공: 지면 그림자(회색) + 높이만큼 화면 위로 띄운 흰 원판(반지름도 z에 비례) — 3D 정보를 2D에 투영.
 · 궤적: 감쇠 트레일 레이어(공 백색, 선수 팀색 저휘도) — 프레임당 전체 배열 곱 1회로 페이드.
 · 이벤트: 소유권은 팀색 실선 링, 공에 가한 힘은 터치코드별 색 점선 링으로
   분리한다. 파울은 actor 적색 링+victim 연결선, 득점은 GOAL 오버레이, 카드/퇴장
   온셋 링. HUD 티커는 canonical event feed의 패스·슛 결과, 징계·교체를 실제 결과
   프레임에 표시한다. 오프사이드 참조선·플래그 선수, 재개 수행자·준비 카운트다운,
   공 회전 tick도 경량 래스터로 표시한다. 포메이션 변경·교체는 골 알림처럼 중앙 감독
   명령 카드로 2.4초간 유지하되, 골과 다른 하늘색/녹색 팔레트로 구분한다.

원본 대비 구현 차이(클론 규약 준수): `on_pitch`→`active_player`, 터치코드에 DEFLECT(9)/PARRY(10)
추가, 득점은 Frame.scored가 아니라 score 차분으로 검출, 파울은 state.foul_kind/actor/victim 직접
사용. 월드 좌표로만 그리므로 obs 폴딩 규약(z회전 vs x미러)과 무관.

사용:
    frames = [state0, state1, ...]          # step_env가 반환한 State들(원 State 그대로)
    env.render_mp4(frames)                  # → mp4 + events/tracking/metadata JSONL 경로 dict
    env.render_mp4(frames, "path/to/out.mp4", fps=25)   # 경로 직접 지정도 가능
    env.render_mp4(frames, dump_replay=False)            # 영상만 필요할 때
"""
from __future__ import annotations

import json
import numbers
import os
import queue
import threading
import time
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from . import formation as formation_module
from .constants import (
    BALL_ALIVE,
    BALL_EVENT_CORNER,
    BALL_EVENT_GOAL,
    BALL_EVENT_GOALKICK,
    BALL_EVENT_NONE,
    BALL_EVENT_THROWIN,
    DIM_ALL,
    DIM_X,
    FOUL_CHARGE,
    FOUL_SETPIECE,
    FOUL_TACKLE,
    FOUL_THROW,
    GK_HANDLING_RELEASE_OFFSET,
    NO_PLAYER,
    NO_TEAM,
    RK_CORNER,
    RK_FREEKICK,
    RK_GK_HOLD,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_NONE,
    RK_OFFSIDE,
    RK_PENALTY,
    RK_THROWIN,
    TEAM_0,
    TEAM_1,
    TEAM_COUNT,
    TOUCH_BODY_TRAP,
    TOUCH_COUNT,
    TOUCH_DEFLECT,
    TOUCH_DRIBBLE,
    TOUCH_EVENT_FORCE,
    TOUCH_EVENT_PHASE_COUNT,
    TOUCH_GK_CATCH,
    TOUCH_INTERCEPT,
    TOUCH_NONE,
    TOUCH_PARRY,
    TOUCH_PASS,
    TOUCH_PASS_HEAD,
    TOUCH_SHOOT,
    TOUCH_SHOOT_HEAD,
    TOUCH_TACKLE,
    WOODWORK_CROSSBAR,
    WOODWORK_NONE,
    WOODWORK_POST,
)
from .render_defaults import DEFAULT_RENDER_FPS
from .setpiece_taker import ROLE_NAMES, classify_roles

LEGACY_EVENTS_SCHEMA = "uos-footballmarl.events/3"
# Historical aggregate JSON envelope, including its formation ``immediate`` field.
# payload가 늘면 버전이 오른다. /5에서 ``formation`` 이벤트와 골·아웃 판정의
# ``crossing{pos,vel,substep}``이 들어왔다. /6은 formation을 즉시 바뀐 **목표**로
# 정의하고, 실제 선수 이동과 혼동되던 ``immediate`` phase 필드를 제거했다.
CANONICAL_EVENTS_SCHEMA = "uos-footballmarl.events/6"

# State snapshots identify the frame of most rule events but only physical
# touches carry an exact substep.  This table therefore defines the narrow
# causal partial order that is actually observable.  Events in one stage keep
# their stable detection order; adding a new type must classify it here instead
# of silently falling back to alphabetic order.
_EVENT_CAUSAL_STAGE = {
    "touch": 0,
    # Outcomes are source-touch annotations.  ``outcome_frame`` separately
    # records when the future result became known.
    "pass_outcome": 1,
    "shot_outcome": 1,
    # 프레임 충돌은 접촉의 물리적 결과이고 골·재개의 원인이다 — 그 사이 단계에 둔다.
    "woodwork": 1,
    "foul": 2,
    "goal": 2,
    "card": 3,
    "sent_off": 4,
    "possession": 5,
    "offside_flag": 5,
    "gk_hold_start": 5,
    "gk_hold_end": 5,
    # 구간 경계는 같은 단계에 두고 삽입 순서(end → period → start)를 안정 정렬이 지킨다.
    "period_end": 6,
    "period": 6,
    "period_start": 6,
    "substitution": 7,
    # 포메이션 지휘는 교체와 같은 단계의 팀 결정이다. 같은 데드볼에 둘 다 일어날 수 있고,
    # 교체로 들어온 선수도 그 모양을 보고 서므로 교체 뒤에 온다.
    "formation": 7,
    # A restart is the consequence of an out, foul, goal, period boundary, or
    # goalkeeper hold, so it follows every same-frame cause above.
    "restart": 8,
    # 리플레이 종결은 같은 프레임의 모든 원인 뒤에 온다.
    "match_end": 9,
    "censored": 9,
}

DEFAULT_DYNAMIC_CAMERA = False
RICH_MINIMAP_RECT = (0.381, 0.014, 0.238, 0.225)

# Rendering is an optional package surface. Keep these dependencies out of the
# environment import graph; actual render entry points populate the globals.
imageio = Image = ImageDraw = ImageFont = None


def _ensure_render_dependencies():
    """Load light/video dependencies only when a render is requested."""

    global imageio, Image, ImageDraw, ImageFont
    if imageio is not None and Image is not None:
        return
    try:
        import imageio.v2 as _imageio
        from PIL import Image as _Image
        from PIL import ImageDraw as _ImageDraw
        from PIL import ImageFont as _ImageFont
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Rendering requires the optional dependencies; install "
            "SoccerWorld with `pip install soccerworld[render]`."
        ) from exc
    imageio, Image, ImageDraw, ImageFont = _imageio, _Image, _ImageDraw, _ImageFont

# imageio-ffmpeg가 ffmpeg를 fork+exec로 띄울 때, JAX가 스레드를 띄운 상태라 CPython이
# "os.fork() was called ... multithreaded" RuntimeWarning을 낸다. 이 fork는 즉시 exec하는
# async-signal-safe 경로(그 사이 파이썬 락 안 잡음)라 실제 데드락 위험이 없어 렌더는 매번 완료된다 —
# 과보수적인 이 특정 경고만 좁게 억제한다(다른 fork 경고엔 영향 없음).
import warnings

warnings.filterwarnings("ignore", message=r".*os\.fork\(\) was called.*",
                        category=RuntimeWarning)

from types import SimpleNamespace  # rich 프레임 뷰(아래 RichRenderer)에서 사용

# ── rich 모드 전용 matplotlib 의존은 지연 로드 ────────────────────────────────
# light 모드(순수 numpy)는 matplotlib 없이 돌아야 하므로 아래 심볼들을 모듈 전역에 '빈 자리'로
# 두고, rich 렌더가 실제로 호출될 때 _ensure_mpl()이 채운다(RichRenderer 메서드들은 이 전역을
# 참조 — render_rich.py가 모듈 상단 import로 쓰던 것과 동일한 이름 해석). 단일 파일 통합 후에도
# 경량 경로는 무의존 유지.
plt = None
LineCollection = PolyCollection = Rectangle = MplPolygon = FontProperties = None
MplPath = None


def _ensure_mpl():
    """rich 렌더 진입 시 1회 호출 — matplotlib을 로드해 모듈 전역(plt/LineCollection/…)을 채운다."""
    global plt, LineCollection, PolyCollection, Rectangle, MplPolygon, FontProperties
    global MplPath
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        from matplotlib.collections import LineCollection as _LC
        from matplotlib.collections import PolyCollection as _PC
        from matplotlib.font_manager import FontProperties as _FP
        from matplotlib.patches import Polygon as _MP
        from matplotlib.patches import Rectangle as _R
        from matplotlib.path import Path as _Path
        plt, LineCollection, PolyCollection, Rectangle, MplPolygon, FontProperties = (
            _plt, _LC, _PC, _R, _MP, _FP
        )
        MplPath = _Path


def _rich_title_font():
    """matplotlib font cache와 무관하게 설치된 한글 글꼴 파일을 직접 선택한다."""
    _ensure_mpl()
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
    ):
        if os.path.isfile(path):
            return FontProperties(fname=path)
    return FontProperties(family="monospace")

TOUCH_LABEL = {TOUCH_PASS: "PASS", TOUCH_SHOOT: "SHOOT", TOUCH_PASS_HEAD: "H-PASS",
               TOUCH_SHOOT_HEAD: "H-SHOOT", TOUCH_DRIBBLE: "DRIBBLE", TOUCH_TACKLE: "TACKLE",
               TOUCH_GK_CATCH: "GK-CATCH", TOUCH_INTERCEPT: "INTERCEPT",
               TOUCH_DEFLECT: "DEFLECT", TOUCH_PARRY: "PARRY",
               TOUCH_BODY_TRAP: "BODY-TRAP"}



_BONES = 12
"""`_skeletons`가 선수당 내놓는 선분 수(몸통·팔·다리). 뼈대 경로 병합의 스트라이드 기준."""

_BONE_VERTS = _BONES * 3
"""병합 경로의 정점 수 — 선분마다 [시작, 끝, NaN] 세 칸."""

_MARKER_POLY_SIDES = 32
"""원형 마커를 대신할 정다각형의 변 수. 32각형이면 반지름 대비 최대 이탈이 0.12%라
Full HD에서 반지름 10 px짜리 머리 마커도 0.02 px 안쪽으로 원과 일치한다."""


def _flatten_circle_markers(*collections):
    """산점도의 원형 마커 경로를 같은 크기의 정다각형으로 바꾼다 — 프레임당 7 ms를 아낀다.

    matplotlib의 ``Collection.draw``는 '경로 1개 + 단색' 산점도에 단일경로 최적화를 걸지만
    그 판정을 위해 **매 프레임** ``Path.get_extents``를 부른다. 베지어 원(CURVE4 4개)이면
    극값을 ``np.roots``(companion 행렬 고윳값)로 풀어 한 번에 1.8 ms가 드는데, 이 렌더는
    그런 산점도를 넷 그리므로 프레임당 7.4 ms — Full HD 그리기 시간의 15%가 마커
    바운딩박스 계산에 쓰였다. 다각형은 같은 호출이 0.08 ms다.

    반지름은 **원래 경로의 범위에서 그대로 가져온다**. scatter가 보관하는 경로는 마커
    변환(‘o’는 ×0.5)이 이미 반영된 것이라 단위원으로 가정하면 마커가 두 배로 커진다.
    정점을 원 위에 얹으면 다각형이 원보다 조금 작으므로 면적이 같아지는 반지름
    ``sqrt(2π / (n·sin(2π/n)))``을 곱해 오차를 안팎으로 나눈다.
    """
    n = _MARKER_POLY_SIDES
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    grow = np.sqrt(2.0 * np.pi / (n * np.sin(2.0 * np.pi / n)))
    unit = np.column_stack([np.cos(th), np.sin(th)]) * grow
    for col in collections:
        paths = col.get_paths()
        if len(paths) != 1 or paths[0].codes is None:
            continue
        if MplPath.CURVE4 not in paths[0].codes:
            continue                      # 이미 다각형(사각형·삼각형 마커)
        ext = paths[0].get_extents()      # 초기화 때 한 번 — 여기서만 베지어 극값을 푼다
        cx, cy = ext.x0 + 0.5 * ext.width, ext.y0 + 0.5 * ext.height
        verts = np.column_stack([cx + 0.5 * ext.width * unit[:, 0],
                                 cy + 0.5 * ext.height * unit[:, 1]])
        col.set_paths([MplPath(verts, closed=True)])


_FRAME_RING = 6
"""light 렌더가 돌려쓰는 프레임 버퍼 수. 인코더 지터를 흡수하면서 메모리는 6 × 0.9 MB뿐이다."""

_FRAME_QUEUE = _FRAME_RING - 2
"""라이터 큐 길이. 링 크기보다 **둘** 작아야 한다.

프로듀서가 슬롯 ``f % R``에 쓰기 시작하는 순간 아직 인코딩이 끝나지 않았을 수 있는 프레임은
큐에 든 Q개와 워커가 쥐고 쓰는 중인 1개, 즉 ``f-1 … f-(Q+1)``이다. 이 슬롯을 마지막으로 쓴
프레임은 ``f-R``이므로 겹치지 않으려면 ``R ≥ Q+2``여야 한다. 처음엔 ``Q = R-1``로 두었는데
그 경우 ``f-R = f-(Q+1)``이 정확히 경계에 걸려, 워커가 그 프레임을 파이프에 쓰는 동안
프로듀서가 같은 버퍼를 덮어썼다 — 같은 입력을 두 번 렌더하면 프레임 151/161이 달랐다."""

_ENCODE_THREADS = 4
"""x264에 넘길 스레드 수 상한.

ffmpeg 기본값은 코어 수를 자동 감지하는데, 128코어 장비에서 672×448짜리 작은 프레임을
그만큼 쪼개면 스레드 관리 비용이 인코딩 자체를 압도한다(실측: 자동 267 fps, 2스레드 421,
4스레드 468, 8스레드 381). 프레임이 작을수록 최적점이 낮아 4로 고정한다 — 코어가 4개
미만인 장비에서는 그 수에 맞춘다."""


def _encode_params(crf):
    """인코더 공통 파라미터. 스레드 상한을 한 군데서 정한다."""
    threads = max(1, min(_ENCODE_THREADS, os.cpu_count() or 1))
    return ["-preset", "veryfast", "-crf", str(crf), "-threads", str(threads)]


class _ThreadedFrameWriter:
    """mp4 인코딩을 전담 스레드로 옮기는 얇은 래퍼.

    ffmpeg는 이미 별도 프로세스지만 ``append_data``가 파이프에 **동기로** 쓰기 때문에
    래스터 루프가 매 프레임 인코더를 기다린다. 쓰기를 스레드로 넘기면 총시간이 두 작업의
    합이 아니라 큰 쪽이 된다. 실패는 삼켜서는 안 되므로 예외를 보관했다가 다음 append나
    close에서 그대로 올린다 — 그러지 않으면 인코딩이 죽어도 빈 mp4가 남는다.
    """

    def __init__(self, path, fps, output_params):
        _ensure_render_dependencies()
        self._writer = imageio.get_writer(
            path, fps=fps, codec="libx264", quality=None,
            pixelformat="yuv420p", macro_block_size=None,
            output_params=list(output_params),
        )
        self._queue = queue.Queue(maxsize=max(1, _FRAME_QUEUE))
        self._error = None
        # 스레드는 **첫 프레임을 쓴 뒤에** 띄운다. imageio는 프레임 크기를 알아야 인코더를
        # 띄울 수 있어 ffmpeg fork가 첫 append_data에서 일어나는데, 그 fork가 워커 스레드에서
        # 발생하면 다른 스레드가 잡은 락을 물려받은 자식이 생긴다. 첫 프레임만 메인 스레드에서
        # 동기로 쓰면 fork가 메인 스레드에서 일어난다.
        self._thread = None

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            try:
                if self._error is None:
                    self._writer.append_data(item)
            except BaseException as exc:  # noqa: BLE001 - 호출자에게 그대로 전달한다
                self._error = exc
            finally:
                self._queue.task_done()

    def _raise_if_failed(self):
        if self._error is not None:
            error, self._error = self._error, None
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001 - 원래 오류를 가리지 않는다
                pass
            raise error

    def append_data(self, frame):
        self._raise_if_failed()
        if self._thread is None:
            self._writer.append_data(frame)      # 이 호출이 ffmpeg를 띄운다(메인 스레드)
            self._thread = threading.Thread(
                target=self._run, name="render-encode", daemon=True
            )
            self._thread.start()
            return
        self._queue.put(frame)

    def close(self):
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
        try:
            self._raise_if_failed()
        finally:
            self._writer.close()


def _mmss(seconds):
    """초 → MM:SS. 이벤트 레코드와 HUD가 같은 표기를 쓰도록 한 군데에 둔다."""
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


_VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".mov", ".webm", ".avi", ".gif"})
"""``render_mp4``가 받는 컨테이너 확장자 — imageio/ffmpeg가 쓸 수 있는 것들."""


def _video_time_lookup(video_t, render_fps):
    """경기 시계(control tick)를 **영상 시각(초)** 으로 옮기는 함수를 만든다.

    이벤트·트래킹 스택 ``F``와 실제로 인코딩된 영상은 같은 격자가 아닐 수 있다
    (자세한 사정은 ``render_mp4`` 안의 주석). 두 격자가 공유하는 유일한 양은 경기
    시계이므로, 그 시계로 **영상 격자에서의 표본 번호**를 되찾은 뒤 render fps로
    나눈다. 격자가 같으면 표본 번호가 곧 행 번호라 결과는 ``행번호/render_fps``와
    정확히 일치한다.

    ``video_t``가 비어 있거나 None이면(격자 정보를 주지 않은 호출) 시계를 그대로
    영상 시각으로 쓰는 대신 0초 기준의 항등 대응으로 되돌아간다.
    """

    fps = float(render_fps)
    if video_t is None or len(video_t) == 0:
        return lambda t: 0.0
    grid = np.asarray(video_t, dtype=np.float64).reshape(-1)
    last = grid.size - 1

    def lookup(t):
        # side="left" — 그 제어 틱을 처음 보여주는 영상 프레임.
        index = int(np.searchsorted(grid, float(t), side="left"))
        return min(max(index, 0), last) / fps

    return lookup


def _remap_event_frames(records, source_t, video_t):
    """Project canonical event frame indices onto the encoded State grid.

    ``event_states`` may be the reset-inclusive control-rate trajectory while
    the video uses a physics-resampled grid.  Event ``f``/``outcome_frame``
    indices therefore cannot be consumed by a renderer until their source
    control tick has been located on the video grid.  Replay records retain
    their canonical source indices; this helper returns display-only copies.
    """

    if records is None:
        return None
    source = np.asarray(source_t, dtype=np.float64).reshape(-1)
    target = np.asarray(video_t, dtype=np.float64).reshape(-1)
    # The common path needs no projection.  Preserve the exact canonical
    # object so the HUD and replay writer demonstrably consume one extraction
    # result rather than two independently inferred event streams.
    if source.shape == target.shape and np.array_equal(source, target):
        return records
    copied = [dict(row) for row in records]
    if not copied or source.size == 0 or target.size == 0:
        return copied

    last = target.size - 1

    def project(index):
        index = int(index)
        if not (0 <= index < source.size):
            return index
        mapped = int(np.searchsorted(target, source[index], side="left"))
        return min(max(mapped, 0), last)

    for row in copied:
        if row.get("f") is not None:
            row["f"] = project(row["f"])
        if row.get("outcome_frame") is not None:
            row["outcome_frame"] = project(row["outcome_frame"])
    return copied


def _jax_backend():
    """실행 백엔드 이름. 조회 실패가 리플레이 기록을 막아서는 안 된다."""
    try:
        return str(jax.default_backend())
    except Exception:
        return "unknown"


def _jax_device_kind():
    """첫 디바이스의 종류(예: "cpu", "NVIDIA A100-SXM4-40GB")."""
    try:
        return str(jax.devices()[0].device_kind)
    except Exception:
        return "unknown"


def _jax_device_count():
    try:
        return int(len(jax.devices()))
    except Exception:
        return 0


def _restart_onsets(env, restart_t, restart_kind, restart_team):
    """Vectorised restart-instance onset mask for render/event consumers.

    Render used to infer an onset from ``kind != previous_kind``.  That loses a
    perfectly real THROWIN -> THROWIN (or any same-code) restart whose timer is
    rewound, and it also loses the opening kickoff because there is no previous
    frame.  Do not maintain a second rule here: prepend the inactive sentinel and
    call :meth:`Restart.restart_reopened`, the transition's instance SSOT, once for
    the complete trajectory.
    """

    restart_t = np.asarray(restart_t)
    restart_kind = np.asarray(restart_kind)
    restart_team = np.asarray(restart_team)
    if restart_t.size == 0:
        return np.zeros((0,), dtype=bool)
    before_t = np.concatenate([
        np.asarray([0], dtype=restart_t.dtype), restart_t[:-1]
    ])
    before_kind = np.concatenate([
        np.asarray([RK_NONE], dtype=restart_kind.dtype), restart_kind[:-1]
    ])
    before_team = np.concatenate([
        np.asarray([NO_TEAM], dtype=restart_team.dtype), restart_team[:-1]
    ])
    after = SimpleNamespace(
        restart_t=restart_t,
        restart_kind=restart_kind,
        restart_team=restart_team,
    )
    return np.asarray(
        env.restart_reopened(before_t, before_kind, before_team, after), dtype=bool
    )


def _new_touch_mask(current, previous):
    """Touch events newly observable in ``current``.

    ``State.touch`` is cleared *inside* every control step, so two returned
    control frames may legitimately contain the same non-zero code for the same
    actor (PASS -> PASS).  Comparing code values alone therefore drops the second
    event.  Conversely, a substep trajectory exposes the accumulator repeatedly
    while ``clock`` is unchanged, where emitting it on every sample would duplicate
    one contact.  A new control tick makes every non-zero code new; within one tick
    only a changed code is new.
    """

    current_touch = np.asarray(current.touch)
    previous_touch = np.asarray(previous.touch)
    new_control_tick = current.clock != previous.clock
    return (current_touch > TOUCH_NONE) & (
        new_control_tick | (current_touch != previous_touch)
    )


def _visual_possession_owner(
    team, player_pos, active, poss_team, ball_pos, restart_t=0, taker=NO_PLAYER
):
    """Return the player slot that should carry the visual possession ring.

    State deliberately stores team possession, not an individual owner.  The
    renderer therefore attributes a valid team possession to its designated
    restart taker, or otherwise to its nearest active team-mate.  Distance is
    never a validity gate: while ``poss_team`` remains valid, the ring remains
    on that team even if an opponent is closer or the ball is in flight.
    """

    team = np.asarray(team).astype(int)
    player_pos = np.asarray(player_pos)
    active = np.asarray(active, dtype=bool)
    poss_team = int(poss_team)
    eligible = active & (team == poss_team)
    if poss_team not in (TEAM_0, TEAM_1) or not bool(np.any(eligible)):
        return NO_PLAYER
    taker = int(taker)
    if int(restart_t) > 0 and 0 <= taker < team.size and bool(eligible[taker]):
        return taker
    distance = np.where(
        eligible,
        np.linalg.norm(player_pos - np.asarray(ball_pos)[:2][None, :], axis=1),
        np.inf,
    )
    return int(np.argmin(distance))


def _visual_offside_line(
    team, player_pos, active, attack_dir, poss_team, last_touch_team,
    ball_state, restart_t,
):
    """Return ``(attacking_team, world_x)`` for the live reference line.

    This is a display aid, not a second Law-11 implementation: the actual
    offence continues to use :mod:`offside`.  It shows the stricter of the
    second-last-defender and halfway references used by the existing rich HUD.
    Fewer than two active defenders means there is no finite defender line to
    draw, so the overlay is suppressed instead of fabricating one at halfway.
    """

    if int(ball_state) != BALL_ALIVE or int(restart_t) > 0:
        return NO_TEAM, None
    poss_team = int(poss_team)
    last_touch_team = int(last_touch_team)
    attacking = poss_team if poss_team in (TEAM_0, TEAM_1) else last_touch_team
    if attacking not in (TEAM_0, TEAM_1):
        return NO_TEAM, None
    team = np.asarray(team).astype(int)
    player_pos = np.asarray(player_pos)
    active = np.asarray(active, dtype=bool)
    attack_dir = np.asarray(attack_dir)
    attacking_slots = np.flatnonzero((team == attacking) & active)
    if attacking_slots.size == 0:
        return NO_TEAM, None
    direction = float(attack_dir[int(attacking_slots[0])])
    defenders = (team != attacking) & active
    if int(np.count_nonzero(defenders)) < 2:
        return NO_TEAM, None
    defender_x = player_pos[defenders, DIM_X] * direction
    second_last = float(np.partition(defender_x, -2)[-2])
    return attacking, max(second_last, 0.0) * direction


TOUCH_RING = {TOUCH_PASS: (240, 240, 240), TOUCH_SHOOT: (255, 150, 40),
              TOUCH_PASS_HEAD: (240, 240, 240), TOUCH_SHOOT_HEAD: (255, 150, 40),
              TOUCH_DRIBBLE: (120, 210, 140), TOUCH_TACKLE: (255, 70, 70),
              TOUCH_GK_CATCH: (90, 200, 255), TOUCH_INTERCEPT: (255, 210, 70),
              TOUCH_DEFLECT: (170, 170, 170), TOUCH_PARRY: (90, 200, 255),
              TOUCH_BODY_TRAP: (130, 225, 180)}

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
RING_POSSESSION_R = 7.5                  # px(근경 기준) — 팀 소유권 실선 링
EVENT_DEDUP_SECONDS = 0.8
RING_CARD_R = 7.0                       # 옐로카드 링 — 힘과 무관한 고정 반지름
RING_SPOT_R = 9.0                        # 퇴장 스팟 · 파울 링(고정)
_RESTART_FLIGHT_OUTCOME = {
    # 공이 실제로 경계를 넘어 죽은 경우만 out_of_play다.
    RK_THROWIN: "out_of_play", RK_GOALKICK: "out_of_play", RK_CORNER: "out_of_play",
    # 심판이 멈춘 경우 — 비행은 반칙/오프사이드로 끝났지 아웃된 것이 아니다.
    RK_OFFSIDE: "offside", RK_FREEKICK: "foul", RK_PENALTY: "foul",
    # 골키퍼가 손으로 잡아 죽은 공.
    RK_GK_HOLD: "saved",
}
"""발사 뒤 다음 접촉보다 먼저 열린 재개를 비행 결과로 옮기는 표."""

_RESTART_TERMINAL_OUTCOMES = frozenset(_RESTART_FLIGHT_OUTCOME.values())

FLIGHT_OUTCOMES = frozenset({
    # 접촉으로 끝난 비행
    "completed", "intercepted", "retained", "rebound_own", "blocked", "saved",
    # 득점으로 끝난 비행
    "goal", "own_goal",
    # 골 프레임(포스트·크로스바)에 맞고 끝난 비행. 축구 통계의 'hit the woodwork'이며
    # 빗나감과 구분해야 한다 — 유효슈팅도 아니고 단순 아웃도 아니다.
    "woodwork",
    # 리플레이가 잘려 결과를 못 본 경우
    "unresolved",
}) | _RESTART_TERMINAL_OUTCOMES
"""패스/슛 비행 결과의 유일한 어휘.

피드는 light/rich 두 벌이고 각자 if/elif로 라벨을 만든다. 어휘가 여기 한 곳에 없으면
새 결과가 한쪽 피드에서만 이름을 얻고 다른 쪽에서는 조용히 일반 폴백으로 떨어진다.
``_extract_events``가 이 집합 밖의 값을 내면 실패한다 — 등록을 강제하는 가드다.
"""

RESTART_LABEL = {RK_KICKOFF: "KICKOFF", RK_THROWIN: "THROW-IN", RK_GOALKICK: "GOALKICK",
                 RK_CORNER: "CORNER", RK_FREEKICK: "FREEKICK", RK_PENALTY: "PENALTY",
                 RK_OFFSIDE: "OFFSIDE-FK", RK_GK_HOLD: "GK-HOLD"}
BALL_EVENT_LABEL = {
    BALL_EVENT_GOAL: "GOAL", BALL_EVENT_CORNER: "OUT(GOAL LINE)",
    BALL_EVENT_GOALKICK: "OUT(GOAL LINE)", BALL_EVENT_THROWIN: "OUT(TOUCHLINE)",
}
"""라인 통과 사건의 표시 이름. 코너/골킥은 같은 골라인 통과이고 최종 터치 팀만 다르다."""
WOODWORK_LABEL = {
    WOODWORK_POST: "POST", WOODWORK_CROSSBAR: "CROSSBAR",
}
"""골 프레임 충돌의 표시 이름."""
BALL_EVENT_RESTART_KIND = {
    BALL_EVENT_CORNER: RK_CORNER, BALL_EVENT_GOALKICK: RK_GOALKICK,
    BALL_EVENT_THROWIN: RK_THROWIN,
}
"""아웃 → 그 아웃이 만든 재개 종류. 재개 스폿이 교차점을 잃는 경우만 담는다.

골로 생긴 킥오프는 여기 없다 — 킥오프의 공은 아무 라인도 넘지 않았고, 그 교차점은
원인인 ``goal`` 레코드에 실린다. 재개 레코드에 붙이면 '이 재개가 라인을 넘었다'로 읽힌다.
"""

FOUL_LABEL = {
    FOUL_TACKLE: "FOUL(TACKLE)", FOUL_CHARGE: "FOUL(CHARGE)",
    FOUL_THROW: "FOUL(THROW RETOUCH)", FOUL_SETPIECE: "FOUL(SETPIECE RETOUCH)",
}

COL_PITCH = (30, 74, 40); COL_LINE = (150, 185, 158); COL_HUD = (16, 20, 17)
def layout_label(index):
    """레이아웃 인덱스 → 사람이 읽는 이름. 표 밖 값은 숨기지 않고 드러낸다."""

    index = int(index)
    names = formation_module.LAYOUT_NAMES
    return names[index] if 0 <= index < len(names) else f"layout#{index}"


TEAM_LABEL = {TEAM_0: "HOME", TEAM_1: "AWAY"}
"""표시용 팀 이름의 단일 진실원천. 슬롯 인덱스(0/1)는 내부 표현이고 사람이 읽는
출력에는 노출하지 않는다 — ``TEAM_0``은 홈, ``TEAM_1``은 원정이다."""


def team_label(value):
    """팀 코드를 표시 이름으로. 중립·미상은 ``"--"``."""
    return TEAM_LABEL.get(int(value), "--") if value is not None else "--"


_TEAM_ROSTER_PREFIX = {TEAM_0: "H", TEAM_1: "A"}


def _roster_display_labels(env):
    """Return stable team-local display numbers keyed by global person ID.

    ``State.player_id`` is a match-global identity and therefore cannot reuse
    11..19 for both teams.  The renderer still needs football-sized labels, so
    starters receive H0..H10/A0..A10 in roster order and bench seats continue
    at H11/A11.  This mapping is presentation metadata only: replay/event data
    retain the authoritative global IDs and substitution provenance stays
    unambiguous.
    """

    labels = {}
    rosters = (
        (TEAM_0, tuple(env.agent_team)),
        (TEAM_1, tuple(env.opponent_team)),
    )
    bench_plan = np.asarray(env._bench_plan["player_id"])
    for team, players in rosters:
        prefix = _TEAM_ROSTER_PREFIX[team]
        for number, player in enumerate(players):
            labels[int(player.id)] = f"{prefix}{number}"
        for seat, player_id in enumerate(bench_plan[team]):
            player_id = int(player_id)
            if player_id >= 0:
                labels[player_id] = f"{prefix}{len(players) + seat}"
    return labels


def _display_person_id(value, labels=None):
    """Format a global person ID without confusing it with a shirt number."""

    try:
        player_id = int(value)
    except (TypeError, ValueError):
        return "?"
    if player_id < 0:
        return "?"
    return (labels or {}).get(player_id, str(player_id))


COL_TEAM = ((208, 68, 58), (66, 118, 208))          # HOME 적 / AWAY 청
COL_GK = ((255, 140, 120), (140, 190, 255))         # GK는 밝은 톤
COL_BALL = (248, 248, 248); COL_SHADOW = (16, 36, 22)
COL_TEXT = (225, 232, 226)
COL_OFFSIDE = (255, 198, 64)
COL_SPIN_POS = (57, 176, 255)
COL_SPIN_NEG = (255, 107, 107)
# 감독 명령은 득점의 노란색과 의미가 섞이지 않게 별도 팔레트를 쓴다. 포메이션은
# 전술판의 하늘색, 교체는 투입을 뜻하는 녹색이다(light/rich가 이 RGB를 함께 사용).
COL_COMMAND_FORMATION = (92, 196, 255)
COL_COMMAND_SUBSTITUTION = (105, 225, 145)


def _coach_command_feed(records, *, frame_count, every=1, person_labels=None):
    """Canonical events/6 감독 이벤트를 중앙 명령 카드용 프레임 목록으로 옮긴다.

    ``every``의 의미는 rich 이벤트 피드와 같다. 원본 프레임이 샘플 사이에 있으면
    그 이벤트 뒤의 첫 retained frame에 보여 미래 명령을 당겨 표시하지 않는다.
    반환 항목은 두 렌더러가 그대로 소비하는 표시 계약이라, light/rich가 서로 다른
    문구나 타이밍을 독립적으로 추론하지 않는다. 포메이션 행은 목표 layout이 명령
    경계에서 바뀌었다는 사실만 말한다. 선수 좌표의 이후 이동을 phase로 추측하지 않는다.
    """

    if isinstance(every, (bool, np.bool_)) or not isinstance(every, numbers.Integral):
        raise TypeError(f"every must be a positive integer, got {every!r}")
    every = int(every)
    if every <= 0:
        raise ValueError(f"every must be > 0, got {every!r}")
    frame_count = int(frame_count)
    out = [[] for _ in range(max(0, frame_count))]
    if not out:
        return out

    for row in records:
        kind = row.get("type")
        if kind not in ("formation", "substitution"):
            continue
        source = row.get("f")
        if source is None or int(source) < 0:
            continue
        target = min((int(source) + every - 1) // every, frame_count - 1)
        tm = row.get("team")
        tm = int(tm) if tm in (TEAM_0, TEAM_1) else None
        if kind == "formation":
            command = {
                "kind": kind,
                "team": tm,
                "headline": f"{team_label(tm)} FORMATION TARGET",
                "detail": f"{row.get('from', '?')}  ->  {row.get('to', '?')}",
                "color": COL_COMMAND_FORMATION,
            }
        else:
            # 한 교체 기회에 여러 명이 동시에 바뀔 수 있다. 행별 카드를 네 개까지만
            # 보이면 다섯 번째 교체를 화면에서 잃으므로, 같은 팀/프레임은 OUT/IN 명단
            # 한 줄로 합친다. 그러면 한 프레임의 최대 카드는 양 팀 × (모양, 교체)=4다.
            command = next(
                (item for item in out[target]
                 if item["kind"] == "substitution" and item["team"] == tm),
                None,
            )
            if command is None:
                command = {
                    "kind": kind,
                    "team": tm,
                    "headline": f"{team_label(tm)} SUBSTITUTION",
                    "out": [],
                    "in": [],
                    "color": COL_COMMAND_SUBSTITUTION,
                }
                out[target].append(command)
            command["out"].append(
                _display_person_id(row.get("player_out"), person_labels)
            )
            command["in"].append(
                _display_person_id(row.get("player_in"), person_labels)
            )
            command["detail"] = (
                f"OUT {','.join(map(str, command['out']))}  ->  "
                f"IN {','.join(map(str, command['in']))}"
            )
            continue
        out[target].append(command)
    return out


def _roster_status_text(
    team, bench_ids, retired_ids, subs_remaining, *, person_labels=None
):
    """rich 하단에 표시할 현재 벤치/교체 OUT 명단.

    감독이 고를 수 있는 사람과 이미 나간 사람을 같은 ``inactive slot`` 그림으로
    뭉개지 않는다. 아주 큰 벤치에서도 HUD 절반을 넘지 않도록 첫 여덟 명을 보이고
    남은 수를 명시한다. 내부 person ID는 이벤트/리플레이에 그대로 남기고, 화면에서는
    ``person_labels``가 있으면 H11/A11 같은 팀별 명단 번호를 쓴다.
    """

    def visible(values, limit):
        ids = [int(value) for value in np.asarray(values).reshape(-1) if int(value) >= 0]
        shown = " ".join(
            _display_person_id(value, person_labels) for value in ids[:limit]
        ) or "--"
        if len(ids) > limit:
            shown += f" +{len(ids) - limit}"
        return shown

    remaining = int(subs_remaining)
    return (
        f"{team_label(team)} BENCH  {visible(bench_ids, 8)}\n"
        f"OUT  {visible(retired_ids, 5)}    SUBS LEFT {remaining}"
    )


def _rich_reserve_people(frame, bench_x0, pitch_width):
    """Return visible reserve/retired people for the rich touchline.

    Bench candidates live outside the fixed field-slot axis, so moving inactive
    field slots into the dugout can never draw them.  This display-only layout
    gives every valid reserve a body in the dugout and places substituted-out
    players on a separate row.  Dismissed field players retain the distinct
    off-pitch coordinates already stored by the environment.
    """

    people = []
    bench_y = -(float(pitch_width) / 2.0 + 3.5)
    retired_y = -(float(pitch_width) / 2.0 + 6.4)
    for team in (TEAM_0, TEAM_1):
        for kind, values, y in (
            ("bench", frame.bench_player_id[team], bench_y),
            ("retired", frame.retired_player_id[team], retired_y),
        ):
            visible = [
                int(value)
                for value in np.asarray(values).reshape(-1)
                if int(value) >= 0
            ]
            for order, player_id in enumerate(visible):
                people.append({
                    "kind": kind,
                    "team": team,
                    "player_id": player_id,
                    "pos": np.asarray(
                        [float(bench_x0[team]) + 2.35 * order, y],
                        np.float64,
                    ),
                })
    return people


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
_DASHED_RING_CACHE = {}


def _ring_cached(radius, width=1.4):
    """반지름/폭을 0.5px 격자로 양자화해 링 스탬프를 캐시. 임펄스 링은 이벤트마다 반지름이
    달라 사전계산 테이블을 못 쓰므로, 재사용되는 이산 크기만 만들어 두고 돌려쓴다."""
    key = (round(float(radius) * 2.0) / 2.0, round(float(width) * 2.0) / 2.0)
    hit = _RING_CACHE.get(key)
    if hit is None:
        hit = _RING_CACHE[key] = _ring(max(key[0], 1.0), key[1])
    return hit


def _dashed_ring_cached(radius, width=1.4, dashes=10, duty=0.56):
    """Cached angularly dashed ring used only for physical ball impulses."""

    key = (
        round(float(radius) * 2.0) / 2.0,
        round(float(width) * 2.0) / 2.0,
        int(dashes),
        round(float(duty), 2),
    )
    hit = _DASHED_RING_CACHE.get(key)
    if hit is None:
        radius_q, width_q, dash_count, duty_q = key
        radius_q = max(radius_q, 1.0)
        r = int(np.ceil(radius_q + width_q))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        d2 = yy * yy + xx * xx
        annulus = (
            (d2 <= (radius_q + width_q) ** 2)
            & (d2 >= (radius_q - width_q) ** 2)
        )
        phase = np.mod(
            (np.arctan2(yy, xx) + np.pi) * dash_count / (2.0 * np.pi),
            1.0,
        )
        mask = annulus & (phase < duty_q)
        hit = _DASHED_RING_CACHE[key] = (yy[mask], xx[mask])
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

    def _light_view_geometry(self):
        """확장 선수 경계를 포함하는 light 렌더 투영 파라미터의 SSOT.

        피치 크기·경계 여유·캔버스 폭에만 의존하는 상수라 한 번만 계산해 캐시한다.
        픽셀 변환마다 다시 계산하면 40초 영상 한 편에서 27 000번 넘게 호출된다
        (래스터 시간의 12 %).
        """

        cached = getattr(self, "_light_view_cache", None)
        if cached is None:
            hx = self.s_cfg.length / 2.0
            hy = self.s_cfg.width / 2.0
            margin = self.e_cfg.player_boundary_margin
            ppm = min(float(self._PPM), (self._W - 8.0) / (2.0 * (hx + margin)))
            cached = (hy + margin, ppm)
            self._light_view_cache = cached
        return cached

    def _light_to_px(self, wx, wy):
        """확장 선수 좌표를 포함하는 월드→light 화면 투영(스칼라)."""

        view_hy, ppm = self._light_view_geometry()
        d = (view_hy - wy) / (2.0 * view_hy)
        return (
            int(self._Y_FAR + d * (self._Y_NEAR - self._Y_FAR)),
            int(self._W // 2 + wx * ppm * (self._FAR_SCALE + (1.0 - self._FAR_SCALE) * d)),
            d,
        )

    def _light_to_px_many(self, pos):
        """(N,2) 월드좌표를 한 번에 투영 — 스칼라 판과 **같은 절단 규칙**을 쓴다.

        프레임마다 선수 수만큼 파이썬 호출을 반복하던 것을 배열 연산 한 번으로 바꾼다.
        int()와 numpy 정수 캐스팅은 둘 다 0 방향 절단이라 결과가 비트 단위로 같다.
        """

        view_hy, ppm = self._light_view_geometry()
        pos = np.asarray(pos, dtype=np.float64)
        d = (view_hy - pos[:, 1]) / (2.0 * view_hy)
        cy = (self._Y_FAR + d * (self._Y_NEAR - self._Y_FAR)).astype(np.int64)
        cx = (self._W // 2
              + pos[:, 0] * ppm * (self._FAR_SCALE + (1.0 - self._FAR_SCALE) * d)).astype(np.int64)
        return cy, cx, d

    def render_mp4(self, states, out_path=None, fps=DEFAULT_RENDER_FPS, mode="light", trail_decay=0.88,
                   event_states=None,
                   event_ttl=8, title=None, verbose=True, batch_index=0,
                   dump_state=False, dump_events=False, dump_replay=True,
                   state_stride=1, frame_annotations=None,
                   replay_metadata=None, **rich_kwargs):
        """State 시퀀스를 mp4로 렌더. states: State 리스트 또는 T-선두축으로 스택된 State pytree.
        out_path=None이면 실행 경로 기준 ./replays/<unix초>/match.mp4에 저장(호출마다 새 폴더,
        같은 초 충돌 시 +1s). title은 HUD 라벨(팀/스타일 표기용).

        mode: "light"(기본) = 순수 numpy 탑다운 경량 렌더(수백 fps). "rich" = 원본 SOCCER 방송형
        렌더(render_rich.RichRenderer — 3D 카메라·관절 스켈레톤·관중석·미니맵 등, matplotlib 필요·저속).
        rich 전용 옵션(dynamic_cam·cam_zoom·feed_n·feed_secs·every·dpi 등)은 **rich_kwargs로 전달**된다.

        dump_state=True면 mp4 옆에 `state.jsonl`(1행=meta, 이후 **행당 1프레임** state — 스트리밍
        기록·부분 읽기/tail 가능), dump_events=True면 `events.json`(구조화 이벤트 레코드:
        접촉·패스/슛 결과·재개·골·파울·카드·퇴장·교체·소유전환·오프사이드·하프타임·GK홀드)과
        `timeline.jsonl`(프레임당 시계·국면만 담은 경량 타임로그)을 함께 쓴다. 텍스트 로그는
        더 이상 쓰지 않는다 — 포맷이 곧 스키마라 필드를 늘릴 때마다 소비자가 깨진다. **이 트래킹/이벤트 덤프는 렌더 모드와 무관하게 동일
        코드경로(_stack_full→_write_*)로 생성**되므로 light·rich가 완전히 같은 산출물을 낸다.
        state_stride>1이면 프레임 행을 그만큼 솎는다(이벤트 로그는 항상 전 프레임 스캔이라 무손실).

        ``dump_replay=True``는 기본 정규 리플레이 계약이다. 영상 옆에 ``events.jsonl``(행당
        이벤트), ``tracking.jsonl``(행당 전체 프레임), ``metadata.jsonl``(행당 메타 주제)을
        쓴다. 영상만 필요한 호출자는 ``dump_replay=False``를 명시한다.
        ``dump_state``/``dump_events``는 개별 산출물만 원하는 호출자를 위해 남긴다.

        ``fps``는 입력 ``states``의 표본률이자 encoder 표본률이다. 이 저수준 API는 State
        배열만 보고 원래 control/physics 표본률을 추측하지 않는다. 서로 다른 Hz를 변환할 때는
        ``SoccerEnv.substep_trajectory``로 먼저 재표본화하거나, 서브스텝 수집까지 소유하는
        ``demo_match.py``를 사용한다.

        batch_index: **배치 차원 호환** — 학습용 vmap 롤아웃처럼 states에 배치 축이 있으면(규약: 시간-
        선두·배치-2번축 `(T, B, ...)`, 즉 lax.scan(시간)+vmap(env)의 자연 출력) 그 중 한 경기만 골라
        `(T, ...)`로 낮춰 렌더한다(기본은 첫 경기). 배치가 없으면 0만 허용한다. light·rich·덤프 전부에 적용.
        반환: 기본/canonical replay 요청 시
        canonical replay면 {"mp4","events","tracking","metadata"}, legacy dump면
        {"mp4","state","events","timeline"} dict. ``dump_replay=False``이고 legacy
        dump도 없을 때만 mp4 경로 문자열을 반환한다."""
        if not isinstance(mode, str) or mode not in ("light", "rich"):
            raise ValueError(
                f"mode must be one of ('light', 'rich'), got {mode!r}"
            )
        rich_ctor_keys = {
            "azim", "elev", "dist", "focal", "figsize", "dpi",
            "trail_len", "player_scale",
        }
        rich_render_keys = {
            "every", "feed_n", "feed_secs", "dynamic_cam", "cam_zoom",
        }
        if mode == "light" and rich_kwargs:
            names = ", ".join(sorted(rich_kwargs))
            raise TypeError(
                f"rich-only render option(s) are not valid in light mode: {names}"
            )
        unknown_rich = set(rich_kwargs) - rich_ctor_keys - rich_render_keys
        if unknown_rich:
            names = ", ".join(sorted(unknown_rich))
            raise TypeError(f"unknown rich render option(s): {names}")

        # 공개 진입점의 공통 옵션은 파일/encoder를 만들기 전에 모두 검증한다.  imageio나
        # 덤프 writer에 맡기면 잘못된 값이 한참 뒤에 실패하면서 빈/부분 mp4와 디렉터리를 남긴다.
        def _bool_option(name, value):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be a boolean, got {value!r}")
            return bool(value)

        def _positive_integral(name, value):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{name} must be a positive integer, got {value!r}")
            value = int(value)
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value!r}")
            return value

        def _finite_real_option(name, value, *, positive=False):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
                raise TypeError(f"{name} must be a finite real number, got {value!r}")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
            if positive and value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value!r}")
            return value

        if isinstance(fps, (bool, np.bool_)) or not isinstance(fps, numbers.Real):
            raise TypeError(f"fps must be a positive finite number, got {fps!r}")
        fps = float(fps)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"fps must be a positive finite number, got {fps!r}")

        if (isinstance(trail_decay, (bool, np.bool_))
                or not isinstance(trail_decay, numbers.Real)):
            raise TypeError(
                f"trail_decay must be a finite number in [0, 1], got {trail_decay!r}"
            )
        trail_decay = float(trail_decay)
        if not np.isfinite(trail_decay) or not (0.0 <= trail_decay <= 1.0):
            raise ValueError(
                f"trail_decay must be a finite number in [0, 1], got {trail_decay!r}"
            )

        event_ttl = _positive_integral("event_ttl", event_ttl)
        state_stride = _positive_integral("state_stride", state_stride)
        verbose = _bool_option("verbose", verbose)
        dump_state = _bool_option("dump_state", dump_state)
        dump_events = _bool_option("dump_events", dump_events)
        dump_replay = _bool_option("dump_replay", dump_replay)
        if title is not None and not isinstance(title, str):
            raise TypeError(f"title must be a string or None, got {title!r}")
        if replay_metadata is not None:
            if not isinstance(replay_metadata, Mapping):
                raise TypeError("replay_metadata must be a mapping or None")
            replay_metadata = dict(replay_metadata)
            try:
                json.dumps(replay_metadata, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "replay_metadata must contain finite JSON-serializable values"
                ) from exc

        # Rich options previously failed only after RichRenderer had built its
        # camera/figure (and after the destination directory was created), or
        # were silently coerced into nonsensical displays such as zero event
        # rows and a zero-length trail.  Canonicalise the complete allowed set
        # here, alongside the common preflight above.
        if mode == "rich":
            for name in ("azim", "elev"):
                if name in rich_kwargs:
                    rich_kwargs[name] = _finite_real_option(name, rich_kwargs[name])
            for name in ("dist", "focal", "dpi", "player_scale", "feed_secs"):
                if name in rich_kwargs:
                    rich_kwargs[name] = _finite_real_option(
                        name, rich_kwargs[name], positive=True
                    )
            for name in ("trail_len", "every", "feed_n"):
                if name in rich_kwargs:
                    rich_kwargs[name] = _positive_integral(name, rich_kwargs[name])
            if "dynamic_cam" in rich_kwargs:
                rich_kwargs["dynamic_cam"] = _bool_option(
                    "dynamic_cam", rich_kwargs["dynamic_cam"]
                )
            if "cam_zoom" in rich_kwargs:
                cam_zoom = _finite_real_option("cam_zoom", rich_kwargs["cam_zoom"])
                if not (0.0 < cam_zoom <= 1.0):
                    raise ValueError(
                        f"cam_zoom must satisfy 0 < cam_zoom <= 1, got {cam_zoom!r}"
                    )
                rich_kwargs["cam_zoom"] = cam_zoom
            if "figsize" in rich_kwargs:
                try:
                    figsize = tuple(rich_kwargs["figsize"])
                except TypeError as exc:
                    raise TypeError(
                        "figsize must be a pair of positive finite numbers"
                    ) from exc
                if len(figsize) != 2:
                    raise ValueError(
                        f"figsize must contain exactly two values, got {figsize!r}"
                    )
                rich_kwargs["figsize"] = tuple(
                    _finite_real_option(f"figsize[{i}]", value, positive=True)
                    for i, value in enumerate(figsize)
                )
        if out_path is not None:
            try:
                out_path = os.fspath(out_path)
            except TypeError as exc:
                raise TypeError(f"out_path must be path-like or None, got {out_path!r}") from exc
            if not isinstance(out_path, str) or not out_path:
                raise ValueError(f"out_path must be a non-empty text path, got {out_path!r}")
            # 확장자는 인코더가 컨테이너를 고르는 유일한 단서다. 여기서 막지 않으면
            # 디렉터리 경로를 넘긴 호출이 **롤아웃과 렌더를 다 끝낸 뒤** imageio 안쪽에서
            # "unknown file extension"으로 죽는다(실측: 3분 경기 렌더를 통째로 버렸다).
            suffix = os.path.splitext(out_path)[1].lower()
            if suffix not in _VIDEO_SUFFIXES:
                raise ValueError(
                    f"out_path must name a video file, got {out_path!r} "
                    f"(suffix {suffix or 'none'!r}); "
                    f"supported: {', '.join(sorted(_VIDEO_SUFFIXES))}"
                )

        # 배치 차원 호환: (T, B, ...) 롤아웃이면 한 경기(batch_index)만 골라 (T, ...)로 — 이후 경로
        # (light·rich·덤프)는 전부 배치 없는 단일 궤적만 본다(단일 진입 정규화).
        states = self._normalize_states(states, batch_index, verbose=verbose)
        if frame_annotations is not None:
            try:
                frame_annotations = list(frame_annotations)
            except TypeError as exc:
                raise TypeError("frame_annotations must be an iterable or None") from exc
            frame_count = int(np.shape(states.ball_pos)[0])
            if len(frame_annotations) != frame_count:
                raise ValueError(
                    "frame_annotations length "
                    f"{len(frame_annotations)} != state frames {frame_count}"
                )

        # 화면 이벤트와 canonical events.jsonl은 모드와 무관하게 반드시 같은
        # 레코드를 소비한다. 패스/슛 결과는 미래 접촉을 봐야 하므로 렌더 전에
        # 한 번만 추출하고 rich/light HUD와 replay writer가 같은 객체를 재사용한다.
        # _normalize_states가 이미 numpy 스택을 만들었으므로 _stack_full의 np.asarray는
        # 복사가 아닌 view이며, 추가 일은 이벤트 O(T·N) 스캔 한 번이다.
        # 이벤트 검출은 영상보다 한 표본 앞을 봐야 할 수 있다. 오프닝 킥오프가 그렇다 —
        # ``_restart_onsets``는 비활성 센티넬을 앞에 붙여 f=0 온셋을 잡도록 만들어져 있지만,
        # 호출자가 리셋 표본을 잘라내면(영상 프레임 수를 맞추려고) 그 프레임 자체가 없어
        # 킥오프가 통째로 사라진다. 영상은 ``states``, 이벤트는 ``event_states``로 가른다.
        render_stack = self._stack_full(
            states if event_states is None else self._normalize_states(
                event_states, batch_index, verbose=False
            )
        )
        # ``event_states`` is the reset-inclusive control-rate trajectory.  Its
        # frame durations belong to control_fps, not the encoded render fps;
        # using 25 here for a 15 Hz source shortened every pass/shot flight by
        # 40%.  Without a separate event grid, F is the video grid itself.
        event_sample_fps = (
            float(self.control_fps) if event_states is not None else fps
        )
        structured_events = self._extract_events(
            render_stack, sample_fps=event_sample_fps
        )
        display_events = _remap_event_frames(
            structured_events,
            render_stack["t"],
            np.asarray(states.t),
        )

        reserved_default_dir = False
        if out_path is None:
            run_id = int(time.time())
            while True:
                output_dir = os.path.join("replays", str(run_id))
                try:
                    # Reserve the default destination atomically.  An
                    # exists()+makedirs(exist_ok=True) pair lets concurrent
                    # calls select and overwrite the same match.mp4.
                    os.makedirs(output_dir)
                except FileExistsError:
                    if not os.path.isdir(output_dir):
                        raise
                    run_id += 1
                    continue
                out_path = os.path.join(output_dir, "match.mp4")
                reserved_default_dir = True
                break
        if os.path.dirname(out_path) and not reserved_default_dir:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if mode == "rich":
            # RichRenderer는 이 파일 하단에 통합됨(구 render_rich.py). matplotlib은 render()에서 지연 로드.
            # rich_kwargs를 생성자(카메라·해상도)와 render()(카메라 팬·피드) 인자로 분배.
            ctor_kw = {k: rich_kwargs[k] for k in rich_ctor_keys if k in rich_kwargs}
            render_kw = {k: rich_kwargs[k] for k in rich_render_keys if k in rich_kwargs}
            renderer = RichRenderer(self, roster=_roster_from_states(states), **ctor_kw)
            renderer.render(states, out_path, fps=fps, title=title, verbose=verbose,
                            frame_annotations=frame_annotations,
                            structured_events=display_events,
                            **render_kw)
        else:
            self._render_light(states, out_path, fps=fps, trail_decay=trail_decay,
                               event_ttl=event_ttl, title=title, verbose=verbose,
                               frame_annotations=frame_annotations,
                               structured_events=display_events)

        # ── 트래킹/이벤트 덤프(모드 공유) ── mp4를 무엇으로 그렸든 동일 산출물.
        if not (dump_state or dump_events or dump_replay):
            return out_path
        outputs = {"mp4": out_path}
        d = os.path.dirname(out_path) or "."
        # HUD와 replay writer가 이미 공유한 full state/event 스캔을 그대로 재사용한다.
        F = render_stack
        # ``F``(이벤트·트래킹 스택)와 **실제로 인코딩된 영상**은 같은 격자가 아니다.
        # ``event_states``를 주면 F는 리셋 표본을 포함한 control 격자이고, 영상은
        # ``states``(재표본화된 render 격자)다. F의 행 **번호**를 render fps로 나누면
        # 두 격자가 섞여 영상 시각이 통째로 어긋난다 — 실측: 90.0초 이벤트가
        # video_time_s=54.0을 가리키고, metadata가 2250프레임 90초 영상을
        # 1351프레임 54.04초라고 적었다. 두 격자가 공유하는 유일한 양은 **경기 시계**이니
        # 시계로 영상 격자의 표본 번호를 되찾은 뒤 render fps로 나눈다. 영상이 실시간
        # 배속이 아닐 수도 있으므로(control 15 Hz를 fps=25로 인코딩하면 1.67배속)
        # 경기 시계를 그대로 영상 시각으로 쓰면 안 된다.
        video_frames = int(np.shape(states.ball_pos)[0])
        video_t = np.asarray(states.t, dtype=np.float64).reshape(-1)
        if dump_replay:
            # ``F`` may be a 25 Hz substep trajectory even when control runs at
            # another rate.  Event look-ahead/flight durations are therefore
            # measured on the rendered sample clock, while the public match
            # clock remains State.t / control_fps.
            events = (
                structured_events
                if structured_events is not None
                else self._extract_events(F, sample_fps=fps)
            )
            ev_path = os.path.join(d, "events.jsonl")
            n_ev = self._write_events_jsonl(
                F, ev_path, events=events, render_fps=fps, video_t=video_t,
            )
            outputs["events"] = ev_path
            tracking_path = os.path.join(d, "tracking.jsonl")
            n_tracking = self._write_tracking_jsonl(
                F, tracking_path, state_stride, render_fps=fps, video_t=video_t,
            )
            outputs["tracking"] = tracking_path
            metadata_path = os.path.join(d, "metadata.jsonl")
            self._write_replay_metadata_jsonl(
                F,
                metadata_path,
                events=events,
                render_fps=fps,
                mode=mode,
                title=title,
                state_stride=state_stride,
                rich_options=rich_kwargs,
                user_metadata=replay_metadata,
                video_filename=os.path.basename(out_path),
                video_frames=video_frames,
            )
            outputs["metadata"] = metadata_path
            if verbose:
                print(f"[render] 이벤트 {n_ev}건 → {ev_path}")
                print(
                    f"[render] tracking {n_tracking}프레임"
                    f"(stride {state_stride}) → {tracking_path}"
                )
                print(f"[render] metadata → {metadata_path}")
        if dump_events:
            ev_path = os.path.join(d, "events.json")
            n_ev = self._write_events_json(F, ev_path, render_fps=fps)
            outputs["events_legacy" if dump_replay else "events"] = ev_path
            tl_path = os.path.join(d, "timeline.jsonl")
            n_tl = self._write_timeline_jsonl(F, tl_path, state_stride)
            outputs["timeline"] = tl_path
            if verbose:
                print(f"[render] 이벤트 {n_ev}건 → {ev_path}")
                print(f"[render] 타임로그 {n_tl}프레임 → {tl_path}")
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
              ``0 < render_fps <= physics_fps``여야 한다. 출력 프레임 수는 물리 궤적 길이를 목표
              fps 시간축에 가장 가깝게 양자화하고, 각 출력 프레임의 끝 시각에 가장 가까운 물리
              표본을 고른다. 마지막 출력은 항상 마지막 물리 표본이라 궤적 endpoint가 보존된다.
        반환: (F, ...)-선두 State pytree — 그대로 `render_mp4(traj, fps=render_fps)`에 전달.

        예)
          def step(c, _):
              o, st, k = c; k, ka, ks = jax.random.split(k, 3)
              o2, st2, _, _, info = env.step_env_array(ks, st, policy(o, ka), collect_substeps=True)
              return (o2, st2, k), info["substeps"]           # (decimation, State)
          (_, _, _), subs = lax.scan(step, init, None, length=T)   # subs: (T, decimation, State)
          traj = env.substep_trajectory(subs, render_fps=25)
          env.render_mp4(traj, fps=25, mode="rich")
        """
        physics_fps = float(self.timebase.physics_fps)
        dec = self.timebase.decimation
        leaves = jax.tree_util.tree_leaves(substeps_stack)
        if not leaves:
            raise ValueError("substeps_stack must contain at least one array leaf")
        T = int(leaves[0].shape[0])
        sample_count = T * dec
        flat = jax.tree_util.tree_map(
            lambda x: x.reshape((sample_count,) + x.shape[2:]), substeps_stack)   # (T*dec, ...) 물리순 평탄화
        if render_fps is not None:
            try:
                target_fps = float(render_fps)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"render_fps must be a finite number, got {render_fps!r}"
                ) from exc
            if not np.isfinite(target_fps) or not (0.0 < target_fps <= physics_fps):
                raise ValueError(
                    f"render_fps must satisfy 0 < render_fps <= {physics_fps:g}, "
                    f"got {render_fps!r}"
                )
        if sample_count == 0:
            return flat

        if render_fps is None:
            # 전 서브스텝 밀도 — 출력 표본이 곧 물리 표본이다. 이 경로도 아래 이벤트
            # 투영을 똑같이 거친다. 원본 버퍼는 제어 프레임 안에서 **누적**이라
            # 그대로 두면 한 접촉이 substep 수만큼 반복되어 보인다. 출력 표현은
            # render_fps와 무관하게 언제나 '구간 버퍼' 하나뿐이어야 한다.
            frame_count = sample_count
            indices = np.arange(sample_count, dtype=np.int64)
            trajectory = flat
        else:
            # 정수 stride는 90 Hz -> 25 fps를 22.5 Hz로 만들어 60초 영상을 54초로 축소한다.
            # 목표 fps의 각 frame-end 시각을 물리 표본 시각에 직접 대응시켜 비정수 비율도 보존한다.
            # 출력 개수는 가장 가까운 정수(half-up)이고, target<=physics라 index는 엄격히 증가한다.
            frame_count = max(
                1, int(np.floor(sample_count * target_fps / physics_fps + 0.5))
            )
            target_ends = (
                np.arange(frame_count, dtype=np.float64) + 1.0
            ) / target_fps
            indices = np.rint(target_ends * physics_fps).astype(np.int64) - 1
            indices = np.clip(indices, 0, sample_count - 1)
            indices[-1] = sample_count - 1
            trajectory = jax.tree_util.tree_map(lambda x: x[indices], flat)

        # ``touch_event_*`` is cumulative only inside one control frame and is
        # reset at the next frame boundary.  Point-sampling the physical states
        # can therefore jump from just before a late contact to just after that
        # reset and silently erase the event (90 Hz -> 25 Hz is a common case).
        # Project every event in each output sample's half-open physical-time
        # interval into that sample instead.  The canonical source is the last
        # substate of each control frame, where all ``decimation`` slots are
        # complete; the local (substep, force/body) coordinates retain causal
        # order for ``_extract_events``.
        identity_event_fields = (
            "touch_event_actor",
            "touch_event_code",
            "touch_event_player_id",
        )
        telemetry_event_specs = (
            ("touch_event_control_t", -1, ()),
            ("touch_event_toi", 0.0, ()),
            ("touch_event_ball_pos", 0.0, (DIM_ALL,)),
            ("touch_event_ball_vel_before", 0.0, (DIM_ALL,)),
            ("touch_event_ball_vel_after", 0.0, (DIM_ALL,)),
            ("touch_event_impulse", 0.0, ()),
        )
        if all(hasattr(substeps_stack, name) for name in identity_event_fields):
            telemetry_presence = [
                hasattr(substeps_stack, name)
                for name, _, _ in telemetry_event_specs
            ]
            if any(telemetry_presence) and not all(telemetry_presence):
                raise ValueError(
                    "ordered touch telemetry fields must be all present or all absent"
                )
            event_specs = [
                ("touch_event_actor", NO_PLAYER, ()),
                ("touch_event_code", TOUCH_NONE, ()),
                ("touch_event_player_id", NO_PLAYER, ()),
            ]
            if all(telemetry_presence):
                event_specs.extend(telemetry_event_specs)

            sources = {
                name: np.asarray(getattr(substeps_stack, name)[:, -1])
                for name, _, _ in event_specs
            }
            expected_prefix = (T, dec, TOUCH_EVENT_PHASE_COUNT)
            for name, _, trailing_shape in event_specs:
                expected_shape = expected_prefix + trailing_shape
                if sources[name].shape != expected_shape:
                    raise ValueError(
                        f"{name} must have shape {expected_shape}, got "
                        f"{sources[name].shape}"
                    )

            interval = {
                name: np.full(
                    (frame_count,) + source.shape[1:],
                    fill_value,
                    dtype=source.dtype,
                )
                for (name, fill_value, _), source in (
                    (spec, sources[spec[0]]) for spec in event_specs
                )
            }
            source_actor = sources["touch_event_actor"]
            source_code = sources["touch_event_code"]
            valid = (
                (source_actor >= 0)
                & (source_actor < self.N)
                & (source_code > TOUCH_NONE)
                & (source_code < TOUCH_COUNT)
            )
            # Terminal transitions are exact State copies, including the last
            # control frame's cumulative touch buffer.  Treating every copied
            # canonical row as a fresh interval repeats that final contact for
            # the rest of a resampled video.  The originating control tick and
            # local cell are a stable event identity.  Rows without that
            # telemetry pair the frozen State.t with identity fields instead.
            source_state_t = np.asarray(substeps_stack.t[:, -1]).astype(int)
            source_control_t = sources.get("touch_event_control_t")
            source_player_id = sources["touch_event_player_id"]
            has_telemetry = all(telemetry_presence)
            source_toi = sources.get("touch_event_toi")
            seen_event_keys = set()
            pending_by_frame = {}
            for control_frame, substep, phase in np.argwhere(valid):
                q = int(substep)
                p = int(phase)
                exact_control_t = (
                    int(source_control_t[control_frame, q, p])
                    if source_control_t is not None else -1
                )
                if exact_control_t >= 0:
                    event_key = ("contact", exact_control_t, q, p)
                else:
                    event_key = (
                        "legacy",
                        int(source_state_t[control_frame]),
                        q,
                        p,
                        int(source_actor[control_frame, q, p]),
                        int(source_code[control_frame, q, p]),
                        int(source_player_id[control_frame, q, p]),
                    )
                if event_key in seen_event_keys:
                    continue
                seen_event_keys.add(event_key)
                physics_index = int(control_frame) * dec + q
                output_frame = int(np.searchsorted(
                    indices, physics_index, side="left"
                ))
                if output_frame >= frame_count:
                    raise RuntimeError(
                        "touch event lies beyond the resampled trajectory endpoint"
                    )
                # 인과 정렬 키 — 제어 tick, 그 안의 서브스텝 + 충돌 시각, 마지막이 위상.
                # ``_extract_events``가 쓰는 순서와 같은 재료다.
                fractional = (
                    q + float(source_toi[control_frame, q, p])
                    if source_toi is not None else float(q)
                )
                order = (
                    exact_control_t if exact_control_t >= 0
                    else int(source_state_t[control_frame]),
                    fractional, q, p,
                )
                pending_by_frame.setdefault(output_frame, []).append(
                    (order, int(control_frame), q, p)
                )

            # 낮은 render_fps에서는 한 출력 표본이 여러 제어 프레임을 덮으므로 서로 다른
            # 접촉이 같은 (substep, phase) 칸을 요구할 수 있다. 칸은 저장 위치일 뿐이고
            # 인과 순서는 ``touch_event_control_t``/``touch_event_toi``가 지고 있으니,
            # 자리를 다투면 빈 칸으로 밀어 담는다. 버퍼 용량을 넘길 때만 실패한다.
            capacity = dec * TOUCH_EVENT_PHASE_COUNT
            for output_frame, items in sorted(pending_by_frame.items()):
                items.sort(key=lambda row: row[0])
                if len(items) > capacity:
                    raise ValueError(
                        "render_fps is too low to preserve ordered touch telemetry; "
                        f"sample {output_frame} would hold {len(items)} contacts but "
                        f"the substep buffer has room for {capacity}"
                    )
                taken = set()
                overflow = []
                for order, control_frame, q, p in items:
                    if (q, p) in taken:
                        overflow.append((order, control_frame, q, p))
                    else:
                        taken.add((q, p))
                        for name, _, _ in event_specs:
                            interval[name][output_frame, q, p] = sources[name][
                                control_frame, q, p
                            ]
                if overflow and not has_telemetry:
                    # 텔레메트리가 없으면 재배치한 접촉의 인과 순서를 되살릴 방법이 없다.
                    raise ValueError(
                        "render_fps is too low to preserve ordered touch telemetry "
                        "without touch_event_control_t/touch_event_toi; raise "
                        "render_fps or record ordered touch telemetry"
                    )
                free = [
                    (fq, fp)
                    for fq in range(dec)
                    for fp in range(TOUCH_EVENT_PHASE_COUNT)
                    if (fq, fp) not in taken
                ]
                for (order, control_frame, q, p), (fq, fp) in zip(overflow, free):
                    for name, _, _ in event_specs:
                        interval[name][output_frame, fq, fp] = sources[name][
                            control_frame, q, p
                        ]

            trajectory = trajectory._replace(**{
                name: jnp.asarray(values) for name, values in interval.items()
            })

        # ``ball_event_*``도 제어 프레임 안에서만 유효한 substep 버퍼다. ``touch_event_*``만
        # 구간 투영하고 이쪽을 점 표본으로 두면 골·아웃 판정의 **교차점과 교차 속도**가
        # 통째로 사라진다(실측: 원본 골 셀이 90 Hz -> 25 Hz 출력에서 소멸). 점수와 재개는
        # 남지만 그 판정을 만든 물리량은 잃는다.
        # 골 프레임 충돌 버퍼도 같은 substep 버퍼라 같은 투영이 필요하다. 두 버퍼는 서로
        # 다른 사건이라 칸을 나눠 쓰지 않으므로(크로스바 맞고 골라인 통과는 한 substep에서
        # 둘 다 일어난다) 투영도 각자 돌린다.
        substep_buffers = (
            (
                ("ball_event_kind", "ball_event_team", "ball_event_pos",
                 "ball_event_vel", "ball_event_control_t"),
                "ball_event_kind", "ball_event_control_t", BALL_EVENT_NONE,
                {"ball_event_kind": BALL_EVENT_NONE,
                 "ball_event_control_t": -1,
                 "ball_event_team": NO_TEAM},
            ),
            (
                ("woodwork_kind", "woodwork_pos", "woodwork_vel_in",
                 "woodwork_control_t"),
                "woodwork_kind", "woodwork_control_t", WOODWORK_NONE,
                {"woodwork_kind": WOODWORK_NONE, "woodwork_control_t": -1},
            ),
        )
        for (ball_fields, kind_field, control_field, none_value,
             blank_values) in substep_buffers:
            if not all(hasattr(flat, name) for name in ball_fields):
                continue
            kind = np.asarray(getattr(flat, kind_field))          # (표본, 슬롯)
            starts = np.concatenate(
                [[0], indices[:-1] + 1]).astype(np.int64) if frame_count > 1 \
                else np.zeros(1, np.int64)
            filled = kind != none_value
            # 구간마다 **마지막으로 채워진 표본**을 슬롯별로 고른다. 프레임마다 파이썬
            # 루프를 돌면 긴 클립에서 느려지므로 reduceat 한 번으로 끝낸다.
            ordinal = np.where(
                filled, np.arange(kind.shape[0], dtype=np.int64)[:, None], -1)
            last = np.maximum.reduceat(ordinal, starts, axis=0)
            # 버퍼는 제어 프레임 안에서 **누적**이라, 한 제어 프레임이 여러 출력 표본에
            # 걸치면 같은 이벤트가 연속 표본에 반복 투영된다(실측: 같은 골 셀이 [0,2]와
            # [1,2]에 중복). ``(control_t, 슬롯)``이 안정된 신원이므로, 앞선 표본에서 이미
            # 나온 것은 지운다 — ``_extract_events``는 ID로 걸러 내지만 raw trajectory와
            # state dump에는 그대로 남기 때문이다.
            control_source = np.asarray(getattr(flat, control_field))
            picked_ct = control_source[np.where(last >= 0, last, 0),
                                       np.arange(kind.shape[1])[None, :]]
            fresh = last >= 0
            previous = np.full(kind.shape[1], -2, np.int64)
            for frame in range(fresh.shape[0]):
                repeat = fresh[frame] & (picked_ct[frame] == previous)
                fresh[frame] &= ~repeat
                previous = np.where(fresh[frame], picked_ct[frame], previous)
            last = np.where(fresh, last, -1)
            take = np.where(last >= 0, last, 0)
            slot = np.arange(kind.shape[1])[None, :]
            projected = {}
            for name in ball_fields:
                source = np.asarray(getattr(flat, name))
                picked = source[take, slot]
                blank = blank_values.get(name, 0)
                keep = last >= 0
                if picked.ndim == 3:
                    keep = keep[:, :, None]
                projected[name] = jnp.asarray(
                    np.where(keep, picked, np.asarray(blank, source.dtype)))
            trajectory = trajectory._replace(**projected)
        return trajectory

    def _restart_status_label(self, restart_kind, restart_t):
        """Return the set-piece readiness badge from the runtime timing SSOT."""

        rk = int(restart_kind)
        rt = int(restart_t)
        timing_cache = getattr(self, "_render_restart_timing_cache", None)
        if timing_cache is None:
            timing_cache = {}
            self._render_restart_timing_cache = timing_cache
        timing = timing_cache.get(rk)
        if timing is None:
            window_value, threshold_value = self._restart_window_ready_threshold(
                np.int32(rk)
            )
            timing = (
                int(np.asarray(window_value)),
                int(np.asarray(threshold_value)),
            )
            timing_cache[rk] = timing
        window, ready_threshold = timing
        prefix = (
            "PK" if rk == RK_PENALTY
            else "HOLD" if rk == RK_GK_HOLD
            else "FK" if rk in (RK_FREEKICK, RK_OFFSIDE)
            else "TAKER"
        )
        if rt >= window:
            return f"{prefix}  SET UP"
        seconds = max(0.0, rt - ready_threshold) * self.e_cfg.dt_phys
        return (
            f"{prefix}  {seconds:0.1f}s"
            if seconds > 0.05
            else f"{prefix}  READY"
        )

    def _light_event_feed(self, records, *, frame_count, person_labels=None):
        """Build one compact Light ticker row per frame from canonical events.

        Launch touches remain handled by the physical impulse ticker.  Outcomes
        are deliberately mapped to ``outcome_frame`` so Light never presents a
        future receiver at launch and agrees exactly with ``events.jsonl``.
        At most two co-located events are retained to fit the 672 px HUD.
        """

        frame_count = max(0, int(frame_count))
        pending = [[] for _ in range(frame_count)]
        if not pending or records is None:
            return [""] * frame_count

        shot_goal_frames = {
            int(row["outcome_frame"])
            for row in records
            if row.get("type") == "shot_outcome"
            and row.get("outcome") == "goal"
            and row.get("outcome_frame") is not None
        }

        def player(value):
            if value is None:
                return "P?"
            value = int(value)
            return f"P{value}" if 0 <= value < self.N else "P?"

        def team(value):
            return team_label(value) if value in (TEAM_0, TEAM_1) else "--"

        def team_for_player(value, explicit=None):
            if explicit in (TEAM_0, TEAM_1):
                return team(explicit)
            if value is None:
                return "--"
            value = int(value)
            if not (0 <= value < self.N):
                return "--"
            return team(TEAM_0 if value < self.n_agents else TEAM_1)

        def emit(row, label, priority, *, outcome=False):
            source = row.get("outcome_frame") if outcome else row.get("f")
            if source is None:
                return
            source = int(source)
            if 0 <= source < frame_count:
                pending[source].append((int(priority), str(label)))

        for row in records:
            kind = row.get("type")
            tm = row.get("team")
            tm_label = team(tm)
            actor = player(row.get("player"))

            if kind == "pass_outcome":
                result = row.get("outcome")
                if result == "unresolved":
                    continue
                if result == "completed":
                    label = f"PASS COMPLETE {tm_label} {actor}>{player(row.get('receiver'))}"
                elif result == "intercepted":
                    label = (
                        f"PASS INTERCEPTED BY "
                        f"{team_for_player(row.get('by'), row.get('by_team'))} "
                        f"{player(row.get('by'))}"
                    )
                elif result == "saved":
                    label = (
                        f"PASS CLAIMED BY "
                        f"{team_for_player(row.get('by'), row.get('by_team'))} "
                        f"{player(row.get('by'))}"
                    )
                elif result == "out_of_play":
                    label = f"PASS OUT {tm_label} {actor}"
                elif result == "retained":
                    # 자기에게 돌아온 킥은 패스가 아니라 캐리다.
                    label = f"CARRY {tm_label} {actor}"
                elif result == "goal":
                    label = f"PASS TO GOAL {tm_label} {actor}"
                elif result == "own_goal":
                    label = f"OWN GOAL {tm_label} {actor}"
                elif result == "woodwork":
                    label = (
                        f"OFF THE {str(row.get('woodwork', 'woodwork')).upper()} "
                        f"{tm_label} {actor}"
                    )
                else:
                    label = f"PASS {str(result).upper()} {tm_label} {actor}"
                emit(row, label, 90, outcome=True)
                continue

            if kind == "shot_outcome":
                result = row.get("outcome")
                if result == "unresolved":
                    continue
                if result == "goal":
                    label = f"GOAL {tm_label} {actor}"
                elif result == "own_goal":
                    label = f"OWN GOAL {tm_label} {actor}"
                elif result == "saved":
                    label = (
                        f"SHOT SAVED BY "
                        f"{team_for_player(row.get('by'), row.get('by_team'))} "
                        f"{player(row.get('by'))}"
                    )
                elif result == "blocked":
                    label = (
                        f"SHOT BLOCKED BY "
                        f"{team_for_player(row.get('by'), row.get('by_team'))} "
                        f"{player(row.get('by'))}"
                    )
                elif result == "out_of_play":
                    label = f"SHOT OFF TARGET {tm_label} {actor}"
                elif result == "rebound_own":
                    label = f"SHOT REBOUND {tm_label} {player(row.get('receiver'))}"
                elif result == "retained":
                    label = f"SHOT RETAINED {tm_label} {actor}"
                elif result == "woodwork":
                    label = (
                        f"OFF THE {str(row.get('woodwork', 'woodwork')).upper()} "
                        f"{tm_label} {actor}"
                    )
                else:
                    label = f"SHOT {str(result).upper()} {tm_label} {actor}"
                emit(row, label, 100, outcome=True)
                continue

            if kind == "goal":
                if int(row.get("f", -1)) not in shot_goal_frames:
                    emit(row, f"GOAL {tm_label}", 100)
                continue
            if kind == "card":
                emit(row, f"YELLOW CARD {tm_label} {actor}", 80)
                continue
            if kind == "sent_off":
                emit(row, f"SENT OFF {tm_label} {actor}", 95)
                continue
            if kind == "substitution":
                emit(
                    row,
                    f"SUB {tm_label} {player(row.get('slot'))} "
                    f"{_display_person_id(row.get('player_out'), person_labels)}"
                    f"->{_display_person_id(row.get('player_in'), person_labels)}",
                    75,
                )
                continue
            if kind == "formation":
                emit(
                    row,
                    f"FORMATION TARGET {tm_label} "
                    f"{row.get('from')} -> {row.get('to')}",
                    76,
                )
                continue
            if kind == "offside_flag":
                emit(row, f"OFFSIDE FLAG {tm_label} {actor}", 85)
                continue
            if kind == "possession":
                previous = row.get("from_team")
                if tm in (TEAM_0, TEAM_1) and previous in (TEAM_0, TEAM_1) and tm != previous:
                    emit(row, f"TURNOVER TO {tm_label}", 45)
                continue
            if kind == "restart":
                label = str(row.get("label", "RESTART"))
                emit(row, f"{label} {tm_label}", 40)
                continue
            if kind == "gk_hold_start":
                emit(row, f"GK HOLD {tm_label}", 40)
                continue
            if kind == "gk_hold_end":
                emit(row, f"GK RELEASE {tm_label}", 40)

        feed = []
        for rows in pending:
            labels = []
            for _, label in sorted(rows, key=lambda item: (-item[0], item[1])):
                if label not in labels:
                    labels.append(label)
                if len(labels) == 2:
                    break
            feed.append(" | ".join(labels))
        return feed

    def _draw_light_coach_commands(self, frame, commands):
        """골 알림처럼 중앙에 오래 남는 감독 명령 카드.

        같은 프레임에 양 팀의 포메이션/교체가 함께 들어올 수 있으므로 한 카드 안에
        최대 네 명령을 모두 표시한다. 득점과 겹치지 않도록 화면 중앙보다 조금 위에
        두며, 노란 골 팔레트 대신 명령 종류별 하늘색/녹색을 쓴다.
        """

        commands = list(commands[:4])
        if not commands:
            return
        specs = []
        if len(commands) == 1:
            command = commands[0]
            specs.append((command["headline"], 3, command["color"]))
            detail_scale = 2
            if self._text_mask(command["detail"], detail_scale).shape[1] > frame.shape[1] - 36:
                detail_scale = 1
            specs.append((command["detail"], detail_scale, command["color"]))
        else:
            specs.append(("COACH COMMANDS", 2, COL_TEXT))
            for command in commands:
                text = f"{command['headline']}  |  {command['detail']}"
                scale = 2 if self._text_mask(text, 2).shape[1] <= frame.shape[1] - 36 else 1
                specs.append((text, scale, command["color"]))

        masks = [self._text_mask(text, scale) for text, scale, _ in specs]
        gap = 4
        content_h = sum(mask.shape[0] for mask in masks) + gap * (len(masks) - 1)
        content_w = max(mask.shape[1] for mask in masks)
        panel_w = min(frame.shape[1] - 12, content_w + 24)
        panel_h = content_h + 18
        cx = frame.shape[1] // 2
        cy = frame.shape[0] // 2 - 68
        x0 = max(6, cx - panel_w // 2)
        x1 = min(frame.shape[1] - 6, x0 + panel_w)
        y0 = max(self._HUD_H + 6, cy - panel_h // 2)
        y1 = min(frame.shape[0] - 6, y0 + panel_h)
        border = tuple(commands[0]["color"])
        frame[y0:y1, x0:x1] = (7, 16, 22)
        frame[y0:y0 + 3, x0:x1] = border
        frame[y1 - 3:y1, x0:x1] = border
        frame[y0:y1, x0:x0 + 3] = border
        frame[y0:y1, x1 - 3:x1] = border
        y = y0 + 9
        for (text, scale, color), mask in zip(specs, masks):
            x = max(x0 + 6, cx - mask.shape[1] // 2)
            self._blit_text(frame, y, x, text, scale, color)
            y += mask.shape[0] + gap

    def _render_light(self, states, out_path, fps=DEFAULT_RENDER_FPS, trail_decay=0.88,
                      event_ttl=8, title=None, verbose=True,
                      frame_annotations=None, structured_events=None):
        """경량 numpy 탑다운 렌더(원 render_mp4 본체) — mp4만 기록.
        프레임 합성은 전부 numpy(플롯 라이브러리 무사용)라 수백 fps로 인코딩 제한까지 닿는다.
        경로 해석·모드 분기·트래킹 덤프는 render_mp4가 담당(여긴 순수 프레임 래스터+인코딩)."""
        S = self._stack_states(states)
        T = len(S["ball_pos"])
        H, W, ppm = self._H, self._W, self._PPM
        _, ppm = self._light_view_geometry()
        far = self._FAR_SCALE
        to_px = self._light_to_px

        base = self._pitch_base(to_px, ppm)
        # 배경은 상수이므로 int16 캐스팅을 한 번만 한다(프레임마다 하면 601프레임 × 0.9 MB).
        # 합성 버퍼도 매 프레임 새로 할당하지 않고 재사용한다 — 출력은 그대로다.
        base_i16 = base.astype(np.int16)
        trail_i16 = np.empty_like(base_i16)
        composite = np.empty_like(base_i16)
        frame_ring = [np.empty_like(base) for _ in range(_FRAME_RING)]
        trail = np.zeros((H, W, 3), np.float32)
        d_player = [_disc(3.0 + 0.45 * i) for i in range(4)]   # 깊이 버킷별 원판(원근 단서)
        d_trail, d_ball_sh = _disc(1.2), _disc(2.0)
        trail_col = [np.array(c, np.float32) * 0.40 for c in COL_TEAM]
        BV = S["ball_vel"]                             # 임펄스 링용 — 프레임 차분 Δv의 원천
        imp_span = max(float(self.e_cfg.f2b_speed_max) - RING_IMPULSE_MIN, 1e-3)
        team = np.asarray(S["team_id"][0]).astype(int)
        gk = np.asarray(S["gk_indices"][0]).astype(int)
        if structured_events is None:
            structured_events = self._extract_events(
                self._stack_full(states), sample_fps=fps
            )
        person_labels = _roster_display_labels(self)
        event_feed = self._light_event_feed(
            structured_events, frame_count=T, person_labels=person_labels
        )
        coach_feed = _coach_command_feed(
            structured_events, frame_count=T, person_labels=person_labels
        )

        # 소유자/오프사이드 참조선은 스탬프 루프 밖에서 한 번만 계산한다.
        # 두 모드가 같은 헬퍼를 쓰므로 거리 gate·비활성 선수 처리가 갈리지 않는다.
        possession_owner = np.full(T, NO_PLAYER, dtype=np.int32)
        offside_team = np.full(T, NO_TEAM, dtype=np.int8)
        offside_x = np.full(T, np.nan, dtype=np.float32)
        for f in range(T):
            possession_owner[f] = _visual_possession_owner(
                team,
                S["player_pos"][f],
                S["active_player"][f],
                S["poss_team"][f],
                S["ball_pos"][f],
                S["restart_t"][f],
                S["pending_taker"][f],
            )
            attacking, line_x = _visual_offside_line(
                team,
                S["player_pos"][f],
                S["active_player"][f],
                S["attack_dir"][f],
                S["poss_team"][f],
                S["last_touch_team"][f],
                S["ball_state"][f],
                S["restart_t"][f],
            )
            offside_team[f] = attacking
            if line_x is not None:
                offside_x[f] = line_x

        # 인코딩은 별도 프로세스(ffmpeg)에서 돌지만, append_data가 파이프에 동기로 쓰기
        # 때문에 래스터와 직렬화된다(측정: 래스터 1.13 s + 인코딩 1.43 s → 합 2.93 s).
        # 쓰기를 전담 스레드로 옮기면 둘이 겹쳐 총시간이 **둘 중 큰 쪽**으로 줄어든다.
        # 프레임 버퍼는 링으로 돌려 복사 없이 넘긴다 — 큐 길이를 링보다 하나 작게 잡아
        # 아직 인코딩되지 않은 버퍼를 덮어쓸 수 없게 한다.
        writer = _ThreadedFrameWriter(out_path, fps=fps, output_params=_encode_params(26))
        effects = []                                   # (ttl, kind, data) 활성 이벤트 플래시
        ticker = ""; goal_ttl = 0
        coach_hold = []; coach_ttl = 0
        annotation_text = ""; annotation_ttl = 0
        prev_score = S["score"][0].copy(); prev_foul = 0
        prev_active = S["active_player"][0].copy(); prev_yellow = S["yellow_cards"][0].copy()
        d_bench = _disc(3.0)
        # 역할 표시용 포메이션 앵커. 프레임마다 env를 부르면 렌더가 롤아웃만큼
        # 느려지므로, 활성 마스크/레이아웃 조합별로 한 번만 계산한다.
        # 앵커는 활성 인원에 따라 달라진다(퇴장하면 남은 선수가 재배치된다). 레이아웃과
        # 활성 마스크의 조합은 클립 안에서 몇 가지뿐이라 그 조합별로 한 번만 계산한다.
        anchor_cache = {}

        def layout_anchors(layout_row, active_row):
            key = (layout_row.tobytes(), active_row.tobytes())
            cached = anchor_cache.get(key)
            if cached is None:
                cached = np.asarray(self.formation_layout_anchors(
                    jnp.asarray(layout_row, jnp.int32),
                    jnp.asarray(active_row, bool)))
                anchor_cache[key] = cached
            return cached

        def layout_roles(layout_row, active_row):
            """레이아웃·생존 마스크 → 역할 코드 (N,).

            앵커 캐시와 **같은 키**를 쓴다 — 역할은 앵커의 함수이므로 앵커가 캐시되는
            순간 역할도 함께 캐시된다. 한 경기에서 서로 다른 키는 대개 한 자릿수라,
            프레임 루프에 드는 비용은 사실상 없다.
            """

            key = (layout_row.tobytes(), active_row.tobytes())
            cached = role_cache.get(key)
            if cached is None:
                cached = np.asarray(classify_roles(
                    jnp.asarray(layout_anchors(layout_row, active_row),
                                jnp.float32),
                    jnp.asarray(gk == 1, bool),
                    jnp.asarray(team, jnp.int32),
                    jnp.asarray(active_row, bool)))
                role_cache[key] = cached
            return cached

        role_cache = {}
        t_start = time.time()

        for f in range(T):
            P = S["player_pos"][f]; B = S["ball_pos"][f]
            stamina_short = S["stamina_short"][f]
            stamina_long = S["stamina_long"][f]
            yellow = S["yellow_cards"][f]
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
            # 선수 투영은 프레임당 한 번만 — 아래의 트레일·링·원판 루프가 모두 이 결과를 쓴다.
            pcy, pcx, pdf = self._light_to_px_many(P)
            for i in range(self.N):
                if not active[i]:
                    continue
                self._stamp(trail, pcy[i], pcx[i], d_trail,
                            trail_col[team[i]])
            bcy, bcx, bd = to_px(B[0], B[1])
            self._stamp(trail, bcy, bcx, d_trail, np.array(COL_BALL, np.float32))

            frame = frame_ring[f % _FRAME_RING]
            np.copyto(trail_i16, trail, casting="unsafe")      # float32 → int16, 0방향 절단
            np.add(base_i16, trail_i16, out=composite)
            np.clip(composite, 0, 255, out=composite)
            np.copyto(frame, composite, casting="unsafe")

            # 오프사이드 참조선: 현재 공격팀의 팀색 선+짧은 설명. 피치 레이어에
            # 먼저 그려 선수·공·이벤트 링이 자연스럽게 위에 올라오게 한다.
            attacking = int(offside_team[f])
            if attacking in (TEAM_0, TEAM_1) and np.isfinite(offside_x[f]):
                line_x = float(offside_x[f])
                hy = self.s_cfg.width / 2.0
                oy0, ox0, _ = to_px(line_x, hy)
                oy1, ox1, _ = to_px(line_x, -hy)
                self._line(frame, oy0, ox0, oy1, ox1, COL_TEAM[attacking])
                off_label = f"OFFSIDE LINE {team_label(attacking)}"
                off_w = self._text_mask(off_label, 1).shape[1]
                self._blit_text(
                    frame,
                    max(self._HUD_H + 2, min(self._H - 12, oy0 + 3)),
                    max(2, min(self._W - off_w - 2, ox0 + 4)),
                    off_label,
                    1,
                    COL_TEAM[attacking],
                )

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
                    dfrac = pdf[i]
                    r_px = ((RING_R_MIN + (RING_R_MAX - RING_R_MIN) * imp)
                            * (far + (1.0 - far) * dfrac))
                    effects.append([max(1, int(round(event_ttl * (0.7 + 0.8 * imp)))), "impulse",
                                    (i, TOUCH_RING.get(code, COL_TEXT), r_px)])
                    ticker = f"{mmss} {TOUCH_LABEL.get(code, '?')} P{i}({team_label(team[i])}) {dv:.1f}m/s"
                if yellow[i] > prev_yellow[i]:
                    effects.append([event_ttl * 2, "ring", (i, (250, 210, 40), RING_CARD_R)])
                    ticker = f"{mmss} YELLOW P{i}({team_label(team[i])})"
                if prev_active[i] and not active[i]:
                    effects.append([event_ttl * 3, "spot", ((pcy[i], pcx[i]), (230, 40, 40))])
                    ticker = f"{mmss} SENT OFF P{i}({team_label(team[i])})"
            if fk > 0 and prev_foul == 0:
                a, v = int(S["foul_actor"][f]), int(S["foul_victim"][f])
                effects.append([event_ttl * 2, "foul", (a, v)])
                # 재터치 반칙 등 피해자 없는 파울은 v=-1 — 음수 인덱싱(P[-1]=마지막 선수) 방지
                ticker = (f"{mmss} {FOUL_LABEL.get(fk, 'FOUL')} P{a}" if v < 0
                          else f"{mmss} {FOUL_LABEL.get(fk, 'FOUL')} P{a}>P{v}")
            score = S["score"][f]
            if int(score[0]) != int(prev_score[0]) or int(score[1]) != int(prev_score[1]):
                goal_ttl = fps; ticker = f"{mmss} GOAL  {int(score[0])}-{int(score[1])}"
            # canonical feed가 같은 프레임의 단순 터치 티커보다 우선한다.
            # 특히 패스/슛 결과는 발사가 아닌 실제 outcome_frame에 나온다.
            if event_feed[f]:
                ticker = f"{mmss} {event_feed[f]}"
            if coach_feed[f]:
                coach_hold = coach_feed[f]
                coach_ttl = max(1, int(round(fps * 2.4)))
            prev_foul = fk; prev_score = score.copy()
            prev_active = active.copy(); prev_yellow = yellow.copy()

            # 이벤트 플래시 렌더
            alive_fx = []
            for fx in effects:
                fx[0] -= 1
                ttl, kind, data = fx
                if kind in ("ring", "impulse"):
                    i, col, r_px = data
                    cy, cx = pcy[i], pcx[i]
                    # 힘 링만 점선, 카드 등 상태 링은 실선으로 두어 의미를 분리.
                    # 폭도 반지름에 따라 굵어진다(큰 힘일수록 진하게). ±1px 깜빡임은 유지.
                    ring_stamp = (
                        _dashed_ring_cached
                        if kind == "impulse"
                        else _ring_cached
                    )
                    self._stamp(frame, cy, cx,
                                ring_stamp(r_px + (1.0 if ttl % 2 else 0.0),
                                           1.2 + 0.06 * r_px),
                                col)
                elif kind == "spot":
                    (cy, cx), col = data
                    self._stamp(frame, cy, cx, _ring_cached(RING_SPOT_R), col)
                elif kind == "foul":
                    a, v = data
                    if a >= 0:
                        ay, ax_ = pcy[a], pcx[a]
                        if v >= 0:                       # 피해자 없는 파울(재터치 등)은 링만
                            vy, vx = pcy[v], pcx[v]
                            self._line(frame, ay, ax_, vy, vx, (255, 80, 80))
                        self._stamp(frame, ay, ax_, _ring_cached(RING_SPOT_R), (255, 80, 80))
                if fx[0] > 0:
                    alive_fx.append(fx)
            effects = alive_fx

            # 환경이 인정한 팀 소유권: 상대와의 거리와 무관한 고정 실선 팀색 링.
            # 점선 임펄스 효과 뒤에 다시 그려 항상 연속된 실선으로 남고,
            # 어두운 받침이 같은 팀색 선수 원판과의 경계를 보존한다.
            owner = int(possession_owner[f])
            if 0 <= owner < self.N:
                owner_scale = far + (1.0 - far) * float(pdf[owner])
                owner_radius = RING_POSSESSION_R * owner_scale
                self._stamp(
                    frame,
                    pcy[owner],
                    pcx[owner],
                    _ring_cached(owner_radius, 2.2),
                    COL_SHADOW,
                )
                self._stamp(
                    frame,
                    pcy[owner],
                    pcx[owner],
                    _ring_cached(owner_radius, 1.15),
                    COL_TEAM[team[owner]],
                )

            # 선수: 원판 + 단기(위)/장기(아래) stamina 바 + 옐로 마커.
            for i in np.argsort(-P[:, 1]):
                if not active[i]:
                    continue
                cy, cx, dfrac = pcy[i], pcx[i], pdf[i]
                col = (COL_GK if gk[i] == 1 else COL_TEAM)[team[i]]
                self._stamp(frame, cy, cx, d_player[min(3, int(dfrac * 4))], col)
                sx = cx - 4
                bars = (
                    (cy - 10, float(stamina_short[i]), "short"),
                    (cy - 7, float(stamina_long[i]), "long"),
                )
                for sy, value, kind in bars:
                    width = int(round(8 * float(np.clip(value, 0, 1))))
                    if 0 <= sy < self._H - 2 and 0 <= sx and sx + 8 < self._W:
                        frame[sy:sy + 2, sx:sx + 8] = (40, 44, 40)
                        if width:
                            value = float(np.clip(value, 0, 1))
                            color = (
                                (int(235 * (1 - value) + 20),
                                 int(165 * value + 65),
                                 int(170 * value + 55))
                                if kind == "short"
                                else (int(220 * (1 - value) + 40),
                                      int(180 * value + 50), 55)
                            )
                            frame[sy:sy + 2, sx:sx + width] = color
                if yellow[i] >= 1:
                    frame[max(cy - 9, 0):max(cy - 9, 0) + 3, min(cx + 5, self._W - 3):min(cx + 5, self._W - 3) + 3] = (250, 210, 40)

            # 패스 순간 래치된 오프사이드 플래그 선수: 상단의 황색 체버론.
            # 소유권/임펄스 링과 형태가 달라 겹쳐도 의미가 헷갈리지 않는다.
            flagged = np.asarray(S["offside_flag"][f], dtype=bool) & active
            for i in np.flatnonzero(flagged):
                my, mx = int(pcy[i]) - 11, int(pcx[i])
                self._line(frame, my - 3, mx - 3, my, mx, COL_OFFSIDE)
                self._line(frame, my - 3, mx + 3, my, mx, COL_OFFSIDE)

            # 데드볼 수행자는 소유권 링에 더해 짧은 TAKER 배지로 특정한다.
            restart_t = int(S["restart_t"][f])
            taker = int(S["pending_taker"][f])
            if restart_t > 0 and 0 <= taker < self.N and bool(active[taker]):
                # 역할까지 찍는다. "TAKER"만으로는 **누가** 찼는지는 보여도 그 선택이
                # 옳은지는 안 보인다 — 스로인을 풀백이 던지는지 센터백이 던지는지가
                # 세트피스 키커 정책이 작동하는지의 유일한 눈 확인 수단이다.
                taker_label = "TAKER " + ROLE_NAMES[int(
                    layout_roles(S["layout_index"][f], active)[taker])]
                taker_w = self._text_mask(taker_label, 1).shape[1]
                self._blit_text(
                    frame,
                    max(self._HUD_H + 2, min(self._H - 12, int(pcy[taker]) + 8)),
                    max(2, min(self._W - taker_w - 2, int(pcx[taker]) - taker_w // 2)),
                    taker_label,
                    1,
                    (255, 220, 120),
                )

            # 경기장 밖 세 구역: **되돌릴 수 있는가**로 줄이 갈린다. state의 _offpitch_zone과
            # 같은 순서로 그린다 — 터치라인에 가까울수록 아직 들어올 수 있는 사람이다.
            #   BENCH 아직 안 쓴 교체 카드 / OUT 교체로 나간 선수 / OFF 퇴장(돌아올 수 없다)
            sent_off_f = S["sent_off"][f]
            bench_ids = S["bench_player_id"][f]
            retired_ids = S["retired_player_id"][f]
            rows = (
                ("BENCH", self._Y_NEAR + 20, None,
                 [(t, int(v)) for t in (TEAM_0, TEAM_1)
                  for v in bench_ids[t] if int(v) >= 0]),
                ("OUT", self._Y_NEAR + 38, (215, 165, 60),
                 [(t, int(v)) for t in (TEAM_0, TEAM_1)
                  for v in retired_ids[t] if int(v) >= 0]),
                ("OFF", self._Y_NEAR + 56, (225, 45, 45),
                 [(int(team[i]), int(i)) for i in range(self.N) if sent_off_f[i]]),
            )
            for label, by, mark, members in rows:
                if not members:
                    continue
                self._blit_text(frame, by - 3, 4, label, 1, (110, 118, 112))
                seen = {TEAM_0: 0, TEAM_1: 0}
                for tid, tag in members:
                    k = seen[tid]; seen[tid] = k + 1
                    # HOME은 왼쪽에서 오른쪽으로, AWAY는 오른쪽에서 왼쪽으로 — 두 팀의
                    # 대기 인원이 한 줄에서 섞이지 않는다.
                    bx = 40 + k * 26 if tid == TEAM_0 else self._W - 40 - k * 26
                    self._stamp(frame, by, bx, d_bench,
                                tuple(int(c * 0.5) for c in COL_TEAM[tid]))
                    if mark is not None:
                        frame[by - 9:by - 5, bx + 3:bx + 6] = mark
                    display_tag = (
                        _display_person_id(tag, person_labels)
                        if label in ("BENCH", "OUT") else str(tag)
                    )
                    self._blit_text(frame, by + 5, bx - 8, display_tag, 1,
                                    (140, 148, 142))

            # 공: 그림자(지면) + 높이 투영 원판(z만큼 위로, 반지름도 z 비례. 깊이 스케일 일관).
            # 공중볼(z>0.5m)은 그림자-공 수직 연결선으로 높이를 즉시 읽히게.
            bz = float(B[2])
            lift = int(bz * ppm * (far + (1.0 - far) * bd))
            self._stamp(frame, bcy, bcx, d_ball_sh, COL_SHADOW)
            if bz > 0.5:
                self._line(frame, bcy, bcx, bcy - lift, bcx, (190, 195, 190))
            self._stamp(frame, bcy - lift, bcx, _disc(2.0 + min(bz, 6.0) * 0.5), COL_BALL)
            spin = np.asarray(S["ball_spin"][f], dtype=np.float64)
            spin_mag = float(np.linalg.norm(spin))
            if spin_mag > 0.5:
                phase = f * 0.9 + spin_mag * 0.3
                radius = 2.0 + 3.5 * min(spin_mag / max(float(self.e_cfg.spin_max), 1e-6), 1.0)
                dx = round(radius * np.cos(phase))
                dy = round(radius * np.sin(phase))
                self._line(
                    frame,
                    bcy - lift - dy,
                    bcx - dx,
                    bcy - lift + dy,
                    bcx + dx,
                    COL_SPIN_POS if spin[2] >= 0.0 else COL_SPIN_NEG,
                )

            # HUD: 시계 · 스코어 · 소유 · 재개 · 티커 · (득점 오버레이)
            frame[:self._HUD_H] = COL_HUD
            poss = int(S["poss_team"][f])
            self._blit_text(frame, 4, 8, f"{mmss}  {int(score[0])} - {int(score[1])}", 2)
            if poss >= 0:
                frame[10:22, 118:130] = COL_TEAM[poss]
            rk = int(S["restart_kind"][f])
            if rk != RK_NONE and restart_t > 0:
                restart_status = (
                    f"{RESTART_LABEL.get(rk, '')}  "
                    f"{self._restart_status_label(rk, restart_t)}"
                )
                self._blit_text(frame, 4, 150, restart_status, 2, (255, 220, 120))
            if title:
                tw = self._text_mask(title, 2).shape[1]
                self._blit_text(frame, 4, max(self._W - tw - 8, 320), title, 2, (190, 200, 192))
            # 양 팀의 현재 목표 모양 — **상시** 표기. 선수 좌표는 명령 경계에서 바뀌지 않고
            # 행동 정책과 물리를 통해 목표를 향해 이동한다.
            shape_y = self._Y_NEAR + 2
            for tm in (TEAM_0, TEAM_1):
                target = int(S["layout_index"][f][tm])
                text = layout_label(target)
                label = f"{team_label(tm)} {text}"
                width = self._text_mask(label, 1).shape[1]
                x = 8 if tm == TEAM_0 else max(8, self._W - width - 8)
                self._blit_text(frame, shape_y, x, label, 1,
                                tuple(int(c * 0.75) for c in COL_TEAM[tm]))
            if ticker:
                self._blit_text(frame, 26, 8, ticker, 1)
            if goal_ttl > 0:
                goal_ttl -= 1
                self._blit_text(frame, self._H // 2 - 16, self._W // 2 - 70, "G O A L", 4, (255, 230, 90))
            if coach_ttl > 0:
                coach_ttl -= 1
                self._draw_light_coach_commands(frame, coach_hold)
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
        if isinstance(batch_index, (bool, np.bool_)) or not isinstance(batch_index, numbers.Integral):
            raise TypeError(f"batch_index must be an integer, got {batch_index!r}")
        bi = int(batch_index)
        if bi < 0:
            raise ValueError(f"batch_index must be >= 0, got {bi}")
        if isinstance(states, (list, tuple)) and not hasattr(states, "_fields"):
            if not states:
                raise ValueError("states must contain at least one frame")
            stacked = jax.tree_util.tree_map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *states)
        else:
            stacked = jax.tree_util.tree_map(np.asarray, states)
        pp = np.asarray(stacked.player_pos)
        if pp.ndim not in (3, 4) or pp.shape[-1] != 2:
            raise ValueError(
                "states.player_pos must have shape (T,N,2) or (T,B,N,2), "
                f"got {pp.shape}"
            )
        T = int(pp.shape[0])
        if T == 0:
            raise ValueError("states must contain at least one frame")
        leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(stacked)]
        bad_time = [x.shape for x in leaves if x.ndim == 0 or x.shape[0] != T]
        if bad_time:
            raise ValueError(
                f"all state leaves must share the time dimension T={T}; got {bad_time[0]}"
            )
        if pp.ndim == 4:                                    # (T, B, N, 2) — 배치 축(1) 존재
            B = pp.shape[1]
            if B == 0 or bi >= B:
                raise ValueError(f"batch_index must satisfy 0 <= index < {B}, got {bi}")
            bad_batch = [x.shape for x in leaves if x.ndim < 2 or x.shape[1] != B]
            if bad_batch:
                raise ValueError(
                    f"all batched state leaves must share batch dimension B={B}; "
                    f"got {bad_batch[0]}"
                )
            stacked = jax.tree_util.tree_map(lambda x: x[:, bi] if np.ndim(x) >= 2 else x, stacked)
            if verbose:
                print(f"[render] 배치 감지 (B={B}) → 경기 {bi}만 렌더")
        elif bi != 0:
            raise ValueError(f"batch_index must be 0 for unbatched states, got {bi}")
        return stacked

    def _stack_states(self, states):
        """State 리스트/스택 pytree → 필요한 필드만 numpy dict (T, ...)로."""
        need = (
            "player_pos", "ball_pos", "ball_vel", "ball_spin", "ball_state",
            "stamina_short", "stamina_long", "yellow_cards", "active_player",
            "touch", "foul_kind",
            "foul_actor", "foul_victim", "restart_kind", "restart_t",
            "restart_team", "pending_taker", "score", "poss_team",
            "last_touch_team", "attack_dir", "offside_flag", "t", "team_id",
            "gk_indices", "sent_off", "bench_player_id", "retired_player_id",
            "layout_index",
        )
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
        fields = {k: np.asarray(getattr(stacked, k)) for k in stacked._fields}
        # active_player는 State 저장 필드가 아니라 on_pitch & ~sent_off projection이지만,
        # 기존 replay consumer가 읽을 수 있도록 dump에는 파생 sidecar로 계속 노출한다.
        fields["active_player"] = np.asarray(stacked.active_player)
        return fields

    # ── 리플레이 산출물 ────────────────────────────────────────────────
    # 사람이 읽는 텍스트 로그 대신 **기계가 읽는 두 파일**을 쓴다.
    #   events.json   무슨 일이 언제 일어났는가 (구조화 레코드)
    #   timeline.jsonl 매 프레임의 시계·국면 (탐색·정렬용 경량 타임로그)
    # 텍스트 로그는 파싱이 불안정하고(포맷이 곧 스키마), 필드를 추가할 때마다 소비자가
    # 깨진다. 같은 정보를 JSON으로 내면 스키마가 명시되고 필드 추가가 안전해진다.

    def _period_index(self, F):
        """전·후반 인덱스 — attack_dir 부호가 뒤집히는 지점이 하프타임이다.

        State에 period 필드가 없어 파생한다. 슬롯 0의 공격방향만 보면 교체·퇴장과 무관하게
        팀 방향 전환만 잡힌다.
        """
        d = np.sign(np.asarray(F["attack_dir"])[:, 0])
        flips = np.flatnonzero(np.diff(d) != 0)
        period = np.zeros(d.size, dtype=np.int16)
        for k, idx in enumerate(flips, start=1):
            period[idx + 1:] = k
        return period

    def _extract_events(self, F, sample_fps=None):
        """전 프레임 스캔으로 구조화된 이벤트 레코드를 만든다.

        기존 텍스트 로그가 잡던 것(터치·재개·골·파울·카드·퇴장)에 더해, 리플레이를 분석에
        쓰려면 반드시 필요한 것들을 새로 잡는다.
          * 패스/슛의 **결과** — 다음 접촉이 같은 팀이면 성공, 아니면 차단/탈취
          * 소유 전환 — 누가 언제 공을 가져갔는가
          * 오프사이드 플래그 온셋, 교체(슬롯 세대 변화), 하프타임 경계
          * GK 홀드 시작/종료
        결과 라벨은 **미래를 봐야** 정해지므로, 접촉을 먼저 모으고 뒤에서 이어 붙인다.
        """
        T = len(F["t"])
        N = self.N
        team = np.asarray(F["team_id"])[0].astype(int)
        control_fps = float(self.control_fps)
        sample_fps = (
            control_fps if sample_fps is None else float(sample_fps)
        )
        if not np.isfinite(sample_fps) or sample_fps <= 0.0:
            raise ValueError("sample_fps must be a positive finite number")
        period = self._period_index(F)
        touch = np.asarray(F["touch"])
        active = np.asarray(F["active_player"])
        ball_pos = np.asarray(F["ball_pos"])
        ball_vel = np.asarray(F["ball_vel"])
        poss = np.asarray(F["poss_team"]).astype(int)
        restart_kind = np.asarray(F["restart_kind"]).astype(int)
        restart_team = np.asarray(F["restart_team"]).astype(int)
        restart_ind = np.asarray(F["restart_indirect"]).astype(bool)
        score = np.asarray(F["score"]).astype(int)
        layout_index = (
            np.asarray(F["layout_index"]).astype(int)
            if "layout_index" in F else None
        )
        foul_kind = np.asarray(F["foul_kind"]).astype(int)
        foul_actor = np.asarray(F["foul_actor"]).astype(int)
        foul_victim = np.asarray(F["foul_victim"]).astype(int)
        yellow = np.asarray(F["yellow_cards"]).astype(int)
        offside = np.asarray(F["offside_flag"])
        player_id = np.asarray(F["player_id"]).astype(int)
        generation = np.asarray(F["slot_generation"]).astype(int)
        last_team = np.asarray(F["last_touch_team"]).astype(int)
        subs_remaining = (
            np.asarray(F["subs_remaining"]).astype(int)
            if "subs_remaining" in F else None
        )
        sub_windows_used = (
            np.asarray(F["sub_windows_used"]).astype(int)
            if "sub_windows_used" in F else None
        )
        ball_event_kind = (
            np.asarray(F["ball_event_kind"]).astype(int)
            if "ball_event_kind" in F else None
        )
        pending = np.asarray(F["pending_taker"]).astype(int)
        ordered_touch_actor = (
            np.asarray(F["touch_event_actor"]).astype(int)
            if "touch_event_actor" in F else None
        )
        ordered_touch_code = (
            np.asarray(F["touch_event_code"]).astype(int)
            if "touch_event_code" in F else None
        )
        ordered_touch_player_id = (
            np.asarray(F["touch_event_player_id"]).astype(int)
            if "touch_event_player_id" in F else None
        )
        ordered_touch_control_t = (
            np.asarray(F["touch_event_control_t"]).astype(int)
            if "touch_event_control_t" in F else None
        )
        ordered_touch_toi = (
            np.asarray(F["touch_event_toi"], dtype=np.float64)
            if "touch_event_toi" in F else None
        )
        ordered_touch_ball_pos = (
            np.asarray(F["touch_event_ball_pos"], dtype=np.float64)
            if "touch_event_ball_pos" in F else None
        )
        ordered_touch_ball_vel_before = (
            np.asarray(F["touch_event_ball_vel_before"], dtype=np.float64)
            if "touch_event_ball_vel_before" in F else None
        )
        ordered_touch_ball_vel_after = (
            np.asarray(F["touch_event_ball_vel_after"], dtype=np.float64)
            if "touch_event_ball_vel_after" in F else None
        )
        ordered_touch_impulse = (
            np.asarray(F["touch_event_impulse"], dtype=np.float64)
            if "touch_event_impulse" in F else None
        )
        ordered_identity = {
            "touch_event_actor": ordered_touch_actor,
            "touch_event_code": ordered_touch_code,
            "touch_event_player_id": ordered_touch_player_id,
        }
        identity_presence = {
            name: value is not None for name, value in ordered_identity.items()
        }
        if any(identity_presence.values()) and not all(identity_presence.values()):
            missing = sorted(
                name for name, present in identity_presence.items() if not present
            )
            raise ValueError(
                "ordered touch identity fields must be all present or all absent; "
                f"missing {missing}"
            )
        has_ordered_touches = all(identity_presence.values())
        if has_ordered_touches:
            expected_identity_shape = (
                T,
                self.timebase.decimation,
                TOUCH_EVENT_PHASE_COUNT,
            )
            malformed = {
                name: value.shape
                for name, value in ordered_identity.items()
                if value.shape != expected_identity_shape
            }
            if malformed:
                raise ValueError(
                    "ordered touch identity fields must all have shape "
                    f"{expected_identity_shape}; got {malformed}"
                )

        ordered_telemetry = {
            "touch_event_control_t": ordered_touch_control_t,
            "touch_event_toi": ordered_touch_toi,
            "touch_event_ball_pos": ordered_touch_ball_pos,
            "touch_event_ball_vel_before": ordered_touch_ball_vel_before,
            "touch_event_ball_vel_after": ordered_touch_ball_vel_after,
            "touch_event_impulse": ordered_touch_impulse,
        }
        telemetry_presence = {
            name: value is not None for name, value in ordered_telemetry.items()
        }
        if any(telemetry_presence.values()) and not all(telemetry_presence.values()):
            missing = sorted(
                name for name, present in telemetry_presence.items() if not present
            )
            raise ValueError(
                "ordered touch telemetry fields must be all present or all absent; "
                f"missing {missing}"
            )
        has_ordered_touch_telemetry = all(telemetry_presence.values())
        if has_ordered_touch_telemetry:
            if not has_ordered_touches:
                raise ValueError(
                    "ordered touch telemetry requires ordered touch identity fields"
                )
            scalar_shape = ordered_touch_actor.shape
            vector_shape = scalar_shape + (DIM_ALL,)
            expected_telemetry_shapes = {
                "touch_event_control_t": scalar_shape,
                "touch_event_toi": scalar_shape,
                "touch_event_ball_pos": vector_shape,
                "touch_event_ball_vel_before": vector_shape,
                "touch_event_ball_vel_after": vector_shape,
                "touch_event_impulse": scalar_shape,
            }
            malformed = {
                name: value.shape
                for name, value in ordered_telemetry.items()
                if value.shape != expected_telemetry_shapes[name]
            }
            if malformed:
                raise ValueError(
                    "ordered touch telemetry fields have invalid shapes; "
                    f"expected {expected_telemetry_shapes}, got {malformed}"
                )
        # ``substep_trajectory`` emits one-shot event buffers for each sampled
        # physical-time interval.  At render rates above control Hz, repeated
        # State.t values distinguish that representation from an ordinary
        # sequence of cumulative control-frame buffers.  Every valid interval
        # cell is new; comparing it with the preceding sample would erase a
        # legitimate repeated equal-code contact in the same control frame.
        ordered_touches_are_intervals = (
            has_ordered_touches
            and sample_fps > control_fps + 1e-9
            and T > 1
            and bool(np.any(np.asarray(F["t"])[1:] == np.asarray(F["t"][:-1])))
        )

        restart_onset = _restart_onsets(
            self, F["restart_t"], F["restart_kind"], F["restart_team"]
        )

        # 라인 통과 telemetry — 판정에 실제로 쓴 교차점/교차 속도다. 재개 스폿은 코너·골킥에서
        # 고정 좌표로 뭉개지고 골은 공을 센터서클로 보내므로 프레임 표본만으로는 복원할 수 없다.
        # 버퍼는 control frame 안에서 누적이고 terminal freeze·리샘플이 프레임을 복제할 수
        # 있으므로 (control_t, substep)으로 한 번만 낸다.
        ball_events_by_frame = {}
        if ball_event_kind is not None:
            ball_event_team_arr = np.asarray(F["ball_event_team"]).astype(int)
            ball_event_pos_arr = np.asarray(F["ball_event_pos"])
            ball_event_vel_arr = np.asarray(F["ball_event_vel"])
            ball_event_ct_arr = np.asarray(F["ball_event_control_t"]).astype(int)
            seen_ball_events = set()
            for f in range(T):
                for q in range(ball_event_kind.shape[1]):
                    kind = int(ball_event_kind[f, q])
                    if kind == BALL_EVENT_NONE:
                        continue
                    identity = (int(ball_event_ct_arr[f, q]), q)
                    if identity in seen_ball_events:
                        continue
                    seen_ball_events.add(identity)
                    team_value = int(ball_event_team_arr[f, q])
                    ball_events_by_frame.setdefault(f, []).append({
                        "substep": q,
                        "kind": kind,
                        "label": BALL_EVENT_LABEL.get(kind, "LINE CROSS"),
                        "team": team_value if team_value >= 0 else None,
                        "control_t": int(ball_event_ct_arr[f, q]),
                        "crossing_pos": [
                            round(float(v), 3) for v in ball_event_pos_arr[f, q]
                        ],
                        "crossing_vel": [
                            round(float(v), 3) for v in ball_event_vel_arr[f, q]
                        ],
                    })

        woodwork_by_frame = {}
        # ``attack_dir``는 슬롯당 스칼라 x부호다(state.py). 슬롯 0의 부호와 팀만 알면
        # 어느 골문을 어느 팀이 지키는지 프레임마다 결정된다.
        attack_x = np.asarray(F["attack_dir"])
        woodwork_kind_arr = (
            np.asarray(F["woodwork_kind"]).astype(int)
            if "woodwork_kind" in F else None
        )
        if woodwork_kind_arr is not None:
            wpos = np.asarray(F["woodwork_pos"])
            wvel = np.asarray(F["woodwork_vel_in"])
            wct = np.asarray(F["woodwork_control_t"]).astype(int)
            seen_woodwork = set()
            for f in range(T):
                for q in range(woodwork_kind_arr.shape[1]):
                    kind = int(woodwork_kind_arr[f, q])
                    if kind == WOODWORK_NONE:
                        continue
                    # ``ball_event_*``와 같은 중복 제거 — 한 제어 프레임이 여러 출력
                    # 표본에 걸치면 같은 충돌이 반복 투영된다.
                    identity = (int(wct[f, q]), q)
                    if identity in seen_woodwork:
                        continue
                    seen_woodwork.add(identity)
                    woodwork_by_frame.setdefault(f, []).append({
                        "substep": q,
                        "kind": kind,
                        "label": WOODWORK_LABEL.get(kind, "WOODWORK"),
                        "control_t": int(wct[f, q]),
                        "pos": [round(float(v), 3) for v in wpos[f, q]],
                        "vel_in": [round(float(v), 3) for v in wvel[f, q]],
                    })

        def crossing_for(f, kinds):
            """프레임 f에서 ``kinds`` 중 하나에 해당하는 라인 통과 telemetry."""
            for row in ball_events_by_frame.get(f, ()):
                if row["kind"] in kinds:
                    return row
            return None

        events = []

        def base(f, kind):
            return {
                "f": int(f),
                "t": int(F["t"][f]),
                "clock_s": round(float(F["t"][f]) / control_fps, 3),
                "clock": _mmss(float(F["t"][f]) / control_fps),
                "period": int(period[f]),
                "type": kind,
            }

        # ── 접촉 ──────────────────────────────────────────────────────────
        touches = []
        for f in range(T):
            if f == 0:
                new_touch = touch[f] > TOUCH_NONE
            else:
                new_touch = _new_touch_mask(
                    SimpleNamespace(touch=touch[f], clock=F["t"][f]),
                    SimpleNamespace(touch=touch[f - 1], clock=F["t"][f - 1]),
                )
            ordered_rows = []
            if has_ordered_touches:
                actor_now = ordered_touch_actor[f]
                code_now = ordered_touch_code[f]
                player_id_now = ordered_touch_player_id[f]
                valid_now = (
                    (actor_now >= 0) & (actor_now < N)
                    & (code_now > TOUCH_NONE) & (code_now < TOUCH_COUNT)
                )
                if ordered_touches_are_intervals:
                    newly_visible = valid_now
                elif f == 0 or F["t"][f] != F["t"][f - 1]:
                    newly_visible = valid_now
                else:
                    newly_visible = valid_now & (
                        (actor_now != ordered_touch_actor[f - 1])
                        | (code_now != ordered_touch_code[f - 1])
                        | (player_id_now != ordered_touch_player_id[f - 1])
                    )
                coordinates = np.argwhere(newly_visible)
                telemetry_coordinates = (
                    has_ordered_touch_telemetry
                    and coordinates.size
                    and all(
                        ordered_touch_control_t[f, int(q), int(p)] >= 0
                        for q, p in coordinates
                    )
                )
                if telemetry_coordinates:
                    # Ordered touch telemetry carries the originating control
                    # tick and fractional body time of impact.  It is authoritative
                    # even when resampling folds the tail of one control frame
                    # and the head of the next into one output sample.
                    coordinates = np.asarray(sorted(
                        coordinates.tolist(),
                        key=lambda row: (
                            int(ordered_touch_control_t[
                                f, int(row[0]), int(row[1])
                            ]),
                            int(row[0]) + float(ordered_touch_toi[
                                f, int(row[0]), int(row[1])
                            ]),
                            int(row[0]),
                            int(row[1]),
                        ),
                    ), dtype=np.int64)
                elif ordered_touches_are_intervals and coordinates.size:
                    # An interval may cross a control boundary: q=5 from the
                    # older frame then q=0 from the newer frame.  Plain
                    # row-major order would reverse those contacts.  Rebuild
                    # the preceding sampled physics index with the same clock
                    # quantisation as ``substep_trajectory`` and rotate local
                    # substep rows into true interval order.  Since
                    # sample_fps > control_fps, an interval is shorter than one
                    # decimation window and each local row occurs at most once.
                    previous_physics_index = (
                        -1
                        if f == 0
                        else int(np.rint(
                            float(f) * float(self.timebase.physics_fps)
                            / sample_fps
                        )) - 1
                    )
                    first_substep = (
                        previous_physics_index + 1
                    ) % self.timebase.decimation
                    coordinates = np.asarray(sorted(
                        coordinates.tolist(),
                        key=lambda row: (
                            (int(row[0]) - first_substep)
                            % self.timebase.decimation,
                            int(row[1]),
                        ),
                    ), dtype=np.int64)
                # Runtime buffers use row-major order (physics substep, then
                # force/body); interval buffers use the rotated causal order
                # above when their sample crosses a control-frame boundary.
                for substep, phase in coordinates:
                    i = int(actor_now[substep, phase])
                    ordered_rows.append((
                        i,
                        int(code_now[substep, phase]),
                        int(player_id_now[substep, phase]),
                        int(substep),
                        int(phase),
                    ))

            # Hand-authored State trajectories have no populated ordered
            # buffer.  Keep their touch-delta path, but never emit a
            # buffered actor twice from the lossy per-player accumulator.
            represented = {row[0] for row in ordered_rows}
            legacy_rows = [
                (int(i), int(touch[f][i]), int(player_id[f][i]), None, None)
                for i in np.flatnonzero(new_touch & active[f])
                if int(i) not in represented
            ]
            for i, code, event_player_id, substep, phase in ordered_rows + legacy_rows:
                rec = base(f, "touch")
                exact_telemetry = (
                    substep is not None
                    and has_ordered_touch_telemetry
                    and ordered_touch_control_t[f, substep, phase] >= 0
                )
                if exact_telemetry:
                    control_t = int(ordered_touch_control_t[f, substep, phase])
                    toi = float(ordered_touch_toi[f, substep, phase])
                    contact_pos = ordered_touch_ball_pos[f, substep, phase]
                    vel_before = ordered_touch_ball_vel_before[f, substep, phase]
                    vel_after = ordered_touch_ball_vel_after[f, substep, phase]
                    impulse = float(ordered_touch_impulse[f, substep, phase])
                    telemetry_values = np.concatenate((
                        np.asarray([toi, impulse], dtype=np.float64),
                        contact_pos,
                        vel_before,
                        vel_after,
                    ))
                    if (
                        not np.all(np.isfinite(telemetry_values))
                        or not 0.0 <= toi <= 1.0
                        or impulse < 0.0
                    ):
                        raise ValueError(
                            "ordered touch telemetry must be finite with "
                            "0 <= toi <= 1 and impulse >= 0"
                        )
                    contact_offset_s = (
                        float(substep) + toi
                    ) * float(self.e_cfg.dt_phys)
                    contact_clock_s = (
                        float(control_t) / control_fps + contact_offset_s
                    )
                    telemetry_source = "contact"
                else:
                    # Fallback for hand-authored trajectories that carry no
                    # ordered touch telemetry.  These values span the whole
                    # sampled frame and cannot distinguish multiple contacts.
                    control_t = None
                    toi = None
                    contact_offset_s = None
                    contact_clock_s = None
                    contact_pos = ball_pos[f]
                    vel_before = (
                        ball_vel[f - 1]
                        if f > 0 else np.zeros(DIM_ALL, dtype=np.float64)
                    )
                    vel_after = ball_vel[f]
                    whistle = (
                        f > 0
                        and foul_kind[f] > 0
                        and foul_kind[f - 1] == 0
                    )
                    impulse = (
                        0.0 if (f == 0 or whistle)
                        else float(np.linalg.norm(vel_after - vel_before))
                    )
                    telemetry_source = "frame_boundary"
                rec.update({
                    "code": code,
                    "label": TOUCH_LABEL.get(code, f"TOUCH{code}"),
                    "player": int(i),
                    "player_id": event_player_id,
                    "team": int(team[i]),
                    "last_touch_team": int(last_team[f]),
                    "ball_speed": round(float(np.linalg.norm(vel_after)), 3),
                    "ball_speed_before": round(
                        float(np.linalg.norm(vel_before)), 3
                    ),
                    "impulse": round(impulse, 3),
                    "ball_pos": [round(float(v), 3) for v in contact_pos],
                    "ball_vel_before": [
                        round(float(v), 3) for v in vel_before
                    ],
                    "ball_vel_after": [
                        round(float(v), 3) for v in vel_after
                    ],
                    "telemetry_source": telemetry_source,
                    "contact_control_t": control_t,
                    "contact_toi": round(toi, 6) if toi is not None else None,
                    "contact_time_in_control_frame_s": (
                        round(contact_offset_s, 6)
                        if contact_offset_s is not None else None
                    ),
                    "contact_clock_s": (
                        round(contact_clock_s, 6)
                        if contact_clock_s is not None else None
                    ),
                    "gk_hold": bool(code == TOUCH_GK_CATCH
                                    and restart_kind[f] == RK_GK_HOLD),
                })
                if substep is not None:
                    rec.update({
                        "substep": substep,
                        "contact_phase": (
                            "force" if phase == TOUCH_EVENT_FORCE else "body"
                        ),
                        "order_in_control_frame": (
                            substep * TOUCH_EVENT_PHASE_COUNT + phase
                        ),
                    })
                # 결과 레코드가 어느 접촉에서 나왔는지 명시하기 위한 안정 키.
                # 같은 선수가 같은 프레임에 같은 종류로 두 번 차면 (프레임, 선수, 코드)
                # 조합만으로는 결과를 어느 쪽에 붙일지 정할 수 없다.
                rec["touch_id"] = len(touches)
                touches.append(rec)
                events.append(rec)

        # ── 패스/슛 결과 — 다음 접촉을 봐야 정해진다 ──────────────────────
        launch_codes = {TOUCH_PASS, TOUCH_PASS_HEAD, TOUCH_SHOOT, TOUCH_SHOOT_HEAD}
        for k, rec in enumerate(touches):
            if rec["code"] not in launch_codes:
                continue
            # 바로 다음 접촉 **레코드**가 결과를 정한다. **다음 프레임**의 접촉만 보면 같은
            # 프레임에서 상대가 곧바로 막은 블록·동시 경합이 누락돼 그런 슛이
            # 'unresolved'로 남는다. 목록은 (프레임, 물리 서브스텝,
            # force/body 단계) 순이므로 같은 프레임의 실제 후속 접촉이 바로 다음 항목이다.
            # 굴절은 결과가 아니다. ``TOUCH_DEFLECT``는 환경 정의상 '탈취 아님, 의도적
            # 플레이 아님'이며 오프사이드 플래그조차 리셋하지 않는다. 그런 접촉을 리셉션으로
            # 읽으면 상대 몸에 맞고 흐른 패스가 'completed'가 되고 아군 몸에 맞고 흐른 패스가
            # 'intercepted'가 된다. 첫 **통제된** 접촉까지 이어 보고, 지나친 굴절은 근거로 남긴다.
            # GK PARRY는 굴절이 아니라 세이브라 그대로 결과를 끝낸다.
            deflections = []
            j = k + 1
            while j < len(touches) and touches[j]["code"] == TOUCH_DEFLECT:
                deflections.append(touches[j])
                j += 1
            nxt = touches[j] if j < len(touches) else None
            is_shot = rec["code"] in (TOUCH_SHOOT, TOUCH_SHOOT_HEAD)
            f0 = rec["f"]
            detail = {}
            if deflections:
                # 굴절한 쪽을 남겨야 소비자가 자책·굴절골을 판별할 수 있다.
                detail["deflections"] = [
                    {
                        "touch_id": d["touch_id"], "player": d["player"],
                        "team": d["team"], "f": d["f"],
                    }
                    for d in deflections
                ]
            # 골은 다음 접촉보다 먼저 확인한다 — 골 직후엔 킥오프 접촉이 오므로 순서를 바꾸면
            # 득점이 '상대에게 넘어감'으로 뒤집힌다.
            goal_after = None
            goal_frame = None
            # A fast shot can be launched and cross the line inside one render
            # sample.  Include the launch frame itself when a previous score is
            # available; starting at f0+1 silently missed that resolvable case.
            goal_search_start = f0 if f0 > 0 else f0 + 1
            # 골 귀속의 올바른 경계는 시간이 아니라 **다음 접촉**이다. 아래에서 이미
            # ``nxt["f"] < goal_frame``이면 귀속을 취소하므로, 고정 시간 창은 그 위에
            # 덧붙은 중복 상한일 뿐이고 느린 굴러들어간 골 같은 참을 놓치기만 한다.
            # 같은 표본 프레임에서 골과 다음 접촉(득점 뒤 킥오프)이 겹칠 수 있으므로
            # 다음 접촉 프레임 자체는 포함한다.
            goal_search_end = T if nxt is None else min(T, nxt["f"] + 1)
            for f in range(goal_search_start, goal_search_end):
                if score[f][0] != score[f - 1][0] or score[f][1] != score[f - 1][1]:
                    goal_after = 0 if score[f][0] > score[f - 1][0] else 1
                    goal_frame = f
                    break
            # 골은 **마지막 접촉**에만 귀속한다. 이 가드가 없으면 득점 직전 몇 초 안의 슛이
            # 전부 득점자로 기록돼 골 1건에 shot_outcome "goal"이 여러 개 생긴다.
            if goal_frame is not None and nxt is not None and nxt["f"] < goal_frame:
                goal_after = None
            # 다음 접촉 전에 공이 나갔는가(재개 온셋이 먼저 오는가)
            restart_before_touch = None
            restart_before_touch_frame = None
            limit = nxt["f"] if nxt else T
            # As with a goal, a near-line launch and the resulting out can share
            # one sampled frame.  An ordinary restart launch does not create an
            # onset here: consuming its timer leaves restart_t=0.  Therefore an
            # onset at f0 is a genuinely new stoppage and belongs to this flight.
            for f in range(f0, min(T, limit)):
                if restart_onset[f]:
                    restart_before_touch = int(restart_kind[f])
                    restart_before_touch_frame = f
                    break

            if goal_after is not None and goal_after == rec["team"]:
                outcome = "goal"
            elif goal_after is not None:
                # 자책골. 이 분기가 없으면 뒤의 재개 분기로 떨어지고, 득점 뒤 킥오프 온셋이
                # ``_RESTART_FLIGHT_OUTCOME``에 없어 기본값 out_of_play가 된다 —
                # 자기 골대로 넣은 클리어가 '라인 밖으로 나갔다'로 기록되는 셈이다.
                outcome = "own_goal"
                detail["conceded_to"] = int(goal_after)
            elif restart_before_touch is not None:
                # 재개 종류를 뭉뚱그리면 안 된다. 공이 실제로 경계를 넘은 것과 심판이
                # 경기를 멈춘 것은 다른 사건이고, 후자를 out_of_play로 적으면 유효슈팅·
                # 아웃 통계가 함께 오염된다(오프사이드 접촉이 '골라인 넘은 슛'이 된다).
                outcome = _RESTART_FLIGHT_OUTCOME.get(
                    restart_before_touch, "out_of_play"
                )
                detail["restart_kind"] = restart_before_touch
                detail["restart_frame"] = int(restart_before_touch_frame)
            elif nxt is None:
                outcome = "unresolved"
            elif nxt["player"] == rec["player"]:
                # 같은 선수가 다시 댔다 — 패스/슛이 아니라 되받은 것이다(리바운드·연속 터치).
                outcome = "retained"
            elif nxt["code"] in (TOUCH_GK_CATCH, TOUCH_PARRY):
                outcome = "saved"
                detail["by"] = nxt["player"]
            elif nxt["team"] == rec["team"]:
                outcome = "rebound_own" if is_shot else "completed"
                detail["receiver"] = nxt["player"]
            else:
                outcome = "blocked" if is_shot else "intercepted"
                detail["by"] = nxt["player"]
            # 슛에서 상대의 굴절은 축구 용어 그대로 **블록**이다 — 막은 선수가 공을
            # 소유하지 못해도 슛은 거기서 끝난다. 패스는 다르다: 상대 몸에 맞고 흘러도
            # 그 선수가 가져간 것이 아니므로 결과는 첫 통제 접촉이 정한다.
            # 골은 위에서 이미 확정됐으므로 여기서 덮지 않는다(굴절 골은 골이다).
            if is_shot and outcome not in ("goal", "own_goal"):
                blocker = next(
                    (row for row in deflections if row["team"] != rec["team"]),
                    None,
                )
                if blocker is not None:
                    outcome = "blocked"
                    detail["by"] = blocker["player"]
            # A restart relocates the ball to a throw/corner/goal-kick spot and
            # a goal recentres it.  Neither administrative position is the
            # physical flight endpoint.  Measuring distance to the *next touch*
            # was worse: it included the entire dead-ball wait and often the
            # restart kick.  Preserve an exact terminal frame/time and leave the
            # unresolved geometric distance null; ordinary touch-ended flights
            # keep their receiver/contact displacement.
            # 자책골도 골과 **같은 프레임에서 끝난다**. 종전에는 ``outcome == "goal"``만
            # 종점으로 인정해 자책골이 ``_RESTART_TERMINAL_OUTCOMES``에도 없는 탓에
            # ``terminal_frame``이 None으로 떨어졌고, 결국 종점이 **다음 접촉**(득점 후
            # 킥오프)이 되어 비행시간과 거리가 데드볼 전체만큼 부풀었다. 바로 위 주석이
            # 말하는 "골이 공을 중앙으로 되돌린다"는 사정은 자책골에도 똑같이 적용된다.
            terminal_frame = (
                goal_frame if outcome in ("goal", "own_goal")
                else restart_before_touch_frame
                if outcome in _RESTART_TERMINAL_OUTCOMES
                else None
            )
            endpoint_frame = terminal_frame if terminal_frame is not None else (
                nxt["f"] if nxt is not None else None
            )
            endpoint_pos = (
                np.asarray(nxt["ball_pos"])[:2]
                if terminal_frame is None and nxt is not None
                else None
            )
            # 골대를 맞은 비행은 축구 통계에서 'hit the woodwork'으로 따로 센다 —
            # 빗나감(out_of_play)도 유효슈팅(saved)도 아니다. 골로 끝난 비행은 결과를
            # 덮지 않고 detail로만 남긴다(굴절 골이 골인 것과 같은 규약).
            woodwork_hit = None
            scan_end = endpoint_frame if endpoint_frame is not None else T - 1
            for wf in range(f0, min(T, int(scan_end) + 1)):
                rows = woodwork_by_frame.get(wf)
                if rows:
                    woodwork_hit = (wf, rows[0])
                    break
            if woodwork_hit is not None:
                wf, wrow = woodwork_hit
                detail["woodwork"] = wrow["label"].lower()
                detail["woodwork_frame"] = int(wf)
                detail["woodwork_pos"] = wrow["pos"]
                if outcome not in ("goal", "own_goal"):
                    outcome = "woodwork"
            if outcome not in FLIGHT_OUTCOMES:
                raise RuntimeError(
                    f"flight outcome {outcome!r} is not registered in "
                    "FLIGHT_OUTCOMES; add it there and give it a label in "
                    "both the light and rich feeds"
                )
            # 골이 된 킥은 슛으로 본다. ``is_shot``은 순수 기하 판정이라
            # (골대 30 m 안 + 조준 원뿔) 30 m 밖의 중거리포나 조준 밖에서 말려 들어간
            # 크로스가 PASS로 분류된다. 의도는 가를 수 없고 결과는 분명하므로,
            # 득점한 킥은 슛 통계에 넣는다 — 패스 분모에 남으면 가장 성공적인 킥이
            # 성공률을 깎는다. 자책골은 '넣은' 것이 아니므로 패스 실패로 남긴다.
            scored_kick = outcome == "goal"
            if scored_kick and not is_shot:
                detail["reclassified_from"] = "pass"
            out = base(
                f0,
                "shot_outcome" if (is_shot or scored_kick) else "pass_outcome",
            )
            out.update({
                "player": rec["player"], "team": rec["team"],
                "outcome": outcome,
                "flight_s": (
                    round((endpoint_frame - f0) / sample_fps, 3)
                    if endpoint_frame is not None else None
                ),
                "distance_m": (round(float(np.linalg.norm(
                    endpoint_pos - np.asarray(rec["ball_pos"])[:2])), 2)
                    if endpoint_pos is not None else None),
                "outcome_frame": (
                    int(endpoint_frame) if endpoint_frame is not None else None
                ),
                "launch_speed": rec["ball_speed"],
                "source_touch_id": rec["touch_id"],
            })
            out.update(detail)
            events.append(out)

        # 경기 생명주기 — 첫 구간의 시작을 명시한다. 구간 경계에서만 파생하면 오프닝
        # ``period_start``가 없어 후반만 잘라 저장한 리플레이와 구분되지 않는다.
        if T > 0:
            opening = base(0, "period_start")
            opening.update({"index": int(period[0])})
            events.append(opening)

        # ── 재개 / 골 / 파울 / 카드 / 퇴장 ────────────────────────────────
        for f in range(T):
            if restart_onset[f]:
                rk = int(restart_kind[f])
                rec = base(f, "restart")
                rec.update({
                    "kind": rk,
                    "label": RESTART_LABEL.get(rk, f"RK{rk}"),
                    "team": int(restart_team[f]),
                    "indirect": bool(restart_ind[f]),
                    "taker": int(pending[f]) if pending[f] >= 0 else None,
                    # 원인 팀은 마지막 접촉 팀과 다를 수 있다 — 파울·백패스·오프사이드는
                    # 반칙한 쪽이 내준 것이고, 그 정보는 아래 ``cause``에 실린다.
                    # 여기서는 관측 사실(마지막 접촉 팀)만 그대로 남기고 이름을 그렇게 붙인다.
                    "last_touch_team": int(last_team[f]) if last_team[f] >= 0 else None,
                    "ball_pos": [round(float(v), 3) for v in ball_pos[f]],
                })
                # 아웃으로 생긴 재개는 '공이 어디서 나갔는가'가 스폿과 다르다. 코너·골킥은
                # 스폿이 고정 좌표라 교차점이 스폿에서 복원되지 않는다.
                crossing = crossing_for(f, tuple(
                    kind for kind, mapped in BALL_EVENT_RESTART_KIND.items()
                    if mapped == rk
                ))
                if crossing is not None:
                    rec["crossing"] = {
                        "pos": crossing["crossing_pos"],
                        "vel": crossing["crossing_vel"],
                        "substep": crossing["substep"],
                        "label": crossing["label"],
                    }
                # 프리킥·페널티는 원인 파울과 붙어야 뜻이 산다. 두 레코드를 따로 두고
                # 소비자에게 프레임으로 조인하라고 하면 매번 같은 코드를 다시 쓰게 된다.
                if rk == RK_OFFSIDE:
                    # 오프사이드 '위치'와 오프사이드 '반칙'은 다른 사건이다. 플래그만으로는
                    # 누가 관여해 콜이 됐는지 알 수 없으므로, 콜을 만든 접촉을 되짚어 남긴다.
                    offender = next(
                        (e for e in reversed(events)
                         if e["type"] == "touch" and e["f"] <= f),
                        None,
                    )
                    if offender is not None:
                        rec["cause"] = {
                            "reason": "offside",
                            "actor": offender["player"],
                            "actor_team": offender["team"],
                            "touch_event_id": offender.get("event_id"),
                        }
                if rk in (RK_FREEKICK, RK_PENALTY) and foul_kind[f] > 0:
                    a, v = int(foul_actor[f]), int(foul_victim[f])
                    rec["cause"] = {
                        "foul_kind": int(foul_kind[f]),
                        "foul_label": FOUL_LABEL.get(int(foul_kind[f]), "FOUL"),
                        "actor": a if a >= 0 else None,
                        "actor_team": int(team[a]) if a >= 0 else None,
                        "victim": v if v >= 0 else None,
                        "victim_team": int(team[v]) if v >= 0 else None,
                    }
                events.append(rec)
            for row in woodwork_by_frame.get(f, ()):
                rec = base(f, "woodwork")
                rec.update({
                    "kind": row["kind"], "label": row["label"],
                    "substep": row["substep"],
                    "pos": row["pos"], "vel_in": row["vel_in"],
                    # 프레임은 어느 팀 것도 아니다 — 기록할 팀은 그 골문을 지키는 쪽이다.
                    # ``attack_dir``는 슬롯별 공격 x부호라 하프타임 교대도 자동으로 따른다.
                    "defending_team": int(
                        team[0] if (attack_x[f, 0] > 0) != (row["pos"][0] > 0)
                        else 1 - team[0]
                    ),
                })
                events.append(rec)
            if f == 0:
                continue
            if score[f][0] != score[f - 1][0] or score[f][1] != score[f - 1][1]:
                scorer = 0 if score[f][0] > score[f - 1][0] else 1
                rec = base(f, "goal")
                rec.update({"team": scorer, "score": [int(score[f][0]), int(score[f][1])]})
                # 골문 어디로 어떤 속도로 들어갔는가 — 공이 센터서클로 옮겨지기 전의 값.
                crossing = crossing_for(f, (BALL_EVENT_GOAL,))
                if crossing is not None:
                    rec["crossing"] = {
                        "pos": crossing["crossing_pos"],
                        "vel": crossing["crossing_vel"],
                        "substep": crossing["substep"],
                    }
                events.append(rec)
            if foul_kind[f] > 0 and foul_kind[f - 1] == 0:
                a, v = int(foul_actor[f]), int(foul_victim[f])
                rec = base(f, "foul")
                rec.update({
                    "kind": int(foul_kind[f]),
                    "label": FOUL_LABEL.get(int(foul_kind[f]), "FOUL"),
                    "actor": a if a >= 0 else None,
                    "actor_team": int(team[a]) if a >= 0 else None,
                    "team": int(team[a]) if a >= 0 else None,
                    "victim": v if v >= 0 else None,
                    "victim_team": int(team[v]) if v >= 0 else None,
                })
                events.append(rec)
            for i in range(N):
                if yellow[f][i] > yellow[f - 1][i]:
                    rec = base(f, "card")
                    rec.update({"color": "yellow", "player": int(i), "team": int(team[i]),
                                "total": int(yellow[f][i])})
                    events.append(rec)
                if active[f - 1][i] and not active[f][i]:
                    rec = base(f, "sent_off")
                    rec.update({"player": int(i), "team": int(team[i])})
                    events.append(rec)
                if generation[f][i] != generation[f - 1][i]:
                    rec = base(f, "substitution")
                    rec.update({"slot": int(i), "team": int(team[i]),
                                "player_out": int(player_id[f - 1][i]),
                                "player_in": int(player_id[f][i])})
                    # [IFAB Law 3] 교체는 데드볼에서만 일어난다. 스케줄된 시각에 공이
                    # 살아 있으면 다음 데드볼로 미뤄지므로, 소비자가 '왜 예정보다 늦게
                    # 들어왔는가'를 알 수 있어야 한다. 잔여 교체·윈도도 같은 이유로 남긴다 —
                    # 이 숫자 없이는 교체가 왜 더 일어나지 않았는지 설명되지 않는다.
                    if subs_remaining is not None:
                        rec["subs_remaining"] = int(subs_remaining[f][int(team[i])])
                        rec["sub_windows_used"] = int(
                            sub_windows_used[f][int(team[i])]
                        )
                    events.append(rec)
            if layout_index is not None:
                for tm in (TEAM_0, TEAM_1):
                    if layout_index[f][tm] == layout_index[f - 1][tm]:
                        continue
                    # 목표 layout은 명령 경계에서 바뀌고 선수는 이후 물리적으로 이동한다.
                    # 명령 프레임을 남겨야 소비자가 그 여파를 자발적 움직임과 구분할 수 있다.
                    rec = base(f, "formation")
                    rec.update({
                        "team": int(tm),
                        "from": layout_label(layout_index[f - 1][tm]),
                        "to": layout_label(layout_index[f][tm]),
                        "from_index": int(layout_index[f - 1][tm]),
                        "to_index": int(layout_index[f][tm]),
                    })
                    events.append(rec)
            if poss[f] != poss[f - 1]:
                # 루즈볼(-1)도 남긴다. 팀→팀 전환만 세면 "빼앗겼다"와 "튕겨나가 아무도
                # 소유하지 않는다"가 구분되지 않아, 소유 스펠 길이가 실제보다 길게 잡힌다.
                rec = base(f, "possession")
                rec.update({
                    "team": int(poss[f]) if poss[f] >= 0 else None,
                    "from_team": int(poss[f - 1]) if poss[f - 1] >= 0 else None,
                    "loose": bool(poss[f] < 0),
                })
                events.append(rec)
            new_off = offside[f].astype(bool) & ~offside[f - 1].astype(bool)
            for i in np.flatnonzero(new_off & active[f]):
                rec = base(f, "offside_flag")
                rec.update({"player": int(i), "team": int(team[i])})
                events.append(rec)
            if period[f] != period[f - 1]:
                # 경계는 이전 구간의 끝이자 새 구간의 시작이다. 두 사건을 함께 남겨야
                # 소비자가 구간을 잘라낼 때 시작/끝을 추론하지 않아도 된다.
                prev = base(f, "period_end")
                prev.update({"index": int(period[f - 1])})
                events.append(prev)
                rec = base(f, "period")
                rec.update({"index": int(period[f])})
                events.append(rec)
                start = base(f, "period_start")
                start.update({"index": int(period[f])})
                events.append(start)
            hold_now = restart_kind[f] == RK_GK_HOLD
            hold_prev = restart_kind[f - 1] == RK_GK_HOLD
            if hold_now != hold_prev:
                rec = base(f, "gk_hold_start" if hold_now else "gk_hold_end")
                rec.update({"team": int(restart_team[f])})
                events.append(rec)

        unknown_types = sorted({
            rec["type"] for rec in events
            if rec["type"] not in _EVENT_CAUSAL_STAGE
        })
        if unknown_types:
            raise RuntimeError(
                "event types need an explicit causal stage: "
                + ", ".join(unknown_types)
            )

        # Collection happens in type-specific passes because launch outcomes
        # need future frames.  Reconstruct the observable causal order instead
        # of sorting by the spelling of ``type``.  Python's stable sort keeps
        # exact physical touch order and deterministic order inside a stage.
        # 리플레이 끝 — 정상 종료와 잘린 저장을 구분할 수 있어야 한다. 구분이 없으면
        # 마지막 미해결 패스나 진행 중이던 재개를 '경기에서 그렇게 끝났다'로 읽게 된다.
        # 종료 판정은 ``SoccerEnv._is_terminal``과 같은 재료(경기 시간·최소 인원)를 쓴다.
        if T > 0:
            last = T - 1
            per_team = [
                int(np.count_nonzero(active[last] & (team == TEAM_0))),
                int(np.count_nonzero(active[last] & (team == TEAM_1))),
            ]
            # minimum_team_players는 팀별 배열이다 — 로스터 크기가 팀마다 다를 수 있다.
            # env._is_terminal과 동일하게 팀별로 비교해야 한쪽만 인원 미달인 경우를 놓치지 않는다.
            minimum = [int(self.minimum_team_players[TEAM_0]),
                       int(self.minimum_team_players[TEAM_1])]
            time_limit = int(F["t"][last]) >= int(self.game_duration)
            abandoned = any(n < m for n, m in zip(per_team, minimum))
            terminal = bool(time_limit or abandoned)
            if terminal:
                # 구간이 실제로 끝났을 때만 닫는다. 잘린 리플레이에 period_end를 내면
                # '여기서 전/후반이 끝났다'로 읽힌다 — 끝난 것은 기록이지 구간이 아니다.
                closing = base(last, "period_end")
                closing.update({"index": int(period[last]), "terminal": True})
                events.append(closing)
            end = base(last, "match_end" if terminal else "censored")
            end.update({
                "terminal": terminal,
                "reason": (
                    "time" if time_limit
                    else "min_players" if abandoned
                    else "truncated_replay"
                ),
                "active_per_team": per_team,
                "minimum_per_team": minimum,
            })
            events.append(end)

        indexed_events = list(enumerate(events))
        indexed_events.sort(key=lambda item: (
            item[1]["f"],
            _EVENT_CAUSAL_STAGE[item[1]["type"]],
            item[0],
        ))
        events = [rec for _, rec in indexed_events]

        frame = None
        sequence_in_frame = 0
        for rec in events:
            if rec["f"] != frame:
                frame = rec["f"]
                sequence_in_frame = 0
            rec["sequence_in_frame"] = sequence_in_frame
            sequence_in_frame += 1
        return events

    def _match_summary(self, F, events, sample_fps=None):
        """이벤트에서 바로 뽑히는 경기 집계.

        "누가 이겼나 · 슛을 몇 개 쐈나 · 패스 성공률이 얼마인가"는 리플레이를 열 때마다
        묻는 것들이라, 소비자가 같은 집계 코드를 매번 다시 쓰게 두지 않는다. 빈도는
        per-90이 아니라 **인플레이 1분당**으로 낸다 — 데드볼 규약이 다른 두 실행을 per-90으로
        비교하면 인플레이 비율 차이가 그대로 빈도 차이로 보인다.
        """
        score = np.asarray(F["score"]).astype(int)
        alive = np.asarray(F["ball_state"]) == BALL_ALIVE
        poss = np.asarray(F["poss_team"]).astype(int)
        team = np.asarray(F["team_id"])[0].astype(int)
        frames = int(alive.size)
        sample_fps = (
            float(self.control_fps) if sample_fps is None else float(sample_fps)
        )
        if not np.isfinite(sample_fps) or sample_fps <= 0.0:
            raise ValueError("sample_fps must be a positive finite number")
        inplay_min = float(alive.sum()) / sample_fps / 60.0

        def team_counts(kind, predicate=None):
            out = [0, 0]
            for rec in events:
                if rec["type"] != kind:
                    continue
                if predicate is not None and not predicate(rec):
                    continue
                t = rec.get("team")
                if t in (0, 1):
                    out[t] += 1
            return out

        passes = [r for r in events if r["type"] == "pass_outcome"]
        shots = [r for r in events if r["type"] == "shot_outcome"]
        completed = [0, 0]
        attempted = [0, 0]
        unresolved = [0, 0]
        carries = [0, 0]
        for rec in passes:
            t = rec.get("team")
            if t not in (0, 1):
                continue
            # 결과가 확정되지 않은 패스(리플레이 절단 등)를 시도에 넣으면 성공률이
            # 구조적으로 낮게 나온다. 분모는 해결된 패스만 세고 미해결은 따로 보고한다.
            if rec["outcome"] == "unresolved":
                unresolved[t] += 1
                continue
            # 공급자와 수신자가 같으면 정의상 **드리블(캐리)**이지 패스가 아니다.
            # 접촉 코드는 킥 속도로 정해지므로 몸싸움 중 강하게 댄 터치가 PASS로 찍히는데,
            # 그것이 자기에게 돌아온 경우까지 패스 시도로 세면 성공할 수 없는 항목이
            # 분모만 키워 성공률을 구조적으로 낮춘다.
            if rec["outcome"] == "retained":
                carries[t] += 1
                continue
            attempted[t] += 1
            completed[t] += int(rec["outcome"] == "completed")
        restarts = {}
        for rec in events:
            if rec["type"] == "restart":
                restarts[rec["label"]] = restarts.get(rec["label"], 0) + 1
        return {
            "score": [int(score[-1][0]), int(score[-1][1])],
            "frames": frames,
            "inplay_fraction": round(float(alive.mean()), 4),
            "inplay_minutes": round(inplay_min, 2),
            # 점유율의 분모는 인플레이 프레임이다. 데드볼 대기(재개 준비 3초 등)를
            # 넣으면 그 시간이 통째로 직전 소유팀에 계상돼 점유율이 왜곡된다.
            "possession_share": [
                round(float((poss[alive] == 0).mean()), 4) if alive.any() else None,
                round(float((poss[alive] == 1).mean()), 4) if alive.any() else None,
                round(float((poss[alive] < 0).mean()), 4) if alive.any() else None,
            ],
            "touches": team_counts("touch"),
            "touches_per_inplay_min": [
                round(c / inplay_min, 2) if inplay_min > 0 else None
                for c in team_counts("touch")
            ],
            "passes_attempted": attempted,
            "passes_completed": completed,
            "passes_unresolved": unresolved,
            # 자기에게 돌아온 킥 — 패스 분모에서 빠지고 여기 따로 잡힌다.
            "carries": carries,
            "pass_completion": [
                round(completed[t] / attempted[t], 4) if attempted[t] else None
                for t in (0, 1)
            ],
            "shots": [len([r for r in shots if r.get("team") == t]) for t in (0, 1)],
            "shots_on_target": [
                len([r for r in shots
                     if r.get("team") == t and r["outcome"] in ("goal", "saved")])
                for t in (0, 1)
            ],
            "fouls": team_counts("foul", lambda r: r.get("actor_team") is not None),
            "cards": team_counts("card"),
            "sent_off": team_counts("sent_off"),
            "substitutions": team_counts("substitution"),
            "restarts": restarts,
            "team_sizes": [int(np.count_nonzero(team == 0)), int(np.count_nonzero(team == 1))],
        }

    def _write_events_json(self, F, path, *, render_fps=DEFAULT_RENDER_FPS):
        """구조화 이벤트 → JSON 하나. 반환: 이벤트 수.

        ``sample_fps``를 넘기지 않으면 검출기가 기본 15 Hz로 시간을 환산해, 같은 궤적의
        같은 사건이 canonical 스트림과 다른 초를 갖게 된다(25 Hz 궤적의 1초 패스가 여기서는
        1.667초). 두 출력의 시간축은 반드시 같아야 한다.

        이 writer는 frozen events/3 소비자를 위한 legacy adapter다. 필드 형태는 유지하되,
        현행 accepted formation target은 모두 명령 경계에서 즉시 적용되므로 ``immediate``는
        항상 true다. events/6와 화면 feed는 목표 변경만 기록하며 이 필드를 생성하지 않는다.
        """
        canonical_events = self._extract_events(F, sample_fps=float(render_fps))
        events = []
        for source in canonical_events:
            rec = dict(source)
            if rec.get("type") == "formation":
                rec["immediate"] = True
            events.append(rec)
        counts = {}
        for rec in events:
            counts[rec["type"]] = counts.get(rec["type"], 0) + 1
        payload = {
            # 이벤트는 바로 위에서 ``render_fps``로 뽑았다. 요약만 기본값(control_fps)을
            # 쓰면 같은 파일 안에서 두 시간축이 갈린다 — 25 fps 렌더에서 인플레이 분이
            # 25/15배로 부풀고 분당 통계가 그만큼 깎인다. 이 함수의 docstring이 금지하는
            # 바로 그 상황이다.
            "summary": self._match_summary(F, events,
                                           sample_fps=float(render_fps)),
            "meta": {
                "schema": LEGACY_EVENTS_SCHEMA,
                "frames": int(len(F["t"])),
                "control_fps": float(self.control_fps),
                "n_players": int(self.N),
                "team_sizes": [int(self.n_agents), int(self.n_opponents)],
                "touch_labels": {int(k): v for k, v in TOUCH_LABEL.items()},
                "restart_labels": {int(k): v for k, v in RESTART_LABEL.items()},
                "foul_labels": {int(k): v for k, v in FOUL_LABEL.items()},
                "counts": counts,
            },
            "events": events,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        return len(events)

    @staticmethod
    def _jsonl_write(fh, row):
        """Write one strict, human-scannable JSON object per line."""

        fh.write(json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n")

    def _write_events_jsonl(self, F, path, *, events=None, render_fps=DEFAULT_RENDER_FPS,
                            video_t=None):
        """Canonical event stream: one event object per line, with both clocks.

        ``video_time_s``는 ``_video_time_lookup``으로 구한다 — F의 행 번호가 아니라
        경기 시계로 되찾은 **영상 표본 번호**다. ``video_t``를 주지 않으면 F가 곧
        영상 격자라고 보고 행 번호를 쓴다. events/6 formation 행은 즉시 바뀐 목표를
        뜻하며, 선수 이동 phase나 legacy ``immediate`` 필드를 싣지 않는다.
        """

        render_fps = float(render_fps)
        if events is None:
            events = self._extract_events(F, sample_fps=render_fps)
        grid = np.asarray(F["t"]).reshape(-1) if video_t is None else video_t
        to_video_time = _video_time_lookup(grid, render_fps)
        with open(path, "w", encoding="utf-8") as fh:
            for event_id, source in enumerate(events):
                row = dict(source)
                video_time = to_video_time(row["t"])
                row.update({
                    "record_type": "event",
                    "event_id": int(event_id),
                    "video_time_s": round(video_time, 3),
                    "sample_end_time_s": round(
                        video_time + 1.0 / render_fps, 3
                    ),
                })
                self._jsonl_write(fh, row)
        return len(events)

    def _write_tracking_jsonl(
        self,
        F,
        path,
        stride=1,
        *,
        render_fps=DEFAULT_RENDER_FPS,
        video_t=None,
    ):
        """Canonical full-state stream: exactly one sampled frame per JSONL row.

        Static schema/configuration belongs in ``metadata.jsonl`` rather than a
        special first row.  This keeps every row in this file homogeneous and
        makes line-oriented consumers (``tail``, DuckDB, jq) straightforward.
        """

        render_fps = float(render_fps)
        grid = np.asarray(F["t"]).reshape(-1) if video_t is None else video_t
        to_video_time = _video_time_lookup(grid, render_fps)
        rows = 0
        with open(path, "w", encoding="utf-8") as fh:
            for f in range(0, len(F["t"]), int(stride)):
                row = self._state_row(F, f)
                video_time = to_video_time(F["t"][f])
                row.update({
                    "record_type": "tracking",
                    "frame": int(f),
                    "video_time_s": round(video_time, 3),
                    "sample_end_time_s": round(
                        video_time + 1.0 / render_fps, 3
                    ),
                    "sim_time_s": round(
                        float(F["t"][f]) / float(self.control_fps), 3
                    ),
                })
                self._jsonl_write(fh, row)
                rows += 1
        return rows

    @staticmethod
    def read_replay_metadata(path, *, require_matching_runtime=True):
        """리플레이 매니페스트를 읽는다. 기본값은 **백엔드 불일치에서 실패**한다.

        같은 시드·같은 코드라도 백엔드가 다르면 궤적이 갈린다(실측: step 0부터
        5.96e-08 m, 300 s 뒤 접촉 392 대 363). 기록만 남기고 강제하지 않으면 GPU에서
        만든 데이터와 CPU에서 만든 데이터가 조용히 섞여, 나중에 원인 모를 불일치로
        되돌아온다. 그래서 읽는 쪽에서 막는다.

        의도적으로 다른 백엔드의 리플레이를 볼 때만 ``require_matching_runtime=False``를
        준다 — 그때는 궤적을 재현할 수 없다는 것을 호출자가 아는 상태다.

        반환: ``{record_type: row}`` dict.
        """

        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        topics = {row["record_type"]: row for row in rows}
        runtime = topics.get("runtime")
        if not require_matching_runtime:
            return topics
        if runtime is None:
            raise ValueError(
                f"{path}: runtime 레코드가 없어 어느 백엔드에서 나온 리플레이인지 "
                "알 수 없다. 이 매니페스트로는 궤적 재현을 보장할 수 없다 "
                "(확인했다면 require_matching_runtime=False)"
            )
        recorded = str(runtime.get("backend"))
        current = _jax_backend()
        if recorded != current:
            raise ValueError(
                f"{path}: 이 리플레이는 {recorded!r}에서 생성됐는데 지금은 "
                f"{current!r}에서 읽고 있다. 백엔드가 다르면 같은 시드라도 궤적이 "
                "갈린다 — 데이터셋을 만든 백엔드에서 실행하거나, 재현이 필요 없다면 "
                "require_matching_runtime=False를 명시하라"
            )
        # x64는 백엔드보다 궤적을 더 크게 가른다 — 같은 CPU라도 float64로 돌린
        # 리플레이는 float32 프로세스에서 재현되지 않는다. backend만 비교하면
        # 그 조합이 조용히 통과한다.
        if "x64_enabled" not in runtime:
            raise ValueError(
                f"{path}: runtime 레코드에 x64_enabled가 없어 어느 정밀도에서 나온 "
                "리플레이인지 알 수 없다 "
                "(확인했다면 require_matching_runtime=False)"
            )
        recorded_x64 = bool(runtime["x64_enabled"])
        current_x64 = bool(jax.config.jax_enable_x64)
        if recorded_x64 != current_x64:
            raise ValueError(
                f"{path}: 이 리플레이는 x64_enabled={recorded_x64}에서 생성됐는데 "
                f"지금은 {current_x64}로 읽고 있다. 정밀도가 다르면 같은 시드라도 "
                "궤적이 갈린다 — 만든 정밀도로 실행하거나, 재현이 필요 없다면 "
                "require_matching_runtime=False를 명시하라"
            )
        return topics

    def _write_replay_metadata_jsonl(
        self,
        F,
        path,
        *,
        events,
        render_fps,
        mode,
        title,
        state_stride,
        rich_options,
        user_metadata=None,
        video_filename="match.mp4",
        video_frames=None,
    ):
        """Write the canonical replay manifest as one independently typed row/topic."""

        T = int(len(F["t"]))
        every = int(rich_options.get("every", 1)) if mode == "rich" else 1
        # ``F``는 이벤트·트래킹 스택이고 영상은 별도 격자일 수 있다. 프레임 수를 F에서
        # 세면 매니페스트가 **다른 파일의 길이**를 영상 길이로 적는다(실측: 2250프레임
        # 90초 영상을 1351프레임 54.04초로 기록). 실제 인코딩 수를 받아 쓴다.
        source_frames = T if video_frames is None else int(video_frames)
        encoded_frames = (source_frames + every - 1) // every
        encoded_fps = float(render_fps) / every
        counts = {}
        for rec in events:
            counts[rec["type"]] = counts.get(rec["type"], 0) + 1

        dynamics = (
            self.dynamics_metadata()
            if callable(getattr(self, "dynamics_metadata", None))
            else None
        )
        rows = [
            {
                "record_type": "replay",
                "schema": "uos-footballmarl.replay/3",
                "title": title,
                "render_mode": mode,
                "camera": (
                    "dynamic"
                    if mode == "rich" and bool(rich_options.get("dynamic_cam", False))
                    else "fixed"
                ),
                "files": {
                    "video": str(video_filename),
                    "events": "events.jsonl",
                    "tracking": "tracking.jsonl",
                    "metadata": "metadata.jsonl",
                },
                "source_frames": T,
                "encoded_frames": encoded_frames,
                "tracking_stride": int(state_stride),
            },
            {
                # 같은 시드·같은 코드라도 백엔드가 다르면 **다른 경기**가 된다.
                # 실측: seed 4242 · 300 s를 CPU와 GPU에서 돌리면 2 s 지점부터
                # 위치가 갈리기 시작해(차이는 덤프 반올림 한계인 1 mm) 5분 뒤에는
                # 접촉이 392 대 363, 이벤트가 675 대 654로 벌어졌다. float 잡음이
                # 접촉 성사 여부 같은 이산 판정을 뒤집기 때문이다.
                # 데이터셋을 만든 백엔드를 남겨 두지 않으면 사후에 구분할 수 없다.
                "record_type": "runtime",
                "backend": _jax_backend(),
                "device_kind": _jax_device_kind(),
                "device_count": _jax_device_count(),
                "x64_enabled": bool(jax.config.jax_enable_x64),
                "note": (
                    "trajectories are bit-reproducible only within one backend; "
                    "compare or mix datasets only across matching runtime rows"
                ),
            },
            {
                "record_type": "timebase",
                "physics_fps": float(self.timebase.physics_fps),
                "control_fps": float(self.control_fps),
                "render_sample_fps": float(render_fps),
                "encoded_fps": encoded_fps,
                "render_every": every,
                "video_duration_s": round(encoded_frames / encoded_fps, 6),
                "sim_clock": "state.t/control_fps",
                "video_clock": "source_frame/render_sample_fps",
                "sample_clock": (
                    "(source_frame+1)/render_sample_fps; post-physics "
                    "frame-end state relative to replay start"
                ),
            },
            {
                "record_type": "coordinates",
                "units": {"position": "m", "velocity": "m/s", "time": "s"},
                "origin": "pitch_center",
                "axes": {"x": "goal_to_goal", "y": "touchline_to_touchline", "z": "up"},
                "pitch": {
                    "length": float(self.s_cfg.length),
                    "width": float(self.s_cfg.width),
                    "goal_width": float(self.goal_w),
                    "goal_height": float(self.goal_h),
                    "player_boundary_margin": float(self.e_cfg.player_boundary_margin),
                },
            },
            {
                "record_type": "roster",
                "n_players": int(self.N),
                "team_sizes": [int(self.n_agents), int(self.n_opponents)],
                "agent_keys": list(self._agent_keys),
                "starter_player_ids": [int(p.id) for p in self.players],
                "team_id": np.asarray(F["team_id"])[0].astype(int).tolist(),
                "gk": np.asarray(F["gk_indices"])[0].astype(int).tolist(),
            },
            {
                "record_type": "schemas",
                "events": CANONICAL_EVENTS_SCHEMA,
                "tracking": "uos-footballmarl.tracking/2",
                "state_field_paths": self._state_field_paths(),
                "touch_labels": {str(int(k)): v for k, v in TOUCH_LABEL.items()},
                "restart_labels": {str(int(k)): v for k, v in RESTART_LABEL.items()},
                "foul_labels": {str(int(k)): v for k, v in FOUL_LABEL.items()},
                "gk_handling_restriction_labels": {
                    "-1": "none",
                    "0": "team0_backpass_or_throw",
                    "1": "team1_backpass_or_throw",
                    "2": "team0_own_release",
                    "3": "team1_own_release",
                },
                "event_counts": counts,
            },
            {
                "record_type": "environment",
                "dynamics_fingerprint": getattr(self, "dynamics_fingerprint", None),
                "dynamics": dynamics,
            },
            {
                "record_type": "summary",
                "data": self._match_summary(F, events, sample_fps=render_fps),
            },
        ]
        if user_metadata is not None:
            rows.append({"record_type": "user", "data": dict(user_metadata)})

        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                self._jsonl_write(fh, row)
        return len(rows)

    def _write_timeline_jsonl(self, F, path, stride=1):
        """프레임별 시계·국면만 담은 경량 타임로그(JSONL).

        state.jsonl은 선수 22명의 전 필드를 담아 한 경기가 수백 MB가 된다. 그런데 리플레이를
        다룰 때 대부분의 질의는 "언제 인플레이였나", "그 순간 점수가 몇이었나", "재개 중이었나"
        같은 것이라 전 필드가 필요 없다. 이 파일은 그 질의를 위한 색인이며 프레임당 100 바이트
        남짓이라 통째로 메모리에 올려 이벤트와 정렬할 수 있다.
        """
        T = len(F["t"])
        fps = float(self.control_fps)
        period = self._period_index(F)
        ball_pos = np.asarray(F["ball_pos"])
        ball_vel = np.asarray(F["ball_vel"])
        active = np.asarray(F["active_player"])
        team = np.asarray(F["team_id"])[0].astype(int)
        rows = 0
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "meta": {
                    "schema": "uos-footballmarl.timeline/2",
                    "frames": int(T),
                    "stride": int(stride),
                    "control_fps": fps,
                    "fields": ["f", "t", "clock_s", "period", "ball_state", "restart_kind",
                               "restart_t", "poss_team", "last_touch_team",
                               "gk_handling_restriction_code", "score",
                               "ball", "ball_speed", "n_active"],
                }
            }, ensure_ascii=False) + "\n")
            for f in range(0, T, max(1, int(stride))):
                fh.write(json.dumps({
                    "f": int(f),
                    "t": int(F["t"][f]),
                    "clock_s": round(float(F["t"][f]) / fps, 3),
                    "period": int(period[f]),
                    "ball_state": int(F["ball_state"][f]),
                    "restart_kind": int(F["restart_kind"][f]),
                    "restart_t": int(F["restart_t"][f]),
                    "poss_team": int(F["poss_team"][f]),
                    "last_touch_team": int(F["last_touch_team"][f]),
                    "gk_handling_restriction_code": int(
                        F["gk_handling_restricted_team"][f]
                    ),
                    "score": [int(F["score"][f][0]), int(F["score"][f][1])],
                    "ball": [round(float(v), 3) for v in ball_pos[f]],
                    "ball_speed": round(float(np.linalg.norm(ball_vel[f])), 3),
                    "n_active": [int(np.count_nonzero(active[f] & (team == 0))),
                                 int(np.count_nonzero(active[f] & (team == 1)))],
                }, separators=(",", ":")) + "\n")
                rows += 1
        return rows

    @staticmethod
    def _state_field_paths():
        """Map every stored State field to its nested replay JSON path."""

        return {
            "t": "t",
            "ball_pos": "ball.pos", "ball_vel": "ball.vel",
            "ball_spin": "ball.spin", "ball_state": "ball.state",
            "player_ctrl": "players.ball_ctrl", "player_pos": "players.pos",
            "player_vel": "players.vel", "player_facing": "players.facing",
            "attack_dir": "players.attack_dir", "team_id": "players.team_id",
            "gk_indices": "players.gk", "vmax": "players.vmax",
            "reach_z": "players.reach_z", "head_z": "players.head_z",
            "endurance_factor": "players.endurance_factor",
            "cooldown": "players.cooldown", "contact_lock_t": "players.contact_lock",
            "aerial_recovery_t": "players.aerial_recovery",
            "stamina_long": "players.stamina_long",
            "stamina_short": "players.stamina_short",
            "ctrl_lock_t": "players.ctrl_lock",
            "kickoff_team": "kickoff_team", "poss_team": "poss",
            "possession_t": "poss_t", "previous_poss_team": "previous_poss",
            "last_touch_team": "last_touch",
            "gk_handling_restricted_team": "gk_handling_restriction.code",
            "restart_team": "restart.team",
            "restart_t": "restart.t", "restart_kind": "restart.kind",
            "offside_flag": "players.offside", "pass_team": "pass.team",
            "pass_t": "pass.t", "foul_kind": "foul.kind",
            "foul_actor": "foul.actor", "foul_victim": "foul.victim",
            "pending_taker": "restart.taker", "setpiece_taker": "restart.setpiece_taker",
            "throw_taker": "restart.throw_taker", "touch": "touch",
            "touch_event_actor": "touch_events.actor",
            "touch_event_code": "touch_events.code",
            "touch_event_player_id": "touch_events.player_id",
            "touch_event_control_t": "touch_events.control_t",
            "touch_event_toi": "touch_events.toi",
            "touch_event_ball_pos": "touch_events.ball_pos",
            "touch_event_ball_vel_before": "touch_events.ball_vel_before",
            "touch_event_ball_vel_after": "touch_events.ball_vel_after",
            "touch_event_impulse": "touch_events.impulse",
            "woodwork_kind": "woodwork.kind",
            "woodwork_pos": "woodwork.pos",
            "woodwork_vel_in": "woodwork.vel_in",
            "woodwork_control_t": "woodwork.control_t",
            "ball_event_kind": "ball_events.kind",
            "ball_event_team": "ball_events.team",
            "ball_event_pos": "ball_events.pos",
            "ball_event_vel": "ball_events.vel",
            "ball_event_control_t": "ball_events.control_t",
            "bench_player_id": "bench.player_id",
            "bench_vmax": "bench.vmax",
            "bench_reach_z": "bench.reach_z",
            "bench_head_z": "bench.head_z",
            "bench_player_ctrl": "bench.player_ctrl",
            "bench_endurance_factor": "bench.endurance_factor",
            "bench_is_gk": "bench.is_gk",
            "bench_role_pos": "bench.role_pos",
            "retired_player_id": "bench.retired_player_id",
            "subs_remaining": "bench.subs_remaining",
            "sub_windows_used": "bench.sub_windows_used",
            "sub_window_open_t": "bench.sub_window_open_t",
            "layout_index": "formation.layout_index",
            "layout_since_t": "formation.layout_since_t",
            "last_touch_actor": "last_touch_actor",
            "episode_seed": "episode_seed",
            "score": "score",
            "yellow_cards": "players.yellow", "player_id": "players.player_id",
            "slot_generation": "players.slot_generation", "on_pitch": "players.on_pitch",
            "sent_off": "players.sent_off", "restart_indirect": "restart.indirect",
            "last_touch_code": "last_touch_code", "role_pos": "players.role_pos",
            "role_pos_count": "players.role_pos_count",
        }

    def _state_row(self, F, f):
        """Loss-bounded, JSON-safe representation of one complete State frame."""

        r3 = lambda a: np.round(np.asarray(a, np.float64), 3).tolist()
        handling_code = int(F["gk_handling_restricted_team"][f])
        handling_team = (
            handling_code - GK_HANDLING_RELEASE_OFFSET
            if handling_code >= GK_HANDLING_RELEASE_OFFSET
            else handling_code
        )
        handling_cause = (
            "own_release"
            if handling_code >= GK_HANDLING_RELEASE_OFFSET
            else ("backpass_or_throw" if handling_code >= 0 else "none")
        )
        return {
            "f": int(f), "t": int(F["t"][f]),
            "clock": round(int(F["t"][f]) / self.control_fps, 3),
            # 에피소드 신원. 세트피스 키커 결정자가 이 값을 tick·종류와 fold-in 해서
            # 쓰므로, 이것이 없으면 덤프만 보고 그 선택을 재현할 수 없다.
            "episode_seed": int(F["episode_seed"][f]),
            "ball": {"pos": r3(F["ball_pos"][f]), "vel": r3(F["ball_vel"][f]),
                     "spin": r3(F["ball_spin"][f]), "state": int(F["ball_state"][f])},
            "poss": int(F["poss_team"][f]), "last_touch": int(F["last_touch_team"][f]),
            "poss_t": int(F["possession_t"][f]),
            "previous_poss": int(F["previous_poss_team"][f]),
            "gk_handling_restriction": {
                "code": handling_code,
                "team": handling_team,
                "cause": handling_cause,
            },
            "kickoff_team": int(F["kickoff_team"][f]),
            "last_touch_code": int(F["last_touch_code"][f]),
            "pass": {"team": int(F["pass_team"][f]), "t": int(F["pass_t"][f])},
            "score": F["score"][f].astype(int).tolist(),
            "restart": {"kind": int(F["restart_kind"][f]), "t": int(F["restart_t"][f]),
                        "team": int(F["restart_team"][f]), "taker": int(F["pending_taker"][f]),
                        "setpiece_taker": int(F["setpiece_taker"][f]),
                        "throw_taker": int(F["throw_taker"][f]),
                        "indirect": bool(F["restart_indirect"][f])},
            "players": {"pos": r3(F["player_pos"][f]), "vel": r3(F["player_vel"][f]),
                        "facing": r3(F["player_facing"][f]),
                        "stamina_short": r3(F["stamina_short"][f]),
                        "stamina_long": r3(F["stamina_long"][f]),
                        "endurance_factor": r3(F["endurance_factor"][f]),
                        "attack_dir": r3(F["attack_dir"][f]),
                        "team_id": F["team_id"][f].astype(int).tolist(),
                        "gk": F["gk_indices"][f].astype(int).tolist(),
                        "vmax": r3(F["vmax"][f]), "reach_z": r3(F["reach_z"][f]),
                        "head_z": r3(F["head_z"][f]),
                        "ball_ctrl": r3(F["player_ctrl"][f]),
                        "active": F["active_player"][f].astype(int).tolist(),
                        "on_pitch": F["on_pitch"][f].astype(int).tolist(),
                        "sent_off": F["sent_off"][f].astype(int).tolist(),
                        "player_id": F["player_id"][f].astype(int).tolist(),
                        "slot_generation": F["slot_generation"][f].astype(int).tolist(),
                        "yellow": F["yellow_cards"][f].astype(int).tolist(),
                        "cooldown": r3(F["cooldown"][f]),
                        "contact_lock": F["contact_lock_t"][f].astype(int).tolist(),
                        "aerial_recovery": F["aerial_recovery_t"][f].astype(int).tolist(),
                        "ctrl_lock": F["ctrl_lock_t"][f].astype(int).tolist(),
                        "offside": F["offside_flag"][f].astype(int).tolist(),
                        "role_pos": r3(F["role_pos"][f]),
                        "role_pos_count": r3(F["role_pos_count"][f])},
            "touch": F["touch"][f].astype(int).tolist(),
            "touch_events": {
                "actor": F["touch_event_actor"][f].astype(int).tolist(),
                "code": F["touch_event_code"][f].astype(int).tolist(),
                "player_id": F["touch_event_player_id"][f].astype(int).tolist(),
                "control_t": F["touch_event_control_t"][f].astype(int).tolist(),
                "toi": r3(F["touch_event_toi"][f]),
                "ball_pos": r3(F["touch_event_ball_pos"][f]),
                "ball_vel_before": r3(
                    F["touch_event_ball_vel_before"][f]
                ),
                "ball_vel_after": r3(F["touch_event_ball_vel_after"][f]),
                "impulse": r3(F["touch_event_impulse"][f]),
            },
            "formation": {
                "layout_index": F["layout_index"][f].astype(int).tolist(),
                "layout_since_t": F["layout_since_t"][f].astype(int).tolist(),
            },
            "bench": {
                "player_id": F["bench_player_id"][f].astype(int).tolist(),
                "vmax": r3(F["bench_vmax"][f]),
                "reach_z": r3(F["bench_reach_z"][f]),
                "head_z": r3(F["bench_head_z"][f]),
                "player_ctrl": r3(F["bench_player_ctrl"][f]),
                "endurance_factor": r3(F["bench_endurance_factor"][f]),
                "is_gk": F["bench_is_gk"][f].astype(bool).tolist(),
                "role_pos": r3(F["bench_role_pos"][f]),
                "retired_player_id": F["retired_player_id"][f].astype(int).tolist(),
                "subs_remaining": F["subs_remaining"][f].astype(int).tolist(),
                "sub_windows_used": F["sub_windows_used"][f].astype(int).tolist(),
                "sub_window_open_t": F["sub_window_open_t"][f].astype(int).tolist(),
            },
            "last_touch_actor": int(F["last_touch_actor"][f]),
            "woodwork": {
                "kind": F["woodwork_kind"][f].astype(int).tolist(),
                "pos": r3(F["woodwork_pos"][f]),
                "vel_in": r3(F["woodwork_vel_in"][f]),
                "control_t": F["woodwork_control_t"][f].astype(int).tolist(),
            },
            "ball_events": {
                "kind": F["ball_event_kind"][f].astype(int).tolist(),
                "team": F["ball_event_team"][f].astype(int).tolist(),
                "pos": r3(F["ball_event_pos"][f]),
                "vel": r3(F["ball_event_vel"][f]),
                "control_t": F["ball_event_control_t"][f].astype(int).tolist(),
            },
            "foul": {"kind": int(F["foul_kind"][f]), "actor": int(F["foul_actor"][f]),
                     "victim": int(F["foul_victim"][f])},
        }

    def _write_state_jsonl(self, F, path, stride=1):
        """프레임별 전체 state를 JSONL로 — 1행 = {"meta": ...}(정적 정보·스키마), 이후
        **행당 1프레임**. 교체 경계에서 바뀌는 identity/profile/role도 프레임 행에 기록한다.

        starter profile을 meta에만 두고 ``player_id``·``slot_generation``·능력치·``role_pos``를
        생략하면 교체 뒤 JSONL을 읽을 때 새 사람이 선발 능력치로 재생된다.
        ``state_field_paths``가 :class:`State`의 모든 저장 필드를 JSON 경로에
        대응시키므로 구조적으로 State를 복원할 수 있다(연속값은 덤프 규약대로 소수 3자리).
        스트리밍 기록이라 풀경기도 메모리 상주 없이 쓰고, 반환값은 기록 프레임 수다.
        """
        T = len(F["t"]); N = self.N
        r3 = lambda a: np.round(np.asarray(a, np.float64), 3).tolist()
        meta = {
            "n_agents": self.n_agents, "n_opponents": self.n_opponents, "N": N,
            "control_fps": self.control_fps, "num_frames_total": T, "stride": stride,
            "pitch": {"length": self.s_cfg.length, "width": self.s_cfg.width,
                      "goal_width": self.goal_w, "goal_height": self.goal_h,
                      "player_boundary_margin": self.e_cfg.player_boundary_margin},
            "agent_keys": list(self._agent_keys),
            "starter_player_ids": [int(p.id) for p in self.players],
            "team_id": F["team_id"][0].astype(int).tolist(),
            "gk": F["gk_indices"][0].astype(int).tolist(),
            "vmax": r3(F["vmax"][0]), "reach_z": r3(F["reach_z"][0]), "head_z": r3(F["head_z"][0]),
            "restart_kinds": {v: k for k, v in RESTART_LABEL.items()},
            "touch_codes": {TOUCH_LABEL[k]: k for k in TOUCH_LABEL},
            # payload가 늘면 버전도 오른다. 8에서 bench·formation·ball_events·
            # last_touch_actor가 들어왔고, 아래 ``state_field_shapes``도 이때 신설됐다.
            # 9에서 episode_seed가, 10에서 골 프레임 충돌 버퍼 woodwork가 들어왔다.
            # 11은 포메이션 보간 전용 필드를 제거하고 목표 layout 하나만 남긴다.
            # 12는 지구력 능력치와 고공 시도 회복 락을 추가한다.
            "frame_schema_version": 12,
            "state_field_paths": self._state_field_paths(),
            # 중첩 리스트는 **빈 배열의 마지막 차원**을 표현하지 못한다. 벤치가 없는 env의
            # ``bench_role_pos``는 (2,0,2)인데 JSON에서는 [[],[]]가 되어 (2,0)으로 읽힌다.
            # 선언 shape를 함께 실어 소비자가 정확히 복원하게 한다.
            "state_field_shapes": {
                name: list(np.asarray(F[name][0]).shape)
                for name in self._state_field_paths()
            },
            "frame_schema": "ball{pos,vel,spin,state} poss last_touch "
                            "gk_handling_restriction{code,team,cause} score "
                            "kickoff_team pass{team,t} last_touch_code "
                            "restart{kind,t,team,taker,setpiece_taker,throw_taker,indirect} "
                            "players{pos,vel,facing,attack_dir,team_id,gk,vmax,reach_z,head_z,"
                            "ball_ctrl,endurance_factor,stamina_short,stamina_long,cooldown,"
                            "contact_lock,aerial_recovery,ctrl_lock,yellow,offside,"
                            "player_id,slot_generation,on_pitch,sent_off,active,role_pos,"
                            "role_pos_count} touch "
                            "touch_events{actor,code,player_id,control_t,toi,ball_pos,"
                            "ball_vel_before,ball_vel_after,impulse} "
                            "ball_events{kind,team,pos,vel,control_t} "
                            "woodwork{kind,pos,vel_in,control_t} "
                            "last_touch_actor episode_seed "
                            "bench{player_id,vmax,reach_z,head_z,player_ctrl,endurance_factor,is_gk,"
                            "role_pos,retired_player_id,subs_remaining,"
                            "sub_windows_used,sub_window_open_t} "
                            "formation{layout_index,layout_since_t} "
                            "foul{kind,actor,victim}",
        }
        n = 0
        with open(path, "w") as fh:
            fh.write(json.dumps({"meta": meta}, separators=(",", ":")) + "\n")
            for f in range(0, T, stride):
                row = self._state_row(F, f)
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

    def _pitch_base(self, to_px, ppm):
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
        margin = self.e_cfg.player_boundary_margin
        outer = (70, 108, 76)
        seg((-hx - margin, hy + margin), (hx + margin, hy + margin), outer)
        seg((hx + margin, hy + margin), (hx + margin, -hy - margin), outer)
        seg((hx + margin, -hy - margin), (-hx - margin, -hy - margin), outer)
        seg((-hx - margin, -hy - margin), (-hx - margin, hy + margin), outer)
        seg((0, hy), (0, -hy))                                  # 하프웨이
        th = np.linspace(0, 2 * np.pi, 200)
        center_r = self.s_cfg.center_circle_radius
        pts = [to_px(center_r * np.cos(a), center_r * np.sin(a)) for a in th]
        for (y1, x1, _), (y2, x2, _) in zip(pts[:-1], pts[1:]):
            self._line(img, y1, x1, y2, x2, COL_LINE)
        pl = self.s_cfg.penalty_area_length; pw2 = self.s_cfg.penalty_area_width / 2.0
        gl = self.s_cfg.goal_area_length; ga2 = self.s_cfg.goal_area_width / 2.0
        gw2 = self.s_cfg.goal_width / 2.0; gh = self.s_cfg.goal_height
        far = self._FAR_SCALE
        for side in (-1, 1):
            xg, xb = side * hx, side * (hx - pl)
            seg((xg, pw2), (xb, pw2)); seg((xb, pw2), (xb, -pw2)); seg((xb, -pw2), (xg, -pw2))
            xa = side * (hx - gl)
            seg((xg, ga2), (xa, ga2)); seg((xa, ga2), (xa, -ga2)); seg((xa, -ga2), (xg, -ga2))
            for first, second in self._penalty_arc_segments(side):
                seg(first, second)
            # 골대 3D 프레임: 골라인(지면) + 양 포스트(높이 goal_height 투영) + 크로스바.
            # 정적이라 사전계산에 포함 — 프레임당 비용 0.
            gy1, gx1, d1 = to_px(xg, gw2); gy2, gx2, d2 = to_px(xg, -gw2)
            self._line(img, gy1, gx1, gy2, gx2, (235, 235, 235))
            h1 = int(gh * ppm * (far + (1.0 - far) * d1))
            h2 = int(gh * ppm * (far + (1.0 - far) * d2))
            for off in (0, side):                               # 포스트/크로스바 2px 두께
                self._line(img, gy1, gx1 + off, gy1 - h1, gx1 + off, (250, 250, 250))
                self._line(img, gy2, gx2 + off, gy2 - h2, gx2 + off, (250, 250, 250))
                self._line(img, gy1 - h1 - (0 if off == 0 else 1), gx1,
                           gy2 - h2 - (0 if off == 0 else 1), gx2, (250, 250, 250))
        return img

    def _penalty_arc_points(self, side, samples=48):
        """페널티 아크(D)의 점열 — 스폿 중심 원에서 **박스 밖** 부분만.

        규격은 이미 config에 있는데(``penalty_arc_radius``·``engine.penalty_spot``)
        선만 그려지지 않고 있었다. 재개 규칙은 이 반경을 쓰므로(페널티 시 나머지 선수가
        비켜야 하는 원) 화면에 없으면 왜 그 자리에 못 서는지 읽히지 않는다.

        수치 마스킹 대신 반각을 직접 푼다 — 마스킹은 한쪽 골대에서 t=0을 넘어 감겨
        선분이 경기장을 가로지른다.
        """

        radius = self.s_cfg.penalty_arc_radius
        spot = self.e_cfg.penalty_spot
        front = self.s_cfg.penalty_area_length - spot   # 스폿에서 박스 앞선까지
        if not 0.0 < front < radius:
            return []                                   # 아크가 박스 밖으로 안 나온다
        half = float(np.arccos(front / radius))
        base = np.pi if side > 0 else 0.0               # 골대 반대 방향
        angles = np.linspace(base - half, base + half, samples)
        cx = side * (self.s_cfg.length / 2.0 - spot)
        # ``seg``와 같은 (x, y) 순서로 돌려준다.
        return list(zip(cx + radius * np.cos(angles), radius * np.sin(angles)))

    def _penalty_arc_segments(self, side):
        points = self._penalty_arc_points(side)
        return list(zip(points[:-1], points[1:]))

    _FONT = None
    _TEXT_CACHE = {}

    def _text_mask(self, text, scale=1):
        """PIL 기본 비트맵 폰트로 텍스트 마스크를 1회 렌더 후 캐시(프레임당 텍스트 비용 ≈ 복사)."""
        _ensure_render_dependencies()
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
# 반지름을 픽셀이 아닌 **미터**로 주면 원근이 투영에서 저절로 처리된다(지면 링 = possession_ring 계열).
TOUCH_RING_RGB = {code: tuple(c / 255.0 for c in col) for code, col in TOUCH_RING.items()}
RING_R_MIN_M, RING_R_MAX_M = 0.7, 3.5     # m — Δv MIN..f2b_speed_max를 선형으로 채운다(비교: 소유링 1.5m)

# 프레임 뷰가 참조하는 State 필드(전부 numpy로 스택). 원본 Frame 속성명으로 매핑된다.
_NEED = ("player_pos", "player_vel", "player_facing", "touch", "ball_pos", "ball_vel",
         "ball_spin", "ball_state", "poss_team", "last_touch_team", "restart_kind",
         "restart_t", "restart_team", "pending_taker", "attack_dir", "offside_flag",
         "pass_t", "stamina_short", "stamina_long", "yellow_cards", "active_player", "foul_kind",
         "foul_actor", "foul_victim", "score", "t", "team_id", "gk_indices")


def _long_stamina_color(s):
    s = float(np.clip(s, 0, 1))
    return (np.clip(2 * (1 - s), 0, 1), np.clip(1.4 * s, 0, 1), 0.18)


def _short_stamina_color(s):
    s = float(np.clip(s, 0, 1))
    return (np.clip(1.8 * (1 - s), 0, 1), 0.25 + 0.65 * s, 0.25 + 0.70 * s)


class _Cam:
    def __init__(self, target, azim, elev, dist, focal):
        self.t = np.asarray(target, float); self.focal = focal
        a, e = np.radians(azim), np.radians(elev)
        d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        self.pos = self.t + dist * d
        f = self.t - self.pos; f /= np.linalg.norm(f)
        # ``elev``가 ±90°면 시선이 월드 up과 평행해 ``cross(f, up)``이 영벡터가 된다.
        # 그대로 정규화하면 회전행렬 전체가 NaN이 되고 모든 투영이 사라진다 — ``elev``는
        # rich 렌더의 공개 옵션이고 유한성만 검사하므로 ±90은 실제로 들어올 수 있다.
        # 그 경우 화면 오른쪽은 방위각이 정한다(수직 시점에서 자연스러운 규약이다).
        r = np.cross(f, [0.0, 0.0, 1.0])
        r_norm = float(np.linalg.norm(r))
        if r_norm < 1e-9:
            r = np.array([-np.sin(a), np.cos(a), 0.0])
            r_norm = 1.0
        r = r / r_norm
        u = np.cross(r, f)
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


def _env_roster(env):
    """rich 렌더가 쓰는 정적 명부(team_id·gk_indices)를 싸게 얻는다.

    원래는 ``env.reset(PRNGKey(0))``으로 뽑았는데, reset은 jit 캐시가 걸리지 않아 **호출마다
    약 2초**를 쓴다(161프레임 렌더 6.4초 중 30%가 여기였다). 필요한 건 경기 내내 불변인
    소속·GK 마스크뿐이므로 (a) 렌더할 states가 이미 갖고 있으면 그걸 쓰고(`roster=` 인자),
    (b) 없으면 한 번만 reset해 env에 붙여 재사용한다."""
    cached = getattr(env, "_rich_roster", None)
    if cached is None:
        st = env.reset(jax.random.PRNGKey(0))[1]
        cached = (np.asarray(st.team_id).astype(int),
                  np.asarray(st.gk_indices).astype(int))
        try:
            env._rich_roster = cached
        except Exception:      # noqa: BLE001 - 불변 env여도 렌더는 계속돼야 한다
            pass
    return cached


def _roster_from_states(states):
    """정규화된 states에서 명부를 뽑는다. 못 뽑으면 None(호출자가 env.reset로 폴백)."""
    try:
        team = np.asarray(states.team_id)
        gk = np.asarray(states.gk_indices)
    except AttributeError:
        return None
    if team.ndim == 2 and gk.ndim == 2 and team.shape == gk.shape:
        return team[0].astype(int), gk[0].astype(int)
    if team.ndim == 1 and gk.ndim == 1:
        return team.astype(int), gk.astype(int)
    return None


class RichRenderer:
    """State 시퀀스 → 방송형 mp4. `RichRenderer(env).render(states, out_path, fps=...)`."""
    GRASS = ("#2f8f3e", "#37a449"); LINE = "#f4f4f4"; BG = "#0c1622"
    APRON = "#1c2a17"; STAND = "#3a4150"
    MINIMAP_RECT = RICH_MINIMAP_RECT
    DEFAULT_DYNAMIC_CAMERA = DEFAULT_DYNAMIC_CAMERA
    BALL_TRAIL_RGB = (1.0, 1.0, 1.0)

    def __init__(self, env, *, roster=None, azim=-90.0, elev=42.0, dist=150.0, focal=820.0,
                 figsize=(12.8, 7.2), dpi=150, trail_len=18, player_scale=1.7):
        # figsize×dpi = 1920×1080(Full HD, 짝수 해상도). 느리면 dpi를 낮춰 호출.
        self.env = env
        self.E = env.e_cfg
        # 정적 속성(team_id·gk)만 필요하다 — 호출자가 states에서 넘겨주면 env.reset을 건너뛴다.
        team, gk = roster if roster is not None else _env_roster(env)
        self.team = np.asarray(team).astype(int); self.gk = np.asarray(gk).astype(int)
        self.N = env.N
        self.L = env.s_cfg.length; self.W = env.s_cfg.width
        self.player_boundary_margin = env.e_cfg.player_boundary_margin
        self.bench_x0 = (-24.0, 4.0)
        self.gw = env.s_cfg.goal_width; self.ch = env.s_cfg.goal_height
        self.control_fps = float(env.control_fps)
        self.cam = _Cam([0, 0, 0], azim, elev, dist, focal)
        self.figsize, self.dpi = figsize, dpi
        self.trail_len, self.player_scale = trail_len, player_scale
        self.pcol = [GKC[self.team[i]] if self.gk[i] else TEAM[self.team[i]] for i in range(self.N)]
        self.person_labels = _roster_display_labels(env)
        self.pnum = [""] * self.N
        for tm in (0, 1):
            c = 1
            for i in np.where(self.team == tm)[0]:
                self.pnum[i] = "GK" if self.gk[i] else str(c); c += 0 if self.gk[i] else 1
        self._precompute_static()

    def _restart_status_label(self, restart_kind, restart_t):
        """Return the rich set-piece badge from the environment timing SSOT."""

        return self.env._restart_status_label(restart_kind, restart_t)

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
                stamina_short=F["stamina_short"][t],
                stamina_long=F["stamina_long"][t],
                foul_kind=int(F["foul_kind"][t]),
                ball_spin=F["ball_spin"][t], ball_vel=F["ball_vel"][t],
                foul_actor=int(F["foul_actor"][t]), foul_victim=int(F["foul_victim"][t]),
                attack_dir=F["attack_dir"][t], offside_flag=F["offside_flag"][t].astype(bool),
                sp_taker=int(F["pending_taker"][t]), ball_state=int(F["ball_state"][t]),
                restart_t=int(F["restart_t"][t]), restart_team=int(F["restart_team"][t]),
                pass_t=int(F["pass_t"][t]), yellow_cards=F["yellow_cards"][t],
                on_pitch=F["active_player"][t].astype(bool),
                sent_off=F["sent_off"][t].astype(bool),
                player_id=F["player_id"][t].astype(int),
                bench_player_id=F["bench_player_id"][t].astype(int),
                retired_player_id=F["retired_player_id"][t].astype(int),
                subs_remaining=F["subs_remaining"][t].astype(int),
                layout_index=F["layout_index"][t].astype(int),
                clock=float(F["t"][t]) / self.control_fps))
        return frames

    def _precompute_static(self):
        L, W = self.L, self.W; hx, hy = L / 2, W / 2
        lines = []

        def rect(x0, x1, y0, y1):
            lines.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], float))
        rect(-hx, hx, -hy, hy); lines.append(np.array([[0, -hy], [0, hy]], float))
        m = self.player_boundary_margin
        self.player_boundary_line = np.array(
            [[-hx - m, -hy - m], [hx + m, -hy - m],
             [hx + m, hy + m], [-hx - m, hy + m], [-hx - m, -hy - m]],
            float,
        )
        center_r = self.env.s_cfg.center_circle_radius
        th = np.linspace(0, 2 * np.pi, 60)
        lines.append(np.stack([center_r * np.cos(th), center_r * np.sin(th)], 1))
        pen_len = self.env.s_cfg.penalty_area_length
        pen_hw = self.env.s_cfg.penalty_area_width / 2.0
        goal_len = self.env.s_cfg.goal_area_length
        goal_hw = self.env.s_cfg.goal_area_width / 2.0
        arc_r = self.env.s_cfg.penalty_arc_radius
        spot = self.env.e_cfg.penalty_spot
        arc_front = pen_len - spot
        for s in (-1, 1):
            gx = s * hx
            rect(gx, gx - s * pen_len, -pen_hw, pen_hw)
            rect(gx, gx - s * goal_len, -goal_hw, goal_hw)
            # 페널티 아크(D) — 박스 밖으로 나오는 부분만. 규격은 config에 있었는데
            # 선만 빠져 있었다.
            if 0.0 < arc_front < arc_r:
                half = float(np.arccos(arc_front / arc_r))
                base = np.pi if s > 0 else 0.0
                a = np.linspace(base - half, base + half, 40)
                cx = s * (hx - spot)
                lines.append(np.stack([cx + arc_r * np.cos(a),
                                       arc_r * np.sin(a)], 1))
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
        ap, sd, sh = self.player_boundary_margin + 1.0, 6.0, 4.0
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

    def _precompute(self, frames, *, structured_events, every=1):
        T, N = len(frames), self.N
        ph = np.zeros(N); kf = np.zeros(N); kh = np.zeros(N); kt = np.zeros(N, int)
        PH = np.zeros((T, N)); AMP = np.zeros((T, N)); KF = np.zeros((T, N)); KH = np.zeros((T, N)); KT = np.zeros((T, N), int)
        score = np.zeros((T, TEAM_COUNT), int)
        shots = np.zeros((T, TEAM_COUNT), int)
        passes = np.zeros((T, TEAM_COUNT), int)
        owner = np.full(T, NO_PLAYER, int)
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
            for tm in (0, 1):
                m = (self.team == tm)
                sh[tm] += int(np.sum(m & np.isin(tc, list(shotc)))); pa[tm] += int(np.sum(m & np.isin(tc, list(passc))))
            if t > 0 and fr.scored >= 0 and frames[t - 1].scored < 0: sc[fr.scored] += 1
            score[t] = sc; shots[t] = sh; passes[t] = pa
            active = (
                np.asarray(fr.on_pitch, dtype=bool)
                if fr.on_pitch is not None else np.ones(N, dtype=bool)
            )
            owner[t] = _visual_possession_owner(
                self.team,
                fr.P,
                active,
                fr.poss_team,
                fr.ball,
                fr.restart_t,
                fr.sp_taker if fr.sp_taker is not None else NO_PLAYER,
            )
        return dict(PH=PH, AMP=AMP, KF=KF, KH=KH, KT=KT, score=score,
                    shots=shots, passes=passes, owner=owner,
                    events=self._structured_event_feed(
                        structured_events, frame_count=T, every=every
                    ))

    def _player_feed_label(self, player):
        """Stable rich-feed label for a state slot."""

        if player is None:
            return "?"
        player = int(player)
        if not (0 <= player < self.N):
            return "?"
        return "GK" if self.gk[player] else f"#{self.pnum[player]}"

    def _shirt_number(self, player_id, slot):
        """Team-local shirt number for the person currently occupying a slot."""

        label = self.person_labels.get(int(player_id))
        if label is not None and len(label) > 1:
            return label[1:]
        return self.pnum[int(slot)]

    def _structured_event_feed(self, records, *, frame_count, every=1):
        """Map canonical replay event records to rich HUD rows.

        Launch rows stay on the launch frame, while pass/shot results are shown
        at ``outcome_frame``.  Consequently ``PASS A -> B`` no longer leaks a
        future first toucher at launch or masquerades as the policy's intended
        receiver.  ``every`` maps an event to the first retained render frame at
        or after its source frame.
        """

        if isinstance(every, (bool, np.bool_)) or not isinstance(every, numbers.Integral):
            raise TypeError(f"every must be a positive integer, got {every!r}")
        every = int(every)
        if every <= 0:
            raise ValueError(f"every must be > 0, got {every!r}")
        frame_count = int(frame_count)
        out = [[] for _ in range(max(0, frame_count))]
        if not out:
            return out

        SP = {
            RK_THROWIN: ("THROW-IN", (0.80, 0.86, 0.55)),
            RK_GOALKICK: ("GOAL KICK", (0.55, 0.80, 0.95)),
            RK_CORNER: ("CORNER", (0.55, 0.90, 0.75)),
            RK_FREEKICK: ("FREE KICK", (0.95, 0.80, 0.45)),
            RK_KICKOFF: ("KICK-OFF", (0.92, 0.92, 0.96)),
            RK_PENALTY: ("PENALTY!", (1.0, 0.45, 0.45)),
            RK_OFFSIDE: ("OFFSIDE FK", (1.0, 0.78, 0.30)),
            RK_GK_HOLD: ("GK HOLD", (0.55, 0.90, 0.98)),
        }
        DRIB_COL = (0.55, 0.80, 0.98)
        last_drib = {}
        restart_cause_frames = {
            int(row["f"])
            for row in records
            if row.get("type") == "restart" and row.get("cause")
        }
        shot_goal_frames = {
            int(row["outcome_frame"])
            for row in records
            if row.get("type") == "shot_outcome"
            and row.get("outcome") == "goal"
            and row.get("outcome_frame") is not None
        }

        def mapped_frame(row, *, outcome=False):
            source = row.get("outcome_frame") if outcome else row.get("f")
            if source is None:
                return None
            source = int(source)
            if source < 0:
                return None
            return min((source + every - 1) // every, frame_count - 1)

        def emit(row, label, color, *, outcome=False):
            target = mapped_frame(row, outcome=outcome)
            if target is not None:
                out[target].append(("* " + label, color))

        for row in records:
            kind = row.get("type")
            tm = row.get("team")
            player = row.get("player")
            ptxt = self._player_feed_label(player)

            if kind == "touch":
                code = int(row.get("code", TOUCH_NONE))
                if code == TOUCH_GK_CATCH:
                    emit(row, f"GK CATCH  {team_label(tm)} {ptxt}", CATCH_COL)
                elif code == TOUCH_PARRY:
                    emit(row, f"PARRY  {team_label(tm)} {ptxt}", CATCH_COL)
                elif code == TOUCH_TACKLE:
                    emit(row, f"TACKLE  {team_label(tm)} {ptxt}", TACKLE_COL)
                elif code == TOUCH_INTERCEPT:
                    emit(row, f"INTERCEPT  {team_label(tm)} {ptxt}", INTERCEPT_COL)
                elif code == TOUCH_DEFLECT:
                    emit(row, f"DEFLECT  {team_label(tm)} {ptxt}", MISS_COL)
                elif code in (TOUCH_SHOOT, TOUCH_SHOOT_HEAD):
                    label = "HEADER SHOT" if code == TOUCH_SHOOT_HEAD else "SHOT"
                    emit(row, f"{label}  {team_label(tm)} {ptxt}", SHOOT_COL)
                elif code in (TOUCH_PASS, TOUCH_PASS_HEAD):
                    label = "HEADER PASS" if code == TOUCH_PASS_HEAD else "PASS"
                    emit(row, f"{label}  {team_label(tm)} {ptxt}", PASS_COL)
                elif code == TOUCH_DRIBBLE:
                    clock = float(row.get("clock_s", 0.0))
                    if clock - last_drib.get(player, -np.inf) >= EVENT_DEDUP_SECONDS:
                        emit(row, f"DRIBBLE  {team_label(tm)} {ptxt}", DRIB_COL)
                    last_drib[player] = clock
                elif code == TOUCH_BODY_TRAP:
                    emit(row, f"BODY TRAP  {team_label(tm)} {ptxt}", DRIB_COL)
                continue

            if kind == "pass_outcome":
                result = row.get("outcome")
                if result == "unresolved":
                    continue
                if result == "completed":
                    receiver = self._player_feed_label(row.get("receiver"))
                    label = f"PASS COMPLETE  {team_label(tm)} {ptxt} -> {receiver}"
                    color = PASS_COL
                elif result == "intercepted":
                    by = row.get("by")
                    btm = int(self.team[int(by)]) if by is not None else "?"
                    label = f"PASS INTERCEPTED  by {team_label(btm)} {self._player_feed_label(by)}"
                    color = INTERCEPT_COL
                elif result == "saved":
                    by = row.get("by")
                    btm = int(self.team[int(by)]) if by is not None else "?"
                    label = f"PASS CLAIMED  by {team_label(btm)} {self._player_feed_label(by)}"
                    color = CATCH_COL
                elif result == "out_of_play":
                    label = f"PASS OUT  {team_label(tm)} {ptxt}"
                    color = MISS_COL
                elif result == "retained":
                    label = f"CARRY  {team_label(tm)} {ptxt}"
                    color = DRIB_COL
                elif result == "goal":
                    label = f"PASS -> GOAL  {team_label(tm)} {ptxt}"
                    color = TEAM[int(tm)]
                elif result == "own_goal":
                    label = f"OWN GOAL  {team_label(tm)} {ptxt}"
                    color = MISS_COL
                elif result == "woodwork":
                    label = (
                        f"OFF THE {str(row.get('woodwork', 'woodwork')).upper()}"
                        f"  {team_label(tm)} {ptxt}"
                    )
                    color = MISS_COL
                else:
                    label = f"PASS {str(result).upper()}  {team_label(tm)} {ptxt}"
                    color = MISS_COL
                emit(row, label, color, outcome=True)
                continue

            if kind == "shot_outcome":
                result = row.get("outcome")
                if result == "unresolved":
                    continue
                if result == "goal":
                    label = f"GOAL!  {team_label(tm)} {ptxt}"
                    color = TEAM[int(tm)]
                elif result == "own_goal":
                    label = f"OWN GOAL  {team_label(tm)} {ptxt}"
                    color = MISS_COL
                elif result == "saved":
                    by = row.get("by")
                    btm = int(self.team[int(by)]) if by is not None else "?"
                    label = f"SHOT SAVED  by {team_label(btm)} {self._player_feed_label(by)}"
                    color = CATCH_COL
                elif result == "blocked":
                    by = row.get("by")
                    btm = int(self.team[int(by)]) if by is not None else "?"
                    label = f"SHOT BLOCKED  by {team_label(btm)} {self._player_feed_label(by)}"
                    color = INTERCEPT_COL
                elif result == "out_of_play":
                    label = f"SHOT OFF TARGET  {team_label(tm)} {ptxt}"
                    color = MISS_COL
                elif result == "rebound_own":
                    receiver = self._player_feed_label(row.get("receiver"))
                    label = f"SHOT REBOUND  {team_label(tm)} {receiver}"
                    color = SHOOT_COL
                elif result == "retained":
                    label = f"SHOT RETAINED  {team_label(tm)} {ptxt}"
                    color = SHOOT_COL
                elif result == "woodwork":
                    label = (
                        f"OFF THE {str(row.get('woodwork', 'woodwork')).upper()}"
                        f"!  {team_label(tm)} {ptxt}"
                    )
                    color = SHOOT_COL
                else:
                    label = f"SHOT {str(result).upper()}  {team_label(tm)} {ptxt}"
                    color = MISS_COL
                emit(row, label, color, outcome=True)
                continue

            if kind == "restart":
                rk = int(row.get("kind", RK_NONE))
                name, color = SP.get(rk, (row.get("label", "RESTART"), MISS_COL))
                cause = row.get("cause")
                if cause:
                    actor = cause.get("actor")
                    victim = cause.get("victim")
                    at = cause.get("actor_team")
                    vt = cause.get("victim_team")
                    detail = str(cause.get("foul_label", "FOUL")).replace("FOUL(", "").rstrip(")")
                    if actor is not None and victim is not None:
                        label = (
                            f"{name}  {team_label(tm)} | {team_label(at)} {self._player_feed_label(actor)} "
                            f"fouled {team_label(vt)} {self._player_feed_label(victim)} [{detail.lower()}]"
                        )
                    else:
                        label = f"{name}  {team_label(tm)} [{detail.lower()}]"
                else:
                    label = f"{name}" + (f"  {team_label(tm)}" if tm is not None else "")
                emit(row, label, color)
                continue

            if kind == "goal":
                # A shot outcome carries the scorer slot.  Suppress the less
                # informative team-only goal row on that same terminal frame.
                if int(row.get("f", -1)) not in shot_goal_frames:
                    emit(row, f"GOAL!  {team_label(tm)}", TEAM[int(tm)])
                continue

            if kind == "foul":
                if int(row.get("f", -1)) in restart_cause_frames:
                    continue
                victim = row.get("victim")
                detail = str(row.get("label", "FOUL"))
                label = f"{detail}  {team_label(tm)} {ptxt}"
                if victim is not None:
                    vtm = int(self.team[int(victim)])
                    label += f" -> {team_label(vtm)} {self._player_feed_label(victim)}"
                emit(row, label, (1.0, 0.42, 0.42))
                continue

            if kind == "card":
                emit(row, f"YELLOW CARD  {team_label(tm)} {ptxt}", (1.0, 0.84, 0.12))
                continue
            if kind == "sent_off":
                emit(row, f"SENT OFF  {team_label(tm)} {ptxt}", (1.0, 0.24, 0.24))
                continue
            if kind == "substitution":
                slot = row.get("slot")
                emit(
                    row,
                    f"SUBSTITUTION  {team_label(tm)} {self._player_feed_label(slot)} "
                    f"({_display_person_id(row.get('player_out'), self.person_labels)} "
                    f"-> {_display_person_id(row.get('player_in'), self.person_labels)})",
                    (0.68, 0.92, 0.72),
                )
                continue
            if kind == "formation":
                emit(
                    row,
                    f"FORMATION TARGET  {team_label(tm)}  "
                    f"{row.get('from')} -> {row.get('to')}",
                    (0.62, 0.80, 1.0),
                )
                continue
            if kind == "possession":
                previous = row.get("from_team")
                if tm is not None and previous is not None and int(tm) != int(previous):
                    emit(row, f"TURNOVER -> {team_label(tm)}", (0.9, 0.9, 0.95))
                continue
            if kind == "offside_flag":
                emit(row, f"OFFSIDE FLAG  {team_label(tm)} {ptxt}", (1.0, 0.78, 0.30))

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
               every=1, feed_n=10, feed_secs=12.0,
               dynamic_cam=DEFAULT_DYNAMIC_CAMERA, cam_zoom=0.66,
               frame_annotations=None, structured_events=None):
        _ensure_render_dependencies()
        _ensure_mpl()   # matplotlib 지연 로드(모듈 전역 plt/LineCollection/… 채움) — rich 진입점
        source_fps = float(fps)
        if structured_events is None:
            # RichRenderer를 공개 Render.render_mp4 우회로 직접 쓰는 호출도 같은 SSOT를
            # 유지한다. 정상 진입점에서는 이미 추출한 목록이 전달되어 이 계산은 생략된다.
            structured_events = self.env._extract_events(
                self.env._stack_full(states), sample_fps=source_fps
            )
        frames = self._frames_from_states(states)[::every]
        if frame_annotations is not None:
            frame_annotations = list(frame_annotations)[::every]
        # ``every`` is a render-cost sampling knob, not a time-lapse control.
        # Lower the encoder rate by the same factor so match duration and all
        # time-based overlays remain invariant.
        fps = float(fps) / int(every)
        S = self._precompute(
            frames, structured_events=structured_events, every=every
        )
        T = len(frames); N = self.N
        coach_feed = _coach_command_feed(
            structured_events, frame_count=T, every=every,
            person_labels=self.person_labels,
        )
        cam_win = self._camera_path(frames, zoom=cam_zoom) if dynamic_cam else None
        t_start = time.time()
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi); fig.patch.set_facecolor(self.BG)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0]); ax.set_facecolor(self.BG)
        ax.set_xlim(*self.xlim); ax.set_ylim(*self.ylim); ax.set_aspect("equal"); ax.axis("off")
        axm = fig.add_axes(self.MINIMAP_RECT); axm.set_zorder(30)

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
        rich_anchor_cache = {}

        def rich_layout_anchors(layout_row, active_row):
            key = (np.asarray(layout_row).tobytes(),
                   np.asarray(active_row).tobytes())
            cached = rich_anchor_cache.get(key)
            if cached is None:
                cached = np.asarray(self.env.formation_layout_anchors(
                    jnp.asarray(layout_row, jnp.int32),
                    jnp.asarray(active_row, bool)))
                rich_anchor_cache[key] = cached
            return cached

        rich_role_cache = {}

        def rich_layout_roles(layout_row, active_row):
            """rich 쪽 역할 코드 — light의 ``layout_roles``와 같은 규약·같은 캐시 키."""

            key = (np.asarray(layout_row).tobytes(),
                   np.asarray(active_row).tobytes())
            cached = rich_role_cache.get(key)
            if cached is None:
                cached = np.asarray(classify_roles(
                    jnp.asarray(rich_layout_anchors(layout_row, active_row),
                                jnp.float32),
                    jnp.asarray(self.gk == 1, bool),
                    jnp.asarray(self.team, jnp.int32),
                    jnp.asarray(active_row, bool)))
                rich_role_cache[key] = cached
            return cached

        boundary_uv = self.cam.proj(
            np.c_[self.player_boundary_line, np.zeros(len(self.player_boundary_line))]
        )
        ax.add_collection(LineCollection(
            [boundary_uv], colors="#7ca888", linewidths=1.0,
            linestyles="dashed", alpha=0.75, zorder=0.9,
        ))
        net_uv = [self.cam.proj(s) for s in self.net_segs]
        ax.add_collection(LineCollection(net_uv, colors="#cfd8e2", linewidths=0.6, alpha=0.38, zorder=1.5))
        goal_uv = [self.cam.proj(s) for s in self.goal_segs]
        ax.add_collection(LineCollection(goal_uv, colors="#ffffff", linewidths=2.8, zorder=2))
        ax.add_patch(Rectangle((0, 0.90), 1, 0.10, transform=ax.transAxes, facecolor="#0a121c", alpha=0.84, edgecolor="none", zorder=20))
        ax.text(0.445, 0.965, team_label(TEAM_0), transform=ax.transAxes, ha="right", va="center", color=TEAM[0], fontsize=13, fontweight="bold", zorder=21)
        ax.text(0.555, 0.965, team_label(TEAM_1), transform=ax.transAxes, ha="left", va="center", color=TEAM[1], fontsize=13, fontweight="bold", zorder=21)
        # 현재 목표 모양 — **상시** 표기. 실제 선수 좌표는 정책·물리를 통해 뒤따른다.
        shape_text = [
            ax.text(0.020, 0.128, "", transform=ax.transAxes, ha="left", va="center",
                    color=TEAM[0], fontsize=9.0, family="monospace", zorder=21),
            ax.text(0.980, 0.128, "", transform=ax.transAxes, ha="right", va="center",
                    color=TEAM[1], fontsize=9.0, family="monospace", zorder=21),
        ]
        # 실제 벤치 후보는 22개 field slot 밖의 State 배열이다. 비활성 field slot을
        # dugout으로 옮기는 것만으로는 후보 선수가 한 명도 보이지 않으므로, 현재 후보와
        # 교체 OUT 명단/잔여 횟수를 별도의 상시 로스터 카드로 표시한다.
        roster_text = [
            ax.text(0.018, 0.076, "", transform=ax.transAxes, ha="left", va="center",
                    color=TEAM[0], fontsize=8.2, fontweight="bold",
                    family="monospace", linespacing=1.35,
                    bbox={"boxstyle": "square,pad=0.35", "fc": "#071019",
                          "ec": TEAM[0], "lw": 0.8, "alpha": 0.84}, zorder=21),
            ax.text(0.982, 0.076, "", transform=ax.transAxes, ha="right", va="center",
                    color=TEAM[1], fontsize=8.2, fontweight="bold",
                    family="monospace", linespacing=1.35,
                    bbox={"boxstyle": "square,pad=0.35", "fc": "#071019",
                          "ec": TEAM[1], "lw": 0.8, "alpha": 0.84}, zorder=21),
        ]
        # BenchPlayer rows are not field slots.  Draw them as small people in
        # the actual dugouts instead of exposing only their numeric IDs in the
        # roster card.  A used reserve becomes a retired figure on the row
        # behind it, while a dismissed field player remains at State's farther
        # ``sent_off`` coordinate and is rendered by the normal skeleton path.
        reserve_capacity = int(
            np.asarray(frames[0].bench_player_id).size
            + np.asarray(frames[0].retired_player_id).size
        )
        reserve_torsos = ax.scatter(
            np.zeros(reserve_capacity), np.zeros(reserve_capacity),
            s=np.zeros(reserve_capacity), marker="s", c="#000000",
            edgecolors="#000000", linewidths=0.8, zorder=6.2,
        )
        reserve_heads = ax.scatter(
            np.zeros(reserve_capacity), np.zeros(reserve_capacity),
            s=np.zeros(reserve_capacity), marker="o", c="#d3a27d",
            edgecolors="#000000", linewidths=0.7, zorder=7.2,
        )
        for tm in (TEAM_0, TEAM_1):
            bench_caption = self.cam.proj(np.asarray([[
                self.bench_x0[tm], -(self.W / 2.0 + 2.0), 0.0,
            ]]))[0]
            ax.text(
                bench_caption[0], bench_caption[1],
                f"{team_label(tm)} BENCH", color=TEAM[tm], fontsize=6.8,
                fontweight="bold", family="monospace", zorder=8,
            )
        ax.add_patch(Rectangle((0.006, 0.165), 0.165, 0.715, transform=ax.transAxes,
                               facecolor="#070e18", alpha=0.55, edgecolor="#24405e", linewidth=1.0, zorder=18))
        ax.text(0.020, 0.860, "MATCH EVENTS", transform=ax.transAxes, ha="left", va="center",
                color="#8fb6e0", fontsize=10.5, fontweight="bold", family="monospace", zorder=19)
        if title:
            ax.text(0.020, 0.924, str(title), transform=ax.transAxes, ha="left", va="center",
                    color="#9fb4c8", fontsize=9.5, fontweight="bold",
                    fontproperties=_rich_title_font(), zorder=22)
        self._draw_minimap_static(axm)

        ps = self.player_scale
        shadows = ax.scatter(np.zeros(N), np.zeros(N), s=34, c="#06140b", alpha=0.22, zorder=5)
        # 선수 한 명의 뼈대 12개 선분은 색이 항상 같다 — NaN으로 끊어 이어 붙이면 경로 264개가
        # 22개가 된다(Agg는 비유한 정점에서 스트로크를 끊으므로 화면은 동일). draw_path_collection
        # 호출 비용이 경로 수에 선형이라 프레임당 4 ms가 줄고, 색 배열도 264→22로 짧아진다.
        bodies = LineCollection([np.zeros((_BONE_VERTS, 2))] * N,
                                colors=list(self.pcol), linewidths=2.4, zorder=6)
        ax.add_collection(bodies)
        heads = ax.scatter(np.zeros(N), np.zeros(N), s=100, c=self.pcol, edgecolors="#0a0a0a", linewidths=0.8, zorder=7)
        shown_player_ids = np.asarray(frames[0].player_id, dtype=np.int64).copy()
        initial_numbers = [
            self._shirt_number(shown_player_ids[i], i) for i in range(N)
        ]
        nums = [ax.text(0, 0, initial_numbers[i], color="white", fontsize=7.0, fontweight="bold",
                        ha="center", va="center", zorder=11) for i in range(N)]
        stamina_short_bg = LineCollection(
            [np.zeros((2, 2))] * N, colors=[(0, 0, 0, 0.5)] * N,
            linewidths=2.5, zorder=9
        )
        stamina_short_fg = LineCollection(
            [np.zeros((2, 2))] * N, colors=[(0.2, 0.9, 0.9)] * N,
            linewidths=2.5, zorder=10
        )
        stamina_long_bg = LineCollection(
            [np.zeros((2, 2))] * N, colors=[(0, 0, 0, 0.5)] * N,
            linewidths=2.5, zorder=9
        )
        stamina_long_fg = LineCollection(
            [np.zeros((2, 2))] * N, colors=[(0, 1, 0)] * N,
            linewidths=2.5, zorder=10
        )
        for collection in (
            stamina_short_bg, stamina_short_fg,
            stamina_long_bg, stamina_long_fg,
        ):
            ax.add_collection(collection)
        ball_sh = ax.scatter([0], [0], s=30, c="#06140b", alpha=0.35, zorder=3)
        ball_drop, = ax.plot([0, 0], [0, 0], color="#d8d8d8", lw=0.8, ls=(0, (2, 2)), zorder=4)
        ball = ax.scatter([0], [0], s=44, c="#fbfbfb", edgecolors="#222", linewidths=0.7, zorder=8)
        spin_tick, = ax.plot([0, 0], [0, 0], color="#ffd400", lw=1.8, zorder=9, alpha=0.0)
        ball_trail = LineCollection([], zorder=3); ax.add_collection(ball_trail)
        # 고정 반지름+팀색 glow = 환경이 인정한 소유권. 아래 imp_rings의 가변
        # 반지름+터치색 pulse = 공에 가한 힘. 두 링은 의미·수명·collection을 공유하지 않는다.
        possession_glow, = ax.plot([], [], lw=6.5, zorder=2, alpha=0.0, solid_capstyle="round")
        possession_ring, = ax.plot([], [], lw=2.0, zorder=3, alpha=0.0,
                                   solid_capstyle="round", color=TEAM[0])
        imp_rings = LineCollection(
            [], zorder=2.4, linestyles=(0, (2.2, 1.5))
        )  # 임펄스 링: 점선·가변 반지름·ttl 페이드(소유권의 실선 고정 링과 분리)
        ax.add_collection(imp_rings)
        sp_label = ax.text(0, 0, "", color="#ffe14d", fontsize=8.5, fontweight="bold",
                           ha="center", va="bottom", zorder=24,
                           bbox=dict(boxstyle="square,pad=0.2", fc="#1a1200", ec="#ffd400", lw=0.8, alpha=0.85))
        sp_label.set_visible(False)
        zone_fill = ax.add_patch(MplPolygon([[0, 0]], closed=True, facecolor="#ffffff", edgecolor="none", alpha=0.0, zorder=1.9))
        zone_edge, = ax.plot([], [], lw=1.6, ls="--", alpha=0.0, zorder=2.3)
        zone_box_fill = ax.add_patch(MplPolygon([[0, 0]], closed=True, facecolor="#ffffff", edgecolor="none", alpha=0.0, zorder=1.9))
        zone_box_edge, = ax.plot([], [], lw=1.6, ls="--", alpha=0.0, zorder=2.3)
        zone_mark_edge, = ax.plot([], [], lw=1.4, ls=(0, (3, 3)), alpha=0.0, zorder=2.3)
        enc_marks = ax.scatter(np.zeros(N), np.zeros(N), s=np.zeros(N), marker="o",
                               facecolors="none", edgecolors="#ff3b3b", linewidths=1.8, zorder=12)
        offside_glow, = ax.plot([], [], lw=5.5, zorder=2.6, alpha=0.0, solid_capstyle="round")
        offside_line, = ax.plot([], [], lw=2.2, zorder=2.8, alpha=0.0, solid_capstyle="round")
        offside_label = ax.text(
            0, 0, "", color="#ffffff", fontsize=7.2, fontweight="bold",
            ha="center", va="bottom", zorder=13, family="monospace",
            bbox=dict(boxstyle="square,pad=0.15", fc="#071019", ec="#64788d",
                      lw=0.6, alpha=0.78),
        )
        offside_label.set_visible(False)
        off_marks = ax.scatter(np.zeros(N), np.zeros(N), s=np.zeros(N), marker="v",
                               facecolors="#ff4d4d", edgecolors="#ffdada", linewidths=0.5, zorder=13)
        card_marks = ax.scatter(np.zeros(N), np.zeros(N), s=np.zeros(N), marker="s",
                                facecolors=[(0, 0, 0, 0)] * N, edgecolors="#141414", linewidths=0.6, zorder=14)
        score_txt = ax.text(0.50, 0.965, "0 : 0", transform=ax.transAxes, ha="center", va="center", color="#fff", fontsize=15, fontweight="bold", zorder=21)
        clock_txt = ax.text(0.50, 0.924, "00:00", transform=ax.transAxes, ha="center", va="center", color="#9fb4c8", fontsize=10, fontweight="bold", family="monospace", zorder=22)
        goal_flash = ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#ffffff", alpha=0.0, edgecolor="none", zorder=19))
        goal_txt = ax.text(0.50, 0.55, "", transform=ax.transAxes, ha="center", va="center", color="#ffd400", fontsize=30, fontweight="bold", zorder=23)
        # 감독 명령은 GOAL과 같은 중앙 이벤트 계층이지만, 위쪽 카드와 별도 팔레트로
        # 의미를 분리한다. 네 행이면 같은 프레임의 양 팀 교체+포메이션을 모두 담는다.
        coach_panel = ax.add_patch(Rectangle(
            (0.245, 0.555), 0.510, 0.170, transform=ax.transAxes,
            facecolor="#071016", edgecolor=_rgb_hex(COL_COMMAND_FORMATION),
            linewidth=2.2, alpha=0.92, zorder=25,
        ))
        coach_title = ax.text(
            0.50, 0.700, "", transform=ax.transAxes, ha="center", va="center",
            color=_rgb_hex(COL_COMMAND_FORMATION), fontsize=15, fontweight="bold",
            family="monospace", zorder=26,
        )
        coach_rows = [
            ax.text(0.50, 0.663 - 0.030 * k, "", transform=ax.transAxes,
                    ha="center", va="center", color="#ffffff", fontsize=9.2,
                    fontweight="bold", family="monospace", zorder=26)
            for k in range(4)
        ]
        coach_panel.set_visible(False)
        coach_title.set_visible(False)
        for row in coach_rows:
            row.set_visible(False)
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

        _flatten_circle_markers(
            shadows, heads, reserve_heads, enc_marks,
            ball, ball_sh, mm_dots, mm_ball,
        )

        # blit 목록에 넣어야 실제로 그려진다 — ``ax.add_collection``만으로는 배경에 남고
        # 프레임마다 갱신되지 않는다. 등록과 그리기가 갈려 있는 구조라 둘 다 해야 한다.
        dyn_ax = [*shape_text, *roster_text,
                  reserve_torsos, reserve_heads, shadows, ball_trail,
                  offside_glow, offside_line, offside_label,
                  possession_glow, possession_ring,
                  imp_rings,
                  zone_fill, zone_edge, zone_box_fill, zone_box_edge, zone_mark_edge, enc_marks,
                  bodies, heads, *nums,
                  stamina_short_bg, stamina_short_fg,
                  stamina_long_bg, stamina_long_fg,
                  off_marks, card_marks, ball_sh, ball_drop, ball, spin_tick,
                  goal_flash, sp_label, coach_panel, coach_title, *coach_rows,
                  annotation_border, annotation_txt, score_txt, clock_txt,
                  goal_txt, *feed_txt]
        dyn_axm = [mm_view, mm_dots, mm_ball]
        bg = None
        if not dynamic_cam:
            for art in dyn_ax + dyn_axm: art.set_animated(True)
            fig.canvas.draw(); bg = fig.canvas.copy_from_bbox(fig.bbox)

        is_gif = str(out_path).lower().endswith(".gif")
        # mp4는 light와 같은 이유로 인코딩을 전담 스레드에 넘긴다(파이프 쓰기가 matplotlib
        # 드로잉과 직렬화된다). gif는 imageio가 전 프레임을 모아 한 번에 쓰는 경로라 겹칠
        # 여지가 없어 그대로 둔다.
        writer = (imageio.get_writer(out_path, mode="I", duration=1.0 / fps, loop=0) if is_gif
                  else _ThreadedFrameWriter(out_path, fps=fps,
                                            output_params=_encode_params(23)))
        feed = []; feed_life = max(40, int(fps * feed_secs))
        feed_state = [None] * feed_n
        trail_buf = []
        # 임펄스 링 상태: [ttl0, ttl, 선수, rgb, 반지름m, 선폭]. light의 event_ttl=8@25fps와 같은 체감 길이.
        imp_fx = []
        imp_ttl = max(2, int(round(fps * 0.32)))
        imp_span = max(float(self.E.f2b_speed_max) - RING_IMPULSE_MIN, 1e-3)
        goal_hold_team = -1; goal_hold_until = -1
        coach_hold = []; coach_hold_until = -1
        annotation_hold = ""; annotation_until = -1

        def _set_text(art, s):
            if art.get_text() != s: art.set_text(s)
        HW = 7.0
        for t, fr in enumerate(frames):
            if coach_feed[t]:
                coach_hold = coach_feed[t]
                coach_hold_until = t + max(1, int(round(fps * 2.4)))
            coach_visible = t < coach_hold_until and bool(coach_hold)
            coach_panel.set_visible(coach_visible)
            coach_title.set_visible(coach_visible)
            if coach_visible:
                first_color = tuple(value / 255.0 for value in coach_hold[0]["color"])
                coach_panel.set_edgecolor(first_color)
                if len(coach_hold) == 1:
                    command = coach_hold[0]
                    _set_text(coach_title, command["headline"])
                    coach_title.set_color(first_color)
                    rows = [(command["detail"], first_color)]
                else:
                    _set_text(coach_title, "COACH COMMANDS")
                    coach_title.set_color("#e1edf7")
                    rows = [
                        (f"{command['headline']}  |  {command['detail']}",
                         tuple(value / 255.0 for value in command["color"]))
                        for command in coach_hold[:4]
                    ]
                for k, row in enumerate(coach_rows):
                    visible = k < len(rows)
                    row.set_visible(visible)
                    if visible:
                        _set_text(row, rows[k][0])
                        row.set_color(rows[k][1])
            else:
                for row in coach_rows:
                    row.set_visible(False)
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
            dismissed = (
                np.asarray(fr.sent_off, dtype=bool)
                if fr.sent_off is not None else np.zeros(N, dtype=bool)
            )
            for tm in (TEAM_0, TEAM_1):
                _set_text(
                    roster_text[tm],
                    _roster_status_text(
                        tm,
                        fr.bench_player_id[tm],
                        fr.retired_player_id[tm],
                        fr.subs_remaining[tm],
                        person_labels=self.person_labels,
                    ),
                )
            # State already separates bench/retired/sent-off field coordinates.
            # Repacking every inactive slot into the dugout collapsed a dismissed
            # player onto the bench and contradicted both the state contract and
            # the light renderer.  Preserve those authoritative coordinates.
            Pd = fr.P
            facing_d = fr.facing

            reserve_people = _rich_reserve_people(
                fr, self.bench_x0, self.W
            )
            if reserve_people:
                reserve_pos = np.stack(
                    [person["pos"] for person in reserve_people]
                )
                torso_uv = self.cam.proj(np.c_[
                    reserve_pos,
                    np.full(len(reserve_people), 0.65),
                ])
                reserve_head_uv = self.cam.proj(np.c_[
                    reserve_pos,
                    np.full(len(reserve_people), 1.45),
                ])
                reserve_torsos.set_offsets(torso_uv)
                reserve_heads.set_offsets(reserve_head_uv)
                reserve_torsos.set_sizes(np.asarray([
                    52.0 if person["kind"] == "bench" else 40.0
                    for person in reserve_people
                ]))
                reserve_heads.set_sizes(np.asarray([
                    28.0 if person["kind"] == "bench" else 23.0
                    for person in reserve_people
                ]))
                reserve_torsos.set_facecolors([
                    TEAM[person["team"]]
                    if person["kind"] == "bench" else "#555e68"
                    for person in reserve_people
                ])
                reserve_torsos.set_edgecolors([
                    TEAM[person["team"]] for person in reserve_people
                ])
                reserve_heads.set_facecolors([
                    "#d3a27d" if person["kind"] == "bench" else "#827a76"
                    for person in reserve_people
                ])
                reserve_heads.set_edgecolors([
                    TEAM[person["team"]] for person in reserve_people
                ])
            else:
                empty = np.empty((0, 2), np.float64)
                reserve_torsos.set_offsets(empty)
                reserve_heads.set_offsets(empty)
                reserve_torsos.set_sizes(np.empty(0))
                reserve_heads.set_sizes(np.empty(0))
            for tm in (TEAM_0, TEAM_1):
                name = layout_label(int(fr.layout_index[tm]))
                _set_text(shape_text[tm], f"{team_label(tm)}  {name}")

            segs, hd, _ = self._skeletons(Pd, facing_d, S["PH"][t], S["AMP"][t], S["KF"][t], S["KH"][t])
            seg_uv = self.cam.proj(segs.reshape(-1, 3)).reshape(N, _BONES, 2, 2)
            bone_uv = np.full((N, _BONE_VERTS, 2), np.nan)
            bone_uv[:, 0::3] = seg_uv[:, :, 0]      # 각 선분 시작점
            bone_uv[:, 1::3] = seg_uv[:, :, 1]      # 각 선분 끝점 (2::3은 NaN = 끊김)
            bodies.set_segments(list(bone_uv))
            kcol = []
            for i in range(N):
                act = (S["KF"][t][i] > 0.25) or (S["KH"][t][i] > 0.25)
                col = ("#8f2834" if dismissed[i] else
                       "#5a626e" if not op[i] else
                       (SHOOT_COL if kt[i] in (TOUCH_SHOOT, TOUCH_SHOOT_HEAD) else PASS_COL) if act else self.pcol[i])
                kcol.append(col)
            bodies.set_color(kcol)
            head_uv = self.cam.proj(hd); heads.set_offsets(head_uv)
            shadows.set_offsets(self.cam.proj(np.c_[Pd, np.zeros(N)]))
            num_uv = self.cam.proj(np.c_[Pd, np.full(N, 0.95 * ps)])
            current_player_ids = np.asarray(fr.player_id, dtype=np.int64)
            changed_ids = np.flatnonzero(current_player_ids != shown_player_ids)
            for i in changed_ids:
                _set_text(nums[i], self._shirt_number(current_player_ids[i], i))
            if changed_ids.size:
                shown_player_ids[:] = current_player_ids
            for i in range(N): nums[i].set_position((num_uv[i, 0], num_uv[i, 1]))
            x0 = head_uv[:, 0] - HW; x1 = head_uv[:, 0] + HW
            short_y = head_uv[:, 1] + 7.0
            long_y = head_uv[:, 1] + 3.5
            short_x = x0 + 2 * HW * np.clip(fr.stamina_short, 0, 1)
            long_x = x0 + 2 * HW * np.clip(fr.stamina_long, 0, 1)
            short_bg_segments = np.stack(
                [np.stack([x0, short_y], 1), np.stack([x1, short_y], 1)], 1
            )
            short_fg_segments = np.stack(
                [np.stack([x0, short_y], 1), np.stack([short_x, short_y], 1)], 1
            )
            long_bg_segments = np.stack(
                [np.stack([x0, long_y], 1), np.stack([x1, long_y], 1)], 1
            )
            long_fg_segments = np.stack(
                [np.stack([x0, long_y], 1), np.stack([long_x, long_y], 1)], 1
            )
            stamina_short_bg.set_segments(short_bg_segments)
            stamina_short_fg.set_segments(short_fg_segments)
            stamina_long_bg.set_segments(long_bg_segments)
            stamina_long_fg.set_segments(long_fg_segments)
            stamina_short_fg.set_color([
                _short_stamina_color(fr.stamina_short[i]) for i in range(N)
            ])
            stamina_long_fg.set_color([
                _long_stamina_color(fr.stamina_long[i]) for i in range(N)
            ])
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
            alive = (int(fr.ball_state) == BALL_ALIVE)
            trail_buf.append(fr.ball.copy())
            if len(trail_buf) > self.trail_len: trail_buf.pop(0)
            if len(trail_buf) >= 2:
                tuv = self.cam.proj(np.array(trail_buf))
                tsegs = np.stack([tuv[:-1], tuv[1:]], 1); m = len(tsegs)
                tcol = np.zeros((m, 4)); tcol[:, :3] = self.BALL_TRAIL_RGB
                tcol[:, 3] = np.linspace(0.04, 0.55, m)
                ball_trail.set_segments(list(tsegs)); ball_trail.set_color(tcol)
                ball_trail.set_linewidths(np.linspace(0.5, 2.2, m))
            else:
                ball_trail.set_segments([])
            ball.set_color("#fbfbfb" if alive else "#8a94a0")
            own = int(S["owner"][t])
            if own >= 0:
                ring = self._ground_ring(fr.P[own, 0], fr.P[own, 1], 1.5)
                possession_glow.set_data(ring[:, 0], ring[:, 1]); possession_glow.set_color(self.pcol[own]); possession_glow.set_alpha(0.55)
                possession_ring.set_data(ring[:, 0], ring[:, 1]); possession_ring.set_color(self.pcol[own]); possession_ring.set_alpha(0.95)
            else:
                possession_glow.set_alpha(0.0); possession_ring.set_alpha(0.0)
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
                # Runtime restart SSOT와 같은 종류별 총 지연을 표시한다.
                lbl = self._restart_status_label(
                    fr.restart_kind, fr.restart_t
                )
                # 역할을 함께 찍는다 — light의 TAKER 배지와 같은 이유다. 누가 찼는지가
                # 아니라 **왜 그 사람이 찼는지**가 보여야 키커 정책을 눈으로 검증할 수 있다.
                lbl = f"{lbl}  {ROLE_NAMES[int(rich_layout_roles(fr.layout_index, op)[spk])]}"
                hd_uv = self.cam.proj(np.array([[fr.P[spk, 0], fr.P[spk, 1], 2.2 * ps]]))[0]
                sp_label.set_position((hd_uv[0], hd_uv[1])); _set_text(sp_label, lbl); sp_label.set_visible(True)
            else:
                sp_label.set_visible(False)
            rk_now = int(fr.restart_kind)
            rteam = int(fr.restart_team)
            if rteam < 0:
                rteam = int(fr.poss_team)
            # restart._encroach_geometry의 종류별 기하를 rich overlay에 동일하게 투영한다.
            # 골킥은 박스만(거짓 9.15m 원 제거), 자기 박스 안 FK/오프사이드는
            # 박스∩9.15m 제약, 페널티는 박스∪아크와 GK 골라인을 그린다.
            zone_kinds = (RK_KICKOFF, RK_THROWIN, RK_GOALKICK, RK_CORNER,
                          RK_FREEKICK, RK_OFFSIDE, RK_PENALTY)
            if (int(fr.restart_t) > 0 and rk_now in zone_kinds
                    and 0 <= rteam < len(TEAM) and fr.attack_dir is not None):
                E = self.E; zcol = TEAM[rteam]
                zone_center = np.asarray(fr.ball[:2], dtype=float).copy()
                rteam_slots = np.where(self.team == rteam)[0]
                adir_rt = float(fr.attack_dir[rteam_slots[0]])
                if rk_now == RK_THROWIN:
                    side = (np.sign(zone_center[1]) if zone_center[1] != 0.0
                            else adir_rt)
                    zone_center[1] = side * self.W / 2.0
                elif rk_now == RK_CORNER:
                    zone_center[0] = ((np.sign(zone_center[0])
                                      if zone_center[0] != 0.0 else adir_rt)
                                     * self.L / 2.0)
                    zone_center[1] = ((np.sign(zone_center[1])
                                      if zone_center[1] != 0.0 else adir_rt)
                                     * self.W / 2.0)
                d_spot = np.linalg.norm(fr.P - zone_center[None, :], axis=1)
                inz = np.zeros(N, dtype=bool)
                own_gx_all = -np.asarray(fr.attack_dir) * self.L / 2.0
                goal_line_depth = (
                    (fr.P[:, 0] - own_gx_all) * np.asarray(fr.attack_dir)
                )
                on_line = (
                    (np.abs(goal_line_depth) <= E.goal_line_tolerance)
                    & (np.abs(fr.P[:, 1])
                       <= self.env.goal_w / 2.0 + E.goal_post_tolerance)
                )
                penalty_gk_legal = (
                    (goal_line_depth <= E.goal_line_tolerance)
                    & (np.abs(fr.P[:, 1])
                       <= self.env.goal_w / 2.0 + E.goal_post_tolerance)
                )
                radial = rk_now in (RK_KICKOFF, RK_THROWIN, RK_CORNER,
                                    RK_FREEKICK, RK_OFFSIDE)
                if radial or rk_now == RK_PENALTY:
                    zr = (self.env.s_cfg.center_circle_radius if rk_now == RK_KICKOFF
                          else E.throwin_clear if rk_now == RK_THROWIN
                          else E.clear_dist + self.env.s_cfg.corner_arc_radius
                          if rk_now == RK_CORNER
                          else self.env.s_cfg.penalty_arc_radius if rk_now == RK_PENALTY
                          else E.clear_dist)
                    ring = self._ground_ring(zone_center[0], zone_center[1], zr, n=48)
                    zone_fill.set_xy(ring); zone_fill.set_facecolor(zcol); zone_fill.set_alpha(0.10)
                    zone_edge.set_data(ring[:, 0], ring[:, 1]); zone_edge.set_color(zcol); zone_edge.set_alpha(0.8)
                    if radial:
                        inz = (self.team != rteam) & op & (d_spot < zr) & (~on_line)
                        if rk_now == RK_KICKOFF:
                            is_kicker = np.arange(N) == spk
                            # ``attack_dir`` points toward the attacking half,
                            # so x*adir>0 is the Law-8 violation.  Keep the
                            # expression written as the runtime signed margin
                            # (margin=-x*adir) to make parity review mechanical.
                            own_half_illegal = (
                                -fr.P[:, 0] * np.asarray(fr.attack_dir) < 0.0
                            )
                            inz = (op & (~is_kicker)
                                   & (own_half_illegal
                                      | ((self.team != rteam) & (d_spot < zr))))
                else:
                    zone_fill.set_alpha(0.0); zone_edge.set_alpha(0.0)

                own_gx = -adir_rt * self.L / 2.0
                own_bf = own_gx + adir_rt * self.env.pen_len
                spot_in_own_box = (
                    min(own_gx, own_bf) <= zone_center[0] <= max(own_gx, own_bf)
                    and abs(zone_center[1]) <= self.env.pen_hw
                )
                box_freekick = (
                    rk_now in (RK_FREEKICK, RK_OFFSIDE)
                    and spot_in_own_box
                )
                show_box = rk_now in (RK_GOALKICK, RK_PENALTY) or box_freekick
                if show_box:
                    own_box = rk_now == RK_GOALKICK or box_freekick
                    gx = ((-adir_rt) if own_box else adir_rt) * self.L / 2.0
                    bf = gx + ((adir_rt) if own_box else (-adir_rt)) * self.env.pen_len
                    hw = self.env.pen_hw
                    corners = np.array([[gx, -hw, 0.0], [gx, hw, 0.0],
                                        [bf, hw, 0.0], [bf, -hw, 0.0]])
                    buv = self.cam.proj(corners)
                    zone_box_fill.set_xy(buv); zone_box_fill.set_facecolor(zcol); zone_box_fill.set_alpha(0.10)
                    bcl = np.vstack([buv, buv[:1]])
                    zone_box_edge.set_data(bcl[:, 0], bcl[:, 1]); zone_box_edge.set_color(zcol); zone_box_edge.set_alpha(0.8)
                    xlo, xhi = min(gx, bf), max(gx, bf)
                    in_box = ((fr.P[:, 0] >= xlo) & (fr.P[:, 0] <= xhi)
                              & (np.abs(fr.P[:, 1]) <= hw))
                    if rk_now == RK_GOALKICK:
                        inz = (self.team != rteam) & op & in_box & (~on_line)
                        zone_mark_edge.set_alpha(0.0)
                    elif box_freekick:
                        inz = inz | (
                            (self.team != rteam) & op & in_box & (~on_line)
                        )
                        zone_mark_edge.set_alpha(0.0)
                    else:
                        is_kicker = np.arange(N) == spk
                        def_gk = (self.team == (1 - rteam)) & (self.gk == 1)
                        ahead_of_mark = (
                            (fr.P[:, 0] - fr.ball[0]) * adir_rt >= 0.0
                        )
                        inz = (op & (~is_kicker)
                               & (((in_box
                                    | (d_spot < self.env.s_cfg.penalty_arc_radius)
                                    | ahead_of_mark) & (~def_gk))
                                  | (def_gk & (~penalty_gk_legal))))
                        # Law-14's behind-mark half-plane is not visible from
                        # the box/arc overlays alone.  Draw its transverse
                        # boundary across the same 5 m player domain used by
                        # movement and projection.
                        mark_line = np.array([
                            [fr.ball[0], -self.W / 2.0 - E.player_boundary_margin, 0.0],
                            [fr.ball[0], self.W / 2.0 + E.player_boundary_margin, 0.0],
                        ])
                        mark_uv = self.cam.proj(mark_line)
                        zone_mark_edge.set_data(mark_uv[:, 0], mark_uv[:, 1])
                        zone_mark_edge.set_color(zcol)
                        zone_mark_edge.set_alpha(0.8)
                else:
                    zone_box_fill.set_alpha(0.0); zone_box_edge.set_alpha(0.0)
                    if rk_now == RK_KICKOFF:
                        half_line = np.array([
                            [0.0, -self.W / 2.0 - E.player_boundary_margin, 0.0],
                            [0.0, self.W / 2.0 + E.player_boundary_margin, 0.0],
                        ])
                        half_uv = self.cam.proj(half_line)
                        zone_mark_edge.set_data(half_uv[:, 0], half_uv[:, 1])
                        zone_mark_edge.set_color(zcol)
                        zone_mark_edge.set_alpha(0.65)
                    else:
                        zone_mark_edge.set_alpha(0.0)
                enc_uv = self.cam.proj(np.c_[fr.P, np.zeros(N)])
                enc_marks.set_offsets(enc_uv); enc_marks.set_sizes(np.where(inz, 150.0, 0.0))
            else:
                zone_fill.set_alpha(0.0); zone_edge.set_alpha(0.0)
                zone_box_fill.set_alpha(0.0); zone_box_edge.set_alpha(0.0)
                zone_mark_edge.set_alpha(0.0)
                enc_marks.set_sizes(np.zeros(N))
            adir = fr.attack_dir; off = fr.offside_flag
            att, lx = _visual_offside_line(
                self.team,
                fr.P,
                op,
                adir,
                fr.poss_team,
                fr.last_team,
                fr.ball_state,
                fr.restart_t,
            )
            if att in (TEAM_0, TEAM_1) and lx is not None:
                ol = self.cam.proj(np.array([[lx, -self.W / 2, 0.0], [lx, self.W / 2, 0.0]]))
                offside_glow.set_data(ol[:, 0], ol[:, 1]); offside_glow.set_color(TEAM[att]); offside_glow.set_alpha(0.22)
                offside_line.set_data(ol[:, 0], ol[:, 1]); offside_line.set_color(TEAM[att]); offside_line.set_alpha(0.95)
                offside_label.set_position((ol[1, 0], ol[1, 1]))
                _set_text(offside_label, f"OFFSIDE LINE  {team_label(att)}")
                offside_label.set_color(TEAM[att]); offside_label.set_visible(True)
            else:
                offside_glow.set_alpha(0.0); offside_line.set_alpha(0.0)
                offside_label.set_visible(False)
            if off is not None and bool(np.any(off)):
                mk_uv = self.cam.proj(np.c_[fr.P, np.full(N, 2.0 * ps)])
                off_marks.set_offsets(mk_uv); off_marks.set_sizes(np.where(off, 55.0, 0.0))
            else:
                off_marks.set_sizes(np.zeros(N))
            yc = fr.yellow_cards if fr.yellow_cards is not None else np.zeros(N)
            card_uv = head_uv.copy(); card_uv[:, 0] += 10.0; card_uv[:, 1] += 2.0
            card_marks.set_offsets(card_uv)
            ccol = np.zeros((N, 4)); csz = np.zeros(N)
            for i in range(N):
                if dismissed[i]:
                    ccol[i] = (0.90, 0.15, 0.15, 1.0); csz[i] = 62.0
                elif yc[i] >= 1:
                    ccol[i] = (0.98, 0.82, 0.10, 1.0); csz[i] = 48.0
            card_marks.set_facecolors(ccol); card_marks.set_sizes(csz)
            if not bool(op.all()):
                heads.set_color([
                    "#8f2834" if dismissed[i]
                    else self.pcol[i] if op[i]
                    else "#6a7078"
                    for i in range(N)
                ])
            else:
                heads.set_color(self.pcol)
            sc = S["score"][t]
            _set_text(score_txt, f"{sc[0]} : {sc[1]}")
            msec = int(round(fr.clock))
            _set_text(clock_txt, f"{msec // 60:02d}:{msec % 60:02d}")
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
                m = self.player_boundary_margin
                gv[:, 0] = np.clip(gv[:, 0], -self.L / 2 - m, self.L / 2 + m)
                gv[:, 1] = np.clip(gv[:, 1], -self.W / 2 - m, self.W / 2 + m)
                mm_view.set_data(gv[:, 0], gv[:, 1])
                fig.canvas.draw()
            else:
                fig.canvas.restore_region(bg)
                for art in dyn_ax: ax.draw_artist(art)
                for art in dyn_axm: axm.draw_artist(art)
                fig.canvas.blit(fig.bbox)
            # buffer_rgba()는 캔버스 내부 버퍼를 그대로 가리키고 다음 draw가 그 자리를
            # 덮어쓴다. 인코딩이 비동기가 된 이상 반드시 복사해서 넘겨야 한다.
            # 알파를 여기서 떼면(`[..., :3]`) 연속이 아닌 gather가 되어 Full HD 한 장에
            # 23.7 ms가 든다 — 캔버스 버퍼를 통째로 memcpy하면 0.38 ms이고, 알파는
            # ffmpeg가 yuv420p로 변환하며 어차피 버린다(imageio는 4채널 입력을 rgba로 받는다).
            writer.append_data(np.array(fig.canvas.buffer_rgba(), copy=True))
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
        m = self.player_boundary_margin
        axm.add_patch(Rectangle(
            (-hx - m, -hy - m), 2.0 * (hx + m), 2.0 * (hy + m),
            fill=False, edgecolor="#7ca888", linewidth=0.7, linestyle="--", zorder=1,
        ))
        axm.set_xlim(-hx - m - 1, hx + m + 1)
        axm.set_ylim(-hy - m - 1, hy + m + 1)
        axm.set_aspect("equal"); axm.axis("off")
