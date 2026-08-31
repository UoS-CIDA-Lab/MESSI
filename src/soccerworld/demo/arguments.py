"""Numerical-stack-free command-line parser for the packaged demo."""

from __future__ import annotations

import argparse

from soccerworld._engine.render_defaults import DEFAULT_RENDER_FPS
from soccerworld._engine.timebase import DEFAULT_TIMEBASE

__all__ = ["build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """Build the compatibility-stable demo command line."""

    parser = argparse.ArgumentParser(
        description=(
            "SoccerWorld 팀별 rule/optional learned 행동 정책 경기를 실행하고 "
            "light/rich 모드로 렌더링합니다."
        )
    )
    parser.add_argument(
        "--home",
        default="random",
        help="HOME(적) 스타일 프리셋 또는 random",
    )
    parser.add_argument(
        "--away",
        default="random",
        help="AWAY(청) 스타일 프리셋 또는 random",
    )
    parser.add_argument(
        "--home-policy",
        default="rule",
        help="HOME 행동 정책: rule 또는 package.module:factory::checkpoint",
    )
    parser.add_argument(
        "--away-policy",
        default="rule",
        help="AWAY 행동 정책: rule 또는 package.module:factory::checkpoint",
    )
    parser.add_argument(
        "--learned-deterministic",
        action="store_true",
        help="학습 정책 분포에서 표본을 뽑지 않고 결정론적으로 디코딩",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=60.0,
        help="시뮬레이션 경기 길이(초, 기본 60)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--control-fps",
        "--fps",
        dest="control_fps",
        type=float,
        default=DEFAULT_TIMEBASE.control_fps,
        help="환경 제어 fps(--fps는 호환 alias, 기본 15 Hz)",
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        default=DEFAULT_RENDER_FPS,
        help=(
            "렌더 상태 샘플링 fps. control fps와 독립이며, 서로 다르면 "
            f"물리 궤적을 재표본화(기본 {DEFAULT_RENDER_FPS:g} Hz)"
        ),
    )
    parser.add_argument(
        "--compressed-match",
        action="store_true",
        help=(
            "--seconds를 한 경기 demo로 보고 하프타임을 포함. stamina는 항상 5,400초 기준; "
            "생략하면 실제 물리시간 clip"
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "출력 mp4 경로. 생략 시 ./replays/<unix초>/match.mp4; "
            "replay JSONL도 같은 디렉터리에 저장"
        ),
    )
    parser.add_argument(
        "--mode",
        "--render-mode",
        dest="mode",
        default="light",
        choices=("light", "rich"),
        help="light=빠른 탑다운 / rich=방송형 3D(느림)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="롤아웃과 경기 요약만 실행",
    )
    parser.add_argument(
        "--video-only",
        dest="replay_data",
        action="store_false",
        help="MP4만 저장하고 기본 replay JSONL(events/tracking/metadata)은 생략",
    )
    parser.set_defaults(replay_data=True)
    parser.add_argument(
        "--state-stride",
        type=int,
        default=1,
        help="tracking.jsonl 저장 프레임 간격(기본 1)",
    )
    parser.add_argument(
        "--dynamic-camera",
        action="store_true",
        help="rich 모드에서 공을 따라가는 동적 카메라 사용",
    )
    parser.add_argument(
        "--camera-zoom",
        type=float,
        default=0.66,
        help="rich 동적 카메라 화면 범위 비율(0 초과 1 이하)",
    )
    parser.add_argument(
        "--bench",
        type=int,
        default=9,
        help="팀당 교체 명단 인원(기본 9). 0이면 교체 정책을 끈다",
    )
    parser.add_argument(
        "--manager",
        default="auto",
        choices=("auto", "idle", "off"),
        help=(
            "auto=규칙 감독이 교체와 포메이션을 함께 지휘 / idle=아무 것도 하지 않는 "
            "감독 / off=감독 없이 두 축을 분리 운용(환경 기본값)"
        ),
    )
    parser.add_argument(
        "--no-manager",
        dest="manager",
        action="store_const",
        const="off",
        help="--manager off 의 별칭",
    )
    parser.add_argument(
        "--taker",
        default="auto",
        choices=("auto", "nearest"),
        help=(
            "세트피스 키커 선택. auto=역할 기반 규칙(실측 캘리브) / "
            "nearest=공 최근접(환경 기본값, 호환용)"
        ),
    )
    parser.add_argument(
        "--start-stamina",
        type=float,
        default=1.0,
        help=(
            "킥오프 시점의 장기 stamina(0~1, 기본 1.0). 짧은 데모에서는 아무도 지치지 "
            "않아 교체가 한 번도 일어나지 않는다 — 후반 상황을 눈으로 확인하려면 낮춰라"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="스타일 프리셋 목록을 출력하고 종료",
    )
    return parser
