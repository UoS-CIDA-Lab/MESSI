"""Numerically lazy bootstrap for the packaged demo."""

from __future__ import annotations

from soccerworld.demo.arguments import build_parser


def main(argv: list[str] | None = None) -> int:
    """Parse lightweight flags, then initialize the numerical runtime."""

    parser = build_parser()
    args = parser.parse_args(argv)

    from soccerworld.runtime import enable_compilation_cache

    enable_compilation_cache()

    from soccerworld.demo.cli import run

    return run(parser, args)


if __name__ == "__main__":
    raise SystemExit(main())
