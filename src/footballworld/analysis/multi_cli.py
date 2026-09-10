"""Command-line entry point for FootballWorld tactical matrix reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from footballworld.analysis.multi_report import write_tactical_matrix_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate verified single-match reports into a tactical matrix report."
    )
    parser.add_argument("matrix_summary", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    html_path, json_path = write_tactical_matrix_report(
        args.matrix_summary, args.output_dir
    )
    print(
        json.dumps(
            {"html": str(html_path.resolve()), "json": str(json_path.resolve())},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
