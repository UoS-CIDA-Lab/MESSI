"""Command-line entry point for verified FootballWorld match reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from footballworld.analysis.dataset import MatchDataset
from footballworld.analysis.report import write_match_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate JSON metrics and an interactive HTML match report."
    )
    parser.add_argument("replay", type=Path, help="published replay directory")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument(
        "--allow-diagnostic",
        action="store_true",
        help="allow a partial or non-authoritative replay and label it clearly",
    )
    parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help="skip potentially expensive artifact hash checks and record a warning",
    )
    parser.add_argument(
        "--policy-reference",
        type=Path,
        help="validated aggregate guideline to compare against current match facts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset = MatchDataset.open(
        args.replay,
        output_name=args.output_name,
        allow_diagnostic=args.allow_diagnostic,
        verify_hashes=not args.skip_hash_verification,
    )
    output_dir = args.output_dir or args.replay / "report"
    html_path, json_path = write_match_report(
        dataset,
        output_dir,
        policy_reference=args.policy_reference,
    )
    print(
        json.dumps(
            {
                "html": str(html_path.resolve()),
                "json": str(json_path.resolve()),
                "authoritative": dataset.authoritative,
                "warnings": dataset.warnings,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
