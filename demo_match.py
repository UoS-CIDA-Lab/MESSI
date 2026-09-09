#!/usr/bin/env python3
"""Run FootballWorld's reproducible full-match demo.

The implementation remains in ``examples.render_full_match`` so there is one
capture/render path and one CLI contract to maintain. The small ``src`` path
bootstrap makes a source checkout runnable before an editable install.
"""

import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
for import_root in (ROOT, SOURCE):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

main = import_module("examples.render_full_match").main


if __name__ == "__main__":
    raise SystemExit(main())
