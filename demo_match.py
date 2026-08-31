#!/usr/bin/env python3
"""Run one SoccerWorld match. Example: ``python demo_match.py --seconds 30``.

By default outputs go to ``./replays/<unix-seconds>/``. Use ``--out PATH`` to choose the MP4
location; replay JSONL files are written beside it unless ``--video-only`` is supplied."""

from soccerworld.demo.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
