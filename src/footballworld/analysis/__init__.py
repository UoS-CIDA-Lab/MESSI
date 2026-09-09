"""Host-only analysis and report generation for published match replays."""

from footballworld.analysis.dataset import MatchDataset
from footballworld.analysis.metrics import build_match_report
from footballworld.analysis.report import write_match_report

__all__ = ["MatchDataset", "build_match_report", "write_match_report"]
