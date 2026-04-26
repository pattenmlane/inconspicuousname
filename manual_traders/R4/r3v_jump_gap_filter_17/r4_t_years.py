"""Round 4 TTE: CSV day 1→4 DTE at open, 2→3, 3→2 (see round4work/round4description.txt example)."""
from __future__ import annotations


def dte_from_csv_day(day: int) -> int:
    return 5 - int(day)


def intraday_progress(timestamp: int) -> float:
    return (int(timestamp) // 100) / 10_000.0


def dte_effective(day: int, timestamp: int) -> float:
    return max(float(dte_from_csv_day(day)) - intraday_progress(timestamp), 1e-6)


def t_years_effective(day: int, timestamp: int) -> float:
    return dte_effective(day, timestamp) / 365.0
