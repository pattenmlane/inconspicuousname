#!/usr/bin/env python3
"""Compare pepper 70/10+slope vs 80/0 on Prosperity4Data (R1+R2), match_trades=worse."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "Prosperity4Data"
HERE = Path(__file__).resolve().parent
PEPPER = "INTARIAN_PEPPER_ROOT"

for p in (
    REPO / "imc-prosperity-4-backtester",
    REPO / "imc-prosperity-4-backtester" / "prosperity4bt",
):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from prosperity4bt.models.test_options import TradeMatchingMode  # noqa: E402
from prosperity4bt.test_runner import TestRunner  # noqa: E402
from prosperity4bt.tools.data_reader import FileSystemReader  # noqa: E402


def pepper_pnl(result) -> float:
    for row in result.final_activities():
        if row.symbol == PEPPER:
            return float(row.profit_loss)
    return 0.0


def run_day(cls, dr, r, d, mm):
    return pepper_pnl(
        TestRunner(cls(), dr, r, d, show_progress_bar=False, print_output=False, trade_matching_mode=mm).run()
    )


def main() -> None:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import pepper_70_10_slope_safeguard_standalone as m70
    import pepper_80_0_long_only_standalone as m80

    dr = FileSystemReader(DATA)
    mm = TradeMatchingMode.worse
    t70, t80 = m70.Trader, m80.Trader
    a = b = 0.0
    print(f"match_trades={mm.value}  data={DATA}")
    print(f"{'R':>2} {'day':>5}  {'70/10+slope':>14}  {'80/0':>14}  {'70-80':>12}")
    print("-" * 52)
    for r, days in ((1, dr.available_days(1)), (2, dr.available_days(2))):
        for d in days:
            p0 = run_day(t70, dr, r, d, mm)
            p1 = run_day(t80, dr, r, d, mm)
            print(f"{r:>2} {d:>5}  {p0:>14,.2f}  {p1:>14,.2f}  {p0 - p1:>12,.2f}")
            a += p0
            b += p1
    print("-" * 52)
    print(f"{'**':>2} {'all':>5}  {a:>14,.2f}  {b:>14,.2f}  {a - b:>12,.2f}")


if __name__ == "__main__":
    main()
