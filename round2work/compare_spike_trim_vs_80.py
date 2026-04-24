#!/usr/bin/env python3
"""Compare pepper_80_spike_trim_rebuy vs pepper_80_0_long_only on Prosperity4Data R1+R2."""

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
    import importlib

    import pepper_80_0_long_only_standalone as m80
    import pepper_80_spike_trim_rebuy as msp

    dr = FileSystemReader(DATA)
    mm = TradeMatchingMode.worse
    rows = []
    tot0 = tot1 = 0.0
    for r, days in ((1, dr.available_days(1)), (2, dr.available_days(2))):
        for d in days:
            importlib.reload(msp)
            msp.SPIKE_SIGNALS = 0
            a = run_day(m80.Trader, dr, r, d, mm)
            b = run_day(msp.Trader, dr, r, d, mm)
            rows.append((r, d, a, b, b - a, msp.SPIKE_SIGNALS))
            tot0 += a
            tot1 += b
    print(f"match_trades={mm.value}  data={DATA}")
    print(f"{'R':>2} {'day':>5}  {'80/0':>14}  {'80 spike-trim':>14}  {'trim-80':>10}  {'spikes':>7}")
    print("-" * 62)
    for r, d, a, b, dd, ns in rows:
        print(f"{r:>2} {d:>5}  {a:>14,.2f}  {b:>14,.2f}  {dd:>10,.2f}  {ns:>7}")
    print("-" * 62)
    print(f"{'**':>2} {'all':>5}  {tot0:>14,.2f}  {tot1:>14,.2f}  {tot1 - tot0:>10,.2f}")


if __name__ == "__main__":
    main()
