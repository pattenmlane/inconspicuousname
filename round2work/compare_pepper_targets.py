#!/usr/bin/env python3
"""
Compare pepper drift+MM+slope with different (target / slack) pairs.

Slack = ``PEPPER_POSITION_LIMIT - PEPPER_TARGET_LONG`` with limit fixed at **80**.

- **70/10** — target 70, slack 10 (defaults in ``pepper_70_10_slope_safeguard_standalone``)
- **75/5** — target 75, slack 5
- **72/8** — target 72, slack 8

Run from repo root::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 round2work/compare_pepper_targets.py
"""

from __future__ import annotations

import importlib
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


def run_day(trader_cls, dr, r, d, mm):
    return pepper_pnl(
        TestRunner(trader_cls(), dr, r, d, show_progress_bar=False, print_output=False, trade_matching_mode=mm).run()
    )


def main() -> None:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    dr = FileSystemReader(DATA)
    mm = TradeMatchingMode.worse

    configs = (
        ("70/10", 70),
        ("75/5", 75),
        ("72/8", 72),
    )

    rows: list[tuple[int, int, float, float, float]] = []
    totals = [0.0, 0.0, 0.0]
    modname = "pepper_70_10_slope_safeguard_standalone"

    for r, days in ((1, dr.available_days(1)), (2, dr.available_days(2))):
        for d in days:
            pnls: list[float] = []
            for i, (_, tgt) in enumerate(configs):
                m = importlib.reload(importlib.import_module(modname))
                m.PEPPER_TARGET_LONG = int(tgt)
                m.PEPPER_POSITION_LIMIT = 80
                p = run_day(m.Trader, dr, r, d, mm)
                pnls.append(p)
                totals[i] += p
            rows.append((r, d, *pnls))

    print(f"match_trades={mm.value}  limit=80  data={DATA}")
    print()
    h = f"{'R':>2} {'day':>5}  {'70/10':>14}  {'75/5':>14}  {'72/8':>14}"
    print(h)
    print("-" * len(h))
    for r, d, p0, p1, p2 in rows:
        print(f"{r:>2} {d:>5}  {p0:>14,.2f}  {p1:>14,.2f}  {p2:>14,.2f}")
    print("-" * len(h))
    print(f"{'**':>2} {'all':>5}  {totals[0]:>14,.2f}  {totals[1]:>14,.2f}  {totals[2]:>14,.2f}")


if __name__ == "__main__":
    main()
