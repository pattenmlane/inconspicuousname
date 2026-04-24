#!/usr/bin/env python3
"""
Sweep ``MM_PASSIVE_SELL_CLIP`` on ``pepper_80_long_sellside_mm_standalone`` (rounds 1+2, Prosperity4Data).

Run from repo root::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 round2work/sweep_hybrid_mm_clip.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "Prosperity4Data"
HERE = Path(__file__).resolve().parent

PEPPER = "INTARIAN_PEPPER_ROOT"
CLIPS = (20, 15, 10, 5, 3, 0)

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


def pepper_pnl_from_result(result) -> float:
    for row in result.final_activities():
        if row.symbol == PEPPER:
            return float(row.profit_loss)
    return 0.0


def run_day(trader_cls, data_reader, round_num: int, day_num: int, match_mode: TradeMatchingMode) -> float:
    runner = TestRunner(
        trader_cls(),
        data_reader,
        round_num,
        day_num,
        show_progress_bar=False,
        print_output=False,
        trade_matching_mode=match_mode,
    )
    return pepper_pnl_from_result(runner.run())


def total_pepper(trader_cls, data_reader, match_mode: TradeMatchingMode) -> float:
    t = 0.0
    for r, days in ((1, data_reader.available_days(1)), (2, data_reader.available_days(2))):
        for d in days:
            t += run_day(trader_cls, data_reader, r, d, match_mode)
    return t


def main() -> None:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import pepper_70_10_slope_safeguard_standalone as m70
    import pepper_80_0_long_only_standalone as m80
    import pepper_80_long_sellside_mm_standalone as m80m

    if not DATA.is_dir():
        raise SystemExit(f"Missing {DATA}")

    dr = FileSystemReader(DATA)
    if not dr.available_days(2):
        raise SystemExit("No round 2 days")

    mm = TradeMatchingMode.worse
    ref70 = total_pepper(m70.Trader, dr, mm)
    ref80 = total_pepper(m80.Trader, dr, mm)

    print(f"match_trades={mm.value}  data={DATA}")
    print(f"reference totals (all R1+R2 days): 70/10+slope={ref70:,.2f}  80/0={ref80:,.2f}")
    print()
    print(f"{'clip':>6}  {'80+sell-MM total':>18}  {'vs 70/10':>12}  {'vs 80/0':>12}")
    print("-" * 54)
    for clip in CLIPS:
        m80m.MM_PASSIVE_SELL_CLIP = int(clip)
        tot = total_pepper(m80m.Trader, dr, mm)
        print(f"{clip:>6}  {tot:>18,.2f}  {tot - ref70:>12,.2f}  {tot - ref80:>12,.2f}")


if __name__ == "__main__":
    main()
