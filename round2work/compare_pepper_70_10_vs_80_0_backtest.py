#!/usr/bin/env python3
"""
Backtest three pepper-only strategies on Prosperity4Data (rounds 1 and 2):

- **70/10+slope** — ``pepper_70_10_slope_safeguard_standalone``
- **80/0** — ``pepper_80_0_long_only_standalone`` (aggressive to +80 only)
- **80+sell-MM** — ``pepper_80_long_sellside_mm_standalone`` (aggressive while <80,
  then drift fair + sell-side-only Emerald at/above 80)

Uses ``match_trades=worse``. Pepper PnL from last activity row per day.

Run from repo root::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 round2work/compare_pepper_70_10_vs_80_0_backtest.py
"""

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
    result = runner.run()
    return pepper_pnl_from_result(result)


def main() -> None:
    if not DATA.is_dir():
        raise SystemExit(f"Missing data dir: {DATA}")

    data_reader = FileSystemReader(DATA)
    days_r1 = data_reader.available_days(1)
    days_r2 = data_reader.available_days(2)
    if not days_r1:
        raise SystemExit("No round 1 days found under Prosperity4Data")
    if not days_r2:
        raise SystemExit("No round 2 days found (expected ROUND_2 with prices_round_2_day_*.csv)")

    match_mode = TradeMatchingMode.worse
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import pepper_70_10_slope_safeguard_standalone as m70
    import pepper_80_0_long_only_standalone as m80
    import pepper_80_long_sellside_mm_standalone as m80m

    t70, t80, t80m = m70.Trader, m80.Trader, m80m.Trader

    rows: list[tuple[int, int, float, float, float, float]] = []

    for r, days in ((1, days_r1), (2, days_r2)):
        for d in days:
            p70 = run_day(t70, data_reader, r, d, match_mode)
            p80 = run_day(t80, data_reader, r, d, match_mode)
            p80m = run_day(t80m, data_reader, r, d, match_mode)
            rows.append((r, d, p70, p80, p80m))

    print("match_trades:", match_mode.value)
    print(f"data: {DATA}")
    print()
    hdr = (
        f"{'R':>2} {'day':>5}  {'70/10+slope':>14}  {'80/0 aggr':>14}  "
        f"{'80+sell-MM':>14}  {'hyb-70':>10}  {'hyb-80':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    tot70 = tot80 = tot80m = 0.0
    for r, d, p70, p80, p80m in rows:
        print(
            f"{r:>2} {d:>5}  {p70:>14,.2f}  {p80:>14,.2f}  {p80m:>14,.2f}  "
            f"{p80m - p70:>10,.2f}  {p80m - p80:>10,.2f}"
        )
        tot70 += p70
        tot80 += p80
        tot80m += p80m
    print("-" * len(hdr))
    print(
        f"{'**':>2} {'all':>5}  {tot70:>14,.2f}  {tot80:>14,.2f}  {tot80m:>14,.2f}  "
        f"{tot80m - tot70:>10,.2f}  {tot80m - tot80:>10,.2f}"
    )
    print()
    print("hyb-70 / hyb-80: 80+sell-MM minus 70/10 or minus 80/0 (positive => hybrid best of the pair).")


if __name__ == "__main__":
    main()
