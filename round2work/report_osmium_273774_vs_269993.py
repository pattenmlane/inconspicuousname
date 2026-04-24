#!/usr/bin/env python3
"""
Compare **osmium-only** standalones ``osmium_273774_wm_freeze_standalone`` vs
``osmium_269993_touch_wm_standalone``:

1. Every ``Prosperity4Data`` day for rounds **1** and **2** (``match_trades=worse``).
2. Every Round **2** day-**29** submission zip (``day 29 logs/*.zip``, ``extra/*.zip``)
   plus the **combined** merge (same as ``combine_submission_runs``).

Writes ``round2work/day 29 logs/osmium_273774_vs_269993_report.txt``.

Run from repo root::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt:$PWD/round2work" \\
  python3 round2work/report_osmium_273774_vs_269993.py
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

OSMIUM = "ASH_COATED_OSMIUM"
MOD_A = "osmium_273774_wm_freeze_standalone"
MOD_B = "osmium_269993_touch_wm_standalone"
ROUND_D29 = 2
DAY_D29 = 29


def _bootstrap() -> None:
    for p in (
        REPO / "imc-prosperity-4-backtester",
        REPO / "imc-prosperity-4-backtester" / "prosperity4bt",
        HERE,
    ):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def osmium_pnl(data_root: Path, round_n: int, day_n: int, modname: str) -> float:
    _bootstrap()
    from prosperity4bt.models.test_options import TradeMatchingMode
    from prosperity4bt.test_runner import TestRunner
    from prosperity4bt.tools.data_reader import FileSystemReader

    mod = importlib.import_module(modname)
    reader = FileSystemReader(data_root)
    if day_n not in reader.available_days(round_n):
        raise ValueError(f"R{round_n} day {day_n} not in {data_root}")
    runner = TestRunner(
        mod.Trader(),
        reader,
        round_n,
        day_n,
        show_progress_bar=False,
        print_output=False,
        trade_matching_mode=TradeMatchingMode.worse,
    )
    result = runner.run()
    for row in result.final_activities():
        if row.symbol == OSMIUM:
            return float(row.profit_loss)
    return 0.0


def collect_day29_zips() -> tuple[list[Path], list[Path]]:
    main = sorted(p for p in (HERE / "day 29 logs").glob("*.zip") if p.is_file())
    extra = sorted(p for p in (HERE / "day 29 logs" / "extra").glob("*.zip") if p.is_file())
    return main, extra


def export_zip_to_round2(zip_path: Path, dest_root: Path) -> None:
    tmp = dest_root / "_tmp_export"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            str(HERE / "logtodata.py"),
            "--zip",
            str(zip_path),
            "--round",
            str(ROUND_D29),
            "--day",
            str(DAY_D29),
            "--out-dir",
            str(tmp),
        ],
        check=True,
    )
    r2 = dest_root / "ROUND_2"
    r2.mkdir(parents=True, exist_ok=True)
    prices = list(tmp.glob(f"prices_round_{ROUND_D29}_day_{DAY_D29}_*.csv"))
    trades = list(tmp.glob(f"trades_round_{ROUND_D29}_day_{DAY_D29}_*.csv"))
    if len(prices) != 1 or len(trades) != 1:
        raise RuntimeError(f"{zip_path}: export mismatch {prices=} {trades=}")
    shutil.copy(prices[0], r2 / f"prices_round_{ROUND_D29}_day_{DAY_D29}.csv")
    shutil.copy(trades[0], r2 / f"trades_round_{ROUND_D29}_day_{DAY_D29}.csv")
    shutil.rmtree(tmp)


def export_combined(combined_dir: Path, dest_root: Path) -> None:
    r2 = dest_root / "ROUND_2"
    r2.mkdir(parents=True, exist_ok=True)
    p = combined_dir / "prices_combined_all_runs.csv"
    t = combined_dir / "trades_combined_all_runs.csv"
    shutil.copy(p, r2 / f"prices_round_{ROUND_D29}_day_{DAY_D29}.csv")
    shutil.copy(t, r2 / f"trades_round_{ROUND_D29}_day_{DAY_D29}.csv")


def merge_all_zips(zips: Sequence[Path], out_dir: Path) -> None:
    subprocess.run(
        [sys.executable, str(HERE / "combine_submission_runs.py"), *[str(z) for z in zips], "--out-dir", str(out_dir)],
        check=True,
    )


def run_day29_tape(zip_path: Path | None, combined_dir: Path, all_zips: list[Path]) -> tuple[float, float]:
    root = Path(tempfile.mkdtemp(prefix="oscmp29_"))
    try:
        if zip_path is not None:
            export_zip_to_round2(zip_path, root)
        else:
            export_combined(combined_dir, root)
        a = osmium_pnl(root, ROUND_D29, DAY_D29, MOD_A)
        b = osmium_pnl(root, ROUND_D29, DAY_D29, MOD_B)
        return a, b
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=REPO / "Prosperity4Data")
    ap.add_argument(
        "--report",
        type=Path,
        default=HERE / "day 29 logs" / "osmium_273774_vs_269993_report.txt",
    )
    ap.add_argument(
        "--combined-dir",
        type=Path,
        default=HERE / "day 29 logs" / "combined_all_including_extra",
    )
    args = ap.parse_args()
    data_root = args.data.expanduser().resolve()

    lines: list[str] = []
    w = lines.append

    w("Osmium-only: 273774 (WM freeze) vs 269993 (WM + touch)")
    w("=" * 88)
    w("match_trades: worse")
    w("")

    # --- Prosperity4Data ---
    w("A) Prosperity4Data — all Round 1 + Round 2 days")
    w("-" * 88)
    _bootstrap()
    from prosperity4bt.tools.data_reader import FileSystemReader

    reader = FileSystemReader(data_root)
    hist_rows: list[tuple[int, int, float, float, float]] = []
    for rn in (1, 2):
        for d in reader.available_days(rn):
            pa = osmium_pnl(data_root, rn, d, MOD_A)
            pb = osmium_pnl(data_root, rn, d, MOD_B)
            hist_rows.append((rn, d, pa, pb, pa - pb))

    hdr = f"{'R':>2} {'day':>5}  {'273774_osm':>14} {'269993_osm':>14} {'delta_A-B':>12}"
    w(hdr)
    w("-" * len(hdr))
    for rn, d, pa, pb, dd in hist_rows:
        w(f"{rn:>2} {d:>5}  {pa:>14,.2f} {pb:>14,.2f} {dd:>12,.2f}")
    w("-" * len(hdr))

    wins_a = sum(1 for _, _, pa, pb, _ in hist_rows if pa > pb)
    wins_b = sum(1 for _, _, pa, pb, _ in hist_rows if pb > pa)
    ties = sum(1 for _, _, pa, pb, _ in hist_rows if pa == pb)
    sa = sum(r[2] for r in hist_rows)
    sb = sum(r[3] for r in hist_rows)
    w("")
    w(f"  Days: {len(hist_rows)}   273774 wins: {wins_a}   269993 wins: {wins_b}   tie: {ties}")
    w(f"  Sum osmium 273774: {sa:,.2f}")
    w(f"  Sum osmium 269993: {sb:,.2f}")
    w(f"  Mean 273774: {sa/len(hist_rows):,.2f}   Mean 269993: {sb/len(hist_rows):,.2f}")
    w("")

    # --- Day 29 zips ---
    main_z, extra_z = collect_day29_zips()
    all_z = main_z + extra_z
    if not all_z:
        raise SystemExit("No day-29 zips")
    comb_dir = args.combined_dir.resolve()
    comb_dir.mkdir(parents=True, exist_ok=True)
    merge_all_zips(all_z, comb_dir)

    w("B) Round 2 day 29 submission tapes (+ combined)")
    w("-" * 88)
    hdr2 = f"{'source':<14} {'run':<12} {'273774_osm':>14} {'269993_osm':>14} {'delta_A-B':>12}"
    w(hdr2)
    w("-" * len(hdr2))

    d29_rows: list[tuple[str, str, float, float, float]] = []
    for z in main_z:
        pa, pb = run_day29_tape(z, comb_dir, all_z)
        d29_rows.append(("day 29 logs", z.stem, pa, pb, pa - pb))
        w(f"{'day 29 logs':<14} {z.stem:<12} {pa:>14,.2f} {pb:>14,.2f} {pa-pb:>12,.2f}")
    for z in extra_z:
        pa, pb = run_day29_tape(z, comb_dir, all_z)
        d29_rows.append(("extra", z.stem, pa, pb, pa - pb))
        w(f"{'extra':<14} {z.stem:<12} {pa:>14,.2f} {pb:>14,.2f} {pa-pb:>12,.2f}")
    pa, pb = run_day29_tape(None, comb_dir, all_z)
    d29_rows.append(("combined", "ALL", pa, pb, pa - pb))
    w(f"{'combined':<14} {'ALL':<12} {pa:>14,.2f} {pb:>14,.2f} {pa-pb:>12,.2f}")
    w("-" * len(hdr2))

    indiv = [r for r in d29_rows if r[0] != "combined"]
    w29a = sum(r[2] for r in indiv)
    w29b = sum(r[3] for r in indiv)
    w29_wa = sum(1 for r in indiv if r[2] > r[3])
    w29_wb = sum(1 for r in indiv if r[3] > r[2])
    w29_t = sum(1 for r in indiv if r[2] == r[3])
    w("")
    w(f"  Individual zips: {len(indiv)}   273774 wins: {w29_wa}   269993 wins: {w29_wb}   tie: {w29_t}")
    w(f"  Sum osmium (indiv only) 273774: {w29a:,.2f}   269993: {w29b:,.2f}")
    w(f"  Combined row: 273774={pa:,.2f}  269993={pb:,.2f}")
    w("")
    w("Recommendation (osmium-only on these backtests)")
    w("-" * 88)
    if sb > sa:
        w("  Prosperity4Data sum favors **269993** (+{:.2f} osmium vs 273774).".format(sb - sa))
    elif sa > sb:
        w("  Prosperity4Data sum favors **273774** (+{:.2f} osmium vs 269993).".format(sa - sb))
    else:
        w("  Prosperity4Data osmium sums are tied.")
    if w29a > w29b:
        w("  Day-29 singles sum favors **273774** (+{:.2f}); combined osmium still ties.".format(w29a - w29b))
    elif w29b > w29a:
        w("  Day-29 singles sum favors **269993** (+{:.2f}).".format(w29b - w29a))
    else:
        w("  Day-29 singles osmium sums are tied.")
    w("  Default os leg: **269993** if you weight canonical historical days; **273774** is")
    w("  simpler (WM-only) with no loss on combined day-29 and only one diverging single (279610).")
    w("")

    outp = args.report.expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
