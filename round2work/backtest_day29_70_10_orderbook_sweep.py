#!/usr/bin/env python3
"""
Backtest ``pepper_70_10_slope_safeguard_standalone`` (``match_trades=worse``):

**A.** Round 2 day-29 submission tapes — each zip under ``day 29 logs/`` and
    ``day 29 logs/extra/``, plus a **combined** merge of all zips
    (``combine_submission_runs.py``).

**B.** All historical days under ``Prosperity4Data/`` discovered by
    ``FileSystemReader`` (Round 1 and Round 2 CSVs matching
    ``prices_round_{n}_day_{d}.csv``).

Writes one report (default: ``day 29 logs/backtest_70_10_day29_orderbook_report.txt``).

Run from repo root::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt:$PWD/round2work" \\
  python3 round2work/backtest_day29_70_10_orderbook_sweep.py
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

PEPPER = "INTARIAN_PEPPER_ROOT"
ROUND = 2
DAY = 29


def _bootstrap_sys_path() -> None:
    for p in (
        REPO / "imc-prosperity-4-backtester",
        REPO / "imc-prosperity-4-backtester" / "prosperity4bt",
        HERE,
    ):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def collect_zip_paths() -> tuple[list[Path], list[Path]]:
    main = sorted(p for p in (HERE / "day 29 logs").glob("*.zip") if p.is_file())
    extra = sorted(p for p in (HERE / "day 29 logs" / "extra").glob("*.zip") if p.is_file())
    return main, extra


def export_zip_to_round2(zip_path: Path, dest_root: Path) -> None:
    """Run logtodata then place files under dest_root/ROUND_2/*."""
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
            str(ROUND),
            "--day",
            str(DAY),
            "--out-dir",
            str(tmp),
        ],
        check=True,
    )
    r2 = dest_root / "ROUND_2"
    r2.mkdir(parents=True, exist_ok=True)
    prices = list(tmp.glob(f"prices_round_{ROUND}_day_{DAY}_*.csv"))
    trades = list(tmp.glob(f"trades_round_{ROUND}_day_{DAY}_*.csv"))
    if len(prices) != 1 or len(trades) != 1:
        raise RuntimeError(f"{zip_path}: expected one prices/trades export, got {prices=} {trades=}")
    shutil.copy(prices[0], r2 / f"prices_round_{ROUND}_day_{DAY}.csv")
    shutil.copy(trades[0], r2 / f"trades_round_{ROUND}_day_{DAY}.csv")
    shutil.rmtree(tmp)


def export_combined_csvs(combined_dir: Path, dest_root: Path) -> None:
    r2 = dest_root / "ROUND_2"
    r2.mkdir(parents=True, exist_ok=True)
    p = combined_dir / "prices_combined_all_runs.csv"
    t = combined_dir / "trades_combined_all_runs.csv"
    if not p.is_file() or not t.is_file():
        raise FileNotFoundError(f"Missing combined exports under {combined_dir}")
    shutil.copy(p, r2 / f"prices_round_{ROUND}_day_{DAY}.csv")
    shutil.copy(t, r2 / f"trades_round_{ROUND}_day_{DAY}.csv")


def merge_all_zips(zips: Sequence[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(HERE / "combine_submission_runs.py"), *[str(z) for z in zips], "--out-dir", str(out_dir)]
    subprocess.run(cmd, check=True)


def count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


@dataclass
class RunResult:
    label: str
    zip_name: str
    bucket: str
    pepper_pnl: float
    total_pnl: float
    n_prices: int
    n_trades: int


def run_70_10_backtest(data_root: Path, round_n: int, day_n: int) -> RunResult:
    """Run pepper 70/10 on ``data_root`` for ``round_n`` / ``day_n`` (FileSystemReader layout)."""
    _bootstrap_sys_path()
    from prosperity4bt.models.test_options import TradeMatchingMode
    from prosperity4bt.test_runner import TestRunner
    from prosperity4bt.tools.data_reader import FileSystemReader

    import pepper_70_10_slope_safeguard_standalone as m70

    reader = FileSystemReader(data_root)
    avail = reader.available_days(round_n)
    if day_n not in avail:
        raise RuntimeError(f"Round {round_n} day {day_n} not in reader (available: {avail})")

    runner = TestRunner(
        m70.Trader(),
        reader,
        round_n,
        day_n,
        show_progress_bar=False,
        print_output=False,
        trade_matching_mode=TradeMatchingMode.worse,
    )
    result = runner.run()
    acts = result.final_activities()
    total = sum(float(a.profit_loss) for a in acts)
    pepper = 0.0
    for a in acts:
        if a.symbol == PEPPER:
            pepper = float(a.profit_loss)
            break

    rdir = reader._round_dir(round_n)
    if rdir is None:
        np_ = nt = 0
    else:
        np_ = count_csv_rows(rdir / f"prices_round_{round_n}_day_{day_n}.csv")
        nt = count_csv_rows(rdir / f"trades_round_{round_n}_day_{day_n}.csv")

    return RunResult("", "", "", pepper, total, np_, nt)


def backtest_one(data_root: Path) -> RunResult:
    return run_70_10_backtest(data_root, ROUND, DAY)


def format_money(x: float) -> str:
    return f"{x:,.2f}"


def scaled_pepper(pepper: float, n_prices: int, ref_rows: int) -> float | None:
    """Rough ``if this tape had ref_rows price rows`` extrapolation (linear in row count)."""
    if n_prices <= 0 or ref_rows <= 0:
        return None
    return pepper * (ref_rows / n_prices)


def format_scaled(pepper: float, n_prices: int, ref_rows: int) -> str:
    s = scaled_pepper(pepper, n_prices, ref_rows)
    return format_money(s) if s is not None else "         n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--report",
        type=Path,
        default=HERE / "day 29 logs" / "backtest_70_10_day29_orderbook_report.txt",
        help="Output report path",
    )
    ap.add_argument(
        "--combined-out",
        type=Path,
        default=HERE / "day 29 logs" / "combined_all_including_extra",
        help="Directory for merged prices/trades (regenerated each run)",
    )
    ap.add_argument(
        "--prosperity-data",
        type=Path,
        default=REPO / "Prosperity4Data",
        help="Root with ROUND1/ROUND_2 prices CSVs (skip section if missing)",
    )
    ap.add_argument(
        "--normalize-rows",
        type=int,
        default=20_000,
        metavar="N",
        help="Reference price-row count for ``pepper@N`` column (default 20000)",
    )
    args = ap.parse_args()
    ref_n = max(1, int(args.normalize_rows))

    main_zips, extra_zips = collect_zip_paths()
    all_zips = main_zips + extra_zips
    if not all_zips:
        raise SystemExit(f"No zips under {HERE / 'day 29 logs'}")

    merge_all_zips(all_zips, args.combined_out.resolve())

    results: list[RunResult] = []

    def run_labeled(zip_path: Path | None, bucket: str, label: str) -> None:
        root = Path(tempfile.mkdtemp(prefix="bt29_"))
        try:
            if zip_path is not None:
                export_zip_to_round2(zip_path, root)
                zn = zip_path.name
            else:
                export_combined_csvs(args.combined_out.resolve(), root)
                zn = "(merged CSV)"
            rr = backtest_one(root)
            rr.label = label
            rr.zip_name = zn
            rr.bucket = bucket
            results.append(rr)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    for z in main_zips:
        run_labeled(z, "day 29 logs", z.stem)
    for z in extra_zips:
        run_labeled(z, "extra", z.stem)
    run_labeled(None, "combined", "COMBINED_ALL_RUNS")

    indiv = [r for r in results if r.bucket != "combined"]
    comb = next(r for r in results if r.bucket == "combined")

    pep_vals = [r.pepper_pnl for r in indiv]
    tot_vals = [r.total_pnl for r in indiv]
    avg_pepper = statistics.mean(pep_vals)
    avg_total = statistics.mean(tot_vals)
    scaled_indiv = [scaled_pepper(r.pepper_pnl, r.n_prices, ref_n) for r in indiv]
    scaled_indiv_f = [x for x in scaled_indiv if x is not None]
    avg_scaled_indiv = statistics.mean(scaled_indiv_f) if scaled_indiv_f else 0.0
    comb_scaled = scaled_pepper(comb.pepper_pnl, comb.n_prices, ref_n)
    delta_scaled_comb = (
        (comb_scaled - avg_scaled_indiv) if comb_scaled is not None else float("nan")
    )

    lines: list[str] = []
    w = lines.append
    w("Round 2 — Day 29 order-book / tape sweep")
    w("=" * 72)
    w("")
    w("Strategy: pepper_70_10_slope_safeguard_standalone.Trader")
    w("match_trades: worse")
    w(f"Individual zips: {len(main_zips)} in `day 29 logs/` + {len(extra_zips)} in `extra/`")
    w(f"Combined tape: merge of all {len(all_zips)} zips (combine_submission_runs.py)")
    w(f"Combined data directory: {args.combined_out.resolve()}")
    w("")
    w(f"Length-normalized pepper (``pepper@{ref_n}``) = pepper_pnl × ({ref_n} / price_rows).")
    w("Assumes PnL scales ~linearly with tick count — ballpark only, not a claim about longer runs.")
    w("")

    w("Per-run results (same replay length; different book/trade tape)")
    w("-" * 96)
    hdr = (
        f"{'source':<18} {'run':<12} {'pepper_pnl':>14} {f'pepper@{ref_n}':>14} "
        f"{'total_pnl':>14} {'prices':>8} {'trades':>8}"
    )
    w(hdr)
    w("-" * len(hdr))
    for r in results:
        w(
            f"{r.bucket:<18} {r.label:<12} {format_money(r.pepper_pnl):>14} "
            f"{format_scaled(r.pepper_pnl, r.n_prices, ref_n):>14} "
            f"{format_money(r.total_pnl):>14} {r.n_prices:>8} {r.n_trades:>8}"
        )
    w("-" * len(hdr))
    w("")

    w("Aggregate — individuals only")
    w("-" * 72)
    w(f"  Count:                {len(indiv)}")
    w(f"  Avg pepper PnL:       {format_money(avg_pepper)}")
    w(f"  Avg pepper@{ref_n}:    {format_money(avg_scaled_indiv)}")
    w(f"  Avg total PnL:        {format_money(avg_total)}")
    w(f"  Min / max pepper:     {format_money(min(pep_vals))}  /  {format_money(max(pep_vals))}")
    if len(pep_vals) >= 2:
        w(f"  Stdev pepper:         {statistics.pstdev(pep_vals):,.2f}")
    if len(scaled_indiv_f) >= 2:
        w(f"  Stdev pepper@{ref_n}:  {statistics.pstdev(scaled_indiv_f):,.2f}")
    w("")

    w("Combined vs average individual")
    w("-" * 72)
    w(f"  Combined pepper PnL:           {format_money(comb.pepper_pnl)}")
    w(f"  Combined pepper@{ref_n}:        {format_scaled(comb.pepper_pnl, comb.n_prices, ref_n)}")
    w(f"  Mean individual pepper PnL:  {format_money(avg_pepper)}")
    w(f"  Mean individual pepper@{ref_n}: {format_money(avg_scaled_indiv)}")
    w(
        f"  Delta pepper@{ref_n} (comb − mean indiv): "
        f"{format_money(delta_scaled_comb) if delta_scaled_comb == delta_scaled_comb else 'n/a'}"
    )
    w("")
    w(f"  Combined total PnL:           {format_money(comb.total_pnl)}")
    w(f"  Mean individual total PnL:  {format_money(avg_total)}")
    w(f"  Delta (combined − mean):     {format_money(comb.total_pnl - avg_total)}")
    w("")

    w("Notes")
    w("-" * 72)
    w("  * Pepper PnL is from the last activity row for INTARIAN_PEPPER_ROOT.")
    w("  * Total PnL is the sum of last-row PnL across products in that tape.")
    w("  * The combined tape is not an average of books; it merges cells across")
    w("    runs (majority vote) and unions trades. Performance can differ from")
    w("    the mean of single-run backtests because fills depend on the merged book.")
    w("")

    # --- Prosperity4Data (Round 1 + Round 2 bundled CSVs) ---
    pdata = args.prosperity_data.expanduser().resolve()
    if pdata.is_dir():
        _bootstrap_sys_path()
        from prosperity4bt.tools.data_reader import FileSystemReader

        fs_reader = FileSystemReader(pdata)
        p4d_results: list[tuple[int, int, RunResult]] = []
        for round_n in (1, 2):
            for d in fs_reader.available_days(round_n):
                rr = run_70_10_backtest(pdata, round_n, d)
                rr.bucket = "Prosperity4Data"
                rr.label = f"R{round_n}d{d}"
                rr.zip_name = f"prices_round_{round_n}_day_{d}.csv"
                p4d_results.append((round_n, d, rr))

        w("")
        w("")
        w("Prosperity4Data — same 70/10 strategy on bundled historical tapes")
        w("=" * 72)
        w(f"Data root: {pdata}")
        w("(Days are those with a matching ``prices_round_<n>_day_<d>.csv`` file.)")
        w(f"(Same ``pepper@{ref_n}`` scaling as above.)")
        w("")

        hdr2 = (
            f"{'round':>7} {'day':>6} {'pepper_pnl':>14} {f'pepper@{ref_n}':>14} "
            f"{'total_pnl':>14} {'prices':>8} {'trades':>8}"
        )
        w(hdr2)
        w("-" * len(hdr2))
        for round_n, d, r in p4d_results:
            w(
                f"{round_n:>7} {d:>6} {format_money(r.pepper_pnl):>14} "
                f"{format_scaled(r.pepper_pnl, r.n_prices, ref_n):>14} "
                f"{format_money(r.total_pnl):>14} {r.n_prices:>8} {r.n_trades:>8}"
            )
        w("-" * len(hdr2))

        if p4d_results:
            pep_p4d = [r.pepper_pnl for _, _, r in p4d_results]
            tot_p4d = [r.total_pnl for _, _, r in p4d_results]
            scaled_p4d = [
                scaled_pepper(r.pepper_pnl, r.n_prices, ref_n) for _, _, r in p4d_results
            ]
            scaled_p4d_f = [x for x in scaled_p4d if x is not None]
            w("")
            w("Aggregate — Prosperity4Data days")
            w("-" * 72)
            w(f"  Day count:            {len(p4d_results)}")
            w(f"  Sum pepper PnL (raw): {format_money(sum(pep_p4d))}")
            w(f"  Sum total PnL:        {format_money(sum(tot_p4d))}")
            w(f"  Avg pepper / day:     {format_money(statistics.mean(pep_p4d))}")
            w(f"  Avg pepper@{ref_n} / day: {format_money(statistics.mean(scaled_p4d_f))}")
            w(f"  Min / max pepper:     {format_money(min(pep_p4d))}  /  {format_money(max(pep_p4d))}")
            if len(pep_p4d) >= 2:
                w(f"  Stdev pepper (days):  {statistics.pstdev(pep_p4d):,.2f}")
            if len(scaled_p4d_f) >= 2:
                w(f"  Stdev pepper@{ref_n}:    {statistics.pstdev(scaled_p4d_f):,.2f}")
            r2_only = [r for rn, _, r in p4d_results if rn == 2]
            if r2_only:
                p2 = [r.pepper_pnl for r in r2_only]
                s2 = [scaled_pepper(r.pepper_pnl, r.n_prices, ref_n) for r in r2_only]
                s2f = [x for x in s2 if x is not None]
                w("")
                w("  Round 2 only (Prosperity4Data days −1, 0, 1 — different tapes than day-29 zips)")
                w("-" * 72)
                w(f"    Count:              {len(r2_only)}")
                w(f"    Sum pepper PnL:     {format_money(sum(p2))}")
                w(f"    Avg pepper / day:   {format_money(statistics.mean(p2))}")
                if s2f:
                    w(f"    Avg pepper@{ref_n}/day: {format_money(statistics.mean(s2f))}")
        w("")
        w("Notes")
        w("-" * 72)
        w("  * Same trader and match_trades=worse as the day-29 zip section above.")
        w("  * Files such as ``prices_round_1_day_19_enriched.csv`` are ignored unless")
        w("    renamed to match the engine's ``..._day_<int>.csv`` pattern.")
    else:
        w("")
        w("")
        w("Prosperity4Data section skipped (directory not found).")
        w(f"  Expected: {pdata}")

    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
