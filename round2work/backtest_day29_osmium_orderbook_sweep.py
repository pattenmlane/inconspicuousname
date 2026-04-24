#!/usr/bin/env python3
"""
Day-29 order-book value sweep for **osmium-only** standalone strategies (same
layout as ``backtest_day29_70_10_orderbook_sweep.py`` for pepper):

* Each zip under ``day 29 logs/`` and ``extra/``, plus **combined** merge of all.
* For each strategy module, ``TestRunner`` on Round 2 day 29, ``match_trades=worse``.
* Report **osmium PnL** and ``osmium@N`` = ``osmium_pnl * (N / price_rows)`` (default N=20000).

Strategies (see ``round2work/osmium_*_standalone.py``):

* **273774** — wall mid + WM spike freeze.
* **269993** — WM spike freeze + touch stress freeze + width boost.

Does **not** re-run Prosperity4Data historical days (day-29 tapes only).

Run from repo root::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt:$PWD/round2work" \\
  python3 round2work/backtest_day29_osmium_orderbook_sweep.py
"""

from __future__ import annotations

import argparse
import importlib
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

OSMIUM = "ASH_COATED_OSMIUM"
ROUND = 2
DAY = 29

TRADER_MODULES: tuple[tuple[str, str], ...] = (
    ("273774 osmium (WM spike freeze)", "osmium_273774_wm_freeze_standalone"),
    ("269993 osmium (WM + touch freeze)", "osmium_269993_touch_wm_standalone"),
)


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
class OsmiumRunResult:
    label: str
    zip_name: str
    bucket: str
    osmium_pnl: float
    total_pnl: float
    n_prices: int
    n_trades: int


def run_osmium_trader(data_root: Path, trader_cls: type, round_n: int, day_n: int) -> OsmiumRunResult:
    _bootstrap_sys_path()
    from prosperity4bt.models.test_options import TradeMatchingMode
    from prosperity4bt.test_runner import TestRunner
    from prosperity4bt.tools.data_reader import FileSystemReader

    reader = FileSystemReader(data_root)
    avail = reader.available_days(round_n)
    if day_n not in avail:
        raise RuntimeError(f"Round {round_n} day {day_n} not in reader (available: {avail})")

    runner = TestRunner(
        trader_cls(),
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
    osm = 0.0
    for a in acts:
        if a.symbol == OSMIUM:
            osm = float(a.profit_loss)
            break

    rdir = reader._round_dir(round_n)
    if rdir is None:
        np_ = nt = 0
    else:
        np_ = count_csv_rows(rdir / f"prices_round_{round_n}_day_{day_n}.csv")
        nt = count_csv_rows(rdir / f"trades_round_{round_n}_day_{day_n}.csv")

    return OsmiumRunResult("", "", "", osm, total, np_, nt)


def format_money(x: float) -> str:
    return f"{x:,.2f}"


def scaled_osmium(osm: float, n_prices: int, ref_rows: int) -> float | None:
    if n_prices <= 0 or ref_rows <= 0:
        return None
    return osm * (ref_rows / n_prices)


def format_scaled(osm: float, n_prices: int, ref_rows: int) -> str:
    s = scaled_osmium(osm, n_prices, ref_rows)
    return format_money(s) if s is not None else "         n/a"


def section_lines(
    strategy_title: str,
    module_name: str,
    results: list[OsmiumRunResult],
    indiv: list[OsmiumRunResult],
    comb: OsmiumRunResult,
    ref_n: int,
) -> list[str]:
    lines: list[str] = []
    w = lines.append
    w(strategy_title)
    w("=" * min(96, max(72, len(strategy_title) + 4)))
    w(f"Module: {module_name}")
    w("")
    w(f"Length-normalized (``osmium@{ref_n}``) = osmium_pnl × ({ref_n} / price_rows).")
    w("Assumes linear scaling in row count — ballpark only.")
    w("")

    hdr = (
        f"{'source':<18} {'run':<12} {'osmium_pnl':>14} {f'osm@{ref_n}':>14} "
        f"{'total_pnl':>14} {'prices':>8} {'trades':>8}"
    )
    w(hdr)
    w("-" * len(hdr))
    for r in results:
        w(
            f"{r.bucket:<18} {r.label:<12} {format_money(r.osmium_pnl):>14} "
            f"{format_scaled(r.osmium_pnl, r.n_prices, ref_n):>14} "
            f"{format_money(r.total_pnl):>14} {r.n_prices:>8} {r.n_trades:>8}"
        )
    w("-" * len(hdr))

    pep_vals = [r.osmium_pnl for r in indiv]
    scaled_f = [scaled_osmium(r.osmium_pnl, r.n_prices, ref_n) for r in indiv]
    scaled_f = [x for x in scaled_f if x is not None]
    avg_osm = statistics.mean(pep_vals)
    avg_scaled = statistics.mean(scaled_f) if scaled_f else 0.0
    comb_scaled = scaled_osmium(comb.osmium_pnl, comb.n_prices, ref_n)
    d_scaled = (comb_scaled - avg_scaled) if comb_scaled is not None else float("nan")

    w("")
    w("Aggregate — individuals only")
    w("-" * 72)
    w(f"  Count:                {len(indiv)}")
    w(f"  Avg osmium PnL:       {format_money(avg_osm)}")
    w(f"  Avg osmium@{ref_n}:   {format_money(avg_scaled)}")
    w(f"  Min / max osmium:     {format_money(min(pep_vals))}  /  {format_money(max(pep_vals))}")
    if len(pep_vals) >= 2:
        w(f"  Stdev osmium:         {statistics.pstdev(pep_vals):,.2f}")
    if len(scaled_f) >= 2:
        w(f"  Stdev osm@{ref_n}:     {statistics.pstdev(scaled_f):,.2f}")

    w("")
    w("Combined vs average individual")
    w("-" * 72)
    w(f"  Combined osmium PnL:          {format_money(comb.osmium_pnl)}")
    w(f"  Combined osmium@{ref_n}:       {format_scaled(comb.osmium_pnl, comb.n_prices, ref_n)}")
    w(f"  Mean individual osmium PnL: {format_money(avg_osm)}")
    w(f"  Mean individual osm@{ref_n}: {format_money(avg_scaled)}")
    w(
        f"  Delta osm@{ref_n} (comb − mean indiv): "
        f"{format_money(d_scaled) if d_scaled == d_scaled else 'n/a'}"
    )
    w("")
    w(f"  Combined total PnL:          {format_money(comb.total_pnl)}")
    w(f"  Mean individual total PnL: {format_money(statistics.mean([r.total_pnl for r in indiv]))}")
    w("")
    w("Notes")
    w("-" * 72)
    w("  * Osmium PnL from last activity row for ASH_COATED_OSMIUM.")
    w("  * These traders do not quote pepper; total usually equals osmium.")
    w("  * Combined tape: merge of all zips (see combine_submission_runs.py).")
    w("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--report",
        type=Path,
        default=HERE / "day 29 logs" / "backtest_day29_osmium_orderbook_report.txt",
    )
    ap.add_argument(
        "--combined-out",
        type=Path,
        default=HERE / "day 29 logs" / "combined_all_including_extra",
    )
    ap.add_argument(
        "--normalize-rows",
        type=int,
        default=20_000,
        metavar="N",
    )
    args = ap.parse_args()
    ref_n = max(1, int(args.normalize_rows))

    main_zips, extra_zips = collect_zip_paths()
    all_zips = main_zips + extra_zips
    if not all_zips:
        raise SystemExit(f"No zips under {HERE / 'day 29 logs'}")

    merge_all_zips(all_zips, args.combined_out.resolve())

    out_lines: list[str] = []
    ow = out_lines.append
    ow("Round 2 day 29 — osmium-only strategies on submission tapes + combined")
    ow("=" * 96)
    ow("")
    ow("METHODOLOGY (read this before interpreting deltas)")
    ow("-" * 96)
    ow("  * Individual zips: one submission replay’s book + that run’s market trades.")
    ow("  * Combined tape: ``combine_submission_runs.py`` merges **books** (majority / fill")
    ow("    empty cells) and **unions** all ``tradeHistory`` rows across zips (deduped).")
    ow("  * The combined **trade tape is not a single real run** — it can contain more")
    ow("    prints at a timestamp than any one submission saw. ``OrderMatchMaker`` uses")
    ow("    those trades for ``match_trades=worse`` fills, so combined can get **many**")
    ow("    more fills than a typical single zip even when **price row count** is still")
    ow("    2000. PnL vs trade-count is **not** linear, but **~2.5× trades vs ~2.4× osmium")
    ow("    PnL** on combined vs mean singles is in the same ballpark — not a CSV bug.")
    ow("  * Input ``profit_and_loss`` / ``mid_price`` columns are for logging / MTM in")
    ow("    activity rows; **matching** uses reconstructed ``OrderDepth`` + trade list.")
    ow("  * If two osmium variants match PnL on almost every zip, their **fair / quote**")
    ow("    path coincided on those tapes (touch-freeze branch rarely changed behavior).")
    ow("")

    for strategy_title, modname in TRADER_MODULES:
        _bootstrap_sys_path()
        mod = importlib.import_module(modname)
        trader_cls = mod.Trader

        results: list[OsmiumRunResult] = []

        def run_labeled(zip_path: Path | None, bucket: str, label: str) -> None:
            root = Path(tempfile.mkdtemp(prefix="bt29osm_"))
            try:
                if zip_path is not None:
                    export_zip_to_round2(zip_path, root)
                    zn = zip_path.name
                else:
                    export_combined_csvs(args.combined_out.resolve(), root)
                    zn = "(merged CSV)"
                rr = run_osmium_trader(root, trader_cls, ROUND, DAY)
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

        out_lines.extend(
            section_lines(strategy_title, modname, results, indiv, comb, ref_n)
        )

    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")

    for name in ("_tmp_273774_full.py", "_tmp_269993_full.py"):
        p = HERE / name
        if p.is_file():
            p.unlink()


if __name__ == "__main__":
    main()
