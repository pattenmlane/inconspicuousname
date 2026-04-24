#!/usr/bin/env python3
"""
Count how many timestamps ``r2_submission.Trader`` would set ``slope_crash``
(pepper rolling-mid safeguard) on:

* All Prosperity4Data R1 + R2 days
* Each Round 2 day-29 zip + combined

Uses the same preconditions as ``Trader.run`` (hist from ``traderData``, append
current mid, OLS on last PEPPER_SLOPE_WINDOW points). ``super().run`` is still
called so state stays consistent.

From repo root::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 round2work/litests/count_pepper_slope_safeguard_triggers.py
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import tempfile
import types
import zipfile
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
R2WORK = HERE.parent
REPO = R2WORK.parent

ROUND_D29 = 2
DAY_D29 = 29


def _bootstrap_bt() -> None:
    for p in (
        REPO / "imc-prosperity-4-backtester",
        REPO / "imc-prosperity-4-backtester" / "prosperity4bt",
    ):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _unload(name: str) -> None:
    if name in sys.modules:
        del sys.modules[name]
    for k in list(sys.modules):
        if k.startswith(name + "."):
            del sys.modules[k]


def _load_r2_module():
    dm = types.ModuleType("datamodel")
    _bootstrap_bt()
    from prosperity4bt import datamodel as _dm

    for n in ("Order", "OrderDepth", "TradingState"):
        setattr(dm, n, getattr(_dm, n))
    sys.modules["datamodel"] = dm
    sys.path.insert(0, str(R2WORK))
    try:
        _unload("r2_submission")
        return importlib.import_module("r2_submission")
    finally:
        sys.path.remove(str(R2WORK))


def collect_day29_zips() -> tuple[list[Path], list[Path]]:
    main = sorted(p for p in (R2WORK / "day 29 logs").glob("*.zip") if p.is_file())
    extra = sorted(p for p in (R2WORK / "day 29 logs" / "extra").glob("*.zip") if p.is_file())
    return main, extra


def export_zip_to_round2(zip_path: Path, dest_root: Path) -> None:
    tmp = dest_root / "_tmp_export"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            str(R2WORK / "logtodata.py"),
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
    shutil.copy(combined_dir / "prices_combined_all_runs.csv", r2 / f"prices_round_{ROUND_D29}_day_{DAY_D29}.csv")
    shutil.copy(combined_dir / "trades_combined_all_runs.csv", r2 / f"trades_round_{ROUND_D29}_day_{DAY_D29}.csv")


def merge_all_zips(zips: Sequence[Path], out_dir: Path) -> None:
    subprocess.run(
        [sys.executable, str(R2WORK / "combine_submission_runs.py"), *[str(z) for z in zips], "--out-dir", str(out_dir)],
        check=True,
    )


def build_counting_trader_class(R):
    class CountingTrader(R.Trader):
        def __init__(self) -> None:
            super().__init__()
            self.slope_crash_ticks = 0

        def run(self, state):
            store = R._parse_trader_data(getattr(state, "traderData", None))
            depths = getattr(state, "order_depths", None)
            if not isinstance(depths, dict):
                depths = {}
            depth_pe = depths.get(R.PEPPER)
            mid_pe = R._micro_mid(depth_pe) if depth_pe else None
            hist = R._sanitize_pepper_slope_hist(store.get(R.PEPPER_SLOPE_HIST), R.PEPPER_SLOPE_WINDOW)
            ts = R._timestamp_int(state)
            if mid_pe is not None:
                hist = list(hist)
                hist.append([float(ts), float(mid_pe)])
                if len(hist) > R.PEPPER_SLOPE_WINDOW * 2:
                    hist = hist[-(R.PEPPER_SLOPE_WINDOW * 2) :]
            slope_crash = False
            if (
                len(hist) >= R.PEPPER_SLOPE_WINDOW
                and depth_pe is not None
                and depth_pe.buy_orders
                and depth_pe.sell_orders
            ):
                slope = R._ols_slope_mid_vs_time(hist[-R.PEPPER_SLOPE_WINDOW :])
                slope_crash = slope < R.PEPPER_SLOPE_SAFEGUARD
            if slope_crash and R.PEPPER in depths:
                self.slope_crash_ticks += 1
            return super().run(state)

    return CountingTrader


def run_count(data_root: Path, rn: int, dn: int, TraderCls) -> int:
    _bootstrap_bt()
    from prosperity4bt.models.test_options import TradeMatchingMode
    from prosperity4bt.test_runner import TestRunner
    from prosperity4bt.tools.data_reader import FileSystemReader

    reader = FileSystemReader(data_root)
    t = TraderCls()
    runner = TestRunner(
        t,
        reader,
        rn,
        dn,
        show_progress_bar=False,
        print_output=False,
        trade_matching_mode=TradeMatchingMode.worse,
    )
    runner.run()
    return int(t.slope_crash_ticks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=REPO / "Prosperity4Data")
    ap.add_argument(
        "--combined-dir",
        type=Path,
        default=R2WORK / "day 29 logs" / "combined_all_including_extra",
    )
    ap.add_argument("--no-combined", action="store_true")
    args = ap.parse_args()

    R = _load_r2_module()
    TraderCls = build_counting_trader_class(R)
    data_root = args.data.expanduser().resolve()

    _bootstrap_bt()
    from prosperity4bt.tools.data_reader import FileSystemReader

    reader = FileSystemReader(data_root)
    lines: list[str] = []
    w = lines.append
    w("Pepper slope_crash tick counts (same logic as r2_submission.run)")
    w(f"PEPPER_SLOPE_WINDOW={R.PEPPER_SLOPE_WINDOW} PEPPER_SLOPE_SAFEGUARD={R.PEPPER_SLOPE_SAFEGUARD}")
    w("")

    w("Historical (Prosperity4Data)")
    w("-" * 50)
    hist_total = 0
    for rn in (1, 2):
        for d in sorted(reader.available_days(rn)):
            n = run_count(data_root, rn, d, TraderCls)
            hist_total += n
            w(f"  R{rn} day {d:>5}: {n} ticks")
    w(f"  SUM R1+R2: {hist_total}")
    w("")

    main_z, extra_z = collect_day29_zips()
    all_z = main_z + extra_z
    comb_dir = args.combined_dir.resolve()
    comb_dir.mkdir(parents=True, exist_ok=True)
    merge_all_zips(all_z, comb_dir)

    w("Round 2 day 29 tapes")
    w("-" * 50)
    d29_total = 0
    for z in main_z:
        root = Path(tempfile.mkdtemp(prefix="slope29_"))
        try:
            export_zip_to_round2(z, root)
            n = run_count(root, ROUND_D29, DAY_D29, TraderCls)
            d29_total += n
            w(f"  day 29 logs/{z.name}: {n} ticks")
        finally:
            shutil.rmtree(root, ignore_errors=True)
    for z in extra_z:
        root = Path(tempfile.mkdtemp(prefix="slope29_"))
        try:
            export_zip_to_round2(z, root)
            n = run_count(root, ROUND_D29, DAY_D29, TraderCls)
            d29_total += n
            w(f"  extra/{z.name}: {n} ticks")
        finally:
            shutil.rmtree(root, ignore_errors=True)
    if not args.no_combined:
        root = Path(tempfile.mkdtemp(prefix="slope29c_"))
        try:
            export_combined(comb_dir, root)
            n = run_count(root, ROUND_D29, DAY_D29, TraderCls)
            d29_total += n
            w(f"  combined/ALL: {n} ticks")
        finally:
            shutil.rmtree(root, ignore_errors=True)
    w(f"  SUM day29 (+combined unless --no-combined): {d29_total}")
    w("")

    outp = HERE / "pepper_slope_safeguard_trigger_counts.txt"
    outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
