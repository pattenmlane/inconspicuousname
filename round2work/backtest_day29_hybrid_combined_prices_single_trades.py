#!/usr/bin/env python3
"""
Hybrid Round 2 day-29 tape for backtesting:

* **Prices:** merged ``prices_combined_all_runs.csv`` (same as ``combine_submission_runs``).
* **Trades:** from **one** submission zip only (via ``logtodata.py``), not the union.

Runs both osmium-only standalones and prints (and optionally writes) osmium PnL.

Example::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt:$PWD/round2work" \\
  python3 round2work/backtest_day29_hybrid_combined_prices_single_trades.py \\
    --trades-zip "round2work/day 29 logs/278346.zip"
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

ROUND = 2
DAY = 29
OSMIUM = "ASH_COATED_OSMIUM"

TRADERS: tuple[tuple[str, str], ...] = (
    ("273774 WM spike freeze", "osmium_273774_wm_freeze_standalone"),
    ("269993 WM + touch", "osmium_269993_touch_wm_standalone"),
)


def _bootstrap() -> None:
    for p in (
        REPO / "imc-prosperity-4-backtester",
        REPO / "imc-prosperity-4-backtester" / "prosperity4bt",
        HERE,
    ):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def ensure_combined_prices(combined_dir: Path, all_zips: list[Path]) -> Path:
    prices = combined_dir / "prices_combined_all_runs.csv"
    if not prices.is_file():
        combined_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, str(HERE / "combine_submission_runs.py"), *[str(z) for z in all_zips], "--out-dir", str(combined_dir)],
            check=True,
        )
    if not prices.is_file():
        raise FileNotFoundError(f"Missing combined prices: {prices}")
    return prices


def export_trades_only(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
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
            str(out_dir),
        ],
        check=True,
    )
    trades = list(out_dir.glob(f"trades_round_{ROUND}_day_{DAY}_*.csv"))
    if len(trades) != 1:
        raise RuntimeError(f"Expected one trades export in {out_dir}, got {trades}")
    return trades[0]


def run_osmium_pnl(trader_mod_name: str, data_root: Path) -> float:
    _bootstrap()
    from prosperity4bt.models.test_options import TradeMatchingMode
    from prosperity4bt.test_runner import TestRunner
    from prosperity4bt.tools.data_reader import FileSystemReader

    mod = importlib.import_module(trader_mod_name)
    reader = FileSystemReader(data_root)
    runner = TestRunner(
        mod.Trader(),
        reader,
        ROUND,
        DAY,
        show_progress_bar=False,
        print_output=False,
        trade_matching_mode=TradeMatchingMode.worse,
    )
    result = runner.run()
    for row in result.final_activities():
        if row.symbol == OSMIUM:
            return float(row.profit_loss)
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trades-zip",
        type=Path,
        default=HERE / "day 29 logs" / "278346.zip",
        help="Submission zip whose tradeHistory becomes the hybrid trades file",
    )
    ap.add_argument(
        "--combined-dir",
        type=Path,
        default=HERE / "day 29 logs" / "combined_all_including_extra",
        help="Directory with prices_combined_all_runs.csv (and zips for rebuild if missing)",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=HERE / "day 29 logs" / "backtest_day29_hybrid_combined_prices_single_trades.txt",
    )
    args = ap.parse_args()

    main_zips = sorted(p for p in (HERE / "day 29 logs").glob("*.zip") if p.is_file())
    extra_zips = sorted(p for p in (HERE / "day 29 logs" / "extra").glob("*.zip") if p.is_file())
    all_zips = main_zips + extra_zips
    if not all_zips:
        raise SystemExit("No day-29 zips found")

    trades_zip = args.trades_zip.expanduser().resolve()
    if not trades_zip.is_file():
        raise FileNotFoundError(trades_zip)

    combined_dir = args.combined_dir.expanduser().resolve()
    prices_path = ensure_combined_prices(combined_dir, all_zips)

    tmp_trade_export = Path(tempfile.mkdtemp(prefix="hybrid_tr_"))
    try:
        trades_src = export_trades_only(trades_zip, tmp_trade_export)
        data_root = Path(tempfile.mkdtemp(prefix="hybrid_data_"))
        try:
            r2 = data_root / "ROUND_2"
            r2.mkdir(parents=True)
            shutil.copy(prices_path, r2 / f"prices_round_{ROUND}_day_{DAY}.csv")
            shutil.copy(trades_src, r2 / f"trades_round_{ROUND}_day_{DAY}.csv")

            lines = [
                "Hybrid tape: combined prices + single-zip trades",
                "=" * 72,
                f"Prices: {prices_path}",
                f"Trades: {trades_zip.name} → {trades_src.name}",
                f"match_trades: worse",
                "",
                f"{'strategy':<28} {'osmium_pnl':>14}",
                "-" * 44,
            ]
            for label, modname in TRADERS:
                pnl = run_osmium_pnl(modname, data_root)
                lines.append(f"{label:<28} {pnl:>14,.2f}")

            text = "\n".join(lines) + "\n"
            print(text, end="")
            outp = args.report.expanduser().resolve()
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(text, encoding="utf-8")
            print(f"Wrote {outp}")
        finally:
            shutil.rmtree(data_root, ignore_errors=True)
    finally:
        shutil.rmtree(tmp_trade_export, ignore_errors=True)


if __name__ == "__main__":
    main()
