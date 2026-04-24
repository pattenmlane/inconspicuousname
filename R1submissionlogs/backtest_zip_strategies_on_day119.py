#!/usr/bin/env python3
"""
Run each **submission strategy** from ``R1submissionlogs/*.zip`` (the ``<id>.py``
inside the zip) on **canonical** Round 1 day **119** data under ``Prosperity4Data``.

Each zip must contain ``<stem>.py`` with a ``Trader`` class using
``from datamodel import Order, OrderDepth, TradingState``. A tiny ``datamodel``
shim is written next to the extracted file so imports resolve to
``prosperity4bt.datamodel``.

Writes ``R1submissionlogs/backtest_day119_strategies_breakdown.txt``.

From repo root::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 R1submissionlogs/backtest_zip_strategies_on_day119.py
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"
ROUND = 1
DAY = 119

DATAMODEL_SHIM = """\
from prosperity4bt.datamodel import Order, OrderDepth, TradingState
"""


def _bootstrap_paths() -> None:
    for p in (
        REPO / "imc-prosperity-4-backtester",
        REPO / "imc-prosperity-4-backtester" / "prosperity4bt",
    ):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def collect_zips() -> list[Path]:
    z = sorted(p for p in HERE.glob("*.zip") if p.is_file())
    if len(z) != 5:
        print(f"Warning: expected 5 zips in {HERE}, found {len(z)}", file=sys.stderr)
    return z


def extract_strategy_module(zip_path: Path, work: Path) -> str:
    stem = zip_path.stem
    py_name = f"{stem}.py"
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if py_name not in names:
            raise FileNotFoundError(f"{zip_path}: missing {py_name}, have: {names}")
        (work / py_name).write_bytes(zf.read(py_name))
    (work / "datamodel.py").write_text(DATAMODEL_SHIM)
    return stem


def unload_module(name: str) -> None:
    if name in sys.modules:
        del sys.modules[name]
    for k in list(sys.modules):
        if k.startswith(name + "."):
            del sys.modules[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=REPO / "Prosperity4Data",
        help="Prosperity4Data root (must contain ROUND1 day 119 CSVs)",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=HERE / "backtest_day119_strategies_breakdown.txt",
        help="Output report path",
    )
    ap.add_argument(
        "--match",
        choices=("worse", "all"),
        default="worse",
        help="Trade matching mode (default: worse)",
    )
    args = ap.parse_args()

    data_root = args.data.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"Missing data dir: {data_root}")

    zips = collect_zips()
    if not zips:
        raise SystemExit(f"No .zip files in {HERE}")

    rows: list[dict] = []
    for zp in zips:
        rows.append(run_one_strategy_with_match(zp, data_root, args.match))

    lines: list[str] = []
    w = lines.append
    w("Round 1 day 119 — each R1submissionlogs strategy on canonical Prosperity4Data")
    w("=" * 78)
    w("")
    w(f"Data: {data_root}")
    w(f"Trade matching: {args.match}")
    w(f"Zips ({len(zips)}): " + ", ".join(z.name for z in zips))
    w("")
    hdr = f"{'submission':<12} {'pepper_pnl':>14} {'osmium_pnl':>14} {'total_pnl':>14}"
    w(hdr)
    w("-" * len(hdr))
    for r in rows:
        w(
            f"{r['stem']:<12} {r['pepper']:>14,.2f} {r['osmium']:>14,.2f} {r['total']:>14,.2f}"
        )
    w("-" * len(hdr))
    peppers = [r["pepper"] for r in rows]
    osms = [r["osmium"] for r in rows]
    tots = [r["total"] for r in rows]
    w("")
    w("Summary")
    w("-" * 78)
    w(f"  Mean pepper:   {sum(peppers)/len(peppers):>14,.2f}")
    w(f"  Mean osmium:   {sum(osms)/len(osms):>14,.2f}")
    w(f"  Mean total:    {sum(tots)/len(tots):>14,.2f}")
    w(f"  Best total:    {max(tots):>14,.2f}  ({rows[tots.index(max(tots))]['stem']})")
    w(f"  Worst total:   {min(tots):>14,.2f}  ({rows[tots.index(min(tots))]['stem']})")
    w("")
    w("Per-product keys in final activity (for debugging non-pepper/osmium)")
    w("-" * 78)
    for r in rows:
        w(f"  {r['stem']}: {sorted(r['products'].keys())}")

    out = args.report.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


def run_one_strategy_with_match(zip_path: Path, data_root: Path, match: str) -> dict:
    _bootstrap_paths()
    from prosperity4bt.models.test_options import TradeMatchingMode
    from prosperity4bt.test_runner import TestRunner
    from prosperity4bt.tools.data_reader import FileSystemReader

    mode = TradeMatchingMode.worse if match == "worse" else TradeMatchingMode.all

    work = Path(tempfile.mkdtemp(prefix="r1d119_"))
    sys.path.insert(0, str(work))
    stem = ""
    try:
        stem = extract_strategy_module(zip_path, work)
        unload_module(stem)
        mod = importlib.import_module(stem)
        if not hasattr(mod, "Trader"):
            raise AttributeError(f"{stem}.py has no Trader class")

        reader = FileSystemReader(data_root)
        if DAY not in reader.available_days(ROUND):
            raise SystemExit(f"No round {ROUND} day {DAY} under {data_root}; have {reader.available_days(ROUND)}")

        runner = TestRunner(
            mod.Trader(),
            reader,
            ROUND,
            DAY,
            show_progress_bar=False,
            print_output=False,
            trade_matching_mode=mode,
        )
        result = runner.run()
        by_sym: dict[str, float] = {}
        for row in result.final_activities():
            by_sym[row.symbol] = float(row.profit_loss)
        total = sum(by_sym.values())
        return {
            "zip": zip_path.name,
            "stem": stem,
            "pepper": by_sym.get(PEPPER, 0.0),
            "osmium": by_sym.get(OSMIUM, 0.0),
            "total": total,
            "products": by_sym,
        }
    finally:
        sys.path.remove(str(work))
        if stem:
            unload_module(stem)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
