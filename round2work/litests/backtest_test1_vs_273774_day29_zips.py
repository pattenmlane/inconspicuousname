#!/usr/bin/env python3
"""
Backtest ``test1.Trader`` vs ``273774.zip`` on each **Round 2 day 29** submission
tape (``round2work/day 29 logs/*.zip``, ``.../extra/*.zip``), plus **combined**
merge (same as ``combine_submission_runs``).

Per-product PnL: pepper, osmium, total. ``match_trades=worse`` by default.

Writes ``round2work/litests/test1_vs_273774_day29_zips.txt``.

From repo root::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt:$PWD/round2work/litests" \\
  python3 round2work/litests/backtest_test1_vs_273774_day29_zips.py
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
R2WORK = HERE.parent
REPO = R2WORK.parent

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"
ROUND = 2
DAY = 29

DATAMODEL_SHIM = """\
from prosperity4bt.datamodel import Order, OrderDepth, TradingState
"""


def _bootstrap_paths() -> None:
    lit = str(HERE)
    if lit not in sys.path:
        sys.path.insert(0, lit)
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


def _load_trader_class_from_zip(zip_path: Path) -> type:
    stem = zip_path.stem
    work = Path(tempfile.mkdtemp(prefix="t1d29_"))
    sys.path.insert(0, str(work))
    try:
        py_name = f"{stem}.py"
        with zipfile.ZipFile(zip_path) as zf:
            if py_name not in zf.namelist():
                raise FileNotFoundError(f"{zip_path}: missing {py_name}")
            (work / py_name).write_bytes(zf.read(py_name))
        (work / "datamodel.py").write_text(DATAMODEL_SHIM)
        _unload(stem)
        mod = importlib.import_module(stem)
        if not hasattr(mod, "Trader"):
            raise AttributeError(f"{stem}.py: no Trader")
        return mod.Trader
    finally:
        sys.path.remove(str(work))
        _unload(stem)
        shutil.rmtree(work, ignore_errors=True)


def _run_day(
    trader_cls: type,
    data_root: Path,
    match: str,
) -> dict[str, float]:
    _bootstrap_paths()
    from prosperity4bt.models.test_options import TradeMatchingMode
    from prosperity4bt.test_runner import TestRunner
    from prosperity4bt.tools.data_reader import FileSystemReader

    mode = TradeMatchingMode.worse if match == "worse" else TradeMatchingMode.all
    reader = FileSystemReader(data_root)
    if DAY not in reader.available_days(ROUND):
        raise ValueError(f"R{ROUND} day {DAY} not in {data_root}: {reader.available_days(ROUND)}")
    runner = TestRunner(
        trader_cls(),
        reader,
        ROUND,
        DAY,
        show_progress_bar=False,
        print_output=False,
        trade_matching_mode=mode,
    )
    result = runner.run()
    return {row.symbol: float(row.profit_loss) for row in result.final_activities()}


def collect_zips() -> tuple[list[Path], list[Path]]:
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


def export_combined(combined_dir: Path, dest_root: Path) -> None:
    r2 = dest_root / "ROUND_2"
    r2.mkdir(parents=True, exist_ok=True)
    p = combined_dir / "prices_combined_all_runs.csv"
    t = combined_dir / "trades_combined_all_runs.csv"
    shutil.copy(p, r2 / f"prices_round_{ROUND}_day_{DAY}.csv")
    shutil.copy(t, r2 / f"trades_round_{ROUND}_day_{DAY}.csv")


def merge_all_zips(zips: Sequence[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(R2WORK / "combine_submission_runs.py"), *[str(z) for z in zips], "--out-dir", str(out_dir)],
        check=True,
    )


def run_tape(
    trader_cls: type,
    zip_path: Path | None,
    combined_dir: Path,
    all_zips: list[Path],
    match: str,
) -> dict[str, float]:
    root = Path(tempfile.mkdtemp(prefix="t1d29run_"))
    try:
        if zip_path is not None:
            export_zip_to_round2(zip_path, root)
        else:
            export_combined(combined_dir, root)
        return _run_day(trader_cls, root, match)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--zip-273774",
        type=Path,
        default=REPO / "R1submissionlogs" / "273774.zip",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=HERE / "test1_vs_273774_day29_zips.txt",
    )
    ap.add_argument(
        "--combined-dir",
        type=Path,
        default=R2WORK / "day 29 logs" / "combined_all_including_extra",
    )
    ap.add_argument("--match", choices=("worse", "all"), default="worse")
    ap.add_argument("--no-combined", action="store_true", help="Skip merged-book combined row")
    args = ap.parse_args()

    zip_273 = args.zip_273774.expanduser().resolve()
    if not zip_273.is_file():
        raise SystemExit(f"Missing {zip_273}")

    main_z, extra_z = collect_zips()
    all_z = main_z + extra_z
    if not all_z:
        raise SystemExit(f"No zips under {R2WORK / 'day 29 logs'}")

    comb_dir = args.combined_dir.resolve()
    comb_dir.mkdir(parents=True, exist_ok=True)
    merge_all_zips(all_z, comb_dir)

    _bootstrap_paths()
    Trader273 = _load_trader_class_from_zip(zip_273)
    from test1 import Trader as TraderTest1

    rows: list[dict] = []

    def one_row(source: str, stem: str, zip_path: Path | None) -> None:
        t1 = run_tape(TraderTest1, zip_path, comb_dir, all_z, args.match)
        z7 = run_tape(Trader273, zip_path, comb_dir, all_z, args.match)
        t1p, t1o = t1.get(PEPPER, 0.0), t1.get(OSMIUM, 0.0)
        z7p, z7o = z7.get(PEPPER, 0.0), z7.get(OSMIUM, 0.0)
        t1tot, z7tot = sum(t1.values()), sum(z7.values())
        rows.append(
            {
                "source": source,
                "run": stem,
                "t1_pepper": t1p,
                "t1_osmium": t1o,
                "t1_total": t1tot,
                "z_pepper": z7p,
                "z_osmium": z7o,
                "z_total": z7tot,
            }
        )

    for z in main_z:
        one_row("day 29 logs", z.stem, z)
    for z in extra_z:
        one_row("extra", z.stem, z)
    if not args.no_combined:
        one_row("combined", "ALL", None)

    def fmt(x: float) -> str:
        return f"{x:,.2f}"

    lines: list[str] = []
    w = lines.append
    w("test1 vs 273774 — Round 2 day 29 submission tapes (each zip + combined)")
    w("=" * 118)
    w(f"273774 zip: {zip_273.name}")
    w(f"Trade matching: {args.match}")
    w("")
    hdr = (
        f"{'source':<14} {'run':<12} "
        f"{'t1_pepper':>12} {'t1_os':>10} {'t1_tot':>10}  "
        f"{'774_pepper':>12} {'774_os':>10} {'774_tot':>10}  "
        f"{'d_pep':>9} {'d_os':>9} {'d_tot':>9}"
    )
    w(hdr)
    w("-" * len(hdr))
    for r in rows:
        dp = r["t1_pepper"] - r["z_pepper"]
        do = r["t1_osmium"] - r["z_osmium"]
        dt = r["t1_total"] - r["z_total"]
        w(
            f"{r['source']:<14} {r['run']:<12} "
            f"{fmt(r['t1_pepper']):>12} {fmt(r['t1_osmium']):>10} {fmt(r['t1_total']):>10}  "
            f"{fmt(r['z_pepper']):>12} {fmt(r['z_osmium']):>10} {fmt(r['z_total']):>10}  "
            f"{fmt(dp):>9} {fmt(do):>9} {fmt(dt):>9}"
        )
    w("-" * len(hdr))

    indiv = [r for r in rows if r["source"] != "combined"]
    w("")
    w(f"Individual zips: {len(indiv)}")
    w(
        f"  test1 wins total: {sum(1 for r in indiv if r['t1_total'] > r['z_total'])} / {len(indiv)}  "
        f"wins osmium: {sum(1 for r in indiv if r['t1_osmium'] > r['z_osmium'])}  "
        f"wins pepper: {sum(1 for r in indiv if r['t1_pepper'] > r['z_pepper'])}"
    )

    outp = args.report.expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
