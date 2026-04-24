#!/usr/bin/env python3
"""
Compare ``round2work/r2_submission.py`` (Round 2 working Trader)
vs ``R1submissionlogs/273774.zip`` (Trader) on:

* All ``Prosperity4Data`` round 1 + 2 days
* Each Round 2 day 29 zip under ``round2work/day 29 logs/`` and ``.../extra/``
* Combined day 29 merge (``combine_submission_runs``)

If PnL matches on every row, the strategies are equivalent under this harness;
otherwise the report lists deltas per product.

Writes ``round2work/litests/pepper7010_combined_vs_273774_zip.txt``.

From repo root::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 round2work/litests/backtest_7010combined_vs_273774.py
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
    work = Path(tempfile.mkdtemp(prefix="cmp273_"))
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


def _load_trader_class_from_submissions_py(py_path: Path) -> type:
    work = Path(tempfile.mkdtemp(prefix="cmp7010_"))
    sys.path.insert(0, str(work))
    modname = "pf7010combined"
    try:
        shutil.copy(py_path.resolve(), work / f"{modname}.py")
        (work / "datamodel.py").write_text(DATAMODEL_SHIM)
        _unload(modname)
        mod = importlib.import_module(modname)
        if not hasattr(mod, "Trader"):
            raise AttributeError(f"{py_path}: no Trader")
        return mod.Trader
    finally:
        sys.path.remove(str(work))
        _unload(modname)
        shutil.rmtree(work, ignore_errors=True)


def _run_day(
    trader_cls: type,
    data_root: Path,
    round_n: int,
    day_n: int,
    match: str,
) -> dict[str, float]:
    _bootstrap_paths()
    from prosperity4bt.models.test_options import TradeMatchingMode
    from prosperity4bt.test_runner import TestRunner
    from prosperity4bt.tools.data_reader import FileSystemReader

    mode = TradeMatchingMode.worse if match == "worse" else TradeMatchingMode.all
    reader = FileSystemReader(data_root)
    if day_n not in reader.available_days(round_n):
        raise ValueError(f"R{round_n} day {day_n} not in {data_root}")
    runner = TestRunner(
        trader_cls(),
        reader,
        round_n,
        day_n,
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
        raise RuntimeError(f"{zip_path}: export mismatch {prices=} {trades=}")
    shutil.copy(prices[0], r2 / f"prices_round_{ROUND}_day_{DAY}.csv")
    shutil.copy(trades[0], r2 / f"trades_round_{ROUND}_day_{DAY}.csv")
    shutil.rmtree(tmp)


def export_combined(combined_dir: Path, dest_root: Path) -> None:
    r2 = dest_root / "ROUND_2"
    r2.mkdir(parents=True, exist_ok=True)
    shutil.copy(combined_dir / "prices_combined_all_runs.csv", r2 / f"prices_round_{ROUND}_day_{DAY}.csv")
    shutil.copy(combined_dir / "trades_combined_all_runs.csv", r2 / f"trades_round_{ROUND}_day_{DAY}.csv")


def merge_all_zips(zips: Sequence[Path], out_dir: Path) -> None:
    subprocess.run(
        [sys.executable, str(R2WORK / "combine_submission_runs.py"), *[str(z) for z in zips], "--out-dir", str(out_dir)],
        check=True,
    )


def run_tape(
    trader_cls: type,
    zip_path: Path | None,
    combined_dir: Path,
    match: str,
) -> dict[str, float]:
    root = Path(tempfile.mkdtemp(prefix="d29cmp_"))
    try:
        if zip_path is not None:
            export_zip_to_round2(zip_path, root)
        else:
            export_combined(combined_dir, root)
        return _run_day(trader_cls, root, ROUND, DAY, match)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def fmt(x: float) -> str:
    return f"{x:,.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=REPO / "Prosperity4Data")
    ap.add_argument(
        "--combined-py",
        type=Path,
        default=R2WORK / "r2_submission.py",
        help="Trader module (default: round2work/r2_submission.py)",
    )
    ap.add_argument("--zip-273774", type=Path, default=REPO / "R1submissionlogs" / "273774.zip")
    ap.add_argument("--report", type=Path, default=HERE / "pepper7010_combined_vs_273774_zip.txt")
    ap.add_argument("--combined-dir", type=Path, default=R2WORK / "day 29 logs" / "combined_all_including_extra")
    ap.add_argument("--match", choices=("worse", "all"), default="worse")
    ap.add_argument("--no-combined", action="store_true")
    args = ap.parse_args()

    data_root = args.data.expanduser().resolve()
    comb_py = args.combined_py.expanduser().resolve()
    zip_273 = args.zip_273774.expanduser().resolve()
    if not comb_py.is_file():
        raise SystemExit(f"Missing {comb_py}")
    if not zip_273.is_file():
        raise SystemExit(f"Missing {zip_273}")

    TraderComb = _load_trader_class_from_submissions_py(comb_py)
    TraderZip = _load_trader_class_from_zip(zip_273)

    _bootstrap_paths()
    from prosperity4bt.tools.data_reader import FileSystemReader

    reader = FileSystemReader(data_root)
    hist_days: list[tuple[int, int]] = []
    for rn in (1, 2):
        for d in reader.available_days(rn):
            hist_days.append((rn, d))
    hist_days.sort()

    main_z, extra_z = collect_zips()
    all_z = main_z + extra_z
    if not all_z:
        raise SystemExit("No day-29 zips")
    comb_dir = args.combined_dir.resolve()
    comb_dir.mkdir(parents=True, exist_ok=True)
    merge_all_zips(all_z, comb_dir)

    rows: list[dict] = []

    def add_hist(rn: int, d: int) -> None:
        a = _run_day(TraderComb, data_root, rn, d, args.match)
        b = _run_day(TraderZip, data_root, rn, d, args.match)
        apn, ao = a.get(PEPPER, 0.0), a.get(OSMIUM, 0.0)
        bp, bo = b.get(PEPPER, 0.0), b.get(OSMIUM, 0.0)
        rows.append(
            {
                "section": "historical",
                "label": f"R{rn} day {d}",
                "comb_pepper": apn,
                "comb_osmium": ao,
                "comb_total": sum(a.values()),
                "zip_pepper": bp,
                "zip_osmium": bo,
                "zip_total": sum(b.values()),
            }
        )

    for rn, d in hist_days:
        add_hist(rn, d)

    def add_d29(source: str, stem: str, zp: Path | None) -> None:
        a = run_tape(TraderComb, zp, comb_dir, args.match)
        b = run_tape(TraderZip, zp, comb_dir, args.match)
        rows.append(
            {
                "section": "day29",
                "label": f"{source}/{stem}",
                "comb_pepper": a.get(PEPPER, 0.0),
                "comb_osmium": a.get(OSMIUM, 0.0),
                "comb_total": sum(a.values()),
                "zip_pepper": b.get(PEPPER, 0.0),
                "zip_osmium": b.get(OSMIUM, 0.0),
                "zip_total": sum(b.values()),
            }
        )

    for z in main_z:
        add_d29("day 29 logs", z.stem, z)
    for z in extra_z:
        add_d29("extra", z.stem, z)
    if not args.no_combined:
        add_d29("combined", "ALL", None)

    for r in rows:
        r["d_pepper"] = r["comb_pepper"] - r["zip_pepper"]
        r["d_osmium"] = r["comb_osmium"] - r["zip_osmium"]
        r["d_total"] = r["comb_total"] - r["zip_total"]

    EPS = 1e-6
    diffs = [r for r in rows if abs(r["d_total"]) > EPS or abs(r["d_pepper"]) > EPS or abs(r["d_osmium"]) > EPS]

    lines: list[str] = []
    w = lines.append
    w("r2_submission.py  vs  273774.zip")
    w("=" * 110)
    w(f"Combined py: {comb_py}")
    w(f"Zip:         {zip_273}")
    w(f"Data:        {data_root}")
    w(f"Matching:    {args.match}")
    w("")

    hdr = (
        f"{'section':<12} {'label':<22} "
        f"{'comb_pep':>12} {'comb_os':>10} {'comb_tot':>10}  "
        f"{'zip_pep':>12} {'zip_os':>10} {'zip_tot':>10}  "
        f"{'d_pep':>9} {'d_os':>9} {'d_tot':>9}"
    )
    w(hdr)
    w("-" * len(hdr))
    for r in rows:
        w(
            f"{r['section']:<12} {r['label']:<22} "
            f"{fmt(r['comb_pepper']):>12} {fmt(r['comb_osmium']):>10} {fmt(r['comb_total']):>10}  "
            f"{fmt(r['zip_pepper']):>12} {fmt(r['zip_osmium']):>10} {fmt(r['zip_total']):>10}  "
            f"{fmt(r['d_pepper']):>9} {fmt(r['d_osmium']):>9} {fmt(r['d_total']):>9}"
        )
    w("-" * len(hdr))

    w("")
    if not diffs:
        w("RESULT: No differences (within float tolerance) on any row — treat as identical under this backtester.")
    else:
        w(f"RESULT: {len(diffs)} row(s) differ (comb minus zip):")
        for r in diffs:
            w(
                f"  {r['section']} {r['label']}:  d_pepper={fmt(r['d_pepper'])}  "
                f"d_osmium={fmt(r['d_osmium'])}  d_total={fmt(r['d_total'])}"
            )

    outp = args.report.expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {outp}")
    if diffs:
        print(f"Differing rows: {len(diffs)}")
    else:
        print("All rows match.")


if __name__ == "__main__":
    main()
