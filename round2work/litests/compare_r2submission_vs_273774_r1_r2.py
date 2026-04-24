#!/usr/bin/env python3
"""
Compare ``round2work/r2_submission.py`` vs ``R1submissionlogs/273774.zip`` on
all Prosperity4Data days for **round 1** and **round 2** (``match_trades=worse``).

Writes ``round2work/litests/r2_vs_273774_r1_r2.txt``.

From repo root::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 round2work/litests/compare_r2submission_vs_273774_r1_r2.py
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import tempfile
import types
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
R2WORK = HERE.parent
REPO = R2WORK.parent

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

DATAMODEL_SHIM = """\
from prosperity4bt.datamodel import Order, OrderDepth, TradingState
"""


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


def _load_trader_from_zip(zip_path: Path) -> type:
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


def _load_r2_submission_trader() -> type:
    dm = types.ModuleType("datamodel")
    _bootstrap_bt()
    from prosperity4bt import datamodel as _dm

    for n in ("Order", "OrderDepth", "TradingState"):
        setattr(dm, n, getattr(_dm, n))
    sys.modules["datamodel"] = dm
    sys.path.insert(0, str(R2WORK))
    try:
        _unload("r2_submission")
        mod = importlib.import_module("r2_submission")
        if not hasattr(mod, "Trader"):
            raise AttributeError("r2_submission: no Trader")
        return mod.Trader
    finally:
        sys.path.remove(str(R2WORK))


def _run_day(trader_cls: type, data_root: Path, round_n: int, day_n: int) -> dict[str, float]:
    _bootstrap_bt()
    from prosperity4bt.models.test_options import TradeMatchingMode
    from prosperity4bt.test_runner import TestRunner
    from prosperity4bt.tools.data_reader import FileSystemReader

    reader = FileSystemReader(data_root)
    if day_n not in reader.available_days(round_n):
        raise ValueError(f"R{round_n} day {day_n} not available")
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
    return {row.symbol: float(row.profit_loss) for row in result.final_activities()}


def fmt(x: float) -> str:
    return f"{x:,.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=REPO / "Prosperity4Data")
    ap.add_argument("--zip-273774", type=Path, default=REPO / "R1submissionlogs" / "273774.zip")
    ap.add_argument("--report", type=Path, default=HERE / "r2_vs_273774_r1_r2.txt")
    args = ap.parse_args()

    data_root = args.data.expanduser().resolve()
    zip_p = args.zip_273774.expanduser().resolve()
    if not zip_p.is_file():
        raise SystemExit(f"Missing {zip_p}")
    if not (R2WORK / "r2_submission.py").is_file():
        raise SystemExit(f"Missing {R2WORK / 'r2_submission.py'}")

    TraderR2 = _load_r2_submission_trader()
    TraderZip = _load_trader_from_zip(zip_p)

    _bootstrap_bt()
    from prosperity4bt.tools.data_reader import FileSystemReader

    reader = FileSystemReader(data_root)
    rows: list[dict] = []
    for rn in (1, 2):
        for d in sorted(reader.available_days(rn)):
            a = _run_day(TraderR2, data_root, rn, d)
            b = _run_day(TraderZip, data_root, rn, d)
            apn, ao = a.get(PEPPER, 0.0), a.get(OSMIUM, 0.0)
            bp, bo = b.get(PEPPER, 0.0), b.get(OSMIUM, 0.0)
            at, bt = sum(a.values()), sum(b.values())
            rows.append(
                {
                    "r": rn,
                    "d": d,
                    "r2_pepper": apn,
                    "r2_osmium": ao,
                    "r2_total": at,
                    "z_pepper": bp,
                    "z_osmium": bo,
                    "z_total": bt,
                    "d_pepper": apn - bp,
                    "d_osmium": ao - bo,
                    "d_total": at - bt,
                }
            )

    lines: list[str] = []
    w = lines.append
    w("r2_submission.py vs 273774.zip — Prosperity4Data R1 + R2")
    w("=" * 110)
    w(f"Data: {data_root}")
    w(f"Zip:  {zip_p}")
    w("Trade matching: worse")
    w("")
    hdr = (
        f"{'R':>2} {'day':>5}  "
        f"{'r2_pepper':>14} {'r2_os':>12} {'r2_tot':>12}  "
        f"{'774_pepper':>14} {'774_os':>12} {'774_tot':>12}  "
        f"{'d_pep':>10} {'d_os':>10} {'d_tot':>10}"
    )
    w(hdr)
    w("-" * len(hdr))
    for r in rows:
        w(
            f"{r['r']:>2} {r['d']:>5}  "
            f"{fmt(r['r2_pepper']):>14} {fmt(r['r2_osmium']):>12} {fmt(r['r2_total']):>12}  "
            f"{fmt(r['z_pepper']):>14} {fmt(r['z_osmium']):>12} {fmt(r['z_total']):>12}  "
            f"{fmt(r['d_pepper']):>10} {fmt(r['d_osmium']):>10} {fmt(r['d_total']):>10}"
        )
    w("-" * len(hdr))

    EPS = 1e-6
    diffs = [r for r in rows if abs(r["d_total"]) > EPS or abs(r["d_pepper"]) > EPS or abs(r["d_osmium"]) > EPS]
    w("")
    if not diffs:
        w("RESULT: No PnL differences (r2_submission vs 273774.zip) on any day.")
    else:
        w(f"RESULT: {len(diffs)} row(s) differ (r2 − zip):")
        for r in diffs:
            w(
                f"  R{r['r']} day {r['d']}: d_pepper={fmt(r['d_pepper'])} d_osmium={fmt(r['d_osmium'])} d_total={fmt(r['d_total'])}"
            )

    br2 = TraderR2()
    w("")
    w(f"r2_submission.bid() = {br2.bid()} (MAF; not used in TestRunner PnL)")

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
