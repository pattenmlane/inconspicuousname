#!/usr/bin/env python3
"""
Backtest ``test1.Trader`` vs full submission ``273774.zip`` on every
``Prosperity4Data`` day (rounds 1 and 2). Per-product PnL: pepper, osmium, total.

Writes ``round2work/litests/test1_vs_273774_all_historical.txt``.

From repo root::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt:$PWD/round2work/litests" \\
  python3 round2work/litests/backtest_test1_vs_273774_all_historical.py
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
REPO = HERE.parent.parent

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

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
    work = Path(tempfile.mkdtemp(prefix="t1cmp_"))
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
        raise ValueError(f"R{round_n} day {day_n} not available: {reader.available_days(round_n)}")
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
    out: dict[str, float] = {}
    for row in result.final_activities():
        out[row.symbol] = float(row.profit_loss)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=REPO / "Prosperity4Data")
    ap.add_argument(
        "--zip-273774",
        type=Path,
        default=REPO / "R1submissionlogs" / "273774.zip",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=HERE / "test1_vs_273774_all_historical.txt",
    )
    ap.add_argument("--match", choices=("worse", "all"), default="worse")
    args = ap.parse_args()

    data_root = args.data.expanduser().resolve()
    zip_273 = args.zip_273774.expanduser().resolve()
    if not zip_273.is_file():
        raise SystemExit(f"Missing {zip_273}")

    _bootstrap_paths()
    from prosperity4bt.tools.data_reader import FileSystemReader

    reader = FileSystemReader(data_root)
    days: list[tuple[int, int]] = []
    for rn in (1, 2):
        for d in reader.available_days(rn):
            days.append((rn, d))
    days.sort()

    Trader273 = _load_trader_class_from_zip(zip_273)
    from test1 import Trader as TraderTest1

    rows: list[dict] = []
    for rn, d in days:
        t1 = _run_day(TraderTest1, data_root, rn, d, args.match)
        z7 = _run_day(Trader273, data_root, rn, d, args.match)
        t1p, t1o = t1.get(PEPPER, 0.0), t1.get(OSMIUM, 0.0)
        z7p, z7o = z7.get(PEPPER, 0.0), z7.get(OSMIUM, 0.0)
        t1tot = sum(t1.values())
        z7tot = sum(z7.values())
        rows.append(
            {
                "round": rn,
                "day": d,
                "t1_pepper": t1p,
                "t1_osmium": t1o,
                "t1_total": t1tot,
                "z_pepper": z7p,
                "z_osmium": z7o,
                "z_total": z7tot,
                "d_pepper": t1p - z7p,
                "d_osmium": t1o - z7o,
                "d_total": t1tot - z7tot,
            }
        )

    def fmt(x: float) -> str:
        return f"{x:,.2f}"

    lines: list[str] = []
    w = lines.append
    w("test1 (litests/test1.py) vs 273774.zip — all Prosperity4Data days")
    w("=" * 120)
    w("")
    w(f"Data root: {data_root}")
    w(f"Trade matching: {args.match}")
    w(f"273774 zip: {zip_273.name}")
    w("")
    hdr = (
        f"{'R':>2} {'day':>5}  "
        f"{'t1_pepper':>14} {'t1_os':>12} {'t1_tot':>12}  "
        f"{'774_pepper':>14} {'774_os':>12} {'774_tot':>12}  "
        f"{'d_pep':>10} {'d_os':>10} {'d_tot':>10}"
    )
    w(hdr)
    w("-" * len(hdr))
    for r in rows:
        w(
            f"{r['round']:>2} {r['day']:>5}  "
            f"{fmt(r['t1_pepper']):>14} {fmt(r['t1_osmium']):>12} {fmt(r['t1_total']):>12}  "
            f"{fmt(r['z_pepper']):>14} {fmt(r['z_osmium']):>12} {fmt(r['z_total']):>12}  "
            f"{fmt(r['d_pepper']):>10} {fmt(r['d_osmium']):>10} {fmt(r['d_total']):>10}"
        )
    w("-" * len(hdr))

    def agg(key: str) -> tuple[float, float]:
        return sum(x[key] for x in rows), sum(x[key] for x in rows) / len(rows)

    w("")
    w("Aggregates (sums then means over days)")
    w("-" * 120)
    for label, keys in (
        ("test1 pepper", ("t1_pepper",)),
        ("test1 osmium", ("t1_osmium",)),
        ("test1 total", ("t1_total",)),
        ("273774 pepper", ("z_pepper",)),
        ("273774 osmium", ("z_osmium",)),
        ("273774 total", ("z_total",)),
    ):
        k = keys[0]
        s, m = agg(k)
        w(f"  {label:16} sum {s:>14,.2f}   mean {m:>14,.2f}")

    w("")
    w("test1 minus 273774 — win counts (strictly > 0)")
    w("-" * 120)
    n = len(rows)
    w(f"  Days: {n}")
    w(f"  test1 wins pepper:  {sum(1 for r in rows if r['d_pepper'] > 0)} / {n}")
    w(f"  test1 wins osmium: {sum(1 for r in rows if r['d_osmium'] > 0)} / {n}")
    w(f"  test1 wins total:   {sum(1 for r in rows if r['d_total'] > 0)} / {n}")
    w(f"  tie pepper:  {sum(1 for r in rows if r['d_pepper'] == 0)} / {n}")
    w(f"  tie osmium: {sum(1 for r in rows if r['d_osmium'] == 0)} / {n}")
    w(f"  tie total:   {sum(1 for r in rows if r['d_total'] == 0)} / {n}")

    outp = args.report.expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
