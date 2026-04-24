#!/usr/bin/env python3
"""
Backtest **full submission traders** from ``269993.zip`` (#1 osmium) and
``273774.zip`` (#2 osmium tie) on **every** day under ``Prosperity4Data`` that
``FileSystemReader`` exposes for **round 1** and **round 2**.

Output is **per-symbol PnL** (pepper vs osmium) plus totals. The two zips use
**different pepper logic** (269993: slope-based targeting; 273774: 70/10 drift
MM), so the **osmium** column is the clean read for “os strat #1 vs #2” on the
same tape; pepper is included for transparency.

Writes ``R1submissionlogs/osstrat_269993_vs_273774_all_historical.txt``.

From repo root::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 R1submissionlogs/compare_osstrat_269993_vs_273774_all_days.py
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
STEM_A = "269993"  # #1 osmium (touch + WM freeze)
STEM_B = "273774"  # #2 osmium (WM freeze only)

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


def _load_trader_class(zip_path: Path) -> type:
    stem = zip_path.stem
    work = Path(tempfile.mkdtemp(prefix="oscmp_"))
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
        "--report",
        type=Path,
        default=HERE / "osstrat_269993_vs_273774_all_historical.txt",
    )
    ap.add_argument("--match", choices=("worse", "all"), default="worse")
    args = ap.parse_args()

    data_root = args.data.expanduser().resolve()
    zip_a = HERE / f"{STEM_A}.zip"
    zip_b = HERE / f"{STEM_B}.zip"
    for z in (zip_a, zip_b):
        if not z.is_file():
            raise SystemExit(f"Missing {z}")

    _bootstrap_paths()
    from prosperity4bt.tools.data_reader import FileSystemReader

    reader = FileSystemReader(data_root)
    days: list[tuple[int, int]] = []
    for rn in (1, 2):
        for d in reader.available_days(rn):
            days.append((rn, d))
    days.sort()

    # Load each Trader class once (same class for all days)
    TraderA = _load_trader_class(zip_a)
    TraderB = _load_trader_class(zip_b)

    rows: list[dict] = []
    for rn, d in days:
        pa = _run_day(TraderA, data_root, rn, d, args.match)
        pb = _run_day(TraderB, data_root, rn, d, args.match)
        oa = pa.get(OSMIUM, 0.0)
        ob = pb.get(OSMIUM, 0.0)
        rows.append(
            {
                "round": rn,
                "day": d,
                "a_pepper": pa.get(PEPPER, 0.0),
                "a_osmium": oa,
                "a_total": sum(pa.values()),
                "b_pepper": pb.get(PEPPER, 0.0),
                "b_osmium": ob,
                "b_total": sum(pb.values()),
                "d_osmium": oa - ob,
            }
        )

    def fmt(x: float) -> str:
        return f"{x:,.2f}"

    lines: list[str] = []
    w = lines.append
    w("Osmium-focused comparison: 269993 (#1) vs 273774 (#2) — all Prosperity4Data days")
    w("=" * 96)
    w("")
    w(f"Data root: {data_root}")
    w(f"Trade matching: {args.match}")
    w(f"269993 zip: {zip_a.name}  (WM spike freeze + touch stress freeze + width boost)")
    w(f"273774 zip: {zip_b.name}  (WM spike freeze only; 70/10 pepper)")
    w("")
    w("NOTE: Each row runs the **entire** Trader from that zip. Pepper logic differs,")
    w("      so use **osmium** columns for os-strategy comparison; pepper is diagnostic.")
    w("")

    hdr = (
        f"{'R':>2} {'day':>5}  "
        f"{STEM_A+'_pepper':>14} {STEM_A+'_os':>12} {STEM_A+'_tot':>12}  "
        f"{STEM_B+'_pepper':>14} {STEM_B+'_os':>12} {STEM_B+'_tot':>12}  "
        f"{'d_osm':>10}"
    )
    w(hdr)
    w("-" * len(hdr))
    for r in rows:
        w(
            f"{r['round']:>2} {r['day']:>5}  "
            f"{fmt(r['a_pepper']):>14} {fmt(r['a_osmium']):>12} {fmt(r['a_total']):>12}  "
            f"{fmt(r['b_pepper']):>14} {fmt(r['b_osmium']):>12} {fmt(r['b_total']):>12}  "
            f"{fmt(r['d_osmium']):>10}"
        )
    w("-" * len(hdr))

    aos = [r["a_osmium"] for r in rows]
    bos = [r["b_osmium"] for r in rows]
    d = [r["d_osmium"] for r in rows]
    w("")
    w("Aggregate — osmium only")
    w("-" * 96)
    w(f"  Days:                    {len(rows)}")
    w(f"  Mean {STEM_A} osmium:     {sum(aos)/len(aos):>14,.2f}")
    w(f"  Mean {STEM_B} osmium:     {sum(bos)/len(bos):>14,.2f}")
    w(f"  Mean (269993 − 273774):  {sum(d)/len(d):>14,.2f}")
    w(f"  Sum osmium {STEM_A}:      {sum(aos):>14,.2f}")
    w(f"  Sum osmium {STEM_B}:      {sum(bos):>14,.2f}")
    w(f"  269993 wins (osm):       {sum(1 for x in d if x > 0)} / {len(d)} days")
    w(f"  273774 wins (osm):       {sum(1 for x in d if x < 0)} / {len(d)} days")
    w(f"  Tie (osm):               {sum(1 for x in d if x == 0)} / {len(d)} days")

    out = args.report.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
