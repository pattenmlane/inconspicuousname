#!/usr/bin/env python3
"""
Compare **potential1**, **potential2**, and the **frozen r2 baseline** JSON
(``baseline_r2_submission_pnls.json``) on the **same tapes** as that baseline.

Reports **pepper**, **osmium**, and **total** PnL for:

* All tapes in the baseline (typically 27)
* **R1 day 19** only
* **R1 day 119** only
* Sum over **individual day-29 zips** only (excludes ``combined/ALL`` and historical)

Runs ``potential1`` / ``potential2`` full ``Trader`` backtests (``match_trades=worse``).
Baseline values are read from the JSON (no re-sim of ``r2_submission``).

Requires an existing baseline file (from ``compare_test1_to_r2_baseline.py --refresh-baseline``).

From repo root::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt:$PWD/round2work/litests" \\
  python3 round2work/litests/compare_potential1_potential2_baseline.py

Writes ``round2work/litests/potential1_potential2_baseline_compare.txt``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import compare_test1_to_r2_baseline as bt

HERE = Path(__file__).resolve().parent
BASELINE_DEFAULT = HERE / "baseline_r2_submission_pnls.json"
REPORT_DEFAULT = HERE / "potential1_potential2_baseline_compare.txt"


@dataclass
class PnL:
    pepper: float = 0.0
    osmium: float = 0.0
    total: float = 0.0

    def add(self, d: dict[str, float]) -> None:
        self.pepper += float(d["pepper"])
        self.osmium += float(d["osmium"])
        self.total += float(d["total"])


def fmt(x: float) -> str:
    return f"{x:,.2f}"


def _unload(name: str) -> None:
    if name in sys.modules:
        del sys.modules[name]
    for k in list(sys.modules):
        if k.startswith(name + "."):
            del sys.modules[k]


def _load_trader(module: str) -> type:
    bt._bootstrap_bt()
    _unload(module)
    m = importlib.import_module(module)
    if not hasattr(m, "Trader"):
        raise AttributeError(f"{module}: no Trader")
    return m.Trader


@contextlib.contextmanager
def _silence_stdout() -> Any:
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def _spec_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row[k] for k in ("kind", "round", "day", "bucket", "stem") if k in row}


def _baseline_pnls(row: dict[str, Any]) -> dict[str, float]:
    return {
        "pepper": float(row["pepper"]),
        "osmium": float(row["osmium"]),
        "total": float(row["total"]),
    }


def _is_r1_day(row: dict[str, Any], day: int) -> bool:
    return row.get("kind") == "historical" and int(row.get("round", 0)) == 1 and int(row.get("day", 0)) == day


def _emit_block(
    w: Any,
    title: str,
    b: PnL,
    p1: PnL,
    p2: PnL,
) -> None:
    w("")
    w(title)
    w("-" * min(100, len(title) + 40))
    hdr = f"{'':22} {'pepper':>14} {'osmium':>14} {'total':>14}"
    w(hdr)
    w(f"{'baseline (r2 JSON)':22} {fmt(b.pepper):>14} {fmt(b.osmium):>14} {fmt(b.total):>14}")
    w(f"{'potential1':22} {fmt(p1.pepper):>14} {fmt(p1.osmium):>14} {fmt(p1.total):>14}")
    w(f"{'potential2':22} {fmt(p2.pepper):>14} {fmt(p2.osmium):>14} {fmt(p2.total):>14}")
    w(f"{'Δ potential1 − baseline':22} {fmt(p1.pepper - b.pepper):>14} {fmt(p1.osmium - b.osmium):>14} {fmt(p1.total - b.total):>14}")
    w(f"{'Δ potential2 − baseline':22} {fmt(p2.pepper - b.pepper):>14} {fmt(p2.osmium - b.osmium):>14} {fmt(p2.total - b.total):>14}")
    w(f"{'Δ potential2 − potential1':22} {fmt(p2.pepper - p1.pepper):>14} {fmt(p2.osmium - p1.osmium):>14} {fmt(p2.total - p1.total):>14}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, default=BASELINE_DEFAULT)
    ap.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    ap.add_argument("--mod-p1", type=str, default="potential1")
    ap.add_argument("--mod-p2", type=str, default="potential2")
    args = ap.parse_args()

    bp = args.baseline.expanduser().resolve()
    if not bp.is_file():
        raise SystemExit(f"Missing baseline JSON: {bp}\nRun compare_test1_to_r2_baseline.py --refresh-baseline first.")

    payload = json.loads(bp.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = payload["rows"]
    data_root = Path(payload["data_root"]).expanduser().resolve()
    combined_dir = Path(payload["combined_dir"]).expanduser().resolve()

    Trader1 = _load_trader(args.mod_p1)
    Trader2 = _load_trader(args.mod_p2)

    all_b = PnL()
    all_p1 = PnL()
    all_p2 = PnL()
    d19_b = PnL()
    d19_p1 = PnL()
    d19_p2 = PnL()
    d119_b = PnL()
    d119_p1 = PnL()
    d119_p2 = PnL()
    d29_b = PnL()
    d29_p1 = PnL()
    d29_p2 = PnL()

    lines: list[str] = []
    w = lines.append
    w("potential1 vs potential2 vs baseline (r2_submission JSON)")
    w("=" * 100)
    w(f"baseline file: {bp}")
    w(f"match (baseline meta): {payload.get('match', '?')}")
    w(f"data_root: {data_root}")
    w(f"combined_dir: {combined_dir}")
    w(f"tapes: {len(rows)}")

    for row in rows:
        spec = _spec_from_row(row)
        base = _baseline_pnls(row)
        with _silence_stdout():
            p1 = bt.run_tape_for_spec(Trader1, spec, data_root, combined_dir)
        with _silence_stdout():
            p2 = bt.run_tape_for_spec(Trader2, spec, data_root, combined_dir)

        all_b.add(base)
        all_p1.add(p1)
        all_p2.add(p2)

        if _is_r1_day(row, 19):
            d19_b.add(base)
            d19_p1.add(p1)
            d19_p2.add(p2)
        if _is_r1_day(row, 119):
            d119_b.add(base)
            d119_p1.add(p1)
            d119_p2.add(p2)
        if bt._is_individual_day29(spec):
            d29_b.add(base)
            d29_p1.add(p1)
            d29_p2.add(p2)

    _emit_block(w, f"ALL TAPES ({len(rows)} rows)", all_b, all_p1, all_p2)
    _emit_block(w, "R1 DAY 19 ONLY", d19_b, d19_p1, d19_p2)
    _emit_block(w, "R1 DAY 119 ONLY", d119_b, d119_p1, d119_p2)
    n_d29 = sum(1 for r in rows if bt._is_individual_day29(_spec_from_row(r)))
    _emit_block(w, f"INDIVIDUAL DAY-29 ZIPS ONLY (sum of {n_d29} zips)", d29_b, d29_p1, d29_p2)

    text = "\n".join(lines) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
