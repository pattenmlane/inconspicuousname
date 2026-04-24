#!/usr/bin/env python3
"""
Compare **pepper** PnL: ``round2work/r2_submission_target75.py`` (PEPPER_TARGET_LONG=75)
vs ``potential1`` on the same tapes as ``baseline_r2_submission_pnls.json``.

Uses paths from that JSON (no need to match frozen baseline PnL).

From repo root::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt:$PWD/round2work/litests" \\
  python3 round2work/litests/compare_r2_target75_vs_potential1_pepper.py
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import sys
import types
from pathlib import Path

import compare_test1_to_r2_baseline as bt

HERE = Path(__file__).resolve().parent
R2WORK = HERE.parent
BASELINE_DEFAULT = HERE / "baseline_r2_submission_pnls.json"
REPORT_DEFAULT = HERE / "r2_target75_vs_potential1_pepper_report.txt"


def _unload(name: str) -> None:
    if name in sys.modules:
        del sys.modules[name]
    for k in list(sys.modules):
        if k.startswith(name + "."):
            del sys.modules[k]


def _load_r2_module_trader(module_name: str) -> type:
    """``module_name`` is e.g. ``r2_submission_target75`` (file under ``round2work/``)."""
    dm = types.ModuleType("datamodel")
    bt._bootstrap_bt()
    from prosperity4bt import datamodel as _dm

    for n in ("Order", "OrderDepth", "TradingState"):
        setattr(dm, n, getattr(_dm, n))
    sys.modules["datamodel"] = dm
    sys.path.insert(0, str(R2WORK))
    try:
        _unload(module_name)
        mod = importlib.import_module(module_name)
        if not hasattr(mod, "Trader"):
            raise AttributeError(f"{module_name}: no Trader")
        return mod.Trader
    finally:
        sys.path.remove(str(R2WORK))


def _load_potential1() -> type:
    bt._bootstrap_bt()
    _unload("potential1")
    import potential1 as m

    return m.Trader


@contextlib.contextmanager
def _silence_stdout():
    o, sys.stdout = sys.stdout, io.StringIO()
    try:
        yield
    finally:
        sys.stdout = o


def fmt(x: float) -> str:
    return f"{x:,.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, default=BASELINE_DEFAULT, help="Tape list + paths only")
    ap.add_argument("--r2-module", type=str, default="r2_submission_target75")
    ap.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    args = ap.parse_args()

    bp = args.baseline.expanduser().resolve()
    if not bp.is_file():
        raise SystemExit(f"Missing {bp}")
    payload = json.loads(bp.read_text(encoding="utf-8"))
    rows = payload["rows"]
    data_root = Path(payload["data_root"]).expanduser().resolve()
    combined_dir = Path(payload["combined_dir"]).expanduser().resolve()

    Tr2 = _load_r2_module_trader(args.r2_module)
    Tp1 = _load_potential1()

    lines: list[str] = []
    w = lines.append
    w(f"Pepper PnL: {args.r2_module} (target long 75) vs potential1")
    w("=" * 90)

    sr2 = sp1 = 0.0
    wr2 = wp1 = 0
    hdr = f"{'label':<32} {'r2_pepper':>14} {'p1_pepper':>14} {'d(p1-r2)':>12}"
    w(hdr)
    w("-" * len(hdr))
    for row in rows:
        sp = {k: row[k] for k in ("kind", "round", "day", "bucket", "stem") if k in row}
        with _silence_stdout():
            pr2 = bt.run_tape_for_spec(Tr2, sp, data_root, combined_dir)["pepper"]
        with _silence_stdout():
            pp1 = bt.run_tape_for_spec(Tp1, sp, data_root, combined_dir)["pepper"]
        lab = bt.label_for_spec(row)
        d = pp1 - pr2
        sr2 += pr2
        sp1 += pp1
        if pr2 > pp1:
            wr2 += 1
        elif pp1 > pr2:
            wp1 += 1
        w(f"{lab:<32} {fmt(pr2):>14} {fmt(pp1):>14} {fmt(d):>12}")
    w("-" * len(hdr))
    w("")
    w(f"# Strict pepper wins: {args.r2_module} {wr2}/{len(rows)}, potential1 {wp1}/{len(rows)}")
    w("")
    w("Sums (all tapes)")
    w(f"  {args.r2_module:24} {fmt(sr2):>14}")
    w(f"  {'potential1':24} {fmt(sp1):>14}")
    w(f"  {'Δ (p1 − r2)':24} {fmt(sp1 - sr2):>14}")

    text = "\n".join(lines) + "\n"
    args.report.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
