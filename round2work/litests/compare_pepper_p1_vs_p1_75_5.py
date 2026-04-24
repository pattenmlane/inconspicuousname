#!/usr/bin/env python3
"""Compare **pepper** PnL only: ``potential1`` (80 / take 7) vs ``potential1_pepper_75_5`` (75 / take 5)."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import sys
from pathlib import Path

import compare_test1_to_r2_baseline as bt

HERE = Path(__file__).resolve().parent
BASELINE_DEFAULT = HERE / "baseline_r2_submission_pnls.json"
REPORT_DEFAULT = HERE / "pepper_p1_vs_p1_75_5_report.txt"


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
    ap.add_argument("--baseline", type=Path, default=BASELINE_DEFAULT)
    ap.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    args = ap.parse_args()

    bp = args.baseline.expanduser().resolve()
    if not bp.is_file():
        raise SystemExit(f"Missing {bp}")
    payload = json.loads(bp.read_text(encoding="utf-8"))
    rows = payload["rows"]
    data_root = Path(payload["data_root"]).expanduser().resolve()
    combined_dir = Path(payload["combined_dir"]).expanduser().resolve()

    T1 = _load_trader("potential1")
    T75 = _load_trader("potential1_pepper_75_5")

    lines: list[str] = []
    w = lines.append
    w("Pepper PnL: potential1 (80 cap, take 7) vs potential1_pepper_75_5 (75 cap, take 5)")
    w("=" * 90)

    s1 = s75 = 0.0
    w1 = w75 = 0
    hdr = f"{'label':<32} {'p1_pepper':>14} {'p75_pepper':>14} {'d(75-1)':>12}"
    w(hdr)
    w("-" * len(hdr))
    for row in rows:
        sp = {k: row[k] for k in ("kind", "round", "day", "bucket", "stem") if k in row}
        with _silence_stdout():
            p1 = bt.run_tape_for_spec(T1, sp, data_root, combined_dir)["pepper"]
        with _silence_stdout():
            p75 = bt.run_tape_for_spec(T75, sp, data_root, combined_dir)["pepper"]
        lab = bt.label_for_spec(row)
        d = p75 - p1
        s1 += p1
        s75 += p75
        if p1 > p75:
            w1 += 1
        elif p75 > p1:
            w75 += 1
        w(f"{lab:<32} {fmt(p1):>14} {fmt(p75):>14} {fmt(d):>12}")
    w("-" * len(hdr))
    w("")
    w(f"# Strict pepper wins: potential1 {w1}/{len(rows)}, 75/5 {w75}/{len(rows)}")
    w("")
    w("Sums (all tapes)")
    w(f"  potential1 pepper           {fmt(s1):>14}")
    w(f"  potential1_pepper_75_5      {fmt(s75):>14}")
    w(f"  Δ (75/5 − p1)               {fmt(s75 - s1):>14}")

    text = "\n".join(lines) + "\n"
    args.report.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
