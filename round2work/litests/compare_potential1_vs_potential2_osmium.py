#!/usr/bin/env python3
"""
Full-suite osmium PnL: ``potential1.Trader`` vs ``potential2.Trader``.

Same tapes as ``compare_test1_to_r2_baseline.py`` (R1+R2 Prosperity4Data,
every day-29 zip + combined). Runs each **full** submission (both products);
report is **osmium-only** columns and deltas.

From repo root::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt:$PWD/round2work/litests" \\
  python3 round2work/litests/compare_potential1_vs_potential2_osmium.py

Writes ``round2work/litests/potential1_vs_potential2_osmium_report.txt``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import sys
from pathlib import Path

import compare_test1_to_r2_baseline as bt

HERE = Path(__file__).resolve().parent
REPORT_DEFAULT = HERE / "potential1_vs_potential2_osmium_report.txt"
# ``run_tape_for_spec`` returns keys pepper / osmium / total (not symbol strings).
OSM_KEY = "osmium"


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


def fmt(x: float) -> str:
    return f"{x:,.2f}"


@contextlib.contextmanager
def _silence_stdout():
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        yield
    finally:
        sys.stdout = old


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=bt.REPO / "Prosperity4Data")
    ap.add_argument("--combined-dir", type=Path, default=bt.R2WORK / "day 29 logs" / "combined_all_including_extra")
    ap.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    ap.add_argument("--mod-a", type=str, default="potential1", help="Python module name (no .py)")
    ap.add_argument("--mod-b", type=str, default="potential2", help="Python module name (no .py)")
    args = ap.parse_args()

    data_root = args.data.expanduser().resolve()
    comb = args.combined_dir.expanduser().resolve()
    ma, mb = args.mod_a, args.mod_b

    specs = bt.collect_all_tape_specs(data_root, comb)
    TraderA = _load_trader(ma)
    TraderB = _load_trader(mb)

    lines: list[str] = []
    w = lines.append
    w(f"osmium-only PnL: {ma} vs {mb} (full Trader backtest, match=worse)")
    w("=" * 100)
    w(f"data: {data_root}")
    w(f"combined dir: {comb}")
    w("")

    sum_a = sum_b = 0.0
    win_a = win_b = 0
    d29_a = d29_b = 0.0
    n_d29 = 0

    hdr = f"{'label':<30} {('osm_'+ma)[:12]:>12} {('osm_'+mb)[:12]:>12} {'d_'+mb+'-'+ma:>12}"
    w(hdr)
    w("-" * len(hdr))
    for spec in specs:
        sp = {k: spec[k] for k in spec if k in ("kind", "round", "day", "bucket", "stem")}
        with _silence_stdout():
            pa = bt.run_tape_for_spec(TraderA, sp, data_root, comb)[OSM_KEY]
        with _silence_stdout():
            pb = bt.run_tape_for_spec(TraderB, sp, data_root, comb)[OSM_KEY]
        lab = bt.label_for_spec(spec)
        d = pb - pa
        sum_a += pa
        sum_b += pb
        if pa > pb:
            win_a += 1
        elif pb > pa:
            win_b += 1
        w(f"{lab:<30} {fmt(pa):>12} {fmt(pb):>12} {fmt(d):>12}")
        if bt._is_individual_day29(spec):
            n_d29 += 1
            d29_a += pa
            d29_b += pb

    w("-" * len(hdr))
    w("")
    w(f"# Per-tape wins on osmium (strict >): {ma} {win_a} / {len(specs)}, {mb} {win_b} / {len(specs)}")
    w("")
    w("Sums (all tapes)")
    w("-" * 60)
    w(f"  {ma:12}  osmium {fmt(sum_a):>14}")
    w(f"  {mb:12}  osmium {fmt(sum_b):>14}")
    w(f"  Δ({mb}−{ma})   osmium {fmt(sum_b - sum_a):>14}")
    w("")
    w(f"Sums (individual day-29 zips only, {n_d29} zips)")
    w("-" * 60)
    w(f"  {ma:12}  osmium {fmt(d29_a):>14}")
    w(f"  {mb:12}  osmium {fmt(d29_b):>14}")
    w(f"  Δ({mb}−{ma})   osmium {fmt(d29_b - d29_a):>14}")

    text = "\n".join(lines) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
