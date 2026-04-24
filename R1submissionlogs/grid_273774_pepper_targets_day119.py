#!/usr/bin/env python3
"""
Grid over **PEPPER_TARGET_LONG** (with ``PEPPER_POSITION_LIMIT = 80``) for the
exact 273774 submission trader (``replay_273774_submission.py`` from zip).

Pairs (target / buffer): 0/80, 50/30, 60/20, 65/15, 70/10, 75/5, 80/0.

* Round **1**, day **119** (``Prosperity4Data/ROUND1/prices_round_1_day_119.csv``)
* ``TradeMatchingMode.worse``

Run from repo root::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 R1submissionlogs/grid_273774_pepper_targets_day119.py
"""
from __future__ import annotations

import importlib.util
import re
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "submissions" / "replay_273774_submission.py"
ZIP_FALLBACK = REPO / "R1submissionlogs" / "273774.zip"
OUT_DIR = REPO / "submissions" / "_grid_tmp"
DATA = REPO / "Prosperity4Data"

# (target_long, buffer) — buffer = 80 - target for display only
GRID = [(0, 80), (50, 30), (60, 20), (65, 15), (70, 10), (75, 5), (80, 0)]


def load_template() -> str:
    if TEMPLATE.is_file():
        return TEMPLATE.read_text(encoding="utf-8")
    if ZIP_FALLBACK.is_file():
        with zipfile.ZipFile(ZIP_FALLBACK) as zf:
            return zf.read("273774.py").decode("utf-8")
    raise SystemExit(f"Need {TEMPLATE} or {ZIP_FALLBACK}")


def inject_target_long(src: str, target_long: int) -> str:
    new, n = re.subn(
        r"^PEPPER_TARGET_LONG = \d+\s*$",
        f"PEPPER_TARGET_LONG = {target_long}",
        src,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise SystemExit("Could not find single PEPPER_TARGET_LONG = … line to patch")
    return new


def run_backtest(module_path: Path) -> tuple[float, float, float]:
    name = f"_grid_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("spec_from_file_location failed")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    from prosperity4bt.models.test_options import TradeMatchingMode
    from prosperity4bt.test_runner import TestRunner
    from prosperity4bt.tools.data_reader import FileSystemReader

    dr = FileSystemReader(DATA)
    runner = TestRunner(
        mod.Trader(),
        dr,
        1,
        119,
        show_progress_bar=False,
        print_output=False,
        trade_matching_mode=TradeMatchingMode.worse,
    )
    result = runner.run()
    final = result.final_activities()
    by_sym = {a.symbol: float(a.profit_loss) for a in final}
    pe = by_sym.get("INTARIAN_PEPPER_ROOT", 0.0)
    os = by_sym.get("ASH_COATED_OSMIUM", 0.0)
    return pe, os, pe + os


def main() -> None:
    src0 = load_template()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[int, int, float, float, float]] = []
    for tgt, buf in GRID:
        text = inject_target_long(src0, tgt)
        path = OUT_DIR / f"replay_273774_day119_tl{tgt}.py"
        path.write_text(text, encoding="utf-8")
        pe, osm, tot = run_backtest(path)
        rows.append((tgt, buf, pe, osm, tot))
        mod_name = f"_grid_{path.stem}"
        sys.modules.pop(mod_name, None)

    rows.sort(key=lambda r: -r[4])
    print("273774 replay | round 1 day 119 | match_trades worse\n")
    print(f"{'tgt':>3} {'buf':>3}  {'pepper':>12}  {'osmium':>10}  {'total':>12}")
    print("-" * 48)
    for tgt, buf, pe, osm, tot in sorted(rows, key=lambda r: (r[0], r[1])):
        print(f"{tgt:>3} {buf:>3}  {pe:>12,.0f}  {osm:>10,.0f}  {tot:>12,.0f}")
    print("-" * 48)
    print("Best total:")
    tgt, buf, pe, osm, tot = rows[0]
    print(f"  target={tgt} buffer={buf}  pepper={pe:,.0f}  osmium={osm:,.0f}  total={tot:,.0f}")
    print(f"\nTemp traders under {OUT_DIR}/ (safe to delete).")


if __name__ == "__main__":
    main()
