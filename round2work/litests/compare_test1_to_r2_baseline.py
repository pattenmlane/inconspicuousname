#!/usr/bin/env python3
"""
Compare the **current** ``litests/test1.py`` ``Trader`` to a **frozen baseline**
of ``round2work/r2_submission.py`` on:

* All Prosperity4Data round 1 + 2 days
* Each Round 2 day 29 zip (``day 29 logs/``, ``extra/``)
* Combined day 29 merge

**One-time baseline (re-run when you change r2_submission or day-29 zips set):**

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt:$PWD/round2work/litests" \\
  python3 round2work/litests/compare_test1_to_r2_baseline.py --refresh-baseline

Writes ``round2work/litests/baseline_r2_submission_pnls.json``.

**Each test1 iteration (reloads ``test1`` from disk every run):**

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt:$PWD/round2work/litests" \\
  python3 round2work/litests/compare_test1_to_r2_baseline.py

Writes ``round2work/litests/test1_vs_r2_baseline_report.txt`` and prints a short summary.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
R2WORK = HERE.parent
REPO = R2WORK.parent

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"
ROUND_D29 = 2
DAY_D29 = 29

BASELINE_DEFAULT = HERE / "baseline_r2_submission_pnls.json"
REPORT_DEFAULT = HERE / "test1_vs_r2_baseline_report.txt"


def _bootstrap_bt() -> None:
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


def _load_r2_trader_class() -> type:
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


def _load_test1_trader_class() -> type:
    """Always reload ``test1`` so edits on disk are picked up."""
    _bootstrap_bt()
    _unload("test1")
    import test1 as m

    if not hasattr(m, "Trader"):
        raise AttributeError("test1.py: no Trader")
    return m.Trader


def _pnl_from_activities(pnl: dict[str, float]) -> dict[str, float]:
    return {
        "pepper": float(pnl.get(PEPPER, 0.0)),
        "osmium": float(pnl.get(OSMIUM, 0.0)),
        "total": float(sum(pnl.values())),
    }


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
    raw = {row.symbol: float(row.profit_loss) for row in result.final_activities()}
    return _pnl_from_activities(raw)


def collect_day29_zips() -> tuple[list[Path], list[Path]]:
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
            str(ROUND_D29),
            "--day",
            str(DAY_D29),
            "--out-dir",
            str(tmp),
        ],
        check=True,
    )
    r2 = dest_root / "ROUND_2"
    r2.mkdir(parents=True, exist_ok=True)
    prices = list(tmp.glob(f"prices_round_{ROUND_D29}_day_{DAY_D29}_*.csv"))
    trades = list(tmp.glob(f"trades_round_{ROUND_D29}_day_{DAY_D29}_*.csv"))
    if len(prices) != 1 or len(trades) != 1:
        raise RuntimeError(f"{zip_path}: export mismatch {prices=} {trades=}")
    shutil.copy(prices[0], r2 / f"prices_round_{ROUND_D29}_day_{DAY_D29}.csv")
    shutil.copy(trades[0], r2 / f"trades_round_{ROUND_D29}_day_{DAY_D29}.csv")
    shutil.rmtree(tmp)


def export_combined(combined_dir: Path, dest_root: Path) -> None:
    r2 = dest_root / "ROUND_2"
    r2.mkdir(parents=True, exist_ok=True)
    shutil.copy(combined_dir / "prices_combined_all_runs.csv", r2 / f"prices_round_{ROUND_D29}_day_{DAY_D29}.csv")
    shutil.copy(combined_dir / "trades_combined_all_runs.csv", r2 / f"trades_round_{ROUND_D29}_day_{DAY_D29}.csv")


def merge_all_zips(zips: Sequence[Path], out_dir: Path) -> None:
    subprocess.run(
        [sys.executable, str(R2WORK / "combine_submission_runs.py"), *[str(z) for z in zips], "--out-dir", str(out_dir)],
        check=True,
    )


def _run_day29_tape(trader_cls: type, zip_path: Path | None, combined_dir: Path) -> dict[str, float]:
    root = Path(tempfile.mkdtemp(prefix="t1r2_"))
    try:
        if zip_path is not None:
            export_zip_to_round2(zip_path, root)
        else:
            export_combined(combined_dir, root)
        return _run_day(trader_cls, root, ROUND_D29, DAY_D29)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def collect_all_tape_specs(data_root: Path, combined_dir: Path) -> list[dict[str, Any]]:
    _bootstrap_bt()
    from prosperity4bt.tools.data_reader import FileSystemReader

    reader = FileSystemReader(data_root)
    rows: list[dict[str, Any]] = []
    for rn in (1, 2):
        for d in sorted(reader.available_days(rn)):
            rows.append({"kind": "historical", "round": rn, "day": d})
    main_z, extra_z = collect_day29_zips()
    all_z = main_z + extra_z
    if not all_z:
        raise SystemExit("No day-29 zips under round2work/day 29 logs")
    comb_dir = combined_dir.resolve()
    comb_dir.mkdir(parents=True, exist_ok=True)
    merge_all_zips(all_z, comb_dir)
    for z in main_z:
        rows.append({"kind": "day29", "bucket": "day 29 logs", "stem": z.stem})
    for z in extra_z:
        rows.append({"kind": "day29", "bucket": "extra", "stem": z.stem})
    rows.append({"kind": "day29", "bucket": "combined", "stem": "ALL"})
    return rows


def _find_zip_for_stem(stem: str) -> Path:
    for folder in (R2WORK / "day 29 logs", R2WORK / "day 29 logs" / "extra"):
        p = folder / f"{stem}.zip"
        if p.is_file():
            return p
    raise FileNotFoundError(f"No zip for stem {stem}")


def run_tape_for_spec(trader_cls: type, spec: dict[str, Any], data_root: Path, combined_dir: Path) -> dict[str, float]:
    if spec["kind"] == "historical":
        return _run_day(trader_cls, data_root, int(spec["round"]), int(spec["day"]))
    if spec["bucket"] == "combined":
        return _run_day29_tape(trader_cls, None, combined_dir)
    zp = _find_zip_for_stem(str(spec["stem"]))
    return _run_day29_tape(trader_cls, zp, combined_dir)


def label_for_spec(spec: dict[str, Any]) -> str:
    if spec["kind"] == "historical":
        return f"R{spec['round']} day {spec['day']}"
    return f"{spec['bucket']}/{spec['stem']}"


def _is_individual_day29(spec: dict[str, Any]) -> bool:
    return spec.get("kind") == "day29" and spec.get("bucket") != "combined"


def build_baseline(data_root: Path, combined_dir: Path, baseline_path: Path, match: str) -> None:
    TraderR2 = _load_r2_trader_class()
    specs = collect_all_tape_specs(data_root, combined_dir)
    rows_out: list[dict[str, Any]] = []
    for spec in specs:
        pnl = run_tape_for_spec(TraderR2, spec, data_root, combined_dir)
        row = {**spec, **pnl}
        rows_out.append(row)

    payload = {
        "version": 1,
        "match": match,
        "r2_submission": str((R2WORK / "r2_submission.py").resolve()),
        "data_root": str(data_root.resolve()),
        "combined_dir": str(combined_dir.resolve()),
        "rows": rows_out,
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote baseline ({len(rows_out)} tapes): {baseline_path}")


def fmt(x: float) -> str:
    return f"{x:,.2f}"


def compare_to_baseline(data_root: Path, combined_dir: Path, baseline_path: Path, report_path: Path) -> None:
    if not baseline_path.is_file():
        raise SystemExit(
            f"Missing baseline {baseline_path}\n"
            f"Run: python3 {Path(__file__).name} --refresh-baseline"
        )
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    rows_base: list[dict[str, Any]] = payload["rows"]
    combined_dir = Path(payload.get("combined_dir", str(combined_dir))).expanduser().resolve()
    data_root = Path(payload.get("data_root", str(data_root))).expanduser().resolve()

    TraderTest1 = _load_test1_trader_class()

    lines: list[str] = []
    w = lines.append
    w("test1 (current litests/test1.py) vs stored r2_submission baseline")
    w("=" * 120)
    w(f"Baseline file: {baseline_path}")
    w(f"Match mode:    {payload.get('match', 'worse')}")
    w("")

    n_tapes = len(rows_base)
    r2_win_pepper = 0
    r2_win_osmium = 0

    hdr = (
        f"{'label':<28} "
        f"{'r2_pepper':>12} {'r2_os':>10} {'r2_tot':>10}  "
        f"{'t1_pepper':>12} {'t1_os':>10} {'t1_tot':>10}  "
        f"{'d_pep':>9} {'d_os':>9} {'d_tot':>9}"
    )
    sep = "-" * len(hdr)
    body: list[str] = []

    sum_r2_pe = sum_r2_os = sum_r2_tot = 0.0
    sum_t1_pe = sum_t1_os = sum_t1_tot = 0.0
    d29_r2_pe = d29_r2_os = d29_r2_tot = 0.0
    d29_t1_pe = d29_t1_os = d29_t1_tot = 0.0
    n_d29_indiv = 0
    for spec in rows_base:
        p_r2 = {k: float(spec[k]) for k in ("pepper", "osmium", "total")}
        sp = {k: spec[k] for k in spec if k in ("kind", "round", "day", "bucket", "stem")}
        p_t1 = run_tape_for_spec(TraderTest1, sp, data_root, combined_dir)
        lab = label_for_spec(spec)
        dp = p_t1["pepper"] - p_r2["pepper"]
        do = p_t1["osmium"] - p_r2["osmium"]
        dt = p_t1["total"] - p_r2["total"]
        if p_r2["pepper"] > p_t1["pepper"]:
            r2_win_pepper += 1
        if p_r2["osmium"] > p_t1["osmium"]:
            r2_win_osmium += 1
        sum_r2_pe += p_r2["pepper"]
        sum_r2_os += p_r2["osmium"]
        sum_r2_tot += p_r2["total"]
        sum_t1_pe += p_t1["pepper"]
        sum_t1_os += p_t1["osmium"]
        sum_t1_tot += p_t1["total"]
        if _is_individual_day29(spec):
            n_d29_indiv += 1
            d29_r2_pe += p_r2["pepper"]
            d29_r2_os += p_r2["osmium"]
            d29_r2_tot += p_r2["total"]
            d29_t1_pe += p_t1["pepper"]
            d29_t1_os += p_t1["osmium"]
            d29_t1_tot += p_t1["total"]
        body.append(
            f"{lab:<28} "
            f"{fmt(p_r2['pepper']):>12} {fmt(p_r2['osmium']):>10} {fmt(p_r2['total']):>10}  "
            f"{fmt(p_t1['pepper']):>12} {fmt(p_t1['osmium']):>10} {fmt(p_t1['total']):>10}  "
            f"{fmt(dp):>9} {fmt(do):>9} {fmt(dt):>9}"
        )

    w("# Per-tape head-to-head (baseline r2_submission PnL > test1 PnL, strict >)")
    w(f"#   pepper: {r2_win_pepper} / {n_tapes}")
    w(f"#   osmium: {r2_win_osmium} / {n_tapes}")
    w("")
    w(hdr)
    w(sep)
    for ln in body:
        w(ln)
    w(sep)

    w("")
    w("Sums (all tapes in baseline)")
    w("-" * 80)

    w(f"  r2:  pepper {fmt(sum_r2_pe):>14}  osmium {fmt(sum_r2_os):>14}  total {fmt(sum_r2_tot):>14}")
    w(f"  t1:  pepper {fmt(sum_t1_pe):>14}  osmium {fmt(sum_t1_os):>14}  total {fmt(sum_t1_tot):>14}")
    w(f"  Δ(t1−r2): pepper {fmt(sum_t1_pe - sum_r2_pe):>10}  osmium {fmt(sum_t1_os - sum_r2_os):>10}  total {fmt(sum_t1_tot - sum_r2_tot):>10}")

    w("")
    w(f"Sums (individual day-29 zips only, {n_d29_indiv} zips; excludes combined/ALL and historical)")
    w("-" * 80)
    w(f"  r2:  pepper {fmt(d29_r2_pe):>14}  osmium {fmt(d29_r2_os):>14}  total {fmt(d29_r2_tot):>14}")
    w(f"  t1:  pepper {fmt(d29_t1_pe):>14}  osmium {fmt(d29_t1_os):>14}  total {fmt(d29_t1_tot):>14}")
    w(f"  Δ(t1−r2): pepper {fmt(d29_t1_pe - d29_r2_pe):>10}  osmium {fmt(d29_t1_os - d29_r2_os):>10}  total {fmt(d29_t1_tot - d29_r2_tot):>10}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh-baseline", action="store_true", help="Run r2_submission once and write JSON baseline")
    ap.add_argument("--data", type=Path, default=REPO / "Prosperity4Data")
    ap.add_argument("--combined-dir", type=Path, default=R2WORK / "day 29 logs" / "combined_all_including_extra")
    ap.add_argument("--baseline", type=Path, default=BASELINE_DEFAULT)
    ap.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    args = ap.parse_args()

    data_root = args.data.expanduser().resolve()
    comb = args.combined_dir.expanduser().resolve()

    if args.refresh_baseline:
        build_baseline(data_root, comb, args.baseline, "worse")
        return

    compare_to_baseline(data_root, comb, args.baseline, args.report)


if __name__ == "__main__":
    main()
