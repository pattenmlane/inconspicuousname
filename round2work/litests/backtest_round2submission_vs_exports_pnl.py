#!/usr/bin/env python3
"""
Verify ``round2submission/*.py`` matches **PnL** of export originals in ``round2work/``:

* ``potential2_pepper_only.py`` vs ``submission_exports_pepper_variants/``
* ``potential3_osmium_only.py`` vs ``submission_exports_p2_variants/``

Same tape set as ``backtest_p2_pepper_robustness_pnl_check.py`` (Prosperity4Data
R1+R2 days, all day-29 zips + combined). ``match_trades=worse``.

From repo root::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt:$PWD/round2work/litests" \\
  python3 round2work/litests/backtest_round2submission_vs_exports_pnl.py
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
R2WORK = HERE.parent
REPO = R2WORK.parent

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"
ROUND = 2
DAY = 29


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


def _unload_module(prefix: str) -> None:
    for k in list(sys.modules):
        if k == prefix or k.startswith(prefix + "."):
            del sys.modules[k]


def load_trader_from_path(py_path: Path, module_name: str) -> type:
    _unload_module(module_name)
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "Trader"):
        raise AttributeError(f"{py_path}: no Trader")
    return mod.Trader  # type: ignore[attr-defined]


def _run_day(
    trader_cls: type,
    data_root: Path,
    round_n: int,
    day_n: int,
) -> dict[str, float]:
    _bootstrap_paths()
    from prosperity4bt.models.test_options import TradeMatchingMode
    from prosperity4bt.test_runner import TestRunner
    from prosperity4bt.tools.data_reader import FileSystemReader

    reader = FileSystemReader(data_root)
    if day_n not in reader.available_days(round_n):
        raise ValueError(f"R{round_n} day {day_n} not in {data_root}: {reader.available_days(round_n)}")
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
        capture_output=True,
        text=True,
    )
    r2 = dest_root / "ROUND_2"
    r2.mkdir(parents=True, exist_ok=True)
    prices = list(tmp.glob(f"prices_round_{ROUND}_day_{DAY}_*.csv"))
    trades = list(tmp.glob(f"trades_round_{ROUND}_day_{DAY}_*.csv"))
    if len(prices) != 1 or len(trades) != 1:
        raise RuntimeError(f"{zip_path}: expected one prices/trades export, got {prices=} {trades=}")
    shutil.copy(prices[0], r2 / f"prices_round_{ROUND}_day_{DAY}.csv")
    shutil.copy(trades[0], r2 / f"trades_round_{ROUND}_day_{DAY}.csv")
    shutil.rmtree(tmp)


def export_combined(combined_dir: Path, dest_root: Path) -> None:
    r2 = dest_root / "ROUND_2"
    r2.mkdir(parents=True, exist_ok=True)
    p = combined_dir / "prices_combined_all_runs.csv"
    t = combined_dir / "trades_combined_all_runs.csv"
    shutil.copy(p, r2 / f"prices_round_{ROUND}_day_{DAY}.csv")
    shutil.copy(t, r2 / f"trades_round_{ROUND}_day_{DAY}.csv")


def run_day29_tape(
    trader_cls: type,
    zip_path: Path | None,
    combined_dir: Path,
) -> dict[str, float]:
    root = Path(tempfile.mkdtemp(prefix="r2subpar_"))
    try:
        if zip_path is not None:
            export_zip_to_round2(zip_path, root)
        else:
            export_combined(combined_dir, root)
        return _run_day(trader_cls, root, ROUND, DAY)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def collect_day29_zips() -> tuple[list[Path], list[Path]]:
    main = sorted(p for p in (R2WORK / "day 29 logs").glob("*.zip") if p.is_file())
    extra = sorted(p for p in (R2WORK / "day 29 logs" / "extra").glob("*.zip") if p.is_file())
    return main, extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=REPO / "Prosperity4Data")
    ap.add_argument(
        "--combined-dir",
        type=Path,
        default=R2WORK / "day 29 logs" / "combined_all_including_extra",
    )
    ap.add_argument("--skip-day29", action="store_true")
    ap.add_argument("--skip-historical", action="store_true")
    args = ap.parse_args()

    data_root = args.data.expanduser().resolve()
    comb_dir = args.combined_dir.expanduser().resolve()

    pairs: list[tuple[str, Path, Path, str]] = [
        (
            "pepper",
            REPO / "round2submission" / "potential2_pepper_only.py",
            R2WORK / "submission_exports_pepper_variants" / "potential2_pepper_only.py",
            PEPPER,
        ),
        (
            "osmium",
            REPO / "round2submission" / "potential3_osmium_only.py",
            R2WORK / "submission_exports_p2_variants" / "potential3_osmium_only.py",
            OSMIUM,
        ),
    ]

    failures: list[str] = []
    rows: list[tuple[str, str, float, float, float, float]] = []

    def check_one(
        suite: str,
        product: str,
        run_sub: Callable[[type], dict[str, float]],
        run_exp: Callable[[type], dict[str, float]],
        TraderSub: type,
        TraderExp: type,
    ) -> None:
        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            d_sub = run_sub(TraderSub)
            d_exp = run_exp(TraderExp)
        finally:
            sys.stdout = old
        ps, pe = float(d_sub.get(product, 0.0)), float(d_exp.get(product, 0.0))
        ts, te = float(sum(d_sub.values())), float(sum(d_exp.values()))
        rows.append((suite, product, ps, pe, ts, te))
        if ps != pe or ts != te:
            failures.append(
                f"{suite} [{product}]: submission prod={ps} export prod={pe} | sub_tot={ts} exp_tot={te}"
            )

    for name, p_sub, p_exp, product in pairs:
        if not p_sub.is_file():
            raise SystemExit(f"Missing {p_sub}")
        if not p_exp.is_file():
            raise SystemExit(f"Missing {p_exp}")

        mod_sub = f"_r2sub_{name}_submission"
        mod_exp = f"_r2sub_{name}_export"
        TraderSub = load_trader_from_path(p_sub, mod_sub)
        TraderExp = load_trader_from_path(p_exp, mod_exp)

        if not args.skip_historical:
            if not data_root.is_dir():
                raise SystemExit(f"Missing historical data: {data_root}")
            from prosperity4bt.tools.data_reader import FileSystemReader

            reader = FileSystemReader(data_root)
            days: list[tuple[int, int]] = []
            for rn in (1, 2):
                for d in reader.available_days(rn):
                    days.append((rn, d))
            days.sort()
            for rn, d in days:
                check_one(
                    f"{name} historical R{rn} day {d}",
                    product,
                    lambda T, r_=rn, d_=d: _run_day(T, data_root, r_, d_),
                    lambda T, r_=rn, d_=d: _run_day(T, data_root, r_, d_),
                    TraderSub,
                    TraderExp,
                )

        if not args.skip_day29:
            main_z, extra_z = collect_day29_zips()
            if not main_z and not extra_z:
                print(f"WARN [{name}]: no day-29 zips", file=sys.stderr)
            else:
                for z in main_z:
                    check_one(
                        f"{name} day29 {z.name}",
                        product,
                        lambda T, zp=z: run_day29_tape(T, zp, comb_dir),
                        lambda T, zp=z: run_day29_tape(T, zp, comb_dir),
                        TraderSub,
                        TraderExp,
                    )
                for z in extra_z:
                    check_one(
                        f"{name} day29 extra/{z.name}",
                        product,
                        lambda T, zp=z: run_day29_tape(T, zp, comb_dir),
                        lambda T, zp=z: run_day29_tape(T, zp, comb_dir),
                        TraderSub,
                        TraderExp,
                    )
                p = comb_dir / "prices_combined_all_runs.csv"
                t = comb_dir / "trades_combined_all_runs.csv"
                if p.is_file() and t.is_file():
                    check_one(
                        f"{name} day29 combined",
                        product,
                        lambda T: run_day29_tape(T, None, comb_dir),
                        lambda T: run_day29_tape(T, None, comb_dir),
                        TraderSub,
                        TraderExp,
                    )

    w = sys.stdout.write
    w("round2submission/ vs round2work export originals — PnL parity (worse)\n")
    w("suite | product | sub_prod | exp_prod | sub_tot | exp_tot\n")
    w("-" * 90 + "\n")
    for suite, prod, ps, pe, ts, te in rows:
        ok = "OK" if (ps == pe and ts == te) else "MISMATCH"
        w(f"{ok}  {suite[:48]:<48}  {prod[:12]:<12}  {ps:,.2f}  {pe:,.2f}  {ts:,.2f}  {te:,.2f}\n")

    if failures:
        w("\nFAILURES:\n")
        for f in failures:
            w(f"  {f}\n")
        raise SystemExit(1)
    w("\nAll checks passed: submission folder matches exports on every run.\n")


if __name__ == "__main__":
    main()
