#!/usr/bin/env python3
"""
Backtest four osmium-only litest traders:

1. **Historical:** ``Prosperity4Data`` — rounds **1** and **2** (all days), one run per strategy.
2. **Day-29 submission tapes:** every ``*.zip`` under ``round2work/day 29 logs/`` (root,
   ``extra/``, ``newww/``), deduped by submission stem; full ``tradeHistory`` export.

Uses ``--match-trades all`` unless overridden.

Writes a TSV of per-tape PnLs to ``--day29-tsv`` and prints summaries to stdout.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R2WORK = HERE.parent
REPO = R2WORK.parent
DAY29_LOGS = R2WORK / "day 29 logs"
LOGTODATA = R2WORK / "logtodata.py"
ROUND = 2
DAY = 29

STRATEGIES: tuple[tuple[str, Path], ...] = (
    ("r2_submission_osmium_only", HERE / "r2_submission_osmium_only.py"),
    ("potential1_osmium_only", HERE / "potential1_osmium_only.py"),
    ("potential2_osmium_only", HERE / "potential2_osmium_only.py"),
    ("potential2_osmium_only_edge1p0", HERE / "potential2_osmium_only_edge1p0.py"),
)


def _pp() -> str:
    return f"{REPO / 'imc-prosperity-4-backtester'}:{REPO / 'imc-prosperity-4-backtester' / 'prosperity4bt'}"


def collect_day29_zips() -> list[Path]:
    candidates: list[Path] = []
    for folder in (DAY29_LOGS, DAY29_LOGS / "extra", DAY29_LOGS / "newww"):
        if folder.is_dir():
            candidates.extend(sorted(folder.glob("*.zip")))
    by_stem: dict[str, Path] = {}
    for z in sorted(candidates, key=lambda p: p.as_posix()):
        by_stem.setdefault(z.stem, z)
    return sorted(by_stem.values(), key=lambda p: p.stem)


def export_tape(zip_path: Path, dest_root: Path) -> None:
    tmp = dest_root / "_tmp_export"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            str(LOGTODATA),
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
        stdout=subprocess.DEVNULL,
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


def run_bt(
    algo: Path, data_root: Path, days: list[str], match: str
) -> tuple[int, dict[str, int], str]:
    """Parse ASH_COATED_OSMIUM and Total profit from stdout when present."""
    cp = subprocess.run(
        [
            sys.executable,
            "-m",
            "prosperity4bt",
            str(algo),
            *days,
            "--data",
            str(data_root),
            "--match-trades",
            match,
            "--no-vis",
            "--no-progress",
            "--no-out",
        ],
        cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": _pp()},
        capture_output=True,
        text=True,
    )
    out = (cp.stderr or "") + (cp.stdout or "")
    if cp.returncode != 0:
        return cp.returncode, {}, out[-2000:]
    nums: dict[str, int] = {}
    osmium_sum = 0
    for line in cp.stdout.splitlines():
        if line.strip().startswith("ASH_COATED_OSMIUM:"):
            m = re.search(r"([\d,]+)\s*$", line)
            if m:
                osmium_sum += int(m.group(1).replace(",", ""))
        if "Total profit:" in line:
            found = re.findall(r"[\d,]+", line.split("Total profit:", 1)[-1])
            if found:
                nums["total"] = int(found[-1].replace(",", ""))
    if osmium_sum:
        nums["osmium"] = osmium_sum
    return 0, nums, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data",
        type=Path,
        default=REPO / "Prosperity4Data",
        help="Historical data root (default: Prosperity4Data)",
    )
    ap.add_argument(
        "--tape-root",
        type=Path,
        default=Path("/tmp/day29_all_full_trades"),
        help="Where per-zip Round 2 day-29 tape trees are written",
    )
    ap.add_argument(
        "--skip-export",
        action="store_true",
        help="Reuse existing folders under --tape-root (one subdir per zip stem)",
    )
    ap.add_argument(
        "--match-trades",
        type=str,
        default="all",
        choices=("all", "worse", "none"),
    )
    ap.add_argument(
        "--day29-tsv",
        type=Path,
        default=Path("/tmp/osmium4_day29_all_tapes.tsv"),
        help="Write per-stem × strategy total PnL (TSV)",
    )
    args = ap.parse_args()

    data_hist = args.data.expanduser().resolve()
    if not data_hist.is_dir():
        raise SystemExit(f"Missing historical data: {data_hist}")

    match = args.match_trades
    names = [n for n, _ in STRATEGIES]

    print("=== Historical: Prosperity4Data rounds 1 + 2 (all days) ===")
    print(f"data={data_hist}\tmatch-trades={match}\n")
    print("strategy\tosmium_pnl\ttotal_profit")
    for name, path in STRATEGIES:
        code, nums, err = run_bt(path, data_hist, ["1", "2"], match)
        if code != 0:
            print(f"{name}\tERROR\t{err[:500]}")
            continue
        o = nums.get("osmium", "")
        t = nums.get("total", "")
        print(f"{name}\t{o}\t{t}")

    zips = collect_day29_zips()
    tape_root = args.tape_root.expanduser().resolve()
    if not args.skip_export:
        tape_root.mkdir(parents=True, exist_ok=True)
        for z in zips:
            dest = tape_root / z.stem
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            export_tape(z, dest)
            print(f"exported day29 {z.stem}", file=sys.stderr)

    print(f"\n=== Day-29 submission tapes: {len(zips)} zips under {DAY29_LOGS} ===")
    print(f"tape_root={tape_root}\tmatch-trades={match}\n")

    wins = {name: 0 for name in names}
    totals = {name: 0 for name in names}
    rows: list[tuple[str, dict[str, int]]] = []

    for z in zips:
        stem = z.stem
        root = tape_root / stem
        if not (root / "ROUND_2" / f"prices_round_{ROUND}_day_{DAY}.csv").is_file():
            print(f"missing tape {root}", file=sys.stderr)
            continue
        pnl_row: dict[str, int] = {}
        for name, path in STRATEGIES:
            code, nums, err = run_bt(path, root, [f"{ROUND}-{DAY}"], match)
            if code != 0 or "total" not in nums:
                print(f"{stem} {name} failed: {err[:400]}", file=sys.stderr)
                continue
            v = nums["total"]
            pnl_row[name] = v
            totals[name] += v
        if len(pnl_row) != len(STRATEGIES):
            continue
        best = max(pnl_row.values())
        for name, v in pnl_row.items():
            if v == best:
                wins[name] += 1
        rows.append((stem, pnl_row))

    tsv_path = args.day29_tsv.expanduser().resolve()
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("stem\t" + "\t".join(names) + "\n")
        for stem, pr in rows:
            f.write(stem + "\t" + "\t".join(str(pr[n]) for n in names) + "\n")
    print(f"Wrote {tsv_path}")

    print("\n# day29 wins (tied for best total each +1)")
    for n in names:
        print(f"  {n}: {wins[n]}")
    print("\n# day29 sum of total profit")
    for n in names:
        print(f"  {n}: {totals[n]:,}")


if __name__ == "__main__":
    main()
