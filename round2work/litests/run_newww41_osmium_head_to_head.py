#!/usr/bin/env python3
"""
Reconstruct Round 2 day-29 tapes from ``round2work/day 29 logs/newww/*.zip``,
then backtest four osmium-only traders with ``--match-trades all``.

By default, ``logtodata`` omits trades where buyer or seller is SUBMISSION;
pass ``--include-submission-trades`` to export the full ``tradeHistory``.

Reports per-tape winner(s) (tied for max total profit each get a win) and sums.
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
LOGTODATA = R2WORK / "logtodata.py"
NEWWW = R2WORK / "day 29 logs" / "newww"
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


def export_tape(zip_path: Path, dest_root: Path, *, exclude_submission_trades: bool) -> None:
    tmp = dest_root / "_tmp_export"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    cmd = [
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
    ]
    if exclude_submission_trades:
        cmd.append("--exclude-submission-trades")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    r2 = dest_root / "ROUND_2"
    r2.mkdir(parents=True, exist_ok=True)
    prices = list(tmp.glob(f"prices_round_{ROUND}_day_{DAY}_*.csv"))
    trades = list(tmp.glob(f"trades_round_{ROUND}_day_{DAY}_*.csv"))
    if len(prices) != 1 or len(trades) != 1:
        raise RuntimeError(f"{zip_path}: expected one prices/trades export, got {prices=} {trades=}")
    shutil.copy(prices[0], r2 / f"prices_round_{ROUND}_day_{DAY}.csv")
    shutil.copy(trades[0], r2 / f"trades_round_{ROUND}_day_{DAY}.csv")
    shutil.rmtree(tmp)


def run_bt(algo: Path, data_root: Path) -> tuple[int, int | None, str]:
    cp = subprocess.run(
        [
            sys.executable,
            "-m",
            "prosperity4bt",
            str(algo),
            f"{ROUND}-{DAY}",
            "--data",
            str(data_root),
            "--match-trades",
            "all",
            "--no-vis",
            "--no-progress",
            "--no-out",
        ],
        cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": _pp()},
        capture_output=True,
        text=True,
    )
    if cp.returncode != 0:
        return cp.returncode, None, (cp.stderr or "") + (cp.stdout or "")
    for line in cp.stdout.splitlines():
        if "Total profit:" in line:
            nums = re.findall(r"[\d,]+", line.split("Total profit:", 1)[-1])
            if nums:
                return 0, int(nums[-1].replace(",", "")), ""
    return cp.returncode, None, (cp.stderr or "") + (cp.stdout or "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tape-root",
        type=Path,
        default=None,
        help="Directory for per-zip tape folders (default: /tmp/... based on trade filter)",
    )
    ap.add_argument(
        "--include-submission-trades",
        action="store_true",
        help="Keep SUBMISSION rows in trades CSV (same as older ~4k-style replays)",
    )
    ap.add_argument(
        "--skip-export",
        action="store_true",
        help="Assume tape-root already has one subfolder per zip stem",
    )
    args = ap.parse_args()

    exclude_sub = not args.include_submission_trades
    if args.tape_root is not None:
        tape_root = args.tape_root.expanduser().resolve()
    elif args.include_submission_trades:
        tape_root = Path("/tmp/newww_r2_day29_full_trades").resolve()
    else:
        tape_root = Path("/tmp/newww_r2_day29_no_submission_trades").resolve()

    zips = sorted(NEWWW.glob("*.zip"))
    if not zips:
        raise SystemExit(f"No zips under {NEWWW}")

    if not args.skip_export:
        tape_root.mkdir(parents=True, exist_ok=True)
        for z in zips:
            stem = z.stem
            dest = tape_root / stem
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            export_tape(z, dest, exclude_submission_trades=exclude_sub)
            print(f"exported {stem}", file=sys.stderr)

    wins: dict[str, int] = {name: 0 for name, _ in STRATEGIES}
    totals: dict[str, int] = {name: 0 for name, _ in STRATEGIES}
    errors: list[str] = []

    print("stem\t" + "\t".join(n for n, _ in STRATEGIES))
    for z in zips:
        stem = z.stem
        data = tape_root / stem
        if not (data / "ROUND_2" / f"prices_round_{ROUND}_day_{DAY}.csv").is_file():
            errors.append(f"missing tape {data}")
            continue
        row: dict[str, int] = {}
        for name, path in STRATEGIES:
            code, pnl, err_blob = run_bt(path, data)
            if code != 0 or pnl is None:
                tail = (err_blob or "")[-1200:].replace("\n", " ")
                errors.append(f"{stem} {name}: exit={code} …{tail}")
                continue
            row[name] = pnl
            totals[name] += pnl
        if len(row) != len(STRATEGIES):
            continue
        best = max(row.values())
        for name, pnl in row.items():
            if pnl == best:
                wins[name] += 1
        print(f"{stem}\t" + "\t".join(str(row[n]) for n, _ in STRATEGIES))

    print("\n# wins (tied for best on a tape each get +1)", file=sys.stderr)
    for name, _ in STRATEGIES:
        print(f"  {name}: {wins[name]}", file=sys.stderr)
    print("\n# total profit (sum over tapes)", file=sys.stderr)
    for name, _ in STRATEGIES:
        print(f"  {name}: {totals[name]:,}", file=sys.stderr)
    if errors:
        print("\nErrors:", file=sys.stderr)
        for e in errors[:30]:
            print(e, file=sys.stderr)


if __name__ == "__main__":
    main()
