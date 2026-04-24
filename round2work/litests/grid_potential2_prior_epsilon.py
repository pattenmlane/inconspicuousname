#!/usr/bin/env python3
"""Grid search ``PRIOR_STRENGTH`` × ``eplison`` (epsilon) for ``potential2_osmium_only``.

Runs ``python3 -m prosperity4bt`` on ``Prosperity4Data`` for rounds 1+2 (all days).

Example::

  cd /path/to/ProsperityRepo
  PYTHONPATH=\"$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt\" \\
  python3 round2work/litests/grid_potential2_prior_epsilon.py

  python3 round2work/litests/grid_potential2_prior_epsilon.py --match worse \\
    --priors 500,2000,8000 --epsilons 0.5,0.65,0.8
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIT = REPO / "round2work" / "litests"
SRC = LIT / "potential2_osmium_only.py"
DATA_DEFAULT = REPO / "Prosperity4Data"


def _patch_source(text: str, prior: int, epsilon: float) -> str:
    """Replace the two assignment lines (must match ``potential2_osmium_only.py``)."""
    t = re.sub(
        r"^(\s*)eplison = 0\.65\s*$",
        rf"\1eplison = {epsilon}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    t = re.sub(
        r"^(\s*)PRIOR_STRENGTH = 2000\s*$",
        rf"\1PRIOR_STRENGTH = {prior}",
        t,
        count=1,
        flags=re.MULTILINE,
    )
    if t == text:
        raise SystemExit(
            "Patch failed: expected `eplison = 0.65` and `PRIOR_STRENGTH = 2000` "
            f"in {SRC}"
        )
    return t


def _run_bt(py: Path, algo: Path, data: Path, match: str) -> int:
    pp = f"{REPO / 'imc-prosperity-4-backtester'}:{REPO / 'imc-prosperity-4-backtester' / 'prosperity4bt'}"
    cp = subprocess.run(
        [
            str(py),
            "-m",
            "prosperity4bt",
            str(algo),
            "1",
            "2",
            "--data",
            str(data),
            "--match-trades",
            match,
            "--no-vis",
            "--no-progress",
            "--no-out",
        ],
        cwd=str(REPO),
        env={**dict(**__import__("os").environ), "PYTHONPATH": pp},
        capture_output=True,
        text=True,
    )
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr + "\n" + cp.stdout)
    m = None
    for line in cp.stdout.splitlines():
        if "Total profit:" in line:
            m = line
    if not m:
        raise RuntimeError("no Total profit in stdout:\n" + cp.stdout)
    nums = re.findall(r"[\d,]+", m.split("Total profit:")[-1])
    if not nums:
        raise RuntimeError(m)
    return int(nums[-1].replace(",", ""))


def _parse_num_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--priors",
        type=str,
        default="500,1000,2000,4000,8000",
        help="Comma-separated PRIOR_STRENGTH values",
    )
    ap.add_argument(
        "--epsilons",
        type=str,
        default="0.35,0.5,0.65,0.8,1.0",
        help="Comma-separated epsilon (eplison) values",
    )
    ap.add_argument("--data", type=Path, default=DATA_DEFAULT, help="Prosperity4Data root")
    ap.add_argument("--match", choices=("all", "worse"), default="all")
    ap.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python with typer installed (default: current interpreter)",
    )
    args = ap.parse_args()

    priors = _parse_int_list(args.priors)
    epsilons = _parse_num_list(args.epsilons)
    text = SRC.read_text(encoding="utf-8")

    rows: list[tuple[int, float, int]] = []
    td = Path(tempfile.mkdtemp(prefix="p2grid_"))
    try:
        for prior in priors:
            for eps in epsilons:
                body = _patch_source(text, prior, eps)
                name = f"p{prior}_e{str(eps).replace('.', 'p')}.py"
                path = td / name
                path.write_text(body, encoding="utf-8")
                total = _run_bt(args.python, path, args.data, args.match)
                rows.append((prior, eps, total))
    finally:
        shutil.rmtree(td, ignore_errors=True)

    rows.sort(key=lambda r: -r[2])
    w = 80
    print("potential2_osmium_only grid", f"data={args.data.name}", f"match={args.match}", sep=" | ")
    print("=" * w)
    print(f"{'prior':>8}  {'epsilon':>8}  {'total_pnl':>12}")
    print("-" * w)
    for prior, eps, total in rows:
        print(f"{prior:8d}  {eps:8.4g}  {total:12,}")
    print("-" * w)
    best = rows[0]
    print(f"best: PRIOR_STRENGTH={best[0]}  epsilon={best[1]}  total={best[2]:,}")


if __name__ == "__main__":
    main()
