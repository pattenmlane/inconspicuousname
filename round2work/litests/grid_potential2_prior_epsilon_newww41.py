#!/usr/bin/env python3
"""Grid PRIOR_STRENGTH × epsilon (``eplison``) for ``potential2_osmium_only`` and
``potential2_osmium_only_edge1p0`` on the **41** ``newww`` submission tapes
(full trades), ``--match-trades all``.

Example::

  cd /path/to/ProsperityRepo
  python3 round2work/litests/grid_potential2_prior_epsilon_newww41.py

  python3 round2work/litests/grid_potential2_prior_epsilon_newww41.py \\
    --priors 1000,2000,4000 --epsilons 0.5,0.65,0.8
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SRC_P2 = HERE / "potential2_osmium_only.py"
SRC_P2E = HERE / "potential2_osmium_only_edge1p0.py"
NEWWW = HERE.parent / "day 29 logs" / "newww"
ROUND = 2
DAY = 29
TAPE_ROOT_DEFAULT = Path("/tmp/newww_r2_day29_full_trades")


def _pp() -> str:
    return f"{REPO / 'imc-prosperity-4-backtester'}:{REPO / 'imc-prosperity-4-backtester' / 'prosperity4bt'}"


def _patch_source(text: str, prior: int, epsilon: float) -> str:
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
            "Patch failed: expected `eplison = 0.65` and `PRIOR_STRENGTH = 2000` in source"
        )
    return t


def _parse_num_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]


def collect_stems(tape_root: Path) -> list[str]:
    if not tape_root.is_dir():
        raise SystemExit(
            f"Missing tape root {tape_root}. Export with:\n"
            "  python3 round2work/litests/run_newww41_osmium_head_to_head.py --include-submission-trades"
        )
    stems = sorted(
        d.name
        for d in tape_root.iterdir()
        if d.is_dir() and (d / "ROUND_2" / f"prices_round_{ROUND}_day_{DAY}.csv").is_file()
    )
    return stems


def run_total(py: Path, algo: Path, data_root: Path, match: str) -> int:
    cp = subprocess.run(
        [
            str(py),
            "-m",
            "prosperity4bt",
            str(algo),
            f"{ROUND}-{DAY}",
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
    if cp.returncode != 0:
        raise RuntimeError(f"{algo} {data_root.name}:\n{cp.stderr}\n{cp.stdout}")
    for line in cp.stdout.splitlines():
        if "Total profit:" in line:
            nums = re.findall(r"[\d,]+", line.split("Total profit:", 1)[-1])
            if nums:
                return int(nums[-1].replace(",", ""))
    raise RuntimeError("no Total profit in stdout:\n" + cp.stdout)


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
    ap.add_argument(
        "--tape-root",
        type=Path,
        default=TAPE_ROOT_DEFAULT,
        help="Per-stem tape folders (ROUND_2 day 29 CSVs)",
    )
    ap.add_argument("--match-trades", choices=("all", "worse", "none"), default="all")
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    args = ap.parse_args()

    priors = _parse_int_list(args.priors)
    epsilons = _parse_num_list(args.epsilons)
    tape_root = args.tape_root.expanduser().resolve()
    stems = collect_stems(tape_root)
    if len(stems) != 41:
        print(f"warning: expected 41 tapes, found {len(stems)} under {tape_root}", file=sys.stderr)

    text_p2 = SRC_P2.read_text(encoding="utf-8")
    text_p2e = SRC_P2E.read_text(encoding="utf-8")

    rows: list[tuple[int, float, int, int]] = []
    td = Path(tempfile.mkdtemp(prefix="p2grid41_"))
    try:
        for prior in priors:
            for eps in epsilons:
                es = str(eps).replace(".", "p")
                body_p2 = _patch_source(text_p2, prior, eps)
                body_p2e = _patch_source(text_p2e, prior, eps)
                name_p2 = td / f"p{prior}_e{es}_p2.py"
                name_p2e = td / f"p{prior}_e{es}_p2e.py"
                name_p2.write_text(body_p2, encoding="utf-8")
                name_p2e.write_text(body_p2e, encoding="utf-8")

                s_p2 = s_p2e = 0
                for stem in stems:
                    root = tape_root / stem
                    s_p2 += run_total(args.python, name_p2, root, args.match_trades)
                    s_p2e += run_total(args.python, name_p2e, root, args.match_trades)
                rows.append((prior, eps, s_p2, s_p2e))
                print(f"prior={prior} eps={eps} p2_sum={s_p2} p2e1_sum={s_p2e}", file=sys.stderr)
    finally:
        shutil.rmtree(td, ignore_errors=True)

    w = 88
    print()
    print(
        "potential2_osmium_only (+ edge1p0) | newww41 |",
        f"tapes={len(stems)}",
        f"root={tape_root}",
        f"match={args.match_trades}",
        sep=" ",
    )
    print("=" * w)
    print(f"{'prior':>8}  {'epsilon':>8}  {'p2_sum41':>12}  {'p2e1_sum41':>12}  {'both':>12}")
    print("-" * w)
    for prior, eps, s2, se in sorted(rows, key=lambda r: -(r[2] + r[3])):
        print(f"{prior:8d}  {eps:8.4g}  {s2:12,}  {se:12,}  {s2 + se:12,}")
    print("-" * w)

    best_p2 = max(rows, key=lambda r: r[2])
    best_e = max(rows, key=lambda r: r[3])
    best_sum = max(rows, key=lambda r: r[2] + r[3])
    print(
        f"best p2_sum41:   prior={best_p2[0]} eps={best_p2[1]} total={best_p2[2]:,}",
    )
    print(
        f"best p2e1_sum41: prior={best_e[0]} eps={best_e[1]} total={best_e[3]:,}",
    )
    print(
        f"best p2+p2e1:    prior={best_sum[0]} eps={best_sum[1]} total={best_sum[2] + best_sum[3]:,}",
    )


if __name__ == "__main__":
    main()
