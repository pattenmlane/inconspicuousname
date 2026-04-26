#!/usr/bin/env python3
"""Grid **trader_v10** under ``--match-trades worse``; write TSV of (half,size,total,hydro)."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

DIR = Path(__file__).resolve().parent
TRADER = DIR / "trader_v10.py"
OUT = DIR / "outputs" / "phase10_v10_hydro_grid_worse.tsv"
# ``DIR`` = …/manual_traders/R4/r4_phase1_marks → repo root is ``parents[2]``
ROOT = DIR.parents[2]
CMD_BASE = [
    "python3",
    "-m",
    "prosperity4bt",
    str(TRADER.relative_to(ROOT)),
    "4",
    "--data",
    "Prosperity4Data",
    "--match-trades",
    "worse",
    "--no-out",
    "--no-vis",
    "--no-progress",
]
ANSI = re.compile(r"\x1b\[[0-9;]*[mK]")
ENV = os.environ.copy()
ENV["PYTHONPATH"] = "imc-prosperity-4-backtester:imc-prosperity-4-backtester/prosperity4bt"

HALVES = [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
SIZES = [8, 12, 16, 20]


def parse_log(text: str) -> tuple[int | None, int | None]:
    hydro_sum = 0
    for m in re.finditer(r"HYDROGEL_PACK:\s*([\d,]+)", text):
        hydro_sum += int(m.group(1).replace(",", ""))
    totals = [
        int(m.group(1).replace(",", ""))
        for m in re.finditer(r"Total profit:\s*([\d,]+)", text)
    ]
    grand = totals[-1] if totals else None
    return grand, (hydro_sum if hydro_sum else None)


def main() -> None:
    rows = ["half\tsize\ttotal_profit\thydro_profit\n"]
    for h in HALVES:
        for s in SIZES:
            e = ENV.copy()
            e["R4_HYDRO_HALF"] = str(h)
            e["R4_HYDRO_SIZE"] = str(s)
            r = subprocess.run(
                CMD_BASE,
                cwd=str(ROOT),
                env=e,
                capture_output=True,
                text=True,
                timeout=240,
            )
            text = ANSI.sub("", (r.stdout or "") + (r.stderr or ""))
            tot, hy = parse_log(text)
            rows.append(f"{h}\t{s}\t{tot}\t{hy}\n")
            print(f"half={h} size={s} -> total={tot} hydro={hy}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(rows), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
