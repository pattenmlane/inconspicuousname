#!/usr/bin/env python3
"""
Parse Prosperity submission ``*.json`` ``activitiesLog`` (CSV) and plot
touch mid (``mid_price``) vs timestamp for pepper and osmium.

Usage::

  python3 R1submissionlogs/plot_mids_from_submission_json.py --zip R1submissionlogs/273774.zip

Opens an interactive figure (``plt.show()``). Optional: ``--save path.png`` to also write a file.

If ``--zip`` is omitted, uses the first ``*.zip`` in ``R1submissionlogs/``.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
from pathlib import Path

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"


def load_activities_csv_from_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        json_names = [n for n in zf.namelist() if n.endswith(".json")]
        if not json_names:
            raise SystemExit(f"No .json in {zip_path}")
        raw = zf.read(json_names[0]).decode("utf-8", errors="replace")
    data = json.loads(raw)
    log = data.get("activitiesLog")
    if not isinstance(log, str) or not log.strip():
        raise SystemExit("JSON missing activitiesLog string")
    return log


def parse_mids_by_product(activities_csv: str) -> dict[str, list[tuple[int, float]]]:
    rows: dict[str, list[tuple[int, float]]] = {PEPPER: [], OSMIUM: []}
    r = csv.reader(io.StringIO(activities_csv), delimiter=";")
    header = next(r)
    idx_day = header.index("day")
    idx_ts = header.index("timestamp")
    idx_prod = header.index("product")
    idx_mid = header.index("mid_price")

    for parts in r:
        if len(parts) <= max(idx_day, idx_ts, idx_prod, idx_mid):
            continue
        prod = parts[idx_prod]
        if prod not in rows:
            continue
        mid_s = parts[idx_mid].strip()
        if not mid_s:
            continue
        try:
            ts = int(parts[idx_ts])
            mid = float(mid_s)
        except ValueError:
            continue
        # Engine often emits 0 when there is no two-sided book; skip for readability.
        if not math.isfinite(mid) or mid <= 0.0:
            continue
        rows[prod].append((ts, mid))

    for prod in rows:
        rows[prod].sort(key=lambda x: x[0])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot mid_price from submission JSON (zip).")
    ap.add_argument(
        "--zip",
        type=Path,
        help="Path to submission zip containing .json (default: first zip in this folder)",
    )
    ap.add_argument(
        "--save",
        type=Path,
        metavar="PATH.png",
        help="If set, also save figure to this PNG path",
    )
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    zip_path = args.zip
    if zip_path is None:
        zips = sorted(here.glob("*.zip"))
        if not zips:
            raise SystemExit(f"No .zip in {here}")
        zip_path = zips[0]

    activities = load_activities_csv_from_zip(zip_path)
    series = parse_mids_by_product(activities)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)

    for ax, (prod, label) in zip(
        axes,
        [(PEPPER, "Pepper (INTARIAN_PEPPER_ROOT)"), (OSMIUM, "Osmium (ASH_COATED_OSMIUM)")],
    ):
        pts = series[prod]
        if not pts:
            ax.set_title(f"{label} — no mid rows")
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, lw=0.8, color="#1f77b4" if prod == PEPPER else "#d62728")
        ax.set_ylabel("mid_price")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("timestamp (activity log)")
    fig.suptitle(f"Touch mid from submission log\n{zip_path.name}", fontsize=11)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Also saved {args.save.resolve()}")

    print("Close the plot window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
