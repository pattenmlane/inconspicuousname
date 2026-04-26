#!/usr/bin/env python3
"""
After v27: quantify how long **Sonic joint tight** (5200+5300 spread<=2) persists
**after** each v26 trigger (Mark67 aggr extract + joint tight at print).

Uses price grid only (same as sim BBO). Writes text summary for analysis.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
SIG = Path(__file__).resolve().parent / "outputs" / "r4_v26_signals.json"
TH = 2
DAYS = [1, 2, 3]
W = 50_000
STEP = 100  # tape tick step


def load_px() -> pd.DataFrame:
    fs = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if p.is_file():
            df = pd.read_csv(p, sep=";")
            df["day"] = d
            fs.append(df)
    return pd.concat(fs, ignore_index=True)


def joint_series(px: pd.DataFrame) -> pd.DataFrame:
    sp = px[px["product"].isin(["VEV_5200", "VEV_5300"])].copy()
    sp["spread"] = sp["ask_price_1"] - sp["bid_price_1"]
    sp52 = sp[sp["product"] == "VEV_5200"][["day", "timestamp", "spread"]].rename(columns={"spread": "s52"})
    sp53 = sp[sp["product"] == "VEV_5300"][["day", "timestamp", "spread"]].rename(columns={"spread": "s53"})
    jt = sp52.merge(sp53, on=["day", "timestamp"])
    jt["jt"] = (jt["s52"] <= TH) & (jt["s53"] <= TH)
    return jt[["day", "timestamp", "jt"]]


def merged_to_day_ts(t: int, cum: dict[int, int]) -> tuple[int, int]:
    for d in sorted(cum.keys(), reverse=True):
        if t >= cum[d]:
            return d, int(t - cum[d])
    return 1, int(t)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    obj = json.loads(SIG.read_text())
    triggers = sorted(int(x) for x in obj["mark67_extract_buy_aggr_filtered_merged_ts"])
    cum = {int(k): int(v) for k, v in obj["day_cum_offset"].items()}
    max_ts = {int(k): int(v) for k, v in obj["day_max_ts"].items()}

    px = load_px()
    jdf = joint_series(px)
    jmap = jdf.set_index(["day", "timestamp"])["jt"].to_dict()

    def jt_at(d: int, ts: int) -> bool:
        return bool(jmap.get((d, ts), False))

    first_break: list[int] = []
    frac_tight_in_window: list[float] = []
    lines = [
        f"v26 triggers n={len(triggers)} window={W} step={STEP}",
        "Per trigger: first tape timestamp offset (from print) where joint NOT tight (nan if tight through min(W, end-of-day-print));",
        "plus fraction of sampled offsets in [0,W] cap to day end that stay tight.",
    ]

    for T in triggers:
        d, t0 = merged_to_day_ts(T, cum)
        end = min(t0 + W, max_ts[d])
        offs = list(range(0, min(W, end - t0) + 1, STEP))
        if not offs:
            continue
        tight_flags = []
        fb: int | None = None
        for dt in offs:
            ts = t0 + dt
            ok = jt_at(d, ts)
            tight_flags.append(1.0 if ok else 0.0)
            if not ok and fb is None and dt > 0:
                fb = dt
        never = W + STEP
        first_break.append(int(fb) if fb is not None else never)
        frac_tight_in_window.append(float(np.mean(tight_flags)))

    arr = np.array(first_break, dtype=float)
    arr2 = np.array(frac_tight_in_window)
    never_sent = W + STEP
    lines.append(
        f"first_break_offset_ticks (STEP={STEP}): mean={arr.mean():.0f} median={np.median(arr):.0f} "
        f"frac_no_break_through_window_end={(arr >= never_sent).mean():.3f}  # sentinel {never_sent}=no break on grid"
    )
    lines.append(
        f"mean_frac_tight_samples_in_window={arr2.mean():.3f} median={np.median(arr2):.3f}"
    )
    lines.append("Interpretation: if joint gate opens often shortly after print, v27-style immediate exit on widen will churn; v26 holds through transient noise.")

    out_path = OUT / "r4_p6_joint_tight_persistence_after_v26_triggers.txt"
    out_path.write_text("\n".join(lines) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
