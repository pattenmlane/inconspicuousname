#!/usr/bin/env python3
"""Mark14 seller aggr_buy EXTRACT: mean fwd_mid_k20 when Sonic joint tight vs not (Phase-1 enriched)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
P1 = Path(__file__).resolve().parent / "outputs_r4_phase1" / "r4_p1_trades_enriched.csv"
OUT = Path(__file__).resolve().parent / "outputs_r4_phase3" / "r4_p12_mark14_fwd20_gate_tight_vs_loose.csv"
TH = 2
DAYS = [1, 2, 3]


def tight_map_day(day: int) -> dict[int, bool]:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    rows = []
    for sym in ["VEV_5200", "VEV_5300"]:
        sub = df[df["product"] == sym].drop_duplicates("timestamp").sort_values("timestamp")
        bid = pd.to_numeric(sub["bid_price_1"], errors="coerce")
        ask = pd.to_numeric(sub["ask_price_1"], errors="coerce")
        sub = sub.assign(spread=(ask - bid).astype(float))[["timestamp", "spread"]]
        sub["sym"] = sym
        rows.append(sub)
    x = pd.concat(rows, ignore_index=True)
    p52 = x[x["sym"] == "VEV_5200"].rename(columns={"spread": "s5200"})[["timestamp", "s5200"]]
    p53 = x[x["sym"] == "VEV_5300"].rename(columns={"spread": "s5300"})[["timestamp", "s5300"]]
    m = p52.merge(p53, on="timestamp", how="inner")
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    return dict(zip(m["timestamp"].astype(int), m["tight"].astype(bool)))


def main() -> None:
    maps = {d: tight_map_day(d) for d in DAYS}
    tr = pd.read_csv(P1)
    sub = tr[
        (tr["symbol"] == "VELVETFRUIT_EXTRACT")
        & (tr["seller"] == "Mark 14")
        & (tr["aggressor_bucket"] == "aggr_buy")
    ].copy()
    sub["fwd20"] = pd.to_numeric(sub["fwd_mid_k20"], errors="coerce")

    def is_tight(r) -> bool:
        return bool(maps[int(r["day"])].get(int(r["timestamp"]), False))

    sub["joint_tight"] = sub.apply(is_tight, axis=1)

    rows = []
    for d in DAYS:
        s = sub[sub["day"] == d].dropna(subset=["fwd20"])
        for lab, m in [("tight", s[s["joint_tight"]]), ("loose", s[~s["joint_tight"]])]:
            v = m["fwd20"]
            if len(v) < 3:
                continue
            rows.append(
                {
                    "day": d,
                    "gate": lab,
                    "n": int(len(v)),
                    "mean_fwd20": float(v.mean()),
                    "median_fwd20": float(v.median()),
                }
            )
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
