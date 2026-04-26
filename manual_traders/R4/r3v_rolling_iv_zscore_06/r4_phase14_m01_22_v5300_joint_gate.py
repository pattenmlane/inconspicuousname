#!/usr/bin/env python3
"""
Phase 14: **VEV_5300** prints with **Mark 01** and **Mark 22** as counterparties,
**Sonic joint gate** at print, forward **VEV_5300** mid K∈{5,20,100} (price-grid rows;
same `forward_delta` convention as Phase 1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
TH = 2
DAYS = [1, 2, 3]
KS = (5, 20, 100)
SYM = "VEV_5300"


def load_px() -> pd.DataFrame:
    fs = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if p.is_file():
            df = pd.read_csv(p, sep=";")
            df["day"] = d
            fs.append(df)
    return pd.concat(fs, ignore_index=True)


def load_tr() -> pd.DataFrame:
    fs = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if p.is_file():
            df = pd.read_csv(p, sep=";")
            df["day"] = d
            fs.append(df)
    return pd.concat(fs, ignore_index=True)


def joint_map(px: pd.DataFrame) -> dict[tuple[int, int], bool]:
    sp = px[px["product"].isin(["VEV_5200", "VEV_5300"])].copy()
    sp["spread"] = sp["ask_price_1"] - sp["bid_price_1"]
    s52 = sp[sp["product"] == "VEV_5200"][["day", "timestamp", "spread"]].rename(columns={"spread": "s52"})
    s53 = sp[sp["product"] == "VEV_5300"][["day", "timestamp", "spread"]].rename(columns={"spread": "s53"})
    jt = s52.merge(s53, on=["day", "timestamp"])
    jt["joint_tight"] = (jt["s52"] <= TH) & (jt["s53"] <= TH)
    return jt.set_index(["day", "timestamp"])["joint_tight"].to_dict()


def series_pair(px: pd.DataFrame, day: int, sym: str) -> tuple[np.ndarray, np.ndarray]:
    g = px[(px["day"] == day) & (px["product"] == sym)].sort_values("timestamp")
    return g["timestamp"].to_numpy(dtype=np.int64), g["mid_price"].astype(float).to_numpy()


def forward_delta(ts: np.ndarray, mid: np.ndarray, t0: int, k: int) -> float:
    i = int(np.searchsorted(ts, t0, side="left"))
    if i >= len(ts):
        return float("nan")
    if ts[i] != t0:
        i = int(np.searchsorted(ts, t0, side="right") - 1)
    i = max(0, min(i, len(ts) - 1))
    j = min(i + k, len(mid) - 1)
    if j <= i:
        return float("nan")
    return float(mid[j] - mid[i])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    px = load_px()
    tr = load_tr()
    jmap = joint_map(px)
    px53 = px.rename(columns={"product": "symbol"})
    rows = []
    for d in DAYS:
        t53 = tr[(tr["day"] == d) & (tr["symbol"] == SYM)].copy()
        psub = px53[(px53["day"] == d) & (px53["symbol"] == SYM)][["timestamp", "bid_price_1", "ask_price_1"]]
        mm = pd.merge_asof(
            t53.sort_values("timestamp"),
            psub.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        mm["day"] = d
        for _, r in mm.iterrows():
            b, s = str(r["buyer"]), str(r["seller"])
            if not ((b == "Mark 01" and s == "Mark 22") or (b == "Mark 22" and s == "Mark 01")):
                continue
            t0 = int(r["timestamp"])
            bid1, ask1 = float(r["bid_price_1"]), float(r["ask_price_1"])
            pr = float(r["price"])
            if pr >= ask1 - 1e-9:
                ag = "buy_aggr"
            elif pr <= bid1 + 1e-9:
                ag = "sell_aggr"
            else:
                ag = "ambiguous"
            jt = bool(jmap.get((d, t0), False))
            ts_arr, mid_arr = series_pair(px, d, SYM)
            fd = {f"k{k}": forward_delta(ts_arr, mid_arr, t0, k) for k in KS}
            rows.append(
                {
                    "day": d,
                    "timestamp": t0,
                    "buyer": b,
                    "seller": s,
                    "pair_order": "01_22" if b == "Mark 01" else "22_01",
                    "aggr": ag,
                    "joint_tight": jt,
                    **fd,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "r4_p14_m01_22_v5300_rows.csv", index=False)

    lines = ["VEV_5300 trades: Mark01/Mark22 only; K = forward price-grid rows\n"]
    for tight_label, sub in (("joint_tight", df[df["joint_tight"]]), ("joint_loose", df[~df["joint_tight"]])):
        lines.append(f"\n=== {tight_label} n={len(sub)} ===\n")
        for k in KS:
            col = f"k{k}"
            v = sub[col].dropna()
            if len(v) < 3:
                lines.append(f"  K={k}: n={len(v)} skip\n")
                continue
            lines.append(
                f"  K={k}: n={len(v)} mean={float(v.mean()):.4f} median={float(v.median()):.4f} "
                f"frac_pos={float((v>0).mean()):.3f}\n"
            )
    (OUT / "r4_p14_m01_22_v5300_joint_gate.txt").write_text("".join(lines))
    print(f"Wrote {OUT / 'r4_p14_m01_22_v5300_rows.csv'} and {OUT / 'r4_p14_m01_22_v5300_joint_gate.txt'}")


if __name__ == "__main__":
    main()
