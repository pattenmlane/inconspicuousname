#!/usr/bin/env python3
"""
Phase 1 follow-up: **VEV_5300** trade prints, forward mid K∈{5,20,100} price-tape rows
(not calendar ms), **stratified by Sonic joint gate** (5200&5300 spread<=2 at same ts).

K = number of **subsequent** price rows for that symbol (same convention as
`r4_phase1_counterparty.py` forward_delta).
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
    sp = px[px["product"].isin(["VEV_5200", "VEV_5300"])].copy()
    sp["spread"] = sp["ask_price_1"] - sp["bid_price_1"]
    sp52 = sp[sp["product"] == "VEV_5200"][["day", "timestamp", "spread"]].rename(columns={"spread": "s52"})
    sp53 = sp[sp["product"] == "VEV_5300"][["day", "timestamp", "spread"]].rename(columns={"spread": "s53"})
    jt = sp52.merge(sp53, on=["day", "timestamp"])
    jt["joint_tight"] = (jt["s52"] <= TH) & (jt["s53"] <= TH)
    jmap = jt.set_index(["day", "timestamp"])["joint_tight"].to_dict()

    t53 = tr[tr["symbol"] == SYM].copy()
    px30 = px[(px["product"] == SYM)][["day", "timestamp", "bid_price_1", "ask_price_1", "mid_price"]]
    t53 = t53.merge(
        px30,
        left_on=["day", "timestamp"],
        right_on=["day", "timestamp"],
        how="left",
    )
    rows = []
    for d in DAYS:
        ts_arr, mid_arr = series_pair(px, d, SYM)
        for _, r in t53[t53["day"] == d].iterrows():
            t0 = int(r["timestamp"])
            jt_ = jmap.get((d, t0), False)
            pxb, pxa = float(r["bid_price_1"]), float(r["ask_price_1"])
            pr = float(r["price"])
            if pr >= pxa - 1e-9:
                side = "buy_aggr"
            elif pr <= pxb + 1e-9:
                side = "sell_aggr"
            else:
                side = "mid_or_ambiguous"
            dlt = {f"k{k}": forward_delta(ts_arr, mid_arr, t0, k) for k in KS}
            rows.append(
                {
                    "day": d,
                    "timestamp": t0,
                    "joint_tight": jt_,
                    "aggr": side,
                    "buyer": r["buyer"],
                    "seller": r["seller"],
                    **dlt,
                }
            )
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "r4_p12_v5300_trades_per_print.csv", index=False)

    lines: list[str] = []
    lines.append("VEV_5300 prints R4 days 1-3, joint gate vs loose, K = forward rows\n")
    for label, sub in (("joint_tight", m[m["joint_tight"]]), ("joint_loose", m[~m["joint_tight"]])):
        lines.append(f"\n=== {label} n={len(sub)} ===\n")
        for k in KS:
            col = f"k{k}"
            v = sub[col].dropna()
            if len(v) < 10:
                lines.append(f"  K={k}: n={len(v)} (skip)\n")
                continue
            mu = float(v.mean())
            med = float(v.median())
            fp = float((v > 0).mean())
            lines.append(f"  K={k}: n={len(v)} mean={mu:.4f} median={med:.4f} frac_pos={fp:.3f}\n")
        for ag in ("buy_aggr", "sell_aggr"):
            s2 = sub[sub["aggr"] == ag]
            if s2.empty:
                continue
            lines.append(f"  {ag} n={len(s2)}\n")
            for k in KS:
                col = f"k{k}"
                v = s2[col].dropna()
                if len(v) < 5:
                    continue
                lines.append(
                    f"    K={k}: mean={float(v.mean()):.4f} frac_pos={float((v>0).mean()):.3f} n={len(v)}\n"
                )

    pth = OUT / "r4_p12_v5300_joint_gate_markout.txt"
    pth.write_text("".join(lines))
    print(f"Wrote {pth} and {OUT / 'r4_p12_v5300_trades_per_print.csv'}")


if __name__ == "__main__":
    main()
