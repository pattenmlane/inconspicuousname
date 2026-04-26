#!/usr/bin/env python3
"""
Round 4 Phase 2 — orthogonal conditioning on Phase 1 enriched trades.

Uses r4_p1_trade_enriched.csv (rebuild if missing from Phase 1 script).

Outputs under analysis_outputs/:
- r4_p2_burst_pair_mark67_extract_k20.csv — Mark67|extract within ±W of Mark01|Mark22 burst
- r4_p2_leadlag_m67_signed_flow_vs_extract_fwd.csv — Mark 67 signed flow vs next extract mid change
- r4_p2_leadlag_signed_flow_by_mark.csv (from r4_phase2_leadlag_multi_mark_flow.py) — same for multiple marks
- r4_p2_sonic_gate_mark67_k5_by_day.csv — joint 5200+5300 spread<=2 vs Mark67 buy_agg extract k=5
- r4_p2_microprice_touch_rate.csv — touch rate when spread==2 on 5200/5300
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "analysis_outputs")
ENRICHED = os.path.join(OUT, "r4_p1_trade_enriched.csv")
PRICE_GLOB = "Prosperity4Data/ROUND_4/prices_round_4_day_{d}.csv"
DAYS = (1, 2, 3)
BURST_PAIR = ("Mark 01", "Mark 22")
W_TS = 500  # ±500 timestamp units window for burst neighborhood


def load_prices() -> pd.DataFrame:
    parts = []
    for d in DAYS:
        p = pd.read_csv(PRICE_GLOB.format(d=d), sep=";")
        p["day"] = d
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def sonic_tight(prices: pd.DataFrame) -> pd.DataFrame:
    """Per (day, timestamp): True if both 5200 and 5300 have spread<=2."""
    sub = prices[prices["product"].isin(["VEV_5200", "VEV_5300"])].copy()
    sub["sp"] = sub["ask_price_1"] - sub["bid_price_1"]
    piv = sub.pivot_table(index=["day", "timestamp"], columns="product", values="sp", aggfunc="first")
    if "VEV_5200" not in piv.columns or "VEV_5300" not in piv.columns:
        piv["sonic_tight"] = False
    else:
        piv["sonic_tight"] = (piv["VEV_5200"] <= 2) & (piv["VEV_5300"] <= 2)
    return piv[["sonic_tight"]].reset_index()


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    if not os.path.isfile(ENRICHED):
        raise SystemExit(f"Missing {ENRICHED}; run r4_phase1_counterparty_analysis.py first")

    df = pd.read_csv(ENRICHED)
    prices = load_prices()
    sonic = sonic_tight(prices)
    df = df.merge(sonic, on=["day", "timestamp"], how="left")
    df["sonic_tight"] = df["sonic_tight"].fillna(False)

    # 1) Burst neighborhood: timestamps with at least one Mark 01 -> Mark 22 print
    burst_times = (
        df[(df["buyer"] == BURST_PAIR[0]) & (df["seller"] == BURST_PAIR[1])][["day", "timestamp"]]
        .drop_duplicates()
    )
    neigh = set()
    for _, r in burst_times.iterrows():
        d, t = int(r["day"]), int(r["timestamp"])
        for dt in range(-W_TS, W_TS + 1, 100):
            neigh.add((d, t + dt))

    df["near_m01_m22_burst"] = [((int(a), int(b)) in neigh) for a, b in zip(df["day"], df["timestamp"])]
    m67 = df[
        (df["buyer"] == "Mark 67")
        & (df["symbol"] == "VELVETFRUIT_EXTRACT")
        & (df["agg"] == "buy_agg")
    ]
    summary = []
    for label, mask in [
        ("all_m67_buy_ex", pd.Series(True, index=m67.index)),
        ("near_burst", m67["near_m01_m22_burst"]),
        ("not_near_burst", ~m67["near_m01_m22_burst"]),
        ("sonic_tight", m67["sonic_tight"]),
        ("sonic_loose", ~m67["sonic_tight"]),
    ]:
        g = m67[mask]
        x = g["dm_ex_k20"].dropna()
        n = len(x)
        mean = float(x.mean()) if n else float("nan")
        t = float(mean / (x.std(ddof=1) / np.sqrt(n))) if n > 30 and x.std(ddof=1) > 0 else float("nan")
        summary.append({"slice": label, "n": n, "mean_dm_ex_k20": mean, "t": t})
    pd.DataFrame(summary).to_csv(os.path.join(OUT, "r4_p2_burst_pair_mark67_extract_k20.csv"), index=False)

    # Per-day stability Mark67 buy extract k=5
    byd = []
    for d in DAYS:
        g = m67[m67["day"] == d]
        x = g["dm_ex_k5"].dropna()
        byd.append(
            {
                "day": d,
                "n": len(x),
                "mean": float(x.mean()) if len(x) else float("nan"),
                "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))
                if len(x) > 20 and x.std(ddof=1) > 0
                else float("nan"),
            }
        )
    pd.DataFrame(byd).to_csv(os.path.join(OUT, "r4_p2_mark67_buy_extract_by_day_k5.csv"), index=False)

    # Sonic gate × Mark67
    sg = []
    for st, name in [(True, "tight_gate"), (False, "loose_gate")]:
        g = m67[m67["sonic_tight"] == st]
        x = g["dm_ex_k5"].dropna()
        sg.append(
            {
                "gate": name,
                "n": len(x),
                "mean_k5": float(x.mean()) if len(x) else float("nan"),
                "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))
                if len(x) > 25 and x.std(ddof=1) > 0
                else float("nan"),
            }
        )
    pd.DataFrame(sg).to_csv(os.path.join(OUT, "r4_p2_sonic_gate_mark67_extract_k5.csv"), index=False)

    # 3) Lead-lag: Mark 67 signed notional on extract (per timestamp) vs forward extract mid change on price grid
    ex_px = prices[prices["product"] == "VELVETFRUIT_EXTRACT"][["day", "timestamp", "mid_price"]].rename(
        columns={"mid_price": "mid_ex"}
    )
    ex_px = ex_px.sort_values(["day", "timestamp"])
    ex_tr = df[df["symbol"] == "VELVETFRUIT_EXTRACT"].copy()
    ex_tr["signed_m67"] = np.where(
        (ex_tr["buyer"] == "Mark 67") & (ex_tr["agg"] == "buy_agg"),
        ex_tr["notional"],
        np.where((ex_tr["seller"] == "Mark 67") & (ex_tr["agg"] == "sell_agg"), -ex_tr["notional"], 0.0),
    )
    flow = ex_tr.groupby(["day", "timestamp"])["signed_m67"].sum().reset_index()
    corrs = []
    for lag_rows in [0, 1, 2, 5, 10, 20]:
        chunks = []
        for d in DAYS:
            g = ex_px[ex_px["day"] == d].reset_index(drop=True)
            if len(g) < lag_rows + 100:
                continue
            g["fwd_mid"] = g["mid_ex"].shift(-lag_rows) - g["mid_ex"]
            mg = g.merge(flow[flow["day"] == d], on=["day", "timestamp"], how="left")
            mg["signed_m67"] = mg["signed_m67"].fillna(0.0)
            sub = mg.dropna(subset=["fwd_mid"])
            if len(sub) > 200:
                chunks.append(sub[["signed_m67", "fwd_mid"]])
        if not chunks:
            continue
        z = pd.concat(chunks, ignore_index=True)
        c = float(z["signed_m67"].corr(z["fwd_mid"])) if lag_rows > 0 else float("nan")
        corrs.append(
            {
                "lag_price_rows": lag_rows,
                "lag_ts_units": lag_rows * 100,
                "corr": c,
                "n": len(z),
            }
        )
    pd.DataFrame(corrs).to_csv(os.path.join(OUT, "r4_p2_leadlag_m67_signed_flow_vs_extract_fwd.csv"), index=False)

    # Microstructure: when sonic tight, share spread==2 on 5200
    st_rows = prices.merge(sonic, on=["day", "timestamp"], how="inner")
    st_rows = st_rows[st_rows["sonic_tight"] & st_rows["product"].eq("VEV_5200")]
    touch = (st_rows["ask_price_1"] - st_rows["bid_price_1"]) <= 2
    pd.DataFrame(
        [{"n": len(st_rows), "share_spread_le2": float(touch.mean()) if len(st_rows) else float("nan")}]
    ).to_csv(os.path.join(OUT, "r4_p2_microprice_sonic_tight_5200_spread.csv"), index=False)

    with open(os.path.join(OUT, "r4_phase2_machine_summary.json"), "w") as f:
        json.dump({"burst_neighborhood_rows_m67": int(m67["near_m01_m22_burst"].sum()), "m67_n": int(len(m67))}, f, indent=2)

    print("Phase2 outputs ->", OUT)


if __name__ == "__main__":
    main()
