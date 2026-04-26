#!/usr/bin/env python3
"""
Round 4: merge extract trades with Sonic joint-tight flag at same timestamp (inner join
to aligned 5200/5300/extract panel). Count notional by (buyer,seller) when tight vs not.

Output: r4_counterparty_extract_under_tight.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = Path("Prosperity4Data/ROUND_4")
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

EXTRACT = "VELVETFRUIT_EXTRACT"
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
TH = 2


def _one(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    v = (
        df[df["product"] == sym]
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")
    )
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return v.assign(
        spread=(ask - bid).astype(float),
        mid=mid,
    )[["timestamp", "spread", "mid"]]


def panel(d: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
    df = df[df["day"] == d]
    a = _one(df, VEV_5200).rename(columns={"spread": "s5200"})
    b = _one(df, VEV_5300).rename(columns={"spread": "s5300"})
    e = _one(df, EXTRACT)[["timestamp", "spread"]].rename(columns={"spread": "s_ext"})
    m = a.merge(b, on="timestamp", how="inner").merge(e, on="timestamp", how="inner")
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    m["csv_day"] = d
    return m[["csv_day", "timestamp", "tight"]]


def main() -> None:
    pan = pd.concat([panel(d) for d in (1, 2, 3)], ignore_index=True)
    pan = pan.set_index(["csv_day", "timestamp"])

    agg_t: dict[str, float] = defaultdict(float)
    agg_l: dict[str, float] = defaultdict(float)
    n_t = n_l = 0

    for d in (1, 2, 3):
        tr = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        tr = tr[tr["symbol"] == EXTRACT]
        for _, r in tr.iterrows():
            ts = int(r["timestamp"])
            key = (d, ts)
            if key not in pan.index:
                continue
            tight = bool(pan.loc[key, "tight"])
            b = str(r.get("buyer", "") or "").strip()
            s = str(r.get("seller", "") or "").strip()
            pair = f"{b}->{s}"
            q = float(r["price"]) * float(r["quantity"])
            if tight:
                agg_t[pair] += abs(q)
                n_t += 1
            else:
                agg_l[pair] += abs(q)
                n_l += 1

    top_t = sorted(agg_t.items(), key=lambda x: -x[1])[:12]
    top_l = sorted(agg_l.items(), key=lambda x: -x[1])[:12]
    obj = {
        "n_extract_trades_tight": n_t,
        "n_extract_trades_loose": n_l,
        "top_pairs_notional_tight": [{"pair": p, "notional": v} for p, v in top_t],
        "top_pairs_notional_loose": [{"pair": p, "notional": v} for p, v in top_l],
    }
    pth = OUT / "r4_counterparty_extract_under_tight.json"
    pth.write_text(json.dumps(obj, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
