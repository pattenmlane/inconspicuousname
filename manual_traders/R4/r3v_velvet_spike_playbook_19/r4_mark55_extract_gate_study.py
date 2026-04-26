#!/usr/bin/env python3
"""Mark 55 aggressive buys on VELVETFRUIT_EXTRACT: fwd20 vs Sonic joint gate (R4)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[3] / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
TH, K, DAYS = 2, 20, (1, 2, 3)


def load_prices(day: int) -> pd.DataFrame:
    return pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")


def gate_panel(df: pd.DataFrame) -> pd.DataFrame:
    s5 = (
        df[df["product"] == "VEV_5200"]
        .drop_duplicates("timestamp")
        .assign(
            s5200=lambda x: pd.to_numeric(x["ask_price_1"], errors="coerce")
            - pd.to_numeric(x["bid_price_1"], errors="coerce")
        )[["timestamp", "s5200"]]
    )
    s3 = (
        df[df["product"] == "VEV_5300"]
        .drop_duplicates("timestamp")
        .assign(
            s5300=lambda x: pd.to_numeric(x["ask_price_1"], errors="coerce")
            - pd.to_numeric(x["bid_price_1"], errors="coerce")
        )[["timestamp", "s5300"]]
    )
    ex = df[df["product"] == "VELVETFRUIT_EXTRACT"].drop_duplicates("timestamp").sort_values("timestamp")
    bid = pd.to_numeric(ex["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(ex["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(ex["mid_price"], errors="coerce")
    ex = pd.DataFrame(
        {
            "timestamp": ex["timestamp"].values,
            "bid_ex": bid,
            "ask_ex": ask,
            "mid_ex": mid,
        }
    )
    g = s5.merge(s3, on="timestamp").merge(ex, on="timestamp")
    g["joint_tight"] = (g["s5200"] <= TH) & (g["s5300"] <= TH)
    g = g.sort_values("timestamp").reset_index(drop=True)
    g["fwd20"] = g["mid_ex"].shift(-K) - g["mid_ex"]
    return g


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in DAYS:
        g = gate_panel(load_prices(d))
        tr = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        tr = tr[(tr["symbol"] == "VELVETFRUIT_EXTRACT") & (tr["buyer"] == "Mark 55")].copy()
        tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
        tr = tr.merge(
            g[["timestamp", "joint_tight", "fwd20", "ask_ex"]],
            on="timestamp",
            how="inner",
        )
        tr = tr[tr["price"] >= tr["ask_ex"]]
        for jt, label in [(True, "tight"), (False, "wide")]:
            sub = tr.loc[tr["joint_tight"] == jt, "fwd20"].dropna()
            rows.append(
                {
                    "day": d,
                    "gate": label,
                    "n": int(len(sub)),
                    "mean_fwd20": float(sub.mean()) if len(sub) else float("nan"),
                    "median_fwd20": float(sub.median()) if len(sub) else float("nan"),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "r4_mark55_aggr_buy_extract_fwd20_by_gate_day.csv", index=False)
    summary = {
        "definition": "Aggressive buy: Mark 55 buyer on extract, price >= ask1 at timestamp; joint_tight from 5200/5300 spreads <=2.",
        "rows": out.to_dict(orient="records"),
        "note": "Unlike Mark 67 study, **tight vs wide ordering flips by day** (e.g. day 2–3 wide has higher mean_fwd20 than tight). Do not treat Sonic tight as a universal cleaner for M55 lift.",
    }
    (OUT / "r4_mark55_extract_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
