#!/usr/bin/env python3
"""Mark 67 aggressive buys on VELVETFRUIT_EXTRACT: fwd20 mid vs Sonic joint gate (R4)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[3] / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
TH, K, DAYS = 2, 20, (1, 2, 3)


def load_prices(day: int) -> pd.DataFrame:
    return pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")


def spread_ask_mid(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = df[df["product"] == product].drop_duplicates("timestamp").sort_values("timestamp")
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return pd.DataFrame(
        {
            "timestamp": v["timestamp"].values,
            "spread": (ask - bid).astype(float),
            "ask": ask,
            "mid": mid,
        }
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in DAYS:
        df = load_prices(d)
        s5 = spread_ask_mid(df, "VEV_5200").rename(columns={"spread": "s5200"})[["timestamp", "s5200"]]
        s3 = spread_ask_mid(df, "VEV_5300").rename(columns={"spread": "s5300"})[["timestamp", "s5300"]]
        ex = spread_ask_mid(df, "VELVETFRUIT_EXTRACT").rename(
            columns={"ask": "ask_ex", "mid": "mid_ex"}
        )[["timestamp", "ask_ex", "mid_ex"]]

        g = s5.merge(s3, on="timestamp").merge(ex, on="timestamp")
        g["joint_tight"] = (g["s5200"] <= TH) & (g["s5300"] <= TH)
        g = g.sort_values("timestamp").reset_index(drop=True)
        g["fwd20"] = g["mid_ex"].shift(-K) - g["mid_ex"]

        tr = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        tr = tr[(tr["symbol"] == "VELVETFRUIT_EXTRACT") & (tr["buyer"] == "Mark 67")].copy()
        tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
        tr = tr.merge(g[["timestamp", "joint_tight", "fwd20", "ask_ex"]], on="timestamp", how="inner")
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
    out.to_csv(OUT / "r4_mark67_aggr_buy_extract_fwd20_by_gate_day.csv", index=False)
    (OUT / "r4_mark67_aggr_buy_extract_summary.json").write_text(out.to_json(orient="records", indent=2))
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
