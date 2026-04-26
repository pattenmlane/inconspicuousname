#!/usr/bin/env python3
"""
Phase 3 (inclineGod): spread–spread and spread vs short-horizon price change on R4 aligned panel.

Forward horizon: 1 price row = +100 raw timestamp (groupby day diff(1) on time-sorted panel).

Outputs:
- r4_p3_spread_price_corr_panels.csv — one row: pooled; one row: sonic_tight subset
- r4_p3_incline_panels_by_day.json — by-day correlation blocks (same keys)
"""
from __future__ import annotations

import json
import os

import pandas as pd

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "analysis_outputs")
PRICE_GLOB = os.path.join("Prosperity4Data", "ROUND_4", "prices_round_4_day_{d}.csv")
DAYS = (1, 2, 3)
TH = 2

SPREAD_COLS = ("s5200", "s5300", "s_ext", "s5000")
# Pair each product spread with the 1-tick own-mid change (not cross-joined to other mids).
DELTA_BY_SPREAD = {
    "s_ext": "d1_m_ext",
    "s5000": "d1_mid5000",
    "s5200": "d1_mid5200",
    "s5300": "d1_mid5300",
}


def one_prod(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = df[df["product"] == product].drop_duplicates("timestamp", keep="first").sort_values("timestamp")
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return pd.DataFrame(
        {
            "day": v["day"].values,
            "timestamp": v["timestamp"].values,
            "spread": (ask - bid).astype(float).values,
            "mid": mid.astype(float).values,
        }
    )


def aligned_panel(day: int) -> pd.DataFrame:
    p = pd.read_csv(PRICE_GLOB.format(d=day), sep=";")
    p["day"] = day
    a = one_prod(p, "VEV_5200").rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = one_prod(p, "VEV_5300").rename(columns={"spread": "s5300", "mid": "mid5300"})
    e = one_prod(p, "VELVETFRUIT_EXTRACT").rename(columns={"spread": "s_ext", "mid": "m_ext"})
    v5 = one_prod(p, "VEV_5000").rename(columns={"spread": "s5000", "mid": "mid5000"})
    m = a.merge(b, on=["day", "timestamp"], how="inner").merge(e, on=["day", "timestamp"], how="inner")
    m = m.merge(v5, on=["day", "timestamp"], how="inner")
    return m.sort_values("timestamp").reset_index(drop=True)


def add_deltas(m: pd.DataFrame) -> pd.DataFrame:
    out = m.copy()
    for col, name in [("m_ext", "d1_m_ext"), ("mid5000", "d1_mid5000"), ("mid5200", "d1_mid5200"), ("mid5300", "d1_mid5300")]:
        out[name] = out.groupby("day")[col].diff(1)
    return out


def corr_safe(a: pd.Series, b: pd.Series) -> float:
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 50:
        return float("nan")
    return float(s["a"].corr(s["b"]))


def block_stats(df: pd.DataFrame) -> dict[str, float | int | str]:
    r: dict[str, float | int | str] = {"n_rows": int(len(df))}
    if len(df) < 20:
        return r
    sc = list(SPREAD_COLS)
    for i, s1 in enumerate(sc):
        for j in range(i + 1, len(sc)):
            s2 = sc[j]
            r[f"corr_{s1}_{s2}"] = corr_safe(df[s1], df[s2])
    for sp, dcol in DELTA_BY_SPREAD.items():
        if dcol in df.columns:
            r[f"corr_{sp}__{dcol}"] = corr_safe(df[sp], df[dcol])
    return r


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    parts = []
    for d in DAYS:
        m = aligned_panel(d)
        m = add_deltas(m)
        m["sonic_tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
        parts.append(m)
    panel = pd.concat(parts, ignore_index=True)

    pooled = block_stats(panel)
    pooled["label"] = "pooled_all_rows"
    tight = block_stats(panel.loc[panel["sonic_tight"]])
    tight["label"] = "pooled_sonic_tight"

    by_day: list[dict] = []
    for d, g in panel.groupby("day"):
        b = block_stats(g)
        b["day"] = int(d)
        by_day.append(b)

    with open(os.path.join(OUT, "r4_p3_incline_panels_by_day.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "horizon": "1 price row = +100 timestamp units; delta = first diff within day on sorted panel",
                "pooled": {**pooled},
                "sonic_tight_subset": {**tight},
                "by_day": by_day,
            },
            f,
            indent=2,
        )

    pd.DataFrame([pooled, tight]).to_csv(os.path.join(OUT, "r4_p3_spread_price_corr_panels.csv"), index=False)
    print("wrote r4_p3_spread_price_corr_panels.csv, r4_p3_incline_panels_by_day.json")


if __name__ == "__main__":
    main()
