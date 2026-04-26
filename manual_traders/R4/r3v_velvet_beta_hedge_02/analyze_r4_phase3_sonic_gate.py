"""
Round 4 Phase 3 — Sonic joint gate (VEV_5200 & VEV_5300 BBO spread <= 2) on R4 tape.

Matches round3work/vouchers_final_strategy/analyze_vev_5200_5300_tight_gate_r3.py convention:
spread = ask_price_1 - bid_price_1; inner join 5200+5300 on timestamp (per day); tight = both <= TH.

1) Panel stats: P(tight), corr(s5200,s5300), corr(s5200, s_ext), forward extract K=20 tight vs loose.
2) Merge gate onto trades (day,timestamp); Phase-1 style forward mids on merged trades.
3) Three-way: Mark 01→22 on VEV_5300 × tight × day; Mark 67 aggr buy extract × tight; Mark 22 aggr sell 5300 × tight.

Outputs CSV + JSON under outputs/
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
TH = 2
K_FWD = 20
KS_TRADE = [5, 20]

VEV5200 = "VEV_5200"
VEV5300 = "VEV_5300"
EXTRACT = "VELVETFRUIT_EXTRACT"


def spread_cols(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = df[df["product"] == product].drop_duplicates(subset=["timestamp"], keep="first")
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return pd.DataFrame(
        {
            "timestamp": v["timestamp"].values,
            "spread": (ask - bid).astype(float).values,
            "mid": mid.astype(float).values,
        }
    )


def gate_panel_for_day(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    a = spread_cols(df, VEV5200).rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = spread_cols(df, VEV5300).rename(columns={"spread": "s5300", "mid": "mid5300"})
    ext = spread_cols(df, EXTRACT).rename(columns={"spread": "s_ext", "mid": "m_ext"})
    m = a.merge(b, on="timestamp", how="inner").merge(ext, on="timestamp", how="inner")
    m = m.sort_values("timestamp").reset_index(drop=True)
    m["day"] = day
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    m["m_ext_f"] = m["m_ext"].shift(-K_FWD)
    m["fwd_ext_k"] = m["m_ext_f"] - m["m_ext"]
    return m


def add_trade_forwards(px: pd.DataFrame) -> pd.DataFrame:
    """Per (day, product) row-wise forward mid deltas for K in KS_TRADE."""
    out = []
    for (d, pr), g in px.groupby(["day", "product"], sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        mid = g["mid_price"].astype(float)
        for k in KS_TRADE:
            g[f"fwd_{k}"] = mid.shift(-k) - mid
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    panels: list[pd.DataFrame] = []
    panel_summaries: list[dict] = []
    for d in DAYS:
        m = gate_panel_for_day(d)
        panels.append(m)
        tight = m["tight"].values
        fe = m["fwd_ext_k"].dropna()
        t_mask = m.loc[fe.index, "tight"]
        fwd_t = m.loc[t_mask & m["fwd_ext_k"].notna(), "fwd_ext_k"]
        fwd_l = m.loc[~t_mask & m["fwd_ext_k"].notna(), "fwd_ext_k"]
        panel_summaries.append(
            {
                "day": d,
                "n": len(m),
                "p_tight": float(tight.mean()),
                "mean_fwd_ext_k20_tight": float(fwd_t.mean()) if len(fwd_t) else None,
                "mean_fwd_ext_k20_loose": float(fwd_l.mean()) if len(fwd_l) else None,
                "n_tight_fwd": int(fwd_t.shape[0]),
                "n_loose_fwd": int(fwd_l.shape[0]),
                "corr_s5200_s5300": float(m["s5200"].corr(m["s5300"])),
                "corr_s5200_s_ext": float(m["s5200"].corr(m["s_ext"])),
                "corr_s5300_s_ext": float(m["s5300"].corr(m["s_ext"])),
                "corr_s5200_dmid5200": float(m["s5200"].corr(m["mid5200"].diff())),
                "corr_s5300_dmid5300": float(m["s5300"].corr(m["mid5300"].diff())),
            }
        )
    panel_all = pd.concat(panels, ignore_index=True)
    panel_all.to_csv(OUT / "r4_phase3_gate_panel_long.csv", index=False)
    pd.DataFrame(panel_summaries).to_csv(OUT / "r4_phase3_gate_panel_by_day.csv", index=False)

    # spread-spread scatter summary (correlation matrix pooled)
    corr_mat = panel_all[["s5200", "s5300", "s_ext"]].corr()
    corr_mat.to_csv(OUT / "r4_phase3_spread_spread_corr_pooled.csv")

    # trades + gate + forwards
    px_frames: list[pd.DataFrame] = []
    for d in DAYS:
        df = pd.read_csv(
            DATA / f"prices_round_4_day_{d}.csv",
            sep=";",
            usecols=["day", "timestamp", "product", "bid_price_1", "ask_price_1", "mid_price"],
        )
        px_frames.append(df)
    px = pd.concat(px_frames, ignore_index=True)
    px["spread"] = (px["ask_price_1"] - px["bid_price_1"]).astype(float)
    px = add_trade_forwards(px)

    gate = panel_all[["day", "timestamp", "tight", "s5200", "s5300"]].drop_duplicates(["day", "timestamp"])
    trs: list[pd.DataFrame] = []
    for d in DAYS:
        t = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        t["day"] = d
        trs.append(t)
    tr = pd.concat(trs, ignore_index=True).rename(columns={"symbol": "product"})
    tr["price"] = tr["price"].astype(float)
    merged = tr.merge(px, on=["day", "timestamp", "product"], how="left")
    merged = merged.merge(gate, on=["day", "timestamp"], how="left")
    merged["aggr_buy"] = merged["price"] >= merged["ask_price_1"]
    merged["aggr_sell"] = merged["price"] <= merged["bid_price_1"]

    def summarize(mask: pd.Series, col: str) -> dict:
        sub = merged.loc[mask & merged[col].notna(), col]
        return {"n": int(len(sub)), "mean": float(sub.mean()) if len(sub) else None, "std": float(sub.std()) if len(sub) > 1 else None}

    # Mark 67 extract aggr buy
    m67 = (merged["product"] == EXTRACT) & (merged["buyer"] == "Mark 67") & merged["aggr_buy"]
    inter = {
        "Mark67_aggr_buy_extract_fwd5_tight": summarize(m67 & merged["tight"], "fwd_5"),
        "Mark67_aggr_buy_extract_fwd5_loose": summarize(m67 & ~merged["tight"], "fwd_5"),
        "Mark67_aggr_buy_extract_fwd20_tight": summarize(m67 & merged["tight"], "fwd_20"),
        "Mark67_aggr_buy_extract_fwd20_loose": summarize(m67 & ~merged["tight"], "fwd_20"),
        "Mark22_aggr_sell_5300_fwd5_tight": summarize(
            (merged["product"] == VEV5300) & (merged["seller"] == "Mark 22") & merged["aggr_sell"] & merged["tight"],
            "fwd_5",
        ),
        "Mark22_aggr_sell_5300_fwd5_loose": summarize(
            (merged["product"] == VEV5300) & (merged["seller"] == "Mark 22") & merged["aggr_sell"] & ~merged["tight"],
            "fwd_5",
        ),
    }
    (OUT / "r4_phase3_counterparty_x_gate_extract_5300.json").write_text(json.dumps(inter, indent=2), encoding="utf-8")

    # Mark 01 -> Mark 22 on VEV_5300
    m01 = (merged["buyer"] == "Mark 01") & (merged["seller"] == "Mark 22") & (merged["product"] == VEV5300)
    g01 = (
        merged.loc[m01]
        .groupby(["day", "tight"])
        .agg(n=("fwd_5", "count"), m5=("fwd_5", "mean"), m20=("fwd_20", "mean"))
        .reset_index()
    )
    g01.to_csv(OUT / "r4_phase3_mark01_to_22_on_5300_by_day_tight.csv", index=False)

    # Compare pooled Phase1-style Mark22 sell 5300 without gate split (already have loose/tight)
    merged.loc[
        (merged["product"] == VEV5300) & (merged["seller"] == "Mark 22") & merged["aggr_sell"]
    ].groupby("tight")["fwd_5"].describe().to_csv(OUT / "r4_phase3_mark22_sell_5300_fwd5_describe_by_tight.csv")

    (OUT / "r4_phase3_run_done.txt").write_text("phase3 sonic gate analysis complete\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
