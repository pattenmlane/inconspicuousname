#!/usr/bin/env python3
"""Round 4 Phase 2 tape analysis: burst×pair conditioning, microprice, cross-lag."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
DAYS = (1, 2, 3)


def load_prices(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    df["day"] = day
    for c in ["bid_price_1", "bid_volume_1", "ask_price_1", "ask_volume_1", "mid_price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bv, av = df["bid_volume_1"].fillna(0), df["ask_volume_1"].fillna(0).abs()
    den = bv + av
    df["micro"] = np.where(
        den > 0,
        (df["bid_price_1"] * bv + df["ask_price_1"] * av) / den,
        df["mid_price"],
    )
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    return df


def load_trades(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"trades_round_4_day_{day}.csv", sep=";")
    df["day"] = day
    df["product"] = df["symbol"].astype(str)
    return df


def burst_table(trades: pd.DataFrame) -> pd.DataFrame:
    g = trades.groupby(["day", "timestamp"]).size().reset_index(name="n_trades")
    g["is_burst"] = g["n_trades"] >= 3
    return g


def forward_same_product(prices: pd.DataFrame, k: int) -> pd.DataFrame:
    """Add fwd_k mid per (day, product) row order."""
    out = []
    for (d, p), g in prices.groupby(["day", "product"]):
        g = g.sort_values("timestamp").reset_index(drop=True)
        g["fwd_mid_k"] = g["mid_price"].shift(-k) - g["mid_price"]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trades = pd.concat([load_trades(d) for d in DAYS], ignore_index=True)
    burst = burst_table(trades)
    m = trades.merge(burst[["day", "timestamp", "is_burst", "n_trades"]], on=["day", "timestamp"], how="left")

    # Burst × Mark01→Mark22 × VEV
    mask_b = (
        m["is_burst"]
        & (m["buyer"] == "Mark 01")
        & (m["seller"] == "Mark 22")
        & m["product"].str.startswith("VEV_")
    )
    sub = m.loc[mask_b].copy()
    # merge extract fwd at same ts from prices
    px = pd.concat([load_prices(d) for d in DAYS], ignore_index=True)
    ex = px[px["product"] == "VELVETFRUIT_EXTRACT"][["day", "timestamp", "mid_price"]].rename(
        columns={"mid_price": "ex_mid"}
    )
    sub = sub.merge(ex, on=["day", "timestamp"], how="left")
    # forward extract mid +20 rows in ex series
    exs = []
    for d in DAYS:
        g = px[(px["day"] == d) & (px["product"] == "VELVETFRUIT_EXTRACT")].sort_values("timestamp")
        g["fwd20_ex"] = g["mid_price"].shift(-20) - g["mid_price"]
        exs.append(g[["day", "timestamp", "fwd20_ex"]])
    exf = pd.concat(exs, ignore_index=True)
    sub = sub.merge(exf, on=["day", "timestamp"], how="left")
    sub.to_csv(OUT / "r4_phase2_burst_mark01_22_vev_with_extract_fwd.csv", index=False)
    summ = {
        "n_rows": int(len(sub)),
        "mean_fwd20_ex_at_print": float(sub["fwd20_ex"].dropna().mean()) if len(sub) else float("nan"),
        "median_fwd20_ex": float(sub["fwd20_ex"].dropna().median()) if len(sub) else float("nan"),
    }
    (OUT / "r4_phase2_burst_mark01_22_summary.json").write_text(json.dumps(summ, indent=2))

    # Microprice vs spread compression: correlate Δmicro and |Δmid| next 5 rows
    mic_rows = []
    for d in DAYS:
        p = load_prices(d)
        p = forward_same_product(p, 5)
        p["d_micro"] = p.groupby(["day", "product"])["micro"].diff()
        p["abs_fwd5"] = p["fwd_mid_k"].abs()
        mic_rows.append(p[["day", "product", "spread", "d_micro", "abs_fwd5"]].dropna())
    mp = pd.concat(mic_rows, ignore_index=True)
    mp = mp.replace([np.inf, -np.inf], np.nan).dropna()
    mp = mp[np.isfinite(mp["spread"]) & np.isfinite(mp["abs_fwd5"])]
    corr = float(mp["spread"].corr(mp["abs_fwd5"])) if len(mp) > 10 else float("nan")
    (OUT / "r4_phase2_microprice_spread_fwdcorr.json").write_text(
        json.dumps({"corr_spread_abs_fwd5_mid": corr, "n": int(len(mp))}, indent=2)
    )

    # Cross-lag: signed vol Mark 01 buys extract vs fwd extract mid
    ex_px = px[px["product"] == "VELVETFRUIT_EXTRACT"].copy()
    t_ex = trades[(trades["product"] == "VELVETFRUIT_EXTRACT") & (trades["buyer"] == "Mark 01")]
    flow = t_ex.groupby(["day", "timestamp"])["quantity"].sum().reset_index(name="buy01_qty")
    ex_px = ex_px.merge(flow, on=["day", "timestamp"], how="left").fillna({"buy01_qty": 0})
    ex_px = ex_px.sort_values(["day", "timestamp"])
    lags = {}
    for lag in (0, 1, 2, 5, 10):
        ex_px[f"fwd_{lag}"] = ex_px.groupby("day")["mid_price"].shift(-lag) - ex_px["mid_price"]
    subf = ex_px[ex_px["buy01_qty"] > 0]
    for lag in (0, 1, 2, 5, 10):
        col = f"fwd_{lag}"
        z = subf[["buy01_qty", col]].dropna()
        lags[str(lag)] = float(z["buy01_qty"].corr(z[col])) if len(z) > 30 else float("nan")
    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    (OUT / "r4_phase2_extract_mark01_buy_flow_lagcorr.json").write_text(json.dumps(_json_safe(lags), indent=2))

    print("Phase2 outputs ->", OUT)


if __name__ == "__main__":
    main()
