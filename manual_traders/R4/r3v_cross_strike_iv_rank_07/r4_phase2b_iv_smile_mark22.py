#!/usr/bin/env python3
"""
Round 4 Phase 2 (deferred bullet): IV smile proxy × counterparty on voucher tape.

At each (day, timestamp) where Mark 22 is seller on any VEV_*, load aligned mids for
VELVETFRUIT_EXTRACT + all 10 VEV strikes from the same price row. Implied vol per strike
via Black–Scholes bisection (r=0). Smile "steepness" = quadratic coefficient a in
IV ~ a*m^2 + b*m + c with m = log(K/S)/sqrt(T). TTE calendar proxy: T_years = max(1e-6, (5-day)/365).

Outputs:
- r4_p2b_smile_at_m22_sell_ts.csv — one row per unique (day,timestamp) at Mark22 voucher sells
- r4_p2b_m22_sell_join_smile.csv — every Mark22→voucher sell trade joined to smile + dm_self_k20
- r4_p2b_steep_vs_fwd_k20_by_symbol.csv — mean dm_self_k20 in top vs bottom steepness quartile by symbol
- r4_p2b_machine_summary.json — quantiles of steepness, counts
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np
import pandas as pd

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "analysis_outputs")
ENRICHED = os.path.join(OUT, "r4_p1_trade_enriched.csv")
GATE_CSV = os.path.join(OUT, "r4_p3_trade_enriched_with_gate.csv")
PRICE_GLOB = os.path.join("Prosperity4Data", "ROUND_4", "prices_round_4_day_{d}.csv")
DAYS = (1, 2, 3)
STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
VOUCHERS = [f"VEV_{k}" for k in STRIKES]


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 1e-12:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def implied_vol_call(market: float, S: float, K: float, T: float, r: float = 0.0) -> float:
    intrinsic = max(S - K, 0.0)
    if market <= intrinsic + 1e-9:
        return float("nan")
    if market >= S - 1e-9:
        return float("nan")
    if S <= 0 or K <= 0 or T <= 0:
        return float("nan")

    def f(sig: float) -> float:
        return bs_call_price(S, K, T, sig, r) - market

    lo, hi = 1e-5, 15.0
    fl, fh = f(lo), f(hi)
    if fl > 0 or fh < 0:
        return float("nan")
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def t_years(day: int) -> float:
    return max(1e-6, float(5 - int(day)) / 365.0)


def load_prices_day(day: int) -> pd.DataFrame:
    p = pd.read_csv(PRICE_GLOB.format(d=day), sep=";")
    p["day"] = day
    return p


def smile_row(prices_day: pd.DataFrame, day: int, ts: int) -> dict[str, Any]:
    """Return steepness (quad a), n_iv_ok, S, or NaNs if insufficient."""
    T = t_years(day)
    row: dict[str, Any] = {"day": day, "timestamp": ts, "T_years": T}
    sub = prices_day[(prices_day["timestamp"] == ts)]
    ex = sub[sub["product"] == "VELVETFRUIT_EXTRACT"]
    if ex.empty:
        for k in ("steepness", "n_iv", "S"):
            row[k] = float("nan")
        return row
    S = float(ex.iloc[0]["mid_price"])
    row["S"] = S
    ks: list[float] = []
    ivs: list[float] = []
    mids: list[float] = []
    for prod, K in zip(VOUCHERS, [float(k) for k in STRIKES]):
        g = sub[sub["product"] == prod]
        if g.empty:
            continue
        mid = float(g.iloc[0]["mid_price"])
        iv = implied_vol_call(mid, S, K, T, 0.0)
        if iv == iv and iv > 0:
            ks.append(K)
            ivs.append(iv)
            mids.append(mid)
    row["n_iv"] = len(ivs)
    if len(ivs) < 5:
        row["steepness"] = float("nan")
        row["iv_rms"] = float("nan")
        return row
    sqrtT = math.sqrt(T)
    m_arr = np.array([math.log(k / S) / sqrtT for k in ks], dtype=float)
    y_arr = np.array(ivs, dtype=float)
    try:
        coeff = np.polyfit(m_arr, y_arr, 2)
    except (ValueError, np.linalg.LinAlgError):
        row["steepness"] = float("nan")
        row["iv_rms"] = float("nan")
        return row
    if not np.all(np.isfinite(coeff)):
        row["steepness"] = float("nan")
        row["iv_rms"] = float("nan")
        return row
    row["steepness"] = float(coeff[0])
    pred = np.polyval(coeff, m_arr)
    row["iv_rms"] = float(np.sqrt(np.mean((y_arr - pred) ** 2)))
    return row


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(ENRICHED)
    m22v = df[(df["seller"] == "Mark 22") & (df["symbol"].str.startswith("VEV_"))].copy()
    keys = m22v[["day", "timestamp"]].drop_duplicates()
    price_by_day = {d: load_prices_day(d) for d in DAYS}

    smile_rows = []
    for _, r in keys.iterrows():
        d, ts = int(r["day"]), int(r["timestamp"])
        smile_rows.append(smile_row(price_by_day[d], d, ts))
    smile_ts = pd.DataFrame(smile_rows)
    smile_ts.to_csv(os.path.join(OUT, "r4_p2b_smile_at_m22_sell_ts.csv"), index=False)

    gate = None
    gp = os.path.join(OUT, "r4_p3_trade_enriched_with_gate.csv")
    if os.path.isfile(gp):
        gate = pd.read_csv(gp)[["day", "timestamp", "sonic_tight"]].drop_duplicates()

    j = m22v.merge(smile_ts, on=["day", "timestamp"], how="left")
    if gate is not None:
        j = j.merge(gate, on=["day", "timestamp"], how="left")
    else:
        j["sonic_tight"] = False
    j.to_csv(os.path.join(OUT, "r4_p2b_m22_sell_join_smile.csv"), index=False)

    steep_valid = smile_ts["steepness"].replace([np.inf, -np.inf], np.nan).dropna()
    q25, q50, q75 = (
        float(steep_valid.quantile(0.25)),
        float(steep_valid.quantile(0.50)),
        float(steep_valid.quantile(0.75)),
    )
    j["steep_hi"] = j["steepness"] >= q75
    j["steep_lo"] = j["steepness"] <= q25

    agg_rows = []
    for sym in sorted(j["symbol"].unique()):
        g = j[j["symbol"] == sym]
        hi = g[g["steep_hi"]]["dm_self_k20"].dropna()
        lo = g[g["steep_lo"]]["dm_self_k20"].dropna()
        agg_rows.append(
            {
                "symbol": sym,
                "n_hi": int(len(hi)),
                "mean_k20_hi": float(hi.mean()) if len(hi) else float("nan"),
                "n_lo": int(len(lo)),
                "mean_k20_lo": float(lo.mean()) if len(lo) else float("nan"),
                "diff_hi_minus_lo": float(hi.mean() - lo.mean()) if len(hi) and len(lo) else float("nan"),
            }
        )
    pd.DataFrame(agg_rows).to_csv(os.path.join(OUT, "r4_p2b_steep_vs_fwd_k20_by_symbol.csv"), index=False)

    tight_sub = j[j["sonic_tight"] == True]["steepness"].dropna()  # noqa: E712
    loose_sub = j[j["sonic_tight"] == False]["steepness"].dropna()  # noqa: E712

    summary = {
        "n_unique_ts_m22_vev_sell": int(len(keys)),
        "n_m22_vev_sell_trades": int(len(m22v)),
        "steepness_quantiles_at_those_ts": {"q25": q25, "q50": q50, "q75": q75},
        "n_finite_steepness": int(steep_valid.shape[0]),
        "mean_steepness": float(steep_valid.mean()) if len(steep_valid) else None,
        "sonic_tight_fraction_of_m22_sell_rows": float(j["sonic_tight"].mean()) if len(j) else None,
        "mean_steep_when_sonic_tight": float(tight_sub.mean()) if len(tight_sub) else None,
        "mean_steep_when_sonic_loose": float(loose_sub.mean()) if len(loose_sub) else None,
    }
    with open(os.path.join(OUT, "r4_p2b_machine_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
