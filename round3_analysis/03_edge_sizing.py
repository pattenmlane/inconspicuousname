"""Convert smile residuals to per-contract dollar edges using vega.
Also size the Hydrogel mean-reversion edge: how much profit per ±20 round-trip."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

DATA = Path("Prosperity4Data/ROUND_3")
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
UND = "VELVETFRUIT_EXTRACT"
STEPS_PER_DAY = 10_000
DTE_AT_OPEN = {0: 8, 1: 7, 2: 6}


def t_years(day: int, ts: int) -> float:
    step = ts // 100
    rem = DTE_AT_OPEN[day] * STEPS_PER_DAY - step
    return rem / (365.0 * STEPS_PER_DAY)


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return S * norm.pdf(d1) * math.sqrt(T)


def load_wide(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
    df = df[df["product"].isin([UND] + VOUCHERS)]
    return df.pivot_table(index="timestamp", columns="product", values="mid_price").sort_index()


def main():
    summary = json.loads(Path("round3_analysis/smile_summary.json").read_text())
    resid_mean = {int(k): v for k, v in summary["per_strike_resid_mean"].items() if v is not None}

    # Use day-2 average S (last day of training, closest to production)
    wide = load_wide(2)
    S_avg = float(wide[UND].mean())
    T_mid = t_years(2, 50_000)  # mid-day on day 2
    atm_iv = float(summary["atm_iv_per_day"]["2"])
    print(f"Reference: S={S_avg:.2f}  T={T_mid:.5f} years (~{T_mid*365:.2f} days)  ATM_IV={atm_iv:.4f}\n")

    # Per-strike dollar edge = vega * (resid in vol units), sign:
    #   resid > 0 means market IV is HIGH vs smile -> overpriced -> SELL
    #   resid < 0 means market IV is LOW vs smile  -> underpriced -> BUY
    print(f"{'K':>6} {'resid_iv':>10} {'vega':>10} {'edge_$/contract':>18} {'side':>6}")
    edges = {}
    for K in STRIKES:
        r = resid_mean.get(K)
        if r is None:
            continue
        v = bs_vega(S_avg, K, T_mid, atm_iv)
        edge = v * r
        side = "SELL" if edge > 0 else "BUY"
        print(f"{K:>6} {r:+10.5f} {v:10.4f} {abs(edge):18.4f} {side:>6}")
        edges[K] = {"resid_iv": r, "vega": v, "edge_dollar": edge, "side": side}

    Path("round3_analysis/strike_edges.json").write_text(json.dumps(edges, indent=2))

    # ---- Hydrogel mean-reversion edge ----
    print("\n--- HYDROGEL_PACK mean-reversion edge ---")
    panels = []
    for d in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{d}.csv", sep=";")
        h = df[df["product"] == "HYDROGEL_PACK"][["timestamp", "mid_price"]].copy()
        h["day"] = d
        panels.append(h)
    h_all = pd.concat(panels, ignore_index=True)
    print(f"n_ticks={len(h_all):,}  mean={h_all['mid_price'].mean():.2f}  std={h_all['mid_price'].std():.2f}")

    # Trivial mean-reversion strategy: buy when mid <= mean-T, sell at mean. Compute avg holding gain.
    mu = h_all["mid_price"].mean()
    for thr in (10, 15, 20, 25, 30):
        # entries below mu - thr
        entries_below = h_all[h_all["mid_price"] <= mu - thr]
        # mean future +500 ticks return
        h_idx = h_all.set_index(["day", "timestamp"]).sort_index()
        rets = []
        for _, row in entries_below.iterrows():
            d = int(row["day"])
            ts = int(row["timestamp"])
            future_ts = ts + 50_000  # 500 ticks later
            try:
                future = h_idx.loc[(d, future_ts), "mid_price"]
                rets.append(float(future) - float(row["mid_price"]))
            except KeyError:
                pass
        if rets:
            print(f"  buy@<= mu-{thr:>2}: n_entries={len(entries_below):>5}  "
                  f"avg gain after 500 ticks = ${np.mean(rets):+.3f}  med={np.median(rets):+.3f}")


if __name__ == "__main__":
    main()
