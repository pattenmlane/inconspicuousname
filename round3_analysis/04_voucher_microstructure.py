"""Profile microstructure for VEV_5100, 5200, 5300:
  - bid/ask spread distribution
  - L1 depth (sizes available)
  - mid-price distribution and move sizes
  - trade volume per day, in/out of book
  - option_mid - theo deviation distribution (z-scores)
The goal: figure out where $20K/day/strike comes from.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

DATA = Path("Prosperity4Data/ROUND_3")
STRIKES = [5100, 5200, 5300]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
UND = "VELVETFRUIT_EXTRACT"
STEPS_PER_DAY = 10_000
DTE_AT_OPEN = {0: 8, 1: 7, 2: 6}

SMILE_A = 0.14215151147708086
SMILE_B = -0.0016298611395181932
SMILE_C = 0.23576325646627055


def t_years(day: int, ts: int) -> float:
    step = ts // 100
    rem = DTE_AT_OPEN[day] * STEPS_PER_DAY - step
    return rem / (365.0 * STEPS_PER_DAY)


def bs_call(S, K, T, sig):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def smile(S, K, T):
    m = math.log(K / S) / math.sqrt(T)
    return SMILE_A * m * m + SMILE_B * m + SMILE_C


def load_prices(day):
    df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
    return df


def load_trades(day):
    return pd.read_csv(DATA / f"trades_round_3_day_{day}.csv", sep=";")


def per_voucher(day):
    df = load_prices(day)
    out = {}
    for v in VOUCHERS:
        d = df[df["product"] == v].sort_values("timestamp").reset_index(drop=True)
        d = d.copy()
        d["spread"] = d["ask_price_1"] - d["bid_price_1"]
        d["mid_change"] = d["mid_price"].diff().abs()
        d["bid_size"] = d["bid_volume_1"].fillna(0)
        d["ask_size"] = d["ask_volume_1"].fillna(0)
        out[v] = d
    return out


def main():
    print("=" * 70)
    print("BID-ASK SPREAD & DEPTH (per voucher, per day)")
    print("=" * 70)
    for day in (0, 1, 2):
        pv = per_voucher(day)
        und = load_prices(day)
        und = und[und["product"] == UND].set_index("timestamp")
        for v in VOUCHERS:
            d = pv[v]
            spread_dist = d["spread"].value_counts().sort_index().head(8)
            print(f"\nday={day} {v}:")
            print(f"  ticks: {len(d)},  mid mean={d['mid_price'].mean():.2f}, std={d['mid_price'].std():.3f}")
            print(f"  spread distribution (top 8): {dict(spread_dist)}")
            print(f"  L1 sizes: bid mean={d['bid_size'].mean():.1f}  ask mean={d['ask_size'].mean():.1f}")
            print(f"  mid |change|: mean={d['mid_change'].mean():.3f}  >0 frac={(d['mid_change']>0).mean():.3f}")

    print("\n" + "=" * 70)
    print("TRADE FLOW (from trades csv)")
    print("=" * 70)
    for day in (0, 1, 2):
        t = load_trades(day)
        for v in VOUCHERS:
            tt = t[t["symbol"] == v]
            if len(tt) == 0:
                continue
            print(f"day={day} {v}: n_trades={len(tt)}, total_qty={tt['quantity'].sum()}, "
                  f"avg_size={tt['quantity'].mean():.1f}, "
                  f"price_range=({tt['price'].min():.0f}, {tt['price'].max():.0f})")

    print("\n" + "=" * 70)
    print("MID - THEO  (theo from quadratic smile)  -- distribution + z-scores")
    print("=" * 70)
    for day in (0, 1, 2):
        und = load_prices(day)
        und = und[und["product"] == UND].set_index("timestamp")["mid_price"]
        pv = per_voucher(day)
        for v in VOUCHERS:
            d = pv[v]
            K = int(v.split("_")[1])
            joined = d.set_index("timestamp").join(und.rename("S")).dropna(subset=["S", "mid_price"])
            theos = []
            for ts, row in joined.iterrows():
                T = t_years(day, int(ts))
                if T <= 0:
                    continue
                sig = smile(row["S"], K, T)
                theos.append(bs_call(row["S"], K, T, sig))
            theos = np.asarray(theos)
            mids = joined["mid_price"].values[: len(theos)]
            dev = mids - theos
            print(f"day={day} {v}: n={len(dev)}  mean(mid-theo)={dev.mean():+.3f}  std={dev.std():.3f}  "
                  f"abs p50={np.median(np.abs(dev)):.3f}  abs p90={np.quantile(np.abs(dev),0.9):.3f}  "
                  f"abs p99={np.quantile(np.abs(dev),0.99):.3f}")

    print("\n" + "=" * 70)
    print("MID-PRICE AUTOCORRELATION (mean-reversion test)")
    print("=" * 70)
    for day in (0, 1, 2):
        pv = per_voucher(day)
        for v in VOUCHERS:
            d = pv[v]
            r = d["mid_price"].diff().dropna()
            if len(r) > 10:
                acf1 = r.autocorr(1)
                acf5 = r.autocorr(5)
                print(f"day={day} {v}: lag1_acf(d_mid)={acf1:+.4f}  lag5={acf5:+.4f}  std(d_mid)={r.std():.3f}")


if __name__ == "__main__":
    main()
