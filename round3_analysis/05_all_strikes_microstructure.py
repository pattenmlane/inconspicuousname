"""Why 5100/5200/5300 specifically? Compare to 5000/5400/5500."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("Prosperity4Data/ROUND_3")
STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]


def main():
    print(f"{'Day':>3} {'Voucher':>10} {'spread_med':>10} {'spread_p90':>10} "
          f"{'L1_bid':>7} {'L1_ask':>7} {'lag1_acf':>10} {'std_dmid':>10} "
          f"{'pct_spread1':>11} {'pct_spread2':>11} {'pct_spread3':>11} {'pct_spread4plus':>15}")
    for day in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        for v in VOUCHERS:
            d = df[df["product"] == v].sort_values("timestamp").reset_index(drop=True)
            if d.empty:
                continue
            sp = (d["ask_price_1"] - d["bid_price_1"]).dropna()
            r = d["mid_price"].diff().dropna()
            print(f"{day:>3} {v:>10} {sp.median():>10.0f} {sp.quantile(0.9):>10.0f} "
                  f"{d['bid_volume_1'].fillna(0).mean():>7.1f} {d['ask_volume_1'].fillna(0).mean():>7.1f} "
                  f"{r.autocorr(1):>10.3f} {r.std():>10.3f} "
                  f"{(sp==1).mean()*100:>10.1f}% {(sp==2).mean()*100:>10.1f}% "
                  f"{(sp==3).mean()*100:>10.1f}% {(sp>=4).mean()*100:>14.1f}%")


if __name__ == "__main__":
    main()
