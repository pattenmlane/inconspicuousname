"""Realistic MM PnL simulator for VEV_5100/5200/5300.

Key insight: the IMC backtester only fills against the historical trades.csv
(only 3-45 trades/day/strike in this dataset), which dramatically understates
real fill rates. In the live IMC simulator, house bots aggressively cross
passive orders posted inside the wall.

Fill model: at each tick we post bid+1 / ask-1 (or join the touch if spread<=2).
We assume the order is filled by `fill_size` contracts in the direction the
mid actually moves on the next tick:
  - if next-tick mid > current mid + 0.5: someone lifted offers => OUR ask fills
  - if next-tick mid < current mid - 0.5: someone hit bids => OUR bid fills
This is conservative: it assumes we ALWAYS get filled when there's flow in
the right direction, but only at one side.

We track inventory, mark to market each tick, and compute PnL.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("Prosperity4Data/ROUND_3")
STRIKES = [5100, 5200, 5300]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
POS_LIM = 300
SOFT_CAP = 200
QUOTE_SIZE = 30  # how many we're willing to trade per tick on each side


def simulate_strike(day: int, voucher: str, fill_size: int = QUOTE_SIZE,
                    soft_cap: int = SOFT_CAP, lim: int = POS_LIM,
                    move_threshold: float = 0.5, take_strategy: str = "passive_mm"):
    df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
    d = df[df["product"] == voucher].sort_values("timestamp").reset_index(drop=True)
    if d.empty:
        return None

    pos = 0
    cash = 0.0
    pnl_path = []
    fills = 0
    next_mid = d["mid_price"].shift(-1).values
    bid1 = d["bid_price_1"].values
    ask1 = d["ask_price_1"].values
    mid = d["mid_price"].values
    bsz = d["bid_volume_1"].fillna(0).values.astype(int)
    asz = d["ask_volume_1"].fillna(0).values.astype(int)

    for i in range(len(d)):
        bb = bid1[i]; ba = ask1[i]; m = mid[i]
        if pd.isna(bb) or pd.isna(ba):
            pnl_path.append(cash + pos * (m if not pd.isna(m) else 0))
            continue
        spread = int(ba - bb)

        # Choose quote prices based on spread
        if spread >= 3:
            bid_px = int(bb) + 1
            ask_px = int(ba) - 1
        elif spread == 2:
            bid_px = int(bb)
            ask_px = int(ba)
        else:
            pnl_path.append(cash + pos * m)
            continue

        # Inventory skew: when long, drop our ask (be hit-able); when short, raise bid
        if pos > 50:
            ask_px = max(int(bb) + 1, ask_px - 1)
        elif pos < -50:
            bid_px = min(int(ba) - 1, bid_px + 1)

        # Determine fill outcome from next tick movement
        if i + 1 < len(d) and not np.isnan(next_mid[i]):
            move = next_mid[i] - m
            # Compute available capacity
            buy_cap = min(fill_size, lim - pos, max(0, soft_cap - pos))
            sell_cap = min(fill_size, lim + pos, max(0, soft_cap + pos))

            # Sell side fills when mid moves UP enough
            if move > move_threshold and sell_cap > 0:
                qty = sell_cap
                cash += qty * ask_px
                pos -= qty
                fills += 1
            # Buy side fills when mid moves DOWN enough
            elif move < -move_threshold and buy_cap > 0:
                qty = buy_cap
                cash -= qty * bid_px
                pos += qty
                fills += 1

        pnl_path.append(cash + pos * m)

    final_pnl = cash + pos * (mid[-1] if not pd.isna(mid[-1]) else 0)
    return {
        "voucher": voucher, "day": day,
        "final_pnl": final_pnl,
        "final_pos": pos,
        "n_fills": fills,
        "max_long": int(np.nanmax(np.cumsum([0]))),  # placeholder
        "pnl_series": pnl_path,
    }


def main():
    rows = []
    for day in (0, 1, 2):
        for v in VOUCHERS:
            r = simulate_strike(day, v)
            if r is None:
                continue
            rows.append((day, v, r["final_pnl"], r["final_pos"], r["n_fills"]))
    print(f"{'day':>4} {'voucher':>10} {'final_pnl':>12} {'pos':>6} {'fills':>7}")
    for d, v, p, pos, n in rows:
        print(f"{d:>4} {v:>10} {p:>12.2f} {pos:>6} {n:>7}")
    print(f"\nTotal: {sum(r[2] for r in rows):,.2f}")
    print(f"Per-day average: {sum(r[2] for r in rows)/3:,.2f}")
    print(f"Per-(day, strike) average: {sum(r[2] for r in rows)/9:,.2f}")


if __name__ == "__main__":
    main()
